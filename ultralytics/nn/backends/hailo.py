# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import weakref
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import torch

from .base import BaseBackend


class HailoBackend(BaseBackend):
    """Hailo AI inference backend for Hailo-8/8L/10H/15H/15M/15L accelerators.

    Loads compiled Hailo models (``.hef`` files) and runs inference using HailoRT (``hailo_platform``).

    The chip's input layer normalizes by 255, so the on-chip input contract is RGB ``uint8`` in ``[0, 255]``.
    The standard Ultralytics predict pipeline emits float32 ``[0, 1]`` tensors, which this backend silently
    casts back to uint8.

    Two HEF families are supported, dispatched by ``model_family`` in ``metadata.yaml``:

    - ``yolov8`` / ``yolo11`` — single-output HEF with on-chip NMS (``nms_postprocess(meta_arch=yolov8)``).
      The chip emits a per-class list of ``(N_class, 5)`` arrays which we just rescale to pixel coords.
    - ``yolo26`` — NMS-free end2end. Six raw conv outputs come off the chip (3 strides × {box reg, cls});
      anchor decode + sigmoid + topk run on host because the NPU doesn't support the topk/gather ops.

    Both families produce Ultralytics' end2end ``(1, N, 6)`` format ``[x1, y1, x2, y2, conf, cls]`` for the
    predictor's end2end short path.
    """

    def load_model(self, weight: str | Path) -> None:
        """Open a HailoRT VDevice and configure an InferModel from the supplied .hef.

        Args:
            weight (str | Path): Path to a ``.hef`` file or to a ``_hailo_model/`` directory containing one.
        """
        try:
            from hailo_platform import VDevice
        except ImportError as e:
            raise ImportError(
                "HailoRT ('hailo_platform') is required for Hailo inference but is not installed.\n"
                "Download HailoRT from:\n"
                "  https://hailo.ai/developer-zone/software-downloads/\n"
                "Install the runtime package AND the driver matching your hardware:\n"
                "  - PCIe driver for M.2 / mPCIe accelerator modules\n"
                "  - USB driver for USB dongle accelerators\n"
                "then re-run inference. HailoRT is not on PyPI; manual install is required."
            ) from e

        w = Path(weight)
        hef = w if w.is_file() and w.suffix == ".hef" else next(w.rglob("*.hef"), None)
        if hef is None or not hef.exists():
            raise FileNotFoundError(f"No .hef file found at/under: {w}")

        # Load metadata first — model_family determines whether we expect single-output (on-chip NMS)
        # or multi-output (yolo26 raw heads, host-side postprocess).
        metadata: dict = {}
        metadata_file = (hef.parent if w.is_file() else w) / "metadata.yaml"
        if metadata_file.exists():
            from ultralytics.utils import YAML

            metadata = YAML.load(metadata_file) or {}
            self.apply_metadata(metadata)
        self._model_family = str(metadata.get("model_family", "yolov8"))

        stack = ExitStack()
        vdevice = stack.enter_context(VDevice())
        self._infer_model = vdevice.create_infer_model(str(hef))

        outputs = self._infer_model.outputs
        outputs = outputs() if callable(outputs) else outputs

        if self._model_family == "yolo26":
            # YOLO26 ships with the Hailo Model Zoo's a16_w16 quantization on the conv leaves, so the
            # raw chip outputs are uint16. Request FLOAT32 here so the SDK dequantizes on-the-fly and
            # our host postprocess (anchors + dist2bbox + sigmoid + topk) consumes natural float values.
            # Must be set BEFORE configure() — the binding format is locked in at that point.
            from hailo_platform import FormatType

            for o in outputs:
                o.set_format_type(FormatType.FLOAT32)

        self._configured = stack.enter_context(self._infer_model.configure())
        self._bindings = self._configured.create_bindings()

        if self._model_family == "yolo26":
            # Multi-output: 6 raw conv tensors (3 box reg + 3 cls per stride). Allocate a buffer per
            # output and bind by name so the SDK fills them on each run().
            self._output_names = [getattr(o, "name", None) for o in outputs]
            self._out_bufs = [np.zeros(tuple(o.shape), dtype=np.float32) for o in outputs]
            for name, buf in zip(self._output_names, self._out_bufs):
                if name is not None:
                    self._bindings.output(name).set_buffer(buf)
                else:
                    self._bindings.output().set_buffer(buf)
            self._yolo26_params = {
                "strides": list(metadata.get("strides", [8, 16, 32])),
                "reg_max": int(metadata.get("reg_max", 1)),
                "nc": int(metadata.get("nc", len(getattr(self, "names", {}) or {}) or 80)),
                "max_det": int(metadata.get("max_det", 300)),
                "conf": float(metadata.get("conf", 0.25)),
            }
        else:
            # yolov8 / yolo11: single-output (on-chip NMS).
            if len(outputs) != 1:
                stack.close()
                raise RuntimeError(
                    f"This HEF has {len(outputs)} outputs, but the yolov8 family decoder only handles "
                    f"single-output graphs produced by nms_postprocess(meta_arch=yolov8). If this is a "
                    f"YOLO26 HEF, ensure metadata.yaml is present with model_family: yolo26."
                )
            self._out_shape = tuple(outputs[0].shape)
            self._out_buf = np.zeros(self._out_shape, dtype=np.float32)
            self._bindings.output().set_buffer(self._out_buf)

        self._finalizer = weakref.finalize(self, stack.close)

    def forward(self, im: torch.Tensor) -> list[np.ndarray]:
        """Run synchronous inference on the Hailo accelerator.

        Args:
            im (torch.Tensor): Input tensor in NHWC layout (post AutoBackend permute). Float values in
                ``[0, 1]`` (predictor default) are silently cast back to ``uint8 [0, 255]`` to match the
                chip's normalization layer; native ``uint8`` input is forwarded unchanged.

        Returns:
            (list[np.ndarray]): Single-element list containing decoded predictions of shape ``(1, N, 6)``
            in ``[x1, y1, x2, y2, conf, cls]`` input-pixel coords.
        """
        if im.dtype == torch.uint8:
            x = im.cpu().numpy()
        else:
            x = (im.cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)

        x = np.ascontiguousarray(x)
        # im.shape is (B, H, W, C) post-permute.
        imgsz_h, imgsz_w = int(x.shape[1]), int(x.shape[2])

        self._bindings.input().set_buffer(x)
        self._configured.run([self._bindings], 10_000)  # 10 second timeout

        if self._model_family == "yolo26":
            return [self._yolo26_postprocess(self._out_bufs, imgsz_h, imgsz_w, **self._yolo26_params)]

        # yolov8 / yolo11: on-chip NMS, list[ndarray] of length num_classes.
        raw = self._bindings.output().get_buffer()
        if isinstance(raw, list) and raw and isinstance(raw[0], list):
            raw = raw[0]
        return [self._decode_nms(raw, imgsz_h, imgsz_w)]

    @staticmethod
    def _decode_nms(per_class: list, imgsz_h: int, imgsz_w: int) -> np.ndarray:
        """Convert the on-chip NMS-postprocess output to end2end ``(1, N, 6)`` predictions.

        Args:
            per_class (list): List of length ``num_classes``; element ``i`` is an ``(N_i, 5)`` array of
                ``[ymin, xmin, ymax, xmax, score]`` in normalized ``[0, 1]`` coords for class ``i``. Empty
                arrays / None entries indicate no detections for that class.
            imgsz_h (int): Network input height in pixels.
            imgsz_w (int): Network input width in pixels.

        Returns:
            (np.ndarray): ``(1, N, 6)`` float32 array of ``[x1, y1, x2, y2, conf, cls]`` in input-pixel
            coords, sorted by descending confidence.
        """
        rows = []
        for cls_idx, cls_dets in enumerate(per_class):
            if cls_dets is None or len(cls_dets) == 0:
                continue
            d = np.asarray(cls_dets, dtype=np.float32)
            x1 = d[:, 1] * imgsz_w
            y1 = d[:, 0] * imgsz_h
            x2 = d[:, 3] * imgsz_w
            y2 = d[:, 2] * imgsz_h
            scores = d[:, 4]
            cls = np.full_like(scores, cls_idx, dtype=np.float32)
            rows.append(np.stack([x1, y1, x2, y2, scores, cls], axis=1))
        if not rows:
            return np.zeros((1, 0, 6), dtype=np.float32)
        out = np.concatenate(rows, axis=0).astype(np.float32, copy=False)
        out = out[out[:, 4].argsort()[::-1]]  # sort by descending confidence
        return out[None, ...]

    @staticmethod
    def _yolo26_postprocess(
        out_bufs: list[np.ndarray],
        imgsz_h: int,
        imgsz_w: int,
        strides: list[int],
        reg_max: int,
        nc: int,
        max_det: int,
        conf: float,
    ) -> np.ndarray:
        """Host-side YOLO26 postprocess: anchors + dist2bbox + sigmoid + top-k.

        Mirrors ``Detect._inference`` followed by ``Detect.postprocess`` (for end2end models). YOLO26 has
        ``reg_max=1`` so DFL is ``nn.Identity`` — box conv outputs are direct 4-channel l/t/r/b distances
        in stride units. The xyxy box decode happens in stride units, then we scale by stride to get pixels.

        Args:
            out_bufs (list[np.ndarray]): The 6 raw conv output buffers (NHWC), arbitrary order. Box-reg
                heads have ``4 * reg_max`` channels; class heads have ``nc`` channels.
            imgsz_h (int): Network input height (used only for clamping; predictor handles letterbox unscale).
            imgsz_w (int): Network input width.
            strides (list[int]): Per-level strides, e.g. ``[8, 16, 32]``. Index 0 maps to the largest spatial
                feature map after sorting.
            reg_max (int): DFL ``reg_max`` (1 for YOLO26). Affects box channel count and decode math.
            nc (int): Number of classes.
            max_det (int): Top-k limit on (anchor, class) pairs.
            conf (float): Score threshold; predictions below this are dropped after top-k.

        Returns:
            (np.ndarray): ``(1, N, 6)`` float32 array of ``[x1, y1, x2, y2, conf, cls]``, descending score.
        """
        del imgsz_h, imgsz_w  # boxes are decoded in pixel space directly; predictor handles letterbox

        # HailoRT emits (H, W, C) for batch=1 outputs. Add a leading batch dim if missing.
        out_bufs = [b if b.ndim == 4 else b[None, ...] for b in out_bufs]

        box_ch = 4 * reg_max
        box_outs = [b for b in out_bufs if b.shape[-1] == box_ch]
        cls_outs = [c for c in out_bufs if c.shape[-1] == nc]
        if len(box_outs) != len(strides) or len(cls_outs) != len(strides):
            raise RuntimeError(
                f"YOLO26 host postprocess expects {len(strides)} box ({box_ch} ch) + {len(strides)} cls "
                f"({nc} ch) heads, got {len(box_outs)} / {len(cls_outs)}. Check metadata.yaml's reg_max / nc."
            )

        # Sort by spatial dim descending: largest H = stride[0] (8), smallest = stride[-1] (32).
        box_outs.sort(key=lambda b: -b.shape[1])
        cls_outs.sort(key=lambda b: -b.shape[1])

        all_boxes, all_scores = [], []
        for stride, box, cls in zip(strides, box_outs, cls_outs):
            B, H, W, _ = box.shape
            # Anchor centers in stride units: (x + 0.5, y + 0.5).
            ys = np.arange(H, dtype=np.float32) + 0.5
            xs = np.arange(W, dtype=np.float32) + 0.5
            gy, gx = np.meshgrid(ys, xs, indexing="ij")
            ax = gx.reshape(-1)  # (H*W,)
            ay = gy.reshape(-1)
            dist = box.reshape(B, -1, 4)
            l = dist[..., 0]
            t = dist[..., 1]
            r = dist[..., 2]
            b_ = dist[..., 3]
            x1 = (ax - l) * stride
            y1 = (ay - t) * stride
            x2 = (ax + r) * stride
            y2 = (ay + b_) * stride
            boxes_stride = np.stack([x1, y1, x2, y2], axis=-1)  # (B, H*W, 4)
            scores_stride = 1.0 / (1.0 + np.exp(-cls.reshape(B, -1, nc)))  # sigmoid
            all_boxes.append(boxes_stride)
            all_scores.append(scores_stride)

        boxes = np.concatenate(all_boxes, axis=1)  # (B, N_total, 4)
        scores = np.concatenate(all_scores, axis=1)  # (B, N_total, nc)

        # End2end top-k over (anchor × class) pairs, mirrors Detect.postprocess + get_topk_index.
        B, N, _ = boxes.shape
        flat_scores = scores.reshape(B, -1)  # (B, N*nc)
        k = min(max_det, flat_scores.shape[1])

        if k < flat_scores.shape[1]:
            part = np.argpartition(-flat_scores, k - 1, axis=1)[:, :k]
            unsorted = np.take_along_axis(flat_scores, part, axis=1)
            order = np.argsort(-unsorted, axis=1)
            topk_idx = np.take_along_axis(part, order, axis=1)
        else:
            topk_idx = np.argsort(-flat_scores, axis=1)

        topk_scores = np.take_along_axis(flat_scores, topk_idx, axis=1)  # (B, k)
        anchor_idx = topk_idx // nc  # (B, k)
        cls_idx = (topk_idx % nc).astype(np.float32)  # (B, k)
        batch_arange = np.arange(B)[:, None]
        selected_boxes = boxes[batch_arange, anchor_idx]  # (B, k, 4)

        # Drop predictions below conf threshold (per-batch; assemble (B, k_kept, 6)).
        out = np.concatenate([selected_boxes, topk_scores[..., None], cls_idx[..., None]], axis=-1)
        if conf > 0 and B == 1:
            mask = out[0, :, 4] > conf
            out = out[:, mask]
        return out.astype(np.float32, copy=False)
