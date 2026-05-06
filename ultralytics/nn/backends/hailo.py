# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import weakref
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import torch

from .base import BaseBackend


class HailoBackend(BaseBackend):
    """Hailo AI inference backend for Hailo-8/8L/10H/15H/15L accelerators.

    Loads compiled Hailo models (``.hef`` files) and runs inference using HailoRT (``hailo_platform``).

    The device's input layer normalizes by 255, so the on-device input contract is RGB ``uint8`` in ``[0, 255]``.
    The standard Ultralytics predict pipeline emits float32 ``[0, 1]`` tensors, which this backend silently
    casts back to uint8.

    Two HEF families are supported, dispatched by ``model_family`` in ``metadata.yaml``:

    - ``yolov8`` / ``yolo11`` — single-output HEF with on-device NMS (``nms_postprocess(meta_arch=yolov8)``).
      The device emits a per-class list of ``(N_class, 5)`` arrays which we just rescale to pixel coords.
    - ``yolo26`` — NMS-free end2end. Six raw conv outputs come off the device (3 strides × {box reg, cls});
      anchor decode + sigmoid + topk run on host because the NPU doesn't support the topk/gather ops.

    Both families produce Ultralytics' end2end ``(1, N, 6)`` format ``[x1, y1, x2, y2, conf, cls]`` for the
    predictor's end2end short path.

    Class attributes:
        INFER_TIMEOUT_MS: Per-frame inference timeout passed to ``ConfiguredInferModel.run``. Set
            generously to absorb first-call firmware warmup on smaller devices (Hailo-8L) at larger
            ``imgsz``. Override per-export via ``metadata.yaml``'s ``infer_timeout_ms`` key, or per-process
            by mutating the class attribute before instantiation.
    """

    INFER_TIMEOUT_MS: int = 60_000

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

        # Load metadata first — model_family determines whether we expect single-output (on-device NMS)
        # or multi-output (yolo26 raw heads, host-side postprocess).
        metadata: dict = {}
        metadata_file = (hef.parent if w.is_file() else w) / "metadata.yaml"
        if metadata_file.exists():
            from ultralytics.utils import YAML

            metadata = YAML.load(metadata_file) or {}
            self.apply_metadata(metadata)
        self._model_family = str(metadata.get("model_family", "yolov8"))
        self._max_det = int(metadata.get("max_det", 300))
        self._infer_timeout_ms = int(metadata.get("infer_timeout_ms", self.INFER_TIMEOUT_MS))

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
            if any(name is None for name in self._output_names):
                stack.close()
                raise RuntimeError(
                    "YOLO26 HEF has unnamed output(s); cannot bind multiple outputs unambiguously. "
                    f"output names = {self._output_names}"
                )
            for name, buf in zip(self._output_names, self._out_bufs):
                self._bindings.output(name).set_buffer(buf)
            # Validate the DFL-free assumption baked into _yolo26_postprocess: the box conv head emits
            # raw l/t/r/b distances in 4 channels (reg_max==1 → DFL is Identity). A non-1 reg_max would
            # require a softmax+matmul DFL step we don't implement.
            if int(metadata.get("reg_max", 1)) != 1:
                stack.close()
                raise NotImplementedError(
                    f"YOLO26 host postprocess requires reg_max=1 (DFL-free); got "
                    f"reg_max={metadata.get('reg_max')!r}. Custom DFL is not implemented."
                )
            nc = int(metadata.get("nc", len(getattr(self, "names", {}) or {}) or 80))
            strides = list(metadata.get("strides", [8, 16, 32]))
            # Pre-classify the 6 buffers into [box@s8, box@s16, box@s32] / [cls@s8, cls@s16, cls@s32]
            # using metadata persisted at compile time. Falls back to channel-count if metadata is
            # missing or fails to resolve at runtime.
            try:
                self._yolo26_box_bufs, self._yolo26_cls_bufs = self._classify_yolo26_buffers(
                    metadata.get("head_outputs") or {}, self._output_names, self._out_bufs, nc, strides
                )
            except RuntimeError:
                stack.close()
                raise
            self._yolo26_params = {
                "strides": strides,
                "nc": nc,
                "max_det": int(metadata.get("max_det", 300)),
                "conf": float(metadata.get("conf", 0.25)),
            }
        else:
            # yolov8 / yolo11: single-output (on-device NMS).
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

        # Hold the ExitStack so callers can release the VDevice (and its hardware lock) deterministically
        # via close(). weakref.finalize is registered as a fallback for processes that don't call close()
        # explicitly — but it relies on GC, which can be deferred under circular refs / GIL contention,
        # leaving the device locked for other processes.
        self._exit_stack = stack
        self._finalizer = weakref.finalize(self, stack.close)

    def close(self) -> None:
        """Release the HailoRT VDevice and free the hardware lock.

        Safe to call multiple times. After ``close()``, ``forward()`` will raise.
        """
        finalizer = getattr(self, "_finalizer", None)
        if finalizer is not None and finalizer.alive:
            finalizer()  # invokes stack.close(); detaches itself

    def __del__(self) -> None:
        """Best-effort teardown when the backend goes out of scope.

        The explicit ``close()`` is preferred — ``__del__`` is unreliable under circular references.
        """
        try:
            self.close()
        except Exception:
            pass

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
        self._configured.run([self._bindings], self._infer_timeout_ms)

        if self._model_family == "yolo26":
            return [
                self._yolo26_postprocess(
                    self._yolo26_box_bufs, self._yolo26_cls_bufs, **self._yolo26_params
                )
            ]

        # yolov8 / yolo11: on-device NMS, list[ndarray] of length num_classes.
        raw = self._bindings.output().get_buffer()
        if isinstance(raw, list) and raw and isinstance(raw[0], list):
            raw = raw[0]
        return [self._decode_nms(raw, imgsz_h, imgsz_w, self._max_det)]

    @staticmethod
    def _classify_yolo26_buffers(
        head_outputs: dict,
        output_names: list,
        out_bufs: list[np.ndarray],
        nc: int,
        strides: list[int],
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Pre-classify the 6 YOLO26 output buffers into per-stride box and class lists.

        Prefers the ``head_outputs`` metadata persisted at compile time — names + spatials in stride
        order — to avoid the channel-count ambiguity that breaks for ``nc == 4`` custom datasets
        (where every leaf has 4 channels). Falls back to channel-count + spatial-dim sort if the
        metadata is missing or names cannot be resolved (older HEFs / SDK rename).

        Args:
            head_outputs (dict): ``{"box_layers": [{"name", "spatial"}, ...], "cls_layers": [...]}``
                in stride order, or ``{}`` to force the channel-count fallback.
            output_names (list): Output names from ``infer_model.outputs[i].name`` (runtime).
            out_bufs (list[np.ndarray]): Output buffers in the same order as ``output_names``.
            nc (int): Number of classes (used by the channel-count fallback).
            strides (list[int]): Per-level strides; length is the expected per-role count.

        Returns:
            (tuple[list[np.ndarray], list[np.ndarray]]): ``(box_bufs, cls_bufs)``, each of length
            ``len(strides)``, ordered by stride (largest spatial → smallest).

        Raises:
            RuntimeError: If neither name dispatch nor the channel-count fallback can resolve all
                ``2 * len(strides)`` buffers unambiguously.
        """
        n = len(strides)

        def _resolve_by_name(entries: list) -> list[np.ndarray] | None:
            """Return buffers in entry order, or None if any entry can't be matched."""
            name_to_buf = dict(zip(output_names, out_bufs))
            picked: list[np.ndarray] = []
            for entry in entries:
                want = entry.get("name")
                if want and want in name_to_buf:
                    picked.append(name_to_buf[want])
                    continue
                # Substring match — runtime SDK may prefix / suffix the HN layer name.
                hits = [b for nm, b in name_to_buf.items() if want and (want in nm or nm in want)]
                if len(hits) == 1:
                    picked.append(hits[0])
                    continue
                return None
            return picked

        if head_outputs and len(head_outputs.get("box_layers", [])) == n and len(head_outputs.get("cls_layers", [])) == n:
            box = _resolve_by_name(head_outputs["box_layers"])
            cls = _resolve_by_name(head_outputs["cls_layers"])
            if box is not None and cls is not None:
                return box, cls

        # Fallback: channel-count + spatial sort. Ambiguous when nc == 4; warn the user.
        bufs_4d = [b if b.ndim == 4 else b[None, ...] for b in out_bufs]
        if nc == 4:
            from ultralytics.utils import LOGGER

            LOGGER.warning(
                "Hailo: YOLO26 head_outputs metadata missing/unresolved with nc==4 — channel counts "
                "alone cannot disambiguate box vs cls heads. Re-export with the current Ultralytics "
                "version to embed head_outputs."
            )
        box_outs = [b for b in bufs_4d if b.shape[-1] == 4]
        cls_outs = [b for b in bufs_4d if b.shape[-1] == nc]
        if len(box_outs) != n or len(cls_outs) != n:
            raise RuntimeError(
                f"YOLO26 fallback dispatch expects {n} box (4 ch) + {n} cls ({nc} ch) heads, got "
                f"{len(box_outs)} / {len(cls_outs)}. Check metadata.yaml's nc, or re-export to embed "
                f"head_outputs."
            )
        box_outs.sort(key=lambda b: -b.shape[1])
        cls_outs.sort(key=lambda b: -b.shape[1])
        return box_outs, cls_outs

    @staticmethod
    def _decode_nms(per_class: list, imgsz_h: int, imgsz_w: int, max_det: int) -> np.ndarray:
        """Convert the on-device NMS-postprocess output to end2end ``(1, N, 6)`` predictions.

        Args:
            per_class (list): List of length ``num_classes``; element ``i`` is an ``(N_i, 5)`` array of
                ``[ymin, xmin, ymax, xmax, score]`` in normalized ``[0, 1]`` coords for class ``i``. Empty
                arrays / None entries indicate no detections for that class.
            imgsz_h (int): Network input height in pixels.
            imgsz_w (int): Network input width in pixels.
            max_det (int): Cap on total detections returned. The on-device NMS allows up to
                ``nc * max_proposals_per_class`` outputs (24 000 for COCO defaults); we trim here for
                symmetry with the YOLO26 path and downstream pipeline efficiency.

        Returns:
            (np.ndarray): ``(1, N, 6)`` float32 array of ``[x1, y1, x2, y2, conf, cls]`` in input-pixel
            coords, sorted by descending confidence and capped at ``max_det``.
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
        if max_det > 0 and out.shape[0] > max_det:
            out = out[:max_det]
        return out[None, ...]

    @staticmethod
    def _yolo26_postprocess(
        box_bufs: list[np.ndarray],
        cls_bufs: list[np.ndarray],
        strides: list[int],
        nc: int,
        max_det: int,
        conf: float,
    ) -> np.ndarray:
        """Host-side YOLO26 postprocess: anchors + dist2bbox + sigmoid + top-k.

        Mirrors ``Detect._inference`` followed by ``Detect.postprocess`` (for end2end models). YOLO26 has
        ``reg_max=1`` so DFL is ``nn.Identity`` — box conv outputs are direct 4-channel l/t/r/b distances
        in stride units. The xyxy box decode happens in stride units, then we scale by stride to get pixels.
        ``reg_max`` is hardcoded to 1 here (validated in ``load_model``); a real DFL would need a
        ``softmax(reg_max) @ arange(reg_max)`` step before dist2bbox.

        Args:
            box_bufs (list[np.ndarray]): Per-stride box-reg conv outputs (NHWC, 4 channels), in stride
                order matching ``strides``. Pre-classified in ``load_model`` from
                ``metadata['head_outputs']``.
            cls_bufs (list[np.ndarray]): Per-stride class conv outputs (NHWC, ``nc`` channels), in stride
                order matching ``strides``.
            strides (list[int]): Per-level strides, e.g. ``[8, 16, 32]``. Index 0 maps to the largest
                spatial feature map.
            nc (int): Number of classes.
            max_det (int): Top-k limit on (anchor, class) pairs.
            conf (float): Score threshold; predictions below this are dropped after top-k.

        Returns:
            (np.ndarray): ``(1, N, 6)`` float32 array of ``[x1, y1, x2, y2, conf, cls]``, descending score.
        """
        # HailoRT emits (H, W, C) for batch=1 outputs. Add a leading batch dim if missing. The whole
        # postprocess (anchor sharing, conf filter, predictor's (1, N, 6) contract) is wired for B=1 —
        # if a multi-batch HEF ever reaches us, fail loudly rather than produce wrong results below.
        box_bufs = [b if b.ndim == 4 else b[None, ...] for b in box_bufs]
        cls_bufs = [c if c.ndim == 4 else c[None, ...] for c in cls_bufs]
        for buf in (*box_bufs, *cls_bufs):
            if buf.shape[0] != 1:
                raise NotImplementedError(
                    f"YOLO26 host postprocess only supports batch=1, got buffer shape {buf.shape}. "
                    f"Multi-batch decode is not implemented."
                )
        if len(box_bufs) != len(strides) or len(cls_bufs) != len(strides):
            raise RuntimeError(
                f"YOLO26 host postprocess expected {len(strides)} box / {len(strides)} cls buffers, "
                f"got {len(box_bufs)} / {len(cls_bufs)}."
            )

        all_boxes, all_scores = [], []
        for stride, box, cls in zip(strides, box_bufs, cls_bufs):
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
            # argpartition with a negative kth selects the top-k in unsorted order without the
            # full-array negation that `-flat_scores` would copy.
            part = np.argpartition(flat_scores, -k, axis=1)[:, -k:]
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

        # Drop predictions below conf threshold. B is guaranteed to be 1 (asserted above), so we can
        # filter unconditionally and produce the (1, k_kept, 6) the predictor expects.
        out = np.concatenate([selected_boxes, topk_scores[..., None], cls_idx[..., None]], axis=-1)
        if conf > 0:
            mask = out[0, :, 4] > conf
            out = out[:, mask]
        return out.astype(np.float32, copy=False)
