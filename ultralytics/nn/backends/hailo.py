# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import weakref
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import torch

from ultralytics.utils import LOGGER

from . import _hailo_yolo26
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
        from ultralytics.utils.export.hailo import SOC_DEVICES

        w = Path(weight)
        hef = w if w.is_file() and w.suffix == ".hef" else next(w.rglob("*.hef"), None)
        if hef is None or not hef.exists():
            raise FileNotFoundError(f"No .hef file found at/under: {w}")

        # Load metadata first — model_family determines whether we expect single-output (on-device NMS)
        # or multi-output (yolo26 raw heads, host-side postprocess). Hailo-private fields are stripped
        # before apply_metadata so they don't end up as random public attributes on the backend (the
        # base impl does setattr(self, k, v) for every metadata key).
        metadata: dict = {}
        hailo_meta: dict = {}
        metadata_file = (hef.parent if w.is_file() else w) / "metadata.yaml"
        if metadata_file.exists():
            from ultralytics.utils import YAML

            raw = YAML.load(metadata_file) or {}
            hailo_keys = {
                "model_family",
                "hailo_mz_tag",
                "hw_arch",
                "head_outputs",
                "infer_timeout_ms",
                "strides",
                "reg_max",
                "nc",
                "max_det",
                "conf",
            }
            hailo_meta = {k: raw[k] for k in hailo_keys if k in raw}
            metadata = {k: v for k, v in raw.items() if k not in hailo_keys}
            self.apply_metadata(metadata)

        # Hailo-15H / 15L are SoC targets — the HEF is deployed onto the device and executed there.
        # HailoRT VDevice on a host cannot drive a SoC, so reject up-front (before importing
        # hailo_platform) with a message that points the user at the right deployment path.
        hw_arch = hailo_meta.get("hw_arch")
        if hw_arch in SOC_DEVICES:
            raise NotImplementedError(
                f"This HEF was compiled for hw_arch={hw_arch!r}, a Hailo SoC target "
                f"({sorted(SOC_DEVICES)}). SoC devices do not support host-side inference via "
                f"HailoRT VDevice — the HEF must be deployed onto the device and executed there. "
                f"Host-side inference via Ultralytics is only supported for Hailo accelerators "
                f"(hailo8 / hailo8l / hailo10h)."
            )

        try:
            from hailo_platform import FormatType, VDevice
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

        self._hailo_meta = hailo_meta
        self._model_family = str(hailo_meta.get("model_family", "yolov8"))
        self._max_det = int(hailo_meta.get("max_det", 300))
        self._infer_timeout_ms = int(hailo_meta.get("infer_timeout_ms", self.INFER_TIMEOUT_MS))

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
            if int(hailo_meta.get("reg_max", 1)) != 1:
                stack.close()
                raise NotImplementedError(
                    f"YOLO26 host postprocess requires reg_max=1 (DFL-free); got "
                    f"reg_max={hailo_meta.get('reg_max')!r}. Custom DFL is not implemented."
                )
            nc = int(hailo_meta.get("nc", len(getattr(self, "names", {}) or {}) or 80))
            strides = list(hailo_meta.get("strides", [8, 16, 32]))
            # Pre-classify the 6 buffers into [box@s8, box@s16, box@s32] / [cls@s8, cls@s16, cls@s32]
            # using metadata persisted at compile time. Falls back to channel-count if metadata is
            # missing or fails to resolve at runtime.
            try:
                self._yolo26_box_bufs, self._yolo26_cls_bufs = _hailo_yolo26.classify_buffers(
                    hailo_meta.get("head_outputs") or {}, self._output_names, self._out_bufs, nc, strides
                )
            except RuntimeError:
                stack.close()
                raise
            self._yolo26_params = {
                "strides": strides,
                "nc": nc,
                "max_det": int(hailo_meta.get("max_det", 300)),
                "conf": float(hailo_meta.get("conf", 0.25)),
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
            # On-device NMS bakes nms_scores_th / nms_iou_th into the HEF at compile time. Surface that
            # contract once at load so users don't silently lose runtime predict(conf=, iou=) overrides.
            LOGGER.info(
                "Hailo: this HEF runs NMS on-device — the conf/iou thresholds were baked at export. "
                "Runtime predict(conf=, iou=) overrides are ignored. Re-export with the desired "
                "thresholds to change them."
            )

        # Hold the ExitStack so callers can release the VDevice (and its hardware lock) deterministically
        # via close(). weakref.finalize is registered as a fallback for processes that don't call close()
        # explicitly — but it relies on GC, which can be deferred under circular refs / GIL contention,
        # leaving the device locked for other processes.
        self._exit_stack = stack
        self._finalizer = weakref.finalize(self, stack.close)

    def close(self) -> None:
        """Release the HailoRT VDevice and free the hardware lock.

        Safe to call multiple times. After ``close()``, ``forward()`` will raise. The
        ``weakref.finalize`` registered in ``load_model`` calls this same callback on GC, so callers
        can rely on either explicit ``close()`` or refcount-driven cleanup.
        """
        finalizer = getattr(self, "_finalizer", None)
        if finalizer is not None and finalizer.alive:
            finalizer()  # invokes stack.close(); detaches itself

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
            x = np.ascontiguousarray(im.cpu().numpy())
        else:
            # astype already returns a fresh contiguous array; no extra copy needed.
            x = (im.cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)

        # im.shape is (B, H, W, C) post-permute. Both decode paths emit (1, N, 6); _yolo26_postprocess
        # asserts B==1 internally, but _decode_nms ignores the batch dim, so guard here.
        if x.shape[0] != 1:
            raise NotImplementedError(
                f"HailoBackend currently only supports batch=1 inference, got input shape {x.shape}."
            )
        imgsz_h, imgsz_w = int(x.shape[1]), int(x.shape[2])

        self._bindings.input().set_buffer(x)
        self._configured.run([self._bindings], self._infer_timeout_ms)

        if self._model_family == "yolo26":
            return [
                _hailo_yolo26.postprocess(
                    self._yolo26_box_bufs, self._yolo26_cls_bufs, **self._yolo26_params
                )
            ]

        # yolov8 / yolo11: on-device NMS, list[ndarray] of length num_classes.
        raw = self._bindings.output().get_buffer()
        if isinstance(raw, list) and raw and isinstance(raw[0], list):
            raw = raw[0]
        return [self._decode_nms(raw, imgsz_h, imgsz_w, self._max_det)]

    # YOLO26-specific dispatch + decode lives in ``_hailo_yolo26``; the static-method shim is kept so
    # tests/external callers can reference ``HailoBackend._classify_yolo26_buffers`` and
    # ``HailoBackend._yolo26_postprocess`` without having to know the layout.
    _classify_yolo26_buffers = staticmethod(_hailo_yolo26.classify_buffers)

    @staticmethod
    def _decode_nms(per_class: list, imgsz_h: int, imgsz_w: int, max_det: int) -> np.ndarray:
        """Convert the on-device NMS-postprocess output to end2end ``(1, N, 6)`` predictions.

        Contract: the on-device ``nms_postprocess(meta_arch=yolov8)`` op emits classes in the same
        index order as the network's class head, which Ultralytics persists in ``model.names``.
        Position ``i`` in ``per_class`` is therefore class ``i`` in ``model.names`` — the chip and
        the host agree on this ordering implicitly via the trained weights.

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

    _yolo26_postprocess = staticmethod(_hailo_yolo26.postprocess)
