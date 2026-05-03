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

    HEFs are produced by the Ultralytics exporter with ``nms_postprocess(meta_arch="yolov8")`` for the
    YOLOv8 and YOLOv11 detect heads. On-chip NMS emits per-class detection lists of shape
    ``(num_classes, max_proposals_per_class, 5)``, which this backend decodes into Ultralytics'
    end-to-end ``(1, N, 6)`` format ``[x1, y1, x2, y2, conf, cls]`` in input-pixel coords, so the
    existing predictor end2end short path consumes them directly.
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
                "Download HailoRT (free Hailo Developer Zone account required) from:\n"
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

        stack = ExitStack()
        vdevice = stack.enter_context(VDevice())
        self._infer_model = vdevice.create_infer_model(str(hef))
        self._configured = stack.enter_context(self._infer_model.configure())
        self._bindings = self._configured.create_bindings()

        outputs = self._infer_model.outputs
        outputs = outputs() if callable(outputs) else outputs
        if len(outputs) != 1:
            stack.close()
            raise RuntimeError(
                f"This HEF has {len(outputs)} outputs, but HailoBackend's decoder only handles single-output "
                f"graphs produced by the auto-export path (post-NMS detection lists)."
            )
        self._out_shape = tuple(outputs[0].shape)
        self._out_buf = np.zeros(self._out_shape, dtype=np.float32)
        self._bindings.output().set_buffer(self._out_buf)
        self._finalizer = weakref.finalize(self, stack.close)

        metadata_file = (hef.parent if w.is_file() else w) / "metadata.yaml"
        if metadata_file.exists():
            from ultralytics.utils import YAML

            self.apply_metadata(YAML.load(metadata_file))

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
        raw = self._bindings.output().get_buffer()

        # On-chip nms_postprocess(meta_arch="yolov8") emits (num_classes, max_proposals_per_class, 5):
        # [ymin, xmin, ymax, xmax, score] in normalized [0, 1].
        if raw.ndim == 3 and raw.shape[-1] == 5:
            return [self._decode_nms(raw, imgsz_h, imgsz_w)]
        raise RuntimeError(
            f"Unrecognized HEF output shape {raw.shape} (dtype={raw.dtype}). HailoBackend's decoder handles "
            f"only on-chip NMS output of shape (num_classes, max_proposals, 5)."
        )

    @staticmethod
    def _decode_nms(buf: np.ndarray, imgsz_h: int, imgsz_w: int) -> np.ndarray:
        """Convert the on-chip NMS-postprocess output to end2end ``(1, N, 6)`` predictions.

        Args:
            buf (np.ndarray): Hailo NMS buffer of shape ``(num_classes, max_proposals_per_class, 5)`` with
                ``[ymin, xmin, ymax, xmax, score]`` in normalized ``[0, 1]`` coords. Unused proposal slots
                are zero-padded.
            imgsz_h (int): Network input height in pixels.
            imgsz_w (int): Network input width in pixels.

        Returns:
            (np.ndarray): ``(1, N, 6)`` float32 array of ``[x1, y1, x2, y2, conf, cls]`` in input-pixel
            coords, sorted by descending confidence.
        """
        nc = buf.shape[0]
        rows = []
        for cls_idx in range(nc):
            cls_dets = buf[cls_idx]
            valid = cls_dets[:, 4] > 0  # zero score means empty slot
            if not valid.any():
                continue
            d = cls_dets[valid]
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
