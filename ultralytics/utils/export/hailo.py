# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ultralytics.utils import LOGGER, YAML


def _resolve_model_script(
    model_script: str | Path | None,
    task: str,
    conf: float,
    iou: float,
    max_det: int,
    num_classes: int,
) -> str:
    """Return the Hailo model script ('alls') content to load into the compiler.

    Args:
        model_script (str | Path | None): User override. Path to an alls file, raw alls content,
            or None to auto-generate.
        task (str): Ultralytics task (only ``"detect"`` is supported in this initial release).
        conf (float): Score threshold for the on-chip NMS layer.
        iou (float): IoU threshold for the on-chip NMS layer.
        max_det (int): Max proposals per class for the on-chip NMS layer.
        num_classes (int): Number of object classes.

    Returns:
        (str): The alls script content.
    """
    if model_script is not None:
        s = str(model_script)
        p = Path(s)
        if p.is_file():
            return p.read_text()
        # Heuristic: alls scripts contain function calls (with newlines / parens). A short single-line value
        # without parens is almost certainly a path the user meant to point at — fail loudly rather than
        # ship the literal string to the compiler as alls content.
        if "\n" in s or "(" in s:
            return s
        raise FileNotFoundError(
            f"model_script={model_script!r} is not a file and does not look like alls content. "
            f"Pass either a path to an existing .alls file or raw alls script content."
        )

    if task != "detect":
        raise NotImplementedError(
            f"Hailo export currently only auto-generates a model script for task='detect', got task='{task}'. "
            f"Pass a custom alls via model_script= to compile other tasks."
        )

    optimization = "model_optimization_flavor(optimization_level=2, compression_level=0)"
    # The chip divides input by 255 internally, so the calibration / inference API expects RGB uint8 [0, 255].
    normalization = "normalization1 = normalization([0.0, 0.0, 0.0], [255.0, 255.0, 255.0])"
    nms_postprocess = (
        'nms_postprocess(meta_arch="yolov8", engine="cpu", '
        f"nms_scores_th={float(conf)}, nms_iou_th={float(iou)}, "
        f"classes={int(num_classes)}, max_proposals_per_class={int(max_det)})"
    )
    return f"{optimization}\n{normalization}\n{nms_postprocess}\n"


def _dataloader_to_numpy(dataloader) -> np.ndarray:
    """Stack a YOLO calibration dataloader into a single (N, H, W, C) uint8 RGB array.

    Hailo's on-chip normalization layer expects raw RGB uint8 frames in [0, 255]; we therefore
    keep the dataloader's native uint8 tensors and just permute BCHW → BHWC.
    """
    chunks: list[np.ndarray] = []
    for batch in dataloader:
        img = batch["img"] if isinstance(batch, dict) else batch
        if not isinstance(img, torch.Tensor):
            raise TypeError(f"Calibration batch 'img' must be a torch.Tensor, got {type(img)}.")
        if img.dtype != torch.uint8:
            raise TypeError(
                f"Calibration batch 'img' dtype must be torch.uint8 (raw RGB [0, 255]) for Hailo, "
                f"got {img.dtype}."
            )
        chunks.append(img.permute(0, 2, 3, 1).contiguous().numpy())
    if not chunks:
        raise ValueError("Calibration dataloader yielded no batches.")
    return np.concatenate(chunks, axis=0)


def onnx2hailo(
    onnx_file: str,
    output_dir: Path | str,
    hw_arch: str = "hailo10h",
    task: str = "detect",
    calibration_data: np.ndarray | None = None,
    model_script: str | Path | None = None,
    conf: float = 0.25,
    iou: float = 0.7,
    max_det: int = 300,
    num_classes: int = 80,
    metadata: dict | None = None,
    model_name: str = "model",
    head_module_name: str | None = None,
    prefix: str = "",
) -> str:
    """Compile an ONNX YOLO model to a Hailo HEF using the Hailo Dataflow Compiler.

    Supports YOLOv8 and YOLOv11 detect heads directly via the ``nms_postprocess(meta_arch="yolov8")``
    macro, which adds on-chip NMS to the compiled HEF.

    Args:
        onnx_file (str): Path to the source ONNX file.
        output_dir (Path | str): Directory to write the compiled ``<model_name>.hef`` and metadata.
        hw_arch (str): Hailo hardware target (one of ``HAILO_CHIPS``).
        task (str): Ultralytics task. Only ``"detect"`` is supported in this release.
        calibration_data (np.ndarray): (N, H, W, C) RGB uint8 calibration array [0-255].
        model_script (str | Path | None): Optional alls override (file path or raw alls content).
        conf (float): NMS score threshold.
        iou (float): NMS IoU threshold.
        max_det (int): NMS max proposals per class.
        num_classes (int): Number of object classes.
        metadata (dict | None): Metadata to persist alongside the HEF as ``metadata.yaml``.
        model_name (str): Name of the compiled HEF (without extension).
        head_module_name (str | None): Detect head module name (e.g. ``"model.22"``). When set, the parser is
            cut at the head's ``Sigmoid`` and ``Concat`` leaves so ``nms_postprocess(meta_arch="yolov8")``
            attaches cleanly, avoiding PyTorch-version-specific shape ops in the DFL/decode subgraph.
        prefix (str): Prefix for log messages.

    Returns:
        (str): Path to the produced ``_hailo_model`` directory.

    Raises:
        ImportError: If the Hailo Dataflow Compiler is not installed.
        ValueError: If calibration data is missing.
    """
    if calibration_data is None or len(calibration_data) == 0:
        raise ValueError("Calibration data is required for Hailo quantization.")
    if calibration_data.dtype != np.uint8:
        LOGGER.warning(
            f"{prefix} calibration_data dtype is {calibration_data.dtype} — Hailo expects RGB uint8 in [0, 255]. "
            f"On-chip normalization divides by 255; passing pre-normalized [0, 1] data will severely degrade accuracy."
        )
    elif calibration_data.max() <= 1:
        LOGGER.warning(
            f"{prefix} calibration_data max value <= 1 — looks pre-normalized. Hailo expects raw RGB uint8 frames "
            f"in [0, 255]; the chip normalizes by 255 internally."
        )

    try:
        from hailo_sdk_client import ClientRunner
    except ImportError as e:
        raise ImportError(
            "Hailo Dataflow Compiler ('hailo_sdk_client') is required for Hailo export but is not installed. "
            "Install it from the Hailo Developer Zone: https://hailo.ai/developer-zone/"
        ) from e

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info(f"\n{prefix} starting export with Hailo Dataflow Compiler (hw_arch={hw_arch})...")

    runner = ClientRunner(hw_arch=hw_arch)
    translate_kwargs: dict = {}
    if head_module_name:
        translate_kwargs["end_node_names"] = [
            f"/{head_module_name}/Sigmoid",
            f"/{head_module_name}/Concat",
        ]
    runner.translate_onnx_model(onnx_file, model_name, **translate_kwargs)

    alls = _resolve_model_script(model_script, task, conf, iou, max_det, num_classes)
    runner.load_model_script(alls)
    runner.optimize(calibration_data)

    hef_bytes = runner.compile()
    hef_path = output_dir / f"{model_name}.hef"
    hef_path.write_bytes(hef_bytes)

    if metadata is not None:
        YAML.save(output_dir / "metadata.yaml", metadata)

    return str(output_dir)
