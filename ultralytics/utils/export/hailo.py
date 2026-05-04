# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from ultralytics.utils import LOGGER, YAML
from ultralytics.utils.downloads import safe_download

# Hailo Model Zoo URL pattern for the YOLO26 quantization scripts. We fetch these per variant rather than
# auto-generate, because the MZ alls is tuned for accuracy parity (mixed-precision a16_w16 layers, adaround,
# higher optimization level than our default).
HAILO_MZ_YOLO26_ALLS_URL = (
    "https://raw.githubusercontent.com/hailo-ai/hailo_model_zoo/master/hailo_model_zoo/cfg/alls/generic/"
    "yolo26{scale}.alls"
)


def _resolve_model_script(
    model_script: str | Path | None,
    task: str,
    nms_config_path: str | Path | None,
) -> str:
    """Return the Hailo model script ('alls') content to load into the compiler.

    Args:
        model_script (str | Path | None): User override or family-specific alls path. If a file path,
            read its content; if it looks like alls content (multi-line / contains ``(``), pass through.
        task (str): Ultralytics task (only ``"detect"`` is supported in this initial release).
        nms_config_path (str | Path | None): Path to a ``meta_arch=yolov8`` NMS-postprocess JSON config.
            Used only when ``model_script`` is None (yolov8 / yolo11 path); ignored otherwise.

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
    nms_postprocess = f'nms_postprocess("{nms_config_path}", meta_arch=yolov8, engine=cpu)'
    return f"{optimization}\n{normalization}\n{nms_postprocess}\n"


def _build_nms_config(
    runner,
    conf: float,
    iou: float,
    num_classes: int,
    imgsz: tuple[int, int],
    max_det: int,
    reg_max: int,
) -> dict:
    """Build the ``meta_arch=yolov8`` JSON config for ``nms_postprocess`` from the parsed HailoNN.

    Inspects the HailoNN that came out of ``translate_onnx_model`` to find the 6 conv leaves of the Detect
    head (3 strides x {box reg, class logits}), classifies each by output channel count, sorts by spatial
    dim to recover stride order, and emits a config dict the SDK expects.

    Args:
        runner: HailoSDK ``ClientRunner`` after ``translate_onnx_model`` has run.
        conf (float): NMS score threshold.
        iou (float): NMS IoU threshold.
        num_classes (int): Number of object classes (matches cls conv channel count).
        imgsz (tuple[int, int]): Model input ``(height, width)``.
        max_det (int): Max proposals per class for on-chip NMS.
        reg_max (int): DFL ``reg_max`` (default 16 for YOLOv8). Box reg conv channel count is ``4 * reg_max``.

    Returns:
        (dict): JSON-serializable config dict ready to pass to ``nms_postprocess``.

    Raises:
        RuntimeError: If 3 box-reg and 3 class conv leaves cannot be located in the parsed HailoNN.
    """
    hn = runner.get_hn_model()
    reg_channels = 4 * reg_max
    reg_layers, cls_layers = [], []
    for output in hn.get_output_layers():
        for pred in hn.predecessors(output):
            channels = pred.output_shapes[0][-1]  # NHWC
            if channels == reg_channels:
                reg_layers.append(pred)
            elif channels == num_classes:
                cls_layers.append(pred)

    if len(reg_layers) != 3 or len(cls_layers) != 3:
        raise RuntimeError(
            f"Expected 3 box-reg ({reg_channels} ch) + 3 cls ({num_classes} ch) conv leaves at HailoNN "
            f"outputs, found {len(reg_layers)} reg / {len(cls_layers)} cls. Cannot auto-build the "
            f"nms_postprocess JSON config; pass a custom alls via model_script=."
        )

    # Sort by spatial dim descending: largest = stride 8, then 16, then 32
    reg_layers.sort(key=lambda layer: -layer.output_shapes[0][1])
    cls_layers.sort(key=lambda layer: -layer.output_shapes[0][1])

    strides = (8, 16, 32)
    bbox_decoders = [
        {
            "name": f"bbox_decoder{i}",
            "stride": stride,
            "reg_layer": reg.name,
            "cls_layer": cls.name,
        }
        for i, (stride, reg, cls) in enumerate(zip(strides, reg_layers, cls_layers))
    ]

    return {
        "nms_scores_th": float(conf),
        "nms_iou_th": float(iou),
        "image_dims": [int(imgsz[0]), int(imgsz[1])],
        "max_proposals_per_class": int(max_det),
        "classes": int(num_classes),
        "regression_length": int(reg_max),
        "background_removal": False,
        "bbox_decoders": bbox_decoders,
    }


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
    num_classes: int = 80,
    imgsz: tuple[int, int] = (640, 640),
    max_det: int = 300,
    reg_max: int = 16,
    metadata: dict | None = None,
    model_name: str = "model",
    model_family: str = "yolov8",
    scale: str = "",
    end_node_names: list[str] | None = None,
    prefix: str = "",
) -> str:
    """Compile an ONNX YOLO model to a Hailo HEF using the Hailo Dataflow Compiler.

    Supports two model families with different on-chip / host-side splits:

    - ``yolov8`` / ``yolo11`` detect heads — uses on-chip NMS via ``nms_postprocess(meta_arch=yolov8)``.
      A JSON config is auto-built from the parsed HailoNN and emitted alongside the HEF.
    - ``yolo26`` detect heads — NMS-free end2end. The Hailo NPU does not support topk/gather, so the
      postprocess runs on host (see ``HailoBackend``). The alls is fetched from the Hailo Model Zoo
      (``cfg/alls/generic/yolo26{scale}.alls``) for accuracy parity, no on-chip NMS attached.

    Args:
        onnx_file (str): Path to the source ONNX file.
        output_dir (Path | str): Directory to write the compiled ``<model_name>.hef`` and metadata.
        hw_arch (str): Hailo hardware target (one of ``HAILO_CHIPS``).
        task (str): Ultralytics task. Only ``"detect"`` is supported in this release.
        calibration_data (np.ndarray): (N, H, W, C) RGB uint8 calibration array [0-255].
        model_script (str | Path | None): Optional alls override (file path or raw alls content). When
            provided, family-specific alls fetching and NMS JSON generation are skipped entirely.
        conf (float): NMS score threshold (``nms_scores_th``).
        iou (float): NMS IoU threshold (``nms_iou_th``).
        num_classes (int): Number of object classes.
        imgsz (tuple[int, int]): Model input ``(height, width)``.
        max_det (int): Max proposals per class for on-chip NMS / max detections for host postprocess.
        reg_max (int): DFL ``reg_max`` (16 for YOLOv8, 1 for YOLO26).
        metadata (dict | None): Metadata to persist alongside the HEF as ``metadata.yaml``. The function
            adds family-specific postprocess params (``model_family``, ``strides``, ``reg_max``, etc.) so
            ``HailoBackend`` can configure host-side decode at load time.
        model_name (str): Name of the compiled HEF (without extension).
        model_family (str): ``"yolov8"`` (default) or ``"yolo26"``. Selects the alls + decode path.
        scale (str): YOLO26 scale letter — one of ``n`` / ``s`` / ``m`` / ``l`` (Hailo Model Zoo does not
            publish an alls for ``x``). Required when ``model_family == "yolo26"`` and
            ``model_script`` is None — used to fetch the matching alls from the Hailo Model Zoo.
        end_node_names (list[str] | None): ONNX end-node names to cut the graph at. The caller computes
            these because the conv-leaf prefix is family-specific (``cv2``/``cv3`` for yolov8,
            ``one2one_cv2``/``one2one_cv3`` for yolo26).
        prefix (str): Prefix for log messages.

    Returns:
        (str): Path to the produced ``_hailo_model`` directory.

    Raises:
        ImportError: If the Hailo Dataflow Compiler is not installed.
        ValueError: If calibration data is missing or yolo26 scale is missing.
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
            "Hailo Dataflow Compiler ('hailo_sdk_client') is required for Hailo export but is not installed.\n"
            "Download the DFC installer from:\n"
            "  https://hailo.ai/developer-zone/software-downloads/\n"
            "and follow the install instructions in the bundled user guide. The DFC is not on PyPI; manual "
            "install is required. Once installed, re-run the export."
        ) from e

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info(
        f"\n{prefix} starting export with Hailo Dataflow Compiler "
        f"(hw_arch={hw_arch}, model_family={model_family})..."
    )

    runner = ClientRunner(hw_arch=hw_arch)
    translate_kwargs: dict = {}
    if end_node_names:
        translate_kwargs["end_node_names"] = list(end_node_names)
    runner.translate_onnx_model(onnx_file, model_name, **translate_kwargs)

    nms_config_path: Path | None = None
    resolved_model_script: str | Path | None = model_script

    if model_script is None and task == "detect":
        if model_family == "yolo26":
            # YOLO26: fetch the variant-specific alls from Hailo Model Zoo (no on-chip NMS, host postprocess).
            if not scale:
                raise ValueError(
                    "model_family='yolo26' requires scale (n/s/m/l) to fetch the matching alls. "
                    "Pass a custom alls via model_script= for non-standard variants."
                )
            alls_url = HAILO_MZ_YOLO26_ALLS_URL.format(scale=scale)
            LOGGER.info(f"{prefix} fetching YOLO26 alls from Hailo Model Zoo: {alls_url}")
            try:
                resolved_model_script = safe_download(url=alls_url, dir=output_dir, exist_ok=True)
            except Exception as e:
                # safe_download retries 3x; surfacing here means the URL or the network is the issue.
                # Reraise as a single clear RuntimeError so the user knows exactly what failed and how
                # to work around it (custom alls path), instead of debugging a download stack trace.
                raise RuntimeError(
                    f"Failed to download the YOLO26 quantization script from the Hailo Model Zoo:\n"
                    f"  url    : {alls_url}\n"
                    f"  output : {output_dir}\n"
                    f"  cause  : {type(e).__name__}: {e}\n"
                    f"Common causes: (1) no internet / proxy blocking raw.githubusercontent.com, "
                    f"(2) the Model Zoo restructured or removed yolo26{scale}.alls. "
                    f"Workaround: download the alls manually and pass model_script=<path> to onnx2hailo."
                ) from e
            if not Path(resolved_model_script).is_file():
                raise RuntimeError(
                    f"Hailo Model Zoo alls download succeeded but no file was written at "
                    f"{resolved_model_script!r}. Re-run, or pass model_script= explicitly to bypass the fetch."
                )
        else:
            # yolov8 / yolo11: build the on-chip NMS JSON config from the parsed HailoNN.
            nms_config = _build_nms_config(runner, conf, iou, num_classes, imgsz, max_det, reg_max)
            nms_config_path = output_dir / f"{model_name}_nms_config.json"
            nms_config_path.write_text(json.dumps(nms_config, indent=2))

    alls = _resolve_model_script(resolved_model_script, task, nms_config_path)
    runner.load_model_script(alls)
    runner.optimize(calibration_data)

    hef_bytes = runner.compile()
    hef_path = output_dir / f"{model_name}.hef"
    hef_path.write_bytes(hef_bytes)

    if metadata is not None:
        # Family-specific params HailoBackend needs at load time. yolov8's on-chip NMS path doesn't need
        # most of these, but writing them for both keeps the metadata schema uniform.
        metadata = dict(metadata)
        metadata["model_family"] = model_family
        if model_family == "yolo26":
            metadata.update(
                {
                    "strides": [8, 16, 32],
                    "reg_max": int(reg_max),
                    "nc": int(num_classes),
                    "max_det": int(max_det),
                    "conf": float(conf),
                    "imgsz": [int(imgsz[0]), int(imgsz[1])],
                }
            )
        YAML.save(output_dir / "metadata.yaml", metadata)

    return str(output_dir)
