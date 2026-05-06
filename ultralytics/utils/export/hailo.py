# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import torch

from ultralytics.utils import LOGGER, YAML
from ultralytics.utils.downloads import safe_download

# --- Hailo Model Zoo resolver ------------------------------------------------------------------------
#
# All Hailo exports load an alls (model script) tuned for the (model, hw_arch, DFC version) triple.
# The Model Zoo publishes per-device alls under cfg/alls/{hw_arch}/base/{stem}.alls (preferred) and a
# device-agnostic fallback at cfg/alls/generic/{stem}.alls. yolov8 / yolo11 alls reference a separate
# nms_postprocess(...) JSON also in the zoo; we fetch it alongside the alls and patch user-supplied
# conf / iou / max_det into it. yolo26 alls have no on-device NMS — used as-is.
#
# DFC ↔ MZ tag mapping
#   • Hailo10H / 15H / 15L: DFC X.Y → MZ tag vX.Y.0 (numerically aligned).
#   • Hailo8 / 8L         : MZ v2.x line is not numerically aligned with DFC versions. The mapping is
#                            maintained by hand in ``H8_DFC_TO_MZ_TAG`` below — refresh on each Hailo
#                            DFC release. v2.18 is the first v2.x release with ``supported_hw_arch``.
HAILO_DEVICES = frozenset({"hailo8", "hailo8l", "hailo10h", "hailo15h", "hailo15l"})  # Hailo devices available for export
H8_DEVICES = frozenset({"hailo8", "hailo8l"})
# Hailo SoC targets (full-stack systems, not host-attached accelerators). Compiled HEFs are deployed
# onto the device and executed there; HailoRT VDevice on a host cannot drive these chips, so the
# inference backend rejects HEFs flagged with these arches up-front.
SOC_DEVICES = frozenset({"hailo15h", "hailo15l"})

# DFC (major, minor) → Model Zoo git tag for the v2.x line (Hailo8 / Hailo8L). The h8 line stalled at
# MZ v2.x while h10h+ moved to v4+, so the mapping is per-release. Append a row when Hailo publishes a
# new DFC v3.x; users on a DFC version not in this table are asked to pass ``model_script=`` directly.
H8_DFC_TO_MZ_TAG: dict[tuple[int, int], str] = {
    (3, 33): "v2.18",
}

HAILO_MZ_RAW_BASE = "https://raw.githubusercontent.com/hailo-ai/hailo_model_zoo/{tag}/{path}"
NMS_POSTPROCESS_RE = re.compile(r'(nms_postprocess\(\s*")([^"]+\.json)(")', re.IGNORECASE)


def _resolve_model_script(model_script: str | Path | None) -> str | None:
    """Read a user-supplied alls override; return None when not provided.

    The auto-generation path that previously lived here is gone — alls now come from the Hailo Model
    Zoo via the resolver chain. ``model_script=`` remains as the escape hatch.

    Args:
        model_script (str | Path | None): User override. May be a file path or raw alls content. When
            None, the caller will run the MZ resolver instead.

    Returns:
        (str | None): The alls content as a string, or None if ``model_script`` is None.
    """
    if model_script is None:
        return None
    s = str(model_script)
    # Resolve `~` and relative paths first so a value like `~/scripts/(hailo).alls` is treated as a
    # path (not as content just because it contains `(`).
    p = Path(s).expanduser()
    if p.is_file():
        return p.read_text()
    # Path-like inputs (str | Path that look like a filesystem path) must resolve to a real file —
    # otherwise we'd silently misread `/data/normalization_calib/foo.alls` as raw alls content
    # because `normalization` appears in the path.
    if isinstance(model_script, Path) or (Path(s).suffix and "\n" not in s):
        raise FileNotFoundError(
            f"model_script={model_script!r} looks like a path but does not exist."
        )
    # Treat as raw alls content only with a strong sentinel: multi-line, OR a top-level alls call
    # with its `(` (so `normalization(...)`, `nms_postprocess(...)` etc. match but a bare path
    # containing one of those words does not).
    sentinel_call_re = re.compile(r"\b(normalization|nms_postprocess|model_optimization_flavor|quantization)\s*\(")
    if "\n" in s or sentinel_call_re.search(s):
        return s
    raise FileNotFoundError(
        f"model_script={model_script!r} is not a file and does not look like alls content. "
        f"Pass either a path to an existing .alls file or raw alls script content."
    )


def _resolve_dfc_version() -> tuple[int, int, int]:
    """Read the installed Hailo Dataflow Compiler version. Hard-fail with guidance if absent.

    Returns:
        (tuple[int, int, int]): ``(major, minor, patch)`` (patch defaults to 0 if not exposed).
    """
    try:
        import hailo_sdk_client
    except ImportError as e:
        raise RuntimeError(
            "Hailo Dataflow Compiler ('hailo_sdk_client') is not installed; cannot resolve the matching "
            "Model Zoo tag. Install DFC from https://hailo.ai/developer-zone/, or pass model_script= to "
            "bypass the MZ resolver."
        ) from e
    raw = (
        getattr(hailo_sdk_client, "__version__", None)
        or getattr(getattr(hailo_sdk_client, "version", None), "__version__", None)
        or getattr(hailo_sdk_client, "VERSION", None)
    )
    if not raw:
        raise RuntimeError(
            "Could not determine the installed Hailo DFC version (no __version__ / VERSION attribute on "
            "hailo_sdk_client). Upgrade the SDK or pass model_script= explicitly to bypass the MZ resolver."
        )
    match = re.match(r"v?(\d+)\.(\d+)(?:\.(\d+))?", str(raw))
    if not match:
        raise RuntimeError(
            f"Unrecognized Hailo DFC version string {raw!r}; expected MAJOR.MINOR[.PATCH]. Upgrade the SDK "
            f"or pass model_script= explicitly to bypass the MZ resolver."
        )
    return int(match.group(1)), int(match.group(2)), int(match.group(3) or 0)


def _resolve_mz_tag(dfc_version: tuple[int, int, int], hw_arch: str) -> str:
    """Map a DFC version + target device to the Hailo Model Zoo git tag.

    h10h/h15h/h15l: numerical (``DFC X.Y → vX.Y.0``). h8/h8l: hand-maintained table since the v2.x
    line isn't numerically aligned with DFC.

    Raises:
        NotImplementedError: For h8/h8l DFC versions not in ``H8_DFC_TO_MZ_TAG``.
    """
    if hw_arch in H8_DEVICES:
        key = (dfc_version[0], dfc_version[1])
        if key not in H8_DFC_TO_MZ_TAG:
            raise NotImplementedError(
                f"No Hailo Model Zoo tag mapped for Hailo8/8L on DFC {key[0]}.{key[1]}. Known mappings: "
                f"{sorted(H8_DFC_TO_MZ_TAG)}. Upgrade Ultralytics, or pass model_script= to bypass the "
                f"MZ resolver and supply a custom alls."
            )
        return H8_DFC_TO_MZ_TAG[key]
    return f"v{dfc_version[0]}.{dfc_version[1]}.0"


def _url_exists(url: str, timeout: float = 10.0) -> bool:
    """HEAD ``url`` and return True iff the server responds 2xx.

    ``safe_download`` falls back to ``curl`` on retry, and curl happily writes a 404 HTML body to
    disk and returns success — so the caller can't distinguish a real download from a saved error
    page. A pre-flight HEAD lets ``_fetch_mz_file`` short-circuit on 404 without that ambiguity.
    """
    try:
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return 200 <= resp.status < 300
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as e:
        LOGGER.debug(f"Hailo: HEAD {url} failed: {type(e).__name__}: {e}")
        return False


def _fetch_mz_file(
    mz_tag: str,
    rel_path: str,
    dest_dir: Path,
    *,
    dest_name: str | None = None,
) -> Path | None:
    """Download a single file from the Hailo Model Zoo at ``mz_tag``.

    Args:
        mz_tag (str): Git tag (e.g. ``"v5.2.0"``).
        rel_path (str): Path within the MZ repo (e.g. ``"hailo_model_zoo/cfg/alls/hailo10h/base/yolov8n.alls"``).
        dest_dir (Path): Local destination directory.
        dest_name (str | None): Override the saved filename (defaults to the URL basename).

    Returns:
        (Path | None): Local file path on success, ``None`` when the URL 404s or the network is
        unreachable so the caller can try a fallback URL.
    """
    url = HAILO_MZ_RAW_BASE.format(tag=mz_tag, path=rel_path)
    # Pre-flight HEAD: safe_download's curl-on-retry path silently saves a 404 response body to
    # disk and returns a Path, which would otherwise look like a successful fetch downstream.
    if not _url_exists(url):
        return None
    try:
        result = safe_download(
            url=url,
            file=dest_name,
            dir=dest_dir,
            unzip=False,
            retry=1,
            exist_ok=True,
        )
    except Exception as e:
        LOGGER.debug(f"Hailo: MZ fetch failed: {url} -> {type(e).__name__}: {e}")
        return None
    path = Path(result)
    if not path.is_file():
        return None
    return path


def _validate_hw_support(yaml_path: Path, stem: str, hw_arch: str) -> None:
    """Assert ``hw_arch`` is in the MZ network YAML's ``info.supported_hw_arch`` list.

    Raises:
        NotImplementedError: If the YAML lacks ``info.supported_hw_arch`` or the arch isn't listed,
            naming the actually supported arches so the user can pick a compatible target.
    """
    data = YAML.load(yaml_path) or {}
    supported = (data.get("info") or {}).get("supported_hw_arch")
    if not supported:
        raise NotImplementedError(
            f"Hailo Model Zoo network YAML for {stem!r} has no 'info.supported_hw_arch' field at "
            f"{yaml_path}; cannot validate the target device. Pass model_script= to bypass."
        )
    if hw_arch not in supported:
        raise NotImplementedError(
            f"{stem} is not supported on {hw_arch!r} per the Hailo Model Zoo. Supported devices: "
            f"{sorted(supported)}. Pick a different hw_arch, or pass model_script= for a custom build."
        )


def _resolve_mz_alls(stem: str, mz_tag: str, hw_arch: str, dest_dir: Path) -> Path:
    """Download the alls for ``stem`` at ``mz_tag``, preferring the HW-specific tuning over generic.

    Args:
        stem (str): Model stem (e.g. ``"yolov8n"``, ``"yolo26m"``).
        mz_tag (str): MZ git tag.
        hw_arch (str): Hailo device name.
        dest_dir (Path): Local destination directory.

    Returns:
        (Path): Local path to the downloaded alls.

    Raises:
        RuntimeError: If neither the HW-specific nor the generic alls is available at this tag.
    """
    hw_rel = f"hailo_model_zoo/cfg/alls/{hw_arch}/base/{stem}.alls"
    generic_rel = f"hailo_model_zoo/cfg/alls/generic/{stem}.alls"
    LOGGER.info(f"Hailo: fetching MZ alls {hw_rel}@{mz_tag}")
    path = _fetch_mz_file(mz_tag, hw_rel, dest_dir)
    if path is not None:
        return path
    LOGGER.warning(
        f"Hailo: HW-specific alls not found for {hw_arch} at {mz_tag}; falling back to generic. "
        f"Accuracy may be lower than the device-tuned alls."
    )
    path = _fetch_mz_file(mz_tag, generic_rel, dest_dir)
    if path is not None:
        return path
    raise RuntimeError(
        f"Could not download an alls for {stem} at MZ {mz_tag}. Tried:\n"
        f"  {HAILO_MZ_RAW_BASE.format(tag=mz_tag, path=hw_rel)}\n"
        f"  {HAILO_MZ_RAW_BASE.format(tag=mz_tag, path=generic_rel)}\n"
        f"Check connectivity, or pass model_script= to compile a custom variant."
    )


def _fetch_and_patch_nms_config(
    alls_path: Path,
    stem: str,
    mz_tag: str,
    dest_dir: Path,
    *,
    conf: float,
    iou: float,
    max_det: int,
) -> Path:
    """Download the MZ NMS JSON for a yolov8 / yolo11 alls and patch user thresholds in.

    The alls's ``nms_postprocess(...)`` line references a JSON via a relative path (e.g.
    ``"../../postprocess_config/yolov8n_nms_config.json"``). Both the HW-specific
    (``cfg/alls/{hw_arch}/base/...``) and the generic (``cfg/alls/generic/...``) layouts resolve this
    to ``cfg/postprocess_config/{stem}_nms_config.json`` at the MZ root. We download it, patch
    ``nms_scores_th`` / ``nms_iou_th`` / ``max_proposals_per_class`` with the user's values, and
    rewrite the alls file so its first arg is the absolute path of our local copy.

    Args:
        alls_path (Path): The downloaded MZ alls (mutated in place).
        stem (str): Model stem (used for the JSON filename).
        mz_tag (str): MZ git tag.
        dest_dir (Path): Local directory to write the JSON into.
        conf (float): NMS score threshold (overrides MZ default).
        iou (float): NMS IoU threshold (overrides MZ default).
        max_det (int): Max proposals per class (overrides MZ default).

    Returns:
        (Path): Local path to the patched NMS JSON.

    Raises:
        RuntimeError: If the alls has no ``nms_postprocess(...)`` line, or the JSON 404s.
    """
    alls_text = alls_path.read_text()
    match = NMS_POSTPROCESS_RE.search(alls_text)
    if match is None:
        raise RuntimeError(
            f"Hailo Model Zoo alls for {stem}@{mz_tag} ({alls_path}) has no nms_postprocess(...) line; "
            f"cannot wire the on-device NMS config. Pass model_script= to compile a custom variant."
        )
    json_basename = Path(match.group(2)).name
    rel_path = f"hailo_model_zoo/cfg/postprocess_config/{json_basename}"
    local_name = f"{stem}_nms_config.json"
    json_path = _fetch_mz_file(mz_tag, rel_path, dest_dir, dest_name=local_name)
    if json_path is None:
        raise RuntimeError(
            f"Could not download MZ NMS JSON {rel_path}@{mz_tag}. "
            f"URL: {HAILO_MZ_RAW_BASE.format(tag=mz_tag, path=rel_path)}. "
            f"Pass model_script= to compile a custom variant."
        )
    config = json.loads(json_path.read_text())
    config["nms_scores_th"] = float(conf)
    config["nms_iou_th"] = float(iou)
    # Hailo's max_proposals_per_class is a per-class on-device cap, while Ultralytics' max_det is a
    # total cap. Mirroring max_det here keeps the chip permissive (up to nc * max_det proposals);
    # HailoBackend._decode_nms then sorts and trims to max_det total. A tighter per-class cap would
    # save NPU bandwidth but could starve a dominant class on class-imbalanced frames.
    config["max_proposals_per_class"] = int(max_det)
    json_path.write_text(json.dumps(config, indent=2))
    new_ref = f'{match.group(1)}{json_path.resolve()}{match.group(3)}'
    alls_path.write_text(NMS_POSTPROCESS_RE.sub(new_ref, alls_text, count=1))
    return json_path


def _dataloader_to_numpy(dataloader) -> np.ndarray:
    """Stack a YOLO calibration dataloader into a single (N, H, W, C) uint8 RGB array.

    Hailo's on-device normalization layer expects raw RGB uint8 frames in [0, 255]; we therefore
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
    end_node_names: list[str] | None = None,
    prefix: str = "",
) -> str:
    """Compile an ONNX YOLO model to a Hailo HEF using the Hailo Dataflow Compiler.

    The alls (model script) and on-device NMS JSON are pulled from the Hailo Model Zoo at the
    git tag matching the installed DFC version. Per-device tuning (``cfg/alls/{hw_arch}/base/...``)
    is preferred; a ``cfg/alls/generic/...`` fallback is used when the device-tuned variant is missing.

    Supports two on-device / host-side splits:

    - ``yolov8`` / ``yolo11`` — on-device NMS via ``nms_postprocess(meta_arch=yolov8)``. The MZ alls
      references a JSON config which we fetch alongside it; user-supplied ``conf`` / ``iou`` /
      ``max_det`` are patched into that JSON before compile.
    - ``yolo26`` — NMS-free end2end. The Hailo NPU doesn't support topk/gather; postprocess runs on
      host (see ``HailoBackend``). MZ alls is used as-is, no NMS JSON to fetch.

    Args:
        onnx_file (str): Path to the source ONNX file.
        output_dir (Path | str): Directory to write the compiled ``<model_name>.hef`` and metadata.
        hw_arch (str): Hailo hardware target (one of ``HAILO_DEVICES``).
        task (str): Ultralytics task. Only ``"detect"`` is supported in this release.
        calibration_data (np.ndarray): (N, H, W, C) RGB uint8 calibration array [0-255].
        model_script (str | Path | None): Optional alls override (file path or raw alls content). When
            provided, the entire MZ resolver chain (DFC version detection, hw_arch validation, MZ alls
            + NMS JSON fetch) is skipped.
        conf (float): NMS score threshold (``nms_scores_th``); patched into the MZ NMS JSON.
        iou (float): NMS IoU threshold (``nms_iou_th``); patched into the MZ NMS JSON.
        num_classes (int): Number of object classes. Logged as a warning if != 80 (MZ baseline).
        imgsz (tuple[int, int]): Model input ``(height, width)``.
        max_det (int): Max proposals per class for on-device NMS / max detections for host postprocess.
        reg_max (int): DFL ``reg_max`` (16 for YOLOv8, 1 for YOLO26). Persisted in metadata for the
            host-side YOLO26 decode in ``HailoBackend``.
        metadata (dict | None): Metadata to persist alongside the HEF as ``metadata.yaml``. The function
            adds family-specific postprocess params (``model_family``, ``strides``, ``reg_max``, etc.)
            and the resolved ``hailo_mz_tag`` (when MZ was used) so ``HailoBackend`` can configure
            host-side decode at load time.
        model_name (str): Model file stem (e.g. ``"yolov8n"``); used directly as the MZ filename for
            both the network YAML and the alls.
        model_family (str): ``"yolov8"`` (default) or ``"yolo26"``. Controls whether the on-device NMS
            JSON is fetched + patched (yolov8/yolo11) or skipped (yolo26).
        end_node_names (list[str] | None): ONNX end-node names to cut the graph at. The caller computes
            these because the conv-leaf prefix is family-specific (``cv2``/``cv3`` for yolov8,
            ``one2one_cv2``/``one2one_cv3`` for yolo26).
        prefix (str): Prefix for log messages.

    Returns:
        (str): Path to the produced ``_hailo_model`` directory.

    Raises:
        ImportError: If the Hailo Dataflow Compiler is not installed.
        ValueError: If calibration data is missing.
        RuntimeError: For DFC version detection failures, MZ download failures, or h8/h8l DFC↔MZ
            mapping miss.
        NotImplementedError: For DFC < 3.33 on h8/h8l, or when the model variant is missing /
            unsupported on the requested device per the MZ network YAML.
    """
    if calibration_data is None or len(calibration_data) == 0:
        raise ValueError("Calibration data is required for Hailo quantization.")
    if calibration_data.dtype != np.uint8:
        LOGGER.warning(
            f"{prefix} calibration_data dtype is {calibration_data.dtype} — Hailo expects RGB uint8 in [0, 255]. "
            f"On-device normalization divides by 255; passing pre-normalized [0, 1] data will severely degrade accuracy."
        )
    elif calibration_data.max() <= 1:
        LOGGER.warning(
            f"{prefix} calibration_data max value <= 1 — looks pre-normalized. Hailo expects raw RGB uint8 frames "
            f"in [0, 255]; the device normalizes by 255 internally."
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

    alls_text: str | None = _resolve_model_script(model_script)
    mz_tag: str | None = None
    # MZ-resolver scratch files — deleted after compile so the output dir only ships the HEF + metadata.
    mz_artifacts: list[Path] = []

    if alls_text is not None:
        # User-supplied alls bypasses MZ resolution — including the hw_arch compatibility check we'd
        # otherwise do via the network YAML's supported_hw_arch list. Surface that so the user knows
        # they own the alls↔device match.
        LOGGER.warning(
            f"{prefix} model_script= override skips hw_arch validation against the Hailo Model Zoo. "
            f"Ensure the alls is tuned for hw_arch={hw_arch!r}; a mismatch will compile a misconfigured HEF."
        )

    if alls_text is None:
        if task != "detect":
            raise NotImplementedError(
                f"Hailo export currently only resolves an alls for task='detect', got task='{task}'. "
                f"Pass a custom alls via model_script= to compile other tasks."
            )
        if int(num_classes) != 80:
            LOGGER.warning(
                f"{prefix} num_classes={num_classes} differs from MZ's COCO80 baseline; the MZ alls is "
                f"tuned for 80 classes — quantization quality on a custom-class model may regress. "
                f"Pass model_script= for full custom-class accuracy parity."
            )
        dfc_version = _resolve_dfc_version()
        mz_tag = _resolve_mz_tag(dfc_version, hw_arch)
        LOGGER.info(f"{prefix} using Hailo Model Zoo tag {mz_tag} (DFC {'.'.join(map(str, dfc_version))}).")
        yaml_path = _fetch_mz_file(
            mz_tag,
            f"hailo_model_zoo/cfg/networks/{model_name}.yaml",
            output_dir,
        )
        if yaml_path is None:
            raise NotImplementedError(
                f"{model_name!r} is not published in Hailo Model Zoo at {mz_tag}; pass model_script= "
                f"to compile a custom variant."
            )
        mz_artifacts.append(yaml_path)
        _validate_hw_support(yaml_path, model_name, hw_arch)
        alls_path = _resolve_mz_alls(model_name, mz_tag, hw_arch, output_dir)
        mz_artifacts.append(alls_path)
        if model_family != "yolo26":
            json_path = _fetch_and_patch_nms_config(
                alls_path, model_name, mz_tag, output_dir,
                conf=conf, iou=iou, max_det=max_det,
            )
            mz_artifacts.append(json_path)
        alls_text = alls_path.read_text()

    runner.load_model_script(alls_text)
    runner.optimize(calibration_data)

    hef_bytes = runner.compile()
    hef_path = output_dir / f"{model_name}.hef"
    hef_path.write_bytes(hef_bytes)

    # Resolver scratch files (network YAML + alls + NMS JSON) are no longer needed — the HEF holds the
    # compiled graph and the JSON's thresholds are baked in. Leaving them around (especially the alls,
    # which we rewrote with an absolute local path) leaks host paths if the user tarballs the dir.
    for artifact in mz_artifacts:
        try:
            artifact.unlink(missing_ok=True)
        except OSError as cleanup_error:
            LOGGER.debug(f"Hailo: could not unlink scratch artifact {artifact}: {cleanup_error}")

    if metadata is not None:
        # Family-specific params HailoBackend needs at load time. yolov8's on-device NMS path doesn't need
        # most of these, but writing them for both keeps the metadata schema uniform.
        metadata = dict(metadata)
        metadata["model_family"] = model_family
        metadata["hw_arch"] = hw_arch
        if mz_tag:
            metadata["hailo_mz_tag"] = mz_tag
        if model_family == "yolo26":
            metadata.update(
                {
                    "strides": [8, 16, 32],
                    "reg_max": int(reg_max),
                    "nc": int(num_classes),
                    "max_det": int(max_det),
                    "conf": float(conf),
                    "imgsz": [int(imgsz[0]), int(imgsz[1])],
                    "head_outputs": _yolo26_head_outputs(runner, list(end_node_names or [])),
                }
            )
        YAML.save(output_dir / "metadata.yaml", metadata)

    return str(output_dir)


def _yolo26_head_outputs(runner, end_node_names: list[str]) -> dict:
    """Capture YOLO26 head-leaf metadata so the runtime can dispatch by name, not channel count.

    The exporter cuts the ONNX at 6 conv leaves interleaved as
    ``[cv2_0, cv3_0, cv2_1, cv3_1, cv2_2, cv3_2]``. Even indices are box-reg heads, odd are class heads.
    We persist the HailoNN layer names and their spatial sizes (per stride) so ``HailoBackend`` can
    pre-classify output buffers without relying on channel counts — which is ambiguous when ``nc == 4``
    (e.g. a 4-class custom dataset puts every leaf into both the box and cls bucket).

    Args:
        runner: HailoSDK ``ClientRunner`` after ``compile()``.
        end_node_names (list[str]): Cut node names (interleaved). Used only as a sanity check on count.

    Returns:
        (dict): ``{"box_layers": [{"name": str, "spatial": int}, ...],
        "cls_layers": [...]}`` — per-stride entries sorted by descending spatial dim
        (largest = stride[0]). Empty dict if the layout cannot be determined.
    """
    try:
        hn = runner.get_hn_model()
        output_preds: list = []
        for o in hn.get_output_layers():
            output_preds.extend(hn.predecessors(o))
        if end_node_names and len(output_preds) != len(end_node_names):
            LOGGER.warning(
                f"Hailo: expected {len(end_node_names)} HN output predecessors, got {len(output_preds)}; "
                f"head_outputs metadata will be omitted (runtime falls back to channel-count dispatch)."
            )
            return {}
        # Sort each role by spatial dim (largest -> smallest = stride[0] -> stride[-1]). Layers are
        # NHWC in HN today, so dim 1 is the H spatial. Validate the rank + that the value isn't the
        # channel count (4 for box, nc for cls) so a future SDK that reports NCHW or scalar shapes
        # fails loud instead of silently scrambling stride order.
        box_layers = output_preds[0::2]
        cls_layers = output_preds[1::2]
        for layer in (*box_layers, *cls_layers):
            shape = layer.output_shapes[0]
            if len(shape) != 4:
                LOGGER.warning(
                    f"Hailo: HN layer {layer.name!r} has unexpected rank {len(shape)} (shape={shape}); "
                    f"head_outputs metadata omitted (runtime falls back to channel-count dispatch)."
                )
                return {}
        box_layers.sort(key=lambda layer: -layer.output_shapes[0][1])
        cls_layers.sort(key=lambda layer: -layer.output_shapes[0][1])
        return {
            "box_layers": [{"name": layer.name, "spatial": int(layer.output_shapes[0][1])} for layer in box_layers],
            "cls_layers": [{"name": layer.name, "spatial": int(layer.output_shapes[0][1])} for layer in cls_layers],
        }
    except Exception as e:
        LOGGER.warning(
            f"Hailo: failed to derive YOLO26 head_outputs metadata ({type(e).__name__}: {e}); "
            f"runtime will fall back to channel-count dispatch."
        )
        return {}
