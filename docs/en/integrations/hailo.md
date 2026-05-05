---
comments: true
description: Deploy Ultralytics YOLO models on Hailo AI architectures. Compile YOLOv8 / YOLO11 / YOLO26 detection models to HEF, then run inference via HailoRT.
keywords: Hailo, Hailo AI, Hailo-8, Hailo-10H, Hailo-15H, Hailo-15L, Hailo Dataflow Compiler, HailoRT, Edge AI, YOLOv8, YOLO11, YOLO26, Model Export, Computer Vision, Object Detection, quantization
---

# Hailo AI Export and Deployment

Ultralytics supports exporting YOLO models to [Hailo](https://hailo.ai/) AI accelerators using the **Hailo Dataflow Compiler (DFC)** for compilation and **HailoRT** for runtime inference.

## Supported models

| Family | Detect head | NMS location | Notes |
| :--- | :--- | :--- | :--- |
| **YOLOv8** / **YOLO11** | Anchor-free DFL (`reg_max=16`) | On-chip via `nms_postprocess(meta_arch=yolov8)` | Auto-generated NMS JSON config emitted alongside the HEF |
| **YOLO26** (n / s / m / l) | NMS-free end2end (`reg_max=1`) | Host-side topk + gather | The Hailo NPU does not support topk/gather; postprocess runs on the host inside `HailoBackend`. The variant-specific quantization script (`yolo26{n,s,m,l}.alls`) is fetched automatically from the [Hailo Model Zoo](https://github.com/hailo-ai/hailo_model_zoo/tree/master/hailo_model_zoo/cfg/alls/generic) for accuracy parity. **YOLO26x is not supported** (no MZ-published alls); pass a custom `model_script=` to compile it. |

> **Other YOLO families** (YOLOv9 / YOLOv10 / YOLOv12) are **not yet supported** — the export pipeline raises `NotImplementedError` if you attempt one. Tracked for a future release.

## Important: SDK installation

Both the Hailo Dataflow Compiler and HailoRT are distributed exclusively through the **Hailo Developer Zone** — they are not available on PyPI. You must install them manually before running an export or running inference:

> Download the Hailo Dataflow Compiler and HailoRT packages from <https://hailo.ai/developer-zone/> and follow the included installation instructions.

If either package is missing, Ultralytics raises an `ImportError` directing you to the Developer Zone — there is no automatic install attempt.

## Supported hardware targets

Pass the chip name via the `name=` argument:

| `name=`    | Chip       | Notes                                  |
| :--------- | :--------- | :------------------------------------- |
| `hailo8`   | Hailo-8    | DRAM-less AI Accelerator               |
| `hailo8l`  | Hailo-8L   | Smaller Hailo-8 variant                |
| `hailo10h` | Hailo-10H  | **Default.** AI Accelerator            |
| `hailo15h` | Hailo-15H  | Embedded SoC (high end)                |
| `hailo15l` | Hailo-15L  | Embedded SoC (cost effective)          |

> **Note on `name=`** — Ultralytics' `name` argument also controls the run sub-directory under `runs/<task>/`. Following the same convention used by Rockchip RKNN, the Hailo export reuses it for the chip target, so a default Hailo export lands at `runs/<task>/hailo10h/`.

## Export

### Python

```python
from ultralytics import YOLO

# YOLOv8 / YOLO11 — auto-generated alls adds nms_postprocess(meta_arch=yolov8); NMS runs on-chip.
YOLO("yolov8n.pt").export(format="hailo", data="coco8.yaml", name="hailo10h", imgsz=640)
YOLO("yolo11n.pt").export(format="hailo", data="coco8.yaml", name="hailo10h", imgsz=640)

# YOLO26 — fetches yolo26{n,s,m,l,x}.alls from the Hailo Model Zoo; postprocess runs on host.
YOLO("yolo26s.pt").export(format="hailo", data="coco8.yaml", name="hailo10h", imgsz=640)
```

### CLI

```bash
yolo export model=yolov8n.pt format=hailo data=coco8.yaml name=hailo10h imgsz=640
yolo export model=yolo11n.pt format=hailo data=coco8.yaml name=hailo10h imgsz=640
yolo export model=yolo26s.pt format=hailo data=coco8.yaml name=hailo10h imgsz=640
```

The export produces a `<stem>_hailo_model/` directory containing `<stem>.hef` and `metadata.yaml`.

### Calibration data — RAW RGB uint8 [0, 255]

The auto-generated model script begins with `normalization([0,0,0],[255,255,255])`, meaning the chip divides input by 255 internally. **Calibration data and inference inputs must therefore be raw RGB `uint8` frames in `[0, 255]` — not pre-normalized `float [0, 1]`.** Ultralytics's standard YOLO calibration dataloader already produces uint8 frames, so the default flow should be used.

For optimal quantization accuracy, supply **at least 1024 calibration images** (a warning fires below this threshold). Increase via the `fraction=` arg or pick a larger `data=` dataset.

### Custom model script override (advanced)

For advanced workflows that need a custom alls model script (e.g. tuned NMS thresholds, per-layer quantization hints, alternative output activations), call the lower-level `onnx2hailo()` utility directly. It accepts a `model_script=` argument as either a path to an `.alls` file or raw alls content, which **bypasses both the YOLOv8 auto-generated script and the YOLO26 Hailo Model Zoo fetch**:

```python
from ultralytics.utils.export.hailo import onnx2hailo

# 'yolov8n.onnx' must already exist (e.g. from YOLO(...).export(format='onnx'))
onnx2hailo(
    onnx_file="yolov8n.onnx",
    output_dir="yolov8n_hailo_model",
    hw_arch="hailo10h",
    calibration_data=calibration_array,  # (N, H, W, C) RGB uint8
    model_script="path/to/my_script.alls",  # or raw alls content
)
```

The high-level `YOLO(...).export(format="hailo")` path uses the family-specific default script — it does not currently surface `model_script=` as a CLI/Python kwarg.

## Inference

```python
from ultralytics import YOLO

model = YOLO("yolov8n_hailo_model")  # or yolo11n_hailo_model / yolo26s_hailo_model
results = model.predict("bus.jpg")
```

The Hailo runtime expects raw RGB `uint8` `[0, 255]` input. The standard Ultralytics predict pipeline emits a normalized float tensor; the backend silently casts it back to uint8 (matching the behavior of the Rockchip RKNN backend). `HailoBackend` reads `model_family` from the HEF directory's `metadata.yaml` and dispatches to the right decode path:

- **YOLOv8 / YOLO11** — the chip's on-chip NMS emits per-class detection lists which the backend rescales to input-pixel coords.
- **YOLO26** — the chip emits 6 raw conv outputs (3 strides × {box-reg, class-logits}); the backend builds anchors per stride, applies `dist2bbox` + sigmoid, and runs a top-k selection over (anchor × class) pairs on host (the same math as `Detect.postprocess` for end2end models).

Both paths return `(1, N, 6)` `[x1, y1, x2, y2, conf, cls]` in input-pixel coords and the predictor's end2end branch consumes them directly.

## Troubleshooting

- **`ImportError: Hailo Dataflow Compiler ('hailo_sdk_client') is required ...`** — install the DFC from the [Hailo Developer Zone](https://hailo.ai/developer-zone/).
- **`ImportError: HailoRT ('hailo_platform') is required ...`** — install HailoRT from the [Hailo Developer Zone](https://hailo.ai/developer-zone/).
- **Severely degraded mAP after quantization** — most common causes are (1) calibration data passed as pre-normalized `float [0, 1]` instead of `uint8 [0, 255]`, or (2) fewer than 1024 calibration images. Both emit warnings during export.
- **Tasks other than detect** — pose, segmentation, OBB, and classification are not yet auto-supported. You can still compile them by supplying a custom `model_script=` and using the lower-level `onnx2hailo()` utility directly.
