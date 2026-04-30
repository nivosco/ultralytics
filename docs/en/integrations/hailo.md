---
comments: true
description: Deploy Ultralytics YOLO models on Hailo AI architectures. Compile YOLOv8 with on-chip NMS or YOLO26 end-to-end models to HEF and run them with HailoRT.
keywords: Hailo, Hailo AI, Hailo-8, Hailo-10H, Hailo-15H, Hailo-15L, Hailo Dataflow Compiler, HailoRT, Edge AI, YOLOv8, YOLO26, Model Export, Computer Vision, Object Detection, quantization
---

# Hailo AI Export and Deployment

Ultralytics supports exporting YOLO models to [Hailo](https://hailo.ai/) AI accelerators using the **Hailo Dataflow Compiler (DFC)** for compilation and **HailoRT** for runtime inference. Two model families are supported in this initial release:

- **YOLOv8** detection models — compiled with on-chip NMS via the `nms_postprocess(meta_arch="yolov8")` model script macro, so the chip emits final detections directly.
- **YOLO26** detection models — already end-to-end (the head's `postprocess()` produces final detections), so no on-chip NMS is required.

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
| `hailo15h` | Hailo-15H  | Embedded SoC variant (high)            |
| `hailo15m` | Hailo-15M  | Embedded SoC variant (mid)             |
| `hailo15l` | Hailo-15L  | Embedded SoC variant (low)             |

> **Note on `name=`** — Ultralytics' `name` argument also controls the run sub-directory under `runs/<task>/`. Following the same convention used by Rockchip RKNN, the Hailo export reuses it for the chip target, so a default Hailo export lands at `runs/<task>/hailo10h/`.

## Export

### Python

```python
from ultralytics import YOLO

# YOLOv8 detect — on-chip NMS is added automatically by the auto-generated model script
YOLO("yolov8n.pt").export(format="hailo", data="coco8.yaml", name="hailo10h", imgsz=640)

# YOLO26 detect — end-to-end, no on-chip NMS needed
YOLO("yolo26n.pt").export(format="hailo", data="coco8.yaml", name="hailo10h", imgsz=640)
```

### CLI

```bash
yolo export model=yolov8n.pt format=hailo data=coco8.yaml name=hailo10h imgsz=640
yolo export model=yolo26n.pt format=hailo data=coco8.yaml name=hailo10h imgsz=640
```

The export produces a `<stem>_hailo_model/` directory containing `<stem>.hef` and `metadata.yaml`.

### Calibration data — RAW RGB uint8 [0, 255]

The auto-generated model script begins with `normalization([0,0,0],[255,255,255])`, meaning the chip divides input by 255 internally. **Calibration data and inference inputs must therefore be raw RGB `uint8` frames in `[0, 255]` — not pre-normalized `float [0, 1]`.** Ultralytics's standard YOLO calibration dataloader already produces uint8 frames, so the default flow should be used.

For optimal quantization accuracy, supply **at least 1024 calibration images** (a warning fires below this threshold). Increase via the `fraction=` arg or pick a larger `data=` dataset.

### Custom model script override (advanced)

For advanced workflows that need a custom alls model script (e.g. tuned NMS thresholds, per-layer quantization hints, alternative output activations), call the lower-level `onnx2hailo()` utility directly. It accepts a `model_script=` argument as either a path to an `.alls` file or raw alls content, and bypasses the auto-generated script entirely:

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

The high-level `YOLO(...).export(format="hailo")` path uses only the auto-generated script — it does not currently surface `model_script=` as a CLI/Python kwarg.

## Inference

```python
from ultralytics import YOLO

model = YOLO("yolov8n_hailo_model")  # or "yolo26n_hailo_model"
results = model.predict("bus.jpg")
```

The Hailo runtime expects raw RGB `uint8` `[0, 255]` input. The standard Ultralytics predict pipeline emits a normalized float tensor; the backend silently casts it back to uint8 on the host (matching the behavior of the Rockchip RKNN backend). Both supported HEF flavors are decoded into the predictor's end-to-end short path:

- **YOLOv8 with on-chip NMS** — the chip emits per-class detection lists; the backend converts these to `(B, N, 6)` `[x1, y1, x2, y2, conf, cls]` in input-pixel coords and the predictor's end2end branch consumes them directly.
- **YOLO26 end2end** — the chip's natural `(N, 6)` output is wrapped with a batch dim and passed through.

## Troubleshooting

- **`ImportError: Hailo Dataflow Compiler ('hailo_sdk_client') is required ...`** — install the DFC from the [Hailo Developer Zone](https://hailo.ai/developer-zone/).
- **`ImportError: HailoRT ('hailo_platform') is required ...`** — install HailoRT from the [Hailo Developer Zone](https://hailo.ai/developer-zone/).
- **Severely degraded mAP after quantization** — most common causes are (1) calibration data passed as pre-normalized `float [0, 1]` instead of `uint8 [0, 255]`, or (2) fewer than 1024 calibration images. Both emit warnings during export.
- **Tasks other than detect** — pose, segmentation, OBB, and classification are not yet auto-supported. You can still compile them by supplying a custom `model_script=` and using the lower-level `onnx2hailo()` utility directly.
