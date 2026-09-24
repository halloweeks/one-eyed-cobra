## Author

**One-Eyed Cobra (OEC)** was originally created and developed by **Hallo Weeks**.

The project is open for contributions, improvements, and further development.


# One-Eyed Cobra (OEC)

OEC is built for a specific purpose: real-time object detection running
directly on embedded CPUs / SoCs — think ~1-5 TOPS class hardware — with
no GPU, no cloud, and no external ML runtime on the device. Every design
decision in this project is made for that target first: pure C, no
Python, no PyTorch/TensorFlow/ONNX Runtime, no CUDA requirement, float32
today with INT8/NEON as a planned follow-up once correctness is proven.

OEC is a **dedicated single-object detector, not a general-purpose one.**
It does not aim to detect and classify dozens of arbitrary object
categories in a scene the way general-purpose detectors do — it's built
to reliably find one specific kind of object per frame, as lean and fast
as that narrower job allows. That focus is what makes it practical to run
on constrained embedded hardware in the first place; a general-purpose
multi-class detector carries capacity and compute overhead this project
deliberately avoids.

OEC is also not a YOLO fork, a reimplementation of an existing
architecture, or a repurposed general framework — it's a small, original
detection pipeline built from scratch around this specific target and
this specific job.

> Status: early development. Core training/inference pipeline is functional
> and validated on small overfit tests; not yet trained or benchmarked on a
> full dataset. Accuracy claims below intentionally avoided until real
> numbers exist.

## What OEC is

- Pure C (C11), no Python runtime, no PyTorch/TensorFlow/ONNX Runtime, no
  CUDA requirement.
- Float32 reference implementation. INT8 / NEON / SIMD optimization is a
  planned later phase, not part of the current codebase.
- Detects **one object per frame** — a dedicated single-object detector,
  not a general multi-object one (multi-object support is a possible future
  direction, not implemented today).
- Anchor-free, grid-based detection head: the backbone downsamples the
  input into a spatial grid, and each grid cell independently predicts
  confidence + a bounding box. The cell whose center is closest to the
  object is trained as the positive target; every other cell is trained as
  background.

## What OEC is not

- Not a copy of YOLO, MobileNet, EfficientNet, SSD, or RetinaNet — no
  architecture or terminology is borrowed from those projects.
- Not benchmarked yet. No mAP, precision/recall, latency, or FPS numbers
  are published because none have been measured on a real dataset. Numbers
  will be added here once they exist, with the methodology alongside them.

## Architecture

The network is a flexible stack of convolution and activation layers,
built up programmatically rather than hardcoded — you define the exact
depth, channel progression, kernel sizes, and strides for your use case.
The only structural requirement is that the final layer outputs exactly
5 channels per spatial cell:

```
[confidence, x_offset, y_offset, width, height]
```

- **confidence** — objectness logit for that cell (pass through a sigmoid
  to get a `[0,1]` probability).
- **x_offset / y_offset** — the object's center, expressed as an offset
  *within* the responsible cell, not full-image coordinates.
- **width / height** — box dimensions, normalized to the full image.

Currently supported layer types: standard convolution and activation
(ReLU). Depthwise-separable convolutions, residual connections, and
multi-scale (feature pyramid style) detection heads are on the roadmap but
not yet implemented — today's model is a single flat stack with a single
detection scale.

## Repository layout

```
include/    Public headers (oec_*.h) — one per subsystem
src/        Implementation (oec_*.c)
```

Key modules:

| Module | Responsibility |
|---|---|
| `oec_tensor` | Minimal N-D float tensor (CHW layout) |
| `oec_conv` | Convolution forward/backward |
| `oec_activation` | Activation layers (ReLU) |
| `oec_layer` | Generic layer wrapper over conv/activation |
| `oec_model` | Layer stack, forward/backward, save/load, predict |
| `oec_loss` | Grid-aware detection loss (confidence + box regression) |
| `oec_optimizer` | SGD, Momentum, RMSProp, Adam |
| `oec_dataset` | Dataset loading (train/valid/test splits, YOLO-style labels) |
| `oec_image` | Image I/O (via stb_image) and drawing utilities |

`include/oec.h` aggregates all public headers into a single include.

## Building

Requires CMake 3.16+ and a C11 compiler (GCC or Clang).

```sh
mkdir build && cd build
cmake ..
cmake --build .
```

This produces two binaries:

- **`oec`** — training entry point (builds a model, trains on a dataset,
  saves the best checkpoint).
- **`oec_run`** — inference entry point (loads a saved model, runs a
  single prediction on an image, writes an annotated output image).

Release builds are compiled with `-O3 -march=native`; adjust
`-march` if cross-compiling for a specific embedded target.

## Dataset format

```
dataset/
  train/
    images/
    labels/
  valid/
    images/
    labels/
  test/
    images/
    labels/
```

Labels are YOLO-style, one line per object, normalized `[0,1]`
coordinates:

```
class_id center_x center_y width height
```

Only the first box in a label file is currently used for training (single-
object design); additional boxes in a file are ignored.

## Model file format (`.oec`)

A simple binary format storing: a magic signature, format version, the
input resolution the model was built for, and each layer's type,
configuration, and weights. Not related to ONNX or any other existing
format.

## Validation status

Before scaling to real data, the training pipeline was validated with a
memorization (overfit) test: train on a handful of images and confirm the
model can reproduce their labels almost exactly. This checks that the
architecture, loss, backpropagation, optimizer, and save/load path are all
wired together correctly — a bug anywhere in that chain typically prevents
even a tiny dataset from being memorized.

**Result:** a 4-sample, 64×64 training run converged to a loss of
`~0.0003`, and the saved checkpoint reproduced one training sample almost
exactly on inference:

| | Predicted | Label | Difference |
|---|---|---|---|
| confidence | 0.999977 | 1.0 | 0.00002 |
| x | 0.501324 | 0.501340 | 0.000016 |
| y | 0.338572 | 0.338576 | 0.000004 |
| w | 0.392692 | 0.392507 | 0.000185 |
| h | 0.517463 | 0.517091 | 0.000372 |

This confirms the pipeline is implemented correctly. It is a memorization
result, not a generalization result — it says nothing yet about accuracy
on unseen images, which is the next validation step (holding out samples,
then scaling to a real dataset).

## Roadmap

Rough order of remaining work, correctness before optimization:

1. ~~Tensor / conv / activation primitives~~
2. ~~Backbone + single-scale detection head + forward inference~~
3. Box decoding, NMS, clean inference API
4. ~~Grid-based loss, backprop, optimizers~~
5. ~~Overfit validation on a tiny (single/few-sample) dataset~~ — passed,
   see [Validation status](#validation-status); scaling to a full ~4-image
   held-out set is next
6. Scale training to a full dataset
7. Benchmark accuracy, size, RAM, and latency — publish real numbers
8. INT8 quantization, NEON/SIMD, multi-threading — only after 1–7 hold up

Depthwise-separable convolutions, residual connections, and multi-scale
(P3/P4/P5-style) detection heads are planned architectural additions once
the single-scale pipeline is fully validated end-to-end.

## Engineering principles

- No unverified accuracy claims, and no "better than X" comparisons
  without a real benchmark to back them up.
- No copying another project's source or architecture — techniques that
  are standard (e.g. anchor-free detection, IoU-based box loss) are used
  and credited as established methods, not claimed as original.
- Correctness is validated before any optimization work begins.
- Params/FLOPs are calculated, not estimated by guesswork.

## License

*(add your chosen license here)*
