# WLNeRFLite: Ultra-Lightweight Neural Radiance Fields in Wolfram Language

[![Wolfram Language](https://img.shields.io/badge/Wolfram_Language-13.0%2B-red.svg)](https://www.wolfram.com/language/)
[![NeRF Synthetic](https://img.shields.io/badge/NeRF-transforms.json_Compatible-brightgreen.svg)](#standard-nerf-json-support)
[![RenderMan](https://img.shields.io/badge/RenderMan-27%2F28_EXR-blue.svg)](#renderman-exr-support)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**`WLNeRFLite`** is a standalone, high-performance, ultra-lightweight Neural Radiance Field (NeRF) framework implemented natively in Wolfram Language. It is designed to train coordinate MLPs at high speeds (>25,000 rays/second on GPU) on both **standard NeRF synthetic datasets** (`transforms.json` with PNG/JPG/TIFF images) and **Pixar RenderMan OpenEXR** archives.

---

## Key Features

1. **Dual Dataset Architecture**:
   * **Standard NeRF JSON**: Fully compatible with canonical NeRF synthetic datasets (Mildenhall et al.), Instant-NGP, Nerfstudio, and 3D Gaussian Splatting (`transforms.json`, `transforms_train.json`, `transforms_val.json`, `transforms_test.json`). Supports PNG, JPEG, TIFF, and alpha-masked foregrounds.
   * **Pixar RenderMan EXR**: Ingests multi-layer OpenEXRs directly. Features a built-in pure-Wolfram binary header parser that extracts exact `worldToCamera` $4 \times 4$ matrices and camera FOVs with zero external CLI dependencies.
2. **Exact Bidirectional Camera Transforms**:
   * Mathematical conversion between RenderMan row-major $4 \times 4$ world-to-camera ($M_{\text{w2c}}$) and standard NeRF column-major OpenGL camera-to-world ($M_{\text{c2w}}$):
     $$\text{NeRFToRenderManTransform}[M_{\text{c2w}}] \quad \longleftrightarrow \quad \text{RenderManToNeRFTransform}[M_{\text{w2c}}]$$
   * Automatic convention detection in ray generation (`GenerateCameraRays`).
3. **Ultra-Lightweight Coordinate MLP (~10k Parameters)**:
   * 6-band harmonic positional encoder ($3 \rightarrow 39$ dims).
   * 2-layer MLP with 64 hidden units and `LogisticSigmoid` color activation.
   * Fully differentiable volumetric quadrature integration embedded inside `NetGraph`.
4. **Command-Line Suite & Interactive Demo Notebook**:
   * `notebooks/WLNeRFLiteDemo.nb`: Full interactive Mathematica demonstration notebook walking through dataset downloading, camera rig visualization, NetGraph compilation, model training, validation PSNR/MSE benchmarking, 360° turntable generation, and 3D density field sampling.
   * `scripts/convert_exr_to_nerf_json.wl`: Converts RenderMan EXRs to standard NeRF JSON folders with split generation (`train`, `val`, `test`).
   * `scripts/train_nerflite.wl`: Standalone CLI training with compact progress reporting, checkpoint export, and test turntable generation.
   * `scripts/download_lego_data.py`: High-speed CDN downloader for the canonical synthetic NeRF Lego benchmark dataset.
   * `scripts/grab_and_run_lego.sh`: One-command automated pipeline to grab the Lego dataset, train, and synthesize novel views.
5. **Interactive Visualization**:
   * `VisualizeNeRFCameraRig`: 3D frustum and camera orbit visualizer.
   * `RenderNeRFLiteTurntable`: 360-degree novel-view turntable synthesis with direct GIF export.

---

## Quick Start

### 1. Installation

Clone the repository and load the paclet in Wolfram Language / Mathematica:

```wolfram
PacletDirectoryLoad["/path/to/WLNeRFLite"];
Needs["WLNeRFLite`"];
```

### 2. Loading a Standard NeRF Dataset

```wolfram
(* Load synthetic NeRF dataset (e.g. Lego, Chair, or your custom export) *)
views = LoadNeRFDataset["/path/to/nerf_dataset/", "Split" -> "train"];

(* Inspect camera rig in 3D *)
VisualizeNeRFCameraRig[views]
```

### 3. Training a NeRFLite Model

```wolfram
(* 1. Initialize 10k-parameter coordinate MLP *)
fieldNet = NetInitialize @ CreateNeRFLiteNet[
  "NumSamples" -> 32,
  "HarmonicBands" -> 6,
  "HiddenDim" -> 64,
  "NumLayers" -> 2
];

(* 2. Train on GPU with compact logging *)
trainedModel = TrainNeRFLiteModel[
  fieldNet,
  views,
  "BatchSize" -> 1024,
  "MaxTrainingRounds" -> 5000,
  "LearningRate" -> 5*^-4,
  "TargetDevice" -> "GPU",
  "TrainingProgressReporting" -> "Compact"
];

(* 3. Render 360-degree orbiting turntable *)
RenderNeRFLiteTurntable[
  trainedModel,
  "NumFrames" -> 36,
  "Resolution" -> {256, 256},
  "ExportFile" -> "turntable.gif"
];
```

---

## Camera Matrix Conventions

NeRF and RenderMan use different coordinate systems and matrix storage formats:

| Property | RenderMan (`worldToCamera`) | NeRF Standard (`transform_matrix`) |
| :--- | :--- | :--- |
| **Direction** | World-to-Camera ($M_{\text{w2c}}$) | Camera-to-World ($M_{\text{c2w}}$) |
| **Storage** | Row-major ($v_{\text{cam}} = v_{\text{world}} \cdot M$) | Column-major ($P_{\text{world}} = M \cdot P_{\text{cam}}$) |
| **Look Axis** | Optical axis $+Z$ | Optical axis $-Z$ |
| **Right / Up** | $+X$ Right, $+Y$ Up | $+X$ Right, $+Y$ Up |
| **Translation** | Row 4 in inverse matrix $M^{-1}$ | Column 4: $[t_x, t_y, t_z, 1]^T$ |

`WLNeRFLite` handles this mapping in closed-form:

```wolfram
(* Convert RenderMan w2c to NeRF c2w *)
c2w = RenderManToNeRFTransform[w2c];

(* Convert NeRF c2w to RenderMan w2c *)
w2c = NeRFToRenderManTransform[c2w];
```

---

## Command-Line Tools

### Convert RenderMan EXRs to Canonical NeRF JSON

```bash
wolframscript -script scripts/convert_exr_to_nerf_json.wl \
  --input "/path/to/renders/scene_031_view_*.exr" \
  --output "/path/to/nerf_datasets/scene_031" \
  --split "0.8,0.1,0.1" \
  --format "PNG"
```

This generates:
* `transforms_train.json` (80% train split)
* `transforms_val.json` (10% validation split)
* `transforms_test.json` (10% test split)
* `transforms.json` (master unified catalog)
* `train/`, `val/`, `test/` PNG image directories

### Benchmark: Classic Lego Tractor Dataset

Download the canonical synthetic Lego tractor dataset and train/render in one command:

```bash
# Automated 1-step grab & run:
./scripts/grab_and_run_lego.sh --rounds 2000 --res 200 --turntable

# Or download the dataset separately:
python3 scripts/download_lego_data.py --out-dir data/lego --splits train val test

# Run training directly:
wolframscript -script scripts/train_nerflite.wl \
  --dataset "data/lego" \
  --rounds 3000 \
  --batch-size 1024 \
  --res 200 \
  --ray-bounds "2.0,6.0" \
  --output "output/lego_nerflite.wlnet" \
  --render-test "output/lego_test.png" \
  --render-turntable "output/lego_turntable.gif"
```

### Train NeRFLite from the Terminal

```bash
wolframscript -script scripts/train_nerflite.wl \
  --dataset "/path/to/nerf_datasets/scene_031" \
  --rounds 10000 \
  --batch-size 1024 \
  --device "GPU" \
  --output "checkpoints/nerflite_scene031.wlnet" \
  --render-test "output/test_recon.png" \
  --render-turntable "output/turntable.gif"
```

---

## Testing

Run the comprehensive unit test suite:

```bash
wolframscript -file Tests/RunTests.wl
```

All 11 tests verify bidirectional matrix inversion, ray equivalence, dataset export/import roundtrips, harmonic encoding, NetGraph evaluation, batch generation, and NeRF synthetic dataset loading.

---

## License

MIT License. Copyright (c) 2026 Flip Phillips / RIT MPS.
