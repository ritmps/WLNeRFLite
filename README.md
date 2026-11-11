# WLNeRFLite: Ultra-Lightweight Neural Radiance Fields in Wolfram Language

[![Wolfram Language](https://img.shields.io/badge/Wolfram_Language-13.0%2B-red.svg)](https://www.wolfram.com/language/)
[![NeRF Synthetic](https://img.shields.io/badge/NeRF-transforms.json_Compatible-brightgreen.svg)](#quick-start)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**`WLNeRFLite`** is a standalone, high-performance, ultra-lightweight Neural Radiance Field (NeRF) paclet implemented natively in Wolfram Language. It trains coordinate MLPs at high throughput (>25,000 rays/second on GPU) directly on standard NeRF datasets (`transforms.json`, `transforms_train.json`, `transforms_val.json`, `transforms_test.json`) with zero external Python or C++ dependencies.

---

## Key Features

1. **Pure Wolfram Language Pipeline**:
   * **Native Dataset Downloader (`DownloadNeRFDataset`)**: Downloads canonical NeRF synthetic benchmark scenes (`"lego"`, `"chair"`, `"drums"`, `"ficus"`, `"hotdog"`, `"materials"`, `"mic"`, `"ship"`) directly in Wolfram Language via concurrent `URLDownload`.
   * **Standard NeRF JSON Support (`LoadNeRFDataset`, `ExportNeRFDataset`)**: Compatible with canonical NeRF synthetic datasets (Mildenhall et al., ECCV 2020), Instant-NGP, Nerfstudio, and 3D Gaussian Splatting (`transforms.json` with PNG/JPG/TIFF images and $4 \times 4$ OpenGL camera-to-world pose matrices).
2. **Embedded Differentiable Volumetric Quadrature**:
   * 6- to 10-band harmonic positional encoder ($\gamma(\mathbf{x})$).
   * Configurable coordinate MLP with optional mid-network skip connections and `LogisticSigmoid` radiance head.
   * Continuous optical transmittance and alpha compositing embedded directly inside `NetGraph` via strict lower-triangular matrix multiplication (`trilStrict`), eliminating CPU-GPU transfer bottlenecks.
3. **High-Throughput Ray Pooling & Stratified Sampling**:
   * `PrecomputeNeRFLiteRayPool` and `CreateNeRFLiteBatchGenerator` precompute world-space rays across views and balance foreground/background ray sampling using alpha masks.
4. **Interactive Visualization & Volume Extraction**:
   * `VisualizeNeRFCameraRig`: 3D camera frustum and orbital pose visualizer.
   * `RenderNeRFLiteView` & `RenderNeRFLiteTurntable`: Novel-view synthesis and 360-degree orbital turntable animation with direct GIF export.
   * `SampleNeRFLiteDensityVolume`: Evaluates the learned 3D scalar density field $\sigma(x, y, z)$ on a Euclidean voxel grid for `Image3D` and `ListContourPlot3D` isosurface extraction.
   * `notebooks/LegoNeRFDemo.nb`: Complete interactive Mathematica demonstration notebook walking through dataset ingestion, camera rig inspection, `NetGraph` compilation, training, validation PSNR/MSE benchmarking, 360° turntable synthesis, and 3D density volume extraction.

---

## Quick Start

### 1. Installation

Clone the repository and load the paclet in Wolfram Language / Mathematica:

```wolfram
PacletDirectoryLoad["/path/to/WLNeRFLite"];
Needs["WLNeRFLite`"];
```

### 2. Download & Load a Standard NeRF Dataset

```wolfram
(* Download the canonical NeRF Lego bulldozer dataset directly in Wolfram Language *)
DownloadNeRFDataset["data/lego", "Scene" -> "lego", "Splits" -> {"train", "val"}];

(* Load training views (downsampled to 200x200 with white background) *)
trainViews = LoadNeRFDataset[
  "data/lego",
  "Split" -> "train",
  "TargetDimensions" -> {200, 200},
  "WhiteBackground" -> True
];

(* Inspect 3D camera rig and viewing frustums *)
VisualizeNeRFCameraRig[trainViews]
```

### 3. Train a NeRFLite Model

```wolfram
(* 1. Initialize coordinate MLP with embedded volumetric quadrature *)
fieldNet = NetInitialize @ CreateNeRFLiteNet[
  "NumSamples" -> 64,
  "HarmonicBands" -> 10,
  "HiddenDim" -> 128,
  "NumLayers" -> 4,
  "BackgroundColor" -> {1., 1., 1.},
  "ColorActivation" -> LogisticSigmoid
];

(* 2. Train on GPU *)
trainedModel = TrainNeRFLiteModel[
  fieldNet,
  trainViews,
  "BatchSize" -> 2048,
  "MaxTrainingRounds" -> 3000,
  "LearningRate" -> 1*^-3,
  "Loss" -> "L2",
  "RayBounds" -> {2.0, 6.0},
  "SampleForegroundRatio" -> 0.5,
  "TargetDevice" -> Automatic
];
```

### 4. Novel-View Synthesis, Turntable & 3D Density Extraction

```wolfram
(* Render 360-degree orbiting turntable *)
turntableFrames = RenderNeRFLiteTurntable[
  trainedModel,
  "NumFrames" -> 36,
  "Resolution" -> {200, 200},
  "Radius" -> 4.0,
  "Elevation" -> 30.0 * Degree,
  "UpAxis" -> "Z",
  "RayBounds" -> {2.0, 6.0},
  "ExportFile" -> "lego_turntable.gif"
];

(* Sample continuous 3D density volume for isosurface & Image3D visualization *)
densityGrid = SampleNeRFLiteDensityVolume[
  trainedModel,
  "Bounds" -> {{-1.2, 1.2}, {-1.2, 1.2}, {-1.2, 1.2}},
  "Resolution" -> 64
];
Image3D[Rescale[densityGrid], ColorFunction -> "SunsetColors"]
```

---

## Paclet API Overview

| Context | Symbol | Description |
| :--- | :--- | :--- |
| `WLNeRFLite`Data`` | `DownloadNeRFDataset` | Downloads canonical synthetic NeRF benchmark datasets (`"lego"`, `"chair"`, etc.) via `URLDownload` |
| `WLNeRFLite`Data`` | `LoadNeRFDataset` | Loads `transforms.json` / `transforms_<split>.json` datasets and camera-to-world (`c2w`) matrices |
| `WLNeRFLite`Data`` | `ExportNeRFDataset` | Exports camera views and images to canonical `transforms.json` format with `train`/`val`/`test` splits |
| `WLNeRFLite`Data`` | `GenerateCameraRays` | Generates ray origins and normalized world directions from $4 \times 4$ `c2w` camera matrices |
| `WLNeRFLite`Data`` | `SampleRaysStratified` | Computes stratified 3D sample points and integration step intervals along rays |
| `WLNeRFLite`Data`` | `PrecomputeNeRFLiteRayPool` | Precomputes in-memory foreground and background ray pools across training views |
| `WLNeRFLite`Net`` | `CreateHarmonicEncoder` | Constructs a differentiable Fourier positional encoding network $\gamma(\mathbf{x})$ |
| `WLNeRFLite`Net`` | `CreateNeRFLiteNet` | Constructs a coordinate MLP `NetGraph` with embedded volumetric quadrature integration |
| `WLNeRFLite`Net`` | `CreateNeRFLiteTrainingNet` | Wraps a NeRF prediction `NetGraph` with L2/L1 photometric loss and optional opacity regularization |
| `WLNeRFLite`Net`` | `ExtractNeRFLiteDensityNet` | Extracts the point-wise density evaluation sub-network ($\mathbb{R}^3 \to \mathbb{R}^+$) |
| `WLNeRFLite`Net`` | `ExtractNeRFLiteColorNet` | Extracts the point-wise RGB evaluation sub-network ($\mathbb{R}^3 \to [0,1]^3$) |
| `WLNeRFLite`Trainer`` | `CreateNeRFLiteBatchGenerator` | Creates a stateful stratified ray batch generator for `NetTrain` |
| `WLNeRFLite`Trainer`` | `TrainNeRFLiteModel` | Trains a NeRFLite model with `NetTrain`, compact progress reporting, and checkpointing |
| `WLNeRFLite`Trainer`` | `ExtractNeRFLiteInferenceNet` | Extracts the trained inference `NetGraph` from a training loss graph |
| `WLNeRFLite`Visualizer`` | `RenderNeRFLiteView` | Synthesizes a 2D novel view (RGB, Opacity, Depth) from a $4 \times 4$ camera matrix |
| `WLNeRFLite`Visualizer`` | `RenderNeRFLiteTurntable` | Renders a 360-degree orbital turntable animation around the target |
| `WLNeRFLite`Visualizer`` | `SampleNeRFLiteDensityVolume` | Samples the learned 3D density field across a regular voxel grid |
| `WLNeRFLite`Visualizer`` | `VisualizeNeRFCameraRig` | Renders an interactive `Graphics3D` plot of camera positions and viewing frustums |

---

## Testing

Run the unit test suite via `wolframscript`:

```bash
wolframscript -file Tests/RunTests.wl
```

---

## License

MIT License. Copyright (c) 2026 Flip Phillips / RIT MPS Lab.
