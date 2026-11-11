# WLNeRFLite Agent Instructions

## fp - fall 2026
## guidance for the `WLNeRFLite` Wolfram Language paclet

* You are a Wolfram Language agent tasked with maintaining and improving the `WLNeRFLite` paclet, an ultra-lightweight Neural Radiance Field (NeRF) engine.
* The `WLNeRFLite` Wolfram Language paclet is an ultra-lightweight Neural Radiance Field (NeRF) engine with GPU-embedded differentiable volumetric quadrature and standard `transforms.json` dataset support.

## Core Architecture

* **`WLNeRFLite`Data`` (`Kernel/NeRFLiteData.wl`)**:
  - `DownloadNeRFDataset`: Pure Wolfram Language downloader (`URLDownload`) for canonical NeRF synthetic datasets (`"lego"`, `"chair"`, `"drums"`, `"ficus"`, `"hotdog"`, `"materials"`, `"mic"`, `"ship"`).
  - `LoadNeRFDataset` / `ExportNeRFDataset`: Ingestion and export of standard `transforms.json` datasets using $4 \times 4$ OpenGL camera-to-world (`c2w`: $+X$ right, $+Y$ up, $-Z$ look) matrices.
  - `GenerateCameraRays`, `SampleRaysStratified`, `PrecomputeNeRFLiteRayPool`: Vectorized and compiled ray emission, stratified sampling, and foreground/background ray pooling.
* **`WLNeRFLite`Net`` (`Kernel/NeRFLiteNet.wl`)**:
  - `CreateHarmonicEncoder`, `CreateNeRFLiteNet`, `CreateNeRFLiteTrainingNet`, `ExtractNeRFLiteDensityNet`, `ExtractNeRFLiteColorNet`.
  - Volumetric transmittance and alpha compositing are computed inside `NetGraph` using a strict lower-triangular cumulative sum layer (`trilStrict`).
* **`WLNeRFLite`Trainer`` (`Kernel/NeRFLiteTrainer.wl`)**:
  - `CreateNeRFLiteBatchGenerator`, `TrainNeRFLiteModel`, `ExtractNeRFLiteInferenceNet`.
* **`WLNeRFLite`Visualizer`` (`Kernel/NeRFLiteVisualizer.wl`)**:
  - `RenderNeRFLiteView`, `RenderNeRFLiteTurntable`, `SampleNeRFLiteDensityVolume`, `VisualizeNeRFCameraRig`.

## Coding & Version Standards

* Keep the paclet pure Wolfram Language (`Kernel/*.wl`, `Documentation/`, `Tests/`, `notebooks/`), free of external Python scripts or renderer-specific dependencies.
* Prefer idiomatic, functional, vectorized Wolfram Language (`NumericArray`, `Developer`ToPackedArray`).
* Ensure every text, code, and JSON file ends with a trailing newline `\n`.
* Track all Wolfram Notebooks (`*.nb` in `Documentation/` and `notebooks/`) using Git LFS via `.gitattributes`.
* The `.githooks/pre-commit` script (`wolframscript`) automatically increments the build number (`Major.Minor.Build`) across `PacletInfo.wl` and `Kernel/init.wl` on commit.
