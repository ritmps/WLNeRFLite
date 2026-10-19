# WLNeRFLite Developer Walkthrough & API Guide

This walkthrough provides an in-depth reference for using, extending, and training Neural Radiance Fields with **`WLNeRFLite`**.

---

## 1. Package Architecture

```
WLNeRFLite/
├── PacletInfo.wl                 # Paclet definition (WLNeRFLite v1.2.0)
├── README.md                     # Overview & Quickstart
├── WALKTHROUGH.md                # In-depth technical architecture
├── .gitignore                    # Git tracking rules
├── Kernel/
│   ├── init.wl                   # Package entry point & top-level exports
│   ├── NeRFLiteData.wl           # Transforms, EXR binary header reader, NeRF JSON loader/exporter
│   ├── NeRFLiteNet.wl            # Harmonic encoder, coordinate MLP, differentiable volume renderer
│   ├── NeRFLiteTrainer.wl        # High-throughput ray pooling, batch generator, training loop
│   ├── NeRFLiteVisualizer.wl     # Novel-view rendering, 360 turntables, 3D camera rig visualizer
│   └── MultiLPENeRF.wl           # 9-pass radiometric and differential geometric NeRF
├── scripts/
│   ├── convert_exr_to_nerf_json.wl  # CLI tool: RenderMan EXRs -> transforms.json + PNGs
│   └── train_nerflite.wl            # CLI tool: Standalone model trainer & turntable generator
└── Tests/
    ├── RunTests.wl               # Test runner
    └── NeRFLite.wlt              # 10-point unit test suite
```

---

## 2. Mathematical Formalism

### 2.1 Coordinate Conventions & Transforms

* **RenderMan $M_{\text{w2c}}$**:
  Row-vector multiplication convention:
  $$\mathbf{p}_{\text{cam}} = \mathbf{p}_{\text{world}} \cdot M_{\text{w2c}}$$
  The inverse $M_{\text{c2w}}^{\text{RMan}} = (M_{\text{w2c}})^{-1}$ stores:
  * Row 1: Right direction $\mathbf{r}_x$
  * Row 2: Up direction $\mathbf{r}_y$
  * Row 3: Look direction $\mathbf{r}_z$ (optical axis pointing into scene)
  * Row 4: Camera world position $\mathbf{t}$

* **NeRF / OpenGL $M_{\text{c2w}}$**:
  Column-vector multiplication convention:
  $$\mathbf{p}_{\text{world}} = M_{\text{c2w}}^{\text{NeRF}} \cdot \mathbf{p}_{\text{cam}}$$
  In OpenGL / NeRF convention:
  * Column 1: Right $\mathbf{r}_x$
  * Column 2: Up $\mathbf{r}_y$
  * Column 3: Back $-\mathbf{r}_z$ (camera looks along $-Z$)
  * Column 4: Camera world position $\mathbf{t}$
  * Row 4: $[0, 0, 0, 1]$

* **Exact Closed-Form Mapping**:
  $$\text{RenderManToNeRFTransform}[M_{\text{w2c}}] = (M_{\text{w2c}}^{-1})^T \cdot \text{diag}(1, 1, -1, 1)$$
  $$\text{NeRFToRenderManTransform}[M_{\text{c2w}}] = \left( (M_{\text{c2w}} \cdot \text{diag}(1, 1, -1, 1))^T \right)^{-1}$$

Both transformations are guaranteed to roundtrip to machine precision ($< 10^{-12}$).

---

## 3. Data Ingestion & Export

### 3.1 Loading Standard NeRF Datasets

```wolfram
(* Load canonical NeRF dataset (train split) *)
views = LoadNeRFDataset["/path/to/dataset", "Split" -> "train"];

(* Load all splits into a single unified list *)
allViews = LoadNeRFDataset["/path/to/dataset", "Split" -> "all"];

(* First view metadata *)
views[[1]]
(* Output:
<|
  "CameraMatrix" -> ...,       (* 4x4 RenderMan-compatible w2c *)
  "NeRFMatrix" -> ...,         (* 4x4 standard NeRF c2w *)
  "FOV" -> 48.0,               (* Field of view in degrees *)
  "Dimensions" -> {512, 512},  (* Width, Height *)
  "RGB" -> ...,                (* HxWx3 PackedArray in [0, 1] *)
  "Alpha" -> ...,              (* HxW PackedArray or None *)
  "ForegroundPixels" -> ...    (* Extracted foreground coordinates *)
|>
*)
```

### 3.2 Exporting RenderMan EXRs to NeRF JSON

```wolfram
views = LoadRenderManEXRViews["/path/to/renders/scene_*.exr"];

res = ExportNeRFDataset[
  views,
  "/path/to/output_nerf_dataset",
  "Split" -> <|"Train" -> 0.8, "Val" -> 0.1, "Test" -> 0.1|>,
  "ImageFormat" -> "PNG",
  "WhiteBackground" -> False
];
```

---

## 4. Model Training & Volume Rendering

### 4.1 Embedded Volumetric Quadrature

The continuous radiative transfer equation along ray $\mathbf{r}(t) = \mathbf{o} + t\mathbf{d}$ is:
$$\hat{\mathbf{C}}(\mathbf{r}) = \sum_{i=1}^N T_i \, \alpha_i \, \mathbf{c}_i, \quad T_i = \exp\left(-\sum_{j=1}^{i-1} \sigma_j \, \delta_j\right), \quad \alpha_i = 1 - \exp(-\sigma_i \, \delta_i)$$

In `WLNeRFLite`, this integration is embedded directly within the MXNet computation graph using a strict lower-triangular matrix layer (`trilStrict`), eliminating CPU-GPU memory transfers.

### 4.2 Training Loop

```wolfram
(* Create model and initialize *)
net = NetInitialize @ CreateNeRFLiteNet[
  "NumSamples" -> 32,
  "HarmonicBands" -> 6,
  "HiddenDim" -> 64,
  "NumLayers" -> 2
];

(* High-speed GPU training *)
trained = TrainNeRFLiteModel[
  net,
  views,
  "BatchSize" -> 1024,
  "MaxTrainingRounds" -> 5000,
  "TargetDevice" -> "GPU",
  "TrainingProgressReporting" -> "Compact"
];
```

---

## 5. Visualizations

### 5.1 3D Camera Rig

```wolfram
VisualizeNeRFCameraRig[views, "FrustumScale" -> 0.4]
```

Renders a 3D plot displaying:
* World origin axes ($X$ Red, $Y$ Green, $Z$ Blue).
* Camera centers colored by orbit order.
* Pyramidal view frustums oriented along the true optical axes.

### 5.2 360-Degree Turntable Synthesis

```wolfram
RenderNeRFLiteTurntable[
  trained,
  "NumFrames" -> 36,
  "Resolution" -> {256, 256},
  "TargetDevice" -> "GPU",
  "ExportFile" -> "orbit.gif"
]
```
