(* ::Package:: *)
(* scripts/build_demo_notebook.wl
   Generates notebooks/WLNeRFLiteDemo.nb with native FrontEnd BoxData formatting.
*)

repoRoot = DirectoryName[ExpandFileName[If[StringQ[$InputFileName] && $InputFileName =!= "", $InputFileName, "scripts/build_demo_notebook.wl"]], 2];
PacletDirectoryLoad[repoRoot];
Needs["WLNeRFLite`"];

Print["Building WLNeRFLite Demonstration Notebook..."];

createInputCell[str_String] := Module[{held, boxes},
  held = MakeExpression[str, StandardForm];
  boxes = UsingFrontEnd[MakeBoxes @@ held];
  Cell[BoxData[boxes], "Input"]
];

createTextCell[str_String] := Cell[str, "Text"];
createSectionCell[str_String] := Cell[str, "Section"];
createSubsectionCell[str_String] := Cell[str, "Subsection"];

cells = {
  Cell[CellGroupData[{
    Cell["WLNeRFLite: Neural Radiance Field Demo (Lego Bulldozer)", "Title"],
    Cell["Interactive Reconstruction & Volume Rendering in Wolfram Language", "Subtitle"],
    createTextCell["This interactive demonstration walks through the end-to-end pipeline of training and rendering a Neural Radiance Field (NeRF) on the canonical synthetic Lego bulldozer dataset (Mildenhall et al., ECCV 2020) using WLNeRFLite.\n\nThe entire process\[LongDash]from dataset downloading and 3D camera rig inspection, to differentiable volumetric NetGraph accumulation, high-throughput ray training, validation benchmarking, 360\[Degree] turntable synthesis, and continuous 3D density field sampling\[LongDash]runs natively in Wolfram Language."]
  }, Open]],

  (* Section 1 *)
  Cell[CellGroupData[{
    createSectionCell["1. Package Initialization & Hardware Detection"],
    createTextCell["Load the WLNeRFLite paclet and detect hardware capabilities. On macOS Apple Silicon (M1/M2/M3/M4), native vectorized CPU acceleration is leveraged for optimal NetTrain throughput; on Linux workstations, NVIDIA CUDA GPU acceleration is automatically targeted."],
    createInputCell["repoRoot = Which[
  StringQ[$NotebookFileName] && $NotebookFileName =!= \"\",
    DirectoryName[ExpandFileName[$NotebookFileName], 2],
  DirectoryQ[\"Kernel\"] && FileExistsQ[\"PacletInfo.wl\"],
    Directory[],
  True,
    FileNameJoin[{ParentDirectory[Directory[]], \"WLNeRFLite\"}]
];

PacletDirectoryLoad[repoRoot];
Needs[\"WLNeRFLite`\"];

Print[\"WLNeRFLite Loaded from: \", repoRoot];
Print[\"Wolfram Kernel: \", $Version];
Print[\"System ID: \", $SystemID];"],
    createInputCell["device = Which[
  $SystemID === \"MacOSX-ARM64\", \"CPU\",
  True, \"Automatic\"
];

Print[\"Selected TargetDevice: \", device];"]
  }, Open]],

  (* Section 2 *)
  Cell[CellGroupData[{
    createSectionCell["2. Download Classic NeRF Lego Bulldozer Dataset"],
    createTextCell["The synthetic NeRF benchmark (Mildenhall et al., ECCV 2020) provides multi-view rendering of complex geometry with realistic specularities, occlusion, and alpha transparency.\n\nWe download the Lego bulldozer dataset (camera transforms JSON files and ground truth frames) directly from high-speed CDN mirrors using the repository's built-in downloader script:"],
    createInputCell["dataDir = FileNameJoin[{repoRoot, \"data\", \"lego\"}];
downloadScript = FileNameJoin[{repoRoot, \"scripts\", \"download_lego_data.py\"}];

(* Download train and validation splits if not already cached locally *)
If[!FileExistsQ[FileNameJoin[{dataDir, \"transforms_train.json\"}]] ||
   !DirectoryQ[FileNameJoin[{dataDir, \"train\"}]],
  Print[\"Downloading Lego bulldozer dataset into \", dataDir, \"...\"];
  RunProcess[{
    \"python3\",
    downloadScript,
    \"--out-dir\", dataDir,
    \"--splits\", \"train\", \"val\",
    \"--workers\", \"16\"
  }],
  Print[\"Lego dataset already cached at: \", dataDir]
];

FileNames[\"*.json\", dataDir]"]
  }, Open]],

  (* Section 3 *)
  Cell[CellGroupData[{
    createSectionCell["3. Dataset Ingestion & 3D Camera Rig Visualization"],
    createTextCell["LoadNeRFDataset automatically parses transforms_train.json, normalizes coordinate spaces between NeRF (OpenGL +X right, +Y up, -Z look) and RenderMan row-major conventions, downsamples images to a responsive training resolution (200x200), and composites against a clean background."],
    createInputCell["(* Ingest training split (downsampled to 200x200 for responsive interactive training) *)
trainViews = LoadNeRFDataset[
  dataDir,
  \"Split\" -> \"train\",
  \"TargetDimensions\" -> {200, 200},
  \"WhiteBackground\" -> True
];

Print[\"Loaded \", Length[trainViews], \" training views.\"];
Print[\"Resolution: \", trainViews[[1]][\"Dimensions\"], \" | Camera FOV: \", Round[trainViews[[1]][\"FOV\"], 0.1], \" deg\"];
Print[\"First view metadata keys: \", Keys[trainViews[[1]]]];"],
    createInputCell["(* Inspect ground-truth views: sample RGB frames and foreground alpha mask *)
ImageAssemble[{{
  Image[trainViews[[1]][\"RGB\"], \"Real32\", ColorSpace -> \"RGB\"],
  Image[trainViews[[25]][\"RGB\"], \"Real32\", ColorSpace -> \"RGB\"],
  Image[trainViews[[50]][\"RGB\"], \"Real32\", ColorSpace -> \"RGB\"],
  Image[trainViews[[1]][\"Alpha\"], \"Real32\"]
}}]"],
    createTextCell["Visualize the hemispherical 3D camera rig surrounding the Lego bulldozer with pyramidal view frustums pointing along the true optical axes:"],
    createInputCell["(* 3D Camera Rig Visualization: Hemisphere camera distribution surrounding the Lego bulldozer *)
VisualizeNeRFCameraRig[trainViews, \"FrustumScale\" -> 0.35]"]
  }, Open]],

  (* Section 4 *)
  Cell[CellGroupData[{
    createSectionCell["4. Differentiable Volumetric NetGraph Architecture"],
    createTextCell["In classical NeRF, continuous volumetric radiative transfer along each camera ray r(t) = o + t d is discretized into numerical quadrature:\n\n   C(r) = Sum[T_i alpha_i c_i, {i, 1, N}]\n   T_i = Exp[-Sum[sigma_j delta_j, {j, 1, i-1}]]\n   alpha_i = 1 - Exp[-sigma_i delta_i]\n\nWLNeRFLite embeds this numerical integration directly into the MXNet computation graph using a strict lower-triangular matrix layer (trilStrict). Rays and samples are evaluated entirely on the target device with zero CPU-GPU transfer bottlenecks."],
    createInputCell["(* Instantiate coordinate MLP with harmonic positional encoding and embedded quadrature *)
fieldNet = NetInitialize @ CreateNeRFLiteNet[
  \"NumSamples\" -> 32,
  \"HarmonicBands\" -> 8,
  \"HiddenDim\" -> 96,
  \"NumLayers\" -> 3,
  \"BackgroundColor\" -> {1., 1., 1.},
  \"ColorActivation\" -> LogisticSigmoid
]"]
  }, Open]],

  (* Section 5 *)
  Cell[CellGroupData[{
    createSectionCell["5. NeRF Model Training"],
    createTextCell["Train the coordinate field using TrainNeRFLiteModel. Standard synthetic NeRF objects are centered at the origin with camera distances around 4.0 units, corresponding to ray near/far bounds of [2.0, 6.0]. We sample 70% of training rays from foreground regions to accelerate reconstruction of fine geometric details."],
    createInputCell["(* Train the NeRF model using NetTrain with real-time interactive panel *)
tStart = AbsoluteTime[];

trainedModel = TrainNeRFLiteModel[
  fieldNet,
  trainViews,
  \"BatchSize\" -> 1024,
  \"MaxTrainingRounds\" -> 2000,
  \"LearningRate\" -> 0.001,
  \"Loss\" -> \"L2\",
  \"RayBounds\" -> {2.0, 6.0},
  \"SampleForegroundRatio\" -> 0.7,
  \"TargetDevice\" -> device,
  \"TrainingProgressReporting\" -> \"Panel\"
];

trainSec = Round[AbsoluteTime[] - tStart, 0.1];
Print[\"Training completed in \", trainSec, \" seconds.\"];"],
    createInputCell["(* Export trained model checkpoint *)
outputDir = FileNameJoin[{repoRoot, \"output\"}];
If[!DirectoryQ[outputDir], CreateDirectory[outputDir, CreateIntermediateDirectories -> True]];

Export[FileNameJoin[{outputDir, \"lego_nerflite.wlnet\"}], trainedModel];
Export[FileNameJoin[{outputDir, \"lego_nerflite.wxf\"}], trainedModel, \"WXF\"];
Print[\"Checkpoint saved to: \", FileNameJoin[{outputDir, \"lego_nerflite.wlnet\"}]];"]
  }, Open]],

  (* Section 6 *)
  Cell[CellGroupData[{
    createSectionCell["6. Novel-View Synthesis & Validation Benchmarking"],
    createTextCell["Evaluate the learned volumetric representation on held-out validation camera poses never seen during training. RenderNeRFLiteView traces rays through the learned density and color fields to synthesize novel views."],
    createInputCell["(* Load held-out validation views *)
valViews = LoadNeRFDataset[
  dataDir,
  \"Split\" -> \"val\",
  \"TargetDimensions\" -> {200, 200},
  \"WhiteBackground\" -> True
];

testView = valViews[[1]];

(* Render novel view from camera matrix *)
novelViewRecon = RenderNeRFLiteView[
  trainedModel,
  testView[\"CameraMatrix\"],
  testView[\"Dimensions\"],
  \"FOV\" -> testView[\"FOV\"],
  \"RayBounds\" -> {2.0, 6.0},
  \"TargetDevice\" -> device
];

gtImg = Image[testView[\"RGB\"], \"Real32\", ColorSpace -> \"RGB\"];
diffImg = Image[Clip[3.0 * Abs[ImageData[novelViewRecon] - ImageData[gtImg]], {0.0, 1.0}]];
mse = Mean[Flatten[(ImageData[novelViewRecon] - ImageData[gtImg])^2]];
psnr = If[mse > 0, Round[-10.0 * Log10[mse], 0.01], Infinity];

Print[\"Validation Reconstruction PSNR: \", psnr, \" dB | MSE: \", NumberForm[mse, {5, 4}]];

(* Side-by-side comparison: Ground Truth | NeRF Reconstruction | 3x Difference Error *)
ImageAssemble[{{
  Labeled[gtImg, \"Ground Truth (Validation View 0)\"],
  Labeled[novelViewRecon, StringTemplate[\"NeRF Reconstruction (`` dB)\"][psnr]],
  Labeled[diffImg, \"Error Difference (3\[Times] Amplified)\"]
}}]"]
  }, Open]],

  (* Section 7 *)
  Cell[CellGroupData[{
    createSectionCell["7. 360-Degree Turntable Orbit Animation"],
    createTextCell["Render a continuous 360-degree orbital turntable around the reconstructed Lego bulldozer using RenderNeRFLiteTurntable. The camera follows a circular orbit with 30\[Degree] elevation around the Z-up axis."],
    createInputCell["(* Render 360-degree novel-view circular turntable orbit (36 frames) *)
turntableFrames = RenderNeRFLiteTurntable[
  trainedModel,
  \"NumFrames\" -> 36,
  \"Resolution\" -> {200, 200},
  \"Radius\" -> 4.0,
  \"Elevation\" -> 30.0 Degree,
  \"Target\" -> {0., 0., 0.},
  \"UpAxis\" -> \"Z\",
  \"FOV\" -> testView[\"FOV\"],
  \"RayBounds\" -> {2.0, 6.0},
  \"TargetDevice\" -> device,
  \"ExportFile\" -> FileNameJoin[{outputDir, \"lego_turntable.gif\"}]
];

Print[\"Turntable rendered: \", Length[turntableFrames], \" frames.\"];
ListAnimate[turntableFrames]"]
  }, Open]],

  (* Section 8 *)
  Cell[CellGroupData[{
    createSectionCell["8. 3D Volumetric Density Sampling & Isosurface Visualization"],
    createTextCell["Because NeRF models a continuous 5D neural field (position + view direction), we can probe the underlying continuous density field \[Sigma](x, y, z) directly in 3D Euclidean space using SampleNeRFLiteDensityVolume without casting camera rays."],
    createInputCell["(* Sample continuous 3D density field on a 48x48x48 Euclidean voxel grid *)
densityVolume = SampleNeRFLiteDensityVolume[
  trainedModel,
  \"Resolution\" -> 48,
  \"Bounds\" -> {{-1.2, 1.2}, {-1.2, 1.2}, {-1.2, 1.2}},
  \"TargetDevice\" -> device
];

Print[\"Sampled density volume dimensions: \", Dimensions[densityVolume]];"],
    createInputCell["(* 3D Density Isosurface Reconstruction *)
ListContourPlot3D[
  densityVolume,
  Contours -> {0.3, 0.8, 1.5},
  ContourStyle -> {
    Directive[Orange, Opacity[0.5]],
    Directive[Yellow, Opacity[0.75]],
    Directive[Red, Opacity[0.9]]
  },
  Mesh -> None,
  Boxed -> True,
  Axes -> True,
  AxesLabel -> {\"X\", \"Y\", \"Z\"},
  PlotLabel -> Style[\"NeRF Learned 3D Density Field Isosurfaces\", Bold, 14]
]"],
    createInputCell["(* Volumetric 3D Density Rendering *)
Image3D[
  Rescale[densityVolume],
  ColorFunction -> \"SunsetColors\",
  Boxed -> True
]"]
  }, Open]]
};

outNbPath = FileNameJoin[{repoRoot, "notebooks", "WLNeRFLiteDemo.nb"}];
If[!DirectoryQ[DirectoryName[outNbPath]], CreateDirectory[DirectoryName[outNbPath], CreateIntermediateDirectories -> True]];

nb = Notebook[
  cells,
  WindowTitle -> "WLNeRFLite: Neural Radiance Field Demo (Lego Bulldozer)",
  StyleDefinitions -> "Default.nb"
];

Export[outNbPath, nb];
Print["Successfully generated demo notebook: ", outNbPath];
