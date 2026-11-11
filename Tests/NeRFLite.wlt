(* ::Package:: *)

PacletDirectoryLoad[DirectoryName[$TestFileName, 2]];
Needs["WLNeRFLite`"];

VerificationTest[
  MemberQ[$Packages, "WLNeRFLite`"] && StringQ[$WLNeRFLiteVersion],
  True,
  TestID -> "NeRFLite-PackageLoad"
]

(* 1. Standard NeRF Camera-to-World Ray Generation Test *)
VerificationTest[
  c2w = {
    {1., 0., 0., 2.5},
    {0., 1., 0., -3.0},
    {0., 0., 1., 7.0},
    {0., 0., 0., 1.}
  };
  rays = GenerateCameraRays[c2w, {64, 64}, "FOV" -> 50.0, "PixelCoords" -> {{32, 32}}];
  Max[Abs[rays["CameraPosition"] - {2.5, -3.0, 7.0}]] < 1*^-12 &&
  Max[Abs[rays["Origins"][[1]] - {2.5, -3.0, 7.0}]] < 1*^-12 &&
  rays["Directions"][[1, 3]] < -0.99,
  True,
  TestID -> "NeRFLite-GenerateCameraRays"
]

(* 2. Stratified Ray Sampling Test *)
VerificationTest[
  origs = {{0., 0., 5.}, {1., 1., 5.}};
  dirs = {{0., 0., -1.}, {0., 0., -1.}};
  sampled = SampleRaysStratified[origs, dirs, 32, {2.0, 6.0}, "Perturb" -> False];
  Dimensions[sampled["SamplePoints"]] === {2, 32, 3} &&
  Dimensions[sampled["Steps"]] === {2, 32} &&
  Abs[sampled["Steps"][[1, 1]] - (4.0 / 32)] < 1*^-5,
  True,
  TestID -> "NeRFLite-SampleRaysStratified"
]

(* 3. End-to-End Dataset Export and Load (transforms.json + PNG) *)
VerificationTest[
  testDir = FileNameJoin[{$TemporaryDirectory, "wlnerflite_test_dataset_" <> IntegerString[RandomInteger[{10000, 99999}]]}];
  syntheticViews = Table[
    Module[{
      c2w = {
        {Cos[phi], 0., Sin[phi], 4.0 * Sin[phi]},
        {0., 1., 0., 0.},
        {-Sin[phi], 0., Cos[phi], 4.0 * Cos[phi]},
        {0., 0., 0., 1.}
      },
      imgData = Table[{0.15 * i, 0.4, 0.8}, {y, 32}, {x, 32}],
      alphaData = Table[If[(x - 16)^2 + (y - 16)^2 < 100, 1.0, 0.0], {y, 32}, {x, 32}]
    },
      <|
        "CameraMatrix" -> Developer`ToPackedArray[N[c2w]],
        "FOV" -> 45.0,
        "Dimensions" -> {32, 32},
        "RGB" -> Developer`ToPackedArray[N[imgData]],
        "Alpha" -> Developer`ToPackedArray[N[alphaData]],
        "ForegroundPixels" -> {{16, 16}, {15, 15}}
      |>
    ],
    {i, 5},
    {phi, {0.0, 0.5, 1.0, 1.5, 2.0}[[i ;; i]]}
  ];
  syntheticViews = Flatten[syntheticViews];
  exportRes = ExportNeRFDataset[syntheticViews, testDir, "Split" -> <|"Train" -> 0.6, "Val" -> 0.2, "Test" -> 0.2|>];
  loadedViews = LoadNeRFDataset[testDir, "Split" -> "all"];
  ok = AssociationQ[exportRes] &&
       FileExistsQ[exportRes["JSONFiles", "master"]] &&
       Length[loadedViews] == 5 &&
       loadedViews[[1]]["Dimensions"] === {32, 32} &&
       Abs[loadedViews[[1]]["FOV"] - 45.0] < 1*^-3 &&
       MatrixQ[loadedViews[[1]]["CameraMatrix"], NumericQ] &&
       Dimensions[loadedViews[[1]]["RGB"]] === {32, 32, 3};
  DeleteDirectory[testDir, DeleteContents -> True];
  ok,
  True,
  TestID -> "NeRFLite-ExportAndLoadNeRFDataset"
]

(* 4. Harmonic Positional Encoder Test *)
VerificationTest[
  encoder = CreateHarmonicEncoder[6];
  initEnc = NetInitialize[encoder];
  out = initEnc[{0.1, 0.2, 0.3}];
  Dimensions[out] === {39} && AllTrue[out, NumericQ],
  True,
  TestID -> "NeRFLite-HarmonicEncoder"
]

(* 5. NeRFLite NetGraph Architecture & Evaluation *)
VerificationTest[
  net = CreateNeRFLiteNet["NumSamples" -> 16, "HarmonicBands" -> 4, "HiddenDim" -> 32, "NumLayers" -> 2];
  initNet = NetInitialize[net];
  testPts = RandomReal[{-1., 1.}, {4, 16, 3}];
  testSteps = ConstantArray[0.1, {4, 16}];
  evalRes = initNet[<|"SamplePoints" -> NumericArray[testPts, "Real32"], "Steps" -> NumericArray[testSteps, "Real32"]|>, TargetDevice -> "CPU"];
  rgbArr = Normal[evalRes["RGB"]];
  opacArr = Normal[evalRes["Opacity"]];
  AssociationQ[evalRes] &&
  Dimensions[rgbArr] === {4, 3} &&
  Dimensions[opacArr] === {4, 1} &&
  AllTrue[Flatten[rgbArr], 0.0 <= # <= 1.0 &],
  True,
  TestID -> "NeRFLite-NetGraphEvaluation"
]

(* 6. Subnet Extraction & 3D Density Volume Sampling Test *)
VerificationTest[
  net = NetInitialize @ CreateNeRFLiteNet["NumSamples" -> 8, "HarmonicBands" -> 2, "HiddenDim" -> 16, "NumLayers" -> 2];
  densityNet = ExtractNeRFLiteDensityNet[net];
  colorNet = ExtractNeRFLiteColorNet[net];
  vol = SampleNeRFLiteDensityVolume[net, "Bounds" -> {{-1., 1.}, {-1., 1.}, {-1., 1.}}, "Resolution" -> 8, "TargetDevice" -> "CPU"];
  img3d = SampleNeRFLiteDensityVolume[densityNet, {{-1., 1.}, {-1., 1.}, {-1., 1.}}, 8, "TargetDevice" -> "CPU", "Return" -> "Image3D"];
  MatchQ[densityNet, _NetChain] &&
  MatchQ[colorNet, _NetChain] &&
  Dimensions[vol] === {8, 8, 8} &&
  ImageQ[img3d] && ImageDimensions[img3d] === {8, 8, 8},
  True,
  TestID -> "NeRFLite-DensityAndColorExtraction"
]

(* 7. Batch Generator & Ray Pool Construction *)
VerificationTest[
  syntheticViews = Table[
    <|
      "CameraMatrix" -> IdentityMatrix[4],
      "FOV" -> 50.0,
      "Dimensions" -> {16, 16},
      "RGB" -> Table[{0.5, 0.5, 0.5}, {16}, {16}],
      "Alpha" -> Table[1.0, {16}, {16}],
      "ForegroundPixels" -> {{8, 8}}
    |>,
    {2}
  ];
  gen = CreateNeRFLiteBatchGenerator[syntheticViews, "BatchSize" -> 16, "NumSamples" -> 16, "PrecomputeRays" -> True];
  batch = gen[16];
  AssociationQ[batch] &&
  Dimensions[Normal[batch["SamplePoints"]]] === {16, 16, 3} &&
  Dimensions[Normal[batch["Steps"]]] === {16, 16} &&
  Dimensions[Normal[batch["TargetRGB"]]] === {16, 3},
  True,
  TestID -> "NeRFLite-BatchGenerator"
]

(* 8. Novel-View Rendering & Camera Rig 3D Visualization Test *)
VerificationTest[
  net = NetInitialize @ CreateNeRFLiteNet["NumSamples" -> 8, "HarmonicBands" -> 2, "HiddenDim" -> 16, "NumLayers" -> 2];
  views = Table[
    <|
      "CameraMatrix" -> {
        {Cos[t], 0., Sin[t], 4.0 * Sin[t]},
        {0., 1., 0., 0.},
        {-Sin[t], 0., Cos[t], 4.0 * Cos[t]},
        {0., 0., 0., 1.}
      },
      "FOV" -> 50.0,
      "Dimensions" -> {16, 16}
    |>,
    {t, {0., 1., 2.}}
  ];
  g3d = VisualizeNeRFCameraRig[views];
  renderedImg = RenderNeRFLiteView[net, views[[1]]["CameraMatrix"], {16, 16}, "TargetDevice" -> "CPU"];
  MatchQ[g3d, _Graphics3D] && ImageQ[renderedImg] && ImageDimensions[renderedImg] === {16, 16},
  True,
  TestID -> "NeRFLite-RenderAndVisualizeCameraRig"
]

(* 9. NeRF Synthetic Lego Dataset Loader & Downloader Cache Test *)
VerificationTest[
  legoDir = FileNameJoin[{DirectoryName[$TestFileName, 2], "data", "lego"}];
  If[DirectoryQ[legoDir] && FileExistsQ[FileNameJoin[{legoDir, "transforms_train.json"}]],
    dlInfo = DownloadNeRFDataset[legoDir, "Splits" -> {"train", "val"}, "Overwrite" -> False];
    legoViews = LoadNeRFDataset[legoDir, "MaxViews" -> 2, "TargetDimensions" -> {64, 64}, "WhiteBackground" -> True];
    AssociationQ[dlInfo] &&
    ListQ[legoViews] &&
    Length[legoViews] == 2 &&
    legoViews[[1]]["Dimensions"] === {64, 64} &&
    MatrixQ[legoViews[[1]]["CameraMatrix"], NumericQ] &&
    NumericQ[legoViews[[1]]["FOV"]],
    True
  ],
  True,
  TestID -> "NeRFLite-LegoDatasetLoad"
]
