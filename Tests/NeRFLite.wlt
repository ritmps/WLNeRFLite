(* ::Package:: *)

VerificationTest[
  PacletDirectoryLoad[DirectoryName[$TestFileName, 2]];
  Needs["WLNeRFLite`"];
  MemberQ[$Packages, "WLNeRFLite`"],
  True,
  TestID -> "NeRFLite-PackageLoad"
]

(* 1. Bidirectional Transform Inversion Test *)
VerificationTest[
  w2c = {
    {0.850651, 0., -0.525731, 0.},
    {0., 1., 0., 0.},
    {0.525731, 0., 0.850651, 0.},
    {1.2, -0.5, 4.8, 1.}
  };
  c2wNeRF = RenderManToNeRFTransform[w2c];
  w2cRoundtrip = NeRFToRenderManTransform[c2wNeRF];
  Max[Abs[w2c - w2cRoundtrip]] < 1*^-12,
  True,
  TestID -> "NeRFLite-TransformRoundtrip"
]

(* 2. NeRF Matrix Camera Position & Look Axis Test *)
VerificationTest[
  w2c = {
    {1., 0., 0., 0.},
    {0., 1., 0., 0.},
    {0., 0., 1., 0.},
    {-2.5, 3.0, -7.0, 1.}
  };
  inv = Inverse[w2c];
  expectedPos = inv[[4, 1 ;; 3]];
  c2w = RenderManToNeRFTransform[w2c];
  nerfPos = c2w[[1 ;; 3, 4]];
  Max[Abs[expectedPos - nerfPos]] < 1*^-12 && c2w[[4]] == {0., 0., 0., 1.},
  True,
  TestID -> "NeRFLite-CameraPositionInNeRF"
]

(* 3. Ray Generation Equivalence Between Conventions *)
VerificationTest[
  w2c = {
    {0., 0.850651, 0.525731, 0.},
    {0., 0.525731, -0.850651, 0.},
    {1., 0., 0., 0.},
    {0.317267, -0.701351, 5.32661, 1.}
  };
  c2w = RenderManToNeRFTransform[w2c];
  coords = {{0, 0}, {32, 32}, {63, 63}};
  raysRman = GenerateCameraRays[w2c, {64, 64}, "FOV" -> 50.0, "PixelCoords" -> coords, "Convention" -> "RenderMan"];
  raysNeRF = GenerateCameraRays[c2w, {64, 64}, "FOV" -> 50.0, "PixelCoords" -> coords, "Convention" -> "NeRF"];
  maxOrigDiff = Max[Abs[raysRman["Origins"] - raysNeRF["Origins"]]];
  maxDirDiff  = Max[Abs[raysRman["Directions"] - raysNeRF["Directions"]]];
  maxOrigDiff < 1*^-12 && maxDirDiff < 1*^-6,
  True,
  TestID -> "NeRFLite-RayEquivalence"
]

(* 4. Stratified Ray Sampling Test *)
VerificationTest[
  origs = {{0., 0., 5.}, {1., 1., 5.}};
  dirs = {{0., 0., -1.}, {0., 0., -1.}};
  sampled = SampleRaysStratified[origs, dirs, 32, {2.0, 7.0}, "Perturb" -> False];
  Dimensions[sampled["SamplePoints"]] === {2, 32, 3} &&
  Dimensions[sampled["Steps"]] === {2, 32} &&
  Abs[sampled["Steps"][[1, 1]] - (5.0 / 32)] < 1*^-5,
  True,
  TestID -> "NeRFLite-SampleRaysStratified"
]

(* 5. End-to-End Dataset Export and Load (transforms.json + PNG) *)
VerificationTest[
  testDir = FileNameJoin[{$TemporaryDirectory, "wlnerflite_test_dataset_" <> IntegerString[RandomInteger[{10000, 99999}]]}];
  syntheticViews = Table[
    Module[{
      cam = {
        {Cos[phi], 0., -Sin[phi], 0.},
        {0., 1., 0., 0.},
        {Sin[phi], 0., Cos[phi], 0.},
        {0., 0., 4., 1.}
      },
      imgData = Table[{0.2 * i, 0.4, 0.8}, {y, 32}, {x, 32}],
      alphaData = Table[If[(x - 16)^2 + (y - 16)^2 < 100, 1.0, 0.0], {y, 32}, {x, 32}]
    },
      <|
        "CameraMatrix" -> Developer`ToPackedArray[N[cam]],
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

(* 6. Harmonic Positional Encoder Test *)
VerificationTest[
  encoder = CreateHarmonicEncoder[6];
  initEnc = NetInitialize[encoder];
  out = initEnc[{0.1, 0.2, 0.3}];
  Dimensions[out] === {39} && AllTrue[out, NumericQ],
  True,
  TestID -> "NeRFLite-HarmonicEncoder"
]

(* 7. NeRFLite NetGraph Architecture & Evaluation *)
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

(* 8. Batch Generator Construction *)
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

(* 9. Camera Rig 3D Visualization Test *)
VerificationTest[
  views = Table[
    <|
      "CameraMatrix" -> {
        {Cos[t], 0., -Sin[t], 0.},
        {0., 1., 0., 0.},
        {Sin[t], 0., Cos[t], 0.},
        {0., 0., 5., 1.}
      },
      "FOV" -> 50.0,
      "Dimensions" -> {32, 32}
    |>,
    {t, {0., 1., 2.}}
  ];
  g3d = VisualizeNeRFCameraRig[views];
  MatchQ[g3d, _Graphics3D],
  True,
  TestID -> "NeRFLite-VisualizeCameraRig"
]

(* 10. NeRF Synthetic Lego Dataset Loader Test *)
VerificationTest[
  legoDir = FileNameJoin[{DirectoryName[$TestFileName, 2], "data", "lego"}];
  If[DirectoryQ[legoDir] && FileExistsQ[FileNameJoin[{legoDir, "transforms_train.json"}]],
    legoViews = LoadNeRFDataset[legoDir, "MaxViews" -> 2, "TargetDimensions" -> {64, 64}, "WhiteBackground" -> True];
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
