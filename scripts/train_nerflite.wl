(* ::Package:: *)
(* scripts/train_nerflite.wl
   Standalone NeRF Lite training tool for standard NeRF JSON or RenderMan EXR datasets.
   Usage:
     wolframscript -script scripts/train_nerflite.wl \
       --dataset "data/lego" \
       --rounds 3000 \
       --batch-size 1024 \
       --res 200 \
       --loss "L2" \
       --output "output/lego_nerflite.wlnet" \
       --render-test "output/lego_test.png" \
       --compare-val "output/lego_val_compare.png" \
       --render-turntable "output/lego_turntable.gif"
*)

$HistoryLength = 0;

scriptFile = Which[
  StringQ[$InputFileName] && $InputFileName =!= "", $InputFileName,
  ListQ[$ScriptCommandLine] && Length[$ScriptCommandLine] > 0 && StringQ[First[$ScriptCommandLine]], First[$ScriptCommandLine],
  True, "scripts/train_nerflite.wl"
];
repoDir = DirectoryName[ExpandFileName[scriptFile], 2];
PacletDirectoryLoad[repoDir];
Needs["WLNeRFLite`"];

Print["=================================================================="];
Print["WLNeRFLite Standalone Model Trainer"];
Print["=================================================================="];

args = Which[
  ListQ[$ScriptCommandLine] && Length[$ScriptCommandLine] > 1,
    $ScriptCommandLine[[2 ;;]],
  True,
    {}
];

parseArg[flag_String, default_] := Module[{pos = FirstPosition[args, flag]},
  If[MissingQ[pos] || pos[[1]] >= Length[args], default, args[[pos[[1]] + 1]]]
];

datasetPath    = parseArg["--dataset", None];
roundsStr      = parseArg["--rounds", "3000"];
batchSizeStr   = parseArg["--batch-size", "1024"];
deviceArg      = parseArg["--device", "Automatic"];
lrStr          = parseArg["--lr", "0.001"];
lossTypeArg    = parseArg["--loss", None];
rayBoundsStr   = parseArg["--ray-bounds", None];
downsampleStr  = parseArg["--downsample", None];
targetResStr   = parseArg["--res", None];
samplesStr     = parseArg["--samples", None];
fgRatioStr     = parseArg["--sample-fg-ratio", "0.5"];
whiteBgStr     = parseArg["--white-bg", "True"];
maxViewsStr    = parseArg["--max-views", None];
bandsStr       = parseArg["--bands", None];
hiddenDimStr   = parseArg["--hidden-dim", None];
layersStr      = parseArg["--layers", None];
weightOpStr    = parseArg["--weight-opacity", "0.0"];
outPath        = parseArg["--output", None];
testImgPath    = parseArg["--render-test", None];
compareValPath = parseArg["--compare-val", None];
turntablePath  = parseArg["--render-turntable", None];
upAxisArg      = parseArg["--up-axis", None];

If[datasetPath === None,
  Print["Error: --dataset argument is required."];
  Print["Usage: wolframscript -script scripts/train_nerflite.wl --dataset <path_or_dir> [--rounds 3000] [--batch-size 1024] [--device CPU|GPU|Automatic] [--res 200] [--loss L2|LogL1] [--output model.wlnet]"];
  Exit[1];
];

rounds = ToExpression[roundsStr];
batchSize = ToExpression[batchSizeStr];
lr = ToExpression[lrStr];
fgRatio = ToExpression[fgRatioStr];
weightOpacity = ToExpression[weightOpStr];
whiteBg = MemberQ[{"true", "1", "yes"}, ToLowerCase[ToString[whiteBgStr]]];
maxViews = If[maxViewsStr =!= None, ToExpression[maxViewsStr], All];

(* Device selection *)
device = Which[
  MemberQ[{"gpu", "cuda"}, ToLowerCase[deviceArg]] && $SystemID === "MacOSX-ARM64",
    Print["Note: macOS ARM64 uses native vector CPU for NetTrain. Setting device to CPU."];
    "CPU",
  MemberQ[{"gpu2", "cuda2", "secondary", "dedicated"}, ToLowerCase[deviceArg]],
    {"GPU", 2},
  MemberQ[{"gpu1", "cuda1", "primary"}, ToLowerCase[deviceArg]],
    {"GPU", 1},
  MemberQ[{"multigpu", "allgpu", "dual-gpu", "dual"}, ToLowerCase[deviceArg]],
    "MultiGPU",
  MemberQ[{"gpu", "cuda"}, ToLowerCase[deviceArg]],
    (* On multi-GPU Linux systems, target dedicated GPU 2 to avoid Xorg display contention *)
    {"GPU", 2},
  ToLowerCase[deviceArg] === "automatic",
    If[$SystemID === "MacOSX-ARM64", "CPU", {"GPU", 2}],
  True,
    deviceArg
];

renderDevice = Which[
  device === "MultiGPU", "GPU",
  True, device
];

Print["[1/4] Loading views from: ", datasetPath];

isNeRFDataset = DirectoryQ[datasetPath] && Length[FileNames["transforms*.json", datasetPath]] > 0;
isJSONFile = StringEndsQ[ToLowerCase[datasetPath], ".json"];

(* Model Hyperparameters tailored to dataset type *)
lossType = Which[
  lossTypeArg =!= None, lossTypeArg,
  (isNeRFDataset || isJSONFile), "L2",
  True, "LogL1"
];

bands = Which[
  bandsStr =!= None, ToExpression[bandsStr],
  (isNeRFDataset || isJSONFile), 10,
  True, 6
];

hiddenDim = Which[
  hiddenDimStr =!= None, ToExpression[hiddenDimStr],
  (isNeRFDataset || isJSONFile), 128,
  True, 64
];

numLayers = Which[
  layersStr =!= None, ToExpression[layersStr],
  (isNeRFDataset || isJSONFile), 4,
  True, 2
];

numSamples = Which[
  samplesStr =!= None, ToExpression[samplesStr],
  (isNeRFDataset || isJSONFile), 64,
  True, 32
];

upAxis = Which[
  upAxisArg =!= None, upAxisArg,
  (isNeRFDataset || isJSONFile), "Z",
  True, "Y"
];

(* Determine TargetDimensions *)
targetDims = Which[
  targetResStr =!= None,
    Module[{r = ToExpression[targetResStr]}, {r, r}],
  downsampleStr =!= None,
    Module[{ds = ToExpression[downsampleStr]},
      If[isNeRFDataset || isJSONFile,
        {Round[800 / ds], Round[800 / ds]},
        None
      ]
    ],
  True,
    Automatic
];

loadOpts = Sequence @@ DeleteCases[{
  "TargetDimensions" -> targetDims,
  "WhiteBackground" -> whiteBg,
  "MaxViews" -> maxViews
}, _ -> None];

views = Which[
  isNeRFDataset,
    LoadNeRFDataset[datasetPath, "Split" -> "train", loadOpts],
  isJSONFile,
    LoadNeRFDataset[datasetPath, loadOpts],
  True,
    LoadRenderManEXRViews[datasetPath]
];

If[FailureQ[views] || Length[views] == 0,
  Print["Error: Could not load any valid views from: ", datasetPath];
  Exit[1];
];
Print["      Loaded ", Length[views], " training views."];
Print["      Image Dimensions: ", views[[1]]["Dimensions"], " | Camera FOV: ", Round[views[[1]]["FOV"], 0.1], " deg"];

(* Ray Bounds *)
rayBounds = Which[
  StringQ[rayBoundsStr] && StringContainsQ[rayBoundsStr, ","],
    Map[ToExpression, StringSplit[rayBoundsStr, ","]],
  (isNeRFDataset || isJSONFile),
    (* Standard NeRF Synthetic objects (Lego, etc.) centered at origin with camera distance ~4.0 *)
    {2.0, 6.0},
  True,
    {3.0, 10.0}
];
Print["      Volumetric Ray Bounds: [", rayBounds[[1]], ", ", rayBounds[[2]], "] | Loss: ", lossType];
Print["      Architecture: ", numSamples, " samples/ray, ", bands, " bands (harmonic), ", hiddenDim, " hidden dim, ", numLayers, " layers, up-axis: ", upAxis];

bgColor = Lookup[views[[1]], "BackgroundColor", If[whiteBg, {1., 1., 1.}, None]];

Print["[2/4] Initializing NeRFLite coordinate MLP (embedded volumetric quadrature)..."];
fieldNet = NetInitialize @ CreateNeRFLiteNet[
  "NumSamples" -> numSamples,
  "HarmonicBands" -> bands,
  "HiddenDim" -> hiddenDim,
  "NumLayers" -> numLayers,
  "BackgroundColor" -> bgColor,
  "ColorActivation" -> LogisticSigmoid
];

Print["[3/4] Starting training on target device: ", device, " (", rounds, " rounds, batch size ", batchSize, ", LR ", lr, ")..."];
tStart = AbsoluteTime[];
trainedModel = TrainNeRFLiteModel[
  fieldNet,
  views,
  "BatchSize" -> batchSize,
  "MaxTrainingRounds" -> rounds,
  "LearningRate" -> lr,
  "Loss" -> lossType,
  "RayBounds" -> rayBounds,
  "SampleForegroundRatio" -> fgRatio,
  "WeightOpacity" -> weightOpacity,
  "TargetDevice" -> device,
  "TrainingProgressReporting" -> "Compact"
];
trainSec = Round[AbsoluteTime[] - tStart, 0.1];
Print["      Training completed in ", trainSec, " seconds."];

If[StringQ[outPath] && outPath =!= "",
  Print["[4/4] Exporting trained model checkpoint to: ", outPath];
  If[!DirectoryQ[DirectoryName[outPath]], CreateDirectory[DirectoryName[outPath], CreateIntermediateDirectories -> True]];
  Export[outPath, trainedModel];
  If[StringEndsQ[outPath, ".wlnet"],
    Export[StringDrop[outPath, -6] <> ".wxf", trainedModel, "WXF"];
  ];
];

If[StringQ[testImgPath] && testImgPath =!= "",
  Print["      Rendering training test view to: ", testImgPath];
  If[!DirectoryQ[DirectoryName[testImgPath]], CreateDirectory[DirectoryName[testImgPath], CreateIntermediateDirectories -> True]];
  testImg = RenderNeRFLiteView[
    trainedModel,
    views[[1]]["CameraMatrix"],
    views[[1]]["Dimensions"],
    "FOV" -> views[[1]]["FOV"],
    "RayBounds" -> rayBounds,
    "TargetDevice" -> renderDevice
  ];
  Export[testImgPath, testImg];
];

If[StringQ[compareValPath] && compareValPath =!= "",
  Print["      Evaluating reconstruction against validation set..."];
  valViews = Which[
    isNeRFDataset,
      LoadNeRFDataset[datasetPath, "Split" -> "val", loadOpts],
    True,
      views
  ];
  If[ListQ[valViews] && Length[valViews] > 0,
    Module[{
      vVal = valViews[[1]], valRecon, gtImg, diffImg, psnr, mse, compGrid, w, h
    },
      valRecon = RenderNeRFLiteView[
        trainedModel,
        vVal["CameraMatrix"],
        vVal["Dimensions"],
        "FOV" -> vVal["FOV"],
        "RayBounds" -> rayBounds,
        "TargetDevice" -> renderDevice
      ];
      gtImg = Image[vVal["RGB"], "Real32", ColorSpace -> "RGB"];
      {w, h} = ImageDimensions[gtImg];

      diffImg = Image[Clip[3.0 * Abs[ImageData[valRecon] - ImageData[gtImg]], {0.0, 1.0}]];
      mse = Mean[Flatten[(ImageData[valRecon] - ImageData[gtImg])^2]];
      psnr = If[mse > 0, Round[-10.0 * Log10[mse], 0.01], Infinity];
      Print["      Validation View 0 -> PSNR: ", psnr, " dB | MSE: ", NumberForm[mse, {5, 4}]];

      compGrid = ImageAssemble[{{
        ImageCrop[gtImg, {w, h}],
        ImageCrop[valRecon, {w, h}],
        ImageCrop[diffImg, {w, h}]
      }}];
      If[!DirectoryQ[DirectoryName[compareValPath]], CreateDirectory[DirectoryName[compareValPath], CreateIntermediateDirectories -> True]];
      Export[compareValPath, compGrid];
      Print["      Validation comparison image saved: ", compareValPath, " (Left: GT, Mid: Recon, Right: 3x Error)"];
    ]
  ];
];

If[StringQ[turntablePath] && turntablePath =!= "",
  Print["      Rendering 360 turntable GIF to: ", turntablePath];
  If[!DirectoryQ[DirectoryName[turntablePath]], CreateDirectory[DirectoryName[turntablePath], CreateIntermediateDirectories -> True]];
  camRadius = If[isNeRFDataset || isJSONFile, 4.0, 5.0];
  RenderNeRFLiteTurntable[
    trainedModel,
    "NumFrames" -> 36,
    "Resolution" -> {256, 256},
    "Radius" -> camRadius,
    "Elevation" -> 30.0 * Degree,
    "Target" -> {0., 0., 0.},
    "UpAxis" -> upAxis,
    "FOV" -> views[[1]]["FOV"],
    "RayBounds" -> rayBounds,
    "TargetDevice" -> renderDevice,
    "ExportFile" -> turntablePath
  ];
];

Print["\nAll tasks finished successfully."];
Exit[0];
