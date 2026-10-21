(* ::Package:: *)
(* scripts/train_nerflite.wl
   Standalone NeRF Lite training tool for standard NeRF JSON or RenderMan EXR datasets.
   Usage:
     wolframscript -script scripts/train_nerflite.wl \
       --dataset "data/lego" \
       --rounds 3000 \
       --batch-size 1024 \
       --device "Automatic" \
       --ray-bounds "2.0,6.0" \
       --downsample 2 \
       --output "output/lego_nerflite.wlnet" \
       --render-test "output/lego_test.png" \
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

datasetPath   = parseArg["--dataset", None];
roundsStr     = parseArg["--rounds", "5000"];
batchSizeStr  = parseArg["--batch-size", "1024"];
deviceArg     = parseArg["--device", "Automatic"];
lrStr         = parseArg["--lr", "0.0005"];
rayBoundsStr  = parseArg["--ray-bounds", None];
downsampleStr = parseArg["--downsample", None];
targetResStr  = parseArg["--res", None];
fgRatioStr    = parseArg["--sample-fg-ratio", "0.5"];
whiteBgStr    = parseArg["--white-bg", "True"];
maxViewsStr   = parseArg["--max-views", None];
outPath       = parseArg["--output", None];
testImgPath   = parseArg["--render-test", None];
turntablePath = parseArg["--render-turntable", None];

If[datasetPath === None,
  Print["Error: --dataset argument is required."];
  Print["Usage: wolframscript -script scripts/train_nerflite.wl --dataset <path_or_dir> [--rounds 5000] [--batch-size 1024] [--device CPU|GPU|Automatic] [--ray-bounds 2.0,6.0] [--downsample 2] [--output model.wlnet]"];
  Exit[1];
];

rounds = ToExpression[roundsStr];
batchSize = ToExpression[batchSizeStr];
lr = ToExpression[lrStr];
fgRatio = ToExpression[fgRatioStr];
whiteBg = MemberQ[{"true", "1", "yes"}, ToLowerCase[ToString[whiteBgStr]]];
maxViews = If[maxViewsStr =!= None, ToExpression[maxViewsStr], All];

(* Device selection: on macOS ARM64, CPU is fastest & native for NetTrain *)
device = Which[
  MemberQ[{"gpu", "cuda"}, ToLowerCase[deviceArg]] && $SystemID === "MacOSX-ARM64",
    Print["Note: macOS ARM64 uses native vector CPU for NetTrain. Setting device to CPU."];
    "CPU",
  ToLowerCase[deviceArg] === "automatic",
    If[$SystemID === "MacOSX-ARM64", "CPU", "GPU"],
  True,
    deviceArg
];

Print["[1/4] Loading views from: ", datasetPath];

isNeRFDataset = DirectoryQ[datasetPath] && Length[FileNames["transforms*.json", datasetPath]] > 0;
isJSONFile = StringEndsQ[ToLowerCase[datasetPath], ".json"];

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
    (* Standard NeRF Synthetic benchmark objects (Lego, Chair, etc.) are centered at origin with camera distance ~4.0 *)
    {2.0, 6.0},
  True,
    {3.0, 10.0}
];
Print["      Volumetric Ray Bounds: [", rayBounds[[1]], ", ", rayBounds[[2]], "]"];

Print["[2/4] Initializing NeRFLite coordinate MLP (10k params)..."];
fieldNet = NetInitialize @ CreateNeRFLiteNet[
  "NumSamples" -> 32,
  "HarmonicBands" -> 6,
  "HiddenDim" -> 64,
  "NumLayers" -> 2,
  "ColorActivation" -> LogisticSigmoid
];

Print["[3/4] Starting training on target device: ", device, " (", rounds, " rounds, batch size ", batchSize, ")..."];
tStart = AbsoluteTime[];
trainedModel = TrainNeRFLiteModel[
  fieldNet,
  views,
  "BatchSize" -> batchSize,
  "MaxTrainingRounds" -> rounds,
  "LearningRate" -> lr,
  "RayBounds" -> rayBounds,
  "SampleForegroundRatio" -> fgRatio,
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
  Print["      Rendering test view to: ", testImgPath];
  If[!DirectoryQ[DirectoryName[testImgPath]], CreateDirectory[DirectoryName[testImgPath], CreateIntermediateDirectories -> True]];
  testImg = RenderNeRFLiteView[
    trainedModel,
    views[[1]]["CameraMatrix"],
    views[[1]]["Dimensions"],
    "FOV" -> views[[1]]["FOV"],
    "RayBounds" -> rayBounds,
    "TargetDevice" -> device
  ];
  Export[testImgPath, testImg];
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
    "Target" -> {0., 0., 0.},
    "FOV" -> views[[1]]["FOV"],
    "RayBounds" -> rayBounds,
    "TargetDevice" -> device,
    "ExportFile" -> turntablePath
  ];
];

Print["\nAll tasks finished successfully."];
Exit[0];
