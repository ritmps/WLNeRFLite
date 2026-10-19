(* ::Package:: *)
(* scripts/train_nerflite.wl
   Standalone NeRF Lite training tool for standard NeRF JSON or RenderMan EXR datasets.
   Usage:
     wolframscript -script scripts/train_nerflite.wl \
       --dataset "/home/flip/renders/nerf_datasets/scene_031" \
       --rounds 10000 \
       --batch-size 1024 \
       --device "GPU" \
       --output "/home/flip/renders/nerf_checkpoints/nerflite_scene031.wlnet" \
       --render-test "/home/flip/renders/nerf_checkpoints/scene031_test.png"
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

datasetPath = parseArg["--dataset", None];
roundsStr   = parseArg["--rounds", "5000"];
batchSizeStr = parseArg["--batch-size", "1024"];
device      = parseArg["--device", "GPU"];
lrStr       = parseArg["--lr", "0.0005"];
outPath     = parseArg["--output", None];
testImgPath = parseArg["--render-test", None];
turntablePath = parseArg["--render-turntable", None];

If[datasetPath === None,
  Print["Error: --dataset argument is required."];
  Print["Usage: wolframscript -script scripts/train_nerflite.wl --dataset <path_or_dir> [--rounds 5000] [--batch-size 1024] [--device GPU|CPU] [--output model.wlnet]"];
  Exit[1];
];

rounds = ToExpression[roundsStr];
batchSize = ToExpression[batchSizeStr];
lr = ToExpression[lrStr];

Print["[1/4] Loading views from: ", datasetPath];
views = Which[
  DirectoryQ[datasetPath] && Length[FileNames["transforms*.json", datasetPath]] > 0,
    LoadNeRFDataset[datasetPath, "Split" -> "train"],
  StringEndsQ[ToLowerCase[datasetPath], ".json"],
    LoadNeRFDataset[datasetPath],
  True,
    LoadRenderManEXRViews[datasetPath]
];

If[FailureQ[views] || Length[views] == 0,
  Print["Error: Could not load any valid views from: ", datasetPath];
  Exit[1];
];
Print["      Loaded ", Length[views], " training views."];
Print["      Image Dimensions: ", views[[1]]["Dimensions"], " | Camera FOV: ", Round[views[[1]]["FOV"], 0.1], " deg"];

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
    "TargetDevice" -> device
  ];
  Export[testImgPath, testImg];
];

If[StringQ[turntablePath] && turntablePath =!= "",
  Print["      Rendering 360 turntable GIF to: ", turntablePath];
  If[!DirectoryQ[DirectoryName[turntablePath]], CreateDirectory[DirectoryName[turntablePath], CreateIntermediateDirectories -> True]];
  RenderNeRFLiteTurntable[
    trainedModel,
    "NumFrames" -> 36,
    "Resolution" -> {256, 256},
    "TargetDevice" -> device,
    "ExportFile" -> turntablePath
  ];
];

Print["\nAll tasks finished successfully."];
Exit[0];
