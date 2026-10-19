(* ::Package:: *)
(* scripts/convert_exr_to_nerf_json.wl
   Convert RenderMan EXR view datasets to canonical NeRF transforms.json + PNG images.
   Usage:
     wolframscript -script scripts/convert_exr_to_nerf_json.wl \
       --input "/home/flip/renders/procedural_dataset/scene_031_view_*.exr" \
       --output "/home/flip/renders/nerf_datasets/scene_031" \
       --split "0.8,0.1,0.1" \
       --format "PNG"
*)

$HistoryLength = 0;

scriptFile = Which[
  StringQ[$InputFileName] && $InputFileName =!= "", $InputFileName,
  ListQ[$ScriptCommandLine] && Length[$ScriptCommandLine] > 0 && StringQ[First[$ScriptCommandLine]], First[$ScriptCommandLine],
  True, "scripts/convert_exr_to_nerf_json.wl"
];
repoDir = DirectoryName[ExpandFileName[scriptFile], 2];
PacletDirectoryLoad[repoDir];
Needs["WLNeRFLite`"];

Print["=================================================================="];
Print["WLNeRFLite EXR -> Standard NeRF Dataset Converter"];
Print["=================================================================="];

args = Which[
  ListQ[$ScriptCommandLine] && Length[$ScriptCommandLine] > 1,
    $ScriptCommandLine[[2 ;;]],
  True,
    {}
];

collectArgs[flag_String] := Module[{pos = FirstPosition[args, flag], rest},
  If[MissingQ[pos], Return[{}]];
  rest = args[[pos[[1]] + 1 ;;]];
  TakeWhile[rest, !StringStartsQ[#, "--"] &]
];

parseArg[flag_String, default_] := Module[{vals = collectArgs[flag]},
  If[Length[vals] == 0, default, First[vals]]
];

inputList    = collectArgs["--input"];
outputDir    = parseArg["--output", None];
splitStr     = parseArg["--split", "0.8,0.1,0.1"];
imgFormat    = ToUpperCase @ parseArg["--format", "PNG"];
whiteBg      = MemberQ[args, "--white-background"];

If[Length[inputList] == 0 || outputDir === None,
  Print["Error: --input and --output arguments are required."];
  Print["Usage: wolframscript -script scripts/convert_exr_to_nerf_json.wl --input <pattern> --output <dir> [--split 0.8,0.1,0.1] [--format PNG] [--white-background]"];
  Exit[1];
];

splitParts = ToExpression /@ StringSplit[splitStr, ","];
splitSpec = If[Length[splitParts] == 3 && AllTrue[splitParts, NumericQ],
  <|"Train" -> splitParts[[1]], "Val" -> splitParts[[2]], "Test" -> splitParts[[3]]|>,
  <|"Train" -> 0.8, "Val" -> 0.1, "Test" -> 0.1|>
];

Print["[1/3] Searching input EXRs: ", If[Length[inputList] == 1, First[inputList], ToString[Length[inputList]] <> " input paths"]];
views = LoadRenderManEXRViews[inputList];
If[FailureQ[views] || Length[views] == 0,
  Print["Error: No valid RenderMan EXR views found matching pattern."];
  Exit[1];
];
Print["      Loaded ", Length[views], " views successfully."];
Print["      Image Dimensions: ", views[[1]]["Dimensions"], " | Camera FOV: ", Round[views[[1]]["FOV"], 0.1], " deg"];

Print["[2/3] Exporting to canonical NeRF JSON format at: ", outputDir];
Print["      Split specification: Train ", splitSpec["Train"] * 100, "% | Val ", splitSpec["Val"] * 100, "% | Test ", splitSpec["Test"] * 100, "%"];
res = ExportNeRFDataset[
  views,
  outputDir,
  "Split" -> splitSpec,
  "ImageFormat" -> imgFormat,
  "WhiteBackground" -> whiteBg,
  "ExportImages" -> True
];

If[FailureQ[res],
  Print["Error during dataset export: ", res];
  Exit[1];
];

Print["[3/3] Export completed successfully!"];
Print["      Total frames: ", res["TotalViews"]];
Print["      Train frames: ", res["Splits", "train"], " -> ", res["JSONFiles", "train"]];
If[KeyExistsQ[res["Splits"], "val"], Print["      Val frames:   ", res["Splits", "val"], " -> ", res["JSONFiles", "val"]]];
If[KeyExistsQ[res["Splits"], "test"], Print["      Test frames:  ", res["Splits", "test"], " -> ", res["JSONFiles", "test"]]];
Print["      Master JSON:  ", res["JSONFiles", "master"]];
Print["\nDataset is ready for WLNeRFLite, Instant-NGP, Nerfstudio, and 3D Gaussian Splatting."];
Exit[0];
