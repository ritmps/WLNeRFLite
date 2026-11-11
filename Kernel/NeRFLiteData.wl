(* ::Package:: *)

BeginPackage["WLNeRFLite`Data`"];

DownloadNeRFDataset::usage = "DownloadNeRFDataset[targetDir, opts] downloads a canonical synthetic NeRF dataset (such as \"lego\", \"chair\", \"drums\", \"ficus\", \"hotdog\", \"materials\", \"mic\", or \"ship\") including camera transforms JSON files and split images.";
LoadNeRFDataset::usage = "LoadNeRFDataset[pathOrDir, opts] loads a standard NeRF dataset specified by transforms.json (or transforms_train.json) with PNG/JPG/TIFF images and 4x4 camera-to-world pose matrices.";
ExportNeRFDataset::usage = "ExportNeRFDataset[views, outputDir, opts] exports a list of camera views into canonical NeRF format with transforms.json (and optional train/val/test splits) and images.";
GenerateCameraRays::usage = "GenerateCameraRays[cameraMatrix, {w, h}, opts] returns ray origins and normalized world directions for a 4x4 camera-to-world (c2w) matrix.";
SampleRaysStratified::usage = "SampleRaysStratified[origins, directions, numSamples, {tNear, tFar}, opts] computes stratified 3D sample points and intervals along rays.";
PrecomputeNeRFLiteRayPool::usage = "PrecomputeNeRFLiteRayPool[views, opts] precomputes camera rays and RGB targets across views into a high-performance in-memory ray pool.";

Begin["`Private`"];

(* ========================================================================= *)
(* 1. Pure Wolfram Language NeRF Synthetic Dataset Downloader                *)
(* ========================================================================= *)

$CanonicalNeRFScenes = {"lego", "chair", "drums", "ficus", "hotdog", "materials", "mic", "ship"};
$NeRFDatasetBaseURL = "https://huggingface.co/datasets/rishitdagli/nerf-gs-datasets/resolve/main";

Options[DownloadNeRFDataset] = {
  "Scene" -> Automatic,
  "Splits" -> {"train", "val"},
  "Overwrite" -> False
};

DownloadNeRFDataset[opts : OptionsPattern[]] := DownloadNeRFDataset[FileNameJoin[{"data", "lego"}], opts];

DownloadNeRFDataset[scene_String, targetDir_String, opts : OptionsPattern[]] :=
  DownloadNeRFDataset[targetDir, "Scene" -> scene, opts];

DownloadNeRFDataset[arg_String, opts : OptionsPattern[]] := Module[{
  sceneOpt = OptionValue["Scene"],
  splitsOpt = OptionValue["Splits"],
  overwriteQ = TrueQ @ OptionValue["Overwrite"],
  isSceneNameOnly, scene, outDir, splits,
  jsonFiles = <||>, relImagePaths = {}, missingURLs = {}, missingDests = {},
  nCached = 0, nDownloaded = 0
},
  isSceneNameOnly = MemberQ[$CanonicalNeRFScenes, ToLowerCase[arg]] &&
                    !StringContainsQ[arg, "/" | "\\"] &&
                    !DirectoryQ[arg];

  If[isSceneNameOnly,
    scene = ToLowerCase[arg];
    outDir = ExpandFileName[FileNameJoin[{"data", scene}]];
    ,
    outDir = ExpandFileName[arg];
    scene = Which[
      StringQ[sceneOpt], ToLowerCase[sceneOpt],
      MemberQ[$CanonicalNeRFScenes, ToLowerCase[FileBaseName[outDir]]], ToLowerCase[FileBaseName[outDir]],
      True, "lego"
    ];
  ];

  If[!DirectoryQ[outDir],
    CreateDirectory[outDir, CreateIntermediateDirectories -> True];
  ];

  splits = Which[
    splitsOpt === All || splitsOpt === "all" || (ListQ[splitsOpt] && MemberQ[ToLowerCase /@ ToString /@ splitsOpt, "all"]),
      {"train", "val", "test"},
    StringQ[splitsOpt],
      {ToLowerCase[splitsOpt]},
    ListQ[splitsOpt],
      ToLowerCase /@ ToString /@ splitsOpt,
    True,
      {"train", "val"}
  ];
  splits = Select[splits, MemberQ[{"train", "val", "test"}, #] &];
  If[splits === {}, splits = {"train", "val"}];

  (* 1. Download split transforms_*.json files and collect frame image paths *)
  Do[
    Module[{
      jfName = "transforms_" <> s <> ".json",
      jfDest, jfURL, rawJson, frames
    },
      jfDest = FileNameJoin[{outDir, jfName}];
      jfURL  = $NeRFDatasetBaseURL <> "/" <> scene <> "/" <> jfName;

      If[overwriteQ || !FileExistsQ[jfDest] || Quiet[FileByteCount[jfDest]] === 0,
        Quiet @ URLDownload[jfURL, jfDest];
      ];

      If[FileExistsQ[jfDest],
        jsonFiles[s] = jfDest;
        rawJson = Quiet @ Check[Developer`ReadRawJSONFile[jfDest], Import[jfDest, "RawJSON"]];
        If[AssociationQ[rawJson],
          frames = Lookup[rawJson, "frames", {}];
          Do[
            Module[{fp = Lookup[f, "file_path", ""], cleanRel},
              If[StringQ[fp] && fp =!= "",
                cleanRel = StringReplace[fp, RegularExpression["^\\./+"] -> ""];
                If[FileExtension[cleanRel] === "", cleanRel = cleanRel <> ".png"];
                AppendTo[relImagePaths, cleanRel];
              ]
            ],
            {f, frames}
          ];
        ];
      ];
    ],
    {s, splits}
  ];

  relImagePaths = DeleteDuplicates[relImagePaths];

  (* 2. Check cached vs missing frame images *)
  Do[
    Module[{destPath, subDir, url},
      destPath = FileNameJoin[Prepend[FileNameSplit[relPath], outDir]];
      subDir = DirectoryName[destPath];
      If[!DirectoryQ[subDir],
        CreateDirectory[subDir, CreateIntermediateDirectories -> True];
      ];
      If[!overwriteQ && FileExistsQ[destPath] && Quiet[FileByteCount[destPath]] > 0,
        nCached++,
        url = $NeRFDatasetBaseURL <> "/" <> scene <> "/" <> StringReplace[relPath, "\\" -> "/"];
        AppendTo[missingURLs, url];
        AppendTo[missingDests, destPath];
      ];
    ],
    {relPath, relImagePaths}
  ];

  (* 3. Download missing images in concurrent chunks via URLDownload *)
  If[Length[missingURLs] > 0,
    Module[{urlChunks, destChunks},
      urlChunks = Partition[missingURLs, UpTo[32]];
      destChunks = Partition[missingDests, UpTo[32]];
      MapThread[
        Function[{uBatch, dBatch},
          Quiet @ URLDownload[uBatch, dBatch];
        ],
        {urlChunks, destChunks}
      ];
      nDownloaded = Count[missingDests, _?(FileExistsQ[#] && Quiet[FileByteCount[#]] > 0 &)];
    ];
  ];

  <|
    "Directory" -> outDir,
    "Scene" -> scene,
    "Splits" -> splits,
    "JSONFiles" -> jsonFiles,
    "TotalFrames" -> Length[relImagePaths],
    "DownloadedFiles" -> nDownloaded,
    "CachedFiles" -> nCached
  |>
];

(* ========================================================================= *)
(* 2. Standard NeRF Dataset Loader (transforms.json + PNG/JPG/TIFF)          *)
(* ========================================================================= *)

Options[LoadNeRFDataset] = {
  "Split" -> Automatic,
  "TargetDimensions" -> Automatic,
  "MaxViews" -> All,
  "Shuffle" -> False,
  "WhiteBackground" -> False,
  "AutoDownload" -> False
};

LoadNeRFDataset[pathOrDir_String, opts : OptionsPattern[]] := Module[{
  split = OptionValue["Split"],
  targetDim = OptionValue["TargetDimensions"],
  maxViews = OptionValue["MaxViews"],
  shuffleQ = TrueQ @ OptionValue["Shuffle"],
  whiteBg = TrueQ @ OptionValue["WhiteBackground"],
  autoDownload = TrueQ @ OptionValue["AutoDownload"],
  baseDir, jsonFile, rawJson, camAngleX, flX, frames, views = {}
},
  If[!FileExistsQ[pathOrDir] && !DirectoryQ[pathOrDir],
    If[autoDownload,
      DownloadNeRFDataset[pathOrDir];,
      Return[Failure["PathNotFound", <|"Path" -> pathOrDir|>]]
    ];
  ];

  If[DirectoryQ[pathOrDir],
    baseDir = ExpandFileName[pathOrDir];
    jsonFile = Which[
      MemberQ[{"all", "master", "unified"}, ToLowerCase[ToString[split]]],
        FileNameJoin[{baseDir, "transforms.json"}],
      StringQ[split] && FileExistsQ[FileNameJoin[{baseDir, "transforms_" <> ToLowerCase[split] <> ".json"}]],
        FileNameJoin[{baseDir, "transforms_" <> ToLowerCase[split] <> ".json"}],
      (split === Automatic || split === None) && FileExistsQ[FileNameJoin[{baseDir, "transforms_train.json"}]],
        FileNameJoin[{baseDir, "transforms_train.json"}],
      FileExistsQ[FileNameJoin[{baseDir, "transforms.json"}]],
        FileNameJoin[{baseDir, "transforms.json"}],
      Length[FileNames["transforms*.json", baseDir]] > 0,
        First[FileNames["transforms*.json", baseDir]],
      autoDownload,
        DownloadNeRFDataset[baseDir];
        FileNameJoin[{baseDir, "transforms_train.json"}],
      True,
        Return[Failure["MissingTransformsJSON", <|"Directory" -> baseDir|>]]
    ];
    ,
    jsonFile = ExpandFileName[pathOrDir];
    baseDir = DirectoryName[jsonFile];
  ];

  rawJson = Quiet @ Check[Developer`ReadRawJSONFile[jsonFile], $Failed];
  If[!AssociationQ[rawJson],
    rawJson = Quiet @ Check[Import[jsonFile, "RawJSON"], $Failed];
  ];
  If[!AssociationQ[rawJson],
    Return[Failure["InvalidJSON", <|"File" -> jsonFile|>]]
  ];

  camAngleX = Lookup[rawJson, "camera_angle_x", None];
  flX = Lookup[rawJson, "fl_x", None];

  frames = Lookup[rawJson, "frames", {}];
  If[!ListQ[frames] || Length[frames] == 0,
    Return[Failure["EmptyFrames", <|"File" -> jsonFile|>]]
  ];

  If[shuffleQ, frames = RandomSample[frames]];
  If[IntegerQ[maxViews] && maxViews > 0,
    frames = Take[frames, Min[maxViews, Length[frames]]];
  ];

  Do[
    Module[{
      frame = frames[[i]], relPath, resolvedPath, img, rgb, alpha = None, fgPixels = None,
      w, h, c2w, fov, rawData, bgColor
    },
      relPath = Lookup[frame, "file_path", ""];
      If[relPath == "", Continue[]];

      resolvedPath = SelectFirst[{
        FileNameJoin[{baseDir, relPath}],
        FileNameJoin[{baseDir, relPath <> ".png"}],
        FileNameJoin[{baseDir, relPath <> ".jpg"}],
        FileNameJoin[{baseDir, relPath <> ".jpeg"}],
        FileNameJoin[{baseDir, relPath <> ".tif"}],
        FileNameJoin[{baseDir, relPath <> ".tiff"}]
      }, FileExistsQ[#] &, None];

      If[resolvedPath === None, Continue[]];

      img = Quiet @ Check[Import[resolvedPath, "Image"], $Failed];
      If[!ImageQ[img], Continue[]];

      If[ListQ[targetDim] && Length[targetDim] == 2,
        img = ImageResize[img, targetDim];
      ];

      {w, h} = ImageDimensions[img];
      rawData = ImageData[img, "Real32"];

      If[Length[Dimensions[rawData]] == 3 && Dimensions[rawData][[3]] >= 4,
        alpha = Developer`ToPackedArray @ N[rawData[[All, All, 4]]];
        rgb = rawData[[All, All, 1 ;; 3]];
        If[whiteBg,
          rgb = rgb * alpha + (1.0 - alpha);
        ];
        ,
        rgb = If[Length[Dimensions[rawData]] == 3,
          rawData[[All, All, 1 ;; 3]],
          ArrayReshape[rawData, {h, w, 3}]
        ];
      ];
      rgb = Developer`ToPackedArray @ N[Clip[rgb, {0.0, 1.0}]];

      If[alpha =!= None,
        Module[{pos = SparseArray[UnitStep[alpha - 0.10001]]["NonzeroPositions"]},
          If[Length[pos] > 10,
            fgPixels = Developer`ToPackedArray @ Transpose[{pos[[All, 2]] - 1, pos[[All, 1]] - 1}];
          ];
        ];
      ];

      c2w = Lookup[frame, "transform_matrix", None];
      If[!MatrixQ[c2w, NumericQ] || Dimensions[c2w] =!= {4, 4}, Continue[]];
      c2w = Developer`ToPackedArray[N[c2w]];

      fov = Which[
        KeyExistsQ[frame, "camera_angle_x"] && NumericQ[frame["camera_angle_x"]],
          N[frame["camera_angle_x"] * (180.0 / Pi)],
        NumericQ[camAngleX],
          N[camAngleX * (180.0 / Pi)],
        NumericQ[flX],
          N[2.0 * ArcTan[w / (2.0 * flX)] * (180.0 / Pi)],
        True,
          50.0
      ];

      bgColor = Developer`ToPackedArray @ N[
        (rgb[[1, 1]] + rgb[[1, w]] + rgb[[h, 1]] + rgb[[h, w]]) / 4.0
      ];

      AppendTo[views, <|
        "CameraMatrix" -> c2w,
        "FOV" -> fov,
        "Dimensions" -> {w, h},
        "RGB" -> rgb,
        "Alpha" -> alpha,
        "ForegroundPixels" -> fgPixels,
        "BackgroundColor" -> bgColor,
        "File" -> resolvedPath,
        "FrameIndex" -> i
      |>];
    ],
    {i, Length[frames]}
  ];

  views
];

(* ========================================================================= *)
(* 3. Standard NeRF Dataset Exporter (transforms.json + PNG images)          *)
(* ========================================================================= *)

Options[ExportNeRFDataset] = {
  "Split" -> <|"Train" -> 0.8, "Val" -> 0.1, "Test" -> 0.1|>,
  "ExportImages" -> True,
  "ImageFormat" -> "PNG",
  "WhiteBackground" -> False,
  "UnifiedTransforms" -> True
};

ExportNeRFDataset[views_List, outputDir_String, opts : OptionsPattern[]] := Module[{
  splitSpec = OptionValue["Split"],
  exportImgs = TrueQ @ OptionValue["ExportImages"],
  imgFmt = OptionValue["ImageFormat"],
  unifiedQ = TrueQ @ OptionValue["UnifiedTransforms"],
  nViews = Length[views],
  v1, w, h, fovDeg, fovRad, flX, flY, cx, cy,
  splitGroups = <||>, jsonOutputs = <||>, allFrames = {}
},
  If[nViews == 0,
    Return[Failure["EmptyViews", <|"Message" -> "No views supplied for export."|>]]
  ];

  If[!DirectoryQ[outputDir],
    CreateDirectory[outputDir, CreateIntermediateDirectories -> True];
  ];

  v1 = First[views];
  {w, h} = Lookup[v1, "Dimensions", {512, 512}];
  fovDeg = N @ Lookup[v1, "FOV", 50.0];
  fovRad = fovDeg * Degree;
  flX = N[(w / 2.0) / Tan[fovRad / 2.0]];
  flY = flX;
  cx = N[w / 2.0];
  cy = N[h / 2.0];

  If[AssociationQ[splitSpec] && KeyExistsQ[splitSpec, "Train"],
    Module[{nTrain, nVal, nTest, perm},
      nTrain = Round[Lookup[splitSpec, "Train", 0.8] * nViews];
      nVal   = Round[Lookup[splitSpec, "Val", 0.1] * nViews];
      nTest  = nViews - (nTrain + nVal);
      perm   = views;
      splitGroups["train"] = Take[perm, nTrain];
      splitGroups["val"]   = Take[perm, {nTrain + 1, nTrain + nVal}];
      splitGroups["test"]  = Take[perm, {nTrain + nVal + 1, nViews}];
    ],
    splitGroups["train"] = views;
  ];

  KeyValueMap[
    Function[{splitName, splitViews},
      Module[{splitSubDir, framesList = {}},
        splitSubDir = FileNameJoin[{outputDir, splitName}];
        If[exportImgs && !DirectoryQ[splitSubDir],
          CreateDirectory[splitSubDir, CreateIntermediateDirectories -> True];
        ];

        Do[
          Module[{
            v = splitViews[[idx]], c2w, relPath, imgFile, rgbArr, alphaArr, imgObj
          },
            c2w = v["CameraMatrix"];

            relPath = If[exportImgs,
              imgFile = FileNameJoin[{splitSubDir, "frame_" <> IntegerString[idx - 1, 10, 5] <> "." <> ToLowerCase[imgFmt]}];
              rgbArr = v["RGB"];
              alphaArr = Lookup[v, "Alpha", None];
              imgObj = If[MatrixQ[alphaArr, NumericQ],
                Image[Join[rgbArr, ArrayReshape[alphaArr, {h, w, 1}], 3], "Real32", ColorSpace -> "RGB"],
                Image[rgbArr, "Real32", ColorSpace -> "RGB"]
              ];
              Export[imgFile, imgObj];
              FileNameJoin[{".", splitName, "frame_" <> IntegerString[idx - 1, 10, 5] <> "." <> ToLowerCase[imgFmt]}],
              Lookup[v, "File", "frame_" <> IntegerString[idx - 1, 10, 5]]
            ];

            AppendTo[framesList, <|
              "file_path" -> relPath,
              "transform_matrix" -> Normal[c2w]
            |>];
          ],
          {idx, Length[splitViews]}
        ];

        Module[{jsonRecord, jsonPath},
          jsonRecord = <|
            "camera_angle_x" -> fovRad,
            "camera_angle_y" -> fovRad,
            "fl_x" -> flX,
            "fl_y" -> flY,
            "cx" -> cx,
            "cy" -> cy,
            "w" -> w,
            "h" -> h,
            "frames" -> framesList
          |>;
          jsonPath = FileNameJoin[{outputDir, "transforms_" <> splitName <> ".json"}];
          Export[jsonPath, jsonRecord, "RawJSON"];
          jsonOutputs[splitName] = jsonPath;
          allFrames = Join[allFrames, framesList];
        ];
      ]
    ],
    splitGroups
  ];

  If[unifiedQ,
    Module[{unifiedRecord, masterJsonPath},
      unifiedRecord = <|
        "camera_angle_x" -> fovRad,
        "camera_angle_y" -> fovRad,
        "fl_x" -> flX,
        "fl_y" -> flY,
        "cx" -> cx,
        "cy" -> cy,
        "w" -> w,
        "h" -> h,
        "frames" -> allFrames
      |>;
      masterJsonPath = FileNameJoin[{outputDir, "transforms.json"}];
      Export[masterJsonPath, unifiedRecord, "RawJSON"];
      jsonOutputs["master"] = masterJsonPath;
    ];
  ];

  <|
    "Directory" -> outputDir,
    "TotalViews" -> nViews,
    "Splits" -> Map[Length, splitGroups],
    "JSONFiles" -> jsonOutputs,
    "Dimensions" -> {w, h},
    "FOV" -> fovDeg
  |>
];

(* ========================================================================= *)
(* 4. Ray Generation & Stratified Sampling (Standard NeRF c2w Convention)    *)
(* ========================================================================= *)

Options[GenerateCameraRays] = {
  "FOV" -> 50.0,
  "AspectRatio" -> 1.0,
  "PixelCoords" -> All
};

GenerateCameraRays[matrix_?MatrixQ, {w_Integer, h_Integer}, opts : OptionsPattern[]] := Module[{
  fov = OptionValue["FOV"],
  ar = OptionValue["AspectRatio"],
  pixelCoords = OptionValue["PixelCoords"],
  pos, rx, ry, rz, hw, hh, uvList, nRays, uCoords, vCoords, ndcX, ndcY, rawDirs, norms, dirs, origins
},
  If[!NumericQ[fov] || fov <= 0, fov = 50.0];
  If[!NumericQ[ar] || ar <= 0, ar = N[w / h]];

  (* Standard NeRF / OpenGL 4x4 camera-to-world (c2w) matrix:
     Col 1: Right (+X), Col 2: Up (+Y), Col 3: Back (+Z, -Look), Col 4: Origin *)
  pos = Developer`ToPackedArray[N[matrix[[1 ;; 3, 4]]]];
  rx  = Developer`ToPackedArray[Normalize[N[matrix[[1 ;; 3, 1]]]]];
  ry  = Developer`ToPackedArray[Normalize[N[matrix[[1 ;; 3, 2]]]]];
  rz  = Developer`ToPackedArray[-Normalize[N[matrix[[1 ;; 3, 3]]]]];

  hw = Tan[fov * Degree / 2.0];
  hh = hw / ar;

  If[pixelCoords === All,
    uvList = Flatten[Table[{u, v}, {v, 0, h - 1}, {u, 0, w - 1}], 1],
    uvList = pixelCoords
  ];

  nRays = Length[uvList];
  uCoords = Developer`ToPackedArray[N[uvList[[All, 1]]]];
  vCoords = Developer`ToPackedArray[N[uvList[[All, 2]]]];

  ndcX = ((uCoords + 0.5) / w * 2.0 - 1.0) * hw;
  ndcY = (1.0 - (vCoords + 0.5) / h * 2.0) * hh;

  rawDirs = Outer[Times, ndcX, rx] + Outer[Times, ndcY, ry] + ConstantArray[rz, nRays];
  norms = Sqrt[Total[rawDirs^2, {2}]];
  dirs = rawDirs / norms;
  origins = ConstantArray[pos, nRays];

  <|
    "Origins" -> origins,
    "Directions" -> dirs,
    "PixelCoords" -> uvList,
    "CameraPosition" -> pos
  |>
];

fastStratifiedPointsCompiled = Quiet @ Compile[{
  {orig, _Real, 2},
  {dirs, _Real, 2},
  {tVals, _Real, 2}
},
  Module[{n = Length[orig], s = Length[tVals[[1]]]},
    Table[
      {
        orig[[i, 1]] + tVals[[i, j]] * dirs[[i, 1]],
        orig[[i, 2]] + tVals[[i, j]] * dirs[[i, 2]],
        orig[[i, 3]] + tVals[[i, j]] * dirs[[i, 3]]
      },
      {i, 1, n},
      {j, 1, s}
    ]
  ],
  Parallelization -> True,
  RuntimeOptions -> "Speed"
];

Options[SampleRaysStratified] = {
  "Perturb" -> True
};

SampleRaysStratified[origins_List, directions_List, numSamples_Integer, {tNear_?NumericQ, tFar_?NumericQ}, opts : OptionsPattern[]] := Module[{
  nRays = Length[origins], perturbQ = TrueQ @ OptionValue["Perturb"],
  tSteps, tDeltas, batchT, pts, steps
},
  tSteps = Subdivide[N[tNear], N[tFar], numSamples];
  tDeltas = Differences[tSteps];

  batchT = If[perturbQ,
    ConstantArray[tSteps[[;; -2]], nRays] + ConstantArray[tDeltas, nRays] * RandomReal[{0., 1.}, {nRays, numSamples}],
    ConstantArray[(tSteps[[;; -2]] + tSteps[[2 ;;]]) / 2.0, nRays]
  ];

  pts = Developer`ToPackedArray @ Quiet @ Check[
    fastStratifiedPointsCompiled[Developer`ToPackedArray[N[origins]], Developer`ToPackedArray[N[directions]], Developer`ToPackedArray[N[batchT]]],
    Table[origins[[i]] + batchT[[i, j]] * directions[[i]], {i, nRays}, {j, numSamples}]
  ];

  steps = If[perturbQ,
    Module[{diffs = batchT[[All, 2 ;;]] - batchT[[All, 1 ;; -2]]},
      Developer`ToPackedArray @ Join[diffs, diffs[[All, -1 ;; -1]], 2]
    ],
    Developer`ToPackedArray[ConstantArray[tDeltas, nRays]]
  ];

  <|
    "SamplePoints" -> pts,
    "Steps" -> steps,
    "SampleT" -> batchT
  |>
];

(* ========================================================================= *)
(* 5. High-Performance In-Memory Ray Pool                                    *)
(* ========================================================================= *)

Options[PrecomputeNeRFLiteRayPool] = {
  "Parallel" -> Automatic
};

PrecomputeNeRFLiteRayPool[viewsInput_, opts : OptionsPattern[]] := Module[{
  views = Which[
    ListQ[viewsInput] && AssociationQ[First[viewsInput, None]],
      viewsInput,
    StringQ[viewsInput],
      LoadNeRFDataset[viewsInput],
    True,
      {}
  ],
  parallelOpt = OptionValue["Parallel"],
  useParallel, viewProcessFn, viewPools,
  allFgOrigins = {}, allFgDirs = {}, allFgColors = {},
  allBgOrigins = {}, allBgDirs = {}, allBgColors = {},
  hasAnyFg = False, hasAnyBg = False, bgColor = {0., 0., 0.}
},
  If[!ListQ[views] || Length[views] == 0,
    Return[Failure["EmptyViewsList", <|
      "Message" -> "No views available to precompute ray pool."
    |>]]
  ];

  If[KeyExistsQ[views[[1]], "BackgroundColor"],
    bgColor = views[[1]]["BackgroundColor"];
  ];

  useParallel = Which[
    TrueQ[parallelOpt], True,
    parallelOpt === False, False,
    parallelOpt === Automatic, Length[Kernels[]] > 0,
    True, False
  ];

  viewProcessFn = Function[v,
    Module[{
      w, h, camMat, fov, rgb, alpha,
      allRays, allOrigins, allDirs, allColors,
      hasAlpha, isFg, fgOrig, fgDirs, fgCols,
      bgOrig, bgDirs, bgCols
    },
      {w, h} = v["Dimensions"];
      camMat = v["CameraMatrix"];
      fov = v["FOV"];
      rgb = v["RGB"];
      alpha = Lookup[v, "Alpha", None];

      allRays = GenerateCameraRays[camMat, {w, h}, "FOV" -> fov];
      allOrigins = allRays["Origins"];
      allDirs = allRays["Directions"];
      allColors = Flatten[rgb, 1];

      hasAlpha = MatrixQ[alpha, NumericQ] && (Length[Lookup[v, "ForegroundPixels", {}]] > 10 || Max[alpha] > 0.1);

      If[hasAlpha,
        isFg = UnitStep[Flatten[alpha] - 0.10001];
        fgOrig = Developer`ToPackedArray @ N @ Pick[allOrigins, isFg, 1];
        fgDirs = Developer`ToPackedArray @ N @ Pick[allDirs, isFg, 1];
        fgCols = Developer`ToPackedArray @ N @ Pick[allColors, isFg, 1];

        bgOrig = Developer`ToPackedArray @ N @ Pick[allOrigins, isFg, 0];
        bgDirs = Developer`ToPackedArray @ N @ Pick[allDirs, isFg, 0];
        bgCols = Developer`ToPackedArray @ N @ Pick[allColors, isFg, 0];

        <|
          "Foreground" -> <|"Origins" -> fgOrig, "Directions" -> fgDirs, "RGB" -> fgCols, "Count" -> Length[fgOrig]|>,
          "Background" -> <|"Origins" -> bgOrig, "Directions" -> bgDirs, "RGB" -> bgCols, "Count" -> Length[bgOrig]|>,
          "HasForeground" -> (Length[fgOrig] > 0)
        |>,
        <|
          "Foreground" -> <|"Origins" -> allOrigins, "Directions" -> allDirs, "RGB" -> allColors, "Count" -> Length[allOrigins]|>,
          "Background" -> <|"Origins" -> {}, "Directions" -> {}, "RGB" -> {}, "Count" -> 0|>,
          "HasForeground" -> False
        |>
      ]
    ]
  ];

  viewPools = If[useParallel && Length[Kernels[]] > 0,
    Quiet @ Check[ParallelNeeds["WLNeRFLite`Data`"], Null];
    ParallelMap[viewProcessFn, views],
    Map[viewProcessFn, views]
  ];

  allFgOrigins = Developer`ToPackedArray @ Flatten[Map[#["Foreground", "Origins"] &, viewPools], 1];
  allFgDirs    = Developer`ToPackedArray @ Flatten[Map[#["Foreground", "Directions"] &, viewPools], 1];
  allFgColors  = Developer`ToPackedArray @ Flatten[Map[#["Foreground", "RGB"] &, viewPools], 1];
  hasAnyFg     = Length[allFgOrigins] > 0;

  Module[{bgViews = Select[viewPools, #["Background", "Count"] > 0 &]},
    If[Length[bgViews] > 0,
      allBgOrigins = Developer`ToPackedArray @ Flatten[Map[#["Background", "Origins"] &, bgViews], 1];
      allBgDirs    = Developer`ToPackedArray @ Flatten[Map[#["Background", "Directions"] &, bgViews], 1];
      allBgColors  = Developer`ToPackedArray @ Flatten[Map[#["Background", "RGB"] &, bgViews], 1];
      hasAnyBg     = Length[allBgOrigins] > 0;
    ]
  ];

  <|
    "Type" -> "NeRFLiteRayPool",
    "Foreground" -> <|
      "Origins" -> allFgOrigins,
      "Directions" -> allFgDirs,
      "RGB" -> allFgColors,
      "TargetRGB" -> allFgColors,
      "Count" -> Length[allFgOrigins]
    |>,
    "Background" -> <|
      "Origins" -> allBgOrigins,
      "Directions" -> allBgDirs,
      "RGB" -> allBgColors,
      "TargetRGB" -> allBgColors,
      "Count" -> Length[allBgOrigins]
    |>,
    "All" -> <|
      "Origins" -> If[hasAnyBg, Join[allFgOrigins, allBgOrigins], allFgOrigins],
      "Directions" -> If[hasAnyBg, Join[allFgDirs, allBgDirs], allFgDirs],
      "RGB" -> If[hasAnyBg, Join[allFgColors, allBgColors], allFgColors],
      "TargetRGB" -> If[hasAnyBg, Join[allFgColors, allBgColors], allFgColors],
      "Count" -> Length[allFgOrigins] + Length[allBgOrigins]
    |>,
    "TotalForegroundRays" -> Length[allFgOrigins],
    "TotalBackgroundRays" -> Length[allBgOrigins],
    "TotalAllRays" -> Length[allFgOrigins] + Length[allBgOrigins],
    "HasForeground" -> hasAnyFg,
    "HasBackground" -> hasAnyBg,
    "Dimensions" -> If[Length[views] > 0, views[[1]]["Dimensions"], {512, 512}],
    "BackgroundColor" -> bgColor,
    "NumViews" -> Length[views]
  |>
];

End[];
EndPackage[];
