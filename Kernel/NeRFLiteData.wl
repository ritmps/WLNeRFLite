(* ::Package:: *)

BeginPackage["WLNeRFLite`Data`"];

LoadNeRFDataset::usage = "LoadNeRFDataset[pathOrDir, opts] loads a standard NeRF synthetic / Instant-NGP / Nerfstudio dataset specified by transforms.json (or transforms_train.json) with PNG/JPG/TIFF/EXR images.";
ExportNeRFDataset::usage = "ExportNeRFDataset[views, outputDir, opts] exports a list of camera views into canonical NeRF synthetic format with transforms.json (and optional train/val/test splits) and standard images.";
RenderManToNeRFTransform::usage = "RenderManToNeRFTransform[w2c] converts a RenderMan 4x4 row-major world-to-camera matrix to a standard NeRF 4x4 camera-to-world (c2w) matrix (OpenGL +X right, +Y up, -Z look).";
NeRFToRenderManTransform::usage = "NeRFToRenderManTransform[c2w] converts a standard NeRF 4x4 camera-to-world matrix to a RenderMan 4x4 row-major world-to-camera (w2c) matrix.";
LoadRenderManEXRView::usage = "LoadRenderManEXRView[exrPath, opts] loads camera transforms directly from RenderMan EXR headers and imports only denoised beauty (if present) or beauty RGB.";
LoadRenderManEXRViews::usage = "LoadRenderManEXRViews[pathsOrDir, opts] loads a collection of RenderMan EXR views ready for NeRF training.";
GenerateCameraRays::usage = "GenerateCameraRays[cameraMatrix, {w, h}, opts] returns ray origins and normalized world directions for a camera (auto-detects RenderMan w2c or NeRF c2w).";
SampleRaysStratified::usage = "SampleRaysStratified[origins, directions, numSamples, {tNear, tFar}, opts] computes stratified 3D sample points and intervals along rays.";
PrecomputeNeRFLiteRayPool::usage = "PrecomputeNeRFLiteRayPool[views, opts] precomputes camera rays and RGB targets across views into a high-performance in-memory ray pool.";

Begin["`Private`"];

(* ========================================================================= *)
(* 1. Bidirectional Camera Matrix Transforms (RenderMan <-> NeRF)            *)
(* ========================================================================= *)

RenderManToNeRFTransform[w2c_?MatrixQ] := Module[{inv, c2w},
  inv = Inverse[w2c];
  c2w = Transpose[inv];
  c2w[[All, 3]] = -c2w[[All, 3]];
  Developer`ToPackedArray[N[c2w]]
];

NeRFToRenderManTransform[c2w_?MatrixQ] := Module[{c2wMod, inv, w2c},
  c2wMod = c2w;
  c2wMod[[All, 3]] = -c2wMod[[All, 3]];
  inv = Transpose[c2wMod];
  w2c = Inverse[inv];
  Developer`ToPackedArray[N[w2c]]
];

(* ========================================================================= *)
(* 2. Binary EXR Header Parsing (Zero External CLI Dependencies)             *)
(* ========================================================================= *)

findEXRTool[name_String] := Module[{paths},
  paths = {
    "/opt/pixar/RenderManProServer-28.0b1/bin/exrinfo",
    "/opt/pixar/RenderManProServer-28.0b1/bin/" <> name,
    FileNameJoin[{"/opt", "homebrew", "bin", name}],
    FileNameJoin[{"/usr", "local", "bin", name}],
    FileNameJoin[{"/usr", "bin", name}],
    name
  };
  SelectFirst[paths, FileExistsQ[#] &, name]
];

getEXRHeaderText[path_String] := Module[{tool},
  tool = SelectFirst[{
    findEXRTool["exrinfo"],
    findEXRTool["exrheader"]
  }, FileExistsQ[#] &, "exrinfo"];
  Quiet @ Check[RunProcess[{tool, path}, "StandardOutput"], ""]
];

parseSciNumberString[str_String] := Module[{normalized, val},
  normalized = StringReplace[StringTrim[str], 
    RegularExpression["(?i)e\\+?(-?)0*(\\d+)"] :> "*^$1$2"
  ];
  val = Quiet @ Check[ToExpression @ normalized, Missing["NotANumber", str]];
  If[NumericQ[val], N[val], Indeterminate]
];

parseEXRMatrix[headerText_String, key_String] := Module[{match, rows, mat},
  match = StringCases[headerText, RegularExpression["(?mis)^\\Q" <> key <> "\\E\\s*\\(type\\s+m44f\\):\\s*\\((.*?)\\)"] :> "$1"];
  If[match === {},
    Missing["KeyAbsent", key],
    rows = StringSplit[StringTrim @ First[match], "\n"];
    mat = parseSciNumberString /@ StringSplit[StringTrim[#], Whitespace..] & /@ rows;
    If[MatrixQ[mat, NumericQ] && Det[mat] != 0,
      Developer`ToPackedArray[N[mat]],
      IdentityMatrix[4]
    ]
  ]
];

parseEXRVector[headerText_String, key_String] := Module[{match, vec},
  match = StringCases[headerText, RegularExpression["(?mi)^\\Q" <> key <> "\\E\\s*\\(type\\s+v3f\\):\\s*\\(([^\\)]*)\\)"] :> "$1"];
  If[match === {},
    Missing["KeyAbsent", key],
    vec = parseSciNumberString /@ StringSplit[StringTrim @ First[match], Whitespace..];
    If[VectorQ[vec, NumericQ], Developer`ToPackedArray[N[vec]], {0., 0., 0.}]
  ]
];

readBinaryEXRHeaderAttributes[path_String] := Module[{
  stream, magic, byte, name, type, size, startPos, endPos, attrs = <||>
},
  stream = Quiet @ OpenRead[path, BinaryFormat -> True];
  If[stream === $Failed, Return[<||>]];

  magic = BinaryReadList[stream, "Byte", 4];
  If[magic =!= {118, 47, 49, 1},
    Close[stream];
    Return[<||>];
  ];
  BinaryReadList[stream, "Byte", 4];

  While[True,
    name = "";
    While[(byte = BinaryRead[stream, "Byte"]) =!= EndOfFile && byte != 0,
      name = name <> FromCharacterCode[byte];
    ];
    If[byte === EndOfFile || name == "", Break[]];

    type = "";
    While[(byte = BinaryRead[stream, "Byte"]) =!= EndOfFile && byte != 0,
      type = type <> FromCharacterCode[byte];
    ];
    If[byte === EndOfFile, Break[]];

    size = BinaryRead[stream, "UnsignedInteger32", ByteOrdering -> -1];
    If[size === EndOfFile, Break[]];

    startPos = StreamPosition[stream];
    endPos = startPos + size;

    Switch[type,
      "m44f",
        Module[{floats = BinaryReadList[stream, "Real32", 16, ByteOrdering -> -1]},
          If[Length[floats] == 16,
            attrs[name] = Developer`ToPackedArray[N @ Partition[floats, 4]];
          ];
        ],
      "v3f",
        Module[{vec = BinaryReadList[stream, "Real32", 3, ByteOrdering -> -1]},
          If[Length[vec] == 3,
            attrs[name] = Developer`ToPackedArray[N @ vec];
          ];
        ],
      "float",
        Module[{val = BinaryRead[stream, "Real32", ByteOrdering -> -1]},
          If[NumericQ[val], attrs[name] = N[val]];
        ],
      "int",
        Module[{ival = BinaryRead[stream, "Integer32", ByteOrdering -> -1]},
          If[IntegerQ[ival], attrs[name] = ival];
        ],
      "string",
        Module[{chars = BinaryReadList[stream, "Byte", size]},
          attrs[name] = FromCharacterCode[chars];
        ],
      _,
        Null
    ];
    SetStreamPosition[stream, endPos];
  ];

  Close[stream];
  attrs
];

(* ========================================================================= *)
(* 3. RenderMan EXR View Ingestion                                           *)
(* ========================================================================= *)

Options[LoadRenderManEXRView] = {
  "Tonemap" -> "Reinhard",
  "ExposureGamma" -> 1.0,
  "ClipRGB" -> Automatic,
  "CameraMatrix" -> Automatic,
  "FOV" -> Automatic
};

LoadRenderManEXRView[path_String, opts : OptionsPattern[]] := Module[{
  hdr = "", binAttrs, w2c, fov, chNames, hasDenoised, chType, chData, r, g, b, h, w, rgb,
  tonemapMode, clipOpt, effectiveTonemap, nonNegRgb, alphaRaw, alpha, fgPixels, bgColor,
  exrioAvailable, nativeImg
},
  If[!FileExistsQ[path],
    Return[Failure["FileNotFound", <|"Path" -> path|>]]
  ];

  binAttrs = readBinaryEXRHeaderAttributes[path];

  w2c = OptionValue["CameraMatrix"];
  If[w2c === Automatic,
    w2c = Lookup[binAttrs, "worldToCamera", None];
    If[!MatrixQ[w2c, NumericQ],
      hdr = getEXRHeaderText[path];
      If[StringLength[hdr] > 0,
        w2c = parseEXRMatrix[hdr, "worldToCamera"];
      ];
    ];
    If[!MatrixQ[w2c, NumericQ],
      Module[{eye = Lookup[binAttrs, "cam_eye", If[hdr =!= "", parseEXRVector[hdr, "cam_eye"], None]],
              at  = Lookup[binAttrs, "cam_at",  If[hdr =!= "", parseEXRVector[hdr, "cam_at"], None]],
              up  = Lookup[binAttrs, "cam_up",  If[hdr =!= "", parseEXRVector[hdr, "cam_up"], None]],
              n, u, v},
        If[VectorQ[eye, NumericQ] && VectorQ[at, NumericQ] && VectorQ[up, NumericQ],
          n = Normalize[eye - at];
          u = Normalize[Cross[up, n]];
          v = Normalize[Cross[n, u]];
          w2c = Developer`ToPackedArray @ N @ {
            {u[[1]], v[[1]], -n[[1]], 0.},
            {u[[2]], v[[2]], -n[[2]], 0.},
            {u[[3]], v[[3]], -n[[3]], 0.},
            {-u . eye, -v . eye, n . eye, 1.}
          };
        ]
      ]
    ];
    If[!MatrixQ[w2c, NumericQ] || w2c == IdentityMatrix[4],
      Return[Failure["MissingCameraMatrix", <|
        "Message" -> "Could not extract valid non-identity camera transform from EXR header.",
        "Path" -> path
      |>]];
    ];
  ];

  fov = OptionValue["FOV"];
  If[fov === Automatic,
    fov = Lookup[binAttrs, "fov", None];
    If[!NumericQ[fov],
      If[hdr === "", hdr = getEXRHeaderText[path]];
      With[{m = StringCases[hdr, RegularExpression["(?mi)^fov\\s*\\(type\\s+float\\):\\s*(\\S+)"] :> "$1"]},
        fov = If[Length[m] > 0, parseSciNumberString[First[m]], 48.0];
      ]
    ];
  ];
  If[!NumericQ[fov] || fov <= 0, fov = 48.0];

  exrioAvailable = MemberQ[$Packages, "EXRIO`"] || NameQ["EXRIO`ImportEXRChannels"];
  hasDenoised = False;
  alphaRaw = None;

  If[exrioAvailable,
    Quiet @ Check[
      chNames = Import[path, "ChannelNames"];
      hasDenoised = KeyExistsQ[chNames, "denoised"] || 
                    MemberQ[Flatten[Values[chNames]], "denoised.R"] || 
                    MemberQ[Flatten[Values[chNames]], "denoised.r"];
      chType = If[hasDenoised, "denoised", "default"];
      chData = Import[path, {"ChannelData", chType}];
      If[AssociationQ[chData],
        r = Lookup[chData, "R", Lookup[chData, "r", None]];
        g = Lookup[chData, "G", Lookup[chData, "g", None]];
        b = Lookup[chData, "B", Lookup[chData, "b", None]];
        alphaRaw = Lookup[chData, "A", Lookup[chData, "a", None]];
      ];
      If[r =!= None && g =!= None && b =!= None,
        {h, w} = Dimensions[r];
        rgb = Developer`ToPackedArray @ Transpose[{Normal[r], Normal[g], Normal[b]}, {3, 1, 2}];
      ],
      Null
    ];
  ];

  If[!ListQ[rgb],
    nativeImg = Quiet @ Check[Import[path, "Image"], $Failed];
    If[ImageQ[nativeImg],
      {w, h} = ImageDimensions[nativeImg];
      rgb = Developer`ToPackedArray @ N[ImageData[nativeImg, "Real32"]];
      If[Length[Dimensions[rgb]] == 3 && Dimensions[rgb][[3]] >= 4,
        alphaRaw = rgb[[All, All, 4]];
        rgb = rgb[[All, All, 1 ;; 3]];
      ];
      hasDenoised = False;,
      Return[Failure["ImageReadError", <|"Path" -> path|>]]
    ];
  ];

  tonemapMode = OptionValue["Tonemap"];
  clipOpt = OptionValue["ClipRGB"];

  effectiveTonemap = Which[
    tonemapMode === "Clip" || (clipOpt === True && (tonemapMode === Automatic || MissingQ[tonemapMode])), "Clip",
    tonemapMode === "None" || (clipOpt === False && (tonemapMode === Automatic || MissingQ[tonemapMode])), "None",
    tonemapMode === "Exposure", "Exposure",
    tonemapMode === "Log", "Log",
    StringQ[tonemapMode], tonemapMode,
    True, "Reinhard"
  ];

  nonNegRgb = Clip[rgb, {0.0, Infinity}];
  rgb = Switch[effectiveTonemap,
    "Reinhard",
      Developer`ToPackedArray @ N[nonNegRgb / (1.0 + nonNegRgb)],
    "Clip",
      Developer`ToPackedArray @ N[Clip[rgb, {0.0, 1.0}]],
    "Exposure",
      Module[{gamma = Replace[OptionValue["ExposureGamma"], Except[_?NumericQ] -> 1.0]},
        Developer`ToPackedArray @ N[1.0 - Exp[-gamma * nonNegRgb]]
      ],
    "Log",
      Module[{maxVal = Max[nonNegRgb]},
        If[maxVal > 0.0,
          Developer`ToPackedArray @ N[Log[1.0 + nonNegRgb] / Log[1.0 + maxVal]],
          rgb
        ]
      ],
    "None",
      rgb,
    _,
      Developer`ToPackedArray @ N[nonNegRgb / (1.0 + nonNegRgb)]
  ];

  bgColor = Developer`ToPackedArray @ N[
    (rgb[[1, 1]] + rgb[[1, w]] + rgb[[h, 1]] + rgb[[h, w]]) / 4.0
  ];

  alpha = If[alphaRaw =!= None,
    Developer`ToPackedArray @ N[Normal[alphaRaw]],
    None
  ];

  fgPixels = If[alpha =!= None,
    Module[{pos = Position[alpha, _?(# > 0.1 &)]},
      If[Length[pos] > 10,
        Developer`ToPackedArray @ Map[{#[[2]] - 1, #[[1]] - 1} &, pos],
        None
      ]
    ],
    None
  ];

  <|
    "CameraMatrix" -> Developer`ToPackedArray[N[w2c]],
    "NeRFMatrix" -> RenderManToNeRFTransform[w2c],
    "FOV" -> N[fov],
    "Dimensions" -> {w, h},
    "Denoised" -> hasDenoised,
    "Tonemap" -> effectiveTonemap,
    "BackgroundColor" -> bgColor,
    "RGB" -> rgb,
    "Alpha" -> alpha,
    "ForegroundPixels" -> fgPixels,
    "File" -> path
  |>
];

Options[LoadRenderManEXRViews] = Join[
  Options[LoadRenderManEXRView],
  {
    "MaxViews" -> All,
    "Shuffle" -> False
  }
];

LoadRenderManEXRViews[input_, opts : OptionsPattern[]] := Module[{
  files, maxViews = OptionValue["MaxViews"], shuffleQ = OptionValue["Shuffle"], views
},
  files = Which[
    ListQ[input],
      Flatten[Map[If[StringQ[#] && (StringContainsQ[#, "*"] || StringContainsQ[#, "?"]), FileNames[#], {#}] &, input]],
    StringQ[input] && DirectoryQ[input],
      FileNames["*.exr", input],
    StringQ[input] && (StringContainsQ[input, "*"] || StringContainsQ[input, "?"]),
      FileNames[input],
    StringQ[input] && FileExistsQ[input],
      {input},
    StringQ[input],
      FileNames[input],
    True,
      {}
  ];
  files = Select[files, StringQ[#] && !StringContainsQ[#, ".crypto."] && FileExistsQ[#] &];

  If[shuffleQ, files = RandomSample[files]];
  If[IntegerQ[maxViews] && maxViews > 0,
    files = Take[files, Min[maxViews, Length[files]]];
  ];

  views = Map[LoadRenderManEXRView[#, FilterRules[{opts}, Options[LoadRenderManEXRView]]] &, files];
  Select[views, AssociationQ[#] && KeyExistsQ[#, "CameraMatrix"] &]
];

(* ========================================================================= *)
(* 4. Standard NeRF Dataset Loader (transforms.json with PNG/JPG/TIFF/EXR)    *)
(* ========================================================================= *)

Options[LoadNeRFDataset] = {
  "Split" -> Automatic,
  "TargetDimensions" -> Automatic,
  "MaxViews" -> All,
  "Shuffle" -> False,
  "WhiteBackground" -> False,
  "Tonemap" -> "None"
};

LoadNeRFDataset[pathOrDir_String, opts : OptionsPattern[]] := Module[{
  split = OptionValue["Split"],
  targetDim = OptionValue["TargetDimensions"],
  maxViews = OptionValue["MaxViews"],
  shuffleQ = OptionValue["Shuffle"],
  whiteBg = TrueQ @ OptionValue["WhiteBackground"],
  tonemapOpt = OptionValue["Tonemap"],
  baseDir, jsonFile, rawJson, camAngleX, camAngleY, flX, flY, wGlob, hGlob,
  cxGlob, cyGlob, frames, views = {}
},
  If[!FileExistsQ[pathOrDir] && !DirectoryQ[pathOrDir],
    Return[Failure["PathNotFound", <|"Path" -> pathOrDir|>]]
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
      True,
        Return[Failure["MissingTransformsJSON", <|"Directory" -> baseDir|>]]
    ];,
    (* Specific JSON file provided *)
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
  camAngleY = Lookup[rawJson, "camera_angle_y", None];
  flX = Lookup[rawJson, "fl_x", None];
  flY = Lookup[rawJson, "fl_y", None];
  wGlob = Lookup[rawJson, "w", None];
  hGlob = Lookup[rawJson, "h", None];
  cxGlob = Lookup[rawJson, "cx", None];
  cyGlob = Lookup[rawJson, "cy", None];

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
      w, h, c2w, w2c, fov, rawData, bgColor
    },
      relPath = Lookup[frame, "file_path", ""];
      If[relPath == "", Continue[]];

      resolvedPath = SelectFirst[{
        FileNameJoin[{baseDir, relPath}],
        FileNameJoin[{baseDir, relPath <> ".png"}],
        FileNameJoin[{baseDir, relPath <> ".jpg"}],
        FileNameJoin[{baseDir, relPath <> ".jpeg"}],
        FileNameJoin[{baseDir, relPath <> ".exr"}],
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
        ];,
        rgb = If[Length[Dimensions[rawData]] == 3, rawData[[All, All, 1 ;; 3]], ArrayReshape[rawData, {h, w, 3}]];
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
      w2c = NeRFToRenderManTransform[c2w];

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
        "CameraMatrix" -> w2c,
        "NeRFMatrix" -> c2w,
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
(* 5. Standard NeRF Dataset Exporter (transforms.json + PNG images)           *)
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
  fovDeg = N @ Lookup[v1, "FOV", 48.0];
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
            c2w = If[KeyExistsQ[v, "NeRFMatrix"] && MatrixQ[v["NeRFMatrix"], NumericQ],
              v["NeRFMatrix"],
              RenderManToNeRFTransform[v["CameraMatrix"]]
            ];

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
(* 6. Ray Generation & Sampling (Auto-detecting RenderMan w2c vs NeRF c2w)    *)
(* ========================================================================= *)

Options[GenerateCameraRays] = {
  "FOV" -> 48.0,
  "AspectRatio" -> 1.0,
  "PixelCoords" -> All,
  "Convention" -> Automatic
};

GenerateCameraRays[matrix_?MatrixQ, {w_Integer, h_Integer}, opts : OptionsPattern[]] := Module[{
  fov = OptionValue["FOV"], ar = OptionValue["AspectRatio"], pixelCoords = OptionValue["PixelCoords"],
  conv = OptionValue["Convention"],
  inv, pos, rx, ry, rz, hw, hh, uvList, nRays, uCoords, vCoords, ndcX, ndcY, rawDirs, norms, dirs, origins,
  isNeRFC2W
},
  If[!NumericQ[fov] || fov <= 0, fov = 48.0];
  If[!NumericQ[ar] || ar <= 0, ar = N[w / h]];

  isNeRFC2W = Which[
    conv === "NeRF", True,
    conv === "RenderMan", False,
    True,
      Abs[matrix[[4, 1]]] < 1*^-5 && Abs[matrix[[4, 2]]] < 1*^-5 && 
      Abs[matrix[[4, 3]]] < 1*^-5 && Abs[matrix[[4, 4]] - 1.0] < 1*^-5
  ];

  If[isNeRFC2W,
    pos = Developer`ToPackedArray[N[matrix[[1 ;; 3, 4]]]];
    rx  = Developer`ToPackedArray[Normalize[N[matrix[[1 ;; 3, 1]]]]];
    ry  = Developer`ToPackedArray[Normalize[N[matrix[[1 ;; 3, 2]]]]];
    rz  = Developer`ToPackedArray[-Normalize[N[matrix[[1 ;; 3, 3]]]]];
    ,
    inv = Inverse[matrix];
    pos = Developer`ToPackedArray[N[inv[[4, 1 ;; 3]]]];
    rx  = Developer`ToPackedArray[Normalize[N[inv[[1, 1 ;; 3]]]]];
    ry  = Developer`ToPackedArray[Normalize[N[inv[[2, 1 ;; 3]]]]];
    rz  = Developer`ToPackedArray[Normalize[N[inv[[3, 1 ;; 3]]]]];
  ];

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
(* 7. High-Performance In-Memory Ray Pool                                    *)
(* ========================================================================= *)

Options[PrecomputeNeRFLiteRayPool] = {
  "Parallel" -> Automatic
};

PrecomputeNeRFLiteRayPool[viewsInput_, opts : OptionsPattern[]] := Module[{
  views = Which[
    ListQ[viewsInput] && AssociationQ[First[viewsInput, None]], viewsInput,
    StringQ[viewsInput] && (StringEndsQ[ToLowerCase[viewsInput], ".json"] || (DirectoryQ[viewsInput] && Length[FileNames["transforms*.json", viewsInput]] > 0)),
      LoadNeRFDataset[viewsInput],
    StringQ[viewsInput] || (ListQ[viewsInput] && StringQ[First[viewsInput, None]]),
      LoadRenderManEXRViews[viewsInput],
    True, {}
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
