(* ::Package:: *)

BeginPackage["WLNeRFLite`MultiLPE`", {
  "NeuralNetworks`",
  "WLNeRFLite`Data`",
  "WLNeRFLite`Net`"
}];

LoadRenderManMultiLPEView::usage = "LoadRenderManMultiLPEView[path, opts] loads camera transforms and all LPE/intrinsic channels (Albedo, Direct/Indirect Diffuse, Direct/Indirect Specular, Beauty, Normal) from a RenderMan EXR.";
LoadRenderManMultiLPEViews::usage = "LoadRenderManMultiLPEViews[pathsOrDir, opts] loads a collection of multi-view Multi-LPE RenderMan EXR files.";
PrecomputeMultiLPERayPool::usage = "PrecomputeMultiLPERayPool[views, opts] builds an in-memory ray pool containing multi-task LPE targets.";
CreateMultiLPENeRFNet::usage = "CreateMultiLPENeRFNet[opts] creates a coordinate MLP with view-independent and view-dependent heads and in-graph volumetric quadrature for all LPEs.";
CreateMultiLPETrainingNet::usage = "CreateMultiLPETrainingNet[fieldNet, opts] wraps a Multi-LPE field net with multi-task tonemapped log-L1 and recomposition losses.";
CreateMultiLPEBatchGenerator::usage = "CreateMultiLPEBatchGenerator[viewsOrPool, opts] creates a stateful batch generator for Multi-LPE NetTrain.";
TrainMultiLPENeRFModel::usage = "TrainMultiLPENeRFModel[net, viewsOrPool, opts] trains a Multi-LPE NeRF on GPU.";
ExtractMultiLPEInferenceNet::usage = "ExtractMultiLPEInferenceNet[trainedNet] extracts the core Multi-LPE prediction net.";
RenderMultiLPENeRFView::usage = "RenderMultiLPENeRFView[net, cameraMatrix, {w, h}, opts] renders novel views under arbitrary relighting modulation coefficients.";
RenderMultiLPETurntable::usage = "RenderMultiLPETurntable[net, opts] renders a 360-degree relit orbiting turntable animation.";
InteractiveMultiLPENeRFViewer::usage = "InteractiveMultiLPENeRFViewer[net, viewsOrMatrix, opts] creates an interactive Manipulate panel for real-time relighting in Mathematica.";

Begin["`Private`"];

Quiet @ SetEnvironment["MXNET_USE_FUSION" -> "0"];

findEXRTool[name_String] := Module[{paths},
  paths = {
    "/opt/pixar/RenderManProServer-28.0b1/bin/" <> name,
    "/opt/pixar/RenderManProServer-28.0b1/bin/exrinfo",
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

parseSciNum[str_String] := Module[{norm, val},
  norm = StringReplace[StringTrim[str], RegularExpression["(?i)e\\+?(-?)0*(\\d+)"] :> "*^$1$2"];
  val = Quiet @ Check[ToExpression @ norm, Missing["NotANumber", str]];
  If[NumericQ[val], N[val], Indeterminate]
];

parseEXRMat[headerText_String, key_String] := Module[{match, rows, mat},
  match = StringCases[headerText, RegularExpression["(?mis)^\\Q" <> key <> "\\E\\s*\\(type\\s+m44f\\):\\s*\\((.*?)\\)"] :> "$1"];
  If[match === {},
    Missing["KeyAbsent", key],
    rows = StringSplit[StringTrim @ First[match], "\n"];
    mat = parseSciNum /@ StringSplit[StringTrim[#], Whitespace..] & /@ rows;
    If[MatrixQ[mat, NumericQ] && Det[mat] != 0,
      Developer`ToPackedArray[N[mat]],
      IdentityMatrix[4]
    ]
  ]
];

parseEXRVec[headerText_String, key_String] := Module[{match, vec},
  match = StringCases[headerText, RegularExpression["(?mi)^\\Q" <> key <> "\\E\\s*\\(type\\s+v3f\\):\\s*\\(([^\\)]*)\\)"] :> "$1"];
  If[match === {},
    Missing["KeyAbsent", key],
    vec = parseSciNum /@ StringSplit[StringTrim @ First[match], Whitespace..];
    If[VectorQ[vec, NumericQ], Developer`ToPackedArray[N[vec]], {0., 0., 0.}]
  ]
];

Options[LoadRenderManMultiLPEView] = {
  "FOV" -> Automatic,
  "CameraMatrix" -> Automatic,
  "ClipRGB" -> True,
  "NormalSpace" -> "World"
};

LoadRenderManMultiLPEView[path_String, opts : OptionsPattern[]] := Module[{
  hdr, w2c, eye, at, up, fov,
  albData, lightData, defData, geomData,
  w, h,
  albRGB, diffDirRGB, diffIndRGB, specDirRGB, specIndRGB, beautyRGB, normalXYZ, normalCamXYZ, alpha,
  toRGB3, dist, suggestedBounds
},
  If[!FileExistsQ[path],
    Return[Failure["FileNotFound", <|"Path" -> path|>]]
  ];

  hdr = getEXRHeaderText[path];

  (* 1. Camera Matrix *)
  w2c = OptionValue["CameraMatrix"];
  If[w2c === Automatic || !MatrixQ[w2c, NumericQ],
    w2c = parseEXRMat[hdr, "worldToCamera"];
    If[!MatrixQ[w2c, NumericQ] || w2c == IdentityMatrix[4],
      eye = parseEXRVec[hdr, "cam_eye"];
      at  = parseEXRVec[hdr, "cam_at"];
      up  = parseEXRVec[hdr, "cam_up"];
      If[VectorQ[eye, NumericQ] && VectorQ[at, NumericQ] && VectorQ[up, NumericQ],
        Module[{n = Normalize[eye - at], u, v},
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
  ];

  If[!MatrixQ[w2c, NumericQ], w2c = IdentityMatrix[4]];

  (* 2. FOV and Distances *)
  fov = OptionValue["FOV"];
  If[!NumericQ[fov],
    With[{m = StringCases[hdr, RegularExpression["(?mi)^fov\\s*\\(type\\s+float\\):\\s*(\\S+)"] :> "$1"]},
      fov = If[Length[m] > 0, parseSciNum[First[m]], 48.0];
    ]
  ];
  If[!NumericQ[fov] || fov <= 0, fov = 48.0];

  eye = parseEXRVec[hdr, "cam_eye"];
  at  = parseEXRVec[hdr, "cam_at"];
  dist = If[VectorQ[eye, NumericQ] && VectorQ[at, NumericQ], EuclideanDistance[eye, at], 18.0];
  suggestedBounds = {Max[1.0, dist - 6.5], dist + 6.5};

  (* 3. Read EXR channel dictionaries *)
  albData   = Quiet @ Check[Import[path, {"ChannelData", "albedo"}], <||>];
  lightData = Quiet @ Check[Import[path, {"ChannelData", "lighting"}], <||>];
  defData   = Quiet @ Check[Import[path, {"ChannelData", "denoised"}], <||>];
  If[Length[defData] == 0,
    defData = Quiet @ Check[Import[path, {"ChannelData", "default"}], <||>];
  ];
  geomData  = Quiet @ Check[Import[path, {"ChannelData", "geometry"}], <||>];

  toRGB3[rRaw_, gRaw_, bRaw_] := Module[{r = Normal[rRaw], g = Normal[gRaw], b = Normal[bRaw]},
    Developer`ToPackedArray @ N @ Transpose[{r, g, b}, {3, 1, 2}]
  ];

  (* Extract Albedo *)
  albRGB = If[KeyExistsQ[albData, "r"],
    toRGB3[albData["r"], albData["g"], albData["b"]],
    None
  ];

  (* Extract Direct Diffuse *)
  diffDirRGB = If[KeyExistsQ[lightData, "direct.diffuse.r"],
    toRGB3[lightData["direct.diffuse.r"], lightData["direct.diffuse.g"], lightData["direct.diffuse.b"]],
    None
  ];

  (* Extract Indirect Diffuse *)
  diffIndRGB = If[KeyExistsQ[lightData, "indirect.diffuse.r"],
    toRGB3[lightData["indirect.diffuse.r"], lightData["indirect.diffuse.g"], lightData["indirect.diffuse.b"]],
    None
  ];

  (* Extract Direct Specular *)
  specDirRGB = If[KeyExistsQ[lightData, "direct.specular.r"],
    toRGB3[lightData["direct.specular.r"], lightData["direct.specular.g"], lightData["direct.specular.b"]],
    None
  ];

  (* Extract Indirect Specular *)
  specIndRGB = If[KeyExistsQ[lightData, "indirect.specular.r"],
    toRGB3[lightData["indirect.specular.r"], lightData["indirect.specular.g"], lightData["indirect.specular.b"]],
    None
  ];

  (* Extract Beauty & Alpha *)
  beautyRGB = If[KeyExistsQ[defData, "R"],
    toRGB3[defData["R"], defData["G"], defData["B"]],
    If[KeyExistsQ[defData, "r"], toRGB3[defData["r"], defData["g"], defData["b"]], None]
  ];

  alpha = If[KeyExistsQ[defData, "A"],
    Developer`ToPackedArray @ N @ Normal[defData["A"]],
    If[KeyExistsQ[defData, "a"], Developer`ToPackedArray @ N @ Normal[defData["a"]], None]
  ];

  (* Extract Normal (RenderMan Camera Space) *)
  normalCamXYZ = If[KeyExistsQ[geomData, "Nn.x"],
    toRGB3[geomData["Nn.x"], geomData["Nn.y"], geomData["Nn.z"]],
    None
  ];

  {h, w} = If[beautyRGB =!= None, Dimensions[beautyRGB][[1 ;; 2]], {512, 512}];

  (* Compute Screen-Space Weingarten ShapeIndex & Curvedness if P and Nn exist *)
  Module[{posXYZ, maskHW, signFlip, pu, pv, nu, nv, e, f, g, l, m, nCoeff, detI, validDet, invDetI, w11, w12, w21, w22, trW, detW, sqrtDisc, k1, k2, dk, sk, validK, shapeHW, curvHW},
    posXYZ = If[KeyExistsQ[geomData, "P.x"],
      toRGB3[geomData["P.x"], geomData["P.y"], geomData["P.z"]],
      None
    ];
    If[posXYZ =!= None && normalCamXYZ =!= None,
      (* Enforce outward camera-facing orientation: faceforward(Nn, P) *)
      signFlip = 1.0 - 2.0 * UnitStep[Total[posXYZ * normalCamXYZ, {3}]];
      normalCamXYZ = Developer`ToPackedArray[N[normalCamXYZ * Transpose[ConstantArray[signFlip, 3], {3, 1, 2}]]];
      maskHW = If[alpha =!= None, UnitStep[alpha - 0.05], ConstantArray[1.0, {h, w}]];
      Module[{pSmooth = GaussianFilter[posXYZ, 1.0], nSmooth = GaussianFilter[normalCamXYZ, 1.0]},
        pu = DerivativeFilter[pSmooth, {1, 0}];
        pv = DerivativeFilter[pSmooth, {0, 1}];
        nu = DerivativeFilter[nSmooth, {1, 0}];
        nv = DerivativeFilter[nSmooth, {0, 1}];
      ];
      e = Total[pu * pu, {3}];
      f = Total[pu * pv, {3}];
      g = Total[pv * pv, {3}];
      l = -Total[pu * nu, {3}];
      m = -0.5 * (Total[pu * nv, {3}] + Total[pv * nu, {3}]);
      nCoeff = -Total[pv * nv, {3}];
      detI = e * g - f^2;
      validDet = UnitStep[Abs[detI] - 10^-8];
      invDetI = validDet / (detI + (1.0 - validDet));
      w11 = (g * l - f * m) * invDetI;
      w12 = (g * m - f * nCoeff) * invDetI;
      w21 = (-f * l + e * m) * invDetI;
      w22 = (-f * m + e * nCoeff) * invDetI;
      trW = w11 + w22;
      detW = w11 * w22 - w12 * w21;
      sqrtDisc = Sqrt[Ramp[trW^2 - 4.0 * detW]];
      k1 = 0.5 * (trW + sqrtDisc) * maskHW;
      k2 = 0.5 * (trW - sqrtDisc) * maskHW;
      dk = k1 - k2;
      sk = k1 + k2;
      validK = UnitStep[dk - 10^-8];
      shapeHW = Clip[-(2.0 / Pi) * ArcTan[(sk * validK) / (dk + (1.0 - validK))] * validK * maskHW, {-1.0, 1.0}];
      curvHW = Clip[Sqrt[0.5 * (k1^2 + k2^2)] * maskHW, {0.0, 50.0}];,
      shapeHW = ConstantArray[0.0, {h, w}];
      curvHW = ConstantArray[0.0, {h, w}];
    ];

    (* Transform camera-space Nn into World Space via Transpose[w2c[[1;;3, 1;;3]]] *)
    normalXYZ = If[normalCamXYZ =!= None && OptionValue["NormalSpace"] === "World",
      Developer`ToPackedArray[N[normalCamXYZ . Transpose[w2c[[1 ;; 3, 1 ;; 3]]]]],
      normalCamXYZ
    ];

    <|
      "CameraMatrix" -> Developer`ToPackedArray[N[w2c]],
      "FOV" -> N[fov],
      "Dimensions" -> {w, h},
      "Eye" -> eye,
      "LookAt" -> at,
      "Distance" -> dist,
      "SuggestedRayBounds" -> suggestedBounds,
      "Albedo" -> albRGB,
      "DirectDiffuse" -> diffDirRGB,
      "IndirectDiffuse" -> diffIndRGB,
      "DirectSpecular" -> specDirRGB,
      "IndirectSpecular" -> specIndRGB,
      "Beauty" -> beautyRGB,
      "Normal" -> normalXYZ,
      "NormalCamera" -> normalCamXYZ,
      "ShapeIndex" -> Developer`ToPackedArray[N @ ArrayReshape[shapeHW, {h, w, 1}]],
      "Curvedness" -> Developer`ToPackedArray[N @ ArrayReshape[curvHW, {h, w, 1}]],
      "Alpha" -> alpha,
      "File" -> path
    |>
  ]
];

Options[LoadRenderManMultiLPEViews] = Join[
  Options[LoadRenderManMultiLPEView],
  {
    "MaxViews" -> All
  }
];

LoadRenderManMultiLPEViews[input_, opts : OptionsPattern[]] := Module[{
  files, maxViews = OptionValue["MaxViews"], loaded
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
    True, {}
  ];

  files = Sort[Select[files, !StringEndsQ[#, ".crypto.exr"] &]];
  If[IntegerQ[maxViews] && maxViews > 0, files = Take[files, UpTo[maxViews]]];

  Module[{subOpts = FilterRules[{opts}, Options[LoadRenderManMultiLPEView]]},
    loaded = Map[LoadRenderManMultiLPEView[#, Sequence @@ subOpts] &, files];
  ];
  Select[loaded, !FailureQ[#] &]
];

Options[PrecomputeMultiLPERayPool] = {
  "Parallel" -> False
};

PrecomputeMultiLPERayPool[views_List, opts : OptionsPattern[]] := Module[{
  allFgOrigins = {}, allFgDirs = {},
  allFgAlb = {}, allFgDiffDir = {}, allFgDiffInd = {}, allFgSpecDir = {}, allFgSpecInd = {}, allFgBeauty = {}, allFgNorm = {}, allFgShape = {}, allFgCurv = {},
  allBgOrigins = {}, allBgDirs = {},
  allBgAlb = {}, allBgDiffDir = {}, allBgDiffInd = {}, allBgSpecDir = {}, allBgSpecInd = {}, allBgBeauty = {}, allBgNorm = {}, allBgShape = {}, allBgCurv = {},
  suggestedBounds = {12.0, 25.0}
},
  If[Length[views] == 0, Return[Failure["EmptyViews", <||>]]];

  If[KeyExistsQ[views[[1]], "SuggestedRayBounds"],
    suggestedBounds = views[[1]]["SuggestedRayBounds"];
  ];

  Do[
    Module[{
      w, h, camMat, fov, alpha, rays, origs, dirs,
      alb, diffDir, diffInd, specDir, specInd, beauty, norm, shape, curv,
      isFg
    },
      {w, h} = v["Dimensions"];
      camMat = v["CameraMatrix"];
      fov = v["FOV"];
      alpha = v["Alpha"];

      rays = GenerateCameraRays[camMat, {w, h}, "FOV" -> fov];
      origs = rays["Origins"];
      dirs = rays["Directions"];

      alb     = Flatten[v["Albedo"], 1];
      diffDir = Flatten[v["DirectDiffuse"], 1];
      diffInd = Flatten[v["IndirectDiffuse"], 1];
      specDir = Flatten[v["DirectSpecular"], 1];
      specInd = Flatten[v["IndirectSpecular"], 1];
      beauty  = Flatten[v["Beauty"], 1];
      norm    = If[KeyExistsQ[v, "Normal"] && v["Normal"] =!= None,
        Flatten[v["Normal"], 1],
        ConstantArray[{0., 1., 0.}, Length[origs]]
      ];
      shape   = If[KeyExistsQ[v, "ShapeIndex"] && v["ShapeIndex"] =!= None,
        Flatten[v["ShapeIndex"], 1],
        ConstantArray[{0.}, Length[origs]]
      ];
      curv    = If[KeyExistsQ[v, "Curvedness"] && v["Curvedness"] =!= None,
        Flatten[v["Curvedness"], 1],
        ConstantArray[{0.}, Length[origs]]
      ];

      isFg = If[MatrixQ[alpha, NumericQ], UnitStep[Flatten[alpha] - 0.05], ConstantArray[1, Length[origs]]];

      allFgOrigins = Join[allFgOrigins, Pick[origs, isFg, 1]];
      allFgDirs    = Join[allFgDirs, Pick[dirs, isFg, 1]];
      allFgAlb     = Join[allFgAlb, Pick[alb, isFg, 1]];
      allFgDiffDir = Join[allFgDiffDir, Pick[diffDir, isFg, 1]];
      allFgDiffInd = Join[allFgDiffInd, Pick[diffInd, isFg, 1]];
      allFgSpecDir = Join[allFgSpecDir, Pick[specDir, isFg, 1]];
      allFgSpecInd = Join[allFgSpecInd, Pick[specInd, isFg, 1]];
      allFgBeauty  = Join[allFgBeauty, Pick[beauty, isFg, 1]];
      allFgNorm    = Join[allFgNorm, Pick[norm, isFg, 1]];
      allFgShape   = Join[allFgShape, Pick[shape, isFg, 1]];
      allFgCurv    = Join[allFgCurv, Pick[curv, isFg, 1]];

      allBgOrigins = Join[allBgOrigins, Pick[origs, isFg, 0]];
      allBgDirs    = Join[allBgDirs, Pick[dirs, isFg, 0]];
      allBgAlb     = Join[allBgAlb, Pick[alb, isFg, 0]];
      allBgDiffDir = Join[allBgDiffDir, Pick[diffDir, isFg, 0]];
      allBgDiffInd = Join[allBgDiffInd, Pick[diffInd, isFg, 0]];
      allBgSpecDir = Join[allBgSpecDir, Pick[specDir, isFg, 0]];
      allBgSpecInd = Join[allBgSpecInd, Pick[specInd, isFg, 0]];
      allBgBeauty  = Join[allBgBeauty, Pick[beauty, isFg, 0]];
      allBgNorm    = Join[allBgNorm, Pick[norm, isFg, 0]];
      allBgShape   = Join[allBgShape, Pick[shape, isFg, 0]];
      allBgCurv    = Join[allBgCurv, Pick[curv, isFg, 0]];
    ],
    {v, views}
  ];

  <|
    "Type" -> "MultiLPERayPool",
    "SuggestedRayBounds" -> suggestedBounds,
    "Foreground" -> <|
      "Origins" -> Developer`ToPackedArray[N[allFgOrigins]],
      "Directions" -> Developer`ToPackedArray[N[allFgDirs]],
      "Albedo" -> Developer`ToPackedArray[N[allFgAlb]],
      "DirectDiffuse" -> Developer`ToPackedArray[N[allFgDiffDir]],
      "IndirectDiffuse" -> Developer`ToPackedArray[N[allFgDiffInd]],
      "DirectSpecular" -> Developer`ToPackedArray[N[allFgSpecDir]],
      "IndirectSpecular" -> Developer`ToPackedArray[N[allFgSpecInd]],
      "Beauty" -> Developer`ToPackedArray[N[allFgBeauty]],
      "Normal" -> Developer`ToPackedArray[N[allFgNorm]],
      "ShapeIndex" -> Developer`ToPackedArray[N[allFgShape]],
      "Curvedness" -> Developer`ToPackedArray[N[allFgCurv]],
      "Count" -> Length[allFgOrigins]
    |>,
    "Background" -> <|
      "Origins" -> Developer`ToPackedArray[N[allBgOrigins]],
      "Directions" -> Developer`ToPackedArray[N[allBgDirs]],
      "Albedo" -> Developer`ToPackedArray[N[allBgAlb]],
      "DirectDiffuse" -> Developer`ToPackedArray[N[allBgDiffDir]],
      "IndirectDiffuse" -> Developer`ToPackedArray[N[allBgDiffInd]],
      "DirectSpecular" -> Developer`ToPackedArray[N[allBgSpecDir]],
      "IndirectSpecular" -> Developer`ToPackedArray[N[allBgSpecInd]],
      "Beauty" -> Developer`ToPackedArray[N[allBgBeauty]],
      "Normal" -> Developer`ToPackedArray[N[allBgNorm]],
      "ShapeIndex" -> Developer`ToPackedArray[N[allBgShape]],
      "Curvedness" -> Developer`ToPackedArray[N[allBgCurv]],
      "Count" -> Length[allBgOrigins]
    |>,
    "TotalRays" -> Length[allFgOrigins] + Length[allBgOrigins]
  |>
];

Options[CreateMultiLPENeRFNet] = {
  "NumSamples" -> 64,
  "HarmonicBands" -> 6,
  "HiddenDim" -> 128,
  "NumLayers" -> 4,
  "IncludeNormals" -> True,
  "IncludeDifferentialGeometry" -> False,
  "IncludeDepth" -> False
};

CreateMultiLPENeRFNet[opts : OptionsPattern[]] := Module[{
  numSamples = OptionValue["NumSamples"],
  bands = OptionValue["HarmonicBands"],
  hiddenDim = OptionValue["HiddenDim"],
  numLayers = OptionValue["NumLayers"],
  withNormals = TrueQ @ OptionValue["IncludeNormals"],
  withDiffGeom = TrueQ @ OptionValue["IncludeDifferentialGeometry"],
  withDepth = TrueQ @ OptionValue["IncludeDepth"],
  encodedDim, trilStrict, layers = <||>, edges = {}, backbone, radFeatureNode
},
  If[!IntegerQ[numSamples] || numSamples <= 0, numSamples = 64];
  If[!IntegerQ[bands] || bands <= 0, bands = 6];
  If[!IntegerQ[hiddenDim] || hiddenDim <= 0, hiddenDim = 128];
  If[!IntegerQ[numLayers] || numLayers <= 0, numLayers = 4];

  encodedDim = 3 * (1 + 2 * bands);
  trilStrict = Table[If[j < i, 1.0, 0.0], {i, 1, numSamples}, {j, 1, numSamples}];

  (* 1. Spatial Coordinate Encoder & Shared Trunk *)
  layers["encoder"] = NetMapOperator[CreateHarmonicEncoder[bands]];
  backbone = WLNeRFLite`Net`Private`CreateFieldBackbone[encodedDim, hiddenDim, numLayers, True];
  layers["backbone"] = NetMapOperator[backbone];

  (* 2. Density Head: Feature -> Linear -> Softplus -> Density sigma >= 0 *)
  layers["density_linear"] = NetMapOperator[LinearLayer[1]];
  layers["density_reshape"] = ReshapeLayer[{numSamples}];
  layers["density_softplus"] = ElementwiseLayer["SoftPlus"];

  (* 3. Intrinsic & Differential Geometry Heads ("Tilde Class") *)
  (* Albedo: in [0, 1]^3 *)
  layers["albedo_linear"] = NetMapOperator[LinearLayer[3]];
  layers["albedo_sigmoid"] = ElementwiseLayer[LogisticSigmoid];

  (* Surface Normal: unit vector in [-1, 1]^3 *)
  If[withNormals || withDiffGeom,
    layers["normal_linear"] = NetMapOperator[LinearLayer[3]];
    layers["normal_tanh"] = ElementwiseLayer[Tanh];
  ];

  (* Differential Geometry Invariants: Shape Index S in [-1, 1] & Curvedness C >= 0 *)
  If[withDiffGeom,
    layers["shape_linear"] = NetMapOperator[LinearLayer[1]];
    layers["shape_tanh"] = ElementwiseLayer[Tanh];
    layers["curv_linear"] = NetMapOperator[LinearLayer[1]];
    layers["curv_softplus"] = ElementwiseLayer["SoftPlus"];
    (* Cross-Stream Geometric Conditioning: concatenate [backbone, N, S, C] *)
    layers["geom_cat"] = CatenateLayer[2];
    radFeatureNode = "geom_cat";,
    radFeatureNode = "backbone";
  ];

  (* 4. Radiometric LPE Heads *)
  (* Direct Diffuse >= 0 *)
  layers["diff_dir_linear"] = NetMapOperator[LinearLayer[3]];
  layers["diff_dir_softplus"] = ElementwiseLayer["SoftPlus"];

  (* Indirect Diffuse >= 0 *)
  layers["diff_ind_linear"] = NetMapOperator[LinearLayer[3]];
  layers["diff_ind_softplus"] = ElementwiseLayer["SoftPlus"];

  (* Direct Specular >= 0 *)
  layers["spec_dir_linear"] = NetMapOperator[LinearLayer[3]];
  layers["spec_dir_softplus"] = ElementwiseLayer["SoftPlus"];

  (* Indirect Specular >= 0 *)
  layers["spec_ind_linear"] = NetMapOperator[LinearLayer[3]];
  layers["spec_ind_softplus"] = ElementwiseLayer["SoftPlus"];

  (* 5. Differentiable Volumetric Quadrature Integration *)
  layers["tau"] = ThreadingLayer[Times];
  layers["tau_cum"] = LinearLayer[
    numSamples,
    "Input" -> numSamples,
    "Weights" -> trilStrict,
    "Biases" -> None,
    "LearningRateMultipliers" -> {"Weights" -> 0.0}
  ];
  layers["transmittance"] = ElementwiseLayer[Exp[-#] &];
  layers["alpha"] = ElementwiseLayer[1.0 - Exp[-#] &];
  layers["weights"] = ThreadingLayer[Times];
  layers["weights_unsqueeze"] = ReshapeLayer[{numSamples, 1}];

  (* Independent Quadrature Accumulations *)
  layers["albedo_weighted"] = ThreadingLayer[Times];
  layers["albedo_accum"] = AggregationLayer[Total, 1];

  layers["diff_dir_weighted"] = ThreadingLayer[Times];
  layers["diff_dir_accum"] = AggregationLayer[Total, 1];

  layers["diff_ind_weighted"] = ThreadingLayer[Times];
  layers["diff_ind_accum"] = AggregationLayer[Total, 1];

  layers["spec_dir_weighted"] = ThreadingLayer[Times];
  layers["spec_dir_accum"] = AggregationLayer[Total, 1];

  layers["spec_ind_weighted"] = ThreadingLayer[Times];
  layers["spec_ind_accum"] = AggregationLayer[Total, 1];

  If[withNormals || withDiffGeom,
    layers["normal_weighted"] = ThreadingLayer[Times];
    layers["normal_accum"] = AggregationLayer[Total, 1];
  ];

  If[withDiffGeom,
    layers["shape_weighted"] = ThreadingLayer[Times];
    layers["shape_accum"] = AggregationLayer[Total, 1];
    layers["curv_weighted"] = ThreadingLayer[Times];
    layers["curv_accum"] = AggregationLayer[Total, 1];
  ];

  layers["opacity_accum"] = AggregationLayer[Total, 1];

  (* 6. In-Graph Physical Recomposition:
        Beauty = Albedo * (DirectDiffuse + IndirectDiffuse) + DirectSpecular + IndirectSpecular
  *)
  layers["total_diffuse"] = ThreadingLayer[Plus];
  layers["diffuse_radiance"] = ThreadingLayer[Times];
  layers["total_specular"] = ThreadingLayer[Plus];
  layers["beauty_recomp"] = ThreadingLayer[Plus];

  edges = {
    NetPort["SamplePoints"] -> "encoder" -> "backbone",

    "backbone" -> "density_linear" -> "density_reshape" -> "density_softplus",
    {"density_softplus", NetPort["Steps"]} -> "tau" -> "tau_cum" -> "transmittance",
    "tau" -> "alpha",
    {"transmittance", "alpha"} -> "weights" -> "weights_unsqueeze",
    "weights_unsqueeze" -> "opacity_accum" -> NetPort["Opacity"],

    (* Albedo *)
    "backbone" -> "albedo_linear" -> "albedo_sigmoid",
    {"albedo_sigmoid", "weights_unsqueeze"} -> "albedo_weighted" -> "albedo_accum" -> NetPort["Albedo"],

    (* Direct Diffuse *)
    radFeatureNode -> "diff_dir_linear" -> "diff_dir_softplus",
    {"diff_dir_softplus", "weights_unsqueeze"} -> "diff_dir_weighted" -> "diff_dir_accum" -> NetPort["DirectDiffuse"],

    (* Indirect Diffuse *)
    radFeatureNode -> "diff_ind_linear" -> "diff_ind_softplus",
    {"diff_ind_softplus", "weights_unsqueeze"} -> "diff_ind_weighted" -> "diff_ind_accum" -> NetPort["IndirectDiffuse"],

    (* Direct Specular *)
    radFeatureNode -> "spec_dir_linear" -> "spec_dir_softplus",
    {"spec_dir_softplus", "weights_unsqueeze"} -> "spec_dir_weighted" -> "spec_dir_accum" -> NetPort["DirectSpecular"],

    (* Indirect Specular *)
    radFeatureNode -> "spec_ind_linear" -> "spec_ind_softplus",
    {"spec_ind_softplus", "weights_unsqueeze"} -> "spec_ind_weighted" -> "spec_ind_accum" -> NetPort["IndirectSpecular"],

    (* Recomposition *)
    {"diff_dir_accum", "diff_ind_accum"} -> "total_diffuse",
    {"albedo_accum", "total_diffuse"} -> "diffuse_radiance",
    {"spec_dir_accum", "spec_ind_accum"} -> "total_specular",
    {"diffuse_radiance", "total_specular"} -> "beauty_recomp" -> NetPort["Beauty"]
  };

  If[withNormals || withDiffGeom,
    AppendTo[edges, "backbone" -> "normal_linear" -> "normal_tanh"];
    AppendTo[edges, {"normal_tanh", "weights_unsqueeze"} -> "normal_weighted" -> "normal_accum" -> NetPort["Normal"]];
  ];

  If[withDiffGeom,
    AppendTo[edges, "backbone" -> "shape_linear" -> "shape_tanh"];
    AppendTo[edges, {"shape_tanh", "weights_unsqueeze"} -> "shape_weighted" -> "shape_accum" -> NetPort["ShapeIndex"]];
    AppendTo[edges, "backbone" -> "curv_linear" -> "curv_softplus"];
    AppendTo[edges, {"curv_softplus", "weights_unsqueeze"} -> "curv_weighted" -> "curv_accum" -> NetPort["Curvedness"]];
    AppendTo[edges, {"backbone", "normal_tanh", "shape_tanh", "curv_softplus"} -> "geom_cat"];
  ];

  NetGraph[layers, edges,
    "SamplePoints" -> {numSamples, 3},
    "Steps" -> {numSamples}
  ]
];

Options[CreateMultiLPETrainingNet] = {
  "WeightAlbedo" -> 1.0,
  "WeightDirectDiffuse" -> 1.0,
  "WeightIndirectDiffuse" -> 1.0,
  "WeightDirectSpecular" -> 1.0,
  "WeightIndirectSpecular" -> 1.0,
  "WeightBeauty" -> 2.0,
  "WeightNormal" -> 1.0,
  "WeightShapeIndex" -> 0.5,
  "WeightCurvedness" -> 0.5
};

CreateMultiLPETrainingNet[multiLpeNet_NetGraph, opts : OptionsPattern[]] := Module[{
  wAlb = N @ OptionValue["WeightAlbedo"],
  wDiffDir = N @ OptionValue["WeightDirectDiffuse"],
  wDiffInd = N @ OptionValue["WeightIndirectDiffuse"],
  wSpecDir = N @ OptionValue["WeightDirectSpecular"],
  wSpecInd = N @ OptionValue["WeightIndirectSpecular"],
  wBeauty = N @ OptionValue["WeightBeauty"],
  wNorm = N @ OptionValue["WeightNormal"],
  wShape = N @ OptionValue["WeightShapeIndex"],
  wCurv = N @ OptionValue["WeightCurvedness"],
  outPorts, hasNormals, hasDiffGeom, layers = <||>, edges = {}, inPortSpecs = <||>, totalLossInputs = {},
  scaledLoss
},
  outPorts = Keys @ Quiet @ Information[multiLpeNet, "OutputPorts"];
  hasNormals = MemberQ[outPorts, "Normal"];
  hasDiffGeom = MemberQ[outPorts, "ShapeIndex"] && MemberQ[outPorts, "Curvedness"];

  scaledLoss[lossNet_, weight_] := If[weight == 1.0,
    lossNet,
    NetChain[{lossNet, ElementwiseLayer[# * weight &]}]
  ];

  layers["net"] = multiLpeNet;
  layers["loss_albedo"] = scaledLoss[MeanAbsoluteLossLayer[], wAlb];
  layers["loss_diff_dir"] = scaledLoss[WLNeRFLite`Net`CreateLogL1LossNet[], wDiffDir];
  layers["loss_diff_ind"] = scaledLoss[WLNeRFLite`Net`CreateLogL1LossNet[], wDiffInd];
  layers["loss_spec_dir"] = scaledLoss[WLNeRFLite`Net`CreateLogL1LossNet[], wSpecDir];
  layers["loss_spec_ind"] = scaledLoss[WLNeRFLite`Net`CreateLogL1LossNet[], wSpecInd];
  layers["loss_beauty"] = scaledLoss[WLNeRFLite`Net`CreateLogL1LossNet[], wBeauty];

  totalLossInputs = {
    "loss_albedo", "loss_diff_dir", "loss_diff_ind",
    "loss_spec_dir", "loss_spec_ind", "loss_beauty"
  };

  edges = {
    NetPort["SamplePoints"] -> NetPort["net", "SamplePoints"],
    NetPort["Steps"] -> NetPort["net", "Steps"],

    NetPort["net", "Albedo"] -> NetPort["loss_albedo", "Input"],
    NetPort["TargetAlbedo"] -> NetPort["loss_albedo", "Target"],

    NetPort["net", "DirectDiffuse"] -> NetPort["loss_diff_dir", "Input"],
    NetPort["TargetDirectDiffuse"] -> NetPort["loss_diff_dir", "Target"],

    NetPort["net", "IndirectDiffuse"] -> NetPort["loss_diff_ind", "Input"],
    NetPort["TargetIndirectDiffuse"] -> NetPort["loss_diff_ind", "Target"],

    NetPort["net", "DirectSpecular"] -> NetPort["loss_spec_dir", "Input"],
    NetPort["TargetDirectSpecular"] -> NetPort["loss_spec_dir", "Target"],

    NetPort["net", "IndirectSpecular"] -> NetPort["loss_spec_ind", "Input"],
    NetPort["TargetIndirectSpecular"] -> NetPort["loss_spec_ind", "Target"],

    NetPort["net", "Beauty"] -> NetPort["loss_beauty", "Input"],
    NetPort["TargetBeauty"] -> NetPort["loss_beauty", "Target"]
  };

  If[hasNormals,
    layers["loss_normal"] = scaledLoss[MeanAbsoluteLossLayer[], wNorm];
    AppendTo[totalLossInputs, "loss_normal"];
    AppendTo[edges, NetPort["net", "Normal"] -> NetPort["loss_normal", "Input"]];
    AppendTo[edges, NetPort["TargetNormal"] -> NetPort["loss_normal", "Target"]];
  ];

  If[hasDiffGeom,
    layers["loss_shape"] = scaledLoss[MeanAbsoluteLossLayer[], wShape];
    layers["loss_curv"] = scaledLoss[WLNeRFLite`Net`CreateLogL1LossNet[], wCurv];
    AppendTo[totalLossInputs, "loss_shape"];
    AppendTo[totalLossInputs, "loss_curv"];
    AppendTo[edges, NetPort["net", "ShapeIndex"] -> NetPort["loss_shape", "Input"]];
    AppendTo[edges, NetPort["TargetShapeIndex"] -> NetPort["loss_shape", "Target"]];
    AppendTo[edges, NetPort["net", "Curvedness"] -> NetPort["loss_curv", "Input"]];
    AppendTo[edges, NetPort["TargetCurvedness"] -> NetPort["loss_curv", "Target"]];
  ];

  layers["total_loss"] = TotalLayer[];
  AppendTo[edges, totalLossInputs -> "total_loss" -> NetPort["Loss"]];

  inPortSpecs = <|
    "SamplePoints" -> Information[multiLpeNet, "InputPorts"]["SamplePoints"],
    "Steps" -> Information[multiLpeNet, "InputPorts"]["Steps"],
    "TargetAlbedo" -> {3},
    "TargetDirectDiffuse" -> {3},
    "TargetIndirectDiffuse" -> {3},
    "TargetDirectSpecular" -> {3},
    "TargetIndirectSpecular" -> {3},
    "TargetBeauty" -> {3}
  |>;

  If[hasNormals, inPortSpecs["TargetNormal"] = {3}];
  If[hasDiffGeom,
    inPortSpecs["TargetShapeIndex"] = {1};
    inPortSpecs["TargetCurvedness"] = {1};
  ];

  NetGraph[layers, edges, Sequence @@ Normal[inPortSpecs]]
];

Options[CreateMultiLPEBatchGenerator] = {
  "BatchSize" -> 1024,
  "NumSamples" -> 64,
  "RayBounds" -> {12.0, 25.0},
  "Perturb" -> True,
  "SampleForegroundRatio" -> 0.85
};

CreateMultiLPEBatchGenerator[poolInput_, opts : OptionsPattern[]] := Module[{
  defaultBs = OptionValue["BatchSize"],
  numSamples = OptionValue["NumSamples"],
  rayBounds = OptionValue["RayBounds"],
  perturbQ = OptionValue["Perturb"],
  sampleFgRatio = N @ OptionValue["SampleForegroundRatio"],
  tNear, tFar, pool, hasFg, hasBg, nFgTotal, nBgTotal, hasNormals, hasDiffGeom
},
  {tNear, tFar} = N[rayBounds];
  pool = Which[
    AssociationQ[poolInput] && Lookup[poolInput, "Type", None] === "MultiLPERayPool", poolInput,
    ListQ[poolInput], PrecomputeMultiLPERayPool[poolInput],
    True, poolInput
  ];

  hasFg = pool["Foreground", "Count"] > 0;
  hasBg = pool["Background", "Count"] > 0;
  nFgTotal = pool["Foreground", "Count"];
  nBgTotal = pool["Background", "Count"];
  hasNormals = KeyExistsQ[pool["Foreground"], "Normal"];
  hasDiffGeom = KeyExistsQ[pool["Foreground"], "ShapeIndex"] && KeyExistsQ[pool["Foreground"], "Curvedness"];

  Function[Null,
    Module[{
      args = {##}, batchInfo, bs,
      nFg, nBg, fgIdx, bgIdx,
      origs, dirs, albs, diffDirs, diffInds, specDirs, specInds, beauties, norms, shapes, curvs,
      sampled, res
    },
      batchInfo = If[Length[args] > 0, args[[1]], None];
      bs = Which[
        IntegerQ[batchInfo] && batchInfo > 0, batchInfo,
        AssociationQ[batchInfo], Lookup[batchInfo, "BatchSize", defaultBs],
        True, defaultBs
      ];

      If[hasFg && hasBg,
        nFg = Round[bs * sampleFgRatio];
        nBg = bs - nFg;
        fgIdx = RandomInteger[{1, nFgTotal}, nFg];
        bgIdx = RandomInteger[{1, nBgTotal}, nBg];

        origs = Join[pool["Foreground", "Origins"][[fgIdx]], pool["Background", "Origins"][[bgIdx]]];
        dirs  = Join[pool["Foreground", "Directions"][[fgIdx]], pool["Background", "Directions"][[bgIdx]]];
        albs  = Join[pool["Foreground", "Albedo"][[fgIdx]], pool["Background", "Albedo"][[bgIdx]]];
        diffDirs = Join[pool["Foreground", "DirectDiffuse"][[fgIdx]], pool["Background", "DirectDiffuse"][[bgIdx]]];
        diffInds = Join[pool["Foreground", "IndirectDiffuse"][[fgIdx]], pool["Background", "IndirectDiffuse"][[bgIdx]]];
        specDirs = Join[pool["Foreground", "DirectSpecular"][[fgIdx]], pool["Background", "DirectSpecular"][[bgIdx]]];
        specInds = Join[pool["Foreground", "IndirectSpecular"][[fgIdx]], pool["Background", "IndirectSpecular"][[bgIdx]]];
        beauties = Join[pool["Foreground", "Beauty"][[fgIdx]], pool["Background", "Beauty"][[bgIdx]]];
        norms    = If[hasNormals, Join[pool["Foreground", "Normal"][[fgIdx]], pool["Background", "Normal"][[bgIdx]]], None];
        shapes   = If[hasDiffGeom, Join[pool["Foreground", "ShapeIndex"][[fgIdx]], pool["Background", "ShapeIndex"][[bgIdx]]], None];
        curvs    = If[hasDiffGeom, Join[pool["Foreground", "Curvedness"][[fgIdx]], pool["Background", "Curvedness"][[bgIdx]]], None];,
        
        (* Foreground only *)
        fgIdx = RandomInteger[{1, nFgTotal}, bs];
        origs = pool["Foreground", "Origins"][[fgIdx]];
        dirs  = pool["Foreground", "Directions"][[fgIdx]];
        albs  = pool["Foreground", "Albedo"][[fgIdx]];
        diffDirs = pool["Foreground", "DirectDiffuse"][[fgIdx]];
        diffInds = pool["Foreground", "IndirectDiffuse"][[fgIdx]];
        specDirs = pool["Foreground", "DirectSpecular"][[fgIdx]];
        specInds = pool["Foreground", "IndirectSpecular"][[fgIdx]];
        beauties = pool["Foreground", "Beauty"][[fgIdx]];
        norms    = If[hasNormals, pool["Foreground", "Normal"][[fgIdx]], None];
        shapes   = If[hasDiffGeom, pool["Foreground", "ShapeIndex"][[fgIdx]], None];
        curvs    = If[hasDiffGeom, pool["Foreground", "Curvedness"][[fgIdx]], None];
      ];

      sampled = SampleRaysStratified[origs, dirs, numSamples, {tNear, tFar}, "Perturb" -> perturbQ];

      res = <|
        "SamplePoints" -> NumericArray[sampled["SamplePoints"], "Real32"],
        "Steps" -> NumericArray[sampled["Steps"], "Real32"],
        "TargetAlbedo" -> NumericArray[albs, "Real32"],
        "TargetDirectDiffuse" -> NumericArray[diffDirs, "Real32"],
        "TargetIndirectDiffuse" -> NumericArray[diffInds, "Real32"],
        "TargetDirectSpecular" -> NumericArray[specDirs, "Real32"],
        "TargetIndirectSpecular" -> NumericArray[specInds, "Real32"],
        "TargetBeauty" -> NumericArray[beauties, "Real32"]
      |>;
      If[hasNormals && norms =!= None,
        res["TargetNormal"] = NumericArray[norms, "Real32"];
      ];
      If[hasDiffGeom && shapes =!= None && curvs =!= None,
        res["TargetShapeIndex"] = NumericArray[shapes, "Real32"];
        res["TargetCurvedness"] = NumericArray[curvs, "Real32"];
      ];
      res
    ]
  ]
];

Options[TrainMultiLPENeRFModel] = {
  "MaxTrainingRounds" -> 5000,
  "BatchSize" -> 2048,
  "LearningRate" -> 1*^-3,
  "Method" -> {"ADAM", "GradientClipping" -> 1.0},
  "TargetDevice" -> "GPU",
  "CheckpointDirectory" -> Automatic,
  "CheckpointInterval" -> Quantity[10, "Minutes"],
  "TrainingProgressReporting" -> "Print"
};

TrainMultiLPENeRFModel[net_NetGraph, viewsOrPool_, opts : OptionsPattern[]] := Module[{
  trainNet, gen, trainedNet, maxRounds, bs, lr, method, device, ckptDir, ckptInt, rep,
  inPorts, nSamples
},
  maxRounds = OptionValue["MaxTrainingRounds"];
  bs = OptionValue["BatchSize"];
  lr = OptionValue["LearningRate"];
  method = OptionValue["Method"];
  device = OptionValue["TargetDevice"];
  ckptDir = OptionValue["CheckpointDirectory"];
  ckptInt = OptionValue["CheckpointInterval"];
  rep = OptionValue["TrainingProgressReporting"];

  trainNet = If[KeyExistsQ[Information[net, "OutputPorts"], "Loss"],
    net,
    CreateMultiLPETrainingNet[net]
  ];

  inPorts = Quiet @ Information[trainNet, "InputPorts"];
  nSamples = If[AssociationQ[inPorts] && KeyExistsQ[inPorts, "Steps"],
    Flatten[{inPorts["Steps"]}][[1]],
    32
  ];

  gen = CreateMultiLPEBatchGenerator[viewsOrPool, "BatchSize" -> bs, "NumSamples" -> nSamples];

  trainedNet = NetTrain[
    trainNet,
    gen,
    MaxTrainingRounds -> maxRounds,
    BatchSize -> bs,
    Method -> method,
    TargetDevice -> device,
    TrainingProgressReporting -> rep
  ];

  trainedNet
];

ExtractMultiLPEInferenceNet[trainedNet_NetGraph] := Module[{extracted},
  extracted = Quiet @ Check[NetExtract[trainedNet, "net"], None];
  If[MatchQ[extracted, _NetGraph],
    extracted,
    trainedNet
  ]
];

Options[RenderMultiLPENeRFView] = {
  "FOV" -> 48.0,
  "RayBounds" -> {12.0, 25.0},
  "NumSamples" -> 32,
  "ChunkSize" -> 4096,
  "TargetDevice" -> "GPU",
  "Coefficients" -> <|"DirectDiffuse" -> 1.0, "IndirectDiffuse" -> 1.0, "DirectSpecular" -> 1.0, "IndirectSpecular" -> 1.0|>,
  "Exposure" -> 1.2,
  "Gamma" -> 2.2,
  "Tonemap" -> "sRGB",
  "ImageSize" -> Automatic
};

RenderMultiLPENeRFView[net_NetGraph, matrix_?MatrixQ, {w_Integer, h_Integer}, opts : OptionsPattern[]] := Module[{
  fov = OptionValue["FOV"],
  rayBounds = OptionValue["RayBounds"],
  numSamples = OptionValue["NumSamples"],
  chunkSize = OptionValue["ChunkSize"],
  device = OptionValue["TargetDevice"],
  coeffs = OptionValue["Coefficients"],
  exposure = N @ OptionValue["Exposure"],
  gammaVal = N @ OptionValue["Gamma"],
  tonemapOpt = OptionValue["Tonemap"],
  imgSizeOpt = OptionValue["ImageSize"],
  evalNet, tNear, tFar, rays, nRays, chunks,
  cDiffDir = 1.0, cDiffInd = 1.0, cSpecDir = 1.0, cSpecInd = 1.0,
  allAlb = {}, allDiffDir = {}, allDiffInd = {}, allSpecDir = {}, allSpecInd = {}, allBeauty = {}, allOpacity = {}, allNorm = {}, allShape = {}, allCurv = {},
  imgAlb, imgDiffDir, imgDiffInd, imgSpecDir, imgSpecInd, imgBeauty, imgRelit, imgNorm = None, imgShape = None, imgCurv = None, relitArray,
  toDisplayImage
},
  {tNear, tFar} = N[rayBounds];
  evalNet = ExtractMultiLPEInferenceNet[net];
  Module[{inPorts = Quiet @ Information[evalNet, "InputPorts"]},
    If[AssociationQ[inPorts] && KeyExistsQ[inPorts, "Steps"],
      numSamples = Flatten[{inPorts["Steps"]}][[1]];
    ]
  ];

  If[AssociationQ[coeffs],
    cDiffDir = Lookup[coeffs, "DirectDiffuse", 1.0];
    cDiffInd = Lookup[coeffs, "IndirectDiffuse", 1.0];
    cSpecDir = Lookup[coeffs, "DirectSpecular", 1.0];
    cSpecInd = Lookup[coeffs, "IndirectSpecular", 1.0];
  ];

  rays = GenerateCameraRays[matrix, {w, h}, "FOV" -> fov];
  nRays = Length[rays["Origins"]];
  chunks = Partition[Range[nRays], UpTo[chunkSize]];

  Do[
    Module[{
      origs = rays["Origins"][[c]],
      dirs  = rays["Directions"][[c]],
      sampled, evalInput, out
    },
      sampled = SampleRaysStratified[origs, dirs, numSamples, {tNear, tFar}, "Perturb" -> False];
      evalInput = <|
        "SamplePoints" -> NumericArray[sampled["SamplePoints"], "Real32"],
        "Steps" -> NumericArray[sampled["Steps"], "Real32"]
      |>;
      out = evalNet[evalInput, TargetDevice -> device];

      allAlb     = Join[allAlb, Normal[out["Albedo"]]];
      allDiffDir = Join[allDiffDir, Normal[out["DirectDiffuse"]]];
      allDiffInd = Join[allDiffInd, Normal[out["IndirectDiffuse"]]];
      allSpecDir = Join[allSpecDir, Normal[out["DirectSpecular"]]];
      allSpecInd = Join[allSpecInd, Normal[out["IndirectSpecular"]]];
      allBeauty  = Join[allBeauty, Normal[out["Beauty"]]];
      If[KeyExistsQ[out, "Opacity"], allOpacity = Join[allOpacity, Flatten[Normal[out["Opacity"]]]]];
      If[KeyExistsQ[out, "Normal"], allNorm = Join[allNorm, Normal[out["Normal"]]]];
      If[KeyExistsQ[out, "ShapeIndex"], allShape = Join[allShape, Flatten[Normal[out["ShapeIndex"]]]]];
      If[KeyExistsQ[out, "Curvedness"], allCurv = Join[allCurv, Flatten[Normal[out["Curvedness"]]]]];
    ],
    {c, chunks}
  ];

  allAlb     = Developer`ToPackedArray @ N @ ArrayReshape[allAlb, {h, w, 3}];
  allDiffDir = Developer`ToPackedArray @ N @ ArrayReshape[allDiffDir, {h, w, 3}];
  allDiffInd = Developer`ToPackedArray @ N @ ArrayReshape[allDiffInd, {h, w, 3}];
  allSpecDir = Developer`ToPackedArray @ N @ ArrayReshape[allSpecDir, {h, w, 3}];
  allSpecInd = Developer`ToPackedArray @ N @ ArrayReshape[allSpecInd, {h, w, 3}];
  allBeauty  = Developer`ToPackedArray @ N @ ArrayReshape[allBeauty, {h, w, 3}];

  If[allNorm =!= {},
    Module[{normArr = Developer`ToPackedArray @ N @ ArrayReshape[allNorm, {h, w, 3}]},
      imgNorm = If[IntegerQ[imgSizeOpt] && imgSizeOpt > 0,
        Image[Clip[0.5 * (normArr + 1.0), {0.0, 1.0}], "Real32", ColorSpace -> "RGB", ImageSize -> imgSizeOpt],
        Image[Clip[0.5 * (normArr + 1.0), {0.0, 1.0}], "Real32", ColorSpace -> "RGB"]
      ];
    ];
  ];

  If[allShape =!= {},
    Module[{shapeArr = Developer`ToPackedArray @ N @ ArrayReshape[allShape, {h, w}]},
      imgShape = Colorize[Image[Clip[0.5 * (shapeArr + 1.0), {0.0, 1.0}], "Real32"], ColorFunction -> "Rainbow"];
    ];
  ];

  If[allCurv =!= {},
    Module[{curvArr = Developer`ToPackedArray @ N @ ArrayReshape[allCurv, {h, w}]},
      imgCurv = Image[Clip[Log[1.0 + Ramp[curvArr]] / Log[1.0 + 10.0], {0.0, 1.0}], "Real32"];
    ];
  ];

  relitArray = allAlb * (cDiffDir * allDiffDir + cDiffInd * allDiffInd) + cSpecDir * allSpecDir + cSpecInd * allSpecInd;

  toDisplayImage = Function[{arr, isIntrinsic},
    Module[{clipped, scaled, gammaCorrected},
      clipped = Clip[arr, {0.0, Infinity}];
      scaled = If[TrueQ[isIntrinsic], clipped, clipped * (2.0^exposure)];
      gammaCorrected = Switch[tonemapOpt,
        "sRGB" | "Gamma",
          Clip[scaled^(1.0 / gammaVal), {0.0, 1.0}],
        "Reinhard",
          Clip[(scaled / (1.0 + scaled))^(1.0 / gammaVal), {0.0, 1.0}],
        "Linear" | "None",
          Clip[scaled, {0.0, 1.0}],
        _,
          Clip[scaled^(1.0 / gammaVal), {0.0, 1.0}]
      ];
      If[IntegerQ[imgSizeOpt] && imgSizeOpt > 0,
        Image[gammaCorrected, "Real32", ColorSpace -> "RGB", ImageSize -> imgSizeOpt],
        Image[gammaCorrected, "Real32", ColorSpace -> "RGB"]
      ]
    ]
  ];

  imgAlb     = toDisplayImage[allAlb, True];
  imgDiffDir = toDisplayImage[allDiffDir, False];
  imgDiffInd = toDisplayImage[allDiffInd, False];
  imgSpecDir = toDisplayImage[allSpecDir, False];
  imgSpecInd = toDisplayImage[allSpecInd, False];
  imgBeauty  = toDisplayImage[allBeauty, False];
  imgRelit   = toDisplayImage[relitArray, False];

  <|
    "Relit" -> imgRelit,
    "Beauty" -> imgBeauty,
    "Albedo" -> imgAlb,
    "DirectDiffuse" -> imgDiffDir,
    "IndirectDiffuse" -> imgDiffInd,
    "DirectSpecular" -> imgSpecDir,
    "IndirectSpecular" -> imgSpecInd,
    "Opacity" -> If[allOpacity =!= {}, Image[ArrayReshape[allOpacity, {h, w}], "Real32"], None],
    "Normal" -> imgNorm,
    "ShapeIndex" -> imgShape,
    "Curvedness" -> imgCurv,
    "RawArrays" -> <|
      "Albedo" -> allAlb,
      "DirectDiffuse" -> allDiffDir,
      "IndirectDiffuse" -> allDiffInd,
      "DirectSpecular" -> allSpecDir,
      "IndirectSpecular" -> allSpecInd,
      "Beauty" -> allBeauty
    |>
  |>
];

Options[RenderMultiLPETurntable] = {
  "NumFrames" -> 36,
  "Resolution" -> {256, 256},
  "Distance" -> 18.5,
  "Elevation" -> 25.0,
  "Target" -> {1.53, 1.20, 1.16},
  "FOV" -> 48.0,
  "RayBounds" -> {12.0, 25.0},
  "Coefficients" -> <|"DirectDiffuse" -> 1.0, "IndirectDiffuse" -> 1.0, "DirectSpecular" -> 1.0, "IndirectSpecular" -> 1.0|>,
  "Exposure" -> 1.2,
  "Gamma" -> 2.2,
  "ExportFile" -> None,
  "TargetDevice" -> "GPU"
};

RenderMultiLPETurntable[net_NetGraph, opts : OptionsPattern[]] := Module[{
  nFrames = OptionValue["NumFrames"],
  res = OptionValue["Resolution"],
  dist = N @ OptionValue["Distance"],
  elev = N @ OptionValue["Elevation"] * Degree,
  target = N @ OptionValue["Target"],
  fov = N @ OptionValue["FOV"],
  rayBounds = OptionValue["RayBounds"],
  coeffs = OptionValue["Coefficients"],
  exposure = OptionValue["Exposure"],
  gammaVal = OptionValue["Gamma"],
  exportFile = OptionValue["ExportFile"],
  device = OptionValue["TargetDevice"],
  angles, frames = {}
},
  angles = Table[phi, {phi, 0.0, 2.0 * Pi * (1.0 - 1.0 / nFrames), 2.0 * Pi / nFrames}];

  Do[
    Module[{eye, n, u, v, w2c, rend},
      eye = target + dist * {Cos[elev] * Cos[phi], Sin[elev], Cos[elev] * Sin[phi]};
      n = Normalize[eye - target];
      u = Normalize[Cross[{0., 1., 0.}, n]];
      v = Normalize[Cross[n, u]];
      w2c = Developer`ToPackedArray @ N @ {
        {u[[1]], v[[1]], -n[[1]], 0.},
        {u[[2]], v[[2]], -n[[2]], 0.},
        {u[[3]], v[[3]], -n[[3]], 0.},
        {-u . eye, -v . eye, n . eye, 1.}
      };

      rend = RenderMultiLPENeRFView[net, w2c, res,
        "FOV" -> fov,
        "RayBounds" -> rayBounds,
        "Coefficients" -> coeffs,
        "Exposure" -> exposure,
        "Gamma" -> gammaVal,
        "TargetDevice" -> device
      ];
      AppendTo[frames, rend["Relit"]];
    ],
    {phi, angles}
  ];

  If[StringQ[exportFile] && exportFile =!= "",
    Export[exportFile, frames, "DisplayDurations" -> 1.0 / 24.0];
  ];

  frames
];

Options[InteractiveMultiLPENeRFViewer] = {
  "Resolution" -> {320, 320},
  "DefaultImageSize" -> 380,
  "TargetDevice" -> "GPU"
};

InteractiveMultiLPENeRFViewer[net_NetGraph, viewsOrMatrix_, opts : OptionsPattern[]] := Module[{
  matrices, fovs, boundsList, res = OptionValue["Resolution"], defaultSize = OptionValue["DefaultImageSize"],
  device = OptionValue["TargetDevice"],
  viewCache = <||>, getCachedView
},
  If[ListQ[viewsOrMatrix] && AssociationQ[First[viewsOrMatrix, None]],
    matrices   = Map[Lookup[#, "CameraMatrix", IdentityMatrix[4]] &, viewsOrMatrix];
    fovs       = Map[Lookup[#, "FOV", 48.0] &, viewsOrMatrix];
    boundsList = Map[Lookup[#, "SuggestedRayBounds", {12.0, 25.0}] &, viewsOrMatrix];,
    matrices   = If[MatrixQ[viewsOrMatrix], {viewsOrMatrix}, {IdentityMatrix[4]}];
    fovs       = {48.0};
    boundsList = {{12.0, 25.0}};
  ];

  getCachedView[idx_Integer] := If[KeyExistsQ[viewCache, idx],
    viewCache[idx],
    viewCache[idx] = RenderMultiLPENeRFView[net, matrices[[idx]], res,
      "FOV" -> fovs[[idx]],
      "RayBounds" -> boundsList[[idx]],
      "TargetDevice" -> device
    ]
  ];

  Manipulate[
    Module[{baseRend, raw, relitArr, toImg, imgRelit, imgAlb, imgDiffDir, imgDiffInd, imgSpecDir, imgSpecInd, imgBeauty, imgNorm, imgShape, imgCurv, imgW},
      imgW = Round[imgSize];
      baseRend = getCachedView[viewIdx];
      raw = baseRend["RawArrays"];

      toImg[arr_, isIntrinsic_] := Image[
        If[TrueQ[isIntrinsic],
          Clip[arr, {0.0, 1.0}],
          Clip[(Clip[arr, {0.0, Infinity}] * (2.0^exposure))^(1.0 / gammaVal), {0.0, 1.0}]
        ],
        "Real32", ColorSpace -> "RGB"
      ];

      relitArr   = raw["Albedo"] * (cDiffDir * raw["DirectDiffuse"] + cDiffInd * raw["IndirectDiffuse"]) +
                   cSpecDir * raw["DirectSpecular"] + cSpecInd * raw["IndirectSpecular"];
      imgRelit   = toImg[relitArr, False];
      imgBeauty  = toImg[raw["Beauty"], False];
      imgAlb     = toImg[raw["Albedo"], True];
      imgDiffDir = toImg[raw["DirectDiffuse"], False];
      imgDiffInd = toImg[raw["IndirectDiffuse"], False];
      imgSpecDir = toImg[raw["DirectSpecular"], False];
      imgSpecInd = toImg[raw["IndirectSpecular"], False];
      imgNorm    = Replace[baseRend["Normal"], None :> imgAlb];
      imgShape   = Replace[baseRend["ShapeIndex"], None :> imgNorm];
      imgCurv    = Replace[baseRend["Curvedness"], None :> imgNorm];

      Switch[layoutMode,
        "Hero (Composite + Gallery)",
          Column[{
            Panel[
              Column[{
                Style["Relit Output (\!\(\*OverscriptBox[\(I\), \(~\)]\))", Bold, 15],
                Show[imgRelit, ImageSize -> (imgW + 80)]
              }, Center],
              Appearance -> "Frameless"
            ],
            Spacer[10],
            Style["Constituent Decomposed Passes", Bold, 12],
            Row[{
              Labeled[Show[imgAlb, ImageSize -> Round[imgW * 0.44]], Style["Albedo (\!\(\*OverscriptBox[\(\[Rho]\), \(~\)]\)\_d)", 11], Top],
              Spacer[12],
              Labeled[Show[imgDiffDir, ImageSize -> Round[imgW * 0.44]], Style["Direct Diffuse (\!\(\*OverscriptBox[\(L\), \(^\)]\)\_d)", 11], Top],
              Spacer[12],
              Labeled[Show[imgDiffInd, ImageSize -> Round[imgW * 0.44]], Style["Indirect Diffuse (\!\(\*OverscriptBox[\(L\), \(^\)]\)\_i)", 11], Top],
              Spacer[12],
              Labeled[Show[imgSpecDir, ImageSize -> Round[imgW * 0.44]], Style["Direct Specular (\!\(\*OverscriptBox[\(S\), \(^\)]\)\_d)", 11], Top]
            }]
          }, Center],

        "Perceptual & Differential Geometry (Tilde Class)",
          Grid[{
            {
              Labeled[Show[imgRelit, ImageSize -> imgW], Style["Relit Composite (\!\(\*OverscriptBox[\(I\), \(~\)]\))", Bold, 12], Top],
              Labeled[Show[imgAlb, ImageSize -> imgW], Style["Intrinsic Albedo (\!\(\*OverscriptBox[\(\[Rho]\), \(~\)]\)\_d)", Bold, 12], Top]
            },
            {
              Labeled[Show[imgNorm, ImageSize -> imgW], Style["Surface Normal (\!\(\*OverscriptBox[\(N\), \(~\)]\)\_n)", Bold, 12], Top],
              Labeled[Show[imgShape, ImageSize -> imgW], Style["Shape Index (\!\(\*OverscriptBox[\(S\), \(~\)]\))", Bold, 12], Top]
            }
          }, Spacings -> {2, 2}],

        "2x2 Grid",
          Grid[{
            {
              Labeled[Show[imgRelit, ImageSize -> imgW], Style["Relit Composite (\!\(\*OverscriptBox[\(I\), \(~\)]\))", Bold, 12], Top],
              Labeled[Show[imgAlb, ImageSize -> imgW], Style["Intrinsic Albedo (\!\(\*OverscriptBox[\(\[Rho]\), \(~\)]\)\_d)", Bold, 12], Top]
            },
            {
              Labeled[Show[imgDiffDir, ImageSize -> imgW], Style["Direct Diffuse (\!\(\*OverscriptBox[\(L\), \(^\)]\)\_d)", Bold, 12], Top],
              Labeled[Show[imgSpecDir, ImageSize -> imgW], Style["Direct Specular (\!\(\*OverscriptBox[\(S\), \(^\)]\)\_d)", Bold, 12], Top]
            }
          }, Spacings -> {2, 2}],

        "Single Pass Focused",
          Module[{selImg = Switch[activePass,
            "Relit (Composite)", imgRelit,
            "Beauty (Baseline)", imgBeauty,
            "Albedo", imgAlb,
            "Surface Normal (N_n)", imgNorm,
            "Shape Index (S)", imgShape,
            "Curvedness (C)", imgCurv,
            "Direct Diffuse", imgDiffDir,
            "Indirect Diffuse", imgDiffInd,
            "Direct Specular", imgSpecDir,
            "Indirect Specular", imgSpecInd,
            _, imgRelit
          ]},
            Column[{
              Style[activePass, Bold, 15],
              Show[selImg, ImageSize -> (imgW + 100)]
            }, Center]
          ]
      ]
    ],
    Style["Camera & Display Settings", Bold],
    {{viewIdx, 1, "Camera Viewpoint"}, 1, Length[matrices], 1},
    {{layoutMode, "Hero (Composite + Gallery)", "Layout"}, {"Hero (Composite + Gallery)", "Perceptual & Differential Geometry (Tilde Class)", "2x2 Grid", "Single Pass Focused"}},
    {{activePass, "Relit (Composite)", "Focused Pass"}, {"Relit (Composite)", "Beauty (Baseline)", "Albedo", "Surface Normal (N_n)", "Shape Index (S)", "Curvedness (C)", "Direct Diffuse", "Indirect Diffuse", "Direct Specular", "Indirect Specular"}},
    {{imgSize, defaultSize, "Display Size (px)"}, 200, 650, 25},
    Delimiter,
    Style["Photometric Tone Mapping & Exposure", Bold],
    {{exposure, 1.2, "Exposure Boost (EV)"}, -1.0, 3.5, 0.1},
    {{gammaVal, 2.2, "Display Gamma (sRGB)"}, 1.0, 2.6, 0.1},
    Delimiter,
    Style["Radiometric Modulation Coefficients (LPEs)", Bold],
    {{cDiffDir, 1.0, "Direct Diffuse (Key/Fill Light)"}, 0.0, 3.0, 0.05},
    {{cDiffInd, 1.0, "Indirect Diffuse (Color Bleed/Bounce)"}, 0.0, 3.0, 0.05},
    {{cSpecDir, 1.0, "Direct Specular (Primary Glints)"}, 0.0, 4.0, 0.05},
    {{cSpecInd, 1.0, "Indirect Specular (Environment Reflection)"}, 0.0, 4.0, 0.05},
    ControlPlacement -> Left,
    SynchronousUpdating -> False
  ]
];

End[];
EndPackage[];
