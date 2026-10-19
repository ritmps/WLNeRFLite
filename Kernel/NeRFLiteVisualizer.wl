(* ::Package:: *)

BeginPackage["WLNeRFLite`Visualizer`", {
  "NeuralNetworks`",
  "WLNeRFLite`Data`",
  "WLNeRFLite`Net`"
}];

RenderNeRFLiteView::usage = "RenderNeRFLiteView[net, cameraMatrix, {w, h}, opts] renders a 2D image from a trained lightweight NeRF.";
RenderNeRFLiteTurntable::usage = "RenderNeRFLiteTurntable[net, opts] renders a 360-degree orbiting turntable animation.";
SampleNeRFLiteDensityVolume::usage = "SampleNeRFLiteDensityVolume[densityNet, {xRange, yRange, zRange}, res] samples a 3D density grid for volume visualization.";
VisualizeNeRFCameraRig::usage = "VisualizeNeRFCameraRig[views, opts] creates an interactive 3D visualization of camera frustums, positions, and bounds in world space.";

Begin["`Private`"];

Options[RenderNeRFLiteView] = {
  "FOV" -> 48.0,
  "RayBounds" -> {3.0, 10.0},
  "NumSamples" -> 32,
  "ChunkSize" -> 4096,
  "TargetDevice" -> "GPU",
  "Tonemap" -> Automatic,
  "Return" -> "Image"
};

RenderNeRFLiteView[net_NetGraph, matrix_?MatrixQ, {w_Integer, h_Integer}, opts : OptionsPattern[]] := Module[{
  fov = OptionValue["FOV"],
  rayBounds = OptionValue["RayBounds"],
  numSamples = OptionValue["NumSamples"],
  chunkSize = OptionValue["ChunkSize"],
  device = OptionValue["TargetDevice"],
  returnType = OptionValue["Return"],
  tonemapOpt = OptionValue["Tonemap"],
  tNear, tFar, rays, nRays, chunks, allRGB = {}, allDepth = {}, allOpacity = {},
  inPorts, hasDepth, evalNet
},
  {tNear, tFar} = N[rayBounds];
  inPorts = Quiet @ Information[net, "InputPorts"];
  If[AssociationQ[inPorts] && KeyExistsQ[inPorts, "Steps"], numSamples = Flatten[{inPorts["Steps"]}][[1]]];
  hasDepth = AssociationQ[inPorts] && KeyExistsQ[inPorts, "SampleT"];

  rays = GenerateCameraRays[matrix, {w, h}, "FOV" -> fov];
  nRays = Length[rays["Origins"]];

  evalNet = net;
  chunks = Partition[Range[nRays], UpTo[chunkSize]];

  Do[
    Module[{
      origs = rays["Origins"][[c]],
      dirs = rays["Directions"][[c]],
      sampled, evalInput, out
    },
      sampled = SampleRaysStratified[origs, dirs, numSamples, {tNear, tFar}, "Perturb" -> False];
      evalInput = <|
        "SamplePoints" -> NumericArray[sampled["SamplePoints"], "Real32"],
        "Steps" -> NumericArray[sampled["Steps"], "Real32"]
      |>;
      If[hasDepth, evalInput["SampleT"] = NumericArray[sampled["SampleT"], "Real32"]];

      out = evalNet[evalInput, TargetDevice -> device];

      allRGB = Join[allRGB, Normal[out["RGB"]]];
      If[KeyExistsQ[out, "Opacity"], allOpacity = Join[allOpacity, Flatten[Normal[out["Opacity"]]]]];
      If[KeyExistsQ[out, "Depth"], allDepth = Join[allDepth, Flatten[Normal[out["Depth"]]]]];
    ],
    {c, chunks}
  ];

  Module[{
    rawRgb = Developer`ToPackedArray[N[ArrayReshape[allRGB, {h, w, 3}]]],
    rgbArray, depthArray, opacityArray
  },
    rgbArray = Switch[tonemapOpt,
      "Reinhard",
        Module[{pos = Clip[rawRgb, {0.0, Infinity}]}, pos / (1.0 + pos)],
      "Clip",
        Clip[rawRgb, {0.0, 1.0}],
      "None",
        rawRgb,
      _,
        If[Max[rawRgb] > 1.05,
          Module[{pos = Clip[rawRgb, {0.0, Infinity}]}, pos / (1.0 + pos)],
          Clip[rawRgb, {0.0, 1.0}]
        ]
    ];

    Switch[returnType,
      "RGB",
        rgbArray,
      "Depth",
        ArrayReshape[allDepth, {h, w}],
      "Opacity",
        ArrayReshape[allOpacity, {h, w}],
      "All",
        <|
          "Image" -> Image[rgbArray, "Real32", ColorSpace -> "RGB"],
          "RGB" -> rgbArray,
          "Opacity" -> If[allOpacity =!= {}, ArrayReshape[allOpacity, {h, w}], None],
          "Depth" -> If[allDepth =!= {}, ArrayReshape[allDepth, {h, w}], None]
        |>,
      _,
        Image[rgbArray, "Real32", ColorSpace -> "RGB"]
    ]
  ]
];

makeTurntableMatrix[center_List, radius_?NumericQ, elevation_?NumericQ, azimuth_?NumericQ] := Module[{
  pos, forward, right, up, c2w
},
  pos = center + radius * {
    Cos[elevation] * Sin[azimuth],
    Sin[elevation],
    Cos[elevation] * Cos[azimuth]
  };
  forward = Normalize[center - pos];
  right = Normalize[Cross[forward, {0., 1., 0.}]];
  up = Normalize[Cross[right, forward]];

  c2w = {
    {right[[1]], right[[2]], right[[3]], 0.},
    {up[[1]], up[[2]], up[[3]], 0.},
    {forward[[1]], forward[[2]], forward[[3]], 0.},
    {pos[[1]], pos[[2]], pos[[3]], 1.}
  };
  Inverse[c2w]
];

Options[RenderNeRFLiteTurntable] = Join[
  {
    "NumFrames" -> 36,
    "Radius" -> 5.0,
    "Elevation" -> 25.0 * Degree,
    "Target" -> {0., 0.5, 0.},
    "Resolution" -> {256, 256},
    "FOV" -> 48.0,
    "ChunkSize" -> 4096,
    "TargetDevice" -> "GPU",
    "ExportFile" -> None
  },
  Options[RenderNeRFLiteView]
];

RenderNeRFLiteTurntable[net_NetGraph, opts : OptionsPattern[]] := Module[{
  nFrames = OptionValue["NumFrames"],
  radius = N @ OptionValue["Radius"],
  elevation = N @ OptionValue["Elevation"],
  target = N @ OptionValue["Target"],
  res = OptionValue["Resolution"],
  fov = N @ OptionValue["FOV"],
  chunkSize = OptionValue["ChunkSize"],
  device = OptionValue["TargetDevice"],
  exportFile = OptionValue["ExportFile"],
  viewOpts = FilterRules[{opts}, Options[RenderNeRFLiteView]],
  angles, frames = {}
},
  angles = Table[2.0 * Pi * (i / nFrames), {i, 0, nFrames - 1}];

  Do[
    Module[{mat, img},
      mat = makeTurntableMatrix[target, radius, elevation, phi];
      img = RenderNeRFLiteView[
        net, mat, res,
        "FOV" -> fov,
        "ChunkSize" -> chunkSize,
        "TargetDevice" -> device,
        Sequence @@ viewOpts
      ];
      AppendTo[frames, img];
    ],
    {phi, angles}
  ];

  If[StringQ[exportFile] && exportFile =!= "",
    Export[exportFile, frames, "AnimationRepetitions" -> Infinity, "DisplayDurations" -> 0.08];
  ];

  frames
];

Options[SampleNeRFLiteDensityVolume] = {
  "TargetDevice" -> "GPU",
  "BatchSize" -> 16384
};

SampleNeRFLiteDensityVolume[densityNet_NetChain, {xRange_, yRange_, zRange_}, res_Integer: 32, opts : OptionsPattern[]] := Module[{
  xs, ys, zs, grid, pts, nPts, chunks, allSigma = {}, device = OptionValue["TargetDevice"],
  bs = OptionValue["BatchSize"]
},
  xs = Subdivide[N[xRange[[1]]], N[xRange[[2]]], res - 1];
  ys = Subdivide[N[yRange[[1]]], N[yRange[[2]]], res - 1];
  zs = Subdivide[N[zRange[[1]]], N[zRange[[2]]], res - 1];

  grid = Table[{x, y, z}, {z, zs}, {y, ys}, {x, xs}];
  pts = Developer`ToPackedArray @ Flatten[grid, 2];
  nPts = Length[pts];

  chunks = Partition[Range[nPts], UpTo[bs]];
  Do[
    Module[{batchPts = pts[[c]], out},
      out = densityNet[NumericArray[batchPts, "Real32"], TargetDevice -> device];
      allSigma = Join[allSigma, Flatten[Normal[out]]];
    ],
    {c, chunks}
  ];

  ArrayReshape[allSigma, {res, res, res}]
];

Options[VisualizeNeRFCameraRig] = {
  "FrustumScale" -> 0.35,
  "PointSize" -> 0.02,
  "ShowAxes" -> True,
  "Target" -> {0., 0., 0.}
};

VisualizeNeRFCameraRig[views_List, opts : OptionsPattern[]] := Module[{
  scale = OptionValue["FrustumScale"],
  ptSize = OptionValue["PointSize"],
  showAxes = TrueQ @ OptionValue["ShowAxes"],
  target = OptionValue["Target"],
  nViews = Length[views],
  primitives = {}
},
  If[showAxes,
    AppendTo[primitives, {
      Thick,
      Red,   Arrow[{{0, 0, 0}, {1, 0, 0}}],
      Green, Arrow[{{0, 0, 0}, {0, 1, 0}}],
      Blue,  Arrow[{{0, 0, 0}, {0, 0, 1}}]
    }];
  ];

  If[ListQ[target] && Length[target] == 3,
    AppendTo[primitives, {PointSize[0.025], Magenta, Point[target]}];
  ];

  Do[
    Module[{
      v = views[[i]], camMat, fov, dim, raysCorner, pos, c1, c2, c3, c4, col
    },
      camMat = Lookup[v, "CameraMatrix", Lookup[v, "NeRFMatrix", None]];
      If[!MatrixQ[camMat, NumericQ], Continue[]];
      fov = Lookup[v, "FOV", 48.0];
      dim = Lookup[v, "Dimensions", {512, 512}];

      raysCorner = GenerateCameraRays[camMat, dim, "FOV" -> fov, "PixelCoords" -> {
        {0, 0}, {dim[[1]] - 1, 0}, {dim[[1]] - 1, dim[[2]] - 1}, {0, dim[[2]] - 1}
      }];

      pos = raysCorner["CameraPosition"];
      c1 = pos + scale * raysCorner["Directions"][[1]];
      c2 = pos + scale * raysCorner["Directions"][[2]];
      c3 = pos + scale * raysCorner["Directions"][[3]];
      c4 = pos + scale * raysCorner["Directions"][[4]];

      col = ColorData["Rainbow"][N[(i - 1) / Max[1, nViews - 1]]];

      AppendTo[primitives, {
        col,
        PointSize[ptSize], Point[pos],
        Opacity[0.85],
        Line[{pos, c1}], Line[{pos, c2}], Line[{pos, c3}], Line[{pos, c4}],
        Line[{c1, c2, c3, c4, c1}]
      }];
    ],
    {i, nViews}
  ];

  Graphics3D[
    primitives,
    Boxed -> True,
    Axes -> True,
    AxesLabel -> {"X", "Y", "Z"},
    Background -> GrayLevel[0.15],
    PlotRange -> All,
    ImageSize -> Large
  ]
];

End[];
EndPackage[];
