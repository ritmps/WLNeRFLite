(* ::Package:: *)

BeginPackage["WLNeRFLite`Net`", {"NeuralNetworks`"}];

CreateHarmonicEncoder::usage = "CreateHarmonicEncoder[numBands:6] creates a differentiable positional encoding net mapping {3} to {3 * (1 + 2*numBands)}.";
CreateNeRFLiteNet::usage = "CreateNeRFLiteNet[opts] constructs a super lightweight NeRF NetGraph with embedded harmonic encoder and differentiable volumetric quadrature.";
CreateNeRFLiteTrainingNet::usage = "CreateNeRFLiteTrainingNet[fieldNet, opts] wraps a NeRF prediction NetGraph with photometric loss.";
CreateLogL1LossNet::usage = "CreateLogL1LossNet[] constructs a tonemapped L1 loss net for HDR radiance using log(1 + x).";
CreateLogL2LossNet::usage = "CreateLogL2LossNet[] constructs a tonemapped L2 loss net for HDR radiance using log(1 + x).";
CreateTonemappedL1LossNet::usage = "CreateTonemappedL1LossNet[] constructs a Reinhard tonemapped L1 loss net.";
ExtractNeRFLiteDensityNet::usage = "ExtractNeRFLiteDensityNet[net] extracts the point density evaluation subnet mapping {3} -> {1}.";
ExtractNeRFLiteColorNet::usage = "ExtractNeRFLiteColorNet[net] extracts the point RGB evaluation subnet mapping {3} -> {3}.";

Begin["`Private`"];

CreateHarmonicEncoder[numBands_Integer: 6] := Module[{w, bands = Max[1, numBands]},
  w = NumericArray[
    N[Flatten[Table[2.0^k * Pi * IdentityMatrix[3], {k, 0, bands - 1}], 1]],
    "Real32"
  ];
  NetGraph[
    <|
      "proj" -> LinearLayer[
        3 * bands,
        "Input" -> 3,
        "Weights" -> w,
        "Biases" -> None,
        "LearningRateMultipliers" -> {"Weights" -> 0.0}
      ],
      "sin" -> ElementwiseLayer[Sin],
      "cos" -> ElementwiseLayer[Cos],
      "concat" -> CatenateLayer[]
    |>,
    {
      NetPort["Input"] -> "proj",
      "proj" -> "sin",
      "proj" -> "cos",
      {NetPort["Input"], "sin", "cos"} -> "concat" -> NetPort["Output"]
    }
  ]
];

CreateFieldBackbone[inputDim_Integer, hiddenDim_Integer, numLayers_Integer, withSkip_ : False] := Module[{
  skipLayer, chain1Layers, chain2Layers, i, chain1, chain2
},
  If[!TrueQ[withSkip] || numLayers < 4,
    chain1Layers = {LinearLayer[hiddenDim, "Input" -> inputDim], Ramp};
    Do[
      AppendTo[chain1Layers, LinearLayer[hiddenDim]];
      AppendTo[chain1Layers, Ramp];,
      {i, 1, numLayers - 1}
    ];
    Return[NetChain[chain1Layers]]
  ];

  skipLayer = Floor[numLayers / 2];

  chain1Layers = {LinearLayer[hiddenDim, "Input" -> inputDim], Ramp};
  Do[
    AppendTo[chain1Layers, LinearLayer[hiddenDim]];
    AppendTo[chain1Layers, Ramp];,
    {i, 1, skipLayer - 1}
  ];
  chain1 = NetChain[chain1Layers];

  chain2Layers = {LinearLayer[hiddenDim, "Input" -> (hiddenDim + inputDim)], Ramp};
  Do[
    AppendTo[chain2Layers, LinearLayer[hiddenDim]];
    AppendTo[chain2Layers, Ramp];,
    {i, 1, numLayers - skipLayer - 1}
  ];
  chain2 = NetChain[chain2Layers];

  NetGraph[
    <|
      "chain1" -> chain1,
      "concat" -> CatenateLayer[],
      "chain2" -> chain2
    |>,
    {
      NetPort["Input"] -> "chain1",
      {"chain1", NetPort["Input"]} -> "concat" -> "chain2" -> NetPort["Output"]
    }
  ]
];

Options[CreateNeRFLiteNet] = {
  "NumSamples" -> 32,
  "HarmonicBands" -> 6,
  "HiddenDim" -> 64,
  "NumLayers" -> 2,
  "SkipConnection" -> Automatic,
  "ColorActivation" -> LogisticSigmoid,
  "BackgroundColor" -> None, (* None, Automatic (RenderMan studio grey {0.388, 0.337, 0.298}), or custom {r, g, b} *)
  "IncludeDepth" -> False
};

CreateNeRFLiteNet[opts : OptionsPattern[]] := Module[{
  numSamples = OptionValue["NumSamples"],
  bands = OptionValue["HarmonicBands"],
  hiddenDim = OptionValue["HiddenDim"],
  numLayers = OptionValue["NumLayers"],
  skipOpt = OptionValue["SkipConnection"],
  colorActOpt = OptionValue["ColorActivation"],
  bgColorOpt = OptionValue["BackgroundColor"],
  withDepth = TrueQ @ OptionValue["IncludeDepth"],
  withSkip, effectiveColorAct, effectiveBgColor,
  encodedDim, trilStrict, layers = <||>, edges = {}, backbone
},
  If[!IntegerQ[numSamples] || numSamples <= 0, numSamples = 32];
  If[!IntegerQ[bands] || bands <= 0, bands = 6];
  If[!IntegerQ[hiddenDim] || hiddenDim <= 0, hiddenDim = 64];
  If[!IntegerQ[numLayers] || numLayers <= 0, numLayers = 2];

  effectiveBgColor = Switch[bgColorOpt,
    _List /; VectorQ[bgColorOpt, NumericQ] && Length[bgColorOpt] == 3, N[bgColorOpt],
    _List /; Length[bgColorOpt] >= 3, N[bgColorOpt[[1 ;; 3]]],
    _, None
  ];

  withSkip = Which[
    BooleanQ[skipOpt], skipOpt,
    skipOpt === Automatic, numLayers >= 4,
    True, False
  ];

  encodedDim = 3 * (1 + 2 * bands);

  (* 1. Strict Lower Triangular Matrix for Cumulative Sum:
        tau_cum[[i]] = Sum[tau[[j]], {j, 1, i - 1}]
  *)
  trilStrict = Table[If[j < i, 1.0, 0.0], {i, 1, numSamples}, {j, 1, numSamples}];

  (* 2. Embedded GPU Harmonic Encoder *)
  layers["encoder"] = NetMapOperator[CreateHarmonicEncoder[bands]];

  (* 3. Field Backbone MLP (supports midpoint skip connections) *)
  backbone = CreateFieldBackbone[encodedDim, hiddenDim, numLayers, withSkip];
  layers["backbone"] = NetMapOperator[backbone];

  (* 4. Density Head: Feature -> Linear -> Softplus -> Density sigma >= 0 *)
  layers["density_linear"] = NetMapOperator[LinearLayer[1]];
  layers["density_reshape"] = ReshapeLayer[{numSamples}];
  layers["density_softplus"] = ElementwiseLayer["SoftPlus"];

  (* 5. Color Head *)
  layers["color_linear"] = NetMapOperator[LinearLayer[3]];
  effectiveColorAct = Switch[colorActOpt,
    "SoftPlus", ElementwiseLayer["SoftPlus"],
    "Sigmoid" | LogisticSigmoid, ElementwiseLayer[LogisticSigmoid],
    "None" | None, None,
    _Function, ElementwiseLayer[colorActOpt],
    _, ElementwiseLayer[LogisticSigmoid]
  ];

  If[effectiveColorAct =!= None,
    layers["color_act"] = effectiveColorAct;
  ];

  (* 6. Volumetric Quadrature Integration *)
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

  (* Accumulated RGB *)
  layers["rgb_weighted"] = ThreadingLayer[Times];
  layers["rgb_accum"] = AggregationLayer[Total, 1];

  (* Accumulated Opacity *)
  layers["opacity_accum"] = AggregationLayer[Total, 1];

  edges = {
    NetPort["SamplePoints"] -> "encoder" -> "backbone",
    "backbone" -> "density_linear" -> "density_reshape" -> "density_softplus",
    "backbone" -> "color_linear",

    {"density_softplus", NetPort["Steps"]} -> "tau" -> "tau_cum" -> "transmittance",
    "tau" -> "alpha",
    {"transmittance", "alpha"} -> "weights" -> "weights_unsqueeze",

    "weights_unsqueeze" -> "opacity_accum" -> NetPort["Opacity"]
  };

  If[effectiveColorAct =!= None,
    AppendTo[edges, "color_linear" -> "color_act"];
    AppendTo[edges, {"color_act", "weights_unsqueeze"} -> "rgb_weighted" -> "rgb_accum"];,
    AppendTo[edges, {"color_linear", "weights_unsqueeze"} -> "rgb_weighted" -> "rgb_accum"];
  ];

  If[VectorQ[effectiveBgColor, NumericQ] && Length[effectiveBgColor] == 3,
    layers["transmittance_final"] = ElementwiseLayer[1.0 - # &];
    layers["bg_linear"] = LinearLayer[
      3,
      "Input" -> 1,
      "Biases" -> None,
      "Weights" -> Developer`ToPackedArray[N[List /@ effectiveBgColor]],
      "LearningRateMultipliers" -> {"Weights" -> 0.0}
    ];
    layers["rgb_final"] = ThreadingLayer[Plus];
    AppendTo[edges, "opacity_accum" -> "transmittance_final" -> "bg_linear"];
    AppendTo[edges, {"rgb_accum", "bg_linear"} -> "rgb_final" -> NetPort["RGB"]];,
    AppendTo[edges, "rgb_accum" -> NetPort["RGB"]];
  ];

  (* Optional Depth accumulation *)
  If[withDepth,
    layers["depth_t_unsqueeze"] = ReshapeLayer[{numSamples, 1}];
    layers["depth_weighted"] = ThreadingLayer[Times];
    layers["depth_accum"] = AggregationLayer[Total, 1];
    edges = Join[edges, {
      NetPort["SampleT"] -> "depth_t_unsqueeze",
      {"depth_t_unsqueeze", "weights_unsqueeze"} -> "depth_weighted" -> "depth_accum" -> NetPort["Depth"]
    }];
  ];

  Module[{portSpecs = {
    "SamplePoints" -> {numSamples, 3},
    "Steps" -> {numSamples}
  }},
    If[withDepth, AppendTo[portSpecs, "SampleT" -> {numSamples}]];
    NetGraph[layers, edges, Sequence @@ portSpecs]
  ]
];

CreateLogL1LossNet[] := NetGraph[
  <|
    "log_pred" -> ElementwiseLayer[Log[1.0 + Max[#, 0.0]] &],
    "log_target" -> ElementwiseLayer[Log[1.0 + Max[#, 0.0]] &],
    "l1" -> MeanAbsoluteLossLayer[]
  |>,
  {
    NetPort["Input"] -> "log_pred",
    NetPort["Target"] -> "log_target",
    {"log_pred", "log_target"} -> "l1" -> NetPort["Output"]
  }
];

CreateLogL2LossNet[] := NetGraph[
  <|
    "log_pred" -> ElementwiseLayer[Log[1.0 + Max[#, 0.0]] &],
    "log_target" -> ElementwiseLayer[Log[1.0 + Max[#, 0.0]] &],
    "l2" -> MeanSquaredLossLayer[]
  |>,
  {
    NetPort["Input"] -> "log_pred",
    NetPort["Target"] -> "log_target",
    {"log_pred", "log_target"} -> "l2" -> NetPort["Output"]
  }
];

CreateTonemappedL1LossNet[] := NetGraph[
  <|
    "tm_pred" -> ElementwiseLayer[Max[#, 0.0] / (1.0 + Max[#, 0.0]) &],
    "tm_target" -> ElementwiseLayer[Max[#, 0.0] / (1.0 + Max[#, 0.0]) &],
    "l1" -> MeanAbsoluteLossLayer[]
  |>,
  {
    NetPort["Input"] -> "tm_pred",
    NetPort["Target"] -> "tm_target",
    {"tm_pred", "tm_target"} -> "l1" -> NetPort["Output"]
  }
];

Options[CreateNeRFLiteTrainingNet] = {
  "Loss" -> "LogL1", (* "LogL1", "L2", "L1", "LogL2", "SmoothL1", or "Tonemapped" *)
  "WeightOpacity" -> 0.0
};

CreateNeRFLiteTrainingNet[fieldNet_NetGraph, opts : OptionsPattern[]] := Module[{
  lossType = OptionValue["Loss"],
  wOpacity = N @ OptionValue["WeightOpacity"],
  lossLayer, inPorts, numSamples, hasDepth, layers = <||>, edges = {}, portSpecs
},
  inPorts = Quiet @ Information[fieldNet, "InputPorts"];
  numSamples = If[AssociationQ[inPorts] && KeyExistsQ[inPorts, "Steps"],
    Flatten[{inPorts["Steps"]}][[1]],
    32
  ];
  hasDepth = KeyExistsQ[inPorts, "SampleT"];

  lossLayer = Switch[lossType,
    "LogL1" | "Log", CreateLogL1LossNet[],
    "LogL2", CreateLogL2LossNet[],
    "Tonemapped" | "Reinhard", CreateTonemappedL1LossNet[],
    "L1", MeanAbsoluteLossLayer[],
    "SmoothL1", MeanAbsoluteLossLayer[],
    _, MeanSquaredLossLayer[]
  ];

  layers["nerf"] = fieldNet;
  layers["color_loss"] = lossLayer;

  edges = {
    NetPort["SamplePoints"] -> NetPort["nerf", "SamplePoints"],
    NetPort["Steps"] -> NetPort["nerf", "Steps"],
    NetPort["nerf", "RGB"] -> NetPort["color_loss", "Input"],
    NetPort["TargetRGB"] -> NetPort["color_loss", "Target"]
  };

  If[hasDepth,
    AppendTo[edges, NetPort["SampleT"] -> NetPort["nerf", "SampleT"]]
  ];

  If[NumericQ[wOpacity] && wOpacity > 0.0,
    layers["opacity_penalty"] = NetChain[{
      ElementwiseLayer[#^2 &],
      AggregationLayer[Total],
      ElementwiseLayer[# * wOpacity &]
    }];
    layers["total_loss"] = TotalLayer[];
    edges = Join[edges, {
      NetPort["nerf", "Opacity"] -> "opacity_penalty",
      {"color_loss", "opacity_penalty"} -> "total_loss" -> NetPort["Loss"]
    }];
    ,
    AppendTo[edges, "color_loss" -> NetPort["Loss"]];
  ];

  portSpecs = {
    "SamplePoints" -> {numSamples, 3},
    "Steps" -> {numSamples},
    "TargetRGB" -> {3}
  };
  If[hasDepth, AppendTo[portSpecs, "SampleT" -> {numSamples}]];

  NetGraph[layers, edges, Sequence @@ portSpecs]
];

ExtractNeRFLiteDensityNet[net_NetGraph] := Module[{encoder, backbone, linear, act},
  encoder = Quiet @ Check[NetExtract[net, {"encoder", "Net"}], $Failed];
  backbone = Quiet @ Check[NetExtract[net, {"backbone", "Net"}], $Failed];
  linear = Quiet @ Check[NetExtract[net, {"density_linear", "Net"}], $Failed];
  act = ElementwiseLayer["SoftPlus"];
  
  If[MatchQ[encoder, _NetGraph | _NetChain] && MatchQ[backbone, _NetGraph | _NetChain] && MatchQ[linear, _LinearLayer],
    NetChain[{encoder, backbone, linear, act}, "Input" -> 3],
    $Failed
  ]
];

ExtractNeRFLiteColorNet[net_NetGraph] := Module[{encoder, backbone, linear, act},
  encoder = Quiet @ Check[NetExtract[net, {"encoder", "Net"}], $Failed];
  backbone = Quiet @ Check[NetExtract[net, {"backbone", "Net"}], $Failed];
  linear = Quiet @ Check[NetExtract[net, {"color_linear", "Net"}], $Failed];
  act = Quiet @ Check[NetExtract[net, "color_act"], ElementwiseLayer[LogisticSigmoid]];
  
  If[MatchQ[encoder, _NetGraph | _NetChain] && MatchQ[backbone, _NetGraph | _NetChain] && MatchQ[linear, _LinearLayer],
    If[act =!= $Failed && MatchQ[act, _ElementwiseLayer],
      NetChain[{encoder, backbone, linear, act}, "Input" -> 3],
      NetChain[{encoder, backbone, linear}, "Input" -> 3]
    ],
    $Failed
  ]
];

End[];
EndPackage[];
