(* ::Package:: *)

BeginPackage["WLNeRFLite`Trainer`", {
  "NeuralNetworks`",
  "WLNeRFLite`Data`",
  "WLNeRFLite`Net`"
}];

CreateNeRFLiteBatchGenerator::usage = "CreateNeRFLiteBatchGenerator[views, opts] creates a stateful batch generator function for NetTrain.";
TrainNeRFLiteModel::usage = "TrainNeRFLiteModel[net, viewsOrGen, opts] executes lightweight NeRF training with GPU acceleration and checkpointing.";
ExtractNeRFLiteInferenceNet::usage = "ExtractNeRFLiteInferenceNet[trainedNet] extracts the core inference prediction net from a trained loss graph.";

Begin["`Private`"];

Options[CreateNeRFLiteBatchGenerator] = Join[
  {
    "BatchSize" -> 1024,
    "NumSamples" -> 32,
    "RayBounds" -> {3.0, 10.0},
    "Perturb" -> True,
    "ViewsPerBatch" -> Automatic,
    "SampleForegroundRatio" -> 0.0,
    "IncludeDepth" -> False,
    "PrecomputeRays" -> True,
    "Parallel" -> Automatic
  },
  Options[LoadNeRFDataset],
  Options[LoadRenderManEXRViews]
];

CreateNeRFLiteBatchGenerator[viewsOrPoolInput_, opts : OptionsPattern[]] := Module[{
  defaultBs = OptionValue["BatchSize"],
  numSamples = OptionValue["NumSamples"],
  rayBounds = OptionValue["RayBounds"],
  perturbQ = OptionValue["Perturb"],
  viewsPerBatchOpt = OptionValue["ViewsPerBatch"],
  sampleFgRatio = N @ OptionValue["SampleForegroundRatio"],
  withDepth = TrueQ @ OptionValue["IncludeDepth"],
  precomputeOpt = OptionValue["PrecomputeRays"],
  tNear, tFar, nViews, isPool, pool, views
},
  {tNear, tFar} = N[rayBounds];

  (* 1. Check if input is already a precomputed ray pool *)
  isPool = AssociationQ[viewsOrPoolInput] && (
    Lookup[viewsOrPoolInput, "Type", None] === "NeRFLiteRayPool" ||
    (KeyExistsQ[viewsOrPoolInput, "Foreground"] && KeyExistsQ[viewsOrPoolInput, "All"])
  );

  If[isPool,
    pool = viewsOrPoolInput;,
    (* Input is views or paths *)
    views = Which[
      ListQ[viewsOrPoolInput] && AssociationQ[First[viewsOrPoolInput, None]],
        viewsOrPoolInput,
      StringQ[viewsOrPoolInput] && (StringEndsQ[ToLowerCase[viewsOrPoolInput], ".json"] || (DirectoryQ[viewsOrPoolInput] && Length[FileNames["transforms*.json", viewsOrPoolInput]] > 0)),
        LoadNeRFDataset[viewsOrPoolInput, Sequence @@ FilterRules[{opts}, Options[LoadNeRFDataset]]],
      StringQ[viewsOrPoolInput] || (ListQ[viewsOrPoolInput] && StringQ[First[viewsOrPoolInput, None]]),
        LoadRenderManEXRViews[viewsOrPoolInput, Sequence @@ FilterRules[{opts}, Options[LoadRenderManEXRViews]]],
      True, {}
    ];
    If[FailureQ[views] || Length[views] == 0,
      Return[Failure["EmptyViewsList", <|
        "Message" -> "No views available to create batch generator. Ensure input paths match existing NeRF datasets or RenderMan EXR files.",
        "Input" -> viewsOrPoolInput
      |>]]
    ];

    If[TrueQ[precomputeOpt],
      pool = PrecomputeNeRFLiteRayPool[views];
      If[FailureQ[pool], Return[pool]];
      isPool = True;
    ];
  ];

  (* 2. Fast Path: In-Memory Precomputed Ray Pool Generator *)
  If[isPool,
    Return[
      Function[Null,
        Module[{
          args = {##}, batchInfo, bs,
          hasFg, fgCount, allCount,
          fgIdx, allIdx,
          curOrig, curDirs, curTgt,
          sampled, res
        },
          batchInfo = If[Length[args] > 0, args[[1]], None];
          bs = Which[
            IntegerQ[batchInfo] && batchInfo > 0, batchInfo,
            AssociationQ[batchInfo], Lookup[batchInfo, "BatchSize", defaultBs],
            True, defaultBs
          ];
          If[!IntegerQ[bs] || bs <= 0, bs = defaultBs];

          fgCount = Lookup[pool, "TotalForegroundRays", If[AssociationQ[pool["Foreground"]], Length[Lookup[pool["Foreground"], "Origins", {}]], 0]];
          allCount = Lookup[pool, "TotalAllRays", If[AssociationQ[pool["All"]], Length[Lookup[pool["All"], "Origins", {}]], 0]];
          hasFg = (TrueQ[pool["HasForeground"]] || fgCount > 0) && AssociationQ[pool["Foreground"]];

          If[hasFg && NumericQ[sampleFgRatio] && sampleFgRatio > 0.0 && fgCount > 0,
            Module[{nFg = Round[bs * Min[1.0, Max[0.0, sampleFgRatio]]], nAll},
              nAll = bs - nFg;
              fgIdx = RandomInteger[{1, fgCount}, nFg];
              If[nAll > 0,
                allIdx = RandomInteger[{1, allCount}, nAll];
                curOrig = Join[pool["Foreground", "Origins"][[fgIdx]], pool["All", "Origins"][[allIdx]]];
                curDirs = Join[pool["Foreground", "Directions"][[fgIdx]], pool["All", "Directions"][[allIdx]]];
                curTgt  = Join[pool["Foreground", "TargetRGB"][[fgIdx]], pool["All", "TargetRGB"][[allIdx]]],
                curOrig = pool["Foreground", "Origins"][[fgIdx]];
                curDirs = pool["Foreground", "Directions"][[fgIdx]];
                curTgt  = pool["Foreground", "TargetRGB"][[fgIdx]]
              ]
            ],
            allIdx = RandomInteger[{1, allCount}, bs];
            curOrig = pool["All", "Origins"][[allIdx]];
            curDirs = pool["All", "Directions"][[allIdx]];
            curTgt  = pool["All", "TargetRGB"][[allIdx]]
          ];

          sampled = SampleRaysStratified[curOrig, curDirs, numSamples, {tNear, tFar}, "Perturb" -> perturbQ];

          res = <|
            "SamplePoints" -> NumericArray[sampled["SamplePoints"], "Real32"],
            "Steps" -> NumericArray[sampled["Steps"], "Real32"],
            "TargetRGB" -> NumericArray[curTgt, "Real32"]
          |>;
          If[withDepth, res["SampleT"] = NumericArray[sampled["SampleT"], "Real32"]];
          res
        ]
      ]
    ]
  ];

  (* 3. View-by-View Dynamic Ray Generation *)
  nViews = Length[views];
  Function[Null,
    Module[{
      args = {##}, batchInfo,
      bs, kViews, chosenIndices, numActiveViews, raysPerView, remainder,
      allPts = {}, allSteps = {}, allTgt = {}, allT = {}
    },
      batchInfo = If[Length[args] > 0, args[[1]], None];
      bs = Which[
        IntegerQ[batchInfo] && batchInfo > 0, batchInfo,
        AssociationQ[batchInfo], Lookup[batchInfo, "BatchSize", defaultBs],
        True, defaultBs
      ];
      If[!IntegerQ[bs] || bs <= 0, bs = defaultBs];

      kViews = Which[
        viewsPerBatchOpt === All, nViews,
        IntegerQ[viewsPerBatchOpt] && viewsPerBatchOpt > 0, Min[nViews, viewsPerBatchOpt],
        viewsPerBatchOpt === Automatic, Min[nViews, 4],
        True, Min[nViews, 4]
      ];
      chosenIndices = If[kViews >= nViews,
        Range[nViews],
        RandomSample[Range[nViews], kViews]
      ];
      numActiveViews = Length[chosenIndices];

      raysPerView = Floor[bs / numActiveViews];
      remainder = Mod[bs, numActiveViews];

      Do[
        Module[{
          idx = chosenIndices[[j]],
          v, w, h, count, uvs, rays, sampled, tgt
        },
          v = views[[idx]];
          count = raysPerView + If[j <= remainder, 1, 0];
          If[count > 0,
            {w, h} = v["Dimensions"];
            uvs = If[NumericQ[sampleFgRatio] && sampleFgRatio > 0.0 && KeyExistsQ[v, "ForegroundPixels"] && ListQ[v["ForegroundPixels"]] && Length[v["ForegroundPixels"]] > 10,
              Module[{nFg = Round[count * sampleFgRatio], nBg, fgUVs, bgUVs},
                nBg = count - nFg;
                fgUVs = RandomChoice[v["ForegroundPixels"], nFg];
                bgUVs = Transpose[{RandomInteger[{0, w - 1}, nBg], RandomInteger[{0, h - 1}, nBg]}];
                Join[fgUVs, bgUVs]
              ],
              Transpose[{RandomInteger[{0, w - 1}, count], RandomInteger[{0, h - 1}, count]}]
            ];
            rays = GenerateCameraRays[v["CameraMatrix"], {w, h}, "PixelCoords" -> uvs, "FOV" -> v["FOV"]];
            sampled = SampleRaysStratified[rays["Origins"], rays["Directions"], numSamples, {tNear, tFar}, "Perturb" -> perturbQ];
            tgt = Extract[v["RGB"], Map[{#[[2]] + 1, #[[1]] + 1} &, uvs]];

            allPts = Join[allPts, sampled["SamplePoints"]];
            allSteps = Join[allSteps, sampled["Steps"]];
            allTgt = Join[allTgt, tgt];
            If[withDepth, allT = Join[allT, sampled["SampleT"]]];
          ]
        ],
        {j, numActiveViews}
      ];

      Module[{res = <|
        "SamplePoints" -> NumericArray[allPts, "Real32"],
        "Steps" -> NumericArray[allSteps, "Real32"],
        "TargetRGB" -> NumericArray[allTgt, "Real32"]
      |>},
        If[withDepth, res["SampleT"] = NumericArray[allT, "Real32"]];
        res
      ]
    ]
  ]
];

Options[TrainNeRFLiteModel] = Join[
  {
    "BatchSize" -> 1024,
    "MaxTrainingRounds" -> 2000,
    "LearningRate" -> 5*^-4,
    "TargetDevice" -> "GPU",
    "Method" -> "ADAM",
    "TrainingProgressReporting" -> Automatic,
    "ProgressReporting" -> Automatic,
    "CheckpointDirectory" -> None,
    "CheckpointInterval" -> Quantity[10, "Minutes"],
    "ValidationGenerator" -> None
  },
  Options[CreateNeRFLiteBatchGenerator],
  Options[CreateNeRFLiteTrainingNet],
  Options[NetTrain]
];

TrainNeRFLiteModel[inputNet_NetGraph, viewsOrGen_, opts : OptionsPattern[{TrainNeRFLiteModel, CreateNeRFLiteBatchGenerator, CreateNeRFLiteTrainingNet, NetTrain}]] := Module[{
  batchSize = OptionValue["BatchSize"],
  maxRounds = OptionValue["MaxTrainingRounds"],
  lr = OptionValue["LearningRate"],
  device = OptionValue["TargetDevice"],
  methodOpt = OptionValue["Method"],
  progReport = Replace[OptionValue["TrainingProgressReporting"], Automatic :> OptionValue["ProgressReporting"]],
  ckptDir = OptionValue["CheckpointDirectory"],
  ckptInterval = OptionValue["CheckpointInterval"],
  valGen = OptionValue["ValidationGenerator"],
  netTrainOpts = <||>,
  baseNet = inputNet,
  trainNet, trainGen, trained
},
  (* 1. Wrap with training loss if needed *)
  trainNet = If[KeyExistsQ[Quiet @ Information[baseNet, "OutputPorts"], "Loss"],
    baseNet,
    CreateNeRFLiteTrainingNet[baseNet, Sequence @@ FilterRules[{opts}, Options[CreateNeRFLiteTrainingNet]]]
  ];

  (* 2. Resolve generator *)
  trainGen = If[MatchQ[viewsOrGen, _Function],
    viewsOrGen,
    Module[{inPorts = Quiet @ Information[trainNet, "InputPorts"], numSamples = 32, withDepth = False},
      If[AssociationQ[inPorts] && KeyExistsQ[inPorts, "Steps"], numSamples = Flatten[{inPorts["Steps"]}][[1]]];
      If[AssociationQ[inPorts] && KeyExistsQ[inPorts, "SampleT"], withDepth = True];
      CreateNeRFLiteBatchGenerator[
        viewsOrGen,
        "BatchSize" -> batchSize,
        "NumSamples" -> numSamples,
        "IncludeDepth" -> withDepth,
        Sequence @@ FilterRules[{opts}, Options[CreateNeRFLiteBatchGenerator]]
      ]
    ]
  ];

  If[FailureQ[trainGen],
    Return[trainGen]
  ];

  (* 3. Configure NetTrain options *)
  netTrainOpts = <|
    LossFunction -> "Loss",
    BatchSize -> batchSize,
    MaxTrainingRounds -> maxRounds,
    TargetDevice -> Which[
      device === "MultiGPU" || device === "AllGPU", {"GPU", All},
      True, device
    ],
    Method -> Which[
      MatchQ[methodOpt, {_String, ___}], methodOpt,
      StringQ[methodOpt], {methodOpt, "LearningRate" -> lr},
      True, {"ADAM", "LearningRate" -> lr}
    ]
  |>;

  (* 4. Intelligent Progress Reporting *)
  Module[{inNotebookQ = TrueQ[$Notebooks]},
    Which[
      progReport === None || progReport === False,
        netTrainOpts[TrainingProgressReporting] = None,
      progReport === "Print",
        netTrainOpts[TrainingProgressReporting] = "Print",
      progReport === "Panel",
        netTrainOpts[TrainingProgressReporting] = If[inNotebookQ, Automatic, None],
      progReport === "Compact" || (progReport === Automatic && !inNotebookQ),
        netTrainOpts[TrainingProgressReporting] = None;
        netTrainOpts[TrainingProgressFunction] = {
          Function[info,
            If[KeyExistsQ[info, "Batch"] || KeyExistsQ[info, "Round"] || KeyExistsQ[info, "AbsoluteBatch"],
              Module[{
                curRound = Lookup[info, "Round", Lookup[info, "AbsoluteBatch", Lookup[info, "Batch", 0]]],
                totRounds = Lookup[info, "TotalRounds", maxRounds],
                loss = Which[
                  NumericQ[Lookup[info, "RoundLoss", None]], info["RoundLoss"],
                  NumericQ[Lookup[info, "BatchLoss", None]], info["BatchLoss"],
                  NumericQ[Lookup[info, "Loss", None]], info["Loss"],
                  True, Missing[]
                ],
                rate = Which[
                  NumericQ[Lookup[info, "BatchesPerSecond", None]], info["BatchesPerSecond"] * batchSize,
                  NumericQ[Lookup[info, "ExamplesPerSecond", None]], info["ExamplesPerSecond"],
                  NumericQ[Lookup[info, "MeanBatchesPerSecond", None]], info["MeanBatchesPerSecond"] * batchSize,
                  True, Missing[]
                ],
                timeVal = Lookup[info, "TimeElapsed", Lookup[info, "CumulativeTime", Missing[]]],
                remVal = Lookup[info, "TimeRemaining", Missing[]],
                sec, remSec, roundStr, lossStr, rateStr, timeStr, etaStr
              },
                sec = Which[
                  QuantityQ[timeVal], QuantityMagnitude[timeVal],
                  NumericQ[timeVal], timeVal,
                  True, Missing[]
                ];
                remSec = Which[
                  QuantityQ[remVal], QuantityMagnitude[remVal],
                  NumericQ[remVal], remVal,
                  True, Missing[]
                ];
                roundStr = If[IntegerQ[totRounds] && totRounds > 0,
                  ToString[curRound] <> "/" <> ToString[totRounds],
                  ToString[curRound]
                ];
                lossStr = If[NumericQ[loss], ToString @ NumberForm[loss, {5, 4}], "--"];
                rateStr = If[NumericQ[rate], ToString[Round[rate]], "--"];
                timeStr = If[NumericQ[sec], ToString[Round[sec, 0.1]] <> "s", "--"];
                etaStr = If[NumericQ[remSec], " | ETA: " <> ToString[Round[remSec, 0.1]] <> "s", ""];
                Print[StringTemplate["[NeRF Lite] Round `1` | Loss: `2` | Rate: `3` rays/s | Elapsed: `4``5`"][
                  roundStr, lossStr, rateStr, timeStr, etaStr
                ]]
              ]
            ]
          ],
          "Interval" -> Quantity[3, "Seconds"]
        },
      True,
        netTrainOpts[TrainingProgressReporting] = Automatic
    ];
  ];

  (* Checkpointing *)
  If[StringQ[ckptDir] && ckptDir =!= "None",
    If[!DirectoryQ[ckptDir], CreateDirectory[ckptDir, CreateIntermediateDirectories -> True]];
    netTrainOpts[TrainingProgressCheckpointing] = {"Directory", ckptDir, "Interval" -> ckptInterval};
  ];

  If[valGen =!= None,
    netTrainOpts[ValidationSet] = valGen;
  ];

  Module[{passthrough = FilterRules[{opts}, Options[NetTrain]]},
    Do[netTrainOpts[p[[1]]] = p[[2]], {p, passthrough}]
  ];

  trained = NetTrain[trainNet, trainGen, Sequence @@ Normal[netTrainOpts]];
  ExtractNeRFLiteInferenceNet[trained]
];

ExtractNeRFLiteInferenceNet[trainedNet_NetGraph] := Module[{extracted},
  extracted = Quiet @ Check[NetExtract[trainedNet, "nerf"], $Failed];
  If[MatchQ[extracted, _NetGraph | _NetChain],
    extracted,
    trainedNet
  ]
];

End[];
EndPackage[];
