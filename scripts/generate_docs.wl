(* ::Package:: *)
(* scripts/generate_docs.wl
   Automated documentation generator for WLNeRFLite paclet symbols.
   Generates standard Wolfram Reference Notebooks (.nb) and Guides
   styled with the native Wolfram Reference.nb stylesheet.
*)

Print["=================================================="];
Print["Generating WLNeRFLite Paclet Documentation..."];
Print["=================================================="];

Module[{
  repoRoot, docRootDir, guideDir, subpackages, curatedUsages, seeAlsoMap,
  getSeeAlso, createDocNotebook, createGuideNotebook, totalCreated = 0
},
  repoRoot = DirectoryName[ExpandFileName[If[StringQ[$InputFileName] && $InputFileName =!= "", $InputFileName, "scripts/generate_docs.wl"]], 2];
  PacletDirectoryLoad[repoRoot];
  Needs["WLNeRFLite`"];

  docRootDir = FileNameJoin[{repoRoot, "Documentation", "English", "ReferencePages", "Symbols"}];
  guideDir   = FileNameJoin[{repoRoot, "Documentation", "English", "Guides"}];

  If[!DirectoryQ[docRootDir], CreateDirectory[docRootDir, CreateIntermediateDirectories -> True]];
  If[!DirectoryQ[guideDir], CreateDirectory[guideDir, CreateIntermediateDirectories -> True]];

  subpackages = {
    "WLNeRFLite`Data`",
    "WLNeRFLite`Net`",
    "WLNeRFLite`Trainer`",
    "WLNeRFLite`Visualizer`"
  };

  curatedUsages = <|
    "LoadNeRFDataset" -> "LoadNeRFDataset[pathOrDir, opts] loads a canonical NeRF synthetic / Instant-NGP / Nerfstudio dataset specified by transforms.json (or transforms_train.json) with PNG/JPG/TIFF images and camera poses.",
    "ExportNeRFDataset" -> "ExportNeRFDataset[views, outputDir, opts] exports a list of camera views into canonical NeRF synthetic format with transforms.json (and optional train/val/test splits) and images.",
    "RenderManToNeRFTransform" -> "RenderManToNeRFTransform[w2c] converts a RenderMan 4x4 row-major world-to-camera matrix to a standard NeRF 4x4 camera-to-world (c2w) matrix (OpenGL +X right, +Y up, -Z look).",
    "NeRFToRenderManTransform" -> "NeRFToRenderManTransform[c2w] converts a standard NeRF 4x4 camera-to-world matrix to a RenderMan 4x4 row-major world-to-camera (w2c) matrix.",
    "LoadRenderManEXRView" -> "LoadRenderManEXRView[exrPath, opts] loads camera transforms directly from RenderMan EXR binary headers and imports beauty RGB.",
    "LoadRenderManEXRViews" -> "LoadRenderManEXRViews[pathsOrDir, opts] loads a collection of RenderMan EXR views ready for NeRF training.",
    "GenerateCameraRays" -> "GenerateCameraRays[cameraMatrix, {w, h}, opts] returns ray origins and normalized world directions for a camera (auto-detects RenderMan w2c or NeRF c2w).",
    "SampleRaysStratified" -> "SampleRaysStratified[origins, directions, numSamples, {tNear, tFar}, opts] computes stratified 3D sample points and intervals along rays.",
    "PrecomputeNeRFLiteRayPool" -> "PrecomputeNeRFLiteRayPool[views, opts] precomputes camera rays and RGB targets across views into a high-performance in-memory ray pool.",

    "CreateHarmonicEncoder" -> "CreateHarmonicEncoder[numBands:6] creates a differentiable positional encoding net mapping {3} to {3 * (1 + 2*numBands)} on GPU.",
    "CreateNeRFLiteNet" -> "CreateNeRFLiteNet[opts] constructs an ultra-lightweight NeRF NetGraph (~10k parameters) with embedded harmonic encoder and differentiable volumetric quadrature.",
    "CreateNeRFLiteTrainingNet" -> "CreateNeRFLiteTrainingNet[fieldNet, opts] wraps a NeRF prediction NetGraph with photometric loss.",
    "CreateLogL1LossNet" -> "CreateLogL1LossNet[] constructs a tonemapped L1 loss net for HDR radiance using log(1 + x).",
    "CreateLogL2LossNet" -> "CreateLogL2LossNet[] constructs a tonemapped L2 loss net for HDR radiance using log(1 + x).",
    "CreateTonemappedL1LossNet" -> "CreateTonemappedL1LossNet[] constructs a Reinhard tonemapped L1 loss net.",
    "ExtractNeRFLiteDensityNet" -> "ExtractNeRFLiteDensityNet[net] extracts the point density evaluation subnet mapping {3} -> {1}.",
    "ExtractNeRFLiteColorNet" -> "ExtractNeRFLiteColorNet[net] extracts the point RGB evaluation subnet mapping {3} -> {3}.",

    "CreateNeRFLiteBatchGenerator" -> "CreateNeRFLiteBatchGenerator[views, opts] creates a stateful batch generator function for NetTrain with stratified ray sampling.",
    "TrainNeRFLiteModel" -> "TrainNeRFLiteModel[net, viewsOrGen, opts] executes lightweight NeRF training with GPU acceleration, compact progress reporting, and checkpointing.",
    "ExtractNeRFLiteInferenceNet" -> "ExtractNeRFLiteInferenceNet[trainedNet] extracts the core inference prediction net from a trained loss graph.",

    "RenderNeRFLiteView" -> "RenderNeRFLiteView[net, cameraMatrix, {w, h}, opts] renders a 2D novel-view image from a trained lightweight NeRF.",
    "RenderNeRFLiteTurntable" -> "RenderNeRFLiteTurntable[net, opts] renders a 360-degree orbiting turntable animation around the object.",
    "SampleNeRFLiteDensityVolume" -> "SampleNeRFLiteDensityVolume[densityNet, {xRange, yRange, zRange}, res] samples a 3D density grid for volume visualization.",
    "VisualizeNeRFCameraRig" -> "VisualizeNeRFCameraRig[views, opts] creates an interactive 3D visualization of camera frustums, positions, and bounds in world space."
  |>;

  seeAlsoMap = <|
    "Transforms" -> {"RenderManToNeRFTransform", "NeRFToRenderManTransform", "GenerateCameraRays"},
    "Data" -> {"LoadNeRFDataset", "ExportNeRFDataset", "LoadRenderManEXRView", "LoadRenderManEXRViews", "PrecomputeNeRFLiteRayPool"},
    "Ray" -> {"GenerateCameraRays", "SampleRaysStratified", "PrecomputeNeRFLiteRayPool"},
    "Net" -> {"CreateNeRFLiteNet", "CreateHarmonicEncoder", "CreateNeRFLiteTrainingNet", "ExtractNeRFLiteDensityNet", "ExtractNeRFLiteColorNet"},
    "Train" -> {"TrainNeRFLiteModel", "CreateNeRFLiteBatchGenerator", "ExtractNeRFLiteInferenceNet", "CreateNeRFLiteNet"},
    "Vis" -> {"RenderNeRFLiteView", "RenderNeRFLiteTurntable", "VisualizeNeRFCameraRig", "SampleNeRFLiteDensityVolume"}
  |>;

  getSeeAlso[symName_String] := Module[{matches},
    matches = Select[Values[seeAlsoMap], MemberQ[#, symName] &];
    If[Length[matches] > 0,
      DeleteCases[DeleteDuplicates[Flatten[matches]], symName],
      {"LoadNeRFDataset", "CreateNeRFLiteNet", "TrainNeRFLiteModel", "RenderNeRFLiteView"}
    ]
  ];

  formatSignatureBoxes[sig_String] := Module[{held, boxes, symHead},
    symHead = First[StringCases[sig, RegularExpression["^[A-Za-z0-9`]+"]]];
    held = Quiet @ Check[ToExpression[sig, InputForm, HoldComplete], Null];
    If[held =!= Null && MatchQ[held, HoldComplete[_[___]]],
      boxes = MakeBoxes @@ held;
      boxes = boxes /. s_String /; StringMatchQ[s, RegularExpression["[a-zA-Z][a-zA-Z0-9]*"]] && s =!= symHead && s =!= "List" && s =!= "Rule" && s =!= "Automatic" && s =!= "True" && s =!= "False" :> StyleBox[s, "TI"];
      boxes
    ,
      sig
    ]
  ];

  formatUsageCell[uText_String] := Module[{lines, gridRows},
    lines = Select[StringSplit[uText, "\n"], StringLength[StringTrim[#]] > 0 &];
    gridRows = Map[
      Function[{line},
        Module[{trimmed = StringTrim[line], parsed, sig, desc, sigBox},
          parsed = StringCases[trimmed, RegularExpression["(?s)^([A-Za-z0-9`]+(?:\\[[^\\]]*\\])?)\\s+(.*)$"] -> {"$1", "$2"}];
          If[Length[parsed] > 0,
            sig = parsed[[1, 1]];
            desc = parsed[[1, 2]];
            sigBox = formatSignatureBoxes[sig];
            {"", Cell[TextData[{Cell[BoxData[sigBox], "InlineFormula"], " ", desc}]]}
          ,
            {"", Cell[TextData[{trimmed}]]}
          ]
        ]
      ],
      lines
    ];
    Cell[BoxData[GridBox[gridRows]], "Usage"]
  ];

  createDocNotebook[symName_String, usageInput_, ctx_String, opts_List, seeAlsoList_List] := Module[
    {notesCells, seeAlsoCell, moreAboutCell, optionsRows, optionsTable, nbPath, usageText = ToString[usageInput]},

    If[opts =!= {},
      optionsRows = Map[
        Function[{opt},
          {
            "",
            Cell[BoxData[StyleBox[ToString[First[opt]], "InlineFormula"]]],
            Cell[BoxData[StyleBox[ToString[Last[opt]], "InlineFormula"]]]
          }
        ],
        opts
      ];
      optionsTable = Cell[BoxData[GridBox[optionsRows]], "2ColumnTableMod"];
      notesCells = {
        Cell["The following options can be given:", "Notes"],
        optionsTable
      };
    ,
      notesCells = {
        Cell[symName <> " has no options.", "Notes"]
      };
    ];

    seeAlsoCell = If[Length[seeAlsoList] > 0,
      Cell[TextData[
        Riffle[
          Map[
            Function[{s},
              Cell[BoxData[TemplateBox[{Cell[TextData[s]], "paclet:WLNeRFLite/ref/" <> s}, "RefLink", BaseStyle -> {"InlineFormula"}]]]
            ],
            seeAlsoList
          ],
          " \[FilledSmallSquare] "
        ]
      ], "SeeAlso"]
    ,
      Cell[TextData[{
        Cell[BoxData[TemplateBox[{Cell[TextData["WLNeRFLite Guide"]], "paclet:WLNeRFLite/guide/WLNeRFLite"}, "RefLink", BaseStyle -> {"InlineFormula"}]]]
      }], "SeeAlso"]
    ];

    moreAboutCell = Cell[TextData[{
      Cell[BoxData[
        TemplateBox[{Cell[TextData["WLNeRFLite Guide"]], "paclet:WLNeRFLite/guide/WLNeRFLite"}, "RefLink", BaseStyle -> {"MoreAbout"}]
      ]]
    }], "MoreAbout"];

    nbPath = FileNameJoin[{docRootDir, symName <> ".nb"}];
    Export[nbPath,
      Notebook[{
        Cell[CellGroupData[{
          Cell["Categorization", "CategorizationSection"],
          Cell["Symbol", "Categorization", CellLabel -> "Entity Type"],
          Cell["WLNeRFLite", "Categorization", CellLabel -> "Paclet Name"],
          Cell[ctx, "Categorization", CellLabel -> "Context"],
          Cell["WLNeRFLite/ref/" <> symName, "Categorization", CellLabel -> "URI"]
        }, Closed]],

        Cell[CellGroupData[{
          Cell[symName, "ObjectName"],
          formatUsageCell[usageText],

          Cell[CellGroupData[Join[{Cell["Details and Options", "NotesSection"]}, notesCells], Closed]],

          Cell[CellGroupData[{
            Cell["Examples", "PrimaryExamplesSection"],
            Cell[CellGroupData[{
              Cell["Basic Examples", "ExampleSection"],
              Cell["Load the WLNeRFLite package:", "ExampleText"],
              Cell[BoxData[RowBox[{"Needs", "[", "\"WLNeRFLite`\"", "]"}]], "Input"]
            }, Open]]
          }, Open]],

          Cell[CellGroupData[{
            Cell["See Also", "SeeAlsoSection"],
            seeAlsoCell
          }, Open]],

          Cell[CellGroupData[{
            Cell["Related Guides", "MoreAboutSection"],
            moreAboutCell
          }, Open]]
        }, Open]]
      },
      TaggingRules -> <|
        "Metadata" -> <|
          "context" -> ctx,
          "type" -> "Symbol",
          "paclet" -> "WLNeRFLite",
          "language" -> "en",
          "title" -> symName,
          "windowtitle" -> symName,
          "uri" -> "WLNeRFLite/ref/" <> symName,
          "summary" -> usageText,
          "index" -> True
        |>
      |>,
      ScreenStyleEnvironment -> "Working",
      WindowTitle -> symName,
      StyleDefinitions -> FrontEnd`FileName[{"Wolfram"}, "Reference.nb"],
      FrontEndVersion -> "13.0+"
      ]
    ];
    nbPath
  ];

  createGuideNotebook[] := Module[{nbPath, guideSections, sectionsCells},
    guideSections = {
      "Dataset Ingestion & Camera Transforms" -> {"LoadNeRFDataset", "ExportNeRFDataset", "RenderManToNeRFTransform", "NeRFToRenderManTransform", "LoadRenderManEXRView", "LoadRenderManEXRViews"},
      "Ray Generation & Precomputed Pools" -> {"GenerateCameraRays", "SampleRaysStratified", "PrecomputeNeRFLiteRayPool"},
      "Neural Field Coordinate MLP (~10k Params)" -> {"CreateNeRFLiteNet", "CreateHarmonicEncoder", "CreateNeRFLiteTrainingNet", "CreateLogL1LossNet", "CreateLogL2LossNet", "CreateTonemappedL1LossNet", "ExtractNeRFLiteDensityNet", "ExtractNeRFLiteColorNet"},
      "Model Training & Batch Generators" -> {"TrainNeRFLiteModel", "CreateNeRFLiteBatchGenerator", "ExtractNeRFLiteInferenceNet"},
      "Novel-View Synthesis & Visualizers" -> {"RenderNeRFLiteView", "RenderNeRFLiteTurntable", "VisualizeNeRFCameraRig", "SampleNeRFLiteDensityVolume"}
    };

    sectionsCells = Map[
      Function[{subsec},
        Module[{subTitle = subsec[[1]], symList = subsec[[2]], funcCells},
          funcCells = Map[
            Function[{sName},
              Module[{uText = Lookup[curatedUsages, sName, sName <> " function"], shortDesc, parsed, cleanDesc},
                shortDesc = First[StringSplit[uText, ". "]];
                If[!StringEndsQ[shortDesc, "."], shortDesc = shortDesc <> "."];
                parsed = StringCases[shortDesc, RegularExpression["(?s)^[A-Za-z0-9`]+(?:\\[[^\\]]*\\])?\\s+(.*)$"] -> "$1"];
                cleanDesc = If[Length[parsed] > 0,
                  ToUpperCase[StringTake[parsed[[1]], 1]] <> StringDrop[parsed[[1]], 1]
                ,
                  shortDesc
                ];
                Cell[TextData[{
                  Cell[BoxData[TemplateBox[{Cell[TextData[sName]], "paclet:WLNeRFLite/ref/" <> sName}, "RefLink", BaseStyle -> {"InlineFormula"}]]],
                  " \[LongDash] ",
                  cleanDesc
                }], "GuideText"]
              ]
            ],
            symList
          ];
          Cell[CellGroupData[Join[{Cell[subTitle, "GuideFunctionsSubsection"]}, funcCells], Open]]
        ]
      ],
      guideSections
    ];

    nbPath = FileNameJoin[{guideDir, "WLNeRFLite.nb"}];
    Export[nbPath,
      Notebook[{
        Cell[CellGroupData[{
          Cell["Categorization", "CategorizationSection"],
          Cell["Guide", "Categorization", CellLabel -> "Entity Type"],
          Cell["WLNeRFLite", "Categorization", CellLabel -> "Paclet Name"],
          Cell["WLNeRFLite`", "Categorization", CellLabel -> "Context"],
          Cell["WLNeRFLite/guide/WLNeRFLite", "Categorization", CellLabel -> "URI"]
        }, Closed]],
        Cell[CellGroupData[{
          Cell["WLNeRFLite Guide", "GuideTitle"],
          Cell["WLNeRFLite is a standalone, high-performance, ultra-lightweight Neural Radiance Field (NeRF) framework implemented natively in Wolfram Language. It supports standard NeRF synthetic datasets (transforms.json with PNG/JPG/TIFF) and Pixar RenderMan OpenEXR archives.", "GuideAbstract"]
        }, Open]],
        Cell[CellGroupData[Join[{Cell["Functions", "GuideFunctionsSection"]}, sectionsCells], Open]]
      },
      TaggingRules -> <|
        "Metadata" -> <|
          "context" -> "WLNeRFLite`",
          "type" -> "Guide",
          "paclet" -> "WLNeRFLite",
          "language" -> "en",
          "title" -> "WLNeRFLite Guide",
          "windowtitle" -> "WLNeRFLite Guide",
          "uri" -> "WLNeRFLite/guide/WLNeRFLite",
          "summary" -> "WLNeRFLite Guide",
          "index" -> True
        |>
      |>,
      ScreenStyleEnvironment -> "Working",
      WindowTitle -> "WLNeRFLite Guide",
      StyleDefinitions -> FrontEnd`FileName[{"Wolfram"}, "Reference.nb"],
      FrontEndVersion -> "13.0+"
      ]
    ];
    nbPath
  ];

  (* Generate symbol reference notebooks *)
  Do[
    Module[{symbols = Names[pkg <> "*"]},
      Do[
        Module[{
          shortName = Last[StringSplit[sym, "`"]],
          usage, symObj, optsList, seeAlso
        },
          symObj = ToExpression[pkg <> shortName];
          usage = Which[
            KeyExistsQ[curatedUsages, shortName], curatedUsages[shortName],
            StringQ[Quiet @ MessageName[symObj, "usage"]], MessageName[symObj, "usage"],
            True, shortName <> "[...] evaluates " <> shortName <> "."
          ];
          optsList = Quiet @ Check[DeleteDuplicatesBy[Options[symObj], First], {}];
          seeAlso = getSeeAlso[shortName];

          createDocNotebook[shortName, usage, pkg, optsList, seeAlso];
          totalCreated++;
          Print["  [Doc] ", shortName, " -> Documentation/English/ReferencePages/Symbols/", shortName, ".nb"];
        ],
        {sym, symbols}
      ]
    ],
    {pkg, subpackages}
  ];

  (* Generate Guide notebook *)
  createGuideNotebook[];
  Print["  [Guide] Generated Documentation/English/Guides/WLNeRFLite.nb"];
  Print["Documentation generation complete! Total symbols documented: ", totalCreated];
];
