(* ::Package:: *)
(* scripts/generate_docs.wl
   Automated documentation generator for WLNeRFLite paclet symbols.
   Generates standard Wolfram Reference Notebooks (.nb) and Guides.
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

  createDocNotebook[symName_String, usageInput_, ctx_String, opts_List, seeAlsoList_List] := Module[
    {notesCells, seeAlsoCells, optionsTable, nbPath, usageText = ToString[usageInput]},

    If[opts =!= {},
      optionsTable = Cell[BoxData[
        GridBox[
          Prepend[
            Map[{StyleBox[ToString[First[#]], "InlineFormula"], StyleBox[ToString[Last[#]], "InlineFormula"]}&, opts],
            {StyleBox["Option", FontWeight -> "Bold"], StyleBox["Default Value", FontWeight -> "Bold"]}
          ],
          GridBoxDividers -> {"Columns" -> {{False}}, "Rows" -> {False, True, {False}}},
          GridBoxSpacings -> {"Columns" -> {{2}}, "Rows" -> {{0.8}}}
        ]
      ], "TableText"];
      notesCells = {
        Cell["The following options can be given:", "NotesText"],
        optionsTable
      };
    ,
      notesCells = {
        Cell["Details and options for " <> symName <> ".", "NotesText"]
      };
    ];

    seeAlsoCells = Cell[BoxData[
      RowBox[Riffle[Map[StyleBox[#, "InlineFormula"]&, seeAlsoList], " \[DoubleLongRightArrow] "]]
    ], "SeeAlsoText"];

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
          Cell[BoxData[
            RowBox[{
              StyleBox[symName, "UsageSymbol"],
              " ",
              usageText
            }]
          ], "Usage"],

          Cell[CellGroupData[Join[{Cell["Details and Options", "NotesSection"]}, notesCells], Closed]],

          Cell[CellGroupData[{
            Cell["Examples", "PrimaryExamplesSection"],
            Cell[CellGroupData[{
              Cell["Basic Examples", "ExampleSection"],
              Cell[BoxData[RowBox[{"Needs", "[", "\"WLNeRFLite`\"", "]"}]], "Input"]
            }, Open]]
          }, Open]],

          Cell[CellGroupData[{
            Cell["See Also", "SeeAlsoSection"],
            seeAlsoCells
          }, Open]],

          Cell[CellGroupData[{
            Cell["Related Guides", "MoreAboutSection"],
            Cell[BoxData[RowBox[{StyleBox["WLNeRFLite Guide", "InlineFormula"]}]], "MoreAboutText"]
          }, Open]]
        }, Open]]
      },
      StyleDefinitions -> "Default.nb"
      ]
    ];
    nbPath
  ];

  createGuideNotebook[] := Module[{nbPath, guideSections},
    guideSections = {
      "Dataset Ingestion & Camera Transforms" -> {"LoadNeRFDataset", "ExportNeRFDataset", "RenderManToNeRFTransform", "NeRFToRenderManTransform", "LoadRenderManEXRView", "LoadRenderManEXRViews"},
      "Ray Generation & Precomputed Pools" -> {"GenerateCameraRays", "SampleRaysStratified", "PrecomputeNeRFLiteRayPool"},
      "Neural Field Coordinate MLP (~10k Params)" -> {"CreateNeRFLiteNet", "CreateHarmonicEncoder", "CreateNeRFLiteTrainingNet", "ExtractNeRFLiteDensityNet", "ExtractNeRFLiteColorNet"},
      "Model Training & Batch Generators" -> {"TrainNeRFLiteModel", "CreateNeRFLiteBatchGenerator", "ExtractNeRFLiteInferenceNet"},
      "Novel-View Synthesis & Visualizers" -> {"RenderNeRFLiteView", "RenderNeRFLiteTurntable", "VisualizeNeRFCameraRig", "SampleNeRFLiteDensityVolume"}
    };

    nbPath = FileNameJoin[{guideDir, "WLNeRFLite.nb"}];
    Export[nbPath,
      Notebook[{
        Cell[CellGroupData[{
          Cell["WLNeRFLite Guide", "GuideTitle"],
          Cell["WLNeRFLite is a standalone, high-performance, ultra-lightweight Neural Radiance Field (NeRF) framework implemented natively in Wolfram Language. It supports standard NeRF synthetic datasets (transforms.json with PNG/JPG/TIFF) and Pixar RenderMan OpenEXR archives.", "GuideAbstract"],

          Sequence @@ Flatten @ Map[
            Function[{pair},
              {
                Cell[pair[[1]], "GuideFunctionsSubsection"],
                Cell[BoxData[GridBox[
                  Map[{StyleBox[#, "InlineFormula"], StyleBox[Lookup[curatedUsages, #, ""], "TableText"]}&, pair[[2]]],
                  GridBoxDividers -> {"Columns" -> {{False}}, "Rows" -> {{False}}},
                  GridBoxSpacings -> {"Columns" -> {{2}}, "Rows" -> {{0.8}}}
                ]], "GuideFunctionListing"]
              }
            ],
            guideSections
          ]
        }, Open]]
      },
      StyleDefinitions -> "Default.nb"
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
          symObj = ToExpression[pkg <> shortName, InputForm, Unevaluated];
          usage = Which[
            KeyExistsQ[curatedUsages, shortName], curatedUsages[shortName],
            StringQ[Quiet @ MessageName[symObj, "usage"]], MessageName[symObj, "usage"],
            True, shortName <> "[...] evaluates " <> shortName <> "."
          ];
          optsList = Quiet @ Check[Options[symObj], {}];
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
