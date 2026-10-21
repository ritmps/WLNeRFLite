(* ::Package:: *)

(* WLNeRFLite Context Initialization *)

Module[{dir = DirectoryName[$InputFileName]},
  If[StringQ[dir] && dir =!= "",
    Get[FileNameJoin[{dir, "NeRFLiteData.wl"}]];
    Get[FileNameJoin[{dir, "NeRFLiteNet.wl"}]];
    Get[FileNameJoin[{dir, "NeRFLiteTrainer.wl"}]];
    Get[FileNameJoin[{dir, "NeRFLiteVisualizer.wl"}]];
  ]
];

BeginPackage["WLNeRFLite`", {
  "WLNeRFLite`Data`",
  "WLNeRFLite`Net`",
  "WLNeRFLite`Trainer`",
  "WLNeRFLite`Visualizer`"
}];

(* Top-level aliases for user convenience *)
LoadNeRFDataset = WLNeRFLite`Data`LoadNeRFDataset;
ExportNeRFDataset = WLNeRFLite`Data`ExportNeRFDataset;
RenderManToNeRFTransform = WLNeRFLite`Data`RenderManToNeRFTransform;
NeRFToRenderManTransform = WLNeRFLite`Data`NeRFToRenderManTransform;
LoadRenderManEXRView = WLNeRFLite`Data`LoadRenderManEXRView;
LoadRenderManEXRViews = WLNeRFLite`Data`LoadRenderManEXRViews;
GenerateCameraRays = WLNeRFLite`Data`GenerateCameraRays;
SampleRaysStratified = WLNeRFLite`Data`SampleRaysStratified;
PrecomputeNeRFLiteRayPool = WLNeRFLite`Data`PrecomputeNeRFLiteRayPool;

CreateHarmonicEncoder = WLNeRFLite`Net`CreateHarmonicEncoder;
CreateNeRFLiteNet = WLNeRFLite`Net`CreateNeRFLiteNet;
CreateNeRFLiteTrainingNet = WLNeRFLite`Net`CreateNeRFLiteTrainingNet;
ExtractNeRFLiteDensityNet = WLNeRFLite`Net`ExtractNeRFLiteDensityNet;
ExtractNeRFLiteColorNet = WLNeRFLite`Net`ExtractNeRFLiteColorNet;

CreateNeRFLiteBatchGenerator = WLNeRFLite`Trainer`CreateNeRFLiteBatchGenerator;
TrainNeRFLiteModel = WLNeRFLite`Trainer`TrainNeRFLiteModel;
ExtractNeRFLiteInferenceNet = WLNeRFLite`Trainer`ExtractNeRFLiteInferenceNet;

RenderNeRFLiteView = WLNeRFLite`Visualizer`RenderNeRFLiteView;
RenderNeRFLiteTurntable = WLNeRFLite`Visualizer`RenderNeRFLiteTurntable;
SampleNeRFLiteDensityVolume = WLNeRFLite`Visualizer`SampleNeRFLiteDensityVolume;
VisualizeNeRFCameraRig = WLNeRFLite`Visualizer`VisualizeNeRFCameraRig;

EndPackage[];
