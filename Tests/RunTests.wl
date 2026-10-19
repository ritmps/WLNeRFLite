(* ::Package:: *)

Needs["MUnit`"];

testFile = FileNameJoin[{DirectoryName[$InputFileName], "NeRFLite.wlt"}];
Print["Running WLNeRFLite test suite: ", testFile];

suiteResult = TestRun[testFile];

Print["\n=== WLNeRFLite Test Results ==="];
Print["All Tests Succeeded: ", TrueQ[suiteResult]];

If[TrueQ[suiteResult],
  Exit[0],
  Exit[1]
];
