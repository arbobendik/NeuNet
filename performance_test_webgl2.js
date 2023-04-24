'use strict';

let inputData = [[0, 1, 2], [1, 0, -1], [0, 1, 2], [2, 1, 0], [1, 2, 3], [3, 2, 1], [2, 3, 4], [4, 3, 2], [10, 5, 1], [0, 5, 10], [0, 5, 6], [10, 5, 0], [4, 7, 10], [-1, -2, -3]];
let givenPoints = [[0, 0], [1, -1], [0, 0], [1, -1], [0, 0], [1, -1], [0, 0], [1, -1], [2, -2], [0, 0], [0, 0], [2, -2], [0, 0], [1, -1]];

let predictFor = [[10, 5, 1], [0, 5, 10], [0, 5, 6], [10, 5, 0], [4, 7, 10], [-1, -2, -3]];
let correctPredictions = [[2, -2], [0, 0], [0, 0], [2, -2], [0, 0], [1, -1]];

var TestObject = (runs, passes, structure) => ({
  runs: runs,
  passes: passes,
  structure: structure,
  results: [],
  correctResults: 0,
  trainingTime: 0,
  predictionTime: 0
});

var netCPU = TestObject(1, 10, [3, 100, 2]);
var netWebGL2 = TestObject(1, 10, [3, 100, 2]);

let doTest = async () => {
  for (let r = 0; r < netCPU.runs; r++) {
    let net = new Net(netCPU.structure);
  
    let t0 = performance.now();
    for (let p = 0; p < netCPU.passes; p++) {
      for (let i = 0; i < inputData.length; i++) await net.trainCPU(inputData[i], givenPoints[i]);
    }
  
    let t1 = performance.now();
    netCPU.results = [];

    for (let i = 0; i < predictFor.length; i++) {
      netCPU.results.push((await net.predictCPU(predictFor[i])).map((e) => Math.round(e)));
    }
    if (netCPU.results.reduce((result, item, i) => (item === correctPredictions[i]) && result)) netCPU.correctResults++;
    console.log(net.neurons);
  
    let t2 = performance.now();
    netCPU.trainingTime += Math.round(t1 - t0);
    netCPU.predictionTime += Math.round(t2 - t1);
  }

  for (let r = 0; r < netWebGL2.runs; r++) {
    let net = new Net(netWebGL2.structure);
  
    let t0 = performance.now();
    for (let p = 0; p < netWebGL2.passes; p++) {
      for (let i = 0; i < inputData.length; i++) await net.trainGPU(inputData[i], givenPoints[i]);
    }
    net.saveTraining();
  
    let t1 = performance.now();
  
    netWebGL2.results = [];
    for (let i = 0; i < predictFor.length; i++) {
      netWebGL2.results.push((await net.predictGPU(predictFor[i])).map((e) => Math.round(e)));
    }
  
    if (netWebGL2.results.reduce((result, item, i) => (item === correctPredictions[i]) && result)) netWebGL2.correctResults++;
    console.log(net.neurons);
  
    let t2 = performance.now();
    netWebGL2.trainingTime += Math.round(t1 - t0);
    netWebGL2.predictionTime += Math.round(t2 - t1);
  }

  document.body.innerHTML += "NET / CPU:\n";
  document.body.innerHTML += "[ <n>" + netCPU.results.join("</n>, <n>") + "</n> ]\n";
  document.body.innerHTML += "Timings for " + netCPU.runs + " test runs in a row.\n";
  document.body.innerHTML += "Net accuracy: " + (netCPU.correctResults / netCPU.runs) * 100 + "%\n";
  document.body.innerHTML += "Training: (" + netCPU.passes + " passes) " + netCPU.trainingTime + "ms\n";
  document.body.innerHTML += "Prediction: " + netCPU.predictionTime + "ms\n";
  document.body.innerHTML += "\n";
  document.body.innerHTML += "NET / WebGL2:\n";
  document.body.innerHTML += "[ <n>" + netWebGL2.results.join("</n>, <n>") + "</n> ]\n";
  document.body.innerHTML += "Timings for " + netWebGL2.runs + " test runs in a row.\n";
  document.body.innerHTML += "Net accuracy: " + (netWebGL2.correctResults / netWebGL2.runs) * 100 + "%\n";
  document.body.innerHTML += "Training: (" + netWebGL2.passes + " passes) " + netWebGL2.trainingTime + "ms\n";
  document.body.innerHTML += "Prediction: " + netWebGL2.predictionTime + "ms\n";
}

doTest();
