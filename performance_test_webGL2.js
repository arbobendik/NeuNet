'use strict';

let inputData = [[0, 1, 2], [1, 0, -1], [0, 1, 2], [2, 1, 0], [1, 2, 3], [3, 2, 1], [2, 3, 4], [4, 3, 2], [10, 5, 1], [0, 5, 10], [0, 5, 6], [10, 5, 0], [4, 7, 10], [-1, -2, -3]];
let givenPoints = [0, 1, 0, 1, 0, 1, 0, 1, 2, 0, 0, 2, 0, 1];

let predictFor = [[10, 5, 1], [0, 5, 10], [0, 5, 6], [10, 5, 0], [4, 7, 10], [-1, -2, -3]];
let correctPredictions = [2, 0, 0, 2, 0, 1];

var TestObject = (net, name, trainFunc, predictFunc, runs, passes, structure) => ({
  name: name,
  net: net,
  trainFunc: trainFunc,
  predictFunc: predictFunc,
  runs: runs,
  passes: passes,
  structure: structure,
  results: [],
  correctResults: 0,
  trainingTime: 0,
  predictionTime: 0
});

var netCPU = TestObject(Net, "Net, CPU", "trainCPU", "predictCPU", 5, 10, [3, 2048, 2048, 1]);
var netGPU = TestObject(Net, "Net, GPU", "trainGPU", "predictGPU", 5, 10, [3, 2048, 2048, 1]);

var netWebGL2 = TestObject(NetWebGL2, "NetWebGL2, GPU", "train", "predict", 5, 10, [3, 2048, 2048, 1]);

let benchImplementation = proto => {
  for (let r = 0; r < proto.runs; r++) {
    let net = new proto.net(proto.structure);
  
    let t0 = performance.now();
    for (let p = 0; p < proto.passes; p++) {
      for (let i = 0; i < inputData.length; i++) net[proto.trainFunc](inputData[i], [givenPoints[i]]);
    }
  
    let t1 = performance.now();
  
    proto.results = [];
    for (let i = 0; i < predictFor.length; i++) proto.results.push(Math.round((net[proto.predictFunc](predictFor[i]))[0]));
  
    if (proto.results.reduce((result, item, i) => (item === correctPredictions[i]) && result)) proto.correctResults++;
    console.log(net.neurons);
  
    let t2 = performance.now();
    proto.trainingTime += Math.round(t1 - t0);
    proto.predictionTime += Math.round(t2 - t1);
  }

  document.body.innerHTML += proto.name, ":\n";
  document.body.innerHTML += "[ <n>" + proto.results.join("</n>, <n>") + "</n> ]\n";
  document.body.innerHTML += "Timings for " + proto.runs + " test runs in a row.\n";
  document.body.innerHTML += "Net accuracy: " + (proto.correctResults / proto.runs) * 100 + "%\n";
  document.body.innerHTML += "Training: (" + proto.passes + " passes) " + proto.trainingTime + "ms\n";
  document.body.innerHTML += "Prediction: " + proto.predictionTime + "ms\n";
  document.body.innerHTML += "\n";

}

benchImplementation(netCPU);
benchImplementation(netGPU);
benchImplementation(netWebGL2);
