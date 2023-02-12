"use strict";

let inputData = [[0, 1, 2], [1, 0, -1], [0, 1, 2], [2, 1, 0], [1, 2, 3], [3, 2, 1], [2, 3, 4], [4, 3, 2]];
let givenPoints = [0, 1, 0, 1, 0, 1, 0, 1];

let predictFor = [[10, 5, 1], [0, 5, 10], [0, 5, 6], [10, 5, 0], [4, 7, 10], [-1, -2, -3]];
let correctPredictions = [1, 0, 0, 1, 0, 1];

var TestObject = (runs, passes, structure) => ({
  runs: runs,
  passes: passes,
  structure: structure,
  results: [],
  correctResults: 0,
  trainingTime: 0,
  predictionTime: 0
});

var neuron = TestObject(10, 10000);
var net = TestObject(1, 3, [3, 1000, 1000, 1]);
var netWebGL2 = TestObject(1, 3, [3, 1000, 1000, 1]);

for (let r = 0; r < neuron.runs; r++) {
  let test = new Neuron(3);

  let t0 = performance.now();
  for (let p = 0; p < neuron.passes; p++) {
    inputData.forEach((item, i) => {
      test.train(item, givenPoints[i]);
    });
  }

  let t1 = performance.now();
  neuron.results = [];
  predictFor.forEach((item, i) => {
    neuron.results.push(Math.round(test.forwardPropagation(item)));
  });

  let t2 = performance.now();
  neuron.trainingTime += Math.round(t1 - t0);
  neuron.predictionTime += Math.round(t2 - t1);
  if (neuron.results.reduce((result, item, i) => (item == correctPredictions[i]) && result)) neuron.correctResults++;
}

for (let r = 0; r < net.runs; r++) {
  let test = new Net(net.structure);

  let t0 = performance.now();
  for (let p = 0; p < net.passes; p++) {
    inputData.forEach((item, i) => {
      test.train(item, [givenPoints[i]]);
    });
  }

  let t1 = performance.now();
  net.results = [];
  predictFor.forEach((item, i) => {
    net.results.push(Math.round(test.predict(item)));
  });

  let t2 = performance.now();
  net.trainingTime += Math.round(t1 - t0);
  net.predictionTime += Math.round(t2 - t1);
  if (net.results.reduce((result, item, i) => (item == correctPredictions[i]) && result)) net.correctResults++;
}

for (let r = 0; r < netWebGL2.runs; r++) {
  let net = new NetWebGL2(netWebGL2.structure);

  let t0 = performance.now();
  for (let p = 0; p < netWebGL2.passes; p++) {
    inputData.forEach((item, i) => {
      net.trainGPU(item, [givenPoints[i]]);
    });
  }
  let t1 = performance.now();
  netWebGL2.results = [];
  predictFor.forEach(async (item, i) => {
    netWebGL2.results.push(Math.round(net.predict(item)));
  });

  let t2 = performance.now();
  netWebGL2.trainingTime += Math.round(t1 - t0);
  netWebGL2.predictionTime += Math.round(t2 - t1);
  if (netWebGL2.results.reduce((result, item, i) => (item == correctPredictions[i]) && result)) netWebGL2.correctResults++;
}

document.body.innerHTML = "";
document.body.innerHTML += "NEURON:\n";
document.body.innerHTML += "[ <n>" + neuron.results.join("</n>, <n>") + "</n> ]\n";
document.body.innerHTML += "Timings for " + neuron.runs + " test runs in a row.\n";
document.body.innerHTML += "Neuron accuracy: " + (neuron.correctResults / neuron.runs) * 100 + "%\n";
document.body.innerHTML += "Training: (" + neuron.passes + " passes) " + neuron.trainingTime + "ms\n";
document.body.innerHTML += "Prediction: " + neuron.predictionTime + "ms\n";
document.body.innerHTML += "\n";
document.body.innerHTML += "NET:\n";
document.body.innerHTML += "[ <n>" + net.results.join("</n>, <n>") + "</n> ]\n";
document.body.innerHTML += "Timings for " + net.runs + " test runs in a row.\n";
document.body.innerHTML += "Net accuracy: " + (net.correctResults / net.runs) * 100 + "%\n";
document.body.innerHTML += "Training: (" + net.passes + " passes) " + net.trainingTime + "ms\n";
document.body.innerHTML += "Prediction: " + net.predictionTime + "ms\n";
document.body.innerHTML += "\n";
document.body.innerHTML += "NET / WebGL2:\n";
document.body.innerHTML += "[ <n>" + netWebGL2.results.join("</n>, <n>") + "</n> ]\n";
document.body.innerHTML += "Timings for " + netWebGL2.runs + " test runs in a row.\n";
document.body.innerHTML += "Net accuracy: " + (netWebGL2.correctResults / netWebGL2.runs) * 100 + "%\n";
document.body.innerHTML += "Training: (" + netWebGL2.passes + " passes) " + netWebGL2.trainingTime + "ms\n";
document.body.innerHTML += "Prediction: " + netWebGL2.predictionTime + "ms\n";
