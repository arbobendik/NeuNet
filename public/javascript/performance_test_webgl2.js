let this_input_data = [[0, 1, 2], [1, 0, -1], [0, 1, 2], [2, 1, 0], [1, 2, 3], [3, 2, 1], [2, 3, 4], [4, 3, 2]];
let this_given_points = [0, 1, 0, 1, 0, 1, 0, 1];

let predict_for = [[10, 5, 1], [0, 5, 10], [0, 5, 6], [10, 5, 0], [4, 7, 10], [-1, -2, -3]];
let correct_predictions = [1, 0, 0, 1, 0, 1];

let neuron_test_runs = 4;
let net_test_runs = 1;
let net_webgl2_runs = 1;

let neuron_passes = 10;
let net_passes = 5;
let net_webgl2_passes = 5;

let neuron_results = [];
let net_results = [];
let net_webgl2_results = [];

let correct_neuron = 0;
let correct_net = 0;
let correct_net_webgl2 = 0;

let neuron_training_time = 0;
let net_training_time = 0;
let net_webgl2_training_time = 0;

let neuron_prediction_time = 0;
let net_prediction_time = 0;
let net_webgl2_prediction_time = 0;

let net_structure = [3, 1000, 1000, 1];
let net_structure_webgl2 = [3, 1000, 1000, 1];


for (let r = 0; r < neuron_test_runs; r++) {
  let neuron = new neunet.Neuron(3);

  let t0 = performance.now();
  for (let p = 0; p < neuron_passes; p++) {
    this_input_data.forEach((item, i) => {
      neuron.train(item, this_given_points[i]);
    });
  }

  let t1 = performance.now();
  neuron_results = [];
  predict_for.forEach((item, i) => {
    neuron_results.push(Math.round(neuron.forward_propagation(item)));
  });

  let t2 = performance.now();
  neuron_training_time += t1 - t0;
  neuron_prediction_time += t2 - t1;
  if (neuron_results.reduce((result, item, i) => (item == correct_predictions[i]) && result)) correct_neuron ++;
}

for (let r = 0; r < net_test_runs; r++) {
  let net = new neunet.Net(net_structure);

  let t0 = performance.now();
  for (let p = 0; p < net_passes; p++) {
    this_input_data.forEach((item, i) => {
      net.train(item, [this_given_points[i]]);
    });
  }
  let t1 = performance.now();
  net_results = [];
  predict_for.forEach((item, i) => {
    net_results.push(Math.round(net.predict(item)));
  });

  let t2 = performance.now();
  net_training_time += t1 - t0;
  net_prediction_time += t2 - t1;
  if (net_results.reduce((result, item, i) => (item == correct_predictions[i]) && result)) correct_net ++;
}

for (let r = 0; r < net_webgl2_runs; r++) {
  let net = new neunet.NetWebGL2(net_structure_webgl2);

  let t0 = performance.now();
  for (let p = 0; p < net_webgl2_passes; p++) {
    this_input_data.forEach((item, i) => {
      net.train_gpu(item, [this_given_points[i]]);
    });
  }
  let t1 = performance.now();
  net_webgl2_results = [];
  predict_for.forEach(async (item, i) => {
    net_webgl2_results.push(Math.round(net.predict(item)));
  });

  let t2 = performance.now();
  net_webgl2_training_time += t1 - t0;
  net_webgl2_prediction_time += t2 - t1;
  if (net_webgl2_results.reduce((result, item, i) => (item == correct_predictions[i]) && result)) correct_net_webgl2 ++;
}

console.log();
console.log("NEURON:");
console.log(neuron_results);
console.log("Timings for " + neuron_test_runs + " test runs in a row.")
console.log("Neuron accuracy: " + (correct_neuron / neuron_test_runs) * 100 + "%");
console.log("Training: (" + neuron_passes + " passes) " + neuron_training_time + "ms");
console.log("Prediction: " + neuron_prediction_time + "ms");
console.log();
console.log("NET:");
console.log(net_results);
console.log("Timings for " + net_test_runs + " test runs in a row.")
console.log("Net accuracy: " + (correct_net / net_test_runs) * 100 + "%");
console.log("Training: (" + net_passes + " passes) " + net_training_time + "ms");
console.log("Prediction: " + net_prediction_time + "ms");
console.log();
console.log("NET / WebGL2:");
console.log(net_webgl2_results);
console.log("Timings for " + net_webgl2_runs + " test runs in a row.")
console.log("Net accuracy: " + (correct_net_webgl2 / net_webgl2_runs) * 100 + "%");
console.log("Training: (" + net_webgl2_passes + " passes) " + net_webgl2_training_time + "ms");
console.log("Prediction: " + net_webgl2_prediction_time + "ms");
console.log();
