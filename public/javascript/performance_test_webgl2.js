let this_input_data = [[0, 1, 2], [1, 0, -1], [0, 1, 2], [2, 1, 0], [1, 2, 3], [3, 2, 1], [2, 3, 4], [4, 3, 2]];
let this_given_points = [0, 1, 0, 1, 0, 1, 0, 1];

let predict_for = [[100, 50, 10], [0, 5, 10], [0, 5, 6], [10, 5, 0], [4, 7, 10], [-1, -2, -3]];
let correct_predictions = [1, 0, 0, 1, 0, 1];

let net_webgl2_runs = 1;
let net_webgl2_passes = 20;
let net_webgl2_results = [];
let correct_net_webgl2 = 0;
let net_webgl2_training_time = 0;
let net_webgl2_prediction_time = 0;

for (let r = 0; r < net_webgl2_runs; r++) {
  let net = new neunet.Net([3, 4000, 2000, 1]);

  let t0 = performance.now();
  for (let p = 0; p < net_webgl2_passes; p++) {
    this_input_data.forEach((item, i) => {
      net.train(item, [this_given_points[i]]);
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
console.log("NET / WebGL2:");
console.log(net_webgl2_results);
console.log("Timings for " + net_webgl2_runs + " test runs in a row.")
console.log("Net accuracy: " + (correct_net_webgl2 / net_webgl2_runs) * 100 + "%");
console.log("Training: (" + net_webgl2_passes + " passes) " + net_webgl2_training_time + "ms");
console.log("Prediction: " + net_webgl2_prediction_time + "ms");
