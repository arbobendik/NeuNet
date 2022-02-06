"use-strict";

// Test if script is running through node.js or elsewhere.
var RunsInNode = (typeof process !== 'undefined') &&
process.release.name.search(/node|io.js/) !== -1;

var neunet = {};

neunet.MathLib = {
  // Activation function a(z) = sigmoid(z).
  /*

                   1-|        ,____------
                     |    /---
                     |   /
                     | /
                 0.5-/
                   / |
                 /   |
             ___/    |
  ______----'________|____________________
       -3            0            3

  */
  sigmoid: x => 1 / (1 + Math.E ** (-x)),
  // Derivative of sigmoid function, sigmoid'(z) necessary for backpropagation.
  /*

               0.25-|-,_
               ,'   |   ',
              /     |     \
             |      |      |
            |       |       |
           /        |        \
          /         |         \
      _-'           |           '-_
  _--'______________|______________'--_
         -3         0         3

  */
  sigmoid_prime: x => neunet.MathLib.sigmoid(x) * (1 - neunet.MathLib.sigmoid(x))
}

neunet.Neuron = function (inputs) {
  var neuron = {
    learning_rate: 0.1,
    forward_propagation: (data) => {
      // Propagate forward by multiplying all inputs with their respective weights.

      //             n
      //    z = b +  E   x * w
      //            t=0   t   t
      let activity = neuron.bias;
      for (let i = 0; i < neuron.weight.length; i++) activity += neuron.weight[i] * data[i];
      // Get the neurons activity (a) by applying the activation function (sigmoid) to z.

      //    a = sigmoid(z)
      return neunet.MathLib.sigmoid(activity);
    },
    back_propagation: (data, activity, error) => {
      // Ideally the activity should be y (the control value).
      // Therefore the cost funtion needs to be minimized.

      //               2
      //    c = (a - y)

      // Or:

      //                        2
      //    c = (sigmoid(z) - y)

      // The variables we can primarily influence are the weights and the bias, which are both represented here in the form of z.
      // At the minimum of a function, the derivative is 0.
      // With gradient descent it is possible to reach the minimum of a function f(x) by repeatedly substracting the derivative of f(x) from x.
      // The influence a single step has on x is called learning rate (λ).
      // λ is a hyperparameter and therefore can be choosen freely (0.05 is often recommended for ANNs).

      //    x   = x - λ * f'(x )
      //     n+1   n          n

      // The minimum of c(x, w, b) is the minimum of c(z) alias dc_dz. So the derivative c'(z) is proportional to c'(x, w, b) and necessary to minimize c(x, w, b).

      //    c'(z) = 2 * (sigmoid(z) - y) * sigmoid'(sigmoid(z))

      // Or if we already know the value of activity:

      //    c'(z) = 2 * (a - y) * sigmoid'(a)
      let dc_dz = 2 * error * neunet.MathLib.sigmoid_prime(activity);
      // The x values of c(x, w, b) are representing the activity of neurons in the former layer.
      // Therefore the c'(x) values needed to calculate the ideal y values for those neurons are returned to the neural networks train() function.
      let dx = new Array(data.length).fill(0);
      // The derivatives of z(w) and z(x) are dependant on the respective indices of w and x.
      for (let i = 0; i < neuron.weight.length; i++) {
        //    z'(w ) = x
        //        i     i

        // And:

        //    z'(x ) = w
        //        i     i

        // The sum formula is not relevant, because all values with different indices can be summed up to one constant and are therefore not relevant for the derivative.
        // Because of the chain rule it is possible to simply multiply c'(z) with our derivatives of z
        // to get the respective derivatives of c we need for our gradient descent function.

        // dx[i] is only the change that will be made to x in the train function in the neural network.

        //    dx   = λ * z'(x ) * c'(z)
        //      i     i          i

        //    x   = x - dx
        //    n+1    n    n
        dx[i] = neuron.weight[i] * dc_dz;
        
        // So the changes for the weights can be calculated with gradient descent.

        //    w   = w - λ * z'(w ) * c'(z )
        //     i     i          i
        //     n+1   n          n        n
        neuron.weight[i] -= neuron.learning_rate * data[i] * dc_dz;
      }
      // The derivative of z with respect to the bias is 1. So applying the chainrule will result in c'(z).

      //    b   = b - λ * c'(z )
      //     n+1   n          n
      neuron.bias -= neuron.learning_rate * dc_dz;
      // Return dx values to the neural net's train function.
      return dx;
    },
    // Seperate training funciton for training the neuron without a net.
    train: (data, y) => {
      // Propagate forward to get activities.
      let activity = neuron.forward_propagation(data);
      // Calculate error.
      let error = activity - y;
      // Propagate backwards.
      neuron.back_propagation(data, activity, error);
    }
  };
  // Initialize weights and bias with random values between -1 and 1.
  neuron.bias = 2 * Math.random() - 1;
  neuron.weight = [];
  neuron.weight.length = inputs;
  for (let i = 0; i < inputs; i ++) neuron.weight[i] = 2 * Math.random() - 1;
  // Return initialized object.
  return neuron;
}

neunet.Net = function (structure) {
  var net = {
    neurons: new Array(structure.length - 1),
    // Return only final output of net.
    predict: (data) => net.forward_propagation(data)[structure.length - 1],
    // Forward propagation through all layers.
    forward_propagation: (data) => {
      var training_data = [data];
      var activities = [];
      for (let i = 0; i < net.neurons.length; i++) {
        activities = [];
        for (let j = 0; j < net.neurons[i].length; j++) {
          activities.push(net.neurons[i][j].forward_propagation(training_data[i]));
        }
        training_data = [...training_data, activities];
      }
      return training_data;
    },
    train: (data, y) => {
      // Forward propagate and save activities for backpropagation.
			var training_data = net.forward_propagation(data);
      // Delta_a is an array filled with the errors of the neurons in the current backpropagated layer.
      var delta_a = y.map((item, i) => training_data[structure.length - 1][i] - item);
      // Backpropagate, iterate through layers.
      for (let i = net.neurons.length - 1; i >= 0; i--) {
        // Create new array to accumulate the errors for the next layer to be backpropagated (net.neurons[i - 1]).
        next_delta_a = new Array(structure[i]).fill(0);
        for (let j = 0; j < net.neurons[i].length; j++) {
          // Backpropagate individual neuron.
          changes_delta_a = net.neurons[i][j].back_propagation(training_data[i], training_data[i+1][j], delta_a[j]);
          // Accumulate changes to get an estimation of what the error of the former layer might be.
          next_delta_a = changes_delta_a.map((item, i) => next_delta_a[i] + item);
        }
        // Update error list with errors of next layer.
        delta_a = next_delta_a;
      }
    }
  };
  // Initialize net structure and neurons.
  for (let i = 1; i < structure.length; i++) {
    net.neurons[i - 1] = new Array(structure[i]);
    for (let j = 0; j < structure[i]; j++) {
      net.neurons[i - 1][j] = new neunet.Neuron(structure[i - 1]);
    }
  }
  // Return initialized object.
  return net;
};

// Make objects accessible through node js's export object.
if (RunsInNode) Object.assign(exports, neunet);
