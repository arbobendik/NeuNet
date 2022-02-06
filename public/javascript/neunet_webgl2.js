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
  sigmoid_prime: x => neunet.MathLib.sigmoid(x) * (1 - neunet.MathLib.sigmoid(x)),
  // Add modulo operation, because it's commonly used in those applications.
  mod: (x, y) => x - y * Math.floor(x/y)
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

neunet.WebGl2Lib = {
  // Plain vertex shader which fills whole clip space with two vertices.
  plain_vertex: `#version 300 es
  in vec4 position;
  void main() {
    gl_Position = position;
  }`,
  set_byte_texture: (gl, array, width, height) => {
    let tex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, width, height, 0, gl.RGBA, gl.UNSIGNED_BYTE, array);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    return tex;
  },
  set_float_texture: (gl, array, width, height) => {
    let tex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 4);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.R32F, width, height, 0, gl.RED, gl.FLOAT, array);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    return tex;
  },
  compile: (gl, vertex, fragment) => {
    var shaders = [
      { source: vertex, type: gl.VERTEX_SHADER },
      { source: fragment, type: gl.FRAGMENT_SHADER }
    ];
    // Create Program, compile and append vertex and fragment shader to it.
    let program = gl.createProgram();
    // Compile GLSL shaders.
    shaders.forEach(async (item, i) => {
      let shader = gl.createShader(item.type);
      gl.shaderSource(shader, item.source);
      gl.compileShader(shader);
      // Append shader to Program if GLSL compiled successfully.
      if (gl.getShaderParameter(shader, gl.COMPILE_STATUS)){
        gl.attachShader(program, shader);
      }else{
        // Log debug info and delete shader if shader fails to compile.
        console.warn(gl.getShaderInfoLog(shader));
        gl.deleteShader(shader);
      }
    });
    gl.linkProgram(program);
    // Return program if it links successfully.
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)){
      // Log debug info and delete Program if Program fails to link.
      console.warn(gl.getProgramInfoLog(program));
      gl.deleteProgram(program);
    }else{
      return program;
    }
  }
};

// Webgl 2 accelerated Net elements
neunet.NetWebGL2 = function (structure) {
  // Use backward object to sort all webgl elements
  let forward = {};
  let backward = {};
  // Source code for compute (fragment) shader for forward pass through net.
  forward.source = `#version 300 es
  precision highp float;

  uniform sampler2D neuron_tex;
  uniform sampler2D data_tex;
  out vec4 out_color;

  // Activation function of neuron.
  float sigmoid(float x) {
    return 1.0 / (1.0 + pow(2.718281828459045, - x));
  }

  // Shader to calculate activity of a single neuron.
  void main() {
    vec2 neuron_texel = texelFetch(neuron_tex, ivec2(0, gl_FragCoord.y), 0).xy;
    // Width is always one, so row gl_FragCoord.y in neuronTex is the line of neuron[gl_FragCoord.y].
    // Initialize z with bias.
    float z = ((neuron_texel.x + neuron_texel.y / 255.0) * 2.0 - 1.0);
    // Get width of neuron_tex to get number of weights + 1
    int neuron_tex_width = textureSize(neuron_tex, 0).x;
    // Iterate over inputs and respective weights for this neuron.
    for (int i = 1; i < neuron_tex_width; i++) {
      neuron_texel = texelFetch(neuron_tex, ivec2(i, gl_FragCoord.y), 0).xy;
      vec2 data_texel = texelFetch(data_tex, ivec2(0, i - 1), 0).xy;
      // Add weight[i] * input[i] to z.
      z += ((neuron_texel.x + neuron_texel.y / 255.0) * 2.0 - 1.0) * (data_texel.x + data_texel.y / 255.0);
    }
    vec2 data_texel = texelFetch(data_tex, ivec2(0, 0), 0).xy;
    // Calculate activity.
    float activity = sigmoid(z);
    out_color = vec4(activity, mod(activity * 255.0, 1.0), vec2(0.0));
    out_color = floor(out_color * 255.0) / 255.0;
  }`;
  // Source code for compute (fragment) shader for backpropagation.
  backward.source = `#version 300 es
  precision highp float;

  uniform sampler2D neuron_tex;
  uniform sampler2D data_tex;
  uniform sampler2D activity_tex;
  uniform sampler2D error_tex;

  uniform float learning_rate;

  uniform int y_instead_of_error;

  out vec4 out_color;

  // Activation function of neuron.
  float sigmoid(float x) {
    return 1.0 / (1.0 + pow(2.718281828459045, - x));
  }

  // Derivative of activation function.
  float sigmoid_prime(float x) {
    return sigmoid(x) * (1.0 - sigmoid(x));
  }

  // Shader to calculate activity of a single neuron.
  void main() {
    // Width is always one, so row gl_FragCoord.y in neuronTex is the line of neuron[gl_FragCoord.y].
    int row = int(gl_FragCoord.y);
    int column = int(gl_FragCoord.x);

    vec2 activity_texel = texelFetch(activity_tex, ivec2(0, row), 0).xy;
    float activity = activity_texel.x + activity_texel.y / 255.0;

    vec2 error_texel = texelFetch(error_tex, ivec2(0, row), 0).xy;
    float error = error_texel.x;
    if (y_instead_of_error == 0) error = activity - error_texel.x;

    float dc_dz = 2.0 * error * sigmoid_prime(activity);

    vec2 neuron_texel = texelFetch(neuron_tex, ivec2(column, row), 0).xy;
    float weight = (neuron_texel.x + neuron_texel.y / 255.0) * 2.0 - 1.0;

    float modifier = dc_dz * learning_rate;
    float dx = weight * dc_dz;

    if (column != 0) {
      vec2 data_texel = texelFetch(data_tex, ivec2(0, column), 0).xy;
      modifier *= data_texel.x + data_texel.y / 255.0;
    }

    float updated_weight = weight - modifier;
    vec2 val = vec2(updated_weight, dx) / 2.0 + 0.5;
    out_color = vec4(val.x, mod(val.x * 255.0, 1.0), val.y, mod(val.y * 255.0, 1.0));
    out_color = floor(out_color * 255.0) / 255.0;
  }`;
  // Create webgl context necessary for hardware acceleration.
  canvas = document.createElement("canvas");
  gl = canvas.getContext("webgl2");

  // Compile plain vertex shader and forward_propagation fragment shader to program.
  forward.program = neunet.WebGl2Lib.compile(gl, neunet.WebGl2Lib.plain_vertex, forward.source);
  // Get uniform and attribbuffer locations for forward pass shader.
  forward.position_location = gl.getAttribLocation(forward.program, 'position');
  forward.neuron_tex_location = gl.getUniformLocation(forward.program, 'neuron_tex');
  forward.data_tex_location = gl.getUniformLocation(forward.program, 'data_tex');
  // Create buffer to provide two vertices to vertex shader.
  forward.vertex_buffer = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, forward.vertex_buffer);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1,-1,1,-1,-1,1,-1,1,1,-1,1,1]), gl.STATIC_DRAW);
  // Create vertex array object.
  forward.vao = gl.createVertexArray();
  gl.bindVertexArray(forward.vao);
  // Tell WebGl how to draw vertices.
  gl.enableVertexAttribArray(forward.position_location);
  gl.vertexAttribPointer(forward.position_location, 2, gl.FLOAT, false, 0, 0);

  // Compile plain vertex shader and training / backward fragment shader to program.
  backward.program = neunet.WebGl2Lib.compile(gl, neunet.WebGl2Lib.plain_vertex, backward.source);

  // Get uniform and attribbuffer locations for trainings pass shader.
  backward.position_location = gl.getAttribLocation(backward.program, 'position');
  backward.neuron_tex_location = gl.getUniformLocation(backward.program, 'neuron_tex');
  backward.data_tex_location = gl.getUniformLocation(backward.program, 'data_tex');
  backward.activity_tex_location = gl.getUniformLocation(backward.program, 'activity_tex');
  backward.error_tex_location = gl.getUniformLocation(backward.program, 'error_tex');

  backward.learning_rate_location = gl.getUniformLocation(backward.program, 'learning_rate');
  backward.y_instead_of_error_location = gl.getUniformLocation(backward.program, 'y_instead_of_error');
  // Create buffer to provide two vertices to vertex shader.
  backward.vertex_buffer = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, backward.vertex_buffer);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1,-1,1,-1,-1,1,-1,1,1,-1,1,1]), gl.STATIC_DRAW);
  // Create vertex array object.
  backward.vao = gl.createVertexArray();
  gl.bindVertexArray(backward.vao);
  // Tell WebGl how to draw vertices.
  gl.enableVertexAttribArray(backward.position_location);
  gl.vertexAttribPointer(backward.position_location, 2, gl.FLOAT, false, 0, 0);

  var framebuffer = gl.createFramebuffer();

  var net = {
    training_textures: [],
    layer_textures: [],
    temp_layer_textures: [],
    error_texture: gl.createTexture(),

    learning_rate: 0.05,
    neurons: new Array(structure.length - 1),
    // Return only final output of net.
    predict: (data) => {
      net.forward_propagation_tex(data)
      return net.forward_propagation(data)[structure.length - 1];
    },
    // Forward propagation with numbers as output.
    forward_propagation: (data) => {
      var training_data = [data];
      var activities = [];
      for (let i = 0; i < net.neurons.length; i++) {
        activities = [];
        for (let j = 0; j < net.neurons[i].length; j+= structure[i] + 1) {
          // Initialize activity with bias.
          let activity = net.neurons[i][j];
          // Get the neurons activity (a) by applying the activation function (sigmoid) to z.
          for (let k = 0; k < structure[i]; k++) activity += net.neurons[i][j + k + 1] * training_data[i][k];
          // a = sigmoid(z)
          activities.push(neunet.MathLib.sigmoid(activity));
        }
        training_data.push(activities);
      }
      return training_data;
    },

    train_cpu: async (data, y) => {
      // Forward propagate and save activities for backpropagation.
			var training_data = net.forward_propagation(data);
      // Delta_a is an array filled with the errors of the neurons in the current backpropagated layer.
      var delta_a = y.map((item, i) => training_data[structure.length - 1][i] - item);
      // Backpropagate, iterate through layers.
      for (let i = net.neurons.length - 1; i >= 0; i--) {
        // Create new array to accumulate the errors for the next layer to be backpropagated (net.neurons[i - 1]).
        next_delta_a = new Array(structure[i]).fill(0);
        for (let j = 0; j < net.neurons[i].length / (structure[i] + 1); j++) {
          let pos_in_array = j * (structure[i] + 1)
          //    c'(z) = 2 * (a - y) * sigmoid'(a)
          let dc_dz = 2 * delta_a[j] * neunet.MathLib.sigmoid_prime(training_data[i + 1][j]);
          //    b   = b - λ * c'(z )
          //     n+1   n          n
          net.neurons[i][pos_in_array] -= net.learning_rate * dc_dz;
          for (let k = 1; k < structure[i] + 1; k++) {
            let data_n = training_data[i][k - 1];
            //    dx   = λ * z'(x ) * c'(z)
            //      i     i          i

            //    x   = x - dx
            //    n+1    n    n
            next_delta_a[k - 1] += dc_dz * net.neurons[i][pos_in_array + k];
            //    w   = w - λ * z'(w ) * c'(z )
            //     i     i          i
            //     n+1   n          n        n
            net.neurons[i][pos_in_array + k] -= net.learning_rate * training_data[i][k - 1] * dc_dz;
          }
        }
        // Update error list with errors of next layer.
        delta_a = next_delta_a;
      }
    },

    // Forward propagation with texture array for backpropagation as output.
    forward_propagation_tex: (data) => {
      // Generate new Float32 array from data for shader.
      var tex_data = new Float32Array(data);

      gl.bindTexture(gl.TEXTURE_2D, net.training_textures[0]);
      gl.pixelStorei(gl.UNPACK_ALIGNMENT, 4);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.R32F, 1, data.length, 0, gl.RED, gl.FLOAT, tex_data);
      // Tell webgl which program to use.
      gl.useProgram(forward.program);
      gl.bindVertexArray(forward.vao);
      // Set width to 1, because only one output (activity) shall be calculated per neuron.
      canvas.width = 1;

      var training_data = [data];
      // Iterate over layers and render directly to training_textures array.
      for (var i = 0; i < net.neurons.length; i++) {
        canvas.height = structure[i + 1];
        gl.viewport(0, 0, gl.canvas.width, gl.canvas.height);
        // Tell program which webgl texture slot to use for which texture.
        gl.activeTexture(gl.TEXTURE0);
        // Convert to and set this layer as texture for shader.
        gl.bindTexture(gl.TEXTURE_2D, net.layer_textures[i]);
        gl.activeTexture(gl.TEXTURE1);
        // Set training_data as data texture.
        gl.bindTexture(gl.TEXTURE_2D, net.training_textures[i]);
        // Link variables in shader with texture slots.
        gl.uniform1i(forward.neuron_tex_location, 0);
        gl.uniform1i(forward.data_tex_location, 1);
        // Drawcall.
        gl.bindBuffer(gl.ARRAY_BUFFER, forward.vertex_buffer);
        gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1,-1,1,-1,-1,1,-1,1,1,-1,1,1]), gl.STATIC_DRAW);

        // Set framebuffer.
        gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer);
        // Configure framebuffer for color and depth.
        gl.drawBuffers([gl.COLOR_ATTACHMENT0]);
        gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, net.training_textures[i + 1], 0);

        gl.drawArrays(gl.TRIANGLES, 0, 6);
      }
    },

    train_gpu: async (data, y) => {
      // Forward propagate and save activities for backpropagation.
      net.forward_propagation_tex(data);
      // Backpropagate, iterate through layers.
      var delta_a = [...y];
      // Tell webgl which program to use.
      gl.useProgram(backward.program);
      gl.bindVertexArray(backward.vao);

      for (let i = net.neurons.length - 1; i >= 0; i--) {
        // Rescale canvas.
        canvas.width = structure[i] + 1;
        canvas.height = structure[i + 1];
        gl.viewport(0, 0, gl.canvas.width, gl.canvas.height);
        // Generate new error texture from delta_a.
        gl.bindTexture(gl.TEXTURE_2D, net.error_texture);
        gl.pixelStorei(gl.UNPACK_ALIGNMENT, 4);
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.R32F, 1, delta_a.length, 0, gl.RED, gl.FLOAT, new Float32Array(delta_a));

        // Reset framebuffer.
        gl.bindTexture(gl.TEXTURE_2D, net.temp_layer_textures[i]);
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, canvas.width, canvas.height, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
        // Set framebuffer.
        gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer);
        // Configure framebuffer for color and depth.
        gl.drawBuffers([gl.COLOR_ATTACHMENT0]);
        gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, net.temp_layer_textures[i], 0);
        // Tell program which webgl texture slot to use for which texture.
        // Convert to and set this layer as texture for shader.
        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_2D, net.layer_textures[i]);
        // Set training_data as data texture.
        gl.activeTexture(gl.TEXTURE1);
        gl.bindTexture(gl.TEXTURE_2D, net.training_textures[i]);
        // Set activities of this layer as texture.
        gl.activeTexture(gl.TEXTURE2);
        gl.bindTexture(gl.TEXTURE_2D, net.training_textures[i + 1]);

        gl.activeTexture(gl.TEXTURE3);
        gl.bindTexture(gl.TEXTURE_2D, net.error_texture);

        // Link variables in shader with texture slots.
        gl.uniform1i(backward.neuron_tex_location, 0);
        gl.uniform1i(backward.data_tex_location, 1);
        gl.uniform1i(backward.activity_tex_location, 2);
        gl.uniform1i(backward.error_tex_location, 3);

        gl.uniform1f(backward.learning_rate_location, net.learning_rate);

        // Tell shader to interprete error texture as ys instead of errors for the first run.
        gl.uniform1i(backward.y_instead_of_error_location, i === net.neurons.length - 1 ? 0 : 1);
        // Drawcall.
        gl.bindBuffer(gl.ARRAY_BUFFER, backward.vertex_buffer);
        gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1,-1,1,-1,-1,1,-1,1,1,-1,1,1]), gl.STATIC_DRAW);

        gl.drawArrays(gl.TRIANGLES, 0, 6);

        // Switch framebuffer texture with texture in main array to update values without allocating new RAM / VRAM.
        var temp = net.layer_textures[i];
        net.layer_textures[i] = net.temp_layer_textures[i];
        net.temp_layer_textures[i] = temp;

        var results = new Uint8Array(net.neurons[i].length * 4);
        gl.readPixels(0, 0, canvas.width, canvas.height, gl.RGBA, gl.UNSIGNED_BYTE, results);

        delta_a = new Array(structure[i]).fill(0);
        for (let j = 0; j < net.neurons[i].length * 4; j+=4) {
          // Apply dx values to neural layer texture.
          net.neurons[i][j / 4] = (results[j] + results[j + 1] / 255) / 127.5 - 1;

          let column = (j / 4) % (structure[i] + 1);
          if (column != 0) delta_a[column - 1] += (results[j + 2] + results[j + 3] / 255) / 127.5 - 1;
        }
      }
    },

    load_training: () => {
      for (let i = 0; i < net.neurons.length; i++) {
        let tex_array = new Uint8Array(net.neurons[i].length * 4).fill(0);
        // Iterate over the net's neural layer arrays and convert them to unisgned byte testures.
        for (let j = 0; j < net.neurons[i].length; j++) {
          // Convert numbers in array to only positive numbers between 0 and 1.
          let num = (net.neurons[i][j] + 1) * 127.5;
          // Keep positive range and convert later in the shader, because the unsigned byte format can only hold positive numbers.
          // Increase precision by using a second 8-bit number for more detail.
          // This adds up to a total bit-depth of 15 plus, one bit for prefix.
          tex_array[j * 4] = num;
          tex_array[j * 4 + 1] = neunet.MathLib.mod(num, 1) * 255;
        }
        // Prepare neurons attributes as texture for GPU.
        gl.bindTexture(gl.TEXTURE_2D, net.layer_textures[i]);
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, structure[i] + 1, structure[i + 1], 0, gl.RGBA, gl.UNSIGNED_BYTE, tex_array);
      }
    }
  };

  // Initialize net structure and neurons.
  net.training_textures[0] = neunet.WebGl2Lib.set_float_texture(gl, null, 1, structure[0]);
  // Iterate over layers.
  for (let i = 0; i < structure.length - 1; i++) {
    // Create a Float32Array for each layer which is easily convertible to a texture for the shader later.

    // The array contains all informations about the neurons in this layer and is structured like this:

    // neuron0:   bias, w0, w1, w2, w3, w4
    // neuron1:   bias, w0, w1, w2, w3, w4
    // neuron2:   bias, w0, w1, w2, w3, w4
    // neuron3:   bias, w0, w1, w2, w3, w4

    // structure[i + 1] ==> neurons in current layer
    // structure[i] ==> weights per neuron in this layer
    // 1 ==> the bias value for each neuron
    net.neurons[i] = new Array(structure[i + 1] * (structure[i] + 1));
    // Create same array encoded in 8-bit unsigned ints for later use as a texture.
    // An unsigned byte Texture is more fitting here than any other type, because it is renderable,
    // so the program doesn't need to make a difference between reuse of a texture it has been rendered to before
    // and a new texture created from data points.
    let tex_array = new Uint8Array(net.neurons[i].length * 4).fill(0);
    // Fill array with random values between -1 and 1 to Initialize all biases and weights.
    for (let j = 0; j < net.neurons[i].length; j++) {
      // Stretch rand to a range between -1 and 1.
      net.neurons[i][j] = (2 * Math.random() - 1);
      // Keep positive range and convert later in the shader, because the unsigned byte format can only hold positive numbers.
      let num = (net.neurons[i][j] + 1) * 127.5;
      // Keep positive range and convert later in the shader, because the unsigned byte format can only hold positive numbers.
      // Increase precision by using a second 8-bit number for more detail.
      // This adds up to a total bit-depth of 15 plus, one bit for prefix.
      tex_array[j * 4] = num;
      tex_array[j * 4 + 1] = neunet.MathLib.mod(num, 1) * 255;
    }

    net.training_textures.push(neunet.WebGl2Lib.set_byte_texture(gl, null, 1, structure[i + 1]));
    // Prepare neurons attributes as texture for GPU.
    net.layer_textures.push(neunet.WebGl2Lib.set_byte_texture(gl, tex_array, structure[i] + 1, structure[i + 1]));
    // Prepare second renderable texture for GPU.
    net.temp_layer_textures.push(neunet.WebGl2Lib.set_byte_texture(gl, null, structure[i] + 1, structure[i + 1]));
  }

  net.error_texture = neunet.WebGl2Lib.set_float_texture(gl, null, gl.MAX_TEXTURE_SIZE, 1);

  // Return initialized object.
  return net;
}

// Make objects accessible through node js's export object.
if (RunsInNode) Object.assign(exports, neunet);
