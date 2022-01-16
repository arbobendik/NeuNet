"use-strict";

var neunet = {};

neunet.WebGl2Lib = {
  // Plain vertex shader which fills whole clip space with two vertices.
  plain_vertex: `#version 300 es
  in vec4 position;
  void main() {
    gl_Position = position;
  }`,
  set_data_texture: (gl, array, width, height) => {
    let tex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.R32F, width, height, 0, gl.RED, gl.FLOAT, new Float32Array(array));
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
neunet.Net = function (structure) {
  // Use forward / backward object to sort all webgl elements
  let [forward, backward] = [{}, {}];
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
    // Width is always one, so row gl_FragCoord.y in neuronTex is the line of neuron[gl_FragCoord.y].
    // Initialize z with bias.
    float z = texelFetch(neuron_tex, ivec2(0, gl_FragCoord.y), 0).x;
    // Get width of neuron_tex to get number of weights + 1
    int neuron_tex_width = textureSize(neuron_tex, 0).x;
    // Iterate over inputs and respective weights for this neuron.
    for (int i = 1; i < neuron_tex_width; i++) {
      // Add weight[i] * input[i] to z.
      z += texelFetch(neuron_tex, ivec2(i, gl_FragCoord.y), 0).x *  texelFetch(data_tex, ivec2(i - 1, 0), 0).x;
    }
    // Calculate activity.
    float activity = sigmoid(z);
    out_color = vec4(activity, vec3(0.0));
  }`;
  // Source code for compute (fragment) shader for backpropagation.
  backward.source = `#version 300 es
  precision highp float;

  uniform sampler2D neuron_tex;
  uniform sampler2D data_tex;
  uniform sampler2D activity_tex;
  uniform sampler2D error_tex;

  uniform float learning_rate;

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

    float data_point = texelFetch(data_tex, ivec2(column, 0), 0).x;
    float activity = texelFetch(activity_tex, ivec2(0, row), 0).x;
    float error = texelFetch(error_tex, ivec2(0, row), 0).x;

    float dc_dz = 2.0 * error * sigmoid_prime(activity);

    float bias_or_weight = texelFetch(neuron_tex, ivec2(column, row), 0).x;

    float modifier = dc_dz * learning_rate;
    float dx = bias_or_weight * dc_dz;

    if (column != 0) modifier *= data_point;

    out_color = vec4(sign(modifier), abs(modifier), sign(dx), abs(dx));
  }`;
  // Create webgl context necessary for hardware acceleration.
  canvas = document.createElement("canvas");
  gl = canvas.getContext("webgl2");
  document.body.appendChild(canvas);
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

  var net = {
    learning_rate: 0.1,
    neurons: new Array(structure.length - 1),
    // Return only final output of net.
    predict: (data) => net.forward_propagation(data)[structure.length - 1],
    forward_propagation: (data) => {
      // Tell webgl which program to use.
      gl.useProgram(forward.program);
      gl.bindVertexArray(forward.vao);
      // Set width to 1, because only one output (activity) shall be calculated per neuron.
      canvas.width = 1;
      // Clone data, that operations in this function don't affect the original array.
      let training_data = [data];
      for (let i = 0; i < net.neurons.length; i++) {
        canvas.height = structure[i + 1];
        gl.viewport(0, 0, gl.canvas.width, gl.canvas.height);
        // Tell program which webgl texture slot to use for which texture.
        gl.activeTexture(gl.TEXTURE0);
        // Convert to and set this layer as texture for shader.
        gl.bindTexture(gl.TEXTURE_2D, neunet.WebGl2Lib.set_data_texture(gl, net.neurons[i], structure[i] + 1, structure[i + 1]));
        gl.activeTexture(gl.TEXTURE1);
        // Set training_data as data texture.
        gl.bindTexture(gl.TEXTURE_2D, neunet.WebGl2Lib.set_data_texture(gl, training_data[i], training_data[i].length, 1));
        // Link variables in shader with texture slots.
        gl.uniform1i(forward.neuron_tex_location, 0);
        gl.uniform1i(forward.data_tex_location, 1);
        // Drawcall.
        gl.bindBuffer(gl.ARRAY_BUFFER, forward.vertex_buffer);
        gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1,-1,1,-1,-1,1,-1,1,1,-1,1,1]), gl.STATIC_DRAW);
        gl.drawArrays(gl.TRIANGLES, 0, 6);

        let results = new Uint8Array(structure[i + 1] * 4);
        gl.readPixels(0, 0, 1, structure[i + 1], gl.RGBA, gl.UNSIGNED_BYTE, results);
        let activities = [];
        for (let j = 0; j < structure[i + 1] * 4; j+=4) activities = [...activities, results[j] / 255];
        training_data = [...training_data, activities];
      }
      return training_data;
    },
    train: async (data, y) => {
      // Forward propagate and save activities for backpropagation.
			var training_data = net.forward_propagation(data);
      // console.log("training_data", training_data);
      // Backpropagate, iterate through layers.
      var delta_a = y.map((item, i) => training_data[structure.length - 1][i] - item);

      // Tell webgl which program to use.
      gl.useProgram(backward.program);
      gl.bindVertexArray(backward.vao);

      // console.log(training_data);
      // console.log(structure.length - net.neurons.length);
      // console.log(net.neurons.length);

      for (let i = net.neurons.length - 1; i >= 0; i--) {
        // Prepare neurons attributes as texture for GPU.

        canvas.height = structure[i + 1];
        canvas.width = structure[i] + 1;
        // console.log("canvas", canvas.width, canvas.height);
        gl.viewport(0, 0, gl.canvas.width, gl.canvas.height);
        // Tell program which webgl texture slot to use for which texture.
        // Convert to and set this layer as texture for shader.
        // console.log("neuron_tex", structure[i] + 1, structure[i + 1], net.neurons[i]);
        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_2D, neunet.WebGl2Lib.set_data_texture(gl, net.neurons[i], structure[i] + 1, structure[i + 1]));
        // Set training_data as data texture.
        // console.log("data", training_data[i].length + 1, 1, [1, ...training_data[i]]);
        gl.activeTexture(gl.TEXTURE1);
        gl.bindTexture(gl.TEXTURE_2D, neunet.WebGl2Lib.set_data_texture(gl, [1, ...training_data[i]], training_data[i].length + 1, 1));
        // Set activities of this layer as texture.
        // console.log("activities", 1, training_data[i + 1].length, training_data[i + 1]);
        gl.activeTexture(gl.TEXTURE2);
        gl.bindTexture(gl.TEXTURE_2D, neunet.WebGl2Lib.set_data_texture(gl, training_data[i + 1], 1, training_data[i + 1].length));

        // console.log("error", 1, delta_a.length, delta_a,);
        gl.activeTexture(gl.TEXTURE3);
        gl.bindTexture(gl.TEXTURE_2D, neunet.WebGl2Lib.set_data_texture(gl, delta_a, 1, delta_a.length));

        // Link variables in shader with texture slots.
        gl.uniform1i(backward.neuron_tex_location, 0);
        gl.uniform1i(backward.data_tex_location, 1);
        gl.uniform1i(backward.activity_tex_location, 2);
        gl.uniform1i(backward.error_tex_location, 3);

        gl.uniform1f(backward.learning_rate_location, net.learning_rate);
        // Drawcall.
        gl.bindBuffer(gl.ARRAY_BUFFER, backward.vertex_buffer);
        gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1,-1,1,-1,-1,1,-1,1,1,-1,1,1]), gl.STATIC_DRAW);
        gl.drawArrays(gl.TRIANGLES, 0, 6);

        let results = new Uint8Array(net.neurons[i].length * 4);
        // console.log(net.neurons[i]);
        gl.readPixels(0, 0, canvas.width, canvas.height, gl.RGBA, gl.UNSIGNED_BYTE, results);
        // console.log(results);
        let modifiers = [];
        // console.log(net.neurons[i]);
        // console.log(structure[i]);
        delta_a = new Array(structure[i]).fill(0);
        for (let j = 0; j < net.neurons[i].length * 4; j+=4) {
          // console.log((results[j] / 255 * 2 - 1) * results[j + 1] / 255);
          net.neurons[i][j / 4] -= (results[j] / 255 * 2 - 1) * results[j + 1] / 255;

          let index = j % (structure[i] * 4);
          if (index != 0) delta_a[index - 1] += (results[j + 2] / 255 * 2 - 1) * results[j + 3] / 255;
        }

        // Apply dx values to neural layer texture.



        /*next_delta_a = new Array(structure[i]).fill(0);
        for (let j = 0; j < net.neurons[i].length; j++) {

          changes_delta_a = net.neurons[i][j].back_propagation(training_data[i], training_data[i+1][j], delta_a[j]);
          next_delta_a = changes_delta_a.map((item, i) => next_delta_a[i] + item);
        }
        delta_a = next_delta_a;*/
      }
    }
  };
  // Initialize net structure and neurons.
  // Iterate over layers.
  for (let i = 1; i < structure.length; i++) {
    // Create a Float32Array for each layer which is easily convertible to a texture for the shader later.

    // The array contains all informations about the neurons in this layer and is structured like this:

    // neuron0:   bias, w0, w1, w2, w3, w4
    // neuron1:   bias, w0, w1, w2, w3, w4
    // neuron2:   bias, w0, w1, w2, w3, w4
    // neuron3:   bias, w0, w1, w2, w3, w4

    // structure[i] ==> neurons in current layer
    // structure[i - 1] ==> weights per neuron in this layer
    // 1 ==> the bias value for each neuron
    net.neurons[i - 1] = new Float32Array(structure[i] * (structure[i - 1] + 1));
    // Fill array with random values between -1 and 1 to Initialize all biases and weights.
    for (let j = 0; j < net.neurons[i - 1].length; j++) {
      net.neurons[i - 1][j] = 2 * Math.random() - 1;
    }
  }
  // Return initialized object.
  return net;
}
