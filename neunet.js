"use-strict";

import { GLLib } from './gllib.js';

export class Net {
  // Create webgl context necessary for hardware acceleration.
  canvas = document.createElement("canvas");
  gl = this.canvas.getContext("webgl2");

  forward = {};
  backward = {};
  sumError = {};
  save = {};

  trainingTextures = [];
  layerTextures = [];
  tempLayerTextures = [];

  errorSumTexture = this.gl.createTexture();
  errorTexture = this.gl.createTexture();
  learningRate = 0.01;
  
  neurons;
  structure;
  activationStructure;

  constructor (structure, activationStructure = [... new Array(structure.length - 1).fill('leakyRelu'), 'linear'], normalize = [... new Array(structure.length - 2).fill(true), false]) {

    this.canvas.viewport = {};
    // Use maximum needed canvas size and regulate rendered pixels using viewport
    let maxLayerSize = structure.reduce((p, c) => Math.max(p, c), 0);
    this.canvas.width = maxLayerSize + 1;
    this.canvas.height = maxLayerSize;

    this.structure = structure;
    this.normalize = normalize;
    this.activationStructure = activationStructure;
    this.neurons = new Array(structure.length - 1);
    // Source code for compute (fragment) shader for forward pass through net.
    this.forward.source = `#version 300 es
    precision highp float;
    precision highp int;

    uniform sampler2D neuron_tex;
    uniform sampler2D data_tex;

    uniform int is_sigmoid;
    uniform int is_tanh;
    uniform int is_leaky_relu;
    uniform int is_linear;

    out vec4 activity_out;

    // Convert 4 bytes, texture channels to usable float.
    float to_float(vec4 bytes) {
      ivec4 intBytes = ivec4(bytes * 255.0);
      uint intValue = uint(intBytes.x) | (uint(intBytes.y) << 8) | (uint(intBytes.z) << 16) | (uint(intBytes.w) << 24);
      return uintBitsToFloat(intValue);
    }

    // Split float into 4 8-bit texture channels.
    vec4 to_bytes(float num) {
      uint intValue = floatBitsToUint(num);
      uint byteMask = uint(255);
      vec4 bytes;
      bytes.x = float(intValue & byteMask);
      bytes.y = float((intValue >> 8) & byteMask);
      bytes.z = float((intValue >> 16) & byteMask);
      bytes.w = float((intValue>> 24) & byteMask);
      return round(bytes) / 255.0;
    }

    // Shader to calculate activity of a single neuron.
    void main() {
      // Width is always one, so row gl_FragCoord.y in neuronTex is the line of neuron[gl_FragCoord.y].
      // Initialize z with bias.
      float z = to_float(texelFetch(neuron_tex, ivec2(0, gl_FragCoord.y), 0));
      // Get width of neuron_tex to get number of weights + 1
      int neuron_tex_width = textureSize(neuron_tex, 0).x;
      // Normalize inputs for linear and leaky relu activation functions
      // Iterate over inputs and respective weights for this neuron.
      for (int i = 1; i < neuron_tex_width; i++) {
        float weight = to_float(texelFetch(neuron_tex, ivec2(i, gl_FragCoord.y), 0));
        float data = to_float(texelFetch(data_tex, ivec2(0, i - 1), 0));
        // Add weight[i] * input[i] to z.
        z += weight * data;
      }
      // Calculate activity.
      // Split activity into four bytes.
      if (is_tanh == 1) activity_out = to_bytes(tanh(z));
      if (is_sigmoid == 1) activity_out = to_bytes(1.0 / (1.0 + pow(2.718281828459045, - z)));
      if (is_leaky_relu == 1) activity_out = to_bytes(max(0.01 * z, z));
      if (is_linear == 1) activity_out = to_bytes(z);
    }`;
    // Source code for compute (fragment) shader for backpropagation.
    this.backward.source = `#version 300 es
    precision highp float;
    precision highp int;

    uniform sampler2D neuron_tex;
    uniform sampler2D data_tex;
    uniform sampler2D activity_tex;
    uniform sampler2D error_tex;

    uniform float learning_rate;

    uniform int y_instead_of_error;

    uniform int is_sigmoid;
    uniform int is_tanh;
    uniform int is_leaky_relu;
    uniform int is_linear;

    layout(location = 0) out vec4 neuron_out;
    layout(location = 1) out vec4 error_out;

    // Convert 4 bytes, texture channels to usable float.
    float to_float(vec4 bytes) {
      ivec4 intBytes = ivec4(bytes * 255.0);
      uint intValue = uint(intBytes.x) | (uint(intBytes.y) << 8) | (uint(intBytes.z) << 16) | (uint(intBytes.w) << 24);
      return uintBitsToFloat(intValue);
    }

    // Split float into 4 8-bit texture channels.
    vec4 to_bytes(float num) {
      uint intValue = floatBitsToUint(num);
      uint byteMask = uint(255);
      vec4 bytes;
      bytes.x = float(intValue & byteMask);
      bytes.y = float((intValue >> 8) & byteMask);
      bytes.z = float((intValue >> 16) & byteMask);
      bytes.w = float((intValue>> 24) & byteMask);
      return round(bytes) / 255.0;
    }

    // Shader to calculate activity of a single neuron.
    void main() {
      // Width is always one, so row gl_FragCoord.y in neuronTex is the line of neuron[gl_FragCoord.y].
      int row = int(gl_FragCoord.y);
      int column = int(gl_FragCoord.x);

      float activity = to_float(texelFetch(activity_tex, ivec2(0, row), 0));

      float error = to_float(texelFetch(error_tex, ivec2(row, 0), 0));
      if (y_instead_of_error == 0) error = activity - error;

      float dc_dz = 2.0 * error;
      if (is_leaky_relu == 1) dc_dz *= sign(activity) * 0.495 + 0.505;
      if (is_tanh == 1) dc_dz *= 1.0 - tanh(activity) * tanh(activity);
      if (is_sigmoid == 1)  {
        float sigmoid_activity = 1.0 / (1.0 + pow(2.718281828459045, - activity));
        dc_dz *= sigmoid_activity * (1.0 - sigmoid_activity);
      }

      float weight = to_float(texelFetch(neuron_tex, ivec2(column, row), 0));

      float modifier = dc_dz * learning_rate;
      float dx = weight * dc_dz;

      if (column != 0) {
        float data = to_float(texelFetch(data_tex, ivec2(0, column - 1), 0));
        modifier *= data;
      }

      neuron_out = to_bytes(weight - modifier);
      error_out = to_bytes(dx);
    }`;
    this.sumError.source = `#version 300 es
    precision highp float;
    precision highp int;

    uniform sampler2D error_sum_tex;

    out vec4 error_out;

    // Convert 4 bytes, texture channels to usable float.
    float to_float(vec4 bytes) {
      ivec4 intBytes = ivec4(bytes * 255.0);
      uint intValue = uint(intBytes.x) | (uint(intBytes.y) << 8) | (uint(intBytes.z) << 16) | (uint(intBytes.w) << 24);
      return uintBitsToFloat(intValue);
    }

    // Split float into 4 8-bit texture channels.
    vec4 to_bytes(float num) {
      uint intValue = floatBitsToUint(num);
      uint byteMask = uint(255);
      vec4 bytes;
      bytes.x = float(intValue & byteMask);
      bytes.y = float((intValue >> 8) & byteMask);
      bytes.z = float((intValue >> 16) & byteMask);
      bytes.w = float((intValue>> 24) & byteMask);
      return round(bytes) / 255.0;
    }

    // Sum all errors of one.
    void main() {
      // Width is always one, so row gl_FragCoord.y in neuronTex is the line of neuron[gl_FragCoord.y].
      int row = int(gl_FragCoord.y);
      int column = int(gl_FragCoord.x);

      float sum = 0.0;

      for (int i = 0; i < textureSize(error_sum_tex, 0).y; i++) {
        vec4 error_texel = texelFetch(error_sum_tex, ivec2(column + 1, i), 0);
        // Sum up all values.
        sum += to_float(error_texel);
      }

      error_out = to_bytes(sum);
    }`;
    this.save.source = `#version 300 es
    precision highp float;
    precision highp int;

    uniform sampler2D tex;
    out vec4 outTex;
    void main() {
      outTex = texelFetch(tex, ivec2(gl_FragCoord.x, gl_FragCoord.y), 0);
    }`;

    // Compile plain vertex shader and forward_propagation fragment shader to program.
    this.forward.program = GLLib.compile(this.gl, GLLib.computeVertex, this.forward.source);
    // Get uniform and attribbuffer locations for forward pass shader.
    this.forward.positionLocation = this.gl.getAttribLocation(this.forward.program, 'position');
    this.forward.neuronTexLocation = this.gl.getUniformLocation(this.forward.program, 'neuron_tex');
    this.forward.dataTexLocation = this.gl.getUniformLocation(this.forward.program, 'data_tex');
    this.forward.sigmoidLocation = this.gl.getUniformLocation(this.forward.program, 'is_sigmoid');
    this.forward.tanhLocation = this.gl.getUniformLocation(this.forward.program, 'is_tanh');
    this.forward.leakyReluLocation = this.gl.getUniformLocation(this.forward.program, 'is_leaky_relu');
    this.forward.linearLocation = this.gl.getUniformLocation(this.forward.program, 'is_linear');
    GLLib.initShaderObj(this.gl, this.forward);

    // Compile plain vertex shader and training / backward fragment shader to program.
    this.backward.program = GLLib.compile(this.gl, GLLib.computeVertex, this.backward.source);
    // Get uniform and attribbuffer locations for trainings pass shader.
    this.backward.positionLocation = this.gl.getAttribLocation(this.backward.program, 'position');
    this.backward.neuronTexLocation = this.gl.getUniformLocation(this.backward.program, 'neuron_tex');
    this.backward.dataTexLocation = this.gl.getUniformLocation(this.backward.program, 'data_tex');
    this.backward.activityTexLocation = this.gl.getUniformLocation(this.backward.program, 'activity_tex');
    this.backward.errorTexLocation = this.gl.getUniformLocation(this.backward.program, 'error_tex');
    this.backward.learningRateLocation = this.gl.getUniformLocation(this.backward.program, 'learning_rate');
    this.backward.yInsteadOfErrorLocation = this.gl.getUniformLocation(this.backward.program, 'y_instead_of_error');
    this.backward.sigmoidLocation = this.gl.getUniformLocation(this.backward.program, 'is_sigmoid');
    this.backward.tanhLocation = this.gl.getUniformLocation(this.backward.program, 'is_tanh');
    this.backward.leakyReluLocation = this.gl.getUniformLocation(this.backward.program, 'is_leaky_relu');
    this.backward.linearLocation = this.gl.getUniformLocation(this.backward.program, 'is_linear');
    GLLib.initShaderObj(this.gl, this.backward);

    // Compile plain vertex shader and error summing shader to program.
    this.sumError.program = GLLib.compile(this.gl, GLLib.computeVertex, this.sumError.source);
    // Get uniform and attribbuffer locations for shader, which sums all errors found in trainings pass.
    this.sumError.positionLocation = this.gl.getAttribLocation(this.sumError.program, 'position');
    this.sumError.errorSumTexLocation = this.gl.getUniformLocation(this.sumError.program, 'error_sum_tex');
    GLLib.initShaderObj(this.gl, this.sumError);

    // Compile plain vertex shader and training transfer shader to program.
    this.save.program = GLLib.compile(this.gl, GLLib.computeVertex, this.save.source);
    // Get uniform and attribbuffer locations for shader, which sums all errors found in trainings pass.
    this.save.positionLocation = this.gl.getAttribLocation(this.save.program, 'position');
    this.save.texLocation = this.gl.getUniformLocation(this.save.program, 'tex');
    GLLib.initShaderObj(this.gl, this.save);

    // Initialize net structure and neurons.
    this.trainingTextures[0] = GLLib.setByteTexture(this.gl, null, 1, this.structure[0]);
    // Iterate over layers.
    for (let i = 0; i < this.structure.length - 1; i++) {
      // Create a Float32Array for each layer which is easily convertible to a texture for the shader later.

      // The array contains all informations about the neurons in this layer and is structured like this:

      // neuron0:   bias, w0, w1, w2, w3, w4
      // neuron1:   bias, w0, w1, w2, w3, w4
      // neuron2:   bias, w0, w1, w2, w3, w4
      // neuron3:   bias, w0, w1, w2, w3, w4

      // structure[i + 1] ==> neurons in current layer
      // structure[i] ==> weights per neuron in this layer
      // 1 ==> the bias value for each neuron
      this.neurons[i] = new Array(this.structure[i + 1] * (this.structure[i] + 1));
      // Create same array encoded in 8-bit unsigned ints for later use as a texture.
      // An unsigned byte Texture is more fitting here than any other type, because it is renderable,
      // so the program doesn't need to make a difference between reuse of a texture it has been rendered to before
      // and a new texture created from data points.
      // Fill array with random values between -1 and 1 to Initialize all biases and weights.
      for (let j = 0; j < this.neurons[i].length; j++) this.neurons[i][j] = (2 * Math.random() - 1);
      let texArray = GLLib.FloatsToBytes(this.neurons[i]);

      this.trainingTextures.push(GLLib.setByteTexture(this.gl, null, 1, this.structure[i + 1]));
      // Prepare neurons attributes as texture for GPU.
      this.layerTextures.push(GLLib.setByteTexture(this.gl, texArray, this.structure[i] + 1, this.structure[i + 1]));
      // Prepare second renderable texture for GPU.
      this.tempLayerTextures.push(GLLib.setByteTexture(this.gl, null, this.structure[i] + 1, this.structure[i + 1]));
    }

    // Initialize error_texture and error_sum_texture with max texture, that they don't have to be reallocated in vram later.
    this.errorSumTexture = GLLib.setByteTexture(this.gl, null, this.gl.MAX_TEXTURE_SIZE, this.gl.MAX_TEXTURE_SIZE);
    this.errorTexture = GLLib.setByteTexture(this.gl, null, this.gl.MAX_TEXTURE_SIZE, 1);
    this.tempErrorTexture = GLLib.setByteTexture(this.gl, null, this.gl.MAX_TEXTURE_SIZE, 1);
  }

  #normalize = (arr) => {
    let sum = 0;
    for (let i = 0; i < arr.length; i++) sum += arr[i] * arr[i];
    let multip = 1 / Math.sqrt(sum);
    for (let i = 0; i < arr.length; i++) arr[i] *= multip;
    return arr;
  }

  // Return only final output of net.
  predictGPU = (data) => this.forwardPropagationGPU(data);
  predictCPU = (data) => (this.forwardPropagationCPU(data))[this.structure.length - 1];
  // Forward propagation with numbers as output.
  forwardPropagationCPU = (data) => {
    let trainingData = [Array.from(data)];
    let activities = [];
    for (let i = 0; i < this.neurons.length; i++) {
      activities = [];
      for (let j = 0; j < this.neurons[i].length; j+= this.structure[i] + 1) {
        this.#normalize(trainingData[i]);
        // Initialize activity with bias.to_float(texelFetch(neuron_tex, ivec2(0, gl_FragCoord.y), 0));
        let activity = this.neurons[i][j];
        // Get the neurons activity (a) by applying the activation function (sigmoid) to z.
        for (let k = 0; k < this.structure[i]; k++) activity += this.neurons[i][j + k + 1] * trainingData[i][k];
        // a = sigmoid(z)
        if (this.activationStructure[i + 1] === 'sigmoid') activities.push(Math.sigmoid(activity));
        if (this.activationStructure[i + 1] === 'tanh') activities.push(Math.tanh(activity));
        if (this.activationStructure[i + 1] === 'leakyRelu') activities.push(Math.max(0.01 * activity, activity));
        if (this.activationStructure[i + 1] === 'linear') activities.push(activity);
      }
      trainingData.push(activities);
    }
    return trainingData;
  };

  trainCPU = (data, y) => {
    // Forward propagate and save activities for backpropagation.
    let trainingData = this.forwardPropagationCPU (data);
    // Delta_a is an array filled with the errors of the neurons in the current backpropagated layer.
    let deltaA = y.map((item, i) => trainingData[this.structure.length - 1][i] - item);
    // Backpropagate, iterate through layers.
    for (let i = this.neurons.length - 1; i >= 0; i--) {
      // Create new array to accumulate the errors for the next layer to be backpropagated (net.neurons[i - 1]).
      let nextDeltaA = new Array(this.structure[i]).fill(0);
      for (let j = 0; j < this.neurons[i].length / (this.structure[i] + 1); j++) {
        let posInArray = j * (this.structure[i] + 1);
        //    c'(z) = 2 * (a - y) * sigmoid'(a)
        let dcDz = 2 * deltaA[j];
        if (this.activationStructure[i + 1] === 'sigmoid') dcDz *= Math.sigmoidPrime(trainingData[i + 1][j]);
        if (this.activationStructure[i + 1] === 'tanh') dcDz *= 1 - Math.tanh(trainingData[i + 1][j] ** 2);
        if (this.activationStructure[i + 1] === 'leakyRelu') dcDz *= (trainingData[i + 1][j] >= 0) ? 1 : 0.01;
        //    b   = b - λ * c'(z )
        //     n+1   n          n
        this.neurons[i][posInArray] -= this.learningRate * dcDz;
        for (let k = 1; k < this.structure[i] + 1; k++) {
          //    dx   = λ * z'(x ) * c'(z)
          //      i     i          i

          //    x   = x - dx
          //    n+1    n    n
          nextDeltaA[k - 1] += dcDz * this.neurons[i][posInArray + k];
          //    w   = w - λ * z'(w ) * c'(z )
          //     i     i          i
          //     n+1   n          n        n
          this.neurons[i][posInArray + k] -= this.learningRate * trainingData[i][k - 1] * dcDz;
        }
      }
      // Update error list with errors of next layer.
      deltaA = nextDeltaA;
    }
  };

  // Forward propagation with texture array for backpropagation as output.
  forwardPropagationTex = (data) => {
    let dataCopy = Array.from(data);
    // Generate new Uint8 array from data for shader.
    let texData = GLLib.FloatsToBytes(this.#normalize(dataCopy));

    this.gl.bindTexture(this.gl.TEXTURE_2D, this.trainingTextures[0]);
    this.gl.texImage2D(this.gl.TEXTURE_2D, 0, this.gl.RGBA8, 1, data.length, 0, this.gl.RGBA, this.gl.UNSIGNED_BYTE, texData);

    // Tell webgl which program to use.
    this.gl.useProgram(this.forward.program);
    this.gl.bindVertexArray(this.forward.vao);
    // Set width to 1, because only one output (activity) shall be calculated per neuron.
    this.canvas.viewport.width = 1;

    // Iterate over layers and render directly to training_textures array.
    for (let i = 0; i < this.neurons.length; i++) {
      this.canvas.viewport.height = this.structure[i + 1];
      this.gl.viewport(0, 0, this.gl.canvas.viewport.width, this.gl.canvas.viewport.height);
      // Tell program which webgl texture slot to use for which texture.
      this.gl.activeTexture(this.gl.TEXTURE0);
      // Convert to and set this layer as texture for shader.
      this.gl.bindTexture(this.gl.TEXTURE_2D, this.layerTextures[i]);
      this.gl.activeTexture(this.gl.TEXTURE1);
      // Set training_data as data texture.
      this.gl.bindTexture(this.gl.TEXTURE_2D, this.trainingTextures[i]);
      // Link variables in shader with texture slots.
      this.gl.uniform1i(this.forward.neuronTexLocation, 0);
      this.gl.uniform1i(this.forward.dataTexLocation, 1);
      
      this.gl.uniform1i(this.forward.sigmoidLocation, (this.activationStructure[i + 1] === 'sigmoid') ? 1 : 0);
      this.gl.uniform1i(this.forward.tanhLocation, (this.activationStructure[i + 1] === 'tanh') ? 1 : 0);
      this.gl.uniform1i(this.forward.leakyReluLocation, (this.activationStructure[i + 1] === 'leakyRelu') ? 1 : 0);
      this.gl.uniform1i(this.forward.linearLocation, (this.activationStructure[i + 1] === 'linear') ? 1 : 0);
      // Drawcall.
      this.gl.bindBuffer(this.gl.ARRAY_BUFFER, this.forward.vertexBuffer);
      this.gl.bufferData(this.gl.ARRAY_BUFFER, new Float32Array([-1,-1,1,-1,-1,1,-1,1,1,-1,1,1]), this.gl.STATIC_DRAW);
      // Set framebuffer.
      this.gl.bindFramebuffer(this.gl.FRAMEBUFFER, this.sumError.framebuffer);
      // Configure framebuffer for color and depth.
      this.gl.drawBuffers([this.gl.COLOR_ATTACHMENT0]);
      this.gl.framebufferTexture2D(this.gl.FRAMEBUFFER, this.gl.COLOR_ATTACHMENT0, this.gl.TEXTURE_2D, this.trainingTextures[i + 1], 0);
      // Clear depth and color buffers from last frame.
      this.gl.clear(this.gl.COLOR_BUFFER_BIT | this.gl.DEPTH_BUFFER_BIT);
      this.gl.drawArrays(this.gl.TRIANGLES, 0, 6);
      // Normalize layer
      if (this.normalize[i]) {
        let results = new Uint8Array(this.canvas.viewport.width * this.canvas.viewport.height * 4);
        this.gl.readPixels(0, 0, this.canvas.viewport.width, this.canvas.viewport.height, this.gl.RGBA, this.gl.UNSIGNED_BYTE, results);
        // Convert to float, normalize, convert back to bytes.
        let texArray = GLLib.FloatsToBytes(this.#normalize(Array.from(GLLib.BytesToFloats(results))));
        // Prepare neurons attributes as texture for GPU.
        this.gl.bindTexture(this.gl.TEXTURE_2D, this.trainingTextures[i + 1]);
        this.gl.texImage2D(this.gl.TEXTURE_2D, 0, this.gl.RGBA8, this.canvas.viewport.width, this.canvas.viewport.height, 0, this.gl.RGBA, this.gl.UNSIGNED_BYTE, texArray);
      }
    }
  };

  // Forward propagation with texture array for backpropagation as output.
  forwardPropagationGPU = (data) => {
    // Generate new Uint8 array from data for shader.
    this.forwardPropagationTex(data);
    var results = new Uint8Array(this.structure[this.neurons.length] * 4);
    this.gl.readPixels(0, 0, this.canvas.viewport.width, this.canvas.viewport.height, this.gl.RGBA, this.gl.UNSIGNED_BYTE, results);
    return Array.from(GLLib.BytesToFloats(results));
  };

  trainGPU = (data, y) => {
    // Forward propagate and save activities for backpropagation.
    this.forwardPropagationTex(data);
    // Generate new error texture from y.
    let deltaA = GLLib.FloatsToBytes(y);
    // Tell webgl which program to use.
    this.gl.useProgram(this.backward.program);
    this.gl.bindVertexArray(this.backward.vao);
    this.gl.bindTexture(this.gl.TEXTURE_2D, this.errorTexture);
    this.gl.texImage2D(this.gl.TEXTURE_2D, 0, this.gl.RGBA8, y.length, 1, 0, this.gl.RGBA, this.gl.UNSIGNED_BYTE, deltaA);
    // Backpropagate, iterate through layers.
    for (let i = this.neurons.length - 1; i >= 0; i--) {
      // Tell webgl which program to use.
      this.gl.useProgram(this.backward.program);
      this.gl.bindVertexArray(this.backward.vao);
      // Rescale canvas for trainings pass.
      this.canvas.viewport.width = this.structure[i] + 1;
      this.canvas.viewport.height = this.structure[i + 1];
      this.gl.viewport(0, 0, this.gl.canvas.viewport.width, this.gl.canvas.viewport.height);
      // Reset neuron render texture.
      this.gl.bindTexture(this.gl.TEXTURE_2D, this.tempLayerTextures[i]);
      this.gl.texImage2D(this.gl.TEXTURE_2D, 0, this.gl.RGBA8, this.canvas.viewport.width, this.canvas.viewport.height, 0, this.gl.RGBA, this.gl.UNSIGNED_BYTE, null);
      // Reset error sum render texture.
      this.gl.bindTexture(this.gl.TEXTURE_2D, this.errorSumTexture);
      this.gl.texImage2D(this.gl.TEXTURE_2D, 0, this.gl.RGBA8, this.canvas.viewport.width, this.canvas.viewport.height, 0, this.gl.RGBA, this.gl.UNSIGNED_BYTE, null);
      // Set framebuffer.
      this.gl.bindFramebuffer(this.gl.FRAMEBUFFER, this.backward.framebuffer);
      // Configure framebuffer for neuron texture and error_sum_texture.
      this.gl.drawBuffers([this.gl.COLOR_ATTACHMENT0, this.gl.COLOR_ATTACHMENT1]);
      this.gl.framebufferTexture2D(this.gl.FRAMEBUFFER, this.gl.COLOR_ATTACHMENT0, this.gl.TEXTURE_2D, this.tempLayerTextures[i], 0);
      this.gl.framebufferTexture2D(this.gl.FRAMEBUFFER, this.gl.COLOR_ATTACHMENT1, this.gl.TEXTURE_2D, this.errorSumTexture, 0);
      // Clear depth and color buffers from last frame.
      this.gl.clear(this.gl.COLOR_BUFFER_BIT | this.gl.DEPTH_BUFFER_BIT);
      // Tell program which webgl texture slot to use for which texture.
      this.gl.activeTexture(this.gl.TEXTURE0);
      this.gl.bindTexture(this.gl.TEXTURE_2D, this.layerTextures[i]);
      // Set training_data as data texture.
      this.gl.activeTexture(this.gl.TEXTURE1);
      this.gl.bindTexture(this.gl.TEXTURE_2D, this.trainingTextures[i]);
      // Set activities of this layer as texture.
      this.gl.activeTexture(this.gl.TEXTURE2);
      this.gl.bindTexture(this.gl.TEXTURE_2D, this.trainingTextures[i + 1]);

      this.gl.activeTexture(this.gl.TEXTURE3);
      this.gl.bindTexture(this.gl.TEXTURE_2D, this.errorTexture);

      // Link variables in shader with texture slots.
      this.gl.uniform1i(this.backward.neuronTexLocation, 0);
      this.gl.uniform1i(this.backward.dataTexLocation, 1);
      this.gl.uniform1i(this.backward.activityTexLocation, 2);
      this.gl.uniform1i(this.backward.errorTexLocation, 3);

      this.gl.uniform1f(this.backward.learningRateLocation, this.learningRate);
      
      // Tell shader to interprete error texture as ys instead of errors for the first run.
      this.gl.uniform1i(this.backward.yInsteadOfErrorLocation, i !== this.neurons.length - 1 ? 1 : 0);
      
      this.gl.uniform1i(this.backward.sigmoidLocation, (this.activationStructure[i + 1] === 'sigmoid') ? 1 : 0);
      this.gl.uniform1i(this.backward.tanhLocation, (this.activationStructure[i + 1] === 'tanh') ? 1 : 0);
      this.gl.uniform1i(this.backward.leakyReluLocation, (this.activationStructure[i + 1] === 'leakyRelu') ? 1 : 0);
      this.gl.uniform1i(this.backward.linearLocation, (this.activationStructure[i + 1] === 'linear') ? 1 : 0);
      // Drawcall.
      this.gl.bindBuffer(this.gl.ARRAY_BUFFER, this.backward.vertexBuffer);
      this.gl.bufferData(this.gl.ARRAY_BUFFER, new Float32Array([- 1, - 1, 1, - 1, - 1, 1, - 1, 1, 1, - 1, 1, 1]), this.gl.STATIC_DRAW);

      this.gl.drawArrays(this.gl.TRIANGLES, 0, 6);

      // Switch framebuffer texture with texture in main array to update values without allocating new RAM / VRAM.
      let temp = this.layerTextures[i];
      this.layerTextures[i] = this.tempLayerTextures[i];
      this.tempLayerTextures[i] = temp;

      // Rescale canvas for error summing pass.
      this.canvas.viewport.width = this.structure[i];
      this.canvas.viewport.height = 1;
      this.gl.viewport(0, 0, this.gl.canvas.viewport.width, this.gl.canvas.viewport.height);

      // Tell webgl which program to use.
      this.gl.useProgram(this.sumError.program);
      this.gl.bindVertexArray(this.sumError.vao);
      // Reset error texture.
      this.gl.bindTexture(this.gl.TEXTURE_2D, this.errorTexture);
      this.gl.texImage2D(this.gl.TEXTURE_2D, 0, this.gl.RGBA8, this.canvas.viewport.width, this.canvas.viewport.height, 0, this.gl.RGBA, this.gl.UNSIGNED_BYTE, null);

      // Set framebuffer.
      this.gl.bindFramebuffer(this.gl.FRAMEBUFFER, this.sumError.framebuffer);
      // Configure framebuffer for color and depth.
      this.gl.drawBuffers([this.gl.COLOR_ATTACHMENT0]);
      this.gl.framebufferTexture2D(this.gl.FRAMEBUFFER, this.gl.COLOR_ATTACHMENT0, this.gl.TEXTURE_2D, this.errorTexture, 0);
      // Clear depth and color buffers from last frame.
      this.gl.clear(this.gl.COLOR_BUFFER_BIT | this.gl.DEPTH_BUFFER_BIT);

      // Tell program which webgl texture slot to use for which texture.
      this.gl.activeTexture(this.gl.TEXTURE0);
      this.gl.bindTexture(this.gl.TEXTURE_2D, this.errorSumTexture);

      // Link variables in shader with texture slots.
      this.gl.uniform1i(this.sumError.errorSumTexLocation, 0);

      // Feed rasterizer.
      this.gl.bindBuffer(this.gl.ARRAY_BUFFER, this.sumError.vertexBuffer);
      this.gl.bufferData(this.gl.ARRAY_BUFFER, new Float32Array([- 1, - 1, 1, - 1, - 1, 1, - 1, 1, 1, - 1, 1, 1]), this.gl.STATIC_DRAW);

      // Drawcall.
      this.gl.drawArrays(this.gl.TRIANGLES, 0, 6);
    }
  };

  loadTraining = () => {
    for (let i = 0; i < this.neurons.length; i++) {
      let texArray = GLLib.FloatsToBytes(this.neurons[i]);
      // Prepare neurons attributes as texture for GPU.
      this.gl.bindTexture(this.gl.TEXTURE_2D, this.layerTextures[i]);
      this.gl.texImage2D(this.gl.TEXTURE_2D, 0, this.gl.RGBA8, this.structure[i] + 1, this.structure[i + 1], 0, this.gl.RGBA, this.gl.UNSIGNED_BYTE, texArray);
    }
  };

  saveTraining = () => {
    for (let i = this.neurons.length - 1; i >= 0; i--) {
      // Tell webgl which program to use.
      this.gl.useProgram(this.save.program);
      this.gl.bindVertexArray(this.save.vao);
      // Rescale canvas for trainings pass.
      this.canvas.viewport.width = this.structure[i] + 1;
      this.canvas.viewport.height = this.structure[i + 1];
      this.gl.viewport(0, 0, this.canvas.viewport.width, this.canvas.viewport.height);
      // Set framebuffer.
      this.gl.bindFramebuffer(this.gl.FRAMEBUFFER, null);
      // Clear depth and color buffers from last frame.
      this.gl.clear(this.gl.COLOR_BUFFER_BIT | this.gl.DEPTH_BUFFER_BIT);
      // Tell program which webgl texture slot to use for which texture.
      this.gl.activeTexture(this.gl.TEXTURE0);
      this.gl.bindTexture(this.gl.TEXTURE_2D, this.layerTextures[i]);
      // Link variables in shader with texture slots.
      this.gl.uniform1i(this.save.texLocation, 0);
      // Drawcall.
      this.gl.bindBuffer(this.gl.ARRAY_BUFFER, this.save.vertexBuffer);
      this.gl.bufferData(this.gl.ARRAY_BUFFER, new Float32Array([- 1, - 1, 1, - 1, - 1, 1, - 1, 1, 1, - 1, 1, 1]), this.gl.STATIC_DRAW);
      this.gl.drawArrays(this.gl.TRIANGLES, 0, 6);
      
      let results = new Uint8Array(this.neurons[i].length * 4);
      this.gl.readPixels(0, 0, this.canvas.viewport.width, this.canvas.viewport.height, this.gl.RGBA, this.gl.UNSIGNED_BYTE, results);

      this.neurons[i] = Array.from(GLLib.BytesToFloats(results));
    }
  };
}
