"use-strict";

import { GLLib } from './gllib.js';

const fetchSync = url => {
  var request = new XMLHttpRequest();
  request.open('GET', url, false); // Set the third parameter to false for a synchronous request
  request.send(null);
  
  if (request.status === 200) {
    return request.responseText;
  } else {
    // Handle HTTP error (404, 500, etc.)
    console.error("Failed to fetch data: ", request.statusText);
  }
}

export class NetWebGL2 {
  // Create webgl context necessary for hardware acceleration.
  canvas = document.createElement("canvas");
  gl = this.canvas.getContext("webgl2");

  forward = {};
  backprop = {};
  sum_error = {};
  fetch_model = {};

  trainingTextures = [];
  layerTextures = [];
  tempLayerTextures = [];

  errorSumTexture = this.gl.createTexture();
  errorTexture = this.gl.createTexture();
  learningRate = 0.01;
  
  neurons;
  structure;
  activationStructure;

  constructor (structure, activationStructure, normalize) {
    this.structure = structure;
    this.activationStructure = activationStructure ?? [... new Array(structure.length - 1).fill('leakyRelu'), 'linear'];
    this.normalize = normalize ?? [... new Array(structure.length - 2).fill(true), false];

    this.canvas.viewport = {};
    // Use maximum needed canvas size and regulate rendered pixels using viewport
    let maxLayerSize = structure.reduce((p, c) => Math.max(p, c), 0);
    this.canvas.width = maxLayerSize + 1;
    this.canvas.height = maxLayerSize;

    this.neurons = new Array(structure.length - 1);

    let protoPrograms = [
      {
        name: 'forward',
        attribs: ['position'],
        uniforms: ['neuron_tex', 'data_tex', 'is_sigmoid', 'is_tanh', 'is_leaky_relu', 'is_linear']
      },
      {
        name: 'backprop',
        attribs: ['position'],
        uniforms: ['neuron_tex', 'data_tex', 'activity_tex', 'error_tex', 'learning_rate', 'y_instead_of_error', 'is_sigmoid', 'is_tanh', 'is_leaky_relu', 'is_linear']
      },
      {
        name: 'sum_error',
        attribs: ['position'],
        uniforms: ['error_sum_tex']
      },
      {
        name: 'fetch_model',
        attribs: ['position'],
        uniforms: ['tex']
      }
    ];

    protoPrograms.forEach(proto => {
      const {name, attribs, uniforms} = proto;
      let ref = this[name];
      // Fetch shader source.
      let source = fetchSync('./shaders/MLP_' + name + '.glsl');
      console.log('/shaders/MLP_' + name + '.glsl');
      // Compile plain vertex shader and forward_propagation fragment shader to program.
      ref.program = GLLib.compile(this.gl, GLLib.computeVertex, source);
      // Get uniform and attribbuffer locations for forward pass shader.
      attribs.forEach(param => ref[param] = this.gl.getAttribLocation(ref.program, param));
      uniforms.forEach(param => ref[param] = this.gl.getUniformLocation(ref.program, param));
      GLLib.initShaderObj(this.gl, ref);
    });

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

  // Forward propagation with texture array for backpropagation as output.
  forwardPropagationTex = (data) => {
    let dataCopy = Array.from(data);
    // Generate new Uint8 array from data for shader.
    // Generate new Uint8 array from data for shader.
    let texData = GLLib.FloatsToBytes(this.#normalize(dataCopy));
    // let texData = GLLib.FloatsToBytes(dataCopy);

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
      // console.log(this.forward);
      this.gl.uniform1i(this.forward.neuron_tex, 0);
      this.gl.uniform1i(this.forward.data_tex, 1);
      
      this.gl.uniform1i(this.forward.is_sigmoid, (this.activationStructure[i + 1] === 'sigmoid') ? 1 : 0);
      this.gl.uniform1i(this.forward.is_tanh, (this.activationStructure[i + 1] === 'tanh') ? 1 : 0);
      this.gl.uniform1i(this.forward.is_leaky_relu, (this.activationStructure[i + 1] === 'leakyRelu') ? 1 : 0);
      this.gl.uniform1i(this.forward.is_linear, (this.activationStructure[i + 1] === 'linear') ? 1 : 0);
      // Drawcall.
      this.gl.bindBuffer(this.gl.ARRAY_BUFFER, this.forward.vertexBuffer);
      this.gl.bufferData(this.gl.ARRAY_BUFFER, new Float32Array([- 1, - 1, 1, - 1, - 1, 1, - 1, 1, 1, - 1, 1, 1]), this.gl.STATIC_DRAW);
      // Set framebuffer.
      this.gl.bindFramebuffer(this.gl.FRAMEBUFFER, this.sum_error.framebuffer);
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
  predict = data => {
    // Generate new Uint8 array from data for shader.
    this.forwardPropagationTex(data);
    var results = new Uint8Array(this.structure[this.neurons.length] * 4);
    this.gl.readPixels(0, 0, this.canvas.viewport.width, this.canvas.viewport.height, this.gl.RGBA, this.gl.UNSIGNED_BYTE, results);
    return Array.from(GLLib.BytesToFloats(results));
  };

  train = (data, y) => {
    // Forward propagate and fetch_model activities for backpropagation.
    this.forwardPropagationTex(data);
    // Generate new error texture from y.
    let deltaA = GLLib.FloatsToBytes(y);
    // Tell webgl which program to use.
    this.gl.useProgram(this.backprop.program);
    this.gl.bindVertexArray(this.backprop.vao);
    this.gl.bindTexture(this.gl.TEXTURE_2D, this.errorTexture);
    this.gl.texImage2D(this.gl.TEXTURE_2D, 0, this.gl.RGBA8, y.length, 1, 0, this.gl.RGBA, this.gl.UNSIGNED_BYTE, deltaA);
    // Backpropagate, iterate through layers.
    for (let i = this.neurons.length - 1; i >= 0; i--) {
      // Tell webgl which program to use.
      this.gl.useProgram(this.backprop.program);
      this.gl.bindVertexArray(this.backprop.vao);
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
      this.gl.bindFramebuffer(this.gl.FRAMEBUFFER, this.backprop.framebuffer);
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
      this.gl.uniform1i(this.backprop.neuron_tex, 0);
      this.gl.uniform1i(this.backprop.data_tex, 1);
      this.gl.uniform1i(this.backprop.activity_tex, 2);
      this.gl.uniform1i(this.backprop.error_tex, 3);

      this.gl.uniform1f(this.backprop.learning_rate, this.learningRate);
      
      // Tell shader to interprete error texture as ys instead of errors for the first run.
      this.gl.uniform1i(this.backprop.y_instead_of_error, i !== this.neurons.length - 1 ? 1 : 0);
      
      this.gl.uniform1i(this.backprop.is_sigmoid, (this.activationStructure[i + 1] === 'sigmoid') ? 1 : 0);
      this.gl.uniform1i(this.backprop.is_tanh, (this.activationStructure[i + 1] === 'tanh') ? 1 : 0);
      this.gl.uniform1i(this.backprop.is_leaky_relu, (this.activationStructure[i + 1] === 'leakyRelu') ? 1 : 0);
      this.gl.uniform1i(this.backprop.is_linear, (this.activationStructure[i + 1] === 'linear') ? 1 : 0);
      // Drawcall.
      this.gl.bindBuffer(this.gl.ARRAY_BUFFER, this.backprop.vertexBuffer);
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
      this.gl.useProgram(this.sum_error.program);
      this.gl.bindVertexArray(this.sum_error.vao);
      // Reset error texture.
      this.gl.bindTexture(this.gl.TEXTURE_2D, this.errorTexture);
      this.gl.texImage2D(this.gl.TEXTURE_2D, 0, this.gl.RGBA8, this.canvas.viewport.width, this.canvas.viewport.height, 0, this.gl.RGBA, this.gl.UNSIGNED_BYTE, null);

      // Set framebuffer.
      this.gl.bindFramebuffer(this.gl.FRAMEBUFFER, this.sum_error.framebuffer);
      // Configure framebuffer for color and depth.
      this.gl.drawBuffers([this.gl.COLOR_ATTACHMENT0]);
      this.gl.framebufferTexture2D(this.gl.FRAMEBUFFER, this.gl.COLOR_ATTACHMENT0, this.gl.TEXTURE_2D, this.errorTexture, 0);
      // Clear depth and color buffers from last frame.
      this.gl.clear(this.gl.COLOR_BUFFER_BIT | this.gl.DEPTH_BUFFER_BIT);

      // Tell program which webgl texture slot to use for which texture.
      this.gl.activeTexture(this.gl.TEXTURE0);
      this.gl.bindTexture(this.gl.TEXTURE_2D, this.errorSumTexture);

      // Link variables in shader with texture slots.
      this.gl.uniform1i(this.sum_error.error_sum_tex, 0);

      // Feed rasterizer.
      this.gl.bindBuffer(this.gl.ARRAY_BUFFER, this.sum_error.vertexBuffer);
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

  fetch_modelTraining = () => {
    for (let i = this.neurons.length - 1; i >= 0; i--) {
      // Tell webgl which program to use.
      this.gl.useProgram(this.fetch_model.program);
      this.gl.bindVertexArray(this.fetch_model.vao);
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
      this.gl.uniform1i(this.fetch_model.tex, 0);
      // Drawcall.
      this.gl.bindBuffer(this.gl.ARRAY_BUFFER, this.fetch_model.vertexBuffer);
      this.gl.bufferData(this.gl.ARRAY_BUFFER, new Float32Array([- 1, - 1, 1, - 1, - 1, 1, - 1, 1, 1, - 1, 1, 1]), this.gl.STATIC_DRAW);
      this.gl.drawArrays(this.gl.TRIANGLES, 0, 6);
      
      let results = new Uint8Array(this.neurons[i].length * 4);
      this.gl.readPixels(0, 0, this.canvas.viewport.width, this.canvas.viewport.height, this.gl.RGBA, this.gl.UNSIGNED_BYTE, results);

      this.neurons[i] = Array.from(GLLib.BytesToFloats(results));
    }
  };
}
