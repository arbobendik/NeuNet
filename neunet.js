"use-strict";

import { GLLib } from './gllib.js';
import { Math } from './math.js';

export class Neuron {
  learningRate = 0.1;
  bias;
  weights;

  constructor (inputs) {
    // Initialize weights and bias with random values between -1 and 1.
    this.bias = 2 * Math.random() - 1;
    this.weights = [];
    this.weights.length = inputs;
    for (let i = 0; i < inputs; i ++) this.weights[i] = 2 * Math.random() - 1;
  }
    
  forwardPropagation = (data) => {
    // Propagate forward by multiplying all inputs with their respective weights.

    //             n
    //    z = b +  E   x * w
    //            t=0   t   t
    let activity = this.bias;
    for (let i = 0; i < this.weights.length; i++) activity += this.weights[i] * data[i];
    // Get the neurons activity (a) by applying the activation function (sigmoid) to z.

    //    a = sigmoid(z)
    return Math.sigmoid(activity);
  };
  
  backPropagation = (data, activity, error) => {
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
    let dc_dz = 2 * error * Math.sigmoidPrime(activity);
    // The x values of c(x, w, b) are representing the activity of neurons in the former layer.
    // Therefore the c'(x) values needed to calculate the ideal y values for those neurons are returned to the neural networks train() function.
    let dx = new Array(data.length).fill(0);
    // The derivatives of z(w) and z(x) are dependant on the respective indices of w and x.
    for (let i = 0; i < this.weights.length; i++) {
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
      //      i     i      i

      //    x   = x - dx
      //    n+1    n    n
      dx[i] = this.weights[i] * dc_dz;

      // So the changes for the weights can be calculated with gradient descent.

      //    w   = w - λ * z'(w ) * c'(z )
      //     i     i          i
      //     n+1   n          n        n
      this.weights[i] -= this.learningRate * data[i] * dc_dz;
    }
    // The derivative of z with respect to the bias is 1. So applying the chainrule will result in c'(z).

    //    b   = b - λ * c'(z )
    //     n+1   n          n
    this.bias -= this.learningRate * dc_dz;
    // Return dx values to the neural net's train function.
    return dx;
  };
  // Seperate training funciton for training the neuron without a net.
  train = (data, y) => {
    // Propagate forward to get activities.
    let activity = this.forwardPropagation(data);
    // Calculate error.
    let error = activity - y;
    // Propagate backwards.
    this.backPropagation(data, activity, error);
  };
}

export class Net {
  neurons = [];
  structure;

  constructor (structure) {
    this.structure = structure;
    // Initialize net structure and neurons.
    this.neurons = new Array(structure.length - 1);
    for (let i = 1; i < structure.length; i++) {
      this.neurons[i - 1] = new Array (structure[i]);
      for (let j = 0; j < structure[i]; j++) this.neurons[i - 1][j] = new Neuron (structure[i - 1]);
    }
  };

  // Return only final output of net.
  predict = (data) => this.forwardPropagation (data) [this.structure.length - 1];
  // Forward propagation through all layers.
  forwardPropagation = (data) => {
    let trainingData = [data];
    let activities = [];
    for (let i = 0; i < this.neurons.length; i++) {
      activities = [];
      for (let j = 0; j < this.neurons[i].length; j++) activities.push(this.neurons[i][j].forwardPropagation(trainingData[i]));
      trainingData.push(activities);
    }
    return trainingData;
  };

  train = (data, y) => {
    // Forward propagate and save activities for backpropagation.
    let trainingData = this.forwardPropagation(data);
    // DeltaA is an array filled with the errors of the neurons in the current backpropagated layer.
    let deltaA = y.map((item, i) => trainingData[this.structure.length - 1][i] - item);
    // Backpropagate, iterate through layers.
    for (let i = this.neurons.length - 1; i >= 0; i--) {
      // Create new array to accumulate the errors for the next layer to be backpropagated (net.neurons[i - 1]).
      let nextDeltaA = new Array(this.structure[i]).fill(0);
      for (let j = 0; j < this.neurons[i].length; j++) {
        // Backpropagate individual neuron.
        let changesDeltaA = this.neurons[i][j].backPropagation (trainingData[i], trainingData[i+1][j], deltaA[j]);
        // Accumulate changes to get an estimation of what the error of the former layer might be.
        nextDeltaA = changesDeltaA.map((item, i) => nextDeltaA[i] + item);
      }
      // Update error list with errors of next layer.
      deltaA = nextDeltaA;
    }
  };
}

export class NetWebGL2 {
  // Create webgl context necessary for hardware acceleration.
  canvas = document.createElement("canvas");
  gl = this.canvas.getContext("webgl2");

  forward = {};
  backward = {};
  sumError = {};

  trainingTextures = [];
  layerTextures = [];
  tempLayerTextures = [];

  errorSumTexture = this.gl.createTexture();
  errorTexture = this.gl.createTexture();
  learningRate = 0.1;
  
  neurons;
  structure;

  constructor (structure) {
    this.structure = structure;
    this.neurons = new Array(structure.length - 1);
    // Source code for compute (fragment) shader for forward pass through net.
    this.forward.source = `#version 300 es
    precision highp float;
    precision highp int;

    uniform sampler2D neuron_tex;
    uniform sampler2D data_tex;

    out vec4 activity_out;

    // Activation function of neuron.
    float sigmoid(float x) {
      return 1.0 / (1.0 + pow(2.718281828459045, - x));
    }

    // Convert 4 bytes, texture channels to usable float.
    float to_float(vec4 bytes) {
      return (bytes.x * 255.0 + bytes.y + bytes.z / 255.0 + bytes.w / 65025.0) * 2.0 - 255.0;
    }

    // Split float into 4 8-bit texture channels.
    vec4 to_bytes(float num) {
      float f = (num + 255.0) / 2.0;
      vec4 bytes = vec4(
        f / 255.0,
        f,
        f * 255.0,
        f * 65025.0
      );
      // Use modulo that the sum of all bytes is num.
      bytes = mod(bytes, 1.0);
      // Ensure that values won't be rounded.
      return floor(bytes * 255.0) / 255.0;
    }

    // Shader to calculate activity of a single neuron.
    void main() {
      vec4 neuron_texel = texelFetch(neuron_tex, ivec2(0, gl_FragCoord.y), 0);
      // Width is always one, so row gl_FragCoord.y in neuronTex is the line of neuron[gl_FragCoord.y].
      // Initialize z with bias.
      float z = to_float(neuron_texel);
      // Get width of neuron_tex to get number of weights + 1
      int neuron_tex_width = textureSize(neuron_tex, 0).x;
      // Iterate over inputs and respective weights for this neuron.
      for (int i = 1; i < neuron_tex_width; i++) {
        neuron_texel = texelFetch(neuron_tex, ivec2(i, gl_FragCoord.y), 0);
        vec4 data_texel = texelFetch(data_tex, ivec2(0, i - 1), 0);
        // Add weight[i] * input[i] to z.
        z += to_float(neuron_texel) * to_float(data_texel);
      }
      vec4 data_texel = texelFetch(data_tex, ivec2(0, 0), 0);
      // Calculate activity.
      // Split activity into four bytes.
      activity_out = to_bytes(sigmoid(z));
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

    layout(location = 0) out vec4 neuron_out;
    layout(location = 1) out vec4 error_out;

    // Activation function of neuron.
    float sigmoid(float x) {
      return 1.0 / (1.0 + pow(2.718281828459045, - x));
    }

    // Derivative of activation function.
    float sigmoid_prime(float x) {
      return sigmoid(x) * (1.0 - sigmoid(x));
    }

    // Convert 4 bytes, texture channels to usable float.
    float to_float(vec4 bytes) {
      return (bytes.x * 255.0 + bytes.y + bytes.z / 255.0 + bytes.w / 65025.0) * 2.0 - 255.0;
    }

    // Split float into 4 8-bit texture channels.
    vec4 to_bytes(float num) {
      float f = (num + 255.0) / 2.0;
      vec4 bytes = vec4(
        f / 255.0,
        f,
        f * 255.0,
        f * 65025.0
      );
      // Use modulo that the sum of all bytes is num.
      bytes = mod(bytes, 1.0);
      // Ensure that values won't be rounded.
      return floor(bytes * 255.0) / 255.0;
    }

    // Shader to calculate activity of a single neuron.
    void main() {
      // Width is always one, so row gl_FragCoord.y in neuronTex is the line of neuron[gl_FragCoord.y].
      int row = int(gl_FragCoord.y);
      int column = int(gl_FragCoord.x);

      vec4 activity_texel = texelFetch(activity_tex, ivec2(0, row), 0);
      float activity = to_float(activity_texel);

      vec4 error_texel = texelFetch(error_tex, ivec2(row, 0), 0);
      float error = 0.0;
      if (y_instead_of_error == 0) {
        error = activity - to_float(error_texel);
      } else {
        error = to_float(error_texel);
      }

      float dc_dz = 2.0 * error * sigmoid_prime(activity);

      vec4 neuron_texel = texelFetch(neuron_tex, ivec2(column, row), 0);
      float weight = to_float(neuron_texel);

      float modifier = dc_dz * learning_rate;
      float dx = weight * dc_dz;

      if (column != 0) {
        vec4 data_texel = texelFetch(data_tex, ivec2(0, column - 1), 0);
        modifier *= to_float(data_texel);
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
      return (bytes.x * 255.0 + bytes.y + bytes.z / 255.0 + bytes.w / 65025.0) * 2.0 - 255.0;
    }

    // Split float into 4 8-bit texture channels.
    vec4 to_bytes(float num) {
      float f = (num + 255.0) / 2.0;
      vec4 bytes = vec4(
        f / 255.0,
        f,
        f * 255.0,
        f * 65025.0
      );
      // Use modulo that the sum of all bytes is num.
      bytes = mod(bytes, 1.0);
      // Ensure that values won't be rounded.
      return floor(bytes * 255.0) / 255.0;
    }

    // Sum all errors of one .
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

    // Compile plain vertex shader and forward_propagation fragment shader to program.
    this.forward.program = GLLib.compile(this.gl, GLLib.computeVertex, this.forward.source);
    // Get uniform and attribbuffer locations for forward pass shader.
    this.forward.positionLocation = this.gl.getAttribLocation(this.forward.program, 'position');
    this.forward.neuronTexLocation = this.gl.getUniformLocation(this.forward.program, 'neuron_tex');
    this.forward.dataTexLocation = this.gl.getUniformLocation(this.forward.program, 'data_tex');
    // Create buffer to provide two vertices to vertex shader.
    this.forward.vertexBuffer = this.gl.createBuffer();
    this.gl.bindBuffer(this.gl.ARRAY_BUFFER, this.forward.vertexBuffer);
    this.gl.bufferData(this.gl.ARRAY_BUFFER, new Float32Array ([- 1 , - 1, 1, - 1, - 1, 1, - 1, 1, 1, - 1, 1, 1]), this.gl.STATIC_DRAW);
    // Create vertex array object.
    this.forward.vao = this.gl.createVertexArray();
    this.gl.bindVertexArray(this.forward.vao);
    // Tell WebGl how to draw vertices.
    this.gl.enableVertexAttribArray(this.forward.positionLocation);
    this.gl.vertexAttribPointer(this.forward.positionLocation, 2, this.gl.FLOAT, false, 0, 0);

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
    // Create buffer to provide two vertices to vertex shader.
    this.backward.vertexBuffer = this.gl.createBuffer();
    this.gl.bindBuffer(this.gl.ARRAY_BUFFER, this.backward.vertexBuffer);
    this.gl.bufferData(this.gl.ARRAY_BUFFER, new Float32Array([- 1 , - 1, 1, - 1, - 1, 1, - 1, 1, 1, - 1, 1, 1]), this.gl.STATIC_DRAW);
    // Create vertex array object.
    this.backward.vao = this.gl.createVertexArray();
    this.backward.framebuffer = this.gl.createFramebuffer();
    this.gl.bindVertexArray(this.backward.vao);
    // Tell WebGl how to draw vertices.
    this.gl.enableVertexAttribArray(this.backward.positionLocation);
    this.gl.vertexAttribPointer(this.backward.positionLocation, 2, this.gl.FLOAT, false, 0, 0);

    // Compile plain vertex shader and error summing shader to program.
    this.sumError.program = GLLib.compile(this.gl, GLLib.computeVertex, this.sumError.source);
    // Get uniform and attribbuffer locations for shader, which sums all errors found in trainings pass.
    this.sumError.positionLocation = this.gl.getAttribLocation(this.sumError.program, 'position');
    this.sumError.errorSumTexLocation = this.gl.getUniformLocation(this.sumError.program, 'error_sum_tex');
    // Create buffer to provide two vertices to vertex shader.
    this.sumError.vertexBuffer = this.gl.createBuffer();
    this.gl.bindBuffer(this.gl.ARRAY_BUFFER, this.sumError.vertexBuffer);
    this.gl.bufferData(this.gl.ARRAY_BUFFER, new Float32Array([- 1 , - 1, 1, - 1, - 1, 1, - 1, 1, 1, - 1, 1, 1]), this.gl.STATIC_DRAW);
    // Create vertex array object.
    this.sumError.vao = this.gl.createVertexArray();
    this.sumError.framebuffer = this.gl.createFramebuffer();
    this.gl.bindVertexArray(this.sumError.vao);
    // Tell WebGl how to draw vertices.
    this.gl.enableVertexAttribArray(this.sumError.positionLocation);
    this.gl.vertexAttribPointer(this.sumError.positionLocation, 2, this.gl.FLOAT, false, 0, 0);

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
      let texArray = new Uint8Array(this.neurons[i].length * 4).fill(0);
      // Fill array with random values between -1 and 1 to Initialize all biases and weights.
      for (let j = 0; j < this.neurons[i].length; j++) {
        // Stretch rand to a range between -1 and 1.
        this.neurons[i][j] = (2 * Math.random() - 1);
        // Keep positive range and convert later in the shader, because the unsigned byte format can only hold positive numbers.
        let bytes = GLLib.toBytes(this.neurons[i][j]);
        // Keep positive range and convert later in the shader, because the unsigned byte format can only hold positive numbers.
        // Increase precision by using a second 8-bit number for more detail.
        // This adds up to a total bit-depth of 15 plus, one bit for prefix.
        texArray[j * 4] = bytes[0];
        texArray[j * 4 + 1] = bytes[1];
        texArray[j * 4 + 2] = bytes[2];
        texArray[j * 4 + 3] = bytes[3];
      }

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

  // Return only final output of net.
  predict = (data) => this.forwardPropagationGPU(data);
  // Forward propagation with numbers as output.
  forwardPropagationCPU = (data) => {
    var trainingData = [data];
    var activities = [];
    for (let i = 0; i < this.neurons.length; i++) {
      activities = [];
      for (let j = 0; j < this.neurons[i].length; j+= this.structure[i] + 1) {
        // Initialize activity with bias.
        let activity = this.neurons[i][j];
        // Get the neurons activity (a) by applying the activation function (sigmoid) to z.
        for (let k = 0; k < this.structure[i]; k++) activity += this.neurons[i][j + k + 1] * trainingData[i][k];
        // a = sigmoid(z)
        activities.push(Math.sigmoid(activity));
      }
      trainingData.push(activities);
    }
    return trainingData;
  };

  trainCPU = async (data, y) => {
    // Forward propagate and save activities for backpropagation.
    let trainingData = this.forwardPropagation (data);
    // Delta_a is an array filled with the errors of the neurons in the current backpropagated layer.
    var deltaA = y.map((item, i) => trainingData[this.structure.length - 1][i] - item);
    // Backpropagate, iterate through layers.
    for (let i = this.neurons.length - 1; i >= 0; i--) {
      // Create new array to accumulate the errors for the next layer to be backpropagated (net.neurons[i - 1]).
      nextDeltaA = new Array(this.structure[i]).fill(0);
      for (let j = 0; j < this.neurons[i].length / (this.structure[i] + 1); j++) {
        let posInArray = j * (this.structure[i] + 1);
        //    c'(z) = 2 * (a - y) * sigmoid'(a)
        let dcDz = 2 * deltaA[j] * Math.sigmoidPrime(trainingData[i + 1][j]);
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
    // Generate new Uint8 array from data for shader.
    var texData = new Uint8Array(data.length * 4);
    for (let i = 0; i < data.length * 4; i+=4) {
      let bytes = GLLib.toBytes(data[i / 4]);
      texData[i] = bytes[0];
      texData[i + 1] = bytes[1];
      texData[i + 2] = bytes[2];
      texData[i + 3] = bytes[3];
    }

    this.gl.bindTexture(this.gl.TEXTURE_2D, this.trainingTextures[0]);
    this.gl.texImage2D(this.gl.TEXTURE_2D, 0, this.gl.RGBA8, 1, data.length, 0, this.gl.RGBA, this.gl.UNSIGNED_BYTE, texData);

    // Tell webgl which program to use.
    this.gl.useProgram(this.forward.program);
    this.gl.bindVertexArray(this.forward.vao);
    // Set width to 1, because only one output (activity) shall be calculated per neuron.
    this.canvas.width = 1;
    // Iterate over layers and render directly to training_textures array.
    for (var i = 0; i < this.neurons.length; i++) {
      this.canvas.height = this.structure[i + 1];
      this.gl.viewport(0, 0, this.gl.canvas.width, this.gl.canvas.height);
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
      // Drawcall.
      this.gl.bindBuffer(this.gl.ARRAY_BUFFER, this.forward.vertexBuffer);
      this.gl.bufferData(this.gl.ARRAY_BUFFER, new Float32Array([-1,-1,1,-1,-1,1,-1,1,1,-1,1,1]), this.gl.STATIC_DRAW);

      // Set framebuffer.
      this.gl.bindFramebuffer(this.gl.FRAMEBUFFER, this.sumError.framebuffer);
      // Configure framebuffer for color and depth.
      this.gl.drawBuffers([this.gl.COLOR_ATTACHMENT0]);
      this.gl.framebufferTexture2D(this.gl.FRAMEBUFFER, this.gl.COLOR_ATTACHMENT0, this.gl.TEXTURE_2D, this.trainingTextures[i + 1], 0);

      this.gl.drawArrays(this.gl.TRIANGLES, 0, 6);
    }
  };

  // Forward propagation with texture array for backpropagation as output.
  forwardPropagationGPU = (data) => {
    // Generate new Uint8 array from data for shader.
    var texData = new Uint8Array(data.length * 4);
    for (let i = 0; i < data.length * 4; i+=4) {
      let bytes = GLLib.toBytes(data[i / 4]);
      texData[i] = bytes[0];
      texData[i + 1] = bytes[1];
      texData[i + 2] = bytes[2];
      texData[i + 3] = bytes[3];
    }

    let results = [];

    this.gl.bindTexture(this.gl.TEXTURE_2D, this.trainingTextures[0]);
    this.gl.texImage2D(this.gl.TEXTURE_2D, 0, this.gl.RGBA8, 1, data.length, 0, this.gl.RGBA, this.gl.UNSIGNED_BYTE, texData);

    // Tell webgl which program to use.
    this.gl.useProgram(this.forward.program);
    this.gl.bindVertexArray(this.forward.vao);
    // Set width to 1, because only one output (activity) shall be calculated per neuron.
    this.canvas.width = 1;
    // Iterate over layers and render directly to training_textures array.
    for (var i = 0; i < this.neurons.length; i++) {
      this.canvas.height = this.structure[i + 1];
      this.gl.viewport(0, 0, this.gl.canvas.width, this.gl.canvas.height);
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
      // Drawcall.
      this.gl.bindBuffer(this.gl.ARRAY_BUFFER, this.forward.vertexBuffer);
      this.gl.bufferData(this.gl.ARRAY_BUFFER, new Float32Array([- 1, - 1, 1, - 1, - 1, 1, - 1, 1, 1, - 1, 1, 1]), this.gl.STATIC_DRAW);

      // Set framebuffer.
      this.gl.bindFramebuffer(this.gl.FRAMEBUFFER, this.sumError.framebuffer);
      // Configure framebuffer for color and depth.
      this.gl.drawBuffers([this.gl.COLOR_ATTACHMENT0]);
      this.gl.framebufferTexture2D(this.gl.FRAMEBUFFER, this.gl.COLOR_ATTACHMENT0, this.gl.TEXTURE_2D, this.trainingTextures[i + 1], 0);

      this.gl.drawArrays(this.gl.TRIANGLES, 0, 6);

      if (i === this.neurons.length - 1) {
        var vals = new Uint8Array(this.structure[i + 1] * 4);
        this.gl.readPixels(0, 0, this.canvas.width, this.canvas.height, this.gl.RGBA, this.gl.UNSIGNED_BYTE, vals);

        for (let j = 0; j < this.structure[i + 1] * 4; j += 4) {
          // Apply dx values to neural layer texture.
          results.push(Math.abs(GLLib.toFloat([vals[j], vals[j + 1], vals[j + 2], vals[j + 3]])));
        }
      }
    }
    return results;
  };

  trainGPU = async (data, y) => {
    // Forward propagate and save activities for backpropagation.
    this.forwardPropagationTex(data);

    // Generate new error texture from y.
    var deltaA = new Uint8Array(y.length * 4);
    for (let i = 0; i < y.length * 4; i+=4) {
      let bytes = GLLib.toBytes(y[i / 4]);
      deltaA[i] = bytes[0];
      deltaA[i + 1] = bytes[1];
      deltaA[i + 2] = bytes[2];
      deltaA[i + 3] = bytes[3];
    }

    // Tell webgl which program to use.
    this.gl.useProgram(this.backward.program);
    this.gl.bindVertexArray(this.backward.vao);

    this.gl.bindTexture(this.gl.TEXTURE_2D, this.errorTexture);
    this.gl.texImage2D(this.gl.TEXTURE_2D, 0, this.gl.RGBA8, 1, y.length, 0, this.gl.RGBA, this.gl.UNSIGNED_BYTE, deltaA);
    // Backpropagate, iterate through layers.
    for (let i = this.neurons.length - 1; i >= 0; i--) {
      // Tell webgl which program to use.
      this.gl.useProgram(this.backward.program);
      this.gl.bindVertexArray(this.backward.vao);
      // Rescale canvas for trainings pass.
      this.canvas.width = this.structure[i] + 1;
      this.canvas.height = this.structure[i + 1];
      this.gl.viewport(0, 0, this.gl.canvas.width, this.gl.canvas.height);
      // Reset neuron render texture.
      this.gl.bindTexture(this.gl.TEXTURE_2D, this.tempLayerTextures[i]);
      this.gl.texImage2D(this.gl.TEXTURE_2D, 0, this.gl.RGBA8, this.canvas.width, this.canvas.height, 0, this.gl.RGBA, this.gl.UNSIGNED_BYTE, null);
      // Reset error sum render texture.
      this.gl.bindTexture(this.gl.TEXTURE_2D, this.errorSumTexture);
      this.gl.texImage2D(this.gl.TEXTURE_2D, 0, this.gl.RGBA8, this.canvas.width, this.canvas.height, 0, this.gl.RGBA, this.gl.UNSIGNED_BYTE, null);
      // Set framebuffer.
      this.gl.bindFramebuffer(this.gl.FRAMEBUFFER, this.backward.framebuffer);
      // Configure framebuffer for neuron texture and error_sum_texture.
      this.gl.drawBuffers([this.gl.COLOR_ATTACHMENT0, this.gl.COLOR_ATTACHMENT1]);
      this.gl.framebufferTexture2D(this.gl.FRAMEBUFFER, this.gl.COLOR_ATTACHMENT0, this.gl.TEXTURE_2D, this.tempLayerTextures[i], 0);
      this.gl.framebufferTexture2D(this.gl.FRAMEBUFFER, this.gl.COLOR_ATTACHMENT1, this.gl.TEXTURE_2D, this.errorSumTexture, 0);
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
      this.gl.uniform1i(this.backward.yInsteadOfErrorLocation, i === this.neurons.length - 1 ? 0 : 1);
      // Drawcall.
      this.gl.bindBuffer(this.gl.ARRAY_BUFFER, this.backward.vertexBuffer);
      this.gl.bufferData(this.gl.ARRAY_BUFFER, new Float32Array([- 1, - 1, 1, - 1, - 1, 1, - 1, 1, 1, - 1, 1, 1]), this.gl.STATIC_DRAW);

      this.gl.drawArrays(this.gl.TRIANGLES, 0, 6);

      // Switch framebuffer texture with texture in main array to update values without allocating new RAM / VRAM.
      var temp = this.layerTextures[i];
      this.layerTextures[i] = this.tempLayerTextures[i];
      this.tempLayerTextures[i] = temp;

      var results = new Uint8Array(this.neurons[i].length * 4);
      this.gl.readPixels(0, 0, this.canvas.width, this.canvas.height, this.gl.RGBA, this.gl.UNSIGNED_BYTE, results);

      for (let j = 0; j < this.neurons[i].length * 4; j+=4) {
        // Apply dx values to neural layer texture.
        this.neurons[i][j / 4] = GLLib.toFloat([results[j], results[j + 1], results[j + 2], results[j + 3]]);
      }
      // console.log("neurons: ", net.neurons[i]);
      // Rescale canvas for error summing pass.
      this.canvas.width = this.structure[i];
      this.canvas.height = 1;
      this.gl.viewport(0, 0, this.gl.canvas.width, this.gl.canvas.height);

      // Tell webgl which program to use.
      this.gl.useProgram(this.sumError.program);
      this.gl.bindVertexArray(this.sumError.vao);
      // Reset error texture.
      this.gl.bindTexture(this.gl.TEXTURE_2D, this.errorTexture);
      this.gl.texImage2D(this.gl.TEXTURE_2D, 0, this.gl.RGBA8, this.canvas.width, this.canvas.height, 0, this.gl.RGBA, this.gl.UNSIGNED_BYTE, null);

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
    for (let i = 0; i < net.neurons.length; i++) {
      let texArray = new Uint8Array(net.neurons[i].length * 4).fill(0);
      // Iterate over the net's neural layer arrays and convert them to unisgned byte testures.
      for (let j = 0; j < net.neurons[i].length; j++) {
        // Convert numbers in array to only positive numbers between 0 and 1.
        let bytes = neunet.WebGL2Lib.toBytes(net.neurons[i][j]);
        // Keep positive range and convert later in the shader, because the unsigned byte format can only hold positive numbers.
        // Increase precision by using a second 8-bit number for more detail.
        // This adds up to a total bit-depth of 15 plus, one bit for prefix.
        texArray[j * 4] = bytes[0];
        texArray[j * 4 + 1] = bytes[1];
        texArray[j * 4 + 2] = bytes[2];
        texArray[j * 4 + 3] = bytes[3];
      }
      // Prepare neurons attributes as texture for GPU.
      gl.bindTexture(gl.TEXTURE_2D, net.layerTextures[i]);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, structure[i] + 1, structure[i + 1], 0, gl.RGBA, gl.UNSIGNED_BYTE, texArray);
    }
  };
}
