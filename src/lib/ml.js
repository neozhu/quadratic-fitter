import * as tf from '@tensorflow/tfjs';

// Coordinate system bounds (for canvas display)
export const X_MIN = -10;
export const X_MAX = 10;
export const Y_MIN = -10;
export const Y_MAX = 10;

/**
 * Normalize x from [X_MIN, X_MAX] to [-1, 1].
 * Using [-1, 1] instead of [0, 1] is critical for tanh/sigmoid activations.
 */
export function normalizeX(x) {
  return 2 * (x - X_MIN) / (X_MAX - X_MIN) - 1;
}

export function denormalizeX(normX) {
  return ((normX + 1) / 2) * (X_MAX - X_MIN) + X_MIN;
}

export function normalizeY(y) {
  return 2 * (y - Y_MIN) / (Y_MAX - Y_MIN) - 1;
}

export function denormalizeY(normY) {
  return ((normY + 1) / 2) * (Y_MAX - Y_MIN) + Y_MIN;
}

function isLinearQuadraticModel(config) {
  return config.activation === 'linear';
}

function modelInputSize(model) {
  return model.inputs?.[0]?.shape?.[1] ?? 1;
}

function normalizedXsToInputTensor(xs, inputSize) {
  if (inputSize === 2) {
    const xArray = Array.from(xs.dataSync());
    const features = xArray.flatMap(x => [x, x * x]);
    return tf.tensor2d(features, [xArray.length, 2]);
  }

  return xs.reshape([xs.shape[0], 1]);
}

/**
 * Build a neural network model based on user config.
 * Input: [x] (normalized to [-1, 1])
 * Output: [y] (normalized to [-1, 1])
 */
export function buildModel(config) {
  const model = tf.sequential();
  const dropoutRate = Number(config.dropoutRate) || 0;

  if (isLinearQuadraticModel(config)) {
    model.add(tf.layers.dense({
      inputShape: [2],
      units: 1,
      activation: 'linear'
    }));

    model.compile({
      optimizer: tf.train.adam(config.learningRate),
      loss: 'meanSquaredError'
    });

    return model;
  }

  // First hidden layer
  model.add(tf.layers.dense({
    inputShape: [1],
    units: config.neuronsPerLayer,
    activation: config.activation,
    kernelInitializer: 'glorotNormal'
  }));
  if (dropoutRate > 0) {
    model.add(tf.layers.dropout({ rate: dropoutRate }));
  }

  // Additional hidden layers
  for (let i = 1; i < config.hiddenLayers; i++) {
    model.add(tf.layers.dense({
      units: config.neuronsPerLayer,
      activation: config.activation,
      kernelInitializer: 'glorotNormal'
    }));
    if (dropoutRate > 0) {
      model.add(tf.layers.dropout({ rate: dropoutRate }));
    }
  }

  // Output layer — linear for regression
  model.add(tf.layers.dense({
    units: 1,
    activation: 'linear'
  }));

  model.compile({
    optimizer: tf.train.adam(config.learningRate),
    loss: 'meanSquaredError'
  });

  return model;
}

/**
 * Prepare training data from user-clicked points.
 * Normalizes both x and y to [-1, 1].
 */
export function prepareData(points, config = {}) {
  const xs = points.map(p => normalizeX(p.x));
  const ys = points.map(p => normalizeY(p.y));
  const inputSize = isLinearQuadraticModel(config) ? 2 : 1;
  const features = inputSize === 2
    ? xs.flatMap(x => [x, x * x])
    : xs;

  return {
    inputs: tf.tensor2d(features, [xs.length, inputSize]),
    labels: tf.tensor2d(ys, [ys.length, 1])
  };
}

/**
 * Generate the prediction curve from the model.
 * Samples normalized x from [-1, 1], predicts y, denormalizes both.
 */
export function predictCurve(model, numPoints = 200) {
  if (!model) return [];

  return tf.tidy(() => {
    // Sample x in normalized space [-1, 1]
    const xs = tf.linspace(-1, 1, numPoints);
    const inputs = normalizedXsToInputTensor(xs, modelInputSize(model));
    const preds = model.predict(inputs);

    const xArray = xs.dataSync();
    const yArray = preds.dataSync();

    const curvePoints = [];
    for (let i = 0; i < numPoints; i++) {
      curvePoints.push({
        x: denormalizeX(xArray[i]),
        y: denormalizeY(yArray[i])
      });
    }
    return curvePoints;
  });
}

/**
 * Extract quadratic coefficients (a, b, c) from the model's prediction curve.
 * Uses least-squares fitting: y = ax² + bx + c
 */
export function extractQuadraticCoeffs(model, numSamples = 200) {
  if (!model) return { a: 0, b: 0, c: 0 };

  return tf.tidy(() => {
    const xs = tf.linspace(-1, 1, numSamples);
    const inputs = normalizedXsToInputTensor(xs, modelInputSize(model));
    const preds = model.predict(inputs);

    const xArr = Array.from(xs.dataSync()).map(nx => denormalizeX(nx));
    const yArr = Array.from(preds.dataSync()).map(ny => denormalizeY(ny));

    // Solve least-squares: y = ax² + bx + c via normal equations
    const n = xArr.length;
    let sx1 = 0, sx2 = 0, sx3 = 0, sx4 = 0;
    let sy = 0, sxy = 0, sx2y = 0;

    for (let i = 0; i < n; i++) {
      const x = xArr[i];
      const y = yArr[i];
      const x2 = x * x;
      sx1 += x;
      sx2 += x2;
      sx3 += x2 * x;
      sx4 += x2 * x2;
      sy += y;
      sxy += x * y;
      sx2y += x2 * y;
    }

    // Cramer's rule for 3x3 system
    const A = [
      [sx4, sx3, sx2],
      [sx3, sx2, sx1],
      [sx2, sx1, n]
    ];
    const B = [sx2y, sxy, sy];

    const det3 = (m) =>
      m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) -
      m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) +
      m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);

    const detA = det3(A);
    if (Math.abs(detA) < 1e-12) return { a: 0, b: 0, c: 0 };

    const Aa = [
      [B[0], A[0][1], A[0][2]],
      [B[1], A[1][1], A[1][2]],
      [B[2], A[2][1], A[2][2]]
    ];
    const Ab = [
      [A[0][0], B[0], A[0][2]],
      [A[1][0], B[1], A[1][2]],
      [A[2][0], B[2], A[2][2]]
    ];
    const Ac = [
      [A[0][0], A[0][1], B[0]],
      [A[1][0], A[1][1], B[1]],
      [A[2][0], A[2][1], B[2]]
    ];

    return {
      a: det3(Aa) / detA,
      b: det3(Ab) / detA,
      c: det3(Ac) / detA
    };
  });
}
