import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import { buildModel, prepareData } from '../src/lib/ml.js';

const supportedActivations = ['tanh', 'relu', 'elu', 'softplus', 'swish', 'mish', 'sigmoid', 'linear'];

const baseConfig = {
  hiddenLayers: 3,
  neuronsPerLayer: 16,
  activation: 'tanh',
  learningRate: 0.01,
  epochs: 50,
};

test('buildModel omits dropout layers when dropoutRate is 0', () => {
  const model = buildModel({ ...baseConfig, dropoutRate: 0 });

  try {
    const dropoutLayers = model.layers.filter(layer => layer.getClassName() === 'Dropout');
    assert.equal(dropoutLayers.length, 0);
  } finally {
    model.dispose();
  }
});

test('buildModel adds one dropout layer after each hidden layer when dropoutRate is set', () => {
  const model = buildModel({ ...baseConfig, dropoutRate: 0.2 });

  try {
    const dropoutLayers = model.layers.filter(layer => layer.getClassName() === 'Dropout');
    assert.equal(dropoutLayers.length, baseConfig.hiddenLayers);
    assert.deepEqual(
      dropoutLayers.map(layer => layer.getConfig().rate),
      [0.2, 0.2, 0.2]
    );
  } finally {
    model.dispose();
  }
});

test('buildModel supports all activation options exposed by the UI', () => {
  for (const activation of supportedActivations) {
    const model = buildModel({ ...baseConfig, activation, dropoutRate: 0 });

    try {
      assert.equal(model.layers[0].getConfig().activation, activation);
    } finally {
      model.dispose();
    }
  }
});

test('Controls exposes the linear activation option', () => {
  const controlsSource = readFileSync(new URL('../src/components/Controls.jsx', import.meta.url), 'utf8');

  assert.match(controlsSource, /<option value="linear">Linear<\/option>/);
});

test('TeachingPanel explains each activation option exposed by the UI', () => {
  const teachingPanelSource = readFileSync(new URL('../src/components/TeachingPanel.jsx', import.meta.url), 'utf8');

  for (const activation of supportedActivations) {
    assert.match(teachingPanelSource, new RegExp(`${activation}:`));
  }
});

test('buildModel uses quadratic feature input for the linear activation option', () => {
  const model = buildModel({ ...baseConfig, activation: 'linear', dropoutRate: 0.2 });

  try {
    assert.deepEqual(model.inputs[0].shape, [null, 2]);
    assert.equal(model.layers.length, 1);
    assert.equal(model.layers[0].getConfig().activation, 'linear');
  } finally {
    model.dispose();
  }
});

test('prepareData expands linear activation inputs to x and x squared features', () => {
  const { inputs, labels } = prepareData([{ x: -10, y: 0 }, { x: 0, y: 5 }, { x: 10, y: 10 }], {
    ...baseConfig,
    activation: 'linear',
  });

  try {
    assert.deepEqual(inputs.shape, [3, 2]);
    assert.deepEqual(Array.from(inputs.dataSync()), [-1, 1, 0, 0, 1, 1]);
    assert.deepEqual(labels.shape, [3, 1]);
  } finally {
    inputs.dispose();
    labels.dispose();
  }
});
