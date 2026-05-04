import React from 'react';
import { Settings2, RotateCcw, Play, Square, Trash2, Undo2 } from 'lucide-react';

export default function Controls({
  config,
  setConfig,
  isTraining,
  onStartTraining,
  onStopTraining,
  onResetModel,
  pointsCount,
  onUndoPoint,
  onClearPoints,
  configChanged,
}) {

  const numericFields = new Set(['hiddenLayers', 'neuronsPerLayer', 'learningRate', 'epochs']);

  const handleConfigChange = (e) => {
    const { name, value } = e.target;
    setConfig(prev => ({
      ...prev,
      [name]: numericFields.has(name) ? Number(value) : value
    }));
  };

  return (
    <div className="sidebar">
      <div className="header">
        <h1>🧠 Quadratic Fitter</h1>
        <p>Machine Learning — y = ax² + bx + c</p>
      </div>

      {/* Neural Network Architecture */}
      <div className="section">
        <div className="section-title">
          <Settings2 size={16} /> Network Architecture
        </div>

        <div className="control-group">
          <label htmlFor="ctrl-hiddenLayers">Hidden Layers</label>
          <div className="slider-row">
            <input
              id="ctrl-hiddenLayers"
              type="range"
              name="hiddenLayers"
              min="1" max="5" step="1"
              value={config.hiddenLayers}
              onChange={handleConfigChange}
              disabled={isTraining}
            />
            <span className="slider-value">{config.hiddenLayers}</span>
          </div>
        </div>

        <div className="control-group">
          <label htmlFor="ctrl-neuronsPerLayer">Neurons per Layer</label>
          <div className="slider-row">
            <input
              id="ctrl-neuronsPerLayer"
              type="range"
              name="neuronsPerLayer"
              min="4" max="64" step="4"
              value={config.neuronsPerLayer}
              onChange={handleConfigChange}
              disabled={isTraining}
            />
            <span className="slider-value">{config.neuronsPerLayer}</span>
          </div>
        </div>

        <div className="control-group">
          <label htmlFor="ctrl-activation">Activation Function</label>
          <select
            id="ctrl-activation"
            name="activation"
            value={config.activation}
            onChange={handleConfigChange}
            disabled={isTraining}
          >
            <option value="relu">ReLU</option>
            <option value="tanh">Tanh</option>
            <option value="sigmoid">Sigmoid</option>
            <option value="elu">ELU</option>
          </select>
        </div>
      </div>

      {/* Training Parameters */}
      <div className="section">
        <div className="section-title">
          <Play size={16} /> Training Parameters
        </div>

        <div className="control-group">
          <label htmlFor="ctrl-learningRate">Learning Rate</label>
          <select
            id="ctrl-learningRate"
            name="learningRate"
            value={config.learningRate}
            onChange={handleConfigChange}
            disabled={isTraining}
          >
            <option value={0.1}>0.1 (Fast)</option>
            <option value={0.05}>0.05</option>
            <option value={0.01}>0.01</option>
            <option value={0.005}>0.005</option>
            <option value={0.001}>0.001 (Slow)</option>
          </select>
        </div>

        <div className="control-group">
          <label htmlFor="ctrl-epochs">Epochs</label>
          <div className="slider-row">
            <input
              id="ctrl-epochs"
              type="range"
              name="epochs"
              min="50" max="1000" step="50"
              value={config.epochs}
              onChange={handleConfigChange}
              disabled={isTraining}
            />
            <span className="slider-value">{config.epochs}</span>
          </div>
        </div>
      </div>

      {/* Training Actions */}
      <div className="section">
        <div className="control-group" style={{ marginBottom: 0 }}>
          {!isTraining ? (
            <button
              className={`btn primary ${configChanged ? 'config-changed' : ''}`}
              onClick={onStartTraining}
              disabled={pointsCount < 3}
            >
              <Play size={18} />
              {configChanged ? 'Retrain with New Config' : 'Start Training'}
            </button>
          ) : (
            <button className="btn danger" onClick={onStopTraining}>
              <Square size={18} /> Stop Training
            </button>
          )}
          <button className="btn secondary" onClick={onResetModel} disabled={isTraining}>
            <RotateCcw size={18} /> Reset Model
          </button>
        </div>
        {pointsCount < 3 && (
          <p className="hint-text">Click at least 3 points on the canvas</p>
        )}
      </div>

      {/* Data Points */}
      <div className="section">
        <div className="section-title">
          Data Points ({pointsCount})
        </div>
        <div className="row">
          <button className="btn secondary" onClick={onUndoPoint} disabled={pointsCount === 0 || isTraining}>
            <Undo2 size={16} /> Undo
          </button>
          <button className="btn secondary" onClick={onClearPoints} disabled={pointsCount === 0 || isTraining}>
            <Trash2 size={16} /> Clear All
          </button>
        </div>
      </div>
    </div>
  );
}
