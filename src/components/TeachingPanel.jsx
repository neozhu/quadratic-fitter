import React from 'react';
import { BrainCircuit, TrendingDown, Zap, Info } from 'lucide-react';

export default function TeachingPanel({
  isTraining, epoch, totalEpochs, loss, lossHistory,
  coefficients, trainedWith, configChanged
}) {

  const displayLoss = loss !== null ? loss.toFixed(6) : '—';
  const progressPct = totalEpochs > 0 ? Math.round((epoch / totalEpochs) * 100) : 0;

  // Mini loss chart
  const renderLossChart = () => {
    if (!lossHistory || lossHistory.length < 2) return null;

    const chartW = 280;
    const chartH = 80;
    const maxLoss = Math.max(...lossHistory);
    const minLoss = Math.min(...lossHistory);
    const range = maxLoss - minLoss || 1;

    const pathPoints = lossHistory.map((l, i) => {
      const x = (i / (lossHistory.length - 1)) * chartW;
      const y = chartH - ((l - minLoss) / range) * (chartH - 8) - 4;
      return `${x},${y}`;
    });

    return (
      <div className="loss-chart-container">
        <svg width={chartW} height={chartH} viewBox={`0 0 ${chartW} ${chartH}`}>
          <defs>
            <linearGradient id="lossGrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#f59e0b" stopOpacity="0.3" />
              <stop offset="100%" stopColor="#f59e0b" stopOpacity="0.02" />
            </linearGradient>
          </defs>
          {/* Area fill */}
          <path
            d={`M0,${chartH} L${pathPoints.join(' L')} L${chartW},${chartH} Z`}
            fill="url(#lossGrad)"
          />
          {/* Line */}
          <polyline
            points={pathPoints.join(' ')}
            fill="none"
            stroke="#f59e0b"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        </svg>
      </div>
    );
  };

  return (
    <div className="right-panel">
      {/* Training Status */}
      <div className={`section ${isTraining ? 'training-active' : ''}`}>
        <div className="section-title">
          <BrainCircuit size={16} /> Training Status
          {isTraining && <span className="badge training">TRAINING</span>}
        </div>

        <div className="stat-grid">
          <div className="stat-box">
            <div className="stat-value">{epoch}</div>
            <div className="stat-label">Epoch</div>
          </div>
          <div className="stat-box">
            <div className="stat-value loss">{displayLoss}</div>
            <div className="stat-label">Loss (MSE)</div>
          </div>
        </div>

        {/* Progress bar */}
        <div className="progress-track">
          <div className="progress-fill" style={{ width: `${progressPct}%` }} />
        </div>
        <div className="progress-label">{progressPct}% complete</div>

        {/* Mini loss chart */}
        {renderLossChart()}

        {/* Show what config was used for this training */}
        {trainedWith && !isTraining && (
          <div className="trained-with-info">
            <Info size={12} />
            <span>
              Trained: {trainedWith.hiddenLayers} layers × {trainedWith.neuronsPerLayer} neurons,{' '}
              {trainedWith.activation}, lr={trainedWith.learningRate}, {trainedWith.epochs} epochs
            </span>
          </div>
        )}
      </div>

      {/* Fitted Coefficients — the KEY feature */}
      <div className={`section coefficients-section ${configChanged ? 'stale' : ''}`}>
        <div className="section-title">
          <Zap size={16} /> Fitted Quadratic: y = ax² + bx + c
        </div>

        <div className="coeff-row">
          <div className="coeff-label">a</div>
          <div className="coeff-bar-container">
            <div
              className="coeff-bar a-bar"
              style={{ width: `${Math.min(Math.abs(coefficients.a) * 20, 100)}%` }}
            />
          </div>
          <div className="coeff-value a-color">{coefficients.a.toFixed(4)}</div>
        </div>
        <div className="coeff-row">
          <div className="coeff-label">b</div>
          <div className="coeff-bar-container">
            <div
              className="coeff-bar b-bar"
              style={{ width: `${Math.min(Math.abs(coefficients.b) * 10, 100)}%` }}
            />
          </div>
          <div className="coeff-value b-color">{coefficients.b.toFixed(4)}</div>
        </div>
        <div className="coeff-row">
          <div className="coeff-label">c</div>
          <div className="coeff-bar-container">
            <div
              className="coeff-bar c-bar"
              style={{ width: `${Math.min(Math.abs(coefficients.c) * 10, 100)}%` }}
            />
          </div>
          <div className="coeff-value c-color">{coefficients.c.toFixed(4)}</div>
        </div>

        <div className="formula-display">
          y = <span className="a-color">{coefficients.a.toFixed(3)}</span>x²
          {coefficients.b >= 0 ? ' + ' : ' − '}
          <span className="b-color">{Math.abs(coefficients.b).toFixed(3)}</span>x
          {coefficients.c >= 0 ? ' + ' : ' − '}
          <span className="c-color">{Math.abs(coefficients.c).toFixed(3)}</span>
        </div>
      </div>

      {/* Instructions */}
      <div className="section instructions-section">
        <div className="section-title">
          <TrendingDown size={16} /> How It Works
        </div>
        <div className="explanation">
          <div className="step">
            <span className="step-num">1</span>
            <p><strong>Click</strong> on the canvas to place training points</p>
          </div>
          <div className="step">
            <span className="step-num">2</span>
            <p><strong>Configure</strong> the neural network (layers, activation, learning rate)</p>
          </div>
          <div className="step">
            <span className="step-num">3</span>
            <p><strong>Train</strong> — watch the <span className="highlight-green">green curve</span> fit your points</p>
          </div>
          <div className="step">
            <span className="step-num">4</span>
            <p><strong>Compare</strong> — change params and retrain to see how different configs fit the same data</p>
          </div>
        </div>
      </div>
    </div>
  );
}
