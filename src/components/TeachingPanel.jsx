import { BrainCircuit, TrendingDown, Zap, Info, Activity } from 'lucide-react';

const activationGuide = {
  tanh: {
    label: 'Tanh',
    effect: 'Smooth and centered around zero, so it usually fits balanced quadratic curves well.',
    tip: 'Good default for this normalized canvas.'
  },
  relu: {
    label: 'ReLU',
    effect: 'Builds piecewise-linear shapes, so the curve can look sharper or less smooth.',
    tip: 'Use a smaller learning rate if the curve jumps.'
  },
  elu: {
    label: 'ELU',
    effect: 'A smoother ReLU-like option that often bends more naturally than ReLU.',
    tip: 'Useful for continuous curves.'
  },
  softplus: {
    label: 'Softplus',
    effect: 'A very smooth ReLU-like function, often producing gentle curve transitions.',
    tip: 'Good when ReLU looks too angular.'
  },
  swish: {
    label: 'Swish',
    effect: 'A smooth modern activation that can learn flexible curved shapes.',
    tip: 'Often stable with lr 0.01 or 0.005.'
  },
  mish: {
    label: 'Mish',
    effect: 'A smooth non-monotonic activation that can create expressive curved fits.',
    tip: 'Good for experimenting with smooth nonlinear fits.'
  },
  sigmoid: {
    label: 'Sigmoid',
    effect: 'Can saturate, which may flatten the prediction into a weak curve or line.',
    tip: 'Try lr 0.005 or 0.001 if it underfits.'
  },
  linear: {
    label: 'Linear',
    effect: 'Uses quadratic features [x, x²] with a linear output, so it directly learns a parabola.',
    tip: 'Best for showing linear regression on engineered quadratic features.'
  }
};

export default function TeachingPanel({
  isTraining, epoch, totalEpochs, loss, lossHistory,
  coefficients, currentActivation, trainedWith, configChanged
}) {

  const displayLoss = loss !== null ? loss.toFixed(6) : '—';
  const progressPct = totalEpochs > 0 ? Math.round((epoch / totalEpochs) * 100) : 0;
  const activationInfo = activationGuide[currentActivation] ?? activationGuide.tanh;

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
              {trainedWith.activation}, lr={trainedWith.learningRate}, drop={trainedWith.dropoutRate ?? 0},{' '}
              {trainedWith.epochs} epochs
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

      <div className="section activation-guide-section">
        <div className="section-title">
          <Activity size={16} /> Activation Effect: {activationInfo.label}
        </div>
        <div className="explanation">
          <div className="step">
            <span className="step-num">ƒ</span>
            <p>{activationInfo.effect}</p>
          </div>
          <div className="step">
            <span className="step-num">i</span>
            <p>{activationInfo.tip}</p>
          </div>
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
