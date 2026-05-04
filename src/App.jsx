import { useState, useRef } from 'react';
import Controls from './components/Controls';
import Canvas from './components/Canvas';
import TeachingPanel from './components/TeachingPanel';
import { buildModel, prepareData, predictCurve, extractQuadraticCoeffs } from './lib/ml';

export default function App() {
  // User-configurable ML parameters
  const [config, setConfig] = useState({
    hiddenLayers: 2,
    neuronsPerLayer: 64,
    activation: 'tanh',
    learningRate: 0.1,
    dropoutRate: 0,
    epochs: 500,
  });

  // Data points clicked by user
  const [points, setPoints] = useState([]);

  // Training state
  const [isTraining, setIsTraining] = useState(false);
  const [epoch, setEpoch] = useState(0);
  const [loss, setLoss] = useState(null);
  const [lossHistory, setLossHistory] = useState([]);
  const [predictionCurve, setPredictionCurve] = useState([]);
  const [coefficients, setCoefficients] = useState({ a: 0, b: 0, c: 0 });

  // Track what config was used for the current/last training run
  const [trainedWith, setTrainedWith] = useState(null);

  const stopTrainingRef = useRef(false);
  const modelRef = useRef(null);

  const handleAddPoint = (pt) => {
    setPoints(prev => [...prev, pt]);
  };

  const handleUndoPoint = () => {
    setPoints(prev => prev.slice(0, -1));
  };

  const handleClearPoints = () => {
    setPoints([]);
    // Also reset curves and coefficients when clearing points
    handleResetModel();
  };

  const handleResetModel = () => {
    if (modelRef.current) {
      modelRef.current.dispose();
      modelRef.current = null;
    }
    setEpoch(0);
    setLoss(null);
    setLossHistory([]);
    setCoefficients({ a: 0, b: 0, c: 0 });
    setPredictionCurve([]);
    setTrainedWith(null);
  };

  /**
   * Start Training:
   * Build-then-dispose: build new model first, only dispose old on success.
   * If build fails (bad config), old model & curves are preserved.
   */
  const handleStartTraining = async () => {
    if (points.length < 3) return;

    // Build fresh model with CURRENT config — do this BEFORE disposing old model
    let freshModel;
    try {
      freshModel = buildModel(config);
    } catch (buildError) {
      console.error("Failed to build model:", buildError);
      return;
    }

    // Build succeeded — now safe to dispose old model
    if (modelRef.current) {
      modelRef.current.dispose();
      modelRef.current = null;
    }
    modelRef.current = freshModel;

    setIsTraining(true);
    stopTrainingRef.current = false;
    setEpoch(0);
    setLoss(null);
    setLossHistory([]);

    // Record what config this training uses
    setTrainedWith({ ...config });

    const { inputs, labels } = prepareData(points, config);

    try {
      await freshModel.fit(inputs, labels, {
        epochs: config.epochs,
        batchSize: Math.min(32, points.length),
        shuffle: true,
        yieldEvery: 'epoch',
        callbacks: {
          onEpochEnd: async (e, logs) => {
            if (stopTrainingRef.current) {
              freshModel.stopTraining = true;
            }
            setEpoch(e + 1);
            setLoss(logs.loss);
            setLossHistory(prev => [...prev, logs.loss]);

            // Update prediction curve every 5 epochs for performance
            if (e % 5 === 0 || e === config.epochs - 1) {
              const curve = predictCurve(freshModel);
              setPredictionCurve(curve);
              const coeffs = extractQuadraticCoeffs(freshModel);
              setCoefficients(coeffs);
            }
          }
        }
      });

      // Final update after training completes
      const curve = predictCurve(freshModel);
      setPredictionCurve(curve);
      const coeffs = extractQuadraticCoeffs(freshModel);
      setCoefficients(coeffs);
    } catch (error) {
      console.error("Training error:", error);
    }

    inputs.dispose();
    labels.dispose();
    setIsTraining(false);
  };

  const handleStopTraining = () => {
    stopTrainingRef.current = true;
  };

  // Detect if config has changed since last training (for UI indicator)
  const configChanged = trainedWith !== null && (
    trainedWith.hiddenLayers !== config.hiddenLayers ||
    trainedWith.neuronsPerLayer !== config.neuronsPerLayer ||
    trainedWith.activation !== config.activation ||
    trainedWith.learningRate !== config.learningRate ||
    trainedWith.dropoutRate !== config.dropoutRate ||
    trainedWith.epochs !== config.epochs
  );

  return (
    <div className="app-container">
      <Controls
        config={config}
        setConfig={setConfig}
        isTraining={isTraining}
        onStartTraining={handleStartTraining}
        onStopTraining={handleStopTraining}
        onResetModel={handleResetModel}
        pointsCount={points.length}
        onUndoPoint={handleUndoPoint}
        onClearPoints={handleClearPoints}
        configChanged={configChanged}
      />

      <main className="main-content">
        <Canvas
          points={points}
          onAddPoint={handleAddPoint}
          predictionCurve={predictionCurve}
          coefficients={coefficients}
        />
      </main>

      <TeachingPanel
        isTraining={isTraining}
        epoch={epoch}
        totalEpochs={config.epochs}
        loss={loss}
        lossHistory={lossHistory}
        coefficients={coefficients}
        currentActivation={config.activation}
        trainedWith={trainedWith}
        configChanged={configChanged}
      />
    </div>
  );
}
