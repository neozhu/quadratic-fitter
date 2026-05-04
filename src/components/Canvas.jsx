import { useRef, useEffect, useCallback } from 'react';
import { X_MIN, X_MAX, Y_MIN, Y_MAX } from '../lib/ml';

export default function Canvas({
  points,
  onAddPoint,
  predictionCurve,
  coefficients
}) {
  const canvasRef = useRef(null);
  const containerRef = useRef(null);
  const sizeRef = useRef({ width: 600, height: 600 });
  const animationRef = useRef(null);
  const resizeRafRef = useRef(null);
  const lastSizeRef = useRef(0);

  const mapX = useCallback((x, w) => ((x - X_MIN) / (X_MAX - X_MIN)) * w, []);
  const mapY = useCallback((y, h) => ((Y_MAX - y) / (Y_MAX - Y_MIN)) * h, []);
  const unmapX = useCallback((px, w) => (px / w) * (X_MAX - X_MIN) + X_MIN, []);
  const unmapY = useCallback((py, h) => Y_MAX - (py / h) * (Y_MAX - Y_MIN), []);

  // Resize handler — deferred via rAF to avoid ResizeObserver loop
  useEffect(() => {
    const applySize = (width, height) => {
      // Width-first: use available width, clamp to max and reasonable height
      const size = Math.max(280, Math.min(width - 24, height - 24, 700));

      // Skip no-op to prevent feedback loops
      if (Math.abs(size - lastSizeRef.current) < 1) return;
      lastSizeRef.current = size;

      sizeRef.current = { width: size, height: size };
      const canvas = canvasRef.current;
      if (canvas) {
        const dpr = window.devicePixelRatio || 1;
        canvas.width = size * dpr;
        canvas.height = size * dpr;
        canvas.style.width = `${size}px`;
        canvas.style.height = `${size}px`;
      }
    };

    const resizeObserver = new ResizeObserver(entries => {
      for (const entry of entries) {
        const { width, height } = entry.contentRect;
        // Defer size mutations to next frame to avoid loop warnings
        if (resizeRafRef.current) cancelAnimationFrame(resizeRafRef.current);
        resizeRafRef.current = requestAnimationFrame(() => applySize(width, height));
      }
    });
    if (containerRef.current) {
      resizeObserver.observe(containerRef.current);
    }
    return () => {
      resizeObserver.disconnect();
      if (resizeRafRef.current) cancelAnimationFrame(resizeRafRef.current);
    };
  }, []);

  // Main draw loop
  useEffect(() => {
    const draw = () => {
      const canvas = canvasRef.current;
      if (!canvas) return;
      const ctx = canvas.getContext('2d');
      const dpr = window.devicePixelRatio || 1;
      const { width: w, height: h } = sizeRef.current;

      ctx.save();
      ctx.scale(dpr, dpr);

      // Background
      ctx.fillStyle = '#0b1120';
      ctx.fillRect(0, 0, w, h);

      // Grid lines (minor)
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.06)';
      ctx.lineWidth = 1;
      for (let i = X_MIN; i <= X_MAX; i++) {
        const px = mapX(i, w);
        ctx.beginPath();
        ctx.moveTo(px, 0);
        ctx.lineTo(px, h);
        ctx.stroke();
      }
      for (let i = Y_MIN; i <= Y_MAX; i++) {
        const py = mapY(i, h);
        ctx.beginPath();
        ctx.moveTo(0, py);
        ctx.lineTo(w, py);
        ctx.stroke();
      }

      // Axis labels
      ctx.fillStyle = 'rgba(255, 255, 255, 0.25)';
      ctx.font = '11px Inter, sans-serif';
      ctx.textAlign = 'center';
      for (let i = X_MIN; i <= X_MAX; i += 2) {
        if (i === 0) continue;
        const px = mapX(i, w);
        const py = mapY(0, h);
        ctx.fillText(i.toString(), px, Math.min(py + 16, h - 4));
      }
      ctx.textAlign = 'right';
      for (let i = Y_MIN; i <= Y_MAX; i += 2) {
        if (i === 0) continue;
        const px = mapX(0, w);
        const py = mapY(i, h);
        ctx.fillText(i.toString(), Math.max(px - 6, 24), py + 4);
      }

      // Main axes
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.3)';
      ctx.lineWidth = 1.5;
      const originX = mapX(0, w);
      const originY = mapY(0, h);
      // Y axis
      ctx.beginPath();
      ctx.moveTo(originX, 0);
      ctx.lineTo(originX, h);
      ctx.stroke();
      // X axis
      ctx.beginPath();
      ctx.moveTo(0, originY);
      ctx.lineTo(w, originY);
      ctx.stroke();


      // Prediction curve (neural network output — the glowing green line)
      if (predictionCurve && predictionCurve.length > 0) {
        // Glow effect
        ctx.shadowBlur = 20;
        ctx.shadowColor = '#10b981';
        ctx.strokeStyle = '#10b981';
        ctx.lineWidth = 3;
        ctx.beginPath();
        for (let i = 0; i < predictionCurve.length; i++) {
          const px = mapX(predictionCurve[i].x, w);
          const py = mapY(predictionCurve[i].y, h);
          if (i === 0) ctx.moveTo(px, py);
          else ctx.lineTo(px, py);
        }
        ctx.stroke();
        ctx.shadowBlur = 0;
      }

      // Fitted quadratic curve (dashed gold, drawn ON TOP of green)
      if (coefficients && (coefficients.a !== 0 || coefficients.b !== 0 || coefficients.c !== 0)) {
        ctx.strokeStyle = 'rgba(251, 191, 36, 0.8)';
        ctx.lineWidth = 2.5;
        ctx.setLineDash([8, 5]);
        ctx.beginPath();
        let started = false;
        for (let px = 0; px <= w; px += 2) {
          const x = unmapX(px, w);
          const y = coefficients.a * x * x + coefficients.b * x + coefficients.c;
          const py = mapY(y, h);
          if (py >= -200 && py <= h + 200) {
            if (!started) {
              ctx.moveTo(px, py);
              started = true;
            } else {
              ctx.lineTo(px, py);
            }
          }
        }
        ctx.stroke();
        ctx.setLineDash([]);
      }

      // User data points
      for (const pt of points) {
        const px = mapX(pt.x, w);
        const py = mapY(pt.y, h);

        // Outer glow
        ctx.beginPath();
        ctx.arc(px, py, 8, 0, Math.PI * 2);
        ctx.fillStyle = 'rgba(236, 72, 153, 0.2)';
        ctx.fill();

        // Inner dot
        ctx.beginPath();
        ctx.arc(px, py, 5, 0, Math.PI * 2);
        ctx.fillStyle = '#ec4899';
        ctx.fill();
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.6)';
        ctx.lineWidth = 1.5;
        ctx.stroke();
      }

      // Formula overlay
      if (coefficients && (coefficients.a !== 0 || coefficients.b !== 0 || coefficients.c !== 0)) {
        const a = coefficients.a;
        const b = coefficients.b;
        const c = coefficients.c;
        const formula = `y = ${a.toFixed(3)}x² ${b >= 0 ? '+' : '−'} ${Math.abs(b).toFixed(3)}x ${c >= 0 ? '+' : '−'} ${Math.abs(c).toFixed(3)}`;

        ctx.font = 'bold 15px "JetBrains Mono", "Fira Code", monospace';
        const metrics = ctx.measureText(formula);
        const textW = metrics.width + 24;
        const textH = 32;
        const tx = w - textW - 12;
        const ty = 12;

        // Background pill
        ctx.fillStyle = 'rgba(0, 0, 0, 0.6)';
        ctx.beginPath();
        ctx.roundRect(tx, ty, textW, textH, 8);
        ctx.fill();
        ctx.strokeStyle = 'rgba(251, 191, 36, 0.4)';
        ctx.lineWidth = 1;
        ctx.stroke();

        // Text
        ctx.fillStyle = '#fbbf24';
        ctx.textAlign = 'center';
        ctx.fillText(formula, tx + textW / 2, ty + 21);
      }

      // Hint
      if (points.length === 0) {
        ctx.fillStyle = 'rgba(255, 255, 255, 0.3)';
        ctx.font = '14px Inter, sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText('Click to add training points', w / 2, 30);
      }

      ctx.restore();
      animationRef.current = requestAnimationFrame(draw);
    };

    draw();
    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [points, predictionCurve, coefficients, mapX, mapY, unmapX]);

  // Click handler
  const handleClick = (e) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const { width: w, height: h } = sizeRef.current;
    const px = (e.clientX - rect.left) * (w / rect.width);
    const py = (e.clientY - rect.top) * (h / rect.height);
    const x = unmapX(px, w);
    const y = unmapY(py, h);
    onAddPoint({ x: Math.round(x * 100) / 100, y: Math.round(y * 100) / 100 });
  };

  return (
    <div className="canvas-container" ref={containerRef}>
      <canvas
        ref={canvasRef}
        onClick={handleClick}
        style={{ cursor: 'crosshair', borderRadius: '12px' }}
      />
    </div>
  );
}
