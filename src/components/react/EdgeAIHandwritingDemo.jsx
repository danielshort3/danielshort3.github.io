import { useMemo, useRef, useState } from 'react';
import { ReactSketchCanvas } from 'react-sketch-canvas';

const DUMMY_MODEL_URL = '/models/edge-ai-handwriting-dummy.onnx';

function loadImage(src) {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = reject;
    image.src = src;
  });
}

async function canvasToTensorData(dataUrl) {
  const image = await loadImage(dataUrl);
  const canvas = document.createElement('canvas');
  canvas.width = 28;
  canvas.height = 28;

  const context = canvas.getContext('2d');
  if (!context) {
    throw new Error('Canvas context unavailable.');
  }

  context.fillStyle = '#000';
  context.fillRect(0, 0, 28, 28);
  context.drawImage(image, 0, 0, 28, 28);

  const rgba = context.getImageData(0, 0, 28, 28).data;
  const tensorData = new Float32Array(28 * 28);
  for (let i = 0; i < 28 * 28; i += 1) {
    const offset = i * 4;
    const grayscale = (rgba[offset] + rgba[offset + 1] + rgba[offset + 2]) / 3;
    tensorData[i] = grayscale / 255;
  }

  return tensorData;
}

function fallbackLabel(tensorData) {
  const density = tensorData.reduce((sum, value) => sum + value, 0) / tensorData.length;
  const label = Math.min(9, Math.floor(density * 10));
  const confidence = Math.max(0.51, Math.min(0.98, density + 0.25));
  return { label, confidence, mode: 'heuristic-placeholder' };
}

export default function EdgeAIHandwritingDemo() {
  const canvasRef = useRef(null);
  const sessionRef = useRef(null);

  const [prediction, setPrediction] = useState('Draw a character, then release pointer for inference.');
  const [isRunning, setIsRunning] = useState(false);

  const providerInfo = useMemo(() => {
    return 'ONNX Runtime Web (WASM) with heuristic fallback until model export is finalized.';
  }, []);

  const runInference = async () => {
    if (!canvasRef.current || isRunning) {
      return;
    }

    setIsRunning(true);

    try {
      const dataUrl = await canvasRef.current.exportImage('png');
      const tensorData = await canvasToTensorData(dataUrl);
      const ort = await import('onnxruntime-web');

      const inputTensor = new ort.Tensor('float32', tensorData, [1, 1, 28, 28]);

      let result = null;
      try {
        if (!sessionRef.current) {
          sessionRef.current = await ort.InferenceSession.create(DUMMY_MODEL_URL, {
            executionProviders: ['wasm']
          });
        }

        const outputMap = await sessionRef.current.run({ input: inputTensor });
        const firstOutput = outputMap[Object.keys(outputMap)[0]];
        const logits = Array.from(firstOutput.data);
        const maxLogit = Math.max(...logits);
        const label = logits.indexOf(maxLogit);
        const confidence = 1 / (1 + Math.exp(-maxLogit));
        result = { label, confidence, mode: 'onnx-dummy-model' };
      } catch {
        result = fallbackLabel(tensorData);
      }

      setPrediction(
        `Predicted class ${result.label} (${Math.round(result.confidence * 100)}% confidence) via ${result.mode}.`
      );
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Unknown inference error';
      setPrediction(`Inference unavailable: ${message}`);
    } finally {
      setIsRunning(false);
    }
  };

  const clearCanvas = async () => {
    if (!canvasRef.current) {
      return;
    }

    await canvasRef.current.clearCanvas();
    setPrediction('Canvas cleared. Draw again to test client-side inference flow.');
  };

  return (
    <section className="surface-card h-full">
      <div className="mb-4 flex items-center justify-between gap-3">
        <h3 className="text-lg font-semibold text-cyan-50">Edge AI Handwriting Recognition</h3>
        <span className="metric-badge text-cyan-100">Sub-10ms target latency</span>
      </div>

      <p className="text-sm text-cyan-100/85">
        Draw directly in-browser. On pointer release, canvas pixels are converted into a tensor and sent through
        ONNX Runtime Web.
      </p>

      <div className="mt-4 overflow-hidden rounded-xl border border-cyan-300/25" onPointerUp={runInference}>
        <ReactSketchCanvas
          ref={canvasRef}
          width="100%"
          height="240px"
          strokeWidth={14}
          strokeColor="#00bcd4"
          canvasColor="#020617"
        />
      </div>

      <div className="mt-4 flex flex-wrap items-center gap-3">
        <button type="button" className="accent-button" onClick={clearCanvas}>
          Clear Canvas
        </button>
        <span className="text-sm text-cyan-100/90">{prediction}</span>
      </div>

      <p className="mt-3 text-xs text-cyan-100/65">{providerInfo}</p>
    </section>
  );
}
