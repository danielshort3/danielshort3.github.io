(function (root) {
  'use strict';

  let predictor;
  let model;
  let worker;
  let nextId = 0;
  const pending = new Map();

  const getModel = () => {
    if (!model) {
      const config = root.PizzaTipsModel;
      const boundaries = root.PizzaTipsMeta?.cityBoundaries;
      if (!config || !boundaries || !root.createPizzaTipsPredictor) {
        throw new Error('The saved model could not load. Reload the demo to try again.');
      }
      model = {
        ...config,
        categories: {
          ...config.categories,
          city: { ...config.categories.city, boundaries }
        }
      };
    }
    return model;
  };

  const unwrap = (result) => {
    if (result.statusCode !== 200) {
      const error = new Error(result.data?.error || 'The estimate could not be calculated.');
      error.details = result.data?.details;
      throw error;
    }
    return result.data;
  };

  const stopWorker = (error) => {
    worker?.terminate();
    worker = null;
    pending.forEach(({ reject, timer }) => {
      root.clearTimeout(timer);
      reject(error);
    });
    pending.clear();
  };

  const getWorker = () => {
    if (!worker) {
      const instance = new root.Worker('/js/demos/pizza-tips-worker.js');
      worker = instance;
      instance.onmessage = ({ data }) => {
        if (worker !== instance) return;
        const request = pending.get(data.id);
        if (!request) return;
        pending.delete(data.id);
        root.clearTimeout(request.timer);
        if (data.error) request.reject(new Error(data.error));
        else request.resolve(data.result);
      };
      instance.onerror = (event) => {
        if (worker !== instance) return;
        event.preventDefault();
        stopWorker(new Error('The comparison map could not be calculated. Select Retry map to try again.'));
      };
      try {
        // Reuse the already-loaded polygons; no second dataset download is needed.
        instance.postMessage({ type: 'init', model: getModel() });
      } catch (error) {
        stopWorker(error);
        throw error;
      }
    }
    return worker;
  };

  const predict = async (payload) => {
    if (!payload.grid || typeof root.Worker !== 'function') {
      predictor ||= root.createPizzaTipsPredictor(getModel());
      return unwrap(predictor.predict(payload));
    }
    const instance = getWorker();
    const result = await new Promise((resolve, reject) => {
      const id = ++nextId;
      const timer = root.setTimeout(() => {
        stopWorker(new Error('The comparison map took too long. Select Retry map to try again.'));
      }, 30000);
      pending.set(id, { resolve, reject, timer });
      try {
        instance.postMessage({ type: 'predict', id, payload });
      } catch (error) {
        stopWorker(error);
      }
    });
    return unwrap(result);
  };

  root.PizzaTipsRuntime = { predict };
})(window);
