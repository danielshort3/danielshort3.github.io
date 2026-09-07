'use strict';

importScripts('pizza-tips-predictor.js');

let predictor;
self.onmessage = ({ data }) => {
  if (data.type === 'init') {
    predictor = self.createPizzaTipsPredictor(data.model);
    return;
  }
  if (data.type !== 'predict') return;
  try {
    self.postMessage({ id: data.id, result: predictor.predict(data.payload) });
  } catch (error) {
    self.postMessage({ id: data.id, error: String(error.message || error) });
  }
};
