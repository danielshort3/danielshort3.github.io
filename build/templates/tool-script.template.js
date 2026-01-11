(() => {
  'use strict';

  const TOOL_ID = '__TOOL_ID__';

  const markSessionDirty = () => {
    try {
      document.dispatchEvent(new CustomEvent('tools:session-dirty', { detail: { toolId: TOOL_ID } }));
    } catch {}
  };

  const getToolRoot = () => document.getElementById('main');

  const getToolSnapshotOutput = () => {
    return {
      kind: 'text',
      text: '',
      summary: ''
    };
  };

  const getToolInputs = () => {
    return {};
  };

  document.addEventListener('DOMContentLoaded', () => {
    const root = getToolRoot();
    if (!root) return;

    root.addEventListener('tools:session-capture', (event) => {
      const detail = event?.detail || {};
      if (detail.toolId !== TOOL_ID) return;

      const payload = detail.payload || {};
      const output = getToolSnapshotOutput();
      const inputs = getToolInputs();

      payload.outputSummary = String(output?.summary || payload.outputSummary || '').trim();
      payload.inputs = inputs;

      if (detail.snapshot && typeof detail.snapshot === 'object') {
        detail.snapshot.output = output;
        detail.snapshot.inputs = inputs;
      }
    });

    root.addEventListener('tools:session-applied', (event) => {
      const detail = event?.detail || {};
      if (detail.toolId !== TOOL_ID) return;
      markSessionDirty();
    });

    root.addEventListener('submit', () => markSessionDirty());
  });
})();

