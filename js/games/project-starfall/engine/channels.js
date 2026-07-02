(function initProjectStarfallEngineChannels(global) {
  'use strict';

  const CoreIds = (typeof require === 'function' ? require('../core/ids.js') : null) || global.ProjectStarfallCore || {};
  const normalizeId = CoreIds.normalizeId || function normalizeIdFallback(value) {
    return String(value || '').trim();
  };

  const MAP_CHANNEL_COUNT = 8;
  const DEFAULT_MAP_CHANNEL_ID = 'ch1';
  const MAP_CHANNELS = Object.freeze(Array.from({ length: MAP_CHANNEL_COUNT }, (_, index) => Object.freeze({
    id: `ch${index + 1}`,
    label: `Ch. ${index + 1}`,
    name: `Channel ${index + 1}`
  })));

  function normalizeMapChannelId(channelId) {
    const id = normalizeId(channelId) || DEFAULT_MAP_CHANNEL_ID;
    return MAP_CHANNELS.some((channel) => channel.id === id) ? id : DEFAULT_MAP_CHANNEL_ID;
  }

  function getMapChannelLabel(channelId) {
    const id = normalizeMapChannelId(channelId);
    const channel = MAP_CHANNELS.find((candidate) => candidate.id === id);
    return channel ? channel.label : 'Ch. 1';
  }

  function createMapChannelSnapshot(channelId) {
    const currentId = normalizeMapChannelId(channelId);
    return {
      currentId,
      currentLabel: getMapChannelLabel(currentId),
      channels: MAP_CHANNELS.map((channel) => ({
        id: channel.id,
        label: channel.label,
        name: channel.name,
        current: channel.id === currentId
      }))
    };
  }

  const api = {
    MAP_CHANNEL_COUNT,
    DEFAULT_MAP_CHANNEL_ID,
    MAP_CHANNELS,
    normalizeMapChannelId,
    getMapChannelLabel,
    createMapChannelSnapshot
  };

  const modules = global.ProjectStarfallEngineModules || {};
  modules.channels = Object.assign({}, modules.channels || {}, api);
  global.ProjectStarfallEngineModules = modules;

  if (typeof module === 'object' && module.exports) {
    module.exports = api;
  }
})(typeof window !== 'undefined' ? window : globalThis);
