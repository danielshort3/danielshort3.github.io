'use strict';

const AI_USER_AGENT_TOKENS = [
  'GPTBot',
  'ChatGPT-User',
  'OAI-SearchBot',
  'ClaudeBot',
  'Claude-User',
  'Claude-SearchBot',
  'anthropic-ai',
  'PerplexityBot',
  'Perplexity-User',
  'Perplexity',
  'Google-Extended',
  'GoogleOther',
  'GoogleOther-Image',
  'GoogleOther-Video',
  'Applebot-Extended',
  'Bytespider',
  'Meta-ExternalAgent',
  'Meta-ExternalFetcher',
  'FacebookBot',
  'CCBot',
  'cohere-ai',
  'Diffbot',
  'YouBot',
  'Timpibot',
  'PhindBot',
  'DuckAssistBot',
  'MistralAI-User',
  'Amazonbot',
  'omgili',
  'omgilibot',
  'ImagesiftBot'
];

function escapeRegex(value) {
  return String(value || '').replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

const AI_USER_AGENT_VALUE = `.*(?:${AI_USER_AGENT_TOKENS.map(escapeRegex).join('|')}).*`;
const AI_USER_AGENT_RE = new RegExp(AI_USER_AGENT_TOKENS.map(escapeRegex).join('|'), 'i');

function isAiUserAgent(value) {
  return AI_USER_AGENT_RE.test(String(value || ''));
}

module.exports = {
  AI_USER_AGENT_TOKENS,
  AI_USER_AGENT_VALUE,
  isAiUserAgent
};
