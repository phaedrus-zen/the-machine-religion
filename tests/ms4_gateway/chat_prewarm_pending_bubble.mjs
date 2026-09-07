import { readFileSync } from 'node:fs';
import vm from 'node:vm';
import { pathToFileURL } from 'node:url';

const htmlPath = process.argv[2];
const mode = process.argv[3] || 'prewarm-503';
const html = readFileSync(htmlPath, 'utf8');

function extractFunction(source, name) {
  const token = `function ${name}(`;
  const asyncToken = `async function ${name}(`;
  let start = source.indexOf(asyncToken);
  if (start < 0) start = source.indexOf(token);
  if (start < 0) throw new Error(`missing function ${name}`);
  const openParen = source.indexOf('(', start);
  if (openParen < 0) throw new Error(`missing signature for ${name}`);
  const closeParen = matchPair(source, openParen, '(', ')');
  const openBrace = skipSpace(source, closeParen + 1);
  if (source[openBrace] !== '{') throw new Error(`missing body for ${name}`);
  const closeBrace = matchPair(source, openBrace, '{', '}');
  return source.slice(start, closeBrace + 1);
}

function skipSpace(source, index) {
  let i = index;
  while (i < source.length && /\s/.test(source[i])) i += 1;
  return i;
}

function matchPair(source, openIndex, openCh, closeCh) {
  let depth = 0;
  let quote = null;
  let escaped = false;
  const templateDepth = [];
  for (let i = openIndex; i < source.length; i += 1) {
    const ch = source[i];
    const next = source[i + 1];
    if (quote === '//') {
      if (ch === '\n') quote = null;
      continue;
    }
    if (quote === '/*') {
      if (ch === '*' && next === '/') {
        quote = null;
        i += 1;
      }
      continue;
    }
    if (quote === '`') {
      if (escaped) {
        escaped = false;
        continue;
      }
      if (ch === '\\') {
        escaped = true;
        continue;
      }
      if (ch === '`') {
        quote = null;
        continue;
      }
      if (ch === '$' && next === '{') {
        quote = null;
        templateDepth.push(depth);
        i += 1;
        if (openCh === '{') depth += 1;
        continue;
      }
      continue;
    }
    if (quote) {
      if (escaped) {
        escaped = false;
        continue;
      }
      if (ch === '\\') {
        escaped = true;
        continue;
      }
      if (ch === quote) quote = null;
      continue;
    }
    if (ch === '/' && next === '/') {
      quote = '//';
      i += 1;
      continue;
    }
    if (ch === '/' && next === '*') {
      quote = '/*';
      i += 1;
      continue;
    }
    if (ch === "'" || ch === '"' || ch === '`') {
      quote = ch;
      continue;
    }
    if (ch === openCh) depth += 1;
    else if (ch === closeCh) {
      depth -= 1;
      if (
        templateDepth.length
        && depth === templateDepth[templateDepth.length - 1]
      ) {
        templateDepth.pop();
        quote = '`';
        continue;
      }
      if (depth === 0) return i;
    }
  }
  throw new Error(`unterminated ${openCh}${closeCh} pair`);
}

function el(tag, id) {
  const children = [];
  const listeners = {};
  const dataset = {};
  const classSet = new Set();
  let text = '';
  const node = {
    tagName: String(tag || 'div').toUpperCase(),
    id: id || '',
    children,
    dataset,
    disabled: false,
    value: '',
    checked: true,
    nodeValue: '',
    className: '',
    scrollTop: 0,
    scrollHeight: 0,
    style: {},
    appendChild(child) {
      children.push(child);
      return child;
    },
    addEventListener(type, fn) {
      listeners[type] = fn;
    },
    dispatchEvent(event) {
      const fn = listeners[event.type];
      if (fn) return fn(event);
      return undefined;
    },
    querySelector(sel) {
      if (sel === '.send-state') {
        return children.find((c) => String(c.className).split(/\s+/).includes('send-state')) || null;
      }
      return null;
    },
    querySelectorAll(sel) {
      if (sel === '.msg.assistant') return children.filter((c) => String(c.className).includes('assistant'));
      if (sel === '.msg.user') return children.filter((c) => String(c.className).includes('user'));
      return [];
    },
  };
  Object.defineProperty(node, 'textContent', {
    get() {
      return `${text}${children.map((c) => c.textContent || c.nodeValue || '').join('')}`;
    },
    set(v) {
      text = String(v ?? '');
      children.length = 0;
    },
  });
  Object.defineProperty(node, 'classList', {
    value: {
      add: (...names) => names.forEach((n) => classSet.add(n)),
      remove: (...names) => names.forEach((n) => classSet.delete(n)),
      contains: (n) => classSet.has(n) || String(node.className).split(/\s+/).includes(n),
    },
  });
  return node;
}

const messages = el('div', 'messages');
const messageInput = el('input', 'messageInput');
const sendButton = el('button', 'sendButton');
const streamToggle = el('input', 'streamToggle');
streamToggle.checked = true;
const chatForm = el('form', 'chatForm');
const statusEl = el('div', 'status');
const sessionEl = el('span', 'session');
const modelSelect = el('select', 'modelSelect');
modelSelect.value = 'nemotron-3-nano:4b';
const depthModelSelect = el('select', 'depthModelSelect');
depthModelSelect.value = '';
const ids = {
  messages,
  messageInput,
  sendButton,
  streamToggle,
  chatForm,
  status: statusEl,
  session: sessionEl,
  modelSelect,
  depthModelSelect,
};

const calls = { prewarm: [], chat: [], tts: [], recent: [] };
const store = {};
let lastChatTerminal = 'none';

function jsonResponse(status, body) {
  return {
    ok: status >= 200 && status < 300,
    status,
    statusText: status === 503 ? 'Service Unavailable' : 'OK',
    json: async () => body,
    body: null,
  };
}

function sseResponse(text) {
  const encoder = new TextEncoder();
  const bytes = encoder.encode(text);
  let sent = false;
  return {
    ok: true,
    status: 200,
    statusText: 'OK',
    body: {
      getReader: () => ({
        async read() {
          if (!sent) {
            sent = true;
            return { value: bytes, done: false };
          }
          return { value: undefined, done: true };
        },
        cancel: async () => {},
      }),
    },
  };
}

const context = vm.createContext({
  console,
  setTimeout,
  clearTimeout,
  AbortController,
  TextDecoder,
  JSON,
  String,
  Boolean,
  Error,
  Promise,
  document: {
    getElementById: (id) => ids[id] || el('div', id),
    createElement: (tag) => el(tag),
    createTextNode: (text) => {
      const node = el('span');
      node.nodeValue = String(text || '');
      node.textContent = String(text || '');
      return node;
    },
  },
  localStorage: {
    getItem: (k) => (Object.prototype.hasOwnProperty.call(store, k) ? store[k] : null),
    setItem: (k, v) => { store[k] = String(v); },
    removeItem: (k) => { delete store[k]; },
  },
  fetch: async (url, options = {}) => {
    const target = String(url);
    if (target.includes('/voice/prewarm/face')) {
      calls.prewarm.push(target);
      if (mode === 'prewarm-503') {
        return jsonResponse(503, { error: 'selected Face model prewarm failed', warmed: false });
      }
      return jsonResponse(200, { warmed: true, frozen_model: 'nemotron-3-nano:4b' });
    }
    if (target.includes('/chat/stream') || target === '/chat') {
      calls.chat.push(target);
      return sseResponse(
        'event: done\ndata: {"text":"Exact warm route.","completed":true,"cancelled":false,"session_id":"s1"}\n\n',
      );
    }
    if (target.includes('/voice/recent-turns')) {
      calls.recent.push(target);
      return jsonResponse(200, { turns: [] });
    }
    if (target.includes('/voice/') || target.includes('tts') || target.includes('/audio')) {
      calls.tts.push(target);
      return jsonResponse(200, {});
    }
    return jsonResponse(404, { error: target });
  },
  sessionId: 's0',
  sessionEl,
  statusEl,
  messages,
  messageInput,
  sendButton,
  streamToggle,
  modelSelect,
  depthModelSelect,
  currentVoiceTurnId: 1,
  activeChatTurnState: null,
  modelCatalogReady: true,
  modelCatalogReadyPromise: null,
  selectedFacePrewarmState: null,
  selectedFacePrewarmGeneration: 0,
  oracleVoicePageInstanceId: 'test-page',
  FACE_MODEL_INVALIDATION_TIMEOUT_MS: 5000,
  bargeIn: () => {},
  setOracleStageTranscript: () => {},
  setOracleStageState: () => {},
  isOracleAbortError: () => false,
  isSupersededFacePrewarm: () => false,
  supersededFacePrewarmError: () => {
    const error = new Error('superseded');
    error.name = 'AbortError';
    error.facePrewarmSuperseded = true;
    return error;
  },
  selectedFaceModelId: () => String(modelSelect.value || ''),
  awaitSelectedFaceModelPrewarm: async () => {
    calls.prewarm.push('/voice/prewarm/face');
    if (mode === 'prewarm-503') {
      throw new Error('selected Face model prewarm failed');
    }
    return { warmed: true, frozen_model: 'nemotron-3-nano:4b' };
  },
  speakTextChatReply: async (...args) => {
    calls.tts.push(['speakTextChatReply', ...args]);
    return false;
  },
  adoptAssistantRoutingMeta: () => {},
  renderAssistantMeta: () => {},
  finishAssistantMessage: (elNode) => elNode,
  markStreamingAssistantIncomplete: () => {},
  reportChatTurnFailure: () => {},
  lastChatTerminal: 'none',
  handleSseEvent: async (eventName, eventData, state) => {
    if (eventName === 'done') {
      state.text = String(eventData.text || '');
      if (state.textNode) state.textNode.nodeValue = state.text;
      if (state.el) state.el.textContent = state.text;
      if (state.turnState) state.turnState.completed = true;
      lastChatTerminal = 'done';
      return 'done';
    }
    return false;
  },
});

const required = [
  'addMessage',
  'addUserMessagePending',
  'markUserMessageFailedNotSent',
  'markUserMessageSent',
  'createStreamingAssistantMessage',
  'sendStreamingChat',
  'sendBlockingChat',
  'submitChatTurn',
];
let source = '"use strict";\n';
for (const name of required) {
  source += extractFunction(html, name) + '\n';
}

vm.runInContext(source, context);
messageInput.value = 'hello oracle';
const run = vm.runInContext(
  `(async () => {
    await submitChatTurn(messageInput.value.trim());
    const users = messages.children.filter((c) => String(c.className).includes('user'));
    const assistants = messages.children.filter((c) => String(c.className).includes('assistant'));
    const user = users[0] || {};
    const userText = String(user.textContent || '');
    return {
      userState: user.dataset ? user.dataset.sendState : null,
      userText,
      userLooksDelivered: Boolean(user.dataset && user.dataset.sendState === 'sent'),
      assistantCount: assistants.length,
      assistantText: assistants[0] ? String(assistants[0].textContent || assistants[0].nodeValue || '') : '',
      terminal: activeChatTurnState && activeChatTurnState.completed ? 'done' : 'none',
    };
  })()`,
  context,
);

const result = await run;
const out = {
  ...result,
  terminal: lastChatTerminal,
  prewarmCalls: calls.prewarm.length,
  chatCalls: calls.chat.length,
  ttsCalls: calls.tts.length,
  recentCalls: calls.recent.length,
};
process.stdout.write(`${JSON.stringify(out)}\n`);
void pathToFileURL;
