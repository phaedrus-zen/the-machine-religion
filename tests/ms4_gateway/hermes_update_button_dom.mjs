import { readFileSync } from 'node:fs';
import vm from 'node:vm';

const htmlPath = process.argv[2];
const mode = process.argv[3] || 'success';
const html = readFileSync(htmlPath, 'utf8');

const buttonTag = html.match(/<button\b[^>]*\bid=["']hermesUpdateBtn["'][^>]*>/i)?.[0];
if (!buttonTag) throw new Error('missing #hermesUpdateBtn');

function element(tag = 'div', id = '') {
  const listeners = new Map();
  const classes = new Set();
  const children = [];
  let text = '';
  let markup = '';
  const node = {
    tagName: tag.toUpperCase(),
    id,
    children,
    disabled: false,
    value: '',
    style: {},
    appendChild(child) {
      children.push(child);
      return child;
    },
    addEventListener(type, listener) {
      listeners.set(type, listener);
    },
    click() {
      if (!node.disabled) return listeners.get('click')?.({type: 'click', target: node});
      return undefined;
    },
    showModal() {},
    close() {},
  };
  Object.defineProperty(node, 'textContent', {
    get: () => text + children.map((child) => child.textContent || '').join(''),
    set(value) {
      text = String(value ?? '');
      markup = text;
      children.length = 0;
    },
  });
  Object.defineProperty(node, 'innerHTML', {
    get: () => markup,
    set(value) {
      markup = String(value ?? '');
      text = markup.replace(/<[^>]+>/g, '');
      children.length = 0;
    },
  });
  Object.defineProperty(node, 'classList', {
    value: {
      add: (...names) => names.forEach((name) => classes.add(name)),
      remove: (...names) => names.forEach((name) => classes.delete(name)),
      contains: (name) => classes.has(name),
    },
  });
  return node;
}

const ids = Object.fromEntries([
  ['hermesUpdateBanner', element('div', 'hermesUpdateBanner')],
  ['hermesUpdateHeadline', element('strong', 'hermesUpdateHeadline')],
  ['hermesUpdateSubline', element('span', 'hermesUpdateSubline')],
  ['hermesUpdateBtn', element('button', 'hermesUpdateBtn')],
  ['hermesUpdatePinBtn', element('button', 'hermesUpdatePinBtn')],
  ['hermesVersionLine', element('div', 'hermesVersionLine')],
  ['hermesVersionCurrent', element('span', 'hermesVersionCurrent')],
  ['hermesPinDialog', element('dialog', 'hermesPinDialog')],
  ['hermesPinSelect', element('select', 'hermesPinSelect')],
  ['hermesPinCancel', element('button', 'hermesPinCancel')],
  ['hermesPinConfirm', element('button', 'hermesPinConfirm')],
]);
ids.hermesUpdateBtn.disabled = /\sdisabled(?:\s|=|>)/i.test(buttonTag);
ids.hermesUpdateBtn.textContent = 'Update Hermes';

const recent = new Date().toISOString();
const available = {
  current: '0.19.0',
  latest: '0.20.0',
  published_latest: '0.20.0',
  latest_signature_state: 'signed',
  update_available: true,
  update_in_progress: false,
  operator_state: 'available',
  install_mode: 'wheel',
  last_update: null,
};
const terminal = {
  current: '0.20.0',
  latest: '0.20.0',
  published_latest: '0.20.0',
  latest_signature_state: 'signed',
  update_available: false,
  update_in_progress: false,
  operator_state: 'current',
  install_mode: 'wheel',
  last_update: {
    status: 'success',
    phase: 'done',
    from_version: '0.19.0',
    to_version: '0.20.0',
    finished_at: recent,
  },
};
const blocked = {
  current: '0.19.0',
  latest: '0.20.0',
  published_latest: '0.20.0',
  latest_signature_state: 'unsigned',
  release_signature_policy: 'require_signed',
  update_available: false,
  update_in_progress: false,
  operator_state: 'blocked',
  update_blocked_reason: 'official_tag_unsigned',
  install_mode: 'wheel',
  last_update: null,
};
// Live shape 2026-09-07 under HERMES_RELEASE_SIGNATURE_POLICY=allow_unsigned:
// installed 0.20.0 (v2026.8.3 signed), newest 0.21.0 (v2026.8.31 unsigned).
const allowUnsigned = {
  current: '0.20.0',
  latest: '0.21.0',
  latest_tag: 'v2026.8.31',
  published_latest: '0.21.0',
  published_latest_tag: 'v2026.8.31',
  latest_signature_state: 'unsigned',
  published_latest_signature_state: 'unsigned',
  release_signature_policy: 'allow_unsigned',
  selected_signed_fallback: false,
  update_available: true,
  update_in_progress: false,
  operator_state: 'update_available',
  update_blocked_reason: null,
  installed_relation: 'older',
  latest_published_at: '2026-08-31T19:29:49Z',
  install_mode: 'editable',
  last_update: null,
};
const unknown = {
  current: '0.19.0',
  latest: null,
  published_latest: null,
  latest_signature_state: 'unknown',
  update_available: false,
  update_in_progress: false,
  operator_state: 'unknown',
  install_mode: 'editable',
  last_update: null,
};

const calls = {gets: [], posts: []};
function response(status, body) {
  return {
    ok: status >= 200 && status < 300,
    status,
    statusText: status === 400 ? 'Bad Request' : 'OK',
    json: async () => body,
  };
}

async function fetchMock(url, options = {}) {
  const path = String(url);
  const method = String(options.method || 'GET').toUpperCase();
  if (path === '/api/v1/hermes/version' && method === 'GET') {
    calls.gets.push(path);
    if (mode === 'blocked') return response(200, blocked);
    if (mode === 'unknown') return response(200, unknown);
    if (mode === 'allow-unsigned') return response(200, allowUnsigned);
    if (mode === 'success' && calls.gets.length > 1) return response(200, terminal);
    return response(200, available);
  }
  if (path === '/api/v1/hermes/update' && method === 'POST') {
    calls.posts.push({path, method, body: JSON.parse(options.body)});
    if (mode === 'preflight-400') {
      return response(400, {error: 'disk preflight failed: insufficient free space'});
    }
    const accepted = mode === 'allow-unsigned' ? '0.21.0' : '0.20.0';
    return response(202, {accepted: true, target_version: accepted, to_version: accepted});
  }
  throw new Error(`unexpected fetch ${method} ${path}`);
}

const context = vm.createContext({
  console,
  Date,
  fetch: fetchMock,
  setInterval: () => 1,
  clearInterval: () => {},
  document: {
    getElementById: (id) => ids[id] || element('div', id),
    createElement: (tag) => element(tag),
  },
});

const start = html.indexOf("const hermesBanner = document.getElementById('hermesUpdateBanner')");
const end = html.indexOf("const daPanel = document.getElementById('doubleAgentPanel')", start);
if (start < 0 || end < 0) throw new Error('missing Hermes UI script');
vm.runInContext(`"use strict";\n${html.slice(start, end)}`, context);

const coldDisabled = ids.hermesUpdateBtn.disabled;
await vm.runInContext('refreshHermesVersion()', context);
const filled = {
  disabled: ids.hermesUpdateBtn.disabled,
  text: ids.hermesUpdateBtn.textContent,
};
ids.hermesUpdateBtn.click();
for (let turn = 0; turn < 4; turn += 1) {
  await new Promise((resolve) => setImmediate(resolve));
}
if (mode === 'preflight-400') {
  await vm.runInContext('refreshHermesVersion()', context);
}

process.stdout.write(`${JSON.stringify({
  mode,
  coldDisabled,
  filled,
  getCalls: calls.gets.length,
  postCalls: calls.posts,
  buttonDisabled: ids.hermesUpdateBtn.disabled,
  buttonText: ids.hermesUpdateBtn.textContent,
  bannerVisible: ids.hermesUpdateBanner.classList.contains('visible'),
  bannerSuccess: ids.hermesUpdateBanner.classList.contains('success'),
  bannerError: ids.hermesUpdateBanner.classList.contains('error'),
  headline: ids.hermesUpdateHeadline.textContent,
  subline: ids.hermesUpdateSubline.textContent,
})}\n`);
