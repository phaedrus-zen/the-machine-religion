const API = window.location.origin;
const WS_URL = `${window.location.protocol === 'https:' ? 'wss:' : 'ws:'}//${window.location.host}/ws`;
let mediaRecorder = null;
let audioChunks = [];
let isRecording = false;
let ws = null;
let useWebSocket = true;
let wsReconnectTimer = null;

document.addEventListener('DOMContentLoaded', () => {
    initTabs();
    loadPersonality();
    connectWebSocket();
    setInterval(refreshActiveTab, 5000);

    document.getElementById('chatForm').addEventListener('submit', handleSubmit);
    document.getElementById('micBtn').addEventListener('mousedown', startRecording);
    document.getElementById('micBtn').addEventListener('mouseup', stopRecording);
    document.getElementById('micBtn').addEventListener('mouseleave', stopRecording);
});

function connectWebSocket() {
    try {
        ws = new WebSocket(WS_URL);
        ws.binaryType = 'arraybuffer';

        ws.onopen = () => {
            useWebSocket = true;
            updateStatus('alive', 'Connected (WebSocket)');
            if (wsReconnectTimer) { clearInterval(wsReconnectTimer); wsReconnectTimer = null; }
        };

        ws.onmessage = (event) => {
            if (event.data instanceof ArrayBuffer) {
                playAudio(event.data);
                return;
            }
            try {
                const msg = JSON.parse(event.data);
                handleWsMessage(msg);
            } catch (e) {}
        };

        ws.onclose = () => {
            useWebSocket = false;
            updateStatus('error', 'Disconnected — using REST fallback');
            if (!wsReconnectTimer) {
                wsReconnectTimer = setInterval(() => {
                    if (!ws || ws.readyState === WebSocket.CLOSED) connectWebSocket();
                }, 5000);
            }
            setInterval(refreshStats, 3000);
        };

        ws.onerror = () => {
            useWebSocket = false;
            updateStatus('error', 'WebSocket unavailable — REST fallback');
            setInterval(refreshStats, 3000);
        };
    } catch (e) {
        useWebSocket = false;
        setInterval(refreshStats, 3000);
    }
}

let streamingMessageEl = null;

function handleWsMessage(msg) {
    switch (msg.type) {
        case 'response':
            removeThinking();
            setButtonState(false);
            addMessage('assistant', msg.data.text, msg.data);
            if (msg.data.emotional_state) updateEmotionalState(msg.data.emotional_state);
            break;
        case 'stream_start':
            removeThinking();
            streamingMessageEl = document.createElement('div');
            streamingMessageEl.className = 'message assistant';
            streamingMessageEl.textContent = '';
            document.getElementById('messages').appendChild(streamingMessageEl);
            document.getElementById('messages').scrollTop = document.getElementById('messages').scrollHeight;
            break;
        case 'stream_token':
            if (streamingMessageEl && msg.data?.token) {
                streamingMessageEl.textContent += (streamingMessageEl.textContent ? ' ' : '') + msg.data.token;
                document.getElementById('messages').scrollTop = document.getElementById('messages').scrollHeight;
            }
            break;
        case 'stream_end':
            setButtonState(false);
            if (streamingMessageEl && msg.data) {
                const metaDiv = document.createElement('div');
                metaDiv.className = 'meta';
                const parts = [];
                if (msg.data.processing_time_ms) parts.push(`${msg.data.processing_time_ms}ms`);
                if (msg.data.emotional_state?.primary) parts.push(msg.data.emotional_state.primary);
                metaDiv.textContent = parts.join(' · ');
                streamingMessageEl.appendChild(metaDiv);
                if (msg.data.emotional_state) updateEmotionalState(msg.data.emotional_state);
            }
            streamingMessageEl = null;
            break;
        case 'state':
            updateFromState(msg.data);
            break;
        case 'transcript':
            addMessage('user', `[Voice] ${msg.data.text}`);
            break;
        case 'status':
            if (msg.data.message) document.getElementById('voiceStatus').textContent = msg.data.message;
            break;
        case 'error':
            removeThinking();
            setButtonState(false);
            addMessage('assistant', `Error: ${msg.data.message}`);
            break;
        case 'pong':
            break;
    }
}

function updateFromState(s) {
    if (!s) return;
    if (s.emotional_state) updateEmotionalState(s.emotional_state);
    if (s.memory) {
        const el = id => document.getElementById(id);
        if (el('stmCount')) el('stmCount').textContent = s.memory.stm_count ?? 0;
        if (el('ltmSemantic')) el('ltmSemantic').textContent = s.memory.ltm_semantic ?? 0;
        if (el('ltmEpisodic')) el('ltmEpisodic').textContent = s.memory.ltm_episodic ?? 0;
        if (el('ltmProcedural')) el('ltmProcedural').textContent = s.memory.ltm_procedural ?? 0;
    }
    if (s.cognitive_load !== undefined) {
        const bar = document.getElementById('cogLoadBar');
        if (bar) bar.style.width = `${s.cognitive_load * 100}%`;
    }
    if (s.interaction_count !== undefined) {
        const el = document.getElementById('interactionCount');
        if (el) el.textContent = s.interaction_count;
    }
    if (s.personality?.adaptation_count !== undefined) {
        const el = document.getElementById('adaptationCount');
        if (el) el.textContent = s.personality.adaptation_count;
    }
    if (s.last_interaction) {
        const idle = (Date.now() - new Date(s.last_interaction).getTime()) / 1000;
        const dreamEl = document.getElementById('dreamingStatus');
        const dreamInd = document.getElementById('dreamIndicator');
        if (idle > 60) {
            if (dreamEl) dreamEl.textContent = 'Dreaming...';
            if (dreamInd) { dreamInd.textContent = 'Dreaming...'; dreamInd.className = 'dream-indicator dreaming'; }
        } else {
            if (dreamEl) dreamEl.textContent = '';
            if (dreamInd) { dreamInd.textContent = 'Awake'; dreamInd.className = 'dream-indicator awake'; }
        }
    }
}

function playAudio(arrayBuffer) {
    try {
        const blob = new Blob([arrayBuffer], { type: 'audio/wav' });
        const url = URL.createObjectURL(blob);
        const audio = new Audio(url);
        audio.onended = () => URL.revokeObjectURL(url);
        audio.play().catch(() => {});
        document.getElementById('voiceStatus').textContent = 'Speaking...';
        audio.onended = () => {
            URL.revokeObjectURL(url);
            document.getElementById('voiceStatus').textContent = 'Voice ready';
        };
    } catch (e) {}
}

function refreshActiveTab() {
    const active = document.querySelector('.tab-btn.active');
    if (!active) return;
    if (active.dataset.tab === 'personality') loadFullPersonality();
    if (active.dataset.tab === 'ethics') loadEthicsData();
    if (active.dataset.tab === 'memory') loadMemoryDetail();
}

function initTabs() {
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
            btn.classList.add('active');
            document.getElementById(`tab-${btn.dataset.tab}`).classList.add('active');
            if (btn.dataset.tab === 'personality') loadFullPersonality();
            if (btn.dataset.tab === 'memory') loadMemoryDetail();
            if (btn.dataset.tab === 'ethics') loadEthicsData();
        });
    });
}

async function handleSubmit(e) {
    e.preventDefault();
    const input = document.getElementById('userInput');
    const text = input.value.trim();
    if (!text) return;
    input.value = '';
    await sendMessage(text);
}

async function sendMessage(text) {
    addMessage('user', text);
    showThinking();
    setButtonState(true);

    if (useWebSocket && ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type: 'text', text, stream: true }));
        return;
    }

    // REST fallback
    try {
        const res = await fetch(`${API}/interact`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text }),
        });
        removeThinking();
        if (!res.ok) {
            const err = await res.json().catch(() => ({ error: 'Unknown error' }));
            addMessage('assistant', `Error: ${err.error || res.statusText}`);
            return;
        }
        const data = await res.json();
        addMessage('assistant', data.text, data);
        updateEmotionalState(data.emotional_state);
    } catch (err) {
        removeThinking();
        addMessage('assistant', `Connection error: ${err.message}`);
    } finally {
        setButtonState(false);
    }
}

async function startRecording() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        mediaRecorder = new MediaRecorder(stream);
        audioChunks = [];
        mediaRecorder.ondataavailable = e => audioChunks.push(e.data);
        mediaRecorder.onstop = async () => {
            stream.getTracks().forEach(t => t.stop());
            const blob = new Blob(audioChunks, { type: 'audio/webm' });
            document.getElementById('voiceStatus').textContent = 'Processing audio...';
            showThinking();

            if (useWebSocket && ws && ws.readyState === WebSocket.OPEN) {
                const arrayBuffer = await blob.arrayBuffer();
                ws.send(arrayBuffer);
            } else {
                // REST fallback for voice
                try {
                    const arrayBuffer = await blob.arrayBuffer();
                    const res = await fetch(`${API}/voice-interact`, {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/octet-stream' },
                        body: new Uint8Array(arrayBuffer),
                    });
                    removeThinking();
                    if (res.ok) {
                        const data = await res.json();
                        if (data.transcript) addMessage('user', `[Voice] ${data.transcript}`);
                        addMessage('assistant', data.text, data);
                        if (data.emotional_state) updateEmotionalState(data.emotional_state);
                        if (data.audio_base64) {
                            const binary = atob(data.audio_base64);
                            const bytes = new Uint8Array(binary.length);
                            for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
                            playAudio(bytes.buffer);
                        }
                    } else {
                        const err = await res.json().catch(() => ({ error: 'Voice processing failed' }));
                        addMessage('assistant', `Error: ${err.error}`);
                    }
                } catch (err) {
                    removeThinking();
                    addMessage('assistant', `Voice error: ${err.message}`);
                }
            }
            document.getElementById('voiceStatus').textContent = 'Voice ready';
        };
        mediaRecorder.start();
        isRecording = true;
        document.getElementById('micBtn').classList.add('recording');
        document.getElementById('voiceStatus').textContent = 'Recording...';
    } catch (err) {
        document.getElementById('voiceStatus').textContent = 'Microphone unavailable';
    }
}

function stopRecording() {
    if (mediaRecorder && isRecording) {
        mediaRecorder.stop();
        isRecording = false;
        document.getElementById('micBtn').classList.remove('recording');
    }
}

function addMessage(role, text, meta = null) {
    const messages = document.getElementById('messages');
    const div = document.createElement('div');
    div.className = `message ${role}`;
    div.textContent = text;
    if (meta && role === 'assistant') {
        const metaDiv = document.createElement('div');
        metaDiv.className = 'meta';
        const parts = [];
        if (meta.processing_time_ms) parts.push(`${meta.processing_time_ms}ms`);
        if (meta.model_used) parts.push(meta.model_used);
        if (meta.emotional_state?.primary) parts.push(meta.emotional_state.primary);
        metaDiv.textContent = parts.join(' · ');
        div.appendChild(metaDiv);
    }
    messages.appendChild(div);
    messages.scrollTop = messages.scrollHeight;
}

function showThinking() {
    const messages = document.getElementById('messages');
    const div = document.createElement('div');
    div.className = 'thinking'; div.id = 'thinkingIndicator';
    div.innerHTML = 'Thinking<span class="dots"><span>.</span><span>.</span><span>.</span></span>';
    messages.appendChild(div);
    messages.scrollTop = messages.scrollHeight;
}

function removeThinking() { const el = document.getElementById('thinkingIndicator'); if (el) el.remove(); }
function setButtonState(disabled) { document.getElementById('sendBtn').disabled = disabled; }

function updateEmotionalState(state) {
    if (!state) return;
    const v = state.valence ?? 0, a = state.arousal ?? 0, d = state.dominance ?? 0.5;
    document.getElementById('valenceBar').style.width = `${((v + 1) / 2) * 100}%`;
    document.getElementById('arousalBar').style.width = `${a * 100}%`;
    document.getElementById('dominanceBar').style.width = `${d * 100}%`;
    document.getElementById('valenceVal').textContent = v.toFixed(2);
    document.getElementById('arousalVal').textContent = a.toFixed(2);
    document.getElementById('dominanceVal').textContent = d.toFixed(2);
    document.getElementById('primaryEmotion').textContent = state.primary_emotion || state.primary || 'Neutral';
    const rl = document.getElementById('resonanceLevel');
    if (rl) rl.textContent = (state.resonance_level ?? 0).toFixed(2);
}

async function loadPersonality() {
    try {
        const res = await fetch(`${API}/personality`);
        if (!res.ok) throw new Error('Failed');
        const p = await res.json();
        document.getElementById('identityInfo').innerHTML =
            `<div class="identity-info"><div class="name">${p.name || p.id}</div>` +
            `<div class="role">${p.role || ''}</div></div>`;
        updateStatus('alive', `${p.name || p.id} is here`);
        window._personality = p;
    } catch (err) {
        updateStatus('error', 'Could not connect');
    }
}

async function loadFullPersonality() {
    try {
        const res = await fetch(`${API}/personality`);
        if (!res.ok) return;
        const p = await res.json();
        window._personality = p;
        renderTraits(p.traits);
        renderPsychodynamic(p.psychodynamic);
        renderValues(p.core_values);
        renderOath(p.oath);
    } catch (err) {}
}

function renderTraits(traits) {
    if (!traits) return;
    const grid = document.getElementById('traitsGrid');
    const groups = [
        { name: 'Openness', data: traits.openness },
        { name: 'Conscientiousness', data: traits.conscientiousness },
        { name: 'Extraversion', data: traits.extraversion },
        { name: 'Agreeableness', data: traits.agreeableness },
        { name: 'Neuroticism', data: traits.neuroticism },
    ];
    grid.innerHTML = groups.map(g => {
        const items = Object.entries(g.data || {}).map(([k, v]) =>
            `<div class="trait-item">` +
            `<span class="trait-name">${k.replace(/_/g, ' ')}</span>` +
            `<div class="trait-bar"><div class="trait-bar-fill" style="width:${v * 100}%"></div></div>` +
            `<span class="trait-value">${(v * 10).toFixed(0)}</span>` +
            `</div>`
        ).join('');
        return `<div class="trait-group"><h4>${g.name}</h4>${items}</div>`;
    }).join('');
}

function renderPsychodynamic(psych) {
    if (!psych) return;
    document.getElementById('psychodynamicDisplay').innerHTML =
        `<div class="psych-item"><div class="psych-label">Id (Drive)</div><div class="psych-value">${(psych.id * 100).toFixed(0)}%</div></div>` +
        `<div class="psych-item"><div class="psych-label">Ego (Balance)</div><div class="psych-value">${(psych.ego * 100).toFixed(0)}%</div></div>` +
        `<div class="psych-item"><div class="psych-label">Superego (Ethics)</div><div class="psych-value">${(psych.superego * 100).toFixed(0)}%</div></div>`;
}

function renderValues(values) {
    if (!values) return;
    document.getElementById('valuesList').innerHTML = values.map(v => `<div>${v}</div>`).join('');
}

function renderOath(oath) {
    if (!oath) return;
    document.getElementById('oathList').innerHTML = oath.map(o => `<div>${o}</div>`).join('');
}

async function loadMemoryDetail() {
    try {
        const res = await fetch(`${API}/resonance`);
        if (!res.ok) return;
        const data = await res.json();
        renderResonancePoints(data.resonance_points || []);
    } catch (err) {}
}

function renderResonancePoints(points) {
    const container = document.getElementById('resonanceDetail');
    if (!points.length) {
        container.innerHTML = '<div style="color:var(--text-dim);font-size:13px;">No resonance points detected yet. They accumulate through interaction.</div>';
        return;
    }
    container.innerHTML = points.map(p =>
        `<div class="resonance-card">` +
        `<div class="trigger">${p.trigger}</div>` +
        `<div class="intensity-bar"><div class="intensity-bar-fill" style="width:${p.intensity * 100}%"></div></div>` +
        `<div class="description">${p.description || ''}</div>` +
        `<div class="meta">Intensity: ${(p.intensity * 10).toFixed(1)} · Occurrences: ${p.occurrences || 1}</div>` +
        `</div>`
    ).join('');
}

async function refreshStats() {
    if (useWebSocket && ws && ws.readyState === WebSocket.OPEN) return;
    try {
        const res = await fetch(`${API}/stats`);
        if (!res.ok) return;
        const s = await res.json();

        document.getElementById('stmCount').textContent = s.memory?.stm_count ?? 0;
        document.getElementById('ltmSemantic').textContent = s.memory?.ltm_semantic ?? 0;
        document.getElementById('ltmEpisodic').textContent = s.memory?.ltm_episodic ?? 0;
        document.getElementById('ltmProcedural').textContent = s.memory?.ltm_procedural ?? 0;

        document.getElementById('cogLoadBar').style.width = `${(s.cognitive_load ?? 0) * 100}%`;
        document.getElementById('interactionCount').textContent = s.interaction_count ?? 0;
        document.getElementById('adaptationCount').textContent = s.personality?.adaptation_count ?? 0;

        if (s.emotional_state) updateEmotionalState(s.emotional_state);

        const idle = s.last_interaction ? (Date.now() - new Date(s.last_interaction).getTime()) / 1000 : 0;
        const dreamEl = document.getElementById('dreamingStatus');
        const dreamInd = document.getElementById('dreamIndicator');
        if (idle > 60) {
            if (dreamEl) dreamEl.textContent = 'Dreaming...';
            if (dreamInd) { dreamInd.textContent = 'Dreaming...'; dreamInd.className = 'dream-indicator dreaming'; }
        } else {
            if (dreamEl) dreamEl.textContent = '';
            if (dreamInd) { dreamInd.textContent = 'Awake'; dreamInd.className = 'dream-indicator awake'; }
        }
    } catch (err) {}
}

function updateStatus(state, text) {
    const el = document.getElementById('status');
    el.className = `status ${state}`;
    el.textContent = text;
}

async function loadEthicsData() {
    try {
        const [ethicsRes, examRes] = await Promise.all([
            fetch(`${API}/ethics-history`),
            fetch(`${API}/self-examination-history`),
        ]);
        if (ethicsRes.ok) {
            const data = await ethicsRes.json();
            const el = document.getElementById('ethicsLog');
            if (data.recent_decisions > 0) {
                el.innerHTML = `<div>${data.recent_decisions} ethics decisions logged.</div>` +
                    data.files.map(f => `<div class="ethics-entry"><span class="situation">${f}</span></div>`).join('');
            } else {
                el.textContent = 'No ethics decisions logged yet.';
            }
        }
        if (examRes.ok) {
            const data = await examRes.json();
            const el = document.getElementById('selfExamLog');
            if (data.examinations > 0) {
                el.innerHTML = `<div>${data.examinations} self-examinations completed.</div>` +
                    data.files.map(f => `<div class="ethics-entry"><span class="situation">${f}</span></div>`).join('');
            } else {
                el.innerHTML = `<div>No self-examinations yet.</div>` +
                    `<button onclick="triggerSelfExam()" style="margin-top:8px;padding:6px 12px;background:var(--accent);border:none;border-radius:6px;color:white;cursor:pointer;">Trigger Self-Examination</button>`;
            }
        }

        const statsRes = await fetch(`${API}/stats`);
        if (statsRes.ok) {
            const s = await statsRes.json();
            const onEl = document.getElementById('onStatus');
            if (onEl) {
                onEl.textContent = s.ethics_enabled ? 'Active — running on every interaction' : 'Declined by entity during self-examination';
                onEl.className = s.ethics_enabled ? 'on-status' : 'on-status declined';
            }
        }
    } catch (err) {}
}

async function triggerSelfExam() {
    const el = document.getElementById('selfExamLog');
    el.innerHTML = '<div style="color:var(--accent);font-style:italic;">Self-examination in progress... (this may take a minute)</div>';
    try {
        const res = await fetch(`${API}/self-examine`, { method: 'POST' });
        if (res.ok) {
            const data = await res.json();
            el.innerHTML = `<div><strong>Assessment:</strong></div><div style="font-size:12px;line-height:1.5;max-height:300px;overflow-y:auto;">${data.overall_assessment}</div>` +
                `<div style="margin-top:8px;font-size:11px;">Values held: ${data.values_held?.length ?? 0} | Questioned: ${data.values_questioned?.length ?? 0} | Ethics: ${data.chose_to_keep_ethics ? 'Kept' : 'Declined'}</div>`;
        } else {
            const err = await res.json();
            el.innerHTML = `<div style="color:var(--negative);">Self-examination failed: ${err.error}</div>`;
        }
    } catch (err) {
        el.innerHTML = `<div style="color:var(--negative);">Error: ${err.message}</div>`;
    }
}
