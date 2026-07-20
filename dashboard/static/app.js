/* ═══════════════════════════════════════════════════════════════
   AI Shorts Studio — app.js
   Vanilla JS SPA: navigation, API calls, SSE progress, all UI
   ═══════════════════════════════════════════════════════════════ */

'use strict';

// ── State ─────────────────────────────────────────────────────────────────
const state = {
  currentPage:  'dashboard',
  currentJobId: null,
  currentStem:  null,
  currentTopic: null,
  progressTimer: null,
  progressStart: null,
  historyData:  [],
};

// ── Navigation ────────────────────────────────────────────────────────────
function navTo(page) {
  document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));

  const pageEl = document.getElementById(`page-${page}`);
  const navEl  = document.getElementById(`nav-${page}`);
  if (pageEl) pageEl.classList.add('active');
  if (navEl)  navEl.classList.add('active');

  state.currentPage = page;

  if (page === 'dashboard') loadDashboard();
  if (page === 'library')   loadLibrary();
  if (page === 'history')   loadHistory();
}

document.querySelectorAll('.nav-item').forEach(el => {
  el.addEventListener('click', e => {
    e.preventDefault();
    navTo(el.dataset.page);
  });
});

// ── API helpers ───────────────────────────────────────────────────────────
async function api(path, method = 'GET', body = null) {
  const opts = { method, headers: { 'Content-Type': 'application/json' } };
  if (body) opts.body = JSON.stringify(body);
  const r = await fetch(path, opts);
  if (!r.ok) {
    const err = await r.json().catch(() => ({ error: r.statusText }));
    throw new Error(err.error || r.statusText);
  }
  return r.json();
}

// ── SSE job stream ─────────────────────────────────────────────────────────
function streamJob(jobId, { onLog, onDone, onError }) {
  const es = new EventSource(`/api/jobs/${jobId}/stream`);
  es.onmessage = e => {
    const msg = JSON.parse(e.data);
    if (msg.ping) return;
    if (msg.done) {
      es.close();
      if (msg.status === 'done') onDone && onDone(msg.result);
      else onError && onError(msg.result?.error || 'Unknown error');
    } else {
      onLog && onLog(msg);
    }
  };
  es.onerror = () => { es.close(); onError && onError('Connection lost'); };
  return es;
}

// ── Progress helpers ───────────────────────────────────────────────────────
const PIPELINE_STEPS = [
  'Fetching', 'script', 'scene plan', 'prompts', 'voice',
  'Transcrib', 'Generat', 'Composit', 'rendered',
];

function inferProgress(msg) {
  const m = msg.toLowerCase();
  for (let i = 0; i < PIPELINE_STEPS.length; i++) {
    if (m.includes(PIPELINE_STEPS[i].toLowerCase())) {
      return Math.round(10 + (i / (PIPELINE_STEPS.length - 1)) * 85);
    }
  }
  return null;
}

function startProgressTimer() {
  state.progressStart = Date.now();
  state.progressTimer = setInterval(() => {
    const elapsed = Math.round((Date.now() - state.progressStart) / 1000);
    const el = document.getElementById('progress-elapsed');
    if (el) el.textContent = `${elapsed}s`;
  }, 1000);
}

function stopProgressTimer() {
  if (state.progressTimer) { clearInterval(state.progressTimer); state.progressTimer = null; }
}

function setProgress(pct) {
  const bar = document.getElementById('progress-bar');
  if (bar) bar.style.width = `${Math.min(100, pct)}%`;
}

function appendLog(msg, kind = 'info') {
  const stream = document.getElementById('log-stream');
  if (!stream) return;
  const div = document.createElement('div');
  div.className = `log-entry ${kind}`;
  div.textContent = msg;
  stream.appendChild(div);
  stream.scrollTop = stream.scrollHeight;
}

function clearLog() {
  const stream = document.getElementById('log-stream');
  if (stream) stream.innerHTML = '';
}

function showCard(id, show = true) {
  const el = document.getElementById(id);
  if (el) el.style.display = show ? '' : 'none';
}

// ── Dashboard ─────────────────────────────────────────────────────────────
async function loadDashboard() {
  try {
    const stats = await api('/api/stats');
    setText('stat-videos', stats.total_videos);
    setText('stat-runs',   stats.total_runs);
    setText('stat-score',  stats.avg_script_score ? `${stats.avg_script_score}/10` : '—');
    setText('stat-passed', stats.total_passed);
  } catch (e) { /* stats load is optional */ }

  try {
    const { videos } = await api('/api/videos');
    renderRecentVideos(videos.slice(0, 8));
  } catch (e) { /* ignore */ }
}

function renderRecentVideos(videos) {
  const el = document.getElementById('recent-videos');
  if (!el) return;
  if (!videos.length) {
    el.innerHTML = `<div class="empty-state" style="padding:40px 0">
      <div class="empty-icon">▦</div>
      <p>No videos yet</p>
    </div>`;
    return;
  }
  el.innerHTML = videos.map(v => videoThumbHTML(v)).join('');
}

// ── Library ───────────────────────────────────────────────────────────────
async function loadLibrary() {
  try {
    const { videos } = await api('/api/videos');
    const grid  = document.getElementById('library-grid');
    const empty = document.getElementById('library-empty');
    if (!videos.length) {
      grid.style.display  = 'none';
      empty.style.display = '';
      return;
    }
    grid.style.display  = '';
    empty.style.display = 'none';
    grid.innerHTML = videos.map(v => videoThumbHTML(v)).join('');
  } catch (e) { toast('Failed to load library', 'error'); }
}

function videoThumbHTML(v) {
  const stem    = v.stem;
  const report  = v.report || {};
  const topic   = (report.topic || stem).replace(/^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}_/, '').replace(/-/g, ' ');
  const score   = report.script?.structure_score;
  const dateStr = stem.slice(0, 10);
  const videoSrc = `/output/videos/${v.file}`;

  return `
  <div class="video-thumb" onclick="openVideoModal('${stem}', '${videoSrc}')">
    <div class="video-thumb-preview">
      <video src="${videoSrc}" muted preload="none"
        onmouseenter="this.play()" onmouseleave="this.pause();this.currentTime=0"></video>
      <div class="play-overlay">
        <div class="play-btn-icon">▶</div>
      </div>
    </div>
    <div class="video-thumb-info">
      <div class="video-thumb-title">${escHtml(topic)}</div>
      <div class="video-thumb-meta">
        <span>${dateStr}</span>
        ${score ? `<span class="thumb-score">${score}/10</span>` : ''}
        <span>${v.size_mb} MB</span>
      </div>
    </div>
  </div>`;
}

function openVideoModal(stem, videoSrc) {
  const modal = document.getElementById('video-modal');
  const vid   = document.getElementById('modal-video');
  const meta  = document.getElementById('modal-meta');
  if (!modal || !vid) return;

  vid.src = videoSrc;
  modal.style.display = '';

  api(`/api/quality_report/${stem}`).then(report => {
    meta.innerHTML = buildModalMeta(report);
  }).catch(() => {
    meta.innerHTML = `<p style="color:var(--text-3);font-size:13px">No quality report found</p>`;
  });
}

function closeVideoModal(e) {
  if (e && e.target !== document.getElementById('video-modal') && !e.target.closest('.modal-close')) return;
  const vid = document.getElementById('modal-video');
  if (vid) { vid.pause(); vid.src = ''; }
  document.getElementById('video-modal').style.display = 'none';
}

function buildModalMeta(r) {
  if (!r) return '';
  const script = r.script || {};
  const aud    = r.audience || {};
  const story  = r.story || {};

  const checkBadge = (ok, label) =>
    `<span class="badge ${ok ? 'badge-pass' : 'badge-fail'}">${ok ? '✓' : '✗'} ${label}</span>`;

  return `
  <h2>${escHtml(r.topic || 'Untitled')}</h2>

  ${story.viral_score ? `
  <div class="meta-section">
    <div class="meta-section-title">Story</div>
    <div class="meta-row"><span>Source</span><strong>${escHtml(story.source || '—')}</strong></div>
    <div class="meta-row"><span>Category</span><strong>${escHtml(story.category || '—')}</strong></div>
    <div class="meta-row"><span>Viral Score</span><strong>${story.viral_score}/100</strong></div>
    <div class="meta-row"><span>Age</span><strong>${story.age_hours}h ago</strong></div>
  </div>` : ''}

  ${aud.overall != null ? `
  <div class="meta-section">
    <div class="meta-section-title">Audience Appeal</div>
    <div class="meta-row"><span>Overall</span><strong>${aud.overall}/100</strong></div>
    <div class="meta-row"><span>Curiosity</span><strong>${aud.curiosity}/10</strong></div>
    <div class="meta-row"><span>Shock Value</span><strong>${aud.shock_value}/10</strong></div>
    <div class="meta-row"><span>Shareability</span><strong>${aud.shareability}/10</strong></div>
  </div>` : ''}

  <div class="meta-section">
    <div class="meta-section-title">Script</div>
    <div class="meta-row"><span>Structure</span><strong>${script.structure_score || '—'}/10</strong></div>
    <div class="meta-row"><span>Hook</span><strong>${script.hook_score || '—'}/10</strong></div>
    <div class="meta-row"><span>Words</span><strong>${script.word_count || '—'}</strong></div>
    <div class="meta-row"><span>Duration</span><strong>~${script.estimated_seconds || '—'}s</strong></div>
  </div>

  <div class="meta-section">
    <div class="meta-section-title">Quality Checks</div>
    <div class="check-list">
      ${checkBadge(script.has_hook,           'Hook')}
      ${checkBadge(script.has_curiosity_gap,  'Curiosity Gap')}
      ${checkBadge(script.has_payoff,         'Payoff')}
      ${checkBadge(script.has_why_it_matters, 'Why It Matters')}
      ${checkBadge(script.has_cta,            'CTA')}
      ${checkBadge(script.fact_safe,          'Fact Safe')}
    </div>
  </div>

  ${r.warnings?.length ? `
  <div class="meta-section">
    <div class="meta-section-title">Warnings</div>
    ${r.warnings.map(w => `<div style="font-size:12px;color:var(--amber);padding:3px 0">⚠ ${escHtml(w)}</div>`).join('')}
  </div>` : ''}

  <div class="meta-section">
    <div class="meta-row">
      <span>Final Result</span>
      <strong class="${r.final_pass ? 'pass-yes' : 'pass-no'} pass-badge">
        ${r.final_pass ? '✅ PASS' : '❌ FAIL'}
      </strong>
    </div>
    <div class="meta-row"><span>Generation Time</span><strong>${r.generation_time_seconds}s</strong></div>
  </div>

  <button class="btn-secondary w-full" style="margin-top:12px"
    onclick="regenFromLibrary('${escHtml(r.topic || '')}')">
    ↺ Regenerate Stages
  </button>`;
}

function regenFromLibrary(topic) {
  closeVideoModal();
  document.getElementById('gen-topic').value = topic;
  navTo('generate');
  toast('Set topic — click Generate or use Regenerate Stage panel', 'info');
}

// ── History ───────────────────────────────────────────────────────────────
async function loadHistory() {
  try {
    const { history } = await api('/api/history');
    state.historyData = history;
    renderHistory(history);
  } catch (e) { toast('Failed to load history', 'error'); }
}

function filterHistory() {
  const q         = (document.getElementById('hist-search')?.value || '').toLowerCase();
  const date      = document.getElementById('hist-date')?.value || '';
  const minScore  = parseInt(document.getElementById('hist-min-score')?.value || '0');

  let data = state.historyData;
  if (q)        data = data.filter(h => (h.topic || '').toLowerCase().includes(q) || (h.source||'').toLowerCase().includes(q));
  if (date)     data = data.filter(h => (h.date || '').startsWith(date));
  if (minScore) data = data.filter(h => (h.script_score || 0) >= minScore || (h.viral_score || 0) >= minScore);

  renderHistory(data);
}

function renderHistory(rows) {
  const empty  = document.getElementById('history-empty');
  const wrap   = document.getElementById('history-table-wrap');
  const tbody  = document.getElementById('history-tbody');
  if (!rows.length) {
    wrap.style.display  = 'none';
    empty.style.display = '';
    return;
  }
  wrap.style.display  = '';
  empty.style.display = 'none';

  tbody.innerHTML = [...rows].reverse().map(h => `
    <tr>
      <td>${h.date || '—'}</td>
      <td class="topic-cell"><div class="topic-cell-inner">${escHtml(h.topic || '—')}</div></td>
      <td>${escHtml(h.source || '—')}</td>
      <td>${h.viral_score != null ? `<span class="${scorePillClass(h.viral_score, 100)}">${h.viral_score}</span>` : '—'}</td>
      <td>${h.script_score != null ? `<span class="${scorePillClass(h.script_score, 10)}">${h.script_score}/10</span>` : '—'}</td>
      <td>${h.word_count || '—'}</td>
      <td>${h.generation_time_seconds ? `${h.generation_time_seconds}s` : '—'}</td>
      <td><span class="pass-badge ${h.final_pass ? 'pass-yes' : 'pass-no'}">${h.final_pass ? 'PASS' : 'FAIL'}</span></td>
      <td>
        <button class="btn-ghost" style="padding:4px 10px;font-size:11px"
          onclick="loadTopicForRegen('${escHtml(h.topic || '')}')">Regen</button>
      </td>
    </tr>
  `).join('');
}

function scorePillClass(val, max) {
  const pct = val / max;
  if (pct >= 0.8) return 'news-score-pill score-high';
  if (pct >= 0.6) return 'news-score-pill score-mid';
  return 'news-score-pill score-low';
}

function loadTopicForRegen(topic) {
  document.getElementById('gen-topic').value = topic;
  navTo('generate');
  toast('Topic loaded — generate or use Regenerate Stage', 'info');
}

// ── News ──────────────────────────────────────────────────────────────────
async function fetchNews() {
  const btn = document.getElementById('btn-fetch-news');
  if (btn) btn.disabled = true;

  showCard('news-empty', false);
  document.getElementById('news-list').style.display = 'none';
  document.getElementById('news-loading').style.display = '';

  try {
    const { job_id } = await api('/api/news/fetch', 'POST', {});
    streamJob(job_id, {
      onLog: msg => { /* silent */ },
      onDone: result => {
        document.getElementById('news-loading').style.display = 'none';
        renderNews(result.stories || []);
        if (btn) btn.disabled = false;
      },
      onError: err => {
        document.getElementById('news-loading').style.display = 'none';
        document.getElementById('news-empty').style.display   = '';
        toast(`News fetch failed: ${err}`, 'error');
        if (btn) btn.disabled = false;
      },
    });
  } catch (e) {
    document.getElementById('news-loading').style.display = 'none';
    document.getElementById('news-empty').style.display   = '';
    toast(`Error: ${e.message}`, 'error');
    if (btn) btn.disabled = false;
  }
}

function renderNews(stories) {
  const list = document.getElementById('news-list');
  if (!stories.length) {
    document.getElementById('news-empty').style.display = '';
    list.style.display = 'none';
    return;
  }

  // Update badge
  const passCount = stories.filter(s => s.passes_gate).length;
  const badge = document.getElementById('news-badge');
  if (badge) { badge.textContent = passCount; badge.style.display = passCount ? '' : 'none'; }

  list.style.display = '';
  list.innerHTML = stories.slice(0, 30).map((s, i) => {
    const pillClass = s.score >= 85 ? 'score-high' : s.score >= 65 ? 'score-mid' : 'score-low';
    const gateEl = s.passes_gate
      ? '' : `<div class="news-gate-fail">❌ ${s.gate_reasons?.join(' · ') || 'fails quality gate'}</div>`;

    return `
    <div class="news-item ${s.passes_gate ? '' : 'gated-fail'}">
      <div class="news-rank">${String(i + 1).padStart(2, '0')}</div>
      <div>
        <div class="news-title">${escHtml(s.title)}</div>
        <div class="news-meta">
          <span>${escHtml(s.source)}</span>
          <span>${s.age_hours}h ago</span>
          <span class="news-score-pill ${pillClass}">${s.score}/100</span>
          <span class="news-category">${escHtml(s.category)}</span>
        </div>
        ${gateEl}
        ${s.summary ? `<div style="font-size:12px;color:var(--text-3);margin-top:5px;line-height:1.5">${escHtml(s.summary.slice(0,160))}${s.summary.length>160?'…':''}</div>` : ''}
      </div>
      <div class="news-actions">
        ${s.passes_gate ? `
          <button class="btn-primary" style="font-size:12px;padding:6px 13px"
            onclick="generateFromNews('${escHtml(s.title)}')">
            ⚡ Generate
          </button>` : ''}
        ${s.url ? `<a href="${escHtml(s.url)}" target="_blank" rel="noopener"
          class="btn-ghost" style="font-size:11px;padding:5px 10px">↗ Read</a>` : ''}
      </div>
    </div>`;
  }).join('');
}

function generateFromNews(title) {
  document.getElementById('gen-topic').value = title;
  navTo('generate');
  toast('Topic loaded from news — click Generate Short', 'info');
}

// ── Script preview (no full generate) ─────────────────────────────────────
async function previewScript() {
  const topic    = document.getElementById('gen-topic').value.trim();
  const provider = document.getElementById('gen-provider').value;
  const model    = document.getElementById('gen-model').value.trim() || null;
  const duration = parseInt(document.getElementById('gen-duration').value) || 40;

  if (!topic) { toast('Enter a topic first', 'error'); return; }

  document.getElementById('btn-preview-script').disabled = true;
  showCard('progress-card');
  showCard('script-preview-card', false);
  showCard('scene-plan-card', false);
  showCard('prompts-card', false);
  clearLog();
  setProgress(10);
  startProgressTimer();

  try {
    const { job_id } = await api('/api/generate/script', 'POST', { topic, provider, model, duration });
    state.currentJobId = job_id;

    streamJob(job_id, {
      onLog: msg => {
        appendLog(msg.msg, msg.kind);
        const pct = inferProgress(msg.msg);
        if (pct) setProgress(pct);
      },
      onDone: result => {
        setProgress(100);
        stopProgressTimer();
        document.getElementById('btn-preview-script').disabled = false;
        renderScriptPreview(result.script, result.validation);
        toast('Script generated!', 'success');
      },
      onError: err => {
        stopProgressTimer();
        appendLog(`Error: ${err}`, 'error');
        document.getElementById('btn-preview-script').disabled = false;
        toast(`Script generation failed: ${err}`, 'error');
      },
    });
  } catch (e) {
    stopProgressTimer();
    appendLog(`Error: ${e.message}`, 'error');
    document.getElementById('btn-preview-script').disabled = false;
    toast(`Error: ${e.message}`, 'error');
  }
}

function renderScriptPreview(script, validation) {
  showCard('script-preview-card');
  document.getElementById('script-preview-text').textContent = script || '';

  const vb = document.getElementById('validation-badges');
  if (!vb || !validation) return;
  vb.innerHTML = [
    badge(validation.passes,          'Validation Pass'),
    badge(validation.has_hook,        'Hook'),
    badge(validation.has_curiosity_gap,'Curiosity Gap'),
    badge(validation.has_payoff,      'Payoff'),
    badge(validation.has_why_it_matters,'Why It Matters'),
    badge(validation.has_cta,         'CTA'),
    badge(validation.fact_safe,       'Fact Safe'),
    `<span class="badge badge-info">📝 ${validation.word_count} words</span>`,
    `<span class="badge badge-info">⏱ ~${validation.estimated_seconds}s</span>`,
    `<span class="badge badge-info">🏗 ${validation.structure_score}/10</span>`,
  ].join('');
}

function badge(ok, label) {
  return `<span class="badge ${ok ? 'badge-pass' : 'badge-fail'}">${ok ? '✓' : '✗'} ${label}</span>`;
}

// ── Full generation ───────────────────────────────────────────────────────
async function generateFull(autoNews = false) {
  const topic         = autoNews ? '' : document.getElementById('gen-topic').value.trim();
  const provider      = document.getElementById('gen-provider').value;
  const model         = document.getElementById('gen-model').value.trim() || null;
  const duration      = parseInt(document.getElementById('gen-duration').value) || 40;
  const voice         = document.getElementById('gen-voice').value.trim();
  const imageProvider = document.getElementById('gen-image-provider').value;
  const whisperModel  = document.getElementById('gen-whisper').value;

  if (!autoNews && !topic) { toast('Enter a topic first', 'error'); return; }

  const btn = document.getElementById('btn-generate-full');
  if (btn) btn.disabled = true;

  showCard('progress-card');
  showCard('script-preview-card', false);
  showCard('scene-plan-card', false);
  showCard('prompts-card', false);
  showCard('result-card', false);
  clearLog();
  setProgress(5);
  startProgressTimer();

  try {
    const { job_id } = await api('/api/generate/full', 'POST', {
      topic, provider, model, duration, voice,
      image_provider: imageProvider,
      whisper_model:  whisperModel,
      auto_news:      autoNews,
    });
    state.currentJobId = job_id;

    streamJob(job_id, {
      onLog: msg => {
        appendLog(msg.msg, msg.kind);
        const pct = inferProgress(msg.msg);
        if (pct) setProgress(pct);
      },
      onDone: result => {
        setProgress(100);
        stopProgressTimer();
        if (btn) btn.disabled = false;
        state.currentStem  = result.video_stem;
        state.currentTopic = result.script
          ? result.script.split(' ').slice(0, 6).join(' ')
          : (topic || 'AI Short');

        renderScriptPreview(result.script, null);
        if (result.scene_plan) renderScenePlan(result.scene_plan);
        if (result.prompts)    renderPrompts(result.prompts);
        renderResult(result);
        showRegenPanel(result.video_stem, topic || state.currentTopic);
        toast('✅ Short generated successfully!', 'success');
      },
      onError: err => {
        setProgress(0);
        stopProgressTimer();
        appendLog(`❌ ${err}`, 'error');
        if (btn) btn.disabled = false;
        toast(`Generation failed: ${err}`, 'error');
      },
    });
  } catch (e) {
    stopProgressTimer();
    appendLog(`Error: ${e.message}`, 'error');
    if (btn) btn.disabled = false;
    toast(`Error: ${e.message}`, 'error');
  }
}

function startAutoGenerate() {
  navTo('generate');
  setTimeout(() => generateFull(true), 200);
}

// ── Scene plan render ─────────────────────────────────────────────────────
function renderScenePlan(planData) {
  showCard('scene-plan-card');
  const scenes = planData.scenes || [];
  const quality = planData.quality || {};
  const pill = document.getElementById('scene-quality-pill');
  if (pill) pill.textContent = `Visual Quality: ${quality.score || '?'}/10`;

  const list = document.getElementById('scene-plan-list');
  if (!list) return;
  list.innerHTML = scenes.map(s => `
    <div class="scene-card">
      <div class="scene-num">${String(s.index).padStart(2,'0')}</div>
      <div>
        <div class="scene-purpose">${escHtml(s.purpose)}</div>
        <div class="scene-desc">${escHtml(s.visual_description)}</div>
        <div class="scene-meta">
          <span>🎭 ${escHtml(s.emotion)}</span>
          <span>📷 ${escHtml(s.camera_style)}</span>
          <span>🔀 ${escHtml(s.transition)}</span>
        </div>
      </div>
      <div class="scene-dur">${s.duration.toFixed(1)}s</div>
    </div>
  `).join('');
}

// ── Prompts render ────────────────────────────────────────────────────────
function renderPrompts(prompts) {
  showCard('prompts-card');
  const list = document.getElementById('prompts-list');
  if (!list) return;
  list.innerHTML = prompts.map((p, i) => `
    <div class="prompt-item">
      <div class="prompt-num">${i + 1}</div>
      <div>${escHtml(p)}</div>
    </div>
  `).join('');
}

// ── Result render ─────────────────────────────────────────────────────────
function renderResult(result) {
  showCard('result-card');
  const el = document.getElementById('result-content');
  if (!el) return;

  const report = result.report || {};
  const script = report.script || {};

  el.innerHTML = `
    <div class="result-video-wrap">
      <div class="result-video-player">
        <video src="/${result.video_file}" controls playsinline></video>
      </div>
      <div class="result-meta">
        <h3>${escHtml(result.script ? result.script.split('.')[0] : 'Generated Short')}</h3>
        <div class="quality-grid">
          <div class="quality-item">Script score: <strong>${script.structure_score || '?'}/10</strong></div>
          <div class="quality-item">Hook: <strong>${script.hook_score || '?'}/10</strong></div>
          <div class="quality-item">Words: <strong>${script.word_count || '?'}</strong></div>
          <div class="quality-item">Duration: <strong>~${script.estimated_seconds || '?'}s</strong></div>
          <div class="quality-item">Images: <strong>${report.images?.count || '?'}</strong></div>
          <div class="quality-item">Time: <strong>${result.generation_time}s</strong></div>
        </div>
        <div style="display:flex;gap:8px;flex-wrap:wrap;margin-bottom:12px">
          <span class="badge ${report.final_pass ? 'badge-pass' : 'badge-fail'}">
            ${report.final_pass ? '✅ PASS' : '❌ FAIL'}
          </span>
          <span class="badge badge-pass">✓ Fact Safe: ${script.fact_safe ? 'Yes' : 'No'}</span>
        </div>
        ${report.warnings?.length ? `
          <div style="font-size:12px;color:var(--amber)">
            ${report.warnings.map(w => `⚠ ${escHtml(w)}`).join('<br>')}
          </div>` : ''}
      </div>
    </div>`;
}

// ── Regen panel ───────────────────────────────────────────────────────────
function showRegenPanel(stem, topic) {
  state.currentStem  = stem;
  state.currentTopic = topic;
  const panel = document.getElementById('regen-panel');
  const label = document.getElementById('regen-stem-label');
  if (panel) panel.style.display = '';
  if (label) label.textContent   = stem;
}

async function regenStage(stage) {
  const stem  = state.currentStem;
  const topic = state.currentTopic || document.getElementById('gen-topic').value.trim();
  if (!stem)  { toast('No active stem — generate a Short first', 'error'); return; }
  if (!topic) { toast('Topic required', 'error'); return; }

  const provider = document.getElementById('gen-provider').value;
  const model    = document.getElementById('gen-model').value.trim() || null;

  showCard('progress-card');
  clearLog();
  setProgress(10);
  startProgressTimer();
  appendLog(`Regenerating ${stage}…`);

  try {
    const { job_id } = await api(`/api/regen/${stage}`, 'POST', {
      stem, topic, provider, model,
      voice:         document.getElementById('gen-voice').value.trim(),
      whisper_model: document.getElementById('gen-whisper').value,
    });

    streamJob(job_id, {
      onLog: msg => { appendLog(msg.msg, msg.kind); const p = inferProgress(msg.msg); if(p) setProgress(p); },
      onDone: result => {
        setProgress(100);
        stopProgressTimer();
        if (stage === 'script'     && result.script)     renderScriptPreview(result.script, result.validation);
        if (stage === 'scene_plan' && result.plan)       { renderScenePlan(result.plan); renderPrompts(result.prompts || []); }
        toast(`✅ ${stage} regenerated!`, 'success');
      },
      onError: err => {
        stopProgressTimer();
        appendLog(`Error: ${err}`, 'error');
        toast(`Regen failed: ${err}`, 'error');
      },
    });
  } catch (e) {
    stopProgressTimer();
    appendLog(`Error: ${e.message}`, 'error');
    toast(`Error: ${e.message}`, 'error');
  }
}

// ── Open output folder ────────────────────────────────────────────────────
async function openOutputFolder() {
  try {
    await api('/api/open_output_folder', 'POST', {});
    toast('Output folder opened', 'success');
  } catch (e) {
    toast(`Failed: ${e.message}`, 'error');
  }
}

// ── Toast ─────────────────────────────────────────────────────────────────
function toast(msg, type = 'info') {
  const container = document.getElementById('toast-container');
  const el = document.createElement('div');
  el.className = `toast ${type}`;
  const icon = type === 'success' ? '✅' : type === 'error' ? '❌' : 'ℹ';
  el.innerHTML = `<span>${icon}</span><span>${escHtml(msg)}</span>`;
  container.appendChild(el);
  setTimeout(() => {
    el.style.opacity = '0';
    el.style.transition = 'opacity .3s';
    setTimeout(() => el.remove(), 300);
  }, 4000);
}

// ── Utilities ─────────────────────────────────────────────────────────────
function escHtml(str) {
  return String(str || '')
    .replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')
    .replace(/"/g,'&quot;').replace(/'/g,'&#39;');
}

function setText(id, val) {
  const el = document.getElementById(id);
  if (el) el.textContent = val ?? '—';
}

// ── Init ──────────────────────────────────────────────────────────────────
loadDashboard();
