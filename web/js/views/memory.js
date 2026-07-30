// views/memory.js — memory pressure per running job + free/kill control.
// Checkpointing itself is a server launch-time flag, not a runtime gateway
// action (see the explainer card already in index.html for view-memory).

function renderMemory(){
  const running = Object.values(state.jobs).filter(j => j.running || j.status === 'running');
  const grid = document.getElementById('memJobCards');
  const empty = document.getElementById('memEmpty');

  if(running.length === 0){
    grid.innerHTML = '';
    empty.style.display = 'block';
    return;
  }
  empty.style.display = 'none';

  grid.innerHTML = running.map(j => {
    const m = state.metrics[j.id] || {};
    const rss = (m.rounds && m.rounds.length && m.rounds[m.rounds.length - 1].rss_mb) || m.hyper?.param_ram_mb || 0;
    const ramTotal = m.profiler?.ram_total_mb || 0;
    const pct = ramTotal ? Math.min(100, (rss / ramTotal) * 100) : (rss ? Math.min(100, rss / 16) : 8);
    const warnCls = pct > 80 ? 'bad' : pct > 50 ? 'warn' : '';
    return `<div class="card">
      <div class="card-title">${j.kind} · <span class="job-id">${j.id.slice(0,10)}</span></div>
      <div style="display:flex; justify-content:space-between; align-items:baseline;">
        <span style="font-family:var(--mono); font-size:22px; font-weight:700;">${rss ? rss.toFixed(0) + ' MB' : '—'}</span>
        <span class="job-status running"><span class="dot"></span>running · pid ${j.pid}</span>
      </div>
      <div class="mem-bar-track"><div class="mem-bar-fill ${warnCls}" style="width:${pct}%"></div></div>
      <div class="pill-row">
        <span class="pill">command <b style="font-weight:500;">${escapeHtml((j.command || '').split('/').pop())}</b></span>
      </div>
      <div style="margin-top:12px;">
        <button class="btn sm danger" onclick="killJob('${j.id}')">⏻ Free / kill process</button>
        <button class="btn sm" onclick="goToJob('${j.id}')">View logs</button>
      </div>
    </div>`;
  }).join('');
}
