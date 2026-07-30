// views/overview.js

let lossChartInstance = null;

function renderOverview(){
  const jobs = Object.values(state.jobs);
  const running = jobs.filter(j => j.running || j.status === 'running');
  const failed = jobs.filter(j => j.status === 'failed');
  const totalLossPoints = Object.values(state.metrics).reduce((a, m) => a + (m.loss?.length || 0), 0);
  const totalRounds = Object.values(state.metrics).reduce((a, m) => a + (m.rounds?.length || 0), 0);

  document.getElementById('overviewStats').innerHTML = `
    <div class="stat ${running.length ? 'ok' : ''}"><div class="stat-label">Running jobs</div><div class="stat-value">${running.length}</div><div class="stat-sub">of ${jobs.length} tracked</div></div>
    <div class="stat ${failed.length ? 'bad' : 'ok'}"><div class="stat-label">Failed jobs</div><div class="stat-value">${failed.length}</div><div class="stat-sub">since gateway start</div></div>
    <div class="stat"><div class="stat-label">FL rounds observed</div><div class="stat-value">${totalRounds}</div><div class="stat-sub">across server + client logs</div></div>
    <div class="stat"><div class="stat-label">Loss samples</div><div class="stat-value">${totalLossPoints}</div><div class="stat-sub">parsed from stdout</div></div>
  `;

  renderJobRows(document.querySelector('#overviewJobsTable tbody'), jobs.slice(-8).reverse());
  renderLossChart();
  renderActivityFeed();
}

function renderActivityFeed(){
  const el = document.getElementById('activityFeed');
  const jobs = Object.values(state.jobs).sort((a, b) => (b.log_seq || 0) - (a.log_seq || 0));
  if(jobs.length === 0){
    el.innerHTML = '<div class="empty-state" style="padding:24px;">No activity yet.</div>';
    return;
  }
  el.innerHTML = jobs.slice(0, 10).map(j => {
    const lines = state.jobLogs[j.id] || [];
    const last = lines[lines.length - 1] || '(no output yet)';
    return `<div style="border-left:2px solid var(--line); padding-left:10px;">
      <div style="font-family:var(--mono); font-size:11px; color:var(--cyan);">${j.kind} · ${j.id.slice(0,10)}</div>
      <div style="font-size:12px; color:var(--text-dim); font-family:var(--mono); overflow:hidden; text-overflow:ellipsis; white-space:nowrap;">${escapeHtml(last)}</div>
    </div>`;
  }).join('');
}

function renderLossChart(){
  const ctx = document.getElementById('lossChart');
  const datasets = [];
  let ci = 0;

  Object.entries(state.metrics).forEach(([id, m]) => {
    if(!m.loss || m.loss.length === 0) return;
    const job = state.jobs[id];
    datasets.push({
      label: `${job ? job.kind : 'job'} ${id.slice(0,6)}`,
      data: m.loss.map(p => ({ x: p.idx, y: p.val })),
      borderColor: CHART_COLORS[ci % CHART_COLORS.length],
      backgroundColor: 'transparent',
      pointRadius: 2,
      tension: .25,
      borderWidth: 2,
    });
    ci++;
  });

  document.getElementById('lossLegend').textContent = datasets.length ? `(${datasets.length} job${datasets.length > 1 ? 's' : ''})` : '';

  if(lossChartInstance) lossChartInstance.destroy();
  if(datasets.length === 0){
    ctx.getContext('2d').clearRect(0, 0, ctx.width, ctx.height);
    return;
  }
  lossChartInstance = new Chart(ctx, { type: 'line', data: { datasets }, options: chartBaseOptions('sample', 'loss') });
}
