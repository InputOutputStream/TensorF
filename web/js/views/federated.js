// views/federated.js — FedAvg/FedDistill round telemetry, per‑instance filtering
let flTimingChartInst = null, flClientsChartInst = null, flBandwidthChartInst = null;
let selectedFederatedInstance = null;

function getFlInstances() {
  return Object.keys(state.metrics).filter(id =>
    state.metrics[id].rounds && state.metrics[id].rounds.length > 0
  );
}

function renderInstanceCards() {
  const container = document.getElementById('flInstanceCards');
  const instances = getFlInstances();
  if (instances.length === 0) {
    container.innerHTML = '<div class="empty-state">No FL instances with round data yet.</div>';
    return;
  }
  container.innerHTML = instances.map(id => {
    const job = state.jobs[id];
    const m = state.metrics[id];
    const lastRound = m.rounds[m.rounds.length - 1];
    const loss = m.loss.length ? m.loss[m.loss.length - 1].val : '—';
    return `<div class="instance-card" data-id="${id}">
      <span class="badge ${job?.status || 'running'}">${job?.status || 'running'}</span>
      <strong>${job?.kind || 'unknown'}</strong> (${id.slice(0,8)})
      <span>Round ${lastRound.round || '—'}</span>
      <span>Loss: ${typeof loss === 'number' ? loss.toFixed(4) : loss}</span>
      <button class="btn sm view-fl-instance" data-id="${id}">View</button>
    </div>`;
  }).join('');

  container.querySelectorAll('.view-fl-instance').forEach(btn => {
    btn.addEventListener('click', () => {
      selectedFederatedInstance = btn.dataset.id;
      renderFederatedDetail(selectedFederatedInstance);
    });
  });
}

function renderFederatedDetail(instanceId) {
  const m = state.metrics[instanceId];
  if (!m) return;
  const rounds = m.rounds || [];
  const serverRounds = rounds.filter(r => r.side === 'server');
  const clientRounds = rounds.filter(r => r.side === 'client');

  const maxRound = rounds.reduce((a, r) => Math.max(a, r.round || 0), 0);
  const lastClientsN = serverRounds.length ? serverRounds[serverRounds.length - 1].clients : '—';
  document.getElementById('flStats').innerHTML = `
    <div class="stat"><div class="stat-label">Latest round</div><div class="stat-value">${maxRound || '—'}</div><div class="stat-sub">for this instance</div></div>
    <div class="stat"><div class="stat-label">Server rounds</div><div class="stat-value">${serverRounds.length}</div><div class="stat-sub">logged</div></div>
    <div class="stat"><div class="stat-label">Client rounds</div><div class="stat-value">${clientRounds.length}</div><div class="stat-sub">logged</div></div>
    <div class="stat"><div class="stat-label">Clients (last)</div><div class="stat-value">${lastClientsN}</div><div class="stat-sub">reported</div></div>
  `;

  renderTimingChart(serverRounds);
  renderClientsChart(serverRounds);
  renderBandwidthChart(clientRounds);
  renderRoundsTable(rounds);
}

function renderTimingChart(serverRounds) {
  const ctx = document.getElementById('flTimingChart');
  if (flTimingChartInst) flTimingChartInst.destroy();
  const timed = serverRounds.filter(r => r.total_ms != null);
  if (!timed.length) {
    document.getElementById('flTimingChart').parentElement.innerHTML = '<div class="empty-state">No timing data for this instance.</div>';
    return;
  }
  flTimingChartInst = safeChart(ctx, {
    type: 'line',
    data: {
      datasets: [
        { label: 'aggregate ms', data: timed.map(r => ({ x: r.round, y: r.aggregate_ms || 0 })), borderColor: '#5ecbd8', borderWidth: 2, pointRadius: 2, tension: .2 },
        { label: 'round total ms', data: timed.map(r => ({ x: r.round, y: r.total_ms || 0 })), borderColor: '#e8a33d', borderWidth: 2, pointRadius: 2, tension: .2 },
      ]
    },
    options: chartBaseOptions('round', 'ms'),
  });
  enableChartInteractions(ctx.canvas.id, flTimingChartInst);
}

function renderClientsChart(serverRounds) {
  const ctx = document.getElementById('flClientsChart');
  if (flClientsChartInst) flClientsChartInst.destroy();
  const withClients = serverRounds.filter(r => r.clients != null);
  if (!withClients.length) {
    document.getElementById('flClientsChart').parentElement.innerHTML = '<div class="empty-state">No client count data.</div>';
    return;
  }
  flClientsChartInst = safeChart(ctx, {
    type: 'bar',
    data: { datasets: [{ label: 'clients', data: withClients.map(r => ({ x: r.round, y: r.clients })), backgroundColor: '#7a5a25' }] },
    options: chartBaseOptions('round', 'clients'),
  });
  enableChartInteractions(ctx.canvas.id, flClientsChartInst);
}

function renderBandwidthChart(clientRounds) {
  const ctx = document.getElementById('flBandwidthChart');
  if (flBandwidthChartInst) flBandwidthChartInst.destroy();
  const bwRounds = clientRounds.filter(r => r.send_mbps != null || r.recv_mbps != null);
  if (!bwRounds.length) {
    document.getElementById('flBandwidthChart').parentElement.innerHTML = '<div class="empty-state">No bandwidth data.</div>';
    return;
  }
  flBandwidthChartInst = safeChart(ctx, {
    type: 'line',
    data: {
      datasets: [
        { label: 'send deltas MB/s', data: bwRounds.map(r => ({ x: r.round, y: r.send_mbps || 0 })), borderColor: '#d9645a', borderWidth: 2, pointRadius: 2, tension: .2 },
        { label: 'recv weights MB/s', data: bwRounds.map(r => ({ x: r.round, y: r.recv_mbps || 0 })), borderColor: '#6fbf73', borderWidth: 2, pointRadius: 2, tension: .2 },
      ]
    },
    options: chartBaseOptions('round', 'MB/s'),
  });
  enableChartInteractions(ctx.canvas.id, flBandwidthChartInst);
}

function renderRoundsTable(allRounds) {
  const tbody = document.querySelector('#flRoundsTable tbody');
  if (allRounds.length === 0) {
    tbody.innerHTML = `<tr class="empty-row"><td colspan="6">no rounds for this instance</td></tr>`;
    return;
  }
  tbody.innerHTML = allRounds.slice(-40).reverse().map(r => {
    return `<tr>
      <td class="job-id">${r.side} ${r.mode ? ('/' + r.mode) : ''}</td>
      <td>${r.round}</td>
      <td>${r.side}</td>
      <td>${r.loss ?? '—'}</td>
      <td>${r.total_ms ?? '—'}</td>
      <td>${r.clients ?? '—'}</td>
    </tr>`;
  }).join('');
}

function renderFederated() {
  const container = document.getElementById('view-federated');
  if (!container.querySelector('#flInstanceCards')) {
    container.innerHTML = `
      <div class="view-header"><div><div class="view-title">Federated learning rounds</div><div class="view-desc">Per‑instance round telemetry.</div></div></div>
      <div id="flInstanceCards" class="instance-grid"></div>
      <div class="grid cols-4" id="flStats"></div>
      <div class="section-sub">Round timing breakdown (server)</div>
      <div class="two-col">
        <div class="card"><div class="card-title">Aggregate time vs round total (ms)</div><div class="chart-wrap tall"><canvas id="flTimingChart"></canvas></div></div>
        <div class="card"><div class="card-title">Clients per round</div><div class="chart-wrap tall"><canvas id="flClientsChart"></canvas></div></div>
      </div>
      <div class="section-sub">Client transfer — bandwidth per round</div>
      <div class="card"><div class="card-title">Send deltas / receive weights (MB/s)</div><div class="chart-wrap"><canvas id="flBandwidthChart"></canvas></div></div>
      <div class="section-sub">Round log</div>
      <div class="card" style="padding:0;"><table class="jobs-table" id="flRoundsTable"><thead><tr><th>Job</th><th>Round</th><th>Side</th><th>Loss</th><th>Time (ms)</th><th>Clients</th></tr></thead><tbody></tbody></table></div>
    `;
  }

  renderInstanceCards();
  const instances = getFlInstances();
  if (instances.length && (!selectedFederatedInstance || !instances.includes(selectedFederatedInstance))) {
    selectedFederatedInstance = instances[0];
  }
  if (selectedFederatedInstance) {
    renderFederatedDetail(selectedFederatedInstance);
  }
}