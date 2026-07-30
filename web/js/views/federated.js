// views/federated.js — FedAvg/FedDistill round telemetry, parsed live
// from server & client job logs (see js/parser.js for the regexes).

let flTimingChartInst = null, flClientsChartInst = null, flBandwidthChartInst = null;

function renderFederated(){
  const allRounds = [];
  Object.entries(state.metrics).forEach(([id, m]) => {
    (m.rounds || []).forEach(r => allRounds.push({ ...r, jobId: id, jobKind: state.jobs[id]?.kind }));
  });

  const serverRounds = allRounds.filter(r => r.side === 'server');
  const clientRounds = allRounds.filter(r => r.side === 'client');
  const maxRound = allRounds.reduce((a, r) => Math.max(a, r.round || 0), 0);
  const lastClientsN = serverRounds.length ? serverRounds[serverRounds.length - 1].clients : '—';

  document.getElementById('flStats').innerHTML = `
    <div class="stat"><div class="stat-label">Latest round</div><div class="stat-value">${maxRound || '—'}</div><div class="stat-sub">max observed across jobs</div></div>
    <div class="stat"><div class="stat-label">Server rounds logged</div><div class="stat-value">${serverRounds.length}</div><div class="stat-sub">from server job stdout</div></div>
    <div class="stat"><div class="stat-label">Client rounds logged</div><div class="stat-value">${clientRounds.length}</div><div class="stat-sub">FedAvg + FedDistill</div></div>
    <div class="stat"><div class="stat-label">Clients (last round)</div><div class="stat-value">${lastClientsN}</div><div class="stat-sub">reported by server</div></div>
  `;

  renderTimingChart(serverRounds);
  renderClientsChart(serverRounds);
  renderBandwidthChart(clientRounds);
  renderRoundsTable(allRounds);
}

function renderTimingChart(serverRounds){
  const ctx = document.getElementById('flTimingChart');
  if(flTimingChartInst) flTimingChartInst.destroy();
  const timed = serverRounds.filter(r => r.total_ms != null);
  if(!timed.length) return;
  flTimingChartInst = new Chart(ctx, {
    type: 'line',
    data: { datasets: [
      { label: 'aggregate ms', data: timed.map(r => ({ x: r.round, y: r.aggregate_ms || 0 })), borderColor: '#5ecbd8', borderWidth: 2, pointRadius: 2, tension: .2 },
      { label: 'round total ms', data: timed.map(r => ({ x: r.round, y: r.total_ms || 0 })), borderColor: '#e8a33d', borderWidth: 2, pointRadius: 2, tension: .2 },
    ]},
    options: chartBaseOptions('round', 'ms'),
  });
}

function renderClientsChart(serverRounds){
  const ctx = document.getElementById('flClientsChart');
  if(flClientsChartInst) flClientsChartInst.destroy();
  const withClients = serverRounds.filter(r => r.clients != null);
  if(!withClients.length) return;
  flClientsChartInst = new Chart(ctx, {
    type: 'bar',
    data: { datasets: [{ label: 'clients', data: withClients.map(r => ({ x: r.round, y: r.clients })), backgroundColor: '#7a5a25' }] },
    options: chartBaseOptions('round', 'clients'),
  });
}

function renderBandwidthChart(clientRounds){
  const ctx = document.getElementById('flBandwidthChart');
  if(flBandwidthChartInst) flBandwidthChartInst.destroy();
  const bwRounds = clientRounds.filter(r => r.send_mbps != null || r.recv_mbps != null);
  if(!bwRounds.length) return;
  flBandwidthChartInst = new Chart(ctx, {
    type: 'line',
    data: { datasets: [
      { label: 'send deltas MB/s', data: bwRounds.map(r => ({ x: r.round, y: r.send_mbps || 0 })), borderColor: '#d9645a', borderWidth: 2, pointRadius: 2, tension: .2 },
      { label: 'recv weights MB/s', data: bwRounds.map(r => ({ x: r.round, y: r.recv_mbps || 0 })), borderColor: '#6fbf73', borderWidth: 2, pointRadius: 2, tension: .2 },
    ]},
    options: chartBaseOptions('round', 'MB/s'),
  });
}

function renderRoundsTable(allRounds){
  const tbody = document.querySelector('#flRoundsTable tbody');
  if(allRounds.length === 0){
    tbody.innerHTML = `<tr class="empty-row"><td colspan="6">no round data yet — launch a server + client to populate this</td></tr>`;
    return;
  }
  tbody.innerHTML = allRounds.slice(-40).reverse().map(r => {
    const lossForJob = state.metrics[r.jobId]?.loss || [];
    const lossVal = lossForJob.length ? lossForJob[lossForJob.length - 1].val : '—';
    return `<tr>
      <td class="job-id">${(r.jobKind || '?')} ${r.jobId.slice(0,8)}</td>
      <td>${r.round}</td>
      <td>${r.side}${r.mode ? (' / ' + r.mode) : ''}</td>
      <td>${lossVal}</td>
      <td>${r.total_ms ?? '—'}</td>
      <td>${r.clients ?? '—'}</td>
    </tr>`;
  }).join('');
}
