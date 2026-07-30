// views/profiler.js — renders the Profiler fingerprint block every
// `client` job prints on startup (see docs/api/profiler.html).

let benchChartInst = null, memStageChartInst = null;
const ISA_LEVELS = ['BASELINE', 'AVX', 'AVX2', 'AVX512', 'AVX512_VNNI', 'AMX'];

function renderProfiler(){
  const sel = document.getElementById('profJobSelect');
  const candidates = Object.entries(state.metrics).filter(([, m]) => m.profiler && (m.profiler.cpu_model || m.profiler.isa));

  if(candidates.length === 0){
    document.getElementById('profilerEmpty').style.display = 'block';
    document.getElementById('profilerContent').style.display = 'none';
    sel.innerHTML = '<option>no profiler data</option>';
    return;
  }

  document.getElementById('profilerEmpty').style.display = 'none';
  document.getElementById('profilerContent').style.display = 'block';

  const prevVal = sel.value;
  sel.innerHTML = candidates.map(([id]) => `<option value="${id}">${state.jobs[id]?.kind || 'job'} · ${id.slice(0,10)}</option>`).join('');
  if(candidates.find(([id]) => id === prevVal)) sel.value = prevVal;
  sel.onchange = () => renderProfilerContent(sel.value);
  renderProfilerContent(sel.value || candidates[candidates.length - 1][0]);
}

function renderProfilerContent(id){
  const m = state.metrics[id];
  if(!m) return;
  const p = m.profiler || {};
  const b = m.bench || {};
  const h = m.hyper || {};

  document.getElementById('probeCpu').innerHTML = `
    <tr><td>Model</td><td>${p.cpu_model || '—'}</td></tr>
    <tr><td>ISA (best)</td><td>${p.isa || '—'}</td></tr>
    <tr><td>Cores</td><td>${p.phys_cores ? `${p.phys_cores}p / ${p.log_cores}l` : '—'}</td></tr>
    <tr><td>Max freq</td><td>${p.max_freq_mhz ? p.max_freq_mhz + ' MHz' : '—'}</td></tr>
  `;
  document.getElementById('isaBadges').innerHTML = ISA_LEVELS.map(l =>
    `<span class="isa-badge ${p.isa === l ? 'has' : ''}">${l}</span>`).join('');

  document.getElementById('probeCache').innerHTML = `
    <tr><td>L3</td><td>${p.l3_mb ? p.l3_mb + ' MB' : '—'}</td></tr>
    <tr><td>L1 latency</td><td>${b.l1_ns ? b.l1_ns + ' ns' : '—'}</td></tr>
    <tr><td>L3 latency</td><td>${b.l3_ns ? b.l3_ns + ' ns' : '—'}</td></tr>
    <tr><td>RAM latency</td><td>${b.ram_ns ? b.ram_ns + ' ns' : '—'}</td></tr>
  `;

  document.getElementById('probeRam').innerHTML = `
    <tr><td>Type</td><td>${p.ram_type || '—'} ${p.ram_speed ? ('@ ' + p.ram_speed + ' MT/s') : ''}</td></tr>
    <tr><td>Total</td><td>${p.ram_total_mb ? p.ram_total_mb + ' MB' : '—'}</td></tr>
    <tr><td>Available</td><td>${(p.ram_avail_mb2 || p.ram_avail_mb) ? (p.ram_avail_mb2 || p.ram_avail_mb) + ' MB' : '—'}</td></tr>
    <tr><td>Bandwidth</td><td>${p.ram_bw_gbs ? p.ram_bw_gbs + ' GB/s' : (b.mem_read_gbs ? b.mem_read_gbs + ' GB/s' : '—')}</td></tr>
  `;

  document.getElementById('probeHyper').innerHTML = `
    <tr><td>batch_size</td><td>${h.batch_size ?? '—'}</td></tr>
    <tr><td>block_size</td><td>${h.block_size ?? '—'}</td></tr>
    <tr><td>quant policy</td><td>${h.quant || '—'}</td></tr>
    <tr><td>param RAM est.</td><td>${h.param_ram_mb ? h.param_ram_mb + ' MB' : '—'}</td></tr>
  `;

  renderBenchChart(b);
  renderMemStageChart(p, h, m);
}

function renderBenchChart(b){
  const ctx = document.getElementById('benchGflopsChart');
  if(benchChartInst) benchChartInst.destroy();
  benchChartInst = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: ['Matmul (L3)', 'Matmul (RAM)'],
      datasets: [{ label: 'GFLOP/s', data: [b.matmul_l3 || 0, b.matmul_ram || 0], backgroundColor: ['#e8a33d', '#5ecbd8'] }],
    },
    options: chartBarOptions(),
  });
}

function renderMemStageChart(p, h, m){
  const ctx = document.getElementById('memStageChart');
  if(memStageChartInst) memStageChartInst.destroy();
  const rss = m.rounds && m.rounds.length ? m.rounds[m.rounds.length - 1].rss_mb : null;
  memStageChartInst = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: ['baseline', 'loaded', 'train', 'infer'],
      datasets: [{
        label: 'RSS MB',
        data: [
          p.ram_avail_mb ? (p.ram_total_mb - p.ram_avail_mb) : 0,
          h.param_ram_mb || 0,
          rss || 0,
          rss || 0,
        ],
        backgroundColor: '#7a5a25',
      }],
    },
    options: chartBarOptions(),
  });
}
