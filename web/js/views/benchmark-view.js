// views/benchmark-view.js
function renderBenchmark(job) {
  const container = document.getElementById('view-benchmark');
  const log = Array.isArray(job.log) ? job.log.join('\n') : job.log || '';
  const parsed = parseBenchmark(log);

  let html = `<h2>Benchmark Results</h2>`;

  if (parsed.results) {
    html += `<div class="results-grid">
      <div class="result-item">Matmul Peak: ${parsed.results.matmul_peak || '—'} GFLOP/s</div>
      <div class="result-item">Matmul L3: ${parsed.results.matmul_l3 || '—'} GFLOP/s</div>
      <div class="result-item">Mem BW Read: ${parsed.results.mem_bw_read || '—'} GB/s</div>
      <div class="result-item">Mem BW Write: ${parsed.results.mem_bw_write || '—'} GB/s</div>
      <div class="result-item">L3 Latency: ${parsed.results.l3_latency || '—'} ns</div>
      <div class="result-item">RAM Latency: ${parsed.results.ram_latency || '—'} ns</div>
    </div>`;
    html += `<div class="chart-container"><canvas id="benchChart"></canvas></div>`;
  }

  if (parsed.advisor) {
    html += `<div class="advisor-box">
      <h3>Hyperparameter Advisor</h3>
      <div>Hardware Score: ${parsed.advisor.hw_score}/100</div>
      <div>Config Fit: ${parsed.advisor.fit_score}/100</div>
      <div>n_embed: ${parsed.advisor.n_embed}</div>
      <div>n_heads: ${parsed.advisor.n_heads}</div>
      <div>n_layers: ${parsed.advisor.n_layers}</div>
      <div>block_size: ${parsed.advisor.block_size}</div>
      <div>batch_size: ${parsed.advisor.batch_size}</div>
      <div>Quantization: ${parsed.advisor.quant}</div>
      <div>Threads: ${parsed.advisor.threads}</div>
      <div>Params: ${parsed.advisor.params_mb} MB</div>
      <div>Training Peak: ${parsed.advisor.train_peak_mb} MB</div>
      <div>Inference Peak: ${parsed.advisor.infer_peak_mb} MB</div>
    </div>`;
  }

  if (parsed.memory) {
    html += `<div class="memory-box">
      <h3>Memory Profile</h3>
      <div>Baseline RSS: ${parsed.memory.baseline_rss} MB, Heap: ${parsed.memory.baseline_heap} MB</div>
      <div>Loaded RSS: ${parsed.memory.loaded_rss} MB, Heap: ${parsed.memory.loaded_heap} MB</div>
      <div>Infer RSS: ${parsed.memory.infer_rss} MB, Heap: ${parsed.memory.infer_heap} MB</div>
      <div>Peak RSS: ${parsed.memory.peak_rss} MB</div>
      <div>Recommended Free: ${parsed.memory.recommended_free} MB</div>
    </div>`;
  }

  container.innerHTML = html;

  setTimeout(() => {
    const ctx = document.getElementById('benchChart')?.getContext('2d');
    if (ctx && parsed.results) {
      const data = {
        labels: ['Matmul Peak', 'Matmul L3', 'Mem Read', 'Mem Write'],
        datasets: [{
          label: 'Performance',
          data: [
            parsed.results.matmul_peak || 0,
            parsed.results.matmul_l3 || 0,
            parsed.results.mem_bw_read || 0,
            parsed.results.mem_bw_write || 0
          ],
          backgroundColor: ['#4caf50', '#2196f3', '#ff9800', '#f44336']
        }]
      };
      safeChart(ctx, { type: 'bar', data, options: { responsive: true } });
    }
  }, 0);
}