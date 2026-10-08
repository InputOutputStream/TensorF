// views/tests-view.js
function renderTests(job) {
  const container = document.getElementById('view-tests');
  const log = Array.isArray(job.log) ? job.log.join('\n') : job.log || '';
  const parsed = parseTests(log);

  let html = `<h2>Basic Tests</h2>`;

  if (parsed.sections) {
    html += `<div class="test-summary">`;
    for (const [section, results] of Object.entries(parsed.sections)) {
      const pass = results.filter(r => r === 'PASS').length;
      const fail = results.filter(r => r === 'FAIL').length;
      const color = fail === 0 ? 'green' : 'red';
      html += `<div class="test-section">
        <h4>${section}</h4>
        <span style="color:${color}">${pass} passed, ${fail} failed</span>
      </div>`;
    }
    html += `</div>`;
  }

  if (parsed.losses && parsed.losses.length > 0) {
    html += `<div class="chart-container"><canvas id="lossChart"></canvas></div>`;
  }

  html += `<div class="log-console">
    <button class="copy-log">Copy Log</button>
    <pre>${job.log || ''}</pre>
  </div>`;

  container.innerHTML = html;

  if (parsed.losses && parsed.losses.length > 0) {
    setTimeout(() => {
      const ctx = document.getElementById('lossChart')?.getContext('2d');
      if (ctx) {
        const data = {
          labels: parsed.losses.map((_, i) => i),
          datasets: [{ label: 'Training Loss', data: parsed.losses, borderColor: 'purple', fill: false }]
        };
        safeChart(ctx, { type: 'line', data, options: { responsive: true } });
      }
    }, 0);
  }

  container.querySelector('.copy-log')?.addEventListener('click', () => {
    const logText = Array.isArray(job.log) ? job.log.join('\n') : job.log || '';
    navigator.clipboard.writeText(logText).catch(() => {});
  });
}