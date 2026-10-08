// charts.js — shared Chart.js helpers (all functions global)

function safeChart(ctx, config) {
  if (typeof Chart === 'undefined') {
    console.error('Chart.js library not loaded.');
    const parent = ctx?.parentNode;
    if (parent) {
      parent.innerHTML = `<div class="chart-error">⚠️ Chart library unavailable. Check network / ad‑blocker.</div>`;
    }
    return null;
  }
  try {
    return new Chart(ctx, config);
  } catch (e) {
    console.error('Chart creation failed:', e);
    return null;
  }
}

const CHART_COLORS = ['#e8a33d', '#5ecbd8', '#6fbf73', '#d9645a', '#c792e0', '#f0d264'];

function chartBaseOptions(xlabel, ylabel) {
  return {
    responsive: true,
    maintainAspectRatio: false,
    interaction: { mode: 'nearest', intersect: false },
    plugins: {
      legend: { display: true, labels: { color: '#93998f', font: { family: 'JetBrains Mono', size: 10 } } },
    },
    scales: {
      x: { type: 'linear', title: { display: true, text: xlabel, color: '#5c6259' }, ticks: { color: '#5c6259' }, grid: { color: '#202620' } },
      y: { title: { display: true, text: ylabel, color: '#5c6259' }, ticks: { color: '#5c6259' }, grid: { color: '#202620' } },
    },
  };
}

function chartBarOptions() {
  return {
    responsive: true,
    maintainAspectRatio: false,
    plugins: { legend: { display: false } },
    scales: {
      x: { ticks: { color: '#93998f' }, grid: { display: false } },
      y: { ticks: { color: '#5c6259' }, grid: { color: '#202620' } },
    },
  };
}

function enableChartInteractions(canvasId, chartInstance) {
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;
  canvas.addEventListener('click', () => {
    const modal = document.getElementById('chartZoomModal');
    const modalImg = document.getElementById('chartZoomImg');
    if (modal && modalImg) {
      modalImg.src = canvas.toDataURL('image/png');
      modal.style.display = 'block';
    }
  });
  const container = canvas.closest('.chart-wrap') || canvas.parentElement;
  if (container && !container.querySelector('.print-chart-btn')) {
    const btn = document.createElement('button');
    btn.className = 'btn sm print-chart-btn';
    btn.textContent = '🖨 Print';
    btn.style.position = 'absolute';
    btn.style.top = '8px';
    btn.style.right = '8px';
    btn.style.zIndex = '10';
    btn.onclick = (e) => {
      e.stopPropagation();
      const printContents = container.innerHTML;
      const win = window.open('', '_blank');
      win.document.write(`
        <html><head><title>Print Chart</title>
        <style>body { background: white; display: flex; justify-content: center; align-items: center; height: 100vh; } img { max-width: 100%; }</style>
        </head><body>${printContents}</body></html>
      `);
      win.document.close();
      win.focus();
      win.print();
    };
    container.appendChild(btn);
  }
}

function closeZoomModal() {
  document.getElementById('chartZoomModal').style.display = 'none';
}