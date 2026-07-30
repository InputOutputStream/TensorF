// charts.js — one shared Chart.js option builder so every view's chart
// looks consistent without copy-pasting the theme block everywhere.

const CHART_COLORS = ['#e8a33d', '#5ecbd8', '#6fbf73', '#d9645a', '#c792e0', '#f0d264'];

function chartBaseOptions(xlabel, ylabel){
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

function chartBarOptions(){
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
