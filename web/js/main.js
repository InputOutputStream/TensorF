// main.js — nav router + app boot. Runs after every other module has
// loaded (script order enforced in index.html).

const VIEW_RENDERERS = {
  overview: renderOverview,
  jobs: renderJobsTable,
  federated: renderFederated,
  profiler: renderProfiler,
  memory: renderMemory,
};

function refreshCurrentView(light = false){
  const activeView = document.querySelector('.view.active').id.replace('view-', '');
  const renderer = VIEW_RENDERERS[activeView];
  if(!renderer) return;
  renderer();
  if(activeView === 'jobs' && light && state.selectedJobId) renderJobDetailMetrics(state.selectedJobId);
}

function initNav(){
  document.querySelectorAll('.nav-item[data-view]').forEach(el => {
    el.addEventListener('click', () => {
      document.querySelectorAll('.nav-item[data-view]').forEach(n => n.classList.remove('active'));
      el.classList.add('active');
      document.querySelectorAll('.view').forEach(v => v.classList.remove('active'));
      document.getElementById('view-' + el.dataset.view).classList.add('active');
      const renderer = VIEW_RENDERERS[el.dataset.view];
      if(renderer) renderer();
    });
  });
}

function initTopbar(){
  document.getElementById('btnReconnect').addEventListener('click', () => { pollJobs(); connectWs(); });
  document.getElementById('gwUrl').addEventListener('change', () => {
    document.getElementById('sbGwLabel').textContent = gwBase().replace(/^https?:\/\//, '');
  });
}

function boot(){
  renderDocs();
  initNav();
  initTopbar();
  initLaunchView();
  initConsoleView();

  document.getElementById('sbGwLabel').textContent = gwBase().replace(/^https?:\/\//, '');

  if(state.pollTimer) clearInterval(state.pollTimer);
  state.pollTimer = setInterval(pollJobs, 6000);

  pollJobs();
  connectWs();
  renderOverview();
}

document.addEventListener('DOMContentLoaded', boot);
