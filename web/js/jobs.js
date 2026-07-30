// jobs.js — single entry point for merging a job object (from REST or WS)
// into shared state, and fanning its bulk log tail through the parser.

function upsertJob(job){
  state.jobs[job.id] = { ...(state.jobs[job.id] || {}), ...job };
  if(!state.jobLogs[job.id]) state.jobLogs[job.id] = [];
  if(!state.metrics[job.id]) state.metrics[job.id] = emptyMetrics();

  if(Array.isArray(job.log)){
    job.log.forEach(line => ingestLogLine(job.id, line, /* fromBulk */ true));
  }

  document.getElementById('jobsBadge').textContent = Object.keys(state.jobs).length;
  document.getElementById('sbJobCount').textContent = Object.keys(state.jobs).length;
}

function goToJob(id){
  document.querySelector('.nav-item[data-view="jobs"]').click();
  selectJob(id);
}
