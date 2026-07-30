// utils.js — small pure helpers shared across views.

function escapeHtml(s){
  return String(s).replace(/[&<>"]/g, c => ({ '&':'&amp;', '<':'&lt;', '>':'&gt;', '"':'&quot;' }[c]));
}
function escapeAttr(s){ return escapeHtml(s); }

function statusClass(job){
  if(job.running || job.status === 'running') return 'running';
  if(job.status === 'failed') return 'failed';
  if(job.status === 'killed') return 'killed';
  return 'exited';
}
function statusDotColor(cls){
  return { running:'var(--green)', failed:'var(--red)', killed:'var(--amber)', exited:'var(--text-faint)' }[cls];
}
