// api.js — talks to the TensorF gateway (docs/api/gateway.html):
//   REST:  GET /api/health, GET /api/jobs, GET /api/jobs/{id},
//          POST /api/jobs/{benchmark|tests|gpt2|smollm|server|client},
//          POST /api/jobs/{id}/kill
//   WS:    /ws  -> {"type":"job_started"|"log", ...}

async function api(path, opts = {}){
  const url = gwBase() + path;
  const res = await fetch(url, { headers: {'Content-Type':'application/json'}, ...opts });
  if(!res.ok){
    let body = {};
    try{ body = await res.json(); }catch(e){}
    throw new Error(body.error || ('HTTP ' + res.status));
  }
  return res.json();
}

function connectWs(){
  const wsUrl = gwBase().replace(/^http/, 'ws') + '/ws';
  if(state.ws){ try{ state.ws.close(); }catch(e){} }
  setWsStatus('connecting');

  let ws;
  try{ ws = new WebSocket(wsUrl); }
  catch(e){ setWsStatus('down'); return; }

  state.ws = ws;
  ws.onopen = () => setWsStatus('live');
  ws.onclose = () => {
    setWsStatus('down');
    setTimeout(() => { if(state.ws === ws) connectWs(); }, 4000);
  };
  ws.onerror = () => setWsStatus('down');
  ws.onmessage = (ev) => {
    let msg;
    try{ msg = JSON.parse(ev.data); }catch(e){ return; }
    handleWsMessage(msg);
  };
}

function setWsStatus(s){
  const dot = document.getElementById('wsDot');
  const label = document.getElementById('wsLabel');
  dot.className = 'dot' + (s === 'live' ? ' live' : s === 'down' ? ' down' : '');
  label.textContent = s === 'live' ? 'live' : s === 'down' ? 'disconnected' : 'connecting…';
}

function handleWsMessage(msg){
  appendGlobalConsole(msg);

  if(msg.type === 'job_started' && msg.job){
    upsertJob(msg.job);
    toast(`Job started: ${msg.job.kind} (${msg.job.id.slice(0,8)})`, 'ok');
    refreshCurrentView();
  } else if(msg.type === 'log' && msg.job_id){
    ingestLogLine(msg.job_id, msg.line);
    if(state.selectedJobId === msg.job_id) appendJobConsole(msg.job_id, msg.line);
    refreshCurrentView(true);
  }
}

// REST fallback poll — the gateway has no job_exited WS event yet
// (documented as "not yet implemented" in docs/api/gateway.html),
// so we poll to catch status transitions the WS stream won't announce.
async function pollJobs(){
  try{
    const data = await api('/api/jobs');
    (data.jobs || []).forEach(j => upsertJob(j));
    refreshCurrentView();
    setWsStatus(state.ws && state.ws.readyState === 1 ? 'live' : 'connecting');
  }catch(e){
    setWsStatus('down');
  }
}

async function killJob(id){
  try{
    const res = await api(`/api/jobs/${id}/kill`, { method: 'POST' });
    if(res.killed) toast('SIGTERM sent — memory will free shortly', 'ok');
    else toast('Job already stopped or not found', 'err');
    setTimeout(pollJobs, 800);
  }catch(e){
    toast('Kill failed: ' + e.message, 'err');
  }
}

async function launchJob(kind, body){
  const job = await api(`/api/jobs/${kind}`, { method: 'POST', body: JSON.stringify(body || {}) });
  upsertJob(job);
  return job;
}

async function fetchJobDetail(id){
  const detail = await api(`/api/jobs/${id}`);
  upsertJob(detail);
  return detail;
}
