// views/jobs-view.js — jobs table + selected-job detail panel

let jdChartInstance = null;

function renderJobRows(tbody, jobs){
  if(jobs.length === 0){
    tbody.innerHTML = `<tr class="empty-row"><td colspan="7">no jobs yet — launch one to get started</td></tr>`;
    return;
  }
  tbody.innerHTML = jobs.map(j => {
    const cls = statusClass(j);
    return `<tr data-id="${j.id}" class="${state.selectedJobId === j.id ? 'selected' : ''}">
      <td><span class="job-kind">${j.kind}</span></td>
      <td class="job-id">${j.id.slice(0,12)}</td>
      <td><span class="job-status ${cls}"><span class="dot" style="background:${statusDotColor(cls)}"></span>${j.status || 'running'}</span></td>
      <td class="job-id">${j.pid ?? '—'}</td>
      <td class="job-cmd" title="${escapeAttr(j.command || '')}">${escapeHtml(j.command || '')}</td>
      <td style="text-align:right; white-space:nowrap;">
        <button class="btn sm" data-view-job="${j.id}">View</button>
        ${(j.running || j.status === 'running') ? `<button class="btn sm danger" data-kill-job="${j.id}">Kill</button>` : ''}
      </td>
    </tr>`;
  }).join('');

  tbody.querySelectorAll('[data-view-job]').forEach(b => b.addEventListener('click', () => goToJob(b.dataset.viewJob)));
  tbody.querySelectorAll('[data-kill-job]').forEach(b => b.addEventListener('click', (e) => { e.stopPropagation(); killJob(b.dataset.killJob); }));
  tbody.querySelectorAll('tr[data-id]').forEach(tr => tr.addEventListener('click', () => selectJob(tr.dataset.id)));
}

function renderJobsTable(){
  const jobs = Object.values(state.jobs).sort((a, b) => (b.started_at || '').localeCompare(a.started_at || ''));
  renderJobRows(document.querySelector('#jobsTable tbody'), jobs);
}

async function selectJob(id){
  state.selectedJobId = id;
  document.getElementById('jobDetailWrap').style.display = 'block';
  document.getElementById('jdId').textContent = id;
  renderJobsTable();

  try{ await fetchJobDetail(id); }catch(e){ /* fall back to cached data */ }

  const job = state.jobs[id];
  const cls = statusClass(job);
  document.getElementById('jdPills').innerHTML = `
    <span class="pill">kind <b>${job.kind}</b></span>
    <span class="pill">status <b style="color:${statusDotColor(cls)}">${job.status || 'running'}</b></span>
    <span class="pill">pid <b>${job.pid}</b></span>
    <span class="pill">log lines <b>${job.log_seq ?? (state.jobLogs[id] || []).length}</b></span>
    ${(job.running || job.status === 'running') ? `<button class="btn sm danger" onclick="killJob('${id}')">⏻ Kill / free</button>` : ''}
  `;

  const consoleEl = document.getElementById('jdConsole');
  const lines = state.jobLogs[id] || [];
  document.getElementById('jdLogCount').textContent = `(${lines.length})`;
  consoleEl.innerHTML = lines.length ? lines.map(renderLogLine).join('') : '<div class="console-empty">no output yet</div>';
  consoleEl.scrollTop = consoleEl.scrollHeight;

  renderJobDetailMetrics(id);
}

function renderJobDetailMetrics(id){
  if(state.selectedJobId !== id) return;
  const m = state.metrics[id] || { loss: [] };
  const ctx = document.getElementById('jdChart');
  const emptyEl = document.getElementById('jdMetricsEmpty');

  if(jdChartInstance) jdChartInstance.destroy();
  if(!m.loss || m.loss.length === 0){
    emptyEl.style.display = 'block';
    return;
  }
  emptyEl.style.display = 'none';
  jdChartInstance = new Chart(ctx, {
    type: 'line',
    data: { datasets: [{
      label: 'loss', data: m.loss.map(p => ({ x: p.idx, y: p.val })),
      borderColor: '#e8a33d', backgroundColor: 'rgba(232,163,61,.08)', fill: true, pointRadius: 2, tension: .25, borderWidth: 2,
    }]},
    options: chartBaseOptions('sample', 'loss'),
  });

  const consoleEl = document.getElementById('jdConsole');
  const lines = state.jobLogs[id] || [];
  document.getElementById('jdLogCount').textContent = `(${lines.length})`;
}

function appendJobConsole(id, line){
  if(state.selectedJobId !== id) return;
  const consoleEl = document.getElementById('jdConsole');
  if(consoleEl.querySelector('.console-empty')) consoleEl.innerHTML = '';
  consoleEl.insertAdjacentHTML('beforeend', renderLogLine(line));
  consoleEl.scrollTop = consoleEl.scrollHeight;
}
