// views/console-view.js — the unfiltered /ws firehose, color-coded.

function appendGlobalConsole(msg){
  const el = document.getElementById('globalConsole');
  if(el.querySelector('.console-empty')) el.innerHTML = '';

  let line, cls = '';
  if(msg.type === 'job_started'){
    line = `▶ job_started  kind=${msg.job.kind}  id=${msg.job.id}  pid=${msg.job.pid}`;
    cls = 'hl-round';
  }else if(msg.type === 'log'){
    line = `[${msg.job_id.slice(0,8)}] ${msg.line}`;
    cls = classifyLogLine(msg.line);
  }else{
    line = JSON.stringify(msg);
  }

  el.insertAdjacentHTML('beforeend', `<div class="ln ${cls}">${escapeHtml(line)}</div>`);
  if(document.getElementById('consoleAutoscroll').checked) el.scrollTop = el.scrollHeight;

  while(el.children.length > 2000) el.removeChild(el.firstChild);
}

function initConsoleView(){
  document.getElementById('consoleClear').addEventListener('click', () => {
    document.getElementById('globalConsole').innerHTML = '<div class="console-empty">cleared</div>';
  });
}
