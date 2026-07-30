// views/launch.js — wires each launch card button to POST /api/jobs/{kind}

function initLaunchView(){
  document.querySelectorAll('[data-launch]').forEach(btn => {
    btn.addEventListener('click', async () => {
      const kind = btn.dataset.launch;
      btn.disabled = true;
      const orig = btn.innerHTML;
      btn.innerHTML = '<span class="spinner"></span> launching…';
      try{
        let body = {};
        if(kind === 'server') body = { port: parseInt(document.getElementById('srvPort').value) || 9000 };
        if(kind === 'client') body = {
          host: document.getElementById('cliHost').value || '127.0.0.1',
          port: parseInt(document.getElementById('cliPort').value) || 9000,
        };
        const job = await launchJob(kind, body);
        toast(`Launched ${kind} → ${job.id.slice(0,10)}`, 'ok');
        document.getElementById('launchResult').innerHTML = `
          <div class="card"><div class="card-title">Launched</div>
            <div class="pill-row">
              <span class="pill">kind <b>${job.kind}</b></span>
              <span class="pill">id <b>${job.id}</b></span>
              <span class="pill">pid <b>${job.pid}</b></span>
            </div>
            <div style="margin-top:10px;"><button class="btn sm" onclick="goToJob('${job.id}')">View job →</button></div>
          </div>`;
      }catch(e){
        toast('Launch failed: ' + e.message, 'err');
      }finally{
        btn.disabled = false;
        btn.innerHTML = orig;
      }
    });
  });
}
