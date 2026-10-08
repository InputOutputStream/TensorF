// views/launch.js — dynamic job launch cards
let cardCounter = 0;

function addJobCard(kind) {
  const container = document.getElementById('jobCards');
  const cardId = `jobCard-${cardCounter++}`;
  const flags = FLAG_HELP[kind] || [];

  let flagInputs = flags.map(f => {
    let input;
    if (f.type === 'select') {
      const options = f.options.map(o => `<option value="${o}" ${o===f.default?'selected':''}>${o}</option>`).join('');
      input = `<select data-flag="${f.flag}" class="param-input">${options}</select>`;
    } else if (f.type === 'boolean') {
      input = `<input type="checkbox" data-flag="${f.flag}" class="param-input" />`;
    } else {
      input = `<input type="${f.type}" data-flag="${f.flag}" value="${f.default || ''}" class="param-input" />`;
    }
    return `<div class="flag-row"><label>${f.flag}</label>${input}<span class="help">${f.help}</span></div>`;
  }).join('');

  const card = document.createElement('div');
  card.className = 'job-card';
  card.id = cardId;
  card.innerHTML = `
  <div class="card-header">
    <span class="job-kind">${kind}</span>
    <button class="remove-card" data-card="${cardId}">✕</button>
  </div>
  <div class="card-body">
    <div class="flag-grid">${flagInputs}</div>
    <button class="launch-btn primary" data-kind="${kind}" data-card="${cardId}">Launch</button>
  </div>
`;
  container.appendChild(card);

  card.querySelector('.launch-btn').addEventListener('click', () => {
    const inputs = card.querySelectorAll('.param-input');
    const args = [];
    inputs.forEach(inp => {
      const flag = inp.dataset.flag;
      if (flag) {
        if (inp.type === 'checkbox') {
          if (inp.checked) args.push(flag);
        } else {
          const val = inp.value.trim();
          if (val !== '') {
            args.push(flag, val);
          }
        }
      }
    });
    const kind = card.querySelector('.launch-btn').dataset.kind;
    launchJobRaw(kind, args);
  });

  card.querySelector('.remove-card').addEventListener('click', () => {
    card.remove();
  });
}

function initLaunchView() {
  const container = document.getElementById('launchControls');
  container.innerHTML = `
    <div class="preset-bar">
      <select id="presetSelect">
        <option value="">— Select a preset —</option>
        ${Object.keys(PRESETS).map(p => `<option value="${p}">${p}</option>`).join('')}
      </select>
      <button id="applyPreset">Apply</button>
    </div>
    <div id="jobCards"></div>
    <div class="add-buttons">
      <button id="addServerBtn">+ Add Server</button>
      <button id="addClientBtn">+ Add Client</button>
      <button id="addBenchmarkBtn">+ Add Benchmark</button>
      <button id="addTestsBtn">+ Add Tests</button>
      <button id="addGpt2Btn">+ Add GPT-2</button>
      <button id="addSmollmBtn">+ Add SmolLM</button>
    </div>
  `;

  document.getElementById('applyPreset').addEventListener('click', () => {
    const sel = document.getElementById('presetSelect');
    const key = sel.value;
    if (key && PRESETS[key]) {
      const preset = PRESETS[key];
      addJobCard(preset.kind);
    }
  });

  document.getElementById('addServerBtn').addEventListener('click', () => addJobCard('server'));
  document.getElementById('addClientBtn').addEventListener('click', () => addJobCard('client'));
  document.getElementById('addBenchmarkBtn').addEventListener('click', () => addJobCard('benchmark'));
  document.getElementById('addTestsBtn').addEventListener('click', () => addJobCard('tests'));
  document.getElementById('addGpt2Btn').addEventListener('click', () => addJobCard('gpt2'));
  document.getElementById('addSmollmBtn').addEventListener('click', () => addJobCard('smollm'));

  addJobCard('server');
}