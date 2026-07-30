// views/docs.js

function renderDocs(){
  const grid = document.getElementById('docGrid');
  grid.innerHTML = DOCS.map(d => `
    <a class="doc-card" href="${RAW_DOCS_BASE}${d.href}" target="_blank" rel="noopener">
      <div class="dc-cat">${d.cat}</div>
      <div class="dc-title">${d.title}</div>
      <div class="dc-desc">${d.desc}</div>
    </a>
  `).join('') + `
    <a class="doc-card" href="${REPO}" target="_blank" rel="noopener">
      <div class="dc-cat">Source</div>
      <div class="dc-title">GitHub repository ↗</div>
      <div class="dc-desc">InputOutputStream/TensorF — issues, source, README.</div>
    </a>`;
}
