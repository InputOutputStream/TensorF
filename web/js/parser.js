// parser.js — the gateway only ships raw stdout/stderr per job (no metrics
// API exists: see docs/api/gateway.html). TensorF's binaries already print
// structured, regex-friendly lines, so we parse those live as they arrive.
//
// Every pattern below was matched against the real printf/cout calls in:
//   include/net/Network/Client.cpp   (round boxes, loss, bandwidth)
//   include/net/Network/Server.cpp   (round header, aggregate/total ms)
//   include/tools/Profiler/Profiler.hpp (hardware fingerprint block)
//
// If the gateway ever gains a real metrics endpoint, only this file and
// ingestLogLine()'s caller need to change — render functions don't care
// where numbers came from.

const RX = {
  trainLoss:      /Training loss\s*:\s*([\d.]+)/,
  distillLoss:    /Distillation loss:\s*([\d.]+)/,
  clientRound:    /──\s*(FedAvg|FedDistill) Round (\d+)/,
  clientBoxRound: /CLIENT\s+Round\s+(\d+)/,
  serverRound:    /\[(\w+)\]\s*Round\s*(\d+)\s*\|\s*(\d+)\s*clients/,
  sendDeltas:     /Sent deltas in ([\d.]+) ms \(([\d.]+) MB, ([\d.]+) MB\/s\)/,
  recvWeights:    /Received weights in ([\d.]+) ms \(([\d.]+) MB, ([\d.]+) MB\/s\)/,
  sendDeltasBox:  /Send deltas\s*:\s*([\d.]+) ms\s*\(\s*([\d.]+) MB,\s*([\d.]+) MB\/s\)/,
  recvWeightsBox: /Receive weights\s*:\s*([\d.]+) ms\s*\(\s*([\d.]+) MB,\s*([\d.]+) MB\/s\)/,
  aggregateMs:    /Aggregate\s*:\s*([\d.]+) ms/,
  roundTotalMs:   /Round total\s*:\s*([\d.]+) ms/,
  tokensPerSec:   /Tokens\/sec\s*:\s*([\d.]+)/,
  rssMb:          /RSS\s*:\s*([\d.]+) MB/,
  cpuModel:       /\[TensorF Profiler\] CPU:\s*(.+)/,
  isa:            /\[TensorF Profiler\] ISA:\s*(\S+)/,
  ramLine:        /\[TensorF Profiler\] RAM:\s*(\d+) MB available, (\S+) ([\d.]+) MT\/s/,
  l3cache:        /\[TensorF Profiler\] L3 cache:\s*(\d+) MB/,
  matmulL3:       /Matmul \(L3\)\s*:\s*([\d.]+) GFLOP\/s/,
  matmulRam:      /Matmul \(RAM\)\s*:\s*([\d.]+) GFLOP\/s/,
  memBw:          /Mem BW \(read\)\s*:\s*([\d.]+) GB\/s/,
  l1lat:          /L1 latency\s*:\s*([\d.]+) ns/,
  l3lat:          /L3 latency\s*:\s*([\d.]+) ns/,
  ramlat:         /RAM latency\s*:\s*([\d.]+) ns/,
  storageRd:      /Storage rd\s*:\s*([\d.]+) MB\/s/,
  recBatch:       /Recommended batch_size\s*:\s*(\d+)/,
  recBlock:       /Recommended block_size\s*:\s*(\d+)/,
  recQuant:       /Recommended quant\s*:\s*(\S+)/,
  paramRam:       /Estimated param RAM\s*:\s*(\d+) MB/,
  physCores:      /Cores\s*:\s*(\d+) physical \/ (\d+) logical/,
  maxFreq:        /Freq\s*:\s*([\d.]+) MHz/,
  totalRam:       /Total\s*:\s*(\d+) MB/,
  availRam:       /Available\s*:\s*(\d+) MB/,
  bandwidthGbs:   /Theor BW\s*:\s*([\d.]+) GB\/s/,
  passLine:       /\[PASS\]/,
  failLine:       /\[FAIL\]|error|Error|ERROR/,
};

function emptyMetrics(){
  return { loss: [], rounds: [], profiler: null, bench: null, hyper: null };
}

function lastRound(m){
  if(m.rounds.length === 0) m.rounds.push({ side: '?', round: 0 });
  return m.rounds[m.rounds.length - 1];
}

function parseMetricLine(jobId, line){
  const m = state.metrics[jobId] || (state.metrics[jobId] = emptyMetrics());
  let mm;

  if((mm = RX.trainLoss.exec(line)))   m.loss.push({ idx: m.loss.length, kind:'train', val: parseFloat(mm[1]) });
  if((mm = RX.distillLoss.exec(line))) m.loss.push({ idx: m.loss.length, kind:'distill', val: parseFloat(mm[1]) });

  if((mm = RX.clientRound.exec(line))) m.rounds.push({ side:'client', mode: mm[1], round: parseInt(mm[2]), ts: Date.now() });
  if((mm = RX.serverRound.exec(line))) m.rounds.push({ side:'server', mode: mm[1], round: parseInt(mm[2]), clients: parseInt(mm[3]), ts: Date.now() });

  if((mm = RX.aggregateMs.exec(line)))  lastRound(m).aggregate_ms = parseFloat(mm[1]);
  if((mm = RX.roundTotalMs.exec(line))) lastRound(m).total_ms = parseFloat(mm[1]);

  if((mm = RX.sendDeltas.exec(line)) || (mm = RX.sendDeltasBox.exec(line))){
    lastRound(m).send_ms = parseFloat(mm[1]); lastRound(m).send_mb = parseFloat(mm[2]); lastRound(m).send_mbps = parseFloat(mm[3]);
  }
  if((mm = RX.recvWeights.exec(line)) || (mm = RX.recvWeightsBox.exec(line))){
    lastRound(m).recv_ms = parseFloat(mm[1]); lastRound(m).recv_mb = parseFloat(mm[2]); lastRound(m).recv_mbps = parseFloat(mm[3]);
  }
  if((mm = RX.rssMb.exec(line))) lastRound(m).rss_mb = parseFloat(mm[1]);

  // hardware fingerprint (Profiler::run / print_summary)
  if(!m.profiler) m.profiler = {};
  if((mm = RX.cpuModel.exec(line))) m.profiler.cpu_model = mm[1].trim();
  if((mm = RX.isa.exec(line)))      m.profiler.isa = mm[1];
  if((mm = RX.ramLine.exec(line))){ m.profiler.ram_avail_mb = parseInt(mm[1]); m.profiler.ram_type = mm[2]; m.profiler.ram_speed = parseFloat(mm[3]); }
  if((mm = RX.l3cache.exec(line)))  m.profiler.l3_mb = parseInt(mm[1]);
  if((mm = RX.physCores.exec(line))){ m.profiler.phys_cores = parseInt(mm[1]); m.profiler.log_cores = parseInt(mm[2]); }
  if((mm = RX.maxFreq.exec(line)))  m.profiler.max_freq_mhz = parseFloat(mm[1]);
  if((mm = RX.totalRam.exec(line))) m.profiler.ram_total_mb = parseInt(mm[1]);
  if((mm = RX.availRam.exec(line))) m.profiler.ram_avail_mb2 = parseInt(mm[1]);
  if((mm = RX.bandwidthGbs.exec(line))) m.profiler.ram_bw_gbs = parseFloat(mm[1]);

  // benchmark probes
  if(!m.bench) m.bench = {};
  if((mm = RX.matmulL3.exec(line)))  m.bench.matmul_l3 = parseFloat(mm[1]);
  if((mm = RX.matmulRam.exec(line))) m.bench.matmul_ram = parseFloat(mm[1]);
  if((mm = RX.memBw.exec(line)))     m.bench.mem_read_gbs = parseFloat(mm[1]);
  if((mm = RX.l1lat.exec(line)))     m.bench.l1_ns = parseFloat(mm[1]);
  if((mm = RX.l3lat.exec(line)))     m.bench.l3_ns = parseFloat(mm[1]);
  if((mm = RX.ramlat.exec(line)))    m.bench.ram_ns = parseFloat(mm[1]);
  if((mm = RX.storageRd.exec(line))) m.bench.storage_mbs = parseFloat(mm[1]);

  // derived hyperparameters (HyperparamAdvisor)
  if(!m.hyper) m.hyper = {};
  if((mm = RX.recBatch.exec(line))) m.hyper.batch_size = parseInt(mm[1]);
  if((mm = RX.recBlock.exec(line))) m.hyper.block_size = parseInt(mm[1]);
  if((mm = RX.recQuant.exec(line))) m.hyper.quant = mm[1];
  if((mm = RX.paramRam.exec(line))) m.hyper.param_ram_mb = parseInt(mm[1]);
}

function ingestLogLine(jobId, line, fromBulk = false){
  if(!state.jobLogs[jobId]) state.jobLogs[jobId] = [];
  if(!fromBulk || !state.jobLogs[jobId].includes(line)){
    state.jobLogs[jobId].push(line);
    if(state.jobLogs[jobId].length > 5000) state.jobLogs[jobId].shift();
  }
  parseMetricLine(jobId, line);
}

function classifyLogLine(line){
  if(RX.trainLoss.test(line) || RX.distillLoss.test(line)) return 'hl-loss';
  if(RX.clientRound.test(line) || RX.serverRound.test(line) || RX.clientBoxRound.test(line)) return 'hl-round';
  if(RX.failLine.test(line)) return 'hl-err';
  if(RX.passLine.test(line)) return 'hl-ok';
  return '';
}

function renderLogLine(line){
  return `<div class="ln ${classifyLogLine(line)}">${escapeHtml(line)}</div>`;
}

function parseBenchmark(log) {
  if (Array.isArray(log)) log = log.join('\n');

  const results = {};
  const advisor = {};
  const memory = {};

  // Match [Results] block
  const resultsMatch = log.match(/\[Results\]([\s\S]*?)(?=\n╔|$)/);
  if (resultsMatch) {
    const lines = resultsMatch[1].split('\n');
    for (const line of lines) {
      const m = line.match(/\s*([\w\s]+)\s*:\s*([\d.]+)\s*([\w\/]+)?/);
      if (m) {
        const key = m[1].trim().toLowerCase().replace(/ /g, '_');
        const val = parseFloat(m[2]);
        if (!isNaN(val)) results[key] = val;
      }
    }
  }

  // Hyperparameter Advisor box
  const advisorMatch = log.match(/HYPERPARAMETER ADVISOR REPORT([\s\S]*?)(?=\n╚|$)/);
  if (advisorMatch) {
    const text = advisorMatch[1];
    const lines = text.split('\n');
    for (const line of lines) {
      const m = line.match(/\s*([\w\s]+)\s*:\s*([\d.]+)/);
      if (m) {
        const key = m[1].trim().toLowerCase().replace(/ /g, '_');
        const val = parseFloat(m[2]);
        if (!isNaN(val)) advisor[key] = val;
      }
      // Extract other fields like n_embed, n_heads, etc.
      const dims = line.match(/n_embed\s*:\s*(\d+)/);
      if (dims) advisor.n_embed = parseInt(dims[1]);
      const heads = line.match(/n_heads\s*:\s*(\d+)/);
      if (heads) advisor.n_heads = parseInt(heads[1]);
      const layers = line.match(/n_layers\s*:\s*(\d+)/);
      if (layers) advisor.n_layers = parseInt(layers[1]);
      const block = line.match(/block_size\s*:\s*(\d+)/);
      if (block) advisor.block_size = parseInt(block[1]);
      const batch = line.match(/batch_size\s*:\s*(\d+)/);
      if (batch) advisor.batch_size = parseInt(batch[1]);
      const quant = line.match(/quantization\s*:\s*(\w+)/);
      if (quant) advisor.quant = quant[1];
      const threads = line.match(/num_threads\s*:\s*(\d+)/);
      if (threads) advisor.threads = parseInt(threads[1]);
      const params = line.match(/Params only\s*:\s*([\d.]+)\s*MB/);
      if (params) advisor.params_mb = parseFloat(params[1]);
      const train = line.match(/Training peak\s*:\s*([\d.]+)\s*MB/);
      if (train) advisor.train_peak_mb = parseFloat(train[1]);
      const infer = line.match(/Inference peak\s*:\s*([\d.]+)\s*MB/);
      if (infer) advisor.infer_peak_mb = parseFloat(infer[1]);
    }
    // Scores
    const hwScore = text.match(/Hardware score\s*:\s*([\d.]+)/);
    if (hwScore) advisor.hw_score = parseFloat(hwScore[1]);
    const fitScore = text.match(/Config fit score\s*:\s*([\d.]+)/);
    if (fitScore) advisor.fit_score = parseFloat(fitScore[1]);
  }

  // Memory Profile box
  const memMatch = log.match(/MEMORY PROFILE REPORT([\s\S]*?)(?=\n╚|$)/);
  if (memMatch) {
    const text = memMatch[1];
    const lines = text.split('\n');
    for (const line of lines) {
      const m = line.match(/(BASELINE|LOADED|INFER)\s+RSS:\s*([\d.]+)\s*MB\s+Heap:\s*([\d.]+)\s*MB/);
      if (m) {
        const stage = m[1].toLowerCase();
        memory[stage + '_rss'] = parseFloat(m[2]);
        memory[stage + '_heap'] = parseFloat(m[3]);
      }
      const peak = line.match(/Peak RSS\s*:\s*([\d.]+)\s*MB/);
      if (peak) memory.peak_rss = parseFloat(peak[1]);
      const rec = line.match(/Recommended free\s*:\s*([\d.]+)\s*MB/);
      if (rec) memory.recommended_free = parseFloat(rec[1]);
    }
  }

  return { results, advisor, memory };
}

function parseTests(log) {
  if (Array.isArray(log)) log = log.join('\n');
  const sections = {};
  const losses = [];

  // Split by "=== Section ==="
  const sectionRegex = /=== (.*?) ===([\s\S]*?)(?=(?:===|$))/g;
  let match;
  while ((match = sectionRegex.exec(log)) !== null) {
    const sectionName = match[1].trim();
    const content = match[2];
    const results = [];
    const lines = content.split('\n');
    for (const line of lines) {
      if (line.includes('[PASS]')) results.push('PASS');
      else if (line.includes('[FAIL]')) results.push('FAIL');
    }
    sections[sectionName] = results;
  }

  // Loss lines: "iter: X loss: [Y]"
  const lossRegex = /iter:\s*(\d+)\s+loss:\s*\[([\d.]+)\]/g;
  let lossMatch;
  while ((lossMatch = lossRegex.exec(log)) !== null) {
    losses.push(parseFloat(lossMatch[2]));
  }

  return { sections, losses };
}