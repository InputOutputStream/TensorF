// js/param-schema.js
const PRESETS = {
  'server:fedavg': {
    kind: 'server',
    args: ['--fed', 'fedavg', '--port', '8080', '--rounds', '100']
  },
  'server:fedprox': {
    kind: 'server',
    args: ['--fed', 'fedprox', '--port', '8080', '--rounds', '100', '--mu', '0.1']
  },
  'server:fedasync': {
    kind: 'server',
    args: ['--fed', 'fedasync', '--port', '8080', '--rounds', '50']
  },
  'client:default': {
    kind: 'client',
    args: ['--server', '127.0.0.1', '--port', '8081', '--batch', '32', '--lr', '0.001']
  },
  'client:gpu': {
    kind: 'client',
    args: ['--server', '127.0.0.1', '--port', '8081', '--batch', '64', '--device', 'cuda']
  },
  'benchmark:quick': {
    kind: 'benchmark',
    args: ['--quick']
  },
  'benchmark:full': {
    kind: 'benchmark',
    args: ['--full']
  }
};

const FLAG_HELP = {
  server: [
    { flag: '--port', type: 'number', default: 8080, help: 'Server port' },
    { flag: '--fed', type: 'select', options: ['fedavg', 'fedprox', 'fedasync'], default: 'fedavg', help: 'Federated algorithm' },
    { flag: '--rounds', type: 'number', default: 100, help: 'Number of rounds' },
    { flag: '--mu', type: 'number', default: 0.01, help: 'FedProx proximal term' },
    { flag: '--min_clients', type: 'number', default: 2, help: 'Minimum clients per round' },
    { flag: '--max_clients', type: 'number', default: 10, help: 'Maximum clients per round' }
  ],
  client: [
    { flag: '--server', type: 'text', default: '127.0.0.1', help: 'Server IP' },
    { flag: '--port', type: 'number', default: 8081, help: 'Client port' },
    { flag: '--batch', type: 'number', default: 32, help: 'Batch size' },
    { flag: '--lr', type: 'number', default: 0.001, help: 'Learning rate' },
    { flag: '--epochs', type: 'number', default: 5, help: 'Local epochs' },
    { flag: '--device', type: 'select', options: ['cpu', 'cuda'], default: 'cpu', help: 'Device' },
    { flag: '--client_id', type: 'text', default: '', help: 'Optional client identifier' }
  ],
  benchmark: [
    { flag: '--quick', type: 'boolean', help: 'Run quick benchmarks' },
    { flag: '--full', type: 'boolean', help: 'Run full benchmarks' },
    { flag: '--iterations', type: 'number', default: 10, help: 'Number of iterations' }
  ],
  tests: [
    { flag: '--verbose', type: 'boolean', help: 'Verbose output' }
  ],
  gpt2: [],
  smollm: []
};

window.PRESETS = PRESETS;
window.FLAG_HELP = FLAG_HELP;