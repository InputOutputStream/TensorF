// docs-data.js — links out to TensorF's own docs site / GitHub. Kept as
// plain data so updating a link never touches render logic.

const REPO = 'https://github.com/InputOutputStream/TensorF';
const RAW_DOCS_BASE = 'https://github.com/InputOutputStream/TensorF/blob/main/';

const DOCS = [
  { cat:'Start',       title:'Overview',              desc:'What TensorF is and why it exists.',            href:'docs/index.html' },
  { cat:'Start',       title:'Install & Build',       desc:'Toolchain, deps, make targets.',                href:'docs/install.html' },
  { cat:'Start',       title:'Quickstart',            desc:'First tensor, first training loop.',            href:'docs/quickstart.html' },
  { cat:'Start',       title:'Examples',              desc:'GPT-2, SmolLM, transformer walkthroughs.',      href:'docs/examples.html' },
  { cat:'Core',        title:'Architecture',          desc:'Autograd design end to end.',                   href:'docs/architecture.html' },
  { cat:'Core',        title:'Tensor<T> / Matrix<T>', desc:'Core numeric types.',                           href:'docs/api/tensor.html' },
  { cat:'Core',        title:'Operations (19)',       desc:'The full op set.',                              href:'docs/api/operations.html' },
  { cat:'NN',          title:'Modules / nn',          desc:'GPT · Llama · LoRA.',                           href:'docs/api/modelloader.html' },
  { cat:'Data',        title:'DataLoader',            desc:'Batching & corpora.',                           href:'docs/api/dataloader.html' },
  { cat:'Data',        title:'Tokenizer',             desc:'Encode/decode.',                                href:'docs/api/tokenizer.html' },
  { cat:'Distributed', title:'Federated Learning',    desc:'FedAvg vs FedDistill, wire protocols.',         href:'docs/federated.html' },
  { cat:'Distributed', title:'Profiler',              desc:'Hardware fingerprint → hyperparameters.',       href:'docs/api/profiler.html' },
  { cat:'Web',         title:'Gateway API',           desc:'The REST/WS backend this dashboard talks to.',  href:'docs/api/gateway.html' },
  { cat:'Reference',   title:'Known Issues',          desc:'Current limitations.',                          href:'docs/known-issues.html' },
];
