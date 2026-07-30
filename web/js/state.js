// state.js — single shared mutable store. No framework: every module
// reads/writes this object directly and calls the relevant render*() fn.

const state = {
  ws: null,
  jobs: {},         // id -> job object (from REST/WS)
  jobLogs: {},      // id -> array of raw log lines
  metrics: {},      // id -> {loss:[], rounds:[], profiler:{}, bench:{}, hyper:{}}
  selectedJobId: null,
  pollTimer: null,
};

function gwBase(){
  return document.getElementById('gwUrl').value.replace(/\/$/, '');
}
