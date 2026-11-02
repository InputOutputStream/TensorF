#!/usr/bin/env bash
# Download the GCN benchmark datasets into Datasets/<name>/ so coratrain can read them.
#
#   cora      LINQS files from the pygcn repository           -> cora.content / cora.cites
#   citeseer  Planetoid files from tkipf/gcn                  -> citeseer.content / citeseer.cites
#   pubmed    Planetoid files from tkipf/gcn                  -> pubmed.content / pubmed.cites
#   nell      Planetoid NELL archive (cs.cmu.edu)             -> unpacked only, see note below
#
# Citeseer and Pubmed are only published as Python pickles; they are converted to the LINQS
# text format by include/tools/planetoid2linqs.hpp, through the small driver
# tools/planetoid2linqs.cpp (built here with $CXX, default g++; no Python).
#
# Usage: scripts/download_datasets.sh [cora] [citeseer] [pubmed] [nell]     (default: all four)
#
# NELL note: its feature matrix is 65,755 x 61,278 (5,414 entity features plus one identity
# column per relation node).  That is far too large as dense LINQS text, so the archive is
# only downloaded and unpacked; training on it needs a sparse feature input.

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA="$ROOT/Datasets"
BIN="$ROOT/bin"
GCN_BASE="https://raw.githubusercontent.com/tkipf/gcn/master/gcn/data"
PYGCN_BASE="https://raw.githubusercontent.com/tkipf/pygcn/master/data/cora"
NELL_URL="http://www.cs.cmu.edu/~zhiliny/data/nell_data.tar.gz"

fetch() { # url dest
  [ -s "$2" ] && return 0
  curl -fL --retry 3 -o "$2.part" "$1" && mv "$2.part" "$2"
}

converter() {
  if [ ! -x "$BIN/planetoid2linqs" ] || [ "$ROOT/include/tools/planetoid2linqs.cpp" -nt "$BIN/planetoid2linqs" ] \
     || [ "$ROOT/include/tools/planetoid2linqs.hpp" -nt "$BIN/planetoid2linqs" ]; then
    mkdir -p "$BIN"
    "${CXX:-g++}" -std=c++17 -O2 -I"$ROOT/include/tools" "$ROOT/include/tools/planetoid2linqs.cpp" -o "$BIN/planetoid2linqs"
  fi
}

do_cora() {
  mkdir -p "$DATA/cora"
  for f in cora.content cora.cites; do fetch "$PYGCN_BASE/$f" "$DATA/cora/$f"; done
  echo "cora: content $(wc -l < "$DATA/cora/cora.content") lines (expect 2708), cites $(wc -l < "$DATA/cora/cora.cites") lines (expect 5429)"
}

do_planetoid() { # name
  local n="$1" raw="$DATA/$1/raw"
  mkdir -p "$raw"
  for p in allx ally tx ty graph test.index; do fetch "$GCN_BASE/ind.$n.$p" "$raw/ind.$n.$p"; done
  converter
  "$BIN/planetoid2linqs" "$raw" "$n" "$DATA/$n"
}

do_nell() {
  local raw="$DATA/nell/raw"
  mkdir -p "$raw"
  fetch "$NELL_URL" "$raw/nell_data.tar.gz"
  tar -xzf "$raw/nell_data.tar.gz" -C "$raw"
  echo "nell: unpacked into $raw (not converted: needs a sparse feature input, see header)"
}

targets=("$@"); [ ${#targets[@]} -eq 0 ] && targets=(cora citeseer pubmed nell)
for t in "${targets[@]}"; do
  case "$t" in
    cora) do_cora ;;
    citeseer|pubmed) do_planetoid "$t" ;;
    nell) do_nell ;;
    *) echo "unknown dataset: $t (expected cora, citeseer, pubmed or nell)" >&2; exit 1 ;;
  esac
done
