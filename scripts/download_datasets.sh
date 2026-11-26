#!/usr/bin/env bash
# Download graph datasets into Datasets/<name>/ (idempotent, resumable).
#   ./scripts/download_datasets.sh list
#   ./scripts/download_datasets.sh cora ogbn-arxiv ogbn-products
#   ./scripts/download_datasets.sh --from /path/to/arxiv.zip ogbn-arxiv      # zip you already have (no network)
#
# cora          LINQS format, from the pygcn repository (2708 nodes).
# ogbn-arxiv    169,343 nodes / 1.17M edges, 128 features, 40 classes, official time split.
# ogbn-products 2,449,029 nodes / 61.9M edges, 100 features, 47 classes, official sales-ranking split.
#               (needs several GB of RAM to load; start with arxiv)
# OGB archives are converted once to the compact Datasets/<name>/graph.tfg (see tools/include/ogb2tfg.cpp).
# Not supported: ogbn-proteins (multi-label), ogbn-mag (heterogeneous; RGCN later), ogbn-papers100M (out-of-core).
#
# Env: OGB_BASE   mirror/base URL   (default https://snap.stanford.edu/ogb/data/nodeproppred)
#      KEEP_RAW=1 keep the extracted CSVs   DATASETS_DIR target directory (default ./Datasets)   OGB_TMP scratch (default Datasets/.tmp)
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT"
OGB_BASE="${OGB_BASE:-https://snap.stanford.edu/ogb/data/nodeproppred}"
DS="${DATASETS_DIR:-$ROOT/Datasets}"; TMP="${OGB_TMP:-$DS/.tmp}"; FROM=""

usage() { sed -n '2,15p' "$0" | sed 's/^# \{0,1\}//'; }
need() { command -v "$1" >/dev/null 2>&1 || { echo "error: '$1' is required ($2)" >&2; exit 1; }; }

get_cora() {
  local dest="$DS/cora" base="https://raw.githubusercontent.com/tkipf/pygcn/master/data/cora"
  mkdir -p "$dest"
  for f in cora.content cora.cites; do
    [ -s "$dest/$f" ] && { echo "have $f"; continue; }
    curl -fL --retry 3 -o "$dest/$f" "$base/$f"
  done
  echo "cora: content lines $(wc -l < "$dest/cora.content") (expect 2708), cites lines $(wc -l < "$dest/cora.cites") (expect 5429)"
}

build_converter() {
  [ -x "$ROOT/bin/ogb2tfg" ] && [ "$ROOT/bin/ogb2tfg" -nt "$ROOT/tools/ogb2tfg.cpp" ] && return
  need g++ "to build the converter"; mkdir -p "$ROOT/bin"
  g++ -std=c++17 -O2 -I"$ROOT/include" "$ROOT/include/tools/ogb2tfg.cpp" -o "$ROOT/bin/ogb2tfg"
}

get_ogb() {
  local name="$1" short="${1#ogbn-}" dest="$DS/$1"
  case "$short" in
    arxiv|products) ;;
    proteins|mag|papers100M) echo "error: $name is not supported (multi-label / heterogeneous / out-of-core)." >&2; return 1 ;;
    *) echo "error: unknown OGB dataset '$name'" >&2; return 1 ;;
  esac
  if [ -s "$dest/graph.tfg" ]; then echo "have $name/graph.tfg"; return 0; fi
  need gzip "to read the OGB CSVs"; need unzip "or use python3 -m zipfile"
  mkdir -p "$TMP" "$dest"
  local zip="$TMP/$short.zip"
  if [ -n "$FROM" ]; then zip="$FROM"; else
    echo "downloading $OGB_BASE/$short.zip (resumable)…"
    curl -fL --retry 3 -C - -o "$zip" "$OGB_BASE/$short.zip"
  fi
  echo "extracting…"; rm -rf "$TMP/x_$short"; mkdir -p "$TMP/x_$short"
  unzip -q -o "$zip" -d "$TMP/x_$short"
  local raw; raw="$(find "$TMP/x_$short" -maxdepth 3 -type d -name raw | head -1)"
  [ -n "$raw" ] || { echo "error: no raw/ directory in the archive" >&2; return 1; }
  build_converter
  "$ROOT/bin/ogb2tfg" --dir "$(dirname "$raw")" --out "$dest/graph.tfg.part"
  mv "$dest/graph.tfg.part" "$dest/graph.tfg"
  if [ "${KEEP_RAW:-0}" != 1 ]; then rm -rf "$TMP/x_$short"; [ -z "$FROM" ] && rm -f "$zip"; fi
  echo "$name ready: $dest/graph.tfg  (use --opt data=$dest)"
}

[ $# -eq 0 ] && { usage; exit 0; }
names=()
while [ $# -gt 0 ]; do
  case "$1" in
    list|-h|--help) usage; exit 0 ;;
    --from) FROM="$2"; shift 2 ;;
    *) names+=("$1"); shift ;;
  esac
done
need curl "to download"
for n in "${names[@]}"; do
  case "$n" in
    cora) get_cora ;;
    ogbn-*) get_ogb "$n" ;;
    *) echo "error: unknown dataset '$n' (try: list)" >&2; exit 1 ;;
  esac
done
