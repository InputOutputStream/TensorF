#!/usr/bin/env bash
# Download the Cora citation dataset (LINQS format) into Datasets/cora/.
# Source: the copy shipped with the pygcn repository (2708 papers, 5429 citations, 1433 word features, 7 classes).

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$ROOT/Datasets/cora"
BASE="https://raw.githubusercontent.com/tkipf/pygcn/master/data/cora"
mkdir -p "$DEST"
for f in cora.content cora.cites; do
  [ -s "$DEST/$f" ] && { echo "have $f"; continue; }
  curl -fL --retry 3 -o "$DEST/$f" "$BASE/$f"
done
echo "content lines: $(wc -l < "$DEST/cora.content") (expect 2708)   cites lines: $(wc -l < "$DEST/cora.cites") (expect 5429)"