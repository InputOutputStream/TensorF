#!/usr/bin/env bash
# Kept for compatibility: see scripts/download_datasets.sh (cora, ogbn-arxiv, ogbn-products, ...)
exec "$(dirname "${BASH_SOURCE[0]}")/download_datasets.sh" cora
