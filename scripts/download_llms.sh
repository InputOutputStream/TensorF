#!/usr/bin/env bash

BASE="https://huggingface.co/ddh0"

cd SLM

# ── GPT-2 Small (124 M paramètres) ──────────────────────────
# Précision complète F32 — 654 MB — mapping 1:1, aucune perte de précision
wget -c "${BASE}/GPT-2-GGUF/resolve/main/GPT-2-f32.gguf"         -O gpt2-small-f32.gguf

# Quantifié Q8_0 — 178 MB — très proche F32, recommandé pour du fine-tuning
wget -c "${BASE}/GPT-2-GGUF/resolve/main/GPT-2-q8_0.gguf"        -O gpt2-small-q8.gguf

# Quantifié Q4_K_S — 135 MB — pour inférence uniquement
wget -c "${BASE}/GPT-2-GGUF/resolve/main/GPT-2-q4_k_s.gguf"      -O gpt2-small-q4.gguf

# # ── GPT-2 Medium (345 M paramètres) ─────────────────────────
# # 24 layers, 16 heads, d_model=1024
# # Q4_K_S — 367 MB
wget -c "${BASE}/GPT-2-medium-GGUF/resolve/main/GPT-2-medium-q4_k_s.gguf" \
#      -O gpt2-medium-q4.gguf

# # Q8_0 — 484 MB
wget -c "${BASE}/GPT-2-medium-GGUF/resolve/main/GPT-2-medium-q8_0.gguf" \
#      -O gpt2-medium-q8.gguf

# # ── GPT-2 Large (774 M paramètres) ──────────────────────────
# # 36 layers, 20 heads, d_model=1280
# # Q4_K_S — ~824 MB
# wget -c "${BASE}/GPT-2-large-GGUF/resolve/main/GPT-2-large-q4_k_s.gguf" \
#      -O gpt2-large-q4.gguf

# # ── GPT-2 XL (1.5 B paramètres) ─────────────────────────────
# # 48 layers, 25 heads, d_model=1600
# # Q4_K_S — ~1.6 GB
# wget -c "${BASE}/GPT-2-xl-GGUF/resolve/main/GPT-2-xl-q4_k_s.gguf" \
#      -O gpt2-xl-q4.gguf

echo ""
echo "Telechargements termines. Fichiers dans $(pwd) :"
ls -lh *.gguf 2>/dev/null