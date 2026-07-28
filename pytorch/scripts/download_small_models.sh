#!/usr/bin/env bash
# Download Mac-friendly small HF chat models + multimodal encoders
# (config + tokenizer + weights) for vLLM Omni multimodal stress.
#
# Text chat backbones:
#   - Qwen/Qwen2.5-0.5B-Instruct          (~1.0 GB)  native qwen2
#   - deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B (~3.6 GB)  qwen2 arch
#   - unsloth/Llama-3.2-1B-Instruct       (~2.5 GB)  native llama
#   - openai-community/gpt2               (~0.55 GB) native gpt2
#   - zai-org/glm-edge-1.5b-chat          (~3.2 GB)  glm-edge
#
# Multimodal encoders (MediaEncoderRegistry):
#   - facebook/dinov2-small               vision
#   - openai/clip-vit-base-patch32        vision-text
#   - HuggingFaceTB/SmolVLM-256M-Instruct small VLM tower
#   - Qwen/Qwen3-VL-2B-Instruct-FP8      vision tower only (~814MB via extract script)
#   - google/siglip-base-patch16-224     DeepSeek-VL vision stand-in (~800MB)
#   - openai/whisper-tiny                 audio / ASR
#   - sentence-transformers/all-MiniLM-L6-v2  embedding
#
# Usage:
#   export HF_TOKEN=hf_xxx          # optional for public models, required for gated
#   ./scripts/download_small_models.sh
#   ./scripts/download_small_models.sh --only llama
#   ./scripts/download_small_models.sh --only smolvlm
#   ./scripts/download_small_models.sh --only qwen3vl
#   ./scripts/download_small_models.sh --only siglip
#   HF_ENDPOINT=https://hf-mirror.com ./scripts/download_small_models.sh
#
# Notes:
#   Uses Range-request chunked curl via hf-mirror by default — full-file
#   downloads to cas-bridge.xethub often hang on some networks; 64MB chunks work.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODELS_DIR="${MODELS_DIR:-$ROOT/models}"
MIRROR="${HF_ENDPOINT:-https://hf-mirror.com}"
TOKEN="${HF_TOKEN:-}"
CHUNK=$((64 * 1024 * 1024))
ONLY=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --only) ONLY="$2"; shift 2 ;;
    --models-dir) MODELS_DIR="$2"; shift 2 ;;
    --mirror) MIRROR="$2"; shift 2 ;;
    --chunk-mb) CHUNK=$(( $2 * 1024 * 1024 )); shift 2 ;;
    -h|--help)
      sed -n '1,30p' "$0"; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; exit 2 ;;
  esac
done

mkdir -p "$MODELS_DIR"
AUTH=()
if [[ -n "$TOKEN" ]]; then
  AUTH=(-H "Authorization: Bearer $TOKEN")
fi

log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }

download_small() {
  local repo="$1" dest_dir="$2" file="$3"
  local dest="$dest_dir/$file"
  mkdir -p "$dest_dir"
  if [[ -f "$dest" && -s "$dest" ]]; then
    return 0
  fi
  log "GET $repo/$file"
  if curl -fsSL --http1.1 --retry 5 --retry-delay 2 \
      "${AUTH[@]}" \
      -o "$dest.partial" \
      "$MIRROR/$repo/resolve/main/$file"; then
    mv "$dest.partial" "$dest"
  else
    rm -f "$dest.partial"
    return 1
  fi
}

probe_size() {
  local repo="$1"
  local cr
  cr=$(curl -sI --http1.1 "${AUTH[@]}" -H "Range: bytes=0-0" \
    "$MIRROR/$repo/resolve/main/model.safetensors" \
    | tr -d '\r' | awk -F/ 'tolower($0) ~ /content-range/ {print $2; exit}')
  echo "${cr:-0}"
}

download_weight_chunked() {
  local repo="$1" dest_dir="$2" total="$3"
  local dest="$dest_dir/model.safetensors"
  local partial="$dest.partial"
  mkdir -p "$dest_dir"

  if [[ -f "$dest" ]]; then
    local sz
    sz=$(stat -f%z "$dest" 2>/dev/null || stat -c%s "$dest")
    if (( sz > 100000000 )); then
      log "SKIP weight $repo ($sz bytes)"
      return 0
    fi
  fi

  if (( total <= 0 )); then
    total=$(probe_size "$repo")
  fi
  if (( total <= 0 )); then
    log "FAIL cannot probe size for $repo"
    return 1
  fi

  local start=0
  if [[ -f "$partial" ]]; then
    start=$(stat -f%z "$partial" 2>/dev/null || stat -c%s "$partial")
  fi
  log "WEIGHT $repo resume=$start total=$total"

  local pos=$start
  while (( pos < total )); do
    local end=$(( pos + CHUNK - 1 ))
    if (( end >= total )); then end=$(( total - 1 )); fi
    local tmp="$partial.chunk"
    log "  chunk $pos-$end"
    if curl -fL --http1.1 --retry 8 --retry-delay 2 \
        "${AUTH[@]}" \
        -H "Range: bytes=$pos-$end" \
        -o "$tmp" \
        "$MIRROR/$repo/resolve/main/model.safetensors" \
        -w "    http=%{http_code} got=%{size_download} speed=%{speed_download}\n"; then
      cat "$tmp" >> "$partial"
      rm -f "$tmp"
      pos=$(stat -f%z "$partial" 2>/dev/null || stat -c%s "$partial")
      python3 -c "print(f'    progress {100.0*$pos/$total:.1f}% ({$pos}/{$total})')" 2>/dev/null \
        || log "    progress $pos / $total"
    else
      rm -f "$tmp"
      log "FAIL chunk $pos-$end for $repo"
      return 1
    fi
  done

  mv "$partial" "$dest"
  ls -lh "$dest"
  log "OK $repo"
}

fetch_model() {
  local name="$1" repo="$2" dir_name="$3" expected="$4"
  if [[ -n "$ONLY" && "$name" != *"$ONLY"* && "$repo" != *"$ONLY"* ]]; then
    log "skip $name (--only=$ONLY)"
    return 0
  fi
  local dest="$MODELS_DIR/$dir_name"
  log "======== $name ($repo) ========"
  for f in config.json generation_config.json tokenizer.json tokenizer_config.json \
           special_tokens_map.json chat_template.jinja merges.txt vocab.json \
           tokenizer.model README.md LICENSE; do
    download_small "$repo" "$dest" "$f" 2>/dev/null || true
  done
  download_weight_chunked "$repo" "$dest" "$expected"
}

# expected sizes (0 = probe)
# --- text chat backbones (OmniLLM) ---
fetch_model "qwen2.5-0.5b" "Qwen/Qwen2.5-0.5B-Instruct" "Qwen__Qwen2.5-0.5B-Instruct" 988097824
fetch_model "gpt2" "openai-community/gpt2" "openai-community__gpt2" 548105171
fetch_model "llama3.2-1b" "unsloth/Llama-3.2-1B-Instruct" "unsloth__Llama-3.2-1B-Instruct" 2471645608
fetch_model "deepseek-r1-1.5b" "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B" "deepseek-ai__DeepSeek-R1-Distill-Qwen-1.5B" 0
fetch_model "glm-edge-1.5b" "zai-org/glm-edge-1.5b-chat" "zai-org__glm-edge-1.5b-chat" 0

# --- multimodal encoders (Mac-friendly; used by MediaEncoderRegistry) ---
# vision: DINOv2 / CLIP / SmolVLM ; audio: Whisper-tiny (ASR)
fetch_model "dinov2-small" "facebook/dinov2-small" "facebook__dinov2-small" 0
fetch_model "clip-vit-b32" "openai/clip-vit-base-patch32" "openai__clip-vit-base-patch32" 0
fetch_model "smolvlm-256m" "HuggingFaceTB/SmolVLM-256M-Instruct" "HuggingFaceTB__SmolVLM-256M-Instruct" 0
fetch_model "whisper-tiny" "openai/whisper-tiny" "openai__whisper-tiny" 0
fetch_model "minilm-l6" "sentence-transformers/all-MiniLM-L6-v2" "sentence-transformers__all-MiniLM-L6-v2" 0

# DeepSeek-VL vision stand-in (SigLIP-base ~775MB) — full deepseek-ai/deepseek-vl-1.3b-chat is ~4GB.
# Known size required: xet/cas-bridge redirects break Range Content-Range probe.
fetch_model "siglip-base" "google/siglip-base-patch16-224" "google__siglip-base-patch16-224" 812672320

# Qwen3-VL vision tower only (~814MB BF16 extracted from 3.5GB FP8 checkpoint)
if [[ -z "$ONLY" || "$ONLY" == *qwen3* || "$ONLY" == *qwen-vl* || "$ONLY" == *vl* ]]; then
  log "======== qwen3-vl vision extract ========"
  bash "$ROOT/scripts/extract_qwen3vl_vision.sh" --dest "$MODELS_DIR/Qwen__Qwen3-VL-2B-Instruct-FP8"
fi

log "======== SUMMARY ========"
du -sh "$MODELS_DIR"/* 2>/dev/null || true
find "$MODELS_DIR" -name 'model.safetensors' -o -name 'vision_weights.safetensors' 2>/dev/null | while read -r f; do ls -lh "$f"; done
log "done"
