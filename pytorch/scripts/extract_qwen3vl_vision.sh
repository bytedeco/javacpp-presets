#!/usr/bin/env bash
# Extract Qwen3-VL vision tower (~814MB BF16) from the full 3.5GB FP8 checkpoint
# via HTTP Range requests, rewrite as a standalone vision_weights.safetensors.
#
# Usage:
#   ./scripts/extract_qwen3vl_vision.sh
#   ./scripts/extract_qwen3vl_vision.sh --dest models/Qwen__Qwen3-VL-2B-Instruct-FP8
#   HF_ENDPOINT=https://hf-mirror.com ./scripts/extract_qwen3vl_vision.sh
#
# Also downloads config.json + preprocessor_config.json into the dest dir.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MIRROR="${HF_ENDPOINT:-https://hf-mirror.com}"
REPO="Qwen/Qwen3-VL-2B-Instruct-FP8"
WEIGHT_FILE="model-00001-of-00001.safetensors"
DEST="${DEST:-$ROOT/models/Qwen__Qwen3-VL-2B-Instruct-FP8}"
TOKEN="${HF_TOKEN:-}"
CHUNK=$((64 * 1024 * 1024))
CACHE="${TMPDIR:-/tmp}/qwen3vl_extract"
ONLY_CONFIG=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dest) DEST="$2"; shift 2 ;;
    --mirror) MIRROR="$2"; shift 2 ;;
    --repo) REPO="$2"; shift 2 ;;
    --only-config) ONLY_CONFIG=1; shift ;;
    -h|--help) sed -n '1,20p' "$0"; exit 0 ;;
    *) echo "Unknown: $1" >&2; exit 2 ;;
  esac
done

mkdir -p "$DEST" "$CACHE"
AUTH=()
if [[ -n "$TOKEN" ]]; then AUTH=(-H "Authorization: Bearer $TOKEN"); fi
log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }

download_small() {
  local file="$1"
  local dest="$DEST/$file"
  if [[ -f "$dest" && -s "$dest" ]]; then return 0; fi
  log "GET $REPO/$file"
  curl -fsSL --http1.1 --retry 5 "${AUTH[@]}" \
    -o "$dest.partial" \
    "$MIRROR/$REPO/resolve/main/$file" \
    && mv "$dest.partial" "$dest" || rm -f "$dest.partial"
}

download_small config.json
download_small preprocessor_config.json
download_small video_preprocessor_config.json
download_small generation_config.json
download_small README.md

if (( ONLY_CONFIG )); then
  log "only-config done → $DEST"
  exit 0
fi

# If vision_weights already present and large enough, skip
if [[ -f "$DEST/vision_weights.safetensors" ]]; then
  sz=$(stat -f%z "$DEST/vision_weights.safetensors" 2>/dev/null || stat -c%s "$DEST/vision_weights.safetensors")
  if (( sz > 500000000 )); then
    log "SKIP vision_weights already present ($sz bytes)"
    ls -lh "$DEST"
    exit 0
  fi
fi
if [[ -f "$DEST/model.safetensors" ]]; then
  sz=$(stat -f%z "$DEST/model.safetensors" 2>/dev/null || stat -c%s "$DEST/model.safetensors")
  if (( sz > 500000000 )); then
    log "model.safetensors present ($sz) — will try extract if vision-only needed"
  fi
fi

HEADER="$CACHE/header.bin"
INDEX="$CACHE/index.json"
# Download index for weight map
if [[ ! -s "$INDEX" ]]; then
  log "GET model.safetensors.index.json"
  curl -fsSL --http1.1 --retry 5 "${AUTH[@]}" \
    -o "$INDEX" \
    "$MIRROR/$REPO/resolve/main/model.safetensors.index.json"
fi

# Download first 2MB of weight file for safetensors header
if [[ ! -s "$HEADER" ]] || [[ $(stat -f%z "$HEADER" 2>/dev/null || stat -c%s "$HEADER") -lt 200000 ]]; then
  log "GET weight header (first 2MB)"
  curl -fL --http1.1 --retry 5 "${AUTH[@]}" \
    -H "Range: bytes=0-2097151" \
    -o "$HEADER" \
    "$MIRROR/$REPO/resolve/main/$WEIGHT_FILE"
fi

OUT_ST="$DEST/vision_weights.safetensors"
PARTIAL="$OUT_ST.partial"
RAW_CHUNK="$CACHE/vision_raw.bin"

python3 - "$HEADER" "$RAW_CHUNK" "$OUT_ST" "$MIRROR" "$REPO" "$WEIGHT_FILE" "$TOKEN" "$CHUNK" <<'PY'
import struct, json, sys, os, subprocess, pathlib

header_path, raw_path, out_path, mirror, repo, weight_file, token, chunk = sys.argv[1:9]
chunk = int(chunk)

with open(header_path, 'rb') as f:
    n = struct.unpack('<Q', f.read(8))[0]
    header = json.loads(f.read(n))
meta = header.pop('__metadata__', {'format': 'pt'})
vis = {k: v for k, v in header.items() if 'visual' in k}
if not vis:
    print('ERROR: no visual tensors in header', file=sys.stderr)
    sys.exit(1)

offsets = []
for k, v in vis.items():
    s, e = v['data_offsets']
    offsets.append((s, e, k, v))
offsets.sort()
rel_start, rel_end = offsets[0][0], offsets[-1][1]
data_start = 8 + n
file_start = data_start + rel_start
file_end = data_start + rel_end  # exclusive
span = file_end - file_start
print(f'[extract] visual tensors={len(vis)} span_mb={span/1e6:.1f} file_bytes={file_start}-{file_end-1}')

# Download raw vision payload via Range if needed
need = True
if os.path.isfile(raw_path):
    sz = os.path.getsize(raw_path)
    if sz >= span:
        need = False
        print(f'[extract] raw cache hit {sz} bytes')
    else:
        print(f'[extract] raw partial {sz}/{span}, resume')

auth = []
if token:
    auth = ['-H', f'Authorization: Bearer {token}']

if need:
    start = os.path.getsize(raw_path) if os.path.isfile(raw_path) else 0
    mode = 'ab' if start else 'wb'
    url = f'{mirror}/{repo}/resolve/main/{weight_file}'
    with open(raw_path, mode) as out:
        pos = start
        while pos < span:
            end = min(pos + chunk, span) - 1  # inclusive relative to vision span
            abs_start = file_start + pos
            abs_end = file_start + end
            print(f'[extract] chunk {pos}-{end} abs={abs_start}-{abs_end}')
            tmp = raw_path + '.chunk'
            cmd = [
                'curl', '-fL', '--http1.1', '--retry', '8', '--retry-delay', '2',
                *auth,
                '-H', f'Range: bytes={abs_start}-{abs_end}',
                '-o', tmp, url,
                '-w', '    http=%{http_code} got=%{size_download}\n'
            ]
            r = subprocess.run(cmd)
            if r.returncode != 0 or not os.path.isfile(tmp):
                print('FAIL chunk', pos, file=sys.stderr)
                sys.exit(1)
            with open(tmp, 'rb') as t:
                data = t.read()
            os.remove(tmp)
            out.write(data)
            pos += len(data)
            print(f'    progress {100.0*pos/span:.1f}% ({pos}/{span})')

# Rebuild safetensors with remapped offsets
# New header: only visual tensors, offsets relative to new data section
new_header = {'__metadata__': meta if meta else {'format': 'pt'}}
# sort by original offset for sequential write
ordered = sorted(vis.items(), key=lambda kv: kv[1]['data_offsets'][0])
cursor = 0
# We will write data in order of original offsets (already contiguous)
# remap: new_offset = old_offset - rel_start
for k, v in ordered:
    s, e = v['data_offsets']
    ns, ne = s - rel_start, e - rel_start
    entry = {kk: vv for kk, vv in v.items() if kk != 'data_offsets'}
    entry['data_offsets'] = [ns, ne]
    new_header[k] = entry
    cursor = max(cursor, ne)

header_bytes = json.dumps(new_header, separators=(',', ':')).encode('utf-8')
# pad header to 8-byte alignment (optional but nice)
pad = (8 - (len(header_bytes) % 8)) % 8
header_bytes = header_bytes + b' ' * pad

out_tmp = out_path + '.writing'
with open(out_tmp, 'wb') as f:
    f.write(struct.pack('<Q', len(header_bytes)))
    f.write(header_bytes)
    with open(raw_path, 'rb') as r:
        # write exactly span bytes
        left = span
        while left:
            buf = r.read(min(8 * 1024 * 1024, left))
            if not buf:
                break
            f.write(buf)
            left -= len(buf)

os.replace(out_tmp, out_path)
print(f'[extract] wrote {out_path} size={os.path.getsize(out_path)}')
# quick validate
with open(out_path, 'rb') as f:
    nn = struct.unpack('<Q', f.read(8))[0]
    hh = json.loads(f.read(nn))
    hh.pop('__metadata__', None)
    print(f'[extract] validate keys={len(hh)} first={next(iter(hh))}')
PY

log "OK vision extract → $OUT_ST"
ls -lh "$DEST"
# also symlink/copy as model.safetensors so tryLoad finds weights
if [[ ! -f "$DEST/model.safetensors" ]]; then
  ln -sf vision_weights.safetensors "$DEST/model.safetensors" 2>/dev/null \
    || cp "$DEST/vision_weights.safetensors" "$DEST/model.safetensors"
fi
log "done"
