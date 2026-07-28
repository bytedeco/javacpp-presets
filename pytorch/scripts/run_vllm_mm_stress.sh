#!/usr/bin/env bash
# Run OmniLLM multimodal multi-backbone stress on Mac and save results for human review.
#
# Usage:
#   ./scripts/run_vllm_mm_stress.sh                 # prefer backbones (qwen/deepseek/llama/glm)
#   ./scripts/run_vllm_mm_stress.sh --encoder-only  # encoders only (fast)
#   ./scripts/run_vllm_mm_stress.sh --only qwen
#   ./scripts/run_vllm_mm_stress.sh --only qwen,deepseek --tokens 24 --rounds 2
#
# Results:
#   samples/out/vllm_mm_stress/RESULTS.md
#   samples/out/vllm_mm_stress/results.jsonl
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

mkdir -p target/samples-compile samples/out/vllm_mm_stress

CP_BASE="target/classes"
if [[ -f target/cp.txt ]]; then
  CP_BASE="$CP_BASE:$(cat target/cp.txt)"
fi
# platform natives (openblas / opencv / ffmpeg classifiers) + local pytorch jars
EXTRA=""
if [[ -d target/dependency ]]; then
  for j in target/dependency/*-macosx-arm64.jar; do
    [[ -f "$j" ]] && EXTRA="$EXTRA:$j"
  done
fi
[[ -f target/pytorch.jar ]] && EXTRA="$EXTRA:target/pytorch.jar"
[[ -f target/pytorch-macosx-arm64.jar ]] && EXTRA="$EXTRA:target/pytorch-macosx-arm64.jar"
# compile classpath must include pytorch.jar (Tensor / torch natives bindings)
CP_COMPILE="$CP_BASE$EXTRA"

echo "[compile] safetensors/FP8 + Qwen3 registry + encoders + OmniLLM + samples"
javac -cp "$CP_COMPILE" -d target/classes \
  src/main/java/org/bytedeco/pytorch/data/safetensors/SafeDType.java \
  src/main/java/org/bytedeco/pytorch/data/safetensors/SafeTensors.java \
  src/main/java/org/bytedeco/pytorch/llm/transformers/loading/Fp8WeightDequant.java \
  src/main/java/org/bytedeco/pytorch/llm/transformers/loading/SnapshotFiles.java \
  src/main/java/org/bytedeco/pytorch/llm/transformers/PretrainedConfig.java \
  src/main/java/org/bytedeco/pytorch/llm/transformers/mapping/WeightMaps.java \
  src/main/java/org/bytedeco/pytorch/llm/transformers/mapping/ModelRegistry.java \
  src/main/java/org/bytedeco/pytorch/llm/transformers/modeling/Qwen3ForCausalLM.java \
  src/main/java/org/bytedeco/pytorch/llm/transformers/AutoModelForCausalLM.java \
  src/main/java/org/bytedeco/pytorch/llm/vllm/multimodal/encoders/VideoEncoder.java \
  src/main/java/org/bytedeco/pytorch/llm/vllm/multimodal/encoders/OcrEncoder.java \
  src/main/java/org/bytedeco/pytorch/llm/vllm/multimodal/encoders/AsrEncoder.java \
  src/main/java/org/bytedeco/pytorch/llm/vllm/multimodal/encoders/FunctionalVisionEncoder.java \
  src/main/java/org/bytedeco/pytorch/llm/vllm/multimodal/encoders/Qwen3VLEncoder.java \
  src/main/java/org/bytedeco/pytorch/llm/vllm/multimodal/encoders/DeepSeekVLEncoder.java \
  src/main/java/org/bytedeco/pytorch/llm/vllm/multimodal/encoders/MediaEncoderRegistry.java \
  src/main/java/org/bytedeco/pytorch/llm/vllm/multimodal/CompositeMultimodalProcessor.java \
  src/main/java/org/bytedeco/pytorch/llm/vllm/LLM.java \
  src/main/java/org/bytedeco/pytorch/llm/vllm/OmniLLM.java
javac -cp "target/samples-compile:$CP_COMPILE" -d target/samples-compile \
  samples/BenchmarkVllmMultimodalStress.java \
  samples/BenchmarkOmniMultimodal.java

NAT="target/native/org/bytedeco/pytorch/macosx-arm64"
NAT2="${HOME}/.javacpp/cache/openblas-0.3.33-1.5.14-SNAPSHOT-macosx-arm64.jar/org/bytedeco/openblas/macosx-arm64"
RCP="target/samples-compile:target/classes:$CP_BASE$EXTRA"

# 2B FP8→BF16 dequant needs headroom; default 14g (override with JAVA_XMX)
XMX="${JAVA_XMX:-14g}"
echo "[run] BenchmarkVllmMultimodalStress -Xmx${XMX} $*"
exec java --enable-native-access=ALL-UNNAMED -Xmx"${XMX}" \
  -Djava.library.path="${NAT}:${NAT2}" \
  -cp "$RCP" \
  samples.BenchmarkVllmMultimodalStress \
  --models-dir models \
  --fixtures samples/fixtures/multimodal \
  --out samples/out/vllm_mm_stress \
  "$@"
