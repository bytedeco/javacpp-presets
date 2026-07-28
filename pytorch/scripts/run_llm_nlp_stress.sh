#!/usr/bin/env bash
# Multi-dimensional full-API stress for llm NLP stack:
#   spacy | text | ragas | nltk | peft | sentence | tokenizers
#
# Usage:
#   ./scripts/run_llm_nlp_stress.sh              # all 7
#   ./scripts/run_llm_nlp_stress.sh nltk ragas   # subset
#   ./scripts/run_llm_nlp_stress.sh --list
#
# Results:
#   samples/out/llm_nlp_stress/SUMMARY.md
#   samples/out/llm_nlp_stress/<name>.log
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

OUT="samples/out/llm_nlp_stress"
mkdir -p target/samples-compile "$OUT"

ALL=(spacy text ragas nltk peft sentence tokenizers)

if [[ "${1:-}" == "--list" ]]; then
  printf '%s\n' "${ALL[@]}"
  exit 0
fi

# resolve selection
SELECT=()
if [[ $# -eq 0 ]]; then
  SELECT=("${ALL[@]}")
else
  for a in "$@"; do
    SELECT+=("$a")
  done
fi

# classpath
CP_BASE="target/classes"
if [[ -f target/cp.txt ]]; then
  CP_BASE="$CP_BASE:$(cat target/cp.txt)"
fi
EXTRA=""
if [[ -d target/dependency ]]; then
  for j in target/dependency/*-macosx-arm64.jar target/dependency/*.jar; do
    [[ -f "$j" ]] && EXTRA="$EXTRA:$j"
  done
fi
[[ -f target/pytorch.jar ]] && EXTRA="$EXTRA:target/pytorch.jar"
[[ -f target/pytorch-macosx-arm64.jar ]] && EXTRA="$EXTRA:target/pytorch-macosx-arm64.jar"
CP_COMPILE="$CP_BASE$EXTRA"

NAT="target/native/org/bytedeco/pytorch/macosx-arm64"
NAT2="${HOME}/.javacpp/cache/openblas-0.3.33-1.5.14-SNAPSHOT-macosx-arm64.jar/org/bytedeco/openblas/macosx-arm64"
# also pick any openblas cache if version differs
if [[ ! -d "$NAT2" ]]; then
  NAT2="$(ls -d "${HOME}"/.javacpp/cache/openblas-*-macosx-arm64.jar/org/bytedeco/openblas/macosx-arm64 2>/dev/null | head -1 || true)"
fi

XMX="${JAVA_XMX:-4g}"
RCP="target/samples-compile:target/classes:$CP_BASE$EXTRA"
JAVA_OPTS=(--enable-native-access=ALL-UNNAMED --add-opens=java.base/java.nio=ALL-UNNAMED -Xmx"${XMX}")
[[ -n "${NAT2:-}" ]] && JAVA_OPTS+=(-Djava.library.path="${NAT}:${NAT2}")
[[ -z "${NAT2:-}" ]] && JAVA_OPTS+=(-Djava.library.path="${NAT}")

class_for() {
  case "$1" in
    spacy) echo samples.BenchmarkSpacy ;;
    text) echo samples.BenchmarkText ;;
    ragas) echo samples.BenchmarkRagas ;;
    nltk) echo samples.BenchmarkNltk ;;
    peft) echo samples.BenchmarkPeft ;;
    sentence) echo samples.BenchmarkSentenceTransformers ;;
    tokenizers) echo samples.BenchmarkTokenizers ;;
    *) echo "" ;;
  esac
}

src_for() {
  case "$1" in
    spacy) echo samples/BenchmarkSpacy.java ;;
    text) echo samples/BenchmarkText.java ;;
    ragas) echo samples/BenchmarkRagas.java ;;
    nltk) echo samples/BenchmarkNltk.java ;;
    peft) echo samples/BenchmarkPeft.java ;;
    sentence) echo samples/BenchmarkSentenceTransformers.java ;;
    tokenizers) echo samples/BenchmarkTokenizers.java ;;
    *) echo "" ;;
  esac
}

echo "[compile] llm nlp stress samples → target/samples-compile"
SRCS=()
for name in "${SELECT[@]}"; do
  s="$(src_for "$name")"
  if [[ -z "$s" || ! -f "$s" ]]; then
    echo "ERROR: unknown module '$name'" >&2
    exit 2
  fi
  SRCS+=("$s")
done

javac -cp "target/samples-compile:$CP_COMPILE" -d target/samples-compile "${SRCS[@]}"

SUMMARY="$OUT/SUMMARY.md"
{
  echo "# LLM NLP multi-dimensional stress"
  echo
  echo "Date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo
  echo "| Module | Status | passed | failed | log |"
  echo "|--------|--------|--------|--------|-----|"
} > "$SUMMARY"

pass_n=0
fail_n=0
for name in "${SELECT[@]}"; do
  cls="$(class_for "$name")"
  log="$OUT/${name}.log"
  echo
  echo "════════════════════════════════════════════════════════"
  echo "[run] $cls"
  echo "════════════════════════════════════════════════════════"
  set +e
  java "${JAVA_OPTS[@]}" -cp "$RCP" "$cls" 2>&1 | tee "$log"
  rc=${PIPESTATUS[0]}
  set -e

  # parse passed/failed from log tail
  p=$(rg -o 'passed=([0-9]+)' -r '$1' "$log" | tail -1 || echo "?")
  f=$(rg -o 'failed=([0-9]+)' -r '$1' "$log" | tail -1 || echo "?")
  if [[ $rc -eq 0 ]]; then
    status="PASS"
    pass_n=$((pass_n + 1))
  else
    status="FAIL"
    fail_n=$((fail_n + 1))
  fi
  echo "| $name | $status | $p | $f | \`${name}.log\` |" >> "$SUMMARY"
done

{
  echo
  echo "## Totals"
  echo
  echo "- modules ok: **$pass_n**"
  echo "- modules failed: **$fail_n**"
} >> "$SUMMARY"

echo
echo "════════════════════════════════════════════════════════"
echo "SUMMARY: ok=$pass_n failed=$fail_n  → $SUMMARY"
echo "════════════════════════════════════════════════════════"
[[ $fail_n -eq 0 ]]
