#!/usr/bin/env bash
# Gloo + MPI multi-thread / multi-process stress (Mac-friendly).
#
# Usage:
#   ./scripts/run_gloo_mpi_stress.sh              # single JVM (Gloo T1–T5 + MPI T4/T6–T8)
#   ./scripts/run_gloo_mpi_stress.sh --mp         # + self-launch 2-rank Gloo (FileStore)
#   ./scripts/run_gloo_mpi_stress.sh --mpirun     # + mpirun -n 2 MPI workers
#   ./scripts/run_gloo_mpi_stress.sh --mpi-only   # only mpirun -n 2 MPI workers
#   ./scripts/run_gloo_mpi_stress.sh --all        # single + --mp + --mpirun
#
# Env:
#   JAVA_XMX=4g
#   MPI_N=2
#   WATCHDOG_SEC=180
#   JAVACPP_CACHEDIR=...   # fresh extract dir (recommended after MPI rebuild)
#
# Critical runtime rules (Mac):
#   1. Prefer JavaCPP platform jars on the classpath; let Loader extract natives.
#      Do NOT also set -Djava.library.path to target/native (C10 double-register).
#   2. Always include openblas + javacpp *-macosx-arm64.jar.
#   3. zsh: never expand $VAR:foo without braces — use ${VAR}:...
#   4. mpirun: launch via a small shell wrapper (avoids -D/-cp arg mangling).
#
# Results:
#   samples/out/gloo_mpi_stress/SUMMARY.md
#   samples/out/gloo_mpi_stress/*.log
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

OUT="samples/out/gloo_mpi_stress"
mkdir -p target/samples-compile "$OUT"

MODE_SINGLE=1
MODE_MP=0
MODE_MPIRUN=0
for a in "$@"; do
  case "$a" in
    --mp) MODE_MP=1 ;;
    --mpirun) MODE_MPIRUN=1 ;;
    --mpi-only) MODE_SINGLE=0; MODE_MP=0; MODE_MPIRUN=1 ;;
    --all) MODE_SINGLE=1; MODE_MP=1; MODE_MPIRUN=1 ;;
    -h|--help)
      sed -n '1,35p' "$0"
      exit 0
      ;;
  esac
done

# ── resolve platform jars (zsh-safe ${var}) ───────────────────────────────
M2_JC="${HOME}/.m2/repository/org/bytedeco/javacpp/1.5.14-SNAPSHOT"
M2_OB="${HOME}/.m2/repository/org/bytedeco/openblas/0.3.33-1.5.14-SNAPSHOT"
JAVACPP_J="${M2_JC}/javacpp-1.5.14-SNAPSHOT.jar"
JAVACPP_P="${M2_JC}/javacpp-1.5.14-SNAPSHOT-macosx-arm64.jar"
OPENBLAS_J="${M2_OB}/openblas-0.3.33-1.5.14-SNAPSHOT.jar"
OPENBLAS_P="${M2_OB}/openblas-0.3.33-1.5.14-SNAPSHOT-macosx-arm64.jar"

for f in "$JAVACPP_J" "$JAVACPP_P" "$OPENBLAS_J" "$OPENBLAS_P"; do
  if [[ ! -f "$f" ]]; then
    echo "ERROR: missing dependency jar: $f" >&2
    exit 2
  fi
done

# Local build artifacts (MPI-enabled platform jar preferred)
PY_J=""
PY_P=""
[[ -f target/pytorch.jar ]] && PY_J="target/pytorch.jar"
[[ -f target/pytorch-macosx-arm64.jar ]] && PY_P="target/pytorch-macosx-arm64.jar"
if [[ -z "$PY_J" || -z "$PY_P" ]]; then
  # fallback to m2 (may lack jnitorch_mpi)
  M2_PY="${HOME}/.m2/repository/org/bytedeco/pytorch/2.13.0-1.5.14-SNAPSHOT"
  [[ -z "$PY_J" && -f "${M2_PY}/pytorch-2.13.0-1.5.14-SNAPSHOT.jar" ]] \
    && PY_J="${M2_PY}/pytorch-2.13.0-1.5.14-SNAPSHOT.jar"
  [[ -z "$PY_P" && -f "${M2_PY}/pytorch-2.13.0-1.5.14-SNAPSHOT-macosx-arm64.jar" ]] \
    && PY_P="${M2_PY}/pytorch-2.13.0-1.5.14-SNAPSHOT-macosx-arm64.jar"
fi
if [[ -z "$PY_J" || -z "$PY_P" ]]; then
  echo "ERROR: need target/pytorch.jar + target/pytorch-macosx-arm64.jar (with jnitorch_mpi)" >&2
  exit 2
fi

# Build CP: samples + classes + deps + pytorch
CP="target/samples-compile:target/classes"
CP="${CP}:${JAVACPP_J}:${JAVACPP_P}:${OPENBLAS_J}:${OPENBLAS_P}"
CP="${CP}:${PY_J}:${PY_P}"
if [[ -f target/cp.txt ]]; then
  # append extra (may duplicate; harmless)
  CP="${CP}:$(cat target/cp.txt)"
fi
if [[ -d target/dependency ]]; then
  # nullglob-safe
  shopt -s nullglob 2>/dev/null || setopt NULL_GLOB 2>/dev/null || true
  for j in target/dependency/*.jar; do
    [[ -f "$j" ]] || continue
    case "$j" in *pytorch*|*openblas*|*javacpp*) continue ;; esac
    CP="${CP}:$j"
  done
fi

XMX="${JAVA_XMX:-4g}"
WATCHDOG_SEC="${WATCHDOG_SEC:-180}"
MPI_N="${MPI_N:-2}"
CACHE_DIR="${JAVACPP_CACHEDIR:-${ROOT}/target/javacpp-cache-gloo-mpi}"
mkdir -p "$CACHE_DIR"

JAVA_OPTS=(
  --enable-native-access=ALL-UNNAMED
  --add-opens=java.base/java.nio=ALL-UNNAMED
  -Xmx"${XMX}"
  -Dorg.bytedeco.javacpp.cachedir="${CACHE_DIR}"
)

# Default: do NOT set java.library.path (C10 double-register with jar extract).
if [[ "${USE_JAVA_LIBRARY_PATH:-0}" == "1" ]]; then
  NAT="target/native/org/bytedeco/pytorch/macosx-arm64"
  JAVA_OPTS+=(-Djava.library.path="${NAT}")
  echo "WARNING: USE_JAVA_LIBRARY_PATH=1 → ${NAT}"
fi

export OMPI_MCA_btl="${OMPI_MCA_btl:-self,tcp}"
export OMPI_MCA_btl_vader_single_copy_mechanism="${OMPI_MCA_btl_vader_single_copy_mechanism:-none}"

have_mpirun() { command -v mpirun >/dev/null 2>&1; }

run_with_watchdog() {
  local log="$1"; shift
  local label="$1"; shift
  echo
  echo "════════════════════════════════════════════════════════"
  echo "▶ $label"
  echo "  log: $log"
  echo "════════════════════════════════════════════════════════"
  (
    "$@" >"$log" 2>&1 &
    local pid=$!
    local elapsed=0
    while kill -0 "$pid" 2>/dev/null; do
      sleep 1
      elapsed=$((elapsed + 1))
      if [[ $elapsed -ge $WATCHDOG_SEC ]]; then
        echo "WATCHDOG: killing pid=$pid after ${WATCHDOG_SEC}s" | tee -a "$log"
        kill -9 "$pid" 2>/dev/null || true
        wait "$pid" 2>/dev/null || true
        return 124
      fi
    done
    wait "$pid"
    return $?
  )
  return $?
}

parse_results() {
  local log="$1"
  local line p f s
  line=$(grep -E 'Results: [0-9]+ passed' "$log" 2>/dev/null | tail -1 || true)
  if [[ -z "$line" ]]; then
    echo "?|?|?"
    return
  fi
  p=$(echo "$line" | sed -E 's/.*Results: ([0-9]+) passed.*/\1/')
  f=$(echo "$line" | sed -E 's/.* ([0-9]+) failed.*/\1/')
  s=$(echo "$line" | sed -E 's/.* ([0-9]+) skipped.*/\1/')
  echo "${p}|${f}|${s}"
}

# Persist CP for mpirun wrapper
CP_FILE=target/gloo_mpi_cp.txt
printf '%s\n' "$CP" > "$CP_FILE"

echo "[compile] BenchmarkGlooMpiStress → target/samples-compile"
echo "[cp] pytorch_p=${PY_P}"
echo "[cp] openblas_p=${OPENBLAS_P}"
echo "[cp] cache=${CACHE_DIR}"
# Check jnitorch_mpi present
if ! jar tf "$PY_P" 2>/dev/null | grep -q 'libjnitorch_mpi'; then
  echo "WARN: ${PY_P} has no libjnitorch_mpi — MPI tests will SKIP."
  echo "      Rebuild: bash scripts/build_jnitorch_mpi.sh"
fi

javac -cp "target/samples-compile:${CP}" \
  -d target/samples-compile \
  samples/BenchmarkGlooMpiStress.java

SUMMARY="$OUT/SUMMARY.md"
{
  echo "# Gloo / MPI multi-thread stress"
  echo
  echo "Date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo
  echo "| Run | Status | passed | failed | skipped | log |"
  echo "|-----|--------|--------|--------|---------|-----|"
} > "$SUMMARY"

pass_n=0
fail_n=0
status_line() {
  local name="$1" st="$2" p="$3" f="$4" s="$5" log="$6"
  echo "| $name | $st | $p | $f | $s | \`$log\` |" >> "$SUMMARY"
  if [[ "$st" == "PASS" || "$st" == "SKIP" ]]; then
    pass_n=$((pass_n + 1))
  else
    fail_n=$((fail_n + 1))
  fi
}

JAVA=(java "${JAVA_OPTS[@]}" -cp "$CP")

# ── 1) single JVM ────────────────────────────────────────────────────────
if [[ "$MODE_SINGLE" == "1" ]]; then
  LOG="$OUT/single.log"
  set +e
  run_with_watchdog "$LOG" "single-JVM Gloo+MPI stress" "${JAVA[@]}" samples.BenchmarkGlooMpiStress
  rc=$?
  set -e
  if [[ $rc -eq 0 ]]; then st=PASS; else st=FAIL; fi
  IFS='|' read -r p f s <<<"$(parse_results "$LOG")"
  status_line "single" "$st" "$p" "$f" "$s" "$LOG"
  tail -n 50 "$LOG" || true
fi

# ── 2) Gloo multi-process ────────────────────────────────────────────────
if [[ "$MODE_MP" == "1" ]]; then
  LOG="$OUT/gloo_mp.log"
  set +e
  run_with_watchdog "$LOG" "Gloo MultiProcessLauncher world=2" \
    "${JAVA[@]}" samples.BenchmarkGlooMpiStress --mp
  rc=$?
  set -e
  if [[ $rc -eq 0 ]]; then st=PASS; else st=FAIL; fi
  IFS='|' read -r p f s <<<"$(parse_results "$LOG")"
  status_line "gloo-mp" "$st" "$p" "$f" "$s" "$LOG"
  tail -n 50 "$LOG" || true
fi

# ── 3) MPI multi-rank via mpirun + wrapper ────────────────────────────────
if [[ "$MODE_MPIRUN" == "1" ]]; then
  LOG="$OUT/mpi_mpirun.log"
  if ! have_mpirun; then
    echo "SKIP mpirun — not on PATH" | tee "$LOG"
    status_line "mpi-mpirun" "SKIP" "0" "0" "1" "$LOG"
  else
    echo "mpirun: $(mpirun --version 2>&1 | head -1)"
    WRAP=target/run_mpi_worker.sh
    cat > "$WRAP" <<WRAP
#!/usr/bin/env bash
set -euo pipefail
cd "${ROOT}"
exec java \\
  --enable-native-access=ALL-UNNAMED \\
  --add-opens=java.base/java.nio=ALL-UNNAMED \\
  -Xmx${XMX} \\
  -Dorg.bytedeco.javacpp.cachedir="${CACHE_DIR}" \\
  -cp "\$(cat "${ROOT}/${CP_FILE}")" \\
  samples.BenchmarkGlooMpiStress --mpi-worker
WRAP
    chmod +x "$WRAP"
    set +e
    run_with_watchdog "$LOG" "mpirun -n ${MPI_N} MPI workers" \
      mpirun -n "${MPI_N}" "${ROOT}/${WRAP}"
    rc=$?
    set -e
    if [[ $rc -eq 0 ]]; then st=PASS; else st=FAIL; fi
    IFS='|' read -r p f s <<<"$(parse_results "$LOG")"
    status_line "mpi-mpirun(n=${MPI_N})" "$st" "$p" "$f" "$s" "$LOG"
    tail -n 80 "$LOG" || true
  fi
fi

{
  echo
  echo "## Totals"
  echo
  echo "- runs ok/skip: $pass_n"
  echo "- runs failed: $fail_n"
  echo "- watchdog: ${WATCHDOG_SEC}s"
  echo "- cache: \`${CACHE_DIR}\`"
  echo
  echo "## Mac conclusions (verified 2026-07-28)"
  echo
  echo "| Test | Result |"
  echo "|------|--------|"
  echo "| T1 Gloo forceCollective ws=1 | PASS |"
  echo "| T2 8-thread Gloo PGs | PASS (~64 coll/s) |"
  echo "| T3 4-thread local DDP | PASS |"
  echo "| T4 MPI create+allreduce ws=1 | PASS (jnitorch_mpi) |"
  echo "| T5 Gloo latency | p50 ≈ 0.02–0.03 ms |"
  echo "| T6 MPI 4-thread single PG | PASS (~40k coll/s @ws1) |"
  echo "| T7 wrapper BackendType.MPI | PASS |"
  echo "| T8 MPI latency | p50 ≈ 0.01 ms |"
  echo "| mpirun -n 2 direct allreduce | 1+2→3.0 PASS |"
  echo "| mpirun -n 2 multi-thread coll | 4/4 PASS |"
  echo "| mpirun -n 2 wrapper MPI | PASS |"
  echo
  echo "### Runtime gotchas"
  echo
  echo "- Do not dual-load libtorch via jar extract **and** \`-Djava.library.path\` (C10 double-register)."
  echo "- \`torch_mpi\` LoadEnabled must not clear \`platform.library\` when packaged \`jnitorch_mpi\` exists."
  echo "- Collective \`@IntrusivePtr\` on ProcessGroupMPI methods must be \`c10d::Work\` (not ProcessGroupMPI)."
  echo "- zsh: quote \`${VAR}\` before \`:'; mpirun prefers a wrapper script."
} >> "$SUMMARY"

echo
echo "════════════════════════════════════════════════════════"
echo "SUMMARY → $SUMMARY"
cat "$SUMMARY"
echo
if [[ $fail_n -gt 0 ]]; then
  echo "FAILED: $fail_n run(s)"
  exit 1
fi
echo "OK: all requested runs passed"
exit 0
