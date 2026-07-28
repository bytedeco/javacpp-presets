#!/usr/bin/env bash
# Rebuild libtorch with optional ProcessGroupMPI / ProcessGroupUCC, then note
# how to re-run JavaCPP generation for profiler + distributed backends.
#
# Usage (from pytorch/ module root):
#   bash scripts/rebuild_distributed_backends.sh           # auto-detect MPI/UCC
#   USE_MPI=1 bash scripts/rebuild_distributed_backends.sh # force MPI
#   USE_UCC=1 UCC_HOME=/path bash scripts/rebuild_distributed_backends.sh
#
# After cppbuild finishes you still need jnitorch_mpi:
#   bash scripts/build_jnitorch_mpi.sh
#   # or full:
#   JAVACPP_ENABLE_MPI_NATIVE=1 mvn -DskipTests process-classes package
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# OpenMPI (Homebrew)
if [[ -d /opt/homebrew/opt/open-mpi ]]; then
  export MPI_HOME="${MPI_HOME:-/opt/homebrew/opt/open-mpi}"
  export PATH="$MPI_HOME/bin:$PATH"
  export DYLD_LIBRARY_PATH="${MPI_HOME}/lib:${DYLD_LIBRARY_PATH:-}"
elif [[ -d /usr/local/opt/open-mpi ]]; then
  export MPI_HOME="${MPI_HOME:-/usr/local/opt/open-mpi}"
  export PATH="$MPI_HOME/bin:$PATH"
fi

# UCC headers-only tree used for JavaCPP parse (not enough to link libtorch UCC)
DEPS_UCC="$ROOT/cppbuild/deps/install"
if [[ -f "$DEPS_UCC/include/ucc/api/ucc.h" ]]; then
  export UCC_INCLUDE="${UCC_INCLUDE:-$DEPS_UCC/include}"
  echo "UCC headers available for JavaCPP parse at $UCC_INCLUDE"
  echo "  (full USE_UCC=1 libtorch link still needs libucc on Linux)"
fi

echo "=== rebuild flags ==="
echo "  MPI_HOME=${MPI_HOME:-<unset>}"
echo "  USE_MPI will auto-enable if mpi.h / mpicc found (see cppbuild.sh)"
echo "  USE_UCC only if libucc is installed (not headers-only)"
echo

# Parent javacpp-presets cppbuild entry
if [[ -f "$ROOT/../cppbuild.sh" ]]; then
  cd "$ROOT/.."
  # Normalize uname platform to javacpp platform id
  _os=$(uname -s | tr '[:upper:]' '[:lower:]')
  _arch=$(uname -m)
  case "$_arch" in
    arm64|aarch64) _arch=arm64 ;;
    x86_64|amd64) _arch=x86_64 ;;
  esac
  case "$_os" in
    darwin) _os=macosx ;;
    linux) _os=linux ;;
  esac
  bash cppbuild.sh -platform "${_os}-${_arch}" "$@" pytorch
else
  echo "Run from javacpp-presets tree: bash cppbuild.sh pytorch"
  echo "With OpenMPI installed, cppbuild.sh sets USE_MPI=1 automatically."
fi

# Verify MPI symbols after rebuild
CPU_LIB=""
for c in \
  "$ROOT/cppbuild/macosx-arm64/lib/libtorch_cpu.dylib" \
  "$ROOT/cppbuild/macosx-arm64/pytorch/torch/lib/libtorch_cpu.dylib" \
  "$ROOT/cppbuild/linux-x86_64/lib/libtorch_cpu.so" \
  "$ROOT/cppbuild/linux-x86_64/pytorch/torch/lib/libtorch_cpu.so"
do
  if [[ -f "$c" ]]; then CPU_LIB="$c"; break; fi
done
if [[ -n "$CPU_LIB" ]]; then
  if nm -gU "$CPU_LIB" 2>/dev/null | grep -q createProcessGroupMPI; then
    echo "OK: $CPU_LIB exports createProcessGroupMPI"
  else
    echo "WARN: $CPU_LIB does NOT export createProcessGroupMPI (USE_MPI may be off)"
  fi
fi

# Ensure platform include symlink is not a stale absolute Downloads/libtorch path
for inc in "$ROOT"/cppbuild/*/include; do
  [[ -e "$inc" || -L "$inc" ]] || continue
  if [[ -L "$inc" ]]; then
    target=$(readlink "$inc" || true)
    case "$target" in
      pytorch/torch/include|*/pytorch/torch/include) ;;
      *)
        echo "Fixing stale include symlink $inc -> $target"
        rm -f "$inc"
        # platform dir is cppbuild/<plat>
        plat_dir=$(dirname "$inc")
        if [[ -d "$plat_dir/pytorch/torch/include" ]]; then
          ln -sfn pytorch/torch/include "$inc"
        fi
        ;;
    esac
  fi
done

cat <<'EOF'

Next steps
----------
1. Build jnitorch_mpi into the platform classifier jar (fast, MPI-only):

     bash scripts/build_jnitorch_mpi.sh

   Or full JavaCPP re-parse + all natives:

     JAVACPP_ENABLE_MPI_NATIVE=1 USE_MPI=1 mvn -DskipTests process-classes package

2. Profiler demo (no libtorch rebuild required — USE_KINETO already on):

     # after regenerate, run MemoryProfilerExample
     # expect memory_profile.json from enableProfiler/disableProfiler/save

3. ProcessGroupMPI needs libtorch rebuilt with USE_MPI=1 (this script / cppbuild).
   jnitorch_mpi auto-enables when createProcessGroupMPI is exported, or set
   JAVACPP_ENABLE_MPI_NATIVE=1.

4. ProcessGroupUCC native: Linux + UCC/UCX install + USE_UCC=1 rebuild.
   Mac: Java glue only (headers under cppbuild/deps/install).

5. Runtime: system OpenMPI still required for transitive deps (libopen-pal,
   libevent, hwloc, pmix). Prefer `mpirun -n N java ...` for multi-rank.

EOF
