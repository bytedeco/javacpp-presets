#!/usr/bin/env bash
# Build only libjnitorch_mpi and package it into pytorch-<platform>.jar.
#
# Prerequisites:
#   - libtorch built with USE_MPI=1 (createProcessGroupMPI in libtorch_cpu)
#   - OpenMPI / MPICH on the machine (mpi.h + libmpi)
#   - target/classes already compiled (mvn compile at least once)
#
# Usage (from pytorch/ module root):
#   bash scripts/build_jnitorch_mpi.sh
#   JAVACPP_ENABLE_MPI_NATIVE=1 bash scripts/build_jnitorch_mpi.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# --- platform ---
_os=$(uname -s | tr '[:upper:]' '[:lower:]')
_arch=$(uname -m)
case "$_arch" in arm64|aarch64) _arch=arm64 ;; x86_64|amd64) _arch=x86_64 ;; esac
case "$_os" in darwin) PLATFORM=macosx-$_arch ;; linux) PLATFORM=linux-$_arch ;; *)
  echo "Unsupported OS: $_os"; exit 1 ;;
esac
echo "platform=$PLATFORM"

# --- MPI ---
if [[ -z "${MPI_HOME:-}" ]]; then
  for d in /opt/homebrew/opt/open-mpi /usr/local/opt/open-mpi /usr/lib64/openmpi /usr; do
    if [[ -f "$d/include/mpi.h" ]]; then export MPI_HOME="$d"; break; fi
  done
fi
if [[ -z "${MPI_HOME:-}" || ! -f "${MPI_HOME}/include/mpi.h" ]]; then
  echo "ERROR: mpi.h not found. brew install open-mpi  (or set MPI_HOME)"
  exit 1
fi
export PATH="${MPI_HOME}/bin:${PATH}"
export DYLD_LIBRARY_PATH="${MPI_HOME}/lib:${DYLD_LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="${MPI_HOME}/lib:${LD_LIBRARY_PATH:-}"
export JAVACPP_ENABLE_MPI_NATIVE=1
export USE_MPI=1
echo "MPI_HOME=$MPI_HOME"

# --- fix stale include symlink (Downloads/libtorch etc.) ---
INC_LINK="cppbuild/${PLATFORM}/include"
if [[ -L "$INC_LINK" ]]; then
  tgt=$(readlink "$INC_LINK" || true)
  if [[ "$tgt" != "pytorch/torch/include" && "$tgt" != */pytorch/torch/include ]]; then
    echo "Fixing stale include symlink $INC_LINK -> $tgt"
    rm -f "$INC_LINK"
    ln -sfn pytorch/torch/include "$INC_LINK"
  fi
elif [[ ! -e "$INC_LINK" && -d "cppbuild/${PLATFORM}/pytorch/torch/include" ]]; then
  ln -sfn pytorch/torch/include "$INC_LINK"
fi

# --- verify libtorch MPI ---
CPU_LIB=""
for c in \
  "cppbuild/${PLATFORM}/lib/libtorch_cpu.dylib" \
  "cppbuild/${PLATFORM}/lib/libtorch_cpu.so" \
  "cppbuild/${PLATFORM}/pytorch/torch/lib/libtorch_cpu.dylib" \
  "cppbuild/${PLATFORM}/pytorch/torch/lib/libtorch_cpu.so"
do
  if [[ -f "$c" ]]; then CPU_LIB="$c"; break; fi
done
if [[ -z "$CPU_LIB" ]]; then
  echo "ERROR: libtorch_cpu not found under cppbuild/${PLATFORM}"
  exit 1
fi
if ! nm -gU "$CPU_LIB" 2>/dev/null | grep -q createProcessGroupMPI; then
  echo "ERROR: $CPU_LIB does not export createProcessGroupMPI"
  echo "  Rebuild libtorch first: USE_MPI=1 bash scripts/rebuild_distributed_backends.sh"
  exit 1
fi
echo "OK: $CPU_LIB has ProcessGroupMPI"

# --- ensure classes compiled ---
if [[ ! -f target/classes/org/bytedeco/pytorch/presets/torch_mpi.class ]]; then
  echo "Compiling Java classes..."
  mvn -q -DskipTests compiler:compile
fi
# Ensure ProcessGroupMPI peer is up to date in target/classes
if [[ -f src/gen/java/org/bytedeco/pytorch/distributed/ProcessGroupMPI.java ]]; then
  javac -cp "target/classes:target/dependency/*" \
    -d target/classes \
    src/gen/java/org/bytedeco/pytorch/distributed/ProcessGroupMPI.java \
    src/gen/java/org/bytedeco/pytorch/global/torch_mpi.java \
    2>/dev/null || true
fi

OUT="target/native/org/bytedeco/pytorch/${PLATFORM}"
CFG="target/native/META-INF/native-image/${PLATFORM}"
mkdir -p "$OUT" "$CFG"

# Include / link paths (pom-equivalent + MPI + resource shims)
INC_PATHS=(
  "cppbuild/${PLATFORM}/include/torch/csrc/api/include"
  "cppbuild/${PLATFORM}/include"
  "target/classes/org/bytedeco/pytorch/include"
  "${MPI_HOME}/include"
  "../openblas/cppbuild/${PLATFORM}/include"
  "../openblas/src/main/resources/org/bytedeco/openblas/include"
  "../openblas/target/classes/org/bytedeco/openblas/include"
)
LINK_PATHS=(
  "cppbuild/${PLATFORM}/lib"
  "${MPI_HOME}/lib"
  # openblas required by jnitorch_mpi link line (-lopenblas via inherit torch)
  "../openblas/cppbuild/${PLATFORM}/lib"
  "/opt/homebrew/opt/openblas/lib"
  "/usr/local/opt/openblas/lib"
)
INC=$(IFS=:; echo "${INC_PATHS[*]}")
LINK=$(IFS=:; echo "${LINK_PATHS[*]}")

CP="target/classes:target/dependency/*"
echo "Building jnitorch_mpi..."
rm -f "$OUT/jnitorch_mpi.cpp" "$OUT/libjnitorch_mpi.dylib" "$OUT/libjnitorch_mpi.so"

java -cp "$CP" org.bytedeco.javacpp.tools.Builder \
  -classpath "target/classes" \
  -d "$OUT" \
  -configdir "$CFG" \
  -copylibs \
  -copyresources \
  -Dplatform.includepath="$INC" \
  -Dplatform.linkpath="$LINK" \
  org.bytedeco.pytorch.global.torch_mpi \
  org.bytedeco.pytorch.distributed.ProcessGroupMPI

# Locate built library
MPI_JNI=""
for f in "$OUT"/libjnitorch_mpi.dylib "$OUT"/libjnitorch_mpi.so "$OUT"/jnitorch_mpi.dll; do
  if [[ -f "$f" ]]; then MPI_JNI="$f"; break; fi
done
if [[ -z "$MPI_JNI" ]]; then
  echo "ERROR: jnitorch_mpi native library was not produced in $OUT"
  ls -la "$OUT"
  exit 1
fi
echo "Built: $MPI_JNI ($(wc -c < "$MPI_JNI") bytes)"

# macOS: make libmpi dependency relocatable via @rpath when we also ship libmpi
if [[ "$PLATFORM" == macosx-* && -f "$OUT/libmpi.dylib" ]]; then
  chmod u+w "$MPI_JNI" "$OUT/libmpi.dylib" || true
  # Real OpenMPI dylib is libmpi.40.dylib; copyLibs may have copied the symlink target as libmpi.dylib
  if [[ -f "${MPI_HOME}/lib/libmpi.40.dylib" ]]; then
    cp -p "${MPI_HOME}/lib/libmpi.40.dylib" "$OUT/libmpi.40.dylib"
    install_name_tool -id @rpath/libmpi.40.dylib "$OUT/libmpi.40.dylib" 2>/dev/null || true
    codesign --force -s - "$OUT/libmpi.40.dylib" 2>/dev/null || true
  fi
  install_name_tool -id @rpath/libmpi.dylib "$OUT/libmpi.dylib" 2>/dev/null || true
  # Retarget absolute Homebrew path → @rpath
  old_mpi=$(otool -L "$MPI_JNI" | awk '/libmpi/{print $1; exit}')
  if [[ -n "$old_mpi" && "$old_mpi" == /* ]]; then
    # Prefer soname .40 if present
    if [[ -f "$OUT/libmpi.40.dylib" ]]; then
      install_name_tool -change "$old_mpi" @rpath/libmpi.40.dylib "$MPI_JNI" || true
    else
      install_name_tool -change "$old_mpi" @rpath/libmpi.dylib "$MPI_JNI" || true
    fi
  fi
  codesign --force -s - "$MPI_JNI" 2>/dev/null || true
  codesign --force -s - "$OUT/libmpi.dylib" 2>/dev/null || true
  echo "otool -L jnitorch_mpi:"
  otool -L "$MPI_JNI" | head -10
fi

# Package classifier jar without recompiling natives
echo "Packaging pytorch-${PLATFORM}.jar ..."
mvn -q -DskipTests -Djavacpp.parser.skip=true -Djavacpp.compiler.skip=true package

echo "=== jar contents (mpi) ==="
jar tf "target/pytorch-${PLATFORM}.jar" | grep -E 'jnitorch_mpi|libmpi' || {
  echo "WARN: mpi libs not found inside jar — check native output path"
  exit 1
}
echo "DONE: jnitorch_mpi packaged into target/pytorch-${PLATFORM}.jar"
echo "Runtime tip: mpirun -n 2 java -cp target/pytorch.jar:target/pytorch-${PLATFORM}.jar:... YourMain"
