#!/bin/bash
# Checks that chunks written by this preset (C-Blosc2) can be decoded by a real C-Blosc 1.x reader.
# Requires: pixi (to obtain C-Blosc 1.x from conda-forge), a C compiler, and a Java build of this preset.
# Usage: CP=<classpath with javacpp, blosc2 and blosc2-<platform> jars> ./run.sh
# Without CP, the jars in ../../target and the javacpp jar of ~/.m2 are used.
set -eu
cd "$(dirname "$0")"
OUT=${OUT:-$(mktemp -d)}
if [[ -z ${CP:-} ]]; then
    JAVACPP=$(find ~/.m2 -name 'javacpp-1.5.8-SNAPSHOT.jar' | head -1)
    CP=$(ls ../../target/blosc2.jar ../../target/blosc2-*.jar | grep -v 'sources\|javadoc' | tr '\n' ':')$JAVACPP
fi

pixi install >/dev/null
PREFIX=$(pixi run -q sh -c 'echo $CONDA_PREFIX')
${CC:-cc} -O1 -I"$PREFIX/include" reader.c -L"$PREFIX/lib" -lblosc -Wl,-rpath,"$PREFIX/lib" -o "$OUT/reader"
javac -cp "$CP" -d "$OUT" BloscChunkWriter.java

status=0
# Expected: "default" fails (stock C-Blosc2 chunks), the other two succeed
java -cp "$CP:$OUT" BloscChunkWriter default "$OUT/default"
echo "[stock blosc1_compress(), expected to FAIL in C-Blosc 1.x]"
"$OUT/reader" "$OUT/default" || true

BLOSC_BLOSC1_COMPAT=1 java -cp "$CP:$OUT" BloscChunkWriter default "$OUT/env"
echo "[blosc1_compress() with BLOSC_BLOSC1_COMPAT=1, expected to pass]"
"$OUT/reader" "$OUT/env" || status=1

java -cp "$CP:$OUT" BloscChunkWriter ctx "$OUT/ctx"
echo "[blosc2_ctx_set_blosc1_format(), expected to pass]"
"$OUT/reader" "$OUT/ctx" || status=1
exit $status
