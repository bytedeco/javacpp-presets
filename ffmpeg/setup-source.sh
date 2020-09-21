#!/bin/bash

EXTENSION=$1

if [[ "$EXTENSION" == *gpl ]]; then
    cp ./src/main/java/org/bytedeco/ffmpeg/presets/postproc.java.enabled ./src/main/java/org/bytedeco/ffmpeg/presets/postproc.java
else
    cp ./src/main/java/org/bytedeco/ffmpeg/presets/postproc.java.disabled ./src/main/java/org/bytedeco/ffmpeg/presets/postproc.java
fi
rm -rf ./src/gen/
