#!/bin/bash
echo "Using prog script for $1 project.."
while true; do echo .; tail -10 ~/build/vb216/javacpp-presets/buildlogs/$1.log; sleep 300; done

