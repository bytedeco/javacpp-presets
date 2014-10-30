#!/bin/bash

set -Eeux
set -o pipefail

distname=precise
tgtdir="$(pwd)/osinst"

for m in opencv ffmpeg; do
    bash cppbuild.sh -platform x86_64 install $m
done
mvn install --projects opencv,ffmpeg

exit 0 

apt-get update
apt-get install debootstrap
debootstrap \
    --arch=amd64 \
    --variant=minbase \
    --keyring=/usr/share/keyrings/ubuntu-archive-keyring.gpg \
    $distname $tgtdir \
    http://de.archive.ubuntu.com/ubuntu/

find $tgtdir

