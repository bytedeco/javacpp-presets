#!/bin/bash

set -Eeux
set -o pipefail

export LC_ALL=C
export TZ=UTC

distname=precise
tgtdir="$(pwd)/osinst"

set
pwd
whoami
uname -a
java -version
mvn -V clean
for m in opencv ffmpeg; do
    bash cppbuild.sh -platform linux-x86_64 install $m
done
mvn -V install --projects opencv,ffmpeg

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

