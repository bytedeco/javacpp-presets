#!/bin/bash

# TODO:
# - use travis-ci's directory caching for downloads
# - use travis-ci's apt caching for debootstrap

set -Eex
set -o pipefail

export LC_ALL=C
export TZ=UTC
DISTNAME=precise
DISTURL="http://de.archive.ubuntu.com/ubuntu"
DISTKEYRING="/usr/share/keyrings/ubuntu-archive-keyring.gpg"
DISTARCH="amd64"
PROJECTS="opencv ffmpeg"
BASEDIR="$(pwd)"
TGTDIR="$BASEDIR/osinst"
CACHEDIR="$BASEDIR/.cache"
INCHROOT="$1"

if ! NCPUS=$(grep -c ^proc /proc/cpuinfo); then
    NCPUS=4
fi


if [[ "$INCHROOT" == "build" ]]; then
    set -Eex

    set
    pwd
    whoami
    uname -a

    export JAVA_HOME="/usr/lib/jvm/java-7-openjdk-amd64"
    export PATH="$JAVA_HOME/bin:$PATH"
    export LD_LIBRARY_PATH="$JAVA_HOME/lib:$LD_LIBRARY_PATH"

    which java
    java -version
    which javac
    javac -version
    which mvn

    for project in $PROJECTS; do
        bash cppbuild.sh -platform linux-x86_64 install $project
    done
    mvn -V install -Djava.awt.headless=true --projects "${PROJECTS// /,}",tests

    exit 0 
fi

connect_chroot() {
    if ! mount | grep "$TGTDIR/proc"; then
        sudo mount -t proc proc "$TGTDIR/proc"
    fi
#    if ! mount | grep "$TGTDIR/dev"; then
#        sudo mount -o bind /etc "$TGTDIR/dev"
#    fi
}

release_chroot() {
    if mount | grep "$TGTDIR/proc"; then
        sudo umount "$TGTDIR/proc"
    fi
#    if mount | grep "$TGTDIR/dev"; then
#        sudo umount "$TGTDIR/dev"
#    fi
}

chroot_do() {
    connect_chroot
    sudo chroot "$TGTDIR" "$@"
}

function download {
    local url="$1"
    local destfile="$2"
    local cachefile="$CACHEDIR/${destfile##*/}"
    local tmpfile="$cachefile.tmp"
    if ! test -e "$cachefile"; then
        rm -f "$tmpfile"
        if curl -L -o "$tmpfile" "$url"; then
            mv -f "$tmpfile" "$cachefile"
        else
            echo "failed to retrieve $url" >&2
            exit 1
        fi
    fi
    cp -vf "$cachefile" "$destfile"
    return 0
}

function getgit {
    local project="$1"
    local tagname="$2"
    local destfile="$3"
    download "https://codeload.github.com/$project/tar.gz/$tagname" "$destfile"
}

function install_yasm {
    local curdir=$(pwd)
    if [[ ! -e "/usr/local/bin/yasm" ]]; then
        getgit yasm/yasm v1.3.0 yasm.tgz
        tar xzf yasm.tgz
        cd yasm-1.3.0
        cmake .
        make -j$NCPUS
        make install DESTDIR="$curdir/tools"
        cd ..
    fi
    yasm --version
}

if [[ "$INCHROOT" == "install" ]]; then
    freeuid="$2"
    echo "deb $DISTURL $DISTNAME main restricted universe
deb $DISTURL $DISTNAME-updates main restricted universe
deb $DISTURL $DISTNAME-security main restricted universe
" > "/etc/apt/sources.list"
    apt-get update
    apt-get -y dist-upgrade
    apt-get -y install openjdk-7-jdk maven build-essential
    install_yasm
    useradd -u $freeuid -d /build build
    touch .installed
    exit 0
fi

if [[ -n "$INCHROOT" ]]; then
    echo "unknown chroot cmd: $INCHROOT" >&2
    exit 1
fi


trap "release_chroot" EXIT

if ! test -e "$TGTDIR/.installed"; then
    sudo rm -rf "$TGTDIR"
    if ! which debootstrap; then
        sudo apt-get update
        sudo apt-get install debootstrap
    fi
    sudo debootstrap --arch="$DISTARCH" --variant=minbase \
        --keyring="$DISTKEYRING" "$DISTNAME" "$TGTDIR" "$DISTURL"
    sudo cp "$0" "$TGTDIR/inchroot.sh"
    chroot_do chmod 755 "inchroot.sh"

    # for more security, select a uid that is not used inside the host system
    freeuid=11723; while grep $freeuid /etc/passwd; do let 'freeuid++'; done
    chroot_do "./inchroot.sh" install $freeuid
fi

sudo rsync -av --del --exclude=/osinst/ "$BASEDIR/" "$TGTDIR/build"
chroot_do chown -R build build
sudo cp "$0" "$TGTDIR/inchroot.sh"
chroot_do chmod 755 "inchroot.sh"
chroot_do su - build -c "/inchroot.sh build"

