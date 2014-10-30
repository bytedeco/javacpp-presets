#!/bin/bash

# TODO: expire items in .cache/

set -Eex
set -o pipefail

export LC_ALL=C
export TZ=UTC
DISTNAME="precise"
DISTURL="http://archive.ubuntu.com/ubuntu"
DISTKEYRING="/usr/share/keyrings/ubuntu-archive-keyring.gpg"
DISTARCH="amd64"
PROJECTS="opencv ffmpeg"
BASEDIR="$(pwd)"
TGTDIR="$BASEDIR/osinst.$DISTNAME.$DISTARCH"
CACHEDIR="$BASEDIR/.cache"
M2REPODIR="$CACHEDIR/m2repo"
INCHROOT="$1"
export CCACHE_DIR="$CACHEDIR/ccache"

if ! NCPUS=$(grep -c ^proc /proc/cpuinfo); then
    NCPUS=4
fi
if (( NCPUS > 4 )); then NCPUS=4; fi

if [[ ! -e "$CACHEDIR" ]]; then mkdir -p "$CACHEDIR"; fi

if [[ -n "$INCHROOT" ]]; then
    export PATH="/usr/local/bin:$PATH"
    export LD_LIBRARY_PATH="/usr/local/lib:$LD_LIBRARY_PATH"
fi

if [[ "$INCHROOT" == "build" ]]; then
    set -Eex

    set
    pwd
    whoami
    uname -a
    free
    df -h || :
    du -sh "$CACHEDIR" || :
    cat /proc/cpuinfo | tail -n50

    export JAVA_HOME="/usr/lib/jvm/java-7-openjdk-amd64"
    export PATH="$JAVA_HOME/bin:$PATH"
    export LD_LIBRARY_PATH="$JAVA_HOME/lib:$LD_LIBRARY_PATH"
    export MAVEN_OPTS="-Dmaven.repo.local=$M2REPODIR -Djava.awt.headless=true -Dmaven.test.failure.ignore=false"

    which java
    java -version
    which javac
    javac -version
    which mvn

    for project in $PROJECTS; do
        bash cppbuild.sh -platform linux-x86_64 install $project
    done
    mvn -V -B install \
        --projects "${PROJECTS// /,}",tests

    exit 0 
fi

connectDir() {
    local outsideDir="$1"
    local insideDir="$2"
    if mount | grep "$TGTDIR$insideDir"; then return 0; fi
    if [[ ! -e "$outsideDir" ]]; then mkdir -p "$outsideDir"; fi
    sudo mount -o bind "$outsideDir" "$TGTDIR$insideDir" || :
    if mount | grep "$TGTDIR$insideDir"; then return 0; fi
    sudo mkdir -p "$TGTDIR$insideDir"
    sudo mount -o bind "$outsideDir" "$TGTDIR$insideDir"
}

disconnectDir() {
    local insideDir="$1"
    if ! mount | grep "$TGTDIR$insideDir"; then return 0; fi
    sudo umount "$TGTDIR$insideDir"
}

connect_chroot() {
    if ! mount | grep "$TGTDIR/proc"; then
        if [[ ! -e "$TGTDIR/proc" ]]; then sudo mkdir -p "$TGTDIR/proc"; fi
        sudo mount -t proc proc "$TGTDIR/proc"
    fi
    connectDir "$CACHEDIR" "/build/.cache"
    connectDir "$CACHEDIR/debs" "/var/cache/apt/archives"
}

release_chroot() {
    disconnectDir "/proc"
    disconnectDir "/build/.cache"
    disconnectDir "/var/cache/apt/archives"
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
        mkdir -p "$CACHEDIR"
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
        make install
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
    apt-get -y install openjdk-7-jdk maven build-essential curl cmake ccache
    install_yasm
    useradd -u $freeuid -d /build build
    exit 0
fi

if [[ -n "$INCHROOT" ]]; then
    echo "unknown chroot cmd: $INCHROOT" >&2
    exit 1
fi


trap "release_chroot" EXIT

if ! test -e "$TGTDIR/.installed"; then
    sudo rm -rf "$TGTDIR"
    sudo mkdir -p "$TGTDIR"
    connect_chroot
    sudo ln -sf build/.cache "$TGTDIR/.cache"
#    if [[ -e "$CACHEDIR" ]]; then
#        sudo mkdir -p "$TGTDIR/.cache"
#        sudo rsync -a --exclude=/debs/ "$CACHEDIR/" "$TGTDIR/.cache"
#    fi    
#    if [[ -e "$CACHEDIR/debs" ]]; then
#        sudo mkdir -p "$TGTDIR/var/cache/apt/archives"
#        sudo rsync -a "$CACHEDIR/debs/" "$TGTDIR/var/cache/apt/archives"
#    fi
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
#    mkdir -p "$CACHEDIR/debs"
#    sudo rsync -a "$TGTDIR/var/cache/apt/archives/" "$CACHEDIR/debs"
#    sudo rsync -a "$TGTDIR/.cache/" "$CACHEDIR/."
#    chroot_do rm -rf .cache
    chroot_do touch .installed
fi

sudo rsync -av -del --exclude=/.cache/ --exclude=/osinst.*/ "$BASEDIR/" "$TGTDIR/build"
chroot_do chown -R build build
sudo cp "$0" "$TGTDIR/inchroot.sh"
chroot_do chmod 755 "inchroot.sh"
chroot_do su - build -c "/inchroot.sh build"

