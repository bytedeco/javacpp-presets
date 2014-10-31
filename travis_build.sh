#!/bin/bash

# TODO: expire items in .cache/

set -Eex
set -o pipefail

export LC_ALL=C
export TZ=UTC
export DISTNAME="${DISTNAME:-precise}"
DISTURL="http://archive.ubuntu.com/ubuntu"
DISTKEYRING="/usr/share/keyrings/ubuntu-archive-keyring.gpg"
export DISTARCH="${DISTARCH:-amd64}"
DISTPARTS="main universe" # main restricted universe multiverse
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

du -sh "$CACHEDIR" || :

if [[ -n "$INCHROOT" ]]; then
    export PATH="/usr/local/bin:$PATH"
    export LD_LIBRARY_PATH="/usr/local/lib:$LD_LIBRARY_PATH"
fi

init_build_env() {
    export JAVA_HOME="/usr/lib/jvm/java-7-openjdk-amd64"
    if [[ ! -e "$JAVA_HOME" ]]; then
        export JAVA_HOME="/usr/lib/jvm/java-6-openjdk-amd64"
    fi
    export PATH="$JAVA_HOME/bin:/opt/maven/bin:/usr/lib/ccache:$PATH"
    export LD_LIBRARY_PATH="$JAVA_HOME/lib:$LD_LIBRARY_PATH"
    export MAVEN_OPTS="-Dmaven.repo.local=$M2REPODIR -Djava.awt.headless=true -Dmaven.test.failure.ignore=false"

}

if [[ "$INCHROOT" == "enter_stage2" ]]; then
    init_build_env
    su - build -c /bin/bash -l
    exit 0
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

    init_build_env

    which java
    java -version
    which javac
    javac -version
    which mvn

    ls -l /usr/bin/g++*

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
    sudo -E chroot "$TGTDIR" "$@"
}

function download {
    local url="$1"
    local destfile="$2"
    if [[ -z "$destfile" ]]; then
        destfile="${url##*/}"
    fi
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
    if [[ -z "$destfile" ]]; then
        destfile="${project##*/}-$tagname.tar.gz"
    fi
    download "https://codeload.github.com/$project/tar.gz/$tagname" "$destfile"
}

function install_yasm {
    local curdir=$(pwd)
    if [[ ! -e "/usr/local/bin/yasm" ]]; then
        getgit yasm/yasm v1.3.0
        tar xzf yasm-v1.3.0.tar.gz
        cd yasm-1.3.0
        cmake .
        make -j$NCPUS
        make install
        cd ..
    fi
    yasm --version
}

install_maven() {
    mkdir -p /opt/maven
    pushd /opt/maven
    download http://ftp-stud.hs-esslingen.de/pub/Mirrors/ftp.apache.org/dist/maven/maven-3/3.2.3/binaries/apache-maven-3.2.3-bin.tar.gz
    tar xzf apache-maven-3.2.3-bin.tar.gz --strip-components=1
    popd
}

install_cmake() {
    local tmpd=$(mktemp -d)
    pushd "$tmpd"
    download http://www.cmake.org/files/v2.8/cmake-2.8.12.2.tar.gz
    tar xzf cmake-2.8.12.2.tar.gz --strip-components=1
    ./configure --parallel=$NCPUS
    make -j$NCPUS
    make install
    popd
}

if [[ "$INCHROOT" == "install" ]]; then
    freeuid="$2"
    if ! test -e /.debs.done; then
        echo "deb $DISTURL $DISTNAME $DISTPARTS
deb $DISTURL $DISTNAME-updates   $DISTPARTS
deb $DISTURL $DISTNAME-security  $DISTPARTS
#deb $DISTURL $DISTNAME-backports $DISTPARTS
" > "/etc/apt/sources.list"
        apt-get -y install gpgv
        apt-get update
        apt-get -y dist-upgrade
        apt-get -y install openjdk-7-jdk || apt-get -y install openjdk-6-jdk
        apt-get -y install build-essential curl ccache # cmake
        touch /.debs.done
    fi
    install_cmake
    install_yasm
    install_maven
    useradd -u $freeuid -d /build build
    exit 0
fi

trap "release_chroot" EXIT

if [[ "$INCHROOT" == "enter" ]]; then
    chroot_do su - build -c "/build/${0##*/} enter_stage2"
fi

aptUpdated=""

aptinst() {
    local testbin="$1"
    local prog="$2"
    if [[ -z "$prog" ]]; then prog="$testbin"; fi
    if which $testbin; then return 0; fi
    if [[ -z "$aptUpdated" ]]; then
        sudo apt-get update
        aptUpdated=1
    fi        
    sudo apt-get install $prog
}

if ! test -e "$TGTDIR/.installed"; then
    if ! "$TGTDIR/.debs.done"; then
        sudo rm -rf "$TGTDIR"
        sudo mkdir -p "$TGTDIR"
    fi
    connect_chroot
    sudo ln -sf build/.cache "$TGTDIR/.cache"
    aptinst debootstrap
    sudo debootstrap --arch="$DISTARCH" --variant=minbase \
        --keyring="$DISTKEYRING" "$DISTNAME" "$TGTDIR" "$DISTURL"
    sudo cp "$0" "$TGTDIR/inchroot.sh"
    chroot_do chmod 755 "inchroot.sh"

    # for more security, select a uid that is not used inside the host system
    freeuid=11723; while grep $freeuid /etc/passwd; do let 'freeuid++'; done
    chroot_do "./inchroot.sh" install $freeuid
    chroot_do touch .installed
fi

sudo rsync -av -del --exclude=/.cache/ --exclude=/.git/ --exclude=/osinst.*/ "$BASEDIR/" "$TGTDIR/build"
chroot_do chown -R build build
chroot_do su - build -c "/build/${0##*/} build"

