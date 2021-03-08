#!/bin/bash
# This file is meant to be included by the parent cppbuild.sh script
if [[ -z "$PLATFORM" ]]; then
    pushd ..
    bash cppbuild.sh "$@" arpack-ng
    popd
    exit
fi

ARPACK_NG_VERSION=3.8.0
download https://github.com/opencollab/arpack-ng/archive/$ARPACK_NG_VERSION.tar.gz arpack-ng-$ARPACK_NG_VERSION.tar.gz

mkdir -p $PLATFORM
cd $PLATFORM
INSTALL_PATH=`pwd`
echo "Decompressing archives..."
tar --totals -xzf ../arpack-ng-$ARPACK_NG_VERSION.tar.gz

cd arpack-ng-$ARPACK_NG_VERSION
if [[ "${ACLOCAL_PATH:-}" == C:\\msys64\\* ]]; then
    export ACLOCAL_PATH=/mingw64/share/aclocal:/usr/share/aclocal
fi
patch -Np1 < ../../../arpack-ng-configure.patch || true # bash bootstrap
chmod 755 configure build-aux/install-sh
sedinplace 's/std::real(sigma) + std::imag(sigma) \* I/*reinterpret_cast<_Complex double*>(\&sigma)/g' ICB/arpack.hpp
sedinplace 's/std::real(sigma) + _Complex_I \* std::imag(sigma)/*reinterpret_cast<_Complex double*>(\&sigma)/g' ICB/arpack.hpp
sedinplace 's/internal::s/s/g' ICB/arpack.hpp
sedinplace 's/internal::d/d/g' ICB/arpack.hpp
sedinplace 's/internal::cn/cn/g' ICB/arpack.hpp
sedinplace 's/internal::z/z/g' ICB/arpack.hpp
cp SRC/arpack.pc.in .

OPENBLAS_PATH="$INSTALL_PATH/../../../openblas/cppbuild/$PLATFORM/"

if [[ -n "${BUILD_PATH:-}" ]]; then
    PREVIFS="$IFS"
    IFS="$BUILD_PATH_SEPARATOR"
    for P in $BUILD_PATH; do
        if [[ -f "$P/include/openblas_config.h" ]]; then
            OPENBLAS_PATH="$P"
        fi
    done
    IFS="$PREVIFS"
fi

OPENBLAS_PATH="${OPENBLAS_PATH//\\//}"

export LDFLAGS="-L$OPENBLAS_PATH/ -L$OPENBLAS_PATH/lib/"
export LD_LIBRARY_PATH="$OPENBLAS_PATH/:$OPENBLAS_PATH/lib/"

case $PLATFORM in
    linux-x86)
        LIBS=
        if echo "int main() { }" | gcc -x c - -lgfortran_nonshared; then
            LIBS="-lgfortran_nonshared"
        fi
        CC="gcc -m32" CXX="g++ -m32" FC="gfortran -m32 $LIBS" F77="$FC" FLIBS="-lgfortran" ./configure --prefix=$INSTALL_PATH --enable-icb --with-blas=openblas --with-lapack=openblas
        make -j $MAKEJ
        make install-strip
        ;;
    linux-x86_64)
        LIBS=
        if echo "int main() { }" | gcc -x c - -lgfortran_nonshared; then
            LIBS="-lgfortran_nonshared"
        fi
        CC="gcc -m64" CXX="g++ -m64" FC="gfortran -m64 $LIBS" F77="$FC" FLIBS="-lgfortran" ./configure --prefix=$INSTALL_PATH --enable-icb --with-blas=openblas --with-lapack=openblas
        make -j $MAKEJ
        make install-strip
        ;;
    linux-armhf)
        CC="arm-linux-gnueabihf-gcc" CXX="arm-linux-gnueabihf-g++" FC="arm-linux-gnueabihf-gfortran" F77="$FC" FLIBS="-lgfortran" ./configure --prefix=$INSTALL_PATH --enable-icb --with-blas=openblas --with-lapack=openblas --host=arm-linux-gnueabihf
        make -j $MAKEJ
        make install-strip
        ;;
    linux-arm64)
        CC="aarch64-linux-gnu-gcc" CXX="aarch64-linux-gnu-g++" FC="aarch64-linux-gnu-gfortran" F77="$FC" FLIBS="-lgfortran" ./configure --prefix=$INSTALL_PATH --enable-icb --with-blas=openblas --with-lapack=openblas --host=aarch64-linux-gnu
        make -j $MAKEJ
        make install-strip
        ;;
    linux-ppc64le)
        sed -i s/elf64ppc/elf64lppc/ configure
        CC="powerpc64le-linux-gnu-gcc -m64" CXX="powerpc64le-linux-gnu-g++ -m64" FC="powerpc64le-linux-gnu-gfortran -m64" F77="$FC" FLIBS="-lgfortran" ./configure --prefix=$INSTALL_PATH --enable-icb --with-blas=openblas --with-lapack=openblas --host=powerpc64le-linux-gnu --build=ppc64le-linux
        make -j $MAKEJ
        make install-strip
        ;;
    macosx-*)
        sed -i="" 's/install_name \\$rpath/install_name @rpath/g' configure m4/libtool.m4
        export CC="$(ls -1 /usr/local/bin/gcc-? | head -n 1)"
        export CXX="$(ls -1 /usr/local/bin/g++-? | head -n 1)"
        export FC="$(ls -1 /usr/local/bin/gfortran-? | head -n 1) -nodefaultlibs -lSystem -static-libgcc -static-libgfortran -lgcc -lgfortran $(ls -1 /usr/local/lib/gcc/?/libquadmath.a | head -n 1)"
        export F77="$FC"
        export FLIBS="-lgfortran"
        ./configure --prefix=$INSTALL_PATH --enable-icb --with-blas=openblas --with-lapack=openblas
        make -j $MAKEJ
        make install-strip
        ;;
    windows-x86)
        CC="gcc -m32" CXX="g++ -m32" FC="gfortran -fallow-argument-mismatch -m32 -static-libgcc -static-libgfortran -Wl,-Bstatic,--whole-archive,--allow-multiple-definition -lwinpthread -lquadmath -lgfortran -Wl,-Bdynamic,--no-whole-archive" F77="$FC" FLIBS="-lgfortran" ./configure --prefix=$INSTALL_PATH --enable-icb --with-blas=openblas --with-lapack=openblas --build=i686-w64-mingw32
        make -j $MAKEJ
        make install-strip
        ;;
    windows-x86_64)
        CC="gcc -m64" CXX="g++ -m64" FC="gfortran -fallow-argument-mismatch -m64 -static-libgcc -static-libgfortran -Wl,-Bstatic,--whole-archive,--allow-multiple-definition -lwinpthread -lquadmath -lgfortran -Wl,-Bdynamic,--no-whole-archive" F77="$FC" FLIBS="-lgfortran" ./configure --prefix=$INSTALL_PATH --enable-icb --with-blas=openblas --with-lapack=openblas --build=x86_64-w64-mingw32
        make -j $MAKEJ
        make install-strip
        ;;
    *)
        echo "Error: Platform \"$PLATFORM\" is not supported"
        ;;
esac
unset CC
unset CXX
unset FC
unset F77
unset LDFLAGS
unset LD_LIBRARY_PATH

cd ../..
