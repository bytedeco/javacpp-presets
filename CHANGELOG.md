
 * Add cross-compilation support for `linux-ppc64le` ([pull #471](https://github.com/bytedeco/javacpp-presets/pull/471))
 * Fix definitions of `fftw_iodim` and `fftw_iodim64` for FFTW ([issue #466](https://github.com/bytedeco/javacpp-presets/issues/466))
 * Add a `Mat(Scalar)` constructor to OpenCV for convenience ([issue bytedeco/javacv#738](https://github.com/bytedeco/javacv/issues/738))
 * Include the `libavutil/time.h` for FFmpeg ([issue bytedeco/javacv#735](https://github.com/bytedeco/javacv/issues/735))
 * Map `AVERROR()` for error codes from `errno.h` used by FFmpeg ([pull #459](https://github.com/bytedeco/javacpp-presets/pull/459))
 * Use `long` instead of `int` for constants starting with `AV_CH_LAYOUT_` in `avutil` ([pull #455](https://github.com/bytedeco/javacpp-presets/pull/455))
 * Build FFmpeg with FreeType, enabling the `drawtext` filter ([pull #452](https://github.com/bytedeco/javacpp-presets/pull/452)), and support for Opus ([pull #457](https://github.com/bytedeco/javacpp-presets/pull/457))
 * Enable cuDNN support for Caffe by default ([issue #187](https://github.com/bytedeco/javacpp-presets/issues/187))
 * Use the new `BUILD_PATH` environment variable passed by the `javacpp` plugin to the `cppbuild.sh` script files
 * Let presets pick up `include` or `link` paths from cached resources ([pull #101](https://github.com/bytedeco/javacpp-presets/pull/101))
 * Update presets for Visual Studio 2015 ([issue #298](https://github.com/bytedeco/javacpp-presets/issues/298))
 * Reuse the presets for HDF5 and OpenBLAS with Caffe and MXNet
 * Replace the `exec-maven-plugin` with the `javacpp` plugin itself to execute the `cppbuild.sh` script files
 * Link TensorFlow statically with `cudart` to avoid dependency on CUDA ([issue #396](https://github.com/bytedeco/javacpp-presets/issues/396))
 * Add missing call to `Loader.load()` in helper class for `opencv_ml` ([issue bytedeco/javacv#638](https://github.com/bytedeco/javacv/issues/638))
 * Work around issues with TensorFlow on some versions of Mac OS X ([issue #335](https://github.com/bytedeco/javacpp-presets/issues/335))
 * Upgrade presets for OpenCV 3.3.0, FFmpeg 3.4, FlyCapture 2.11.3.121 ([pull #424](https://github.com/bytedeco/javacpp-presets/pull/424)), libdc1394 2.2.5, Chilitags, HDF5 1.10.1, OpenBLAS 0.2.20, FFTW 3.3.7, GSL 2.4, LLVM 4.0.0 ([pull #404](https://github.com/bytedeco/javacpp-presets/pull/404)), Leptonica 1.74.4, Tesseract 3.05.01, Caffe 1.0, CUDA 9.0, cuDNN 7.0, MXNet 0.12.0, TensorFlow 1.4.0, and their dependencies
 * Add presets for libfreenect2 ([pull #340](https://github.com/bytedeco/javacpp-presets/pull/340)), MKL 2018.0 ([issue #112](https://github.com/bytedeco/javacpp-presets/issues/112)), The Arcade Learning Environment, LiquidFun ([pull #356](https://github.com/bytedeco/javacpp-presets/pull/356)), and Skia ([pull #418](https://github.com/bytedeco/javacpp-presets/pull/418))
 * Fix the `FlyCapture2` module for some versions on Windows ([issue #337](https://github.com/bytedeco/javacpp-presets/issues/337))
 * Add functions missing from the presets of MXNet ([issue #332](https://github.com/bytedeco/javacpp-presets/issues/332))
 * Add presets for the `text` module of OpenCV 3.x ([pull #333](https://github.com/bytedeco/javacpp-presets/pull/333))

### December 7, 2016 version 1.3
 * Fix FFmpeg builds on ARM when not using a cross compiler ([issue #322](https://github.com/bytedeco/javacpp-presets/issues/322))
 * Add `blas_extra.h` to presets for OpenBLAS, containing `blas_set_num_threads()` and `blas_get_vendor()` functions
 * Introduce platform artifacts that depend on binaries for all available platforms and work with any build system (sbt, Gradle, M2Eclipse, etc)
 * Map more functions of the OpenCV Transparent API with `UMat` and `UMatVector` parameters ([issue bytedeco/javacv#518](https://github.com/bytedeco/javacv/issues/518))
 * Add support for `android-arm` and `android-x86` platforms to TensorFlow presets ([pull #297](https://github.com/bytedeco/javacpp-presets/pull/297))
 * Keep a reference of `tensorflow.SessionOptions` in `AbstractSession` to prevent premature deallocation ([pull #297](https://github.com/bytedeco/javacpp-presets/pull/297))
 * Enable CUDA in `cppbuild.sh` script for TensorFlow ([issue #294](https://github.com/bytedeco/javacpp-presets/issues/294))
 * Bundle `libgomp.so.1` in JAR files of OpenCV for the sake of some Linux distributions ([issue bytedeco/javacv#436](https://github.com/bytedeco/javacv/issues/436))
 * Fix `linux-armhf` and `linux-ppc64le` builds for all presets ([pull #279](https://github.com/bytedeco/javacpp-presets/pull/279))
 * Fix `libdc1394` not properly linking with `libusb-1.0` on Mac OS X ([issue bytedeco/javacv#501](https://github.com/bytedeco/javacv/issues/501))
 * Add presets for the `bioinspired` module of OpenCV 3.1 ([pull #282](https://github.com/bytedeco/javacpp-presets/pull/282))
 * Include `tensorflow/core/graph/dot.h` header file from TensorFlow ([pull #272](https://github.com/bytedeco/javacpp-presets/pull/272))
 * Add presets for librealsense, HDF5, and OpenBLAS/MKL ([issue #112](https://github.com/bytedeco/javacpp-presets/issues/112))
 * Make Caffe work on CPU-only machines ([issue #219](https://github.com/bytedeco/javacpp-presets/issues/219))
 * Fix loading issue with `opencv_face` ([issue bytedeco/javacv#470](https://github.com/bytedeco/javacv/issues/470))
 * Fix presets for CUDA on the `linux-ppc64le` platform
 * Upgrade presets for FFmpeg 3.2.1, FFTW 3.3.5, GSL 2.2.1, LLVM 3.9.0, CUDA 8.0, cuDNN 5.1, Caffe, MXNet, TensorFlow 0.11.0, and their dependencies
 * Set default options in `tensorflow/cppbuild.sh` to prevent console reads during build
 * Add `Tensor.createStringArray()` method to access `DT_STRING` data ([issue #249](https://github.com/bytedeco/javacpp-presets/issues/249))
 * Fix Javadoc links for externally referenced classes
 * Work around build issues for TensorFlow on some Linux distributions ([issue bazelbuild/bazel#1322](https://github.com/bazelbuild/bazel/issues/1322))
 * Fix `cppbuild.sh` script to build properly FFmpeg from source on Windows
 * Stop using Zeranoe FFmpeg builds, often not available for release versions ([issue #225](https://github.com/bytedeco/javacpp-presets/issues/225))
 * Add `linux-ppc64le` to `cppbuild.sh` scripts of OpenCV, FFmpeg, Leptonica, and Tesseract

### May 15, 2016 version 1.2
 * Build libdc1394 for the Windows platform as well ([issue bytedeco/procamcalib#4](https://github.com/bytedeco/procamcalib/issues/4))
 * Lower Maven prerequisite in the `pom.xml` file to 3.0 ([issue bytedeco/javacpp#93](https://github.com/bytedeco/javacpp/issues/93))
 * Include the `Descriptor` and `Message` APIs in the presets for Caffe ([issue #196](https://github.com/bytedeco/javacpp-presets/issues/196))
 * Prevent creating text relocations for shared libraries on Android ([issue bytedeco/javacv#245](https://github.com/bytedeco/javacv/issues/245))
 * Make sure to include only native libraries in platform specific JAR files ([pull bytedeco/javacpp#89](https://github.com/bytedeco/javacpp/pull/89))
 * Execute the `cppbuild.sh` scripts within the Maven build lifecycle, can be skipped with `-Djavacpp.cppbuild.skip` ([pull #175](https://github.com/bytedeco/javacpp-presets/pull/175))
 * Fix Caffe crashing in GPU mode: Do not define `CPU_ONLY` ([issue #147](https://github.com/bytedeco/javacpp-presets/issues/147))
 * Make OpenBLAS build for Caffe more generic ([issue #154](https://github.com/bytedeco/javacpp-presets/issues/154))
 * Include missing `graph_constructor.h` header file from the `tensorflow` module ([issue #165](https://github.com/bytedeco/javacpp-presets/issues/165))
 * Add missing `GraphDefBuilder.Options.WithAttr()` methods from the `tensorflow` module ([issue #160](https://github.com/bytedeco/javacpp-presets/issues/160))
 * Add `linux-armhf` platform to the `cppbuild.sh` scripts of OpenCV, FFmpeg, etc ([pull #177](https://github.com/bytedeco/javacpp-presets/pull/177))
 * Add support for Motion JPEG to the minimal configuration proposed for FFmpeg in the `cppbuild.sh` file
 * Make `mvn -Djavacpp.platform=...` and `mvn -Djavacpp.platform.dependency=...` commands work correctly
 * Add presets for the `dnn` module of OpenCV 3.1 ([issue #145](https://github.com/bytedeco/javacpp-presets/issues/145))
 * Prepend "javacpp." to all properties associated with Maven in the `pom.xml` files to avoid name clashes
 * Add a `Mat(CvArr arr)` constructor for convenience ([issue bytedeco/javacv#317](https://github.com/bytedeco/javacv/issues/317))
 * Fix loading issue with `opencv_stitching` and `opencv_xfeatures2d` ([issue bytedeco/javacv#316](https://github.com/bytedeco/javacv/issues/316), [issue bytedeco/javacv#336](https://github.com/bytedeco/javacv/issues/336))
 * Virtualize all `Solver` classes from Caffe ([issue #143](https://github.com/bytedeco/javacpp-presets/issues/143))
 * Work around GSL not loading on Android ([issue bytedeco/javacpp#55](https://github.com/bytedeco/javacpp/issues/55))
 * Fix Android builds of FFmpeg, FFTW, GSL, Leptonica, and Tesseract causing errors under Mac OS X ([issue #45](https://github.com/bytedeco/javacpp-presets/issues/45))
 * Avoid versioning of FFTW and GSL libraries, preventing them from working on Android ([issue #127](https://github.com/bytedeco/javacpp-presets/issues/127))
 * Upgrade presets for OpenCV 3.1.0, FFmpeg 3.0.2, OpenSSL 1.0.2h, x265 1.9, FlyCapture 2.9.3.43, libdc1394 2.2.4, videoInput, GSL 2.1, LLVM 3.8.0, Leptonica 1.73, Tesseract 3.04.01, cuDNN 5, and Caffe rc3, including the latest versions of their dependencies ([issue bytedeco/javacpp#55](https://github.com/bytedeco/javacpp/issues/55))
 * Add presets for MXNet and TensorFlow 0.8.0 ([issue #111](https://github.com/bytedeco/javacpp-presets/issues/111))
 * Virtualize `opencv_videostab.IFrameSource` to let us implement it in Java ([issue bytedeco/javacv#277](https://github.com/bytedeco/javacv/issues/277))
 * Fix MinGW-w64 builds with recent versions of GCC 5.x and potential issue with using "w32threads" for FFmpeg
 * Add missing `StatModel.loadXXX()` methods in the `opencv_ml` module ([issue #109](https://github.com/bytedeco/javacpp-presets/issues/109))
 * Define commonly used Caffe `std::vector` types (`DatumVector`, `FloatCallbackVector`, and `DoubleCallbackVector`) for ease of use and performance reasons
 * Fix the `cppbuild.sh` script for FFmpeg, failing to build x264 and OpenH264 properly on Windows

### October 25, 2015 version 1.1
 * Build the Maven artifacts for Linux in a CentOS 6 Docker container, for maximum compatibility ([issue #22](https://github.com/bytedeco/javacpp-presets/issues/22))
 * Cache files downloaded by `cppbuild.sh` in the `downloads` subdirectory to prevent having to redownload everything after a clean
 * Add the `clang` module to the presets for LLVM
 * Propose for FFmpeg in the `cppbuild.sh` file a minimal configuration to support MPEG-4 streams with H.264 and AAC
 * Add the non-GPL OpenH264 as an alternative H.264 encoder to x264 in FFmpeg
 * Hack together `log_callback.h` to be able to redirect to Java log messages from FFmpeg
 * Pick up `OLDCC`, `OLDCXX`, and `OLDFC` environment variables in `cppbuild.sh` and `platform.oldcompiler` system property in Maven to build with it libraries that can tolerate an older version of the C/C++ compiler on Linux
 * Upgrade all Maven dependencies and plugins to latest versions, thus bumping minimum requirements to Java SE 7, Android 4.0, and Maven 3.0
 * Provide `cppbuild.sh` script for Caffe that includes all dependencies ([pull #77](https://github.com/bytedeco/javacpp-presets/pull/77))
 * Upgrade presets for Caffe, CUDA 7.5, cuDNN 3, FFmpeg 2.8.1, Speex 1.2rc2, x265 1.8, FlyCapture 2.8.3.1, libfreenect 0.5.3, LLVM 3.7.0, Tesseract 3.04
 * Include `motion_vector.h`, `fifo.h`, and `audio_fifo.h` header files in the `avutil` module of FFmpeg ([issue #79](https://github.com/bytedeco/javacpp-presets/issues/79))
 * Add presets for Chilitags, thanks to Chris Walters for the financial contribution
 * Let users choose the runtime type of `layer_by_name()` from `FloatNet` or `DoubleNet` in `caffe` ([issue bytedeco/javacpp#25](https://github.com/bytedeco/javacpp/issues/25))
 * Add presets for the `face`, `optflow`, and `xfeatures2d` modules of OpenCV 3.0 ([issue bytedeco/javacv#196](https://github.com/bytedeco/javacv/issues/196), [issue bytedeco/javacv#239](https://github.com/bytedeco/javacv/issues/239), [issue #54](https://github.com/bytedeco/javacpp-presets/issues/54))
 * Switch to GCC 4.9 by default on Android, probably dropping support for Android 2.2, because GCC 4.6 has issues with OpenMP ([issue bytedeco/javacv#179](https://github.com/bytedeco/javacv/issues/179))
 * Resolve missing dependency for GSL on `windows-x86` by linking statically whatever it needs from `libgcc_s_dw2-1.dll`

### July 11, 2015 version 1.0
 * Build OpenCV, GSL, Leptonica, and Tesseract from source code on all platforms for better consistency in functionality across platforms
 * Add presets for CUDA 7.0 (including cuBLAS, cuDNN, cuFFT, cuRAND, cuSOLVER, cuSPARSE, and NPP)
 * Offer the Apache License, Version 2.0, as a new choice of license, in addition to the GPLv2 with Classpath exception
 * Add libvpx in the `cppbuild.sh` script for FFmpeg to support the WebM format ([issue #33](https://github.com/bytedeco/javacpp-presets/issues/33))
 * Upgrade presets for OpenCV 3.0.0, FFmpeg 2.7.1, OpenSSL 1.0.2d, x265 1.7, Leptonica 1.72, LLVM 3.6.1, and the latest of Caffe
 * Define commonly used OpenCV `std::vector` types (`PointVector`, `Point2fVector`, `Point2dVector`, `SizeVector`, `RectVector`, `KeyPointVector`, `DMatchVector`) for ease of use and performance reasons
 * Map `cv::saturate_cast<>()` in a more meaningful way ([issue #53](https://github.com/bytedeco/javacpp-presets/issues/53)) and name these functions more consistently
 * In addition to Leptonica and Tesseract, use only the officially supported GCC compiler for FFmpeg, FFTW, and GSL under Windows as well, to prevent compatibility issues ([issue bytedeco/javacv#137](https://github.com/bytedeco/javacv/issues/137))
 * Make `flycapture/cppbuild.sh` fail if FlyCapture is not found installed on the system ([issue #46](https://github.com/bytedeco/javacpp-presets/issues/46))
 * Patch libdc1394, libdcfreenect, FFTW, GSL, Leptonica and Tesseract with missing `@rpath` needed by Mac OS X ([issue #46](https://github.com/bytedeco/javacpp-presets/issues/46))

### April 4, 2015 version 0.11
 * Remove unneeded `@Opaque` types from `gsl` and replace them with their definitions whose names end with "_struct"
 * Segregate methods using `java.awt` classes into the new `Java2DFrameConverter` class of JavaCV ([issue #12](https://github.com/bytedeco/javacpp-presets/issues/12))
 * Emphasize the need to install separately the parent `pom.xml` file ([issue #42](https://github.com/bytedeco/javacpp-presets/issues/42))
 * Make CMake configurable via `CMAKE` variable in `cppbuild.sh` ([pull #41](https://github.com/bytedeco/javacpp-presets/pull/41))
 * Add presets for Caffe ([issue #34](https://github.com/bytedeco/javacpp-presets/issues/34))
 * Let `createBuffer()` return `UByteIndexer` and `UShortIndexer` when appropriate for unsigned data types
 * Remove the need to set manually the `platform.dependency` system property for downstream modules without a `cppbuild.sh` file
 * Fix failing `cppbuild.sh` for FFmpeg on Ubuntu ([issue #32](https://github.com/bytedeco/javacpp-presets/issues/32))
 * Disable iconv, XCB, and SDL for more portables builds of FFmpeg, but enable `x11grab` and `avfoundation` to allow screen capture ([issue #39](https://github.com/bytedeco/javacpp-presets/issues/39))
 * Avoid versioning of Leptonica and Tesseract libraries, preventing them from working on Android ([issue #38](https://github.com/bytedeco/javacpp-presets/issues/38))
 * Add x265 in the `cppbuild.sh` script for FFmpeg, thanks to Mark Bolstad ([issue bytedeco/javacv#41](https://github.com/bytedeco/javacv/issues/41))
 * Upgrade presets for OpenCV 2.4.11, FFmpeg 2.6.1, OpenSSL 1.0.2a, FlyCapture 2.7.3.19, libdc1394 2.2.3, libfreenect 0.5.2, LLVM 3.6.0
 * Switch from `IntPointer` to `BoolPointer` for the `BOOL*` pointer type of Leptonica ([issue #36](https://github.com/bytedeco/javacpp-presets/issues/36))
 * Add `preload` for `gif`, `jpeg`, `png`, `tiff`, and `webp` libraries in presets for Leptonica ([issue #36](https://github.com/bytedeco/javacpp-presets/issues/36))
 * Include missing `ltrresultiterator.h` header file in the presets for Tesseract ([issue #36](https://github.com/bytedeco/javacpp-presets/issues/36))
 * Append `@Documented` to annotation types to have them picked up by Javadoc

### December 23, 2014 version 0.10
 * Update instructions in the `README.md` file for manual installation in Android Studio
 * Include presets for Leptonica 1.71 and Tesseract 3.03-rc1 on Windows too
 * Fix `Mat.createFrom(BufferedImage)` ([issue #30](https://github.com/bytedeco/javacpp-presets/issues/30))
 * Add Speex, OpenCORE (AMR-NB and AMR-WB), and OpenSSL in the `cppbuild.sh` script for FFmpeg to support common RTMPS streams, among other things ([issue #2](https://github.com/bytedeco/javacpp-presets/issues/2) and [issue bytedeco/javacv#71](https://github.com/bytedeco/javacv/issues/71))
 * Deprecate slow `get()` and `put()` methods of `CvMat` in favor of the fast ones from `createIndexer()` ([issue javacv:317](http://code.google.com/p/javacv/issues/detail?id=317))
 * Include `operations.hpp` and `mat.hpp` in `opencv_core` to get a few important functions such as `read()` and `write()` for `FileStorage`
 * Replace `install_name_tool` hack to set `@rpath` on Mac OS X with patches to do it properly on install ([issue bytedeco/javacpp#6](https://github.com/bytedeco/javacpp/issues/6) and [issue bytedeco/javacv#49](https://github.com/bytedeco/javacv/issues/49))
 * Disable DocLint, which prevents the build from succeeding on Java 8 ([issue bytedeco/javacpp#5](https://github.com/bytedeco/javacpp/issues/5))
 * Disable OpenCL detection when building FFmpeg, causing link failures ([issue #19](https://github.com/bytedeco/javacpp-presets/issues/19))
 * Document a bit the `create()` factory methods in the `helper` package of the OpenCV module, and their relationship with the `release()` methods
 * Include new `createIndexer()` method in `CvMat`, `IplImage`, `Mat`, etc. for easy and efficient multidimensional access of data ([issue #317](http://code.google.com/p/javacv/issues/detail?id=317))
 * Deprecate `get*Buffer()` methods in favor of a better named and generic `createBuffer()` method
 * Fix `java.lang.UnsatisfiedLinkError` when allocating `opencv_core.Mat`, among others ([issue bytedeco/javacv#9](https://github.com/bytedeco/javacv/issues/9) and [issue bytedeco/javacv#28](https://github.com/bytedeco/javacv/issues/28))
 * Force OpenCV to build with GCC 4.6, as newer versions are known to hang on Android 2.2 ([issue android:43819](https://code.google.com/p/android/issues/detail?id=43819))
 * Upgrade presets for OpenCV 2.4.10, FFmpeg 2.5.1, FlyCapture 2.7.3.13, libfreenect 0.5.1, ARToolKitPlus 2.3.1, LLVM 3.5.0, and videoInput 0.200, where the latest code got merged into the master branch
 * Add callbacks for Tesseract according to new functionality in JavaCPP
 * Fix missing dependency of `opencv_contrib` on `opencv_nonfree` ([issue javacv:490](https://code.google.com/p/javacv/issues/detail?id=490))
 * Skip functions that are not actually implemented in `avdevice`, causing load failures on Android
 * Update presets for FFmpeg where `avcodec` now inherits from `swresample` ([issue #13](https://github.com/bytedeco/javacpp-presets/issues/13))
 * Add a `README.md` file to each presets with links to original project, Java API documentation, and sample usage
 * Add missing overloaded methods with `PointerPointer` parameters in LLVM module

### July 27, 2014 version 0.9
 * Add libmp3lame to FFmpeg builds ([issue javacv:411](https://code.google.com/p/javacv/issues/detail?id=448))
 * Upgrade presets for FFmpeg 2.3, FlyCapture 2.6.3.4 ([pull #6](https://github.com/bytedeco/javacpp-presets/pull/6), [issue #8](https://github.com/bytedeco/javacpp-presets/issues/8)), libfreenect 0.5
 * Make the `cppbuild.sh` scripts install native libraries inside the `cppbuild` subdirectories, instead of on the system
 * Include new `platform.dependency` and `platform.dependencies` properties to let users depend easily on the artifacts that contain native libraries
 * Add presets for flandmark 1.07 ([pull #9](https://github.com/bytedeco/javacpp-presets/pull/9)), FFTW 3.3.4, GSL 1.16, LLVM 3.4.2, Leptonica 1.71, Tesseract 3.03-rc1
 * Fix missing `static` keyword on methods annotated with an `@Adapter` ([issue #3](https://github.com/bytedeco/javacpp-presets/issues/3))
 * Turn `Mat.createFrom()` into a static factory method, and make `Mat.copyFrom()` call `Mat.create()` as appropriate ([issue #1](https://github.com/bytedeco/javacpp-presets/issues/1))
 * Add missing `native_camera` modules of `opencv_highgui` for Android
 * Fix functions from `opencv_stitching` not accepting a `MatVector` as apparently intended by the API (issue javacv:466)

### April 28, 2014 version 0.8
 * Move from Google Code to GitHub as main source code repository
 * Rename the `com.googlecode.javacpp` package to `org.bytedeco.javacpp`
 * Appended the version of the parent artifact to the ones of the child modules, in an effort to avoid conflicts
 * Updated `cppbuild.sh` scripts with support for the "android-x86" platform (issue javacv:411), thanks to Xavier Hallade
 * Added presets for PGR FlyCapture 1.7
 * Fixed compilation errors on Android, Mac OS X, and Windows
 * Upgraded to OpenCV 2.4.9, FFmpeg 2.2.1, libdc1394 2.2.2, and libfreenect 0.4
 * Introduced build scripts, based on the CPPJARs package of JavaCV, to install native C/C++ libraries
 * Ported various helper classes and methods from JavaCV
 * Inserted missing dependency entries in the `pom.xml` files of the child modules
 * Added presets for the C++ API of OpenCV 2.4.8, which can now be parsed due to the latest changes in JavaCPP

### January 6, 2014 version 0.7
 * Fixed JavaCPP properties not getting set by the parent `pom.xml` file
 * Added presets for the C API of OpenCV 2.4.8, FFmpeg 2.1.x, libfreenect 0.2 (OpenKinect), videoInput 0.200, and ARToolkitPlus 2.3.0

### September 15, 2013 version 0.6
Initial release


Acknowledgments
---------------
This project was conceived at the [Okutomi & Tanaka Laboratory](http://www.ok.ctrl.titech.ac.jp/), Tokyo Institute of Technology, where I was supported for my doctoral research program by a generous scholarship from the Ministry of Education, Culture, Sports, Science and Technology (MEXT) of the Japanese Government. I extend my gratitude further to all who have reported bugs, donated code, or made suggestions for improvements (details above)!
