JavaCPP Presets
===============

[![Join the chat at https://gitter.im/bytedeco/javacpp](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/bytedeco/javacpp?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge) [![Maven Central](https://maven-badges.herokuapp.com/maven-central/org.bytedeco/javacpp-presets/badge.svg)](https://maven-badges.herokuapp.com/maven-central/org.bytedeco/javacpp-presets) [![Sonatype Nexus (Snapshots)](https://img.shields.io/nexus/s/https/oss.sonatype.org/org.bytedeco/javacpp-presets.svg)](http://bytedeco.org/builds/) <sup>Android, iOS, Linux, Mac OS X:</sup> [![Travis CI](https://travis-ci.org/bytedeco/javacpp-presets.svg?branch=master)](https://travis-ci.org/bytedeco/javacpp-presets) <sup>Windows:</sup> [![AppVeyor](https://ci.appveyor.com/api/projects/status/github/bytedeco/javacpp-presets?branch=master&svg=true)](https://ci.appveyor.com/project/bytedeco/javacpp-presets)


Introduction
------------
The JavaCPP Presets module contains Java configuration and interface classes for widely used C/C++ libraries. The configuration files in the `org.bytedeco.javacpp.presets` package are used by the `Parser` to create from C/C++ header files the Java interface files targeting the `org.bytedeco.javacpp` package, which is turn are used by the `Generator` and the native C++ compiler to produce the required JNI libraries. Moreover, helper classes make their functionality easier to use on the Java platform, including Android.

Please refer to the wiki page for more information about how to [create new presets](https://github.com/bytedeco/javacpp-presets/wiki/Create-New-Presets). Since additional documentation is currently lacking, please also feel free to ask questions on [the mailing list](http://groups.google.com/group/javacpp-project).


Downloads
---------
JAR files containing binaries for all child modules and builds for all supported platforms (Android, iOS, Linux, Mac OS X, and Windows) can be obtained from the [Maven Central Repository](http://search.maven.org/#search|ga|1|bytedeco).

To install manually the JAR files, follow the instructions in the [Manual Installation](#manual-installation) section below.

We can also have everything downloaded and installed automatically with:

 * Maven (inside the `pom.xml` file)
```xml
  <dependency>
    <groupId>org.bytedeco.javacpp-presets</groupId>
    <artifactId>${moduleName}-platform</artifactId>
    <version>${moduleVersion}-1.4.4</version>
  </dependency>
```

 * Gradle (inside the `build.gradle` file)
```groovy
  dependencies {
    compile group: 'org.bytedeco.javacpp-presets', name: moduleName + '-platform', version: moduleVersion + '-1.4.4'
  }
```

 * Leiningen (inside the `project.clj` file)
```clojure
  :dependencies [
    [~(symbol (str "org.bytedeco.javacpp-presets/" moduleName "-platform")) ~(str moduleVersion "-1.4.4")]
  ]
```

 * sbt (inside the `build.sbt` file)
```scala
  libraryDependencies += "org.bytedeco.javacpp-presets" % moduleName + "-platform" % moduleVersion + "-1.4.4"
```

where the `moduleName` and `moduleVersion` variables correspond to the desired module. This downloads binaries for all platforms, but to get binaries for only one platform we can set the `javacpp.platform` system property (via the `-D` command line option) to something like `android-arm`, `linux-x86_64`, `macosx-x86_64`, `windows-x86_64`, etc. Another option available for Scala users is [sbt-javacpp](https://github.com/bytedeco/sbt-javacpp).


Required Software
-----------------
To use the JavaCPP Presets, you will need to download and install the following software:

 * An implementation of Java SE 7 or newer:
   * OpenJDK  http://openjdk.java.net/install/  or
   * Oracle JDK  http://www.oracle.com/technetwork/java/javase/downloads/  or
   * IBM JDK  http://www.ibm.com/developerworks/java/jdk/

Further, in the case of Android, the JavaCPP Presets also rely on:

 * Android SDK API 14 or newer  http://developer.android.com/sdk/


Manual Installation
-------------------
Simply put all the desired JAR files (`opencv*.jar`, `ffmpeg*.jar`, etc.), in addition to `javacpp.jar`, somewhere in your class path. The JAR files available as pre-built artifacts are meant to be used with [JavaCPP](https://github.com/bytedeco/javacpp). The binaries for Linux were built for CentOS 6 and 7, so they should work on most distributions currently in use. The ones for Android were compiled for ARMv7 processors featuring an FPU, so they will not work on ancient devices such as the HTC Magic or some others with an ARMv6 CPU. Here are some more specific instructions for common cases:

NetBeans (Java SE 7 or newer):

 1. In the Projects window, right-click the Libraries node of your project, and select "Add JAR/Folder...".
 2. Locate the JAR files, select them, and click OK.

Eclipse (Java SE 7 or newer):

 1. Navigate to Project > Properties > Java Build Path > Libraries and click "Add External JARs...".
 2. Locate the JAR files, select them, and click OK.

IntelliJ IDEA (Android 4.0 or newer):

 1. Follow the instructions on this page: http://developer.android.com/training/basics/firstapp/
 2. Copy all the JAR files into the `app/libs` subdirectory.
 3. Navigate to File > Project Structure > app > Dependencies, click `+`, and select "2 File dependency".
 4. Select all the JAR files from the `libs` subdirectory.

After that, we can access almost transparently the corresponding C/C++ APIs through the interface classes found in the `org.bytedeco.javacpp` package. Indeed, the `Parser` translates the code comments from the C/C++ header files into the Java interface files, (almost) ready to be consumed by Javadoc. However, since their translation still leaves to be desired, one may wish to refer to the original documentation pages. For instance, the ones for OpenCV and FFmpeg can be found online at:

 * [OpenCV documentation](http://docs.opencv.org/master/)
 * [FFmpeg documentation](http://ffmpeg.org/doxygen/trunk/)


Build Instructions
------------------
If the binary files available above are not enough for your needs, you might need to rebuild them from the source code. To this end, project files on the Java side were created as [Maven modules](#the-maven-modules). By default, the Maven build also installs the native libraries on the native C/C++ side with the [`cppbuild.sh` scripts](#the-cppbuildsh-scripts), but they can also be installed by other means.

Additionally, one can find on the wiki page additional information about the recommended [build environments](https://github.com/bytedeco/javacpp-presets/wiki/Build-Environments) for the major platforms.


### The Maven modules
The JavaCPP Presets depend on Maven, a powerful build system for Java, so before attempting a build, be sure to install and read up on:

 * Maven 3.x  http://maven.apache.org/download.html
 * JavaCPP 1.4.4  https://github.com/bytedeco/javacpp

Each child module in turn relies by default on the included [`cppbuild.sh` scripts](#the-cppbuildsh-scripts), explained below, to install its corresponding native libraries in the `cppbuild` subdirectory. To use native libraries already installed somewhere else on the system, other installation directories than `cppbuild` can also be specified either in the `pom.xml` files or in the `.java` configuration files. The following versions are supported:

 * OpenCV 4.0.1  https://opencv.org/releases.html
 * FFmpeg 4.1.x  http://ffmpeg.org/download.html
 * FlyCapture 2.13.x  http://www.ptgrey.com/flycapture-sdk
 * Spinnaker 1.19.x https://www.ptgrey.com/spinnaker-sdk
 * libdc1394 2.1.x or 2.2.x  http://sourceforge.net/projects/libdc1394/files/
 * libfreenect 0.5.3  https://github.com/OpenKinect/libfreenect
 * libfreenect2 0.2.0  https://github.com/OpenKinect/libfreenect2
 * librealsense 1.12.1  https://github.com/IntelRealSense/librealsense
 * videoInput 0.200  https://github.com/ofTheo/videoInput/
 * ARToolKitPlus 2.3.1  https://launchpad.net/artoolkitplus
 * Chilitags  https://github.com/chili-epfl/chilitags
 * flandmark 1.07  http://cmp.felk.cvut.cz/~uricamic/flandmark/#download
 * HDF5 1.10.4  https://www.hdfgroup.org/downloads/
 * MKL 2019.1  https://software.intel.com/intel-mkl
 * MKL-DNN 0.17.2  https://github.com/intel/mkl-dnn
 * OpenBLAS 0.3.5  http://www.openblas.net/
 * ARPACK-NG 3.6.3  https://github.com/opencollab/arpack-ng
 * CMINPACK 1.3.6  https://github.com/devernay/cminpack
 * FFTW 3.3.8  http://www.fftw.org/download.html
 * GSL 2.5  http://www.gnu.org/software/gsl/#downloading
 * CPython 3.6.x  https://www.python.org/downloads/
 * LLVM 7.0.1  http://llvm.org/releases/download.html
 * libpostal 1.1-alpha  https://github.com/openvenues/libpostal
 * Leptonica 1.77.0  http://www.leptonica.org/download.html
 * Tesseract 4.0.0  https://github.com/tesseract-ocr/tesseract
 * Caffe 1.0  https://github.com/BVLC/caffe
 * CUDA 10.0  https://developer.nvidia.com/cuda-downloads
   * cuDNN 7.4  https://developer.nvidia.com/cudnn
 * MXNet 1.4.0.rc0  https://github.com/dmlc/mxnet
 * TensorFlow 1.12.0  https://github.com/tensorflow/tensorflow
 * TensorRT 5.0  https://developer.nvidia.com/tensorrt
 * The Arcade Learning Environment 0.6.0  https://github.com/mgbellemare/Arcade-Learning-Environment
 * ONNX 1.3.0  https://github.com/onnx/onnx
 * LiquidFun  http://google.github.io/liquidfun/
 * Skia  https://skia.org
 * System APIs of the build environments:
   * Linux (glibc)  https://www.gnu.org/software/libc/
   * Mac OS X (XNU libc)  https://opensource.apple.com/
   * Windows (Win32)  https://developer.microsoft.com/en-us/windows/

Once everything installed and configured, simply execute
```bash
$ mvn install --projects .,opencv,ffmpeg,flycapture,libdc1394,libfreenect,videoinput,artoolkitplus,etc.
```
inside the directory containing the parent `pom.xml` file, by specifying only the desired child modules in the command, but **without the leading period "." in the comma-separated list of projects, the parent `poml.xml` file itself might not get installed.** Also specify `-Djavacpp.cppbuild.skip` as option to skip the execution of the `cppbuild.sh` scripts. In addition to `-Djavacpp.platform=...`, some of the presets can also be built against CUDA with `-Djavacpp.platform.extension=-gpu`. Please refer to the comments inside the `pom.xml` file for further details. From the "platform" subdirectory, we can also install the "platform" artifacts with a similar command:

```bash
$ cd platform
$ mvn install --projects ../opencv/platform,../ffmpeg/platform,etc. -Djavacpp.platform.host
```


### The `cppbuild.sh` scripts
Running the scripts allows us to install easily the native libraries on multiple platforms, but additional software is required:

 * A recent version of Linux, Mac OS X, or Windows with MSYS and Visual Studio
 * Android NDK r7 or newer  http://developer.android.com/ndk/downloads/  (required only for Android builds)

With the above in working order, the scripts get launched automatically as part of the Maven build lifecycle, but we can also manually execute
```bash
$ ANDROID_NDK=/path/to/android-ndk/ bash cppbuild.sh [-platform <name>] [-extension <name>] <install | clean> [projects]
```
where possible platform names are: `android-arm`, `android-x86`, `linux-x86`, `linux-x86_64`, `linux-armhf`, `linux-ppc64le`, `linux-mips64el`, `macosx-x86_64`, `windows-x86`, `windows-x86_64`, etc. The only extension currently supported by some builds is `-gpu`, requiring CUDA to be installed. (The `ANDROID_NDK` variable is required only for Android builds.) Please note that the scripts download source archives from appropriate sites as necessary.

To compile binaries for an Android device with no FPU, first make sure this is what you want. Without FPU, the performance of either OpenCV or FFmpeg is bound to be unacceptable. If you still wish to continue down that road, then replace "armeabi-v7a" by "armeabi" and "-march=armv7-a -mfloat-abi=softfp -mfpu=vfpv3-d16" with "-march=armv5te -mtune=xscale -msoft-float", inside various files.

Although JavaCPP can pick up native libraries installed on the system, the scripts exist to facilitate the build process across multiple platforms. They also allow JavaCPP to copy the native libraries and load them at runtime from the JAR files created above by Maven, a useful feature for standalone applications or Java applets. Moreover, tricks such as the following work with JNLP:
```xml
    <resources os="Linux" arch="x86 i386 i486 i586 i686">
        <jar href="lib/opencv-linux-x86.jar"/>
        <jar href="lib/ffmpeg-linux-x86.jar"/>
    </resources>
    <resources os="Linux" arch="x86_64 amd64">
        <jar href="lib/opencv-linux-x86_64.jar"/>
        <jar href="lib/ffmpeg-linux-x86_64.jar"/>
    </resources>
```

Thanks to Jose Gómez for testing this out!


How Can I Help?
---------------
Contributions of any kind are highly welcome! At the moment, the `Parser` has limited capabilities, so I plan to improve it gradually to the point where it can successfully parse large C++ header files that are even more convoluted than the ones from OpenCV, Caffe, or TensorFlow, but the build system could also be improved. Consequently, I am looking for help especially with the five following tasks, in no particular order:

 * Setting up continuous integration, preferably free on the cloud ([Travis CI](https://travis-ci.org/)?)
 * Improving the `Parser` (by using the [presets for Clang](llvm/src/main/java/org/bytedeco/javacpp/clang.java)?)
 * Providing builds for more platforms, as with `linux-armhf` for [Raspberry Pi](https://www.raspberrypi.org/), etc.
 * Replacing the Bash/Maven build combo by something easier to use ([Gradle](http://gradle.org/)?)
 * Adding new presets as child modules for other C/C++ libraries (Caffe2, OpenNI, OpenMesh, PCL, etc.)

To contribute, please fork and create pull requests, or post your suggestions [as a new "issue"](https://github.com/bytedeco/javacpp-presets/issues). Thank you very much in advance for your contribution!


----
Project lead: Samuel Audet [samuel.audet `at` gmail.com](mailto:samuel.audet&nbsp;at&nbsp;gmail.com)  
Developer site: https://github.com/bytedeco/javacpp-presets  
Discussion group: http://groups.google.com/group/javacpp-project
