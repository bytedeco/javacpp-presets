
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
