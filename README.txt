=JavaCPP Presets=

==Introduction==
The JavaCPP Presets module contains Java configuration and interface classes for widely used C/C++ libraries. The configuration files in the `com.googlecode.javacpp.presets` package are used by the `Parser` to create from C/C++ header files the Java interface files targeting the `com.googlecode.javacpp` package, which is turn are used by the `Generator` and the native C++ compiler to produce the required JNI libraries.

More details to come shortly... In the meantime, please feel free to ask questions on [http://groups.google.com/group/javacpp-project the mailing list].


==Required Software==
To use JavaCPP Presets, you will need to download and install the following software:
 * An implementation of Java SE 6 or 7
  * OpenJDK  http://openjdk.java.net/install/  or
  * Sun JDK  http://www.oracle.com/technetwork/java/javase/downloads/  or
  * IBM JDK  http://www.ibm.com/developerworks/java/jdk/  or
  * Java SE for Mac OS X  http://developer.apple.com/java/  etc.


==Build Instructions==
The source code can be found at this repository: 
 * https://code.google.com/p/javacpp.presets/

To rebuild the source code, please note that the project files were created for:
 * Maven 2 or 3  http://maven.apache.org/download.html
 * JavaCPP 0.8  http://code.google.com/p/javacpp/
 * msinttypes  http://code.google.com/p/msinttypes/ (under Windows for the Microsoft C/C++ Compiler)

Each child module in turn relies on its corresponding native library being installed in the directory specified in its `.java` configuration file or, by default, on the native system in `/usr/local/`, or `C:/MinGW/local/` (under Windows), or `${platform.root}/../` (for Android):
 * OpenCV 2.4.8  http://opencv.org/downloads.html
 * FFmpeg 2.1.x  http://ffmpeg.org/download.html
 * libdc1394 2.1.x or 2.2.x  http://sourceforge.net/projects/libdc1394/files/
 * libfreenect 0.2  https://github.com/OpenKinect/libfreenect
 * videoInput 0.200  https://github.com/ofTheo/videoInput/tree/update2013
 * ARToolKitPlus 2.3.0  https://launchpad.net/artoolkitplus

Once everything installed, simply execute
{{{
$ mvn install --projects opencv,ffmpeg,libdc1394,libfreenect,videoinput,artoolkitplus,distribution
}}}
in the root directory, by specifying only the desired child modules in the command. Please refer to the comments inside the parent `pom.xml` file for further details.


==Quick Start==
Simply put all the desired JAR files (`opencv*.jar`, `ffmpeg*.jar`, `libdc1394*.jar`, `libfreenect*.jar`, `videoinput*.jar`, and `artoolkitplus*.jar`), in addition to `javacpp.jar`, somewhere in your CLASSPATH, or point your `pom.xml` file to the Maven repository http://maven2.javacpp.googlecode.com/git/, when the binary files are present. Here are some more specific instructions for common cases:

NetBeans (Java SE 6 or 7):
 # In the Projects window, right-click the Libraries node of your project, and select "Add JAR/Folder...".
 # Locate the JAR files, select them, and click OK.

Eclipse (Java SE 6 or 7):
 # Navigate to Project > Properties > Java Build Path > Libraries and click "Add External JARs...".
 # Locate the JAR files, select them, and click OK.

Eclipse (Android 2.2 or newer):
 # Follow the instructions on this page: http://developer.android.com/training/basics/firstapp/
 # Go to File > New > Folder, select your project as parent folder, type "libs/armeabi" as Folder name, and click Finish.
 # Copy `javacpp.jar`, `opencv.jar`, `ffmpeg.jar`, and `artoolkitplus.jar` into the newly created "libs" folder.
 # Extract all the `*.so` files from `opencv-android-arm.jar`, `ffmpeg-android-arm.jar` and `artoolkitplus-android-arm.jar` directly into the newly created "libs/armeabi" folder, without creating any of the subdirectories found in the JAR files.
 # Navigate to Project > Properties > Java Build Path > Libraries and click "Add JARs...".
 # Select all of `javacpp.jar`, `ffmpeg.jar`, and `artoolkitplus.jar` from the newly created "libs" folder.

After that, we can access almost transparently the corresponding C/C++ APIs through the interface classes found in the `com.googlecode.javacpp` package. Indeed, the `Parser` translates the code comments from the C/C++ header files into the Java interface files, (almost) ready to be consumed by Javadoc.


==How Can I Help?==
Contribution of any kind is highly welcome! At the moment, the `Parser` has limited capabilities, but I plan to improve it gradually to the point where it can successfully parse large C++ header files that are even more convoluted than the ones from OpenCV. Consequently, I am looking for help especially with the two following tasks:
 # Improving the `Parser`
 # Adding new presets as child modules for other C/C++ libraries (LLVM, OpenMesh, Tesseract, etc)

Please post your suggestions and patches [http://code.google.com/p/javacpp/issues/ as a new "issue"]. Thank you very much in advance for your contribution!


==Changes==

 * Added presets for the C++ API of OpenCV 2.4.8, which can now be parsed due to the latest changes in JavaCPP

===January 6, 2014 version 0.7===
 * Fixed JavaCPP properties not getting set by the parent `pom.xml` file
 * Added presets for the C API of OpenCV 2.4.8, FFmpeg 2.1.x, libfreenect 0.2 (OpenKinect), videoInput 0.200, and ARToolkitPlus 2.3.0

===September 15, 2013 version 0.6===
Initial release


----
Copyright (C) 2013-2014 Samuel Audet <samuel.audet@gmail.com>
Project site: http://code.google.com/p/javacpp/

Licensed under the GNU General Public License version 2 (GPLv2) with Classpath exception.
Please refer to LICENSE.txt or http://www.gnu.org/licenses/ for details.

