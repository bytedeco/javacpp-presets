JavaCPP Presets for Qt
======================

[![Gitter](https://badges.gitter.im/bytedeco/javacpp.svg)](https://gitter.im/bytedeco/javacpp) [![Maven Central](https://maven-badges.herokuapp.com/maven-central/org.bytedeco/qt/badge.svg)](https://maven-badges.herokuapp.com/maven-central/org.bytedeco/qt) [![Sonatype Nexus (Snapshots)](https://img.shields.io/nexus/s/https/oss.sonatype.org/org.bytedeco/qt.svg)](http://bytedeco.org/builds/)  
<sup>Build status for all platforms:</sup> [![qt](https://github.com/bytedeco/javacpp-presets/workflows/qt/badge.svg)](https://github.com/bytedeco/javacpp-presets/actions?query=workflow%3Aqt)  <sup>Commercial support:</sup> [![xscode](https://img.shields.io/badge/Available%20on-xs%3Acode-blue?style=?style=plastic&logo=appveyor&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEAAAABACAMAAACdt4HsAAAAGXRFWHRTb2Z0d2FyZQBBZG9iZSBJbWFnZVJlYWR5ccllPAAAAAZQTFRF////////VXz1bAAAAAJ0Uk5T/wDltzBKAAAAlUlEQVR42uzXSwqAMAwE0Mn9L+3Ggtgkk35QwcnSJo9S+yGwM9DCooCbgn4YrJ4CIPUcQF7/XSBbx2TEz4sAZ2q1RAECBAiYBlCtvwN+KiYAlG7UDGj59MViT9hOwEqAhYCtAsUZvL6I6W8c2wcbd+LIWSCHSTeSAAECngN4xxIDSK9f4B9t377Wd7H5Nt7/Xz8eAgwAvesLRjYYPuUAAAAASUVORK5CYII=)](https://xscode.com/bytedeco/javacpp-presets)


Introduction
------------
This directory contains the JavaCPP Presets module for:

 * Qt 5.15.2  https://www.qt.io/

Please refer to the parent README.md file for more detailed information about the JavaCPP Presets.


Documentation
-------------
Java API documentation is available here:

 * http://bytedeco.org/javacpp-presets/qt/apidocs/

&lowast; Call `Loader.load(Qt5Core.class)` to get the path for [`-platformpluginpath`](http://doc.qt.io/qt-5/qguiapplication.html#QGuiApplication) before using this API.  
&lowast; Java must be started with the '-XstartOnFirstThread' command-line option on macOS.


Sample Usage
------------
Here is a simple example of Qt ported to Java from this C++ source file:

 * https://doc.qt.io/archives/qt-4.8/gettingstartedqt.html

We can use [Maven 3](http://maven.apache.org/) to download and install automatically all the class files as well as the native binaries. To run this sample code, after creating the `pom.xml` and `GettingStarted.java` source files below, simply execute on the command line:
```bash
 $ mvn compile exec:java
```
Please see the samples directory for more Qt examples ported to Java from C++.

### The `pom.xml` build file
```xml
<project>
    <modelVersion>4.0.0</modelVersion>
    <groupId>org.bytedeco.qt</groupId>
    <artifactId>gettingstarted</artifactId>
    <version>1.5.7</version>
    <properties>
        <exec.mainClass>GettingStarted</exec.mainClass>
    </properties>
    <dependencies>
        <dependency>
            <groupId>org.bytedeco</groupId>
            <artifactId>qt-platform</artifactId>
            <version>5.15.2-1.5.7</version>
        </dependency>
    </dependencies>
    <build>
        <sourceDirectory>.</sourceDirectory>
    </build>
</project>
```

### The `GettingStarted.java` source file
```java
import java.io.File;
import org.bytedeco.javacpp.*;
import org.bytedeco.qt.Qt5Core.*;
import org.bytedeco.qt.Qt5Gui.*;
import org.bytedeco.qt.Qt5Widgets.*;
import static org.bytedeco.qt.global.Qt5Core.*;
import static org.bytedeco.qt.global.Qt5Gui.*;
import static org.bytedeco.qt.global.Qt5Widgets.*;

public class GettingStarted {
    private static IntPointer argc;
    private static PointerPointer argv;

    public static void main(String[] args) {
        String path = Loader.load(org.bytedeco.qt.global.Qt5Core.class);
        argc = new IntPointer(new int[]{3});
        argv = new PointerPointer("gettingstarted", "-platformpluginpath", new File(path).getParent(), null);

        QApplication app = new QApplication(argc, argv);

        QTextEdit textEdit = new QTextEdit();
        textEdit.show();

        System.exit(app.exec());
    }
}
```
