JavaCPP Presets for Systems
===========================

[![Gitter](https://badges.gitter.im/bytedeco/javacpp.svg)](https://gitter.im/bytedeco/javacpp) [![Maven Central](https://maven-badges.herokuapp.com/maven-central/org.bytedeco/systems/badge.svg)](https://maven-badges.herokuapp.com/maven-central/org.bytedeco/systems) [![Sonatype Nexus (Snapshots)](https://img.shields.io/nexus/s/https/oss.sonatype.org/org.bytedeco/systems.svg)](http://bytedeco.org/builds/)  
<sup>Build status for all platforms:</sup> [![systems](https://github.com/bytedeco/javacpp-presets/workflows/systems/badge.svg)](https://github.com/bytedeco/javacpp-presets/actions?query=workflow%3Asystems)  <sup>Commercial support:</sup> [![xscode](https://img.shields.io/badge/Available%20on-xs%3Acode-blue?style=?style=plastic&logo=appveyor&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEAAAABACAMAAACdt4HsAAAAGXRFWHRTb2Z0d2FyZQBBZG9iZSBJbWFnZVJlYWR5ccllPAAAAAZQTFRF////////VXz1bAAAAAJ0Uk5T/wDltzBKAAAAlUlEQVR42uzXSwqAMAwE0Mn9L+3Ggtgkk35QwcnSJo9S+yGwM9DCooCbgn4YrJ4CIPUcQF7/XSBbx2TEz4sAZ2q1RAECBAiYBlCtvwN+KiYAlG7UDGj59MViT9hOwEqAhYCtAsUZvL6I6W8c2wcbd+LIWSCHSTeSAAECngN4xxIDSK9f4B9t377Wd7H5Nt7/Xz8eAgwAvesLRjYYPuUAAAAASUVORK5CYII=)](https://xscode.com/bytedeco/javacpp-presets)


Introduction
------------
This directory contains the JavaCPP Presets module for system APIs:

 * Linux (glibc)  https://www.gnu.org/software/libc/
 * Mac OS X (XNU libc)  https://opensource.apple.com/
 * Windows (Win32)  https://developer.microsoft.com/en-us/windows/

For now, basic functionality is supported. The range will be expanded based on demand, so please make sure to communicate your needs.

Please refer to the parent README.md file for more detailed information about the JavaCPP Presets.


Documentation
-------------
Java API documentation is available here:

 * http://bytedeco.org/javacpp-presets/systems/apidocs/


Sample Usage
------------
Here is a simple example to detect some AVX capabilities by using CPUID functions exposed by systems APIs for x86 processors:

 * https://en.wikipedia.org/wiki/CPUID

We can use [Maven 3](http://maven.apache.org/) to download and install automatically all the class files as well as the native binaries. To run this sample code, after creating the `pom.xml` and `TestAVX.java` source files below, simply execute on the command line:
```bash
 $ mvn compile exec:java
```

### The `pom.xml` build file
```xml
<project>
    <modelVersion>4.0.0</modelVersion>
    <groupId>org.bytedeco.systems</groupId>
    <artifactId>testavx</artifactId>
    <version>1.5.7</version>
    <properties>
        <exec.mainClass>TestAVX</exec.mainClass>
    </properties>
    <dependencies>
        <dependency>
            <groupId>org.bytedeco</groupId>
            <artifactId>systems-platform</artifactId>
            <version>1.5.7</version>
        </dependency>
    </dependencies>
    <build>
        <sourceDirectory>.</sourceDirectory>
    </build>
</project>
```

### The `TestAVX.java` source file
```java
import org.bytedeco.javacpp.*;
import org.bytedeco.systems.global.*;

public class TestAVX {
    public static void main(String[] args) {
        int AVXbit = 0x10000000;
        int AVX2bit = 0x20;
        boolean hasAVX = false;
        boolean hasAVX2 = false;
        String platform = Loader.getPlatform();

        if (platform.startsWith("linux-x86")) {
            int[] eax = {0}, ebx = {0}, ecx = {0}, edx = {0};
            linux.__cpuid_count(1, 0, eax, ebx, ecx, edx);
            hasAVX = (ecx[0] & AVXbit) != 0;
            linux.__cpuid_count(7, 0, eax, ebx, ecx, edx);
            hasAVX2 = hasAVX && (ebx[0] & AVX2bit) != 0;
        } else if (platform.startsWith("macosx-x86")) {
            int[] eax = {0}, ebx = {0}, ecx = {0}, edx = {0};
            macosx.__cpuid_count(1, 0, eax, ebx, ecx, edx);
            hasAVX = (ecx[0] & AVXbit) != 0;
            macosx.__cpuid_count(7, 0, eax, ebx, ecx, edx);
            hasAVX2 = hasAVX && (ebx[0] & AVX2bit) != 0;
        } else if (platform.startsWith("windows-x86")) {
            int[] cpuinfo = new int[4];
            windows.__cpuidex(cpuinfo, 1, 0);
            hasAVX = (cpuinfo[2] & AVXbit) != 0;
            windows.__cpuidex(cpuinfo, 7, 0);
            hasAVX2 = hasAVX && (cpuinfo[1] & AVX2bit) != 0;
        }

        System.out.println("hasAVX = " + hasAVX);
        System.out.println("hasAVX2 = " + hasAVX2);
    }
}
```
