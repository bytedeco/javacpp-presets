JavaCPP Presets for Leptonica
=============================

[![Gitter](https://badges.gitter.im/bytedeco/javacpp.svg)](https://gitter.im/bytedeco/javacpp) [![Maven Central](https://maven-badges.herokuapp.com/maven-central/org.bytedeco/leptonica/badge.svg)](https://maven-badges.herokuapp.com/maven-central/org.bytedeco/leptonica) [![Sonatype Nexus (Snapshots)](https://img.shields.io/nexus/s/https/oss.sonatype.org/org.bytedeco/leptonica.svg)](http://bytedeco.org/builds/)  
<sup>Build status for all platforms:</sup> [![leptonica](https://github.com/bytedeco/javacpp-presets/workflows/leptonica/badge.svg)](https://github.com/bytedeco/javacpp-presets/actions?query=workflow%3Aleptonica)  <sup>Commercial support:</sup> [![xscode](https://img.shields.io/badge/Available%20on-xs%3Acode-blue?style=?style=plastic&logo=appveyor&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEAAAABACAMAAACdt4HsAAAAGXRFWHRTb2Z0d2FyZQBBZG9iZSBJbWFnZVJlYWR5ccllPAAAAAZQTFRF////////VXz1bAAAAAJ0Uk5T/wDltzBKAAAAlUlEQVR42uzXSwqAMAwE0Mn9L+3Ggtgkk35QwcnSJo9S+yGwM9DCooCbgn4YrJ4CIPUcQF7/XSBbx2TEz4sAZ2q1RAECBAiYBlCtvwN+KiYAlG7UDGj59MViT9hOwEqAhYCtAsUZvL6I6W8c2wcbd+LIWSCHSTeSAAECngN4xxIDSK9f4B9t377Wd7H5Nt7/Xz8eAgwAvesLRjYYPuUAAAAASUVORK5CYII=)](https://xscode.com/bytedeco/javacpp-presets)


Introduction
------------
This directory contains the JavaCPP Presets module for:

 * Leptonica 1.84.0  http://www.leptonica.org/

Please refer to the parent README.md file for more detailed information about the JavaCPP Presets.


Documentation
-------------
Java API documentation is available here:

 * http://bytedeco.org/javacpp-presets/leptonica/apidocs/


Sample Usage
------------
Here is a simple example of Leptonica ported to Java from this C source file:

 * http://tpgit.github.io/Leptonica/sudokutest_8c_source.html

We can use [Maven 3](http://maven.apache.org/) to download and install automatically all the class files as well as the native binaries. To run this sample code, after creating the `pom.xml` and `SudokuTest.java` source files below, simply execute on the command line:
```bash
 $ mvn compile exec:java -Dexec.args="[filein]"
```

### The `pom.xml` build file
```xml
<project>
    <modelVersion>4.0.0</modelVersion>
    <groupId>org.bytedeco.leptonica</groupId>
    <artifactId>sudokutest</artifactId>
    <version>1.5.10-SNAPSHOT</version>
    <properties>
        <exec.mainClass>SudokuTest</exec.mainClass>
    </properties>
    <dependencies>
        <dependency>
            <groupId>org.bytedeco</groupId>
            <artifactId>leptonica-platform</artifactId>
            <version>1.84.0-1.5.10-SNAPSHOT</version>
        </dependency>
    </dependencies>
    <build>
        <sourceDirectory>.</sourceDirectory>
    </build>
</project>
```

### The `SudokuTest.java` source file
```java
/*====================================================================*
 -  Copyright (C) 2001 Leptonica.  All rights reserved.
 -
 -  Redistribution and use in source and binary forms, with or without
 -  modification, are permitted provided that the following conditions
 -  are met:
 -  1. Redistributions of source code must retain the above copyright
 -     notice, this list of conditions and the following disclaimer.
 -  2. Redistributions in binary form must reproduce the above
 -     copyright notice, this list of conditions and the following
 -     disclaimer in the documentation and/or other materials
 -     provided with the distribution.
 -
 -  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 -  ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 -  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 -  A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL ANY
 -  CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 -  EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 -  PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 -  PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 -  OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 -  NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 -  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *====================================================================*/

/*
 * sudokutest.c
 *
 *   Tests sudoku solver and generator.
 */

import org.bytedeco.javacpp.*;
import org.bytedeco.leptonica.*;
import static org.bytedeco.leptonica.global.lept.*;

public class SudokuTest {
    static final String startsol = "3 8 7 2 6 4 1 9 5 "
                                 + "2 6 5 8 9 1 4 3 7 "
                                 + "1 4 9 5 3 7 6 8 2 "
                                 + "5 2 3 7 1 6 8 4 9 "
                                 + "7 1 6 9 4 8 2 5 3 "
                                 + "8 9 4 3 5 2 7 1 6 "
                                 + "9 7 2 1 8 5 3 6 4 "
                                 + "4 3 1 6 7 9 5 2 8 "
                                 + "6 5 8 4 2 3 9 7 1";

    public static void main(String[] args) {
        Loader.load(org.bytedeco.leptonica.global.lept.class);

        IntPointer   unique = new IntPointer(1);
        IntPointer   array;
        L_SUDOKU     sud;

        if (args.length != 0 && args.length != 1) {
            System.err.println("Syntax: SudokuTest [filein]");
            System.exit(1);
        }

        if (args.length == 0) {
            /* Generate a new sudoku by element elimination */
            array = sudokuReadString(new BytePointer(startsol));
            sud = sudokuGenerate(array, 3693, 28, 7);
            sudokuDestroy(sud);
            lept_free(array);
            System.exit(0);
        }

        /* Solve the input sudoku */
        if ((array = sudokuReadFile(new BytePointer(args[0]))) == null) {
            System.err.println("invalid input");
            System.exit(1);
        }
        if ((sud = sudokuCreate(array)) == null) {
            System.err.println("sud not made");
            System.exit(1);
        }
        sudokuOutput(sud, L_SUDOKU_INIT);
        startTimer();
        sudokuSolve(sud);
        System.err.printf("Time: %7.3f sec\n", stopTimer());
        sudokuOutput(sud, L_SUDOKU_STATE);
        sudokuDestroy(sud);

        /* Test for uniqueness */
        sudokuTestUniqueness(array, unique);
        if (unique.get(0) != 0) {
            System.err.println("Sudoku is unique");
        } else {
            System.err.println("Sudoku is NOT unique");
        }
        lept_free(array);
    }
}
```
