JavaCPP Presets for libecl
==========================

Introduction
------------
This directory contains the JavaCPP Presets module for:

 * libecl 2.9.1  https://github.com/equinor/libecl

Please refer to the parent README.md file for more detailed information about the JavaCPP Presets.


Documentation
-------------
Java API documentation is available here:

 * http://bytedeco.org/javacpp-presets/libecl/apidocs/


Sample Usage
------------
Here is a simple example of libecl ported to Java from this C source file:

 * https://github.com/equinor/libecl/blob/2.9.1/applications/ecl/kw_list.c

We can use [Maven 3](http://maven.apache.org/) to download and install automatically all the class files as well as the native binaries. To run this sample code, after creating the `pom.xml` and `KeywordsList.java` source files below, simply execute on the command line:
```bash
 $ mvn compile exec:java
```

### The `pom.xml` build file
```xml
<project>
    <modelVersion>4.0.0</modelVersion>
    <groupId>org.bytedeco.libecl</groupId>
    <artifactId>KeywordsList</artifactId>
    <version>1.5.4</version>
    <properties>
        <exec.mainClass>KeywordsList</exec.mainClass>
    </properties>
    <dependencies>
        <dependency>
            <groupId>org.bytedeco</groupId>
            <artifactId>libecl-platform</artifactId>
            <version>2.9.1-1.5.4</version>
        </dependency>
    </dependencies>
    <build>
        <sourceDirectory>.</sourceDirectory>
    </build>
</project>
```

### The `KeywordsList.java` source file
```java
/*
 * Copyright (C) 2011  Statoil ASA, Norway.
 * The file 'kw_list.c' is part of ERT - Ensemble based Reservoir Tool.
 * ERT is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * ERT is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE.
 * See the GNU General Public License at <http://www.gnu.org/licenses/gpl.html>
 * for more details.
 */
import org.bytedeco.libecl.fortio_type;
import org.bytedeco.libecl.ecl_kw_type;
import static org.bytedeco.libecl.global.libecl.*;

public class KeywordsList {

    public static void main(String[] args) {
        for (int i = 0; i < args.length; i++) {
            kw_list(args[i]);
        }
    }

    private static void kw_list(String filename) {
        fortio_type fortio;
        ecl_kw_type ecl_kw = ecl_kw_alloc_empty();
        BooleanPointer fmt_file = new BooleanPointer();
        if (ecl_util_fmt_file(filename,  fmt_file)) {

            System.out.println("-----------------------------------------------------------------");
            System.out.printf("%s: %n", filename);
            fortio = fortio_open_reader(filename, fmt_file, ECL_ENDIAN_FLIP);
            while (ecl_kw_fread_realloc(ecl_kw, fortio)) {
                ecl_kw_summarize(ecl_kw);
            }
            System.out.println("-----------------------------------------------------------------");

            ecl_kw_free(ecl_kw);
            fortio_fclose(fortio);
        } else {
            System.err.printf("Could not determine formatted/unformatted status of:%s - skipping%n", filename);
        }
    }

}
```

