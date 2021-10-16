JavaCPP Presets for veoffload
=============================

[![Gitter](https://badges.gitter.im/bytedeco/javacpp.svg)](https://gitter.im/bytedeco/javacpp) [![Maven Central](https://maven-badges.herokuapp.com/maven-central/org.bytedeco/veoffload/badge.svg)](https://maven-badges.herokuapp.com/maven-central/org.bytedeco/veoffload) [![Sonatype Nexus (Snapshots)](https://img.shields.io/nexus/s/https/oss.sonatype.org/org.bytedeco/veoffload.svg)](http://bytedeco.org/builds/)  
<sup>Build status for all platforms:</sup> [![veoffload](https://github.com/bytedeco/javacpp-presets/workflows/veoffload/badge.svg)](https://github.com/bytedeco/javacpp-presets/actions?query=workflow%3Aveoffload)  <sup>Commercial support:</sup> [![xscode](https://img.shields.io/badge/Available%20on-xs%3Acode-blue?style=?style=plastic&logo=appveyor&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEAAAABACAMAAACdt4HsAAAAGXRFWHRTb2Z0d2FyZQBBZG9iZSBJbWFnZVJlYWR5ccllPAAAAAZQTFRF////////VXz1bAAAAAJ0Uk5T/wDltzBKAAAAlUlEQVR42uzXSwqAMAwE0Mn9L+3Ggtgkk35QwcnSJo9S+yGwM9DCooCbgn4YrJ4CIPUcQF7/XSBbx2TEz4sAZ2q1RAECBAiYBlCtvwN+KiYAlG7UDGj59MViT9hOwEqAhYCtAsUZvL6I6W8c2wcbd+LIWSCHSTeSAAECngN4xxIDSK9f4B9t377Wd7H5Nt7/Xz8eAgwAvesLRjYYPuUAAAAASUVORK5CYII=)](https://xscode.com/bytedeco/javacpp-presets)


Introduction
------------
This directory contains the JavaCPP Presets module for:

 * veoffload  https://github.com/SX-Aurora/aveo

Please refer to the parent README.md file for more detailed information about the JavaCPP Presets.


Documentation
-------------
Java API documentation is available here:

 * http://bytedeco.org/javacpp-presets/veoffload/apidocs/


Sample Usage
------------
Here is a simple example of veoffload ported to Java from the C source file on this page:

 * https://www.hpc.nec/documents/veos/en/aveo/md_GettingStarted.html

We can use [Maven 3](http://maven.apache.org/) to download and install automatically all the class files as well as the native binaries. To run this sample code, after creating the `pom.xml` and `vehello.java` source files below, simply execute on the command line:
```bash
 $ mvn compile exec:java
```

### The `pom.xml` build file
```xml
<project>
    <modelVersion>4.0.0</modelVersion>
    <groupId>org.bytedeco.veoffload</groupId>
    <artifactId>vehello</artifactId>
    <version>1.5.7-SNAPSHOT</version>
    <properties>
        <exec.mainClass>vehello</exec.mainClass>
    </properties>
    <dependencies>
        <dependency>
            <groupId>org.bytedeco</groupId>
            <artifactId>veoffload-platform</artifactId>
            <version>2.8.2-1.5.7-SNAPSHOT</version>
        </dependency>
    </dependencies>
    <build>
        <sourceDirectory>.</sourceDirectory>
    </build>
</project>
```

### The `vehello.java` source file
```java
import org.bytedeco.javacpp.*;
import org.bytedeco.veoffload.*;
import static org.bytedeco.veoffload.global.veo.*;

public class vehello {
    public static void main(String[] args) {
        /* Load "vehello" on VE node 0 */
        veo_proc_handle proc = veo_proc_create(0);
        long handle = veo_load_library(proc, "./libvehello.so");
        veo_thr_ctxt ctx = veo_context_open(proc);

        veo_args argp = veo_args_alloc();
        long id = veo_call_async_by_name(ctx, handle, "hello", argp);
        long retval[] = {0};
        veo_call_wait_result(ctx, id, retval);
        veo_args_free(argp);
        veo_context_close(ctx);
        veo_proc_destroy(proc);
        System.exit(0);
    }
}
```
