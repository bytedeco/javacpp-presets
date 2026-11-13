JavaCPP Presets for Blosc2
=======================

[![Gitter](https://badges.gitter.im/bytedeco/javacpp.svg)](https://gitter.im/bytedeco/javacpp) [![Maven Central](https://maven-badges.herokuapp.com/maven-central/org.bytedeco/blosc2/badge.svg)](https://maven-badges.herokuapp.com/maven-central/org.bytedeco/blosc2) [![Sonatype Nexus (Snapshots)](https://img.shields.io/nexus/s/https/oss.sonatype.org/org.bytedeco/blosc2.svg)](http://bytedeco.org/builds/)  
<sup>Build status for all platforms:</sup> [![blosc2](https://github.com/bytedeco/javacpp-presets/workflows/blosc2/badge.svg)](https://github.com/bytedeco/javacpp-presets/actions?query=workflow%3Ablosc2)  <sup>Commercial support:</sup> [![xscode](https://img.shields.io/badge/Available%20on-xs%3Acode-blue?style=?style=plastic&logo=appveyor&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEAAAABACAMAAACdt4HsAAAAGXRFWHRTb2Z0d2FyZQBBZG9iZSBJbWFnZVJlYWR5ccllPAAAAAZQTFRF////////VXz1bAAAAAJ0Uk5T/wDltzBKAAAAlUlEQVR42uzXSwqAMAwE0Mn9L+3Ggtgkk35QwcnSJo9S+yGwM9DCooCbgn4YrJ4CIPUcQF7/XSBbx2TEz4sAZ2q1RAECBAiYBlCtvwN+KiYAlG7UDGj59MViT9hOwEqAhYCtAsUZvL6I6W8c2wcbd+LIWSCHSTeSAAECngN4xxIDSK9f4B9t377Wd7H5Nt7/Xz8eAgwAvesLRjYYPuUAAAAASUVORK5CYII=)](https://xscode.com/bytedeco/javacpp-presets)


Introduction
------------
This directory contains the JavaCPP Presets module for:

 * Blosc2 3.6.2  https://github.com/Blosc/c-blosc2

C-Blosc2 reads and writes Blosc1-compatible chunks (`blosc1_compress()` and `blosc1_decompress()`), including a context option, `blosc2_ctx_set_blosc1_format()`, that writes chunks which C-Blosc 1.x readers can decode (as needed for Zarr and N5; stock C-Blosc2 chunks, even from `blosc1_compress()`, are not readable by C-Blosc 1.x, unless `BLOSC_BLOSC1_COMPAT` is set), as well as native Blosc2 chunks, super-chunks, frames and N-dimensional arrays (`b2nd`). The LZ4, Zlib and Zstd codecs are bundled in the native library.

Please refer to the parent README.md file for more detailed information about the JavaCPP Presets.


Documentation
-------------
Java API documentation is available here:

 * http://bytedeco.org/javacpp-presets/blosc2/apidocs/


Sample Usage
------------
Here is a simple example of Blosc compression and decompression.

We can use [Maven 3](http://maven.apache.org/) to download and install automatically all the class files as well as the native binaries. To run this sample code, after creating the `pom.xml` and `BloscRoundTripExample.java` source files below, simply execute on the command line:
```bash
 $ mvn compile exec:java
```

### The `pom.xml` build file
```xml
<project>
    <modelVersion>4.0.0</modelVersion>
    <groupId>org.bytedeco.blosc2</groupId>
    <artifactId>blosc2-round-trip-example</artifactId>
    <version>1.5.7</version>
    <properties>
        <exec.mainClass>BloscRoundTripExample</exec.mainClass>
        <maven.compiler.source>1.7</maven.compiler.source>
        <maven.compiler.target>1.7</maven.compiler.target>
    </properties>
    <dependencies>
        <dependency>
            <groupId>org.bytedeco</groupId>
            <artifactId>blosc2-platform</artifactId>
            <version>3.6.2-1.5.7</version>
        </dependency>
    </dependencies>
    <build>
        <sourceDirectory>.</sourceDirectory>
    </build>
</project>
```

### The `BloscRoundTripExample.java` source file
```java
import org.bytedeco.javacpp.*;
import org.bytedeco.blosc2.*;
import org.bytedeco.blosc2.global.blosc2;

/**
 * Compresses and decompresses a buffer of 32-bit integers the way Zarr and N5 chunks are encoded with Blosc.
 * Stock C-Blosc2 chunks cannot be decoded by C-Blosc 1.x readers (such as numcodecs), not even those written
 * by blosc1_compress(). blosc2_ctx_set_blosc1_format() (C-Blosc2 3.6.2 and later) makes the context write
 * chunks that C-Blosc 1.x can decode. C-Blosc2 decodes both kinds with blosc1_decompress().
 */
public class BloscRoundTripExample {
    public static void main(String[] args) {
        blosc2.blosc2_init();
        try {
            blosc2.blosc2_set_nthreads((short)4);

            int n = 1 << 20;
            IntPointer src = new IntPointer(n);
            for (int i = 0; i < n; i++) {
                src.put(i, i / 16);
            }
            long nbytes = (long)n * Integer.BYTES;

            // clevel 5, byte shuffle, typesize 4, written in the Blosc1 chunk format
            blosc2_cparams cparams = new blosc2_cparams();
            cparams.compcode((byte)blosc2.BLOSC_ZSTD);
            cparams.clevel((byte)5);
            cparams.typesize(Integer.BYTES);
            cparams.nthreads((short)4);
            cparams.splitmode(blosc2.BLOSC_FORWARD_COMPAT_SPLIT);
            cparams.filters(blosc2.BLOSC2_MAX_FILTERS - 1, (byte)blosc2.BLOSC_SHUFFLE);
            blosc2_context cctx = blosc2.blosc2_create_cctx(cparams);
            if (blosc2.blosc2_ctx_set_blosc1_format(cctx, true) < 0) {
                throw new RuntimeException("Blosc1 format is not available for this setup");
            }
            BytePointer compressed = new BytePointer(nbytes + blosc2.BLOSC2_MAX_OVERHEAD);
            int csize = blosc2.blosc2_compress_ctx(cctx, src, (int)nbytes, compressed, (int)compressed.capacity());
            blosc2.blosc2_free_ctx(cctx);
            if (csize <= 0) {
                throw new RuntimeException("Compression failed: " + csize);
            }
            System.out.println("Compressed " + nbytes + " bytes into " + csize + " bytes");

            // The uncompressed size is stored in the chunk header
            SizeTPointer uncompressedSize = new SizeTPointer(1);
            SizeTPointer compressedSize = new SizeTPointer(1);
            SizeTPointer blockSize = new SizeTPointer(1);
            blosc2.blosc1_cbuffer_sizes(compressed, uncompressedSize, compressedSize, blockSize);

            IntPointer dest = new IntPointer(n);
            int dsize = blosc2.blosc1_decompress(compressed, dest, uncompressedSize.get());
            if (dsize != nbytes) {
                throw new RuntimeException("Decompression failed: " + dsize);
            }
            for (int i = 0; i < n; i++) {
                if (src.get(i) != dest.get(i)) {
                    throw new RuntimeException("Mismatch at index " + i);
                }
            }
            System.out.println("Round trip OK");
        } finally {
            blosc2.blosc2_destroy();
        }
    }
}
```


---
Copyright (C) 2022 Mark Kittisopikul
Project Info: https://github.com/bytedeco/javacpp-presets/tree/master/blosc2
