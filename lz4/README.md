JavaCPP Presets for LZ4
=======================

[![Gitter](https://badges.gitter.im/bytedeco/javacpp.svg)](https://gitter.im/bytedeco/javacpp) [![Maven Central](https://maven-badges.herokuapp.com/maven-central/org.bytedeco/lz4/badge.svg)](https://maven-badges.herokuapp.com/maven-central/org.bytedeco/lz4) [![Sonatype Nexus (Snapshots)](https://img.shields.io/nexus/s/https/oss.sonatype.org/org.bytedeco/lz4.svg)](http://bytedeco.org/builds/)  
<sup>Build status for all platforms:</sup> [![lz4](https://github.com/bytedeco/javacpp-presets/workflows/lz4/badge.svg)](https://github.com/bytedeco/javacpp-presets/actions?query=workflow%3Alz4)  <sup>Commercial support:</sup> [![xscode](https://img.shields.io/badge/Available%20on-xs%3Acode-blue?style=?style=plastic&logo=appveyor&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEAAAABACAMAAACdt4HsAAAAGXRFWHRTb2Z0d2FyZQBBZG9iZSBJbWFnZVJlYWR5ccllPAAAAAZQTFRF////////VXz1bAAAAAJ0Uk5T/wDltzBKAAAAlUlEQVR42uzXSwqAMAwE0Mn9L+3Ggtgkk35QwcnSJo9S+yGwM9DCooCbgn4YrJ4CIPUcQF7/XSBbx2TEz4sAZ2q1RAECBAiYBlCtvwN+KiYAlG7UDGj59MViT9hOwEqAhYCtAsUZvL6I6W8c2wcbd+LIWSCHSTeSAAECngN4xxIDSK9f4B9t377Wd7H5Nt7/Xz8eAgwAvesLRjYYPuUAAAAASUVORK5CYII=)](https://xscode.com/bytedeco/javacpp-presets)


Introduction
------------
This directory contains the JavaCPP Presets module for:

 * LZ4 1.9.3  https://github.com/lz4/lz4/

Please refer to the parent README.md file for more detailed information about the JavaCPP Presets.


Documentation
-------------
Java API documentation is available here:

 * http://bytedeco.org/javacpp-presets/lz4/apidocs/


Sample Usage
------------
Here is a simple example of LZ4 frame compression.

We can use [Maven 3](http://maven.apache.org/) to download and install automatically all the class files as well as the native binaries. To run this sample code, after creating the `pom.xml` and `LZ4FrameCompressionExample.java` source files below, simply execute on the command line:
```bash
 $ mvn compile exec:java
```

### The `pom.xml` build file
```xml
<project>
    <modelVersion>4.0.0</modelVersion>
    <groupId>org.bytedeco.lz4</groupId>
    <artifactId>lz4-frame-compression-example</artifactId>
    <version>1.5.7</version>
    <properties>
        <exec.mainClass>LZ4FrameCompressionExample</exec.mainClass>
        <maven.compiler.source>1.7</maven.compiler.source>
        <maven.compiler.target>1.7</maven.compiler.target>
    </properties>
    <dependencies>
        <dependency>
            <groupId>org.bytedeco</groupId>
            <artifactId>lz4-platform</artifactId>
            <version>1.9.3-1.5.7</version>
        </dependency>
    </dependencies>
    <build>
        <sourceDirectory>.</sourceDirectory>
    </build>
</project>
```

### The `LZ4FrameCompressionExample.java` source file
```java
import java.nio.ByteBuffer;
import org.bytedeco.javacpp.*;
import org.bytedeco.lz4.*;
import org.bytedeco.lz4.global.lz4;

public final class LZ4FrameCompressionExample {

    private static final int NUM_VALUES = 10 * 1024 * 1024; // 10MB

    public static void main(String[] args) throws LZ4Exception {
        // Print LZ4 version
        System.out.println("LZ4 Version: " + lz4.LZ4_VERSION_STRING.getString());

        // Generate some data
        final ByteBuffer data = ByteBuffer.allocateDirect(NUM_VALUES);
        for (int i = 0; i < NUM_VALUES; i++) {
            data.put((byte) i);
        }
        data.position(0);

        // Compress
        final ByteBuffer compressed = compress(data);
        System.out.println("Uncompressed size: " + data.limit());
        System.out.println("Compressed size: " + compressed.limit());

        // Decompress
        final ByteBuffer decompressed = decompress(compressed, data.limit());

        // Verify that decompressed == data
        for (int i = 0; i < NUM_VALUES; i++) {
            if (data.get(i) != decompressed.get(i)) {
                throw new IllegalStateException("Input and output differ.");
            }
        }
        System.out.println("Verified that input data == output data");
    }

    private static ByteBuffer compress(ByteBuffer data) {
        // Output buffer
        final int maxCompressedSize = (int) lz4.LZ4F_compressFrameBound(data.limit(), null);
        final ByteBuffer compressed = ByteBuffer.allocateDirect(maxCompressedSize);

        final Pointer dataPointer = new Pointer(data);
        final Pointer dstPointer = new Pointer(compressed);
        final long compressedSize = lz4.LZ4F_compressFrame(dstPointer, compressed.limit(), dataPointer, data.limit(),
                null);
        compressed.limit((int) compressedSize);
        return compressed;
    }

    private static ByteBuffer decompress(ByteBuffer compressed, int uncompressedSize) throws LZ4Exception {
        final LZ4FDecompressionContext dctx = new LZ4FDecompressionContext();
        final long ctxError = lz4.LZ4F_createDecompressionContext(dctx, lz4.LZ4F_VERSION);
        checkForError(ctxError);

        // Output buffer
        final ByteBuffer decompressed = ByteBuffer.allocateDirect(uncompressedSize);

        final SizeTPointer dstSize = new SizeTPointer(1);
        final SizeTPointer srcSize = new SizeTPointer(1);

        try {
            long ret;
            do {
                dstSize.put(decompressed.remaining());
                srcSize.put(compressed.limit());
                final Pointer dstPointer = new Pointer(decompressed);
                final Pointer compressedPointer = new Pointer(compressed);

                ret = lz4.LZ4F_decompress(dctx, dstPointer, dstSize, compressedPointer, srcSize, null);
                checkForError(ret);
                decompressed.position(decompressed.position() + (int) dstSize.get());
                compressed.position(compressed.position() + (int) srcSize.get());
            } while (ret != 0);

        } finally {
            lz4.LZ4F_freeDecompressionContext(dctx);
        }

        decompressed.position(0);
        return decompressed;
    }

    private static void checkForError(long errorCode) throws LZ4Exception {
        if (lz4.LZ4F_isError(errorCode) != 0) {
            throw new LZ4Exception(lz4.LZ4F_getErrorName(errorCode).getString());
        }
    }

    private static final class LZ4Exception extends Exception {
        public LZ4Exception(final String message) {
            super(message);
        }
    }
}
```
