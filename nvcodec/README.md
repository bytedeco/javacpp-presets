JavaCPP Presets for NVIDIA Video Codec SDK
==========================================

[![Gitter](https://badges.gitter.im/bytedeco/javacpp.svg)](https://gitter.im/bytedeco/javacpp) [![Maven Central](https://maven-badges.herokuapp.com/maven-central/org.bytedeco/nvcodec/badge.svg)](https://maven-badges.herokuapp.com/maven-central/org.bytedeco/nvcodec) [![Sonatype Nexus (Snapshots)](https://img.shields.io/nexus/s/https/oss.sonatype.org/org.bytedeco/nvcodec.svg)](http://bytedeco.org/builds/)  
<sup>Build status for all platforms:</sup> [![nvcodec](https://github.com/bytedeco/javacpp-presets/workflows/nvcodec/badge.svg)](https://github.com/bytedeco/javacpp-presets/actions?query=workflow%3Anvcodec)  <sup>Commercial support:</sup> [![xscode](https://img.shields.io/badge/Available%20on-xs%3Acode-blue?style=?style=plastic&logo=appveyor&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEAAAABACAMAAACdt4HsAAAAGXRFWHRTb2Z0d2FyZQBBZG9iZSBJbWFnZVJlYWR5ccllPAAAAAZQTFRF////////VXz1bAAAAAJ0Uk5T/wDltzBKAAAAlUlEQVR42uzXSwqAMAwE0Mn9L+3Ggtgkk35QwcnSJo9S+yGwM9DCooCbgn4YrJ4CIPUcQF7/XSBbx2TEz4sAZ2q1RAECBAiYBlCtvwN+KiYAlG7UDGj59MViT9hOwEqAhYCtAsUZvL6I6W8c2wcbd+LIWSCHSTeSAAECngN4xxIDSK9f4B9t377Wd7H5Nt7/Xz8eAgwAvesLRjYYPuUAAAAASUVORK5CYII=)](https://xscode.com/bytedeco/javacpp-presets)


License Agreements
------------------
By downloading these archives, you agree to the terms of the license agreements for NVIDIA software included in the archives.

### NVIDIA Video Codec SDK
To view the license for NVIDIA Video Codec SDK included in these archives, click [here](https://docs.nvidia.com/video-technologies/video-codec-sdk/license/)


Introduction
------------
This directory contains the JavaCPP Presets module for:

 * NVIDIA Video Codec SDK 11.1.5  https://developer.nvidia.com/nvidia-video-codec-sdk

Please refer to the parent README.md file for more detailed information about the JavaCPP Presets.


Documentation
-------------
Java API documentation is available here:

 * http://bytedeco.org/javacpp-presets/nvcodec/apidocs/


Sample Usage
------------
Here is a simple example ported to Java from C code based on `Samples/AppEncode/AppEncCuda` and `Samples/AppDecode/AppDec` included in `Video_Codec_SDK_11.1.5.zip` available at:

 * https://developer.nvidia.com/nvidia-video-codec-sdk/download

We can use [Maven 3](http://maven.apache.org/) to download and install automatically all the class files as well as the native binaries. To run this sample code, after creating the `pom.xml` and `SampleEncodeDecode.java` source files below, simply execute on the command line:
```bash
 $ mvn compile exec:java
```
You can find more encoder and decoder samples in the [`samples`](samples) subdirectory.

### The `pom.xml` build file
```xml
<project>
    <modelVersion>4.0.0</modelVersion>
    <groupId>org.bytedeco.nvcodec</groupId>
    <artifactId>sampleencodedecode</artifactId>
    <version>1.5.7</version>
    <properties>
        <exec.mainClass>SampleEncodeDecode</exec.mainClass>
    </properties>
    <dependencies>
        <dependency>
            <groupId>org.bytedeco</groupId>
            <artifactId>nvcodec-platform</artifactId>
            <version>11.1.5-1.5.7</version>
        </dependency>

        <!-- Additional dependencies to use bundled CUDA -->
        <dependency>
            <groupId>org.bytedeco</groupId>
            <artifactId>cuda-platform-redist</artifactId>
            <version>11.6-8.3-1.5.7</version>
        </dependency>
    </dependencies>
    <build>
        <sourceDirectory>.</sourceDirectory>
    </build>
</project>
```

### The `SampleEncodeDecode.java` source file
```java
import org.bytedeco.cuda.cudart.CUctx_st;
import org.bytedeco.javacpp.IntPointer;
import org.bytedeco.nvcodec.nvcuvid.CUVIDDECODECAPS;

import static org.bytedeco.cuda.global.cudart.*;
import static org.bytedeco.nvcodec.global.nvcuvid.*;
import static org.bytedeco.nvcodec.global.nvencodeapi.*;

public class SampleEncodeDecode {
    public static void checkEncodeApiCall(String functionName, int result) {
        if (result != NV_ENC_SUCCESS) {
            System.err.printf("ERROR: %s returned '%d' \r\n", functionName, result);
            System.exit(-1);
        }
    }

    public static void checkCudaApiCall(String functionName, int result) {
        if (result != CUDA_SUCCESS) {
            System.err.printf("ERROR: %s returned '%d' \r\n", functionName, result);
            System.exit(-1);
        }
    }

    public static void main(String[] args) {
        int targetGpu = 0; // If you use NVIDIA GPU not '0', changing it.

        CUctx_st cuContext = new CUctx_st();

        checkCudaApiCall("cuInit", cuInit(0));
        checkCudaApiCall("cuCtxCreate", cuCtxCreate(cuContext, 0, targetGpu));
        // Check encoder max supported version
        {
            IntPointer version = new IntPointer(1);

            checkEncodeApiCall("NvEncodeAPIGetMaxSupportedVersion", NvEncodeAPIGetMaxSupportedVersion(version));

            System.out.printf("Encoder Max Supported Version\t : %d \r\n", version.get());
        }

        // Query decoder capability 'MPEG-1' codec
        {
            CUVIDDECODECAPS decodeCaps = new CUVIDDECODECAPS();
            decodeCaps.eCodecType(cudaVideoCodec_HEVC);
            decodeCaps.eChromaFormat(cudaVideoChromaFormat_420);
            decodeCaps.nBitDepthMinus8(2); // 10 bit

            checkCudaApiCall("cuvidGetDecoderCaps", cuvidGetDecoderCaps(decodeCaps));

            System.out.printf("Decoder Capability MPEG-1 Codec\t : %s \r\n", (decodeCaps.bIsSupported() != 0));
        }
    }
}
```
