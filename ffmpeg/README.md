JavaCPP Presets for FFmpeg
==========================

Introduction
------------
This directory contains the JavaCPP Presets module for:

 * FFmpeg 3.0.2  http://ffmpeg.org/

Please refer to the parent README.md file for more detailed information about the JavaCPP Presets.


Documentation
-------------
Java API documentation is available here:

 * http://bytedeco.org/javacpp-presets/ffmpeg/apidocs/


Sample Usage
------------
Here is a simple example of FFmpeg ported to Java from this C source file:

 * https://github.com/chelyaev/ffmpeg-tutorial/blob/master/tutorial01.c

We can use [Maven 3](http://maven.apache.org/) to download and install automatically all the class files as well as the native binaries. To run this sample code, after creating the `pom.xml` and `src/main/java/Tutorial01.java` source files below, simply execute on the command line:
```bash
 $ mvn compile exec:java -Dexec.args="myvideofile.mpg"
```

### The `pom.xml` build file
```xml
<project>
    <modelVersion>4.0.0</modelVersion>
    <groupId>org.bytedeco.javacpp-presets.ffmpeg</groupId>
    <artifactId>tutorial01</artifactId>
    <version>1.2</version>
    <properties>
        <exec.mainClass>Tutorial01</exec.mainClass>
    </properties>
    <dependencies>
        <dependency>
            <groupId>org.bytedeco.javacpp-presets</groupId>
            <artifactId>ffmpeg</artifactId>
            <version>3.0.2-1.2</version>
        </dependency>
    </dependencies>
</project>
```

### The `src/main/java/Tutorial01.java` source file
```java
// tutorial01.c
//
// This tutorial was written by Stephen Dranger (dranger@gmail.com).
//
// Code based on a tutorial by Martin Bohme (boehme@inb.uni-luebeckREMOVETHIS.de)
// Tested on Gentoo, CVS version 5/01/07 compiled with GCC 4.1.1

// A small sample program that shows how to use libavformat and libavcodec to
// read video from a file.
//
// Use the Makefile to build all examples.
//
// Run using
//
// tutorial01 myvideofile.mpg
//
// to write the first five frames from "myvideofile.mpg" to disk in PPM
// format.

import java.io.*;
import org.bytedeco.javacpp.*;
import static org.bytedeco.javacpp.avcodec.*;
import static org.bytedeco.javacpp.avformat.*;
import static org.bytedeco.javacpp.avutil.*;
import static org.bytedeco.javacpp.swscale.*;

public class Tutorial01 {
    static void SaveFrame(AVFrame pFrame, int width, int height, int iFrame)
            throws IOException {
        // Open file
        OutputStream stream = new FileOutputStream("frame" + iFrame + ".ppm");

        // Write header
        stream.write(("P6\n" + width + " " + height + "\n255\n").getBytes());

        // Write pixel data
        BytePointer data = pFrame.data(0);
        byte[] bytes = new byte[width * 3];
        int l = pFrame.linesize(0);
        for(int y = 0; y < height; y++) {
            data.position(y * l).get(bytes);
            stream.write(bytes);
        }

        // Close file
        stream.close();
    }

    public static void main(String[] args) throws IOException {
        AVFormatContext pFormatCtx = new AVFormatContext(null);
        int             i, videoStream;
        AVCodecContext  pCodecCtx = null;
        AVCodec         pCodec = null;
        AVFrame         pFrame = null;
        AVFrame         pFrameRGB = null;
        AVPacket        packet = new AVPacket();
        int[]           frameFinished = new int[1];
        int             numBytes;
        BytePointer     buffer = null;

        AVDictionary    optionsDict = null;
        SwsContext      sws_ctx = null;

        if (args.length < 1) {
            System.out.println("Please provide a movie file");
            System.exit(-1);
        }
        // Register all formats and codecs
        av_register_all();

        // Open video file
        if (avformat_open_input(pFormatCtx, args[0], null, null) != 0) {
            System.exit(-1); // Couldn't open file
        }

        // Retrieve stream information
        if (avformat_find_stream_info(pFormatCtx, (PointerPointer)null) < 0) {
            System.exit(-1); // Couldn't find stream information
        }

        // Dump information about file onto standard error
        av_dump_format(pFormatCtx, 0, args[0], 0);

        // Find the first video stream
        videoStream = -1;
        for (i = 0; i < pFormatCtx.nb_streams(); i++) {
            if (pFormatCtx.streams(i).codec().codec_type() == AVMEDIA_TYPE_VIDEO) {
                videoStream = i;
                break;
            }
        }
        if (videoStream == -1) {
            System.exit(-1); // Didn't find a video stream
        }

        // Get a pointer to the codec context for the video stream
        pCodecCtx = pFormatCtx.streams(videoStream).codec();

        // Find the decoder for the video stream
        pCodec = avcodec_find_decoder(pCodecCtx.codec_id());
        if (pCodec == null) {
            System.err.println("Unsupported codec!");
            System.exit(-1); // Codec not found
        }
        // Open codec
        if (avcodec_open2(pCodecCtx, pCodec, optionsDict) < 0) {
            System.exit(-1); // Could not open codec
        }

        // Allocate video frame
        pFrame = av_frame_alloc();

        // Allocate an AVFrame structure
        pFrameRGB = av_frame_alloc();
        if(pFrameRGB == null) {
            System.exit(-1);
        }

        // Determine required buffer size and allocate buffer
        numBytes = avpicture_get_size(AV_PIX_FMT_RGB24,
                pCodecCtx.width(), pCodecCtx.height());
        buffer = new BytePointer(av_malloc(numBytes));

        sws_ctx = sws_getContext(pCodecCtx.width(), pCodecCtx.height(),
                pCodecCtx.pix_fmt(), pCodecCtx.width(), pCodecCtx.height(),
                AV_PIX_FMT_RGB24, SWS_BILINEAR, null, null, (DoublePointer)null);

        // Assign appropriate parts of buffer to image planes in pFrameRGB
        // Note that pFrameRGB is an AVFrame, but AVFrame is a superset
        // of AVPicture
        avpicture_fill(new AVPicture(pFrameRGB), buffer, AV_PIX_FMT_RGB24,
                pCodecCtx.width(), pCodecCtx.height());

        // Read frames and save first five frames to disk
        i = 0;
        while (av_read_frame(pFormatCtx, packet) >= 0) {
            // Is this a packet from the video stream?
            if (packet.stream_index() == videoStream) {
                // Decode video frame
                avcodec_decode_video2(pCodecCtx, pFrame, frameFinished, packet);

                // Did we get a video frame?
                if (frameFinished[0] != 0) {
                    // Convert the image from its native format to RGB
                    sws_scale(sws_ctx, pFrame.data(), pFrame.linesize(), 0,
                            pCodecCtx.height(), pFrameRGB.data(), pFrameRGB.linesize());

                    // Save the frame to disk
                    if (++i<=5) {
                        SaveFrame(pFrameRGB, pCodecCtx.width(), pCodecCtx.height(), i);
                    }
                }
            }

            // Free the packet that was allocated by av_read_frame
            av_free_packet(packet);
        }

        // Free the RGB image
        av_free(buffer);
        av_free(pFrameRGB);

        // Free the YUV frame
        av_free(pFrame);

        // Close the codec
        avcodec_close(pCodecCtx);

        // Close the video file
        avformat_close_input(pFormatCtx);

        System.exit(0);
    }
}
```
