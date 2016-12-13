JavaCPP Presets for FFmpeg
==========================

Introduction
------------
This directory contains the JavaCPP Presets module for:

 * FFmpeg 3.1.4  http://ffmpeg.org/

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
    <version>1.2.5-SNAPSHOT</version>
    <properties>
        <exec.mainClass>Tutorial01</exec.mainClass>
    </properties>
    <dependencies>
        <dependency>
            <groupId>org.bytedeco.javacpp-presets</groupId>
            <artifactId>ffmpeg</artifactId>
            <version>3.1.4-1.2.5-SNAPSHOT</version>
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
    private static void saveFrame(final AVFrame pFrame, final int iFrame) throws IOException {
        // Open file
        final int width = pFrame.width();
        final int height = pFrame.height();
        try (OutputStream stream = new FileOutputStream("frame" + iFrame + ".ppm")) {
            // Write header
            stream.write(("P6\n" + width + " " + height + "\n255\n").getBytes());

            // Write pixel data
            final BytePointer data = pFrame.data(0);
            final byte[] bytes = new byte[width * 3];
            final int l = pFrame.linesize(0);
            for (int y = 0; y < height; y++) {
                data.position(y * l).get(bytes);
                stream.write(bytes);
            }
        }
    }

    private static AVStream findVideoStream(final AVFormatContext pFormatCtx) {
        // Find the first video stream
        for (int i = 0; i < pFormatCtx.nb_streams(); i++) {
            final AVStream videoStream = pFormatCtx.streams(i);
            if (videoStream.codecpar().codec_type() == AVMEDIA_TYPE_VIDEO) {
                return videoStream;
            }
        }
        throw new IllegalStateException("Couldn't find a video stream");
    }

    public static void main(final String[] args) throws IOException {
        if (args.length < 1) {
            throw new IllegalStateException("Please provide a movie file");
        }
        // Register all formats and codecs
        av_register_all();

        // Open video file
        final AVFormatContext pFormatCtx = new AVFormatContext(null);
        if (avformat_open_input(pFormatCtx, args[0], null, null) != 0) {
            System.exit(-1); // Couldn't open file
        }

        // Retrieve stream information
        if (avformat_find_stream_info(pFormatCtx, (PointerPointer<?>) null) < 0) {
            System.exit(-1); // Couldn't find stream information
        }

        // Dump information about file onto standard error
        av_dump_format(pFormatCtx, 0, args[0], 0);

        // Iterate through all metadata
        for (AVDictionaryEntry tag = null; (tag = av_dict_get(pFormatCtx.metadata(), "", tag, AV_DICT_IGNORE_SUFFIX)) != null;) {
            final String key = tag.key().getString();
            final String value = tag.value().getString();
            System.out.println(key + ": " + value);
        }

        final AVStream videoStream = findVideoStream(pFormatCtx);
        final AVCodecContext pCodecCtx = videoStream.codec();
        final AVCodecParameters pCodecParameters = videoStream.codecpar();

        // Find the decoder for the video stream
        final AVCodec pCodec = avcodec_find_decoder(pCodecParameters.codec_id());
        if (pCodec == null) {
            throw new IllegalStateException("Unsupported codec!");
        }
        // Open codec
        if (avcodec_open2(pCodecCtx, pCodec, (AVDictionary) null) < 0) {
            throw new IllegalStateException("Could not open codec");
        }

        // Allocate video frame
        final AVFrame pFrameYUV = av_frame_alloc();

        // Allocate an RGB AVFrame structure
        final AVFrame pFrameRGB = av_frame_alloc();
        pFrameRGB.width(pCodecCtx.width());
        pFrameRGB.height(pCodecCtx.height());
        pFrameRGB.format(AV_PIX_FMT_RGB24);

        // Determine required buffer size and allocate buffer
        final int numBytes = av_image_get_buffer_size(pFrameRGB.format(), pFrameRGB.width(), pFrameRGB.height(), 1);
        final BytePointer buffer = new BytePointer(av_malloc(numBytes));

        final SwsContext sws_ctx = sws_getContext(pCodecParameters.width(), pCodecParameters.height(),
            pCodecParameters.format(), pFrameRGB.width(), pFrameRGB.height(),
            pFrameRGB.format(), SWS_BICUBIC, null, null, (DoublePointer) null);

        // Assign appropriate parts of buffer to image planes in pFrameRGB
        // Note that pFrameRGB is an AVFrame, but AVFrame is a superset
        // of AVPicture
        av_image_fill_arrays(pFrameRGB.data(), pFrameRGB.linesize(), buffer,
            pFrameRGB.format(), pFrameRGB.width(), pFrameRGB.height(), 1);

        // Read frames and save first five frames to disk
        try {
            final AVPacket packet = new AVPacket();
            for (int i = 0; av_read_frame(pFormatCtx, packet) >= 0;) {
                // Is this a packet from the video stream?
                if (packet.stream_index() == videoStream.index()) {
                    // Decode video frame
                    if (avcodec_send_packet(pCodecCtx, packet) != 0) {
                        throw new IllegalStateException("Frame read failed!");
                    }

                    if (avcodec_receive_frame(pCodecCtx, pFrameYUV) == 0) {
                        // Convert the image from its native format to RGB
                        sws_scale(sws_ctx, pFrameYUV.data(), pFrameYUV.linesize(), 0,
                            pCodecCtx.height(), pFrameRGB.data(), pFrameRGB.linesize());

                        // Save the frame to disk
                        if (++i <= 5) {
                            saveFrame(pFrameRGB, i);
                        }
                        else {
                            // Free the packet that was allocated by av_read_frame
                            av_packet_unref(packet);
                            return;
                        }
                    }
                }

                // Free the packet that was allocated by av_read_frame
                av_packet_unref(packet);
            }
        }
        finally {
            // Free the RGB image
            av_free(buffer);
            av_free(pFrameRGB);

            // Free the YUV frame
            av_free(pFrameYUV);

            // Close the codec
            avcodec_close(pCodecCtx);

            // Close the video file
            avformat_close_input(pFormatCtx);
        }
    }
}
```
