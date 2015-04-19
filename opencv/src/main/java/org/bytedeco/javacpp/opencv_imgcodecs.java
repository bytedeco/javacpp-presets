// Targeted by JavaCPP version 0.11

package org.bytedeco.javacpp;

import java.nio.*;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

import static org.bytedeco.javacpp.opencv_core.*;
import static org.bytedeco.javacpp.opencv_imgproc.*;

public class opencv_imgcodecs extends org.bytedeco.javacpp.helper.opencv_imgcodecs {
    static { Loader.load(); }

// Parsed from <opencv2/imgcodecs/imgcodecs_c.h>

/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                        Intel License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of Intel Corporation may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

// #ifndef __OPENCV_IMGCODECS_H__
// #define __OPENCV_IMGCODECS_H__

// #include "opencv2/core/core_c.h"

// #ifdef __cplusplus
// #endif /* __cplusplus */

/** enum  */
public static final int
/* 8bit, color or not */
    CV_LOAD_IMAGE_UNCHANGED  = -1,
/* 8bit, gray */
    CV_LOAD_IMAGE_GRAYSCALE  = 0,
/* ?, color */
    CV_LOAD_IMAGE_COLOR      = 1,
/* any depth, ? */
    CV_LOAD_IMAGE_ANYDEPTH   = 2,
/* ?, any color */
    CV_LOAD_IMAGE_ANYCOLOR   = 4;

/* load image from file
  iscolor can be a combination of above flags where CV_LOAD_IMAGE_UNCHANGED
  overrides the other flags
  using CV_LOAD_IMAGE_ANYCOLOR alone is equivalent to CV_LOAD_IMAGE_UNCHANGED
  unless CV_LOAD_IMAGE_ANYDEPTH is specified images are converted to 8bit
*/
public static native IplImage cvLoadImage( @Cast("const char*") BytePointer filename, int iscolor/*=CV_LOAD_IMAGE_COLOR*/);
public static native IplImage cvLoadImage( @Cast("const char*") BytePointer filename);
public static native IplImage cvLoadImage( String filename, int iscolor/*=CV_LOAD_IMAGE_COLOR*/);
public static native IplImage cvLoadImage( String filename);
public static native CvMat cvLoadImageM( @Cast("const char*") BytePointer filename, int iscolor/*=CV_LOAD_IMAGE_COLOR*/);
public static native CvMat cvLoadImageM( @Cast("const char*") BytePointer filename);
public static native CvMat cvLoadImageM( String filename, int iscolor/*=CV_LOAD_IMAGE_COLOR*/);
public static native CvMat cvLoadImageM( String filename);

/** enum  */
public static final int
    CV_IMWRITE_JPEG_QUALITY = 1,
    CV_IMWRITE_JPEG_PROGRESSIVE = 2,
    CV_IMWRITE_JPEG_OPTIMIZE = 3,
    CV_IMWRITE_JPEG_RST_INTERVAL = 4,
    CV_IMWRITE_JPEG_LUMA_QUALITY = 5,
    CV_IMWRITE_JPEG_CHROMA_QUALITY = 6,
    CV_IMWRITE_PNG_COMPRESSION = 16,
    CV_IMWRITE_PNG_STRATEGY = 17,
    CV_IMWRITE_PNG_BILEVEL = 18,
    CV_IMWRITE_PNG_STRATEGY_DEFAULT = 0,
    CV_IMWRITE_PNG_STRATEGY_FILTERED = 1,
    CV_IMWRITE_PNG_STRATEGY_HUFFMAN_ONLY = 2,
    CV_IMWRITE_PNG_STRATEGY_RLE = 3,
    CV_IMWRITE_PNG_STRATEGY_FIXED = 4,
    CV_IMWRITE_PXM_BINARY = 32,
    CV_IMWRITE_WEBP_QUALITY = 64;

/* save image to file */
public static native int cvSaveImage( @Cast("const char*") BytePointer filename, @Const CvArr image,
                        @Const IntPointer params/*=0*/ );
public static native int cvSaveImage( @Cast("const char*") BytePointer filename, @Const CvArr image );
public static native int cvSaveImage( String filename, @Const CvArr image,
                        @Const IntBuffer params/*=0*/ );
public static native int cvSaveImage( String filename, @Const CvArr image );
public static native int cvSaveImage( @Cast("const char*") BytePointer filename, @Const CvArr image,
                        @Const int[] params/*=0*/ );
public static native int cvSaveImage( String filename, @Const CvArr image,
                        @Const IntPointer params/*=0*/ );
public static native int cvSaveImage( @Cast("const char*") BytePointer filename, @Const CvArr image,
                        @Const IntBuffer params/*=0*/ );
public static native int cvSaveImage( String filename, @Const CvArr image,
                        @Const int[] params/*=0*/ );

/* decode image stored in the buffer */
public static native IplImage cvDecodeImage( @Const CvMat buf, int iscolor/*=CV_LOAD_IMAGE_COLOR*/);
public static native IplImage cvDecodeImage( @Const CvMat buf);
public static native CvMat cvDecodeImageM( @Const CvMat buf, int iscolor/*=CV_LOAD_IMAGE_COLOR*/);
public static native CvMat cvDecodeImageM( @Const CvMat buf);

/* encode image and store the result as a byte vector (single-row 8uC1 matrix) */
public static native CvMat cvEncodeImage( @Cast("const char*") BytePointer ext, @Const CvArr image,
                             @Const IntPointer params/*=0*/ );
public static native CvMat cvEncodeImage( @Cast("const char*") BytePointer ext, @Const CvArr image );
public static native CvMat cvEncodeImage( String ext, @Const CvArr image,
                             @Const IntBuffer params/*=0*/ );
public static native CvMat cvEncodeImage( String ext, @Const CvArr image );
public static native CvMat cvEncodeImage( @Cast("const char*") BytePointer ext, @Const CvArr image,
                             @Const int[] params/*=0*/ );
public static native CvMat cvEncodeImage( String ext, @Const CvArr image,
                             @Const IntPointer params/*=0*/ );
public static native CvMat cvEncodeImage( @Cast("const char*") BytePointer ext, @Const CvArr image,
                             @Const IntBuffer params/*=0*/ );
public static native CvMat cvEncodeImage( String ext, @Const CvArr image,
                             @Const int[] params/*=0*/ );

/** enum  */
public static final int
    CV_CVTIMG_FLIP      = 1,
    CV_CVTIMG_SWAP_RB   = 2;

/* utility function: convert one image to another with optional vertical flip */
public static native void cvConvertImage( @Const CvArr src, CvArr dst, int flags/*=0*/);
public static native void cvConvertImage( @Const CvArr src, CvArr dst);

public static native int cvHaveImageReader(@Cast("const char*") BytePointer filename);
public static native int cvHaveImageReader(String filename);
public static native int cvHaveImageWriter(@Cast("const char*") BytePointer filename);
public static native int cvHaveImageWriter(String filename);


/****************************************************************************************\
*                              Obsolete functions/synonyms                               *
\****************************************************************************************/

public static native IplImage cvvLoadImage(@Cast("const char*") BytePointer name);
public static native IplImage cvvLoadImage(String name);
public static native int cvvSaveImage(@Cast("const char*") BytePointer arg1, CvArr arg2, IntPointer arg3);
public static native int cvvSaveImage(String arg1, CvArr arg2, IntBuffer arg3);
public static native int cvvSaveImage(@Cast("const char*") BytePointer arg1, CvArr arg2, int[] arg3);
public static native int cvvSaveImage(String arg1, CvArr arg2, IntPointer arg3);
public static native int cvvSaveImage(@Cast("const char*") BytePointer arg1, CvArr arg2, IntBuffer arg3);
public static native int cvvSaveImage(String arg1, CvArr arg2, int[] arg3);
public static native void cvvConvertImage(CvArr arg1, CvArr arg2, int arg3);


// #ifdef __cplusplus
// #endif

// #endif // __OPENCV_IMGCODECS_H__


// Parsed from <opencv2/imgcodecs.hpp>

/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                          License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

// #ifndef __OPENCV_IMGCODECS_HPP__
// #define __OPENCV_IMGCODECS_HPP__

// #include "opencv2/core.hpp"

//////////////////////////////// image codec ////////////////////////////////

/** enum cv:: */
public static final int IMREAD_UNCHANGED  = -1, // 8bit, color or not
       IMREAD_GRAYSCALE  = 0,  // 8bit, gray
       IMREAD_COLOR      = 1,  // ?, color
       IMREAD_ANYDEPTH   = 2,  // any depth, ?
       IMREAD_ANYCOLOR   = 4,  // ?, any color
       IMREAD_LOAD_GDAL  = 8;   // Use gdal driver

/** enum cv:: */
public static final int IMWRITE_JPEG_QUALITY        = 1,
       IMWRITE_JPEG_PROGRESSIVE    = 2,
       IMWRITE_JPEG_OPTIMIZE       = 3,
       IMWRITE_JPEG_RST_INTERVAL   = 4,
       IMWRITE_JPEG_LUMA_QUALITY   = 5,
       IMWRITE_JPEG_CHROMA_QUALITY = 6,
       IMWRITE_PNG_COMPRESSION     = 16,
       IMWRITE_PNG_STRATEGY        = 17,
       IMWRITE_PNG_BILEVEL         = 18,
       IMWRITE_PXM_BINARY          = 32,
       IMWRITE_WEBP_QUALITY        = 64;

/** enum cv:: */
public static final int IMWRITE_PNG_STRATEGY_DEFAULT      = 0,
       IMWRITE_PNG_STRATEGY_FILTERED     = 1,
       IMWRITE_PNG_STRATEGY_HUFFMAN_ONLY = 2,
       IMWRITE_PNG_STRATEGY_RLE          = 3,
       IMWRITE_PNG_STRATEGY_FIXED        = 4;

@Namespace("cv") public static native @ByVal Mat imread( @Str BytePointer filename, int flags/*=IMREAD_COLOR*/ );
@Namespace("cv") public static native @ByVal Mat imread( @Str BytePointer filename );
@Namespace("cv") public static native @ByVal Mat imread( @Str String filename, int flags/*=IMREAD_COLOR*/ );
@Namespace("cv") public static native @ByVal Mat imread( @Str String filename );

@Namespace("cv") public static native @Cast("bool") boolean imwrite( @Str BytePointer filename, @ByVal Mat img,
              @StdVector IntPointer params/*=std::vector<int>()*/);
@Namespace("cv") public static native @Cast("bool") boolean imwrite( @Str BytePointer filename, @ByVal Mat img);
@Namespace("cv") public static native @Cast("bool") boolean imwrite( @Str String filename, @ByVal Mat img,
              @StdVector IntBuffer params/*=std::vector<int>()*/);
@Namespace("cv") public static native @Cast("bool") boolean imwrite( @Str String filename, @ByVal Mat img);
@Namespace("cv") public static native @Cast("bool") boolean imwrite( @Str BytePointer filename, @ByVal Mat img,
              @StdVector int[] params/*=std::vector<int>()*/);
@Namespace("cv") public static native @Cast("bool") boolean imwrite( @Str String filename, @ByVal Mat img,
              @StdVector IntPointer params/*=std::vector<int>()*/);
@Namespace("cv") public static native @Cast("bool") boolean imwrite( @Str BytePointer filename, @ByVal Mat img,
              @StdVector IntBuffer params/*=std::vector<int>()*/);
@Namespace("cv") public static native @Cast("bool") boolean imwrite( @Str String filename, @ByVal Mat img,
              @StdVector int[] params/*=std::vector<int>()*/);

@Namespace("cv") public static native @ByVal Mat imdecode( @ByVal Mat buf, int flags );

@Namespace("cv") public static native @ByVal Mat imdecode( @ByVal Mat buf, int flags, Mat dst);

@Namespace("cv") public static native @Cast("bool") boolean imencode( @Str BytePointer ext, @ByVal Mat img,
                            @Cast("uchar*") @StdVector BytePointer buf,
                            @StdVector IntPointer params/*=std::vector<int>()*/);
@Namespace("cv") public static native @Cast("bool") boolean imencode( @Str BytePointer ext, @ByVal Mat img,
                            @Cast("uchar*") @StdVector BytePointer buf);
@Namespace("cv") public static native @Cast("bool") boolean imencode( @Str String ext, @ByVal Mat img,
                            @Cast("uchar*") @StdVector ByteBuffer buf,
                            @StdVector IntBuffer params/*=std::vector<int>()*/);
@Namespace("cv") public static native @Cast("bool") boolean imencode( @Str String ext, @ByVal Mat img,
                            @Cast("uchar*") @StdVector ByteBuffer buf);
@Namespace("cv") public static native @Cast("bool") boolean imencode( @Str BytePointer ext, @ByVal Mat img,
                            @Cast("uchar*") @StdVector byte[] buf,
                            @StdVector int[] params/*=std::vector<int>()*/);
@Namespace("cv") public static native @Cast("bool") boolean imencode( @Str BytePointer ext, @ByVal Mat img,
                            @Cast("uchar*") @StdVector byte[] buf);
@Namespace("cv") public static native @Cast("bool") boolean imencode( @Str String ext, @ByVal Mat img,
                            @Cast("uchar*") @StdVector BytePointer buf,
                            @StdVector IntPointer params/*=std::vector<int>()*/);
@Namespace("cv") public static native @Cast("bool") boolean imencode( @Str String ext, @ByVal Mat img,
                            @Cast("uchar*") @StdVector BytePointer buf);
@Namespace("cv") public static native @Cast("bool") boolean imencode( @Str BytePointer ext, @ByVal Mat img,
                            @Cast("uchar*") @StdVector ByteBuffer buf,
                            @StdVector IntBuffer params/*=std::vector<int>()*/);
@Namespace("cv") public static native @Cast("bool") boolean imencode( @Str BytePointer ext, @ByVal Mat img,
                            @Cast("uchar*") @StdVector ByteBuffer buf);
@Namespace("cv") public static native @Cast("bool") boolean imencode( @Str String ext, @ByVal Mat img,
                            @Cast("uchar*") @StdVector byte[] buf,
                            @StdVector int[] params/*=std::vector<int>()*/);
@Namespace("cv") public static native @Cast("bool") boolean imencode( @Str String ext, @ByVal Mat img,
                            @Cast("uchar*") @StdVector byte[] buf);

 // cv

// #endif //__OPENCV_IMGCODECS_HPP__


}
