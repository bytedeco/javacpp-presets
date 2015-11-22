// Targeted by JavaCPP version 1.2-SNAPSHOT

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

/** \addtogroup imgcodecs_c
  \{
  */

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

/** \} imgcodecs_c */

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

/**
  \defgroup imgcodecs Image file reading and writing
  \{
    \defgroup imgcodecs_c C API
    \defgroup imgcodecs_ios iOS glue
  \}
*/

//////////////////////////////// image codec ////////////////////////////////

/** \addtogroup imgcodecs
 *  \{
 <p>
 *  Imread flags */
/** enum cv::ImreadModes */
public static final int
       /** If set, return the loaded image as is (with alpha channel, otherwise it gets cropped). */
       IMREAD_UNCHANGED  = -1,
       /** If set, always convert image to the single channel grayscale image. */
       IMREAD_GRAYSCALE  = 0,
       /** If set, always convert image to the 3 channel BGR color image. */
       IMREAD_COLOR      = 1,
       /** If set, return 16-bit/32-bit image when the input has the corresponding depth, otherwise convert it to 8-bit. */
       IMREAD_ANYDEPTH   = 2,
       /** If set, the image is read in any possible color format. */
       IMREAD_ANYCOLOR   = 4,
       /** If set, use the gdal driver for loading the image. */
       IMREAD_LOAD_GDAL  = 8;

/** Imwrite flags */
/** enum cv::ImwriteFlags */
public static final int
       /** For JPEG, it can be a quality from 0 to 100 (the higher is the better). Default value is 95. */
       IMWRITE_JPEG_QUALITY        = 1,
       /** Enable JPEG features, 0 or 1, default is False. */
       IMWRITE_JPEG_PROGRESSIVE    = 2,
       /** Enable JPEG features, 0 or 1, default is False. */
       IMWRITE_JPEG_OPTIMIZE       = 3,
       /** JPEG restart interval, 0 - 65535, default is 0 - no restart. */
       IMWRITE_JPEG_RST_INTERVAL   = 4,
       /** Separate luma quality level, 0 - 100, default is 0 - don't use. */
       IMWRITE_JPEG_LUMA_QUALITY   = 5,
       /** Separate chroma quality level, 0 - 100, default is 0 - don't use. */
       IMWRITE_JPEG_CHROMA_QUALITY = 6,
       /** For PNG, it can be the compression level from 0 to 9. A higher value means a smaller size and longer compression time. Default value is 3. */
       IMWRITE_PNG_COMPRESSION     = 16,
       /** One of cv::ImwritePNGFlags, default is IMWRITE_PNG_STRATEGY_DEFAULT. */
       IMWRITE_PNG_STRATEGY        = 17,
       /** Binary level PNG, 0 or 1, default is 0. */
       IMWRITE_PNG_BILEVEL         = 18,
       /** For PPM, PGM, or PBM, it can be a binary format flag, 0 or 1. Default value is 1. */
       IMWRITE_PXM_BINARY          = 32,
       /** For WEBP, it can be a quality from 1 to 100 (the higher is the better). By default (without any parameter) and for quality above 100 the lossless compression is used. */
       IMWRITE_WEBP_QUALITY        = 64;

/** Imwrite PNG specific flags */
/** enum cv::ImwritePNGFlags */
public static final int
       IMWRITE_PNG_STRATEGY_DEFAULT      = 0,
       IMWRITE_PNG_STRATEGY_FILTERED     = 1,
       IMWRITE_PNG_STRATEGY_HUFFMAN_ONLY = 2,
       IMWRITE_PNG_STRATEGY_RLE          = 3,
       IMWRITE_PNG_STRATEGY_FIXED        = 4;

/** \brief Loads an image from a file.
<p>
\anchor imread
<p>
@param filename Name of file to be loaded.
@param flags Flag that can take values of \ref cv::ImreadModes
<p>
The function imread loads an image from the specified file and returns it. If the image cannot be
read (because of missing file, improper permissions, unsupported or invalid format), the function
returns an empty matrix ( Mat::data==NULL ). Currently, the following file formats are supported:
<p>
-   Windows bitmaps - \*.bmp, \*.dib (always supported)
-   JPEG files - \*.jpeg, \*.jpg, \*.jpe (see the *Notes* section)
-   JPEG 2000 files - \*.jp2 (see the *Notes* section)
-   Portable Network Graphics - \*.png (see the *Notes* section)
-   WebP - \*.webp (see the *Notes* section)
-   Portable image format - \*.pbm, \*.pgm, \*.ppm (always supported)
-   Sun rasters - \*.sr, \*.ras (always supported)
-   TIFF files - \*.tiff, \*.tif (see the *Notes* section)
<p>
\note
<p>
-   The function determines the type of an image by the content, not by the file extension.
-   On Microsoft Windows\* OS and MacOSX\*, the codecs shipped with an OpenCV image (libjpeg,
    libpng, libtiff, and libjasper) are used by default. So, OpenCV can always read JPEGs, PNGs,
    and TIFFs. On MacOSX, there is also an option to use native MacOSX image readers. But beware
    that currently these native image loaders give images with different pixel values because of
    the color management embedded into MacOSX.
-   On Linux\*, BSD flavors and other Unix-like open-source operating systems, OpenCV looks for
    codecs supplied with an OS image. Install the relevant packages (do not forget the development
    files, for example, "libjpeg-dev", in Debian\* and Ubuntu\*) to get the codec support or turn
    on the OPENCV_BUILD_3RDPARTY_LIBS flag in CMake.
<p>
\note In the case of color images, the decoded images will have the channels stored in B G R order.
 */
@Namespace("cv") public static native @ByVal Mat imread( @Str BytePointer filename, int flags/*=cv::IMREAD_COLOR*/ );
@Namespace("cv") public static native @ByVal Mat imread( @Str BytePointer filename );
@Namespace("cv") public static native @ByVal Mat imread( @Str String filename, int flags/*=cv::IMREAD_COLOR*/ );
@Namespace("cv") public static native @ByVal Mat imread( @Str String filename );

/** \brief Loads a multi-page image from a file. (see imread for details.)
<p>
@param filename Name of file to be loaded.
@param flags Flag that can take values of \ref cv::ImreadModes, default with IMREAD_ANYCOLOR.
@param mats A vector of Mat objects holding each page, if more than one.
<p>
*/
@Namespace("cv") public static native @Cast("bool") boolean imreadmulti(@Str BytePointer filename, @ByRef MatVector mats, int flags/*=cv::IMREAD_ANYCOLOR*/);
@Namespace("cv") public static native @Cast("bool") boolean imreadmulti(@Str BytePointer filename, @ByRef MatVector mats);
@Namespace("cv") public static native @Cast("bool") boolean imreadmulti(@Str String filename, @ByRef MatVector mats, int flags/*=cv::IMREAD_ANYCOLOR*/);
@Namespace("cv") public static native @Cast("bool") boolean imreadmulti(@Str String filename, @ByRef MatVector mats);

/** \brief Saves an image to a specified file.
<p>
@param filename Name of the file.
@param img Image to be saved.
@param params Format-specific save parameters encoded as pairs, see \ref cv::ImwriteFlags
paramId_1, paramValue_1, paramId_2, paramValue_2, ... .
<p>
The function imwrite saves the image to the specified file. The image format is chosen based on the
filename extension (see imread for the list of extensions). Only 8-bit (or 16-bit unsigned (CV_16U)
in case of PNG, JPEG 2000, and TIFF) single-channel or 3-channel (with 'BGR' channel order) images
can be saved using this function. If the format, depth or channel order is different, use
Mat::convertTo , and cvtColor to convert it before saving. Or, use the universal FileStorage I/O
functions to save the image to XML or YAML format.
<p>
It is possible to store PNG images with an alpha channel using this function. To do this, create
8-bit (or 16-bit) 4-channel image BGRA, where the alpha channel goes last. Fully transparent pixels
should have alpha set to 0, fully opaque pixels should have alpha set to 255/65535. The sample below
shows how to create such a BGRA image and store to PNG file. It also demonstrates how to set custom
compression parameters :
<pre>{@code
    #include <vector>
    #include <stdio.h>
    #include <opencv2/opencv.hpp>

    using namespace cv;
    using namespace std;

    void createAlphaMat(Mat &mat)
    {
        CV_Assert(mat.channels() == 4);
        for (int i = 0; i < mat.rows; ++i) {
            for (int j = 0; j < mat.cols; ++j) {
                Vec4b& bgra = mat.at<Vec4b>(i, j);
                bgra[0] = UCHAR_MAX; // Blue
                bgra[1] = saturate_cast<uchar>((float (mat.cols - j)) / ((float)mat.cols) * UCHAR_MAX); // Green
                bgra[2] = saturate_cast<uchar>((float (mat.rows - i)) / ((float)mat.rows) * UCHAR_MAX); // Red
                bgra[3] = saturate_cast<uchar>(0.5 * (bgra[1] + bgra[2])); // Alpha
            }
        }
    }

    int main(int argv, char **argc)
    {
        // Create mat with alpha channel
        Mat mat(480, 640, CV_8UC4);
        createAlphaMat(mat);

        vector<int> compression_params;
        compression_params.push_back(IMWRITE_PNG_COMPRESSION);
        compression_params.push_back(9);

        try {
            imwrite("alpha.png", mat, compression_params);
        }
        catch (runtime_error& ex) {
            fprintf(stderr, "Exception converting image to PNG format: %s\n", ex.what());
            return 1;
        }

        fprintf(stdout, "Saved PNG file with alpha data.\n");
        return 0;
    }
}</pre>
 */
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

/** \overload */
@Namespace("cv") public static native @ByVal Mat imdecode( @ByVal Mat buf, int flags );

/** \brief Reads an image from a buffer in memory.
<p>
@param buf Input array or vector of bytes.
@param flags The same flags as in imread, see \ref cv::ImreadModes.
@param dst The optional output placeholder for the decoded matrix. It can save the image
reallocations when the function is called repeatedly for images of the same size.
<p>
The function reads an image from the specified buffer in the memory. If the buffer is too short or
contains invalid data, the empty matrix/image is returned.
<p>
See imread for the list of supported formats and flags description.
<p>
\note In the case of color images, the decoded images will have the channels stored in B G R order.
 */
@Namespace("cv") public static native @ByVal Mat imdecode( @ByVal Mat buf, int flags, Mat dst);

/** \brief Encodes an image into a memory buffer.
<p>
@param ext File extension that defines the output format.
@param img Image to be written.
@param buf Output buffer resized to fit the compressed image.
@param params Format-specific parameters. See imwrite and \ref cv::ImwriteFlags.
<p>
The function compresses the image and stores it in the memory buffer that is resized to fit the
result. See imwrite for the list of supported formats and flags description.
<p>
\note cvEncodeImage returns single-row matrix of type CV_8UC1 that contains encoded image as array
of bytes.
 */
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

/** \} imgcodecs */

 // cv

// #endif //__OPENCV_IMGCODECS_HPP__


}
