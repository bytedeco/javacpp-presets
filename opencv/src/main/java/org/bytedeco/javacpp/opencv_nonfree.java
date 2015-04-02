// Targeted by JavaCPP version 0.11-SNAPSHOT

package org.bytedeco.javacpp;

import java.nio.*;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

import static org.bytedeco.javacpp.opencv_core.*;
import static org.bytedeco.javacpp.opencv_imgproc.*;
import static org.bytedeco.javacpp.opencv_highgui.*;
import static org.bytedeco.javacpp.opencv_flann.*;
import static org.bytedeco.javacpp.opencv_features2d.*;
import static org.bytedeco.javacpp.opencv_calib3d.*;
import static org.bytedeco.javacpp.opencv_objdetect.*;
import static org.bytedeco.javacpp.opencv_photo.*;
import static org.bytedeco.javacpp.opencv_ml.*;
import static org.bytedeco.javacpp.opencv_video.*;
import static org.bytedeco.javacpp.opencv_legacy.*;

public class opencv_nonfree extends org.bytedeco.javacpp.presets.opencv_nonfree {
    static { Loader.load(); }

// Parsed from <opencv2/nonfree/nonfree.hpp>

/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009-2012, Willow Garage Inc., all rights reserved.
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

// #ifndef __OPENCV_NONFREE_HPP__
// #define __OPENCV_NONFREE_HPP__

// #include "opencv2/nonfree/features2d.hpp"

@Namespace("cv") public static native @Cast("bool") boolean initModule_nonfree();



// #endif

/* End of file. */


// Parsed from <opencv2/nonfree/features2d.hpp>

/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
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

// #ifndef __OPENCV_NONFREE_FEATURES_2D_HPP__
// #define __OPENCV_NONFREE_FEATURES_2D_HPP__

// #include "opencv2/features2d/features2d.hpp"

// #ifdef __cplusplus

/**
 SIFT implementation.

 The class implements SIFT algorithm by D. Lowe.
*/
@Namespace("cv") @NoOffset public static class SIFT extends Feature2D {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public SIFT(Pointer p) { super(p); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public SIFT(int size) { allocateArray(size); }
    private native void allocateArray(int size);
    @Override public SIFT position(int position) {
        return (SIFT)super.position(position);
    }

    public SIFT( int nfeatures/*=0*/, int nOctaveLayers/*=3*/,
              double contrastThreshold/*=0.04*/, double edgeThreshold/*=10*/,
              double sigma/*=1.6*/) { allocate(nfeatures, nOctaveLayers, contrastThreshold, edgeThreshold, sigma); }
    private native void allocate( int nfeatures/*=0*/, int nOctaveLayers/*=3*/,
              double contrastThreshold/*=0.04*/, double edgeThreshold/*=10*/,
              double sigma/*=1.6*/);
    public SIFT() { allocate(); }
    private native void allocate();

    /** returns the descriptor size in floats (128) */
    public native int descriptorSize();

    /** returns the descriptor type */
    public native int descriptorType();

    /** finds the keypoints using SIFT algorithm */
    public native @Name("operator()") void apply(@ByVal Mat img, @ByVal Mat mask,
                        @StdVector KeyPoint keypoints);
    /** finds the keypoints and computes descriptors for them using SIFT algorithm.
     *  Optionally it can compute descriptors for the user-provided keypoints */
    public native @Name("operator()") void apply(@ByVal Mat img, @ByVal Mat mask,
                        @StdVector KeyPoint keypoints,
                        @ByVal Mat descriptors,
                        @Cast("bool") boolean useProvidedKeypoints/*=false*/);
    public native @Name("operator()") void apply(@ByVal Mat img, @ByVal Mat mask,
                        @StdVector KeyPoint keypoints,
                        @ByVal Mat descriptors);

    public native AlgorithmInfo info();

    public native void buildGaussianPyramid( @Const @ByRef Mat base, @ByRef MatVector pyr, int nOctaves );
    public native void buildDoGPyramid( @Const @ByRef MatVector pyr, @ByRef MatVector dogpyr );
    public native void findScaleSpaceExtrema( @Const @ByRef MatVector gauss_pyr, @Const @ByRef MatVector dog_pyr,
                                    @StdVector KeyPoint keypoints );
}

/**
 SURF implementation.

 The class implements SURF algorithm by H. Bay et al.
 */
@Namespace("cv") @NoOffset public static class SURF extends Feature2D {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public SURF(Pointer p) { super(p); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public SURF(int size) { allocateArray(size); }
    private native void allocateArray(int size);
    @Override public SURF position(int position) {
        return (SURF)super.position(position);
    }

    /** the default constructor */
    public SURF() { allocate(); }
    private native void allocate();
    /** the full constructor taking all the necessary parameters */
    public SURF(double hessianThreshold,
                      int nOctaves/*=4*/, int nOctaveLayers/*=2*/,
                      @Cast("bool") boolean extended/*=true*/, @Cast("bool") boolean upright/*=false*/) { allocate(hessianThreshold, nOctaves, nOctaveLayers, extended, upright); }
    private native void allocate(double hessianThreshold,
                      int nOctaves/*=4*/, int nOctaveLayers/*=2*/,
                      @Cast("bool") boolean extended/*=true*/, @Cast("bool") boolean upright/*=false*/);
    public SURF(double hessianThreshold) { allocate(hessianThreshold); }
    private native void allocate(double hessianThreshold);

    /** returns the descriptor size in float's (64 or 128) */
    public native int descriptorSize();

    /** returns the descriptor type */
    public native int descriptorType();

    /** finds the keypoints using fast hessian detector used in SURF */
    public native @Name("operator()") void apply(@ByVal Mat img, @ByVal Mat mask,
                        @StdVector KeyPoint keypoints);
    /** finds the keypoints and computes their descriptors. Optionally it can compute descriptors for the user-provided keypoints */
    public native @Name("operator()") void apply(@ByVal Mat img, @ByVal Mat mask,
                        @StdVector KeyPoint keypoints,
                        @ByVal Mat descriptors,
                        @Cast("bool") boolean useProvidedKeypoints/*=false*/);
    public native @Name("operator()") void apply(@ByVal Mat img, @ByVal Mat mask,
                        @StdVector KeyPoint keypoints,
                        @ByVal Mat descriptors);

    public native AlgorithmInfo info();

    public native double hessianThreshold(); public native SURF hessianThreshold(double hessianThreshold);
    public native int nOctaves(); public native SURF nOctaves(int nOctaves);
    public native int nOctaveLayers(); public native SURF nOctaveLayers(int nOctaveLayers);
    public native @Cast("bool") boolean extended(); public native SURF extended(boolean extended);
    public native @Cast("bool") boolean upright(); public native SURF upright(boolean upright);
}

 /* namespace cv */

// #endif /* __cplusplus */

// #endif

/* End of file. */


}
