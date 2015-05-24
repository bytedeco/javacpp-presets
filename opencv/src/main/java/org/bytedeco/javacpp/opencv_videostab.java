// Targeted by JavaCPP version 0.11

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
import static org.bytedeco.javacpp.opencv_nonfree.*;

public class opencv_videostab extends org.bytedeco.javacpp.presets.opencv_videostab {
    static { Loader.load(); }

// Parsed from <opencv2/videostab/frame_source.hpp>

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
// Copyright (C) 2009-2011, Willow Garage Inc., all rights reserved.
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

// #ifndef __OPENCV_VIDEOSTAB_FRAME_SOURCE_HPP__
// #define __OPENCV_VIDEOSTAB_FRAME_SOURCE_HPP__

// #include <vector>
// #include <string>
// #include "opencv2/core/core.hpp"
// #include "opencv2/highgui/highgui.hpp"

@Namespace("cv::videostab") public static class IFrameSource extends Pointer {
    static { Loader.load(); }
    /** Empty constructor. */
    public IFrameSource() { }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public IFrameSource(Pointer p) { super(p); }

    public native void reset();
    public native @ByVal Mat nextFrame();
}

@Namespace("cv::videostab") public static class NullFrameSource extends IFrameSource {
    static { Loader.load(); }
    /** Default native constructor. */
    public NullFrameSource() { allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public NullFrameSource(int size) { allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public NullFrameSource(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public NullFrameSource position(int position) {
        return (NullFrameSource)super.position(position);
    }

    public native void reset();
    public native @ByVal Mat nextFrame();
}

@Namespace("cv::videostab") @NoOffset public static class VideoFileSource extends IFrameSource {
    static { Loader.load(); }
    /** Empty constructor. */
    public VideoFileSource() { }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public VideoFileSource(Pointer p) { super(p); }

    public VideoFileSource(@StdString BytePointer path, @Cast("bool") boolean volatileFrame/*=false*/) { allocate(path, volatileFrame); }
    private native void allocate(@StdString BytePointer path, @Cast("bool") boolean volatileFrame/*=false*/);
    public VideoFileSource(@StdString BytePointer path) { allocate(path); }
    private native void allocate(@StdString BytePointer path);
    public VideoFileSource(@StdString String path, @Cast("bool") boolean volatileFrame/*=false*/) { allocate(path, volatileFrame); }
    private native void allocate(@StdString String path, @Cast("bool") boolean volatileFrame/*=false*/);
    public VideoFileSource(@StdString String path) { allocate(path); }
    private native void allocate(@StdString String path);

    public native void reset();
    public native @ByVal Mat nextFrame();

    public native int frameCount();
    public native double fps();
}

 // namespace videostab
 // namespace cv

// #endif


// Parsed from <opencv2/videostab/log.hpp>

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
// Copyright (C) 2009-2011, Willow Garage Inc., all rights reserved.
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

// #ifndef __OPENCV_VIDEOSTAB_LOG_HPP__
// #define __OPENCV_VIDEOSTAB_LOG_HPP__

// #include "opencv2/core/core.hpp"

@Namespace("cv::videostab") public static class ILog extends Pointer {
    static { Loader.load(); }
    /** Empty constructor. */
    public ILog() { }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public ILog(Pointer p) { super(p); }

    public native void print(@Cast("const char*") BytePointer format);
    public native void print(String format);
}

@Namespace("cv::videostab") public static class NullLog extends ILog {
    static { Loader.load(); }
    /** Default native constructor. */
    public NullLog() { allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public NullLog(int size) { allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public NullLog(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public NullLog position(int position) {
        return (NullLog)super.position(position);
    }

    public native void print(@Cast("const char*") BytePointer arg0);
    public native void print(String arg0);
}

@Namespace("cv::videostab") public static class LogToStdout extends ILog {
    static { Loader.load(); }
    /** Default native constructor. */
    public LogToStdout() { allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public LogToStdout(int size) { allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public LogToStdout(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public LogToStdout position(int position) {
        return (LogToStdout)super.position(position);
    }

    public native void print(@Cast("const char*") BytePointer format);
    public native void print(String format);
}

 // namespace videostab
 // namespace cv

// #endif


// Parsed from <opencv2/videostab/fast_marching.hpp>

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
// Copyright (C) 2009-2011, Willow Garage Inc., all rights reserved.
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

// #ifndef __OPENCV_VIDEOSTAB_FAST_MARCHING_HPP__
// #define __OPENCV_VIDEOSTAB_FAST_MARCHING_HPP__

// #include <cmath>
// #include <queue>
// #include <algorithm>
// #include "opencv2/core/core.hpp"

// See http://iwi.eldoc.ub.rug.nl/FILES/root/2004/JGraphToolsTelea/2004JGraphToolsTelea.pdf
@Namespace("cv::videostab") @NoOffset public static class FastMarchingMethod extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public FastMarchingMethod(Pointer p) { super(p); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public FastMarchingMethod(int size) { allocateArray(size); }
    private native void allocateArray(int size);
    @Override public FastMarchingMethod position(int position) {
        return (FastMarchingMethod)super.position(position);
    }

    public FastMarchingMethod() { allocate(); }
    private native void allocate();

    public native @ByVal Mat distanceMap();
}

 // namespace videostab
 // namespace cv

// #include "fast_marching_inl.hpp"

// #endif


// Parsed from <opencv2/videostab/optical_flow.hpp>

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
// Copyright (C) 2009-2011, Willow Garage Inc., all rights reserved.
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

// #ifndef __OPENCV_VIDEOSTAB_OPTICAL_FLOW_HPP__
// #define __OPENCV_VIDEOSTAB_OPTICAL_FLOW_HPP__

// #include "opencv2/core/core.hpp"
// #include "opencv2/opencv_modules.hpp"

// #if defined(HAVE_OPENCV_GPU) && !defined(ANDROID)
// #  include "opencv2/gpu/gpu.hpp"
// #endif

@Namespace("cv::videostab") public static class ISparseOptFlowEstimator extends Pointer {
    static { Loader.load(); }
    /** Empty constructor. */
    public ISparseOptFlowEstimator() { }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public ISparseOptFlowEstimator(Pointer p) { super(p); }

    public native void run(
                @ByVal Mat frame0, @ByVal Mat frame1, @ByVal Mat points0, @ByVal Mat points1,
                @ByVal Mat status, @ByVal Mat errors);
}

@Namespace("cv::videostab") public static class IDenseOptFlowEstimator extends Pointer {
    static { Loader.load(); }
    /** Empty constructor. */
    public IDenseOptFlowEstimator() { }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public IDenseOptFlowEstimator(Pointer p) { super(p); }

    public native void run(
                @ByVal Mat frame0, @ByVal Mat frame1, @ByVal Mat flowX, @ByVal Mat flowY,
                @ByVal Mat errors);
}

@Namespace("cv::videostab") @NoOffset public static class PyrLkOptFlowEstimatorBase extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public PyrLkOptFlowEstimatorBase(Pointer p) { super(p); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public PyrLkOptFlowEstimatorBase(int size) { allocateArray(size); }
    private native void allocateArray(int size);
    @Override public PyrLkOptFlowEstimatorBase position(int position) {
        return (PyrLkOptFlowEstimatorBase)super.position(position);
    }

    public PyrLkOptFlowEstimatorBase() { allocate(); }
    private native void allocate();

    public native void setWinSize(@ByVal Size val);
    public native @ByVal Size winSize();

    public native void setMaxLevel(int val);
    public native int maxLevel();
}

@Namespace("cv::videostab") public static class SparsePyrLkOptFlowEstimator extends PyrLkOptFlowEstimatorBase {
    static { Loader.load(); }
    /** Default native constructor. */
    public SparsePyrLkOptFlowEstimator() { allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public SparsePyrLkOptFlowEstimator(int size) { allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public SparsePyrLkOptFlowEstimator(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public SparsePyrLkOptFlowEstimator position(int position) {
        return (SparsePyrLkOptFlowEstimator)super.position(position);
    }
    public ISparseOptFlowEstimator asISparseOptFlowEstimator() { return asISparseOptFlowEstimator(this); }
    @Namespace public static native @Name("static_cast<cv::videostab::ISparseOptFlowEstimator*>") ISparseOptFlowEstimator asISparseOptFlowEstimator(SparsePyrLkOptFlowEstimator pointer);

    public native void run(
                @ByVal Mat frame0, @ByVal Mat frame1, @ByVal Mat points0, @ByVal Mat points1,
                @ByVal Mat status, @ByVal Mat errors);
}

// #if defined(HAVE_OPENCV_GPU) && !defined(ANDROID)
@Platform(not="android") @Namespace("cv::videostab") @NoOffset public static class DensePyrLkOptFlowEstimatorGpu extends PyrLkOptFlowEstimatorBase {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public DensePyrLkOptFlowEstimatorGpu(Pointer p) { super(p); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public DensePyrLkOptFlowEstimatorGpu(int size) { allocateArray(size); }
    private native void allocateArray(int size);
    @Override public DensePyrLkOptFlowEstimatorGpu position(int position) {
        return (DensePyrLkOptFlowEstimatorGpu)super.position(position);
    }
    public IDenseOptFlowEstimator asIDenseOptFlowEstimator() { return asIDenseOptFlowEstimator(this); }
    @Namespace public static native @Name("static_cast<cv::videostab::IDenseOptFlowEstimator*>") IDenseOptFlowEstimator asIDenseOptFlowEstimator(DensePyrLkOptFlowEstimatorGpu pointer);

    public DensePyrLkOptFlowEstimatorGpu() { allocate(); }
    private native void allocate();

    public native void run(
                @ByVal Mat frame0, @ByVal Mat frame1, @ByVal Mat flowX, @ByVal Mat flowY,
                @ByVal Mat errors);
}
// #endif

 // namespace videostab
 // namespace cv

// #endif


// Parsed from <opencv2/videostab/global_motion.hpp>

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
// Copyright (C) 2009-2011, Willow Garage Inc., all rights reserved.
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

// #ifndef __OPENCV_VIDEOSTAB_GLOBAL_MOTION_HPP__
// #define __OPENCV_VIDEOSTAB_GLOBAL_MOTION_HPP__

// #include <vector>
// #include "opencv2/core/core.hpp"
// #include "opencv2/features2d/features2d.hpp"
// #include "opencv2/videostab/optical_flow.hpp"

/** enum cv::videostab::MotionModel */
public static final int
    TRANSLATION = 0,
    TRANSLATION_AND_SCALE = 1,
    LINEAR_SIMILARITY = 2,
    AFFINE = 3;

@Namespace("cv::videostab") public static native @ByVal Mat estimateGlobalMotionLeastSquares(
        @StdVector Point2f points0, @StdVector Point2f points1,
        int model/*=cv::videostab::AFFINE*/, FloatPointer rmse/*=0*/);
@Namespace("cv::videostab") public static native @ByVal Mat estimateGlobalMotionLeastSquares(
        @StdVector Point2f points0, @StdVector Point2f points1);
@Namespace("cv::videostab") public static native @ByVal Mat estimateGlobalMotionLeastSquares(
        @StdVector Point2f points0, @StdVector Point2f points1,
        int model/*=cv::videostab::AFFINE*/, FloatBuffer rmse/*=0*/);
@Namespace("cv::videostab") public static native @ByVal Mat estimateGlobalMotionLeastSquares(
        @StdVector Point2f points0, @StdVector Point2f points1,
        int model/*=cv::videostab::AFFINE*/, float[] rmse/*=0*/);

@Namespace("cv::videostab") @NoOffset public static class RansacParams extends Pointer {
    static { Loader.load(); }
    /** Empty constructor. */
    public RansacParams() { }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public RansacParams(Pointer p) { super(p); }

    public native int size(); public native RansacParams size(int size); // subset size
    public native float thresh(); public native RansacParams thresh(float thresh); // max error to classify as inlier
    public native float eps(); public native RansacParams eps(float eps); // max outliers ratio
    public native float prob(); public native RansacParams prob(float prob); // probability of success

    public RansacParams(int _size, float _thresh, float _eps, float _prob) { allocate(_size, _thresh, _eps, _prob); }
    private native void allocate(int _size, float _thresh, float _eps, float _prob);

    public static native @ByVal RansacParams translationMotionStd();
    public static native @ByVal RansacParams translationAndScale2dMotionStd();
    public static native @ByVal RansacParams linearSimilarityMotionStd();
    public static native @ByVal RansacParams affine2dMotionStd();
}

@Namespace("cv::videostab") public static native @ByVal Mat estimateGlobalMotionRobust(
        @StdVector Point2f points0, @StdVector Point2f points1,
        int model/*=cv::videostab::AFFINE*/, @Const @ByRef(nullValue = "cv::videostab::RansacParams::affine2dMotionStd()") RansacParams params/*=cv::videostab::RansacParams::affine2dMotionStd()*/,
        FloatPointer rmse/*=0*/, IntPointer ninliers/*=0*/);
@Namespace("cv::videostab") public static native @ByVal Mat estimateGlobalMotionRobust(
        @StdVector Point2f points0, @StdVector Point2f points1);
@Namespace("cv::videostab") public static native @ByVal Mat estimateGlobalMotionRobust(
        @StdVector Point2f points0, @StdVector Point2f points1,
        int model/*=cv::videostab::AFFINE*/, @Const @ByRef(nullValue = "cv::videostab::RansacParams::affine2dMotionStd()") RansacParams params/*=cv::videostab::RansacParams::affine2dMotionStd()*/,
        FloatBuffer rmse/*=0*/, IntBuffer ninliers/*=0*/);
@Namespace("cv::videostab") public static native @ByVal Mat estimateGlobalMotionRobust(
        @StdVector Point2f points0, @StdVector Point2f points1,
        int model/*=cv::videostab::AFFINE*/, @Const @ByRef(nullValue = "cv::videostab::RansacParams::affine2dMotionStd()") RansacParams params/*=cv::videostab::RansacParams::affine2dMotionStd()*/,
        float[] rmse/*=0*/, int[] ninliers/*=0*/);

@Namespace("cv::videostab") public static class IGlobalMotionEstimator extends Pointer {
    static { Loader.load(); }
    /** Empty constructor. */
    public IGlobalMotionEstimator() { }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public IGlobalMotionEstimator(Pointer p) { super(p); }

    public native @ByVal Mat estimate(@Const @ByRef Mat frame0, @Const @ByRef Mat frame1);
}

@Namespace("cv::videostab") @NoOffset public static class PyrLkRobustMotionEstimator extends IGlobalMotionEstimator {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public PyrLkRobustMotionEstimator(Pointer p) { super(p); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public PyrLkRobustMotionEstimator(int size) { allocateArray(size); }
    private native void allocateArray(int size);
    @Override public PyrLkRobustMotionEstimator position(int position) {
        return (PyrLkRobustMotionEstimator)super.position(position);
    }

    public PyrLkRobustMotionEstimator() { allocate(); }
    private native void allocate();

    public native void setDetector(@Ptr FeatureDetector val);
    public native @Ptr FeatureDetector detector();

    public native void setOptFlowEstimator(@Ptr ISparseOptFlowEstimator val);
    public native @Ptr ISparseOptFlowEstimator optFlowEstimator();

    public native void setMotionModel(@Cast("cv::videostab::MotionModel") int val);
    public native @Cast("cv::videostab::MotionModel") int motionModel();

    public native void setRansacParams(@Const @ByRef RansacParams val);
    public native @ByVal RansacParams ransacParams();

    public native void setMaxRmse(float val);
    public native float maxRmse();

    public native void setMinInlierRatio(float val);
    public native float minInlierRatio();

    public native @ByVal Mat estimate(@Const @ByRef Mat frame0, @Const @ByRef Mat frame1);
}

@Namespace("cv::videostab") public static native @ByVal Mat getMotion(int from, int to, @Const Mat motions, int size);

@Namespace("cv::videostab") public static native @ByVal Mat getMotion(int from, int to, @Const @ByRef MatVector motions);

 // namespace videostab
 // namespace cv

// #endif


// Parsed from <opencv2/videostab/motion_stabilizing.hpp>

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
// Copyright (C) 2009-2011, Willow Garage Inc., all rights reserved.
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

// #ifndef __OPENCV_VIDEOSTAB_MOTION_STABILIZING_HPP__
// #define __OPENCV_VIDEOSTAB_MOTION_STABILIZING_HPP__

// #include <vector>
// #include "opencv2/core/core.hpp"

@Namespace("cv::videostab") public static class IMotionStabilizer extends Pointer {
    static { Loader.load(); }
    /** Empty constructor. */
    public IMotionStabilizer() { }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public IMotionStabilizer(Pointer p) { super(p); }

    public native void stabilize(@Const Mat motions, int size, Mat stabilizationMotions);

// #ifdef OPENCV_CAN_BREAK_BINARY_COMPATIBILITY
// #endif
}

@Namespace("cv::videostab") @NoOffset public static class MotionFilterBase extends IMotionStabilizer {
    static { Loader.load(); }
    /** Empty constructor. */
    public MotionFilterBase() { }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public MotionFilterBase(Pointer p) { super(p); }


    public native void setRadius(int val);
    public native int radius();

    public native void update();

    public native @ByVal Mat stabilize(int index, @Const Mat motions, int size);
    public native void stabilize(@Const Mat motions, int size, Mat stabilizationMotions);
}

@Namespace("cv::videostab") @NoOffset public static class GaussianMotionFilter extends MotionFilterBase {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public GaussianMotionFilter(Pointer p) { super(p); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public GaussianMotionFilter(int size) { allocateArray(size); }
    private native void allocateArray(int size);
    @Override public GaussianMotionFilter position(int position) {
        return (GaussianMotionFilter)super.position(position);
    }

    public GaussianMotionFilter() { allocate(); }
    private native void allocate();

    public native void setStdev(float val);
    public native float stdev();

    public native void update();

    public native @ByVal Mat stabilize(int index, @Const Mat motions, int size);
}

@Namespace("cv::videostab") public static native @ByVal Mat ensureInclusionConstraint(@Const @ByRef Mat M, @ByVal Size size, float trimRatio);

@Namespace("cv::videostab") public static native float estimateOptimalTrimRatio(@Const @ByRef Mat M, @ByVal Size size);

 // namespace videostab
 // namespace

// #endif


// Parsed from <opencv2/videostab/inpainting.hpp>

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
// Copyright (C) 2009-2011, Willow Garage Inc., all rights reserved.
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

// #ifndef __OPENCV_VIDEOSTAB_INPAINTINT_HPP__
// #define __OPENCV_VIDEOSTAB_INPAINTINT_HPP__

// #include <vector>
// #include "opencv2/core/core.hpp"
// #include "opencv2/videostab/optical_flow.hpp"
// #include "opencv2/videostab/fast_marching.hpp"
// #include "opencv2/photo/photo.hpp"

@Namespace("cv::videostab") @NoOffset public static class InpainterBase extends Pointer {
    static { Loader.load(); }
    /** Empty constructor. */
    public InpainterBase() { }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public InpainterBase(Pointer p) { super(p); }


    public native void setRadius(int val);
    public native int radius();

    public native void setFrames(@Const @ByRef MatVector val);
    public native @Const @ByRef MatVector frames();

    public native void setMotions(@Const @ByRef MatVector val);
    public native @Const @ByRef MatVector motions();

    public native void setStabilizedFrames(@Const @ByRef MatVector val);
    public native @Const @ByRef MatVector stabilizedFrames();

    public native void setStabilizationMotions(@Const @ByRef MatVector val);
    public native @Const @ByRef MatVector stabilizationMotions();

    public native void update();

    public native void inpaint(int idx, @ByRef Mat frame, @ByRef Mat mask);
}

@Namespace("cv::videostab") public static class NullInpainter extends InpainterBase {
    static { Loader.load(); }
    /** Default native constructor. */
    public NullInpainter() { allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public NullInpainter(int size) { allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public NullInpainter(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public NullInpainter position(int position) {
        return (NullInpainter)super.position(position);
    }

    public native void inpaint(int arg0, @ByRef Mat arg1, @ByRef Mat arg2);
}

@Namespace("cv::videostab") @NoOffset public static class InpaintingPipeline extends InpainterBase {
    static { Loader.load(); }
    /** Default native constructor. */
    public InpaintingPipeline() { allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public InpaintingPipeline(int size) { allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public InpaintingPipeline(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public InpaintingPipeline position(int position) {
        return (InpaintingPipeline)super.position(position);
    }

    public native void pushBack(@Ptr InpainterBase inpainter);
    public native @Cast("bool") boolean empty();

    public native void setRadius(int val);
    public native void setFrames(@Const @ByRef MatVector val);
    public native void setMotions(@Const @ByRef MatVector val);
    public native void setStabilizedFrames(@Const @ByRef MatVector val);
    public native void setStabilizationMotions(@Const @ByRef MatVector val);

    public native void update();

    public native void inpaint(int idx, @ByRef Mat frame, @ByRef Mat mask);
}

@Namespace("cv::videostab") @NoOffset public static class ConsistentMosaicInpainter extends InpainterBase {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public ConsistentMosaicInpainter(Pointer p) { super(p); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public ConsistentMosaicInpainter(int size) { allocateArray(size); }
    private native void allocateArray(int size);
    @Override public ConsistentMosaicInpainter position(int position) {
        return (ConsistentMosaicInpainter)super.position(position);
    }

    public ConsistentMosaicInpainter() { allocate(); }
    private native void allocate();

    public native void setStdevThresh(float val);
    public native float stdevThresh();

    public native void inpaint(int idx, @ByRef Mat frame, @ByRef Mat mask);
}

@Namespace("cv::videostab") @NoOffset public static class MotionInpainter extends InpainterBase {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public MotionInpainter(Pointer p) { super(p); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public MotionInpainter(int size) { allocateArray(size); }
    private native void allocateArray(int size);
    @Override public MotionInpainter position(int position) {
        return (MotionInpainter)super.position(position);
    }

    public MotionInpainter() { allocate(); }
    private native void allocate();

    public native void setOptFlowEstimator(@Ptr IDenseOptFlowEstimator val);
    public native @Ptr IDenseOptFlowEstimator optFlowEstimator();

    public native void setFlowErrorThreshold(float val);
    public native float flowErrorThreshold();

    public native void setDistThreshold(float val);
    public native float distThresh();

    public native void setBorderMode(int val);
    public native int borderMode();

    public native void inpaint(int idx, @ByRef Mat frame, @ByRef Mat mask);
}

@Namespace("cv::videostab") @NoOffset public static class ColorAverageInpainter extends InpainterBase {
    static { Loader.load(); }
    /** Default native constructor. */
    public ColorAverageInpainter() { allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public ColorAverageInpainter(int size) { allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public ColorAverageInpainter(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public ColorAverageInpainter position(int position) {
        return (ColorAverageInpainter)super.position(position);
    }

    public native void inpaint(int idx, @ByRef Mat frame, @ByRef Mat mask);
}

@Namespace("cv::videostab") @NoOffset public static class ColorInpainter extends InpainterBase {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public ColorInpainter(Pointer p) { super(p); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public ColorInpainter(int size) { allocateArray(size); }
    private native void allocateArray(int size);
    @Override public ColorInpainter position(int position) {
        return (ColorInpainter)super.position(position);
    }

    public ColorInpainter(int method/*=cv::INPAINT_TELEA*/, double _radius/*=2.*/) { allocate(method, _radius); }
    private native void allocate(int method/*=cv::INPAINT_TELEA*/, double _radius/*=2.*/);
    public ColorInpainter() { allocate(); }
    private native void allocate();

    public native void inpaint(int idx, @ByRef Mat frame, @ByRef Mat mask);
}

@Namespace("cv::videostab") public static native void calcFlowMask(
        @Const @ByRef Mat flowX, @Const @ByRef Mat flowY, @Const @ByRef Mat errors, float maxError,
        @Const @ByRef Mat mask0, @Const @ByRef Mat mask1, @ByRef Mat flowMask);

@Namespace("cv::videostab") public static native void completeFrameAccordingToFlow(
        @Const @ByRef Mat flowMask, @Const @ByRef Mat flowX, @Const @ByRef Mat flowY, @Const @ByRef Mat frame1, @Const @ByRef Mat mask1,
        float distThresh, @ByRef Mat frame0, @ByRef Mat mask0);

 // namespace videostab
 // namespace cv

// #endif


// Parsed from <opencv2/videostab/deblurring.hpp>

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
// Copyright (C) 2009-2011, Willow Garage Inc., all rights reserved.
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

// #ifndef __OPENCV_VIDEOSTAB_DEBLURRING_HPP__
// #define __OPENCV_VIDEOSTAB_DEBLURRING_HPP__

// #include <vector>
// #include "opencv2/core/core.hpp"

@Namespace("cv::videostab") public static native float calcBlurriness(@Const @ByRef Mat frame);

@Namespace("cv::videostab") @NoOffset public static class DeblurerBase extends Pointer {
    static { Loader.load(); }
    /** Empty constructor. */
    public DeblurerBase() { }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public DeblurerBase(Pointer p) { super(p); }


    public native void setRadius(int val);
    public native int radius();

    public native void setFrames(@Const @ByRef MatVector val);
    public native @Const @ByRef MatVector frames();

    public native void setMotions(@Const @ByRef MatVector val);
    public native @Const @ByRef MatVector motions();

    public native void setBlurrinessRates(@StdVector FloatPointer val);
    public native void setBlurrinessRates(@StdVector FloatBuffer val);
    public native void setBlurrinessRates(@StdVector float[] val);
    public native @StdVector FloatPointer blurrinessRates();

    public native void update();

    public native void deblur(int idx, @ByRef Mat frame);
}

@Namespace("cv::videostab") public static class NullDeblurer extends DeblurerBase {
    static { Loader.load(); }
    /** Default native constructor. */
    public NullDeblurer() { allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public NullDeblurer(int size) { allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public NullDeblurer(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public NullDeblurer position(int position) {
        return (NullDeblurer)super.position(position);
    }

    public native void deblur(int arg0, @ByRef Mat arg1);
}

@Namespace("cv::videostab") @NoOffset public static class WeightingDeblurer extends DeblurerBase {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public WeightingDeblurer(Pointer p) { super(p); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public WeightingDeblurer(int size) { allocateArray(size); }
    private native void allocateArray(int size);
    @Override public WeightingDeblurer position(int position) {
        return (WeightingDeblurer)super.position(position);
    }

    public WeightingDeblurer() { allocate(); }
    private native void allocate();

    public native void setSensitivity(float val);
    public native float sensitivity();

    public native void deblur(int idx, @ByRef Mat frame);
}

 // namespace videostab
 // namespace cv

// #endif


// Parsed from <opencv2/videostab/stabilizer.hpp>

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
// Copyright (C) 2009-2011, Willow Garage Inc., all rights reserved.
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

// #ifndef __OPENCV_VIDEOSTAB_STABILIZER_HPP__
// #define __OPENCV_VIDEOSTAB_STABILIZER_HPP__

// #include <vector>
// #include "opencv2/core/core.hpp"
// #include "opencv2/imgproc/imgproc.hpp"
// #include "opencv2/videostab/global_motion.hpp"
// #include "opencv2/videostab/motion_stabilizing.hpp"
// #include "opencv2/videostab/frame_source.hpp"
// #include "opencv2/videostab/log.hpp"
// #include "opencv2/videostab/inpainting.hpp"
// #include "opencv2/videostab/deblurring.hpp"

@Namespace("cv::videostab") @NoOffset public static class StabilizerBase extends Pointer {
    static { Loader.load(); }
    /** Empty constructor. */
    public StabilizerBase() { }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public StabilizerBase(Pointer p) { super(p); }


    public native void setLog(@Ptr ILog _log);
    public native @Ptr ILog log();

    public native void setRadius(int val);
    public native int radius();

    public native void setFrameSource(@Ptr IFrameSource val);
    public native @Ptr IFrameSource frameSource();

    public native void setMotionEstimator(@Ptr IGlobalMotionEstimator val);
    public native @Ptr IGlobalMotionEstimator motionEstimator();

    public native void setDeblurer(@Ptr DeblurerBase val);
    public native @Ptr DeblurerBase deblurrer();

    public native void setTrimRatio(float val);
    public native float trimRatio();

    public native void setCorrectionForInclusion(@Cast("bool") boolean val);
    public native @Cast("bool") boolean doCorrectionForInclusion();

    public native void setBorderMode(int val);
    public native int borderMode();

    public native void setInpainter(@Ptr InpainterBase val);
    public native @Ptr InpainterBase inpainter();
}

@Namespace("cv::videostab") @NoOffset public static class OnePassStabilizer extends StabilizerBase {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public OnePassStabilizer(Pointer p) { super(p); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public OnePassStabilizer(int size) { allocateArray(size); }
    private native void allocateArray(int size);
    @Override public OnePassStabilizer position(int position) {
        return (OnePassStabilizer)super.position(position);
    }
    public IFrameSource asIFrameSource() { return asIFrameSource(this); }
    @Namespace public static native @Name("static_cast<cv::videostab::IFrameSource*>") IFrameSource asIFrameSource(OnePassStabilizer pointer);

    public OnePassStabilizer() { allocate(); }
    private native void allocate();

    public native void setMotionFilter(@Ptr MotionFilterBase val);
    public native @Ptr MotionFilterBase motionFilter();

    public native void reset();
    public native @ByVal Mat nextFrame();
}

@Namespace("cv::videostab") @NoOffset public static class TwoPassStabilizer extends StabilizerBase {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public TwoPassStabilizer(Pointer p) { super(p); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public TwoPassStabilizer(int size) { allocateArray(size); }
    private native void allocateArray(int size);
    @Override public TwoPassStabilizer position(int position) {
        return (TwoPassStabilizer)super.position(position);
    }
    public IFrameSource asIFrameSource() { return asIFrameSource(this); }
    @Namespace public static native @Name("static_cast<cv::videostab::IFrameSource*>") IFrameSource asIFrameSource(TwoPassStabilizer pointer);

    public TwoPassStabilizer() { allocate(); }
    private native void allocate();

    public native void setMotionStabilizer(@Ptr IMotionStabilizer val);
    public native @Ptr IMotionStabilizer motionStabilizer();

    public native void setEstimateTrimRatio(@Cast("bool") boolean val);
    public native @Cast("bool") boolean mustEstimateTrimaRatio();

    public native void reset();
    public native @ByVal Mat nextFrame();

    // available after pre-pass, before it's empty
    public native @ByVal MatVector motions();
}

 // namespace videostab
 // namespace cv

// #endif


// Parsed from <opencv2/videostab/videostab.hpp>

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
// Copyright (C) 2009-2011, Willow Garage Inc., all rights reserved.
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

// #ifndef __OPENCV_VIDEOSTAB_HPP__
// #define __OPENCV_VIDEOSTAB_HPP__

// #include "opencv2/videostab/stabilizer.hpp"

// #endif


}
