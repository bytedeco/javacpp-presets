// Targeted by JavaCPP version 0.11-SNAPSHOT

package org.bytedeco.javacpp;

import java.nio.*;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

import static org.bytedeco.javacpp.opencv_core.*;

public class opencv_superres extends org.bytedeco.javacpp.presets.opencv_superres {
    static { Loader.load(); }

// Parsed from <opencv2/superres/superres.hpp>

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

// #ifndef __OPENCV_SUPERRES_HPP__
// #define __OPENCV_SUPERRES_HPP__

// #include "opencv2/core/core.hpp"
        @Namespace("cv::superres") public static native @Cast("bool") boolean initModule_superres();

        @Namespace("cv::superres") public static class FrameSource extends Pointer {
            static { Loader.load(); }
            public FrameSource() { }
            public FrameSource(Pointer p) { super(p); }
        

            public native void nextFrame(@ByVal Mat frame);
            public native void reset();
        }

        @Namespace("cv::superres") public static native @Ptr FrameSource createFrameSource_Empty();

        @Namespace("cv::superres") public static native @Ptr FrameSource createFrameSource_Video(@StdString BytePointer fileName);
        @Namespace("cv::superres") public static native @Ptr FrameSource createFrameSource_Video(@StdString String fileName);
        @Namespace("cv::superres") public static native @Ptr FrameSource createFrameSource_Video_GPU(@StdString BytePointer fileName);
        @Namespace("cv::superres") public static native @Ptr FrameSource createFrameSource_Video_GPU(@StdString String fileName);

        @Namespace("cv::superres") public static native @Ptr FrameSource createFrameSource_Camera(int deviceId/*=0*/);
        @Namespace("cv::superres") public static native @Ptr FrameSource createFrameSource_Camera();

        @Namespace("cv::superres") @NoOffset public static class SuperResolution extends Algorithm {
            static { Loader.load(); }
            public SuperResolution() { }
            public SuperResolution(Pointer p) { super(p); }
            public FrameSource asFrameSource() { return asFrameSource(this); }
            @Namespace public static native @Name("static_cast<cv::superres::FrameSource*>") FrameSource asFrameSource(SuperResolution pointer);
        
            public native void setInput(@Ptr FrameSource frameSource);

            public native void nextFrame(@ByVal Mat frame);
            public native void reset();

            public native void collectGarbage();
        }

        // S. Farsiu , D. Robinson, M. Elad, P. Milanfar. Fast and robust multiframe super resolution.
        // Dennis Mitzel, Thomas Pock, Thomas Schoenemann, Daniel Cremers. Video Super Resolution using Duality Based TV-L1 Optical Flow.
        @Namespace("cv::superres") public static native @Ptr SuperResolution createSuperResolution_BTVL1();
        @Namespace("cv::superres") public static native @Ptr SuperResolution createSuperResolution_BTVL1_GPU();
        @Namespace("cv::superres") public static native @Ptr SuperResolution createSuperResolution_BTVL1_OCL();
    


// #endif // __OPENCV_SUPERRES_HPP__


// Parsed from <opencv2/superres/optical_flow.hpp>

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

// #ifndef __OPENCV_SUPERRES_OPTICAL_FLOW_HPP__
// #define __OPENCV_SUPERRES_OPTICAL_FLOW_HPP__

// #include "opencv2/core/core.hpp"
        @Namespace("cv::superres") public static class DenseOpticalFlowExt extends Algorithm {
            static { Loader.load(); }
            public DenseOpticalFlowExt() { }
            public DenseOpticalFlowExt(Pointer p) { super(p); }
        
            public native void calc(@ByVal Mat frame0, @ByVal Mat frame1, @ByVal Mat flow1, @ByVal Mat flow2/*=noArray()*/);
            public native void calc(@ByVal Mat frame0, @ByVal Mat frame1, @ByVal Mat flow1);
            public native void collectGarbage();
        }

        @Namespace("cv::superres") public static native @Ptr DenseOpticalFlowExt createOptFlow_Farneback();
        @Namespace("cv::superres") public static native @Ptr DenseOpticalFlowExt createOptFlow_Farneback_GPU();
        

        @Namespace("cv::superres") public static native @Ptr DenseOpticalFlowExt createOptFlow_Simple();

        @Namespace("cv::superres") public static native @Ptr DenseOpticalFlowExt createOptFlow_DualTVL1();
        @Namespace("cv::superres") public static native @Ptr DenseOpticalFlowExt createOptFlow_DualTVL1_GPU();
        

        @Namespace("cv::superres") public static native @Ptr DenseOpticalFlowExt createOptFlow_Brox_GPU();

        @Namespace("cv::superres") public static native @Ptr DenseOpticalFlowExt createOptFlow_PyrLK_GPU();
        
    


// #endif // __OPENCV_SUPERRES_OPTICAL_FLOW_HPP__


}
