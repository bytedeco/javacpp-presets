// Targeted by JavaCPP version 0.11

package org.bytedeco.javacpp;

import java.nio.*;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

import static org.bytedeco.javacpp.opencv_core.*;
import static org.bytedeco.javacpp.opencv_imgproc.*;
import static org.bytedeco.javacpp.opencv_video.*;

public class opencv_shape extends org.bytedeco.javacpp.presets.opencv_shape {
    static { Loader.load(); }

// Parsed from <opencv2/shape.hpp>

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

// #ifndef __OPENCV_SHAPE_HPP__
// #define __OPENCV_SHAPE_HPP__

// #include "opencv2/shape/emdL1.hpp"
// #include "opencv2/shape/shape_transformer.hpp"
// #include "opencv2/shape/hist_cost.hpp"
// #include "opencv2/shape/shape_distance.hpp"
@Namespace("cv") public static native @Cast("bool") boolean initModule_shape();


// #endif

/* End of file. */


// Parsed from <opencv2/shape/emdL1.hpp>

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

// #ifndef __OPENCV_EMD_L1_HPP__
// #define __OPENCV_EMD_L1_HPP__

// #include "opencv2/core.hpp"
/****************************************************************************************\
*                                   EMDL1 Function                                      *
\****************************************************************************************/

@Namespace("cv") public static native float EMDL1(@ByVal Mat signature1, @ByVal Mat signature2);

//namespace cv

// #endif


// Parsed from <opencv2/shape/shape_transformer.hpp>

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
// Copyright (C) 2013, OpenCV Foundation, all rights reserved.
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

// #ifndef __OPENCV_SHAPE_SHAPE_TRANSFORM_HPP__
// #define __OPENCV_SHAPE_SHAPE_TRANSFORM_HPP__
// #include <vector>
// #include "opencv2/core.hpp"
// #include "opencv2/imgproc.hpp"

/**
 * The base class for ShapeTransformer.
 * This is just to define the common interface for
 * shape transformation techniques.
 */
@Namespace("cv") public static class ShapeTransformer extends Algorithm {
    static { Loader.load(); }
    /** Empty constructor. */
    public ShapeTransformer() { }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public ShapeTransformer(Pointer p) { super(p); }

    /* Estimate, Apply Transformation and return Transforming cost*/
    public native void estimateTransformation(@ByVal Mat transformingShape, @ByVal Mat targetShape,
                                                     @StdVector DMatch matches);

    public native float applyTransformation(@ByVal Mat input, @ByVal Mat output/*=noArray()*/);
    public native float applyTransformation(@ByVal Mat input);

    public native void warpImage(@ByVal Mat transformingImage, @ByVal Mat output,
                                       int flags/*=INTER_LINEAR*/, int borderMode/*=BORDER_CONSTANT*/,
                                       @Const @ByRef Scalar borderValue/*=Scalar()*/);
    public native void warpImage(@ByVal Mat transformingImage, @ByVal Mat output);
}

/***********************************************************************************/
/***********************************************************************************/
/**
 * Thin Plate Spline Transformation
 * Implementation of the TPS transformation
 * according to "Principal Warps: Thin-Plate Splines and the
 * Decomposition of Deformations" by Juan Manuel Perez for the GSOC 2013
 */

@Namespace("cv") public static class ThinPlateSplineShapeTransformer extends ShapeTransformer {
    static { Loader.load(); }
    /** Empty constructor. */
    public ThinPlateSplineShapeTransformer() { }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public ThinPlateSplineShapeTransformer(Pointer p) { super(p); }

    public native void setRegularizationParameter(double beta);
    public native double getRegularizationParameter();
}

/* Complete constructor */
@Namespace("cv") public static native @Ptr ThinPlateSplineShapeTransformer createThinPlateSplineShapeTransformer(double regularizationParameter/*=0*/);
@Namespace("cv") public static native @Ptr ThinPlateSplineShapeTransformer createThinPlateSplineShapeTransformer();

/***********************************************************************************/
/***********************************************************************************/
/**
 * Affine Transformation as a derivated from ShapeTransformer
 */

@Namespace("cv") public static class AffineTransformer extends ShapeTransformer {
    static { Loader.load(); }
    /** Empty constructor. */
    public AffineTransformer() { }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public AffineTransformer(Pointer p) { super(p); }

    public native void setFullAffine(@Cast("bool") boolean fullAffine);
    public native @Cast("bool") boolean getFullAffine();
}

/* Complete constructor */
@Namespace("cv") public static native @Ptr AffineTransformer createAffineTransformer(@Cast("bool") boolean fullAffine);

 // cv
// #endif


// Parsed from <opencv2/shape/hist_cost.hpp>

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
// Copyright (C) 2013, OpenCV Foundation, all rights reserved.
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

// #ifndef __OPENCV_HIST_COST_HPP__
// #define __OPENCV_HIST_COST_HPP__

// #include "opencv2/imgproc.hpp"

/**
 * The base class for HistogramCostExtractor.
 */
@Namespace("cv") public static class HistogramCostExtractor extends Algorithm {
    static { Loader.load(); }
    /** Empty constructor. */
    public HistogramCostExtractor() { }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public HistogramCostExtractor(Pointer p) { super(p); }

    public native void buildCostMatrix(@ByVal Mat descriptors1, @ByVal Mat descriptors2, @ByVal Mat costMatrix);

    public native void setNDummies(int nDummies);
    public native int getNDummies();

    public native void setDefaultCost(float defaultCost);
    public native float getDefaultCost();
}

/**  */
@Namespace("cv") public static class NormHistogramCostExtractor extends HistogramCostExtractor {
    static { Loader.load(); }
    /** Empty constructor. */
    public NormHistogramCostExtractor() { }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public NormHistogramCostExtractor(Pointer p) { super(p); }

    public native void setNormFlag(int flag);
    public native int getNormFlag();
}

@Namespace("cv") public static native @Ptr HistogramCostExtractor createNormHistogramCostExtractor(int flag/*=DIST_L2*/, int nDummies/*=25*/, float defaultCost/*=0.2f*/);
@Namespace("cv") public static native @Ptr HistogramCostExtractor createNormHistogramCostExtractor();

/**  */
@Namespace("cv") public static class EMDHistogramCostExtractor extends HistogramCostExtractor {
    static { Loader.load(); }
    /** Empty constructor. */
    public EMDHistogramCostExtractor() { }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public EMDHistogramCostExtractor(Pointer p) { super(p); }

    public native void setNormFlag(int flag);
    public native int getNormFlag();
}

@Namespace("cv") public static native @Ptr HistogramCostExtractor createEMDHistogramCostExtractor(int flag/*=DIST_L2*/, int nDummies/*=25*/, float defaultCost/*=0.2f*/);
@Namespace("cv") public static native @Ptr HistogramCostExtractor createEMDHistogramCostExtractor();

/**  */
@Namespace("cv") public static class ChiHistogramCostExtractor extends HistogramCostExtractor {
    static { Loader.load(); }
    /** Empty constructor. */
    public ChiHistogramCostExtractor() { }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public ChiHistogramCostExtractor(Pointer p) { super(p); }
}

@Namespace("cv") public static native @Ptr HistogramCostExtractor createChiHistogramCostExtractor(int nDummies/*=25*/, float defaultCost/*=0.2f*/);
@Namespace("cv") public static native @Ptr HistogramCostExtractor createChiHistogramCostExtractor();

/**  */
@Namespace("cv") public static class EMDL1HistogramCostExtractor extends HistogramCostExtractor {
    static { Loader.load(); }
    /** Empty constructor. */
    public EMDL1HistogramCostExtractor() { }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public EMDL1HistogramCostExtractor(Pointer p) { super(p); }
}

@Namespace("cv") public static native @Ptr HistogramCostExtractor createEMDL1HistogramCostExtractor(int nDummies/*=25*/, float defaultCost/*=0.2f*/);
@Namespace("cv") public static native @Ptr HistogramCostExtractor createEMDL1HistogramCostExtractor();

 // cv
// #endif


// Parsed from <opencv2/shape/shape_distance.hpp>

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
// Copyright (C) 2013, OpenCV Foundation, all rights reserved.
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

// #ifndef __OPENCV_SHAPE_SHAPE_DISTANCE_HPP__
// #define __OPENCV_SHAPE_SHAPE_DISTANCE_HPP__
// #include "opencv2/core.hpp"
// #include "opencv2/shape/hist_cost.hpp"
// #include "opencv2/shape/shape_transformer.hpp"

/**
 * The base class for ShapeDistanceExtractor.
 * This is just to define the common interface for
 * shape comparisson techniques.
 */
@Namespace("cv") public static class ShapeDistanceExtractor extends Algorithm {
    static { Loader.load(); }
    /** Empty constructor. */
    public ShapeDistanceExtractor() { }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public ShapeDistanceExtractor(Pointer p) { super(p); }

    public native float computeDistance(@ByVal Mat contour1, @ByVal Mat contour2);
}

/***********************************************************************************/
/***********************************************************************************/
/***********************************************************************************/
/**
 * Shape Context implementation.
 * The SCD class implements SCD algorithm proposed by Belongie et al.in
 * "Shape Matching and Object Recognition Using Shape Contexts".
 * Implemented by Juan M. Perez for the GSOC 2013.
 */
@Namespace("cv") public static class ShapeContextDistanceExtractor extends ShapeDistanceExtractor {
    static { Loader.load(); }
    /** Empty constructor. */
    public ShapeContextDistanceExtractor() { }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public ShapeContextDistanceExtractor(Pointer p) { super(p); }

    public native void setAngularBins(int nAngularBins);
    public native int getAngularBins();

    public native void setRadialBins(int nRadialBins);
    public native int getRadialBins();

    public native void setInnerRadius(float innerRadius);
    public native float getInnerRadius();

    public native void setOuterRadius(float outerRadius);
    public native float getOuterRadius();

    public native void setRotationInvariant(@Cast("bool") boolean rotationInvariant);
    public native @Cast("bool") boolean getRotationInvariant();

    public native void setShapeContextWeight(float shapeContextWeight);
    public native float getShapeContextWeight();

    public native void setImageAppearanceWeight(float imageAppearanceWeight);
    public native float getImageAppearanceWeight();

    public native void setBendingEnergyWeight(float bendingEnergyWeight);
    public native float getBendingEnergyWeight();

    public native void setImages(@ByVal Mat image1, @ByVal Mat image2);
    public native void getImages(@ByVal Mat image1, @ByVal Mat image2);

    public native void setIterations(int iterations);
    public native int getIterations();

    public native void setCostExtractor(@Ptr HistogramCostExtractor comparer);
    public native @Ptr HistogramCostExtractor getCostExtractor();

    public native void setStdDev(float sigma);
    public native float getStdDev();

    public native void setTransformAlgorithm(@Ptr ShapeTransformer transformer);
    public native @Ptr ShapeTransformer getTransformAlgorithm();
}

/* Complete constructor */
@Namespace("cv") public static native @Ptr ShapeContextDistanceExtractor createShapeContextDistanceExtractor(int nAngularBins/*=12*/, int nRadialBins/*=4*/,
                                        float innerRadius/*=0.2f*/, float outerRadius/*=2*/, int iterations/*=3*/,
                                        @Ptr HistogramCostExtractor comparer/*=createChiHistogramCostExtractor()*/,
                                        @Ptr ShapeTransformer transformer/*=createThinPlateSplineShapeTransformer()*/);
@Namespace("cv") public static native @Ptr ShapeContextDistanceExtractor createShapeContextDistanceExtractor();

/***********************************************************************************/
/***********************************************************************************/
/***********************************************************************************/
/**
 * Hausdorff distace implementation based on
 */
@Namespace("cv") public static class HausdorffDistanceExtractor extends ShapeDistanceExtractor {
    static { Loader.load(); }
    /** Empty constructor. */
    public HausdorffDistanceExtractor() { }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public HausdorffDistanceExtractor(Pointer p) { super(p); }

    public native void setDistanceFlag(int distanceFlag);
    public native int getDistanceFlag();

    public native void setRankProportion(float rankProportion);
    public native float getRankProportion();
}

/* Constructor */
@Namespace("cv") public static native @Ptr HausdorffDistanceExtractor createHausdorffDistanceExtractor(int distanceFlag/*=cv::NORM_L2*/, float rankProp/*=0.6f*/);
@Namespace("cv") public static native @Ptr HausdorffDistanceExtractor createHausdorffDistanceExtractor();

 // cv
// #endif


}
