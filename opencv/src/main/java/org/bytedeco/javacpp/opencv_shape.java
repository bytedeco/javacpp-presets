// Targeted by JavaCPP version 1.2-SNAPSHOT

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

/**
  \defgroup shape Shape Distance and Matching
 */

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

/** \addtogroup shape
/** \{
<p>
/** \brief Computes the "minimal work" distance between two weighted point configurations base on the papers
"EMD-L1: An efficient and Robust Algorithm for comparing histogram-based descriptors", by Haibin
Ling and Kazunori Okuda; and "The Earth Mover's Distance is the Mallows Distance: Some Insights from
Statistics", by Elizaveta Levina and Peter Bickel.
<p>
@param signature1 First signature, a single column floating-point matrix. Each row is the value of
the histogram in each bin.
@param signature2 Second signature of the same format and size as signature1.
 */
@Namespace("cv") public static native float EMDL1(@ByVal Mat signature1, @ByVal Mat signature2);

/** \} */

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

/** \addtogroup shape
 *  \{
<p>
/** \brief Abstract base class for shape transformation algorithms.
 */
@Namespace("cv") public static class ShapeTransformer extends Algorithm {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public ShapeTransformer(Pointer p) { super(p); }

    /** \brief Estimate the transformation parameters of the current transformer algorithm, based on point matches.
    <p>
    @param transformingShape Contour defining first shape.
    @param targetShape Contour defining second shape (Target).
    @param matches Standard vector of Matches between points.
     */
    public native void estimateTransformation(@ByVal Mat transformingShape, @ByVal Mat targetShape,
                                                     @ByRef DMatchVector matches);

    /** \brief Apply a transformation, given a pre-estimated transformation parameters.
    <p>
    @param input Contour (set of points) to apply the transformation.
    @param output Output contour.
     */
    public native float applyTransformation(@ByVal Mat input, @ByVal(nullValue = "cv::noArray()") Mat output/*=cv::noArray()*/);
    public native float applyTransformation(@ByVal Mat input);

    /** \brief Apply a transformation, given a pre-estimated transformation parameters, to an Image.
    <p>
    @param transformingImage Input image.
    @param output Output image.
    @param flags Image interpolation method.
    @param borderMode border style.
    @param borderValue border value.
     */
    public native void warpImage(@ByVal Mat transformingImage, @ByVal Mat output,
                                       int flags/*=cv::INTER_LINEAR*/, int borderMode/*=cv::BORDER_CONSTANT*/,
                                       @Const @ByRef(nullValue = "cv::Scalar()") Scalar borderValue/*=cv::Scalar()*/);
    public native void warpImage(@ByVal Mat transformingImage, @ByVal Mat output);
}

/***********************************************************************************/
/***********************************************************************************/

/** \brief Definition of the transformation
<p>
ocupied in the paper "Principal Warps: Thin-Plate Splines and Decomposition of Deformations", by
F.L. Bookstein (PAMI 1989). :
 */
@Namespace("cv") public static class ThinPlateSplineShapeTransformer extends ShapeTransformer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public ThinPlateSplineShapeTransformer(Pointer p) { super(p); }

    /** \brief Set the regularization parameter for relaxing the exact interpolation requirements of the TPS
    algorithm.
    <p>
    @param beta value of the regularization parameter.
     */
    public native void setRegularizationParameter(double beta);
    public native double getRegularizationParameter();
}

/** Complete constructor */
@Namespace("cv") public static native @Ptr ThinPlateSplineShapeTransformer createThinPlateSplineShapeTransformer(double regularizationParameter/*=0*/);
@Namespace("cv") public static native @Ptr ThinPlateSplineShapeTransformer createThinPlateSplineShapeTransformer();

/***********************************************************************************/
/***********************************************************************************/

/** \brief Wrapper class for the OpenCV Affine Transformation algorithm. :
 */
@Namespace("cv") public static class AffineTransformer extends ShapeTransformer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public AffineTransformer(Pointer p) { super(p); }

    public native void setFullAffine(@Cast("bool") boolean fullAffine);
    public native @Cast("bool") boolean getFullAffine();
}

/** Complete constructor */
@Namespace("cv") public static native @Ptr AffineTransformer createAffineTransformer(@Cast("bool") boolean fullAffine);

/** \} */

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

/** \addtogroup shape
 *  \{
<p>
/** \brief Abstract base class for histogram cost algorithms.
 */
@Namespace("cv") public static class HistogramCostExtractor extends Algorithm {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public HistogramCostExtractor(Pointer p) { super(p); }

    public native void buildCostMatrix(@ByVal Mat descriptors1, @ByVal Mat descriptors2, @ByVal Mat costMatrix);

    public native void setNDummies(int nDummies);
    public native int getNDummies();

    public native void setDefaultCost(float defaultCost);
    public native float getDefaultCost();
}

/** \brief A norm based cost extraction. :
 */
@Namespace("cv") public static class NormHistogramCostExtractor extends HistogramCostExtractor {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public NormHistogramCostExtractor(Pointer p) { super(p); }

    public native void setNormFlag(int flag);
    public native int getNormFlag();
}

@Namespace("cv") public static native @Ptr HistogramCostExtractor createNormHistogramCostExtractor(int flag/*=cv::DIST_L2*/, int nDummies/*=25*/, float defaultCost/*=0.2f*/);
@Namespace("cv") public static native @Ptr HistogramCostExtractor createNormHistogramCostExtractor();

/** \brief An EMD based cost extraction. :
 */
@Namespace("cv") public static class EMDHistogramCostExtractor extends HistogramCostExtractor {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public EMDHistogramCostExtractor(Pointer p) { super(p); }

    public native void setNormFlag(int flag);
    public native int getNormFlag();
}

@Namespace("cv") public static native @Ptr HistogramCostExtractor createEMDHistogramCostExtractor(int flag/*=cv::DIST_L2*/, int nDummies/*=25*/, float defaultCost/*=0.2f*/);
@Namespace("cv") public static native @Ptr HistogramCostExtractor createEMDHistogramCostExtractor();

/** \brief An Chi based cost extraction. :
 */
@Namespace("cv") public static class ChiHistogramCostExtractor extends HistogramCostExtractor {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public ChiHistogramCostExtractor(Pointer p) { super(p); }
}

@Namespace("cv") public static native @Ptr HistogramCostExtractor createChiHistogramCostExtractor(int nDummies/*=25*/, float defaultCost/*=0.2f*/);
@Namespace("cv") public static native @Ptr HistogramCostExtractor createChiHistogramCostExtractor();

/** \brief An EMD-L1 based cost extraction. :
 */
@Namespace("cv") public static class EMDL1HistogramCostExtractor extends HistogramCostExtractor {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public EMDL1HistogramCostExtractor(Pointer p) { super(p); }
}

@Namespace("cv") public static native @Ptr HistogramCostExtractor createEMDL1HistogramCostExtractor(int nDummies/*=25*/, float defaultCost/*=0.2f*/);
@Namespace("cv") public static native @Ptr HistogramCostExtractor createEMDL1HistogramCostExtractor();

/** \} */

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

/** \addtogroup shape
 *  \{
<p>
/** \brief Abstract base class for shape distance algorithms.
 */
@Namespace("cv") public static class ShapeDistanceExtractor extends Algorithm {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public ShapeDistanceExtractor(Pointer p) { super(p); }

    /** \brief Compute the shape distance between two shapes defined by its contours.
    <p>
    @param contour1 Contour defining first shape.
    @param contour2 Contour defining second shape.
     */
    public native float computeDistance(@ByVal Mat contour1, @ByVal Mat contour2);
}

/***********************************************************************************/
/***********************************************************************************/
/***********************************************************************************/
/** \brief Implementation of the Shape Context descriptor and matching algorithm
<p>
proposed by Belongie et al. in "Shape Matching and Object Recognition Using Shape Contexts" (PAMI
2002). This implementation is packaged in a generic scheme, in order to allow you the
implementation of the common variations of the original pipeline.
*/
@Namespace("cv") public static class ShapeContextDistanceExtractor extends ShapeDistanceExtractor {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public ShapeContextDistanceExtractor(Pointer p) { super(p); }

    /** \brief Establish the number of angular bins for the Shape Context Descriptor used in the shape matching
    pipeline.
    <p>
    @param nAngularBins The number of angular bins in the shape context descriptor.
     */
    public native void setAngularBins(int nAngularBins);
    public native int getAngularBins();

    /** \brief Establish the number of radial bins for the Shape Context Descriptor used in the shape matching
    pipeline.
    <p>
    @param nRadialBins The number of radial bins in the shape context descriptor.
     */
    public native void setRadialBins(int nRadialBins);
    public native int getRadialBins();

    /** \brief Set the inner radius of the shape context descriptor.
    <p>
    @param innerRadius The value of the inner radius.
     */
    public native void setInnerRadius(float innerRadius);
    public native float getInnerRadius();

    /** \brief Set the outer radius of the shape context descriptor.
    <p>
    @param outerRadius The value of the outer radius.
     */
    public native void setOuterRadius(float outerRadius);
    public native float getOuterRadius();

    public native void setRotationInvariant(@Cast("bool") boolean rotationInvariant);
    public native @Cast("bool") boolean getRotationInvariant();

    /** \brief Set the weight of the shape context distance in the final value of the shape distance. The shape
    context distance between two shapes is defined as the symmetric sum of shape context matching costs
    over best matching points. The final value of the shape distance is a user-defined linear
    combination of the shape context distance, an image appearance distance, and a bending energy.
    <p>
    @param shapeContextWeight The weight of the shape context distance in the final distance value.
     */
    public native void setShapeContextWeight(float shapeContextWeight);
    public native float getShapeContextWeight();

    /** \brief Set the weight of the Image Appearance cost in the final value of the shape distance. The image
    appearance cost is defined as the sum of squared brightness differences in Gaussian windows around
    corresponding image points. The final value of the shape distance is a user-defined linear
    combination of the shape context distance, an image appearance distance, and a bending energy. If
    this value is set to a number different from 0, is mandatory to set the images that correspond to
    each shape.
    <p>
    @param imageAppearanceWeight The weight of the appearance cost in the final distance value.
     */
    public native void setImageAppearanceWeight(float imageAppearanceWeight);
    public native float getImageAppearanceWeight();

    /** \brief Set the weight of the Bending Energy in the final value of the shape distance. The bending energy
    definition depends on what transformation is being used to align the shapes. The final value of the
    shape distance is a user-defined linear combination of the shape context distance, an image
    appearance distance, and a bending energy.
    <p>
    @param bendingEnergyWeight The weight of the Bending Energy in the final distance value.
     */
    public native void setBendingEnergyWeight(float bendingEnergyWeight);
    public native float getBendingEnergyWeight();

    /** \brief Set the images that correspond to each shape. This images are used in the calculation of the Image
    Appearance cost.
    <p>
    @param image1 Image corresponding to the shape defined by contours1.
    @param image2 Image corresponding to the shape defined by contours2.
     */
    public native void setImages(@ByVal Mat image1, @ByVal Mat image2);
    public native void getImages(@ByVal Mat image1, @ByVal Mat image2);

    public native void setIterations(int iterations);
    public native int getIterations();

    /** \brief Set the algorithm used for building the shape context descriptor cost matrix.
    <p>
    @param comparer Smart pointer to a HistogramCostExtractor, an algorithm that defines the cost
    matrix between descriptors.
     */
    public native void setCostExtractor(@Ptr HistogramCostExtractor comparer);
    public native @Ptr HistogramCostExtractor getCostExtractor();

    /** \brief Set the value of the standard deviation for the Gaussian window for the image appearance cost.
    <p>
    @param sigma Standard Deviation.
     */
    public native void setStdDev(float sigma);
    public native float getStdDev();

    /** \brief Set the algorithm used for aligning the shapes.
    <p>
    @param transformer Smart pointer to a ShapeTransformer, an algorithm that defines the aligning
    transformation.
     */
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
/** \brief A simple Hausdorff distance measure between shapes defined by contours
<p>
according to the paper "Comparing Images using the Hausdorff distance." by D.P. Huttenlocher, G.A.
Klanderman, and W.J. Rucklidge. (PAMI 1993). :
 */
@Namespace("cv") public static class HausdorffDistanceExtractor extends ShapeDistanceExtractor {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public HausdorffDistanceExtractor(Pointer p) { super(p); }

    /** \brief Set the norm used to compute the Hausdorff value between two shapes. It can be L1 or L2 norm.
    <p>
    @param distanceFlag Flag indicating which norm is used to compute the Hausdorff distance
    (NORM_L1, NORM_L2).
     */
    public native void setDistanceFlag(int distanceFlag);
    public native int getDistanceFlag();

    /** \brief This method sets the rank proportion (or fractional value) that establish the Kth ranked value of
    the partial Hausdorff distance. Experimentally had been shown that 0.6 is a good value to compare
    shapes.
    <p>
    @param rankProportion fractional value (between 0 and 1).
     */
    public native void setRankProportion(float rankProportion);
    public native float getRankProportion();
}

/* Constructor */
@Namespace("cv") public static native @Ptr HausdorffDistanceExtractor createHausdorffDistanceExtractor(int distanceFlag/*=cv::NORM_L2*/, float rankProp/*=0.6f*/);
@Namespace("cv") public static native @Ptr HausdorffDistanceExtractor createHausdorffDistanceExtractor();

/** \} */

 // cv
// #endif


}
