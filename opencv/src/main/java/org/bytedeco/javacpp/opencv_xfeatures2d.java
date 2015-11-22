// Targeted by JavaCPP version 1.2-SNAPSHOT

package org.bytedeco.javacpp;

import java.nio.*;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

import static org.bytedeco.javacpp.opencv_core.*;
import static org.bytedeco.javacpp.opencv_imgproc.*;
import static org.bytedeco.javacpp.opencv_imgcodecs.*;
import static org.bytedeco.javacpp.opencv_videoio.*;
import static org.bytedeco.javacpp.opencv_highgui.*;
import static org.bytedeco.javacpp.opencv_flann.*;
import static org.bytedeco.javacpp.opencv_ml.*;
import static org.bytedeco.javacpp.opencv_features2d.*;

public class opencv_xfeatures2d extends org.bytedeco.javacpp.presets.opencv_xfeatures2d {
    static { Loader.load(); }

// Parsed from <opencv2/xfeatures2d.hpp>

/*
By downloading, copying, installing or using the software you agree to this
license. If you do not agree to this license, do not download, install,
copy or use the software.

                          License Agreement
               For Open Source Computer Vision Library
                       (3-clause BSD License)

Copyright (C) 2013, OpenCV Foundation, all rights reserved.
Third party copyrights are property of their respective owners.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

  * Redistributions of source code must retain the above copyright notice,
    this list of conditions and the following disclaimer.

  * Redistributions in binary form must reproduce the above copyright notice,
    this list of conditions and the following disclaimer in the documentation
    and/or other materials provided with the distribution.

  * Neither the names of the copyright holders nor the names of the contributors
    may be used to endorse or promote products derived from this software
    without specific prior written permission.

This software is provided by the copyright holders and contributors "as is" and
any express or implied warranties, including, but not limited to, the implied
warranties of merchantability and fitness for a particular purpose are
disclaimed. In no event shall copyright holders or contributors be liable for
any direct, indirect, incidental, special, exemplary, or consequential damages
(including, but not limited to, procurement of substitute goods or services;
loss of use, data, or profits; or business interruption) however caused
and on any theory of liability, whether in contract, strict liability,
or tort (including negligence or otherwise) arising in any way out of
the use of this software, even if advised of the possibility of such damage.
*/

// #ifndef __OPENCV_XFEATURES2D_HPP__
// #define __OPENCV_XFEATURES2D_HPP__

// #include "opencv2/features2d.hpp"
// #include "opencv2/xfeatures2d/nonfree.hpp"

/** \defgroup xfeatures2d Extra 2D Features Framework
\{
    \defgroup xfeatures2d_experiment Experimental 2D Features Algorithms
<p>
This section describes experimental algorithms for 2d feature detection.
    <p>
    \defgroup xfeatures2d_nonfree Non-free 2D Features Algorithms
<p>
This section describes two popular algorithms for 2d feature detection, SIFT and SURF, that are
known to be patented. Use them at your own risk.
<p>
\}
*/

/** \addtogroup xfeatures2d_experiment
 *  \{
<p>
/** \brief Class implementing the FREAK (*Fast Retina Keypoint*) keypoint descriptor, described in \cite AOV12 .
<p>
The algorithm propose a novel keypoint descriptor inspired by the human visual system and more
precisely the retina, coined Fast Retina Key- point (FREAK). A cascade of binary strings is
computed by efficiently comparing image intensities over a retinal sampling pattern. FREAKs are in
general faster to compute with lower memory load and also more robust than SIFT, SURF or BRISK.
They are competitive alternatives to existing keypoints in particular for embedded applications.
<p>
\note
   -   An example on how to use the FREAK descriptor can be found at
        opencv_source_code/samples/cpp/freak_demo.cpp
 */
@Namespace("cv::xfeatures2d") public static class FREAK extends Feature2D {
    static { Loader.load(); }
    /** Default native constructor. */
    public FREAK() { super((Pointer)null); allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public FREAK(int size) { super((Pointer)null); allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public FREAK(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public FREAK position(int position) {
        return (FREAK)super.position(position);
    }


    /** enum cv::xfeatures2d::FREAK:: */
    public static final int
        NB_SCALES = 64, NB_PAIRS = 512, NB_ORIENPAIRS = 45;

    /**
    @param orientationNormalized Enable orientation normalization.
    @param scaleNormalized Enable scale normalization.
    @param patternScale Scaling of the description pattern.
    @param nOctaves Number of octaves covered by the detected keypoints.
    @param selectedPairs (Optional) user defined selected pairs indexes,
     */
    public static native @Ptr FREAK create(@Cast("bool") boolean orientationNormalized/*=true*/,
                                 @Cast("bool") boolean scaleNormalized/*=true*/,
                                 float patternScale/*=22.0f*/,
                                 int nOctaves/*=4*/,
                                 @StdVector IntPointer selectedPairs/*=std::vector<int>()*/);
    public static native @Ptr FREAK create();
    public static native @Ptr FREAK create(@Cast("bool") boolean orientationNormalized/*=true*/,
                                 @Cast("bool") boolean scaleNormalized/*=true*/,
                                 float patternScale/*=22.0f*/,
                                 int nOctaves/*=4*/,
                                 @StdVector IntBuffer selectedPairs/*=std::vector<int>()*/);
    public static native @Ptr FREAK create(@Cast("bool") boolean orientationNormalized/*=true*/,
                                 @Cast("bool") boolean scaleNormalized/*=true*/,
                                 float patternScale/*=22.0f*/,
                                 int nOctaves/*=4*/,
                                 @StdVector int[] selectedPairs/*=std::vector<int>()*/);
}


/** \brief The class implements the keypoint detector introduced by \cite Agrawal08, synonym of StarDetector. :
 */
@Namespace("cv::xfeatures2d") public static class StarDetector extends Feature2D {
    static { Loader.load(); }
    /** Default native constructor. */
    public StarDetector() { super((Pointer)null); allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public StarDetector(int size) { super((Pointer)null); allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public StarDetector(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public StarDetector position(int position) {
        return (StarDetector)super.position(position);
    }

    /** the full constructor */
    public static native @Ptr StarDetector create(int maxSize/*=45*/, int responseThreshold/*=30*/,
                             int lineThresholdProjected/*=10*/,
                             int lineThresholdBinarized/*=8*/,
                             int suppressNonmaxSize/*=5*/);
    public static native @Ptr StarDetector create();
}

/*
 * BRIEF Descriptor
 */

/** \brief Class for computing BRIEF descriptors described in \cite calon2010 .
<p>
@param bytes legth of the descriptor in bytes, valid values are: 16, 32 (default) or 64 .
@param use_orientation sample patterns using keypoints orientation, disabled by default.
 <p>
 */
@Namespace("cv::xfeatures2d") public static class BriefDescriptorExtractor extends Feature2D {
    static { Loader.load(); }
    /** Default native constructor. */
    public BriefDescriptorExtractor() { super((Pointer)null); allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public BriefDescriptorExtractor(int size) { super((Pointer)null); allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public BriefDescriptorExtractor(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public BriefDescriptorExtractor position(int position) {
        return (BriefDescriptorExtractor)super.position(position);
    }

    public static native @Ptr BriefDescriptorExtractor create( int bytes/*=32*/, @Cast("bool") boolean use_orientation/*=false*/ );
    public static native @Ptr BriefDescriptorExtractor create( );
}

/** \brief Class implementing the locally uniform comparison image descriptor, described in \cite LUCID
<p>
An image descriptor that can be computed very fast, while being
about as robust as, for example, SURF or BRIEF.
 */
@Namespace("cv::xfeatures2d") public static class LUCID extends Feature2D {
    static { Loader.load(); }
    /** Default native constructor. */
    public LUCID() { super((Pointer)null); allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public LUCID(int size) { super((Pointer)null); allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public LUCID(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public LUCID position(int position) {
        return (LUCID)super.position(position);
    }

    /**
     * @param lucid_kernel kernel for descriptor construction, where 1=3x3, 2=5x5, 3=7x7 and so forth
     * @param blur_kernel kernel for blurring image prior to descriptor construction, where 1=3x3, 2=5x5, 3=7x7 and so forth
     */
    public static native @Ptr LUCID create(int lucid_kernel, int blur_kernel);
}


/*
* LATCH Descriptor
*/

/** latch Class for computing the LATCH descriptor.
If you find this code useful, please add a reference to the following paper in your work:
Gil Levi and Tal Hassner, "LATCH: Learned Arrangements of Three Patch Codes", arXiv preprint arXiv:1501.03719, 15 Jan. 2015
<p>
LATCH is a binary descriptor based on learned comparisons of triplets of image patches.
<p>
* bytes is the size of the descriptor - can be 64, 32, 16, 8, 4, 2 or 1
* rotationInvariance - whether or not the descriptor should compansate for orientation changes.
* half_ssd_size - the size of half of the mini-patches size. For example, if we would like to compare triplets of patches of size 7x7x
    then the half_ssd_size should be (7-1)/2 = 3.
<p>
Note: the descriptor can be coupled with any keypoint extractor. The only demand is that if you use set rotationInvariance = True then 
    you will have to use an extractor which estimates the patch orientation (in degrees). Examples for such extractors are ORB and SIFT.
<p>
Note: a complete example can be found under /samples/cpp/tutorial_code/xfeatures2D/latch_match.cpp
<p>
*/
@Namespace("cv::xfeatures2d") public static class LATCH extends Feature2D {
    static { Loader.load(); }
    /** Default native constructor. */
    public LATCH() { super((Pointer)null); allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public LATCH(int size) { super((Pointer)null); allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public LATCH(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public LATCH position(int position) {
        return (LATCH)super.position(position);
    }

	public static native @Ptr LATCH create(int bytes/*=32*/, @Cast("bool") boolean rotationInvariance/*=true*/, int half_ssd_size/*=3*/);
	public static native @Ptr LATCH create();
}

/** \brief Class implementing DAISY descriptor, described in \cite Tola10
<p>
@param radius radius of the descriptor at the initial scale
@param q_radius amount of radial range division quantity
@param q_theta amount of angular range division quantity
@param q_hist amount of gradient orientations range division quantity
@param norm choose descriptors normalization type, where
DAISY::NRM_NONE will not do any normalization (default),
DAISY::NRM_PARTIAL mean that histograms are normalized independently for L2 norm equal to 1.0,
DAISY::NRM_FULL mean that descriptors are normalized for L2 norm equal to 1.0,
DAISY::NRM_SIFT mean that descriptors are normalized for L2 norm equal to 1.0 but no individual one is bigger than 0.154 as in SIFT
@param H optional 3x3 homography matrix used to warp the grid of daisy but sampling keypoints remains unwarped on image
@param interpolation switch to disable interpolation for speed improvement at minor quality loss
@param use_orientation sample patterns using keypoints orientation, disabled by default.
 <p>
 */
@Namespace("cv::xfeatures2d") public static class DAISY extends Feature2D {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public DAISY(Pointer p) { super(p); }

    /** enum cv::xfeatures2d::DAISY:: */
    public static final int
        NRM_NONE = 100, NRM_PARTIAL = 101, NRM_FULL = 102, NRM_SIFT = 103;
    public static native @Ptr DAISY create( float radius/*=15*/, int q_radius/*=3*/, int q_theta/*=8*/,
                    int q_hist/*=8*/, int norm/*=cv::xfeatures2d::DAISY::NRM_NONE*/, @ByVal(nullValue = "cv::noArray()") Mat H/*=cv::noArray()*/,
                    @Cast("bool") boolean interpolation/*=true*/, @Cast("bool") boolean use_orientation/*=false*/ );
    public static native @Ptr DAISY create( );

    /** \overload
     * @param image image to extract descriptors
     * @param keypoints of interest within image
     * @param descriptors resulted descriptors array
     */
    public native void compute( @ByVal Mat image, @ByRef KeyPointVector keypoints, @ByVal Mat descriptors );

    public native void compute( @ByVal MatVector images,
                              @ByRef KeyPointVectorVector keypoints,
                              @ByVal MatVector descriptors );

    /** \overload
     * @param image image to extract descriptors
     * @param roi region of interest within image
     * @param descriptors resulted descriptors array for roi image pixels
     */
    public native void compute( @ByVal Mat image, @ByVal Rect roi, @ByVal Mat descriptors );

    /**\overload
     * @param image image to extract descriptors
     * @param descriptors resulted descriptors array for all image pixels
     */
    public native void compute( @ByVal Mat image, @ByVal Mat descriptors );

    /**
     * @param y position y on image
     * @param x position x on image
     * @param orientation orientation on image (0->360)
     * @param descriptor supplied array for descriptor storage
     */
    public native void GetDescriptor( double y, double x, int orientation, FloatPointer descriptor );
    public native void GetDescriptor( double y, double x, int orientation, FloatBuffer descriptor );
    public native void GetDescriptor( double y, double x, int orientation, float[] descriptor );

    /**
     * @param y position y on image
     * @param x position x on image
     * @param orientation orientation on image (0->360)
     * @param descriptor supplied array for descriptor storage
     * @param H homography matrix for warped grid
     */
    public native @Cast("bool") boolean GetDescriptor( double y, double x, int orientation, FloatPointer descriptor, DoublePointer H );
    public native @Cast("bool") boolean GetDescriptor( double y, double x, int orientation, FloatBuffer descriptor, DoubleBuffer H );
    public native @Cast("bool") boolean GetDescriptor( double y, double x, int orientation, float[] descriptor, double[] H );

    /**
     * @param y position y on image
     * @param x position x on image
     * @param orientation orientation on image (0->360)
     * @param descriptor supplied array for descriptor storage
     */
    public native void GetUnnormalizedDescriptor( double y, double x, int orientation, FloatPointer descriptor );
    public native void GetUnnormalizedDescriptor( double y, double x, int orientation, FloatBuffer descriptor );
    public native void GetUnnormalizedDescriptor( double y, double x, int orientation, float[] descriptor );

    /**
     * @param y position y on image
     * @param x position x on image
     * @param orientation orientation on image (0->360)
     * @param descriptor supplied array for descriptor storage
     * @param H homography matrix for warped grid
     */
    public native @Cast("bool") boolean GetUnnormalizedDescriptor( double y, double x, int orientation, FloatPointer descriptor, DoublePointer H );
    public native @Cast("bool") boolean GetUnnormalizedDescriptor( double y, double x, int orientation, FloatBuffer descriptor, DoubleBuffer H );
    public native @Cast("bool") boolean GetUnnormalizedDescriptor( double y, double x, int orientation, float[] descriptor, double[] H );

}


/** \} */




// #endif


// Parsed from <opencv2/xfeatures2d/nonfree.hpp>

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

// #ifndef __OPENCV_XFEATURES2D_FEATURES_2D_HPP__
// #define __OPENCV_XFEATURES2D_FEATURES_2D_HPP__

// #include "opencv2/features2d.hpp"

/** \addtogroup xfeatures2d_nonfree
 *  \{
<p>
/** \brief Class for extracting keypoints and computing descriptors using the Scale Invariant Feature Transform
(SIFT) algorithm by D. Lowe \cite Lowe04 .
 */
@Namespace("cv::xfeatures2d") public static class SIFT extends Feature2D {
    static { Loader.load(); }
    /** Default native constructor. */
    public SIFT() { super((Pointer)null); allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public SIFT(int size) { super((Pointer)null); allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public SIFT(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public SIFT position(int position) {
        return (SIFT)super.position(position);
    }

    /**
    @param nfeatures The number of best features to retain. The features are ranked by their scores
    (measured in SIFT algorithm as the local contrast)
    <p>
    @param nOctaveLayers The number of layers in each octave. 3 is the value used in D. Lowe paper. The
    number of octaves is computed automatically from the image resolution.
    <p>
    @param contrastThreshold The contrast threshold used to filter out weak features in semi-uniform
    (low-contrast) regions. The larger the threshold, the less features are produced by the detector.
    <p>
    @param edgeThreshold The threshold used to filter out edge-like features. Note that the its meaning
    is different from the contrastThreshold, i.e. the larger the edgeThreshold, the less features are
    filtered out (more features are retained).
    <p>
    @param sigma The sigma of the Gaussian applied to the input image at the octave \#0. If your image
    is captured with a weak camera with soft lenses, you might want to reduce the number.
     */
    public static native @Ptr SIFT create( int nfeatures/*=0*/, int nOctaveLayers/*=3*/,
                                        double contrastThreshold/*=0.04*/, double edgeThreshold/*=10*/,
                                        double sigma/*=1.6*/);
    public static native @Ptr SIFT create();
}

/** \brief Class for extracting Speeded Up Robust Features from an image \cite Bay06 .
<p>
The algorithm parameters:
-   member int extended
    -   0 means that the basic descriptors (64 elements each) shall be computed
    -   1 means that the extended descriptors (128 elements each) shall be computed
-   member int upright
    -   0 means that detector computes orientation of each feature.
    -   1 means that the orientation is not computed (which is much, much faster). For example,
if you match images from a stereo pair, or do image stitching, the matched features
likely have very similar angles, and you can speed up feature extraction by setting
upright=1.
-   member double hessianThreshold
Threshold for the keypoint detector. Only features, whose hessian is larger than
hessianThreshold are retained by the detector. Therefore, the larger the value, the less
keypoints you will get. A good default value could be from 300 to 500, depending from the
image contrast.
-   member int nOctaves
The number of a gaussian pyramid octaves that the detector uses. It is set to 4 by default.
If you want to get very large features, use the larger value. If you want just small
features, decrease it.
-   member int nOctaveLayers
The number of images within each octave of a gaussian pyramid. It is set to 2 by default.
\note
   -   An example using the SURF feature detector can be found at
        opencv_source_code/samples/cpp/generic_descriptor_match.cpp
    -   Another example using the SURF feature detector, extractor and matcher can be found at
        opencv_source_code/samples/cpp/matcher_simple.cpp
 */
@Namespace("cv::xfeatures2d") public static class SURF extends Feature2D {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public SURF(Pointer p) { super(p); }

    /**
    @param hessianThreshold Threshold for hessian keypoint detector used in SURF.
    @param nOctaves Number of pyramid octaves the keypoint detector will use.
    @param nOctaveLayers Number of octave layers within each octave.
    @param extended Extended descriptor flag (true - use extended 128-element descriptors; false - use
    64-element descriptors).
    @param upright Up-right or rotated features flag (true - do not compute orientation of features;
    false - compute orientation).
     */
    public static native @Ptr SURF create(double hessianThreshold/*=100*/,
                      int nOctaves/*=4*/, int nOctaveLayers/*=3*/,
                      @Cast("bool") boolean extended/*=false*/, @Cast("bool") boolean upright/*=false*/);
    public static native @Ptr SURF create();

    public native void setHessianThreshold(double hessianThreshold);
    public native double getHessianThreshold();

    public native void setNOctaves(int nOctaves);
    public native int getNOctaves();

    public native void setNOctaveLayers(int nOctaveLayers);
    public native int getNOctaveLayers();

    public native void setExtended(@Cast("bool") boolean extended);
    public native @Cast("bool") boolean getExtended();

    public native void setUpright(@Cast("bool") boolean upright);
    public native @Cast("bool") boolean getUpright();
}

/** \} */


 /* namespace cv */

// #endif


}
