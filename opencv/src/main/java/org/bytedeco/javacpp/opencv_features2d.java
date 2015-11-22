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

public class opencv_features2d extends org.bytedeco.javacpp.presets.opencv_features2d {
    static { Loader.load(); }

// Parsed from <opencv2/features2d.hpp>

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

// #ifndef __OPENCV_FEATURES_2D_HPP__
// #define __OPENCV_FEATURES_2D_HPP__

// #include "opencv2/core.hpp"
// #include "opencv2/flann/miniflann.hpp"

/**
  \defgroup features2d 2D Features Framework
  \{
    \defgroup features2d_main Feature Detection and Description
    \defgroup features2d_match Descriptor Matchers
<p>
Matchers of keypoint descriptors in OpenCV have wrappers with a common interface that enables you to
easily switch between different algorithms solving the same problem. This section is devoted to
matching descriptors that are represented as vectors in a multidimensional space. All objects that
implement vector descriptor matchers inherit the DescriptorMatcher interface.
<p>
\note
   -   An example explaining keypoint matching can be found at
        opencv_source_code/samples/cpp/descriptor_extractor_matcher.cpp
    -   An example on descriptor matching evaluation can be found at
        opencv_source_code/samples/cpp/detector_descriptor_matcher_evaluation.cpp
    -   An example on one to many image matching can be found at
        opencv_source_code/samples/cpp/matching_to_many_images.cpp
    <p>
    \defgroup features2d_draw Drawing Function of Keypoints and Matches
    \defgroup features2d_category Object Categorization
<p>
This section describes approaches based on local 2D features and used to categorize objects.
<p>
\note
   -   A complete Bag-Of-Words sample can be found at
        opencv_source_code/samples/cpp/bagofwords_classification.cpp
    -   (Python) An example using the features2D framework to perform object categorization can be
        found at opencv_source_code/samples/python2/find_obj.py
  <p>
  \}
 */

/** \addtogroup features2d
 *  \{ */

// //! writes vector of keypoints to the file storage
// CV_EXPORTS void write(FileStorage& fs, const String& name, const std::vector<KeyPoint>& keypoints);
// //! reads vector of keypoints from the specified file storage node
// CV_EXPORTS void read(const FileNode& node, CV_OUT std::vector<KeyPoint>& keypoints);

/** \brief A class filters a vector of keypoints.
 <p>
 Because now it is difficult to provide a convenient interface for all usage scenarios of the
 keypoints filter class, it has only several needed by now static methods.
 */
@Namespace("cv") public static class KeyPointsFilter extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public KeyPointsFilter(Pointer p) { super(p); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public KeyPointsFilter(int size) { super((Pointer)null); allocateArray(size); }
    private native void allocateArray(int size);
    @Override public KeyPointsFilter position(int position) {
        return (KeyPointsFilter)super.position(position);
    }

    public KeyPointsFilter() { super((Pointer)null); allocate(); }
    private native void allocate();

    /*
     * Remove keypoints within borderPixels of an image edge.
     */
    public static native void runByImageBorder( @ByRef KeyPointVector keypoints, @ByVal Size imageSize, int borderSize );
    /*
     * Remove keypoints of sizes out of range.
     */
    public static native void runByKeypointSize( @ByRef KeyPointVector keypoints, float minSize,
                                       float maxSize/*=FLT_MAX*/ );
    public static native void runByKeypointSize( @ByRef KeyPointVector keypoints, float minSize );
    /*
     * Remove keypoints from some image by mask for pixels of this image.
     */
    public static native void runByPixelsMask( @ByRef KeyPointVector keypoints, @Const @ByRef Mat mask );
    /*
     * Remove duplicated keypoints.
     */
    public static native void removeDuplicated( @ByRef KeyPointVector keypoints );

    /*
     * Retain the specified number of the best keypoints (according to the response)
     */
    public static native void retainBest( @ByRef KeyPointVector keypoints, int npoints );
}


/************************************ Base Classes ************************************/

/** \brief Abstract base class for 2D image feature detectors and descriptor extractors
*/
@Namespace("cv") public static class Feature2D extends Algorithm {
    static { Loader.load(); }
    /** Default native constructor. */
    public Feature2D() { super((Pointer)null); allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public Feature2D(int size) { super((Pointer)null); allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public Feature2D(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public Feature2D position(int position) {
        return (Feature2D)super.position(position);
    }


    /** \brief Detects keypoints in an image (first variant) or image set (second variant).
    <p>
    @param image Image.
    @param keypoints The detected keypoints. In the second variant of the method keypoints[i] is a set
    of keypoints detected in images[i] .
    @param mask Mask specifying where to look for keypoints (optional). It must be a 8-bit integer
    matrix with non-zero values in the region of interest.
     */
    public native void detect( @ByVal Mat image,
                                     @ByRef KeyPointVector keypoints,
                                     @ByVal(nullValue = "cv::noArray()") Mat mask/*=cv::noArray()*/ );
    public native void detect( @ByVal Mat image,
                                     @ByRef KeyPointVector keypoints );

    /** \overload
    @param images Image set.
    @param keypoints The detected keypoints. In the second variant of the method keypoints[i] is a set
    of keypoints detected in images[i] .
    @param masks Masks for each input image specifying where to look for keypoints (optional).
    masks[i] is a mask for images[i].
    */
    public native void detect( @ByVal MatVector images,
                             @ByRef KeyPointVectorVector keypoints,
                             @ByVal(nullValue = "cv::noArray()") MatVector masks/*=cv::noArray()*/ );
    public native void detect( @ByVal MatVector images,
                             @ByRef KeyPointVectorVector keypoints );

    /** \brief Computes the descriptors for a set of keypoints detected in an image (first variant) or image set
    (second variant).
    <p>
    @param image Image.
    @param keypoints Input collection of keypoints. Keypoints for which a descriptor cannot be
    computed are removed. Sometimes new keypoints can be added, for example: SIFT duplicates keypoint
    with several dominant orientations (for each orientation).
    @param descriptors Computed descriptors. In the second variant of the method descriptors[i] are
    descriptors computed for a keypoints[i]. Row j is the keypoints (or keypoints[i]) is the
    descriptor for keypoint j-th keypoint.
     */
    public native void compute( @ByVal Mat image,
                                      @ByRef KeyPointVector keypoints,
                                      @ByVal Mat descriptors );

    /** \overload
    <p>
    @param images Image set.
    @param keypoints Input collection of keypoints. Keypoints for which a descriptor cannot be
    computed are removed. Sometimes new keypoints can be added, for example: SIFT duplicates keypoint
    with several dominant orientations (for each orientation).
    @param descriptors Computed descriptors. In the second variant of the method descriptors[i] are
    descriptors computed for a keypoints[i]. Row j is the keypoints (or keypoints[i]) is the
    descriptor for keypoint j-th keypoint.
    */
    public native void compute( @ByVal MatVector images,
                              @ByRef KeyPointVectorVector keypoints,
                              @ByVal MatVector descriptors );

    /** Detects keypoints and computes the descriptors */
    public native void detectAndCompute( @ByVal Mat image, @ByVal Mat mask,
                                               @ByRef KeyPointVector keypoints,
                                               @ByVal Mat descriptors,
                                               @Cast("bool") boolean useProvidedKeypoints/*=false*/ );
    public native void detectAndCompute( @ByVal Mat image, @ByVal Mat mask,
                                               @ByRef KeyPointVector keypoints,
                                               @ByVal Mat descriptors );

    public native int descriptorSize();
    public native int descriptorType();
    public native int defaultNorm();

    /** Return true if detector object is empty */
    public native @Cast("bool") boolean empty();
}

/** Feature detectors in OpenCV have wrappers with a common interface that enables you to easily switch
between different algorithms solving the same problem. All objects that implement keypoint detectors
inherit the FeatureDetector interface. */

/** Extractors of keypoint descriptors in OpenCV have wrappers with a common interface that enables you
to easily switch between different algorithms solving the same problem. This section is devoted to
computing descriptors represented as vectors in a multidimensional space. All objects that implement
the vector descriptor extractors inherit the DescriptorExtractor interface.
 */

/** \addtogroup features2d_main
 *  \{
<p>
/** \brief Class implementing the BRISK keypoint detector and descriptor extractor, described in \cite LCS11 .
 */
@Namespace("cv") public static class BRISK extends Feature2D {
    static { Loader.load(); }
    /** Default native constructor. */
    public BRISK() { super((Pointer)null); allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public BRISK(int size) { super((Pointer)null); allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public BRISK(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public BRISK position(int position) {
        return (BRISK)super.position(position);
    }

    /** \brief The BRISK constructor
    <p>
    @param thresh AGAST detection threshold score.
    @param octaves detection octaves. Use 0 to do single scale.
    @param patternScale apply this scale to the pattern used for sampling the neighbourhood of a
    keypoint.
     */
    public static native @Ptr BRISK create(int thresh/*=30*/, int octaves/*=3*/, float patternScale/*=1.0f*/);
    public static native @Ptr BRISK create();

    /** \brief The BRISK constructor for a custom pattern
    <p>
    @param radiusList defines the radii (in pixels) where the samples around a keypoint are taken (for
    keypoint scale 1).
    @param numberList defines the number of sampling points on the sampling circle. Must be the same
    size as radiusList..
    @param dMax threshold for the short pairings used for descriptor formation (in pixels for keypoint
    scale 1).
    @param dMin threshold for the long pairings used for orientation determination (in pixels for
    keypoint scale 1).
    @param indexChange index remapping of the bits. */
    public static native @Ptr BRISK create(@StdVector FloatPointer radiusList, @StdVector IntPointer numberList,
            float dMax/*=5.85f*/, float dMin/*=8.2f*/, @StdVector IntPointer indexChange/*=std::vector<int>()*/);
    public static native @Ptr BRISK create(@StdVector FloatPointer radiusList, @StdVector IntPointer numberList);
    public static native @Ptr BRISK create(@StdVector FloatBuffer radiusList, @StdVector IntBuffer numberList,
            float dMax/*=5.85f*/, float dMin/*=8.2f*/, @StdVector IntBuffer indexChange/*=std::vector<int>()*/);
    public static native @Ptr BRISK create(@StdVector FloatBuffer radiusList, @StdVector IntBuffer numberList);
    public static native @Ptr BRISK create(@StdVector float[] radiusList, @StdVector int[] numberList,
            float dMax/*=5.85f*/, float dMin/*=8.2f*/, @StdVector int[] indexChange/*=std::vector<int>()*/);
    public static native @Ptr BRISK create(@StdVector float[] radiusList, @StdVector int[] numberList);
}

/** \brief Class implementing the ORB (*oriented BRIEF*) keypoint detector and descriptor extractor
<p>
described in \cite RRKB11 . The algorithm uses FAST in pyramids to detect stable keypoints, selects
the strongest features using FAST or Harris response, finds their orientation using first-order
moments and computes the descriptors using BRIEF (where the coordinates of random point pairs (or
k-tuples) are rotated according to the measured orientation).
 */
@Namespace("cv") public static class ORB extends Feature2D {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public ORB(Pointer p) { super(p); }

    /** enum cv::ORB:: */
    public static final int kBytes = 32, HARRIS_SCORE= 0, FAST_SCORE= 1;

    /** \brief The ORB constructor
    <p>
    @param nfeatures The maximum number of features to retain.
    @param scaleFactor Pyramid decimation ratio, greater than 1. scaleFactor==2 means the classical
    pyramid, where each next level has 4x less pixels than the previous, but such a big scale factor
    will degrade feature matching scores dramatically. On the other hand, too close to 1 scale factor
    will mean that to cover certain scale range you will need more pyramid levels and so the speed
    will suffer.
    @param nlevels The number of pyramid levels. The smallest level will have linear size equal to
    input_image_linear_size/pow(scaleFactor, nlevels).
    @param edgeThreshold This is size of the border where the features are not detected. It should
    roughly match the patchSize parameter.
    @param firstLevel It should be 0 in the current implementation.
    @param WTA_K The number of points that produce each element of the oriented BRIEF descriptor. The
    default value 2 means the BRIEF where we take a random point pair and compare their brightnesses,
    so we get 0/1 response. Other possible values are 3 and 4. For example, 3 means that we take 3
    random points (of course, those point coordinates are random, but they are generated from the
    pre-defined seed, so each element of BRIEF descriptor is computed deterministically from the pixel
    rectangle), find point of maximum brightness and output index of the winner (0, 1 or 2). Such
    output will occupy 2 bits, and therefore it will need a special variant of Hamming distance,
    denoted as NORM_HAMMING2 (2 bits per bin). When WTA_K=4, we take 4 random points to compute each
    bin (that will also occupy 2 bits with possible values 0, 1, 2 or 3).
    @param scoreType The default HARRIS_SCORE means that Harris algorithm is used to rank features
    (the score is written to KeyPoint::score and is used to retain best nfeatures features);
    FAST_SCORE is alternative value of the parameter that produces slightly less stable keypoints,
    but it is a little faster to compute.
    @param patchSize size of the patch used by the oriented BRIEF descriptor. Of course, on smaller
    pyramid layers the perceived image area covered by a feature will be larger.
    @param fastThreshold
     */
    public static native @Ptr ORB create(int nfeatures/*=500*/, float scaleFactor/*=1.2f*/, int nlevels/*=8*/, int edgeThreshold/*=31*/,
            int firstLevel/*=0*/, int WTA_K/*=2*/, int scoreType/*=cv::ORB::HARRIS_SCORE*/, int patchSize/*=31*/, int fastThreshold/*=20*/);
    public static native @Ptr ORB create();

    public native void setMaxFeatures(int maxFeatures);
    public native int getMaxFeatures();

    public native void setScaleFactor(double scaleFactor);
    public native double getScaleFactor();

    public native void setNLevels(int nlevels);
    public native int getNLevels();

    public native void setEdgeThreshold(int edgeThreshold);
    public native int getEdgeThreshold();

    public native void setFirstLevel(int firstLevel);
    public native int getFirstLevel();

    public native void setWTA_K(int wta_k);
    public native int getWTA_K();

    public native void setScoreType(int scoreType);
    public native int getScoreType();

    public native void setPatchSize(int patchSize);
    public native int getPatchSize();

    public native void setFastThreshold(int fastThreshold);
    public native int getFastThreshold();
}

/** \brief Maximally stable extremal region extractor. :
<p>
The class encapsulates all the parameters of the MSER extraction algorithm (see
<http://en.wikipedia.org/wiki/Maximally_stable_extremal_regions>). Also see
<http://code.opencv.org/projects/opencv/wiki/MSER> for useful comments and parameters description.
<p>
\note
   -   (Python) A complete example showing the use of the MSER detector can be found at
        opencv_source_code/samples/python2/mser.py
 */
@Namespace("cv") public static class MSER extends Feature2D {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public MSER(Pointer p) { super(p); }

    /** the full constructor */
    public static native @Ptr MSER create( int _delta/*=5*/, int _min_area/*=60*/, int _max_area/*=14400*/,
              double _max_variation/*=0.25*/, double _min_diversity/*=.2*/,
              int _max_evolution/*=200*/, double _area_threshold/*=1.01*/,
              double _min_margin/*=0.003*/, int _edge_blur_size/*=5*/ );
    public static native @Ptr MSER create( );

    public native void detectRegions( @ByVal Mat image,
                                            @ByRef PointVectorVector msers,
                                            @ByRef RectVector bboxes );

    public native void setDelta(int delta);
    public native int getDelta();

    public native void setMinArea(int minArea);
    public native int getMinArea();

    public native void setMaxArea(int maxArea);
    public native int getMaxArea();

    public native void setPass2Only(@Cast("bool") boolean f);
    public native @Cast("bool") boolean getPass2Only();
}

/** \overload */
@Namespace("cv") public static native void FAST( @ByVal Mat image, @ByRef KeyPointVector keypoints,
                      int threshold, @Cast("bool") boolean nonmaxSuppression/*=true*/ );
@Namespace("cv") public static native void FAST( @ByVal Mat image, @ByRef KeyPointVector keypoints,
                      int threshold );

/** \brief Detects corners using the FAST algorithm
<p>
@param image grayscale image where keypoints (corners) are detected.
@param keypoints keypoints detected on the image.
@param threshold threshold on difference between intensity of the central pixel and pixels of a
circle around this pixel.
@param nonmaxSuppression if true, non-maximum suppression is applied to detected corners
(keypoints).
@param type one of the three neighborhoods as defined in the paper:
FastFeatureDetector::TYPE_9_16, FastFeatureDetector::TYPE_7_12,
FastFeatureDetector::TYPE_5_8
<p>
Detects corners using the FAST algorithm by \cite Rosten06 .
<p>
\note In Python API, types are given as cv2.FAST_FEATURE_DETECTOR_TYPE_5_8,
cv2.FAST_FEATURE_DETECTOR_TYPE_7_12 and cv2.FAST_FEATURE_DETECTOR_TYPE_9_16. For corner
detection, use cv2.FAST.detect() method.
 */
@Namespace("cv") public static native void FAST( @ByVal Mat image, @ByRef KeyPointVector keypoints,
                      int threshold, @Cast("bool") boolean nonmaxSuppression, int type );

/** \} features2d_main
 <p>
 *  \addtogroup features2d_main
 *  \{
<p>
/** \brief Wrapping class for feature detection using the FAST method. :
 */
@Namespace("cv") public static class FastFeatureDetector extends Feature2D {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public FastFeatureDetector(Pointer p) { super(p); }

    /** enum cv::FastFeatureDetector:: */
    public static final int
        TYPE_5_8 = 0, TYPE_7_12 = 1, TYPE_9_16 = 2,
        THRESHOLD = 10000, NONMAX_SUPPRESSION= 10001, FAST_N= 10002;

    public static native @Ptr FastFeatureDetector create( int threshold/*=10*/,
                                                        @Cast("bool") boolean nonmaxSuppression/*=true*/,
                                                        int type/*=cv::FastFeatureDetector::TYPE_9_16*/ );
    public static native @Ptr FastFeatureDetector create( );

    public native void setThreshold(int threshold);
    public native int getThreshold();

    public native void setNonmaxSuppression(@Cast("bool") boolean f);
    public native @Cast("bool") boolean getNonmaxSuppression();

    public native void setType(int type);
    public native int getType();
}

/** \overload */
@Namespace("cv") public static native void AGAST( @ByVal Mat image, @ByRef KeyPointVector keypoints,
                      int threshold, @Cast("bool") boolean nonmaxSuppression/*=true*/ );
@Namespace("cv") public static native void AGAST( @ByVal Mat image, @ByRef KeyPointVector keypoints,
                      int threshold );

/** \brief Detects corners using the AGAST algorithm
<p>
@param image grayscale image where keypoints (corners) are detected.
@param keypoints keypoints detected on the image.
@param threshold threshold on difference between intensity of the central pixel and pixels of a
circle around this pixel.
@param nonmaxSuppression if true, non-maximum suppression is applied to detected corners
(keypoints).
@param type one of the four neighborhoods as defined in the paper:
AgastFeatureDetector::AGAST_5_8, AgastFeatureDetector::AGAST_7_12d,
AgastFeatureDetector::AGAST_7_12s, AgastFeatureDetector::OAST_9_16
<p>
Detects corners using the AGAST algorithm by \cite mair2010_agast .
 <p>
 */
@Namespace("cv") public static native void AGAST( @ByVal Mat image, @ByRef KeyPointVector keypoints,
                      int threshold, @Cast("bool") boolean nonmaxSuppression, int type );
/** \} features2d_main
 <p>
 *  \addtogroup features2d_main
 *  \{
<p>
/** \brief Wrapping class for feature detection using the AGAST method. :
 */
@Namespace("cv") public static class AgastFeatureDetector extends Feature2D {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public AgastFeatureDetector(Pointer p) { super(p); }

    /** enum cv::AgastFeatureDetector:: */
    public static final int
        AGAST_5_8 = 0, AGAST_7_12d = 1, AGAST_7_12s = 2, OAST_9_16 = 3,
        THRESHOLD = 10000, NONMAX_SUPPRESSION = 10001;

    public static native @Ptr AgastFeatureDetector create( int threshold/*=10*/,
                                                         @Cast("bool") boolean nonmaxSuppression/*=true*/,
                                                         int type/*=cv::AgastFeatureDetector::OAST_9_16*/ );
    public static native @Ptr AgastFeatureDetector create( );

    public native void setThreshold(int threshold);
    public native int getThreshold();

    public native void setNonmaxSuppression(@Cast("bool") boolean f);
    public native @Cast("bool") boolean getNonmaxSuppression();

    public native void setType(int type);
    public native int getType();
}

/** \brief Wrapping class for feature detection using the goodFeaturesToTrack function. :
 */
@Namespace("cv") public static class GFTTDetector extends Feature2D {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public GFTTDetector(Pointer p) { super(p); }

    public static native @Ptr GFTTDetector create( int maxCorners/*=1000*/, double qualityLevel/*=0.01*/, double minDistance/*=1*/,
                                                 int blockSize/*=3*/, @Cast("bool") boolean useHarrisDetector/*=false*/, double k/*=0.04*/ );
    public static native @Ptr GFTTDetector create( );
    public native void setMaxFeatures(int maxFeatures);
    public native int getMaxFeatures();

    public native void setQualityLevel(double qlevel);
    public native double getQualityLevel();

    public native void setMinDistance(double minDistance);
    public native double getMinDistance();

    public native void setBlockSize(int blockSize);
    public native int getBlockSize();

    public native void setHarrisDetector(@Cast("bool") boolean val);
    public native @Cast("bool") boolean getHarrisDetector();

    public native void setK(double k);
    public native double getK();
}

/** \brief Class for extracting blobs from an image. :
<p>
The class implements a simple algorithm for extracting blobs from an image:
<p>
1.  Convert the source image to binary images by applying thresholding with several thresholds from
    minThreshold (inclusive) to maxThreshold (exclusive) with distance thresholdStep between
    neighboring thresholds.
2.  Extract connected components from every binary image by findContours and calculate their
    centers.
3.  Group centers from several binary images by their coordinates. Close centers form one group that
    corresponds to one blob, which is controlled by the minDistBetweenBlobs parameter.
4.  From the groups, estimate final centers of blobs and their radiuses and return as locations and
    sizes of keypoints.
<p>
This class performs several filtrations of returned blobs. You should set filterBy\* to true/false
to turn on/off corresponding filtration. Available filtrations:
<p>
-   **By color**. This filter compares the intensity of a binary image at the center of a blob to
blobColor. If they differ, the blob is filtered out. Use blobColor = 0 to extract dark blobs
and blobColor = 255 to extract light blobs.
-   **By area**. Extracted blobs have an area between minArea (inclusive) and maxArea (exclusive).
-   **By circularity**. Extracted blobs have circularity
(\f$\frac{4*\pi*Area}{perimeter * perimeter}\f$) between minCircularity (inclusive) and
maxCircularity (exclusive).
-   **By ratio of the minimum inertia to maximum inertia**. Extracted blobs have this ratio
between minInertiaRatio (inclusive) and maxInertiaRatio (exclusive).
-   **By convexity**. Extracted blobs have convexity (area / area of blob convex hull) between
minConvexity (inclusive) and maxConvexity (exclusive).
<p>
Default values of parameters are tuned to extract dark circular blobs.
 */
@Namespace("cv") public static class SimpleBlobDetector extends Feature2D {
    static { Loader.load(); }
    /** Default native constructor. */
    public SimpleBlobDetector() { super((Pointer)null); allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public SimpleBlobDetector(int size) { super((Pointer)null); allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public SimpleBlobDetector(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public SimpleBlobDetector position(int position) {
        return (SimpleBlobDetector)super.position(position);
    }

  @NoOffset public static class Params extends Pointer {
      static { Loader.load(); }
      /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
      public Params(Pointer p) { super(p); }
      /** Native array allocator. Access with {@link Pointer#position(int)}. */
      public Params(int size) { super((Pointer)null); allocateArray(size); }
      private native void allocateArray(int size);
      @Override public Params position(int position) {
          return (Params)super.position(position);
      }
  
      public Params() { super((Pointer)null); allocate(); }
      private native void allocate();
      public native float thresholdStep(); public native Params thresholdStep(float thresholdStep);
      public native float minThreshold(); public native Params minThreshold(float minThreshold);
      public native float maxThreshold(); public native Params maxThreshold(float maxThreshold);
      public native @Cast("size_t") long minRepeatability(); public native Params minRepeatability(long minRepeatability);
      public native float minDistBetweenBlobs(); public native Params minDistBetweenBlobs(float minDistBetweenBlobs);

      public native @Cast("bool") boolean filterByColor(); public native Params filterByColor(boolean filterByColor);
      public native @Cast("uchar") byte blobColor(); public native Params blobColor(byte blobColor);

      public native @Cast("bool") boolean filterByArea(); public native Params filterByArea(boolean filterByArea);
      public native float minArea(); public native Params minArea(float minArea);
      public native float maxArea(); public native Params maxArea(float maxArea);

      public native @Cast("bool") boolean filterByCircularity(); public native Params filterByCircularity(boolean filterByCircularity);
      public native float minCircularity(); public native Params minCircularity(float minCircularity);
      public native float maxCircularity(); public native Params maxCircularity(float maxCircularity);

      public native @Cast("bool") boolean filterByInertia(); public native Params filterByInertia(boolean filterByInertia);
      public native float minInertiaRatio(); public native Params minInertiaRatio(float minInertiaRatio);
      public native float maxInertiaRatio(); public native Params maxInertiaRatio(float maxInertiaRatio);

      public native @Cast("bool") boolean filterByConvexity(); public native Params filterByConvexity(boolean filterByConvexity);
      public native float minConvexity(); public native Params minConvexity(float minConvexity);
      public native float maxConvexity(); public native Params maxConvexity(float maxConvexity);

      public native void read( @Const @ByRef FileNode fn );
      public native void write( @ByRef FileStorage fs );
  }

  public static native @Ptr SimpleBlobDetector create(@Const @ByRef(nullValue = "cv::SimpleBlobDetector::Params()") Params parameters/*=cv::SimpleBlobDetector::Params()*/);
  public static native @Ptr SimpleBlobDetector create();
}

/** \} features2d_main
 <p>
 *  \addtogroup features2d_main
 *  \{
<p>
/** \brief Class implementing the KAZE keypoint detector and descriptor extractor, described in \cite ABD12 .
<p>
\note AKAZE descriptor can only be used with KAZE or AKAZE keypoints .. [ABD12] KAZE Features. Pablo
F. Alcantarilla, Adrien Bartoli and Andrew J. Davison. In European Conference on Computer Vision
(ECCV), Fiorenze, Italy, October 2012.
*/
@Namespace("cv") public static class KAZE extends Feature2D {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public KAZE(Pointer p) { super(p); }

    /** enum cv::KAZE:: */
    public static final int
        DIFF_PM_G1 = 0,
        DIFF_PM_G2 = 1,
        DIFF_WEICKERT = 2,
        DIFF_CHARBONNIER = 3;

    /** \brief The KAZE constructor
    <p>
    @param extended Set to enable extraction of extended (128-byte) descriptor.
    @param upright Set to enable use of upright descriptors (non rotation-invariant).
    @param threshold Detector response threshold to accept point
    @param nOctaves Maximum octave evolution of the image
    @param nOctaveLayers Default number of sublevels per scale level
    @param diffusivity Diffusivity type. DIFF_PM_G1, DIFF_PM_G2, DIFF_WEICKERT or
    DIFF_CHARBONNIER
     */
    public static native @Ptr KAZE create(@Cast("bool") boolean extended/*=false*/, @Cast("bool") boolean upright/*=false*/,
                                        float threshold/*=0.001f*/,
                                        int nOctaves/*=4*/, int nOctaveLayers/*=4*/,
                                        int diffusivity/*=cv::KAZE::DIFF_PM_G2*/);
    public static native @Ptr KAZE create();

    public native void setExtended(@Cast("bool") boolean extended);
    public native @Cast("bool") boolean getExtended();

    public native void setUpright(@Cast("bool") boolean upright);
    public native @Cast("bool") boolean getUpright();

    public native void setThreshold(double threshold);
    public native double getThreshold();

    public native void setNOctaves(int octaves);
    public native int getNOctaves();

    public native void setNOctaveLayers(int octaveLayers);
    public native int getNOctaveLayers();

    public native void setDiffusivity(int diff);
    public native int getDiffusivity();
}

/** \brief Class implementing the AKAZE keypoint detector and descriptor extractor, described in \cite ANB13 . :
<p>
\note AKAZE descriptors can only be used with KAZE or AKAZE keypoints. Try to avoid using *extract*
and *detect* instead of *operator()* due to performance reasons. .. [ANB13] Fast Explicit Diffusion
for Accelerated Features in Nonlinear Scale Spaces. Pablo F. Alcantarilla, Jesús Nuevo and Adrien
Bartoli. In British Machine Vision Conference (BMVC), Bristol, UK, September 2013.
 */
@Namespace("cv") public static class AKAZE extends Feature2D {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public AKAZE(Pointer p) { super(p); }

    // AKAZE descriptor type
    /** enum cv::AKAZE:: */
    public static final int
        /** Upright descriptors, not invariant to rotation */
        DESCRIPTOR_KAZE_UPRIGHT = 2,
        DESCRIPTOR_KAZE = 3,
        /** Upright descriptors, not invariant to rotation */
        DESCRIPTOR_MLDB_UPRIGHT = 4,
        DESCRIPTOR_MLDB = 5;

    /** \brief The AKAZE constructor
    <p>
    @param descriptor_type Type of the extracted descriptor: DESCRIPTOR_KAZE,
    DESCRIPTOR_KAZE_UPRIGHT, DESCRIPTOR_MLDB or DESCRIPTOR_MLDB_UPRIGHT.
    @param descriptor_size Size of the descriptor in bits. 0 -\> Full size
    @param descriptor_channels Number of channels in the descriptor (1, 2, 3)
    @param threshold Detector response threshold to accept point
    @param nOctaves Maximum octave evolution of the image
    @param nOctaveLayers Default number of sublevels per scale level
    @param diffusivity Diffusivity type. DIFF_PM_G1, DIFF_PM_G2, DIFF_WEICKERT or
    DIFF_CHARBONNIER
     */
    public static native @Ptr AKAZE create(int descriptor_type/*=cv::AKAZE::DESCRIPTOR_MLDB*/,
                                         int descriptor_size/*=0*/, int descriptor_channels/*=3*/,
                                         float threshold/*=0.001f*/, int nOctaves/*=4*/,
                                         int nOctaveLayers/*=4*/, int diffusivity/*=cv::KAZE::DIFF_PM_G2*/);
    public static native @Ptr AKAZE create();

    public native void setDescriptorType(int dtype);
    public native int getDescriptorType();

    public native void setDescriptorSize(int dsize);
    public native int getDescriptorSize();

    public native void setDescriptorChannels(int dch);
    public native int getDescriptorChannels();

    public native void setThreshold(double threshold);
    public native double getThreshold();

    public native void setNOctaves(int octaves);
    public native int getNOctaves();

    public native void setNOctaveLayers(int octaveLayers);
    public native int getNOctaveLayers();

    public native void setDiffusivity(int diff);
    public native int getDiffusivity();
}

/** \} features2d_main
<p>
/****************************************************************************************\
*                                      Distance                                          *
\****************************************************************************************/

@Name("cv::Accumulator<unsigned char>") public static class Accumulator extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public Accumulator() { super((Pointer)null); allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public Accumulator(int size) { super((Pointer)null); allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public Accumulator(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public Accumulator position(int position) {
        return (Accumulator)super.position(position);
    }
 }

/*
 * Squared Euclidean distance functor
 */

/*
 * Euclidean distance functor
 */

/*
 * Manhattan distance (city block distance) functor
 */

/****************************************************************************************\
*                                  DescriptorMatcher                                     *
\****************************************************************************************/

/** \addtogroup features2d_match
/** \{
<p>
/** \brief Abstract base class for matching keypoint descriptors.
<p>
It has two groups of match methods: for matching descriptors of an image with another image or with
an image set.
 */
@Namespace("cv") @NoOffset public static class DescriptorMatcher extends Algorithm {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public DescriptorMatcher(Pointer p) { super(p); }


    /** \brief Adds descriptors to train a CPU(trainDescCollectionis) or GPU(utrainDescCollectionis) descriptor
    collection.
    <p>
    If the collection is not empty, the new descriptors are added to existing train descriptors.
    <p>
    @param descriptors Descriptors to add. Each descriptors[i] is a set of descriptors from the same
    train image.
     */
    public native void add( @ByVal MatVector descriptors );

    /** \brief Returns a constant link to the train descriptor collection trainDescCollection .
     */
    public native @Const @ByRef MatVector getTrainDescriptors();

    /** \brief Clears the train descriptor collections.
     */
    public native void clear();

    /** \brief Returns true if there are no train descriptors in the both collections.
     */
    public native @Cast("bool") boolean empty();

    /** \brief Returns true if the descriptor matcher supports masking permissible matches.
     */
    public native @Cast("bool") boolean isMaskSupported();

    /** \brief Trains a descriptor matcher
    <p>
    Trains a descriptor matcher (for example, the flann index). In all methods to match, the method
    train() is run every time before matching. Some descriptor matchers (for example, BruteForceMatcher)
    have an empty implementation of this method. Other matchers really train their inner structures (for
    example, FlannBasedMatcher trains flann::Index ).
     */
    public native void train();

    /** \brief Finds the best match for each descriptor from a query set.
    <p>
    @param queryDescriptors Query set of descriptors.
    @param trainDescriptors Train set of descriptors. This set is not added to the train descriptors
    collection stored in the class object.
    @param matches Matches. If a query descriptor is masked out in mask , no match is added for this
    descriptor. So, matches size may be smaller than the query descriptors count.
    @param mask Mask specifying permissible matches between an input query and train matrices of
    descriptors.
    <p>
    In the first variant of this method, the train descriptors are passed as an input argument. In the
    second variant of the method, train descriptors collection that was set by DescriptorMatcher::add is
    used. Optional mask (or masks) can be passed to specify which query and training descriptors can be
    matched. Namely, queryDescriptors[i] can be matched with trainDescriptors[j] only if
    mask.at\<uchar\>(i,j) is non-zero.
     */
    public native void match( @ByVal Mat queryDescriptors, @ByVal Mat trainDescriptors,
                    @ByRef DMatchVector matches, @ByVal(nullValue = "cv::noArray()") Mat mask/*=cv::noArray()*/ );
    public native void match( @ByVal Mat queryDescriptors, @ByVal Mat trainDescriptors,
                    @ByRef DMatchVector matches );

    /** \brief Finds the k best matches for each descriptor from a query set.
    <p>
    @param queryDescriptors Query set of descriptors.
    @param trainDescriptors Train set of descriptors. This set is not added to the train descriptors
    collection stored in the class object.
    @param mask Mask specifying permissible matches between an input query and train matrices of
    descriptors.
    @param matches Matches. Each matches[i] is k or less matches for the same query descriptor.
    @param k Count of best matches found per each query descriptor or less if a query descriptor has
    less than k possible matches in total.
    @param compactResult Parameter used when the mask (or masks) is not empty. If compactResult is
    false, the matches vector has the same size as queryDescriptors rows. If compactResult is true,
    the matches vector does not contain matches for fully masked-out query descriptors.
    <p>
    These extended variants of DescriptorMatcher::match methods find several best matches for each query
    descriptor. The matches are returned in the distance increasing order. See DescriptorMatcher::match
    for the details about query and train descriptors.
     */
    public native void knnMatch( @ByVal Mat queryDescriptors, @ByVal Mat trainDescriptors,
                       @ByRef DMatchVectorVector matches, int k,
                       @ByVal(nullValue = "cv::noArray()") Mat mask/*=cv::noArray()*/, @Cast("bool") boolean compactResult/*=false*/ );
    public native void knnMatch( @ByVal Mat queryDescriptors, @ByVal Mat trainDescriptors,
                       @ByRef DMatchVectorVector matches, int k );

    /** \brief For each query descriptor, finds the training descriptors not farther than the specified distance.
    <p>
    @param queryDescriptors Query set of descriptors.
    @param trainDescriptors Train set of descriptors. This set is not added to the train descriptors
    collection stored in the class object.
    @param matches Found matches.
    @param compactResult Parameter used when the mask (or masks) is not empty. If compactResult is
    false, the matches vector has the same size as queryDescriptors rows. If compactResult is true,
    the matches vector does not contain matches for fully masked-out query descriptors.
    @param maxDistance Threshold for the distance between matched descriptors. Distance means here
    metric distance (e.g. Hamming distance), not the distance between coordinates (which is measured
    in Pixels)!
    @param mask Mask specifying permissible matches between an input query and train matrices of
    descriptors.
    <p>
    For each query descriptor, the methods find such training descriptors that the distance between the
    query descriptor and the training descriptor is equal or smaller than maxDistance. Found matches are
    returned in the distance increasing order.
     */
    public native void radiusMatch( @ByVal Mat queryDescriptors, @ByVal Mat trainDescriptors,
                          @ByRef DMatchVectorVector matches, float maxDistance,
                          @ByVal(nullValue = "cv::noArray()") Mat mask/*=cv::noArray()*/, @Cast("bool") boolean compactResult/*=false*/ );
    public native void radiusMatch( @ByVal Mat queryDescriptors, @ByVal Mat trainDescriptors,
                          @ByRef DMatchVectorVector matches, float maxDistance );

    /** \overload
    @param queryDescriptors Query set of descriptors.
    @param matches Matches. If a query descriptor is masked out in mask , no match is added for this
    descriptor. So, matches size may be smaller than the query descriptors count.
    @param masks Set of masks. Each masks[i] specifies permissible matches between the input query
    descriptors and stored train descriptors from the i-th image trainDescCollection[i].
    */
    public native void match( @ByVal Mat queryDescriptors, @ByRef DMatchVector matches,
                            @ByVal(nullValue = "cv::noArray()") MatVector masks/*=cv::noArray()*/ );
    public native void match( @ByVal Mat queryDescriptors, @ByRef DMatchVector matches );
    /** \overload
    @param queryDescriptors Query set of descriptors.
    @param matches Matches. Each matches[i] is k or less matches for the same query descriptor.
    @param k Count of best matches found per each query descriptor or less if a query descriptor has
    less than k possible matches in total.
    @param masks Set of masks. Each masks[i] specifies permissible matches between the input query
    descriptors and stored train descriptors from the i-th image trainDescCollection[i].
    @param compactResult Parameter used when the mask (or masks) is not empty. If compactResult is
    false, the matches vector has the same size as queryDescriptors rows. If compactResult is true,
    the matches vector does not contain matches for fully masked-out query descriptors.
    */
    public native void knnMatch( @ByVal Mat queryDescriptors, @ByRef DMatchVectorVector matches, int k,
                               @ByVal(nullValue = "cv::noArray()") MatVector masks/*=cv::noArray()*/, @Cast("bool") boolean compactResult/*=false*/ );
    public native void knnMatch( @ByVal Mat queryDescriptors, @ByRef DMatchVectorVector matches, int k );
    /** \overload
    @param queryDescriptors Query set of descriptors.
    @param matches Found matches.
    @param maxDistance Threshold for the distance between matched descriptors. Distance means here
    metric distance (e.g. Hamming distance), not the distance between coordinates (which is measured
    in Pixels)!
    @param masks Set of masks. Each masks[i] specifies permissible matches between the input query
    descriptors and stored train descriptors from the i-th image trainDescCollection[i].
    @param compactResult Parameter used when the mask (or masks) is not empty. If compactResult is
    false, the matches vector has the same size as queryDescriptors rows. If compactResult is true,
    the matches vector does not contain matches for fully masked-out query descriptors.
    */
    public native void radiusMatch( @ByVal Mat queryDescriptors, @ByRef DMatchVectorVector matches, float maxDistance,
                          @ByVal(nullValue = "cv::noArray()") MatVector masks/*=cv::noArray()*/, @Cast("bool") boolean compactResult/*=false*/ );
    public native void radiusMatch( @ByVal Mat queryDescriptors, @ByRef DMatchVectorVector matches, float maxDistance );

    // Reads matcher object from a file node
    public native void read( @Const @ByRef FileNode arg0 );
    // Writes matcher object to a file storage
    public native void write( @ByRef FileStorage arg0 );

    /** \brief Clones the matcher.
    <p>
    @param emptyTrainData If emptyTrainData is false, the method creates a deep copy of the object,
    that is, copies both parameters and train data. If emptyTrainData is true, the method creates an
    object copy with the current parameters but with empty train data.
     */
    public native @Ptr DescriptorMatcher clone( @Cast("bool") boolean emptyTrainData/*=false*/ );
    public native @Ptr DescriptorMatcher clone( );

    /** \brief Creates a descriptor matcher of a given type with the default parameters (using default
    constructor).
    <p>
    @param descriptorMatcherType Descriptor matcher type. Now the following matcher types are
    supported:
    -   {@code BruteForce} (it uses L2 )
    -   {@code BruteForce-L1}
    -   {@code BruteForce-Hamming}
    -   {@code BruteForce-Hamming(2)}
    -   {@code FlannBased}
     */
    public static native @Ptr DescriptorMatcher create( @Str BytePointer descriptorMatcherType );
    public static native @Ptr DescriptorMatcher create( @Str String descriptorMatcherType );
}

/** \brief Brute-force descriptor matcher.
<p>
For each descriptor in the first set, this matcher finds the closest descriptor in the second set
by trying each one. This descriptor matcher supports masking permissible matches of descriptor
sets.
 */
@Namespace("cv") @NoOffset public static class BFMatcher extends DescriptorMatcher {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public BFMatcher(Pointer p) { super(p); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public BFMatcher(int size) { super((Pointer)null); allocateArray(size); }
    private native void allocateArray(int size);
    @Override public BFMatcher position(int position) {
        return (BFMatcher)super.position(position);
    }

    /** \brief Brute-force matcher constructor.
    <p>
    @param normType One of NORM_L1, NORM_L2, NORM_HAMMING, NORM_HAMMING2. L1 and L2 norms are
    preferable choices for SIFT and SURF descriptors, NORM_HAMMING should be used with ORB, BRISK and
    BRIEF, NORM_HAMMING2 should be used with ORB when WTA_K==3 or 4 (see ORB::ORB constructor
    description).
    @param crossCheck If it is false, this is will be default BFMatcher behaviour when it finds the k
    nearest neighbors for each query descriptor. If crossCheck==true, then the knnMatch() method with
    k=1 will only return pairs (i,j) such that for i-th query descriptor the j-th descriptor in the
    matcher's collection is the nearest and vice versa, i.e. the BFMatcher will only return consistent
    pairs. Such technique usually produces best results with minimal number of outliers when there are
    enough matches. This is alternative to the ratio test, used by D. Lowe in SIFT paper.
     */
    public BFMatcher( int normType/*=cv::NORM_L2*/, @Cast("bool") boolean crossCheck/*=false*/ ) { super((Pointer)null); allocate(normType, crossCheck); }
    private native void allocate( int normType/*=cv::NORM_L2*/, @Cast("bool") boolean crossCheck/*=false*/ );
    public BFMatcher( ) { super((Pointer)null); allocate(); }
    private native void allocate( );

    public native @Cast("bool") boolean isMaskSupported();

    public native @Ptr DescriptorMatcher clone( @Cast("bool") boolean emptyTrainData/*=false*/ );
    public native @Ptr DescriptorMatcher clone( );
}


/** \brief Flann-based descriptor matcher.
<p>
This matcher trains flann::Index_ on a train descriptor collection and calls its nearest search
methods to find the best matches. So, this matcher may be faster when matching a large train
collection than the brute force matcher. FlannBasedMatcher does not support masking permissible
matches of descriptor sets because flann::Index does not support this. :
 */
@Namespace("cv") @NoOffset public static class FlannBasedMatcher extends DescriptorMatcher {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public FlannBasedMatcher(Pointer p) { super(p); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public FlannBasedMatcher(int size) { super((Pointer)null); allocateArray(size); }
    private native void allocateArray(int size);
    @Override public FlannBasedMatcher position(int position) {
        return (FlannBasedMatcher)super.position(position);
    }

    public FlannBasedMatcher( @Ptr IndexParams indexParams/*=makePtr<flann::KDTreeIndexParams>()*/,
                           @Ptr SearchParams searchParams/*=makePtr<flann::SearchParams>()*/ ) { super((Pointer)null); allocate(indexParams, searchParams); }
    private native void allocate( @Ptr IndexParams indexParams/*=makePtr<flann::KDTreeIndexParams>()*/,
                           @Ptr SearchParams searchParams/*=makePtr<flann::SearchParams>()*/ );
    public FlannBasedMatcher( ) { super((Pointer)null); allocate(); }
    private native void allocate( );

    public native void add( @ByVal MatVector descriptors );
    public native void clear();

    // Reads matcher object from a file node
    public native void read( @Const @ByRef FileNode arg0 );
    // Writes matcher object to a file storage
    public native void write( @ByRef FileStorage arg0 );

    public native void train();
    public native @Cast("bool") boolean isMaskSupported();

    public native @Ptr DescriptorMatcher clone( @Cast("bool") boolean emptyTrainData/*=false*/ );
    public native @Ptr DescriptorMatcher clone( );
}

/** \} features2d_match
<p>
/****************************************************************************************\
*                                   Drawing functions                                    *
\****************************************************************************************/

/** \addtogroup features2d_draw
/** \{ */

@Namespace("cv") public static class DrawMatchesFlags extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public DrawMatchesFlags() { super((Pointer)null); allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public DrawMatchesFlags(int size) { super((Pointer)null); allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public DrawMatchesFlags(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public DrawMatchesFlags position(int position) {
        return (DrawMatchesFlags)super.position(position);
    }

    /** enum cv::DrawMatchesFlags:: */
    public static final int /** Output image matrix will be created (Mat::create),
 *  i.e. existing memory of output image may be reused.
 *  Two source image, matches and single keypoints will be drawn.
 *  For each keypoint only the center point will be drawn (without
 *  the circle around keypoint with keypoint size and orientation). */
 DEFAULT = 0,
          /** Output image matrix will not be created (Mat::create).
 *  Matches will be drawn on existing content of output image. */
          DRAW_OVER_OUTIMG = 1,
          /** Single keypoints will not be drawn. */
          NOT_DRAW_SINGLE_POINTS = 2,
          /** For each keypoint the circle around keypoint with keypoint size and
 *  orientation will be drawn. */
          DRAW_RICH_KEYPOINTS = 4;
}

/** \brief Draws keypoints.
<p>
@param image Source image.
@param keypoints Keypoints from the source image.
@param outImage Output image. Its content depends on the flags value defining what is drawn in the
output image. See possible flags bit values below.
@param color Color of keypoints.
@param flags Flags setting drawing features. Possible flags bit values are defined by
DrawMatchesFlags. See details above in drawMatches .
<p>
\note
For Python API, flags are modified as cv2.DRAW_MATCHES_FLAGS_DEFAULT,
cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS, cv2.DRAW_MATCHES_FLAGS_DRAW_OVER_OUTIMG,
cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS
 */
@Namespace("cv") public static native void drawKeypoints( @ByVal Mat image, @Const @ByRef KeyPointVector keypoints, @ByVal Mat outImage,
                               @Const @ByRef(nullValue = "cv::Scalar::all(-1)") Scalar color/*=cv::Scalar::all(-1)*/, int flags/*=cv::DrawMatchesFlags::DEFAULT*/ );
@Namespace("cv") public static native void drawKeypoints( @ByVal Mat image, @Const @ByRef KeyPointVector keypoints, @ByVal Mat outImage );

/** \brief Draws the found matches of keypoints from two images.
<p>
@param img1 First source image.
@param keypoints1 Keypoints from the first source image.
@param img2 Second source image.
@param keypoints2 Keypoints from the second source image.
@param matches1to2 Matches from the first image to the second one, which means that keypoints1[i]
has a corresponding point in keypoints2[matches[i]] .
@param outImg Output image. Its content depends on the flags value defining what is drawn in the
output image. See possible flags bit values below.
@param matchColor Color of matches (lines and connected keypoints). If matchColor==Scalar::all(-1)
, the color is generated randomly.
@param singlePointColor Color of single keypoints (circles), which means that keypoints do not
have the matches. If singlePointColor==Scalar::all(-1) , the color is generated randomly.
@param matchesMask Mask determining which matches are drawn. If the mask is empty, all matches are
drawn.
@param flags Flags setting drawing features. Possible flags bit values are defined by
DrawMatchesFlags.
<p>
This function draws matches of keypoints from two images in the output image. Match is a line
connecting two keypoints (circles). See cv::DrawMatchesFlags.
 */
@Namespace("cv") public static native void drawMatches( @ByVal Mat img1, @Const @ByRef KeyPointVector keypoints1,
                             @ByVal Mat img2, @Const @ByRef KeyPointVector keypoints2,
                             @Const @ByRef DMatchVector matches1to2, @ByVal Mat outImg,
                             @Const @ByRef(nullValue = "cv::Scalar::all(-1)") Scalar matchColor/*=cv::Scalar::all(-1)*/, @Const @ByRef(nullValue = "cv::Scalar::all(-1)") Scalar singlePointColor/*=cv::Scalar::all(-1)*/,
                             @Cast("char*") @StdVector BytePointer matchesMask/*=std::vector<char>()*/, int flags/*=cv::DrawMatchesFlags::DEFAULT*/ );
@Namespace("cv") public static native void drawMatches( @ByVal Mat img1, @Const @ByRef KeyPointVector keypoints1,
                             @ByVal Mat img2, @Const @ByRef KeyPointVector keypoints2,
                             @Const @ByRef DMatchVector matches1to2, @ByVal Mat outImg );
@Namespace("cv") public static native void drawMatches( @ByVal Mat img1, @Const @ByRef KeyPointVector keypoints1,
                             @ByVal Mat img2, @Const @ByRef KeyPointVector keypoints2,
                             @Const @ByRef DMatchVector matches1to2, @ByVal Mat outImg,
                             @Const @ByRef(nullValue = "cv::Scalar::all(-1)") Scalar matchColor/*=cv::Scalar::all(-1)*/, @Const @ByRef(nullValue = "cv::Scalar::all(-1)") Scalar singlePointColor/*=cv::Scalar::all(-1)*/,
                             @Cast("char*") @StdVector ByteBuffer matchesMask/*=std::vector<char>()*/, int flags/*=cv::DrawMatchesFlags::DEFAULT*/ );
@Namespace("cv") public static native void drawMatches( @ByVal Mat img1, @Const @ByRef KeyPointVector keypoints1,
                             @ByVal Mat img2, @Const @ByRef KeyPointVector keypoints2,
                             @Const @ByRef DMatchVector matches1to2, @ByVal Mat outImg,
                             @Const @ByRef(nullValue = "cv::Scalar::all(-1)") Scalar matchColor/*=cv::Scalar::all(-1)*/, @Const @ByRef(nullValue = "cv::Scalar::all(-1)") Scalar singlePointColor/*=cv::Scalar::all(-1)*/,
                             @Cast("char*") @StdVector byte[] matchesMask/*=std::vector<char>()*/, int flags/*=cv::DrawMatchesFlags::DEFAULT*/ );

/** \overload */
@Namespace("cv") public static native @Name("drawMatches") void drawMatchesKnn( @ByVal Mat img1, @Const @ByRef KeyPointVector keypoints1,
                             @ByVal Mat img2, @Const @ByRef KeyPointVector keypoints2,
                             @Const @ByRef DMatchVectorVector matches1to2, @ByVal Mat outImg,
                             @Const @ByRef(nullValue = "cv::Scalar::all(-1)") Scalar matchColor/*=cv::Scalar::all(-1)*/, @Const @ByRef(nullValue = "cv::Scalar::all(-1)") Scalar singlePointColor/*=cv::Scalar::all(-1)*/,
                             @Cast("const std::vector<std::vector<char> >*") @ByRef(nullValue = "std::vector<std::vector<char> >()") ByteVectorVector matchesMask/*=std::vector<std::vector<char> >()*/, int flags/*=cv::DrawMatchesFlags::DEFAULT*/ );
@Namespace("cv") public static native @Name("drawMatches") void drawMatchesKnn( @ByVal Mat img1, @Const @ByRef KeyPointVector keypoints1,
                             @ByVal Mat img2, @Const @ByRef KeyPointVector keypoints2,
                             @Const @ByRef DMatchVectorVector matches1to2, @ByVal Mat outImg );

/** \} features2d_draw
<p>
/****************************************************************************************\
*   Functions to evaluate the feature detectors and [generic] descriptor extractors      *
\****************************************************************************************/

@Namespace("cv") public static native void evaluateFeatureDetector( @Const @ByRef Mat img1, @Const @ByRef Mat img2, @Const @ByRef Mat H1to2,
                                         KeyPointVector keypoints1, KeyPointVector keypoints2,
                                         @ByRef FloatPointer repeatability, @ByRef IntPointer correspCount,
                                         @Cast("cv::FeatureDetector*") @Ptr Feature2D fdetector/*=cv::Ptr<cv::FeatureDetector>()*/ );
@Namespace("cv") public static native void evaluateFeatureDetector( @Const @ByRef Mat img1, @Const @ByRef Mat img2, @Const @ByRef Mat H1to2,
                                         KeyPointVector keypoints1, KeyPointVector keypoints2,
                                         @ByRef FloatPointer repeatability, @ByRef IntPointer correspCount );
@Namespace("cv") public static native void evaluateFeatureDetector( @Const @ByRef Mat img1, @Const @ByRef Mat img2, @Const @ByRef Mat H1to2,
                                         KeyPointVector keypoints1, KeyPointVector keypoints2,
                                         @ByRef FloatBuffer repeatability, @ByRef IntBuffer correspCount,
                                         @Cast("cv::FeatureDetector*") @Ptr Feature2D fdetector/*=cv::Ptr<cv::FeatureDetector>()*/ );
@Namespace("cv") public static native void evaluateFeatureDetector( @Const @ByRef Mat img1, @Const @ByRef Mat img2, @Const @ByRef Mat H1to2,
                                         KeyPointVector keypoints1, KeyPointVector keypoints2,
                                         @ByRef FloatBuffer repeatability, @ByRef IntBuffer correspCount );
@Namespace("cv") public static native void evaluateFeatureDetector( @Const @ByRef Mat img1, @Const @ByRef Mat img2, @Const @ByRef Mat H1to2,
                                         KeyPointVector keypoints1, KeyPointVector keypoints2,
                                         @ByRef float[] repeatability, @ByRef int[] correspCount,
                                         @Cast("cv::FeatureDetector*") @Ptr Feature2D fdetector/*=cv::Ptr<cv::FeatureDetector>()*/ );
@Namespace("cv") public static native void evaluateFeatureDetector( @Const @ByRef Mat img1, @Const @ByRef Mat img2, @Const @ByRef Mat H1to2,
                                         KeyPointVector keypoints1, KeyPointVector keypoints2,
                                         @ByRef float[] repeatability, @ByRef int[] correspCount );

@Namespace("cv") public static native void computeRecallPrecisionCurve( @Const @ByRef DMatchVectorVector matches1to2,
                                             @Cast("const std::vector<std::vector<unsigned char> >*") @ByRef ByteVectorVector correctMatches1to2Mask,
                                             @ByRef Point2fVector recallPrecisionCurve );

@Namespace("cv") public static native float getRecall( @Const @ByRef Point2fVector recallPrecisionCurve, float l_precision );
@Namespace("cv") public static native int getNearestPoint( @Const @ByRef Point2fVector recallPrecisionCurve, float l_precision );

/****************************************************************************************\
*                                     Bag of visual words                                *
\****************************************************************************************/

/** \addtogroup features2d_category
/** \{
<p>
/** \brief Abstract base class for training the *bag of visual words* vocabulary from a set of descriptors.
<p>
For details, see, for example, *Visual Categorization with Bags of Keypoints* by Gabriella Csurka,
Christopher R. Dance, Lixin Fan, Jutta Willamowski, Cedric Bray, 2004. :
 */
@Namespace("cv") @NoOffset public static class BOWTrainer extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public BOWTrainer(Pointer p) { super(p); }


    /** \brief Adds descriptors to a training set.
    <p>
    @param descriptors Descriptors to add to a training set. Each row of the descriptors matrix is a
    descriptor.
    <p>
    The training set is clustered using clustermethod to construct the vocabulary.
     */
    public native void add( @Const @ByRef Mat descriptors );

    /** \brief Returns a training set of descriptors.
    */
    public native @Const @ByRef MatVector getDescriptors();

    /** \brief Returns the count of all descriptors stored in the training set.
    */
    public native int descriptorsCount();

    public native void clear();

    /** \overload */
    public native @ByVal Mat cluster();

    /** \brief Clusters train descriptors.
    <p>
    @param descriptors Descriptors to cluster. Each row of the descriptors matrix is a descriptor.
    Descriptors are not added to the inner train descriptor set.
    <p>
    The vocabulary consists of cluster centers. So, this method returns the vocabulary. In the first
    variant of the method, train descriptors stored in the object are clustered. In the second variant,
    input descriptors are clustered.
     */
    public native @ByVal Mat cluster( @Const @ByRef Mat descriptors );
}

/** \brief kmeans -based class to train visual vocabulary using the *bag of visual words* approach. :
 */
@Namespace("cv") @NoOffset public static class BOWKMeansTrainer extends BOWTrainer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public BOWKMeansTrainer(Pointer p) { super(p); }

    /** \brief The constructor.
    <p>
    @see cv::kmeans
    */
    public BOWKMeansTrainer( int clusterCount, @Const @ByRef(nullValue = "cv::TermCriteria()") TermCriteria termcrit/*=cv::TermCriteria()*/,
                          int attempts/*=3*/, int flags/*=cv::KMEANS_PP_CENTERS*/ ) { super((Pointer)null); allocate(clusterCount, termcrit, attempts, flags); }
    private native void allocate( int clusterCount, @Const @ByRef(nullValue = "cv::TermCriteria()") TermCriteria termcrit/*=cv::TermCriteria()*/,
                          int attempts/*=3*/, int flags/*=cv::KMEANS_PP_CENTERS*/ );
    public BOWKMeansTrainer( int clusterCount ) { super((Pointer)null); allocate(clusterCount); }
    private native void allocate( int clusterCount );

    // Returns trained vocabulary (i.e. cluster centers).
    public native @ByVal Mat cluster();
    public native @ByVal Mat cluster( @Const @ByRef Mat descriptors );
}

/** \brief Class to compute an image descriptor using the *bag of visual words*.
<p>
Such a computation consists of the following steps:
<p>
1.  Compute descriptors for a given image and its keypoints set.
2.  Find the nearest visual words from the vocabulary for each keypoint descriptor.
3.  Compute the bag-of-words image descriptor as is a normalized histogram of vocabulary words
encountered in the image. The i-th bin of the histogram is a frequency of i-th word of the
vocabulary in the given image.
 */
@Namespace("cv") @NoOffset public static class BOWImgDescriptorExtractor extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public BOWImgDescriptorExtractor(Pointer p) { super(p); }

    /** \brief The constructor.
    <p>
    @param dextractor Descriptor extractor that is used to compute descriptors for an input image and
    its keypoints.
    @param dmatcher Descriptor matcher that is used to find the nearest word of the trained vocabulary
    for each keypoint descriptor of the image.
     */
    public BOWImgDescriptorExtractor( @Cast("cv::DescriptorExtractor*") @Ptr Feature2D dextractor,
                                   @Ptr DescriptorMatcher dmatcher ) { super((Pointer)null); allocate(dextractor, dmatcher); }
    private native void allocate( @Cast("cv::DescriptorExtractor*") @Ptr Feature2D dextractor,
                                   @Ptr DescriptorMatcher dmatcher );
    /** \overload */
    public BOWImgDescriptorExtractor( @Ptr DescriptorMatcher dmatcher ) { super((Pointer)null); allocate(dmatcher); }
    private native void allocate( @Ptr DescriptorMatcher dmatcher );

    /** \brief Sets a visual vocabulary.
    <p>
    @param vocabulary Vocabulary (can be trained using the inheritor of BOWTrainer ). Each row of the
    vocabulary is a visual word (cluster center).
     */
    public native void setVocabulary( @Const @ByRef Mat vocabulary );

    /** \brief Returns the set vocabulary.
    */
    public native @Const @ByRef Mat getVocabulary();

    /** \brief Computes an image descriptor using the set visual vocabulary.
    <p>
    @param image Image, for which the descriptor is computed.
    @param keypoints Keypoints detected in the input image.
    @param imgDescriptor Computed output image descriptor.
    @param pointIdxsOfClusters Indices of keypoints that belong to the cluster. This means that
    pointIdxsOfClusters[i] are keypoint indices that belong to the i -th cluster (word of vocabulary)
    returned if it is non-zero.
    @param descriptors Descriptors of the image keypoints that are returned if they are non-zero.
     */
    public native void compute( @ByVal Mat image, @ByRef KeyPointVector keypoints, @ByVal Mat imgDescriptor,
                      IntVectorVector pointIdxsOfClusters/*=0*/, Mat descriptors/*=0*/ );
    public native void compute( @ByVal Mat image, @ByRef KeyPointVector keypoints, @ByVal Mat imgDescriptor );
    /** \overload
    @param keypointDescriptors Computed descriptors to match with vocabulary.
    @param imgDescriptor Computed output image descriptor.
    @param pointIdxsOfClusters Indices of keypoints that belong to the cluster. This means that
    pointIdxsOfClusters[i] are keypoint indices that belong to the i -th cluster (word of vocabulary)
    returned if it is non-zero.
    */
    public native void compute( @ByVal Mat keypointDescriptors, @ByVal Mat imgDescriptor,
                      IntVectorVector pointIdxsOfClusters/*=0*/ );
    public native void compute( @ByVal Mat keypointDescriptors, @ByVal Mat imgDescriptor );
    // compute() is not constant because DescriptorMatcher::match is not constant

    /** \brief Returns an image descriptor size if the vocabulary is set. Otherwise, it returns 0.
    */
    public native int descriptorSize();

    /** \brief Returns an image descriptor type.
     */
    public native int descriptorType();
}

/** \} features2d_category
 <p>
 *  \} features2d */

 /* namespace cv */

// #endif


}
