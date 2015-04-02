// Targeted by JavaCPP version 0.11-SNAPSHOT

package org.bytedeco.javacpp;

import org.bytedeco.javacpp.annotation.Index;
import java.nio.*;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

import static org.bytedeco.javacpp.opencv_core.*;
import static org.bytedeco.javacpp.opencv_imgproc.*;
import static org.bytedeco.javacpp.opencv_highgui.*;
import static org.bytedeco.javacpp.opencv_flann.*;

public class opencv_features2d extends org.bytedeco.javacpp.presets.opencv_features2d {
    static { Loader.load(); }

@Name("std::vector<std::vector<cv::KeyPoint> >") public static class KeyPointVectorVector extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public KeyPointVectorVector(Pointer p) { super(p); }
    public KeyPointVectorVector(KeyPoint[] ... array) { this(array.length); put(array); }
    public KeyPointVectorVector()       { allocate();  }
    public KeyPointVectorVector(long n) { allocate(n); }
    private native void allocate();
    private native void allocate(@Cast("size_t") long n);
    public native @Name("operator=") @ByRef KeyPointVectorVector put(@ByRef KeyPointVectorVector x);

    public native long size();
    public native void resize(@Cast("size_t") long n);
    public native @Index long size(@Cast("size_t") long i);
    public native @Index void resize(@Cast("size_t") long i, @Cast("size_t") long n);

    @Index public native @ByRef KeyPoint get(@Cast("size_t") long i, @Cast("size_t") long j);
    public native KeyPointVectorVector put(@Cast("size_t") long i, @Cast("size_t") long j, KeyPoint value);

    public KeyPointVectorVector put(KeyPoint[] ... array) {
        if (size() != array.length) { resize(array.length); }
        for (int i = 0; i < array.length; i++) {
            if (size(i) != array[i].length) { resize(i, array[i].length); }
            for (int j = 0; j < array[i].length; j++) {
                put(i, j, array[i][j]);
            }
        }
        return this;
    }
}

@Name("std::vector<std::vector<cv::DMatch> >") public static class DMatchVectorVector extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public DMatchVectorVector(Pointer p) { super(p); }
    public DMatchVectorVector(DMatch[] ... array) { this(array.length); put(array); }
    public DMatchVectorVector()       { allocate();  }
    public DMatchVectorVector(long n) { allocate(n); }
    private native void allocate();
    private native void allocate(@Cast("size_t") long n);
    public native @Name("operator=") @ByRef DMatchVectorVector put(@ByRef DMatchVectorVector x);

    public native long size();
    public native void resize(@Cast("size_t") long n);
    public native @Index long size(@Cast("size_t") long i);
    public native @Index void resize(@Cast("size_t") long i, @Cast("size_t") long n);

    @Index public native @ByRef DMatch get(@Cast("size_t") long i, @Cast("size_t") long j);
    public native DMatchVectorVector put(@Cast("size_t") long i, @Cast("size_t") long j, DMatch value);

    public DMatchVectorVector put(DMatch[] ... array) {
        if (size() != array.length) { resize(array.length); }
        for (int i = 0; i < array.length; i++) {
            if (size(i) != array[i].length) { resize(i, array[i].length); }
            for (int j = 0; j < array[i].length; j++) {
                put(i, j, array[i][j]);
            }
        }
        return this;
    }
}

// Parsed from <opencv2/features2d/features2d.hpp>

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

// #include "opencv2/core/core.hpp"
// #include "opencv2/flann/miniflann.hpp"

// #ifdef __cplusplus
// #include <limits>

@Namespace("cv") public static native @Cast("bool") boolean initModule_features2d();

/**
 The Keypoint Class

 The class instance stores a keypoint, i.e. a point feature found by one of many available keypoint detectors, such as
 Harris corner detector, cv::FAST, cv::StarDetector, cv::SURF, cv::SIFT, cv::LDetector etc.

 The keypoint is characterized by the 2D position, scale
 (proportional to the diameter of the neighborhood that needs to be taken into account),
 orientation and some other parameters. The keypoint neighborhood is then analyzed by another algorithm that builds a descriptor
 (usually represented as a feature vector). The keypoints representing the same object in different images can then be matched using
 cv::KDTree or another method.
*/
@Namespace("cv") @NoOffset public static class KeyPoint extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public KeyPoint(Pointer p) { super(p); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public KeyPoint(int size) { allocateArray(size); }
    private native void allocateArray(int size);
    @Override public KeyPoint position(int position) {
        return (KeyPoint)super.position(position);
    }

    /** the default constructor */
    public KeyPoint() { allocate(); }
    private native void allocate();
    /** the full constructor */
    public KeyPoint(@ByVal Point2f _pt, float _size, float _angle/*=-1*/,
                float _response/*=0*/, int _octave/*=0*/, int _class_id/*=-1*/) { allocate(_pt, _size, _angle, _response, _octave, _class_id); }
    private native void allocate(@ByVal Point2f _pt, float _size, float _angle/*=-1*/,
                float _response/*=0*/, int _octave/*=0*/, int _class_id/*=-1*/);
    public KeyPoint(@ByVal Point2f _pt, float _size) { allocate(_pt, _size); }
    private native void allocate(@ByVal Point2f _pt, float _size);
    /** another form of the full constructor */
    public KeyPoint(float x, float y, float _size, float _angle/*=-1*/,
                float _response/*=0*/, int _octave/*=0*/, int _class_id/*=-1*/) { allocate(x, y, _size, _angle, _response, _octave, _class_id); }
    private native void allocate(float x, float y, float _size, float _angle/*=-1*/,
                float _response/*=0*/, int _octave/*=0*/, int _class_id/*=-1*/);
    public KeyPoint(float x, float y, float _size) { allocate(x, y, _size); }
    private native void allocate(float x, float y, float _size);

    public native @Cast("size_t") long hash();

    /** converts vector of keypoints to vector of points */
    public static native void convert(@StdVector KeyPoint keypoints,
                            @StdVector Point2f points2f,
                            @StdVector IntPointer keypointIndexes/*=vector<int>()*/);
    public static native void convert(@StdVector KeyPoint keypoints,
                            @StdVector Point2f points2f);
    public static native void convert(@StdVector KeyPoint keypoints,
                            @StdVector Point2f points2f,
                            @StdVector IntBuffer keypointIndexes/*=vector<int>()*/);
    public static native void convert(@StdVector KeyPoint keypoints,
                            @StdVector Point2f points2f,
                            @StdVector int[] keypointIndexes/*=vector<int>()*/);
    /** converts vector of points to the vector of keypoints, where each keypoint is assigned the same size and the same orientation */
    public static native void convert(@StdVector Point2f points2f,
                            @StdVector KeyPoint keypoints,
                            float size/*=1*/, float response/*=1*/, int octave/*=0*/, int class_id/*=-1*/);
    public static native void convert(@StdVector Point2f points2f,
                            @StdVector KeyPoint keypoints);

    /** computes overlap for pair of keypoints;
     *  overlap is a ratio between area of keypoint regions intersection and
     *  area of keypoint regions union (now keypoint region is circle) */
    public static native float overlap(@Const @ByRef KeyPoint kp1, @Const @ByRef KeyPoint kp2);

    /** coordinates of the keypoints */
    public native @ByRef Point2f pt(); public native KeyPoint pt(Point2f pt);
    /** diameter of the meaningful keypoint neighborhood */
    public native float size(); public native KeyPoint size(float size);
    /** computed orientation of the keypoint (-1 if not applicable);
     *  it's in [0,360) degrees and measured relative to
     *  image coordinate system, ie in clockwise. */
    public native float angle(); public native KeyPoint angle(float angle);
    /** the response by which the most strong keypoints have been selected. Can be used for the further sorting or subsampling */
    public native float response(); public native KeyPoint response(float response);
    /** octave (pyramid layer) from which the keypoint has been extracted */
    public native int octave(); public native KeyPoint octave(int octave);
    /** object class (if the keypoints need to be clustered by an object they belong to) */
    public native int class_id(); public native KeyPoint class_id(int class_id);
}

/** writes vector of keypoints to the file storage */
@Namespace("cv") public static native void write(@ByRef FileStorage fs, @StdString BytePointer name, @StdVector KeyPoint keypoints);
@Namespace("cv") public static native void write(@ByRef FileStorage fs, @StdString String name, @StdVector KeyPoint keypoints);
/** reads vector of keypoints from the specified file storage node */
@Namespace("cv") public static native void read(@Const @ByRef FileNode node, @StdVector KeyPoint keypoints);

/*
 * A class filters a vector of keypoints.
 * Because now it is difficult to provide a convenient interface for all usage scenarios of the keypoints filter class,
 * it has only several needed by now static methods.
 */
@Namespace("cv") public static class KeyPointsFilter extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public KeyPointsFilter(Pointer p) { super(p); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public KeyPointsFilter(int size) { allocateArray(size); }
    private native void allocateArray(int size);
    @Override public KeyPointsFilter position(int position) {
        return (KeyPointsFilter)super.position(position);
    }

    public KeyPointsFilter() { allocate(); }
    private native void allocate();

    /*
     * Remove keypoints within borderPixels of an image edge.
     */
    public static native void runByImageBorder( @StdVector KeyPoint keypoints, @ByVal Size imageSize, int borderSize );
    /*
     * Remove keypoints of sizes out of range.
     */
    public static native void runByKeypointSize( @StdVector KeyPoint keypoints, float minSize,
                                       float maxSize/*=FLT_MAX*/ );
    public static native void runByKeypointSize( @StdVector KeyPoint keypoints, float minSize );
    /*
     * Remove keypoints from some image by mask for pixels of this image.
     */
    public static native void runByPixelsMask( @StdVector KeyPoint keypoints, @Const @ByRef Mat mask );
    /*
     * Remove duplicated keypoints.
     */
    public static native void removeDuplicated( @StdVector KeyPoint keypoints );

    /*
     * Retain the specified number of the best keypoints (according to the response)
     */
    public static native void retainBest( @StdVector KeyPoint keypoints, int npoints );
}


/************************************ Base Classes ************************************/

/*
 * Abstract base class for 2D image feature detectors.
 */
@Namespace("cv") public static class FeatureDetector extends Algorithm {
    static { Loader.load(); }
    /** Empty constructor. */
    public FeatureDetector() { }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public FeatureDetector(Pointer p) { super(p); }


    /*
     * Detect keypoints in an image.
     * image        The image.
     * keypoints    The detected keypoints.
     * mask         Mask specifying where to look for keypoints (optional). Must be a char
     *              matrix with non-zero values in the region of interest.
     */
    public native void detect( @Const @ByRef Mat image, @StdVector KeyPoint keypoints, @Const @ByRef Mat mask/*=Mat()*/ );
    public native void detect( @Const @ByRef Mat image, @StdVector KeyPoint keypoints );

    /*
     * Detect keypoints in an image set.
     * images       Image collection.
     * keypoints    Collection of keypoints detected in an input images. keypoints[i] is a set of keypoints detected in an images[i].
     * masks        Masks for image set. masks[i] is a mask for images[i].
     */
    public native void detect( @Const @ByRef MatVector images, @ByRef KeyPointVectorVector keypoints, @Const @ByRef MatVector masks/*=vector<Mat>()*/ );
    public native void detect( @Const @ByRef MatVector images, @ByRef KeyPointVectorVector keypoints );

    // Return true if detector object is empty
    public native @Cast("bool") boolean empty();

    // Create feature detector by detector name.
    public static native @Ptr FeatureDetector create( @StdString BytePointer detectorType );
    public static native @Ptr FeatureDetector create( @StdString String detectorType );
}


/*
 * Abstract base class for computing descriptors for image keypoints.
 *
 * In this interface we assume a keypoint descriptor can be represented as a
 * dense, fixed-dimensional vector of some basic type. Most descriptors used
 * in practice follow this pattern, as it makes it very easy to compute
 * distances between descriptors. Therefore we represent a collection of
 * descriptors as a Mat, where each row is one keypoint descriptor.
 */
@Namespace("cv") public static class DescriptorExtractor extends Algorithm {
    static { Loader.load(); }
    /** Empty constructor. */
    public DescriptorExtractor() { }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public DescriptorExtractor(Pointer p) { super(p); }


    /*
     * Compute the descriptors for a set of keypoints in an image.
     * image        The image.
     * keypoints    The input keypoints. Keypoints for which a descriptor cannot be computed are removed.
     * descriptors  Copmputed descriptors. Row i is the descriptor for keypoint i.
     */
    public native void compute( @Const @ByRef Mat image, @StdVector KeyPoint keypoints, @ByRef Mat descriptors );

    /*
     * Compute the descriptors for a keypoints collection detected in image collection.
     * images       Image collection.
     * keypoints    Input keypoints collection. keypoints[i] is keypoints detected in images[i].
     *              Keypoints for which a descriptor cannot be computed are removed.
     * descriptors  Descriptor collection. descriptors[i] are descriptors computed for set keypoints[i].
     */
    public native void compute( @Const @ByRef MatVector images, @ByRef KeyPointVectorVector keypoints, @ByRef MatVector descriptors );

    public native int descriptorSize();
    public native int descriptorType();

    public native @Cast("bool") boolean empty();

    public static native @Ptr DescriptorExtractor create( @StdString BytePointer descriptorExtractorType );
    public static native @Ptr DescriptorExtractor create( @StdString String descriptorExtractorType );
}



/*
 * Abstract base class for simultaneous 2D feature detection descriptor extraction.
 */
@Namespace("cv") public static class Feature2D extends FeatureDetector {
    static { Loader.load(); }
    /** Empty constructor. */
    public Feature2D() { }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public Feature2D(Pointer p) { super(p); }
    public DescriptorExtractor asDescriptorExtractor() { return asDescriptorExtractor(this); }
    @Namespace public static native @Name("static_cast<cv::DescriptorExtractor*>") DescriptorExtractor asDescriptorExtractor(Feature2D pointer);

    /*
     * Detect keypoints in an image.
     * image        The image.
     * keypoints    The detected keypoints.
     * mask         Mask specifying where to look for keypoints (optional). Must be a char
     *              matrix with non-zero values in the region of interest.
     * useProvidedKeypoints If true, the method will skip the detection phase and will compute
     *                      descriptors for the provided keypoints
     */
    public native @Name("operator()") void detectAndCompute( @ByVal Mat image, @ByVal Mat mask,
                                         @StdVector KeyPoint keypoints,
                                         @ByVal Mat descriptors,
                                         @Cast("bool") boolean useProvidedKeypoints/*=false*/ );
    public native @Name("operator()") void detectAndCompute( @ByVal Mat image, @ByVal Mat mask,
                                         @StdVector KeyPoint keypoints,
                                         @ByVal Mat descriptors );

    public native void compute( @Const @ByRef Mat image, @StdVector KeyPoint keypoints, @ByRef Mat descriptors );

    // Create feature detector and descriptor extractor by name.
    public static native @Ptr Feature2D create( @StdString BytePointer name );
    public static native @Ptr Feature2D create( @StdString String name );
}

/**
  BRISK implementation
*/
@Namespace("cv") @NoOffset public static class BRISK extends Feature2D {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public BRISK(Pointer p) { super(p); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public BRISK(int size) { allocateArray(size); }
    private native void allocateArray(int size);
    @Override public BRISK position(int position) {
        return (BRISK)super.position(position);
    }

    public BRISK(int thresh/*=30*/, int octaves/*=3*/, float patternScale/*=1.0f*/) { allocate(thresh, octaves, patternScale); }
    private native void allocate(int thresh/*=30*/, int octaves/*=3*/, float patternScale/*=1.0f*/);
    public BRISK() { allocate(); }
    private native void allocate();

    // returns the descriptor size in bytes
    public native int descriptorSize();
    // returns the descriptor type
    public native int descriptorType();

    // Compute the BRISK features on an image
    public native @Name("operator()") void apply(@ByVal Mat image, @ByVal Mat mask, @StdVector KeyPoint keypoints);

    // Compute the BRISK features and descriptors on an image
    public native @Name("operator()") void apply( @ByVal Mat image, @ByVal Mat mask, @StdVector KeyPoint keypoints,
                          @ByVal Mat descriptors, @Cast("bool") boolean useProvidedKeypoints/*=false*/ );
    public native @Name("operator()") void apply( @ByVal Mat image, @ByVal Mat mask, @StdVector KeyPoint keypoints,
                          @ByVal Mat descriptors );

    public native AlgorithmInfo info();

    // custom setup
    public BRISK(@StdVector FloatPointer radiusList, @StdVector IntPointer numberList,
            float dMax/*=5.85f*/, float dMin/*=8.2f*/, @StdVector IntPointer indexChange/*=std::vector<int>()*/) { allocate(radiusList, numberList, dMax, dMin, indexChange); }
    private native void allocate(@StdVector FloatPointer radiusList, @StdVector IntPointer numberList,
            float dMax/*=5.85f*/, float dMin/*=8.2f*/, @StdVector IntPointer indexChange/*=std::vector<int>()*/);
    public BRISK(@StdVector FloatPointer radiusList, @StdVector IntPointer numberList) { allocate(radiusList, numberList); }
    private native void allocate(@StdVector FloatPointer radiusList, @StdVector IntPointer numberList);
    public BRISK(@StdVector FloatBuffer radiusList, @StdVector IntBuffer numberList,
            float dMax/*=5.85f*/, float dMin/*=8.2f*/, @StdVector IntBuffer indexChange/*=std::vector<int>()*/) { allocate(radiusList, numberList, dMax, dMin, indexChange); }
    private native void allocate(@StdVector FloatBuffer radiusList, @StdVector IntBuffer numberList,
            float dMax/*=5.85f*/, float dMin/*=8.2f*/, @StdVector IntBuffer indexChange/*=std::vector<int>()*/);
    public BRISK(@StdVector FloatBuffer radiusList, @StdVector IntBuffer numberList) { allocate(radiusList, numberList); }
    private native void allocate(@StdVector FloatBuffer radiusList, @StdVector IntBuffer numberList);
    public BRISK(@StdVector float[] radiusList, @StdVector int[] numberList,
            float dMax/*=5.85f*/, float dMin/*=8.2f*/, @StdVector int[] indexChange/*=std::vector<int>()*/) { allocate(radiusList, numberList, dMax, dMin, indexChange); }
    private native void allocate(@StdVector float[] radiusList, @StdVector int[] numberList,
            float dMax/*=5.85f*/, float dMin/*=8.2f*/, @StdVector int[] indexChange/*=std::vector<int>()*/);
    public BRISK(@StdVector float[] radiusList, @StdVector int[] numberList) { allocate(radiusList, numberList); }
    private native void allocate(@StdVector float[] radiusList, @StdVector int[] numberList);

    // call this to generate the kernel:
    // circle of radius r (pixels), with n points;
    // short pairings with dMax, long pairings with dMin
    public native void generateKernel(@StdVector FloatPointer radiusList,
            @StdVector IntPointer numberList, float dMax/*=5.85f*/, float dMin/*=8.2f*/,
            @StdVector IntPointer indexChange/*=std::vector<int>()*/);
    public native void generateKernel(@StdVector FloatPointer radiusList,
            @StdVector IntPointer numberList);
    public native void generateKernel(@StdVector FloatBuffer radiusList,
            @StdVector IntBuffer numberList, float dMax/*=5.85f*/, float dMin/*=8.2f*/,
            @StdVector IntBuffer indexChange/*=std::vector<int>()*/);
    public native void generateKernel(@StdVector FloatBuffer radiusList,
            @StdVector IntBuffer numberList);
    public native void generateKernel(@StdVector float[] radiusList,
            @StdVector int[] numberList, float dMax/*=5.85f*/, float dMin/*=8.2f*/,
            @StdVector int[] indexChange/*=std::vector<int>()*/);
    public native void generateKernel(@StdVector float[] radiusList,
            @StdVector int[] numberList);
}


/**
 ORB implementation.
*/
@Namespace("cv") @NoOffset public static class ORB extends Feature2D {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public ORB(Pointer p) { super(p); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public ORB(int size) { allocateArray(size); }
    private native void allocateArray(int size);
    @Override public ORB position(int position) {
        return (ORB)super.position(position);
    }

    // the size of the signature in bytes
    /** enum cv::ORB:: */
    public static final int kBytes = 32, HARRIS_SCORE= 0, FAST_SCORE= 1;

    public ORB(int nfeatures/*=500*/, float scaleFactor/*=1.2f*/, int nlevels/*=8*/, int edgeThreshold/*=31*/,
            int firstLevel/*=0*/, int WTA_K/*=2*/, int scoreType/*=ORB::HARRIS_SCORE*/, int patchSize/*=31*/ ) { allocate(nfeatures, scaleFactor, nlevels, edgeThreshold, firstLevel, WTA_K, scoreType, patchSize); }
    private native void allocate(int nfeatures/*=500*/, float scaleFactor/*=1.2f*/, int nlevels/*=8*/, int edgeThreshold/*=31*/,
            int firstLevel/*=0*/, int WTA_K/*=2*/, int scoreType/*=ORB::HARRIS_SCORE*/, int patchSize/*=31*/ );
    public ORB( ) { allocate(); }
    private native void allocate( );

    // returns the descriptor size in bytes
    public native int descriptorSize();
    // returns the descriptor type
    public native int descriptorType();

    // Compute the ORB features and descriptors on an image
    public native @Name("operator()") void apply(@ByVal Mat image, @ByVal Mat mask, @StdVector KeyPoint keypoints);

    // Compute the ORB features and descriptors on an image
    public native @Name("operator()") void apply( @ByVal Mat image, @ByVal Mat mask, @StdVector KeyPoint keypoints,
                         @ByVal Mat descriptors, @Cast("bool") boolean useProvidedKeypoints/*=false*/ );
    public native @Name("operator()") void apply( @ByVal Mat image, @ByVal Mat mask, @StdVector KeyPoint keypoints,
                         @ByVal Mat descriptors );

    public native AlgorithmInfo info();
}

/**
  FREAK implementation
*/
@Namespace("cv") @NoOffset public static class FREAK extends DescriptorExtractor {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public FREAK(Pointer p) { super(p); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public FREAK(int size) { allocateArray(size); }
    private native void allocateArray(int size);
    @Override public FREAK position(int position) {
        return (FREAK)super.position(position);
    }

    /** Constructor
         * @param orientationNormalized enable orientation normalization
         * @param scaleNormalized enable scale normalization
         * @param patternScale scaling of the description pattern
         * @param nOctaves number of octaves covered by the detected keypoints
         * @param selectedPairs (optional) user defined selected pairs
    */
    public FREAK( @Cast("bool") boolean orientationNormalized/*=true*/,
               @Cast("bool") boolean scaleNormalized/*=true*/,
               float patternScale/*=22.0f*/,
               int nOctaves/*=4*/,
               @StdVector IntPointer selectedPairs/*=vector<int>()*/) { allocate(orientationNormalized, scaleNormalized, patternScale, nOctaves, selectedPairs); }
    private native void allocate( @Cast("bool") boolean orientationNormalized/*=true*/,
               @Cast("bool") boolean scaleNormalized/*=true*/,
               float patternScale/*=22.0f*/,
               int nOctaves/*=4*/,
               @StdVector IntPointer selectedPairs/*=vector<int>()*/);
    public FREAK() { allocate(); }
    private native void allocate();
    public FREAK( @Cast("bool") boolean orientationNormalized/*=true*/,
               @Cast("bool") boolean scaleNormalized/*=true*/,
               float patternScale/*=22.0f*/,
               int nOctaves/*=4*/,
               @StdVector IntBuffer selectedPairs/*=vector<int>()*/) { allocate(orientationNormalized, scaleNormalized, patternScale, nOctaves, selectedPairs); }
    private native void allocate( @Cast("bool") boolean orientationNormalized/*=true*/,
               @Cast("bool") boolean scaleNormalized/*=true*/,
               float patternScale/*=22.0f*/,
               int nOctaves/*=4*/,
               @StdVector IntBuffer selectedPairs/*=vector<int>()*/);
    public FREAK( @Cast("bool") boolean orientationNormalized/*=true*/,
               @Cast("bool") boolean scaleNormalized/*=true*/,
               float patternScale/*=22.0f*/,
               int nOctaves/*=4*/,
               @StdVector int[] selectedPairs/*=vector<int>()*/) { allocate(orientationNormalized, scaleNormalized, patternScale, nOctaves, selectedPairs); }
    private native void allocate( @Cast("bool") boolean orientationNormalized/*=true*/,
               @Cast("bool") boolean scaleNormalized/*=true*/,
               float patternScale/*=22.0f*/,
               int nOctaves/*=4*/,
               @StdVector int[] selectedPairs/*=vector<int>()*/);
    
    

    /** returns the descriptor length in bytes */
    public native int descriptorSize();

    /** returns the descriptor type */
    public native int descriptorType();

    /** select the 512 "best description pairs"
         * @param images grayscale images set
         * @param keypoints set of detected keypoints
         * @param corrThresh correlation threshold
         * @param verbose print construction information
         * @return list of best pair indexes
    */
    public native @StdVector IntPointer selectPairs( @Const @ByRef MatVector images, @ByRef KeyPointVectorVector keypoints,
                          double corrThresh/*=0.7*/, @Cast("bool") boolean verbose/*=true*/ );
    public native @StdVector IntPointer selectPairs( @Const @ByRef MatVector images, @ByRef KeyPointVectorVector keypoints );

    public native AlgorithmInfo info();

    /** enum cv::FREAK:: */
    public static final int
        NB_SCALES = 64, NB_PAIRS = 512, NB_ORIENPAIRS = 45;
}


/**
 Maximal Stable Extremal Regions class.

 The class implements MSER algorithm introduced by J. Matas.
 Unlike SIFT, SURF and many other detectors in OpenCV, this is salient region detector,
 not the salient point detector.

 It returns the regions, each of those is encoded as a contour.
*/
@Namespace("cv") @NoOffset public static class MSER extends FeatureDetector {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public MSER(Pointer p) { super(p); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public MSER(int size) { allocateArray(size); }
    private native void allocateArray(int size);
    @Override public MSER position(int position) {
        return (MSER)super.position(position);
    }

    /** the full constructor */
    public MSER( int _delta/*=5*/, int _min_area/*=60*/, int _max_area/*=14400*/,
              double _max_variation/*=0.25*/, double _min_diversity/*=.2*/,
              int _max_evolution/*=200*/, double _area_threshold/*=1.01*/,
              double _min_margin/*=0.003*/, int _edge_blur_size/*=5*/ ) { allocate(_delta, _min_area, _max_area, _max_variation, _min_diversity, _max_evolution, _area_threshold, _min_margin, _edge_blur_size); }
    private native void allocate( int _delta/*=5*/, int _min_area/*=60*/, int _max_area/*=14400*/,
              double _max_variation/*=0.25*/, double _min_diversity/*=.2*/,
              int _max_evolution/*=200*/, double _area_threshold/*=1.01*/,
              double _min_margin/*=0.003*/, int _edge_blur_size/*=5*/ );
    public MSER( ) { allocate(); }
    private native void allocate( );

    /** the operator that extracts the MSERs from the image or the specific part of it */
    public native @Name("operator()") void detect( @Const @ByRef Mat image, @ByRef PointVectorVector msers,
                                            @Const @ByRef Mat mask/*=Mat()*/ );
    public native @Name("operator()") void detect( @Const @ByRef Mat image, @ByRef PointVectorVector msers );
    public native AlgorithmInfo info();
}

/**
 The "Star" Detector.

 The class implements the keypoint detector introduced by K. Konolige.
*/
@Namespace("cv") @NoOffset public static class StarDetector extends FeatureDetector {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public StarDetector(Pointer p) { super(p); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public StarDetector(int size) { allocateArray(size); }
    private native void allocateArray(int size);
    @Override public StarDetector position(int position) {
        return (StarDetector)super.position(position);
    }

    /** the full constructor */
    public StarDetector(int _maxSize/*=45*/, int _responseThreshold/*=30*/,
                     int _lineThresholdProjected/*=10*/,
                     int _lineThresholdBinarized/*=8*/,
                     int _suppressNonmaxSize/*=5*/) { allocate(_maxSize, _responseThreshold, _lineThresholdProjected, _lineThresholdBinarized, _suppressNonmaxSize); }
    private native void allocate(int _maxSize/*=45*/, int _responseThreshold/*=30*/,
                     int _lineThresholdProjected/*=10*/,
                     int _lineThresholdBinarized/*=8*/,
                     int _suppressNonmaxSize/*=5*/);
    public StarDetector() { allocate(); }
    private native void allocate();

    /** finds the keypoints in the image */
    public native @Name("operator()") void detect(@Const @ByRef Mat image,
                    @StdVector KeyPoint keypoints);

    public native AlgorithmInfo info();
}

/** detects corners using FAST algorithm by E. Rosten */
@Namespace("cv") public static native void FAST( @ByVal Mat image, @StdVector KeyPoint keypoints,
                      int threshold, @Cast("bool") boolean nonmaxSuppression/*=true*/ );
@Namespace("cv") public static native void FAST( @ByVal Mat image, @StdVector KeyPoint keypoints,
                      int threshold );

@Namespace("cv") public static native void FASTX( @ByVal Mat image, @StdVector KeyPoint keypoints,
                      int threshold, @Cast("bool") boolean nonmaxSuppression, int type );

@Namespace("cv") @NoOffset public static class FastFeatureDetector extends FeatureDetector {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public FastFeatureDetector(Pointer p) { super(p); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public FastFeatureDetector(int size) { allocateArray(size); }
    private native void allocateArray(int size);
    @Override public FastFeatureDetector position(int position) {
        return (FastFeatureDetector)super.position(position);
    }


    /** enum cv::FastFeatureDetector:: */
    public static final int // Define it in old class to simplify migration to 2.5
      TYPE_5_8 = 0, TYPE_7_12 = 1, TYPE_9_16 = 2;

    public FastFeatureDetector( int threshold/*=10*/, @Cast("bool") boolean nonmaxSuppression/*=true*/ ) { allocate(threshold, nonmaxSuppression); }
    private native void allocate( int threshold/*=10*/, @Cast("bool") boolean nonmaxSuppression/*=true*/ );
    public FastFeatureDetector( ) { allocate(); }
    private native void allocate( );
    public native AlgorithmInfo info();
}


@Namespace("cv") @NoOffset public static class GFTTDetector extends FeatureDetector {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public GFTTDetector(Pointer p) { super(p); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public GFTTDetector(int size) { allocateArray(size); }
    private native void allocateArray(int size);
    @Override public GFTTDetector position(int position) {
        return (GFTTDetector)super.position(position);
    }

    public GFTTDetector( int maxCorners/*=1000*/, double qualityLevel/*=0.01*/, double minDistance/*=1*/,
                              int blockSize/*=3*/, @Cast("bool") boolean useHarrisDetector/*=false*/, double k/*=0.04*/ ) { allocate(maxCorners, qualityLevel, minDistance, blockSize, useHarrisDetector, k); }
    private native void allocate( int maxCorners/*=1000*/, double qualityLevel/*=0.01*/, double minDistance/*=1*/,
                              int blockSize/*=3*/, @Cast("bool") boolean useHarrisDetector/*=false*/, double k/*=0.04*/ );
    public GFTTDetector( ) { allocate(); }
    private native void allocate( );
    public native AlgorithmInfo info();
}

@Namespace("cv") @NoOffset public static class SimpleBlobDetector extends FeatureDetector {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public SimpleBlobDetector(Pointer p) { super(p); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public SimpleBlobDetector(int size) { allocateArray(size); }
    private native void allocateArray(int size);
    @Override public SimpleBlobDetector position(int position) {
        return (SimpleBlobDetector)super.position(position);
    }

  @NoOffset public static class Params extends Pointer {
      static { Loader.load(); }
      /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
      public Params(Pointer p) { super(p); }
      /** Native array allocator. Access with {@link Pointer#position(int)}. */
      public Params(int size) { allocateArray(size); }
      private native void allocateArray(int size);
      @Override public Params position(int position) {
          return (Params)super.position(position);
      }
  
      public Params() { allocate(); }
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

  public SimpleBlobDetector(@Const @ByRef Params parameters/*=SimpleBlobDetector::Params()*/) { allocate(parameters); }
  private native void allocate(@Const @ByRef Params parameters/*=SimpleBlobDetector::Params()*/);
  public SimpleBlobDetector() { allocate(); }
  private native void allocate();

  public native void read( @Const @ByRef FileNode fn );
  public native void write( @ByRef FileStorage fs );
}


@Namespace("cv") @NoOffset public static class DenseFeatureDetector extends FeatureDetector {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public DenseFeatureDetector(Pointer p) { super(p); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public DenseFeatureDetector(int size) { allocateArray(size); }
    private native void allocateArray(int size);
    @Override public DenseFeatureDetector position(int position) {
        return (DenseFeatureDetector)super.position(position);
    }

    public DenseFeatureDetector( float initFeatureScale/*=1.f*/, int featureScaleLevels/*=1*/,
                                       float featureScaleMul/*=0.1f*/,
                                       int initXyStep/*=6*/, int initImgBound/*=0*/,
                                       @Cast("bool") boolean varyXyStepWithScale/*=true*/,
                                       @Cast("bool") boolean varyImgBoundWithScale/*=false*/ ) { allocate(initFeatureScale, featureScaleLevels, featureScaleMul, initXyStep, initImgBound, varyXyStepWithScale, varyImgBoundWithScale); }
    private native void allocate( float initFeatureScale/*=1.f*/, int featureScaleLevels/*=1*/,
                                       float featureScaleMul/*=0.1f*/,
                                       int initXyStep/*=6*/, int initImgBound/*=0*/,
                                       @Cast("bool") boolean varyXyStepWithScale/*=true*/,
                                       @Cast("bool") boolean varyImgBoundWithScale/*=false*/ );
    public DenseFeatureDetector( ) { allocate(); }
    private native void allocate( );
    public native AlgorithmInfo info();
}

/*
 * Adapts a detector to partition the source image into a grid and detect
 * points in each cell.
 */
@Namespace("cv") @NoOffset public static class GridAdaptedFeatureDetector extends FeatureDetector {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public GridAdaptedFeatureDetector(Pointer p) { super(p); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public GridAdaptedFeatureDetector(int size) { allocateArray(size); }
    private native void allocateArray(int size);
    @Override public GridAdaptedFeatureDetector position(int position) {
        return (GridAdaptedFeatureDetector)super.position(position);
    }

    /*
     * detector            Detector that will be adapted.
     * maxTotalKeypoints   Maximum count of keypoints detected on the image. Only the strongest keypoints
     *                      will be keeped.
     * gridRows            Grid rows count.
     * gridCols            Grid column count.
     */
    public GridAdaptedFeatureDetector( @Ptr FeatureDetector detector/*=0*/,
                                            int maxTotalKeypoints/*=1000*/,
                                            int gridRows/*=4*/, int gridCols/*=4*/ ) { allocate(detector, maxTotalKeypoints, gridRows, gridCols); }
    private native void allocate( @Ptr FeatureDetector detector/*=0*/,
                                            int maxTotalKeypoints/*=1000*/,
                                            int gridRows/*=4*/, int gridCols/*=4*/ );
    public GridAdaptedFeatureDetector( ) { allocate(); }
    private native void allocate( );

    // TODO implement read/write
    public native @Cast("bool") boolean empty();

    public native AlgorithmInfo info();
}

/*
 * Adapts a detector to detect points over multiple levels of a Gaussian
 * pyramid. Useful for detectors that are not inherently scaled.
 */
@Namespace("cv") @NoOffset public static class PyramidAdaptedFeatureDetector extends FeatureDetector {
    static { Loader.load(); }
    /** Empty constructor. */
    public PyramidAdaptedFeatureDetector() { }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public PyramidAdaptedFeatureDetector(Pointer p) { super(p); }

    // maxLevel - The 0-based index of the last pyramid layer
    public PyramidAdaptedFeatureDetector( @Ptr FeatureDetector detector, int maxLevel/*=2*/ ) { allocate(detector, maxLevel); }
    private native void allocate( @Ptr FeatureDetector detector, int maxLevel/*=2*/ );
    public PyramidAdaptedFeatureDetector( @Ptr FeatureDetector detector ) { allocate(detector); }
    private native void allocate( @Ptr FeatureDetector detector );

    // TODO implement read/write
    public native @Cast("bool") boolean empty();
}

/** \brief A feature detector parameter adjuster, this is used by the DynamicAdaptedFeatureDetector
 *  and is a wrapper for FeatureDetector that allow them to be adjusted after a detection
 */
@Namespace("cv") public static class AdjusterAdapter extends FeatureDetector {
    static { Loader.load(); }
    /** Empty constructor. */
    public AdjusterAdapter() { }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public AdjusterAdapter(Pointer p) { super(p); }

    /** pure virtual interface
     */
    /** too few features were detected so, adjust the detector params accordingly
     * \param min the minimum number of desired features
     * \param n_detected the number previously detected
     */
    public native void tooFew(int min, int n_detected);
    /** too many features were detected so, adjust the detector params accordingly
     * \param max the maximum number of desired features
     * \param n_detected the number previously detected
     */
    public native void tooMany(int max, int n_detected);
    /** are params maxed out or still valid?
     * \return false if the parameters can't be adjusted any more
     */
    public native @Cast("bool") boolean good();

    public native @Ptr AdjusterAdapter clone();

    public static native @Ptr AdjusterAdapter create( @StdString BytePointer detectorType );
    public static native @Ptr AdjusterAdapter create( @StdString String detectorType );
}
/** \brief an adaptively adjusting detector that iteratively detects until the desired number
 * of features are detected.
 *  Beware that this is not thread safe - as the adjustment of parameters breaks the const
 *  of the detection routine...
 *  /TODO Make this const correct and thread safe
 *
 *  sample usage:
 //will create a detector that attempts to find 100 - 110 FAST Keypoints, and will at most run
 //FAST feature detection 10 times until that number of keypoints are found
 Ptr<FeatureDetector> detector(new DynamicAdaptedFeatureDetector(new FastAdjuster(20,true),100, 110, 10));

 */
@Namespace("cv") @NoOffset public static class DynamicAdaptedFeatureDetector extends FeatureDetector {
    static { Loader.load(); }
    /** Empty constructor. */
    public DynamicAdaptedFeatureDetector() { }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public DynamicAdaptedFeatureDetector(Pointer p) { super(p); }


    /** \param adjuster an AdjusterAdapter that will do the detection and parameter adjustment
     *  \param max_features the maximum desired number of features
     *  \param max_iters the maximum number of times to try to adjust the feature detector params
     *          for the FastAdjuster this can be high, but with Star or Surf this can get time consuming
     *  \param min_features the minimum desired features
     */
    public DynamicAdaptedFeatureDetector( @Ptr AdjusterAdapter adjuster, int min_features/*=400*/, int max_features/*=500*/, int max_iters/*=5*/ ) { allocate(adjuster, min_features, max_features, max_iters); }
    private native void allocate( @Ptr AdjusterAdapter adjuster, int min_features/*=400*/, int max_features/*=500*/, int max_iters/*=5*/ );
    public DynamicAdaptedFeatureDetector( @Ptr AdjusterAdapter adjuster ) { allocate(adjuster); }
    private native void allocate( @Ptr AdjusterAdapter adjuster );

    public native @Cast("bool") boolean empty();
}

/**\brief an adjust for the FAST detector. This will basically decrement or increment the
 * threshold by 1
 */
@Namespace("cv") @NoOffset public static class FastAdjuster extends AdjusterAdapter {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public FastAdjuster(Pointer p) { super(p); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public FastAdjuster(int size) { allocateArray(size); }
    private native void allocateArray(int size);
    @Override public FastAdjuster position(int position) {
        return (FastAdjuster)super.position(position);
    }

    /**\param init_thresh the initial threshold to start with, default = 20
     * \param nonmax whether to use non max or not for fast feature detection
     * \param min_thresh
     * \param max_thresh
     */
    public FastAdjuster(int init_thresh/*=20*/, @Cast("bool") boolean nonmax/*=true*/, int min_thresh/*=1*/, int max_thresh/*=200*/) { allocate(init_thresh, nonmax, min_thresh, max_thresh); }
    private native void allocate(int init_thresh/*=20*/, @Cast("bool") boolean nonmax/*=true*/, int min_thresh/*=1*/, int max_thresh/*=200*/);
    public FastAdjuster() { allocate(); }
    private native void allocate();

    public native void tooFew(int minv, int n_detected);
    public native void tooMany(int maxv, int n_detected);
    public native @Cast("bool") boolean good();

    public native @Ptr AdjusterAdapter clone();
}


/** An adjuster for StarFeatureDetector, this one adjusts the responseThreshold for now
 * TODO find a faster way to converge the parameters for Star - use CvStarDetectorParams
 */
@Namespace("cv") @NoOffset public static class StarAdjuster extends AdjusterAdapter {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public StarAdjuster(Pointer p) { super(p); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public StarAdjuster(int size) { allocateArray(size); }
    private native void allocateArray(int size);
    @Override public StarAdjuster position(int position) {
        return (StarAdjuster)super.position(position);
    }

    public StarAdjuster(double initial_thresh/*=30.0*/, double min_thresh/*=2.*/, double max_thresh/*=200.*/) { allocate(initial_thresh, min_thresh, max_thresh); }
    private native void allocate(double initial_thresh/*=30.0*/, double min_thresh/*=2.*/, double max_thresh/*=200.*/);
    public StarAdjuster() { allocate(); }
    private native void allocate();

    public native void tooFew(int minv, int n_detected);
    public native void tooMany(int maxv, int n_detected);
    public native @Cast("bool") boolean good();

    public native @Ptr AdjusterAdapter clone();
}

@Namespace("cv") @NoOffset public static class SurfAdjuster extends AdjusterAdapter {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public SurfAdjuster(Pointer p) { super(p); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public SurfAdjuster(int size) { allocateArray(size); }
    private native void allocateArray(int size);
    @Override public SurfAdjuster position(int position) {
        return (SurfAdjuster)super.position(position);
    }

    public SurfAdjuster( double initial_thresh/*=400.f*/, double min_thresh/*=2*/, double max_thresh/*=1000*/ ) { allocate(initial_thresh, min_thresh, max_thresh); }
    private native void allocate( double initial_thresh/*=400.f*/, double min_thresh/*=2*/, double max_thresh/*=1000*/ );
    public SurfAdjuster( ) { allocate(); }
    private native void allocate( );

    public native void tooFew(int minv, int n_detected);
    public native void tooMany(int maxv, int n_detected);
    public native @Cast("bool") boolean good();

    public native @Ptr AdjusterAdapter clone();
}

@Namespace("cv") public static native @ByVal Mat windowedMatchingMask( @StdVector KeyPoint keypoints1, @StdVector KeyPoint keypoints2,
                                     float maxDeltaX, float maxDeltaY );



/*
 * OpponentColorDescriptorExtractor
 *
 * Adapts a descriptor extractor to compute descripors in Opponent Color Space
 * (refer to van de Sande et al., CGIV 2008 "Color Descriptors for Object Category Recognition").
 * Input RGB image is transformed in Opponent Color Space. Then unadapted descriptor extractor
 * (set in constructor) computes descriptors on each of the three channel and concatenate
 * them into a single color descriptor.
 */
@Namespace("cv") @NoOffset public static class OpponentColorDescriptorExtractor extends DescriptorExtractor {
    static { Loader.load(); }
    /** Empty constructor. */
    public OpponentColorDescriptorExtractor() { }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public OpponentColorDescriptorExtractor(Pointer p) { super(p); }

    public OpponentColorDescriptorExtractor( @Ptr DescriptorExtractor descriptorExtractor ) { allocate(descriptorExtractor); }
    private native void allocate( @Ptr DescriptorExtractor descriptorExtractor );

    public native void read( @Const @ByRef FileNode arg0 );
    public native void write( @ByRef FileStorage arg0 );

    public native int descriptorSize();
    public native int descriptorType();

    public native @Cast("bool") boolean empty();
}

/*
 * BRIEF Descriptor
 */
@Namespace("cv") @NoOffset public static class BriefDescriptorExtractor extends DescriptorExtractor {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public BriefDescriptorExtractor(Pointer p) { super(p); }

    @MemberGetter public static native int PATCH_SIZE();
    @MemberGetter public static native int KERNEL_SIZE();

    // bytes is a length of descriptor in bytes. It can be equal 16, 32 or 64 bytes.
    public BriefDescriptorExtractor( int bytes/*=32*/ ) { allocate(bytes); }
    private native void allocate( int bytes/*=32*/ );
    public BriefDescriptorExtractor( ) { allocate(); }
    private native void allocate( );

    public native void read( @Const @ByRef FileNode arg0 );
    public native void write( @ByRef FileStorage arg0 );

    public native int descriptorSize();
    public native int descriptorType();

    /** @todo read and write for brief */

    public native AlgorithmInfo info();
}


/****************************************************************************************\
*                                      Distance                                          *
\****************************************************************************************/

@Name("cv::Accumulator<unsigned char>") public static class Accumulator extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public Accumulator() { allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public Accumulator(int size) { allocateArray(size); }
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

/*
 * Hamming distance functor - counts the bit differences between two strings - useful for the Brief descriptor
 * bit count of A exclusive XOR'ed with B
 */
@Namespace("cv") public static class Hamming extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public Hamming() { allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public Hamming(int size) { allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public Hamming(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public Hamming position(int position) {
        return (Hamming)super.position(position);
    }

    /** enum cv::Hamming:: */
    public static final int normType =  NORM_HAMMING;

    /** this will count the bits in a ^ b
     */
    public native @Cast("cv::Hamming::ResultType") @Name("operator()") int apply( @Cast("const unsigned char*") BytePointer a, @Cast("const unsigned char*") BytePointer b, int size );
    public native @Cast("cv::Hamming::ResultType") @Name("operator()") int apply( @Cast("const unsigned char*") ByteBuffer a, @Cast("const unsigned char*") ByteBuffer b, int size );
    public native @Cast("cv::Hamming::ResultType") @Name("operator()") int apply( @Cast("const unsigned char*") byte[] a, @Cast("const unsigned char*") byte[] b, int size );
}

/****************************************************************************************\
*                                      DMatch                                            *
\****************************************************************************************/
/*
 * Struct for matching: query descriptor index, train descriptor index, train image index and distance between descriptors.
 */
@Namespace("cv") @NoOffset public static class DMatch extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public DMatch(Pointer p) { super(p); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public DMatch(int size) { allocateArray(size); }
    private native void allocateArray(int size);
    @Override public DMatch position(int position) {
        return (DMatch)super.position(position);
    }

    public DMatch() { allocate(); }
    private native void allocate();
    public DMatch( int _queryIdx, int _trainIdx, float _distance ) { allocate(_queryIdx, _trainIdx, _distance); }
    private native void allocate( int _queryIdx, int _trainIdx, float _distance );
    public DMatch( int _queryIdx, int _trainIdx, int _imgIdx, float _distance ) { allocate(_queryIdx, _trainIdx, _imgIdx, _distance); }
    private native void allocate( int _queryIdx, int _trainIdx, int _imgIdx, float _distance );

    public native int queryIdx(); public native DMatch queryIdx(int queryIdx); // query descriptor index
    public native int trainIdx(); public native DMatch trainIdx(int trainIdx); // train descriptor index
    public native int imgIdx(); public native DMatch imgIdx(int imgIdx);   // train image index

    public native float distance(); public native DMatch distance(float distance);

    // less is better
    public native @Cast("bool") @Name("operator<") boolean lessThan( @Const @ByRef DMatch m );
}

/****************************************************************************************\
*                                  DescriptorMatcher                                     *
\****************************************************************************************/
/*
 * Abstract base class for matching two sets of descriptors.
 */
@Namespace("cv") @NoOffset public static class DescriptorMatcher extends Algorithm {
    static { Loader.load(); }
    /** Empty constructor. */
    public DescriptorMatcher() { }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public DescriptorMatcher(Pointer p) { super(p); }


    /*
     * Add descriptors to train descriptor collection.
     * descriptors      Descriptors to add. Each descriptors[i] is a descriptors set from one image.
     */
    public native void add( @Const @ByRef MatVector descriptors );
    /*
     * Get train descriptors collection.
     */
    public native @Const @ByRef MatVector getTrainDescriptors();
    /*
     * Clear train descriptors collection.
     */
    public native void clear();

    /*
     * Return true if there are not train descriptors in collection.
     */
    public native @Cast("bool") boolean empty();
    /*
     * Return true if the matcher supports mask in match methods.
     */
    public native @Cast("bool") boolean isMaskSupported();

    /*
     * Train matcher (e.g. train flann index).
     * In all methods to match the method train() is run every time before matching.
     * Some descriptor matchers (e.g. BruteForceMatcher) have empty implementation
     * of this method, other matchers really train their inner structures
     * (e.g. FlannBasedMatcher trains flann::Index). So nonempty implementation
     * of train() should check the class object state and do traing/retraining
     * only if the state requires that (e.g. FlannBasedMatcher trains flann::Index
     * if it has not trained yet or if new descriptors have been added to the train
     * collection).
     */
    public native void train();
    /*
     * Group of methods to match descriptors from image pair.
     * Method train() is run in this methods.
     */
    // Find one best match for each query descriptor (if mask is empty).
    public native void match( @Const @ByRef Mat queryDescriptors, @Const @ByRef Mat trainDescriptors,
                    @StdVector DMatch matches, @Const @ByRef Mat mask/*=Mat()*/ );
    public native void match( @Const @ByRef Mat queryDescriptors, @Const @ByRef Mat trainDescriptors,
                    @StdVector DMatch matches );
    // Find k best matches for each query descriptor (in increasing order of distances).
    // compactResult is used when mask is not empty. If compactResult is false matches
    // vector will have the same size as queryDescriptors rows. If compactResult is true
    // matches vector will not contain matches for fully masked out query descriptors.
    public native void knnMatch( @Const @ByRef Mat queryDescriptors, @Const @ByRef Mat trainDescriptors,
                       @ByRef DMatchVectorVector matches, int k,
                       @Const @ByRef Mat mask/*=Mat()*/, @Cast("bool") boolean compactResult/*=false*/ );
    public native void knnMatch( @Const @ByRef Mat queryDescriptors, @Const @ByRef Mat trainDescriptors,
                       @ByRef DMatchVectorVector matches, int k );
    // Find best matches for each query descriptor which have distance less than
    // maxDistance (in increasing order of distances).
    public native void radiusMatch( @Const @ByRef Mat queryDescriptors, @Const @ByRef Mat trainDescriptors,
                          @ByRef DMatchVectorVector matches, float maxDistance,
                          @Const @ByRef Mat mask/*=Mat()*/, @Cast("bool") boolean compactResult/*=false*/ );
    public native void radiusMatch( @Const @ByRef Mat queryDescriptors, @Const @ByRef Mat trainDescriptors,
                          @ByRef DMatchVectorVector matches, float maxDistance );
    /*
     * Group of methods to match descriptors from one image to image set.
     * See description of similar methods for matching image pair above.
     */
    public native void match( @Const @ByRef Mat queryDescriptors, @StdVector DMatch matches,
                    @Const @ByRef MatVector masks/*=vector<Mat>()*/ );
    public native void match( @Const @ByRef Mat queryDescriptors, @StdVector DMatch matches );
    public native void knnMatch( @Const @ByRef Mat queryDescriptors, @ByRef DMatchVectorVector matches, int k,
               @Const @ByRef MatVector masks/*=vector<Mat>()*/, @Cast("bool") boolean compactResult/*=false*/ );
    public native void knnMatch( @Const @ByRef Mat queryDescriptors, @ByRef DMatchVectorVector matches, int k );
    public native void radiusMatch( @Const @ByRef Mat queryDescriptors, @ByRef DMatchVectorVector matches, float maxDistance,
                       @Const @ByRef MatVector masks/*=vector<Mat>()*/, @Cast("bool") boolean compactResult/*=false*/ );
    public native void radiusMatch( @Const @ByRef Mat queryDescriptors, @ByRef DMatchVectorVector matches, float maxDistance );

    // Reads matcher object from a file node
    public native void read( @Const @ByRef FileNode arg0 );
    // Writes matcher object to a file storage
    public native void write( @ByRef FileStorage arg0 );

    // Clone the matcher. If emptyTrainData is false the method create deep copy of the object, i.e. copies
    // both parameters and train data. If emptyTrainData is true the method create object copy with current parameters
    // but with empty train data.
    public native @Ptr DescriptorMatcher clone( @Cast("bool") boolean emptyTrainData/*=false*/ );
    public native @Ptr DescriptorMatcher clone( );

    public static native @Ptr DescriptorMatcher create( @StdString BytePointer descriptorMatcherType );
    public static native @Ptr DescriptorMatcher create( @StdString String descriptorMatcherType );
}

/*
 * Brute-force descriptor matcher.
 *
 * For each descriptor in the first set, this matcher finds the closest
 * descriptor in the second set by trying each one.
 *
 * For efficiency, BruteForceMatcher is templated on the distance metric.
 * For float descriptors, a common choice would be cv::L2<float>.
 */
@Namespace("cv") @NoOffset public static class BFMatcher extends DescriptorMatcher {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public BFMatcher(Pointer p) { super(p); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public BFMatcher(int size) { allocateArray(size); }
    private native void allocateArray(int size);
    @Override public BFMatcher position(int position) {
        return (BFMatcher)super.position(position);
    }

    public BFMatcher( int normType/*=NORM_L2*/, @Cast("bool") boolean crossCheck/*=false*/ ) { allocate(normType, crossCheck); }
    private native void allocate( int normType/*=NORM_L2*/, @Cast("bool") boolean crossCheck/*=false*/ );
    public BFMatcher( ) { allocate(); }
    private native void allocate( );

    public native @Cast("bool") boolean isMaskSupported();

    public native @Ptr DescriptorMatcher clone( @Cast("bool") boolean emptyTrainData/*=false*/ );
    public native @Ptr DescriptorMatcher clone( );

    public native AlgorithmInfo info();
}


/*
 * Flann based matcher
 */
@Namespace("cv") @NoOffset public static class FlannBasedMatcher extends DescriptorMatcher {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public FlannBasedMatcher(Pointer p) { super(p); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public FlannBasedMatcher(int size) { allocateArray(size); }
    private native void allocateArray(int size);
    @Override public FlannBasedMatcher position(int position) {
        return (FlannBasedMatcher)super.position(position);
    }

    public FlannBasedMatcher( @Ptr IndexParams indexParams/*=new flann::KDTreeIndexParams()*/,
                           @Ptr SearchParams searchParams/*=new flann::SearchParams()*/ ) { allocate(indexParams, searchParams); }
    private native void allocate( @Ptr IndexParams indexParams/*=new flann::KDTreeIndexParams()*/,
                           @Ptr SearchParams searchParams/*=new flann::SearchParams()*/ );
    public FlannBasedMatcher( ) { allocate(); }
    private native void allocate( );

    public native void add( @Const @ByRef MatVector descriptors );
    public native void clear();

    // Reads matcher object from a file node
    public native void read( @Const @ByRef FileNode arg0 );
    // Writes matcher object to a file storage
    public native void write( @ByRef FileStorage arg0 );

    public native void train();
    public native @Cast("bool") boolean isMaskSupported();

    public native @Ptr DescriptorMatcher clone( @Cast("bool") boolean emptyTrainData/*=false*/ );
    public native @Ptr DescriptorMatcher clone( );

    public native AlgorithmInfo info();
}

/****************************************************************************************\
*                                GenericDescriptorMatcher                                *
\****************************************************************************************/
/*
 *   Abstract interface for a keypoint descriptor and matcher
 */

@Namespace("cv") @NoOffset public static class GenericDescriptorMatcher extends Pointer {
    static { Loader.load(); }
    /** Empty constructor. */
    public GenericDescriptorMatcher() { }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public GenericDescriptorMatcher(Pointer p) { super(p); }


    /*
     * Add train collection: images and keypoints from them.
     * images       A set of train images.
     * ketpoints    Keypoint collection that have been detected on train images.
     *
     * Keypoints for which a descriptor cannot be computed are removed. Such keypoints
     * must be filtered in this method befor adding keypoints to train collection "trainPointCollection".
     * If inheritor class need perform such prefiltering the method add() must be overloaded.
     * In the other class methods programmer has access to the train keypoints by a constant link.
     */
    public native void add( @Const @ByRef MatVector images,
                          @ByRef KeyPointVectorVector keypoints );

    public native @Const @ByRef MatVector getTrainImages();
    public native @Const @ByRef KeyPointVectorVector getTrainKeypoints();

    /*
     * Clear images and keypoints storing in train collection.
     */
    public native void clear();
    /*
     * Returns true if matcher supports mask to match descriptors.
     */
    public native @Cast("bool") boolean isMaskSupported();
    /*
     * Train some inner structures (e.g. flann index or decision trees).
     * train() methods is run every time in matching methods. So the method implementation
     * should has a check whether these inner structures need be trained/retrained or not.
     */
    public native void train();

    /*
     * Classifies query keypoints.
     * queryImage    The query image
     * queryKeypoints   Keypoints from the query image
     * trainImage    The train image
     * trainKeypoints   Keypoints from the train image
     */
    // Classify keypoints from query image under one train image.
    public native void classify( @Const @ByRef Mat queryImage, @StdVector KeyPoint queryKeypoints,
                               @Const @ByRef Mat trainImage, @StdVector KeyPoint trainKeypoints );
    // Classify keypoints from query image under train image collection.
    public native void classify( @Const @ByRef Mat queryImage, @StdVector KeyPoint queryKeypoints );

    /*
     * Group of methods to match keypoints from image pair.
     * Keypoints for which a descriptor cannot be computed are removed.
     * train() method is called here.
     */
    // Find one best match for each query descriptor (if mask is empty).
    public native void match( @Const @ByRef Mat queryImage, @StdVector KeyPoint queryKeypoints,
                    @Const @ByRef Mat trainImage, @StdVector KeyPoint trainKeypoints,
                    @StdVector DMatch matches, @Const @ByRef Mat mask/*=Mat()*/ );
    public native void match( @Const @ByRef Mat queryImage, @StdVector KeyPoint queryKeypoints,
                    @Const @ByRef Mat trainImage, @StdVector KeyPoint trainKeypoints,
                    @StdVector DMatch matches );
    // Find k best matches for each query keypoint (in increasing order of distances).
    // compactResult is used when mask is not empty. If compactResult is false matches
    // vector will have the same size as queryDescriptors rows.
    // If compactResult is true matches vector will not contain matches for fully masked out query descriptors.
    public native void knnMatch( @Const @ByRef Mat queryImage, @StdVector KeyPoint queryKeypoints,
                       @Const @ByRef Mat trainImage, @StdVector KeyPoint trainKeypoints,
                       @ByRef DMatchVectorVector matches, int k,
                       @Const @ByRef Mat mask/*=Mat()*/, @Cast("bool") boolean compactResult/*=false*/ );
    public native void knnMatch( @Const @ByRef Mat queryImage, @StdVector KeyPoint queryKeypoints,
                       @Const @ByRef Mat trainImage, @StdVector KeyPoint trainKeypoints,
                       @ByRef DMatchVectorVector matches, int k );
    // Find best matches for each query descriptor which have distance less than maxDistance (in increasing order of distances).
    public native void radiusMatch( @Const @ByRef Mat queryImage, @StdVector KeyPoint queryKeypoints,
                          @Const @ByRef Mat trainImage, @StdVector KeyPoint trainKeypoints,
                          @ByRef DMatchVectorVector matches, float maxDistance,
                          @Const @ByRef Mat mask/*=Mat()*/, @Cast("bool") boolean compactResult/*=false*/ );
    public native void radiusMatch( @Const @ByRef Mat queryImage, @StdVector KeyPoint queryKeypoints,
                          @Const @ByRef Mat trainImage, @StdVector KeyPoint trainKeypoints,
                          @ByRef DMatchVectorVector matches, float maxDistance );
    /*
     * Group of methods to match keypoints from one image to image set.
     * See description of similar methods for matching image pair above.
     */
    public native void match( @Const @ByRef Mat queryImage, @StdVector KeyPoint queryKeypoints,
                    @StdVector DMatch matches, @Const @ByRef MatVector masks/*=vector<Mat>()*/ );
    public native void match( @Const @ByRef Mat queryImage, @StdVector KeyPoint queryKeypoints,
                    @StdVector DMatch matches );
    public native void knnMatch( @Const @ByRef Mat queryImage, @StdVector KeyPoint queryKeypoints,
                       @ByRef DMatchVectorVector matches, int k,
                       @Const @ByRef MatVector masks/*=vector<Mat>()*/, @Cast("bool") boolean compactResult/*=false*/ );
    public native void knnMatch( @Const @ByRef Mat queryImage, @StdVector KeyPoint queryKeypoints,
                       @ByRef DMatchVectorVector matches, int k );
    public native void radiusMatch( @Const @ByRef Mat queryImage, @StdVector KeyPoint queryKeypoints,
                          @ByRef DMatchVectorVector matches, float maxDistance,
                          @Const @ByRef MatVector masks/*=vector<Mat>()*/, @Cast("bool") boolean compactResult/*=false*/ );
    public native void radiusMatch( @Const @ByRef Mat queryImage, @StdVector KeyPoint queryKeypoints,
                          @ByRef DMatchVectorVector matches, float maxDistance );

    // Reads matcher object from a file node
    public native void read( @Const @ByRef FileNode fn );
    // Writes matcher object to a file storage
    public native void write( @ByRef FileStorage fs );

    // Return true if matching object is empty (e.g. feature detector or descriptor matcher are empty)
    public native @Cast("bool") boolean empty();

    // Clone the matcher. If emptyTrainData is false the method create deep copy of the object, i.e. copies
    // both parameters and train data. If emptyTrainData is true the method create object copy with current parameters
    // but with empty train data.
    public native @Ptr GenericDescriptorMatcher clone( @Cast("bool") boolean emptyTrainData/*=false*/ );
    public native @Ptr GenericDescriptorMatcher clone( );

    public static native @Ptr GenericDescriptorMatcher create( @StdString BytePointer genericDescritptorMatcherType,
                                                     @StdString BytePointer paramsFilename/*=string()*/ );
    public static native @Ptr GenericDescriptorMatcher create( @StdString BytePointer genericDescritptorMatcherType );
    public static native @Ptr GenericDescriptorMatcher create( @StdString String genericDescritptorMatcherType,
                                                     @StdString String paramsFilename/*=string()*/ );
    public static native @Ptr GenericDescriptorMatcher create( @StdString String genericDescritptorMatcherType );
}


/****************************************************************************************\
*                                VectorDescriptorMatcher                                 *
\****************************************************************************************/

/*
 *  A class used for matching descriptors that can be described as vectors in a finite-dimensional space
 */

@Namespace("cv") @NoOffset public static class VectorDescriptorMatcher extends GenericDescriptorMatcher {
    static { Loader.load(); }
    /** Empty constructor. */
    public VectorDescriptorMatcher() { }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public VectorDescriptorMatcher(Pointer p) { super(p); }

    public VectorDescriptorMatcher( @Ptr DescriptorExtractor extractor, @Ptr DescriptorMatcher matcher ) { allocate(extractor, matcher); }
    private native void allocate( @Ptr DescriptorExtractor extractor, @Ptr DescriptorMatcher matcher );

    public native void add( @Const @ByRef MatVector imgCollection,
                          @ByRef KeyPointVectorVector pointCollection );

    public native void clear();

    public native void train();

    public native @Cast("bool") boolean isMaskSupported();

    public native void read( @Const @ByRef FileNode fn );
    public native void write( @ByRef FileStorage fs );
    public native @Cast("bool") boolean empty();

    public native @Ptr GenericDescriptorMatcher clone( @Cast("bool") boolean emptyTrainData/*=false*/ );
    public native @Ptr GenericDescriptorMatcher clone( );
}

/****************************************************************************************\
*                                   Drawing functions                                    *
\****************************************************************************************/
@Namespace("cv") public static class DrawMatchesFlags extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public DrawMatchesFlags() { allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public DrawMatchesFlags(int size) { allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public DrawMatchesFlags(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public DrawMatchesFlags position(int position) {
        return (DrawMatchesFlags)super.position(position);
    }

    /** enum cv::DrawMatchesFlags:: */
    public static final int DEFAULT = 0, // Output image matrix will be created (Mat::create),
                       // i.e. existing memory of output image may be reused.
                       // Two source image, matches and single keypoints will be drawn.
                       // For each keypoint only the center point will be drawn (without
                       // the circle around keypoint with keypoint size and orientation).
          DRAW_OVER_OUTIMG = 1, // Output image matrix will not be created (Mat::create).
                                // Matches will be drawn on existing content of output image.
          NOT_DRAW_SINGLE_POINTS = 2, // Single keypoints will not be drawn.
          DRAW_RICH_KEYPOINTS = 4; // For each keypoint the circle around keypoint with keypoint size and
                                  // orientation will be drawn.
}

// Draw keypoints.
@Namespace("cv") public static native void drawKeypoints( @Const @ByRef Mat image, @StdVector KeyPoint keypoints, @ByRef Mat outImage,
                               @Const @ByRef Scalar color/*=Scalar::all(-1)*/, int flags/*=DrawMatchesFlags::DEFAULT*/ );
@Namespace("cv") public static native void drawKeypoints( @Const @ByRef Mat image, @StdVector KeyPoint keypoints, @ByRef Mat outImage );

// Draws matches of keypints from two images on output image.
@Namespace("cv") public static native void drawMatches( @Const @ByRef Mat img1, @StdVector KeyPoint keypoints1,
                             @Const @ByRef Mat img2, @StdVector KeyPoint keypoints2,
                             @StdVector DMatch matches1to2, @ByRef Mat outImg,
                             @Const @ByRef Scalar matchColor/*=Scalar::all(-1)*/, @Const @ByRef Scalar singlePointColor/*=Scalar::all(-1)*/,
                             @Cast("char*") @StdVector BytePointer matchesMask/*=vector<char>()*/, int flags/*=DrawMatchesFlags::DEFAULT*/ );
@Namespace("cv") public static native void drawMatches( @Const @ByRef Mat img1, @StdVector KeyPoint keypoints1,
                             @Const @ByRef Mat img2, @StdVector KeyPoint keypoints2,
                             @StdVector DMatch matches1to2, @ByRef Mat outImg );
@Namespace("cv") public static native void drawMatches( @Const @ByRef Mat img1, @StdVector KeyPoint keypoints1,
                             @Const @ByRef Mat img2, @StdVector KeyPoint keypoints2,
                             @StdVector DMatch matches1to2, @ByRef Mat outImg,
                             @Const @ByRef Scalar matchColor/*=Scalar::all(-1)*/, @Const @ByRef Scalar singlePointColor/*=Scalar::all(-1)*/,
                             @Cast("char*") @StdVector ByteBuffer matchesMask/*=vector<char>()*/, int flags/*=DrawMatchesFlags::DEFAULT*/ );
@Namespace("cv") public static native void drawMatches( @Const @ByRef Mat img1, @StdVector KeyPoint keypoints1,
                             @Const @ByRef Mat img2, @StdVector KeyPoint keypoints2,
                             @StdVector DMatch matches1to2, @ByRef Mat outImg,
                             @Const @ByRef Scalar matchColor/*=Scalar::all(-1)*/, @Const @ByRef Scalar singlePointColor/*=Scalar::all(-1)*/,
                             @Cast("char*") @StdVector byte[] matchesMask/*=vector<char>()*/, int flags/*=DrawMatchesFlags::DEFAULT*/ );

@Namespace("cv") public static native void drawMatches( @Const @ByRef Mat img1, @StdVector KeyPoint keypoints1,
                             @Const @ByRef Mat img2, @StdVector KeyPoint keypoints2,
                             @Const @ByRef DMatchVectorVector matches1to2, @ByRef Mat outImg,
                             @Const @ByRef Scalar matchColor/*=Scalar::all(-1)*/, @Const @ByRef Scalar singlePointColor/*=Scalar::all(-1)*/,
                             @Cast("const std::vector<std::vector<char> >*") @ByRef ByteVectorVector matchesMask/*=vector<vector<char> >()*/, int flags/*=DrawMatchesFlags::DEFAULT*/ );
@Namespace("cv") public static native void drawMatches( @Const @ByRef Mat img1, @StdVector KeyPoint keypoints1,
                             @Const @ByRef Mat img2, @StdVector KeyPoint keypoints2,
                             @Const @ByRef DMatchVectorVector matches1to2, @ByRef Mat outImg );

/****************************************************************************************\
*   Functions to evaluate the feature detectors and [generic] descriptor extractors      *
\****************************************************************************************/

@Namespace("cv") public static native void evaluateFeatureDetector( @Const @ByRef Mat img1, @Const @ByRef Mat img2, @Const @ByRef Mat H1to2,
                                         @StdVector KeyPoint keypoints1, @StdVector KeyPoint keypoints2,
                                         @ByRef FloatPointer repeatability, @ByRef IntPointer correspCount,
                                         @Ptr FeatureDetector fdetector/*=Ptr<FeatureDetector>()*/ );
@Namespace("cv") public static native void evaluateFeatureDetector( @Const @ByRef Mat img1, @Const @ByRef Mat img2, @Const @ByRef Mat H1to2,
                                         @StdVector KeyPoint keypoints1, @StdVector KeyPoint keypoints2,
                                         @ByRef FloatPointer repeatability, @ByRef IntPointer correspCount );
@Namespace("cv") public static native void evaluateFeatureDetector( @Const @ByRef Mat img1, @Const @ByRef Mat img2, @Const @ByRef Mat H1to2,
                                         @StdVector KeyPoint keypoints1, @StdVector KeyPoint keypoints2,
                                         @ByRef FloatBuffer repeatability, @ByRef IntBuffer correspCount,
                                         @Ptr FeatureDetector fdetector/*=Ptr<FeatureDetector>()*/ );
@Namespace("cv") public static native void evaluateFeatureDetector( @Const @ByRef Mat img1, @Const @ByRef Mat img2, @Const @ByRef Mat H1to2,
                                         @StdVector KeyPoint keypoints1, @StdVector KeyPoint keypoints2,
                                         @ByRef FloatBuffer repeatability, @ByRef IntBuffer correspCount );
@Namespace("cv") public static native void evaluateFeatureDetector( @Const @ByRef Mat img1, @Const @ByRef Mat img2, @Const @ByRef Mat H1to2,
                                         @StdVector KeyPoint keypoints1, @StdVector KeyPoint keypoints2,
                                         @ByRef float[] repeatability, @ByRef int[] correspCount,
                                         @Ptr FeatureDetector fdetector/*=Ptr<FeatureDetector>()*/ );
@Namespace("cv") public static native void evaluateFeatureDetector( @Const @ByRef Mat img1, @Const @ByRef Mat img2, @Const @ByRef Mat H1to2,
                                         @StdVector KeyPoint keypoints1, @StdVector KeyPoint keypoints2,
                                         @ByRef float[] repeatability, @ByRef int[] correspCount );

@Namespace("cv") public static native void computeRecallPrecisionCurve( @Const @ByRef DMatchVectorVector matches1to2,
                                             @Cast("const std::vector<std::vector<unsigned char> >*") @ByRef ByteVectorVector correctMatches1to2Mask,
                                             @StdVector Point2f recallPrecisionCurve );

@Namespace("cv") public static native float getRecall( @StdVector Point2f recallPrecisionCurve, float l_precision );
@Namespace("cv") public static native int getNearestPoint( @StdVector Point2f recallPrecisionCurve, float l_precision );

@Namespace("cv") public static native void evaluateGenericDescriptorMatcher( @Const @ByRef Mat img1, @Const @ByRef Mat img2, @Const @ByRef Mat H1to2,
                                                  @StdVector KeyPoint keypoints1, @StdVector KeyPoint keypoints2,
                                                  DMatchVectorVector matches1to2, @Cast("std::vector<std::vector<unsigned char> >*") ByteVectorVector correctMatches1to2Mask,
                                                  @StdVector Point2f recallPrecisionCurve,
                                                  @Ptr GenericDescriptorMatcher dmatch/*=Ptr<GenericDescriptorMatcher>()*/ );
@Namespace("cv") public static native void evaluateGenericDescriptorMatcher( @Const @ByRef Mat img1, @Const @ByRef Mat img2, @Const @ByRef Mat H1to2,
                                                  @StdVector KeyPoint keypoints1, @StdVector KeyPoint keypoints2,
                                                  DMatchVectorVector matches1to2, @Cast("std::vector<std::vector<unsigned char> >*") ByteVectorVector correctMatches1to2Mask,
                                                  @StdVector Point2f recallPrecisionCurve );


/****************************************************************************************\
*                                     Bag of visual words                                *
\****************************************************************************************/
/*
 * Abstract base class for training of a 'bag of visual words' vocabulary from a set of descriptors
 */
@Namespace("cv") @NoOffset public static class BOWTrainer extends Pointer {
    static { Loader.load(); }
    /** Empty constructor. */
    public BOWTrainer() { }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public BOWTrainer(Pointer p) { super(p); }


    public native void add( @Const @ByRef Mat descriptors );
    public native @Const @ByRef MatVector getDescriptors();
    public native int descripotorsCount();

    public native void clear();

    /*
     * Train visual words vocabulary, that is cluster training descriptors and
     * compute cluster centers.
     * Returns cluster centers.
     *
     * descriptors      Training descriptors computed on images keypoints.
     */
    public native @ByVal Mat cluster();
    public native @ByVal Mat cluster( @Const @ByRef Mat descriptors );
}

/*
 * This is BOWTrainer using cv::kmeans to get vocabulary.
 */
@Namespace("cv") @NoOffset public static class BOWKMeansTrainer extends BOWTrainer {
    static { Loader.load(); }
    /** Empty constructor. */
    public BOWKMeansTrainer() { }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public BOWKMeansTrainer(Pointer p) { super(p); }

    public BOWKMeansTrainer( int clusterCount, @Const @ByRef TermCriteria termcrit/*=TermCriteria()*/,
                          int attempts/*=3*/, int flags/*=KMEANS_PP_CENTERS*/ ) { allocate(clusterCount, termcrit, attempts, flags); }
    private native void allocate( int clusterCount, @Const @ByRef TermCriteria termcrit/*=TermCriteria()*/,
                          int attempts/*=3*/, int flags/*=KMEANS_PP_CENTERS*/ );
    public BOWKMeansTrainer( int clusterCount ) { allocate(clusterCount); }
    private native void allocate( int clusterCount );

    // Returns trained vocabulary (i.e. cluster centers).
    public native @ByVal Mat cluster();
    public native @ByVal Mat cluster( @Const @ByRef Mat descriptors );
}

/*
 * Class to compute image descriptor using bag of visual words.
 */
@Namespace("cv") @NoOffset public static class BOWImgDescriptorExtractor extends Pointer {
    static { Loader.load(); }
    /** Empty constructor. */
    public BOWImgDescriptorExtractor() { }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public BOWImgDescriptorExtractor(Pointer p) { super(p); }

    public BOWImgDescriptorExtractor( @Ptr DescriptorExtractor dextractor,
                                   @Ptr DescriptorMatcher dmatcher ) { allocate(dextractor, dmatcher); }
    private native void allocate( @Ptr DescriptorExtractor dextractor,
                                   @Ptr DescriptorMatcher dmatcher );

    public native void setVocabulary( @Const @ByRef Mat vocabulary );
    public native @Const @ByRef Mat getVocabulary();
    public native void compute( @Const @ByRef Mat image, @StdVector KeyPoint keypoints, @ByRef Mat imgDescriptor,
                      IntVectorVector pointIdxsOfClusters/*=0*/, Mat descriptors/*=0*/ );
    public native void compute( @Const @ByRef Mat image, @StdVector KeyPoint keypoints, @ByRef Mat imgDescriptor );
    // compute() is not constant because DescriptorMatcher::match is not constant

    public native int descriptorSize();
    public native int descriptorType();
}

 /* namespace cv */

// #endif /* __cplusplus */

// #endif

/* End of file. */


}
