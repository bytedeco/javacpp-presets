// Targeted by JavaCPP version 0.8-SNAPSHOT

package com.googlecode.javacpp;

import com.googlecode.javacpp.helper.opencv_objdetect.*;
import com.googlecode.javacpp.*;
import com.googlecode.javacpp.annotation.*;
import java.nio.*;

import static com.googlecode.javacpp.opencv_core.*;
import static com.googlecode.javacpp.opencv_imgproc.*;
import static com.googlecode.javacpp.opencv_highgui.*;

public class opencv_objdetect extends com.googlecode.javacpp.presets.opencv_objdetect {
    static { Loader.load(); }

@Name("std::deque<CvDataMatrixCode>") public static class CvDataMatrixCodeDeque extends Pointer {
    static { Loader.load(); }
    public CvDataMatrixCodeDeque(Pointer p) { super(p); }
    public CvDataMatrixCodeDeque(CvDataMatrixCode ... array) { this(array.length); put(array); }
    public CvDataMatrixCodeDeque()       { allocate();  }
    public CvDataMatrixCodeDeque(long n) { allocate(n); }
    private native void allocate();
    private native void allocate(@Cast("size_t") long n);

    public native long size();
    public native void resize(@Cast("size_t") long n);

    @Index public native @ByRef CvDataMatrixCode get(@Cast("size_t") long i);
    public native CvDataMatrixCodeDeque put(@Cast("size_t") long i, CvDataMatrixCode value);

    public CvDataMatrixCodeDeque put(CvDataMatrixCode ... array) {
        if (size() != array.length) { resize(array.length); }
        for (int i = 0; i < array.length; i++) {
            put(i, array[i]);
        }
        return this;
    }
}

@Name("std::vector<cv::Ptr<cv::linemod::Modality> >") public static class ModalityVector extends Pointer {
    static { Loader.load(); }
    public ModalityVector(Pointer p) { super(p); }
    public ModalityVector(Modality ... array) { this(array.length); put(array); }
    public ModalityVector()       { allocate();  }
    public ModalityVector(long n) { allocate(n); }
    private native void allocate();
    private native void allocate(@Cast("size_t") long n);

    public native long size();
    public native void resize(@Cast("size_t") long n);

    @Index public native @Ptr Modality get(@Cast("size_t") long i);
    public native ModalityVector put(@Cast("size_t") long i, Modality value);

    public ModalityVector put(Modality ... array) {
        if (size() != array.length) { resize(array.length); }
        for (int i = 0; i < array.length; i++) {
            put(i, array[i]);
        }
        return this;
    }
}

// Parsed from header file /usr/local/include/opencv2/objdetect/objdetect.hpp

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

// #ifndef __OPENCV_OBJDETECT_HPP__
// #define __OPENCV_OBJDETECT_HPP__

// #include "opencv2/core/core.hpp"

// #ifdef __cplusplus
// #include <map>
// #include <deque>
// #endif

/****************************************************************************************\
*                         Haar-like Object Detection functions                           *
\****************************************************************************************/

public static final int CV_HAAR_MAGIC_VAL =    0x42500000;
public static final String CV_TYPE_NAME_HAAR =    "opencv-haar-classifier";

// #define CV_IS_HAAR_CLASSIFIER( haar )
//     ((haar) != NULL &&
//     (((const CvHaarClassifierCascade*)(haar))->flags & CV_MAGIC_MASK)==CV_HAAR_MAGIC_VAL)

public static final int CV_HAAR_FEATURE_MAX =  3;

public static class CvHaarFeature extends Pointer {
    static { Loader.load(); }
    public CvHaarFeature() { allocate(); }
    public CvHaarFeature(int size) { allocateArray(size); }
    public CvHaarFeature(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public CvHaarFeature position(int position) {
        return (CvHaarFeature)super.position(position);
    }

    public native int tilted(); public native CvHaarFeature tilted(int tilted);
        @Name({"rect", ".r"}) public native @ByVal CvRect rect_r(int i); public native CvHaarFeature rect_r(int i, CvRect rect_r);
        @Name({"rect", ".weight"}) public native float rect_weight(int i); public native CvHaarFeature rect_weight(int i, float rect_weight);
}

public static class CvHaarClassifier extends Pointer {
    static { Loader.load(); }
    public CvHaarClassifier() { allocate(); }
    public CvHaarClassifier(int size) { allocateArray(size); }
    public CvHaarClassifier(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public CvHaarClassifier position(int position) {
        return (CvHaarClassifier)super.position(position);
    }

    public native int count(); public native CvHaarClassifier count(int count);
    public native CvHaarFeature haar_feature(); public native CvHaarClassifier haar_feature(CvHaarFeature haar_feature);
    public native FloatPointer threshold(); public native CvHaarClassifier threshold(FloatPointer threshold);
    public native IntPointer left(); public native CvHaarClassifier left(IntPointer left);
    public native IntPointer right(); public native CvHaarClassifier right(IntPointer right);
    public native FloatPointer alpha(); public native CvHaarClassifier alpha(FloatPointer alpha);
}

public static class CvHaarStageClassifier extends Pointer {
    static { Loader.load(); }
    public CvHaarStageClassifier() { allocate(); }
    public CvHaarStageClassifier(int size) { allocateArray(size); }
    public CvHaarStageClassifier(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public CvHaarStageClassifier position(int position) {
        return (CvHaarStageClassifier)super.position(position);
    }

    public native int count(); public native CvHaarStageClassifier count(int count);
    public native float threshold(); public native CvHaarStageClassifier threshold(float threshold);
    public native CvHaarClassifier classifier(); public native CvHaarStageClassifier classifier(CvHaarClassifier classifier);

    public native int next(); public native CvHaarStageClassifier next(int next);
    public native int child(); public native CvHaarStageClassifier child(int child);
    public native int parent(); public native CvHaarStageClassifier parent(int parent);
}

@Opaque public static class CvHidHaarClassifierCascade extends Pointer {
    public CvHidHaarClassifierCascade() { }
    public CvHidHaarClassifierCascade(Pointer p) { super(p); }
}

public static class CvHaarClassifierCascade extends AbstractCvHaarClassifierCascade {
    static { Loader.load(); }
    public CvHaarClassifierCascade() { allocate(); }
    public CvHaarClassifierCascade(int size) { allocateArray(size); }
    public CvHaarClassifierCascade(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public CvHaarClassifierCascade position(int position) {
        return (CvHaarClassifierCascade)super.position(position);
    }

    public native int flags(); public native CvHaarClassifierCascade flags(int flags);
    public native int count(); public native CvHaarClassifierCascade count(int count);
    public native @ByVal CvSize orig_window_size(); public native CvHaarClassifierCascade orig_window_size(CvSize orig_window_size);
    public native @ByVal CvSize real_window_size(); public native CvHaarClassifierCascade real_window_size(CvSize real_window_size);
    public native double scale(); public native CvHaarClassifierCascade scale(double scale);
    public native CvHaarStageClassifier stage_classifier(); public native CvHaarClassifierCascade stage_classifier(CvHaarStageClassifier stage_classifier);
    public native CvHidHaarClassifierCascade hid_cascade(); public native CvHaarClassifierCascade hid_cascade(CvHidHaarClassifierCascade hid_cascade);
}

public static class CvAvgComp extends Pointer {
    static { Loader.load(); }
    public CvAvgComp() { allocate(); }
    public CvAvgComp(int size) { allocateArray(size); }
    public CvAvgComp(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public CvAvgComp position(int position) {
        return (CvAvgComp)super.position(position);
    }

    public native @ByVal CvRect rect(); public native CvAvgComp rect(CvRect rect);
    public native int neighbors(); public native CvAvgComp neighbors(int neighbors);
}

/* Loads haar classifier cascade from a directory.
   It is obsolete: convert your cascade to xml and use cvLoad instead */
public static native CvHaarClassifierCascade cvLoadHaarClassifierCascade(
                    @Cast("const char*") BytePointer directory, @ByVal CvSize orig_window_size);
public static native CvHaarClassifierCascade cvLoadHaarClassifierCascade(
                    String directory, @ByVal CvSize orig_window_size);

public static native void cvReleaseHaarClassifierCascade( @Cast("CvHaarClassifierCascade**") PointerPointer cascade );
public static native void cvReleaseHaarClassifierCascade( @ByPtrPtr CvHaarClassifierCascade cascade );

public static final int CV_HAAR_DO_CANNY_PRUNING =    1;
public static final int CV_HAAR_SCALE_IMAGE =         2;
public static final int CV_HAAR_FIND_BIGGEST_OBJECT = 4;
public static final int CV_HAAR_DO_ROUGH_SEARCH =     8;

//CVAPI(CvSeq*) cvHaarDetectObjectsForROC( const CvArr* image,
//                     CvHaarClassifierCascade* cascade, CvMemStorage* storage,
//                     CvSeq** rejectLevels, CvSeq** levelWeightds,
//                     double scale_factor CV_DEFAULT(1.1),
//                     int min_neighbors CV_DEFAULT(3), int flags CV_DEFAULT(0),
//                     CvSize min_size CV_DEFAULT(cvSize(0,0)), CvSize max_size CV_DEFAULT(cvSize(0,0)),
//                     bool outputRejectLevels = false );


public static native CvSeq cvHaarDetectObjects( @Const CvArr image,
                     CvHaarClassifierCascade cascade, CvMemStorage storage,
                     double scale_factor/*CV_DEFAULT(1.1)*/,
                     int min_neighbors/*CV_DEFAULT(3)*/, int flags/*CV_DEFAULT(0)*/,
                     @ByVal CvSize min_size/*CV_DEFAULT(cvSize(0,0))*/, @ByVal CvSize max_size/*CV_DEFAULT(cvSize(0,0))*/);

/* sets images for haar classifier cascade */
public static native void cvSetImagesForHaarClassifierCascade( CvHaarClassifierCascade cascade,
                                                @Const CvArr sum, @Const CvArr sqsum,
                                                @Const CvArr tilted_sum, double scale );

/* runs the cascade on the specified window */
public static native int cvRunHaarClassifierCascade( @Const CvHaarClassifierCascade cascade,
                                       @ByVal CvPoint pt, int start_stage/*CV_DEFAULT(0)*/);


/****************************************************************************************\
*                         Latent SVM Object Detection functions                          *
\****************************************************************************************/

// DataType: STRUCT position
// Structure describes the position of the filter in the feature pyramid
// l - level in the feature pyramid
// (x, y) - coordinate in level l
public static class CvLSVMFilterPosition extends Pointer {
    static { Loader.load(); }
    public CvLSVMFilterPosition() { allocate(); }
    public CvLSVMFilterPosition(int size) { allocateArray(size); }
    public CvLSVMFilterPosition(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public CvLSVMFilterPosition position(int position) {
        return (CvLSVMFilterPosition)super.position(position);
    }

    public native int x(); public native CvLSVMFilterPosition x(int x);
    public native int y(); public native CvLSVMFilterPosition y(int y);
    public native int l(); public native CvLSVMFilterPosition l(int l);
}

// DataType: STRUCT filterObject
// Description of the filter, which corresponds to the part of the object
// V               - ideal (penalty = 0) position of the partial filter
//                   from the root filter position (V_i in the paper)
// penaltyFunction - vector describes penalty function (d_i in the paper)
//                   pf[0] * x + pf[1] * y + pf[2] * x^2 + pf[3] * y^2
// FILTER DESCRIPTION
//   Rectangular map (sizeX x sizeY),
//   every cell stores feature vector (dimension = p)
// H               - matrix of feature vectors
//                   to set and get feature vectors (i,j)
//                   used formula H[(j * sizeX + i) * p + k], where
//                   k - component of feature vector in cell (i, j)
// END OF FILTER DESCRIPTION
public static class CvLSVMFilterObject extends Pointer {
    static { Loader.load(); }
    public CvLSVMFilterObject() { allocate(); }
    public CvLSVMFilterObject(int size) { allocateArray(size); }
    public CvLSVMFilterObject(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public CvLSVMFilterObject position(int position) {
        return (CvLSVMFilterObject)super.position(position);
    }

    public native @ByVal CvLSVMFilterPosition V(); public native CvLSVMFilterObject V(CvLSVMFilterPosition V);
    public native float fineFunction(int i); public native CvLSVMFilterObject fineFunction(int i, float fineFunction);
    @MemberGetter public native FloatPointer fineFunction();
    public native int sizeX(); public native CvLSVMFilterObject sizeX(int sizeX);
    public native int sizeY(); public native CvLSVMFilterObject sizeY(int sizeY);
    public native int numFeatures(); public native CvLSVMFilterObject numFeatures(int numFeatures);
    public native FloatPointer H(); public native CvLSVMFilterObject H(FloatPointer H);
}

// data type: STRUCT CvLatentSvmDetector
// structure contains internal representation of trained Latent SVM detector
// num_filters          - total number of filters (root plus part) in model
// num_components       - number of components in model
// num_part_filters     - array containing number of part filters for each component
// filters              - root and part filters for all model components
// b                    - biases for all model components
// score_threshold      - confidence level threshold
public static class CvLatentSvmDetector extends Pointer {
    static { Loader.load(); }
    public CvLatentSvmDetector() { allocate(); }
    public CvLatentSvmDetector(int size) { allocateArray(size); }
    public CvLatentSvmDetector(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public CvLatentSvmDetector position(int position) {
        return (CvLatentSvmDetector)super.position(position);
    }

    public native int num_filters(); public native CvLatentSvmDetector num_filters(int num_filters);
    public native int num_components(); public native CvLatentSvmDetector num_components(int num_components);
    public native IntPointer num_part_filters(); public native CvLatentSvmDetector num_part_filters(IntPointer num_part_filters);
    public native CvLSVMFilterObject filters(int i); public native CvLatentSvmDetector filters(int i, CvLSVMFilterObject filters);
    @MemberGetter public native @Cast("CvLSVMFilterObject**") PointerPointer filters();
    public native FloatPointer b(); public native CvLatentSvmDetector b(FloatPointer b);
    public native float score_threshold(); public native CvLatentSvmDetector score_threshold(float score_threshold);
}

// data type: STRUCT CvObjectDetection
// structure contains the bounding box and confidence level for detected object
// rect                 - bounding box for a detected object
// score                - confidence level
public static class CvObjectDetection extends Pointer {
    static { Loader.load(); }
    public CvObjectDetection() { allocate(); }
    public CvObjectDetection(int size) { allocateArray(size); }
    public CvObjectDetection(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public CvObjectDetection position(int position) {
        return (CvObjectDetection)super.position(position);
    }

    public native @ByVal CvRect rect(); public native CvObjectDetection rect(CvRect rect);
    public native float score(); public native CvObjectDetection score(float score);
}

//////////////// Object Detection using Latent SVM //////////////


/*
// load trained detector from a file
//
// API
// CvLatentSvmDetector* cvLoadLatentSvmDetector(const char* filename);
// INPUT
// filename             - path to the file containing the parameters of
                        - trained Latent SVM detector
// OUTPUT
// trained Latent SVM detector in internal representation
*/
public static native CvLatentSvmDetector cvLoadLatentSvmDetector(@Cast("const char*") BytePointer filename);
public static native CvLatentSvmDetector cvLoadLatentSvmDetector(String filename);

/*
// release memory allocated for CvLatentSvmDetector structure
//
// API
// void cvReleaseLatentSvmDetector(CvLatentSvmDetector** detector);
// INPUT
// detector             - CvLatentSvmDetector structure to be released
// OUTPUT
*/
public static native void cvReleaseLatentSvmDetector(@Cast("CvLatentSvmDetector**") PointerPointer detector);
public static native void cvReleaseLatentSvmDetector(@ByPtrPtr CvLatentSvmDetector detector);

/*
// find rectangular regions in the given image that are likely
// to contain objects and corresponding confidence levels
//
// API
// CvSeq* cvLatentSvmDetectObjects(const IplImage* image,
//                                  CvLatentSvmDetector* detector,
//                                  CvMemStorage* storage,
//                                  float overlap_threshold = 0.5f,
//                                  int numThreads = -1);
// INPUT
// image                - image to detect objects in
// detector             - Latent SVM detector in internal representation
// storage              - memory storage to store the resultant sequence
//                          of the object candidate rectangles
// overlap_threshold    - threshold for the non-maximum suppression algorithm
                           = 0.5f [here will be the reference to original paper]
// OUTPUT
// sequence of detected objects (bounding boxes and confidence levels stored in CvObjectDetection structures)
*/
public static native CvSeq cvLatentSvmDetectObjects(IplImage image,
                                CvLatentSvmDetector detector,
                                CvMemStorage storage,
                                float overlap_threshold/*CV_DEFAULT(0.5f)*/,
                                int numThreads/*CV_DEFAULT(-1)*/);

// #ifdef __cplusplus

public static native CvSeq cvHaarDetectObjectsForROC( @Const CvArr image,
                     CvHaarClassifierCascade cascade, CvMemStorage storage,
                     @StdVector IntPointer rejectLevels, @StdVector DoublePointer levelWeightds,
                     double scale_factor/*CV_DEFAULT(1.1)*/,
                     int min_neighbors/*CV_DEFAULT(3)*/, int flags/*CV_DEFAULT(0)*/,
                     @ByVal CvSize min_size/*CV_DEFAULT(cvSize(0,0))*/, @ByVal CvSize max_size/*CV_DEFAULT(cvSize(0,0))*/,
                     @Cast("bool") boolean outputRejectLevels/*=false*/ );
public static native CvSeq cvHaarDetectObjectsForROC( @Const CvArr image,
                     CvHaarClassifierCascade cascade, CvMemStorage storage,
                     @StdVector IntBuffer rejectLevels, @StdVector DoubleBuffer levelWeightds,
                     double scale_factor/*CV_DEFAULT(1.1)*/,
                     int min_neighbors/*CV_DEFAULT(3)*/, int flags/*CV_DEFAULT(0)*/,
                     @ByVal CvSize min_size/*CV_DEFAULT(cvSize(0,0))*/, @ByVal CvSize max_size/*CV_DEFAULT(cvSize(0,0))*/,
                     @Cast("bool") boolean outputRejectLevels/*=false*/ );
public static native CvSeq cvHaarDetectObjectsForROC( @Const CvArr image,
                     CvHaarClassifierCascade cascade, CvMemStorage storage,
                     @StdVector int[] rejectLevels, @StdVector double[] levelWeightds,
                     double scale_factor/*CV_DEFAULT(1.1)*/,
                     int min_neighbors/*CV_DEFAULT(3)*/, int flags/*CV_DEFAULT(0)*/,
                     @ByVal CvSize min_size/*CV_DEFAULT(cvSize(0,0))*/, @ByVal CvSize max_size/*CV_DEFAULT(cvSize(0,0))*/,
                     @Cast("bool") boolean outputRejectLevels/*=false*/ );

///////////////////////////// Object Detection ////////////////////////////

/*
 * This is a class wrapping up the structure CvLatentSvmDetector and functions working with it.
 * The class goals are:
 * 1) provide c++ interface;
 * 2) make it possible to load and detect more than one class (model) unlike CvLatentSvmDetector.
 */
@Namespace("cv") @NoOffset public static class LatentSvmDetector extends Pointer {
    static { Loader.load(); }
    public LatentSvmDetector(Pointer p) { super(p); }
    public LatentSvmDetector(int size) { allocateArray(size); }
    private native void allocateArray(int size);
    @Override public LatentSvmDetector position(int position) {
        return (LatentSvmDetector)super.position(position);
    }

    @NoOffset public static class ObjectDetection extends Pointer {
        static { Loader.load(); }
        public ObjectDetection(Pointer p) { super(p); }
        public ObjectDetection(int size) { allocateArray(size); }
        private native void allocateArray(int size);
        @Override public ObjectDetection position(int position) {
            return (ObjectDetection)super.position(position);
        }
    
        public ObjectDetection() { allocate(); }
        private native void allocate();
        public ObjectDetection( @Const @ByRef Rect rect, float score, int classID/*=-1*/ ) { allocate(rect, score, classID); }
        private native void allocate( @Const @ByRef Rect rect, float score, int classID/*=-1*/ );
        public native @ByVal Rect rect(); public native ObjectDetection rect(Rect rect);
        public native float score(); public native ObjectDetection score(float score);
        public native int classID(); public native ObjectDetection classID(int classID);
    }

    public LatentSvmDetector() { allocate(); }
    private native void allocate();
    public LatentSvmDetector( @Const @ByRef StringVector filenames, @Const @ByRef StringVector classNames/*=vector<string>()*/ ) { allocate(filenames, classNames); }
    private native void allocate( @Const @ByRef StringVector filenames, @Const @ByRef StringVector classNames/*=vector<string>()*/ );

    public native void clear();
    public native @Cast("bool") boolean empty();
    public native @Cast("bool") boolean load( @Const @ByRef StringVector filenames, @Const @ByRef StringVector classNames/*=vector<string>()*/ );

    public native void detect( @Const @ByRef Mat image,
                             @StdVector ObjectDetection objectDetections,
                             float overlapThreshold/*=0.5f*/,
                             int numThreads/*=-1*/ );

    public native @Const @ByRef StringVector getClassNames();
    public native @Cast("size_t") long getClassCount();
}

// class for grouping object candidates, detected by Cascade Classifier, HOG etc.
// instance of the class is to be passed to cv::partition (see cxoperations.hpp)
@Namespace("cv") @NoOffset public static class SimilarRects extends Pointer {
    static { Loader.load(); }
    public SimilarRects() { }
    public SimilarRects(Pointer p) { super(p); }

    public SimilarRects(double _eps) { allocate(_eps); }
    private native void allocate(double _eps);
    public native @Cast("bool") @Name("operator()") boolean apply(@Const @ByRef Rect r1, @Const @ByRef Rect r2);
    public native double eps(); public native SimilarRects eps(double eps);
}

@Namespace("cv") public static native void groupRectangles(@StdVector Rect rectList, int groupThreshold, double eps/*=0.2*/);
@Namespace("cv") public static native void groupRectangles(@StdVector Rect rectList, @StdVector IntPointer weights, int groupThreshold, double eps/*=0.2*/);
@Namespace("cv") public static native void groupRectangles(@StdVector Rect rectList, @StdVector IntBuffer weights, int groupThreshold, double eps/*=0.2*/);
@Namespace("cv") public static native void groupRectangles(@StdVector Rect rectList, @StdVector int[] weights, int groupThreshold, double eps/*=0.2*/);
@Namespace("cv") public static native void groupRectangles( @StdVector Rect rectList, int groupThreshold, double eps, @StdVector IntPointer weights, @StdVector DoublePointer levelWeights );
@Namespace("cv") public static native void groupRectangles( @StdVector Rect rectList, int groupThreshold, double eps, @StdVector IntBuffer weights, @StdVector DoubleBuffer levelWeights );
@Namespace("cv") public static native void groupRectangles( @StdVector Rect rectList, int groupThreshold, double eps, @StdVector int[] weights, @StdVector double[] levelWeights );
@Namespace("cv") public static native void groupRectangles(@StdVector Rect rectList, @StdVector IntPointer rejectLevels,
                                @StdVector DoublePointer levelWeights, int groupThreshold, double eps/*=0.2*/);
@Namespace("cv") public static native void groupRectangles(@StdVector Rect rectList, @StdVector IntBuffer rejectLevels,
                                @StdVector DoubleBuffer levelWeights, int groupThreshold, double eps/*=0.2*/);
@Namespace("cv") public static native void groupRectangles(@StdVector Rect rectList, @StdVector int[] rejectLevels,
                                @StdVector double[] levelWeights, int groupThreshold, double eps/*=0.2*/);
@Namespace("cv") public static native void groupRectangles_meanshift(@StdVector Rect rectList, @StdVector DoublePointer foundWeights, @StdVector DoublePointer foundScales,
                                          double detectThreshold/*=0.0*/, @ByVal Size winDetSize/*=Size(64, 128)*/);
@Namespace("cv") public static native void groupRectangles_meanshift(@StdVector Rect rectList, @StdVector DoubleBuffer foundWeights, @StdVector DoubleBuffer foundScales,
                                          double detectThreshold/*=0.0*/, @ByVal Size winDetSize/*=Size(64, 128)*/);
@Namespace("cv") public static native void groupRectangles_meanshift(@StdVector Rect rectList, @StdVector double[] foundWeights, @StdVector double[] foundScales,
                                          double detectThreshold/*=0.0*/, @ByVal Size winDetSize/*=Size(64, 128)*/);


@Namespace("cv") public static class FeatureEvaluator extends Pointer {
    static { Loader.load(); }
    public FeatureEvaluator() { allocate(); }
    public FeatureEvaluator(int size) { allocateArray(size); }
    public FeatureEvaluator(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public FeatureEvaluator position(int position) {
        return (FeatureEvaluator)super.position(position);
    }

    /** enum cv::FeatureEvaluator:: */
    public static final int HAAR = 0, LBP = 1, HOG = 2;

    public native @Cast("bool") boolean read(@Const @ByRef FileNode node);
    public native @Ptr FeatureEvaluator clone();
    public native int getFeatureType();

    public native @Cast("bool") boolean setImage(@Const @ByRef Mat img, @ByVal Size origWinSize);
    public native @Cast("bool") boolean setWindow(@ByVal Point p);

    public native double calcOrd(int featureIdx);
    public native int calcCat(int featureIdx);

    public native @Ptr FeatureEvaluator create(int type);
}



/** enum cv:: */
public static final int
    CASCADE_DO_CANNY_PRUNING= 1,
    CASCADE_SCALE_IMAGE= 2,
    CASCADE_FIND_BIGGEST_OBJECT= 4,
    CASCADE_DO_ROUGH_SEARCH= 8;

@Namespace("cv") @NoOffset public static class CascadeClassifier extends Pointer {
    static { Loader.load(); }
    public CascadeClassifier(Pointer p) { super(p); }
    public CascadeClassifier(int size) { allocateArray(size); }
    private native void allocateArray(int size);
    @Override public CascadeClassifier position(int position) {
        return (CascadeClassifier)super.position(position);
    }

    public CascadeClassifier() { allocate(); }
    private native void allocate();
    public CascadeClassifier( @StdString BytePointer filename ) { allocate(filename); }
    private native void allocate( @StdString BytePointer filename );
    public CascadeClassifier( @StdString String filename ) { allocate(filename); }
    private native void allocate( @StdString String filename );

    public native @Cast("bool") boolean empty();
    public native @Cast("bool") boolean load( @StdString BytePointer filename );
    public native @Cast("bool") boolean load( @StdString String filename );
    public native @Cast("bool") boolean read( @Const @ByRef FileNode node );
    public native void detectMultiScale( @Const @ByRef Mat image,
                                       @StdVector Rect objects,
                                       double scaleFactor/*=1.1*/,
                                       int minNeighbors/*=3*/, int flags/*=0*/,
                                       @ByVal Size minSize/*=Size()*/,
                                       @ByVal Size maxSize/*=Size()*/ );

    public native void detectMultiScale( @Const @ByRef Mat image,
                                       @StdVector Rect objects,
                                       @StdVector IntPointer rejectLevels,
                                       @StdVector DoublePointer levelWeights,
                                       double scaleFactor/*=1.1*/,
                                       int minNeighbors/*=3*/, int flags/*=0*/,
                                       @ByVal Size minSize/*=Size()*/,
                                       @ByVal Size maxSize/*=Size()*/,
                                       @Cast("bool") boolean outputRejectLevels/*=false*/ );
    public native void detectMultiScale( @Const @ByRef Mat image,
                                       @StdVector Rect objects,
                                       @StdVector IntBuffer rejectLevels,
                                       @StdVector DoubleBuffer levelWeights,
                                       double scaleFactor/*=1.1*/,
                                       int minNeighbors/*=3*/, int flags/*=0*/,
                                       @ByVal Size minSize/*=Size()*/,
                                       @ByVal Size maxSize/*=Size()*/,
                                       @Cast("bool") boolean outputRejectLevels/*=false*/ );
    public native void detectMultiScale( @Const @ByRef Mat image,
                                       @StdVector Rect objects,
                                       @StdVector int[] rejectLevels,
                                       @StdVector double[] levelWeights,
                                       double scaleFactor/*=1.1*/,
                                       int minNeighbors/*=3*/, int flags/*=0*/,
                                       @ByVal Size minSize/*=Size()*/,
                                       @ByVal Size maxSize/*=Size()*/,
                                       @Cast("bool") boolean outputRejectLevels/*=false*/ );


    public native @Cast("bool") boolean isOldFormatCascade();
    public native @ByVal Size getOriginalWindowSize();
    public native int getFeatureType();
    public native @Cast("bool") boolean setImage( @Const @ByRef Mat arg0 );
    public static class MaskGenerator extends Pointer {
        static { Loader.load(); }
        public MaskGenerator() { }
        public MaskGenerator(Pointer p) { super(p); }
    
        public native @ByVal Mat generateMask(@Const @ByRef Mat src);
        public native void initializeMask(@Const @ByRef Mat arg0);
    }
    public native void setMaskGenerator(@Ptr MaskGenerator maskGenerator);
    public native @Ptr MaskGenerator getMaskGenerator();

    public native void setFaceDetectionMaskGenerator();
}


//////////////// HOG (Histogram-of-Oriented-Gradients) Descriptor and Object Detector //////////////

// struct for detection region of interest (ROI)
@Namespace("cv") public static class DetectionROI extends Pointer {
    static { Loader.load(); }
    public DetectionROI() { allocate(); }
    public DetectionROI(int size) { allocateArray(size); }
    public DetectionROI(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public DetectionROI position(int position) {
        return (DetectionROI)super.position(position);
    }

   // scale(size) of the bounding box
   public native double scale(); public native DetectionROI scale(double scale);
   // set of requrested locations to be evaluated
   public native @StdVector Point locations(); public native DetectionROI locations(Point locations);
   // vector that will contain confidence values for each location
   public native @StdVector DoublePointer confidences(); public native DetectionROI confidences(DoublePointer confidences);
}

@Namespace("cv") @NoOffset public static class HOGDescriptor extends Pointer {
    static { Loader.load(); }
    public HOGDescriptor(Pointer p) { super(p); }
    public HOGDescriptor(int size) { allocateArray(size); }
    private native void allocateArray(int size);
    @Override public HOGDescriptor position(int position) {
        return (HOGDescriptor)super.position(position);
    }

    /** enum cv::HOGDescriptor:: */
    public static final int L2Hys= 0;
    /** enum cv::HOGDescriptor:: */
    public static final int DEFAULT_NLEVELS= 64;

    public HOGDescriptor() { allocate(); }
    private native void allocate();

    public HOGDescriptor(@ByVal Size _winSize, @ByVal Size _blockSize, @ByVal Size _blockStride,
                      @ByVal Size _cellSize, int _nbins, int _derivAperture/*=1*/, double _winSigma/*=-1*/,
                      int _histogramNormType/*=HOGDescriptor::L2Hys*/,
                      double _L2HysThreshold/*=0.2*/, @Cast("bool") boolean _gammaCorrection/*=false*/,
                      int _nlevels/*=HOGDescriptor::DEFAULT_NLEVELS*/) { allocate(_winSize, _blockSize, _blockStride, _cellSize, _nbins, _derivAperture, _winSigma, _histogramNormType, _L2HysThreshold, _gammaCorrection, _nlevels); }
    private native void allocate(@ByVal Size _winSize, @ByVal Size _blockSize, @ByVal Size _blockStride,
                      @ByVal Size _cellSize, int _nbins, int _derivAperture/*=1*/, double _winSigma/*=-1*/,
                      int _histogramNormType/*=HOGDescriptor::L2Hys*/,
                      double _L2HysThreshold/*=0.2*/, @Cast("bool") boolean _gammaCorrection/*=false*/,
                      int _nlevels/*=HOGDescriptor::DEFAULT_NLEVELS*/);

    public HOGDescriptor(@StdString BytePointer filename) { allocate(filename); }
    private native void allocate(@StdString BytePointer filename);
    public HOGDescriptor(@StdString String filename) { allocate(filename); }
    private native void allocate(@StdString String filename);

    public HOGDescriptor(@Const @ByRef HOGDescriptor d) { allocate(d); }
    private native void allocate(@Const @ByRef HOGDescriptor d);

    public native @Cast("size_t") long getDescriptorSize();
    public native @Cast("bool") boolean checkDetectorSize();
    public native double getWinSigma();

    public native void setSVMDetector(@ByVal Mat _svmdetector);

    public native @Cast("bool") boolean read(@ByRef FileNode fn);
    public native void write(@ByRef FileStorage fs, @StdString BytePointer objname);
    public native void write(@ByRef FileStorage fs, @StdString String objname);

    public native @Cast("bool") boolean load(@StdString BytePointer filename, @StdString BytePointer objname/*=String()*/);
    public native @Cast("bool") boolean load(@StdString String filename, @StdString String objname/*=String()*/);
    public native void save(@StdString BytePointer filename, @StdString BytePointer objname/*=String()*/);
    public native void save(@StdString String filename, @StdString String objname/*=String()*/);
    public native void copyTo(@ByRef HOGDescriptor c);

    public native void compute(@Const @ByRef Mat img,
                             @StdVector FloatPointer descriptors,
                             @ByVal Size winStride/*=Size()*/, @ByVal Size padding/*=Size()*/,
                             @StdVector Point locations/*=vector<Point>()*/);
    public native void compute(@Const @ByRef Mat img,
                             @StdVector FloatBuffer descriptors,
                             @ByVal Size winStride/*=Size()*/, @ByVal Size padding/*=Size()*/,
                             @StdVector Point locations/*=vector<Point>()*/);
    public native void compute(@Const @ByRef Mat img,
                             @StdVector float[] descriptors,
                             @ByVal Size winStride/*=Size()*/, @ByVal Size padding/*=Size()*/,
                             @StdVector Point locations/*=vector<Point>()*/);
    //with found weights output
    public native void detect(@Const @ByRef Mat img, @StdVector Point foundLocations,
                            @StdVector DoublePointer weights,
                            double hitThreshold/*=0*/, @ByVal Size winStride/*=Size()*/,
                            @ByVal Size padding/*=Size()*/,
                            @StdVector Point searchLocations/*=vector<Point>()*/);
    public native void detect(@Const @ByRef Mat img, @StdVector Point foundLocations,
                            @StdVector DoubleBuffer weights,
                            double hitThreshold/*=0*/, @ByVal Size winStride/*=Size()*/,
                            @ByVal Size padding/*=Size()*/,
                            @StdVector Point searchLocations/*=vector<Point>()*/);
    public native void detect(@Const @ByRef Mat img, @StdVector Point foundLocations,
                            @StdVector double[] weights,
                            double hitThreshold/*=0*/, @ByVal Size winStride/*=Size()*/,
                            @ByVal Size padding/*=Size()*/,
                            @StdVector Point searchLocations/*=vector<Point>()*/);
    //without found weights output
    public native void detect(@Const @ByRef Mat img, @StdVector Point foundLocations,
                            double hitThreshold/*=0*/, @ByVal Size winStride/*=Size()*/,
                            @ByVal Size padding/*=Size()*/,
                            @StdVector Point searchLocations/*=vector<Point>()*/);
    //with result weights output
    public native void detectMultiScale(@Const @ByRef Mat img, @StdVector Rect foundLocations,
                                      @StdVector DoublePointer foundWeights, double hitThreshold/*=0*/,
                                      @ByVal Size winStride/*=Size()*/, @ByVal Size padding/*=Size()*/, double scale/*=1.05*/,
                                      double finalThreshold/*=2.0*/,@Cast("bool") boolean useMeanshiftGrouping/*=false*/);
    public native void detectMultiScale(@Const @ByRef Mat img, @StdVector Rect foundLocations,
                                      @StdVector DoubleBuffer foundWeights, double hitThreshold/*=0*/,
                                      @ByVal Size winStride/*=Size()*/, @ByVal Size padding/*=Size()*/, double scale/*=1.05*/,
                                      double finalThreshold/*=2.0*/,@Cast("bool") boolean useMeanshiftGrouping/*=false*/);
    public native void detectMultiScale(@Const @ByRef Mat img, @StdVector Rect foundLocations,
                                      @StdVector double[] foundWeights, double hitThreshold/*=0*/,
                                      @ByVal Size winStride/*=Size()*/, @ByVal Size padding/*=Size()*/, double scale/*=1.05*/,
                                      double finalThreshold/*=2.0*/,@Cast("bool") boolean useMeanshiftGrouping/*=false*/);
    //without found weights output
    public native void detectMultiScale(@Const @ByRef Mat img, @StdVector Rect foundLocations,
                                      double hitThreshold/*=0*/, @ByVal Size winStride/*=Size()*/,
                                      @ByVal Size padding/*=Size()*/, double scale/*=1.05*/,
                                      double finalThreshold/*=2.0*/, @Cast("bool") boolean useMeanshiftGrouping/*=false*/);

    public native void computeGradient(@Const @ByRef Mat img, @ByRef Mat grad, @ByRef Mat angleOfs,
                                     @ByVal Size paddingTL/*=Size()*/, @ByVal Size paddingBR/*=Size()*/);

    public native @StdVector FloatPointer getDefaultPeopleDetector();
    public native @StdVector FloatPointer getDaimlerPeopleDetector();

    public native @ByVal Size winSize(); public native HOGDescriptor winSize(Size winSize);
    public native @ByVal Size blockSize(); public native HOGDescriptor blockSize(Size blockSize);
    public native @ByVal Size blockStride(); public native HOGDescriptor blockStride(Size blockStride);
    public native @ByVal Size cellSize(); public native HOGDescriptor cellSize(Size cellSize);
    public native int nbins(); public native HOGDescriptor nbins(int nbins);
    public native int derivAperture(); public native HOGDescriptor derivAperture(int derivAperture);
    public native double winSigma(); public native HOGDescriptor winSigma(double winSigma);
    public native int histogramNormType(); public native HOGDescriptor histogramNormType(int histogramNormType);
    public native double L2HysThreshold(); public native HOGDescriptor L2HysThreshold(double L2HysThreshold);
    public native @Cast("bool") boolean gammaCorrection(); public native HOGDescriptor gammaCorrection(boolean gammaCorrection);
    public native @StdVector FloatPointer svmDetector(); public native HOGDescriptor svmDetector(FloatPointer svmDetector);
    public native int nlevels(); public native HOGDescriptor nlevels(int nlevels);


   // evaluate specified ROI and return confidence value for each location
   public native void detectROI(@Const @ByRef Mat img, @StdVector Point locations,
                                      @StdVector Point foundLocations, @StdVector DoublePointer confidences,
                                      double hitThreshold/*=0*/, @ByVal Size winStride/*=Size()*/,
                                      @ByVal Size padding/*=Size()*/);
   public native void detectROI(@Const @ByRef Mat img, @StdVector Point locations,
                                      @StdVector Point foundLocations, @StdVector DoubleBuffer confidences,
                                      double hitThreshold/*=0*/, @ByVal Size winStride/*=Size()*/,
                                      @ByVal Size padding/*=Size()*/);
   public native void detectROI(@Const @ByRef Mat img, @StdVector Point locations,
                                      @StdVector Point foundLocations, @StdVector double[] confidences,
                                      double hitThreshold/*=0*/, @ByVal Size winStride/*=Size()*/,
                                      @ByVal Size padding/*=Size()*/);

   // evaluate specified ROI and return confidence value for each location in multiple scales
   public native void detectMultiScaleROI(@Const @ByRef Mat img,
                                                          @StdVector Rect foundLocations,
                                                          @StdVector DetectionROI locations,
                                                          double hitThreshold/*=0*/,
                                                          int groupThreshold/*=0*/);

   // read/parse Dalal's alt model file
   public native void readALTModel(@StdString BytePointer modelfile);
   public native void readALTModel(@StdString String modelfile);
   public native void groupRectangles(@StdVector Rect rectList, @StdVector DoublePointer weights, int groupThreshold, double eps);
   public native void groupRectangles(@StdVector Rect rectList, @StdVector DoubleBuffer weights, int groupThreshold, double eps);
   public native void groupRectangles(@StdVector Rect rectList, @StdVector double[] weights, int groupThreshold, double eps);
}


@Namespace("cv") public static native void findDataMatrix(@ByVal Mat image,
                                 @ByRef StringVector codes,
                                 @ByVal Mat corners/*=noArray()*/,
                                 @ByVal MatVector dmtx/*=noArray()*/);
@Namespace("cv") public static native void drawDataMatrixCodes(@ByVal Mat image,
                                      @Const @ByRef StringVector codes,
                                      @ByVal Mat corners);


/****************************************************************************************\
*                                Datamatrix                                              *
\****************************************************************************************/

public static class CvDataMatrixCode extends Pointer {
    static { Loader.load(); }
    public CvDataMatrixCode() { allocate(); }
    public CvDataMatrixCode(int size) { allocateArray(size); }
    public CvDataMatrixCode(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public CvDataMatrixCode position(int position) {
        return (CvDataMatrixCode)super.position(position);
    }

  public native @Cast("char") byte msg(int i); public native CvDataMatrixCode msg(int i, byte msg);
  @MemberGetter public native @Cast("char*") BytePointer msg();
  public native CvMat original(); public native CvDataMatrixCode original(CvMat original);
  public native CvMat corners(); public native CvDataMatrixCode corners(CvMat corners);
}

public static native @ByVal CvDataMatrixCodeDeque cvFindDataMatrix(CvMat im);

/****************************************************************************************\
*                                 LINE-MOD                                               *
\****************************************************************************************/

/** @todo Convert doxy comments to rst

/**
 * \brief Discriminant feature described by its location and label.
 */
@Namespace("cv::linemod") @NoOffset public static class Feature extends Pointer {
    static { Loader.load(); }
    public Feature(Pointer p) { super(p); }
    public Feature(int size) { allocateArray(size); }
    private native void allocateArray(int size);
    @Override public Feature position(int position) {
        return (Feature)super.position(position);
    }

  /** x offset */
  public native int x(); public native Feature x(int x);
  /** y offset */
  public native int y(); public native Feature y(int y);
  /** Quantization */
  public native int label(); public native Feature label(int label);

  public Feature() { allocate(); }
  private native void allocate();
  public Feature(int x, int y, int label) { allocate(x, y, label); }
  private native void allocate(int x, int y, int label);

  public native void read(@Const @ByRef FileNode fn);
  public native void write(@ByRef FileStorage fs);
}

   

@Namespace("cv::linemod") public static class Template extends Pointer {
    static { Loader.load(); }
    public Template() { allocate(); }
    public Template(int size) { allocateArray(size); }
    public Template(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public Template position(int position) {
        return (Template)super.position(position);
    }

  public native int width(); public native Template width(int width);
  public native int height(); public native Template height(int height);
  public native int pyramid_level(); public native Template pyramid_level(int pyramid_level);
  public native @StdVector Feature features(); public native Template features(Feature features);

  public native void read(@Const @ByRef FileNode fn);
  public native void write(@ByRef FileStorage fs);
}

/**
 * \brief Represents a modality operating over an image pyramid.
 */
@Namespace("cv::linemod") public static class QuantizedPyramid extends Pointer {
    static { Loader.load(); }
    public QuantizedPyramid() { }
    public QuantizedPyramid(Pointer p) { super(p); }

  // Virtual destructor

  /**
   * \brief Compute quantized image at current pyramid level for online detection.
   *
   * \param[out] dst The destination 8-bit image. For each pixel at most one bit is set,
   *                 representing its classification.
   */
  public native void quantize(@ByRef Mat dst);

  /**
   * \brief Extract most discriminant features at current pyramid level to form a new template.
   *
   * \param[out] templ The new template.
   */
  public native @Cast("bool") boolean extractTemplate(@ByRef Template templ);

  /**
   * \brief Go to the next pyramid level.
   *
   * \todo Allow pyramid scale factor other than 2
   */
  public native void pyrDown();
}

  

/**
 * \brief Interface for modalities that plug into the LINE template matching representation.
 *
 * \todo Max response, to allow optimization of summing (255/MAX) features as uint8
 */
@Namespace("cv::linemod") public static class Modality extends Pointer {
    static { Loader.load(); }
    public Modality() { }
    public Modality(Pointer p) { super(p); }

  // Virtual destructor

  /**
   * \brief Form a quantized image pyramid from a source image.
   *
   * \param[in] src  The source image. Type depends on the modality.
   * \param[in] mask Optional mask. If not empty, unmasked pixels are set to zero
   *                 in quantized image and cannot be extracted as features.
   */
  public native @Ptr QuantizedPyramid process(@Const @ByRef Mat src,
                      @Const @ByRef Mat mask/*=Mat()*/);

  public native @StdString BytePointer name();

  public native void read(@Const @ByRef FileNode fn);
  public native void write(@ByRef FileStorage fs);

  /**
   * \brief Create modality by name.
   *
   * The following modality types are supported:
   * - "ColorGradient"
   * - "DepthNormal"
   */
  public static native @Ptr Modality create(@StdString BytePointer modality_type);
  public static native @Ptr Modality create(@StdString String modality_type);

  /**
   * \brief Load a modality from file.
   */
  public static native @Ptr Modality create(@Const @ByRef FileNode fn);
}

/**
 * \brief Modality that computes quantized gradient orientations from a color image.
 */
@Namespace("cv::linemod") @NoOffset public static class ColorGradient extends Modality {
    static { Loader.load(); }
    public ColorGradient(Pointer p) { super(p); }
    public ColorGradient(int size) { allocateArray(size); }
    private native void allocateArray(int size);
    @Override public ColorGradient position(int position) {
        return (ColorGradient)super.position(position);
    }

  /**
   * \brief Default constructor. Uses reasonable default parameter values.
   */
  public ColorGradient() { allocate(); }
  private native void allocate();

  /**
   * \brief Constructor.
   *
   * \param weak_threshold   When quantizing, discard gradients with magnitude less than this.
   * \param num_features     How many features a template must contain.
   * \param strong_threshold Consider as candidate features only gradients whose norms are
   *                         larger than this.
   */
  public ColorGradient(float weak_threshold, @Cast("size_t") long num_features, float strong_threshold) { allocate(weak_threshold, num_features, strong_threshold); }
  private native void allocate(float weak_threshold, @Cast("size_t") long num_features, float strong_threshold);

  public native @StdString BytePointer name();

  public native void read(@Const @ByRef FileNode fn);
  public native void write(@ByRef FileStorage fs);

  public native float weak_threshold(); public native ColorGradient weak_threshold(float weak_threshold);
  public native @Cast("size_t") long num_features(); public native ColorGradient num_features(long num_features);
  public native float strong_threshold(); public native ColorGradient strong_threshold(float strong_threshold);
}

/**
 * \brief Modality that computes quantized surface normals from a dense depth map.
 */
@Namespace("cv::linemod") @NoOffset public static class DepthNormal extends Modality {
    static { Loader.load(); }
    public DepthNormal(Pointer p) { super(p); }
    public DepthNormal(int size) { allocateArray(size); }
    private native void allocateArray(int size);
    @Override public DepthNormal position(int position) {
        return (DepthNormal)super.position(position);
    }

  /**
   * \brief Default constructor. Uses reasonable default parameter values.
   */
  public DepthNormal() { allocate(); }
  private native void allocate();

  /**
   * \brief Constructor.
   *
   * \param distance_threshold   Ignore pixels beyond this distance.
   * \param difference_threshold When computing normals, ignore contributions of pixels whose
   *                             depth difference with the central pixel is above this threshold.
   * \param num_features         How many features a template must contain.
   * \param extract_threshold    Consider as candidate feature only if there are no differing
   *                             orientations within a distance of extract_threshold.
   */
  public DepthNormal(int distance_threshold, int difference_threshold, @Cast("size_t") long num_features,
                int extract_threshold) { allocate(distance_threshold, difference_threshold, num_features, extract_threshold); }
  private native void allocate(int distance_threshold, int difference_threshold, @Cast("size_t") long num_features,
                int extract_threshold);

  public native @StdString BytePointer name();

  public native void read(@Const @ByRef FileNode fn);
  public native void write(@ByRef FileStorage fs);

  public native int distance_threshold(); public native DepthNormal distance_threshold(int distance_threshold);
  public native int difference_threshold(); public native DepthNormal difference_threshold(int difference_threshold);
  public native @Cast("size_t") long num_features(); public native DepthNormal num_features(long num_features);
  public native int extract_threshold(); public native DepthNormal extract_threshold(int extract_threshold);
}

/**
 * \brief Debug function to colormap a quantized image for viewing.
 */
@Namespace("cv::linemod") public static native void colormap(@Const @ByRef Mat quantized, @ByRef Mat dst);

/**
 * \brief Represents a successful template match.
 */
@Namespace("cv::linemod") @NoOffset public static class Match extends Pointer {
    static { Loader.load(); }
    public Match(Pointer p) { super(p); }
    public Match(int size) { allocateArray(size); }
    private native void allocateArray(int size);
    @Override public Match position(int position) {
        return (Match)super.position(position);
    }

  public Match() { allocate(); }
  private native void allocate();

  public Match(int x, int y, float similarity, @StdString BytePointer class_id, int template_id) { allocate(x, y, similarity, class_id, template_id); }
  private native void allocate(int x, int y, float similarity, @StdString BytePointer class_id, int template_id);
  public Match(int x, int y, float similarity, @StdString String class_id, int template_id) { allocate(x, y, similarity, class_id, template_id); }
  private native void allocate(int x, int y, float similarity, @StdString String class_id, int template_id);

  /** Sort matches with high similarity to the front */
  public native @Cast("bool") @Name("operator<") boolean lessThan(@Const @ByRef Match rhs);

  public native @Cast("bool") @Name("operator==") boolean equals(@Const @ByRef Match rhs);

  public native int x(); public native Match x(int x);
  public native int y(); public native Match y(int y);
  public native float similarity(); public native Match similarity(float similarity);
  public native @StdString BytePointer class_id(); public native Match class_id(BytePointer class_id);
  public native int template_id(); public native Match template_id(int template_id);
}

     

/**
 * \brief Object detector using the LINE template matching algorithm with any set of
 * modalities.
 */
@Namespace("cv::linemod") @NoOffset public static class Detector extends Pointer {
    static { Loader.load(); }
    public Detector(Pointer p) { super(p); }
    public Detector(int size) { allocateArray(size); }
    private native void allocateArray(int size);
    @Override public Detector position(int position) {
        return (Detector)super.position(position);
    }

  /**
   * \brief Empty constructor, initialize with read().
   */
  public Detector() { allocate(); }
  private native void allocate();

  /**
   * \brief Constructor.
   *
   * \param modalities       Modalities to use (color gradients, depth normals, ...).
   * \param T_pyramid        Value of the sampling step T at each pyramid level. The
   *                         number of pyramid levels is T_pyramid.size().
   */
  public Detector(@Const @ByRef ModalityVector modalities, @StdVector IntPointer T_pyramid) { allocate(modalities, T_pyramid); }
  private native void allocate(@Const @ByRef ModalityVector modalities, @StdVector IntPointer T_pyramid);
  public Detector(@Const @ByRef ModalityVector modalities, @StdVector IntBuffer T_pyramid) { allocate(modalities, T_pyramid); }
  private native void allocate(@Const @ByRef ModalityVector modalities, @StdVector IntBuffer T_pyramid);
  public Detector(@Const @ByRef ModalityVector modalities, @StdVector int[] T_pyramid) { allocate(modalities, T_pyramid); }
  private native void allocate(@Const @ByRef ModalityVector modalities, @StdVector int[] T_pyramid);

  /**
   * \brief Detect objects by template matching.
   *
   * Matches globally at the lowest pyramid level, then refines locally stepping up the pyramid.
   *
   * \param      sources   Source images, one for each modality.
   * \param      threshold Similarity threshold, a percentage between 0 and 100.
   * \param[out] matches   Template matches, sorted by similarity score.
   * \param      class_ids If non-empty, only search for the desired object classes.
   * \param[out] quantized_images Optionally return vector<Mat> of quantized images.
   * \param      masks     The masks for consideration during matching. The masks should be CV_8UC1
   *                       where 255 represents a valid pixel.  If non-empty, the vector must be
   *                       the same size as sources.  Each element must be
   *                       empty or the same size as its corresponding source.
   */
  public native void match(@Const @ByRef MatVector sources, float threshold, @StdVector Match matches,
               @Const @ByRef StringVector class_ids/*=std::vector<std::string>()*/,
               @ByVal MatVector quantized_images/*=noArray()*/,
               @Const @ByRef MatVector masks/*=std::vector<Mat>()*/);

  /**
   * \brief Add new object template.
   *
   * \param      sources      Source images, one for each modality.
   * \param      class_id     Object class ID.
   * \param      object_mask  Mask separating object from background.
   * \param[out] bounding_box Optionally return bounding box of the extracted features.
   *
   * \return Template ID, or -1 if failed to extract a valid template.
   */
  public native int addTemplate(@Const @ByRef MatVector sources, @StdString BytePointer class_id,
            @Const @ByRef Mat object_mask, Rect bounding_box/*=NULL*/);
  public native int addTemplate(@Const @ByRef MatVector sources, @StdString String class_id,
            @Const @ByRef Mat object_mask, Rect bounding_box/*=NULL*/);

  /**
   * \brief Add a new object template computed by external means.
   */
  public native int addSyntheticTemplate(@StdVector Template templates, @StdString BytePointer class_id);
  public native int addSyntheticTemplate(@StdVector Template templates, @StdString String class_id);

  /**
   * \brief Get the modalities used by this detector.
   *
   * You are not permitted to add/remove modalities, but you may dynamic_cast them to
   * tweak parameters.
   */
  public native @Const @ByRef ModalityVector getModalities();

  /**
   * \brief Get sampling step T at pyramid_level.
   */
  public native int getT(int pyramid_level);

  /**
   * \brief Get number of pyramid levels used by this detector.
   */
  public native int pyramidLevels();

  /**
   * \brief Get the template pyramid identified by template_id.
   *
   * For example, with 2 modalities (Gradient, Normal) and two pyramid levels
   * (L0, L1), the order is (GradientL0, NormalL0, GradientL1, NormalL1).
   */
  public native @StdVector Template getTemplates(@StdString BytePointer class_id, int template_id);
  public native @StdVector Template getTemplates(@StdString String class_id, int template_id);

  public native int numTemplates();
  public native int numTemplates(@StdString BytePointer class_id);
  public native int numTemplates(@StdString String class_id);
  public native int numClasses();

  public native @ByVal StringVector classIds();

  public native void read(@Const @ByRef FileNode fn);
  public native void write(@ByRef FileStorage fs);

  public native @StdString BytePointer readClass(@Const @ByRef FileNode fn, @StdString BytePointer class_id_override/*=""*/);
  public native @StdString String readClass(@Const @ByRef FileNode fn, @StdString String class_id_override/*=""*/);
  public native void writeClass(@StdString BytePointer class_id, @ByRef FileStorage fs);
  public native void writeClass(@StdString String class_id, @ByRef FileStorage fs);

  public native void readClasses(@Const @ByRef StringVector class_ids,
                     @StdString BytePointer format/*="templates_%s.yml.gz"*/);
  public native void readClasses(@Const @ByRef StringVector class_ids,
                     @StdString String format/*="templates_%s.yml.gz"*/);
  public native void writeClasses(@StdString BytePointer format/*="templates_%s.yml.gz"*/);
  public native void writeClasses(@StdString String format/*="templates_%s.yml.gz"*/);
}

/**
 * \brief Factory function for detector using LINE algorithm with color gradients.
 *
 * Default parameter settings suitable for VGA images.
 */
@Namespace("cv::linemod") public static native @Ptr Detector getDefaultLINE();

/**
 * \brief Factory function for detector using LINE-MOD algorithm with color gradients
 * and depth normals.
 *
 * Default parameter settings suitable for VGA images.
 */
@Namespace("cv::linemod") public static native @Ptr Detector getDefaultLINEMOD();


 // namespace cv

// #endif

// #endif


}
