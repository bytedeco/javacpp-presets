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
import static org.bytedeco.javacpp.opencv_features2d.*;
import static org.bytedeco.javacpp.opencv_calib3d.*;
import static org.bytedeco.javacpp.opencv_objdetect.*;
import static org.bytedeco.javacpp.opencv_video.*;
import static org.bytedeco.javacpp.opencv_ml.*;
import static org.bytedeco.javacpp.opencv_photo.*;
import static org.bytedeco.javacpp.opencv_legacy.*;
import static org.bytedeco.javacpp.opencv_nonfree.*;

public class opencv_contrib extends org.bytedeco.javacpp.presets.opencv_contrib {
    static { Loader.load(); }

@Name("std::map<int,std::string>") public static class IntStringMap extends Pointer {
    static { Loader.load(); }
    public IntStringMap(Pointer p) { super(p); }
    public IntStringMap()       { allocate();  }
    private native void allocate();
    public native @Name("operator=") @ByRef IntStringMap put(@ByRef IntStringMap x);

    public native long size();

    @Index public native @StdString BytePointer get(int i);
    public native IntStringMap put(int i, BytePointer value);
}

@Name("std::vector<std::pair<cv::Rect_<int>,int> >") public static class RectIntPairVector extends Pointer {
    static { Loader.load(); }
    public RectIntPairVector(Pointer p) { super(p); }
    public RectIntPairVector()       { allocate();  }
    public RectIntPairVector(long n) { allocate(n); }
    private native void allocate();
    private native void allocate(@Cast("size_t") long n);
    public native @Name("operator=") @ByRef RectIntPairVector put(@ByRef RectIntPairVector x);

    public native long size();
    public native void resize(@Cast("size_t") long n);

    @Index public native @ByRef Rect first(@Cast("size_t") long i); public native RectIntPairVector first(@Cast("size_t") long i, Rect first);
    @Index public native @ByRef int second(@Cast("size_t") long i);  public native RectIntPairVector second(@Cast("size_t") long i, int second);
}

@Name("std::valarray<float>") public static class FloatValArray extends Pointer {
    static { Loader.load(); }
    public FloatValArray(Pointer p) { super(p); }
    public FloatValArray(float ... array) { this(array.length); put(array); }
    public FloatValArray()       { allocate();  }
    public FloatValArray(long n) { allocate(n); }
    private native void allocate();
    private native void allocate(@Cast("size_t") long n);
    public native @Name("operator=") @ByRef FloatValArray put(@ByRef FloatValArray x);

    public native long size();
    public native void resize(@Cast("size_t") long n);

    @Index public native @ByRef float get(@Cast("size_t") long i);
    public native FloatValArray put(@Cast("size_t") long i, float value);

    public FloatValArray put(float ... array) {
        if (size() != array.length) { resize(array.length); }
        for (int i = 0; i < array.length; i++) {
            put(i, array[i]);
        }
        return this;
    }
}

// Parsed from <opencv2/contrib/contrib.hpp>

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

// #ifndef __OPENCV_CONTRIB_HPP__
// #define __OPENCV_CONTRIB_HPP__

// #include "opencv2/core/core.hpp"
// #include "opencv2/imgproc/imgproc.hpp"
// #include "opencv2/features2d/features2d.hpp"
// #include "opencv2/objdetect/objdetect.hpp"

// #ifdef __cplusplus

/****************************************************************************************\
*                                   Adaptive Skin Detector                               *
\****************************************************************************************/

@NoOffset public static class CvAdaptiveSkinDetector extends Pointer {
    static { Loader.load(); }
    public CvAdaptiveSkinDetector(Pointer p) { super(p); }
    public CvAdaptiveSkinDetector(int size) { allocateArray(size); }
    private native void allocateArray(int size);
    @Override public CvAdaptiveSkinDetector position(int position) {
        return (CvAdaptiveSkinDetector)super.position(position);
    }


    /** enum CvAdaptiveSkinDetector:: */
    public static final int
        MORPHING_METHOD_NONE = 0,
        MORPHING_METHOD_ERODE = 1,
        MORPHING_METHOD_ERODE_ERODE = 2,
        MORPHING_METHOD_ERODE_DILATE = 3;

    public CvAdaptiveSkinDetector(int samplingDivider/*=1*/, int morphingMethod/*=MORPHING_METHOD_NONE*/) { allocate(samplingDivider, morphingMethod); }
    private native void allocate(int samplingDivider/*=1*/, int morphingMethod/*=MORPHING_METHOD_NONE*/);
    public CvAdaptiveSkinDetector() { allocate(); }
    private native void allocate();

    public native void process(IplImage inputBGRImage, IplImage outputHueMask);
}


/****************************************************************************************\
 *                                  Fuzzy MeanShift Tracker                               *
 \****************************************************************************************/

@NoOffset public static class CvFuzzyPoint extends Pointer {
    static { Loader.load(); }
    public CvFuzzyPoint() { }
    public CvFuzzyPoint(Pointer p) { super(p); }

    public native double x(); public native CvFuzzyPoint x(double x);
    public native double y(); public native CvFuzzyPoint y(double y);
    public native double value(); public native CvFuzzyPoint value(double value);

    public CvFuzzyPoint(double _x, double _y) { allocate(_x, _y); }
    private native void allocate(double _x, double _y);
}

@NoOffset public static class CvFuzzyCurve extends Pointer {
    static { Loader.load(); }
    public CvFuzzyCurve(Pointer p) { super(p); }
    public CvFuzzyCurve(int size) { allocateArray(size); }
    private native void allocateArray(int size);
    @Override public CvFuzzyCurve position(int position) {
        return (CvFuzzyCurve)super.position(position);
    }

    public CvFuzzyCurve() { allocate(); }
    private native void allocate();

    public native void setCentre(double _centre);
    public native double getCentre();
    public native void clear();
    public native void addPoint(double x, double y);
    public native double calcValue(double param);
    public native double getValue();
    public native void setValue(double _value);
}

@NoOffset public static class CvFuzzyFunction extends Pointer {
    static { Loader.load(); }
    public CvFuzzyFunction(Pointer p) { super(p); }
    public CvFuzzyFunction(int size) { allocateArray(size); }
    private native void allocateArray(int size);
    @Override public CvFuzzyFunction position(int position) {
        return (CvFuzzyFunction)super.position(position);
    }

    public native @StdVector CvFuzzyCurve curves(); public native CvFuzzyFunction curves(CvFuzzyCurve curves);

    public CvFuzzyFunction() { allocate(); }
    private native void allocate();
    public native void addCurve(CvFuzzyCurve curve, double value/*=0*/);
    public native void addCurve(CvFuzzyCurve curve);
    public native void resetValues();
    public native double calcValue();
    public native CvFuzzyCurve newCurve();
}

@NoOffset public static class CvFuzzyRule extends Pointer {
    static { Loader.load(); }
    public CvFuzzyRule(Pointer p) { super(p); }
    public CvFuzzyRule(int size) { allocateArray(size); }
    private native void allocateArray(int size);
    @Override public CvFuzzyRule position(int position) {
        return (CvFuzzyRule)super.position(position);
    }

    public CvFuzzyRule() { allocate(); }
    private native void allocate();
    public native void setRule(CvFuzzyCurve c1, CvFuzzyCurve c2, CvFuzzyCurve o1);
    public native double calcValue(double param1, double param2);
    public native CvFuzzyCurve getOutputCurve();
}

@NoOffset public static class CvFuzzyController extends Pointer {
    static { Loader.load(); }
    public CvFuzzyController(Pointer p) { super(p); }
    public CvFuzzyController(int size) { allocateArray(size); }
    private native void allocateArray(int size);
    @Override public CvFuzzyController position(int position) {
        return (CvFuzzyController)super.position(position);
    }

    public CvFuzzyController() { allocate(); }
    private native void allocate();
    public native void addRule(CvFuzzyCurve c1, CvFuzzyCurve c2, CvFuzzyCurve o1);
    public native double calcOutput(double param1, double param2);
}

@NoOffset public static class CvFuzzyMeanShiftTracker extends Pointer {
    static { Loader.load(); }
    public CvFuzzyMeanShiftTracker(Pointer p) { super(p); }
    public CvFuzzyMeanShiftTracker(int size) { allocateArray(size); }
    private native void allocateArray(int size);
    @Override public CvFuzzyMeanShiftTracker position(int position) {
        return (CvFuzzyMeanShiftTracker)super.position(position);
    }

    /** enum CvFuzzyMeanShiftTracker::TrackingState */
    public static final int
        tsNone          = 0,
        tsSearching     = 1,
        tsTracking      = 2,
        tsSetWindow     = 3,
        tsDisabled      = 10;

    /** enum CvFuzzyMeanShiftTracker::ResizeMethod */
    public static final int
        rmEdgeDensityLinear     = 0,
        rmEdgeDensityFuzzy      = 1,
        rmInnerDensity          = 2;

    /** enum CvFuzzyMeanShiftTracker:: */
    public static final int
        MinKernelMass           = 1000;

    
    public native int searchMode(); public native CvFuzzyMeanShiftTracker searchMode(int searchMode);
    public CvFuzzyMeanShiftTracker() { allocate(); }
    private native void allocate();

    public native void track(IplImage maskImage, IplImage depthMap, int resizeMethod, @Cast("bool") boolean resetSearch, int minKernelMass/*=MinKernelMass*/);
    public native void track(IplImage maskImage, IplImage depthMap, int resizeMethod, @Cast("bool") boolean resetSearch);
}

    @Namespace("cv") @NoOffset public static class Octree extends Pointer {
        static { Loader.load(); }
        public Octree(Pointer p) { super(p); }
        public Octree(int size) { allocateArray(size); }
        private native void allocateArray(int size);
        @Override public Octree position(int position) {
            return (Octree)super.position(position);
        }
    
        @NoOffset public static class Node extends Pointer {
            static { Loader.load(); }
            public Node(Pointer p) { super(p); }
            public Node(int size) { allocateArray(size); }
            private native void allocateArray(int size);
            @Override public Node position(int position) {
                return (Node)super.position(position);
            }
        
            public Node() { allocate(); }
            private native void allocate();
            public native int begin(); public native Node begin(int begin);
            public native int end(); public native Node end(int end);
            public native float x_min(); public native Node x_min(float x_min);
            public native float x_max(); public native Node x_max(float x_max);
            public native float y_min(); public native Node y_min(float y_min);
            public native float y_max(); public native Node y_max(float y_max);
            public native float z_min(); public native Node z_min(float z_min);
            public native float z_max(); public native Node z_max(float z_max);
            public native int maxLevels(); public native Node maxLevels(int maxLevels);
            public native @Cast("bool") boolean isLeaf(); public native Node isLeaf(boolean isLeaf);
            public native int children(int i); public native Node children(int i, int children);
            @MemberGetter public native IntPointer children();
        }

        public Octree() { allocate(); }
        private native void allocate();
        public Octree( @StdVector Point3f points, int maxLevels/*=10*/, int minPoints/*=20*/ ) { allocate(points, maxLevels, minPoints); }
        private native void allocate( @StdVector Point3f points, int maxLevels/*=10*/, int minPoints/*=20*/ );
        public Octree( @StdVector Point3f points ) { allocate(points); }
        private native void allocate( @StdVector Point3f points );

        public native void buildTree( @StdVector Point3f points, int maxLevels/*=10*/, int minPoints/*=20*/ );
        public native void buildTree( @StdVector Point3f points );
        public native void getPointsWithinSphere( @Const @ByRef Point3f center, float radius,
                                                   @StdVector Point3f points );
        public native @StdVector Node getNodes();
    }


    @Namespace("cv") @NoOffset public static class Mesh3D extends Pointer {
        static { Loader.load(); }
        public Mesh3D(Pointer p) { super(p); }
        public Mesh3D(int size) { allocateArray(size); }
        private native void allocateArray(int size);
        @Override public Mesh3D position(int position) {
            return (Mesh3D)super.position(position);
        }
    
        public static class EmptyMeshException extends Pointer {
            static { Loader.load(); }
            public EmptyMeshException() { allocate(); }
            public EmptyMeshException(int size) { allocateArray(size); }
            public EmptyMeshException(Pointer p) { super(p); }
            private native void allocate();
            private native void allocateArray(int size);
            @Override public EmptyMeshException position(int position) {
                return (EmptyMeshException)super.position(position);
            }
        }

        public Mesh3D() { allocate(); }
        private native void allocate();
        public Mesh3D(@StdVector Point3f vtx) { allocate(vtx); }
        private native void allocate(@StdVector Point3f vtx);

        public native void buildOctree();
        public native void clearOctree();
        public native float estimateResolution(float tryRatio/*=0.1f*/);
        public native float estimateResolution();
        public native void computeNormals(float normalRadius, int minNeighbors/*=20*/);
        public native void computeNormals(float normalRadius);
        public native void computeNormals(@StdVector IntPointer subset, float normalRadius, int minNeighbors/*=20*/);
        public native void computeNormals(@StdVector IntPointer subset, float normalRadius);
        public native void computeNormals(@StdVector IntBuffer subset, float normalRadius, int minNeighbors/*=20*/);
        public native void computeNormals(@StdVector IntBuffer subset, float normalRadius);
        public native void computeNormals(@StdVector int[] subset, float normalRadius, int minNeighbors/*=20*/);
        public native void computeNormals(@StdVector int[] subset, float normalRadius);

        public native void writeAsVrml(@StdString BytePointer file, @StdVector Scalar colors/*=vector<Scalar>()*/);
        public native void writeAsVrml(@StdString BytePointer file);
        public native void writeAsVrml(@StdString String file, @StdVector Scalar colors/*=vector<Scalar>()*/);
        public native void writeAsVrml(@StdString String file);

        public native @StdVector Point3f vtx(); public native Mesh3D vtx(Point3f vtx);
        public native @StdVector Point3f normals(); public native Mesh3D normals(Point3f normals);
        public native float resolution(); public native Mesh3D resolution(float resolution);
        public native @ByRef Octree octree(); public native Mesh3D octree(Octree octree);

        
    }

    @Namespace("cv") @NoOffset public static class SpinImageModel extends Pointer {
        static { Loader.load(); }
        public SpinImageModel(Pointer p) { super(p); }
        public SpinImageModel(int size) { allocateArray(size); }
        private native void allocateArray(int size);
        @Override public SpinImageModel position(int position) {
            return (SpinImageModel)super.position(position);
        }
    

        /* model parameters, leave unset for default or auto estimate */
        public native float normalRadius(); public native SpinImageModel normalRadius(float normalRadius);
        public native int minNeighbors(); public native SpinImageModel minNeighbors(int minNeighbors);

        public native float binSize(); public native SpinImageModel binSize(float binSize);
        public native int imageWidth(); public native SpinImageModel imageWidth(int imageWidth);

        public native float lambda(); public native SpinImageModel lambda(float lambda);
        public native float gamma(); public native SpinImageModel gamma(float gamma);

        public native float T_GeometriccConsistency(); public native SpinImageModel T_GeometriccConsistency(float T_GeometriccConsistency);
        public native float T_GroupingCorespondances(); public native SpinImageModel T_GroupingCorespondances(float T_GroupingCorespondances);

        /* public interface */
        public SpinImageModel() { allocate(); }
        private native void allocate();
        public SpinImageModel(@Const @ByRef Mesh3D mesh) { allocate(mesh); }
        private native void allocate(@Const @ByRef Mesh3D mesh);

        public native void setLogger(@Cast("std::ostream*") Pointer log);
        public native void selectRandomSubset(float ratio);
        public native void setSubset(@StdVector IntPointer subset);
        public native void setSubset(@StdVector IntBuffer subset);
        public native void setSubset(@StdVector int[] subset);
        public native void compute();

        public native void match(@Const @ByRef SpinImageModel scene, @Cast("std::vector<std::vector<cv::Vec2i> >*") @ByRef PointVectorVector result);

        public native @ByVal Mat packRandomScaledSpins(@Cast("bool") boolean separateScale/*=false*/, @Cast("size_t") long xCount/*=10*/, @Cast("size_t") long yCount/*=10*/);
        public native @ByVal Mat packRandomScaledSpins();

        public native @Cast("size_t") long getSpinCount();
        public native @ByVal Mat getSpinImage(@Cast("size_t") long index);
        public native @Const @ByRef Point3f getSpinVertex(@Cast("size_t") long index);
        public native @Const @ByRef Point3f getSpinNormal(@Cast("size_t") long index);
        public native @ByRef Mesh3D getMesh();

        /* static utility functions */
        public static native @Cast("bool") boolean spinCorrelation(@Const @ByRef Mat spin1, @Const @ByRef Mat spin2, float lambda, @ByRef FloatPointer result);
        public static native @Cast("bool") boolean spinCorrelation(@Const @ByRef Mat spin1, @Const @ByRef Mat spin2, float lambda, @ByRef FloatBuffer result);
        public static native @Cast("bool") boolean spinCorrelation(@Const @ByRef Mat spin1, @Const @ByRef Mat spin2, float lambda, @ByRef float[] result);

        

        

        
    }

    @Namespace("cv") @NoOffset public static class TickMeter extends Pointer {
        static { Loader.load(); }
        public TickMeter(Pointer p) { super(p); }
        public TickMeter(int size) { allocateArray(size); }
        private native void allocateArray(int size);
        @Override public TickMeter position(int position) {
            return (TickMeter)super.position(position);
        }
    
        public TickMeter() { allocate(); }
        private native void allocate();
        public native void start();
        public native void stop();

        public native @Cast("int64") long getTimeTicks();
        public native double getTimeMicro();
        public native double getTimeMilli();
        public native double getTimeSec();
        public native @Cast("int64") long getCounter();

        public native void reset();
    }

    @Namespace("cv") public static native @Cast("std::ostream*") @ByRef @Name("operator<<") Pointer shiftLeft(@Cast("std::ostream*") @ByRef Pointer out, @Const @ByRef TickMeter tm);

    @Namespace("cv") @NoOffset public static class SelfSimDescriptor extends Pointer {
        static { Loader.load(); }
        public SelfSimDescriptor(Pointer p) { super(p); }
        public SelfSimDescriptor(int size) { allocateArray(size); }
        private native void allocateArray(int size);
        @Override public SelfSimDescriptor position(int position) {
            return (SelfSimDescriptor)super.position(position);
        }
    
        public SelfSimDescriptor() { allocate(); }
        private native void allocate();
        public SelfSimDescriptor(int _ssize, int _lsize,
                                  int _startDistanceBucket/*=DEFAULT_START_DISTANCE_BUCKET*/,
                                  int _numberOfDistanceBuckets/*=DEFAULT_NUM_DISTANCE_BUCKETS*/,
                                  int _nangles/*=DEFAULT_NUM_ANGLES*/) { allocate(_ssize, _lsize, _startDistanceBucket, _numberOfDistanceBuckets, _nangles); }
        private native void allocate(int _ssize, int _lsize,
                                  int _startDistanceBucket/*=DEFAULT_START_DISTANCE_BUCKET*/,
                                  int _numberOfDistanceBuckets/*=DEFAULT_NUM_DISTANCE_BUCKETS*/,
                                  int _nangles/*=DEFAULT_NUM_ANGLES*/);
        public SelfSimDescriptor(int _ssize, int _lsize) { allocate(_ssize, _lsize); }
        private native void allocate(int _ssize, int _lsize);
        public SelfSimDescriptor(@Const @ByRef SelfSimDescriptor ss) { allocate(ss); }
        private native void allocate(@Const @ByRef SelfSimDescriptor ss);
        public native @ByRef @Name("operator=") SelfSimDescriptor put(@Const @ByRef SelfSimDescriptor ss);

        public native @Cast("size_t") long getDescriptorSize();
        public native @ByVal Size getGridSize( @ByVal Size imgsize, @ByVal Size winStride );

        public native void compute(@Const @ByRef Mat img, @StdVector FloatPointer descriptors, @ByVal Size winStride/*=Size()*/,
                                     @StdVector Point locations/*=vector<Point>()*/);
        public native void compute(@Const @ByRef Mat img, @StdVector FloatPointer descriptors);
        public native void compute(@Const @ByRef Mat img, @StdVector FloatBuffer descriptors, @ByVal Size winStride/*=Size()*/,
                                     @StdVector Point locations/*=vector<Point>()*/);
        public native void compute(@Const @ByRef Mat img, @StdVector FloatBuffer descriptors);
        public native void compute(@Const @ByRef Mat img, @StdVector float[] descriptors, @ByVal Size winStride/*=Size()*/,
                                     @StdVector Point locations/*=vector<Point>()*/);
        public native void compute(@Const @ByRef Mat img, @StdVector float[] descriptors);
        public native void computeLogPolarMapping(@ByRef Mat mappingMask);
        public native void SSD(@Const @ByRef Mat img, @ByVal Point pt, @ByRef Mat ssd);

        public native int smallSize(); public native SelfSimDescriptor smallSize(int smallSize);
        public native int largeSize(); public native SelfSimDescriptor largeSize(int largeSize);
        public native int startDistanceBucket(); public native SelfSimDescriptor startDistanceBucket(int startDistanceBucket);
        public native int numberOfDistanceBuckets(); public native SelfSimDescriptor numberOfDistanceBuckets(int numberOfDistanceBuckets);
        public native int numberOfAngles(); public native SelfSimDescriptor numberOfAngles(int numberOfAngles);

        /** enum cv::SelfSimDescriptor:: */
        public static final int DEFAULT_SMALL_SIZE = 5, DEFAULT_LARGE_SIZE = 41,
            DEFAULT_NUM_ANGLES = 20, DEFAULT_START_DISTANCE_BUCKET = 3,
            DEFAULT_NUM_DISTANCE_BUCKETS = 7;
    }


    public static class BundleAdjustCallback extends FunctionPointer {
        static { Loader.load(); }
        public    BundleAdjustCallback(Pointer p) { super(p); }
        protected BundleAdjustCallback() { allocate(); }
        private native void allocate();
        public native @Cast("bool") boolean call(int iteration, double norm_error, Pointer user_data);
    }

    @Namespace("cv") @NoOffset public static class LevMarqSparse extends Pointer {
        static { Loader.load(); }
        public LevMarqSparse(Pointer p) { super(p); }
        public LevMarqSparse(int size) { allocateArray(size); }
        private native void allocateArray(int size);
        @Override public LevMarqSparse position(int position) {
            return (LevMarqSparse)super.position(position);
        }
    
        public LevMarqSparse() { allocate(); }
        private native void allocate();
        @Convention("CV_CDECL") public static class Fjac_int_int_Mat_Mat_Mat_Mat_Pointer extends FunctionPointer {
            static { Loader.load(); }
            public    Fjac_int_int_Mat_Mat_Mat_Mat_Pointer(Pointer p) { super(p); }
            protected Fjac_int_int_Mat_Mat_Mat_Mat_Pointer() { allocate(); }
            private native void allocate();
            public native void call(int i, int j, @ByRef Mat point_params,
                                                     @ByRef Mat cam_params, @ByRef Mat A, @ByRef Mat B, Pointer data);
        }
        @Convention("CV_CDECL") public static class Func_int_int_Mat_Mat_Mat_Pointer extends FunctionPointer {
            static { Loader.load(); }
            public    Func_int_int_Mat_Mat_Mat_Pointer(Pointer p) { super(p); }
            protected Func_int_int_Mat_Mat_Mat_Pointer() { allocate(); }
            private native void allocate();
            public native void call(int i, int j, @ByRef Mat point_params,
                                                     @ByRef Mat cam_params, @ByRef Mat estim, Pointer data);
        }
        public LevMarqSparse(int npoints,
                              int ncameras,
                              int nPointParams,
                              int nCameraParams,
                              int nErrParams,
                              @ByRef Mat visibility,
                              @ByRef Mat P0,
                              @ByRef Mat X,
                              @ByVal TermCriteria criteria,
                              Fjac_int_int_Mat_Mat_Mat_Mat_Pointer fjac,
                              Func_int_int_Mat_Mat_Mat_Pointer func,
                              Pointer data,
                              BundleAdjustCallback cb, Pointer user_data
                              ) { allocate(npoints, ncameras, nPointParams, nCameraParams, nErrParams, visibility, P0, X, criteria, fjac, func, data, cb, user_data); }
        private native void allocate(int npoints,
                              int ncameras,
                              int nPointParams,
                              int nCameraParams,
                              int nErrParams,
                              @ByRef Mat visibility,
                              @ByRef Mat P0,
                              @ByRef Mat X,
                              @ByVal TermCriteria criteria,
                              Fjac_int_int_Mat_Mat_Mat_Mat_Pointer fjac,
                              Func_int_int_Mat_Mat_Mat_Pointer func,
                              Pointer data,
                              BundleAdjustCallback cb, Pointer user_data
                              );

        public native void run( int npoints,
                                 int ncameras,
                                 int nPointParams,
                                 int nCameraParams,
                                 int nErrParams,
                                 @ByRef Mat visibility,
                                 @ByRef Mat P0,
                                 @ByRef Mat X,
                                 @ByVal TermCriteria criteria,
                                 Fjac_int_int_Mat_Mat_Mat_Mat_Pointer fjac,
                                 Func_int_int_Mat_Mat_Mat_Pointer func,
                                 Pointer data
                                 );

        public native void clear();

        // useful function to do simple bundle adjustment tasks
        public static native void bundleAdjust(@StdVector Point3d points,
                                         @Const @ByRef Point2dVectorVector imagePoints,
                                         @Const @ByRef IntVectorVector visibility,
                                         @ByRef MatVector cameraMatrix,
                                         @ByRef MatVector R,
                                         @ByRef MatVector T,
                                         @ByRef MatVector distCoeffs,
                                         @Const @ByRef TermCriteria criteria/*=TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 30, DBL_EPSILON)*/,
                                         BundleAdjustCallback cb/*=0*/, Pointer user_data/*=0*/);
        public static native void bundleAdjust(@StdVector Point3d points,
                                         @Const @ByRef Point2dVectorVector imagePoints,
                                         @Const @ByRef IntVectorVector visibility,
                                         @ByRef MatVector cameraMatrix,
                                         @ByRef MatVector R,
                                         @ByRef MatVector T,
                                         @ByRef MatVector distCoeffs);
        public native void optimize(@ByRef CvMat _vis); //main function that runs minimization

        //iteratively asks for measurement for visible camera-point pairs
        public native void ask_for_proj(@ByRef CvMat _vis,@Cast("bool") boolean once/*=false*/);
        public native void ask_for_proj(@ByRef CvMat _vis);
        //iteratively asks for Jacobians for every camera_point pair
        public native void ask_for_projac(@ByRef CvMat _vis);

        public native CvMat err(); public native LevMarqSparse err(CvMat err); //error X-hX
        public native double prevErrNorm(); public native LevMarqSparse prevErrNorm(double prevErrNorm);
        public native double errNorm(); public native LevMarqSparse errNorm(double errNorm);
        public native double lambda(); public native LevMarqSparse lambda(double lambda);
        public native @ByRef CvTermCriteria criteria(); public native LevMarqSparse criteria(CvTermCriteria criteria);
        public native int iters(); public native LevMarqSparse iters(int iters);

        public native CvMat U(int i); public native LevMarqSparse U(int i, CvMat U);
        @MemberGetter public native @Cast("CvMat**") PointerPointer U(); //size of array is equal to number of cameras
        public native CvMat V(int i); public native LevMarqSparse V(int i, CvMat V);
        @MemberGetter public native @Cast("CvMat**") PointerPointer V(); //size of array is equal to number of points
        public native CvMat inv_V_star(int i); public native LevMarqSparse inv_V_star(int i, CvMat inv_V_star);
        @MemberGetter public native @Cast("CvMat**") PointerPointer inv_V_star(); //inverse of V*

        public native CvMat A(int i); public native LevMarqSparse A(int i, CvMat A);
        @MemberGetter public native @Cast("CvMat**") PointerPointer A();
        public native CvMat B(int i); public native LevMarqSparse B(int i, CvMat B);
        @MemberGetter public native @Cast("CvMat**") PointerPointer B();
        public native CvMat W(int i); public native LevMarqSparse W(int i, CvMat W);
        @MemberGetter public native @Cast("CvMat**") PointerPointer W();

        public native CvMat X(); public native LevMarqSparse X(CvMat X); //measurement
        public native CvMat hX(); public native LevMarqSparse hX(CvMat hX); //current measurement extimation given new parameter vector

        public native CvMat prevP(); public native LevMarqSparse prevP(CvMat prevP); //current already accepted parameter.
        public native CvMat P(); public native LevMarqSparse P(CvMat P); // parameters used to evaluate function with new params
        // this parameters may be rejected

        public native CvMat deltaP(); public native LevMarqSparse deltaP(CvMat deltaP); //computed increase of parameters (result of normal system solution )

        public native CvMat ea(int i); public native LevMarqSparse ea(int i, CvMat ea);
        @MemberGetter public native @Cast("CvMat**") PointerPointer ea(); // sum_i  AijT * e_ij , used as right part of normal equation
        // length of array is j = number of cameras
        public native CvMat eb(int i); public native LevMarqSparse eb(int i, CvMat eb);
        @MemberGetter public native @Cast("CvMat**") PointerPointer eb(); // sum_j  BijT * e_ij , used as right part of normal equation
        // length of array is i = number of points

        public native CvMat Yj(int i); public native LevMarqSparse Yj(int i, CvMat Yj);
        @MemberGetter public native @Cast("CvMat**") PointerPointer Yj(); //length of array is i = num_points

        public native CvMat S(); public native LevMarqSparse S(CvMat S); //big matrix of block Sjk  , each block has size num_cam_params x num_cam_params

        public native CvMat JtJ_diag(); public native LevMarqSparse JtJ_diag(CvMat JtJ_diag); //diagonal of JtJ,  used to backup diagonal elements before augmentation

        public native CvMat Vis_index(); public native LevMarqSparse Vis_index(CvMat Vis_index); // matrix which element is index of measurement for point i and camera j

        public native int num_cams(); public native LevMarqSparse num_cams(int num_cams);
        public native int num_points(); public native LevMarqSparse num_points(int num_points);
        public native int num_err_param(); public native LevMarqSparse num_err_param(int num_err_param);
        public native int num_cam_param(); public native LevMarqSparse num_cam_param(int num_cam_param);
        public native int num_point_param(); public native LevMarqSparse num_point_param(int num_point_param);

        //target function and jacobian pointers, which needs to be initialized
        public native Fjac_int_int_Mat_Mat_Mat_Mat_Pointer fjac(); public native LevMarqSparse fjac(Fjac_int_int_Mat_Mat_Mat_Mat_Pointer fjac);
        public native Func_int_int_Mat_Mat_Mat_Pointer func(); public native LevMarqSparse func(Func_int_int_Mat_Mat_Mat_Pointer func);

        public native Pointer data(); public native LevMarqSparse data(Pointer data);

        public native BundleAdjustCallback cb(); public native LevMarqSparse cb(BundleAdjustCallback cb);
        public native Pointer user_data(); public native LevMarqSparse user_data(Pointer user_data);
    }

    @Namespace("cv") public static native int chamerMatching( @ByRef Mat img, @ByRef Mat templ,
                                      @ByRef PointVectorVector results, @StdVector FloatPointer cost,
                                      double templScale/*=1*/, int maxMatches/*=20*/,
                                      double minMatchDistance/*=1.0*/, int padX/*=3*/,
                                      int padY/*=3*/, int scales/*=5*/, double minScale/*=0.6*/, double maxScale/*=1.6*/,
                                      double orientationWeight/*=0.5*/, double truncate/*=20*/);
    @Namespace("cv") public static native int chamerMatching( @ByRef Mat img, @ByRef Mat templ,
                                      @ByRef PointVectorVector results, @StdVector FloatPointer cost);
    @Namespace("cv") public static native int chamerMatching( @ByRef Mat img, @ByRef Mat templ,
                                      @ByRef PointVectorVector results, @StdVector FloatBuffer cost,
                                      double templScale/*=1*/, int maxMatches/*=20*/,
                                      double minMatchDistance/*=1.0*/, int padX/*=3*/,
                                      int padY/*=3*/, int scales/*=5*/, double minScale/*=0.6*/, double maxScale/*=1.6*/,
                                      double orientationWeight/*=0.5*/, double truncate/*=20*/);
    @Namespace("cv") public static native int chamerMatching( @ByRef Mat img, @ByRef Mat templ,
                                      @ByRef PointVectorVector results, @StdVector FloatBuffer cost);
    @Namespace("cv") public static native int chamerMatching( @ByRef Mat img, @ByRef Mat templ,
                                      @ByRef PointVectorVector results, @StdVector float[] cost,
                                      double templScale/*=1*/, int maxMatches/*=20*/,
                                      double minMatchDistance/*=1.0*/, int padX/*=3*/,
                                      int padY/*=3*/, int scales/*=5*/, double minScale/*=0.6*/, double maxScale/*=1.6*/,
                                      double orientationWeight/*=0.5*/, double truncate/*=20*/);
    @Namespace("cv") public static native int chamerMatching( @ByRef Mat img, @ByRef Mat templ,
                                      @ByRef PointVectorVector results, @StdVector float[] cost);


    @Namespace("cv") @NoOffset public static class StereoVar extends Pointer {
        static { Loader.load(); }
        public StereoVar(Pointer p) { super(p); }
        public StereoVar(int size) { allocateArray(size); }
        private native void allocateArray(int size);
        @Override public StereoVar position(int position) {
            return (StereoVar)super.position(position);
        }
    
        // Flags
        /** enum cv::StereoVar:: */
        public static final int USE_INITIAL_DISPARITY = 1, USE_EQUALIZE_HIST = 2, USE_SMART_ID = 4, USE_AUTO_PARAMS = 8, USE_MEDIAN_FILTERING = 16;
        /** enum cv::StereoVar:: */
        public static final int CYCLE_O = 0, CYCLE_V = 1;
        /** enum cv::StereoVar:: */
        public static final int PENALIZATION_TICHONOV = 0, PENALIZATION_CHARBONNIER = 1, PENALIZATION_PERONA_MALIK = 2;

        /** the default constructor */
        public StereoVar() { allocate(); }
        private native void allocate();

        /** the full constructor taking all the necessary algorithm parameters */
        public StereoVar(int levels, double pyrScale, int nIt, int minDisp, int maxDisp, int poly_n, double poly_sigma, float fi, float lambda, int penalization, int cycle, int flags) { allocate(levels, pyrScale, nIt, minDisp, maxDisp, poly_n, poly_sigma, fi, lambda, penalization, cycle, flags); }
        private native void allocate(int levels, double pyrScale, int nIt, int minDisp, int maxDisp, int poly_n, double poly_sigma, float fi, float lambda, int penalization, int cycle, int flags);

        /** the destructor */

        /** the stereo correspondence operator that computes disparity map for the specified rectified stereo pair */
        public native @Name("operator()") void compute(@Const @ByRef Mat left, @Const @ByRef Mat right, @ByRef Mat disp);

        public native int levels(); public native StereoVar levels(int levels);
        public native double pyrScale(); public native StereoVar pyrScale(double pyrScale);
        public native int nIt(); public native StereoVar nIt(int nIt);
        public native int minDisp(); public native StereoVar minDisp(int minDisp);
        public native int maxDisp(); public native StereoVar maxDisp(int maxDisp);
        public native int poly_n(); public native StereoVar poly_n(int poly_n);
        public native double poly_sigma(); public native StereoVar poly_sigma(double poly_sigma);
        public native float fi(); public native StereoVar fi(float fi);
        public native float lambda(); public native StereoVar lambda(float lambda);
        public native int penalization(); public native StereoVar penalization(int penalization);
        public native int cycle(); public native StereoVar cycle(int cycle);
        public native int flags(); public native StereoVar flags(int flags);
    }

    @Namespace("cv") public static native void polyfit(@Const @ByRef Mat srcx, @Const @ByRef Mat srcy, @ByRef Mat dst, int order);

    @Namespace("cv") public static class Directory extends Pointer {
        static { Loader.load(); }
        public Directory() { allocate(); }
        public Directory(int size) { allocateArray(size); }
        public Directory(Pointer p) { super(p); }
        private native void allocate();
        private native void allocateArray(int size);
        @Override public Directory position(int position) {
            return (Directory)super.position(position);
        }
    
            public static native @ByVal StringVector GetListFiles( @StdString BytePointer path, @StdString BytePointer exten/*="*"*/, @Cast("bool") boolean addPath/*=true*/ );
            public static native @ByVal StringVector GetListFiles( @StdString BytePointer path );
            public static native @ByVal StringVector GetListFiles( @StdString String path, @StdString String exten/*="*"*/, @Cast("bool") boolean addPath/*=true*/ );
            public static native @ByVal StringVector GetListFiles( @StdString String path );
            public static native @ByVal StringVector GetListFilesR( @StdString BytePointer path, @StdString BytePointer exten/*="*"*/, @Cast("bool") boolean addPath/*=true*/ );
            public static native @ByVal StringVector GetListFilesR( @StdString BytePointer path );
            public static native @ByVal StringVector GetListFilesR( @StdString String path, @StdString String exten/*="*"*/, @Cast("bool") boolean addPath/*=true*/ );
            public static native @ByVal StringVector GetListFilesR( @StdString String path );
            public static native @ByVal StringVector GetListFolders( @StdString BytePointer path, @StdString BytePointer exten/*="*"*/, @Cast("bool") boolean addPath/*=true*/ );
            public static native @ByVal StringVector GetListFolders( @StdString BytePointer path );
            public static native @ByVal StringVector GetListFolders( @StdString String path, @StdString String exten/*="*"*/, @Cast("bool") boolean addPath/*=true*/ );
            public static native @ByVal StringVector GetListFolders( @StdString String path );
    }

    /*
     * Generation of a set of different colors by the following way:
     * 1) generate more then need colors (in "factor" times) in RGB,
     * 2) convert them to Lab,
     * 3) choose the needed count of colors from the set that are more different from
     *    each other,
     * 4) convert the colors back to RGB
     */
    @Namespace("cv") public static native void generateColors( @StdVector Scalar colors, @Cast("size_t") long count, @Cast("size_t") long factor/*=100*/ );
    @Namespace("cv") public static native void generateColors( @StdVector Scalar colors, @Cast("size_t") long count );


    /*
     *  Estimate the rigid body motion from frame0 to frame1. The method is based on the paper
     *  "Real-Time Visual Odometry from Dense RGB-D Images", F. Steinbucker, J. Strum, D. Cremers, ICCV, 2011.
     */
    /** enum cv:: */
    public static final int ROTATION          = 1,
           TRANSLATION       = 2,
           RIGID_BODY_MOTION = 4;
    @Namespace("cv") public static native @Cast("bool") boolean RGBDOdometry( @ByRef Mat Rt, @Const @ByRef Mat initRt,
                                      @Const @ByRef Mat image0, @Const @ByRef Mat depth0, @Const @ByRef Mat mask0,
                                      @Const @ByRef Mat image1, @Const @ByRef Mat depth1, @Const @ByRef Mat mask1,
                                      @Const @ByRef Mat cameraMatrix, float minDepth/*=0.f*/, float maxDepth/*=4.f*/, float maxDepthDiff/*=0.07f*/,
                                      @StdVector IntPointer iterCounts/*=std::vector<int>()*/,
                                      @StdVector FloatPointer minGradientMagnitudes/*=std::vector<float>()*/,
                                      int transformType/*=RIGID_BODY_MOTION*/ );
    @Namespace("cv") public static native @Cast("bool") boolean RGBDOdometry( @ByRef Mat Rt, @Const @ByRef Mat initRt,
                                      @Const @ByRef Mat image0, @Const @ByRef Mat depth0, @Const @ByRef Mat mask0,
                                      @Const @ByRef Mat image1, @Const @ByRef Mat depth1, @Const @ByRef Mat mask1,
                                      @Const @ByRef Mat cameraMatrix );
    @Namespace("cv") public static native @Cast("bool") boolean RGBDOdometry( @ByRef Mat Rt, @Const @ByRef Mat initRt,
                                      @Const @ByRef Mat image0, @Const @ByRef Mat depth0, @Const @ByRef Mat mask0,
                                      @Const @ByRef Mat image1, @Const @ByRef Mat depth1, @Const @ByRef Mat mask1,
                                      @Const @ByRef Mat cameraMatrix, float minDepth/*=0.f*/, float maxDepth/*=4.f*/, float maxDepthDiff/*=0.07f*/,
                                      @StdVector IntBuffer iterCounts/*=std::vector<int>()*/,
                                      @StdVector FloatBuffer minGradientMagnitudes/*=std::vector<float>()*/,
                                      int transformType/*=RIGID_BODY_MOTION*/ );
    @Namespace("cv") public static native @Cast("bool") boolean RGBDOdometry( @ByRef Mat Rt, @Const @ByRef Mat initRt,
                                      @Const @ByRef Mat image0, @Const @ByRef Mat depth0, @Const @ByRef Mat mask0,
                                      @Const @ByRef Mat image1, @Const @ByRef Mat depth1, @Const @ByRef Mat mask1,
                                      @Const @ByRef Mat cameraMatrix, float minDepth/*=0.f*/, float maxDepth/*=4.f*/, float maxDepthDiff/*=0.07f*/,
                                      @StdVector int[] iterCounts/*=std::vector<int>()*/,
                                      @StdVector float[] minGradientMagnitudes/*=std::vector<float>()*/,
                                      int transformType/*=RIGID_BODY_MOTION*/ );

    /**
    *Bilinear interpolation technique.
    *
    *The value of a desired cortical pixel is obtained through a bilinear interpolation of the values
    *of the four nearest neighbouring Cartesian pixels to the center of the RF.
    *The same principle is applied to the inverse transformation.
    *
    *More details can be found in http://dx.doi.org/10.1007/978-3-642-23968-7_5
    */
    @Namespace("cv") @NoOffset public static class LogPolar_Interp extends Pointer {
        static { Loader.load(); }
        public LogPolar_Interp(Pointer p) { super(p); }
        public LogPolar_Interp(int size) { allocateArray(size); }
        private native void allocateArray(int size);
        @Override public LogPolar_Interp position(int position) {
            return (LogPolar_Interp)super.position(position);
        }
    

        public LogPolar_Interp() { allocate(); }
        private native void allocate();

        /**
        *Constructor
        *\param w the width of the input image
        *\param h the height of the input image
        *\param center the transformation center: where the output precision is maximal
        *\param R the number of rings of the cortical image (default value 70 pixel)
        *\param ro0 the radius of the blind spot (default value 3 pixel)
        *\param interp interpolation algorithm
        *\param full \a 1 (default value) means that the retinal image (the inverse transform) is computed within the circumscribing circle.
        *            \a 0 means that the retinal image is computed within the inscribed circle.
        *\param S the number of sectors of the cortical image (default value 70 pixel).
        *         Its value is usually internally computed to obtain a pixel aspect ratio equals to 1.
        *\param sp \a 1 (default value) means that the parameter \a S is internally computed.
        *          \a 0 means that the parameter \a S is provided by the user.
        */
        public LogPolar_Interp(int w, int h, @ByVal @Cast("cv::Point2i*") Point center, int R/*=70*/, double ro0/*=3.0*/,
                                int interp/*=INTER_LINEAR*/, int full/*=1*/, int S/*=117*/, int sp/*=1*/) { allocate(w, h, center, R, ro0, interp, full, S, sp); }
        private native void allocate(int w, int h, @ByVal @Cast("cv::Point2i*") Point center, int R/*=70*/, double ro0/*=3.0*/,
                                int interp/*=INTER_LINEAR*/, int full/*=1*/, int S/*=117*/, int sp/*=1*/);
        public LogPolar_Interp(int w, int h, @ByVal @Cast("cv::Point2i*") Point center) { allocate(w, h, center); }
        private native void allocate(int w, int h, @ByVal @Cast("cv::Point2i*") Point center);
        /**
        *Transformation from Cartesian image to cortical (log-polar) image.
        *\param source the Cartesian image
        *\return the transformed image (cortical image)
        */
        public native @Const @ByVal Mat to_cortical(@Const @ByRef Mat source);
        /**
        *Transformation from cortical image to retinal (inverse log-polar) image.
        *\param source the cortical image
        *\return the transformed image (retinal image)
        */
        public native @Const @ByVal Mat to_cartesian(@Const @ByRef Mat source);
        /**
        *Destructor
        */
    }

    /**
    *Overlapping circular receptive fields technique
    *
    *The Cartesian plane is divided in two regions: the fovea and the periphery.
    *The fovea (oversampling) is handled by using the bilinear interpolation technique described above, whereas in
    *the periphery we use the overlapping Gaussian circular RFs.
    *
    *More details can be found in http://dx.doi.org/10.1007/978-3-642-23968-7_5
    */
    @Namespace("cv") @NoOffset public static class LogPolar_Overlapping extends Pointer {
        static { Loader.load(); }
        public LogPolar_Overlapping(Pointer p) { super(p); }
        public LogPolar_Overlapping(int size) { allocateArray(size); }
        private native void allocateArray(int size);
        @Override public LogPolar_Overlapping position(int position) {
            return (LogPolar_Overlapping)super.position(position);
        }
    
        public LogPolar_Overlapping() { allocate(); }
        private native void allocate();

        /**
        *Constructor
        *\param w the width of the input image
        *\param h the height of the input image
        *\param center the transformation center: where the output precision is maximal
        *\param R the number of rings of the cortical image (default value 70 pixel)
        *\param ro0 the radius of the blind spot (default value 3 pixel)
        *\param full \a 1 (default value) means that the retinal image (the inverse transform) is computed within the circumscribing circle.
        *            \a 0 means that the retinal image is computed within the inscribed circle.
        *\param S the number of sectors of the cortical image (default value 70 pixel).
        *         Its value is usually internally computed to obtain a pixel aspect ratio equals to 1.
        *\param sp \a 1 (default value) means that the parameter \a S is internally computed.
        *          \a 0 means that the parameter \a S is provided by the user.
        */
        public LogPolar_Overlapping(int w, int h, @ByVal @Cast("cv::Point2i*") Point center, int R/*=70*/,
                                     double ro0/*=3.0*/, int full/*=1*/, int S/*=117*/, int sp/*=1*/) { allocate(w, h, center, R, ro0, full, S, sp); }
        private native void allocate(int w, int h, @ByVal @Cast("cv::Point2i*") Point center, int R/*=70*/,
                                     double ro0/*=3.0*/, int full/*=1*/, int S/*=117*/, int sp/*=1*/);
        public LogPolar_Overlapping(int w, int h, @ByVal @Cast("cv::Point2i*") Point center) { allocate(w, h, center); }
        private native void allocate(int w, int h, @ByVal @Cast("cv::Point2i*") Point center);
        /**
        *Transformation from Cartesian image to cortical (log-polar) image.
        *\param source the Cartesian image
        *\return the transformed image (cortical image)
        */
        public native @Const @ByVal Mat to_cortical(@Const @ByRef Mat source);
        /**
        *Transformation from cortical image to retinal (inverse log-polar) image.
        *\param source the cortical image
        *\return the transformed image (retinal image)
        */
        public native @Const @ByVal Mat to_cartesian(@Const @ByRef Mat source);
        /**
        *Destructor
        */
    }

    /**
    * Adjacent receptive fields technique
    *
    *All the Cartesian pixels, whose coordinates in the cortical domain share the same integer part, are assigned to the same RF.
    *The precision of the boundaries of the RF can be improved by breaking each pixel into subpixels and assigning each of them to the correct RF.
    *This technique is implemented from: Traver, V., Pla, F.: Log-polar mapping template design: From task-level requirements
    *to geometry parameters. Image Vision Comput. 26(10) (2008) 1354-1370
    *
    *More details can be found in http://dx.doi.org/10.1007/978-3-642-23968-7_5
    */
    @Namespace("cv") @NoOffset public static class LogPolar_Adjacent extends Pointer {
        static { Loader.load(); }
        public LogPolar_Adjacent(Pointer p) { super(p); }
        public LogPolar_Adjacent(int size) { allocateArray(size); }
        private native void allocateArray(int size);
        @Override public LogPolar_Adjacent position(int position) {
            return (LogPolar_Adjacent)super.position(position);
        }
    
        public LogPolar_Adjacent() { allocate(); }
        private native void allocate();

        /**
         *Constructor
         *\param w the width of the input image
         *\param h the height of the input image
         *\param center the transformation center: where the output precision is maximal
         *\param R the number of rings of the cortical image (default value 70 pixel)
         *\param ro0 the radius of the blind spot (default value 3 pixel)
         *\param smin the size of the subpixel (default value 0.25 pixel)
         *\param full \a 1 (default value) means that the retinal image (the inverse transform) is computed within the circumscribing circle.
         *            \a 0 means that the retinal image is computed within the inscribed circle.
         *\param S the number of sectors of the cortical image (default value 70 pixel).
         *         Its value is usually internally computed to obtain a pixel aspect ratio equals to 1.
         *\param sp \a 1 (default value) means that the parameter \a S is internally computed.
         *          \a 0 means that the parameter \a S is provided by the user.
         */
        public LogPolar_Adjacent(int w, int h, @ByVal @Cast("cv::Point2i*") Point center, int R/*=70*/, double ro0/*=3.0*/, double smin/*=0.25*/, int full/*=1*/, int S/*=117*/, int sp/*=1*/) { allocate(w, h, center, R, ro0, smin, full, S, sp); }
        private native void allocate(int w, int h, @ByVal @Cast("cv::Point2i*") Point center, int R/*=70*/, double ro0/*=3.0*/, double smin/*=0.25*/, int full/*=1*/, int S/*=117*/, int sp/*=1*/);
        public LogPolar_Adjacent(int w, int h, @ByVal @Cast("cv::Point2i*") Point center) { allocate(w, h, center); }
        private native void allocate(int w, int h, @ByVal @Cast("cv::Point2i*") Point center);
        /**
         *Transformation from Cartesian image to cortical (log-polar) image.
         *\param source the Cartesian image
         *\return the transformed image (cortical image)
         */
        public native @Const @ByVal Mat to_cortical(@Const @ByRef Mat source);
        /**
         *Transformation from cortical image to retinal (inverse log-polar) image.
         *\param source the cortical image
         *\return the transformed image (retinal image)
         */
        public native @Const @ByVal Mat to_cartesian(@Const @ByRef Mat source);
        /**
         *Destructor
         */
    }

    @Namespace("cv") public static native @ByVal Mat subspaceProject(@ByVal Mat W, @ByVal Mat mean, @ByVal Mat src);
    @Namespace("cv") public static native @ByVal Mat subspaceReconstruct(@ByVal Mat W, @ByVal Mat mean, @ByVal Mat src);

    @Namespace("cv") @NoOffset public static class LDA extends Pointer {
        static { Loader.load(); }
        public LDA(Pointer p) { super(p); }
    
        // Initializes a LDA with num_components (default 0) and specifies how
        // samples are aligned (default dataAsRow=true).
        public LDA(int num_components/*=0*/) { allocate(num_components); }
        private native void allocate(int num_components/*=0*/);
        public LDA() { allocate(); }
        private native void allocate();

        // Initializes and performs a Discriminant Analysis with Fisher's
        // Optimization Criterion on given data in src and corresponding labels
        // in labels. If 0 (or less) number of components are given, they are
        // automatically determined for given data in computation.
        public LDA(@Const @ByRef Mat src, @StdVector IntPointer labels,
                        int num_components/*=0*/) { allocate(src, labels, num_components); }
        private native void allocate(@Const @ByRef Mat src, @StdVector IntPointer labels,
                        int num_components/*=0*/);
        public LDA(@Const @ByRef Mat src, @StdVector IntPointer labels) { allocate(src, labels); }
        private native void allocate(@Const @ByRef Mat src, @StdVector IntPointer labels);
        public LDA(@Const @ByRef Mat src, @StdVector IntBuffer labels,
                        int num_components/*=0*/) { allocate(src, labels, num_components); }
        private native void allocate(@Const @ByRef Mat src, @StdVector IntBuffer labels,
                        int num_components/*=0*/);
        public LDA(@Const @ByRef Mat src, @StdVector IntBuffer labels) { allocate(src, labels); }
        private native void allocate(@Const @ByRef Mat src, @StdVector IntBuffer labels);
        public LDA(@Const @ByRef Mat src, @StdVector int[] labels,
                        int num_components/*=0*/) { allocate(src, labels, num_components); }
        private native void allocate(@Const @ByRef Mat src, @StdVector int[] labels,
                        int num_components/*=0*/);
        public LDA(@Const @ByRef Mat src, @StdVector int[] labels) { allocate(src, labels); }
        private native void allocate(@Const @ByRef Mat src, @StdVector int[] labels);

        // Initializes and performs a Discriminant Analysis with Fisher's
        // Optimization Criterion on given data in src and corresponding labels
        // in labels. If 0 (or less) number of components are given, they are
        // automatically determined for given data in computation.
        public LDA(@ByVal MatVector src, @ByVal Mat labels,
                        int num_components/*=0*/) { allocate(src, labels, num_components); }
        private native void allocate(@ByVal MatVector src, @ByVal Mat labels,
                        int num_components/*=0*/);
        public LDA(@ByVal MatVector src, @ByVal Mat labels) { allocate(src, labels); }
        private native void allocate(@ByVal MatVector src, @ByVal Mat labels);

        // Serializes this object to a given filename.
        public native void save(@StdString BytePointer filename);
        public native void save(@StdString String filename);

        // Deserializes this object from a given filename.
        public native void load(@StdString BytePointer filename);
        public native void load(@StdString String filename);

        // Serializes this object to a given cv::FileStorage.
        public native void save(@ByRef FileStorage fs);

            // Deserializes this object from a given cv::FileStorage.
        public native void load(@Const @ByRef FileStorage node);

        // Destructor.

        /** Compute the discriminants for data in src and labels. */
        public native void compute(@ByVal MatVector src, @ByVal Mat labels);

        // Projects samples into the LDA subspace.
        public native @ByVal Mat project(@ByVal Mat src);

        // Reconstructs projections from the LDA subspace.
        public native @ByVal Mat reconstruct(@ByVal Mat src);

        // Returns the eigenvectors of this LDA.
        public native @ByVal Mat eigenvectors();

        // Returns the eigenvalues of this LDA.
        public native @ByVal Mat eigenvalues();
    }

    @Namespace("cv") public static class FaceRecognizer extends Algorithm {
        static { Loader.load(); }
        public FaceRecognizer() { }
        public FaceRecognizer(Pointer p) { super(p); }
    
        /** virtual destructor */

        // Trains a FaceRecognizer.
        public native void train(@ByVal MatVector src, @ByVal Mat labels);

        // Updates a FaceRecognizer.
        public native void update(@ByVal MatVector src, @ByVal Mat labels);

        // Gets a prediction from a FaceRecognizer.
        public native int predict(@ByVal Mat src);

        // Predicts the label and confidence for a given sample.
        public native void predict(@ByVal Mat src, @ByRef IntPointer label, @ByRef DoublePointer confidence);
        public native void predict(@ByVal Mat src, @ByRef IntBuffer label, @ByRef DoubleBuffer confidence);
        public native void predict(@ByVal Mat src, @ByRef int[] label, @ByRef double[] confidence);

        // Serializes this object to a given filename.
        public native void save(@StdString BytePointer filename);
        public native void save(@StdString String filename);

        // Deserializes this object from a given filename.
        public native void load(@StdString BytePointer filename);
        public native void load(@StdString String filename);

        // Serializes this object to a given cv::FileStorage.
        public native void save(@ByRef FileStorage fs);

        // Deserializes this object from a given cv::FileStorage.
        public native void load(@Const @ByRef FileStorage fs);

        // Sets additional information as pairs label - info.
        public native void setLabelsInfo(@Const @ByRef IntStringMap labelsInfo);

        // Gets string information by label
        public native @StdString BytePointer getLabelInfo(int label);

        // Gets labels by string
        public native @StdVector IntPointer getLabelsByString(@StdString BytePointer str);
        public native @StdVector IntBuffer getLabelsByString(@StdString String str);
    }

    @Namespace("cv") public static native @Ptr FaceRecognizer createEigenFaceRecognizer(int num_components/*=0*/, double threshold/*=DBL_MAX*/);
    @Namespace("cv") public static native @Ptr FaceRecognizer createEigenFaceRecognizer();
    @Namespace("cv") public static native @Ptr FaceRecognizer createFisherFaceRecognizer(int num_components/*=0*/, double threshold/*=DBL_MAX*/);
    @Namespace("cv") public static native @Ptr FaceRecognizer createFisherFaceRecognizer();
    @Namespace("cv") public static native @Ptr FaceRecognizer createLBPHFaceRecognizer(int radius/*=1*/, int neighbors/*=8*/,
                                                                int grid_x/*=8*/, int grid_y/*=8*/, double threshold/*=DBL_MAX*/);
    @Namespace("cv") public static native @Ptr FaceRecognizer createLBPHFaceRecognizer();

    /** enum cv:: */
    public static final int
        COLORMAP_AUTUMN = 0,
        COLORMAP_BONE = 1,
        COLORMAP_JET = 2,
        COLORMAP_WINTER = 3,
        COLORMAP_RAINBOW = 4,
        COLORMAP_OCEAN = 5,
        COLORMAP_SUMMER = 6,
        COLORMAP_SPRING = 7,
        COLORMAP_COOL = 8,
        COLORMAP_HSV = 9,
        COLORMAP_PINK = 10,
        COLORMAP_HOT = 11;

    @Namespace("cv") public static native void applyColorMap(@ByVal Mat src, @ByVal Mat dst, int colormap);

    @Namespace("cv") public static native @Cast("bool") boolean initModule_contrib();


// #include "opencv2/contrib/retina.hpp"

// #include "opencv2/contrib/openfabmap.hpp"

// #endif

// #endif


// Parsed from <opencv2/contrib/detection_based_tracker.hpp>

// #pragma once

// #if defined(__linux__) || defined(LINUX) || defined(__APPLE__) || defined(ANDROID)

// #include <opencv2/core/core.hpp>
// #include <opencv2/objdetect/objdetect.hpp>

// #include <vector>

@NoOffset public static class DetectionBasedTracker extends Pointer {
    static { Loader.load(); }
    public DetectionBasedTracker() { }
    public DetectionBasedTracker(Pointer p) { super(p); }

        @NoOffset public static class Parameters extends Pointer {
            static { Loader.load(); }
            public Parameters(Pointer p) { super(p); }
            public Parameters(int size) { allocateArray(size); }
            private native void allocateArray(int size);
            @Override public Parameters position(int position) {
                return (Parameters)super.position(position);
            }
        
            public native int minObjectSize(); public native Parameters minObjectSize(int minObjectSize);
            public native int maxObjectSize(); public native Parameters maxObjectSize(int maxObjectSize);
            public native double scaleFactor(); public native Parameters scaleFactor(double scaleFactor);
            public native int maxTrackLifetime(); public native Parameters maxTrackLifetime(int maxTrackLifetime);
            public native int minNeighbors(); public native Parameters minNeighbors(int minNeighbors);
            public native int minDetectionPeriod(); public native Parameters minDetectionPeriod(int minDetectionPeriod); //the minimal time between run of the big object detector (on the whole frame) in ms (1000 mean 1 sec), default=0

            public Parameters() { allocate(); }
            private native void allocate();
        }

        public DetectionBasedTracker(@StdString BytePointer cascadeFilename, @Const @ByRef Parameters params) { allocate(cascadeFilename, params); }
        private native void allocate(@StdString BytePointer cascadeFilename, @Const @ByRef Parameters params);
        public DetectionBasedTracker(@StdString String cascadeFilename, @Const @ByRef Parameters params) { allocate(cascadeFilename, params); }
        private native void allocate(@StdString String cascadeFilename, @Const @ByRef Parameters params);

        public native @Cast("bool") boolean run();
        public native void stop();
        public native void resetTracking();

        public native void process(@Const @ByRef Mat imageGray);

        public native @Cast("bool") boolean setParameters(@Const @ByRef Parameters params);
        public native @Const @ByRef Parameters getParameters();
        public native void getObjects(@StdVector Rect result);
        public native void getObjects(@ByRef RectIntPairVector result);
}
 //end of cv namespace

// #endif


// Parsed from <opencv2/contrib/hybridtracker.hpp>

//*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                                License Agreement
//                       For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2008-2011, Willow Garage Inc., all rights reserved.
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

// #ifndef __OPENCV_HYBRIDTRACKER_H_
// #define __OPENCV_HYBRIDTRACKER_H_

// #include "opencv2/core/core.hpp"
// #include "opencv2/core/operations.hpp"
// #include "opencv2/imgproc/imgproc.hpp"
// #include "opencv2/features2d/features2d.hpp"
// #include "opencv2/video/tracking.hpp"
// #include "opencv2/ml/ml.hpp"

// #ifdef __cplusplus

// Motion model for tracking algorithm. Currently supports objects that do not move much.
// To add Kalman filter
@Namespace("cv") @NoOffset public static class CvMotionModel extends Pointer {
    static { Loader.load(); }
    public CvMotionModel(Pointer p) { super(p); }
    public CvMotionModel(int size) { allocateArray(size); }
    private native void allocateArray(int size);
    @Override public CvMotionModel position(int position) {
        return (CvMotionModel)super.position(position);
    }

    /** enum cv::CvMotionModel:: */
    public static final int LOW_PASS_FILTER = 0, KALMAN_FILTER = 1, EM = 2;

    public CvMotionModel() { allocate(); }
    private native void allocate();

    public native float low_pass_gain(); public native CvMotionModel low_pass_gain(float low_pass_gain);    // low pass gain
}

// Mean Shift Tracker parameters for specifying use of HSV channel and CamShift parameters.
@Namespace("cv") @NoOffset public static class CvMeanShiftTrackerParams extends Pointer {
    static { Loader.load(); }
    public CvMeanShiftTrackerParams(Pointer p) { super(p); }
    public CvMeanShiftTrackerParams(int size) { allocateArray(size); }
    private native void allocateArray(int size);
    @Override public CvMeanShiftTrackerParams position(int position) {
        return (CvMeanShiftTrackerParams)super.position(position);
    }

    /** enum cv::CvMeanShiftTrackerParams:: */
    public static final int  H = 0, HS = 1, HSV = 2;
    public CvMeanShiftTrackerParams(int tracking_type/*=CvMeanShiftTrackerParams::HS*/,
                @ByVal CvTermCriteria term_crit/*=CvTermCriteria()*/) { allocate(tracking_type, term_crit); }
    private native void allocate(int tracking_type/*=CvMeanShiftTrackerParams::HS*/,
                @ByVal CvTermCriteria term_crit/*=CvTermCriteria()*/);
    public CvMeanShiftTrackerParams() { allocate(); }
    private native void allocate();

    public native int tracking_type(); public native CvMeanShiftTrackerParams tracking_type(int tracking_type);
    public native @StdVector FloatPointer h_range(); public native CvMeanShiftTrackerParams h_range(FloatPointer h_range);
    public native @StdVector FloatPointer s_range(); public native CvMeanShiftTrackerParams s_range(FloatPointer s_range);
    public native @StdVector FloatPointer v_range(); public native CvMeanShiftTrackerParams v_range(FloatPointer v_range);
    public native @ByRef CvTermCriteria term_crit(); public native CvMeanShiftTrackerParams term_crit(CvTermCriteria term_crit);
}

// Feature tracking parameters
@Namespace("cv") @NoOffset public static class CvFeatureTrackerParams extends Pointer {
    static { Loader.load(); }
    public CvFeatureTrackerParams(Pointer p) { super(p); }
    public CvFeatureTrackerParams(int size) { allocateArray(size); }
    private native void allocateArray(int size);
    @Override public CvFeatureTrackerParams position(int position) {
        return (CvFeatureTrackerParams)super.position(position);
    }

    /** enum cv::CvFeatureTrackerParams:: */
    public static final int  SIFT = 0, SURF = 1, OPTICAL_FLOW = 2;
    public CvFeatureTrackerParams(int featureType/*=0*/, int windowSize/*=0*/) { allocate(featureType, windowSize); }
    private native void allocate(int featureType/*=0*/, int windowSize/*=0*/);
    public CvFeatureTrackerParams() { allocate(); }
    private native void allocate();

    public native int feature_type(); public native CvFeatureTrackerParams feature_type(int feature_type); // Feature type to use
    public native int window_size(); public native CvFeatureTrackerParams window_size(int window_size); // Window size in pixels around which to search for new window
}

// Hybrid Tracking parameters for specifying weights of individual trackers and motion model.
@Namespace("cv") @NoOffset public static class CvHybridTrackerParams extends Pointer {
    static { Loader.load(); }
    public CvHybridTrackerParams(Pointer p) { super(p); }
    public CvHybridTrackerParams(int size) { allocateArray(size); }
    private native void allocateArray(int size);
    @Override public CvHybridTrackerParams position(int position) {
        return (CvHybridTrackerParams)super.position(position);
    }

    public CvHybridTrackerParams(float ft_tracker_weight/*=0.5*/, float ms_tracker_weight/*=0.5*/,
                @ByVal CvFeatureTrackerParams ft_params/*=CvFeatureTrackerParams()*/,
                @ByVal CvMeanShiftTrackerParams ms_params/*=CvMeanShiftTrackerParams()*/,
                @ByVal CvMotionModel model/*=CvMotionModel()*/) { allocate(ft_tracker_weight, ms_tracker_weight, ft_params, ms_params, model); }
    private native void allocate(float ft_tracker_weight/*=0.5*/, float ms_tracker_weight/*=0.5*/,
                @ByVal CvFeatureTrackerParams ft_params/*=CvFeatureTrackerParams()*/,
                @ByVal CvMeanShiftTrackerParams ms_params/*=CvMeanShiftTrackerParams()*/,
                @ByVal CvMotionModel model/*=CvMotionModel()*/);
    public CvHybridTrackerParams() { allocate(); }
    private native void allocate();

    public native float ft_tracker_weight(); public native CvHybridTrackerParams ft_tracker_weight(float ft_tracker_weight);
    public native float ms_tracker_weight(); public native CvHybridTrackerParams ms_tracker_weight(float ms_tracker_weight);
    public native @ByRef CvFeatureTrackerParams ft_params(); public native CvHybridTrackerParams ft_params(CvFeatureTrackerParams ft_params);
    public native @ByRef CvMeanShiftTrackerParams ms_params(); public native CvHybridTrackerParams ms_params(CvMeanShiftTrackerParams ms_params);
    public native int motion_model(); public native CvHybridTrackerParams motion_model(int motion_model);
    public native float low_pass_gain(); public native CvHybridTrackerParams low_pass_gain(float low_pass_gain);
}

// Performs Camshift using parameters from MeanShiftTrackerParams
@Namespace("cv") @NoOffset public static class CvMeanShiftTracker extends Pointer {
    static { Loader.load(); }
    public CvMeanShiftTracker() { }
    public CvMeanShiftTracker(Pointer p) { super(p); }

    public native @ByRef CvMeanShiftTrackerParams params(); public native CvMeanShiftTracker params(CvMeanShiftTrackerParams params);

    
    public CvMeanShiftTracker(@ByVal CvMeanShiftTrackerParams _params) { allocate(_params); }
    private native void allocate(@ByVal CvMeanShiftTrackerParams _params);
    
    public native void newTrackingWindow(@ByVal Mat image, @ByVal Rect selection);
    public native @ByVal RotatedRect updateTrackingWindow(@ByVal Mat image);
    public native @ByVal Mat getHistogramProjection(int type);
    public native void setTrackingWindow(@ByVal Rect _window);
    public native @ByVal Rect getTrackingWindow();
    public native @ByVal RotatedRect getTrackingEllipse();
    public native @ByVal Point2f getTrackingCenter();
}

// Performs SIFT/SURF feature tracking using parameters from FeatureTrackerParams
@Namespace("cv") @NoOffset public static class CvFeatureTracker extends Pointer {
    static { Loader.load(); }
    public CvFeatureTracker() { }
    public CvFeatureTracker(Pointer p) { super(p); }

    public native @ByRef Mat disp_matches(); public native CvFeatureTracker disp_matches(Mat disp_matches);
    public native @ByRef CvFeatureTrackerParams params(); public native CvFeatureTracker params(CvFeatureTrackerParams params);

    
    public CvFeatureTracker(@ByVal CvFeatureTrackerParams params) { allocate(params); }
    private native void allocate(@ByVal CvFeatureTrackerParams params);
    
    public native void newTrackingWindow(@ByVal Mat image, @ByVal Rect selection);
    public native @ByVal Rect updateTrackingWindow(@ByVal Mat image);
    public native @ByVal Rect updateTrackingWindowWithSIFT(@ByVal Mat image);
    public native @ByVal Rect updateTrackingWindowWithFlow(@ByVal Mat image);
    public native void setTrackingWindow(@ByVal Rect _window);
    public native @ByVal Rect getTrackingWindow();
    public native @ByVal Point2f getTrackingCenter();
}

// Performs Hybrid Tracking and combines individual trackers using EM or filters
@Namespace("cv") @NoOffset public static class CvHybridTracker extends Pointer {
    static { Loader.load(); }
    public CvHybridTracker(Pointer p) { super(p); }
    public CvHybridTracker(int size) { allocateArray(size); }
    private native void allocateArray(int size);
    @Override public CvHybridTracker position(int position) {
        return (CvHybridTracker)super.position(position);
    }

    public native @ByRef CvHybridTrackerParams params(); public native CvHybridTracker params(CvHybridTrackerParams params);
    public CvHybridTracker() { allocate(); }
    private native void allocate();
    public CvHybridTracker(@ByVal CvHybridTrackerParams params) { allocate(params); }
    private native void allocate(@ByVal CvHybridTrackerParams params);

    public native void newTracker(@ByVal Mat image, @ByVal Rect selection);
    public native void updateTracker(@ByVal Mat image);
    public native @ByVal Rect getTrackingWindow();
}


// #endif

// #endif


// Parsed from <opencv2/contrib/retina.hpp>

/*#******************************************************************************
 ** IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
 **
 ** By downloading, copying, installing or using the software you agree to this license.
 ** If you do not agree to this license, do not download, install,
 ** copy or use the software.
 **
 **
 ** HVStools : interfaces allowing OpenCV users to integrate Human Vision System models. Presented models originate from Jeanny Herault's original research and have been reused and adapted by the author&collaborators for computed vision applications since his thesis with Alice Caplier at Gipsa-Lab.
 ** Use: extract still images & image sequences features, from contours details to motion spatio-temporal features, etc. for high level visual scene analysis. Also contribute to image enhancement/compression such as tone mapping.
 **
 ** Maintainers : Listic lab (code author current affiliation & applications) and Gipsa Lab (original research origins & applications)
 **
 **  Creation - enhancement process 2007-2011
 **      Author: Alexandre Benoit (benoit.alexandre.vision@gmail.com), LISTIC lab, Annecy le vieux, France
 **
 ** Theses algorithm have been developped by Alexandre BENOIT since his thesis with Alice Caplier at Gipsa-Lab (www.gipsa-lab.inpg.fr) and the research he pursues at LISTIC Lab (www.listic.univ-savoie.fr).
 ** Refer to the following research paper for more information:
 ** Benoit A., Caplier A., Durette B., Herault, J., "USING HUMAN VISUAL SYSTEM MODELING FOR BIO-INSPIRED LOW LEVEL IMAGE PROCESSING", Elsevier, Computer Vision and Image Understanding 114 (2010), pp. 758-773, DOI: http://dx.doi.org/10.1016/j.cviu.2010.01.011
 ** This work have been carried out thanks to Jeanny Herault who's research and great discussions are the basis of all this work, please take a look at his book:
 ** Vision: Images, Signals and Neural Networks: Models of Neural Processing in Visual Perception (Progress in Neural Processing),By: Jeanny Herault, ISBN: 9814273686. WAPI (Tower ID): 113266891.
 **
 ** The retina filter includes the research contributions of phd/research collegues from which code has been redrawn by the author :
 ** _take a look at the retinacolor.hpp module to discover Brice Chaix de Lavarene color mosaicing/demosaicing and the reference paper:
 ** ====> B. Chaix de Lavarene, D. Alleysson, B. Durette, J. Herault (2007). "Efficient demosaicing through recursive filtering", IEEE International Conference on Image Processing ICIP 2007
 ** _take a look at imagelogpolprojection.hpp to discover retina spatial log sampling which originates from Barthelemy Durette phd with Jeanny Herault. A Retina / V1 cortex projection is also proposed and originates from Jeanny's discussions.
 ** ====> more informations in the above cited Jeanny Heraults's book.
 **
 **                          License Agreement
 **               For Open Source Computer Vision Library
 **
 ** Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
 ** Copyright (C) 2008-2011, Willow Garage Inc., all rights reserved.
 **
 **               For Human Visual System tools (hvstools)
 ** Copyright (C) 2007-2011, LISTIC Lab, Annecy le Vieux and GIPSA Lab, Grenoble, France, all rights reserved.
 **
 ** Third party copyrights are property of their respective owners.
 **
 ** Redistribution and use in source and binary forms, with or without modification,
 ** are permitted provided that the following conditions are met:
 **
 ** * Redistributions of source code must retain the above copyright notice,
 **    this list of conditions and the following disclaimer.
 **
 ** * Redistributions in binary form must reproduce the above copyright notice,
 **    this list of conditions and the following disclaimer in the documentation
 **    and/or other materials provided with the distribution.
 **
 ** * The name of the copyright holders may not be used to endorse or promote products
 **    derived from this software without specific prior written permission.
 **
 ** This software is provided by the copyright holders and contributors "as is" and
 ** any express or implied warranties, including, but not limited to, the implied
 ** warranties of merchantability and fitness for a particular purpose are disclaimed.
 ** In no event shall the Intel Corporation or contributors be liable for any direct,
 ** indirect, incidental, special, exemplary, or consequential damages
 ** (including, but not limited to, procurement of substitute goods or services;
 ** loss of use, data, or profits; or business interruption) however caused
 ** and on any theory of liability, whether in contract, strict liability,
 ** or tort (including negligence or otherwise) arising in any way out of
 ** the use of this software, even if advised of the possibility of such damage.
 *******************************************************************************/

// #ifndef __OPENCV_CONTRIB_RETINA_HPP__
// #define __OPENCV_CONTRIB_RETINA_HPP__

/*
 * Retina.hpp
 *
 *  Created on: Jul 19, 2011
 *      Author: Alexandre Benoit
 */

// #include "opencv2/core/core.hpp" // for all OpenCV core functionalities access, including cv::Exception support
// #include <valarray>

/** enum cv::RETINA_COLORSAMPLINGMETHOD */
public static final int
    /** each pixel position is either R, G or B in a random choice */
    RETINA_COLOR_RANDOM = 0,
    /** color sampling is RGBRGBRGB..., line 2 BRGBRGBRG..., line 3, GBRGBRGBR... */
    RETINA_COLOR_DIAGONAL = 1,
    /** standard bayer sampling */
    RETINA_COLOR_BAYER = 2;

@Namespace("cv") @Opaque public static class RetinaFilter extends Pointer {
    public RetinaFilter() { }
    public RetinaFilter(Pointer p) { super(p); }
}

/**
 * a wrapper class which allows the Gipsa/Listic Labs model to be used.
 * This retina model allows spatio-temporal image processing (applied on still images, video sequences).
 * As a summary, these are the retina model properties:
 * => It applies a spectral whithening (mid-frequency details enhancement)
 * => high frequency spatio-temporal noise reduction
 * => low frequency luminance to be reduced (luminance range compression)
 * => local logarithmic luminance compression allows details to be enhanced in low light conditions
 *
 * USE : this model can be used basically for spatio-temporal video effects but also for :
 *      _using the getParvo method output matrix : texture analysiswith enhanced signal to noise ratio and enhanced details robust against input images luminance ranges
 *      _using the getMagno method output matrix : motion analysis also with the previously cited properties
 *
 * for more information, reer to the following papers :
 * Benoit A., Caplier A., Durette B., Herault, J., "USING HUMAN VISUAL SYSTEM MODELING FOR BIO-INSPIRED LOW LEVEL IMAGE PROCESSING", Elsevier, Computer Vision and Image Understanding 114 (2010), pp. 758-773, DOI: http://dx.doi.org/10.1016/j.cviu.2010.01.011
 * Vision: Images, Signals and Neural Networks: Models of Neural Processing in Visual Perception (Progress in Neural Processing),By: Jeanny Herault, ISBN: 9814273686. WAPI (Tower ID): 113266891.
 *
 * The retina filter includes the research contributions of phd/research collegues from which code has been redrawn by the author :
 * _take a look at the retinacolor.hpp module to discover Brice Chaix de Lavarene color mosaicing/demosaicing and the reference paper:
 * ====> B. Chaix de Lavarene, D. Alleysson, B. Durette, J. Herault (2007). "Efficient demosaicing through recursive filtering", IEEE International Conference on Image Processing ICIP 2007
 * _take a look at imagelogpolprojection.hpp to discover retina spatial log sampling which originates from Barthelemy Durette phd with Jeanny Herault. A Retina / V1 cortex projection is also proposed and originates from Jeanny's discussions.
 * ====> more informations in the above cited Jeanny Heraults's book.
 */
@Namespace("cv") @NoOffset public static class Retina extends Pointer {
    static { Loader.load(); }
    public Retina() { }
    public Retina(Pointer p) { super(p); }


    // parameters structure for better clarity, check explenations on the comments of methods : setupOPLandIPLParvoChannel and setupIPLMagnoChannel
    public static class RetinaParameters extends Pointer {
        static { Loader.load(); }
        public RetinaParameters() { allocate(); }
        public RetinaParameters(int size) { allocateArray(size); }
        public RetinaParameters(Pointer p) { super(p); }
        private native void allocate();
        private native void allocateArray(int size);
        @Override public RetinaParameters position(int position) {
            return (RetinaParameters)super.position(position);
        }
    
        @NoOffset public static class OPLandIplParvoParameters extends Pointer {
            static { Loader.load(); }
            public OPLandIplParvoParameters(Pointer p) { super(p); }
            public OPLandIplParvoParameters(int size) { allocateArray(size); }
            private native void allocateArray(int size);
            @Override public OPLandIplParvoParameters position(int position) {
                return (OPLandIplParvoParameters)super.position(position);
            }
         // Outer Plexiform Layer (OPL) and Inner Plexiform Layer Parvocellular (IplParvo) parameters
               public OPLandIplParvoParameters() { allocate(); }
               private native void allocate();// default setup
               public native @Cast("bool") boolean colorMode(); public native OPLandIplParvoParameters colorMode(boolean colorMode);
               public native @Cast("bool") boolean normaliseOutput(); public native OPLandIplParvoParameters normaliseOutput(boolean normaliseOutput);
               public native float photoreceptorsLocalAdaptationSensitivity(); public native OPLandIplParvoParameters photoreceptorsLocalAdaptationSensitivity(float photoreceptorsLocalAdaptationSensitivity);
               public native float photoreceptorsTemporalConstant(); public native OPLandIplParvoParameters photoreceptorsTemporalConstant(float photoreceptorsTemporalConstant);
               public native float photoreceptorsSpatialConstant(); public native OPLandIplParvoParameters photoreceptorsSpatialConstant(float photoreceptorsSpatialConstant);
               public native float horizontalCellsGain(); public native OPLandIplParvoParameters horizontalCellsGain(float horizontalCellsGain);
               public native float hcellsTemporalConstant(); public native OPLandIplParvoParameters hcellsTemporalConstant(float hcellsTemporalConstant);
               public native float hcellsSpatialConstant(); public native OPLandIplParvoParameters hcellsSpatialConstant(float hcellsSpatialConstant);
               public native float ganglionCellsSensitivity(); public native OPLandIplParvoParameters ganglionCellsSensitivity(float ganglionCellsSensitivity);
           }
           @NoOffset public static class IplMagnoParameters extends Pointer {
               static { Loader.load(); }
               public IplMagnoParameters(Pointer p) { super(p); }
               public IplMagnoParameters(int size) { allocateArray(size); }
               private native void allocateArray(int size);
               @Override public IplMagnoParameters position(int position) {
                   return (IplMagnoParameters)super.position(position);
               }
            // Inner Plexiform Layer Magnocellular channel (IplMagno)
               public IplMagnoParameters() { allocate(); }
               private native void allocate();// default setup
               public native @Cast("bool") boolean normaliseOutput(); public native IplMagnoParameters normaliseOutput(boolean normaliseOutput);
               public native float parasolCells_beta(); public native IplMagnoParameters parasolCells_beta(float parasolCells_beta);
               public native float parasolCells_tau(); public native IplMagnoParameters parasolCells_tau(float parasolCells_tau);
               public native float parasolCells_k(); public native IplMagnoParameters parasolCells_k(float parasolCells_k);
               public native float amacrinCellsTemporalCutFrequency(); public native IplMagnoParameters amacrinCellsTemporalCutFrequency(float amacrinCellsTemporalCutFrequency);
               public native float V0CompressionParameter(); public native IplMagnoParameters V0CompressionParameter(float V0CompressionParameter);
               public native float localAdaptintegration_tau(); public native IplMagnoParameters localAdaptintegration_tau(float localAdaptintegration_tau);
               public native float localAdaptintegration_k(); public native IplMagnoParameters localAdaptintegration_k(float localAdaptintegration_k);
           }
            public native @ByRef OPLandIplParvoParameters OPLandIplParvo(); public native RetinaParameters OPLandIplParvo(OPLandIplParvoParameters OPLandIplParvo);
            public native @ByRef IplMagnoParameters IplMagno(); public native RetinaParameters IplMagno(IplMagnoParameters IplMagno);
    }

    /**
     * Main constructor with most commun use setup : create an instance of color ready retina model
     * @param inputSize : the input frame size
     */
    public Retina(@ByVal Size inputSize) { allocate(inputSize); }
    private native void allocate(@ByVal Size inputSize);

    /**
     * Complete Retina filter constructor which allows all basic structural parameters definition
         * @param inputSize : the input frame size
     * @param colorMode : the chosen processing mode : with or without color processing
     * @param colorSamplingMethod: specifies which kind of color sampling will be used
     * @param useRetinaLogSampling: activate retina log sampling, if true, the 2 following parameters can be used
     * @param reductionFactor: only usefull if param useRetinaLogSampling=true, specifies the reduction factor of the output frame (as the center (fovea) is high resolution and corners can be underscaled, then a reduction of the output is allowed without precision leak
     * @param samplingStrenght: only usefull if param useRetinaLogSampling=true, specifies the strenght of the log scale that is applied
     */
    public Retina(@ByVal Size inputSize, @Cast("const bool") boolean colorMode, @Cast("cv::RETINA_COLORSAMPLINGMETHOD") int colorSamplingMethod/*=RETINA_COLOR_BAYER*/, @Cast("const bool") boolean useRetinaLogSampling/*=false*/, double reductionFactor/*=1.0*/, double samplingStrenght/*=10.0*/) { allocate(inputSize, colorMode, colorSamplingMethod, useRetinaLogSampling, reductionFactor, samplingStrenght); }
    private native void allocate(@ByVal Size inputSize, @Cast("const bool") boolean colorMode, @Cast("cv::RETINA_COLORSAMPLINGMETHOD") int colorSamplingMethod/*=RETINA_COLOR_BAYER*/, @Cast("const bool") boolean useRetinaLogSampling/*=false*/, double reductionFactor/*=1.0*/, double samplingStrenght/*=10.0*/);
    public Retina(@ByVal Size inputSize, @Cast("const bool") boolean colorMode) { allocate(inputSize, colorMode); }
    private native void allocate(@ByVal Size inputSize, @Cast("const bool") boolean colorMode);

    /**
    * retreive retina input buffer size
    */
    public native @ByVal Size inputSize();

    /**
    * retreive retina output buffer size
    */
    public native @ByVal Size outputSize();

    /**
     * try to open an XML retina parameters file to adjust current retina instance setup
     * => if the xml file does not exist, then default setup is applied
     * => warning, Exceptions are thrown if read XML file is not valid
     * @param retinaParameterFile : the parameters filename
         * @param applyDefaultSetupOnFailure : set to true if an error must be thrown on error
     */
    public native void setup(@StdString BytePointer retinaParameterFile/*=""*/, @Cast("const bool") boolean applyDefaultSetupOnFailure/*=true*/);
    public native void setup();
    public native void setup(@StdString String retinaParameterFile/*=""*/, @Cast("const bool") boolean applyDefaultSetupOnFailure/*=true*/);


    /**
     * try to open an XML retina parameters file to adjust current retina instance setup
     * => if the xml file does not exist, then default setup is applied
     * => warning, Exceptions are thrown if read XML file is not valid
     * @param fs : the open Filestorage which contains retina parameters
     * @param applyDefaultSetupOnFailure : set to true if an error must be thrown on error
     */
    public native void setup(@ByRef FileStorage fs, @Cast("const bool") boolean applyDefaultSetupOnFailure/*=true*/);
    public native void setup(@ByRef FileStorage fs);

    /**
     * try to open an XML retina parameters file to adjust current retina instance setup
     * => if the xml file does not exist, then default setup is applied
     * => warning, Exceptions are thrown if read XML file is not valid
     * @param newParameters : a parameters structures updated with the new target configuration
     */
    public native void setup(@ByVal RetinaParameters newParameters);

    /**
     * @return the current parameters setup
     */
    public native @ByVal RetinaParameters getParameters();

    /**
     * parameters setup display method
     * @return a string which contains formatted parameters information
     */
    public native @StdString BytePointer printSetup();

    /**
     * write xml/yml formated parameters information
     * @param fs : the filename of the xml file that will be open and writen with formatted parameters information
     */
    public native void write( @StdString BytePointer fs );
    public native void write( @StdString String fs );


    /**
     * write xml/yml formated parameters information
     * @param fs : a cv::Filestorage object ready to be filled
         */
    public native void write( @ByRef FileStorage fs );

    /**
     * setup the OPL and IPL parvo channels (see biologocal model)
     * OPL is referred as Outer Plexiform Layer of the retina, it allows the spatio-temporal filtering which withens the spectrum and reduces spatio-temporal noise while attenuating global luminance (low frequency energy)
     * IPL parvo is the OPL next processing stage, it refers to Inner Plexiform layer of the retina, it allows high contours sensitivity in foveal vision.
     * for more informations, please have a look at the paper Benoit A., Caplier A., Durette B., Herault, J., "USING HUMAN VISUAL SYSTEM MODELING FOR BIO-INSPIRED LOW LEVEL IMAGE PROCESSING", Elsevier, Computer Vision and Image Understanding 114 (2010), pp. 758-773, DOI: http://dx.doi.org/10.1016/j.cviu.2010.01.011
     * @param colorMode : specifies if (true) color is processed of not (false) to then processing gray level image
     * @param normaliseOutput : specifies if (true) output is rescaled between 0 and 255 of not (false)
     * @param photoreceptorsLocalAdaptationSensitivity: the photoreceptors sensitivity renage is 0-1 (more log compression effect when value increases)
     * @param photoreceptorsTemporalConstant: the time constant of the first order low pass filter of the photoreceptors, use it to cut high temporal frequencies (noise or fast motion), unit is frames, typical value is 1 frame
     * @param photoreceptorsSpatialConstant: the spatial constant of the first order low pass filter of the photoreceptors, use it to cut high spatial frequencies (noise or thick contours), unit is pixels, typical value is 1 pixel
     * @param horizontalCellsGain: gain of the horizontal cells network, if 0, then the mean value of the output is zero, if the parameter is near 1, then, the luminance is not filtered and is still reachable at the output, typicall value is 0
     * @param HcellsTemporalConstant: the time constant of the first order low pass filter of the horizontal cells, use it to cut low temporal frequencies (local luminance variations), unit is frames, typical value is 1 frame, as the photoreceptors
     * @param HcellsSpatialConstant: the spatial constant of the first order low pass filter of the horizontal cells, use it to cut low spatial frequencies (local luminance), unit is pixels, typical value is 5 pixel, this value is also used for local contrast computing when computing the local contrast adaptation at the ganglion cells level (Inner Plexiform Layer parvocellular channel model)
     * @param ganglionCellsSensitivity: the compression strengh of the ganglion cells local adaptation output, set a value between 160 and 250 for best results, a high value increases more the low value sensitivity... and the output saturates faster, recommended value: 230
     */
    public native void setupOPLandIPLParvoChannel(@Cast("const bool") boolean colorMode/*=true*/, @Cast("const bool") boolean normaliseOutput/*=true*/, float photoreceptorsLocalAdaptationSensitivity/*=0.7f*/, float photoreceptorsTemporalConstant/*=0.5f*/, float photoreceptorsSpatialConstant/*=0.53f*/, float horizontalCellsGain/*=0*/, float HcellsTemporalConstant/*=1*/, float HcellsSpatialConstant/*=7*/, float ganglionCellsSensitivity/*=0.7f*/);
    public native void setupOPLandIPLParvoChannel();

    /**
     * set parameters values for the Inner Plexiform Layer (IPL) magnocellular channel
     * this channel processes signals outpint from OPL processing stage in peripheral vision, it allows motion information enhancement. It is decorrelated from the details channel. See reference paper for more details.
     * @param normaliseOutput : specifies if (true) output is rescaled between 0 and 255 of not (false)
     * @param parasolCells_beta: the low pass filter gain used for local contrast adaptation at the IPL level of the retina (for ganglion cells local adaptation), typical value is 0
     * @param parasolCells_tau: the low pass filter time constant used for local contrast adaptation at the IPL level of the retina (for ganglion cells local adaptation), unit is frame, typical value is 0 (immediate response)
     * @param parasolCells_k: the low pass filter spatial constant used for local contrast adaptation at the IPL level of the retina (for ganglion cells local adaptation), unit is pixels, typical value is 5
     * @param amacrinCellsTemporalCutFrequency: the time constant of the first order high pass fiter of the magnocellular way (motion information channel), unit is frames, tipicall value is 5
     * @param V0CompressionParameter: the compression strengh of the ganglion cells local adaptation output, set a value between 160 and 250 for best results, a high value increases more the low value sensitivity... and the output saturates faster, recommended value: 200
     * @param localAdaptintegration_tau: specifies the temporal constant of the low pas filter involved in the computation of the local "motion mean" for the local adaptation computation
     * @param localAdaptintegration_k: specifies the spatial constant of the low pas filter involved in the computation of the local "motion mean" for the local adaptation computation
     */
    public native void setupIPLMagnoChannel(@Cast("const bool") boolean normaliseOutput/*=true*/, float parasolCells_beta/*=0*/, float parasolCells_tau/*=0*/, float parasolCells_k/*=7*/, float amacrinCellsTemporalCutFrequency/*=1.2f*/, float V0CompressionParameter/*=0.95f*/, float localAdaptintegration_tau/*=0*/, float localAdaptintegration_k/*=7*/);
    public native void setupIPLMagnoChannel();

    /**
     * method which allows retina to be applied on an input image, after run, encapsulated retina module is ready to deliver its outputs using dedicated acccessors, see getParvo and getMagno methods
     * @param inputImage : the input cv::Mat image to be processed, can be gray level or BGR coded in any format (from 8bit to 16bits)
     */
    public native void run(@Const @ByRef Mat inputImage);

    /**
     * accessor of the details channel of the retina (models foveal vision)
     * @param retinaOutput_parvo : the output buffer (reallocated if necessary), this output is rescaled for standard 8bits image processing use in OpenCV
     */
    public native void getParvo(@ByRef Mat retinaOutput_parvo);

    /**
     * accessor of the details channel of the retina (models foveal vision)
     * @param retinaOutput_parvo : the output buffer (reallocated if necessary), this output is the original retina filter model output, without any quantification or rescaling
     */
    public native void getParvo(@ByRef FloatValArray retinaOutput_parvo);

    /**
     * accessor of the motion channel of the retina (models peripheral vision)
     * @param retinaOutput_magno : the output buffer (reallocated if necessary), this output is rescaled for standard 8bits image processing use in OpenCV
     */
    public native void getMagno(@ByRef Mat retinaOutput_magno);

    /**
     * accessor of the motion channel of the retina (models peripheral vision)
     * @param retinaOutput_magno : the output buffer (reallocated if necessary), this output is the original retina filter model output, without any quantification or rescaling
     */
    public native void getMagno(@ByRef FloatValArray retinaOutput_magno);

    // original API level data accessors : get buffers addresses...
    public native @Const @ByRef FloatValArray getMagno();
    public native @Const @ByRef FloatValArray getParvo();

    /**
     * activate color saturation as the final step of the color demultiplexing process
     * -> this saturation is a sigmoide function applied to each channel of the demultiplexed image.
     * @param saturateColors: boolean that activates color saturation (if true) or desactivate (if false)
     * @param colorSaturationValue: the saturation factor
     */
    public native void setColorSaturation(@Cast("const bool") boolean saturateColors/*=true*/, float colorSaturationValue/*=4.0*/);
    public native void setColorSaturation();

    /**
     * clear all retina buffers (equivalent to opening the eyes after a long period of eye close ;o)
     */
    public native void clearBuffers();

    /**
    * Activate/desactivate the Magnocellular pathway processing (motion information extraction), by default, it is activated
    * @param activate: true if Magnocellular output should be activated, false if not
    */
    public native void activateMovingContoursProcessing(@Cast("const bool") boolean activate);

    /**
    * Activate/desactivate the Parvocellular pathway processing (contours information extraction), by default, it is activated
    * @param activate: true if Parvocellular (contours information extraction) output should be activated, false if not
    */
    public native void activateContoursProcessing(@Cast("const bool") boolean activate);


}


// #endif /* __OPENCV_CONTRIB_RETINA_HPP__ */


// Parsed from <opencv2/contrib/openfabmap.hpp>

/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
// This file originates from the openFABMAP project:
// [http://code.google.com/p/openfabmap/]
//
// For published work which uses all or part of OpenFABMAP, please cite:
// [http://ieeexplore.ieee.org/xpl/articleDetails.jsp?arnumber=6224843]
//
// Original Algorithm by Mark Cummins and Paul Newman:
// [http://ijr.sagepub.com/content/27/6/647.short]
// [http://ieeexplore.ieee.org/xpl/articleDetails.jsp?arnumber=5613942]
// [http://ijr.sagepub.com/content/30/9/1100.abstract]
//
//                           License Agreement
//
// Copyright (C) 2012 Arren Glover [aj.glover@qut.edu.au] and
//                    Will Maddern [w.maddern@qut.edu.au], all rights reserved.
//
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

// #ifndef __OPENCV_OPENFABMAP_H_
// #define __OPENCV_OPENFABMAP_H_

// #include "opencv2/core/core.hpp"
// #include "opencv2/features2d/features2d.hpp"

// #include <vector>
// #include <list>
// #include <map>
// #include <set>
// #include <valarray>

/*
    Return data format of a FABMAP compare call
*/
@Namespace("cv::of2") @NoOffset public static class IMatch extends Pointer {
    static { Loader.load(); }
    public IMatch(Pointer p) { super(p); }
    public IMatch(int size) { allocateArray(size); }
    private native void allocateArray(int size);
    @Override public IMatch position(int position) {
        return (IMatch)super.position(position);
    }


    public IMatch() { allocate(); }
    private native void allocate();
    public IMatch(int _queryIdx, int _imgIdx, double _likelihood, double _match) { allocate(_queryIdx, _imgIdx, _likelihood, _match); }
    private native void allocate(int _queryIdx, int _imgIdx, double _likelihood, double _match);

    public native int queryIdx(); public native IMatch queryIdx(int queryIdx);    //query index
    public native int imgIdx(); public native IMatch imgIdx(int imgIdx);      //test index

    public native double likelihood(); public native IMatch likelihood(double likelihood);  //raw loglikelihood
    public native double match(); public native IMatch match(double match);      //normalised probability

    public native @Cast("bool") @Name("operator<") boolean lessThan(@Const @ByRef IMatch m);

}

/*
    Base FabMap class. Each FabMap method inherits from this class.
*/
@Namespace("cv::of2") @NoOffset public static class FabMap extends Pointer {
    static { Loader.load(); }
    public FabMap() { }
    public FabMap(Pointer p) { super(p); }


    //FabMap options
    /** enum cv::of2::FabMap:: */
    public static final int
        MEAN_FIELD = 1,
        SAMPLED = 2,
        NAIVE_BAYES = 4,
        CHOW_LIU = 8,
        MOTION_MODEL = 16;

    public FabMap(@Const @ByRef Mat clTree, double PzGe, double PzGNe, int flags,
                int numSamples/*=0*/) { allocate(clTree, PzGe, PzGNe, flags, numSamples); }
    private native void allocate(@Const @ByRef Mat clTree, double PzGe, double PzGNe, int flags,
                int numSamples/*=0*/);
    public FabMap(@Const @ByRef Mat clTree, double PzGe, double PzGNe, int flags) { allocate(clTree, PzGe, PzGNe, flags); }
    private native void allocate(@Const @ByRef Mat clTree, double PzGe, double PzGNe, int flags);

    //methods to add training data for sampling method
    public native void addTraining(@Const @ByRef Mat queryImgDescriptor);
    public native void addTraining(@Const @ByRef MatVector queryImgDescriptors);

    //methods to add to the test data
    public native void add(@Const @ByRef Mat queryImgDescriptor);
    public native void add(@Const @ByRef MatVector queryImgDescriptors);

    //accessors
    public native @Const @ByRef MatVector getTrainingImgDescriptors();
    public native @Const @ByRef MatVector getTestImgDescriptors();

    //Main FabMap image comparison
    public native void compare(@Const @ByRef Mat queryImgDescriptor,
                @StdVector IMatch matches, @Cast("bool") boolean addQuery/*=false*/,
                @Const @ByRef Mat mask/*=Mat()*/);
    public native void compare(@Const @ByRef Mat queryImgDescriptor,
                @StdVector IMatch matches);
    public native void compare(@Const @ByRef Mat queryImgDescriptor,
                @Const @ByRef Mat testImgDescriptors, @StdVector IMatch matches,
                @Const @ByRef Mat mask/*=Mat()*/);
    public native void compare(@Const @ByRef Mat queryImgDescriptor,
                @Const @ByRef Mat testImgDescriptors, @StdVector IMatch matches);
    public native void compare(@Const @ByRef Mat queryImgDescriptor,
                @Const @ByRef MatVector testImgDescriptors,
                @StdVector IMatch matches, @Const @ByRef Mat mask/*=Mat()*/);
    public native void compare(@Const @ByRef Mat queryImgDescriptor,
                @Const @ByRef MatVector testImgDescriptors,
                @StdVector IMatch matches);
    public native void compare(@Const @ByRef MatVector queryImgDescriptors, @StdVector IMatch matches, @Cast("bool") boolean addQuery/*=false*/, @Const @ByRef Mat mask/*=Mat()*/);
    public native void compare(@Const @ByRef MatVector queryImgDescriptors, @StdVector IMatch matches);
    public native void compare(@Const @ByRef MatVector queryImgDescriptors,
                @Const @ByRef MatVector testImgDescriptors,
                @StdVector IMatch matches, @Const @ByRef Mat mask/*=Mat()*/);
    public native void compare(@Const @ByRef MatVector queryImgDescriptors,
                @Const @ByRef MatVector testImgDescriptors,
                @StdVector IMatch matches);

}

/*
    The original FAB-MAP algorithm, developed based on:
    http://ijr.sagepub.com/content/27/6/647.short
*/
@Namespace("cv::of2") public static class FabMap1 extends FabMap {
    static { Loader.load(); }
    public FabMap1() { }
    public FabMap1(Pointer p) { super(p); }

    public FabMap1(@Const @ByRef Mat clTree, double PzGe, double PzGNe, int flags,
                int numSamples/*=0*/) { allocate(clTree, PzGe, PzGNe, flags, numSamples); }
    private native void allocate(@Const @ByRef Mat clTree, double PzGe, double PzGNe, int flags,
                int numSamples/*=0*/);
    public FabMap1(@Const @ByRef Mat clTree, double PzGe, double PzGNe, int flags) { allocate(clTree, PzGe, PzGNe, flags); }
    private native void allocate(@Const @ByRef Mat clTree, double PzGe, double PzGNe, int flags);
}

/*
    A computationally faster version of the original FAB-MAP algorithm. A look-
    up-table is used to precompute many of the reoccuring calculations
*/
@Namespace("cv::of2") @NoOffset public static class FabMapLUT extends FabMap {
    static { Loader.load(); }
    public FabMapLUT() { }
    public FabMapLUT(Pointer p) { super(p); }

    public FabMapLUT(@Const @ByRef Mat clTree, double PzGe, double PzGNe,
                int flags, int numSamples/*=0*/, int precision/*=6*/) { allocate(clTree, PzGe, PzGNe, flags, numSamples, precision); }
    private native void allocate(@Const @ByRef Mat clTree, double PzGe, double PzGNe,
                int flags, int numSamples/*=0*/, int precision/*=6*/);
    public FabMapLUT(@Const @ByRef Mat clTree, double PzGe, double PzGNe,
                int flags) { allocate(clTree, PzGe, PzGNe, flags); }
    private native void allocate(@Const @ByRef Mat clTree, double PzGe, double PzGNe,
                int flags);
}

/*
    The Accelerated FAB-MAP algorithm, developed based on:
    http://ieeexplore.ieee.org/xpl/articleDetails.jsp?arnumber=5613942
*/
@Namespace("cv::of2") @NoOffset public static class FabMapFBO extends FabMap {
    static { Loader.load(); }
    public FabMapFBO() { }
    public FabMapFBO(Pointer p) { super(p); }

    public FabMapFBO(@Const @ByRef Mat clTree, double PzGe, double PzGNe, int flags,
                int numSamples/*=0*/, double rejectionThreshold/*=1e-8*/, double PsGd/*=1e-8*/, int bisectionStart/*=512*/, int bisectionIts/*=9*/) { allocate(clTree, PzGe, PzGNe, flags, numSamples, rejectionThreshold, PsGd, bisectionStart, bisectionIts); }
    private native void allocate(@Const @ByRef Mat clTree, double PzGe, double PzGNe, int flags,
                int numSamples/*=0*/, double rejectionThreshold/*=1e-8*/, double PsGd/*=1e-8*/, int bisectionStart/*=512*/, int bisectionIts/*=9*/);
    public FabMapFBO(@Const @ByRef Mat clTree, double PzGe, double PzGNe, int flags) { allocate(clTree, PzGe, PzGNe, flags); }
    private native void allocate(@Const @ByRef Mat clTree, double PzGe, double PzGNe, int flags);
}

/*
    The FAB-MAP2.0 algorithm, developed based on:
    http://ijr.sagepub.com/content/30/9/1100.abstract
*/
@Namespace("cv::of2") @NoOffset public static class FabMap2 extends FabMap {
    static { Loader.load(); }
    public FabMap2() { }
    public FabMap2(Pointer p) { super(p); }


    public FabMap2(@Const @ByRef Mat clTree, double PzGe, double PzGNe, int flags) { allocate(clTree, PzGe, PzGNe, flags); }
    private native void allocate(@Const @ByRef Mat clTree, double PzGe, double PzGNe, int flags);

    //FabMap2 builds the inverted index and requires an additional training/test
    //add function
    public native void addTraining(@Const @ByRef Mat queryImgDescriptors);
    public native void addTraining(@Const @ByRef MatVector queryImgDescriptors);

    public native void add(@Const @ByRef Mat queryImgDescriptors);
    public native void add(@Const @ByRef MatVector queryImgDescriptors);

}
/*
    A Chow-Liu tree is required by FAB-MAP. The Chow-Liu tree provides an
    estimate of the full distribution of visual words using a minimum spanning
    tree. The tree is generated through training data.
*/
@Namespace("cv::of2") @NoOffset public static class ChowLiuTree extends Pointer {
    static { Loader.load(); }
    public ChowLiuTree(Pointer p) { super(p); }
    public ChowLiuTree(int size) { allocateArray(size); }
    private native void allocateArray(int size);
    @Override public ChowLiuTree position(int position) {
        return (ChowLiuTree)super.position(position);
    }

    public ChowLiuTree() { allocate(); }
    private native void allocate();

    //add data to the chow-liu tree before calling make
    public native void add(@Const @ByRef Mat imgDescriptor);
    public native void add(@Const @ByRef MatVector imgDescriptors);

    public native @Const @ByRef MatVector getImgDescriptors();

    public native @ByVal Mat make(double infoThreshold/*=0.0*/);
    public native @ByVal Mat make();

}

/*
    A custom vocabulary training method based on:
    http://www.springerlink.com/content/d1h6j8x552532003/
*/
@Namespace("cv::of2") @NoOffset public static class BOWMSCTrainer extends BOWTrainer {
    static { Loader.load(); }
    public BOWMSCTrainer(Pointer p) { super(p); }
    public BOWMSCTrainer(int size) { allocateArray(size); }
    private native void allocateArray(int size);
    @Override public BOWMSCTrainer position(int position) {
        return (BOWMSCTrainer)super.position(position);
    }

    public BOWMSCTrainer(double clusterSize/*=0.4*/) { allocate(clusterSize); }
    private native void allocate(double clusterSize/*=0.4*/);
    public BOWMSCTrainer() { allocate(); }
    private native void allocate();

    // Returns trained vocabulary (i.e. cluster centers).
    public native @ByVal Mat cluster();
    public native @ByVal Mat cluster(@Const @ByRef Mat descriptors);

}





// #endif /* OPENFABMAP_H_ */


}
