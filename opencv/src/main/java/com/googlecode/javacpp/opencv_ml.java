// Targeted by JavaCPP version 0.8-SNAPSHOT

package com.googlecode.javacpp;

import com.googlecode.javacpp.*;
import com.googlecode.javacpp.annotation.*;
import java.nio.*;

import static com.googlecode.javacpp.opencv_core.*;

public class opencv_ml extends com.googlecode.javacpp.presets.opencv_ml {
    static { Loader.load(); }

@Name("std::map<std::string,int>") public static class StringIntMap extends Pointer {
    static { Loader.load(); }
    public StringIntMap(Pointer p) { super(p); }
    public StringIntMap()       { allocate();  }
    private native void allocate();

    public native long size();

    @Index public native @ByRef int get(@StdString BytePointer i);
    public native StringIntMap put(@StdString BytePointer i, int value);
}

// Parsed from header file /usr/local/include/opencv2/ml/ml.hpp

/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                        Intel License Agreement
//
// Copyright (C) 2000, Intel Corporation, all rights reserved.
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

// #ifndef __OPENCV_ML_HPP__
// #define __OPENCV_ML_HPP__

// #include "opencv2/core/core.hpp"
// #include <limits.h>

// #ifdef __cplusplus

// #include <map>
// #include <string>
// #include <iostream>

// Apple defines a check() macro somewhere in the debug headers
// that interferes with a method definiton in this header
// #undef check

/****************************************************************************************\
*                               Main struct definitions                                  *
\****************************************************************************************/

/* log(2*PI) */
public static final double CV_LOG2PI = (1.8378770664093454835606594728112);

/* columns of <trainData> matrix are training samples */
public static final int CV_COL_SAMPLE = 0;

/* rows of <trainData> matrix are training samples */
public static final int CV_ROW_SAMPLE = 1;

// #define CV_IS_ROW_SAMPLE(flags) ((flags) & CV_ROW_SAMPLE)

public static class CvVectors extends Pointer {
    static { Loader.load(); }
    public CvVectors() { allocate(); }
    public CvVectors(int size) { allocateArray(size); }
    public CvVectors(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public CvVectors position(int position) {
        return (CvVectors)super.position(position);
    }

    public native int type(); public native CvVectors type(int type);
    public native int dims(); public native CvVectors dims(int dims);
    public native int count(); public native CvVectors count(int count);
    public native CvVectors next(); public native CvVectors next(CvVectors next);
        @Name("data.ptr") public native @Cast("uchar*") BytePointer data_ptr(int i); public native CvVectors data_ptr(int i, BytePointer data_ptr);
        @Name("data.ptr") @MemberGetter public native @Cast("uchar**") PointerPointer data_ptr();
        @Name("data.fl") public native FloatPointer data_fl(int i); public native CvVectors data_fl(int i, FloatPointer data_fl);
        @Name("data.fl") @MemberGetter public native @Cast("float**") PointerPointer data_fl();
        @Name("data.db") public native DoublePointer data_db(int i); public native CvVectors data_db(int i, DoublePointer data_db);
        @Name("data.db") @MemberGetter public native @Cast("double**") PointerPointer data_db();
}

// #if 0
// #endif

/* Variable type */
public static final int CV_VAR_NUMERICAL =    0;
public static final int CV_VAR_ORDERED =      0;
public static final int CV_VAR_CATEGORICAL =  1;

public static final String CV_TYPE_NAME_ML_SVM =         "opencv-ml-svm";
public static final String CV_TYPE_NAME_ML_KNN =         "opencv-ml-knn";
public static final String CV_TYPE_NAME_ML_NBAYES =      "opencv-ml-bayesian";
public static final String CV_TYPE_NAME_ML_EM =          "opencv-ml-em";
public static final String CV_TYPE_NAME_ML_BOOSTING =    "opencv-ml-boost-tree";
public static final String CV_TYPE_NAME_ML_TREE =        "opencv-ml-tree";
public static final String CV_TYPE_NAME_ML_ANN_MLP =     "opencv-ml-ann-mlp";
public static final String CV_TYPE_NAME_ML_CNN =         "opencv-ml-cnn";
public static final String CV_TYPE_NAME_ML_RTREES =      "opencv-ml-random-trees";
public static final String CV_TYPE_NAME_ML_ERTREES =     "opencv-ml-extremely-randomized-trees";
public static final String CV_TYPE_NAME_ML_GBT =         "opencv-ml-gradient-boosting-trees";

public static final int CV_TRAIN_ERROR =  0;
public static final int CV_TEST_ERROR =   1;

@NoOffset public static class CvStatModel extends Pointer {
    static { Loader.load(); }
    public CvStatModel(Pointer p) { super(p); }
    public CvStatModel(int size) { allocateArray(size); }
    private native void allocateArray(int size);
    @Override public CvStatModel position(int position) {
        return (CvStatModel)super.position(position);
    }

    public CvStatModel() { allocate(); }
    private native void allocate();

    public native void clear();

    public native void save( @Cast("const char*") BytePointer filename, @Cast("const char*") BytePointer name/*=0*/ );
    public native void save( String filename, String name/*=0*/ );
    public native void load( @Cast("const char*") BytePointer filename, @Cast("const char*") BytePointer name/*=0*/ );
    public native void load( String filename, String name/*=0*/ );

    public native void write( CvFileStorage storage, @Cast("const char*") BytePointer name );
    public native void write( CvFileStorage storage, String name );
    public native void read( CvFileStorage storage, CvFileNode node );
}

/****************************************************************************************\
*                                 Normal Bayes Classifier                                *
\****************************************************************************************/

/* The structure, representing the grid range of statmodel parameters.
   It is used for optimizing statmodel accuracy by varying model parameters,
   the accuracy estimate being computed by cross-validation.
   The grid is logarithmic, so <step> must be greater then 1. */

@NoOffset public static class CvParamGrid extends Pointer {
    static { Loader.load(); }
    public CvParamGrid(Pointer p) { super(p); }
    public CvParamGrid(int size) { allocateArray(size); }
    private native void allocateArray(int size);
    @Override public CvParamGrid position(int position) {
        return (CvParamGrid)super.position(position);
    }

    // SVM params type
    /** enum CvParamGrid:: */
    public static final int SVM_C= 0, SVM_GAMMA= 1, SVM_P= 2, SVM_NU= 3, SVM_COEF= 4, SVM_DEGREE= 5;

    public CvParamGrid() { allocate(); }
    private native void allocate();

    public CvParamGrid( double min_val, double max_val, double log_step ) { allocate(min_val, max_val, log_step); }
    private native void allocate( double min_val, double max_val, double log_step );
    //CvParamGrid( int param_id );
    public native @Cast("bool") boolean check();

    public native double min_val(); public native CvParamGrid min_val(double min_val);
    public native double max_val(); public native CvParamGrid max_val(double max_val);
    public native double step(); public native CvParamGrid step(double step);
}



@NoOffset public static class CvNormalBayesClassifier extends CvStatModel {
    static { Loader.load(); }
    public CvNormalBayesClassifier(Pointer p) { super(p); }
    public CvNormalBayesClassifier(int size) { allocateArray(size); }
    private native void allocateArray(int size);
    @Override public CvNormalBayesClassifier position(int position) {
        return (CvNormalBayesClassifier)super.position(position);
    }

    public CvNormalBayesClassifier() { allocate(); }
    private native void allocate();

    public CvNormalBayesClassifier( @Const CvMat trainData, @Const CvMat responses,
            @Const CvMat varIdx/*=0*/, @Const CvMat sampleIdx/*=0*/ ) { allocate(trainData, responses, varIdx, sampleIdx); }
    private native void allocate( @Const CvMat trainData, @Const CvMat responses,
            @Const CvMat varIdx/*=0*/, @Const CvMat sampleIdx/*=0*/ );

    public native @Cast("bool") boolean train( @Const CvMat trainData, @Const CvMat responses,
            @Const CvMat varIdx/*=0*/, @Const CvMat sampleIdx/*=0*/, @Cast("bool") boolean update/*=false*/ );

    public native float predict( @Const CvMat samples, CvMat results/*=0*/ );
    public native void clear();

    public CvNormalBayesClassifier( @Const @ByRef Mat trainData, @Const @ByRef Mat responses,
                                @Const @ByRef Mat varIdx/*=cv::Mat()*/, @Const @ByRef Mat sampleIdx/*=cv::Mat()*/ ) { allocate(trainData, responses, varIdx, sampleIdx); }
    private native void allocate( @Const @ByRef Mat trainData, @Const @ByRef Mat responses,
                                @Const @ByRef Mat varIdx/*=cv::Mat()*/, @Const @ByRef Mat sampleIdx/*=cv::Mat()*/ );
    public native @Cast("bool") boolean train( @Const @ByRef Mat trainData, @Const @ByRef Mat responses,
                           @Const @ByRef Mat varIdx/*=cv::Mat()*/, @Const @ByRef Mat sampleIdx/*=cv::Mat()*/,
                           @Cast("bool") boolean update/*=false*/ );
    public native float predict( @Const @ByRef Mat samples, Mat results/*=0*/ );

    public native void write( CvFileStorage storage, @Cast("const char*") BytePointer name );
    public native void write( CvFileStorage storage, String name );
    public native void read( CvFileStorage storage, CvFileNode node );
}


/****************************************************************************************\
*                          K-Nearest Neighbour Classifier                                *
\****************************************************************************************/

// k Nearest Neighbors
@NoOffset public static class CvKNearest extends CvStatModel {
    static { Loader.load(); }
    public CvKNearest(Pointer p) { super(p); }
    public CvKNearest(int size) { allocateArray(size); }
    private native void allocateArray(int size);
    @Override public CvKNearest position(int position) {
        return (CvKNearest)super.position(position);
    }


    public CvKNearest() { allocate(); }
    private native void allocate();

    public CvKNearest( @Const CvMat trainData, @Const CvMat responses,
                    @Const CvMat sampleIdx/*=0*/, @Cast("bool") boolean isRegression/*=false*/, int max_k/*=32*/ ) { allocate(trainData, responses, sampleIdx, isRegression, max_k); }
    private native void allocate( @Const CvMat trainData, @Const CvMat responses,
                    @Const CvMat sampleIdx/*=0*/, @Cast("bool") boolean isRegression/*=false*/, int max_k/*=32*/ );

    public native @Cast("bool") boolean train( @Const CvMat trainData, @Const CvMat responses,
                            @Const CvMat sampleIdx/*=0*/, @Cast("bool") boolean is_regression/*=false*/,
                            int maxK/*=32*/, @Cast("bool") boolean updateBase/*=false*/ );

    public native float find_nearest( @Const CvMat samples, int k, CvMat results/*=0*/,
            @Cast("const float**") PointerPointer neighbors/*=0*/, CvMat neighborResponses/*=0*/, CvMat dist/*=0*/ );
    public native float find_nearest( @Const CvMat samples, int k, CvMat results/*=0*/,
            @Const @ByPtrPtr FloatPointer neighbors/*=0*/, CvMat neighborResponses/*=0*/, CvMat dist/*=0*/ );
    public native float find_nearest( @Const CvMat samples, int k, CvMat results/*=0*/,
            @Const @ByPtrPtr FloatBuffer neighbors/*=0*/, CvMat neighborResponses/*=0*/, CvMat dist/*=0*/ );
    public native float find_nearest( @Const CvMat samples, int k, CvMat results/*=0*/,
            @Const @ByPtrPtr float[] neighbors/*=0*/, CvMat neighborResponses/*=0*/, CvMat dist/*=0*/ );

    public CvKNearest( @Const @ByRef Mat trainData, @Const @ByRef Mat responses,
                   @Const @ByRef Mat sampleIdx/*=cv::Mat()*/, @Cast("bool") boolean isRegression/*=false*/, int max_k/*=32*/ ) { allocate(trainData, responses, sampleIdx, isRegression, max_k); }
    private native void allocate( @Const @ByRef Mat trainData, @Const @ByRef Mat responses,
                   @Const @ByRef Mat sampleIdx/*=cv::Mat()*/, @Cast("bool") boolean isRegression/*=false*/, int max_k/*=32*/ );

    public native @Cast("bool") boolean train( @Const @ByRef Mat trainData, @Const @ByRef Mat responses,
                           @Const @ByRef Mat sampleIdx/*=cv::Mat()*/, @Cast("bool") boolean isRegression/*=false*/,
                           int maxK/*=32*/, @Cast("bool") boolean updateBase/*=false*/ );

    public native float find_nearest( @Const @ByRef Mat samples, int k, Mat results/*=0*/,
                                    @Cast("const float**") PointerPointer neighbors/*=0*/, Mat neighborResponses/*=0*/,
                                    Mat dist/*=0*/ );
    public native float find_nearest( @Const @ByRef Mat samples, int k, Mat results/*=0*/,
                                    @Const @ByPtrPtr FloatPointer neighbors/*=0*/, Mat neighborResponses/*=0*/,
                                    Mat dist/*=0*/ );
    public native float find_nearest( @Const @ByRef Mat samples, int k, Mat results/*=0*/,
                                    @Const @ByPtrPtr FloatBuffer neighbors/*=0*/, Mat neighborResponses/*=0*/,
                                    Mat dist/*=0*/ );
    public native float find_nearest( @Const @ByRef Mat samples, int k, Mat results/*=0*/,
                                    @Const @ByPtrPtr float[] neighbors/*=0*/, Mat neighborResponses/*=0*/,
                                    Mat dist/*=0*/ );
    public native float find_nearest( @Const @ByRef Mat samples, int k, @ByRef Mat results,
                                            @ByRef Mat neighborResponses, @ByRef Mat dists);

    public native void clear();
    public native int get_max_k();
    public native int get_var_count();
    public native int get_sample_count();
    public native @Cast("bool") boolean is_regression();

    public native float write_results( int k, int k1, int start, int end,
            @Const FloatPointer neighbor_responses, @Const FloatPointer dist, CvMat _results,
            CvMat _neighbor_responses, CvMat _dist, Cv32suf sort_buf );
    public native float write_results( int k, int k1, int start, int end,
            @Const FloatBuffer neighbor_responses, @Const FloatBuffer dist, CvMat _results,
            CvMat _neighbor_responses, CvMat _dist, Cv32suf sort_buf );
    public native float write_results( int k, int k1, int start, int end,
            @Const float[] neighbor_responses, @Const float[] dist, CvMat _results,
            CvMat _neighbor_responses, CvMat _dist, Cv32suf sort_buf );

    public native void find_neighbors_direct( @Const CvMat _samples, int k, int start, int end,
            FloatPointer neighbor_responses, @Cast("const float**") PointerPointer neighbors, FloatPointer dist );
    public native void find_neighbors_direct( @Const CvMat _samples, int k, int start, int end,
            FloatPointer neighbor_responses, @Const @ByPtrPtr FloatPointer neighbors, FloatPointer dist );
    public native void find_neighbors_direct( @Const CvMat _samples, int k, int start, int end,
            FloatBuffer neighbor_responses, @Const @ByPtrPtr FloatBuffer neighbors, FloatBuffer dist );
    public native void find_neighbors_direct( @Const CvMat _samples, int k, int start, int end,
            float[] neighbor_responses, @Const @ByPtrPtr float[] neighbors, float[] dist );
}

/****************************************************************************************\
*                                   Support Vector Machines                              *
\****************************************************************************************/

// SVM training parameters
@NoOffset public static class CvSVMParams extends Pointer {
    static { Loader.load(); }
    public CvSVMParams(Pointer p) { super(p); }
    public CvSVMParams(int size) { allocateArray(size); }
    private native void allocateArray(int size);
    @Override public CvSVMParams position(int position) {
        return (CvSVMParams)super.position(position);
    }

    public CvSVMParams() { allocate(); }
    private native void allocate();
    public CvSVMParams( int svm_type, int kernel_type,
                     double degree, double gamma, double coef0,
                     double Cvalue, double nu, double p,
                     CvMat class_weights, @ByVal CvTermCriteria term_crit ) { allocate(svm_type, kernel_type, degree, gamma, coef0, Cvalue, nu, p, class_weights, term_crit); }
    private native void allocate( int svm_type, int kernel_type,
                     double degree, double gamma, double coef0,
                     double Cvalue, double nu, double p,
                     CvMat class_weights, @ByVal CvTermCriteria term_crit );

    public native int svm_type(); public native CvSVMParams svm_type(int svm_type);
    public native int kernel_type(); public native CvSVMParams kernel_type(int kernel_type);
    public native double degree(); public native CvSVMParams degree(double degree); // for poly
    public native double gamma(); public native CvSVMParams gamma(double gamma);  // for poly/rbf/sigmoid
    public native double coef0(); public native CvSVMParams coef0(double coef0);  // for poly/sigmoid

    public native double C(); public native CvSVMParams C(double C);  // for CV_SVM_C_SVC, CV_SVM_EPS_SVR and CV_SVM_NU_SVR
    public native double nu(); public native CvSVMParams nu(double nu); // for CV_SVM_NU_SVC, CV_SVM_ONE_CLASS, and CV_SVM_NU_SVR
    public native double p(); public native CvSVMParams p(double p); // for CV_SVM_EPS_SVR
    public native CvMat class_weights(); public native CvSVMParams class_weights(CvMat class_weights); // for CV_SVM_C_SVC
    public native @ByVal CvTermCriteria term_crit(); public native CvSVMParams term_crit(CvTermCriteria term_crit); // termination criteria
}


@NoOffset public static class CvSVMKernel extends Pointer {
    static { Loader.load(); }
    public CvSVMKernel(Pointer p) { super(p); }
    public CvSVMKernel(int size) { allocateArray(size); }
    private native void allocateArray(int size);
    @Override public CvSVMKernel position(int position) {
        return (CvSVMKernel)super.position(position);
    }

    @Namespace("CvSVMKernel") public static class Calc extends FunctionPointer {
        static { Loader.load(); }
        public    Calc(Pointer p) { super(p); }
        public native void call(CvSVMKernel o,  int vec_count, int vec_size, @Const @ByPtrPtr FloatPointer vecs,
                                           @Const FloatPointer another, FloatPointer results );
    }
    public CvSVMKernel() { allocate(); }
    private native void allocate();
    public CvSVMKernel( @Const CvSVMParams params, Calc _calc_func ) { allocate(params, _calc_func); }
    private native void allocate( @Const CvSVMParams params, Calc _calc_func );
    public native @Cast("bool") boolean create( @Const CvSVMParams params, Calc _calc_func );

    public native void clear();
    public native void calc( int vcount, int n, @Cast("const float**") PointerPointer vecs, @Const FloatPointer another, FloatPointer results );
    public native void calc( int vcount, int n, @Const @ByPtrPtr FloatPointer vecs, @Const FloatPointer another, FloatPointer results );
    public native void calc( int vcount, int n, @Const @ByPtrPtr FloatBuffer vecs, @Const FloatBuffer another, FloatBuffer results );
    public native void calc( int vcount, int n, @Const @ByPtrPtr float[] vecs, @Const float[] another, float[] results );

    @MemberGetter public native @Const CvSVMParams params();
    public native Calc calc_func(); public native CvSVMKernel calc_func(Calc calc_func);

    public native void calc_non_rbf_base( int vec_count, int vec_size, @Cast("const float**") PointerPointer vecs,
                                        @Const FloatPointer another, FloatPointer results,
                                        double alpha, double beta );
    public native void calc_non_rbf_base( int vec_count, int vec_size, @Const @ByPtrPtr FloatPointer vecs,
                                        @Const FloatPointer another, FloatPointer results,
                                        double alpha, double beta );
    public native void calc_non_rbf_base( int vec_count, int vec_size, @Const @ByPtrPtr FloatBuffer vecs,
                                        @Const FloatBuffer another, FloatBuffer results,
                                        double alpha, double beta );
    public native void calc_non_rbf_base( int vec_count, int vec_size, @Const @ByPtrPtr float[] vecs,
                                        @Const float[] another, float[] results,
                                        double alpha, double beta );

    public native void calc_linear( int vec_count, int vec_size, @Cast("const float**") PointerPointer vecs,
                                  @Const FloatPointer another, FloatPointer results );
    public native void calc_linear( int vec_count, int vec_size, @Const @ByPtrPtr FloatPointer vecs,
                                  @Const FloatPointer another, FloatPointer results );
    public native void calc_linear( int vec_count, int vec_size, @Const @ByPtrPtr FloatBuffer vecs,
                                  @Const FloatBuffer another, FloatBuffer results );
    public native void calc_linear( int vec_count, int vec_size, @Const @ByPtrPtr float[] vecs,
                                  @Const float[] another, float[] results );
    public native void calc_rbf( int vec_count, int vec_size, @Cast("const float**") PointerPointer vecs,
                               @Const FloatPointer another, FloatPointer results );
    public native void calc_rbf( int vec_count, int vec_size, @Const @ByPtrPtr FloatPointer vecs,
                               @Const FloatPointer another, FloatPointer results );
    public native void calc_rbf( int vec_count, int vec_size, @Const @ByPtrPtr FloatBuffer vecs,
                               @Const FloatBuffer another, FloatBuffer results );
    public native void calc_rbf( int vec_count, int vec_size, @Const @ByPtrPtr float[] vecs,
                               @Const float[] another, float[] results );
    public native void calc_poly( int vec_count, int vec_size, @Cast("const float**") PointerPointer vecs,
                                @Const FloatPointer another, FloatPointer results );
    public native void calc_poly( int vec_count, int vec_size, @Const @ByPtrPtr FloatPointer vecs,
                                @Const FloatPointer another, FloatPointer results );
    public native void calc_poly( int vec_count, int vec_size, @Const @ByPtrPtr FloatBuffer vecs,
                                @Const FloatBuffer another, FloatBuffer results );
    public native void calc_poly( int vec_count, int vec_size, @Const @ByPtrPtr float[] vecs,
                                @Const float[] another, float[] results );
    public native void calc_sigmoid( int vec_count, int vec_size, @Cast("const float**") PointerPointer vecs,
                                   @Const FloatPointer another, FloatPointer results );
    public native void calc_sigmoid( int vec_count, int vec_size, @Const @ByPtrPtr FloatPointer vecs,
                                   @Const FloatPointer another, FloatPointer results );
    public native void calc_sigmoid( int vec_count, int vec_size, @Const @ByPtrPtr FloatBuffer vecs,
                                   @Const FloatBuffer another, FloatBuffer results );
    public native void calc_sigmoid( int vec_count, int vec_size, @Const @ByPtrPtr float[] vecs,
                                   @Const float[] another, float[] results );
}


public static class CvSVMKernelRow extends Pointer {
    static { Loader.load(); }
    public CvSVMKernelRow() { allocate(); }
    public CvSVMKernelRow(int size) { allocateArray(size); }
    public CvSVMKernelRow(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public CvSVMKernelRow position(int position) {
        return (CvSVMKernelRow)super.position(position);
    }

    public native CvSVMKernelRow prev(); public native CvSVMKernelRow prev(CvSVMKernelRow prev);
    public native CvSVMKernelRow next(); public native CvSVMKernelRow next(CvSVMKernelRow next);
    public native FloatPointer data(); public native CvSVMKernelRow data(FloatPointer data);
}


public static class CvSVMSolutionInfo extends Pointer {
    static { Loader.load(); }
    public CvSVMSolutionInfo() { allocate(); }
    public CvSVMSolutionInfo(int size) { allocateArray(size); }
    public CvSVMSolutionInfo(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public CvSVMSolutionInfo position(int position) {
        return (CvSVMSolutionInfo)super.position(position);
    }

    public native double obj(); public native CvSVMSolutionInfo obj(double obj);
    public native double rho(); public native CvSVMSolutionInfo rho(double rho);
    public native double upper_bound_p(); public native CvSVMSolutionInfo upper_bound_p(double upper_bound_p);
    public native double upper_bound_n(); public native CvSVMSolutionInfo upper_bound_n(double upper_bound_n);
    public native double r(); public native CvSVMSolutionInfo r(double r);   // for Solver_NU
}

@NoOffset public static class CvSVMSolver extends Pointer {
    static { Loader.load(); }
    public CvSVMSolver(Pointer p) { super(p); }
    public CvSVMSolver(int size) { allocateArray(size); }
    private native void allocateArray(int size);
    @Override public CvSVMSolver position(int position) {
        return (CvSVMSolver)super.position(position);
    }

    @Namespace("CvSVMSolver") public static class SelectWorkingSet extends FunctionPointer {
        static { Loader.load(); }
        public    SelectWorkingSet(Pointer p) { super(p); }
        public native @Cast("bool") boolean call(CvSVMSolver o,  @ByRef IntPointer i, @ByRef IntPointer j );
    }
    @Namespace("CvSVMSolver") public static class GetRow extends FunctionPointer {
        static { Loader.load(); }
        public    GetRow(Pointer p) { super(p); }
        public native FloatPointer call(CvSVMSolver o,  int i, FloatPointer row, FloatPointer dst, @Cast("bool") boolean existed );
    }
    @Namespace("CvSVMSolver") public static class CalcRho extends FunctionPointer {
        static { Loader.load(); }
        public    CalcRho(Pointer p) { super(p); }
        public native void call(CvSVMSolver o,  @ByRef DoublePointer rho, @ByRef DoublePointer r );
    }

    public CvSVMSolver() { allocate(); }
    private native void allocate();

    public CvSVMSolver( int count, int var_count, @Cast("const float**") PointerPointer samples, @Cast("schar*") BytePointer y,
                     int alpha_count, DoublePointer alpha, double Cp, double Cn,
                     CvMemStorage storage, CvSVMKernel kernel, GetRow get_row,
                     SelectWorkingSet select_working_set, CalcRho calc_rho ) { allocate(count, var_count, samples, y, alpha_count, alpha, Cp, Cn, storage, kernel, get_row, select_working_set, calc_rho); }
    private native void allocate( int count, int var_count, @Cast("const float**") PointerPointer samples, @Cast("schar*") BytePointer y,
                     int alpha_count, DoublePointer alpha, double Cp, double Cn,
                     CvMemStorage storage, CvSVMKernel kernel, GetRow get_row,
                     SelectWorkingSet select_working_set, CalcRho calc_rho );
    public CvSVMSolver( int count, int var_count, @Const @ByPtrPtr FloatPointer samples, @Cast("schar*") BytePointer y,
                     int alpha_count, DoublePointer alpha, double Cp, double Cn,
                     CvMemStorage storage, CvSVMKernel kernel, GetRow get_row,
                     SelectWorkingSet select_working_set, CalcRho calc_rho ) { allocate(count, var_count, samples, y, alpha_count, alpha, Cp, Cn, storage, kernel, get_row, select_working_set, calc_rho); }
    private native void allocate( int count, int var_count, @Const @ByPtrPtr FloatPointer samples, @Cast("schar*") BytePointer y,
                     int alpha_count, DoublePointer alpha, double Cp, double Cn,
                     CvMemStorage storage, CvSVMKernel kernel, GetRow get_row,
                     SelectWorkingSet select_working_set, CalcRho calc_rho );
    public CvSVMSolver( int count, int var_count, @Const @ByPtrPtr FloatBuffer samples, @Cast("schar*") ByteBuffer y,
                     int alpha_count, DoubleBuffer alpha, double Cp, double Cn,
                     CvMemStorage storage, CvSVMKernel kernel, GetRow get_row,
                     SelectWorkingSet select_working_set, CalcRho calc_rho ) { allocate(count, var_count, samples, y, alpha_count, alpha, Cp, Cn, storage, kernel, get_row, select_working_set, calc_rho); }
    private native void allocate( int count, int var_count, @Const @ByPtrPtr FloatBuffer samples, @Cast("schar*") ByteBuffer y,
                     int alpha_count, DoubleBuffer alpha, double Cp, double Cn,
                     CvMemStorage storage, CvSVMKernel kernel, GetRow get_row,
                     SelectWorkingSet select_working_set, CalcRho calc_rho );
    public CvSVMSolver( int count, int var_count, @Const @ByPtrPtr float[] samples, @Cast("schar*") byte[] y,
                     int alpha_count, double[] alpha, double Cp, double Cn,
                     CvMemStorage storage, CvSVMKernel kernel, GetRow get_row,
                     SelectWorkingSet select_working_set, CalcRho calc_rho ) { allocate(count, var_count, samples, y, alpha_count, alpha, Cp, Cn, storage, kernel, get_row, select_working_set, calc_rho); }
    private native void allocate( int count, int var_count, @Const @ByPtrPtr float[] samples, @Cast("schar*") byte[] y,
                     int alpha_count, double[] alpha, double Cp, double Cn,
                     CvMemStorage storage, CvSVMKernel kernel, GetRow get_row,
                     SelectWorkingSet select_working_set, CalcRho calc_rho );
    public native @Cast("bool") boolean create( int count, int var_count, @Cast("const float**") PointerPointer samples, @Cast("schar*") BytePointer y,
                     int alpha_count, DoublePointer alpha, double Cp, double Cn,
                     CvMemStorage storage, CvSVMKernel kernel, GetRow get_row,
                     SelectWorkingSet select_working_set, CalcRho calc_rho );
    public native @Cast("bool") boolean create( int count, int var_count, @Const @ByPtrPtr FloatPointer samples, @Cast("schar*") BytePointer y,
                     int alpha_count, DoublePointer alpha, double Cp, double Cn,
                     CvMemStorage storage, CvSVMKernel kernel, GetRow get_row,
                     SelectWorkingSet select_working_set, CalcRho calc_rho );
    public native @Cast("bool") boolean create( int count, int var_count, @Const @ByPtrPtr FloatBuffer samples, @Cast("schar*") ByteBuffer y,
                     int alpha_count, DoubleBuffer alpha, double Cp, double Cn,
                     CvMemStorage storage, CvSVMKernel kernel, GetRow get_row,
                     SelectWorkingSet select_working_set, CalcRho calc_rho );
    public native @Cast("bool") boolean create( int count, int var_count, @Const @ByPtrPtr float[] samples, @Cast("schar*") byte[] y,
                     int alpha_count, double[] alpha, double Cp, double Cn,
                     CvMemStorage storage, CvSVMKernel kernel, GetRow get_row,
                     SelectWorkingSet select_working_set, CalcRho calc_rho );

    public native void clear();
    public native @Cast("bool") boolean solve_generic( @ByRef CvSVMSolutionInfo si );

    public native @Cast("bool") boolean solve_c_svc( int count, int var_count, @Cast("const float**") PointerPointer samples, @Cast("schar*") BytePointer y,
                                  double Cp, double Cn, CvMemStorage storage,
                                  CvSVMKernel kernel, DoublePointer alpha, @ByRef CvSVMSolutionInfo si );
    public native @Cast("bool") boolean solve_c_svc( int count, int var_count, @Const @ByPtrPtr FloatPointer samples, @Cast("schar*") BytePointer y,
                                  double Cp, double Cn, CvMemStorage storage,
                                  CvSVMKernel kernel, DoublePointer alpha, @ByRef CvSVMSolutionInfo si );
    public native @Cast("bool") boolean solve_c_svc( int count, int var_count, @Const @ByPtrPtr FloatBuffer samples, @Cast("schar*") ByteBuffer y,
                                  double Cp, double Cn, CvMemStorage storage,
                                  CvSVMKernel kernel, DoubleBuffer alpha, @ByRef CvSVMSolutionInfo si );
    public native @Cast("bool") boolean solve_c_svc( int count, int var_count, @Const @ByPtrPtr float[] samples, @Cast("schar*") byte[] y,
                                  double Cp, double Cn, CvMemStorage storage,
                                  CvSVMKernel kernel, double[] alpha, @ByRef CvSVMSolutionInfo si );
    public native @Cast("bool") boolean solve_nu_svc( int count, int var_count, @Cast("const float**") PointerPointer samples, @Cast("schar*") BytePointer y,
                                   CvMemStorage storage, CvSVMKernel kernel,
                                   DoublePointer alpha, @ByRef CvSVMSolutionInfo si );
    public native @Cast("bool") boolean solve_nu_svc( int count, int var_count, @Const @ByPtrPtr FloatPointer samples, @Cast("schar*") BytePointer y,
                                   CvMemStorage storage, CvSVMKernel kernel,
                                   DoublePointer alpha, @ByRef CvSVMSolutionInfo si );
    public native @Cast("bool") boolean solve_nu_svc( int count, int var_count, @Const @ByPtrPtr FloatBuffer samples, @Cast("schar*") ByteBuffer y,
                                   CvMemStorage storage, CvSVMKernel kernel,
                                   DoubleBuffer alpha, @ByRef CvSVMSolutionInfo si );
    public native @Cast("bool") boolean solve_nu_svc( int count, int var_count, @Const @ByPtrPtr float[] samples, @Cast("schar*") byte[] y,
                                   CvMemStorage storage, CvSVMKernel kernel,
                                   double[] alpha, @ByRef CvSVMSolutionInfo si );
    public native @Cast("bool") boolean solve_one_class( int count, int var_count, @Cast("const float**") PointerPointer samples,
                                      CvMemStorage storage, CvSVMKernel kernel,
                                      DoublePointer alpha, @ByRef CvSVMSolutionInfo si );
    public native @Cast("bool") boolean solve_one_class( int count, int var_count, @Const @ByPtrPtr FloatPointer samples,
                                      CvMemStorage storage, CvSVMKernel kernel,
                                      DoublePointer alpha, @ByRef CvSVMSolutionInfo si );
    public native @Cast("bool") boolean solve_one_class( int count, int var_count, @Const @ByPtrPtr FloatBuffer samples,
                                      CvMemStorage storage, CvSVMKernel kernel,
                                      DoubleBuffer alpha, @ByRef CvSVMSolutionInfo si );
    public native @Cast("bool") boolean solve_one_class( int count, int var_count, @Const @ByPtrPtr float[] samples,
                                      CvMemStorage storage, CvSVMKernel kernel,
                                      double[] alpha, @ByRef CvSVMSolutionInfo si );

    public native @Cast("bool") boolean solve_eps_svr( int count, int var_count, @Cast("const float**") PointerPointer samples, @Const FloatPointer y,
                                    CvMemStorage storage, CvSVMKernel kernel,
                                    DoublePointer alpha, @ByRef CvSVMSolutionInfo si );
    public native @Cast("bool") boolean solve_eps_svr( int count, int var_count, @Const @ByPtrPtr FloatPointer samples, @Const FloatPointer y,
                                    CvMemStorage storage, CvSVMKernel kernel,
                                    DoublePointer alpha, @ByRef CvSVMSolutionInfo si );
    public native @Cast("bool") boolean solve_eps_svr( int count, int var_count, @Const @ByPtrPtr FloatBuffer samples, @Const FloatBuffer y,
                                    CvMemStorage storage, CvSVMKernel kernel,
                                    DoubleBuffer alpha, @ByRef CvSVMSolutionInfo si );
    public native @Cast("bool") boolean solve_eps_svr( int count, int var_count, @Const @ByPtrPtr float[] samples, @Const float[] y,
                                    CvMemStorage storage, CvSVMKernel kernel,
                                    double[] alpha, @ByRef CvSVMSolutionInfo si );

    public native @Cast("bool") boolean solve_nu_svr( int count, int var_count, @Cast("const float**") PointerPointer samples, @Const FloatPointer y,
                                   CvMemStorage storage, CvSVMKernel kernel,
                                   DoublePointer alpha, @ByRef CvSVMSolutionInfo si );
    public native @Cast("bool") boolean solve_nu_svr( int count, int var_count, @Const @ByPtrPtr FloatPointer samples, @Const FloatPointer y,
                                   CvMemStorage storage, CvSVMKernel kernel,
                                   DoublePointer alpha, @ByRef CvSVMSolutionInfo si );
    public native @Cast("bool") boolean solve_nu_svr( int count, int var_count, @Const @ByPtrPtr FloatBuffer samples, @Const FloatBuffer y,
                                   CvMemStorage storage, CvSVMKernel kernel,
                                   DoubleBuffer alpha, @ByRef CvSVMSolutionInfo si );
    public native @Cast("bool") boolean solve_nu_svr( int count, int var_count, @Const @ByPtrPtr float[] samples, @Const float[] y,
                                   CvMemStorage storage, CvSVMKernel kernel,
                                   double[] alpha, @ByRef CvSVMSolutionInfo si );

    public native FloatPointer get_row_base( int i, @Cast("bool*") BoolPointer _existed );
    public native FloatPointer get_row( int i, FloatPointer dst );
    public native FloatBuffer get_row( int i, FloatBuffer dst );
    public native float[] get_row( int i, float[] dst );

    public native int sample_count(); public native CvSVMSolver sample_count(int sample_count);
    public native int var_count(); public native CvSVMSolver var_count(int var_count);
    public native int cache_size(); public native CvSVMSolver cache_size(int cache_size);
    public native int cache_line_size(); public native CvSVMSolver cache_line_size(int cache_line_size);
    @MemberGetter public native @Const FloatPointer samples(int i);
    @MemberGetter public native @Cast("const float**") PointerPointer samples();
    @MemberGetter public native @Const CvSVMParams params();
    public native CvMemStorage storage(); public native CvSVMSolver storage(CvMemStorage storage);
    public native @ByVal CvSVMKernelRow lru_list(); public native CvSVMSolver lru_list(CvSVMKernelRow lru_list);
    public native CvSVMKernelRow rows(); public native CvSVMSolver rows(CvSVMKernelRow rows);

    public native int alpha_count(); public native CvSVMSolver alpha_count(int alpha_count);

    public native DoublePointer G(); public native CvSVMSolver G(DoublePointer G);
    public native DoublePointer alpha(); public native CvSVMSolver alpha(DoublePointer alpha);

    // -1 - lower bound, 0 - free, 1 - upper bound
    public native @Cast("schar*") BytePointer alpha_status(); public native CvSVMSolver alpha_status(BytePointer alpha_status);

    public native @Cast("schar*") BytePointer y(); public native CvSVMSolver y(BytePointer y);
    public native DoublePointer b(); public native CvSVMSolver b(DoublePointer b);
    public native FloatPointer buf(int i); public native CvSVMSolver buf(int i, FloatPointer buf);
    @MemberGetter public native @Cast("float**") PointerPointer buf();
    public native double eps(); public native CvSVMSolver eps(double eps);
    public native int max_iter(); public native CvSVMSolver max_iter(int max_iter);
    public native double C(int i); public native CvSVMSolver C(int i, double C);
    @MemberGetter public native DoublePointer C();  // C[0] == Cn, C[1] == Cp
    public native CvSVMKernel kernel(); public native CvSVMSolver kernel(CvSVMKernel kernel);

    public native SelectWorkingSet select_working_set_func(); public native CvSVMSolver select_working_set_func(SelectWorkingSet select_working_set_func);
    public native CalcRho calc_rho_func(); public native CvSVMSolver calc_rho_func(CalcRho calc_rho_func);
    public native GetRow get_row_func(); public native CvSVMSolver get_row_func(GetRow get_row_func);

    public native @Cast("bool") boolean select_working_set( @ByRef IntPointer i, @ByRef IntPointer j );
    public native @Cast("bool") boolean select_working_set( @ByRef IntBuffer i, @ByRef IntBuffer j );
    public native @Cast("bool") boolean select_working_set( @ByRef int[] i, @ByRef int[] j );
    public native @Cast("bool") boolean select_working_set_nu_svm( @ByRef IntPointer i, @ByRef IntPointer j );
    public native @Cast("bool") boolean select_working_set_nu_svm( @ByRef IntBuffer i, @ByRef IntBuffer j );
    public native @Cast("bool") boolean select_working_set_nu_svm( @ByRef int[] i, @ByRef int[] j );
    public native void calc_rho( @ByRef DoublePointer rho, @ByRef DoublePointer r );
    public native void calc_rho( @ByRef DoubleBuffer rho, @ByRef DoubleBuffer r );
    public native void calc_rho( @ByRef double[] rho, @ByRef double[] r );
    public native void calc_rho_nu_svm( @ByRef DoublePointer rho, @ByRef DoublePointer r );
    public native void calc_rho_nu_svm( @ByRef DoubleBuffer rho, @ByRef DoubleBuffer r );
    public native void calc_rho_nu_svm( @ByRef double[] rho, @ByRef double[] r );

    public native FloatPointer get_row_svc( int i, FloatPointer row, FloatPointer dst, @Cast("bool") boolean existed );
    public native FloatBuffer get_row_svc( int i, FloatBuffer row, FloatBuffer dst, @Cast("bool") boolean existed );
    public native float[] get_row_svc( int i, float[] row, float[] dst, @Cast("bool") boolean existed );
    public native FloatPointer get_row_one_class( int i, FloatPointer row, FloatPointer dst, @Cast("bool") boolean existed );
    public native FloatBuffer get_row_one_class( int i, FloatBuffer row, FloatBuffer dst, @Cast("bool") boolean existed );
    public native float[] get_row_one_class( int i, float[] row, float[] dst, @Cast("bool") boolean existed );
    public native FloatPointer get_row_svr( int i, FloatPointer row, FloatPointer dst, @Cast("bool") boolean existed );
    public native FloatBuffer get_row_svr( int i, FloatBuffer row, FloatBuffer dst, @Cast("bool") boolean existed );
    public native float[] get_row_svr( int i, float[] row, float[] dst, @Cast("bool") boolean existed );
}


public static class CvSVMDecisionFunc extends Pointer {
    static { Loader.load(); }
    public CvSVMDecisionFunc() { allocate(); }
    public CvSVMDecisionFunc(int size) { allocateArray(size); }
    public CvSVMDecisionFunc(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public CvSVMDecisionFunc position(int position) {
        return (CvSVMDecisionFunc)super.position(position);
    }

    public native double rho(); public native CvSVMDecisionFunc rho(double rho);
    public native int sv_count(); public native CvSVMDecisionFunc sv_count(int sv_count);
    public native DoublePointer alpha(); public native CvSVMDecisionFunc alpha(DoublePointer alpha);
    public native IntPointer sv_index(); public native CvSVMDecisionFunc sv_index(IntPointer sv_index);
}


// SVM model
@NoOffset public static class CvSVM extends CvStatModel {
    static { Loader.load(); }
    public CvSVM(Pointer p) { super(p); }
    public CvSVM(int size) { allocateArray(size); }
    private native void allocateArray(int size);
    @Override public CvSVM position(int position) {
        return (CvSVM)super.position(position);
    }

    // SVM type
    /** enum CvSVM:: */
    public static final int C_SVC= 100, NU_SVC= 101, ONE_CLASS= 102, EPS_SVR= 103, NU_SVR= 104;

    // SVM kernel type
    /** enum CvSVM:: */
    public static final int LINEAR= 0, POLY= 1, RBF= 2, SIGMOID= 3;

    // SVM params type
    /** enum CvSVM:: */
    public static final int C= 0, GAMMA= 1, P= 2, NU= 3, COEF= 4, DEGREE= 5;

    public CvSVM() { allocate(); }
    private native void allocate();

    public CvSVM( @Const CvMat trainData, @Const CvMat responses,
               @Const CvMat varIdx/*=0*/, @Const CvMat sampleIdx/*=0*/,
               @ByVal CvSVMParams params/*=CvSVMParams()*/ ) { allocate(trainData, responses, varIdx, sampleIdx, params); }
    private native void allocate( @Const CvMat trainData, @Const CvMat responses,
               @Const CvMat varIdx/*=0*/, @Const CvMat sampleIdx/*=0*/,
               @ByVal CvSVMParams params/*=CvSVMParams()*/ );

    public native @Cast("bool") boolean train( @Const CvMat trainData, @Const CvMat responses,
                            @Const CvMat varIdx/*=0*/, @Const CvMat sampleIdx/*=0*/,
                            @ByVal CvSVMParams params/*=CvSVMParams()*/ );

    public native @Cast("bool") boolean train_auto( @Const CvMat trainData, @Const CvMat responses,
            @Const CvMat varIdx, @Const CvMat sampleIdx, @ByVal CvSVMParams params,
            int kfold/*=10*/,
            @ByVal CvParamGrid Cgrid/*=get_default_grid(CvSVM::C)*/,
            @ByVal CvParamGrid gammaGrid/*=get_default_grid(CvSVM::GAMMA)*/,
            @ByVal CvParamGrid pGrid/*=get_default_grid(CvSVM::P)*/,
            @ByVal CvParamGrid nuGrid/*=get_default_grid(CvSVM::NU)*/,
            @ByVal CvParamGrid coeffGrid/*=get_default_grid(CvSVM::COEF)*/,
            @ByVal CvParamGrid degreeGrid/*=get_default_grid(CvSVM::DEGREE)*/,
            @Cast("bool") boolean balanced/*=false*/ );

    public native float predict( @Const CvMat sample, @Cast("bool") boolean returnDFVal/*=false*/ );
    public native float predict( @Const CvMat samples, CvMat results );

    public CvSVM( @Const @ByRef Mat trainData, @Const @ByRef Mat responses,
              @Const @ByRef Mat varIdx/*=cv::Mat()*/, @Const @ByRef Mat sampleIdx/*=cv::Mat()*/,
              @ByVal CvSVMParams params/*=CvSVMParams()*/ ) { allocate(trainData, responses, varIdx, sampleIdx, params); }
    private native void allocate( @Const @ByRef Mat trainData, @Const @ByRef Mat responses,
              @Const @ByRef Mat varIdx/*=cv::Mat()*/, @Const @ByRef Mat sampleIdx/*=cv::Mat()*/,
              @ByVal CvSVMParams params/*=CvSVMParams()*/ );

    public native @Cast("bool") boolean train( @Const @ByRef Mat trainData, @Const @ByRef Mat responses,
                           @Const @ByRef Mat varIdx/*=cv::Mat()*/, @Const @ByRef Mat sampleIdx/*=cv::Mat()*/,
                           @ByVal CvSVMParams params/*=CvSVMParams()*/ );

    public native @Cast("bool") boolean train_auto( @Const @ByRef Mat trainData, @Const @ByRef Mat responses,
                                @Const @ByRef Mat varIdx, @Const @ByRef Mat sampleIdx, @ByVal CvSVMParams params,
                                int k_fold/*=10*/,
                                @ByVal CvParamGrid Cgrid/*=CvSVM::get_default_grid(CvSVM::C)*/,
                                @ByVal CvParamGrid gammaGrid/*=CvSVM::get_default_grid(CvSVM::GAMMA)*/,
                                @ByVal CvParamGrid pGrid/*=CvSVM::get_default_grid(CvSVM::P)*/,
                                @ByVal CvParamGrid nuGrid/*=CvSVM::get_default_grid(CvSVM::NU)*/,
                                @ByVal CvParamGrid coeffGrid/*=CvSVM::get_default_grid(CvSVM::COEF)*/,
                                @ByVal CvParamGrid degreeGrid/*=CvSVM::get_default_grid(CvSVM::DEGREE)*/,
                                @Cast("bool") boolean balanced/*=false*/);
    public native float predict( @Const @ByRef Mat sample, @Cast("bool") boolean returnDFVal/*=false*/ ); public native void predict( @ByVal Mat samples, @ByVal Mat results );

    public native int get_support_vector_count();
    public native @Const FloatPointer get_support_vector(int i);
    public native @ByVal CvSVMParams get_params();
    public native void clear();

    public static native @ByVal CvParamGrid get_default_grid( int param_id );

    public native void write( CvFileStorage storage, @Cast("const char*") BytePointer name );
    public native void write( CvFileStorage storage, String name );
    public native void read( CvFileStorage storage, CvFileNode node );
    public native int get_var_count();
}

/****************************************************************************************\
*                              Expectation - Maximization                                *
\****************************************************************************************/
@Namespace("cv") @NoOffset public static class EM extends Algorithm {
    static { Loader.load(); }
    public EM() { }
    public EM(Pointer p) { super(p); }

    // Type of covariation matrices
    /** enum cv::EM:: */
    public static final int COV_MAT_SPHERICAL= 0, COV_MAT_DIAGONAL= 1, COV_MAT_GENERIC= 2, COV_MAT_DEFAULT= COV_MAT_DIAGONAL;

    // Default parameters
    /** enum cv::EM:: */
    public static final int DEFAULT_NCLUSTERS= 5, DEFAULT_MAX_ITERS= 100;

    // The initial step
    /** enum cv::EM:: */
    public static final int START_E_STEP= 1, START_M_STEP= 2, START_AUTO_STEP= 0;

    public EM(int nclusters/*=EM::DEFAULT_NCLUSTERS*/, int covMatType/*=EM::COV_MAT_DIAGONAL*/,
           @Const @ByRef TermCriteria termCrit/*=TermCriteria(TermCriteria::COUNT+TermCriteria::EPS,
                                                     EM::DEFAULT_MAX_ITERS, FLT_EPSILON)*/) { allocate(nclusters, covMatType, termCrit); }
    private native void allocate(int nclusters/*=EM::DEFAULT_NCLUSTERS*/, int covMatType/*=EM::COV_MAT_DIAGONAL*/,
           @Const @ByRef TermCriteria termCrit/*=TermCriteria(TermCriteria::COUNT+TermCriteria::EPS,
                                                     EM::DEFAULT_MAX_ITERS, FLT_EPSILON)*/);
    public native void clear();

    public native @Cast("bool") boolean train(@ByVal Mat samples,
                           @ByVal Mat logLikelihoods/*=noArray()*/,
                           @ByVal Mat labels/*=noArray()*/,
                           @ByVal Mat probs/*=noArray()*/);

    public native @Cast("bool") boolean trainE(@ByVal Mat samples,
                            @ByVal Mat means0,
                            @ByVal Mat covs0/*=noArray()*/,
                            @ByVal Mat weights0/*=noArray()*/,
                            @ByVal Mat logLikelihoods/*=noArray()*/,
                            @ByVal Mat labels/*=noArray()*/,
                            @ByVal Mat probs/*=noArray()*/);

    public native @Cast("bool") boolean trainM(@ByVal Mat samples,
                            @ByVal Mat probs0,
                            @ByVal Mat logLikelihoods/*=noArray()*/,
                            @ByVal Mat labels/*=noArray()*/,
                            @ByVal Mat probs/*=noArray()*/);

    public native @ByVal Point2d predict(@ByVal Mat sample,
                    @ByVal Mat probs/*=noArray()*/);

    public native @Cast("bool") boolean isTrained();

    public native AlgorithmInfo info();
    public native void read(@Const @ByRef FileNode fn);
}
 // namespace cv

/****************************************************************************************\
*                                      Decision Tree                                     *
\****************************************************************************************/
public static class CvPair16u32s extends Pointer {
    static { Loader.load(); }
    public CvPair16u32s() { allocate(); }
    public CvPair16u32s(int size) { allocateArray(size); }
    public CvPair16u32s(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public CvPair16u32s position(int position) {
        return (CvPair16u32s)super.position(position);
    }

    public native @Cast("unsigned short*") ShortPointer u(); public native CvPair16u32s u(ShortPointer u);
    public native IntPointer i(); public native CvPair16u32s i(IntPointer i);
}


// #define CV_DTREE_CAT_DIR(idx,subset)
//     (2*((subset[(idx)>>5]&(1 << ((idx) & 31)))==0)-1)

public static class CvDTreeSplit extends Pointer {
    static { Loader.load(); }
    public CvDTreeSplit() { allocate(); }
    public CvDTreeSplit(int size) { allocateArray(size); }
    public CvDTreeSplit(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public CvDTreeSplit position(int position) {
        return (CvDTreeSplit)super.position(position);
    }

    public native int var_idx(); public native CvDTreeSplit var_idx(int var_idx);
    public native int condensed_idx(); public native CvDTreeSplit condensed_idx(int condensed_idx);
    public native int inversed(); public native CvDTreeSplit inversed(int inversed);
    public native float quality(); public native CvDTreeSplit quality(float quality);
    public native CvDTreeSplit next(); public native CvDTreeSplit next(CvDTreeSplit next);
        public native int subset(int i); public native CvDTreeSplit subset(int i, int subset);
        @MemberGetter public native IntPointer subset();
            @Name("ord.c") public native float ord_c(); public native CvDTreeSplit ord_c(float ord_c);
            @Name("ord.split_point") public native int ord_split_point(); public native CvDTreeSplit ord_split_point(int ord_split_point);
}

public static class CvDTreeNode extends Pointer {
    static { Loader.load(); }
    public CvDTreeNode() { allocate(); }
    public CvDTreeNode(int size) { allocateArray(size); }
    public CvDTreeNode(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public CvDTreeNode position(int position) {
        return (CvDTreeNode)super.position(position);
    }

    public native int class_idx(); public native CvDTreeNode class_idx(int class_idx);
    public native int Tn(); public native CvDTreeNode Tn(int Tn);
    public native double value(); public native CvDTreeNode value(double value);

    public native CvDTreeNode parent(); public native CvDTreeNode parent(CvDTreeNode parent);
    public native CvDTreeNode left(); public native CvDTreeNode left(CvDTreeNode left);
    public native CvDTreeNode right(); public native CvDTreeNode right(CvDTreeNode right);

    public native CvDTreeSplit split(); public native CvDTreeNode split(CvDTreeSplit split);

    public native int sample_count(); public native CvDTreeNode sample_count(int sample_count);
    public native int depth(); public native CvDTreeNode depth(int depth);
    public native IntPointer num_valid(); public native CvDTreeNode num_valid(IntPointer num_valid);
    public native int offset(); public native CvDTreeNode offset(int offset);
    public native int buf_idx(); public native CvDTreeNode buf_idx(int buf_idx);
    public native double maxlr(); public native CvDTreeNode maxlr(double maxlr);

    // global pruning data
    public native int complexity(); public native CvDTreeNode complexity(int complexity);
    public native double alpha(); public native CvDTreeNode alpha(double alpha);
    public native double node_risk(); public native CvDTreeNode node_risk(double node_risk);
    public native double tree_risk(); public native CvDTreeNode tree_risk(double tree_risk);
    public native double tree_error(); public native CvDTreeNode tree_error(double tree_error);

    // cross-validation pruning data
    public native IntPointer cv_Tn(); public native CvDTreeNode cv_Tn(IntPointer cv_Tn);
    public native DoublePointer cv_node_risk(); public native CvDTreeNode cv_node_risk(DoublePointer cv_node_risk);
    public native DoublePointer cv_node_error(); public native CvDTreeNode cv_node_error(DoublePointer cv_node_error);

    public native int get_num_valid(int vi);
    public native void set_num_valid(int vi, int n);
}


@NoOffset public static class CvDTreeParams extends Pointer {
    static { Loader.load(); }
    public CvDTreeParams(Pointer p) { super(p); }
    public CvDTreeParams(int size) { allocateArray(size); }
    private native void allocateArray(int size);
    @Override public CvDTreeParams position(int position) {
        return (CvDTreeParams)super.position(position);
    }

    public native int max_categories(); public native CvDTreeParams max_categories(int max_categories);
    public native int max_depth(); public native CvDTreeParams max_depth(int max_depth);
    public native int min_sample_count(); public native CvDTreeParams min_sample_count(int min_sample_count);
    public native int cv_folds(); public native CvDTreeParams cv_folds(int cv_folds);
    public native @Cast("bool") boolean use_surrogates(); public native CvDTreeParams use_surrogates(boolean use_surrogates);
    public native @Cast("bool") boolean use_1se_rule(); public native CvDTreeParams use_1se_rule(boolean use_1se_rule);
    public native @Cast("bool") boolean truncate_pruned_tree(); public native CvDTreeParams truncate_pruned_tree(boolean truncate_pruned_tree);
    public native float regression_accuracy(); public native CvDTreeParams regression_accuracy(float regression_accuracy);
    @MemberGetter public native @Const FloatPointer priors();

    public CvDTreeParams() { allocate(); }
    private native void allocate();
    public CvDTreeParams( int max_depth, int min_sample_count,
                       float regression_accuracy, @Cast("bool") boolean use_surrogates,
                       int max_categories, int cv_folds,
                       @Cast("bool") boolean use_1se_rule, @Cast("bool") boolean truncate_pruned_tree,
                       @Const FloatPointer priors ) { allocate(max_depth, min_sample_count, regression_accuracy, use_surrogates, max_categories, cv_folds, use_1se_rule, truncate_pruned_tree, priors); }
    private native void allocate( int max_depth, int min_sample_count,
                       float regression_accuracy, @Cast("bool") boolean use_surrogates,
                       int max_categories, int cv_folds,
                       @Cast("bool") boolean use_1se_rule, @Cast("bool") boolean truncate_pruned_tree,
                       @Const FloatPointer priors );
    public CvDTreeParams( int max_depth, int min_sample_count,
                       float regression_accuracy, @Cast("bool") boolean use_surrogates,
                       int max_categories, int cv_folds,
                       @Cast("bool") boolean use_1se_rule, @Cast("bool") boolean truncate_pruned_tree,
                       @Const FloatBuffer priors ) { allocate(max_depth, min_sample_count, regression_accuracy, use_surrogates, max_categories, cv_folds, use_1se_rule, truncate_pruned_tree, priors); }
    private native void allocate( int max_depth, int min_sample_count,
                       float regression_accuracy, @Cast("bool") boolean use_surrogates,
                       int max_categories, int cv_folds,
                       @Cast("bool") boolean use_1se_rule, @Cast("bool") boolean truncate_pruned_tree,
                       @Const FloatBuffer priors );
    public CvDTreeParams( int max_depth, int min_sample_count,
                       float regression_accuracy, @Cast("bool") boolean use_surrogates,
                       int max_categories, int cv_folds,
                       @Cast("bool") boolean use_1se_rule, @Cast("bool") boolean truncate_pruned_tree,
                       @Const float[] priors ) { allocate(max_depth, min_sample_count, regression_accuracy, use_surrogates, max_categories, cv_folds, use_1se_rule, truncate_pruned_tree, priors); }
    private native void allocate( int max_depth, int min_sample_count,
                       float regression_accuracy, @Cast("bool") boolean use_surrogates,
                       int max_categories, int cv_folds,
                       @Cast("bool") boolean use_1se_rule, @Cast("bool") boolean truncate_pruned_tree,
                       @Const float[] priors );
}


@NoOffset public static class CvDTreeTrainData extends Pointer {
    static { Loader.load(); }
    public CvDTreeTrainData(Pointer p) { super(p); }
    public CvDTreeTrainData(int size) { allocateArray(size); }
    private native void allocateArray(int size);
    @Override public CvDTreeTrainData position(int position) {
        return (CvDTreeTrainData)super.position(position);
    }

    public CvDTreeTrainData() { allocate(); }
    private native void allocate();
    public CvDTreeTrainData( @Const CvMat trainData, int tflag,
                          @Const CvMat responses, @Const CvMat varIdx/*=0*/,
                          @Const CvMat sampleIdx/*=0*/, @Const CvMat varType/*=0*/,
                          @Const CvMat missingDataMask/*=0*/,
                          @Const @ByRef CvDTreeParams params/*=CvDTreeParams()*/,
                          @Cast("bool") boolean _shared/*=false*/, @Cast("bool") boolean _add_labels/*=false*/ ) { allocate(trainData, tflag, responses, varIdx, sampleIdx, varType, missingDataMask, params, _shared, _add_labels); }
    private native void allocate( @Const CvMat trainData, int tflag,
                          @Const CvMat responses, @Const CvMat varIdx/*=0*/,
                          @Const CvMat sampleIdx/*=0*/, @Const CvMat varType/*=0*/,
                          @Const CvMat missingDataMask/*=0*/,
                          @Const @ByRef CvDTreeParams params/*=CvDTreeParams()*/,
                          @Cast("bool") boolean _shared/*=false*/, @Cast("bool") boolean _add_labels/*=false*/ );

    public native void set_data( @Const CvMat trainData, int tflag,
                              @Const CvMat responses, @Const CvMat varIdx/*=0*/,
                              @Const CvMat sampleIdx/*=0*/, @Const CvMat varType/*=0*/,
                              @Const CvMat missingDataMask/*=0*/,
                              @Const @ByRef CvDTreeParams params/*=CvDTreeParams()*/,
                              @Cast("bool") boolean _shared/*=false*/, @Cast("bool") boolean _add_labels/*=false*/,
                              @Cast("bool") boolean _update_data/*=false*/ );
    public native void do_responses_copy();

    public native void get_vectors( @Const CvMat _subsample_idx,
             FloatPointer values, @Cast("uchar*") BytePointer missing, FloatPointer responses, @Cast("bool") boolean get_class_idx/*=false*/ );
    public native void get_vectors( @Const CvMat _subsample_idx,
             FloatBuffer values, @Cast("uchar*") ByteBuffer missing, FloatBuffer responses, @Cast("bool") boolean get_class_idx/*=false*/ );
    public native void get_vectors( @Const CvMat _subsample_idx,
             float[] values, @Cast("uchar*") byte[] missing, float[] responses, @Cast("bool") boolean get_class_idx/*=false*/ );

    public native CvDTreeNode subsample_data( @Const CvMat _subsample_idx );

    public native void write_params( CvFileStorage fs );
    public native void read_params( CvFileStorage fs, CvFileNode node );

    // release all the data
    public native void clear();

    public native int get_num_classes();
    public native int get_var_type(int vi);
    public native int get_work_var_count();

    public native @Const FloatPointer get_ord_responses( CvDTreeNode n, FloatPointer values_buf, IntPointer sample_indices_buf );
    public native @Const FloatBuffer get_ord_responses( CvDTreeNode n, FloatBuffer values_buf, IntBuffer sample_indices_buf );
    public native @Const float[] get_ord_responses( CvDTreeNode n, float[] values_buf, int[] sample_indices_buf );
    public native @Const IntPointer get_class_labels( CvDTreeNode n, IntPointer labels_buf );
    public native @Const IntBuffer get_class_labels( CvDTreeNode n, IntBuffer labels_buf );
    public native @Const int[] get_class_labels( CvDTreeNode n, int[] labels_buf );
    public native @Const IntPointer get_cv_labels( CvDTreeNode n, IntPointer labels_buf );
    public native @Const IntBuffer get_cv_labels( CvDTreeNode n, IntBuffer labels_buf );
    public native @Const int[] get_cv_labels( CvDTreeNode n, int[] labels_buf );
    public native @Const IntPointer get_sample_indices( CvDTreeNode n, IntPointer indices_buf );
    public native @Const IntBuffer get_sample_indices( CvDTreeNode n, IntBuffer indices_buf );
    public native @Const int[] get_sample_indices( CvDTreeNode n, int[] indices_buf );
    public native @Const IntPointer get_cat_var_data( CvDTreeNode n, int vi, IntPointer cat_values_buf );
    public native @Const IntBuffer get_cat_var_data( CvDTreeNode n, int vi, IntBuffer cat_values_buf );
    public native @Const int[] get_cat_var_data( CvDTreeNode n, int vi, int[] cat_values_buf );
    public native void get_ord_var_data( CvDTreeNode n, int vi, FloatPointer ord_values_buf, IntPointer sorted_indices_buf,
                                       @Cast("const float**") PointerPointer ord_values, @Cast("const int**") PointerPointer sorted_indices, IntPointer sample_indices_buf );
    public native void get_ord_var_data( CvDTreeNode n, int vi, FloatPointer ord_values_buf, IntPointer sorted_indices_buf,
                                       @Const @ByPtrPtr FloatPointer ord_values, @Const @ByPtrPtr IntPointer sorted_indices, IntPointer sample_indices_buf );
    public native void get_ord_var_data( CvDTreeNode n, int vi, FloatBuffer ord_values_buf, IntBuffer sorted_indices_buf,
                                       @Const @ByPtrPtr FloatBuffer ord_values, @Const @ByPtrPtr IntBuffer sorted_indices, IntBuffer sample_indices_buf );
    public native void get_ord_var_data( CvDTreeNode n, int vi, float[] ord_values_buf, int[] sorted_indices_buf,
                                       @Const @ByPtrPtr float[] ord_values, @Const @ByPtrPtr int[] sorted_indices, int[] sample_indices_buf );
    public native int get_child_buf_idx( CvDTreeNode n );

    ////////////////////////////////////

    public native @Cast("bool") boolean set_params( @Const @ByRef CvDTreeParams params );
    public native CvDTreeNode new_node( CvDTreeNode parent, int count,
                                       int storage_idx, int offset );

    public native CvDTreeSplit new_split_ord( int vi, float cmp_val,
                    int split_point, int inversed, float quality );
    public native CvDTreeSplit new_split_cat( int vi, float quality );
    public native void free_node_data( CvDTreeNode node );
    public native void free_train_data();
    public native void free_node( CvDTreeNode node );

    public native int sample_count(); public native CvDTreeTrainData sample_count(int sample_count);
    public native int var_all(); public native CvDTreeTrainData var_all(int var_all);
    public native int var_count(); public native CvDTreeTrainData var_count(int var_count);
    public native int max_c_count(); public native CvDTreeTrainData max_c_count(int max_c_count);
    public native int ord_var_count(); public native CvDTreeTrainData ord_var_count(int ord_var_count);
    public native int cat_var_count(); public native CvDTreeTrainData cat_var_count(int cat_var_count);
    public native int work_var_count(); public native CvDTreeTrainData work_var_count(int work_var_count);
    public native @Cast("bool") boolean have_labels(); public native CvDTreeTrainData have_labels(boolean have_labels);
    public native @Cast("bool") boolean have_priors(); public native CvDTreeTrainData have_priors(boolean have_priors);
    public native @Cast("bool") boolean is_classifier(); public native CvDTreeTrainData is_classifier(boolean is_classifier);
    public native int tflag(); public native CvDTreeTrainData tflag(int tflag);

    @MemberGetter public native @Const CvMat train_data();
    @MemberGetter public native @Const CvMat responses();
    public native CvMat responses_copy(); public native CvDTreeTrainData responses_copy(CvMat responses_copy); // used in Boosting

    public native int buf_count(); public native CvDTreeTrainData buf_count(int buf_count);
    public native int buf_size(); public native CvDTreeTrainData buf_size(int buf_size); // buf_size is obsolete, please do not use it, use expression ((int64)buf->rows * (int64)buf->cols / buf_count) instead
    public native @Cast("bool") boolean shared(); public native CvDTreeTrainData shared(boolean shared);
    public native int is_buf_16u(); public native CvDTreeTrainData is_buf_16u(int is_buf_16u);

    public native CvMat cat_count(); public native CvDTreeTrainData cat_count(CvMat cat_count);
    public native CvMat cat_ofs(); public native CvDTreeTrainData cat_ofs(CvMat cat_ofs);
    public native CvMat cat_map(); public native CvDTreeTrainData cat_map(CvMat cat_map);

    public native CvMat counts(); public native CvDTreeTrainData counts(CvMat counts);
    public native CvMat buf(); public native CvDTreeTrainData buf(CvMat buf);
    public native @Cast("size_t") long get_length_subbuf();

    public native CvMat direction(); public native CvDTreeTrainData direction(CvMat direction);
    public native CvMat split_buf(); public native CvDTreeTrainData split_buf(CvMat split_buf);

    public native CvMat var_idx(); public native CvDTreeTrainData var_idx(CvMat var_idx);
    public native CvMat var_type(); public native CvDTreeTrainData var_type(CvMat var_type); // i-th element =
                     //   k<0  - ordered
                     //   k>=0 - categorical, see k-th element of cat_* arrays
    public native CvMat priors(); public native CvDTreeTrainData priors(CvMat priors);
    public native CvMat priors_mult(); public native CvDTreeTrainData priors_mult(CvMat priors_mult);

    public native @ByVal CvDTreeParams params(); public native CvDTreeTrainData params(CvDTreeParams params);

    public native CvMemStorage tree_storage(); public native CvDTreeTrainData tree_storage(CvMemStorage tree_storage);
    public native CvMemStorage temp_storage(); public native CvDTreeTrainData temp_storage(CvMemStorage temp_storage);

    public native CvDTreeNode data_root(); public native CvDTreeTrainData data_root(CvDTreeNode data_root);

    public native CvSet node_heap(); public native CvDTreeTrainData node_heap(CvSet node_heap);
    public native CvSet split_heap(); public native CvDTreeTrainData split_heap(CvSet split_heap);
    public native CvSet cv_heap(); public native CvDTreeTrainData cv_heap(CvSet cv_heap);
    public native CvSet nv_heap(); public native CvDTreeTrainData nv_heap(CvSet nv_heap);

    public native RNG rng(); public native CvDTreeTrainData rng(RNG rng);
}
    @Namespace("cv") @Opaque public static class DTreeBestSplitFinder extends Pointer {
        public DTreeBestSplitFinder() { }
        public DTreeBestSplitFinder(Pointer p) { super(p); }
    }
    @Namespace("cv") @Opaque public static class ForestTreeBestSplitFinder extends Pointer {
        public ForestTreeBestSplitFinder() { }
        public ForestTreeBestSplitFinder(Pointer p) { super(p); }
    }


@NoOffset public static class CvDTree extends CvStatModel {
    static { Loader.load(); }
    public CvDTree(Pointer p) { super(p); }
    public CvDTree(int size) { allocateArray(size); }
    private native void allocateArray(int size);
    @Override public CvDTree position(int position) {
        return (CvDTree)super.position(position);
    }

    public CvDTree() { allocate(); }
    private native void allocate();

    public native @Cast("bool") boolean train( @Const CvMat trainData, int tflag,
                            @Const CvMat responses, @Const CvMat varIdx/*=0*/,
                            @Const CvMat sampleIdx/*=0*/, @Const CvMat varType/*=0*/,
                            @Const CvMat missingDataMask/*=0*/,
                            @ByVal CvDTreeParams params/*=CvDTreeParams()*/ );

    public native @Cast("bool") boolean train( CvMLData trainData, @ByVal CvDTreeParams params/*=CvDTreeParams()*/ );

    // type in {CV_TRAIN_ERROR, CV_TEST_ERROR}
    public native float calc_error( CvMLData trainData, int type, @StdVector FloatPointer resp/*=0*/ );
    public native float calc_error( CvMLData trainData, int type, @StdVector FloatBuffer resp/*=0*/ );
    public native float calc_error( CvMLData trainData, int type, @StdVector float[] resp/*=0*/ );

    public native @Cast("bool") boolean train( CvDTreeTrainData trainData, @Const CvMat subsampleIdx );

    public native CvDTreeNode predict( @Const CvMat sample, @Const CvMat missingDataMask/*=0*/,
                                      @Cast("bool") boolean preprocessedInput/*=false*/ );

    public native @Cast("bool") boolean train( @Const @ByRef Mat trainData, int tflag,
                           @Const @ByRef Mat responses, @Const @ByRef Mat varIdx/*=cv::Mat()*/,
                           @Const @ByRef Mat sampleIdx/*=cv::Mat()*/, @Const @ByRef Mat varType/*=cv::Mat()*/,
                           @Const @ByRef Mat missingDataMask/*=cv::Mat()*/,
                           @ByVal CvDTreeParams params/*=CvDTreeParams()*/ );

    public native CvDTreeNode predict( @Const @ByRef Mat sample, @Const @ByRef Mat missingDataMask/*=cv::Mat()*/,
                                      @Cast("bool") boolean preprocessedInput/*=false*/ );
    public native @ByVal Mat getVarImportance();

    public native @Const CvMat get_var_importance();
    public native void clear();

    public native void read( CvFileStorage fs, CvFileNode node );
    public native void write( CvFileStorage fs, @Cast("const char*") BytePointer name );
    public native void write( CvFileStorage fs, String name );

    // special read & write methods for trees in the tree ensembles
    public native void read( CvFileStorage fs, CvFileNode node,
                           CvDTreeTrainData data );
    public native void write( CvFileStorage fs );

    public native @Const CvDTreeNode get_root();
    public native int get_pruned_tree_idx();
    public native CvDTreeTrainData get_data();
    public native int pruned_tree_idx(); public native CvDTree pruned_tree_idx(int pruned_tree_idx);
}


/****************************************************************************************\
*                                   Random Trees Classifier                              *
\****************************************************************************************/

@NoOffset public static class CvForestTree extends CvDTree {
    static { Loader.load(); }
    public CvForestTree(Pointer p) { super(p); }
    public CvForestTree(int size) { allocateArray(size); }
    private native void allocateArray(int size);
    @Override public CvForestTree position(int position) {
        return (CvForestTree)super.position(position);
    }

    public CvForestTree() { allocate(); }
    private native void allocate();

    public native @Cast("bool") boolean train( CvDTreeTrainData trainData, @Const CvMat _subsample_idx, CvRTrees forest );

    public native int get_var_count();
    public native void read( CvFileStorage fs, CvFileNode node, CvRTrees forest, CvDTreeTrainData _data );

    /* dummy methods to avoid warnings: BEGIN */
    public native @Cast("bool") boolean train( @Const CvMat trainData, int tflag,
                            @Const CvMat responses, @Const CvMat varIdx/*=0*/,
                            @Const CvMat sampleIdx/*=0*/, @Const CvMat varType/*=0*/,
                            @Const CvMat missingDataMask/*=0*/,
                            @ByVal CvDTreeParams params/*=CvDTreeParams()*/ );

    public native @Cast("bool") boolean train( CvDTreeTrainData trainData, @Const CvMat _subsample_idx );
    public native void read( CvFileStorage fs, CvFileNode node );
    public native void read( CvFileStorage fs, CvFileNode node,
                           CvDTreeTrainData data );
}


@NoOffset public static class CvRTParams extends CvDTreeParams {
    static { Loader.load(); }
    public CvRTParams(Pointer p) { super(p); }
    public CvRTParams(int size) { allocateArray(size); }
    private native void allocateArray(int size);
    @Override public CvRTParams position(int position) {
        return (CvRTParams)super.position(position);
    }

    //Parameters for the forest
    public native @Cast("bool") boolean calc_var_importance(); public native CvRTParams calc_var_importance(boolean calc_var_importance); // true <=> RF processes variable importance
    public native int nactive_vars(); public native CvRTParams nactive_vars(int nactive_vars);
    public native @ByVal CvTermCriteria term_crit(); public native CvRTParams term_crit(CvTermCriteria term_crit);

    public CvRTParams() { allocate(); }
    private native void allocate();
    public CvRTParams( int max_depth, int min_sample_count,
                    float regression_accuracy, @Cast("bool") boolean use_surrogates,
                    int max_categories, @Const FloatPointer priors, @Cast("bool") boolean calc_var_importance,
                    int nactive_vars, int max_num_of_trees_in_the_forest,
                    float forest_accuracy, int termcrit_type ) { allocate(max_depth, min_sample_count, regression_accuracy, use_surrogates, max_categories, priors, calc_var_importance, nactive_vars, max_num_of_trees_in_the_forest, forest_accuracy, termcrit_type); }
    private native void allocate( int max_depth, int min_sample_count,
                    float regression_accuracy, @Cast("bool") boolean use_surrogates,
                    int max_categories, @Const FloatPointer priors, @Cast("bool") boolean calc_var_importance,
                    int nactive_vars, int max_num_of_trees_in_the_forest,
                    float forest_accuracy, int termcrit_type );
    public CvRTParams( int max_depth, int min_sample_count,
                    float regression_accuracy, @Cast("bool") boolean use_surrogates,
                    int max_categories, @Const FloatBuffer priors, @Cast("bool") boolean calc_var_importance,
                    int nactive_vars, int max_num_of_trees_in_the_forest,
                    float forest_accuracy, int termcrit_type ) { allocate(max_depth, min_sample_count, regression_accuracy, use_surrogates, max_categories, priors, calc_var_importance, nactive_vars, max_num_of_trees_in_the_forest, forest_accuracy, termcrit_type); }
    private native void allocate( int max_depth, int min_sample_count,
                    float regression_accuracy, @Cast("bool") boolean use_surrogates,
                    int max_categories, @Const FloatBuffer priors, @Cast("bool") boolean calc_var_importance,
                    int nactive_vars, int max_num_of_trees_in_the_forest,
                    float forest_accuracy, int termcrit_type );
    public CvRTParams( int max_depth, int min_sample_count,
                    float regression_accuracy, @Cast("bool") boolean use_surrogates,
                    int max_categories, @Const float[] priors, @Cast("bool") boolean calc_var_importance,
                    int nactive_vars, int max_num_of_trees_in_the_forest,
                    float forest_accuracy, int termcrit_type ) { allocate(max_depth, min_sample_count, regression_accuracy, use_surrogates, max_categories, priors, calc_var_importance, nactive_vars, max_num_of_trees_in_the_forest, forest_accuracy, termcrit_type); }
    private native void allocate( int max_depth, int min_sample_count,
                    float regression_accuracy, @Cast("bool") boolean use_surrogates,
                    int max_categories, @Const float[] priors, @Cast("bool") boolean calc_var_importance,
                    int nactive_vars, int max_num_of_trees_in_the_forest,
                    float forest_accuracy, int termcrit_type );
}


@NoOffset public static class CvRTrees extends CvStatModel {
    static { Loader.load(); }
    public CvRTrees(Pointer p) { super(p); }
    public CvRTrees(int size) { allocateArray(size); }
    private native void allocateArray(int size);
    @Override public CvRTrees position(int position) {
        return (CvRTrees)super.position(position);
    }

    public CvRTrees() { allocate(); }
    private native void allocate();
    public native @Cast("bool") boolean train( @Const CvMat trainData, int tflag,
                            @Const CvMat responses, @Const CvMat varIdx/*=0*/,
                            @Const CvMat sampleIdx/*=0*/, @Const CvMat varType/*=0*/,
                            @Const CvMat missingDataMask/*=0*/,
                            @ByVal CvRTParams params/*=CvRTParams()*/ );

    public native @Cast("bool") boolean train( CvMLData data, @ByVal CvRTParams params/*=CvRTParams()*/ );
    public native float predict( @Const CvMat sample, @Const CvMat missing/*=0*/ );
    public native float predict_prob( @Const CvMat sample, @Const CvMat missing/*=0*/ );

    public native @Cast("bool") boolean train( @Const @ByRef Mat trainData, int tflag,
                           @Const @ByRef Mat responses, @Const @ByRef Mat varIdx/*=cv::Mat()*/,
                           @Const @ByRef Mat sampleIdx/*=cv::Mat()*/, @Const @ByRef Mat varType/*=cv::Mat()*/,
                           @Const @ByRef Mat missingDataMask/*=cv::Mat()*/,
                           @ByVal CvRTParams params/*=CvRTParams()*/ );
    public native float predict( @Const @ByRef Mat sample, @Const @ByRef Mat missing/*=cv::Mat()*/ );
    public native float predict_prob( @Const @ByRef Mat sample, @Const @ByRef Mat missing/*=cv::Mat()*/ );
    public native @ByVal Mat getVarImportance();

    public native void clear();

    public native @Const CvMat get_var_importance();
    public native float get_proximity( @Const CvMat sample1, @Const CvMat sample2,
            @Const CvMat missing1/*=0*/, @Const CvMat missing2/*=0*/ );

    public native float calc_error( CvMLData data, int type, @StdVector FloatPointer resp/*=0*/ );
    public native float calc_error( CvMLData data, int type, @StdVector FloatBuffer resp/*=0*/ );
    public native float calc_error( CvMLData data, int type, @StdVector float[] resp/*=0*/ ); // type in {CV_TRAIN_ERROR, CV_TEST_ERROR}

    public native float get_train_error();

    public native void read( CvFileStorage fs, CvFileNode node );
    public native void write( CvFileStorage fs, @Cast("const char*") BytePointer name );
    public native void write( CvFileStorage fs, String name );

    public native CvMat get_active_var_mask();
    public native @Cast("CvRNG*") LongPointer get_rng();

    public native int get_tree_count();
    public native CvForestTree get_tree(int i);
}

/****************************************************************************************\
*                           Extremely randomized trees Classifier                        *
\****************************************************************************************/
@NoOffset public static class CvERTreeTrainData extends CvDTreeTrainData {
    static { Loader.load(); }
    public CvERTreeTrainData() { allocate(); }
    public CvERTreeTrainData(int size) { allocateArray(size); }
    public CvERTreeTrainData(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public CvERTreeTrainData position(int position) {
        return (CvERTreeTrainData)super.position(position);
    }

    public native void set_data( @Const CvMat trainData, int tflag,
                              @Const CvMat responses, @Const CvMat varIdx/*=0*/,
                              @Const CvMat sampleIdx/*=0*/, @Const CvMat varType/*=0*/,
                              @Const CvMat missingDataMask/*=0*/,
                              @Const @ByRef CvDTreeParams params/*=CvDTreeParams()*/,
                              @Cast("bool") boolean _shared/*=false*/, @Cast("bool") boolean _add_labels/*=false*/,
                              @Cast("bool") boolean _update_data/*=false*/ );
    public native void get_ord_var_data( CvDTreeNode n, int vi, FloatPointer ord_values_buf, IntPointer missing_buf,
                                       @Cast("const float**") PointerPointer ord_values, @Cast("const int**") PointerPointer missing, IntPointer sample_buf/*=0*/ );
    public native void get_ord_var_data( CvDTreeNode n, int vi, FloatPointer ord_values_buf, IntPointer missing_buf,
                                       @Const @ByPtrPtr FloatPointer ord_values, @Const @ByPtrPtr IntPointer missing, IntPointer sample_buf/*=0*/ );
    public native void get_ord_var_data( CvDTreeNode n, int vi, FloatBuffer ord_values_buf, IntBuffer missing_buf,
                                       @Const @ByPtrPtr FloatBuffer ord_values, @Const @ByPtrPtr IntBuffer missing, IntBuffer sample_buf/*=0*/ );
    public native void get_ord_var_data( CvDTreeNode n, int vi, float[] ord_values_buf, int[] missing_buf,
                                       @Const @ByPtrPtr float[] ord_values, @Const @ByPtrPtr int[] missing, int[] sample_buf/*=0*/ );
    public native @Const IntPointer get_sample_indices( CvDTreeNode n, IntPointer indices_buf );
    public native @Const IntBuffer get_sample_indices( CvDTreeNode n, IntBuffer indices_buf );
    public native @Const int[] get_sample_indices( CvDTreeNode n, int[] indices_buf );
    public native @Const IntPointer get_cv_labels( CvDTreeNode n, IntPointer labels_buf );
    public native @Const IntBuffer get_cv_labels( CvDTreeNode n, IntBuffer labels_buf );
    public native @Const int[] get_cv_labels( CvDTreeNode n, int[] labels_buf );
    public native @Const IntPointer get_cat_var_data( CvDTreeNode n, int vi, IntPointer cat_values_buf );
    public native @Const IntBuffer get_cat_var_data( CvDTreeNode n, int vi, IntBuffer cat_values_buf );
    public native @Const int[] get_cat_var_data( CvDTreeNode n, int vi, int[] cat_values_buf );
    public native void get_vectors( @Const CvMat _subsample_idx, FloatPointer values, @Cast("uchar*") BytePointer missing,
                                  FloatPointer responses, @Cast("bool") boolean get_class_idx/*=false*/ );
    public native void get_vectors( @Const CvMat _subsample_idx, FloatBuffer values, @Cast("uchar*") ByteBuffer missing,
                                  FloatBuffer responses, @Cast("bool") boolean get_class_idx/*=false*/ );
    public native void get_vectors( @Const CvMat _subsample_idx, float[] values, @Cast("uchar*") byte[] missing,
                                  float[] responses, @Cast("bool") boolean get_class_idx/*=false*/ );
    public native CvDTreeNode subsample_data( @Const CvMat _subsample_idx );
    @MemberGetter public native @Const CvMat missing_mask();
}

public static class CvForestERTree extends CvForestTree {
    static { Loader.load(); }
    public CvForestERTree() { allocate(); }
    public CvForestERTree(int size) { allocateArray(size); }
    public CvForestERTree(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public CvForestERTree position(int position) {
        return (CvForestERTree)super.position(position);
    }

}

public static class CvERTrees extends CvRTrees {
    static { Loader.load(); }
    public CvERTrees(Pointer p) { super(p); }
    public CvERTrees(int size) { allocateArray(size); }
    private native void allocateArray(int size);
    @Override public CvERTrees position(int position) {
        return (CvERTrees)super.position(position);
    }

    public CvERTrees() { allocate(); }
    private native void allocate();
    public native @Cast("bool") boolean train( @Const CvMat trainData, int tflag,
                            @Const CvMat responses, @Const CvMat varIdx/*=0*/,
                            @Const CvMat sampleIdx/*=0*/, @Const CvMat varType/*=0*/,
                            @Const CvMat missingDataMask/*=0*/,
                            @ByVal CvRTParams params/*=CvRTParams()*/);
    public native @Cast("bool") boolean train( @Const @ByRef Mat trainData, int tflag,
                           @Const @ByRef Mat responses, @Const @ByRef Mat varIdx/*=cv::Mat()*/,
                           @Const @ByRef Mat sampleIdx/*=cv::Mat()*/, @Const @ByRef Mat varType/*=cv::Mat()*/,
                           @Const @ByRef Mat missingDataMask/*=cv::Mat()*/,
                           @ByVal CvRTParams params/*=CvRTParams()*/);
    public native @Cast("bool") boolean train( CvMLData data, @ByVal CvRTParams params/*=CvRTParams()*/ );
}


/****************************************************************************************\
*                                   Boosted tree classifier                              *
\****************************************************************************************/

@NoOffset public static class CvBoostParams extends CvDTreeParams {
    static { Loader.load(); }
    public CvBoostParams(Pointer p) { super(p); }
    public CvBoostParams(int size) { allocateArray(size); }
    private native void allocateArray(int size);
    @Override public CvBoostParams position(int position) {
        return (CvBoostParams)super.position(position);
    }

    public native int boost_type(); public native CvBoostParams boost_type(int boost_type);
    public native int weak_count(); public native CvBoostParams weak_count(int weak_count);
    public native int split_criteria(); public native CvBoostParams split_criteria(int split_criteria);
    public native double weight_trim_rate(); public native CvBoostParams weight_trim_rate(double weight_trim_rate);

    public CvBoostParams() { allocate(); }
    private native void allocate();
    public CvBoostParams( int boost_type, int weak_count, double weight_trim_rate,
                       int max_depth, @Cast("bool") boolean use_surrogates, @Const FloatPointer priors ) { allocate(boost_type, weak_count, weight_trim_rate, max_depth, use_surrogates, priors); }
    private native void allocate( int boost_type, int weak_count, double weight_trim_rate,
                       int max_depth, @Cast("bool") boolean use_surrogates, @Const FloatPointer priors );
    public CvBoostParams( int boost_type, int weak_count, double weight_trim_rate,
                       int max_depth, @Cast("bool") boolean use_surrogates, @Const FloatBuffer priors ) { allocate(boost_type, weak_count, weight_trim_rate, max_depth, use_surrogates, priors); }
    private native void allocate( int boost_type, int weak_count, double weight_trim_rate,
                       int max_depth, @Cast("bool") boolean use_surrogates, @Const FloatBuffer priors );
    public CvBoostParams( int boost_type, int weak_count, double weight_trim_rate,
                       int max_depth, @Cast("bool") boolean use_surrogates, @Const float[] priors ) { allocate(boost_type, weak_count, weight_trim_rate, max_depth, use_surrogates, priors); }
    private native void allocate( int boost_type, int weak_count, double weight_trim_rate,
                       int max_depth, @Cast("bool") boolean use_surrogates, @Const float[] priors );
}

@NoOffset public static class CvBoostTree extends CvDTree {
    static { Loader.load(); }
    public CvBoostTree(Pointer p) { super(p); }
    public CvBoostTree(int size) { allocateArray(size); }
    private native void allocateArray(int size);
    @Override public CvBoostTree position(int position) {
        return (CvBoostTree)super.position(position);
    }

    public CvBoostTree() { allocate(); }
    private native void allocate();

    public native @Cast("bool") boolean train( CvDTreeTrainData trainData,
                            @Const CvMat subsample_idx, CvBoost ensemble );

    public native void scale( double s );
    public native void read( CvFileStorage fs, CvFileNode node,
                           CvBoost ensemble, CvDTreeTrainData _data );
    public native void clear();

    /* dummy methods to avoid warnings: BEGIN */
    public native @Cast("bool") boolean train( @Const CvMat trainData, int tflag,
                            @Const CvMat responses, @Const CvMat varIdx/*=0*/,
                            @Const CvMat sampleIdx/*=0*/, @Const CvMat varType/*=0*/,
                            @Const CvMat missingDataMask/*=0*/,
                            @ByVal CvDTreeParams params/*=CvDTreeParams()*/ );
    public native @Cast("bool") boolean train( CvDTreeTrainData trainData, @Const CvMat _subsample_idx );

    public native void read( CvFileStorage fs, CvFileNode node );
    public native void read( CvFileStorage fs, CvFileNode node,
                           CvDTreeTrainData data );
}


@NoOffset public static class CvBoost extends CvStatModel {
    static { Loader.load(); }
    public CvBoost(Pointer p) { super(p); }
    public CvBoost(int size) { allocateArray(size); }
    private native void allocateArray(int size);
    @Override public CvBoost position(int position) {
        return (CvBoost)super.position(position);
    }

    // Boosting type
    /** enum CvBoost:: */
    public static final int DISCRETE= 0, REAL= 1, LOGIT= 2, GENTLE= 3;

    // Splitting criteria
    /** enum CvBoost:: */
    public static final int DEFAULT= 0, GINI= 1, MISCLASS= 3, SQERR= 4;

    public CvBoost() { allocate(); }
    private native void allocate();

    public CvBoost( @Const CvMat trainData, int tflag,
                 @Const CvMat responses, @Const CvMat varIdx/*=0*/,
                 @Const CvMat sampleIdx/*=0*/, @Const CvMat varType/*=0*/,
                 @Const CvMat missingDataMask/*=0*/,
                 @ByVal CvBoostParams params/*=CvBoostParams()*/ ) { allocate(trainData, tflag, responses, varIdx, sampleIdx, varType, missingDataMask, params); }
    private native void allocate( @Const CvMat trainData, int tflag,
                 @Const CvMat responses, @Const CvMat varIdx/*=0*/,
                 @Const CvMat sampleIdx/*=0*/, @Const CvMat varType/*=0*/,
                 @Const CvMat missingDataMask/*=0*/,
                 @ByVal CvBoostParams params/*=CvBoostParams()*/ );

    public native @Cast("bool") boolean train( @Const CvMat trainData, int tflag,
                 @Const CvMat responses, @Const CvMat varIdx/*=0*/,
                 @Const CvMat sampleIdx/*=0*/, @Const CvMat varType/*=0*/,
                 @Const CvMat missingDataMask/*=0*/,
                 @ByVal CvBoostParams params/*=CvBoostParams()*/,
                 @Cast("bool") boolean update/*=false*/ );

    public native @Cast("bool") boolean train( CvMLData data,
                 @ByVal CvBoostParams params/*=CvBoostParams()*/,
                 @Cast("bool") boolean update/*=false*/ );

    public native float predict( @Const CvMat sample, @Const CvMat missing/*=0*/,
                               CvMat weak_responses/*=0*/, @ByVal CvSlice slice/*=CV_WHOLE_SEQ*/,
                               @Cast("bool") boolean raw_mode/*=false*/, @Cast("bool") boolean return_sum/*=false*/ );

    public CvBoost( @Const @ByRef Mat trainData, int tflag,
                @Const @ByRef Mat responses, @Const @ByRef Mat varIdx/*=cv::Mat()*/,
                @Const @ByRef Mat sampleIdx/*=cv::Mat()*/, @Const @ByRef Mat varType/*=cv::Mat()*/,
                @Const @ByRef Mat missingDataMask/*=cv::Mat()*/,
                @ByVal CvBoostParams params/*=CvBoostParams()*/ ) { allocate(trainData, tflag, responses, varIdx, sampleIdx, varType, missingDataMask, params); }
    private native void allocate( @Const @ByRef Mat trainData, int tflag,
                @Const @ByRef Mat responses, @Const @ByRef Mat varIdx/*=cv::Mat()*/,
                @Const @ByRef Mat sampleIdx/*=cv::Mat()*/, @Const @ByRef Mat varType/*=cv::Mat()*/,
                @Const @ByRef Mat missingDataMask/*=cv::Mat()*/,
                @ByVal CvBoostParams params/*=CvBoostParams()*/ );

    public native @Cast("bool") boolean train( @Const @ByRef Mat trainData, int tflag,
                           @Const @ByRef Mat responses, @Const @ByRef Mat varIdx/*=cv::Mat()*/,
                           @Const @ByRef Mat sampleIdx/*=cv::Mat()*/, @Const @ByRef Mat varType/*=cv::Mat()*/,
                           @Const @ByRef Mat missingDataMask/*=cv::Mat()*/,
                           @ByVal CvBoostParams params/*=CvBoostParams()*/,
                           @Cast("bool") boolean update/*=false*/ );

    public native float predict( @Const @ByRef Mat sample, @Const @ByRef Mat missing/*=cv::Mat()*/,
                                       @Const @ByRef Range slice/*=cv::Range::all()*/, @Cast("bool") boolean rawMode/*=false*/,
                                       @Cast("bool") boolean returnSum/*=false*/ );

    public native float calc_error( CvMLData _data, int type, @StdVector FloatPointer resp/*=0*/ );
    public native float calc_error( CvMLData _data, int type, @StdVector FloatBuffer resp/*=0*/ );
    public native float calc_error( CvMLData _data, int type, @StdVector float[] resp/*=0*/ ); // type in {CV_TRAIN_ERROR, CV_TEST_ERROR}

    public native void prune( @ByVal CvSlice slice );

    public native void clear();

    public native void write( CvFileStorage storage, @Cast("const char*") BytePointer name );
    public native void write( CvFileStorage storage, String name );
    public native void read( CvFileStorage storage, CvFileNode node );
    public native @Const CvMat get_active_vars(@Cast("bool") boolean absolute_idx/*=true*/);

    public native CvSeq get_weak_predictors();

    public native CvMat get_weights();
    public native CvMat get_subtree_weights();
    public native CvMat get_weak_response();
    public native @Const @ByRef CvBoostParams get_params();
    public native @Const CvDTreeTrainData get_data();
}


/****************************************************************************************\
*                                   Gradient Boosted Trees                               *
\****************************************************************************************/

// DataType: STRUCT CvGBTreesParams
// Parameters of GBT (Gradient Boosted trees model), including single
// tree settings and ensemble parameters.
//
// weak_count          - count of trees in the ensemble
// loss_function_type  - loss function used for ensemble training
// subsample_portion   - portion of whole training set used for
//                       every single tree training.
//                       subsample_portion value is in (0.0, 1.0].
//                       subsample_portion == 1.0 when whole dataset is
//                       used on each step. Count of sample used on each
//                       step is computed as
//                       int(total_samples_count * subsample_portion).
// shrinkage           - regularization parameter.
//                       Each tree prediction is multiplied on shrinkage value.


@NoOffset public static class CvGBTreesParams extends CvDTreeParams {
    static { Loader.load(); }
    public CvGBTreesParams(Pointer p) { super(p); }
    public CvGBTreesParams(int size) { allocateArray(size); }
    private native void allocateArray(int size);
    @Override public CvGBTreesParams position(int position) {
        return (CvGBTreesParams)super.position(position);
    }

    public native int weak_count(); public native CvGBTreesParams weak_count(int weak_count);
    public native int loss_function_type(); public native CvGBTreesParams loss_function_type(int loss_function_type);
    public native float subsample_portion(); public native CvGBTreesParams subsample_portion(float subsample_portion);
    public native float shrinkage(); public native CvGBTreesParams shrinkage(float shrinkage);

    public CvGBTreesParams() { allocate(); }
    private native void allocate();
    public CvGBTreesParams( int loss_function_type, int weak_count, float shrinkage,
            float subsample_portion, int max_depth, @Cast("bool") boolean use_surrogates ) { allocate(loss_function_type, weak_count, shrinkage, subsample_portion, max_depth, use_surrogates); }
    private native void allocate( int loss_function_type, int weak_count, float shrinkage,
            float subsample_portion, int max_depth, @Cast("bool") boolean use_surrogates );
}

// DataType: CLASS CvGBTrees
// Gradient Boosting Trees (GBT) algorithm implementation.
//
// data             - training dataset
// params           - parameters of the CvGBTrees
// weak             - array[0..(class_count-1)] of CvSeq
//                    for storing tree ensembles
// orig_response    - original responses of the training set samples
// sum_response     - predicitons of the current model on the training dataset.
//                    this matrix is updated on every iteration.
// sum_response_tmp - predicitons of the model on the training set on the next
//                    step. On every iteration values of sum_responses_tmp are
//                    computed via sum_responses values. When the current
//                    step is complete sum_response values become equal to
//                    sum_responses_tmp.
// sampleIdx       - indices of samples used for training the ensemble.
//                    CvGBTrees training procedure takes a set of samples
//                    (train_data) and a set of responses (responses).
//                    Only pairs (train_data[i], responses[i]), where i is
//                    in sample_idx are used for training the ensemble.
// subsample_train  - indices of samples used for training a single decision
//                    tree on the current step. This indices are countered
//                    relatively to the sample_idx, so that pairs
//                    (train_data[sample_idx[i]], responses[sample_idx[i]])
//                    are used for training a decision tree.
//                    Training set is randomly splited
//                    in two parts (subsample_train and subsample_test)
//                    on every iteration accordingly to the portion parameter.
// subsample_test   - relative indices of samples from the training set,
//                    which are not used for training a tree on the current
//                    step.
// missing          - mask of the missing values in the training set. This
//                    matrix has the same size as train_data. 1 - missing
//                    value, 0 - not a missing value.
// class_labels     - output class labels map.
// rng              - random number generator. Used for spliting the
//                    training set.
// class_count      - count of output classes.
//                    class_count == 1 in the case of regression,
//                    and > 1 in the case of classification.
// delta            - Huber loss function parameter.
// base_value       - start point of the gradient descent procedure.
//                    model prediction is
//                    f(x) = f_0 + sum_{i=1..weak_count-1}(f_i(x)), where
//                    f_0 is the base value.



@NoOffset public static class CvGBTrees extends CvStatModel {
    static { Loader.load(); }
    public CvGBTrees(Pointer p) { super(p); }
    public CvGBTrees(int size) { allocateArray(size); }
    private native void allocateArray(int size);
    @Override public CvGBTrees position(int position) {
        return (CvGBTrees)super.position(position);
    }


    /*
    // DataType: ENUM
    // Loss functions implemented in CvGBTrees.
    //
    // SQUARED_LOSS
    // problem: regression
    // loss = (x - x')^2
    //
    // ABSOLUTE_LOSS
    // problem: regression
    // loss = abs(x - x')
    //
    // HUBER_LOSS
    // problem: regression
    // loss = delta*( abs(x - x') - delta/2), if abs(x - x') > delta
    //           1/2*(x - x')^2, if abs(x - x') <= delta,
    //           where delta is the alpha-quantile of pseudo responses from
    //           the training set.
    //
    // DEVIANCE_LOSS
    // problem: classification
    //
    */
    /** enum CvGBTrees:: */
    public static final int SQUARED_LOSS= 0, ABSOLUTE_LOSS = 1, HUBER_LOSS= 3, DEVIANCE_LOSS = 4;


    /*
    // Default constructor. Creates a model only (without training).
    // Should be followed by one form of the train(...) function.
    //
    // API
    // CvGBTrees();

    // INPUT
    // OUTPUT
    // RESULT
    */
    public CvGBTrees() { allocate(); }
    private native void allocate();


    /*
    // Full form constructor. Creates a gradient boosting model and does the
    // train.
    //
    // API
    // CvGBTrees( const CvMat* trainData, int tflag,
             const CvMat* responses, const CvMat* varIdx=0,
             const CvMat* sampleIdx=0, const CvMat* varType=0,
             const CvMat* missingDataMask=0,
             CvGBTreesParams params=CvGBTreesParams() );

    // INPUT
    // trainData    - a set of input feature vectors.
    //                  size of matrix is
    //                  <count of samples> x <variables count>
    //                  or <variables count> x <count of samples>
    //                  depending on the tflag parameter.
    //                  matrix values are float.
    // tflag         - a flag showing how do samples stored in the
    //                  trainData matrix row by row (tflag=CV_ROW_SAMPLE)
    //                  or column by column (tflag=CV_COL_SAMPLE).
    // responses     - a vector of responses corresponding to the samples
    //                  in trainData.
    // varIdx       - indices of used variables. zero value means that all
    //                  variables are active.
    // sampleIdx    - indices of used samples. zero value means that all
    //                  samples from trainData are in the training set.
    // varType      - vector of <variables count> length. gives every
    //                  variable type CV_VAR_CATEGORICAL or CV_VAR_ORDERED.
    //                  varType = 0 means all variables are numerical.
    // missingDataMask  - a mask of misiing values in trainData.
    //                  missingDataMask = 0 means that there are no missing
    //                  values.
    // params         - parameters of GTB algorithm.
    // OUTPUT
    // RESULT
    */
    public CvGBTrees( @Const CvMat trainData, int tflag,
                 @Const CvMat responses, @Const CvMat varIdx/*=0*/,
                 @Const CvMat sampleIdx/*=0*/, @Const CvMat varType/*=0*/,
                 @Const CvMat missingDataMask/*=0*/,
                 @ByVal CvGBTreesParams params/*=CvGBTreesParams()*/ ) { allocate(trainData, tflag, responses, varIdx, sampleIdx, varType, missingDataMask, params); }
    private native void allocate( @Const CvMat trainData, int tflag,
                 @Const CvMat responses, @Const CvMat varIdx/*=0*/,
                 @Const CvMat sampleIdx/*=0*/, @Const CvMat varType/*=0*/,
                 @Const CvMat missingDataMask/*=0*/,
                 @ByVal CvGBTreesParams params/*=CvGBTreesParams()*/ );


    /*
    // Destructor.
    */


    /*
    // Gradient tree boosting model training
    //
    // API
    // virtual bool train( const CvMat* trainData, int tflag,
             const CvMat* responses, const CvMat* varIdx=0,
             const CvMat* sampleIdx=0, const CvMat* varType=0,
             const CvMat* missingDataMask=0,
             CvGBTreesParams params=CvGBTreesParams(),
             bool update=false );

    // INPUT
    // trainData    - a set of input feature vectors.
    //                  size of matrix is
    //                  <count of samples> x <variables count>
    //                  or <variables count> x <count of samples>
    //                  depending on the tflag parameter.
    //                  matrix values are float.
    // tflag         - a flag showing how do samples stored in the
    //                  trainData matrix row by row (tflag=CV_ROW_SAMPLE)
    //                  or column by column (tflag=CV_COL_SAMPLE).
    // responses     - a vector of responses corresponding to the samples
    //                  in trainData.
    // varIdx       - indices of used variables. zero value means that all
    //                  variables are active.
    // sampleIdx    - indices of used samples. zero value means that all
    //                  samples from trainData are in the training set.
    // varType      - vector of <variables count> length. gives every
    //                  variable type CV_VAR_CATEGORICAL or CV_VAR_ORDERED.
    //                  varType = 0 means all variables are numerical.
    // missingDataMask  - a mask of misiing values in trainData.
    //                  missingDataMask = 0 means that there are no missing
    //                  values.
    // params         - parameters of GTB algorithm.
    // update         - is not supported now. (!)
    // OUTPUT
    // RESULT
    // Error state.
    */
    public native @Cast("bool") boolean train( @Const CvMat trainData, int tflag,
                 @Const CvMat responses, @Const CvMat varIdx/*=0*/,
                 @Const CvMat sampleIdx/*=0*/, @Const CvMat varType/*=0*/,
                 @Const CvMat missingDataMask/*=0*/,
                 @ByVal CvGBTreesParams params/*=CvGBTreesParams()*/,
                 @Cast("bool") boolean update/*=false*/ );


    /*
    // Gradient tree boosting model training
    //
    // API
    // virtual bool train( CvMLData* data,
             CvGBTreesParams params=CvGBTreesParams(),
             bool update=false ) {return false;};

    // INPUT
    // data          - training set.
    // params        - parameters of GTB algorithm.
    // update        - is not supported now. (!)
    // OUTPUT
    // RESULT
    // Error state.
    */
    public native @Cast("bool") boolean train( CvMLData data,
                 @ByVal CvGBTreesParams params/*=CvGBTreesParams()*/,
                 @Cast("bool") boolean update/*=false*/ );


    /*
    // Response value prediction
    //
    // API
    // virtual float predict_serial( const CvMat* sample, const CvMat* missing=0,
             CvMat* weak_responses=0, CvSlice slice = CV_WHOLE_SEQ,
             int k=-1 ) const;

    // INPUT
    // sample         - input sample of the same type as in the training set.
    // missing        - missing values mask. missing=0 if there are no
    //                   missing values in sample vector.
    // weak_responses  - predictions of all of the trees.
    //                   not implemented (!)
    // slice           - part of the ensemble used for prediction.
    //                   slice = CV_WHOLE_SEQ when all trees are used.
    // k               - number of ensemble used.
    //                   k is in {-1,0,1,..,<count of output classes-1>}.
    //                   in the case of classification problem
    //                   <count of output classes-1> ensembles are built.
    //                   If k = -1 ordinary prediction is the result,
    //                   otherwise function gives the prediction of the
    //                   k-th ensemble only.
    // OUTPUT
    // RESULT
    // Predicted value.
    */
    public native float predict_serial( @Const CvMat sample, @Const CvMat missing/*=0*/,
                CvMat weakResponses/*=0*/, @ByVal CvSlice slice/*=CV_WHOLE_SEQ*/,
                int k/*=-1*/ );

    /*
    // Response value prediction.
    // Parallel version (in the case of TBB existence)
    //
    // API
    // virtual float predict( const CvMat* sample, const CvMat* missing=0,
             CvMat* weak_responses=0, CvSlice slice = CV_WHOLE_SEQ,
             int k=-1 ) const;

    // INPUT
    // sample         - input sample of the same type as in the training set.
    // missing        - missing values mask. missing=0 if there are no
    //                   missing values in sample vector.
    // weak_responses  - predictions of all of the trees.
    //                   not implemented (!)
    // slice           - part of the ensemble used for prediction.
    //                   slice = CV_WHOLE_SEQ when all trees are used.
    // k               - number of ensemble used.
    //                   k is in {-1,0,1,..,<count of output classes-1>}.
    //                   in the case of classification problem
    //                   <count of output classes-1> ensembles are built.
    //                   If k = -1 ordinary prediction is the result,
    //                   otherwise function gives the prediction of the
    //                   k-th ensemble only.
    // OUTPUT
    // RESULT
    // Predicted value.
    */
    public native float predict( @Const CvMat sample, @Const CvMat missing/*=0*/,
                CvMat weakResponses/*=0*/, @ByVal CvSlice slice/*=CV_WHOLE_SEQ*/,
                int k/*=-1*/ );

    /*
    // Deletes all the data.
    //
    // API
    // virtual void clear();

    // INPUT
    // OUTPUT
    // delete data, weak, orig_response, sum_response,
    //        weak_eval, subsample_train, subsample_test,
    //        sample_idx, missing, lass_labels
    // delta = 0.0
    // RESULT
    */
    public native void clear();

    /*
    // Compute error on the train/test set.
    //
    // API
    // virtual float calc_error( CvMLData* _data, int type,
    //        std::vector<float> *resp = 0 );
    //
    // INPUT
    // data  - dataset
    // type  - defines which error is to compute: train (CV_TRAIN_ERROR) or
    //         test (CV_TEST_ERROR).
    // OUTPUT
    // resp  - vector of predicitons
    // RESULT
    // Error value.
    */
    public native float calc_error( CvMLData _data, int type,
                @StdVector FloatPointer resp/*=0*/ );
    public native float calc_error( CvMLData _data, int type,
                @StdVector FloatBuffer resp/*=0*/ );
    public native float calc_error( CvMLData _data, int type,
                @StdVector float[] resp/*=0*/ );

    /*
    //
    // Write parameters of the gtb model and data. Write learned model.
    //
    // API
    // virtual void write( CvFileStorage* fs, const char* name ) const;
    //
    // INPUT
    // fs     - file storage to read parameters from.
    // name   - model name.
    // OUTPUT
    // RESULT
    */
    public native void write( CvFileStorage fs, @Cast("const char*") BytePointer name );
    public native void write( CvFileStorage fs, String name );


    /*
    //
    // Read parameters of the gtb model and data. Read learned model.
    //
    // API
    // virtual void read( CvFileStorage* fs, CvFileNode* node );
    //
    // INPUT
    // fs     - file storage to read parameters from.
    // node   - file node.
    // OUTPUT
    // RESULT
    */
    public native void read( CvFileStorage fs, CvFileNode node );


    // new-style C++ interface
    public CvGBTrees( @Const @ByRef Mat trainData, int tflag,
                  @Const @ByRef Mat responses, @Const @ByRef Mat varIdx/*=cv::Mat()*/,
                  @Const @ByRef Mat sampleIdx/*=cv::Mat()*/, @Const @ByRef Mat varType/*=cv::Mat()*/,
                  @Const @ByRef Mat missingDataMask/*=cv::Mat()*/,
                  @ByVal CvGBTreesParams params/*=CvGBTreesParams()*/ ) { allocate(trainData, tflag, responses, varIdx, sampleIdx, varType, missingDataMask, params); }
    private native void allocate( @Const @ByRef Mat trainData, int tflag,
                  @Const @ByRef Mat responses, @Const @ByRef Mat varIdx/*=cv::Mat()*/,
                  @Const @ByRef Mat sampleIdx/*=cv::Mat()*/, @Const @ByRef Mat varType/*=cv::Mat()*/,
                  @Const @ByRef Mat missingDataMask/*=cv::Mat()*/,
                  @ByVal CvGBTreesParams params/*=CvGBTreesParams()*/ );

    public native @Cast("bool") boolean train( @Const @ByRef Mat trainData, int tflag,
                           @Const @ByRef Mat responses, @Const @ByRef Mat varIdx/*=cv::Mat()*/,
                           @Const @ByRef Mat sampleIdx/*=cv::Mat()*/, @Const @ByRef Mat varType/*=cv::Mat()*/,
                           @Const @ByRef Mat missingDataMask/*=cv::Mat()*/,
                           @ByVal CvGBTreesParams params/*=CvGBTreesParams()*/,
                           @Cast("bool") boolean update/*=false*/ );

    public native float predict( @Const @ByRef Mat sample, @Const @ByRef Mat missing/*=cv::Mat()*/,
                               @Const @ByRef Range slice/*=cv::Range::all()*/,
                               int k/*=-1*/ );

}



/****************************************************************************************\
*                              Artificial Neural Networks (ANN)                          *
\****************************************************************************************/

/////////////////////////////////// Multi-Layer Perceptrons //////////////////////////////

@NoOffset public static class CvANN_MLP_TrainParams extends Pointer {
    static { Loader.load(); }
    public CvANN_MLP_TrainParams(Pointer p) { super(p); }
    public CvANN_MLP_TrainParams(int size) { allocateArray(size); }
    private native void allocateArray(int size);
    @Override public CvANN_MLP_TrainParams position(int position) {
        return (CvANN_MLP_TrainParams)super.position(position);
    }

    public CvANN_MLP_TrainParams() { allocate(); }
    private native void allocate();
    public CvANN_MLP_TrainParams( @ByVal CvTermCriteria term_crit, int train_method,
                               double param1, double param2/*=0*/ ) { allocate(term_crit, train_method, param1, param2); }
    private native void allocate( @ByVal CvTermCriteria term_crit, int train_method,
                               double param1, double param2/*=0*/ );

    /** enum CvANN_MLP_TrainParams:: */
    public static final int BACKPROP= 0, RPROP= 1;

    public native @ByVal CvTermCriteria term_crit(); public native CvANN_MLP_TrainParams term_crit(CvTermCriteria term_crit);
    public native int train_method(); public native CvANN_MLP_TrainParams train_method(int train_method);

    // backpropagation parameters
    public native double bp_dw_scale(); public native CvANN_MLP_TrainParams bp_dw_scale(double bp_dw_scale);
    public native double bp_moment_scale(); public native CvANN_MLP_TrainParams bp_moment_scale(double bp_moment_scale);

    // rprop parameters
    public native double rp_dw0(); public native CvANN_MLP_TrainParams rp_dw0(double rp_dw0);
    public native double rp_dw_plus(); public native CvANN_MLP_TrainParams rp_dw_plus(double rp_dw_plus);
    public native double rp_dw_minus(); public native CvANN_MLP_TrainParams rp_dw_minus(double rp_dw_minus);
    public native double rp_dw_min(); public native CvANN_MLP_TrainParams rp_dw_min(double rp_dw_min);
    public native double rp_dw_max(); public native CvANN_MLP_TrainParams rp_dw_max(double rp_dw_max);
}


@NoOffset public static class CvANN_MLP extends CvStatModel {
    static { Loader.load(); }
    public CvANN_MLP(Pointer p) { super(p); }
    public CvANN_MLP(int size) { allocateArray(size); }
    private native void allocateArray(int size);
    @Override public CvANN_MLP position(int position) {
        return (CvANN_MLP)super.position(position);
    }

    public CvANN_MLP() { allocate(); }
    private native void allocate();
    public CvANN_MLP( @Const CvMat layerSizes,
                   int activateFunc/*=CvANN_MLP::SIGMOID_SYM*/,
                   double fparam1/*=0*/, double fparam2/*=0*/ ) { allocate(layerSizes, activateFunc, fparam1, fparam2); }
    private native void allocate( @Const CvMat layerSizes,
                   int activateFunc/*=CvANN_MLP::SIGMOID_SYM*/,
                   double fparam1/*=0*/, double fparam2/*=0*/ );

    public native void create( @Const CvMat layerSizes,
                             int activateFunc/*=CvANN_MLP::SIGMOID_SYM*/,
                             double fparam1/*=0*/, double fparam2/*=0*/ );

    public native int train( @Const CvMat inputs, @Const CvMat outputs,
                           @Const CvMat sampleWeights, @Const CvMat sampleIdx/*=0*/,
                           @ByVal CvANN_MLP_TrainParams params/*=CvANN_MLP_TrainParams()*/,
                           int flags/*=0*/ );
    public native float predict( @Const CvMat inputs, CvMat outputs );

    public CvANN_MLP( @Const @ByRef Mat layerSizes,
                  int activateFunc/*=CvANN_MLP::SIGMOID_SYM*/,
                  double fparam1/*=0*/, double fparam2/*=0*/ ) { allocate(layerSizes, activateFunc, fparam1, fparam2); }
    private native void allocate( @Const @ByRef Mat layerSizes,
                  int activateFunc/*=CvANN_MLP::SIGMOID_SYM*/,
                  double fparam1/*=0*/, double fparam2/*=0*/ );

    public native void create( @Const @ByRef Mat layerSizes,
                            int activateFunc/*=CvANN_MLP::SIGMOID_SYM*/,
                            double fparam1/*=0*/, double fparam2/*=0*/ );

    public native int train( @Const @ByRef Mat inputs, @Const @ByRef Mat outputs,
                          @Const @ByRef Mat sampleWeights, @Const @ByRef Mat sampleIdx/*=cv::Mat()*/,
                          @ByVal CvANN_MLP_TrainParams params/*=CvANN_MLP_TrainParams()*/,
                          int flags/*=0*/ );

    public native float predict( @Const @ByRef Mat inputs, @ByRef Mat outputs );

    public native void clear();

    // possible activation functions
    /** enum CvANN_MLP:: */
    public static final int IDENTITY = 0, SIGMOID_SYM = 1, GAUSSIAN = 2;

    // available training flags
    /** enum CvANN_MLP:: */
    public static final int UPDATE_WEIGHTS = 1, NO_INPUT_SCALE = 2, NO_OUTPUT_SCALE = 4;

    public native void read( CvFileStorage fs, CvFileNode node );
    public native void write( CvFileStorage storage, @Cast("const char*") BytePointer name );
    public native void write( CvFileStorage storage, String name );

    public native int get_layer_count();
    public native @Const CvMat get_layer_sizes();
    public native DoublePointer get_weights(int layer);

    public native void calc_activ_func_deriv( CvMat xf, CvMat deriv, @Const DoublePointer bias );
    public native void calc_activ_func_deriv( CvMat xf, CvMat deriv, @Const DoubleBuffer bias );
    public native void calc_activ_func_deriv( CvMat xf, CvMat deriv, @Const double[] bias );
}

/****************************************************************************************\
*                           Auxilary functions declarations                              *
\****************************************************************************************/

/* Generates <sample> from multivariate normal distribution, where <mean> - is an
   average row vector, <cov> - symmetric covariation matrix */
public static native void cvRandMVNormal( CvMat mean, CvMat cov, CvMat sample,
                           @Cast("CvRNG*") LongPointer rng/*CV_DEFAULT(0)*/ );
public static native void cvRandMVNormal( CvMat mean, CvMat cov, CvMat sample,
                           @Cast("CvRNG*") LongBuffer rng/*CV_DEFAULT(0)*/ );
public static native void cvRandMVNormal( CvMat mean, CvMat cov, CvMat sample,
                           @Cast("CvRNG*") long[] rng/*CV_DEFAULT(0)*/ );

/* Generates sample from gaussian mixture distribution */
public static native void cvRandGaussMixture( @Cast("CvMat**") PointerPointer means,
                               @Cast("CvMat**") PointerPointer covs,
                               FloatPointer weights,
                               int clsnum,
                               CvMat sample,
                               CvMat sampClasses/*CV_DEFAULT(0)*/ );
public static native void cvRandGaussMixture( @ByPtrPtr CvMat means,
                               @ByPtrPtr CvMat covs,
                               FloatPointer weights,
                               int clsnum,
                               CvMat sample,
                               CvMat sampClasses/*CV_DEFAULT(0)*/ );
public static native void cvRandGaussMixture( @ByPtrPtr CvMat means,
                               @ByPtrPtr CvMat covs,
                               FloatBuffer weights,
                               int clsnum,
                               CvMat sample,
                               CvMat sampClasses/*CV_DEFAULT(0)*/ );
public static native void cvRandGaussMixture( @ByPtrPtr CvMat means,
                               @ByPtrPtr CvMat covs,
                               float[] weights,
                               int clsnum,
                               CvMat sample,
                               CvMat sampClasses/*CV_DEFAULT(0)*/ );

public static final int CV_TS_CONCENTRIC_SPHERES = 0;

/* creates test set */
public static native void cvCreateTestSet( int type, @Cast("CvMat**") PointerPointer samples,
                 int num_samples,
                 int num_features,
                 @Cast("CvMat**") PointerPointer responses,
                 int num_classes );
public static native void cvCreateTestSet( int type, @ByPtrPtr CvMat samples,
                 int num_samples,
                 int num_features,
                 @ByPtrPtr CvMat responses,
                 int num_classes );

/****************************************************************************************\
*                                      Data                                             *
\****************************************************************************************/

public static final int CV_COUNT =     0;
public static final int CV_PORTION =   1;

@NoOffset public static class CvTrainTestSplit extends Pointer {
    static { Loader.load(); }
    public CvTrainTestSplit(Pointer p) { super(p); }
    public CvTrainTestSplit(int size) { allocateArray(size); }
    private native void allocateArray(int size);
    @Override public CvTrainTestSplit position(int position) {
        return (CvTrainTestSplit)super.position(position);
    }

    public CvTrainTestSplit() { allocate(); }
    private native void allocate();
    public CvTrainTestSplit( int train_sample_count, @Cast("bool") boolean mix/*=true*/) { allocate(train_sample_count, mix); }
    private native void allocate( int train_sample_count, @Cast("bool") boolean mix/*=true*/);
    public CvTrainTestSplit( float train_sample_portion, @Cast("bool") boolean mix/*=true*/) { allocate(train_sample_portion, mix); }
    private native void allocate( float train_sample_portion, @Cast("bool") boolean mix/*=true*/);

        @Name("train_sample_part.count") public native int train_sample_part_count(); public native CvTrainTestSplit train_sample_part_count(int train_sample_part_count);
        @Name("train_sample_part.portion") public native float train_sample_part_portion(); public native CvTrainTestSplit train_sample_part_portion(float train_sample_part_portion);
    public native int train_sample_part_mode(); public native CvTrainTestSplit train_sample_part_mode(int train_sample_part_mode);

    public native @Cast("bool") boolean mix(); public native CvTrainTestSplit mix(boolean mix);
}

@NoOffset public static class CvMLData extends Pointer {
    static { Loader.load(); }
    public CvMLData(Pointer p) { super(p); }
    public CvMLData(int size) { allocateArray(size); }
    private native void allocateArray(int size);
    @Override public CvMLData position(int position) {
        return (CvMLData)super.position(position);
    }

    public CvMLData() { allocate(); }
    private native void allocate();

    // returns:
    // 0 - OK
    // -1 - file can not be opened or is not correct
    public native int read_csv( @Cast("const char*") BytePointer filename );
    public native int read_csv( String filename );

    public native @Const CvMat get_values();
    public native @Const CvMat get_responses();
    public native @Const CvMat get_missing();

    public native void set_response_idx( int idx ); // old response become predictors, new response_idx = idx
                                      // if idx < 0 there will be no response
    public native int get_response_idx();

    public native void set_train_test_split( @Const CvTrainTestSplit spl );
    public native @Const CvMat get_train_sample_idx();
    public native @Const CvMat get_test_sample_idx();
    public native void mix_train_and_test_idx();

    public native @Const CvMat get_var_idx();
    public native void chahge_var_idx( int vi, @Cast("bool") boolean state ); // misspelled (saved for back compitability),
                                               // use change_var_idx
    public native void change_var_idx( int vi, @Cast("bool") boolean state ); // state == true to set vi-variable as predictor

    public native @Const CvMat get_var_types();
    public native int get_var_type( int var_idx );
    // following 2 methods enable to change vars type
    // use these methods to assign CV_VAR_CATEGORICAL type for categorical variable
    // with numerical labels; in the other cases var types are correctly determined automatically
    public native void set_var_types( @Cast("const char*") BytePointer str );
    public native void set_var_types( String str );  // str examples:
                                            // "ord[0-17],cat[18]", "ord[0,2,4,10-12], cat[1,3,5-9,13,14]",
                                            // "cat", "ord" (all vars are categorical/ordered)
    public native void change_var_type( int var_idx, int type); // type in { CV_VAR_ORDERED, CV_VAR_CATEGORICAL }

    public native void set_delimiter( @Cast("char") byte ch );
    public native @Cast("char") byte get_delimiter();

    public native void set_miss_ch( @Cast("char") byte ch );
    public native @Cast("char") byte get_miss_ch();

    public native @Const @ByRef StringIntMap get_class_labels_map();
}



@Namespace("cv") public static native @Cast("bool") boolean initModule_ml();



// #endif // __cplusplus
// #endif // __OPENCV_ML_HPP__

/* End of file. */


}
