// Targeted by JavaCPP version 0.9

package org.bytedeco.javacpp;

import java.nio.*;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

import static org.bytedeco.javacpp.opencv_core.*;
import static org.bytedeco.javacpp.opencv_imgproc.*;

public class opencv_video extends org.bytedeco.javacpp.helper.opencv_video {
    static { Loader.load(); }

// Parsed from <opencv2/video/video.hpp>

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

// #ifndef __OPENCV_VIDEO_HPP__
// #define __OPENCV_VIDEO_HPP__

// #include "opencv2/video/tracking.hpp"
// #include "opencv2/video/background_segm.hpp"

// #ifdef __cplusplus

@Namespace("cv") public static native @Cast("bool") boolean initModule_video();


// #endif

// #endif //__OPENCV_VIDEO_HPP__


// Parsed from <opencv2/video/tracking.hpp>

/** \file tracking.hpp
 \brief The Object and Feature Tracking
 */

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

// #ifndef __OPENCV_TRACKING_HPP__
// #define __OPENCV_TRACKING_HPP__

// #include "opencv2/core/core.hpp"
// #include "opencv2/imgproc/imgproc.hpp"

// #ifdef __cplusplus
// #endif

/****************************************************************************************\
*                                  Motion Analysis                                       *
\****************************************************************************************/

/************************************ optical flow ***************************************/

public static final int CV_LKFLOW_PYR_A_READY =       1;
public static final int CV_LKFLOW_PYR_B_READY =       2;
public static final int CV_LKFLOW_INITIAL_GUESSES =   4;
public static final int CV_LKFLOW_GET_MIN_EIGENVALS = 8;

/* It is Lucas & Kanade method, modified to use pyramids.
   Also it does several iterations to get optical flow for
   every point at every pyramid level.
   Calculates optical flow between two images for certain set of points (i.e.
   it is a "sparse" optical flow, which is opposite to the previous 3 methods) */
public static native void cvCalcOpticalFlowPyrLK( @Const CvArr prev, @Const CvArr curr,
                                     CvArr prev_pyr, CvArr curr_pyr,
                                     @Const CvPoint2D32f prev_features,
                                     CvPoint2D32f curr_features,
                                     int count,
                                     @ByVal CvSize win_size,
                                     int level,
                                     @Cast("char*") BytePointer status,
                                     FloatPointer track_error,
                                     @ByVal CvTermCriteria criteria,
                                     int flags );
public static native void cvCalcOpticalFlowPyrLK( @Const CvArr prev, @Const CvArr curr,
                                     CvArr prev_pyr, CvArr curr_pyr,
                                     @Cast("const CvPoint2D32f*") FloatBuffer prev_features,
                                     @Cast("CvPoint2D32f*") FloatBuffer curr_features,
                                     int count,
                                     @ByVal CvSize win_size,
                                     int level,
                                     @Cast("char*") ByteBuffer status,
                                     FloatBuffer track_error,
                                     @ByVal CvTermCriteria criteria,
                                     int flags );
public static native void cvCalcOpticalFlowPyrLK( @Const CvArr prev, @Const CvArr curr,
                                     CvArr prev_pyr, CvArr curr_pyr,
                                     @Cast("const CvPoint2D32f*") float[] prev_features,
                                     @Cast("CvPoint2D32f*") float[] curr_features,
                                     int count,
                                     @ByVal CvSize win_size,
                                     int level,
                                     @Cast("char*") byte[] status,
                                     float[] track_error,
                                     @ByVal CvTermCriteria criteria,
                                     int flags );


/* Modification of a previous sparse optical flow algorithm to calculate
   affine flow */
public static native void cvCalcAffineFlowPyrLK( @Const CvArr prev, @Const CvArr curr,
                                    CvArr prev_pyr, CvArr curr_pyr,
                                    @Const CvPoint2D32f prev_features,
                                    CvPoint2D32f curr_features,
                                    FloatPointer matrices, int count,
                                    @ByVal CvSize win_size, int level,
                                    @Cast("char*") BytePointer status, FloatPointer track_error,
                                    @ByVal CvTermCriteria criteria, int flags );
public static native void cvCalcAffineFlowPyrLK( @Const CvArr prev, @Const CvArr curr,
                                    CvArr prev_pyr, CvArr curr_pyr,
                                    @Cast("const CvPoint2D32f*") FloatBuffer prev_features,
                                    @Cast("CvPoint2D32f*") FloatBuffer curr_features,
                                    FloatBuffer matrices, int count,
                                    @ByVal CvSize win_size, int level,
                                    @Cast("char*") ByteBuffer status, FloatBuffer track_error,
                                    @ByVal CvTermCriteria criteria, int flags );
public static native void cvCalcAffineFlowPyrLK( @Const CvArr prev, @Const CvArr curr,
                                    CvArr prev_pyr, CvArr curr_pyr,
                                    @Cast("const CvPoint2D32f*") float[] prev_features,
                                    @Cast("CvPoint2D32f*") float[] curr_features,
                                    float[] matrices, int count,
                                    @ByVal CvSize win_size, int level,
                                    @Cast("char*") byte[] status, float[] track_error,
                                    @ByVal CvTermCriteria criteria, int flags );

/* Estimate rigid transformation between 2 images or 2 point sets */
public static native int cvEstimateRigidTransform( @Const CvArr A, @Const CvArr B,
                                      CvMat M, int full_affine );

/* Estimate optical flow for each pixel using the two-frame G. Farneback algorithm */
public static native void cvCalcOpticalFlowFarneback( @Const CvArr prev, @Const CvArr next,
                                        CvArr flow, double pyr_scale, int levels,
                                        int winsize, int iterations, int poly_n,
                                        double poly_sigma, int flags );

/********************************* motion templates *************************************/

/****************************************************************************************\
*        All the motion template functions work only with single channel images.         *
*        Silhouette image must have depth IPL_DEPTH_8U or IPL_DEPTH_8S                   *
*        Motion history image must have depth IPL_DEPTH_32F,                             *
*        Gradient mask - IPL_DEPTH_8U or IPL_DEPTH_8S,                                   *
*        Motion orientation image - IPL_DEPTH_32F                                        *
*        Segmentation mask - IPL_DEPTH_32F                                               *
*        All the angles are in degrees, all the times are in milliseconds                *
\****************************************************************************************/

/* Updates motion history image given motion silhouette */
public static native void cvUpdateMotionHistory( @Const CvArr silhouette, CvArr mhi,
                                      double timestamp, double duration );

/* Calculates gradient of the motion history image and fills
   a mask indicating where the gradient is valid */
public static native void cvCalcMotionGradient( @Const CvArr mhi, CvArr mask, CvArr orientation,
                                     double delta1, double delta2,
                                     int aperture_size/*=3*/);
public static native void cvCalcMotionGradient( @Const CvArr mhi, CvArr mask, CvArr orientation,
                                     double delta1, double delta2);

/* Calculates average motion direction within a selected motion region
   (region can be selected by setting ROIs and/or by composing a valid gradient mask
   with the region mask) */
public static native double cvCalcGlobalOrientation( @Const CvArr orientation, @Const CvArr mask,
                                        @Const CvArr mhi, double timestamp,
                                        double duration );

/* Splits a motion history image into a few parts corresponding to separate independent motions
   (e.g. left hand, right hand) */
public static native CvSeq cvSegmentMotion( @Const CvArr mhi, CvArr seg_mask,
                                CvMemStorage storage,
                                double timestamp, double seg_thresh );

/****************************************************************************************\
*                                       Tracking                                         *
\****************************************************************************************/

/* Implements CAMSHIFT algorithm - determines object position, size and orientation
   from the object histogram back project (extension of meanshift) */
public static native int cvCamShift( @Const CvArr prob_image, @ByVal CvRect window,
                        @ByVal CvTermCriteria criteria, CvConnectedComp comp,
                        CvBox2D box/*=NULL*/ );
public static native int cvCamShift( @Const CvArr prob_image, @ByVal CvRect window,
                        @ByVal CvTermCriteria criteria, CvConnectedComp comp );

/* Implements MeanShift algorithm - determines object position
   from the object histogram back project */
public static native int cvMeanShift( @Const CvArr prob_image, @ByVal CvRect window,
                         @ByVal CvTermCriteria criteria, CvConnectedComp comp );

/*
standard Kalman filter (in G. Welch' and G. Bishop's notation):

  x(k)=A*x(k-1)+B*u(k)+w(k)  p(w)~N(0,Q)
  z(k)=H*x(k)+v(k),   p(v)~N(0,R)
*/
public static class CvKalman extends AbstractCvKalman {
    static { Loader.load(); }
    public CvKalman() { allocate(); }
    public CvKalman(int size) { allocateArray(size); }
    public CvKalman(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public CvKalman position(int position) {
        return (CvKalman)super.position(position);
    }

    public native int MP(); public native CvKalman MP(int MP);                     /* number of measurement vector dimensions */
    public native int DP(); public native CvKalman DP(int DP);                     /* number of state vector dimensions */
    public native int CP(); public native CvKalman CP(int CP);                     /* number of control vector dimensions */

    /* backward compatibility fields */
// #if 1
    public native FloatPointer PosterState(); public native CvKalman PosterState(FloatPointer PosterState);         /* =state_pre->data.fl */
    public native FloatPointer PriorState(); public native CvKalman PriorState(FloatPointer PriorState);          /* =state_post->data.fl */
    public native FloatPointer DynamMatr(); public native CvKalman DynamMatr(FloatPointer DynamMatr);           /* =transition_matrix->data.fl */
    public native FloatPointer MeasurementMatr(); public native CvKalman MeasurementMatr(FloatPointer MeasurementMatr);     /* =measurement_matrix->data.fl */
    public native FloatPointer MNCovariance(); public native CvKalman MNCovariance(FloatPointer MNCovariance);        /* =measurement_noise_cov->data.fl */
    public native FloatPointer PNCovariance(); public native CvKalman PNCovariance(FloatPointer PNCovariance);        /* =process_noise_cov->data.fl */
    public native FloatPointer KalmGainMatr(); public native CvKalman KalmGainMatr(FloatPointer KalmGainMatr);        /* =gain->data.fl */
    public native FloatPointer PriorErrorCovariance(); public native CvKalman PriorErrorCovariance(FloatPointer PriorErrorCovariance);/* =error_cov_pre->data.fl */
    public native FloatPointer PosterErrorCovariance(); public native CvKalman PosterErrorCovariance(FloatPointer PosterErrorCovariance);/* =error_cov_post->data.fl */
    public native FloatPointer Temp1(); public native CvKalman Temp1(FloatPointer Temp1);               /* temp1->data.fl */
    public native FloatPointer Temp2(); public native CvKalman Temp2(FloatPointer Temp2);               /* temp2->data.fl */
// #endif

    public native CvMat state_pre(); public native CvKalman state_pre(CvMat state_pre);           /* predicted state (x'(k)):
                                    x(k)=A*x(k-1)+B*u(k) */
    public native CvMat state_post(); public native CvKalman state_post(CvMat state_post);          /* corrected state (x(k)):
                                    x(k)=x'(k)+K(k)*(z(k)-H*x'(k)) */
    public native CvMat transition_matrix(); public native CvKalman transition_matrix(CvMat transition_matrix);   /* state transition matrix (A) */
    public native CvMat control_matrix(); public native CvKalman control_matrix(CvMat control_matrix);      /* control matrix (B)
                                   (it is not used if there is no control)*/
    public native CvMat measurement_matrix(); public native CvKalman measurement_matrix(CvMat measurement_matrix);  /* measurement matrix (H) */
    public native CvMat process_noise_cov(); public native CvKalman process_noise_cov(CvMat process_noise_cov);   /* process noise covariance matrix (Q) */
    public native CvMat measurement_noise_cov(); public native CvKalman measurement_noise_cov(CvMat measurement_noise_cov); /* measurement noise covariance matrix (R) */
    public native CvMat error_cov_pre(); public native CvKalman error_cov_pre(CvMat error_cov_pre);       /* priori error estimate covariance matrix (P'(k)):
                                    P'(k)=A*P(k-1)*At + Q)*/
    public native CvMat gain(); public native CvKalman gain(CvMat gain);                /* Kalman gain matrix (K(k)):
                                    K(k)=P'(k)*Ht*inv(H*P'(k)*Ht+R)*/
    public native CvMat error_cov_post(); public native CvKalman error_cov_post(CvMat error_cov_post);      /* posteriori error estimate covariance matrix (P(k)):
                                    P(k)=(I-K(k)*H)*P'(k) */
    public native CvMat temp1(); public native CvKalman temp1(CvMat temp1);               /* temporary matrices */
    public native CvMat temp2(); public native CvKalman temp2(CvMat temp2);
    public native CvMat temp3(); public native CvKalman temp3(CvMat temp3);
    public native CvMat temp4(); public native CvKalman temp4(CvMat temp4);
    public native CvMat temp5(); public native CvKalman temp5(CvMat temp5);
}

/* Creates Kalman filter and sets A, B, Q, R and state to some initial values */
public static native CvKalman cvCreateKalman( int dynam_params, int measure_params,
                                 int control_params/*=0*/);
public static native CvKalman cvCreateKalman( int dynam_params, int measure_params);

/* Releases Kalman filter state */
public static native void cvReleaseKalman( @Cast("CvKalman**") PointerPointer kalman);
public static native void cvReleaseKalman( @ByPtrPtr CvKalman kalman);

/* Updates Kalman filter by time (predicts future state of the system) */
public static native @Const CvMat cvKalmanPredict( CvKalman kalman,
                                      @Const CvMat control/*=NULL*/);
public static native @Const CvMat cvKalmanPredict( CvKalman kalman);

/* Updates Kalman filter by measurement
   (corrects state of the system and internal matrices) */
public static native @Const CvMat cvKalmanCorrect( CvKalman kalman, @Const CvMat measurement );

public static native @Const CvMat cvKalmanUpdateByTime(CvKalman arg1, CvMat arg2);
public static native @Const CvMat cvKalmanUpdateByMeasurement(CvKalman arg1, CvMat arg2);

// #ifdef __cplusplus

/** updates motion history image using the current silhouette */
@Namespace("cv") public static native void updateMotionHistory( @ByVal Mat silhouette, @ByVal Mat mhi,
                                       double timestamp, double duration );

/** computes the motion gradient orientation image from the motion history image */
@Namespace("cv") public static native void calcMotionGradient( @ByVal Mat mhi, @ByVal Mat mask,
                                      @ByVal Mat orientation,
                                      double delta1, double delta2,
                                      int apertureSize/*=3*/ );
@Namespace("cv") public static native void calcMotionGradient( @ByVal Mat mhi, @ByVal Mat mask,
                                      @ByVal Mat orientation,
                                      double delta1, double delta2 );

/** computes the global orientation of the selected motion history image part */
@Namespace("cv") public static native double calcGlobalOrientation( @ByVal Mat orientation, @ByVal Mat mask,
                                           @ByVal Mat mhi, double timestamp,
                                           double duration );

@Namespace("cv") public static native void segmentMotion(@ByVal Mat mhi, @ByVal Mat segmask,
                                @StdVector Rect boundingRects,
                                double timestamp, double segThresh);

/** updates the object tracking window using CAMSHIFT algorithm */
@Namespace("cv") public static native @ByVal RotatedRect CamShift( @ByVal Mat probImage, @ByRef Rect window,
                                   @ByVal TermCriteria criteria );

/** updates the object tracking window using meanshift algorithm */
@Namespace("cv") public static native int meanShift( @ByVal Mat probImage, @ByRef Rect window,
                            @ByVal TermCriteria criteria );

/**
 Kalman filter.

 The class implements standard Kalman filter \u005Curl{http://en.wikipedia.org/wiki/Kalman_filter}.
 However, you can modify KalmanFilter::transitionMatrix, KalmanFilter::controlMatrix and
 KalmanFilter::measurementMatrix to get the extended Kalman filter functionality.
*/
@Namespace("cv") @NoOffset public static class KalmanFilter extends Pointer {
    static { Loader.load(); }
    public KalmanFilter(Pointer p) { super(p); }
    public KalmanFilter(int size) { allocateArray(size); }
    private native void allocateArray(int size);
    @Override public KalmanFilter position(int position) {
        return (KalmanFilter)super.position(position);
    }

    /** the default constructor */
    public KalmanFilter() { allocate(); }
    private native void allocate();
    /** the full constructor taking the dimensionality of the state, of the measurement and of the control vector */
    public KalmanFilter(int dynamParams, int measureParams, int controlParams/*=0*/, int type/*=CV_32F*/) { allocate(dynamParams, measureParams, controlParams, type); }
    private native void allocate(int dynamParams, int measureParams, int controlParams/*=0*/, int type/*=CV_32F*/);
    public KalmanFilter(int dynamParams, int measureParams) { allocate(dynamParams, measureParams); }
    private native void allocate(int dynamParams, int measureParams);
    /** re-initializes Kalman filter. The previous content is destroyed. */
    public native void init(int dynamParams, int measureParams, int controlParams/*=0*/, int type/*=CV_32F*/);
    public native void init(int dynamParams, int measureParams);

    /** computes predicted state */
    public native @Const @ByRef Mat predict(@Const @ByRef Mat control/*=Mat()*/);
    public native @Const @ByRef Mat predict();
    /** updates the predicted state from the measurement */
    public native @Const @ByRef Mat correct(@Const @ByRef Mat measurement);

    /** predicted state (x'(k)): x(k)=A*x(k-1)+B*u(k) */
    public native @ByRef Mat statePre(); public native KalmanFilter statePre(Mat statePre);
    /** corrected state (x(k)): x(k)=x'(k)+K(k)*(z(k)-H*x'(k)) */
    public native @ByRef Mat statePost(); public native KalmanFilter statePost(Mat statePost);
    /** state transition matrix (A) */
    public native @ByRef Mat transitionMatrix(); public native KalmanFilter transitionMatrix(Mat transitionMatrix);
    /** control matrix (B) (not used if there is no control) */
    public native @ByRef Mat controlMatrix(); public native KalmanFilter controlMatrix(Mat controlMatrix);
    /** measurement matrix (H) */
    public native @ByRef Mat measurementMatrix(); public native KalmanFilter measurementMatrix(Mat measurementMatrix);
    /** process noise covariance matrix (Q) */
    public native @ByRef Mat processNoiseCov(); public native KalmanFilter processNoiseCov(Mat processNoiseCov);
    /** measurement noise covariance matrix (R) */
    public native @ByRef Mat measurementNoiseCov(); public native KalmanFilter measurementNoiseCov(Mat measurementNoiseCov);
    /** priori error estimate covariance matrix (P'(k)): P'(k)=A*P(k-1)*At + Q)*/
    public native @ByRef Mat errorCovPre(); public native KalmanFilter errorCovPre(Mat errorCovPre);
    /** Kalman gain matrix (K(k)): K(k)=P'(k)*Ht*inv(H*P'(k)*Ht+R) */
    public native @ByRef Mat gain(); public native KalmanFilter gain(Mat gain);
    /** posteriori error estimate covariance matrix (P(k)): P(k)=(I-K(k)*H)*P'(k) */
    public native @ByRef Mat errorCovPost(); public native KalmanFilter errorCovPost(Mat errorCovPost);

    // temporary matrices
    public native @ByRef Mat temp1(); public native KalmanFilter temp1(Mat temp1);
    public native @ByRef Mat temp2(); public native KalmanFilter temp2(Mat temp2);
    public native @ByRef Mat temp3(); public native KalmanFilter temp3(Mat temp3);
    public native @ByRef Mat temp4(); public native KalmanFilter temp4(Mat temp4);
    public native @ByRef Mat temp5(); public native KalmanFilter temp5(Mat temp5);
}

/** enum cv:: */
public static final int
    OPTFLOW_USE_INITIAL_FLOW =  CV_LKFLOW_INITIAL_GUESSES,
    OPTFLOW_LK_GET_MIN_EIGENVALS =  CV_LKFLOW_GET_MIN_EIGENVALS,
    OPTFLOW_FARNEBACK_GAUSSIAN = 256;

/** constructs a pyramid which can be used as input for calcOpticalFlowPyrLK */
@Namespace("cv") public static native int buildOpticalFlowPyramid(@ByVal Mat img, @ByVal MatVector pyramid,
                                         @ByVal Size winSize, int maxLevel, @Cast("bool") boolean withDerivatives/*=true*/,
                                         int pyrBorder/*=BORDER_REFLECT_101*/, int derivBorder/*=BORDER_CONSTANT*/,
                                         @Cast("bool") boolean tryReuseInputImage/*=true*/);
@Namespace("cv") public static native int buildOpticalFlowPyramid(@ByVal Mat img, @ByVal MatVector pyramid,
                                         @ByVal Size winSize, int maxLevel);

/** computes sparse optical flow using multi-scale Lucas-Kanade algorithm */
@Namespace("cv") public static native void calcOpticalFlowPyrLK( @ByVal Mat prevImg, @ByVal Mat nextImg,
                           @ByVal Mat prevPts, @ByVal Mat nextPts,
                           @ByVal Mat status, @ByVal Mat err,
                           @ByVal Size winSize/*=Size(21,21)*/, int maxLevel/*=3*/,
                           @ByVal TermCriteria criteria/*=TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 30, 0.01)*/,
                           int flags/*=0*/, double minEigThreshold/*=1e-4*/);
@Namespace("cv") public static native void calcOpticalFlowPyrLK( @ByVal Mat prevImg, @ByVal Mat nextImg,
                           @ByVal Mat prevPts, @ByVal Mat nextPts,
                           @ByVal Mat status, @ByVal Mat err);

/** computes dense optical flow using Farneback algorithm */
@Namespace("cv") public static native void calcOpticalFlowFarneback( @ByVal Mat prev, @ByVal Mat next,
                           @ByVal Mat flow, double pyr_scale, int levels, int winsize,
                           int iterations, int poly_n, double poly_sigma, int flags );

/** estimates the best-fit Euqcidean, similarity, affine or perspective transformation
// that maps one 2D point set to another or one image to another. */
@Namespace("cv") public static native @ByVal Mat estimateRigidTransform( @ByVal Mat src, @ByVal Mat dst,
                                         @Cast("bool") boolean fullAffine);

/** computes dense optical flow using Simple Flow algorithm */
@Namespace("cv") public static native void calcOpticalFlowSF(@ByRef Mat from,
                                    @ByRef Mat to,
                                    @ByRef Mat flow,
                                    int layers,
                                    int averaging_block_size,
                                    int max_flow);

@Namespace("cv") public static native void calcOpticalFlowSF(@ByRef Mat from,
                                    @ByRef Mat to,
                                    @ByRef Mat flow,
                                    int layers,
                                    int averaging_block_size,
                                    int max_flow,
                                    double sigma_dist,
                                    double sigma_color,
                                    int postprocess_window,
                                    double sigma_dist_fix,
                                    double sigma_color_fix,
                                    double occ_thr,
                                    int upscale_averaging_radius,
                                    double upscale_sigma_dist,
                                    double upscale_sigma_color,
                                    double speed_up_thr);

@Namespace("cv") public static class DenseOpticalFlow extends Algorithm {
    static { Loader.load(); }
    public DenseOpticalFlow() { }
    public DenseOpticalFlow(Pointer p) { super(p); }

    public native void calc(@ByVal Mat I0, @ByVal Mat I1, @ByVal Mat flow);
    public native void collectGarbage();
}

// Implementation of the Zach, Pock and Bischof Dual TV-L1 Optical Flow method
//
// see reference:
//   [1] C. Zach, T. Pock and H. Bischof, "A Duality Based Approach for Realtime TV-L1 Optical Flow".
//   [2] Javier Sanchez, Enric Meinhardt-Llopis and Gabriele Facciolo. "TV-L1 Optical Flow Estimation".
@Namespace("cv") public static native @Ptr DenseOpticalFlow createOptFlow_DualTVL1();



// #endif

// #endif


// Parsed from <opencv2/video/background_segm.hpp>

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

// #ifndef __OPENCV_BACKGROUND_SEGM_HPP__
// #define __OPENCV_BACKGROUND_SEGM_HPP__

// #include "opencv2/core/core.hpp"
// #include <list>

/**
 The Base Class for Background/Foreground Segmentation

 The class is only used to define the common interface for
 the whole family of background/foreground segmentation algorithms.
*/
@Namespace("cv") public static class BackgroundSubtractor extends Algorithm {
    static { Loader.load(); }
    public BackgroundSubtractor() { allocate(); }
    public BackgroundSubtractor(int size) { allocateArray(size); }
    public BackgroundSubtractor(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public BackgroundSubtractor position(int position) {
        return (BackgroundSubtractor)super.position(position);
    }

    /** the virtual destructor */
    /** the update operator that takes the next video frame and returns the current foreground mask as 8-bit binary image. */
    public native @Name("operator()") void apply(@ByVal Mat image, @ByVal Mat fgmask,
                                                  double learningRate/*=0*/);
    public native @Name("operator()") void apply(@ByVal Mat image, @ByVal Mat fgmask);

    /** computes a background image */
    public native void getBackgroundImage(@ByVal Mat backgroundImage);
}


/**
 Gaussian Mixture-based Backbround/Foreground Segmentation Algorithm

 The class implements the following algorithm:
 "An improved adaptive background mixture model for real-time tracking with shadow detection"
 P. KadewTraKuPong and R. Bowden,
 Proc. 2nd European Workshp on Advanced Video-Based Surveillance Systems, 2001."
 http://personal.ee.surrey.ac.uk/Personal/R.Bowden/publications/avbs01/avbs01.pdf

*/
@Namespace("cv") @NoOffset public static class BackgroundSubtractorMOG extends BackgroundSubtractor {
    static { Loader.load(); }
    public BackgroundSubtractorMOG(Pointer p) { super(p); }
    public BackgroundSubtractorMOG(int size) { allocateArray(size); }
    private native void allocateArray(int size);
    @Override public BackgroundSubtractorMOG position(int position) {
        return (BackgroundSubtractorMOG)super.position(position);
    }

    /** the default constructor */
    public BackgroundSubtractorMOG() { allocate(); }
    private native void allocate();
    /** the full constructor that takes the length of the history, the number of gaussian mixtures, the background ratio parameter and the noise strength */
    public BackgroundSubtractorMOG(int history, int nmixtures, double backgroundRatio, double noiseSigma/*=0*/) { allocate(history, nmixtures, backgroundRatio, noiseSigma); }
    private native void allocate(int history, int nmixtures, double backgroundRatio, double noiseSigma/*=0*/);
    public BackgroundSubtractorMOG(int history, int nmixtures, double backgroundRatio) { allocate(history, nmixtures, backgroundRatio); }
    private native void allocate(int history, int nmixtures, double backgroundRatio);
    /** the destructor */
    /** the update operator */
    public native @Name("operator()") void apply(@ByVal Mat image, @ByVal Mat fgmask, double learningRate/*=0*/);
    public native @Name("operator()") void apply(@ByVal Mat image, @ByVal Mat fgmask);

    /** re-initiaization method */
    public native void initialize(@ByVal Size frameSize, int frameType);

    public native AlgorithmInfo info();
}


/**
 The class implements the following algorithm:
 "Improved adaptive Gausian mixture model for background subtraction"
 Z.Zivkovic
 International Conference Pattern Recognition, UK, August, 2004.
 http://www.zoranz.net/Publications/zivkovic2004ICPR.pdf
*/
@Namespace("cv") @NoOffset public static class BackgroundSubtractorMOG2 extends BackgroundSubtractor {
    static { Loader.load(); }
    public BackgroundSubtractorMOG2(Pointer p) { super(p); }
    public BackgroundSubtractorMOG2(int size) { allocateArray(size); }
    private native void allocateArray(int size);
    @Override public BackgroundSubtractorMOG2 position(int position) {
        return (BackgroundSubtractorMOG2)super.position(position);
    }

    /** the default constructor */
    public BackgroundSubtractorMOG2() { allocate(); }
    private native void allocate();
    /** the full constructor that takes the length of the history, the number of gaussian mixtures, the background ratio parameter and the noise strength */
    public BackgroundSubtractorMOG2(int history,  float varThreshold, @Cast("bool") boolean bShadowDetection/*=true*/) { allocate(history, varThreshold, bShadowDetection); }
    private native void allocate(int history,  float varThreshold, @Cast("bool") boolean bShadowDetection/*=true*/);
    public BackgroundSubtractorMOG2(int history,  float varThreshold) { allocate(history, varThreshold); }
    private native void allocate(int history,  float varThreshold);
    /** the destructor */
    /** the update operator */
    public native @Name("operator()") void apply(@ByVal Mat image, @ByVal Mat fgmask, double learningRate/*=-1*/);
    public native @Name("operator()") void apply(@ByVal Mat image, @ByVal Mat fgmask);

    /** computes a background image which are the mean of all background gaussians */
    public native void getBackgroundImage(@ByVal Mat backgroundImage);

    /** re-initiaization method */
    public native void initialize(@ByVal Size frameSize, int frameType);

    public native AlgorithmInfo info();
    // Tau - shadow threshold. The shadow is detected if the pixel is darker
    //version of the background. Tau is a threshold on how much darker the shadow can be.
    //Tau= 0.5 means that if pixel is more than 2 times darker then it is not shadow
    //See: Prati,Mikic,Trivedi,Cucchiarra,"Detecting Moving Shadows...",IEEE PAMI,2003.
}

/**
 * Background Subtractor module. Takes a series of images and returns a sequence of mask (8UC1)
 * images of the same size, where 255 indicates Foreground and 0 represents Background.
 * This class implements an algorithm described in "Visual Tracking of Human Visitors under
 * Variable-Lighting Conditions for a Responsive Audio Art Installation," A. Godbehere,
 * A. Matsukawa, K. Goldberg, American Control Conference, Montreal, June 2012.
 */
@Namespace("cv") @NoOffset public static class BackgroundSubtractorGMG extends BackgroundSubtractor {
    static { Loader.load(); }
    public BackgroundSubtractorGMG(Pointer p) { super(p); }
    public BackgroundSubtractorGMG(int size) { allocateArray(size); }
    private native void allocateArray(int size);
    @Override public BackgroundSubtractorGMG position(int position) {
        return (BackgroundSubtractorGMG)super.position(position);
    }

    public BackgroundSubtractorGMG() { allocate(); }
    private native void allocate();
    public native AlgorithmInfo info();

    /**
     * Validate parameters and set up data structures for appropriate image size.
     * Must call before running on data.
     * @param frameSize input frame size
     * @param min       minimum value taken on by pixels in image sequence. Usually 0
     * @param max       maximum value taken on by pixels in image sequence. e.g. 1.0 or 255
     */
    public native void initialize(@ByVal Size frameSize, double min, double max);

    /**
     * Performs single-frame background subtraction and builds up a statistical background image
     * model.
     * @param image Input image
     * @param fgmask Output mask image representing foreground and background pixels
     */
    public native @Name("operator()") void apply(@ByVal Mat image, @ByVal Mat fgmask, double learningRate/*=-1.0*/);
    public native @Name("operator()") void apply(@ByVal Mat image, @ByVal Mat fgmask);

    /**
     * Releases all inner buffers.
     */
    public native void release();

    /** Total number of distinct colors to maintain in histogram. */
    public native int maxFeatures(); public native BackgroundSubtractorGMG maxFeatures(int maxFeatures);
    /** Set between 0.0 and 1.0, determines how quickly features are "forgotten" from histograms. */
    public native double learningRate(); public native BackgroundSubtractorGMG learningRate(double learningRate);
    /** Number of frames of video to use to initialize histograms. */
    public native int numInitializationFrames(); public native BackgroundSubtractorGMG numInitializationFrames(int numInitializationFrames);
    /** Number of discrete levels in each channel to be used in histograms. */
    public native int quantizationLevels(); public native BackgroundSubtractorGMG quantizationLevels(int quantizationLevels);
    /** Prior probability that any given pixel is a background pixel. A sensitivity parameter. */
    public native double backgroundPrior(); public native BackgroundSubtractorGMG backgroundPrior(double backgroundPrior);
    /** Value above which pixel is determined to be FG. */
    public native double decisionThreshold(); public native BackgroundSubtractorGMG decisionThreshold(double decisionThreshold);
    /** Smoothing radius, in pixels, for cleaning up FG image. */
    public native int smoothingRadius(); public native BackgroundSubtractorGMG smoothingRadius(int smoothingRadius);
    /** Perform background model update */
    public native @Cast("bool") boolean updateBackgroundModel(); public native BackgroundSubtractorGMG updateBackgroundModel(boolean updateBackgroundModel);
}



// #endif


}
