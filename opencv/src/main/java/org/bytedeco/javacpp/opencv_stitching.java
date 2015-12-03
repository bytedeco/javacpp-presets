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
import static org.bytedeco.javacpp.opencv_calib3d.*;
import static org.bytedeco.javacpp.opencv_objdetect.*;
import static org.bytedeco.javacpp.opencv_video.*;

public class opencv_stitching extends org.bytedeco.javacpp.presets.opencv_stitching {
    static { Loader.load(); }

// Parsed from <opencv2/stitching/detail/warpers.hpp>

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

// #ifndef __OPENCV_STITCHING_WARPERS_HPP__
// #define __OPENCV_STITCHING_WARPERS_HPP__

// #include "opencv2/core.hpp"
// #include "opencv2/core/cuda.hpp"
// #include "opencv2/imgproc.hpp"
// #include "opencv2/opencv_modules.hpp"

/** \addtogroup stitching_warp
 *  \{
<p>
/** \brief Rotation-only model image warper interface.
 */
@Namespace("cv::detail") public static class RotationWarper extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public RotationWarper(Pointer p) { super(p); }


    /** \brief Projects the image point.
    <p>
    @param pt Source point
    @param K Camera intrinsic parameters
    @param R Camera rotation matrix
    @return Projected point
     */
    public native @ByVal Point2f warpPoint(@Const @ByRef Point2f pt, @ByVal Mat K, @ByVal Mat R);

    /** \brief Builds the projection maps according to the given camera data.
    <p>
    @param src_size Source image size
    @param K Camera intrinsic parameters
    @param R Camera rotation matrix
    @param xmap Projection map for the x axis
    @param ymap Projection map for the y axis
    @return Projected image minimum bounding box
     */
    public native @ByVal Rect buildMaps(@ByVal Size src_size, @ByVal Mat K, @ByVal Mat R, @ByVal Mat xmap, @ByVal Mat ymap);

    /** \brief Projects the image.
    <p>
    @param src Source image
    @param K Camera intrinsic parameters
    @param R Camera rotation matrix
    @param interp_mode Interpolation mode
    @param border_mode Border extrapolation mode
    @param dst Projected image
    @return Project image top-left corner
     */
    public native @ByVal Point warp(@ByVal Mat src, @ByVal Mat K, @ByVal Mat R, int interp_mode, int border_mode,
                           @ByVal Mat dst);

    /** \brief Projects the image backward.
    <p>
    @param src Projected image
    @param K Camera intrinsic parameters
    @param R Camera rotation matrix
    @param interp_mode Interpolation mode
    @param border_mode Border extrapolation mode
    @param dst_size Backward-projected image size
    @param dst Backward-projected image
     */
    public native void warpBackward(@ByVal Mat src, @ByVal Mat K, @ByVal Mat R, int interp_mode, int border_mode,
                                  @ByVal Size dst_size, @ByVal Mat dst);

    /**
    @param src_size Source image bounding box
    @param K Camera intrinsic parameters
    @param R Camera rotation matrix
    @return Projected image minimum bounding box
     */
    public native @ByVal Rect warpRoi(@ByVal Size src_size, @ByVal Mat K, @ByVal Mat R);

    public native float getScale();
    public native void setScale(float arg0);
}

/** \brief Base class for warping logic implementation.
 */
@Namespace("cv::detail") public static class ProjectorBase extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public ProjectorBase() { super((Pointer)null); allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public ProjectorBase(int size) { super((Pointer)null); allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public ProjectorBase(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public ProjectorBase position(int position) {
        return (ProjectorBase)super.position(position);
    }

    public native void setCameraParams(@ByVal(nullValue = "cv::Mat::eye(3, 3, CV_32F)") Mat K/*=cv::Mat::eye(3, 3, CV_32F)*/,
                             @ByVal(nullValue = "cv::Mat::eye(3, 3, CV_32F)") Mat R/*=cv::Mat::eye(3, 3, CV_32F)*/,
                             @ByVal(nullValue = "cv::Mat::zeros(3, 1, CV_32F)") Mat T/*=cv::Mat::zeros(3, 1, CV_32F)*/);
    public native void setCameraParams();

    public native float scale(); public native ProjectorBase scale(float scale);
    public native float k(int i); public native ProjectorBase k(int i, float k);
    @MemberGetter public native FloatPointer k();
    public native float rinv(int i); public native ProjectorBase rinv(int i, float rinv);
    @MemberGetter public native FloatPointer rinv();
    public native float r_kinv(int i); public native ProjectorBase r_kinv(int i, float r_kinv);
    @MemberGetter public native FloatPointer r_kinv();
    public native float k_rinv(int i); public native ProjectorBase k_rinv(int i, float k_rinv);
    @MemberGetter public native FloatPointer k_rinv();
    public native float t(int i); public native ProjectorBase t(int i, float t);
    @MemberGetter public native FloatPointer t();
}

/** \brief Base class for rotation-based warper using a detail::ProjectorBase_ derived class.
 */


@Namespace("cv::detail") public static class PlaneProjector extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public PlaneProjector() { super((Pointer)null); allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public PlaneProjector(int size) { super((Pointer)null); allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public PlaneProjector(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public PlaneProjector position(int position) {
        return (PlaneProjector)super.position(position);
    }

    public native void mapForward(float x, float y, @ByRef FloatPointer u, @ByRef FloatPointer v);
    public native void mapForward(float x, float y, @ByRef FloatBuffer u, @ByRef FloatBuffer v);
    public native void mapForward(float x, float y, @ByRef float[] u, @ByRef float[] v);
    public native void mapBackward(float u, float v, @ByRef FloatPointer x, @ByRef FloatPointer y);
    public native void mapBackward(float u, float v, @ByRef FloatBuffer x, @ByRef FloatBuffer y);
    public native void mapBackward(float u, float v, @ByRef float[] x, @ByRef float[] y);
}

/** \brief Warper that maps an image onto the z = 1 plane.
 */
@Name("cv::detail::PlaneWarper") public static class DetailPlaneWarper extends RotationWarper {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public DetailPlaneWarper(Pointer p) { super(p); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public DetailPlaneWarper(int size) { super((Pointer)null); allocateArray(size); }
    private native void allocateArray(int size);
    @Override public DetailPlaneWarper position(int position) {
        return (DetailPlaneWarper)super.position(position);
    }

    /** \brief Construct an instance of the plane warper class.
    <p>
    @param scale Projected image scale multiplier
     */
    public DetailPlaneWarper(float scale/*=1.f*/) { super((Pointer)null); allocate(scale); }
    private native void allocate(float scale/*=1.f*/);
    public DetailPlaneWarper() { super((Pointer)null); allocate(); }
    private native void allocate();

    public native @ByVal Point2f warpPoint(@Const @ByRef Point2f pt, @ByVal Mat K, @ByVal Mat R);
    public native @ByVal Point2f warpPoint(@Const @ByRef Point2f pt, @ByVal Mat K, @ByVal Mat R, @ByVal Mat T);

    public native @ByVal Rect buildMaps(@ByVal Size src_size, @ByVal Mat K, @ByVal Mat R, @ByVal Mat T, @ByVal Mat xmap, @ByVal Mat ymap);
    public native @ByVal Rect buildMaps(@ByVal Size src_size, @ByVal Mat K, @ByVal Mat R, @ByVal Mat xmap, @ByVal Mat ymap);

    public native @ByVal Point warp(@ByVal Mat src, @ByVal Mat K, @ByVal Mat R,
                   int interp_mode, int border_mode, @ByVal Mat dst);
    public native @ByVal Point warp(@ByVal Mat src, @ByVal Mat K, @ByVal Mat R, @ByVal Mat T, int interp_mode, int border_mode,
                   @ByVal Mat dst);

    public native @ByVal Rect warpRoi(@ByVal Size src_size, @ByVal Mat K, @ByVal Mat R);
    public native @ByVal Rect warpRoi(@ByVal Size src_size, @ByVal Mat K, @ByVal Mat R, @ByVal Mat T);
}


@Namespace("cv::detail") public static class SphericalProjector extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public SphericalProjector() { super((Pointer)null); allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public SphericalProjector(int size) { super((Pointer)null); allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public SphericalProjector(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public SphericalProjector position(int position) {
        return (SphericalProjector)super.position(position);
    }

    public native void mapForward(float x, float y, @ByRef FloatPointer u, @ByRef FloatPointer v);
    public native void mapForward(float x, float y, @ByRef FloatBuffer u, @ByRef FloatBuffer v);
    public native void mapForward(float x, float y, @ByRef float[] u, @ByRef float[] v);
    public native void mapBackward(float u, float v, @ByRef FloatPointer x, @ByRef FloatPointer y);
    public native void mapBackward(float u, float v, @ByRef FloatBuffer x, @ByRef FloatBuffer y);
    public native void mapBackward(float u, float v, @ByRef float[] x, @ByRef float[] y);
}


/** \brief Warper that maps an image onto the unit sphere located at the origin.
 <p>
 Projects image onto unit sphere with origin at (0, 0, 0).
 Poles are located at (0, -1, 0) and (0, 1, 0) points.
*/
@Name("cv::detail::SphericalWarper") public static class DetailSphericalWarper extends RotationWarper {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public DetailSphericalWarper(Pointer p) { super(p); }

    /** \brief Construct an instance of the spherical warper class.
    <p>
    @param scale Projected image scale multiplier
     */
    public DetailSphericalWarper(float scale) { super((Pointer)null); allocate(scale); }
    private native void allocate(float scale);

    public native @ByVal Rect buildMaps(@ByVal Size src_size, @ByVal Mat K, @ByVal Mat R, @ByVal Mat xmap, @ByVal Mat ymap);
    public native @ByVal Point warp(@ByVal Mat src, @ByVal Mat K, @ByVal Mat R, int interp_mode, int border_mode, @ByVal Mat dst);
}


@Namespace("cv::detail") public static class CylindricalProjector extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public CylindricalProjector() { super((Pointer)null); allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public CylindricalProjector(int size) { super((Pointer)null); allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public CylindricalProjector(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public CylindricalProjector position(int position) {
        return (CylindricalProjector)super.position(position);
    }

    public native void mapForward(float x, float y, @ByRef FloatPointer u, @ByRef FloatPointer v);
    public native void mapForward(float x, float y, @ByRef FloatBuffer u, @ByRef FloatBuffer v);
    public native void mapForward(float x, float y, @ByRef float[] u, @ByRef float[] v);
    public native void mapBackward(float u, float v, @ByRef FloatPointer x, @ByRef FloatPointer y);
    public native void mapBackward(float u, float v, @ByRef FloatBuffer x, @ByRef FloatBuffer y);
    public native void mapBackward(float u, float v, @ByRef float[] x, @ByRef float[] y);
}


/** \brief Warper that maps an image onto the x\*x + z\*z = 1 cylinder.
 */
@Name("cv::detail::CylindricalWarper") public static class DetailCylindricalWarper extends RotationWarper {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public DetailCylindricalWarper(Pointer p) { super(p); }

    /** \brief Construct an instance of the cylindrical warper class.
    <p>
    @param scale Projected image scale multiplier
     */
    public DetailCylindricalWarper(float scale) { super((Pointer)null); allocate(scale); }
    private native void allocate(float scale);

    public native @ByVal Rect buildMaps(@ByVal Size src_size, @ByVal Mat K, @ByVal Mat R, @ByVal Mat xmap, @ByVal Mat ymap);
    public native @ByVal Point warp(@ByVal Mat src, @ByVal Mat K, @ByVal Mat R, int interp_mode, int border_mode, @ByVal Mat dst);
}


@Namespace("cv::detail") public static class FisheyeProjector extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public FisheyeProjector() { super((Pointer)null); allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public FisheyeProjector(int size) { super((Pointer)null); allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public FisheyeProjector(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public FisheyeProjector position(int position) {
        return (FisheyeProjector)super.position(position);
    }

    public native void mapForward(float x, float y, @ByRef FloatPointer u, @ByRef FloatPointer v);
    public native void mapForward(float x, float y, @ByRef FloatBuffer u, @ByRef FloatBuffer v);
    public native void mapForward(float x, float y, @ByRef float[] u, @ByRef float[] v);
    public native void mapBackward(float u, float v, @ByRef FloatPointer x, @ByRef FloatPointer y);
    public native void mapBackward(float u, float v, @ByRef FloatBuffer x, @ByRef FloatBuffer y);
    public native void mapBackward(float u, float v, @ByRef float[] x, @ByRef float[] y);
}


@Name("cv::detail::FisheyeWarper") public static class DetailFisheyeWarper extends RotationWarper {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public DetailFisheyeWarper(Pointer p) { super(p); }

    public DetailFisheyeWarper(float scale) { super((Pointer)null); allocate(scale); }
    private native void allocate(float scale);
}


@Namespace("cv::detail") public static class StereographicProjector extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public StereographicProjector() { super((Pointer)null); allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public StereographicProjector(int size) { super((Pointer)null); allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public StereographicProjector(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public StereographicProjector position(int position) {
        return (StereographicProjector)super.position(position);
    }

    public native void mapForward(float x, float y, @ByRef FloatPointer u, @ByRef FloatPointer v);
    public native void mapForward(float x, float y, @ByRef FloatBuffer u, @ByRef FloatBuffer v);
    public native void mapForward(float x, float y, @ByRef float[] u, @ByRef float[] v);
    public native void mapBackward(float u, float v, @ByRef FloatPointer x, @ByRef FloatPointer y);
    public native void mapBackward(float u, float v, @ByRef FloatBuffer x, @ByRef FloatBuffer y);
    public native void mapBackward(float u, float v, @ByRef float[] x, @ByRef float[] y);
}


@Name("cv::detail::StereographicWarper") public static class DetailStereographicWarper extends RotationWarper {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public DetailStereographicWarper(Pointer p) { super(p); }

    public DetailStereographicWarper(float scale) { super((Pointer)null); allocate(scale); }
    private native void allocate(float scale);
}


@Namespace("cv::detail") @NoOffset public static class CompressedRectilinearProjector extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public CompressedRectilinearProjector() { super((Pointer)null); allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public CompressedRectilinearProjector(int size) { super((Pointer)null); allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public CompressedRectilinearProjector(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public CompressedRectilinearProjector position(int position) {
        return (CompressedRectilinearProjector)super.position(position);
    }

    public native float a(); public native CompressedRectilinearProjector a(float a);
    public native float b(); public native CompressedRectilinearProjector b(float b);

    public native void mapForward(float x, float y, @ByRef FloatPointer u, @ByRef FloatPointer v);
    public native void mapForward(float x, float y, @ByRef FloatBuffer u, @ByRef FloatBuffer v);
    public native void mapForward(float x, float y, @ByRef float[] u, @ByRef float[] v);
    public native void mapBackward(float u, float v, @ByRef FloatPointer x, @ByRef FloatPointer y);
    public native void mapBackward(float u, float v, @ByRef FloatBuffer x, @ByRef FloatBuffer y);
    public native void mapBackward(float u, float v, @ByRef float[] x, @ByRef float[] y);
}


@Name("cv::detail::CompressedRectilinearWarper") public static class DetailCompressedRectilinearWarper extends RotationWarper {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public DetailCompressedRectilinearWarper(Pointer p) { super(p); }

    public DetailCompressedRectilinearWarper(float scale, float A/*=1*/, float B/*=1*/) { super((Pointer)null); allocate(scale, A, B); }
    private native void allocate(float scale, float A/*=1*/, float B/*=1*/);
    public DetailCompressedRectilinearWarper(float scale) { super((Pointer)null); allocate(scale); }
    private native void allocate(float scale);
}


@Namespace("cv::detail") @NoOffset public static class CompressedRectilinearPortraitProjector extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public CompressedRectilinearPortraitProjector() { super((Pointer)null); allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public CompressedRectilinearPortraitProjector(int size) { super((Pointer)null); allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public CompressedRectilinearPortraitProjector(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public CompressedRectilinearPortraitProjector position(int position) {
        return (CompressedRectilinearPortraitProjector)super.position(position);
    }

    public native float a(); public native CompressedRectilinearPortraitProjector a(float a);
    public native float b(); public native CompressedRectilinearPortraitProjector b(float b);

    public native void mapForward(float x, float y, @ByRef FloatPointer u, @ByRef FloatPointer v);
    public native void mapForward(float x, float y, @ByRef FloatBuffer u, @ByRef FloatBuffer v);
    public native void mapForward(float x, float y, @ByRef float[] u, @ByRef float[] v);
    public native void mapBackward(float u, float v, @ByRef FloatPointer x, @ByRef FloatPointer y);
    public native void mapBackward(float u, float v, @ByRef FloatBuffer x, @ByRef FloatBuffer y);
    public native void mapBackward(float u, float v, @ByRef float[] x, @ByRef float[] y);
}


@Name("cv::detail::CompressedRectilinearPortraitWarper") public static class DetailCompressedRectilinearPortraitWarper extends RotationWarper {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public DetailCompressedRectilinearPortraitWarper(Pointer p) { super(p); }

   public DetailCompressedRectilinearPortraitWarper(float scale, float A/*=1*/, float B/*=1*/) { super((Pointer)null); allocate(scale, A, B); }
   private native void allocate(float scale, float A/*=1*/, float B/*=1*/);
   public DetailCompressedRectilinearPortraitWarper(float scale) { super((Pointer)null); allocate(scale); }
   private native void allocate(float scale);
}


@Namespace("cv::detail") @NoOffset public static class PaniniProjector extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public PaniniProjector() { super((Pointer)null); allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public PaniniProjector(int size) { super((Pointer)null); allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public PaniniProjector(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public PaniniProjector position(int position) {
        return (PaniniProjector)super.position(position);
    }

    public native float a(); public native PaniniProjector a(float a);
    public native float b(); public native PaniniProjector b(float b);

    public native void mapForward(float x, float y, @ByRef FloatPointer u, @ByRef FloatPointer v);
    public native void mapForward(float x, float y, @ByRef FloatBuffer u, @ByRef FloatBuffer v);
    public native void mapForward(float x, float y, @ByRef float[] u, @ByRef float[] v);
    public native void mapBackward(float u, float v, @ByRef FloatPointer x, @ByRef FloatPointer y);
    public native void mapBackward(float u, float v, @ByRef FloatBuffer x, @ByRef FloatBuffer y);
    public native void mapBackward(float u, float v, @ByRef float[] x, @ByRef float[] y);
}


@Name("cv::detail::PaniniWarper") public static class DetailPaniniWarper extends RotationWarper {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public DetailPaniniWarper(Pointer p) { super(p); }

   public DetailPaniniWarper(float scale, float A/*=1*/, float B/*=1*/) { super((Pointer)null); allocate(scale, A, B); }
   private native void allocate(float scale, float A/*=1*/, float B/*=1*/);
   public DetailPaniniWarper(float scale) { super((Pointer)null); allocate(scale); }
   private native void allocate(float scale);
}


@Namespace("cv::detail") @NoOffset public static class PaniniPortraitProjector extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public PaniniPortraitProjector() { super((Pointer)null); allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public PaniniPortraitProjector(int size) { super((Pointer)null); allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public PaniniPortraitProjector(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public PaniniPortraitProjector position(int position) {
        return (PaniniPortraitProjector)super.position(position);
    }

    public native float a(); public native PaniniPortraitProjector a(float a);
    public native float b(); public native PaniniPortraitProjector b(float b);

    public native void mapForward(float x, float y, @ByRef FloatPointer u, @ByRef FloatPointer v);
    public native void mapForward(float x, float y, @ByRef FloatBuffer u, @ByRef FloatBuffer v);
    public native void mapForward(float x, float y, @ByRef float[] u, @ByRef float[] v);
    public native void mapBackward(float u, float v, @ByRef FloatPointer x, @ByRef FloatPointer y);
    public native void mapBackward(float u, float v, @ByRef FloatBuffer x, @ByRef FloatBuffer y);
    public native void mapBackward(float u, float v, @ByRef float[] x, @ByRef float[] y);
}


@Name("cv::detail::PaniniPortraitWarper") public static class DetailPaniniPortraitWarper extends RotationWarper {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public DetailPaniniPortraitWarper(Pointer p) { super(p); }

   public DetailPaniniPortraitWarper(float scale, float A/*=1*/, float B/*=1*/) { super((Pointer)null); allocate(scale, A, B); }
   private native void allocate(float scale, float A/*=1*/, float B/*=1*/);
   public DetailPaniniPortraitWarper(float scale) { super((Pointer)null); allocate(scale); }
   private native void allocate(float scale);

}


@Namespace("cv::detail") public static class MercatorProjector extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public MercatorProjector() { super((Pointer)null); allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public MercatorProjector(int size) { super((Pointer)null); allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public MercatorProjector(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public MercatorProjector position(int position) {
        return (MercatorProjector)super.position(position);
    }

    public native void mapForward(float x, float y, @ByRef FloatPointer u, @ByRef FloatPointer v);
    public native void mapForward(float x, float y, @ByRef FloatBuffer u, @ByRef FloatBuffer v);
    public native void mapForward(float x, float y, @ByRef float[] u, @ByRef float[] v);
    public native void mapBackward(float u, float v, @ByRef FloatPointer x, @ByRef FloatPointer y);
    public native void mapBackward(float u, float v, @ByRef FloatBuffer x, @ByRef FloatBuffer y);
    public native void mapBackward(float u, float v, @ByRef float[] x, @ByRef float[] y);
}


@Name("cv::detail::MercatorWarper") public static class DetailMercatorWarper extends RotationWarper {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public DetailMercatorWarper(Pointer p) { super(p); }

    public DetailMercatorWarper(float scale) { super((Pointer)null); allocate(scale); }
    private native void allocate(float scale);
}


@Namespace("cv::detail") public static class TransverseMercatorProjector extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public TransverseMercatorProjector() { super((Pointer)null); allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public TransverseMercatorProjector(int size) { super((Pointer)null); allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public TransverseMercatorProjector(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public TransverseMercatorProjector position(int position) {
        return (TransverseMercatorProjector)super.position(position);
    }

    public native void mapForward(float x, float y, @ByRef FloatPointer u, @ByRef FloatPointer v);
    public native void mapForward(float x, float y, @ByRef FloatBuffer u, @ByRef FloatBuffer v);
    public native void mapForward(float x, float y, @ByRef float[] u, @ByRef float[] v);
    public native void mapBackward(float u, float v, @ByRef FloatPointer x, @ByRef FloatPointer y);
    public native void mapBackward(float u, float v, @ByRef FloatBuffer x, @ByRef FloatBuffer y);
    public native void mapBackward(float u, float v, @ByRef float[] x, @ByRef float[] y);
}


@Name("cv::detail::TransverseMercatorWarper") public static class DetailTransverseMercatorWarper extends RotationWarper {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public DetailTransverseMercatorWarper(Pointer p) { super(p); }

    public DetailTransverseMercatorWarper(float scale) { super((Pointer)null); allocate(scale); }
    private native void allocate(float scale);
}


@Name("cv::detail::PlaneWarperGpu") public static class DetailPlaneWarperGpu extends RotationWarper {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public DetailPlaneWarperGpu(Pointer p) { super(p); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public DetailPlaneWarperGpu(int size) { super((Pointer)null); allocateArray(size); }
    private native void allocateArray(int size);
    @Override public DetailPlaneWarperGpu position(int position) {
        return (DetailPlaneWarperGpu)super.position(position);
    }

    public DetailPlaneWarperGpu(float scale/*=1.f*/) { super((Pointer)null); allocate(scale); }
    private native void allocate(float scale/*=1.f*/);
    public DetailPlaneWarperGpu() { super((Pointer)null); allocate(); }
    private native void allocate();

    public native @ByVal Rect buildMaps(@ByVal Size src_size, @ByVal Mat K, @ByVal Mat R, @ByVal Mat xmap, @ByVal Mat ymap);

    public native @ByVal Rect buildMaps(@ByVal Size src_size, @ByVal Mat K, @ByVal Mat R, @ByVal Mat T, @ByVal Mat xmap, @ByVal Mat ymap);

    public native @ByVal Point warp(@ByVal Mat src, @ByVal Mat K, @ByVal Mat R, int interp_mode, int border_mode,
                   @ByVal Mat dst);

    public native @ByVal Point warp(@ByVal Mat src, @ByVal Mat K, @ByVal Mat R, @ByVal Mat T, int interp_mode, int border_mode,
                   @ByVal Mat dst);
}


@Name("cv::detail::SphericalWarperGpu") public static class DetailSphericalWarperGpu extends RotationWarper {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public DetailSphericalWarperGpu(Pointer p) { super(p); }

    public DetailSphericalWarperGpu(float scale) { super((Pointer)null); allocate(scale); }
    private native void allocate(float scale);

    public native @ByVal Rect buildMaps(@ByVal Size src_size, @ByVal Mat K, @ByVal Mat R, @ByVal Mat xmap, @ByVal Mat ymap);

    public native @ByVal Point warp(@ByVal Mat src, @ByVal Mat K, @ByVal Mat R, int interp_mode, int border_mode,
                   @ByVal Mat dst);
}


@Name("cv::detail::CylindricalWarperGpu") public static class DetailCylindricalWarperGpu extends RotationWarper {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public DetailCylindricalWarperGpu(Pointer p) { super(p); }

    public DetailCylindricalWarperGpu(float scale) { super((Pointer)null); allocate(scale); }
    private native void allocate(float scale);

    public native @ByVal Rect buildMaps(@ByVal Size src_size, @ByVal Mat K, @ByVal Mat R, @ByVal Mat xmap, @ByVal Mat ymap);

    public native @ByVal Point warp(@ByVal Mat src, @ByVal Mat K, @ByVal Mat R, int interp_mode, int border_mode,
                   @ByVal Mat dst);
}


@Namespace("cv::detail") public static class SphericalPortraitProjector extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public SphericalPortraitProjector() { super((Pointer)null); allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public SphericalPortraitProjector(int size) { super((Pointer)null); allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public SphericalPortraitProjector(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public SphericalPortraitProjector position(int position) {
        return (SphericalPortraitProjector)super.position(position);
    }

    public native void mapForward(float x, float y, @ByRef FloatPointer u, @ByRef FloatPointer v);
    public native void mapForward(float x, float y, @ByRef FloatBuffer u, @ByRef FloatBuffer v);
    public native void mapForward(float x, float y, @ByRef float[] u, @ByRef float[] v);
    public native void mapBackward(float u, float v, @ByRef FloatPointer x, @ByRef FloatPointer y);
    public native void mapBackward(float u, float v, @ByRef FloatBuffer x, @ByRef FloatBuffer y);
    public native void mapBackward(float u, float v, @ByRef float[] x, @ByRef float[] y);
}


// Projects image onto unit sphere with origin at (0, 0, 0).
// Poles are located NOT at (0, -1, 0) and (0, 1, 0) points, BUT at (1, 0, 0) and (-1, 0, 0) points.
@Namespace("cv::detail") public static class SphericalPortraitWarper extends RotationWarper {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public SphericalPortraitWarper(Pointer p) { super(p); }

    public SphericalPortraitWarper(float scale) { super((Pointer)null); allocate(scale); }
    private native void allocate(float scale);
}

@Namespace("cv::detail") public static class CylindricalPortraitProjector extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public CylindricalPortraitProjector() { super((Pointer)null); allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public CylindricalPortraitProjector(int size) { super((Pointer)null); allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public CylindricalPortraitProjector(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public CylindricalPortraitProjector position(int position) {
        return (CylindricalPortraitProjector)super.position(position);
    }

    public native void mapForward(float x, float y, @ByRef FloatPointer u, @ByRef FloatPointer v);
    public native void mapForward(float x, float y, @ByRef FloatBuffer u, @ByRef FloatBuffer v);
    public native void mapForward(float x, float y, @ByRef float[] u, @ByRef float[] v);
    public native void mapBackward(float u, float v, @ByRef FloatPointer x, @ByRef FloatPointer y);
    public native void mapBackward(float u, float v, @ByRef FloatBuffer x, @ByRef FloatBuffer y);
    public native void mapBackward(float u, float v, @ByRef float[] x, @ByRef float[] y);
}


@Namespace("cv::detail") public static class CylindricalPortraitWarper extends RotationWarper {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public CylindricalPortraitWarper(Pointer p) { super(p); }

    public CylindricalPortraitWarper(float scale) { super((Pointer)null); allocate(scale); }
    private native void allocate(float scale);
}

@Namespace("cv::detail") public static class PlanePortraitProjector extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public PlanePortraitProjector() { super((Pointer)null); allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public PlanePortraitProjector(int size) { super((Pointer)null); allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public PlanePortraitProjector(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public PlanePortraitProjector position(int position) {
        return (PlanePortraitProjector)super.position(position);
    }

    public native void mapForward(float x, float y, @ByRef FloatPointer u, @ByRef FloatPointer v);
    public native void mapForward(float x, float y, @ByRef FloatBuffer u, @ByRef FloatBuffer v);
    public native void mapForward(float x, float y, @ByRef float[] u, @ByRef float[] v);
    public native void mapBackward(float u, float v, @ByRef FloatPointer x, @ByRef FloatPointer y);
    public native void mapBackward(float u, float v, @ByRef FloatBuffer x, @ByRef FloatBuffer y);
    public native void mapBackward(float u, float v, @ByRef float[] x, @ByRef float[] y);
}


@Namespace("cv::detail") public static class PlanePortraitWarper extends RotationWarper {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public PlanePortraitWarper(Pointer p) { super(p); }

    public PlanePortraitWarper(float scale) { super((Pointer)null); allocate(scale); }
    private native void allocate(float scale);
}

/** \} stitching_warp */

 // namespace detail
 // namespace cv

// #include "warpers_inl.hpp"

// #endif // __OPENCV_STITCHING_WARPERS_HPP__


// Parsed from <opencv2/stitching/detail/matchers.hpp>

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

// #ifndef __OPENCV_STITCHING_MATCHERS_HPP__
// #define __OPENCV_STITCHING_MATCHERS_HPP__

// #include "opencv2/core.hpp"
// #include "opencv2/features2d.hpp"

// #include "opencv2/opencv_modules.hpp"

// #ifdef HAVE_OPENCV_XFEATURES2D
// #endif

/** \addtogroup stitching_match
 *  \{
<p>
/** \brief Structure containing image keypoints and descriptors. */
@Namespace("cv::detail") public static class ImageFeatures extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public ImageFeatures() { super((Pointer)null); allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public ImageFeatures(int size) { super((Pointer)null); allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public ImageFeatures(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public ImageFeatures position(int position) {
        return (ImageFeatures)super.position(position);
    }

    public native int img_idx(); public native ImageFeatures img_idx(int img_idx);
    public native @ByRef Size img_size(); public native ImageFeatures img_size(Size img_size);
    public native @ByRef KeyPointVector keypoints(); public native ImageFeatures keypoints(KeyPointVector keypoints);
    public native @ByRef UMat descriptors(); public native ImageFeatures descriptors(UMat descriptors);
}

/** \brief Feature finders base class */
@Namespace("cv::detail") public static class FeaturesFinder extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public FeaturesFinder(Pointer p) { super(p); }

    /** \overload */
    public native @Name("operator ()") void apply(@ByVal Mat image, @ByRef ImageFeatures features);
    /** \brief Finds features in the given image.
    <p>
    @param image Source image
    @param features Found features
    @param rois Regions of interest
    <p>
    \sa detail::ImageFeatures, Rect_
    */
    public native @Name("operator ()") void apply(@ByVal Mat image, @ByRef ImageFeatures features, @Const @ByRef RectVector rois);
    /** \brief Frees unused memory allocated before if there is any. */
    public native void collectGarbage();
}

/** \brief SURF features finder.
<p>
\sa detail::FeaturesFinder, SURF
*/
@Namespace("cv::detail") @NoOffset public static class SurfFeaturesFinder extends FeaturesFinder {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public SurfFeaturesFinder(Pointer p) { super(p); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public SurfFeaturesFinder(int size) { super((Pointer)null); allocateArray(size); }
    private native void allocateArray(int size);
    @Override public SurfFeaturesFinder position(int position) {
        return (SurfFeaturesFinder)super.position(position);
    }

    public SurfFeaturesFinder(double hess_thresh/*=300.*/, int num_octaves/*=3*/, int num_layers/*=4*/,
                           int num_octaves_descr/*=3*/, int num_layers_descr/*=4*/) { super((Pointer)null); allocate(hess_thresh, num_octaves, num_layers, num_octaves_descr, num_layers_descr); }
    private native void allocate(double hess_thresh/*=300.*/, int num_octaves/*=3*/, int num_layers/*=4*/,
                           int num_octaves_descr/*=3*/, int num_layers_descr/*=4*/);
    public SurfFeaturesFinder() { super((Pointer)null); allocate(); }
    private native void allocate();
}

/** \brief ORB features finder. :
<p>
\sa detail::FeaturesFinder, ORB
*/
@Namespace("cv::detail") @NoOffset public static class OrbFeaturesFinder extends FeaturesFinder {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public OrbFeaturesFinder(Pointer p) { super(p); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public OrbFeaturesFinder(int size) { super((Pointer)null); allocateArray(size); }
    private native void allocateArray(int size);
    @Override public OrbFeaturesFinder position(int position) {
        return (OrbFeaturesFinder)super.position(position);
    }

    public OrbFeaturesFinder(@ByVal(nullValue = "cv::Size(3,1)") Size _grid_size/*=cv::Size(3,1)*/, int nfeatures/*=1500*/, float scaleFactor/*=1.3f*/, int nlevels/*=5*/) { super((Pointer)null); allocate(_grid_size, nfeatures, scaleFactor, nlevels); }
    private native void allocate(@ByVal(nullValue = "cv::Size(3,1)") Size _grid_size/*=cv::Size(3,1)*/, int nfeatures/*=1500*/, float scaleFactor/*=1.3f*/, int nlevels/*=5*/);
    public OrbFeaturesFinder() { super((Pointer)null); allocate(); }
    private native void allocate();
}


// #ifdef HAVE_OPENCV_XFEATURES2D
// #endif

/** \brief Structure containing information about matches between two images.
<p>
It's assumed that there is a homography between those images.
*/
@Namespace("cv::detail") @NoOffset public static class MatchesInfo extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public MatchesInfo(Pointer p) { super(p); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public MatchesInfo(int size) { super((Pointer)null); allocateArray(size); }
    private native void allocateArray(int size);
    @Override public MatchesInfo position(int position) {
        return (MatchesInfo)super.position(position);
    }

    public MatchesInfo() { super((Pointer)null); allocate(); }
    private native void allocate();
    public MatchesInfo(@Const @ByRef MatchesInfo other) { super((Pointer)null); allocate(other); }
    private native void allocate(@Const @ByRef MatchesInfo other);
    public native @Const @ByRef @Name("operator =") MatchesInfo put(@Const @ByRef MatchesInfo other);

    /** Images indices (optional) */
    public native int src_img_idx(); public native MatchesInfo src_img_idx(int src_img_idx);
    public native int dst_img_idx(); public native MatchesInfo dst_img_idx(int dst_img_idx);
    public native @ByRef DMatchVector matches(); public native MatchesInfo matches(DMatchVector matches);
    /** Geometrically consistent matches mask */
    public native @Cast("uchar*") @StdVector BytePointer inliers_mask(); public native MatchesInfo inliers_mask(BytePointer inliers_mask);
    /** Number of geometrically consistent matches */
    public native int num_inliers(); public native MatchesInfo num_inliers(int num_inliers);
    /** Estimated homography */
    public native @ByRef Mat H(); public native MatchesInfo H(Mat H);
    /** Confidence two images are from the same panorama */
    public native double confidence(); public native MatchesInfo confidence(double confidence);
}

/** \brief Feature matchers base class. */
@Namespace("cv::detail") @NoOffset public static class FeaturesMatcher extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public FeaturesMatcher(Pointer p) { super(p); }


    /** \overload
    @param features1 First image features
    @param features2 Second image features
    @param matches_info Found matches
    */
    public native @Name("operator ()") void apply(@Const @ByRef ImageFeatures features1, @Const @ByRef ImageFeatures features2,
                         @ByRef MatchesInfo matches_info);

    /** \brief Performs images matching.
    <p>
    @param features Features of the source images
    @param pairwise_matches Found pairwise matches
    @param mask Mask indicating which image pairs must be matched
    <p>
    The function is parallelized with the TBB library.
    <p>
    \sa detail::MatchesInfo
    */
    public native @Name("operator ()") void apply(@StdVector ImageFeatures features, @StdVector MatchesInfo pairwise_matches,
                         @Const @ByRef(nullValue = "cv::UMat()") UMat mask/*=cv::UMat()*/);
    public native @Name("operator ()") void apply(@StdVector ImageFeatures features, @StdVector MatchesInfo pairwise_matches);

    /** @return True, if it's possible to use the same matcher instance in parallel, false otherwise
    */
    public native @Cast("bool") boolean isThreadSafe();

    /** \brief Frees unused memory allocated before if there is any.
    */
    public native void collectGarbage();
}

/** \brief Features matcher which finds two best matches for each feature and leaves the best one only if the
ratio between descriptor distances is greater than the threshold match_conf
<p>
\sa detail::FeaturesMatcher
 */
@Namespace("cv::detail") @NoOffset public static class BestOf2NearestMatcher extends FeaturesMatcher {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public BestOf2NearestMatcher(Pointer p) { super(p); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public BestOf2NearestMatcher(int size) { super((Pointer)null); allocateArray(size); }
    private native void allocateArray(int size);
    @Override public BestOf2NearestMatcher position(int position) {
        return (BestOf2NearestMatcher)super.position(position);
    }

    /** \brief Constructs a "best of 2 nearest" matcher.
    <p>
    @param try_use_gpu Should try to use GPU or not
    @param match_conf Match distances ration threshold
    @param num_matches_thresh1 Minimum number of matches required for the 2D projective transform
    estimation used in the inliers classification step
    @param num_matches_thresh2 Minimum number of matches required for the 2D projective transform
    re-estimation on inliers
     */
    public BestOf2NearestMatcher(@Cast("bool") boolean try_use_gpu/*=false*/, float match_conf/*=0.3f*/, int num_matches_thresh1/*=6*/,
                              int num_matches_thresh2/*=6*/) { super((Pointer)null); allocate(try_use_gpu, match_conf, num_matches_thresh1, num_matches_thresh2); }
    private native void allocate(@Cast("bool") boolean try_use_gpu/*=false*/, float match_conf/*=0.3f*/, int num_matches_thresh1/*=6*/,
                              int num_matches_thresh2/*=6*/);
    public BestOf2NearestMatcher() { super((Pointer)null); allocate(); }
    private native void allocate();

    public native void collectGarbage();
}

@Namespace("cv::detail") @NoOffset public static class BestOf2NearestRangeMatcher extends BestOf2NearestMatcher {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public BestOf2NearestRangeMatcher(Pointer p) { super(p); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public BestOf2NearestRangeMatcher(int size) { super((Pointer)null); allocateArray(size); }
    private native void allocateArray(int size);
    @Override public BestOf2NearestRangeMatcher position(int position) {
        return (BestOf2NearestRangeMatcher)super.position(position);
    }

    public BestOf2NearestRangeMatcher(int range_width/*=5*/, @Cast("bool") boolean try_use_gpu/*=false*/, float match_conf/*=0.3f*/,
                                int num_matches_thresh1/*=6*/, int num_matches_thresh2/*=6*/) { super((Pointer)null); allocate(range_width, try_use_gpu, match_conf, num_matches_thresh1, num_matches_thresh2); }
    private native void allocate(int range_width/*=5*/, @Cast("bool") boolean try_use_gpu/*=false*/, float match_conf/*=0.3f*/,
                                int num_matches_thresh1/*=6*/, int num_matches_thresh2/*=6*/);
    public BestOf2NearestRangeMatcher() { super((Pointer)null); allocate(); }
    private native void allocate();

    public native @Name("operator ()") void apply(@StdVector ImageFeatures features, @StdVector MatchesInfo pairwise_matches,
                         @Const @ByRef(nullValue = "cv::UMat()") UMat mask/*=cv::UMat()*/);
    public native @Name("operator ()") void apply(@StdVector ImageFeatures features, @StdVector MatchesInfo pairwise_matches);
}

/** \} stitching_match */

 // namespace detail
 // namespace cv

// #endif // __OPENCV_STITCHING_MATCHERS_HPP__


// Parsed from <opencv2/stitching/detail/util.hpp>

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

// #ifndef __OPENCV_STITCHING_UTIL_HPP__
// #define __OPENCV_STITCHING_UTIL_HPP__

// #include <list>
// #include "opencv2/core.hpp"

// #ifndef ENABLE_LOG
public static final int ENABLE_LOG = 0;
// #endif

// TODO remove LOG macros, add logging class
// #if ENABLE_LOG
// #ifdef ANDROID
//   #include <iostream>
//   #include <sstream>
//   #include <android/log.h>
//   #define LOG_STITCHING_MSG(msg)
//     do {
//         Stringstream _os;
//         _os << msg;
//        __android_log_print(ANDROID_LOG_DEBUG, "STITCHING", "%s", _os.str().c_str());
//     } while(0);
// #else
//   #include <iostream>
//   #define LOG_STITCHING_MSG(msg) for(;;) { std::cout << msg; std::cout.fsh(); break; }
// #endif
// #else
//   #define LOG_STITCHING_MSG(msg)
// #endif

// #define LOG_(_level, _msg)
//     for(;;)
//     {
//         using namespace std;
//         if ((_level) >= ::cv::detail::stitchingLogLevel())
//         {
//             LOG_STITCHING_MSG(_msg);
//         }
//     break;
//     }


// #define LOG(msg) LOG_(1, msg)
// #define LOG_CHAT(msg) LOG_(0, msg)

// #define LOGLN(msg) LOG(msg << std::endl)
// #define LOGLN_CHAT(msg) LOG_CHAT(msg << std::endl)

//#if DEBUG_LOG_CHAT
//  #define LOG_CHAT(msg) LOG(msg)
//  #define LOGLN_CHAT(msg) LOGLN(msg)
//#else
//  #define LOG_CHAT(msg) do{}while(0)
//  #define LOGLN_CHAT(msg) do{}while(0)
//#endif

/** \addtogroup stitching
 *  \{ */

@Namespace("cv::detail") @NoOffset public static class DisjointSets extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public DisjointSets(Pointer p) { super(p); }

    public DisjointSets(int elem_count/*=0*/) { super((Pointer)null); allocate(elem_count); }
    private native void allocate(int elem_count/*=0*/);
    public DisjointSets() { super((Pointer)null); allocate(); }
    private native void allocate();

    public native void createOneElemSets(int elem_count);
    public native int findSetByElem(int elem);
    public native int mergeSets(int set1, int set2);

    public native @StdVector IntPointer parent(); public native DisjointSets parent(IntPointer parent);
    public native @StdVector IntPointer size(); public native DisjointSets size(IntPointer size);
}


@Namespace("cv::detail") @NoOffset public static class GraphEdge extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public GraphEdge(Pointer p) { super(p); }

    public GraphEdge(int from, int to, float weight) { super((Pointer)null); allocate(from, to, weight); }
    private native void allocate(int from, int to, float weight);
    public native @Cast("bool") @Name("operator <") boolean lessThan(@Const @ByRef GraphEdge other);
    public native @Cast("bool") @Name("operator >") boolean greaterThan(@Const @ByRef GraphEdge other);

    public native int from(); public native GraphEdge from(int from);
    public native int to(); public native GraphEdge to(int to);
    public native float weight(); public native GraphEdge weight(float weight);
}




@Namespace("cv::detail") @NoOffset public static class Graph extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public Graph(Pointer p) { super(p); }

    public Graph(int num_vertices/*=0*/) { super((Pointer)null); allocate(num_vertices); }
    private native void allocate(int num_vertices/*=0*/);
    public Graph() { super((Pointer)null); allocate(); }
    private native void allocate();
    public native void create(int num_vertices);
    public native int numVertices();
    public native void addEdge(int from, int to, float weight);
}


//////////////////////////////////////////////////////////////////////////////
// Auxiliary functions

@Namespace("cv::detail") public static native @Cast("bool") boolean overlapRoi(@ByVal Point tl1, @ByVal Point tl2, @ByVal Size sz1, @ByVal Size sz2, @ByRef Rect roi);
@Namespace("cv::detail") public static native @ByVal Rect resultRoi(@Const @ByRef PointVector corners, @Const @ByRef UMatVector images);
@Namespace("cv::detail") public static native @ByVal Rect resultRoi(@Const @ByRef PointVector corners, @Const @ByRef SizeVector sizes);
@Namespace("cv::detail") public static native @ByVal Rect resultRoiIntersection(@Const @ByRef PointVector corners, @Const @ByRef SizeVector sizes);
@Namespace("cv::detail") public static native @ByVal Point resultTl(@Const @ByRef PointVector corners);

// Returns random 'count' element subset of the {0,1,...,size-1} set
@Namespace("cv::detail") public static native void selectRandomSubset(int count, int size, @StdVector IntPointer subset);
@Namespace("cv::detail") public static native void selectRandomSubset(int count, int size, @StdVector IntBuffer subset);
@Namespace("cv::detail") public static native void selectRandomSubset(int count, int size, @StdVector int[] subset);

@Namespace("cv::detail") public static native @ByRef IntPointer stitchingLogLevel();

/** \} */

 // namespace detail
 // namespace cv

// #include "util_inl.hpp"

// #endif // __OPENCV_STITCHING_UTIL_HPP__


// Parsed from <opencv2/stitching/detail/camera.hpp>

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

// #ifndef __OPENCV_STITCHING_CAMERA_HPP__
// #define __OPENCV_STITCHING_CAMERA_HPP__

// #include "opencv2/core.hpp"

/** \addtogroup stitching
 *  \{
<p>
/** \brief Describes camera parameters.
<p>
\note Translation is assumed to be zero during the whole stitching pipeline. :
 */
@Namespace("cv::detail") @NoOffset public static class CameraParams extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public CameraParams(Pointer p) { super(p); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public CameraParams(int size) { super((Pointer)null); allocateArray(size); }
    private native void allocateArray(int size);
    @Override public CameraParams position(int position) {
        return (CameraParams)super.position(position);
    }

    public CameraParams() { super((Pointer)null); allocate(); }
    private native void allocate();
    public CameraParams(@Const @ByRef CameraParams other) { super((Pointer)null); allocate(other); }
    private native void allocate(@Const @ByRef CameraParams other);
    public native @Const @ByRef @Name("operator =") CameraParams put(@Const @ByRef CameraParams other);
    public native @ByVal Mat K();

    public native double focal(); public native CameraParams focal(double focal); // Focal length
    public native double aspect(); public native CameraParams aspect(double aspect); // Aspect ratio
    public native double ppx(); public native CameraParams ppx(double ppx); // Principal point X
    public native double ppy(); public native CameraParams ppy(double ppy); // Principal point Y
    public native @ByRef Mat R(); public native CameraParams R(Mat R); // Rotation
    public native @ByRef Mat t(); public native CameraParams t(Mat t); // Translation
}

/** \} */

 // namespace detail
 // namespace cv

// #endif // #ifndef __OPENCV_STITCHING_CAMERA_HPP__


// Parsed from <opencv2/stitching/detail/motion_estimators.hpp>

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

// #ifndef __OPENCV_STITCHING_MOTION_ESTIMATORS_HPP__
// #define __OPENCV_STITCHING_MOTION_ESTIMATORS_HPP__

// #include "opencv2/core.hpp"
// #include "matchers.hpp"
// #include "util.hpp"
// #include "camera.hpp"

/** \addtogroup stitching_rotation
 *  \{
<p>
/** \brief Rotation estimator base class.
<p>
It takes features of all images, pairwise matches between all images and estimates rotations of all
cameras.
<p>
\note The coordinate system origin is implementation-dependent, but you can always normalize the
rotations in respect to the first camera, for instance. :
 */
@Namespace("cv::detail") public static class Estimator extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public Estimator(Pointer p) { super(p); }


    /** \brief Estimates camera parameters.
    <p>
    @param features Features of images
    @param pairwise_matches Pairwise matches of images
    @param cameras Estimated camera parameters
    @return True in case of success, false otherwise
     */
    public native @Cast("bool") @Name("operator ()") boolean apply(@StdVector ImageFeatures features,
                         @StdVector MatchesInfo pairwise_matches,
                         @StdVector CameraParams cameras);
}

/** \brief Homography based rotation estimator.
 */
@Namespace("cv::detail") @NoOffset public static class HomographyBasedEstimator extends Estimator {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public HomographyBasedEstimator(Pointer p) { super(p); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public HomographyBasedEstimator(int size) { super((Pointer)null); allocateArray(size); }
    private native void allocateArray(int size);
    @Override public HomographyBasedEstimator position(int position) {
        return (HomographyBasedEstimator)super.position(position);
    }

    public HomographyBasedEstimator(@Cast("bool") boolean is_focals_estimated/*=false*/) { super((Pointer)null); allocate(is_focals_estimated); }
    private native void allocate(@Cast("bool") boolean is_focals_estimated/*=false*/);
    public HomographyBasedEstimator() { super((Pointer)null); allocate(); }
    private native void allocate();
}

/** \brief Base class for all camera parameters refinement methods.
 */
@Namespace("cv::detail") @NoOffset public static class BundleAdjusterBase extends Estimator {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public BundleAdjusterBase(Pointer p) { super(p); }

    public native @Const @ByVal Mat refinementMask();
    public native void setRefinementMask(@Const @ByRef Mat mask);

    public native double confThresh();
    public native void setConfThresh(double conf_thresh);

    public native @ByVal TermCriteria termCriteria();
    public native void setTermCriteria(@Const @ByRef TermCriteria term_criteria);
}


/** \brief Implementation of the camera parameters refinement algorithm which minimizes sum of the reprojection
error squares
<p>
It can estimate focal length, aspect ratio, principal point.
You can affect only on them via the refinement mask.
 */
@Namespace("cv::detail") @NoOffset public static class BundleAdjusterReproj extends BundleAdjusterBase {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public BundleAdjusterReproj(Pointer p) { super(p); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public BundleAdjusterReproj(int size) { super((Pointer)null); allocateArray(size); }
    private native void allocateArray(int size);
    @Override public BundleAdjusterReproj position(int position) {
        return (BundleAdjusterReproj)super.position(position);
    }

    public BundleAdjusterReproj() { super((Pointer)null); allocate(); }
    private native void allocate();
}


/** \brief Implementation of the camera parameters refinement algorithm which minimizes sum of the distances
between the rays passing through the camera center and a feature. :
<p>
It can estimate focal length. It ignores the refinement mask for now.
 */
@Namespace("cv::detail") @NoOffset public static class BundleAdjusterRay extends BundleAdjusterBase {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public BundleAdjusterRay(Pointer p) { super(p); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public BundleAdjusterRay(int size) { super((Pointer)null); allocateArray(size); }
    private native void allocateArray(int size);
    @Override public BundleAdjusterRay position(int position) {
        return (BundleAdjusterRay)super.position(position);
    }

    public BundleAdjusterRay() { super((Pointer)null); allocate(); }
    private native void allocate();
}


/** enum cv::detail::WaveCorrectKind */
public static final int
    WAVE_CORRECT_HORIZ = 0,
    WAVE_CORRECT_VERT = 1;

/** \brief Tries to make panorama more horizontal (or vertical).
<p>
@param rmats Camera rotation matrices.
@param kind Correction kind, see detail::WaveCorrectKind.
 */
@Namespace("cv::detail") public static native void waveCorrect(@ByRef MatVector rmats, @Cast("cv::detail::WaveCorrectKind") int kind);


//////////////////////////////////////////////////////////////////////////////
// Auxiliary functions

// Returns matches graph representation in DOT language
@Namespace("cv::detail") public static native @Str BytePointer matchesGraphAsString(@ByRef StringVector pathes, @StdVector MatchesInfo pairwise_matches,
                                            float conf_threshold);

@Namespace("cv::detail") public static native @StdVector IntPointer leaveBiggestComponent(
        @StdVector ImageFeatures features,
        @StdVector MatchesInfo pairwise_matches,
        float conf_threshold);

@Namespace("cv::detail") public static native void findMaxSpanningTree(
        int num_images, @StdVector MatchesInfo pairwise_matches,
        @ByRef Graph span_tree, @StdVector IntPointer centers);
@Namespace("cv::detail") public static native void findMaxSpanningTree(
        int num_images, @StdVector MatchesInfo pairwise_matches,
        @ByRef Graph span_tree, @StdVector IntBuffer centers);
@Namespace("cv::detail") public static native void findMaxSpanningTree(
        int num_images, @StdVector MatchesInfo pairwise_matches,
        @ByRef Graph span_tree, @StdVector int[] centers);

/** \} stitching_rotation */

 // namespace detail
 // namespace cv

// #endif // __OPENCV_STITCHING_MOTION_ESTIMATORS_HPP__


// Parsed from <opencv2/stitching/detail/exposure_compensate.hpp>

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

// #ifndef __OPENCV_STITCHING_EXPOSURE_COMPENSATE_HPP__
// #define __OPENCV_STITCHING_EXPOSURE_COMPENSATE_HPP__

// #include "opencv2/core.hpp"

/** \addtogroup stitching_exposure
 *  \{
<p>
/** \brief Base class for all exposure compensators.
 */
@Namespace("cv::detail") public static class ExposureCompensator extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public ExposureCompensator(Pointer p) { super(p); }


    /** enum cv::detail::ExposureCompensator:: */
    public static final int NO = 0, GAIN = 1, GAIN_BLOCKS = 2;
    public static native @Ptr ExposureCompensator createDefault(int type);

    /**
    @param corners Source image top-left corners
    @param images Source images
    @param masks Image masks to update (second value in pair specifies the value which should be used
    to detect where image is)
     */
    public native void feed(@Const @ByRef PointVector corners, @Const @ByRef UMatVector images,
                  @Const @ByRef UMatVector masks);
    /** \overload */
    public native void feed(@Const @ByRef PointVector corners, @Const @ByRef UMatVector images,
                          @Const @ByRef UMatBytePairVector masks);
    /** \brief Compensate exposure in the specified image.
    <p>
    @param index Image index
    @param corner Image top-left corner
    @param image Image to process
    @param mask Image mask
     */
    public native void apply(int index, @ByVal Point corner, @ByVal Mat image, @ByVal Mat mask);
}

/** \brief Stub exposure compensator which does nothing.
 */
@Namespace("cv::detail") public static class NoExposureCompensator extends ExposureCompensator {
    static { Loader.load(); }
    /** Default native constructor. */
    public NoExposureCompensator() { super((Pointer)null); allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public NoExposureCompensator(int size) { super((Pointer)null); allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public NoExposureCompensator(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public NoExposureCompensator position(int position) {
        return (NoExposureCompensator)super.position(position);
    }

    public native void feed(@Const @ByRef PointVector arg0, @Const @ByRef UMatVector arg1,
                  @Const @ByRef UMatBytePairVector arg2);
    public native void apply(int arg0, @ByVal Point arg1, @ByVal Mat arg2, @ByVal Mat arg3);
}

/** \brief Exposure compensator which tries to remove exposure related artifacts by adjusting image
intensities, see \cite BL07 and \cite WJ10 for details.
 */
@Namespace("cv::detail") public static class GainCompensator extends ExposureCompensator {
    static { Loader.load(); }
    /** Default native constructor. */
    public GainCompensator() { super((Pointer)null); allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public GainCompensator(int size) { super((Pointer)null); allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public GainCompensator(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public GainCompensator position(int position) {
        return (GainCompensator)super.position(position);
    }

    public native void feed(@Const @ByRef PointVector corners, @Const @ByRef UMatVector images,
                  @Const @ByRef UMatBytePairVector masks);
    public native void apply(int index, @ByVal Point corner, @ByVal Mat image, @ByVal Mat mask);
    public native @StdVector DoublePointer gains();
}

/** \brief Exposure compensator which tries to remove exposure related artifacts by adjusting image block
intensities, see \cite UES01 for details.
 */
@Namespace("cv::detail") @NoOffset public static class BlocksGainCompensator extends ExposureCompensator {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public BlocksGainCompensator(Pointer p) { super(p); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public BlocksGainCompensator(int size) { super((Pointer)null); allocateArray(size); }
    private native void allocateArray(int size);
    @Override public BlocksGainCompensator position(int position) {
        return (BlocksGainCompensator)super.position(position);
    }

    public BlocksGainCompensator(int bl_width/*=32*/, int bl_height/*=32*/) { super((Pointer)null); allocate(bl_width, bl_height); }
    private native void allocate(int bl_width/*=32*/, int bl_height/*=32*/);
    public BlocksGainCompensator() { super((Pointer)null); allocate(); }
    private native void allocate();
    public native void feed(@Const @ByRef PointVector corners, @Const @ByRef UMatVector images,
                  @Const @ByRef UMatBytePairVector masks);
    public native void apply(int index, @ByVal Point corner, @ByVal Mat image, @ByVal Mat mask);
}

/** \} */

 // namespace detail
 // namespace cv

// #endif // __OPENCV_STITCHING_EXPOSURE_COMPENSATE_HPP__


// Parsed from <opencv2/stitching/detail/seam_finders.hpp>

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

// #ifndef __OPENCV_STITCHING_SEAM_FINDERS_HPP__
// #define __OPENCV_STITCHING_SEAM_FINDERS_HPP__

// #include <set>
// #include "opencv2/core.hpp"
// #include "opencv2/opencv_modules.hpp"

/** \addtogroup stitching_seam
 *  \{
<p>
/** \brief Base class for a seam estimator.
 */
@Namespace("cv::detail") public static class SeamFinder extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public SeamFinder(Pointer p) { super(p); }

    /** \brief Estimates seams.
    <p>
    @param src Source images
    @param corners Source image top-left corners
    @param masks Source image masks to update
     */
    public native void find(@Const @ByRef UMatVector src, @Const @ByRef PointVector corners,
                          @ByRef UMatVector masks);
}

/** \brief Stub seam estimator which does nothing.
 */
@Namespace("cv::detail") public static class NoSeamFinder extends SeamFinder {
    static { Loader.load(); }
    /** Default native constructor. */
    public NoSeamFinder() { super((Pointer)null); allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public NoSeamFinder(int size) { super((Pointer)null); allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public NoSeamFinder(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public NoSeamFinder position(int position) {
        return (NoSeamFinder)super.position(position);
    }

    public native void find(@Const @ByRef UMatVector arg0, @Const @ByRef PointVector arg1, @ByRef UMatVector arg2);
}

/** \brief Base class for all pairwise seam estimators.
 */
@Namespace("cv::detail") @NoOffset public static class PairwiseSeamFinder extends SeamFinder {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public PairwiseSeamFinder(Pointer p) { super(p); }

    public native void find(@Const @ByRef UMatVector src, @Const @ByRef PointVector corners,
                          @ByRef UMatVector masks);
}

/** \brief Voronoi diagram-based seam estimator.
 */
@Namespace("cv::detail") public static class VoronoiSeamFinder extends PairwiseSeamFinder {
    static { Loader.load(); }
    /** Default native constructor. */
    public VoronoiSeamFinder() { super((Pointer)null); allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public VoronoiSeamFinder(int size) { super((Pointer)null); allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public VoronoiSeamFinder(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public VoronoiSeamFinder position(int position) {
        return (VoronoiSeamFinder)super.position(position);
    }

    public native void find(@Const @ByRef UMatVector src, @Const @ByRef PointVector corners,
                          @ByRef UMatVector masks);
    public native void find(@Const @ByRef SizeVector size, @Const @ByRef PointVector corners,
                          @ByRef UMatVector masks);
}


@Namespace("cv::detail") @NoOffset public static class DpSeamFinder extends SeamFinder {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public DpSeamFinder(Pointer p) { super(p); }

    /** enum cv::detail::DpSeamFinder::CostFunction */
    public static final int COLOR = 0, COLOR_GRAD = 1;

    public DpSeamFinder(@Cast("cv::detail::DpSeamFinder::CostFunction") int costFunc/*=cv::detail::DpSeamFinder::COLOR*/) { super((Pointer)null); allocate(costFunc); }
    private native void allocate(@Cast("cv::detail::DpSeamFinder::CostFunction") int costFunc/*=cv::detail::DpSeamFinder::COLOR*/);
    public DpSeamFinder() { super((Pointer)null); allocate(); }
    private native void allocate();

    public native @Cast("cv::detail::DpSeamFinder::CostFunction") int costFunction();
    public native void setCostFunction(@Cast("cv::detail::DpSeamFinder::CostFunction") int val);

    public native void find(@Const @ByRef UMatVector src, @Const @ByRef PointVector corners,
                          @ByRef UMatVector masks);
}

/** \brief Base class for all minimum graph-cut-based seam estimators.
 */
@Namespace("cv::detail") public static class GraphCutSeamFinderBase extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public GraphCutSeamFinderBase() { super((Pointer)null); allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public GraphCutSeamFinderBase(int size) { super((Pointer)null); allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public GraphCutSeamFinderBase(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public GraphCutSeamFinderBase position(int position) {
        return (GraphCutSeamFinderBase)super.position(position);
    }

    /** enum cv::detail::GraphCutSeamFinderBase::CostType */
    public static final int COST_COLOR = 0, COST_COLOR_GRAD = 1;
}

/** \brief Minimum graph cut-based seam estimator. See details in \cite V03 .
 */
@Namespace("cv::detail") @NoOffset public static class GraphCutSeamFinder extends GraphCutSeamFinderBase {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public GraphCutSeamFinder(Pointer p) { super(p); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public GraphCutSeamFinder(int size) { super((Pointer)null); allocateArray(size); }
    private native void allocateArray(int size);
    @Override public GraphCutSeamFinder position(int position) {
        return (GraphCutSeamFinder)super.position(position);
    }
    public SeamFinder asSeamFinder() { return asSeamFinder(this); }
    @Namespace public static native @Name("static_cast<cv::detail::SeamFinder*>") SeamFinder asSeamFinder(GraphCutSeamFinder pointer);

    public GraphCutSeamFinder(int cost_type/*=COST_COLOR_GRAD*/, float terminal_cost/*=10000.f*/,
                           float bad_region_penalty/*=1000.f*/) { super((Pointer)null); allocate(cost_type, terminal_cost, bad_region_penalty); }
    private native void allocate(int cost_type/*=COST_COLOR_GRAD*/, float terminal_cost/*=10000.f*/,
                           float bad_region_penalty/*=1000.f*/);
    public GraphCutSeamFinder() { super((Pointer)null); allocate(); }
    private native void allocate();

    public native void find(@Const @ByRef UMatVector src, @Const @ByRef PointVector corners,
                  @ByRef UMatVector masks);
}


// #ifdef HAVE_OPENCV_CUDALEGACY
// #endif

/** \} */

 // namespace detail
 // namespace cv

// #endif // __OPENCV_STITCHING_SEAM_FINDERS_HPP__


// Parsed from <opencv2/stitching/detail/blenders.hpp>

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

// #ifndef __OPENCV_STITCHING_BLENDERS_HPP__
// #define __OPENCV_STITCHING_BLENDERS_HPP__

// #include "opencv2/core.hpp"

/** \addtogroup stitching_blend
 *  \{
<p>
/** \brief Base class for all blenders.
<p>
Simple blender which puts one image over another
*/
@Namespace("cv::detail") public static class Blender extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public Blender() { super((Pointer)null); allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public Blender(int size) { super((Pointer)null); allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public Blender(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public Blender position(int position) {
        return (Blender)super.position(position);
    }


    /** enum cv::detail::Blender:: */
    public static final int NO = 0, FEATHER = 1, MULTI_BAND = 2;
    public static native @Ptr Blender createDefault(int type, @Cast("bool") boolean try_gpu/*=false*/);
    public static native @Ptr Blender createDefault(int type);

    /** \brief Prepares the blender for blending.
    <p>
    @param corners Source images top-left corners
    @param sizes Source image sizes
     */
    public native void prepare(@Const @ByRef PointVector corners, @Const @ByRef SizeVector sizes);
    /** \overload */
    public native void prepare(@ByVal Rect dst_roi);
    /** \brief Processes the image.
    <p>
    @param img Source image
    @param mask Source image mask
    @param tl Source image top-left corners
     */
    public native void feed(@ByVal Mat img, @ByVal Mat mask, @ByVal Point tl);
    /** \brief Blends and returns the final pano.
    <p>
    @param dst Final pano
    @param dst_mask Final pano mask
     */
    public native void blend(@ByVal Mat dst, @ByVal Mat dst_mask);
}

/** \brief Simple blender which mixes images at its borders.
 */
@Namespace("cv::detail") @NoOffset public static class FeatherBlender extends Blender {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public FeatherBlender(Pointer p) { super(p); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public FeatherBlender(int size) { super((Pointer)null); allocateArray(size); }
    private native void allocateArray(int size);
    @Override public FeatherBlender position(int position) {
        return (FeatherBlender)super.position(position);
    }

    public FeatherBlender(float sharpness/*=0.02f*/) { super((Pointer)null); allocate(sharpness); }
    private native void allocate(float sharpness/*=0.02f*/);
    public FeatherBlender() { super((Pointer)null); allocate(); }
    private native void allocate();

    public native float sharpness();
    public native void setSharpness(float val);

    public native void prepare(@ByVal Rect dst_roi);
    public native void feed(@ByVal Mat img, @ByVal Mat mask, @ByVal Point tl);
    public native void blend(@ByVal Mat dst, @ByVal Mat dst_mask);

    /** Creates weight maps for fixed set of source images by their masks and top-left corners.
     *  Final image can be obtained by simple weighting of the source images. */
    public native @ByVal Rect createWeightMaps(@Const @ByRef UMatVector masks, @Const @ByRef PointVector corners,
                              @ByRef UMatVector weight_maps);
}



/** \brief Blender which uses multi-band blending algorithm (see \cite BA83).
 */
@Namespace("cv::detail") @NoOffset public static class MultiBandBlender extends Blender {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public MultiBandBlender(Pointer p) { super(p); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public MultiBandBlender(int size) { super((Pointer)null); allocateArray(size); }
    private native void allocateArray(int size);
    @Override public MultiBandBlender position(int position) {
        return (MultiBandBlender)super.position(position);
    }

    public MultiBandBlender(int try_gpu/*=false*/, int num_bands/*=5*/, int weight_type/*=CV_32F*/) { super((Pointer)null); allocate(try_gpu, num_bands, weight_type); }
    private native void allocate(int try_gpu/*=false*/, int num_bands/*=5*/, int weight_type/*=CV_32F*/);
    public MultiBandBlender() { super((Pointer)null); allocate(); }
    private native void allocate();

    public native int numBands();
    public native void setNumBands(int val);

    public native void prepare(@ByVal Rect dst_roi);
    public native void feed(@ByVal Mat img, @ByVal Mat mask, @ByVal Point tl);
    public native void blend(@ByVal Mat dst, @ByVal Mat dst_mask);
}


//////////////////////////////////////////////////////////////////////////////
// Auxiliary functions

@Namespace("cv::detail") public static native void normalizeUsingWeightMap(@ByVal Mat weight, @ByVal Mat src);

@Namespace("cv::detail") public static native void createWeightMap(@ByVal Mat mask, float sharpness, @ByVal Mat weight);

@Namespace("cv::detail") public static native void createLaplacePyr(@ByVal Mat img, int num_levels, @ByRef UMatVector pyr);
@Namespace("cv::detail") public static native void createLaplacePyrGpu(@ByVal Mat img, int num_levels, @ByRef UMatVector pyr);

// Restores source image
@Namespace("cv::detail") public static native void restoreImageFromLaplacePyr(@ByRef UMatVector pyr);
@Namespace("cv::detail") public static native void restoreImageFromLaplacePyrGpu(@ByRef UMatVector pyr);

/** \} */

 // namespace detail
 // namespace cv

// #endif // __OPENCV_STITCHING_BLENDERS_HPP__


// Parsed from <opencv2/stitching/detail/autocalib.hpp>

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

// #ifndef __OPENCV_STITCHING_AUTOCALIB_HPP__
// #define __OPENCV_STITCHING_AUTOCALIB_HPP__

// #include "opencv2/core.hpp"
// #include "matchers.hpp"

/** \addtogroup stitching_autocalib
 *  \{
<p>
/** \brief Tries to estimate focal lengths from the given homography under the assumption that the camera
undergoes rotations around its centre only.
<p>
@param H Homography.
@param f0 Estimated focal length along X axis.
@param f1 Estimated focal length along Y axis.
@param f0_ok True, if f0 was estimated successfully, false otherwise.
@param f1_ok True, if f1 was estimated successfully, false otherwise.
<p>
See "Construction of Panoramic Image Mosaics with Global and Local Alignment"
by Heung-Yeung Shum and Richard Szeliski.
 */
@Namespace("cv::detail") public static native void focalsFromHomography(@Const @ByRef Mat H, @ByRef DoublePointer f0, @ByRef DoublePointer f1, @Cast("bool*") @ByRef BoolPointer f0_ok, @Cast("bool*") @ByRef BoolPointer f1_ok);
@Namespace("cv::detail") public static native void focalsFromHomography(@Const @ByRef Mat H, @ByRef DoubleBuffer f0, @ByRef DoubleBuffer f1, @Cast("bool*") @ByRef boolean[] f0_ok, @Cast("bool*") @ByRef boolean[] f1_ok);
@Namespace("cv::detail") public static native void focalsFromHomography(@Const @ByRef Mat H, @ByRef double[] f0, @ByRef double[] f1, @Cast("bool*") @ByRef BoolPointer f0_ok, @Cast("bool*") @ByRef BoolPointer f1_ok);
@Namespace("cv::detail") public static native void focalsFromHomography(@Const @ByRef Mat H, @ByRef DoublePointer f0, @ByRef DoublePointer f1, @Cast("bool*") @ByRef boolean[] f0_ok, @Cast("bool*") @ByRef boolean[] f1_ok);
@Namespace("cv::detail") public static native void focalsFromHomography(@Const @ByRef Mat H, @ByRef DoubleBuffer f0, @ByRef DoubleBuffer f1, @Cast("bool*") @ByRef BoolPointer f0_ok, @Cast("bool*") @ByRef BoolPointer f1_ok);
@Namespace("cv::detail") public static native void focalsFromHomography(@Const @ByRef Mat H, @ByRef double[] f0, @ByRef double[] f1, @Cast("bool*") @ByRef boolean[] f0_ok, @Cast("bool*") @ByRef boolean[] f1_ok);

/** \brief Estimates focal lengths for each given camera.
<p>
@param features Features of images.
@param pairwise_matches Matches between all image pairs.
@param focals Estimated focal lengths for each camera.
 */
@Namespace("cv::detail") public static native void estimateFocal(@StdVector ImageFeatures features,
                              @StdVector MatchesInfo pairwise_matches,
                              @StdVector DoublePointer focals);
@Namespace("cv::detail") public static native void estimateFocal(@StdVector ImageFeatures features,
                              @StdVector MatchesInfo pairwise_matches,
                              @StdVector DoubleBuffer focals);
@Namespace("cv::detail") public static native void estimateFocal(@StdVector ImageFeatures features,
                              @StdVector MatchesInfo pairwise_matches,
                              @StdVector double[] focals);

@Namespace("cv::detail") public static native @Cast("bool") boolean calibrateRotatingCamera(@Const @ByRef MatVector Hs, @ByRef Mat K);

/** \} stitching_autocalib */

 // namespace detail
 // namespace cv

// #endif // __OPENCV_STITCHING_AUTOCALIB_HPP__


// Parsed from <opencv2/stitching/detail/timelapsers.hpp>

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


// #ifndef __OPENCV_STITCHING_TIMELAPSERS_HPP__
// #define __OPENCV_STITCHING_TIMELAPSERS_HPP__

// #include "opencv2/core.hpp"

/** \addtogroup stitching
 *  \{ */

//  Base Timelapser class, takes a sequence of images, applies appropriate shift, stores result in dst_.

@Namespace("cv::detail") public static class Timelapser extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public Timelapser() { super((Pointer)null); allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public Timelapser(int size) { super((Pointer)null); allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public Timelapser(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public Timelapser position(int position) {
        return (Timelapser)super.position(position);
    }


    /** enum cv::detail::Timelapser:: */
    public static final int AS_IS = 0, CROP = 1;

    public static native @Ptr Timelapser createDefault(int type);

    public native void initialize(@Const @ByRef PointVector corners, @Const @ByRef SizeVector sizes);
    public native void process(@ByVal Mat img, @ByVal Mat mask, @ByVal Point tl);
    public native @Const @ByRef UMat getDst();
}


@Namespace("cv::detail") public static class TimelapserCrop extends Timelapser {
    static { Loader.load(); }
    /** Default native constructor. */
    public TimelapserCrop() { super((Pointer)null); allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public TimelapserCrop(int size) { super((Pointer)null); allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public TimelapserCrop(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public TimelapserCrop position(int position) {
        return (TimelapserCrop)super.position(position);
    }

    public native void initialize(@Const @ByRef PointVector corners, @Const @ByRef SizeVector sizes);
}

/** \} */

 // namespace detail
 // namespace cv

// #endif // __OPENCV_STITCHING_TIMELAPSERS_HPP__


// Parsed from <opencv2/stitching/warpers.hpp>

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

// #ifndef __OPENCV_STITCHING_WARPER_CREATORS_HPP__
// #define __OPENCV_STITCHING_WARPER_CREATORS_HPP__

// #include "opencv2/stitching/detail/warpers.hpp"

/** \addtogroup stitching_warp
 *  \{
<p>
/** \brief Image warper factories base class.
 */
@Namespace("cv") public static class WarperCreator extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public WarperCreator(Pointer p) { super(p); }

    public native @Ptr RotationWarper create(float scale);
}

/** \brief Plane warper factory class.
  \sa detail::PlaneWarper
 */
@Namespace("cv") public static class PlaneWarper extends WarperCreator {
    static { Loader.load(); }
    /** Default native constructor. */
    public PlaneWarper() { super((Pointer)null); allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public PlaneWarper(int size) { super((Pointer)null); allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public PlaneWarper(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public PlaneWarper position(int position) {
        return (PlaneWarper)super.position(position);
    }

    public native @Ptr RotationWarper create(float scale);
}

/** \brief Cylindrical warper factory class.
\sa detail::CylindricalWarper
*/
@Namespace("cv") public static class CylindricalWarper extends WarperCreator {
    static { Loader.load(); }
    /** Default native constructor. */
    public CylindricalWarper() { super((Pointer)null); allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public CylindricalWarper(int size) { super((Pointer)null); allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public CylindricalWarper(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public CylindricalWarper position(int position) {
        return (CylindricalWarper)super.position(position);
    }

    public native @Ptr RotationWarper create(float scale);
}

/** \brief Spherical warper factory class */
@Namespace("cv") public static class SphericalWarper extends WarperCreator {
    static { Loader.load(); }
    /** Default native constructor. */
    public SphericalWarper() { super((Pointer)null); allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public SphericalWarper(int size) { super((Pointer)null); allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public SphericalWarper(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public SphericalWarper position(int position) {
        return (SphericalWarper)super.position(position);
    }

    public native @Ptr RotationWarper create(float scale);
}

@Namespace("cv") public static class FisheyeWarper extends WarperCreator {
    static { Loader.load(); }
    /** Default native constructor. */
    public FisheyeWarper() { super((Pointer)null); allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public FisheyeWarper(int size) { super((Pointer)null); allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public FisheyeWarper(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public FisheyeWarper position(int position) {
        return (FisheyeWarper)super.position(position);
    }

    public native @Ptr RotationWarper create(float scale);
}

@Namespace("cv") public static class StereographicWarper extends WarperCreator {
    static { Loader.load(); }
    /** Default native constructor. */
    public StereographicWarper() { super((Pointer)null); allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public StereographicWarper(int size) { super((Pointer)null); allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public StereographicWarper(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public StereographicWarper position(int position) {
        return (StereographicWarper)super.position(position);
    }

    public native @Ptr RotationWarper create(float scale);
}

@Namespace("cv") @NoOffset public static class CompressedRectilinearWarper extends WarperCreator {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public CompressedRectilinearWarper(Pointer p) { super(p); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public CompressedRectilinearWarper(int size) { super((Pointer)null); allocateArray(size); }
    private native void allocateArray(int size);
    @Override public CompressedRectilinearWarper position(int position) {
        return (CompressedRectilinearWarper)super.position(position);
    }

    public CompressedRectilinearWarper(float A/*=1*/, float B/*=1*/) { super((Pointer)null); allocate(A, B); }
    private native void allocate(float A/*=1*/, float B/*=1*/);
    public CompressedRectilinearWarper() { super((Pointer)null); allocate(); }
    private native void allocate();
    public native @Ptr RotationWarper create(float scale);
}

@Namespace("cv") @NoOffset public static class CompressedRectilinearPortraitWarper extends WarperCreator {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public CompressedRectilinearPortraitWarper(Pointer p) { super(p); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public CompressedRectilinearPortraitWarper(int size) { super((Pointer)null); allocateArray(size); }
    private native void allocateArray(int size);
    @Override public CompressedRectilinearPortraitWarper position(int position) {
        return (CompressedRectilinearPortraitWarper)super.position(position);
    }

    public CompressedRectilinearPortraitWarper(float A/*=1*/, float B/*=1*/) { super((Pointer)null); allocate(A, B); }
    private native void allocate(float A/*=1*/, float B/*=1*/);
    public CompressedRectilinearPortraitWarper() { super((Pointer)null); allocate(); }
    private native void allocate();
    public native @Ptr RotationWarper create(float scale);
}

@Namespace("cv") @NoOffset public static class PaniniWarper extends WarperCreator {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public PaniniWarper(Pointer p) { super(p); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public PaniniWarper(int size) { super((Pointer)null); allocateArray(size); }
    private native void allocateArray(int size);
    @Override public PaniniWarper position(int position) {
        return (PaniniWarper)super.position(position);
    }

    public PaniniWarper(float A/*=1*/, float B/*=1*/) { super((Pointer)null); allocate(A, B); }
    private native void allocate(float A/*=1*/, float B/*=1*/);
    public PaniniWarper() { super((Pointer)null); allocate(); }
    private native void allocate();
    public native @Ptr RotationWarper create(float scale);
}

@Namespace("cv") @NoOffset public static class PaniniPortraitWarper extends WarperCreator {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public PaniniPortraitWarper(Pointer p) { super(p); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public PaniniPortraitWarper(int size) { super((Pointer)null); allocateArray(size); }
    private native void allocateArray(int size);
    @Override public PaniniPortraitWarper position(int position) {
        return (PaniniPortraitWarper)super.position(position);
    }

    public PaniniPortraitWarper(float A/*=1*/, float B/*=1*/) { super((Pointer)null); allocate(A, B); }
    private native void allocate(float A/*=1*/, float B/*=1*/);
    public PaniniPortraitWarper() { super((Pointer)null); allocate(); }
    private native void allocate();
    public native @Ptr RotationWarper create(float scale);
}

@Namespace("cv") public static class MercatorWarper extends WarperCreator {
    static { Loader.load(); }
    /** Default native constructor. */
    public MercatorWarper() { super((Pointer)null); allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public MercatorWarper(int size) { super((Pointer)null); allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public MercatorWarper(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public MercatorWarper position(int position) {
        return (MercatorWarper)super.position(position);
    }

    public native @Ptr RotationWarper create(float scale);
}

@Namespace("cv") public static class TransverseMercatorWarper extends WarperCreator {
    static { Loader.load(); }
    /** Default native constructor. */
    public TransverseMercatorWarper() { super((Pointer)null); allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public TransverseMercatorWarper(int size) { super((Pointer)null); allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public TransverseMercatorWarper(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public TransverseMercatorWarper position(int position) {
        return (TransverseMercatorWarper)super.position(position);
    }

    public native @Ptr RotationWarper create(float scale);
}



// #ifdef HAVE_OPENCV_CUDAWARPING
// #endif

/** \} stitching_warp */

 // namespace cv

// #endif // __OPENCV_STITCHING_WARPER_CREATORS_HPP__


// Parsed from <opencv2/stitching.hpp>

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

// #ifndef __OPENCV_STITCHING_STITCHER_HPP__
// #define __OPENCV_STITCHING_STITCHER_HPP__

// #include "opencv2/core.hpp"
// #include "opencv2/features2d.hpp"
// #include "opencv2/stitching/warpers.hpp"
// #include "opencv2/stitching/detail/matchers.hpp"
// #include "opencv2/stitching/detail/motion_estimators.hpp"
// #include "opencv2/stitching/detail/exposure_compensate.hpp"
// #include "opencv2/stitching/detail/seam_finders.hpp"
// #include "opencv2/stitching/detail/blenders.hpp"
// #include "opencv2/stitching/detail/camera.hpp"

/**
\defgroup stitching Images stitching
<p>
This figure illustrates the stitching module pipeline implemented in the Stitcher class. Using that
class it's possible to configure/remove some steps, i.e. adjust the stitching pipeline according to
the particular needs. All building blocks from the pipeline are available in the detail namespace,
one can combine and use them separately.
<p>
The implemented stitching pipeline is very similar to the one proposed in \cite BL07 .
<p>
![image](StitchingPipeline.jpg)
<p>
\{
    \defgroup stitching_match Features Finding and Images Matching
    \defgroup stitching_rotation Rotation Estimation
    \defgroup stitching_autocalib Autocalibration
    \defgroup stitching_warp Images Warping
    \defgroup stitching_seam Seam Estimation
    \defgroup stitching_exposure Exposure Compensation
    \defgroup stitching_blend Image Blenders
\}
  */

/** \addtogroup stitching
 *  \{
<p>
/** \brief High level image stitcher.
<p>
It's possible to use this class without being aware of the entire stitching pipeline. However, to
be able to achieve higher stitching stability and quality of the final images at least being
familiar with the theory is recommended.
<p>
\note
   -   A basic example on image stitching can be found at
        opencv_source_code/samples/cpp/stitching.cpp
    -   A detailed example on image stitching can be found at
        opencv_source_code/samples/cpp/stitching_detailed.cpp
 */
@Namespace("cv") public static class Stitcher extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public Stitcher() { super((Pointer)null); allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public Stitcher(int size) { super((Pointer)null); allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public Stitcher(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public Stitcher position(int position) {
        return (Stitcher)super.position(position);
    }

    /** enum cv::Stitcher:: */
    public static final int ORIG_RESOL = -1;
    /** enum cv::Stitcher::Status */
    public static final int
        OK = 0,
        ERR_NEED_MORE_IMGS = 1,
        ERR_HOMOGRAPHY_EST_FAIL = 2,
        ERR_CAMERA_PARAMS_ADJUST_FAIL = 3;

   // Stitcher() {}
    /** \brief Creates a stitcher with the default parameters.
    <p>
    @param try_use_gpu Flag indicating whether GPU should be used whenever it's possible.
    @return Stitcher class instance.
     */
    public static native @ByVal Stitcher createDefault(@Cast("bool") boolean try_use_gpu/*=false*/);
    public static native @ByVal Stitcher createDefault();

    public native double registrationResol();
    public native void setRegistrationResol(double resol_mpx);

    public native double seamEstimationResol();
    public native void setSeamEstimationResol(double resol_mpx);

    public native double compositingResol();
    public native void setCompositingResol(double resol_mpx);

    public native double panoConfidenceThresh();
    public native void setPanoConfidenceThresh(double conf_thresh);

    public native @Cast("bool") boolean waveCorrection();
    public native void setWaveCorrection(@Cast("bool") boolean flag);

    public native @Cast("cv::detail::WaveCorrectKind") int waveCorrectKind();
    public native void setWaveCorrectKind(@Cast("cv::detail::WaveCorrectKind") int kind);

    public native @Ptr FeaturesFinder featuresFinder();
    public native void setFeaturesFinder(@Ptr FeaturesFinder features_finder);

    public native @Ptr FeaturesMatcher featuresMatcher();
    public native void setFeaturesMatcher(@Ptr FeaturesMatcher features_matcher);

    public native @Const @ByRef UMat matchingMask();
    public native void setMatchingMask(@Const @ByRef UMat mask);

    public native @Ptr BundleAdjusterBase bundleAdjuster();
    public native void setBundleAdjuster(@Ptr BundleAdjusterBase bundle_adjuster);

    public native @Ptr WarperCreator warper();
    public native void setWarper(@Ptr WarperCreator creator);

    public native @Ptr ExposureCompensator exposureCompensator();
    public native void setExposureCompensator(@Ptr ExposureCompensator exposure_comp);

    public native @Ptr SeamFinder seamFinder();
    public native void setSeamFinder(@Ptr SeamFinder seam_finder);

    public native @Ptr Blender blender();
    public native void setBlender(@Ptr Blender b);

    /** \overload */
    public native @Cast("cv::Stitcher::Status") int estimateTransform(@ByVal MatVector images);
    /** \brief These functions try to match the given images and to estimate rotations of each camera.
    <p>
    \note Use the functions only if you're aware of the stitching pipeline, otherwise use
    Stitcher::stitch.
    <p>
    @param images Input images.
    @param rois Region of interest rectangles.
    @return Status code.
     */
    public native @Cast("cv::Stitcher::Status") int estimateTransform(@ByVal MatVector images, @Const @ByRef RectVectorVector rois);

    /** \overload */
    public native @Cast("cv::Stitcher::Status") int composePanorama(@ByVal Mat pano);
    /** \brief These functions try to compose the given images (or images stored internally from the other function
    calls) into the final pano under the assumption that the image transformations were estimated
    before.
    <p>
    \note Use the functions only if you're aware of the stitching pipeline, otherwise use
    Stitcher::stitch.
    <p>
    @param images Input images.
    @param pano Final pano.
    @return Status code.
     */
    public native @Cast("cv::Stitcher::Status") int composePanorama(@ByVal MatVector images, @ByVal Mat pano);

    /** \overload */
    public native @Cast("cv::Stitcher::Status") int stitch(@ByVal MatVector images, @ByVal Mat pano);
    /** \brief These functions try to stitch the given images.
    <p>
    @param images Input images.
    @param rois Region of interest rectangles.
    @param pano Final pano.
    @return Status code.
     */
    public native @Cast("cv::Stitcher::Status") int stitch(@ByVal MatVector images, @Const @ByRef RectVectorVector rois, @ByVal Mat pano);

    public native @StdVector IntPointer component();
    public native @StdVector CameraParams cameras();
    public native double workScale();
}

@Namespace("cv") public static native @Ptr Stitcher createStitcher(@Cast("bool") boolean try_use_gpu/*=false*/);
@Namespace("cv") public static native @Ptr Stitcher createStitcher();

/** \} stitching */

 // namespace cv

// #endif // __OPENCV_STITCHING_STITCHER_HPP__


}
