// Targeted by JavaCPP version 0.8

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
import static org.bytedeco.javacpp.opencv_photo.*;
import static org.bytedeco.javacpp.opencv_ml.*;
import static org.bytedeco.javacpp.opencv_video.*;
import static org.bytedeco.javacpp.opencv_legacy.*;
import static org.bytedeco.javacpp.opencv_nonfree.*;

public class opencv_stitching extends org.bytedeco.javacpp.presets.opencv_stitching {
    static { Loader.load(); }

@Name("std::vector<std::pair<cv::Mat,unsigned char> >") public static class MatBytePairVector extends Pointer {
    static { Loader.load(); }
    public MatBytePairVector(Pointer p) { super(p); }
    public MatBytePairVector()       { allocate();  }
    public MatBytePairVector(long n) { allocate(n); }
    private native void allocate();
    private native void allocate(@Cast("size_t") long n);

    public native long size();
    public native void resize(@Cast("size_t") long n);

    @Index public native @ByRef Mat first(@Cast("size_t") long i); public native MatBytePairVector first(@Cast("size_t") long i, Mat first);
    @Index public native @ByRef byte second(@Cast("size_t") long i);  public native MatBytePairVector second(@Cast("size_t") long i, byte second);
}

// Parsed from /usr/local/include/opencv2/stitching/detail/warpers.hpp

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

// #include "opencv2/core/core.hpp"
// #include "opencv2/core/gpumat.hpp"
// #include "opencv2/imgproc/imgproc.hpp"

@Namespace("cv::detail") public static class RotationWarper extends Pointer {
    static { Loader.load(); }
    public RotationWarper() { }
    public RotationWarper(Pointer p) { super(p); }


    public native @ByVal Point2f warpPoint(@Const @ByRef Point2f pt, @Const @ByRef Mat K, @Const @ByRef Mat R);

    public native @ByVal Rect buildMaps(@ByVal Size src_size, @Const @ByRef Mat K, @Const @ByRef Mat R, @ByRef Mat xmap, @ByRef Mat ymap);

    public native @ByVal Point warp(@Const @ByRef Mat src, @Const @ByRef Mat K, @Const @ByRef Mat R, int interp_mode, int border_mode,
                           @ByRef Mat dst);

    public native void warpBackward(@Const @ByRef Mat src, @Const @ByRef Mat K, @Const @ByRef Mat R, int interp_mode, int border_mode,
                                  @ByVal Size dst_size, @ByRef Mat dst);

    public native @ByVal Rect warpRoi(@ByVal Size src_size, @Const @ByRef Mat K, @Const @ByRef Mat R);

    public native float getScale();
    public native void setScale(float arg0);
}


@Namespace("cv::detail") public static class ProjectorBase extends Pointer {
    static { Loader.load(); }
    public ProjectorBase() { allocate(); }
    public ProjectorBase(int size) { allocateArray(size); }
    public ProjectorBase(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public ProjectorBase position(int position) {
        return (ProjectorBase)super.position(position);
    }

    public native void setCameraParams(@Const @ByRef Mat K/*=Mat::eye(3, 3, CV_32F)*/,
                             @Const @ByRef Mat R/*=Mat::eye(3, 3, CV_32F)*/,
                             @Const @ByRef Mat T/*=Mat::zeros(3, 1, CV_32F)*/);
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


@Namespace("cv::detail") public static class PlaneProjector extends Pointer {
    static { Loader.load(); }
    public PlaneProjector() { allocate(); }
    public PlaneProjector(int size) { allocateArray(size); }
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


@Name("cv::detail::PlaneWarper") public static class DetailPlaneWarper extends RotationWarper {
    static { Loader.load(); }
    public DetailPlaneWarper(Pointer p) { super(p); }
    public DetailPlaneWarper(int size) { allocateArray(size); }
    private native void allocateArray(int size);
    @Override public DetailPlaneWarper position(int position) {
        return (DetailPlaneWarper)super.position(position);
    }

    public DetailPlaneWarper(float scale/*=1.f*/) { allocate(scale); }
    private native void allocate(float scale/*=1.f*/);
    public DetailPlaneWarper() { allocate(); }
    private native void allocate();

    public native void setScale(float scale);

    public native @ByVal Point2f warpPoint(@Const @ByRef Point2f pt, @Const @ByRef Mat K, @Const @ByRef Mat R, @Const @ByRef Mat T);

    public native @ByVal Rect buildMaps(@ByVal Size src_size, @Const @ByRef Mat K, @Const @ByRef Mat R, @Const @ByRef Mat T, @ByRef Mat xmap, @ByRef Mat ymap);

    public native @ByVal Point warp(@Const @ByRef Mat src, @Const @ByRef Mat K, @Const @ByRef Mat R, @Const @ByRef Mat T, int interp_mode, int border_mode,
                   @ByRef Mat dst);

    public native @ByVal Rect warpRoi(@ByVal Size src_size, @Const @ByRef Mat K, @Const @ByRef Mat R, @Const @ByRef Mat T);
}


@Namespace("cv::detail") public static class SphericalProjector extends Pointer {
    static { Loader.load(); }
    public SphericalProjector() { allocate(); }
    public SphericalProjector(int size) { allocateArray(size); }
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


// Projects image onto unit sphere with origin at (0, 0, 0).
// Poles are located at (0, -1, 0) and (0, 1, 0) points.
@Name("cv::detail::SphericalWarper") public static class DetailSphericalWarper extends RotationWarper {
    static { Loader.load(); }
    public DetailSphericalWarper() { }
    public DetailSphericalWarper(Pointer p) { super(p); }

    public DetailSphericalWarper(float scale) { allocate(scale); }
    private native void allocate(float scale);
}


@Namespace("cv::detail") public static class CylindricalProjector extends Pointer {
    static { Loader.load(); }
    public CylindricalProjector() { allocate(); }
    public CylindricalProjector(int size) { allocateArray(size); }
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


// Projects image onto x * x + z * z = 1 cylinder
@Name("cv::detail::CylindricalWarper") public static class DetailCylindricalWarper extends RotationWarper {
    static { Loader.load(); }
    public DetailCylindricalWarper() { }
    public DetailCylindricalWarper(Pointer p) { super(p); }

    public DetailCylindricalWarper(float scale) { allocate(scale); }
    private native void allocate(float scale);
}


@Namespace("cv::detail") public static class FisheyeProjector extends Pointer {
    static { Loader.load(); }
    public FisheyeProjector() { allocate(); }
    public FisheyeProjector(int size) { allocateArray(size); }
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
    public DetailFisheyeWarper() { }
    public DetailFisheyeWarper(Pointer p) { super(p); }

    public DetailFisheyeWarper(float scale) { allocate(scale); }
    private native void allocate(float scale);
}


@Namespace("cv::detail") public static class StereographicProjector extends Pointer {
    static { Loader.load(); }
    public StereographicProjector() { allocate(); }
    public StereographicProjector(int size) { allocateArray(size); }
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
    public DetailStereographicWarper() { }
    public DetailStereographicWarper(Pointer p) { super(p); }

    public DetailStereographicWarper(float scale) { allocate(scale); }
    private native void allocate(float scale);
}


@Namespace("cv::detail") @NoOffset public static class CompressedRectilinearProjector extends Pointer {
    static { Loader.load(); }
    public CompressedRectilinearProjector() { allocate(); }
    public CompressedRectilinearProjector(int size) { allocateArray(size); }
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
    public DetailCompressedRectilinearWarper() { }
    public DetailCompressedRectilinearWarper(Pointer p) { super(p); }

    public DetailCompressedRectilinearWarper(float scale, float A/*=1*/, float B/*=1*/) { allocate(scale, A, B); }
    private native void allocate(float scale, float A/*=1*/, float B/*=1*/);
    public DetailCompressedRectilinearWarper(float scale) { allocate(scale); }
    private native void allocate(float scale);
}


@Namespace("cv::detail") @NoOffset public static class CompressedRectilinearPortraitProjector extends Pointer {
    static { Loader.load(); }
    public CompressedRectilinearPortraitProjector() { allocate(); }
    public CompressedRectilinearPortraitProjector(int size) { allocateArray(size); }
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
    public DetailCompressedRectilinearPortraitWarper() { }
    public DetailCompressedRectilinearPortraitWarper(Pointer p) { super(p); }

   public DetailCompressedRectilinearPortraitWarper(float scale, float A/*=1*/, float B/*=1*/) { allocate(scale, A, B); }
   private native void allocate(float scale, float A/*=1*/, float B/*=1*/);
   public DetailCompressedRectilinearPortraitWarper(float scale) { allocate(scale); }
   private native void allocate(float scale);
}


@Namespace("cv::detail") @NoOffset public static class PaniniProjector extends Pointer {
    static { Loader.load(); }
    public PaniniProjector() { allocate(); }
    public PaniniProjector(int size) { allocateArray(size); }
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
    public DetailPaniniWarper() { }
    public DetailPaniniWarper(Pointer p) { super(p); }

   public DetailPaniniWarper(float scale, float A/*=1*/, float B/*=1*/) { allocate(scale, A, B); }
   private native void allocate(float scale, float A/*=1*/, float B/*=1*/);
   public DetailPaniniWarper(float scale) { allocate(scale); }
   private native void allocate(float scale);
}


@Namespace("cv::detail") @NoOffset public static class PaniniPortraitProjector extends Pointer {
    static { Loader.load(); }
    public PaniniPortraitProjector() { allocate(); }
    public PaniniPortraitProjector(int size) { allocateArray(size); }
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
    public DetailPaniniPortraitWarper() { }
    public DetailPaniniPortraitWarper(Pointer p) { super(p); }

   public DetailPaniniPortraitWarper(float scale, float A/*=1*/, float B/*=1*/) { allocate(scale, A, B); }
   private native void allocate(float scale, float A/*=1*/, float B/*=1*/);
   public DetailPaniniPortraitWarper(float scale) { allocate(scale); }
   private native void allocate(float scale);

}


@Namespace("cv::detail") public static class MercatorProjector extends Pointer {
    static { Loader.load(); }
    public MercatorProjector() { allocate(); }
    public MercatorProjector(int size) { allocateArray(size); }
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
    public DetailMercatorWarper() { }
    public DetailMercatorWarper(Pointer p) { super(p); }

    public DetailMercatorWarper(float scale) { allocate(scale); }
    private native void allocate(float scale);
}


@Namespace("cv::detail") public static class TransverseMercatorProjector extends Pointer {
    static { Loader.load(); }
    public TransverseMercatorProjector() { allocate(); }
    public TransverseMercatorProjector(int size) { allocateArray(size); }
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
    public DetailTransverseMercatorWarper() { }
    public DetailTransverseMercatorWarper(Pointer p) { super(p); }

    public DetailTransverseMercatorWarper(float scale) { allocate(scale); }
    private native void allocate(float scale);
}


@Platform(not="android") @Name("cv::detail::PlaneWarperGpu") @NoOffset public static class DetailPlaneWarperGpu extends RotationWarper {
    static { Loader.load(); }
    public DetailPlaneWarperGpu(Pointer p) { super(p); }
    public DetailPlaneWarperGpu(int size) { allocateArray(size); }
    private native void allocateArray(int size);
    @Override public DetailPlaneWarperGpu position(int position) {
        return (DetailPlaneWarperGpu)super.position(position);
    }

    public DetailPlaneWarperGpu(float scale/*=1.f*/) { allocate(scale); }
    private native void allocate(float scale/*=1.f*/);
    public DetailPlaneWarperGpu() { allocate(); }
    private native void allocate();

    public native @ByVal Rect buildMaps(@ByVal Size src_size, @Const @ByRef Mat K, @Const @ByRef Mat R, @ByRef Mat xmap, @ByRef Mat ymap);

    public native @ByVal Rect buildMaps(@ByVal Size src_size, @Const @ByRef Mat K, @Const @ByRef Mat R, @Const @ByRef Mat T, @ByRef Mat xmap, @ByRef Mat ymap);

    public native @ByVal Point warp(@Const @ByRef Mat src, @Const @ByRef Mat K, @Const @ByRef Mat R, int interp_mode, int border_mode,
                   @ByRef Mat dst);

    public native @ByVal Point warp(@Const @ByRef Mat src, @Const @ByRef Mat K, @Const @ByRef Mat R, @Const @ByRef Mat T, int interp_mode, int border_mode,
                   @ByRef Mat dst);

    public native @ByVal Rect buildMaps(@ByVal Size src_size, @Const @ByRef Mat K, @Const @ByRef Mat R, @ByRef GpuMat xmap, @ByRef GpuMat ymap);

    public native @ByVal Rect buildMaps(@ByVal Size src_size, @Const @ByRef Mat K, @Const @ByRef Mat R, @Const @ByRef Mat T, @ByRef GpuMat xmap, @ByRef GpuMat ymap);

    public native @ByVal Point warp(@Const @ByRef GpuMat src, @Const @ByRef Mat K, @Const @ByRef Mat R, int interp_mode, int border_mode,
                   @ByRef GpuMat dst);

    public native @ByVal Point warp(@Const @ByRef GpuMat src, @Const @ByRef Mat K, @Const @ByRef Mat R, @Const @ByRef Mat T, int interp_mode, int border_mode,
                   @ByRef GpuMat dst);
}


@Platform(not="android") @Name("cv::detail::SphericalWarperGpu") @NoOffset public static class DetailSphericalWarperGpu extends RotationWarper {
    static { Loader.load(); }
    public DetailSphericalWarperGpu() { }
    public DetailSphericalWarperGpu(Pointer p) { super(p); }

    public DetailSphericalWarperGpu(float scale) { allocate(scale); }
    private native void allocate(float scale);

    public native @ByVal Rect buildMaps(@ByVal Size src_size, @Const @ByRef Mat K, @Const @ByRef Mat R, @ByRef Mat xmap, @ByRef Mat ymap);

    public native @ByVal Point warp(@Const @ByRef Mat src, @Const @ByRef Mat K, @Const @ByRef Mat R, int interp_mode, int border_mode,
                   @ByRef Mat dst);

    public native @ByVal Rect buildMaps(@ByVal Size src_size, @Const @ByRef Mat K, @Const @ByRef Mat R, @ByRef GpuMat xmap, @ByRef GpuMat ymap);

    public native @ByVal Point warp(@Const @ByRef GpuMat src, @Const @ByRef Mat K, @Const @ByRef Mat R, int interp_mode, int border_mode,
                   @ByRef GpuMat dst);
}


@Platform(not="android") @Name("cv::detail::CylindricalWarperGpu") @NoOffset public static class DetailCylindricalWarperGpu extends RotationWarper {
    static { Loader.load(); }
    public DetailCylindricalWarperGpu() { }
    public DetailCylindricalWarperGpu(Pointer p) { super(p); }

    public DetailCylindricalWarperGpu(float scale) { allocate(scale); }
    private native void allocate(float scale);

    public native @ByVal Rect buildMaps(@ByVal Size src_size, @Const @ByRef Mat K, @Const @ByRef Mat R, @ByRef Mat xmap, @ByRef Mat ymap);

    public native @ByVal Point warp(@Const @ByRef Mat src, @Const @ByRef Mat K, @Const @ByRef Mat R, int interp_mode, int border_mode,
                   @ByRef Mat dst);

    public native @ByVal Rect buildMaps(@ByVal Size src_size, @Const @ByRef Mat K, @Const @ByRef Mat R, @ByRef GpuMat xmap, @ByRef GpuMat ymap);

    public native @ByVal Point warp(@Const @ByRef GpuMat src, @Const @ByRef Mat K, @Const @ByRef Mat R, int interp_mode, int border_mode,
                   @ByRef GpuMat dst);
}


@Namespace("cv::detail") public static class SphericalPortraitProjector extends Pointer {
    static { Loader.load(); }
    public SphericalPortraitProjector() { allocate(); }
    public SphericalPortraitProjector(int size) { allocateArray(size); }
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
    public SphericalPortraitWarper() { }
    public SphericalPortraitWarper(Pointer p) { super(p); }

    public SphericalPortraitWarper(float scale) { allocate(scale); }
    private native void allocate(float scale);
}

@Namespace("cv::detail") public static class CylindricalPortraitProjector extends Pointer {
    static { Loader.load(); }
    public CylindricalPortraitProjector() { allocate(); }
    public CylindricalPortraitProjector(int size) { allocateArray(size); }
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
    public CylindricalPortraitWarper() { }
    public CylindricalPortraitWarper(Pointer p) { super(p); }

    public CylindricalPortraitWarper(float scale) { allocate(scale); }
    private native void allocate(float scale);
}

@Namespace("cv::detail") public static class PlanePortraitProjector extends Pointer {
    static { Loader.load(); }
    public PlanePortraitProjector() { allocate(); }
    public PlanePortraitProjector(int size) { allocateArray(size); }
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
    public PlanePortraitWarper() { }
    public PlanePortraitWarper(Pointer p) { super(p); }

    public PlanePortraitWarper(float scale) { allocate(scale); }
    private native void allocate(float scale);
}


 // namespace cv

// #include "warpers_inl.hpp"

// #endif // __OPENCV_STITCHING_WARPERS_HPP__


// Parsed from /usr/local/include/opencv2/stitching/detail/matchers.hpp

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

// #include "opencv2/core/core.hpp"
// #include "opencv2/core/gpumat.hpp"
// #include "opencv2/features2d/features2d.hpp"

// #include "opencv2/opencv_modules.hpp"

// #if defined(HAVE_OPENCV_NONFREE)
//     #include "opencv2/nonfree/gpu.hpp"
// #endif

@Namespace("cv::detail") public static class ImageFeatures extends Pointer {
    static { Loader.load(); }
    public ImageFeatures() { allocate(); }
    public ImageFeatures(int size) { allocateArray(size); }
    public ImageFeatures(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public ImageFeatures position(int position) {
        return (ImageFeatures)super.position(position);
    }

    public native int img_idx(); public native ImageFeatures img_idx(int img_idx);
    public native @ByRef Size img_size(); public native ImageFeatures img_size(Size img_size);
    public native @StdVector KeyPoint keypoints(); public native ImageFeatures keypoints(KeyPoint keypoints);
    public native @ByRef Mat descriptors(); public native ImageFeatures descriptors(Mat descriptors);
}


@Namespace("cv::detail") public static class FeaturesFinder extends Pointer {
    static { Loader.load(); }
    public FeaturesFinder() { }
    public FeaturesFinder(Pointer p) { super(p); }

    public native @Name("operator()") void apply(@Const @ByRef Mat image, @ByRef ImageFeatures features);
    public native @Name("operator()") void apply(@Const @ByRef Mat image, @ByRef ImageFeatures features, @StdVector Rect rois);
    public native void collectGarbage();
}


@Namespace("cv::detail") @NoOffset public static class SurfFeaturesFinder extends FeaturesFinder {
    static { Loader.load(); }
    public SurfFeaturesFinder(Pointer p) { super(p); }
    public SurfFeaturesFinder(int size) { allocateArray(size); }
    private native void allocateArray(int size);
    @Override public SurfFeaturesFinder position(int position) {
        return (SurfFeaturesFinder)super.position(position);
    }

    public SurfFeaturesFinder(double hess_thresh/*=300.*/, int num_octaves/*=3*/, int num_layers/*=4*/,
                           int num_octaves_descr/*=3*/, int num_layers_descr/*=4*/) { allocate(hess_thresh, num_octaves, num_layers, num_octaves_descr, num_layers_descr); }
    private native void allocate(double hess_thresh/*=300.*/, int num_octaves/*=3*/, int num_layers/*=4*/,
                           int num_octaves_descr/*=3*/, int num_layers_descr/*=4*/);
    public SurfFeaturesFinder() { allocate(); }
    private native void allocate();
}

@Namespace("cv::detail") @NoOffset public static class OrbFeaturesFinder extends FeaturesFinder {
    static { Loader.load(); }
    public OrbFeaturesFinder(Pointer p) { super(p); }
    public OrbFeaturesFinder(int size) { allocateArray(size); }
    private native void allocateArray(int size);
    @Override public OrbFeaturesFinder position(int position) {
        return (OrbFeaturesFinder)super.position(position);
    }

    public OrbFeaturesFinder(@ByVal Size _grid_size/*=Size(3,1)*/, int nfeatures/*=1500*/, float scaleFactor/*=1.3f*/, int nlevels/*=5*/) { allocate(_grid_size, nfeatures, scaleFactor, nlevels); }
    private native void allocate(@ByVal Size _grid_size/*=Size(3,1)*/, int nfeatures/*=1500*/, float scaleFactor/*=1.3f*/, int nlevels/*=5*/);
    public OrbFeaturesFinder() { allocate(); }
    private native void allocate();
}


// #if defined(HAVE_OPENCV_NONFREE)
@Platform(not="android") @Namespace("cv::detail") @NoOffset public static class SurfFeaturesFinderGpu extends FeaturesFinder {
    static { Loader.load(); }
    public SurfFeaturesFinderGpu(Pointer p) { super(p); }
    public SurfFeaturesFinderGpu(int size) { allocateArray(size); }
    private native void allocateArray(int size);
    @Override public SurfFeaturesFinderGpu position(int position) {
        return (SurfFeaturesFinderGpu)super.position(position);
    }

    public SurfFeaturesFinderGpu(double hess_thresh/*=300.*/, int num_octaves/*=3*/, int num_layers/*=4*/,
                              int num_octaves_descr/*=4*/, int num_layers_descr/*=2*/) { allocate(hess_thresh, num_octaves, num_layers, num_octaves_descr, num_layers_descr); }
    private native void allocate(double hess_thresh/*=300.*/, int num_octaves/*=3*/, int num_layers/*=4*/,
                              int num_octaves_descr/*=4*/, int num_layers_descr/*=2*/);
    public SurfFeaturesFinderGpu() { allocate(); }
    private native void allocate();

    public native void collectGarbage();
}
// #endif


@Namespace("cv::detail") @NoOffset public static class MatchesInfo extends Pointer {
    static { Loader.load(); }
    public MatchesInfo(Pointer p) { super(p); }
    public MatchesInfo(int size) { allocateArray(size); }
    private native void allocateArray(int size);
    @Override public MatchesInfo position(int position) {
        return (MatchesInfo)super.position(position);
    }

    public MatchesInfo() { allocate(); }
    private native void allocate();
    public MatchesInfo(@Const @ByRef MatchesInfo other) { allocate(other); }
    private native void allocate(@Const @ByRef MatchesInfo other);
    public native @Const @ByRef @Name("operator=") MatchesInfo put(@Const @ByRef MatchesInfo other);

    public native int src_img_idx(); public native MatchesInfo src_img_idx(int src_img_idx);
    public native int dst_img_idx(); public native MatchesInfo dst_img_idx(int dst_img_idx);       // Images indices (optional)
    public native @StdVector DMatch matches(); public native MatchesInfo matches(DMatch matches);
    public native @Cast("uchar*") @StdVector BytePointer inliers_mask(); public native MatchesInfo inliers_mask(BytePointer inliers_mask);    // Geometrically consistent matches mask
    public native int num_inliers(); public native MatchesInfo num_inliers(int num_inliers);                    // Number of geometrically consistent matches
    public native @ByRef Mat H(); public native MatchesInfo H(Mat H);                              // Estimated homography
    public native double confidence(); public native MatchesInfo confidence(double confidence);                  // Confidence two images are from the same panorama
}


@Namespace("cv::detail") @NoOffset public static class FeaturesMatcher extends Pointer {
    static { Loader.load(); }
    public FeaturesMatcher() { }
    public FeaturesMatcher(Pointer p) { super(p); }


    public native @Name("operator()") void apply(@Const @ByRef ImageFeatures features1, @Const @ByRef ImageFeatures features2,
                         @ByRef MatchesInfo matches_info);

    public native @Name("operator()") void apply(@StdVector ImageFeatures features, @StdVector MatchesInfo pairwise_matches,
                         @Const @ByRef Mat mask/*=cv::Mat()*/);
    public native @Name("operator()") void apply(@StdVector ImageFeatures features, @StdVector MatchesInfo pairwise_matches);

    public native @Cast("bool") boolean isThreadSafe();

    public native void collectGarbage();
}


@Namespace("cv::detail") @NoOffset public static class BestOf2NearestMatcher extends FeaturesMatcher {
    static { Loader.load(); }
    public BestOf2NearestMatcher(Pointer p) { super(p); }
    public BestOf2NearestMatcher(int size) { allocateArray(size); }
    private native void allocateArray(int size);
    @Override public BestOf2NearestMatcher position(int position) {
        return (BestOf2NearestMatcher)super.position(position);
    }

    public BestOf2NearestMatcher(@Cast("bool") boolean try_use_gpu/*=false*/, float match_conf/*=0.3f*/, int num_matches_thresh1/*=6*/,
                              int num_matches_thresh2/*=6*/) { allocate(try_use_gpu, match_conf, num_matches_thresh1, num_matches_thresh2); }
    private native void allocate(@Cast("bool") boolean try_use_gpu/*=false*/, float match_conf/*=0.3f*/, int num_matches_thresh1/*=6*/,
                              int num_matches_thresh2/*=6*/);
    public BestOf2NearestMatcher() { allocate(); }
    private native void allocate();

    public native void collectGarbage();
}


 // namespace cv

// #endif // __OPENCV_STITCHING_MATCHERS_HPP__


// Parsed from /usr/local/include/opencv2/stitching/detail/util.hpp

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
// #include "opencv2/core/core.hpp"

public static final int ENABLE_LOG = 0;

// TODO remove LOG macros, add logging class
// #if ENABLE_LOG
// #ifdef ANDROID
//   #include <iostream>
//   #include <sstream>
//   #include <android/log.h>
//   #define LOG_STITCHING_MSG(msg)
//     do {
//         std::stringstream _os;
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

@Namespace("cv::detail") @NoOffset public static class DisjointSets extends Pointer {
    static { Loader.load(); }
    public DisjointSets(Pointer p) { super(p); }

    public DisjointSets(int elem_count/*=0*/) { allocate(elem_count); }
    private native void allocate(int elem_count/*=0*/);
    public DisjointSets() { allocate(); }
    private native void allocate();

    public native void createOneElemSets(int elem_count);
    public native int findSetByElem(int elem);
    public native int mergeSets(int set1, int set2);

    public native @StdVector IntPointer parent(); public native DisjointSets parent(IntPointer parent);
    public native @StdVector IntPointer size(); public native DisjointSets size(IntPointer size);
}


@Namespace("cv::detail") @NoOffset public static class GraphEdge extends Pointer {
    static { Loader.load(); }
    public GraphEdge() { }
    public GraphEdge(Pointer p) { super(p); }

    public GraphEdge(int from, int to, float weight) { allocate(from, to, weight); }
    private native void allocate(int from, int to, float weight);
    public native @Cast("bool") @Name("operator<") boolean lessThan(@Const @ByRef GraphEdge other);
    public native @Cast("bool") @Name("operator>") boolean greaterThan(@Const @ByRef GraphEdge other);

    public native int from(); public native GraphEdge from(int from);
    public native int to(); public native GraphEdge to(int to);
    public native float weight(); public native GraphEdge weight(float weight);
}

   


@Namespace("cv::detail") @NoOffset public static class Graph extends Pointer {
    static { Loader.load(); }
    public Graph(Pointer p) { super(p); }

    public Graph(int num_vertices/*=0*/) { allocate(num_vertices); }
    private native void allocate(int num_vertices/*=0*/);
    public Graph() { allocate(); }
    private native void allocate();
    public native void create(int num_vertices);
    public native int numVertices();
    public native void addEdge(int from, int to, float weight);
}


//////////////////////////////////////////////////////////////////////////////
// Auxiliary functions

@Namespace("cv::detail") public static native @Cast("bool") boolean overlapRoi(@ByVal Point tl1, @ByVal Point tl2, @ByVal Size sz1, @ByVal Size sz2, @ByRef Rect roi);
@Namespace("cv::detail") public static native @ByVal Rect resultRoi(@StdVector Point corners, @Const @ByRef MatVector images);
@Namespace("cv::detail") public static native @ByVal Rect resultRoi(@StdVector Point corners, @StdVector Size sizes);
@Namespace("cv::detail") public static native @ByVal Point resultTl(@StdVector Point corners);

// Returns random 'count' element subset of the {0,1,...,size-1} set
@Namespace("cv::detail") public static native void selectRandomSubset(int count, int size, @StdVector IntPointer subset);
@Namespace("cv::detail") public static native void selectRandomSubset(int count, int size, @StdVector IntBuffer subset);
@Namespace("cv::detail") public static native void selectRandomSubset(int count, int size, @StdVector int[] subset);

@Namespace("cv::detail") public static native @ByRef IntPointer stitchingLogLevel();


 // namespace cv

// #include "util_inl.hpp"

// #endif // __OPENCV_STITCHING_UTIL_HPP__


// Parsed from /usr/local/include/opencv2/stitching/detail/camera.hpp

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

// #include "opencv2/core/core.hpp"

@Namespace("cv::detail") @NoOffset public static class CameraParams extends Pointer {
    static { Loader.load(); }
    public CameraParams(Pointer p) { super(p); }
    public CameraParams(int size) { allocateArray(size); }
    private native void allocateArray(int size);
    @Override public CameraParams position(int position) {
        return (CameraParams)super.position(position);
    }

    public CameraParams() { allocate(); }
    private native void allocate();
    public CameraParams(@Const @ByRef CameraParams other) { allocate(other); }
    private native void allocate(@Const @ByRef CameraParams other);
    public native @Const @ByRef @Name("operator=") CameraParams put(@Const @ByRef CameraParams other);
    public native @ByVal Mat K();

    public native double focal(); public native CameraParams focal(double focal); // Focal length
    public native double aspect(); public native CameraParams aspect(double aspect); // Aspect ratio
    public native double ppx(); public native CameraParams ppx(double ppx); // Principal point X
    public native double ppy(); public native CameraParams ppy(double ppy); // Principal point Y
    public native @ByRef Mat R(); public native CameraParams R(Mat R); // Rotation
    public native @ByRef Mat t(); public native CameraParams t(Mat t); // Translation
}


 // namespace cv

// #endif // #ifndef __OPENCV_STITCHING_CAMERA_HPP__


// Parsed from /usr/local/include/opencv2/stitching/detail/motion_estimators.hpp

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

// #include "opencv2/core/core.hpp"
// #include "matchers.hpp"
// #include "util.hpp"
// #include "camera.hpp"

@Namespace("cv::detail") public static class Estimator extends Pointer {
    static { Loader.load(); }
    public Estimator() { }
    public Estimator(Pointer p) { super(p); }


    public native @Name("operator()") void apply(@StdVector ImageFeatures features, @StdVector MatchesInfo pairwise_matches,
                         @StdVector CameraParams cameras);
}


@Namespace("cv::detail") @NoOffset public static class HomographyBasedEstimator extends Estimator {
    static { Loader.load(); }
    public HomographyBasedEstimator(Pointer p) { super(p); }
    public HomographyBasedEstimator(int size) { allocateArray(size); }
    private native void allocateArray(int size);
    @Override public HomographyBasedEstimator position(int position) {
        return (HomographyBasedEstimator)super.position(position);
    }

    public HomographyBasedEstimator(@Cast("bool") boolean is_focals_estimated/*=false*/) { allocate(is_focals_estimated); }
    private native void allocate(@Cast("bool") boolean is_focals_estimated/*=false*/);
    public HomographyBasedEstimator() { allocate(); }
    private native void allocate();
}


@Namespace("cv::detail") @NoOffset public static class BundleAdjusterBase extends Estimator {
    static { Loader.load(); }
    public BundleAdjusterBase() { }
    public BundleAdjusterBase(Pointer p) { super(p); }

    public native @Const @ByVal Mat refinementMask();
    public native void setRefinementMask(@Const @ByRef Mat mask);

    public native double confThresh();
    public native void setConfThresh(double conf_thresh);

    public native @ByVal CvTermCriteria termCriteria();
    public native void setTermCriteria(@Const @ByRef CvTermCriteria term_criteria);
}


// Minimizes reprojection error.
// It can estimate focal length, aspect ratio, principal point.
// You can affect only on them via the refinement mask.
@Namespace("cv::detail") @NoOffset public static class BundleAdjusterReproj extends BundleAdjusterBase {
    static { Loader.load(); }
    public BundleAdjusterReproj(Pointer p) { super(p); }
    public BundleAdjusterReproj(int size) { allocateArray(size); }
    private native void allocateArray(int size);
    @Override public BundleAdjusterReproj position(int position) {
        return (BundleAdjusterReproj)super.position(position);
    }

    public BundleAdjusterReproj() { allocate(); }
    private native void allocate();
}


// Minimizes sun of ray-to-ray distances.
// It can estimate focal length. It ignores the refinement mask for now.
@Namespace("cv::detail") @NoOffset public static class BundleAdjusterRay extends BundleAdjusterBase {
    static { Loader.load(); }
    public BundleAdjusterRay(Pointer p) { super(p); }
    public BundleAdjusterRay(int size) { allocateArray(size); }
    private native void allocateArray(int size);
    @Override public BundleAdjusterRay position(int position) {
        return (BundleAdjusterRay)super.position(position);
    }

    public BundleAdjusterRay() { allocate(); }
    private native void allocate();
}


/** enum cv::detail::WaveCorrectKind */
public static final int
    WAVE_CORRECT_HORIZ = 0,
    WAVE_CORRECT_VERT = 1;

@Namespace("cv::detail") public static native void waveCorrect(@ByRef MatVector rmats, @Cast("cv::detail::WaveCorrectKind") int kind);


//////////////////////////////////////////////////////////////////////////////
// Auxiliary functions

// Returns matches graph representation in DOT language
@Namespace("cv::detail") public static native @StdString BytePointer matchesGraphAsString(@ByRef StringVector pathes, @StdVector MatchesInfo pairwise_matches,
                                            float conf_threshold);

@Namespace("cv::detail") public static native @StdVector IntPointer leaveBiggestComponent(@StdVector ImageFeatures features, @StdVector MatchesInfo pairwise_matches,
                                                  float conf_threshold);

@Namespace("cv::detail") public static native void findMaxSpanningTree(int num_images, @StdVector MatchesInfo pairwise_matches,
                                    @ByRef Graph span_tree, @StdVector IntPointer centers);
@Namespace("cv::detail") public static native void findMaxSpanningTree(int num_images, @StdVector MatchesInfo pairwise_matches,
                                    @ByRef Graph span_tree, @StdVector IntBuffer centers);
@Namespace("cv::detail") public static native void findMaxSpanningTree(int num_images, @StdVector MatchesInfo pairwise_matches,
                                    @ByRef Graph span_tree, @StdVector int[] centers);


 // namespace cv

// #endif // __OPENCV_STITCHING_MOTION_ESTIMATORS_HPP__


// Parsed from /usr/local/include/opencv2/stitching/detail/exposure_compensate.hpp

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

// #include "opencv2/core/core.hpp"

@Namespace("cv::detail") public static class ExposureCompensator extends Pointer {
    static { Loader.load(); }
    public ExposureCompensator() { }
    public ExposureCompensator(Pointer p) { super(p); }


    /** enum cv::detail::ExposureCompensator:: */
    public static final int NO = 0, GAIN = 1, GAIN_BLOCKS = 2;
    public native @Ptr ExposureCompensator createDefault(int type);

    public native void feed(@StdVector Point corners, @Const @ByRef MatVector images,
                  @Const @ByRef MatVector masks);
    public native void feed(@StdVector Point corners, @Const @ByRef MatVector images,
                          @Const @ByRef MatBytePairVector masks);
    public native void apply(int index, @ByVal Point corner, @ByRef Mat image, @Const @ByRef Mat mask);
}


@Namespace("cv::detail") public static class NoExposureCompensator extends ExposureCompensator {
    static { Loader.load(); }
    public NoExposureCompensator() { allocate(); }
    public NoExposureCompensator(int size) { allocateArray(size); }
    public NoExposureCompensator(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public NoExposureCompensator position(int position) {
        return (NoExposureCompensator)super.position(position);
    }

    public native void feed(@StdVector Point arg0, @Const @ByRef MatVector arg1,
                  @Const @ByRef MatBytePairVector arg2);
    public native void apply(int arg0, @ByVal Point arg1, @ByRef Mat arg2, @Const @ByRef Mat arg3);
}


@Namespace("cv::detail") public static class GainCompensator extends ExposureCompensator {
    static { Loader.load(); }
    public GainCompensator() { allocate(); }
    public GainCompensator(int size) { allocateArray(size); }
    public GainCompensator(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public GainCompensator position(int position) {
        return (GainCompensator)super.position(position);
    }

    public native void feed(@StdVector Point corners, @Const @ByRef MatVector images,
                  @Const @ByRef MatBytePairVector masks);
    public native void apply(int index, @ByVal Point corner, @ByRef Mat image, @Const @ByRef Mat mask);
    public native @StdVector DoublePointer gains();
}


@Namespace("cv::detail") @NoOffset public static class BlocksGainCompensator extends ExposureCompensator {
    static { Loader.load(); }
    public BlocksGainCompensator(Pointer p) { super(p); }
    public BlocksGainCompensator(int size) { allocateArray(size); }
    private native void allocateArray(int size);
    @Override public BlocksGainCompensator position(int position) {
        return (BlocksGainCompensator)super.position(position);
    }

    public BlocksGainCompensator(int bl_width/*=32*/, int bl_height/*=32*/) { allocate(bl_width, bl_height); }
    private native void allocate(int bl_width/*=32*/, int bl_height/*=32*/);
    public BlocksGainCompensator() { allocate(); }
    private native void allocate();
    public native void feed(@StdVector Point corners, @Const @ByRef MatVector images,
                  @Const @ByRef MatBytePairVector masks);
    public native void apply(int index, @ByVal Point corner, @ByRef Mat image, @Const @ByRef Mat mask);
}


 // namespace cv

// #endif // __OPENCV_STITCHING_EXPOSURE_COMPENSATE_HPP__


// Parsed from /usr/local/include/opencv2/stitching/detail/seam_finders.hpp

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
// #include "opencv2/core/core.hpp"
// #include "opencv2/core/gpumat.hpp"

@Namespace("cv::detail") public static class SeamFinder extends Pointer {
    static { Loader.load(); }
    public SeamFinder() { }
    public SeamFinder(Pointer p) { super(p); }

    public native void find(@Const @ByRef MatVector src, @StdVector Point corners,
                          @ByRef MatVector masks);
}


@Namespace("cv::detail") public static class NoSeamFinder extends SeamFinder {
    static { Loader.load(); }
    public NoSeamFinder() { allocate(); }
    public NoSeamFinder(int size) { allocateArray(size); }
    public NoSeamFinder(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public NoSeamFinder position(int position) {
        return (NoSeamFinder)super.position(position);
    }

    public native void find(@Const @ByRef MatVector arg0, @StdVector Point arg1, @ByRef MatVector arg2);
}


@Namespace("cv::detail") @NoOffset public static class PairwiseSeamFinder extends SeamFinder {
    static { Loader.load(); }
    public PairwiseSeamFinder() { }
    public PairwiseSeamFinder(Pointer p) { super(p); }

    public native void find(@Const @ByRef MatVector src, @StdVector Point corners,
                          @ByRef MatVector masks);
}


@Namespace("cv::detail") public static class VoronoiSeamFinder extends PairwiseSeamFinder {
    static { Loader.load(); }
    public VoronoiSeamFinder() { allocate(); }
    public VoronoiSeamFinder(int size) { allocateArray(size); }
    public VoronoiSeamFinder(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public VoronoiSeamFinder position(int position) {
        return (VoronoiSeamFinder)super.position(position);
    }

    public native void find(@StdVector Size size, @StdVector Point corners,
                          @ByRef MatVector masks);
}


@Namespace("cv::detail") @NoOffset public static class DpSeamFinder extends SeamFinder {
    static { Loader.load(); }
    public DpSeamFinder(Pointer p) { super(p); }

    /** enum cv::detail::DpSeamFinder::CostFunction */
    public static final int COLOR = 0, COLOR_GRAD = 1;

    public DpSeamFinder(@Cast("cv::detail::DpSeamFinder::CostFunction") int costFunc/*=COLOR*/) { allocate(costFunc); }
    private native void allocate(@Cast("cv::detail::DpSeamFinder::CostFunction") int costFunc/*=COLOR*/);
    public DpSeamFinder() { allocate(); }
    private native void allocate();

    public native @Cast("cv::detail::DpSeamFinder::CostFunction") int costFunction();
    public native void setCostFunction(@Cast("cv::detail::DpSeamFinder::CostFunction") int val);

    public native void find(@Const @ByRef MatVector src, @StdVector Point corners,
                          @ByRef MatVector masks);
}


@Namespace("cv::detail") public static class GraphCutSeamFinderBase extends Pointer {
    static { Loader.load(); }
    public GraphCutSeamFinderBase() { allocate(); }
    public GraphCutSeamFinderBase(int size) { allocateArray(size); }
    public GraphCutSeamFinderBase(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public GraphCutSeamFinderBase position(int position) {
        return (GraphCutSeamFinderBase)super.position(position);
    }

    /** enum cv::detail::GraphCutSeamFinderBase:: */
    public static final int COST_COLOR = 0, COST_COLOR_GRAD = 1;
}


@Namespace("cv::detail") @NoOffset public static class GraphCutSeamFinder extends GraphCutSeamFinderBase {
    static { Loader.load(); }
    public GraphCutSeamFinder(Pointer p) { super(p); }
    public GraphCutSeamFinder(int size) { allocateArray(size); }
    private native void allocateArray(int size);
    @Override public GraphCutSeamFinder position(int position) {
        return (GraphCutSeamFinder)super.position(position);
    }
    public SeamFinder asSeamFinder() { return asSeamFinder(this); }
    @Namespace public static native @Name("static_cast<cv::detail::SeamFinder*>") SeamFinder asSeamFinder(GraphCutSeamFinder pointer);

    public GraphCutSeamFinder(int cost_type/*=COST_COLOR_GRAD*/, float terminal_cost/*=10000.f*/,
                           float bad_region_penalty/*=1000.f*/) { allocate(cost_type, terminal_cost, bad_region_penalty); }
    private native void allocate(int cost_type/*=COST_COLOR_GRAD*/, float terminal_cost/*=10000.f*/,
                           float bad_region_penalty/*=1000.f*/);
    public GraphCutSeamFinder() { allocate(); }
    private native void allocate();

    public native void find(@Const @ByRef MatVector src, @StdVector Point corners,
                  @ByRef MatVector masks);
}


@Platform(not="android") @Namespace("cv::detail") @NoOffset public static class GraphCutSeamFinderGpu extends GraphCutSeamFinderBase {
    static { Loader.load(); }
    public GraphCutSeamFinderGpu(Pointer p) { super(p); }
    public GraphCutSeamFinderGpu(int size) { allocateArray(size); }
    private native void allocateArray(int size);
    @Override public GraphCutSeamFinderGpu position(int position) {
        return (GraphCutSeamFinderGpu)super.position(position);
    }
    public PairwiseSeamFinder asPairwiseSeamFinder() { return asPairwiseSeamFinder(this); }
    @Namespace public static native @Name("static_cast<cv::detail::PairwiseSeamFinder*>") PairwiseSeamFinder asPairwiseSeamFinder(GraphCutSeamFinderGpu pointer);

    public GraphCutSeamFinderGpu(int cost_type/*=COST_COLOR_GRAD*/, float terminal_cost/*=10000.f*/,
                              float bad_region_penalty/*=1000.f*/) { allocate(cost_type, terminal_cost, bad_region_penalty); }
    private native void allocate(int cost_type/*=COST_COLOR_GRAD*/, float terminal_cost/*=10000.f*/,
                              float bad_region_penalty/*=1000.f*/);
    public GraphCutSeamFinderGpu() { allocate(); }
    private native void allocate();

    public native void find(@Const @ByRef MatVector src, @StdVector Point corners,
                  @ByRef MatVector masks);
    public native void findInPair(@Cast("size_t") long first, @Cast("size_t") long second, @ByVal Rect roi);
}


 // namespace cv

// #endif // __OPENCV_STITCHING_SEAM_FINDERS_HPP__


// Parsed from /usr/local/include/opencv2/stitching/detail/blenders.hpp

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

// #include "opencv2/core/core.hpp"


// Simple blender which puts one image over another
@Namespace("cv::detail") public static class Blender extends Pointer {
    static { Loader.load(); }
    public Blender() { allocate(); }
    public Blender(int size) { allocateArray(size); }
    public Blender(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public Blender position(int position) {
        return (Blender)super.position(position);
    }


    /** enum cv::detail::Blender:: */
    public static final int NO = 0, FEATHER = 1, MULTI_BAND = 2;
    public native @Ptr Blender createDefault(int type, @Cast("bool") boolean try_gpu/*=false*/);
    public native @Ptr Blender createDefault(int type);

    public native void prepare(@StdVector Point corners, @StdVector Size sizes);
    public native void prepare(@ByVal Rect dst_roi);
    public native void feed(@Const @ByRef Mat img, @Const @ByRef Mat mask, @ByVal Point tl);
    public native void blend(@ByRef Mat dst, @ByRef Mat dst_mask);
}


@Namespace("cv::detail") @NoOffset public static class FeatherBlender extends Blender {
    static { Loader.load(); }
    public FeatherBlender(Pointer p) { super(p); }
    public FeatherBlender(int size) { allocateArray(size); }
    private native void allocateArray(int size);
    @Override public FeatherBlender position(int position) {
        return (FeatherBlender)super.position(position);
    }

    public FeatherBlender(float sharpness/*=0.02f*/) { allocate(sharpness); }
    private native void allocate(float sharpness/*=0.02f*/);
    public FeatherBlender() { allocate(); }
    private native void allocate();

    public native float sharpness();
    public native void setSharpness(float val);

    public native void prepare(@ByVal Rect dst_roi);
    public native void feed(@Const @ByRef Mat img, @Const @ByRef Mat mask, @ByVal Point tl);
    public native void blend(@ByRef Mat dst, @ByRef Mat dst_mask);

    // Creates weight maps for fixed set of source images by their masks and top-left corners.
    // Final image can be obtained by simple weighting of the source images.
    public native @ByVal Rect createWeightMaps(@Const @ByRef MatVector masks, @StdVector Point corners,
                              @ByRef MatVector weight_maps);
}




@Namespace("cv::detail") @NoOffset public static class MultiBandBlender extends Blender {
    static { Loader.load(); }
    public MultiBandBlender(Pointer p) { super(p); }
    public MultiBandBlender(int size) { allocateArray(size); }
    private native void allocateArray(int size);
    @Override public MultiBandBlender position(int position) {
        return (MultiBandBlender)super.position(position);
    }

    public MultiBandBlender(int try_gpu/*=false*/, int num_bands/*=5*/, int weight_type/*=CV_32F*/) { allocate(try_gpu, num_bands, weight_type); }
    private native void allocate(int try_gpu/*=false*/, int num_bands/*=5*/, int weight_type/*=CV_32F*/);
    public MultiBandBlender() { allocate(); }
    private native void allocate();

    public native int numBands();
    public native void setNumBands(int val);

    public native void prepare(@ByVal Rect dst_roi);
    public native void feed(@Const @ByRef Mat img, @Const @ByRef Mat mask, @ByVal Point tl);
    public native void blend(@ByRef Mat dst, @ByRef Mat dst_mask); //CV_32F or CV_16S
}


//////////////////////////////////////////////////////////////////////////////
// Auxiliary functions

@Namespace("cv::detail") public static native void normalizeUsingWeightMap(@Const @ByRef Mat weight, @ByRef Mat src);

@Namespace("cv::detail") public static native void createWeightMap(@Const @ByRef Mat mask, float sharpness, @ByRef Mat weight);

@Namespace("cv::detail") public static native void createLaplacePyr(@Const @ByRef Mat img, int num_levels, @ByRef MatVector pyr);
@Namespace("cv::detail") public static native void createLaplacePyrGpu(@Const @ByRef Mat img, int num_levels, @ByRef MatVector pyr);

// Restores source image
@Namespace("cv::detail") public static native void restoreImageFromLaplacePyr(@ByRef MatVector pyr);
@Namespace("cv::detail") public static native void restoreImageFromLaplacePyrGpu(@ByRef MatVector pyr);


 // namespace cv

// #endif // __OPENCV_STITCHING_BLENDERS_HPP__


// Parsed from /usr/local/include/opencv2/stitching/detail/autocalib.hpp

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

// #include "opencv2/core/core.hpp"
// #include "matchers.hpp"

// See "Construction of Panoramic Image Mosaics with Global and Local Alignment"
// by Heung-Yeung Shum and Richard Szeliski.
@Namespace("cv::detail") public static native void focalsFromHomography(@Const @ByRef Mat H, @ByRef DoublePointer f0, @ByRef DoublePointer f1, @Cast("bool*") @ByRef BoolPointer f0_ok, @Cast("bool*") @ByRef BoolPointer f1_ok);
@Namespace("cv::detail") public static native void focalsFromHomography(@Const @ByRef Mat H, @ByRef DoubleBuffer f0, @ByRef DoubleBuffer f1, @Cast("bool*") @ByRef BoolPointer f0_ok, @Cast("bool*") @ByRef BoolPointer f1_ok);
@Namespace("cv::detail") public static native void focalsFromHomography(@Const @ByRef Mat H, @ByRef double[] f0, @ByRef double[] f1, @Cast("bool*") @ByRef BoolPointer f0_ok, @Cast("bool*") @ByRef BoolPointer f1_ok);

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


 // namespace cv

// #endif // __OPENCV_STITCHING_AUTOCALIB_HPP__


// Parsed from /usr/local/include/opencv2/stitching/warpers.hpp

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

@Namespace("cv") public static class WarperCreator extends Pointer {
    static { Loader.load(); }
    public WarperCreator() { }
    public WarperCreator(Pointer p) { super(p); }

    public native @Ptr RotationWarper create(float scale);
}


@Namespace("cv") public static class PlaneWarper extends WarperCreator {
    static { Loader.load(); }
    public PlaneWarper() { allocate(); }
    public PlaneWarper(int size) { allocateArray(size); }
    public PlaneWarper(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public PlaneWarper position(int position) {
        return (PlaneWarper)super.position(position);
    }

    public native @Ptr RotationWarper create(float scale);
}


@Namespace("cv") public static class CylindricalWarper extends WarperCreator {
    static { Loader.load(); }
    public CylindricalWarper() { allocate(); }
    public CylindricalWarper(int size) { allocateArray(size); }
    public CylindricalWarper(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public CylindricalWarper position(int position) {
        return (CylindricalWarper)super.position(position);
    }

    public native @Ptr RotationWarper create(float scale);
}


@Namespace("cv") public static class SphericalWarper extends WarperCreator {
    static { Loader.load(); }
    public SphericalWarper() { allocate(); }
    public SphericalWarper(int size) { allocateArray(size); }
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
    public FisheyeWarper() { allocate(); }
    public FisheyeWarper(int size) { allocateArray(size); }
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
    public StereographicWarper() { allocate(); }
    public StereographicWarper(int size) { allocateArray(size); }
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
    public CompressedRectilinearWarper(Pointer p) { super(p); }
    public CompressedRectilinearWarper(int size) { allocateArray(size); }
    private native void allocateArray(int size);
    @Override public CompressedRectilinearWarper position(int position) {
        return (CompressedRectilinearWarper)super.position(position);
    }

    public CompressedRectilinearWarper(float A/*=1*/, float B/*=1*/) { allocate(A, B); }
    private native void allocate(float A/*=1*/, float B/*=1*/);
    public CompressedRectilinearWarper() { allocate(); }
    private native void allocate();
    public native @Ptr RotationWarper create(float scale);
}

@Namespace("cv") @NoOffset public static class CompressedRectilinearPortraitWarper extends WarperCreator {
    static { Loader.load(); }
    public CompressedRectilinearPortraitWarper(Pointer p) { super(p); }
    public CompressedRectilinearPortraitWarper(int size) { allocateArray(size); }
    private native void allocateArray(int size);
    @Override public CompressedRectilinearPortraitWarper position(int position) {
        return (CompressedRectilinearPortraitWarper)super.position(position);
    }

    public CompressedRectilinearPortraitWarper(float A/*=1*/, float B/*=1*/) { allocate(A, B); }
    private native void allocate(float A/*=1*/, float B/*=1*/);
    public CompressedRectilinearPortraitWarper() { allocate(); }
    private native void allocate();
    public native @Ptr RotationWarper create(float scale);
}

@Namespace("cv") @NoOffset public static class PaniniWarper extends WarperCreator {
    static { Loader.load(); }
    public PaniniWarper(Pointer p) { super(p); }
    public PaniniWarper(int size) { allocateArray(size); }
    private native void allocateArray(int size);
    @Override public PaniniWarper position(int position) {
        return (PaniniWarper)super.position(position);
    }

    public PaniniWarper(float A/*=1*/, float B/*=1*/) { allocate(A, B); }
    private native void allocate(float A/*=1*/, float B/*=1*/);
    public PaniniWarper() { allocate(); }
    private native void allocate();
    public native @Ptr RotationWarper create(float scale);
}

@Namespace("cv") @NoOffset public static class PaniniPortraitWarper extends WarperCreator {
    static { Loader.load(); }
    public PaniniPortraitWarper(Pointer p) { super(p); }
    public PaniniPortraitWarper(int size) { allocateArray(size); }
    private native void allocateArray(int size);
    @Override public PaniniPortraitWarper position(int position) {
        return (PaniniPortraitWarper)super.position(position);
    }

    public PaniniPortraitWarper(float A/*=1*/, float B/*=1*/) { allocate(A, B); }
    private native void allocate(float A/*=1*/, float B/*=1*/);
    public PaniniPortraitWarper() { allocate(); }
    private native void allocate();
    public native @Ptr RotationWarper create(float scale);
}

@Namespace("cv") public static class MercatorWarper extends WarperCreator {
    static { Loader.load(); }
    public MercatorWarper() { allocate(); }
    public MercatorWarper(int size) { allocateArray(size); }
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
    public TransverseMercatorWarper() { allocate(); }
    public TransverseMercatorWarper(int size) { allocateArray(size); }
    public TransverseMercatorWarper(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public TransverseMercatorWarper position(int position) {
        return (TransverseMercatorWarper)super.position(position);
    }

    public native @Ptr RotationWarper create(float scale);
}



@Platform(not="android") @Namespace("cv") public static class PlaneWarperGpu extends WarperCreator {
    static { Loader.load(); }
    public PlaneWarperGpu() { allocate(); }
    public PlaneWarperGpu(int size) { allocateArray(size); }
    public PlaneWarperGpu(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public PlaneWarperGpu position(int position) {
        return (PlaneWarperGpu)super.position(position);
    }

    public native @Ptr RotationWarper create(float scale);
}


@Platform(not="android") @Namespace("cv") public static class CylindricalWarperGpu extends WarperCreator {
    static { Loader.load(); }
    public CylindricalWarperGpu() { allocate(); }
    public CylindricalWarperGpu(int size) { allocateArray(size); }
    public CylindricalWarperGpu(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public CylindricalWarperGpu position(int position) {
        return (CylindricalWarperGpu)super.position(position);
    }

    public native @Ptr RotationWarper create(float scale);
}


@Platform(not="android") @Namespace("cv") public static class SphericalWarperGpu extends WarperCreator {
    static { Loader.load(); }
    public SphericalWarperGpu() { allocate(); }
    public SphericalWarperGpu(int size) { allocateArray(size); }
    public SphericalWarperGpu(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public SphericalWarperGpu position(int position) {
        return (SphericalWarperGpu)super.position(position);
    }

    public native @Ptr RotationWarper create(float scale);
}

 // namespace cv

// #endif // __OPENCV_STITCHING_WARPER_CREATORS_HPP__


// Parsed from /usr/local/include/opencv2/stitching/stitcher.hpp

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

// #include "opencv2/core/core.hpp"
// #include "opencv2/features2d/features2d.hpp"
// #include "opencv2/stitching/warpers.hpp"
// #include "opencv2/stitching/detail/matchers.hpp"
// #include "opencv2/stitching/detail/motion_estimators.hpp"
// #include "opencv2/stitching/detail/exposure_compensate.hpp"
// #include "opencv2/stitching/detail/seam_finders.hpp"
// #include "opencv2/stitching/detail/blenders.hpp"
// #include "opencv2/stitching/detail/camera.hpp"

@Namespace("cv") @NoOffset public static class Stitcher extends Pointer {
    static { Loader.load(); }
    public Stitcher() { }
    public Stitcher(Pointer p) { super(p); }

    /** enum cv::Stitcher:: */
    public static final int ORIG_RESOL = -1;
    /** enum cv::Stitcher::Status */
    public static final int OK = 0, ERR_NEED_MORE_IMGS = 1;

    // Creates stitcher with default parameters
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

    public native @Const @ByRef Mat matchingMask();
    public native void setMatchingMask(@Const @ByRef Mat mask);

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

    public native @Cast("cv::Stitcher::Status") int estimateTransform(@ByVal MatVector images);
    public native @Cast("cv::Stitcher::Status") int estimateTransform(@ByVal MatVector images, @Const @ByRef RectVectorVector rois);

    public native @Cast("cv::Stitcher::Status") int composePanorama(@ByVal Mat pano);
    public native @Cast("cv::Stitcher::Status") int composePanorama(@ByVal MatVector images, @ByVal Mat pano);

    public native @Cast("cv::Stitcher::Status") int stitch(@ByVal MatVector images, @ByVal Mat pano);
    public native @Cast("cv::Stitcher::Status") int stitch(@ByVal MatVector images, @Const @ByRef RectVectorVector rois, @ByVal Mat pano);

    public native @StdVector IntPointer component();
    public native @StdVector CameraParams cameras();
    public native double workScale();
}

 // namespace cv

// #endif // __OPENCV_STITCHING_STITCHER_HPP__


}
