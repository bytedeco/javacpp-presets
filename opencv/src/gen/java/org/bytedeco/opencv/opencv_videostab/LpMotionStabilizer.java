// Targeted by JavaCPP version 1.5.13-SNAPSHOT: DO NOT EDIT THIS FILE

package org.bytedeco.opencv.opencv_videostab;

import java.nio.*;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

import static org.bytedeco.javacpp.presets.javacpp.*;
import static org.bytedeco.openblas.global.openblas_nolapack.*;
import static org.bytedeco.openblas.global.openblas.*;
import org.bytedeco.opencv.opencv_core.*;
import static org.bytedeco.opencv.global.opencv_core.*;
import org.bytedeco.opencv.opencv_imgproc.*;
import static org.bytedeco.opencv.global.opencv_imgproc.*;
import static org.bytedeco.opencv.global.opencv_imgcodecs.*;
import org.bytedeco.opencv.opencv_videoio.*;
import static org.bytedeco.opencv.global.opencv_videoio.*;
import org.bytedeco.opencv.opencv_highgui.*;
import static org.bytedeco.opencv.global.opencv_highgui.*;
import org.bytedeco.opencv.opencv_flann.*;
import static org.bytedeco.opencv.global.opencv_flann.*;
import org.bytedeco.opencv.opencv_features2d.*;
import static org.bytedeco.opencv.global.opencv_features2d.*;
import org.bytedeco.opencv.opencv_calib3d.*;
import static org.bytedeco.opencv.global.opencv_calib3d.*;
import org.bytedeco.opencv.opencv_dnn.*;
import static org.bytedeco.opencv.global.opencv_dnn.*;
import org.bytedeco.opencv.opencv_objdetect.*;
import static org.bytedeco.opencv.global.opencv_objdetect.*;
import org.bytedeco.opencv.opencv_video.*;
import static org.bytedeco.opencv.global.opencv_video.*;
import org.bytedeco.opencv.opencv_ximgproc.*;
import static org.bytedeco.opencv.global.opencv_ximgproc.*;
import org.bytedeco.opencv.opencv_optflow.*;
import static org.bytedeco.opencv.global.opencv_optflow.*;
import org.bytedeco.opencv.opencv_photo.*;
import static org.bytedeco.opencv.global.opencv_photo.*;

import static org.bytedeco.opencv.global.opencv_videostab.*;


@Namespace("cv::videostab") @NoOffset @Properties(inherit = org.bytedeco.opencv.presets.opencv_videostab.class)
public class LpMotionStabilizer extends IMotionStabilizer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public LpMotionStabilizer(Pointer p) { super(p); }

    public LpMotionStabilizer(@Cast("cv::videostab::MotionModel") int model/*=cv::videostab::MM_SIMILARITY*/) { super((Pointer)null); allocate(model); }
    private native void allocate(@Cast("cv::videostab::MotionModel") int model/*=cv::videostab::MM_SIMILARITY*/);
    public LpMotionStabilizer() { super((Pointer)null); allocate(); }
    private native void allocate();

    public native void setMotionModel(@Cast("cv::videostab::MotionModel") int val);
    public native @Cast("cv::videostab::MotionModel") int motionModel();

    public native void setFrameSize(@ByVal Size val);
    public native @ByVal Size frameSize();

    public native void setTrimRatio(float val);
    public native float trimRatio();

    public native void setWeight1(float val);
    public native float weight1();

    public native void setWeight2(float val);
    public native float weight2();

    public native void setWeight3(float val);
    public native float weight3();

    public native void setWeight4(float val);
    public native float weight4();

    public native void stabilize(
                int size, @Const @ByRef MatVector motions, @Const @ByRef Range range,
                Mat stabilizationMotions);
}
