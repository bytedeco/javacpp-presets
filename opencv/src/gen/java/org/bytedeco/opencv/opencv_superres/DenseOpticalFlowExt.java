// Targeted by JavaCPP version 1.5.13-SNAPSHOT: DO NOT EDIT THIS FILE

package org.bytedeco.opencv.opencv_superres;

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

import static org.bytedeco.opencv.global.opencv_superres.*;


/** \addtogroup superres
 *  \{ */

        @Namespace("cv::superres") @Properties(inherit = org.bytedeco.opencv.presets.opencv_superres.class)
public class DenseOpticalFlowExt extends Algorithm {
            static { Loader.load(); }
            /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
            public DenseOpticalFlowExt(Pointer p) { super(p); }
            /** Downcast constructor. */
            public DenseOpticalFlowExt(Algorithm pointer) { super((Pointer)null); allocate(pointer); }
            @Namespace private native @Name("static_cast<cv::superres::DenseOpticalFlowExt*>") void allocate(Algorithm pointer);
            @Override public Algorithm asAlgorithm() { return asAlgorithm(this); }
            @Namespace public static native @Name("static_cast<cv::Algorithm*>") Algorithm asAlgorithm(DenseOpticalFlowExt pointer);
        
            public native void calc(@ByVal Mat frame0, @ByVal Mat frame1, @ByVal Mat flow1, @ByVal(nullValue = "cv::OutputArray(cv::noArray())") Mat flow2);
            public native void calc(@ByVal Mat frame0, @ByVal Mat frame1, @ByVal Mat flow1);
            public native void calc(@ByVal UMat frame0, @ByVal UMat frame1, @ByVal UMat flow1, @ByVal(nullValue = "cv::OutputArray(cv::noArray())") UMat flow2);
            public native void calc(@ByVal UMat frame0, @ByVal UMat frame1, @ByVal UMat flow1);
            public native void calc(@ByVal GpuMat frame0, @ByVal GpuMat frame1, @ByVal GpuMat flow1, @ByVal(nullValue = "cv::OutputArray(cv::noArray())") GpuMat flow2);
            public native void calc(@ByVal GpuMat frame0, @ByVal GpuMat frame1, @ByVal GpuMat flow1);
            public native void collectGarbage();
        }
