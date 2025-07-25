// Targeted by JavaCPP version 1.5.13-SNAPSHOT: DO NOT EDIT THIS FILE

package org.bytedeco.opencv.opencv_cudaoptflow;

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
import org.bytedeco.opencv.opencv_cudaarithm.*;
import static org.bytedeco.opencv.global.opencv_cudaarithm.*;
import org.bytedeco.opencv.opencv_cudafilters.*;
import static org.bytedeco.opencv.global.opencv_cudafilters.*;
import org.bytedeco.opencv.opencv_cudaimgproc.*;
import static org.bytedeco.opencv.global.opencv_cudaimgproc.*;
import static org.bytedeco.opencv.global.opencv_cudawarping.*;

import static org.bytedeco.opencv.global.opencv_cudaoptflow.*;


//
// OpticalFlowDual_TVL1
//

/** \brief Implementation of the Zach, Pock and Bischof Dual TV-L1 Optical Flow method.
 *
 * \note C. Zach, T. Pock and H. Bischof, "A Duality Based Approach for Realtime TV-L1 Optical Flow".
 * \note Javier Sanchez, Enric Meinhardt-Llopis and Gabriele Facciolo. "TV-L1 Optical Flow Estimation".
 */
@Namespace("cv::cuda") @Properties(inherit = org.bytedeco.opencv.presets.opencv_cudaoptflow.class)
public class OpticalFlowDual_TVL1 extends DenseOpticalFlow {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public OpticalFlowDual_TVL1(Pointer p) { super(p); }
    /** Downcast constructor. */
    public OpticalFlowDual_TVL1(Algorithm pointer) { super((Pointer)null); allocate(pointer); }
    @Namespace private native @Name("static_cast<cv::cuda::OpticalFlowDual_TVL1*>") void allocate(Algorithm pointer);

    /**
     * Time step of the numerical scheme.
     */
    public native double getTau();
    public native void setTau(double tau);

    /**
     * Weight parameter for the data term, attachment parameter.
     * This is the most relevant parameter, which determines the smoothness of the output.
     * The smaller this parameter is, the smoother the solutions we obtain.
     * It depends on the range of motions of the images, so its value should be adapted to each image sequence.
     */
    public native double getLambda();
    public native void setLambda(double lambda);

    /**
     * Weight parameter for (u - v)^2, tightness parameter.
     * It serves as a link between the attachment and the regularization terms.
     * In theory, it should have a small value in order to maintain both parts in correspondence.
     * The method is stable for a large range of values of this parameter.
     */
    public native double getGamma();
    public native void setGamma(double gamma);

    /**
     * parameter used for motion estimation. It adds a variable allowing for illumination variations
     * Set this parameter to 1. if you have varying illumination.
     * See: Chambolle et al, A First-Order Primal-Dual Algorithm for Convex Problems with Applications to Imaging
     * Journal of Mathematical imaging and vision, may 2011 Vol 40 issue 1, pp 120-145
     */
    public native double getTheta();
    public native void setTheta(double theta);

    /**
     * Number of scales used to create the pyramid of images.
     */
    public native int getNumScales();
    public native void setNumScales(int nscales);

    /**
     * Number of warpings per scale.
     * Represents the number of times that I1(x+u0) and grad( I1(x+u0) ) are computed per scale.
     * This is a parameter that assures the stability of the method.
     * It also affects the running time, so it is a compromise between speed and accuracy.
     */
    public native int getNumWarps();
    public native void setNumWarps(int warps);

    /**
     * Stopping criterion threshold used in the numerical scheme, which is a trade-off between precision and running time.
     * A small value will yield more accurate solutions at the expense of a slower convergence.
     */
    public native double getEpsilon();
    public native void setEpsilon(double epsilon);

    /**
     * Stopping criterion iterations number used in the numerical scheme.
     */
    public native int getNumIterations();
    public native void setNumIterations(int iterations);

    public native double getScaleStep();
    public native void setScaleStep(double scaleStep);

    public native @Cast("bool") boolean getUseInitialFlow();
    public native void setUseInitialFlow(@Cast("bool") boolean useInitialFlow);

    public static native @Ptr OpticalFlowDual_TVL1 create(
                double tau/*=0.25*/,
                double lambda/*=0.15*/,
                double theta/*=0.3*/,
                int nscales/*=5*/,
                int warps/*=5*/,
                double epsilon/*=0.01*/,
                int iterations/*=300*/,
                double scaleStep/*=0.8*/,
                double gamma/*=0.0*/,
                @Cast("bool") boolean useInitialFlow/*=false*/);
    public static native @Ptr OpticalFlowDual_TVL1 create();
}
