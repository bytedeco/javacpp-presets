// Targeted by JavaCPP version 1.5.13-SNAPSHOT: DO NOT EDIT THIS FILE

package org.bytedeco.opencv.opencv_optflow;

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
import org.bytedeco.opencv.opencv_video.*;
import static org.bytedeco.opencv.global.opencv_video.*;
import org.bytedeco.opencv.opencv_ximgproc.*;
import static org.bytedeco.opencv.global.opencv_ximgproc.*;

import static org.bytedeco.opencv.global.opencv_optflow.*;


/** \brief "Dual TV L1" Optical Flow Algorithm.
<p>
The class implements the "Dual TV L1" optical flow algorithm described in \cite Zach2007 and
\cite Javier2012 .
Here are important members of the class that control the algorithm, which you can set after
constructing the class instance:
<p>
-   member double tau
    Time step of the numerical scheme.
<p>
-   member double lambda
    Weight parameter for the data term, attachment parameter. This is the most relevant
    parameter, which determines the smoothness of the output. The smaller this parameter is,
    the smoother the solutions we obtain. It depends on the range of motions of the images, so
    its value should be adapted to each image sequence.
<p>
-   member double theta
    Weight parameter for (u - v)\^2, tightness parameter. It serves as a link between the
    attachment and the regularization terms. In theory, it should have a small value in order
    to maintain both parts in correspondence. The method is stable for a large range of values
    of this parameter.
<p>
-   member int nscales
    Number of scales used to create the pyramid of images.
<p>
-   member int warps
    Number of warpings per scale. Represents the number of times that I1(x+u0) and grad(
    I1(x+u0) ) are computed per scale. This is a parameter that assures the stability of the
    method. It also affects the running time, so it is a compromise between speed and
    accuracy.
<p>
-   member double epsilon
    Stopping criterion threshold used in the numerical scheme, which is a trade-off between
    precision and running time. A small value will yield more accurate solutions at the
    expense of a slower convergence.
<p>
-   member int iterations
    Stopping criterion iterations number used in the numerical scheme.
<p>
C. Zach, T. Pock and H. Bischof, "A Duality Based Approach for Realtime TV-L1 Optical Flow".
Javier Sanchez, Enric Meinhardt-Llopis and Gabriele Facciolo. "TV-L1 Optical Flow Estimation".
*/
@Namespace("cv::optflow") @Properties(inherit = org.bytedeco.opencv.presets.opencv_optflow.class)
public class DualTVL1OpticalFlow extends DenseOpticalFlow {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public DualTVL1OpticalFlow(Pointer p) { super(p); }
    /** Downcast constructor. */
    public DualTVL1OpticalFlow(Algorithm pointer) { super((Pointer)null); allocate(pointer); }
    @Namespace private native @Name("static_cast<cv::optflow::DualTVL1OpticalFlow*>") void allocate(Algorithm pointer);

    /** \brief Time step of the numerical scheme
    /** @see setTau */
    public native double getTau();
    /** \copybrief getTau @see getTau */
    public native void setTau(double val);
    /** \brief Weight parameter for the data term, attachment parameter
    /** @see setLambda */
    public native double getLambda();
    /** \copybrief getLambda @see getLambda */
    public native void setLambda(double val);
    /** \brief Weight parameter for (u - v)^2, tightness parameter
    /** @see setTheta */
    public native double getTheta();
    /** \copybrief getTheta @see getTheta */
    public native void setTheta(double val);
    /** \brief coefficient for additional illumination variation term
    /** @see setGamma */
    public native double getGamma();
    /** \copybrief getGamma @see getGamma */
    public native void setGamma(double val);
    /** \brief Number of scales used to create the pyramid of images
    /** @see setScalesNumber */
    public native int getScalesNumber();
    /** \copybrief getScalesNumber @see getScalesNumber */
    public native void setScalesNumber(int val);
    /** \brief Number of warpings per scale
    /** @see setWarpingsNumber */
    public native int getWarpingsNumber();
    /** \copybrief getWarpingsNumber @see getWarpingsNumber */
    public native void setWarpingsNumber(int val);
    /** \brief Stopping criterion threshold used in the numerical scheme, which is a trade-off between precision and running time
    /** @see setEpsilon */
    public native double getEpsilon();
    /** \copybrief getEpsilon @see getEpsilon */
    public native void setEpsilon(double val);
    /** \brief Inner iterations (between outlier filtering) used in the numerical scheme
    /** @see setInnerIterations */
    public native int getInnerIterations();
    /** \copybrief getInnerIterations @see getInnerIterations */
    public native void setInnerIterations(int val);
    /** \brief Outer iterations (number of inner loops) used in the numerical scheme
    /** @see setOuterIterations */
    public native int getOuterIterations();
    /** \copybrief getOuterIterations @see getOuterIterations */
    public native void setOuterIterations(int val);
    /** \brief Use initial flow
    /** @see setUseInitialFlow */
    public native @Cast("bool") boolean getUseInitialFlow();
    /** \copybrief getUseInitialFlow @see getUseInitialFlow */
    public native void setUseInitialFlow(@Cast("bool") boolean val);
    /** \brief Step between scales (<1)
    /** @see setScaleStep */
    public native double getScaleStep();
    /** \copybrief getScaleStep @see getScaleStep */
    public native void setScaleStep(double val);
    /** \brief Median filter kernel size (1 = no filter) (3 or 5)
    /** @see setMedianFiltering */
    public native int getMedianFiltering();
    /** \copybrief getMedianFiltering @see getMedianFiltering */
    public native void setMedianFiltering(int val);

    /** \brief Creates instance of cv::DualTVL1OpticalFlow*/
    public static native @Ptr DualTVL1OpticalFlow create(
                                                double tau/*=0.25*/,
                                                double lambda/*=0.15*/,
                                                double theta/*=0.3*/,
                                                int nscales/*=5*/,
                                                int warps/*=5*/,
                                                double epsilon/*=0.01*/,
                                                int innnerIterations/*=30*/,
                                                int outerIterations/*=10*/,
                                                double scaleStep/*=0.8*/,
                                                double gamma/*=0.0*/,
                                                int medianFiltering/*=5*/,
                                                @Cast("bool") boolean useInitialFlow/*=false*/);
    public static native @Ptr DualTVL1OpticalFlow create();
}
