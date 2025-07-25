// Targeted by JavaCPP version 1.5.13-SNAPSHOT: DO NOT EDIT THIS FILE

package org.bytedeco.opencv.opencv_cudastereo;

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

import static org.bytedeco.opencv.global.opencv_cudastereo.*;


/////////////////////////////////////////
// StereoConstantSpaceBP

/** \brief Class computing stereo correspondence using the constant space belief propagation algorithm. :
<p>
The class implements algorithm described in \cite Yang2010 . StereoConstantSpaceBP supports both local
minimum and global minimum data cost initialization algorithms. For more details, see the paper
mentioned above. By default, a local algorithm is used. To enable a global algorithm, set
use_local_init_data_cost to false .
<p>
StereoConstantSpaceBP uses a truncated linear model for the data cost and discontinuity terms:
<p>
<pre>{@code \[DataCost = data \_ weight  \cdot \min ( \lvert I_2-I_1  \rvert , max \_ data \_ term)\]}</pre>
<p>
<pre>{@code \[DiscTerm =  \min (disc \_ single \_ jump  \cdot \lvert f_1-f_2  \rvert , max \_ disc \_ term)\]}</pre>
<p>
For more details, see \cite Yang2010 .
<p>
By default, StereoConstantSpaceBP uses floating-point arithmetics and the CV_32FC1 type for
messages. But it can also use fixed-point arithmetics and the CV_16SC1 message type for better
performance. To avoid an overflow in this case, the parameters must satisfy the following
requirement:
<p>
<pre>{@code \[10  \cdot 2^{levels-1}  \cdot max \_ data \_ term < SHRT \_ MAX\]}</pre>
 <p>
 */
@Namespace("cv::cuda") @Properties(inherit = org.bytedeco.opencv.presets.opencv_cudastereo.class)
public class StereoConstantSpaceBP extends StereoBeliefPropagation {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public StereoConstantSpaceBP(Pointer p) { super(p); }
    /** Downcast constructor. */
    public StereoConstantSpaceBP(Algorithm pointer) { super((Pointer)null); allocate(pointer); }
    @Namespace private native @Name("static_cast<cv::cuda::StereoConstantSpaceBP*>") void allocate(Algorithm pointer);

    /** number of active disparity on the first level */
    public native int getNrPlane();
    public native void setNrPlane(int nr_plane);

    public native @Cast("bool") boolean getUseLocalInitDataCost();
    public native void setUseLocalInitDataCost(@Cast("bool") boolean use_local_init_data_cost);

    /** \brief Uses a heuristic method to compute parameters (ndisp, iters, levelsand nrplane) for the specified
    image size (widthand height).
     */
    public static native void estimateRecommendedParams(int width, int height, @ByRef IntPointer ndisp, @ByRef IntPointer iters, @ByRef IntPointer levels, @ByRef IntPointer nr_plane);
    public static native void estimateRecommendedParams(int width, int height, @ByRef IntBuffer ndisp, @ByRef IntBuffer iters, @ByRef IntBuffer levels, @ByRef IntBuffer nr_plane);
    public static native void estimateRecommendedParams(int width, int height, @ByRef int[] ndisp, @ByRef int[] iters, @ByRef int[] levels, @ByRef int[] nr_plane);
}
