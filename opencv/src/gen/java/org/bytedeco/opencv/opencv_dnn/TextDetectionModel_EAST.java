// Targeted by JavaCPP version 1.5.13-SNAPSHOT: DO NOT EDIT THIS FILE

package org.bytedeco.opencv.opencv_dnn;

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

import static org.bytedeco.opencv.global.opencv_dnn.*;


/** \brief This class represents high-level API for text detection DL networks compatible with EAST model.
 *
 * Configurable parameters:
 * - (float) confThreshold - used to filter boxes by confidences, default: 0.5f
 * - (float) nmsThreshold - used in non maximum suppression, default: 0.0f
 */
@Namespace("cv::dnn") @Properties(inherit = org.bytedeco.opencv.presets.opencv_dnn.class)
public class TextDetectionModel_EAST extends TextDetectionModel {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public TextDetectionModel_EAST(Pointer p) { super(p); }
    /** Native array allocator. Access with {@link Pointer#position(long)}. */
    public TextDetectionModel_EAST(long size) { super((Pointer)null); allocateArray(size); }
    private native void allocateArray(long size);
    @Override public TextDetectionModel_EAST position(long position) {
        return (TextDetectionModel_EAST)super.position(position);
    }
    @Override public TextDetectionModel_EAST getPointer(long i) {
        return new TextDetectionModel_EAST((Pointer)this).offsetAddress(i);
    }

    @Deprecated public TextDetectionModel_EAST() { super((Pointer)null); allocate(); }
    @Deprecated private native void allocate();

    /**
     * \brief Create text detection algorithm from deep learning network
     * @param network [in] Net object
     */
    public TextDetectionModel_EAST(@Const @ByRef Net network) { super((Pointer)null); allocate(network); }
    private native void allocate(@Const @ByRef Net network);

    /**
     * \brief Create text detection model from network represented in one of the supported formats.
     * An order of \p model and \p config arguments does not matter.
     * @param model [in] Binary file contains trained weights.
     * @param config [in] Text file contains network configuration.
     */
    public TextDetectionModel_EAST(@StdString BytePointer model, @StdString BytePointer config/*=""*/) { super((Pointer)null); allocate(model, config); }
    private native void allocate(@StdString BytePointer model, @StdString BytePointer config/*=""*/);
    public TextDetectionModel_EAST(@StdString BytePointer model) { super((Pointer)null); allocate(model); }
    private native void allocate(@StdString BytePointer model);
    public TextDetectionModel_EAST(@StdString String model, @StdString String config/*=""*/) { super((Pointer)null); allocate(model, config); }
    private native void allocate(@StdString String model, @StdString String config/*=""*/);
    public TextDetectionModel_EAST(@StdString String model) { super((Pointer)null); allocate(model); }
    private native void allocate(@StdString String model);

    /**
     * \brief Set the detection confidence threshold
     * @param confThreshold [in] A threshold used to filter boxes by confidences
     */
    public native @ByRef TextDetectionModel_EAST setConfidenceThreshold(float confThreshold);

    /**
     * \brief Get the detection confidence threshold
     */
    public native float getConfidenceThreshold();

    /**
     * \brief Set the detection NMS filter threshold
     * @param nmsThreshold [in] A threshold used in non maximum suppression
     */
    public native @ByRef TextDetectionModel_EAST setNMSThreshold(float nmsThreshold);

    /**
     * \brief Get the detection confidence threshold
     */
    public native float getNMSThreshold();
}
