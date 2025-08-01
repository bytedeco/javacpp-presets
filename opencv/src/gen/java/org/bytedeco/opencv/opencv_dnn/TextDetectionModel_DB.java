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


/** \brief This class represents high-level API for text detection DL networks compatible with DB model.
 *
 * Related publications: \cite liao2020real
 * Paper: https://arxiv.org/abs/1911.08947
 * For more information about the hyper-parameters setting, please refer to https://github.com/MhLiao/DB
 *
 * Configurable parameters:
 * - (float) binaryThreshold - The threshold of the binary map. It is usually set to 0.3.
 * - (float) polygonThreshold - The threshold of text polygons. It is usually set to 0.5, 0.6, and 0.7. Default is 0.5f
 * - (double) unclipRatio - The unclip ratio of the detected text region, which determines the output size. It is usually set to 2.0.
 * - (int) maxCandidates - The max number of the output results.
 */
@Namespace("cv::dnn") @Properties(inherit = org.bytedeco.opencv.presets.opencv_dnn.class)
public class TextDetectionModel_DB extends TextDetectionModel {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public TextDetectionModel_DB(Pointer p) { super(p); }
    /** Native array allocator. Access with {@link Pointer#position(long)}. */
    public TextDetectionModel_DB(long size) { super((Pointer)null); allocateArray(size); }
    private native void allocateArray(long size);
    @Override public TextDetectionModel_DB position(long position) {
        return (TextDetectionModel_DB)super.position(position);
    }
    @Override public TextDetectionModel_DB getPointer(long i) {
        return new TextDetectionModel_DB((Pointer)this).offsetAddress(i);
    }

    @Deprecated public TextDetectionModel_DB() { super((Pointer)null); allocate(); }
    @Deprecated private native void allocate();

    /**
     * \brief Create text detection algorithm from deep learning network.
     * @param network [in] Net object.
     */
    public TextDetectionModel_DB(@Const @ByRef Net network) { super((Pointer)null); allocate(network); }
    private native void allocate(@Const @ByRef Net network);

    /**
     * \brief Create text detection model from network represented in one of the supported formats.
     * An order of \p model and \p config arguments does not matter.
     * @param model [in] Binary file contains trained weights.
     * @param config [in] Text file contains network configuration.
     */
    public TextDetectionModel_DB(@StdString BytePointer model, @StdString BytePointer config/*=""*/) { super((Pointer)null); allocate(model, config); }
    private native void allocate(@StdString BytePointer model, @StdString BytePointer config/*=""*/);
    public TextDetectionModel_DB(@StdString BytePointer model) { super((Pointer)null); allocate(model); }
    private native void allocate(@StdString BytePointer model);
    public TextDetectionModel_DB(@StdString String model, @StdString String config/*=""*/) { super((Pointer)null); allocate(model, config); }
    private native void allocate(@StdString String model, @StdString String config/*=""*/);
    public TextDetectionModel_DB(@StdString String model) { super((Pointer)null); allocate(model); }
    private native void allocate(@StdString String model);

    public native @ByRef TextDetectionModel_DB setBinaryThreshold(float binaryThreshold);
    public native float getBinaryThreshold();

    public native @ByRef TextDetectionModel_DB setPolygonThreshold(float polygonThreshold);
    public native float getPolygonThreshold();

    public native @ByRef TextDetectionModel_DB setUnclipRatio(double unclipRatio);
    public native double getUnclipRatio();

    public native @ByRef TextDetectionModel_DB setMaxCandidates(int maxCandidates);
    public native int getMaxCandidates();
}
