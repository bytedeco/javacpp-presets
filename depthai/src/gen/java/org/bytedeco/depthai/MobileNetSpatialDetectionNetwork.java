// Targeted by JavaCPP version 1.5.11: DO NOT EDIT THIS FILE

package org.bytedeco.depthai;

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

import static org.bytedeco.depthai.global.depthai.*;


/**
 * MobileNetSpatialDetectionNetwork node. Mobilenet-SSD based network with spatial location data.
 */
@Namespace("dai::node") @Properties(inherit = org.bytedeco.depthai.presets.depthai.class)
public class MobileNetSpatialDetectionNetwork extends MobileNetSpatialDetectionNetworkPropertiesNode {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public MobileNetSpatialDetectionNetwork(Pointer p) { super(p); }

    public MobileNetSpatialDetectionNetwork(@SharedPtr PipelineImpl par, @Cast("int64_t") long nodeId) { super((Pointer)null); allocate(par, nodeId); }
    private native void allocate(@SharedPtr PipelineImpl par, @Cast("int64_t") long nodeId);
}
