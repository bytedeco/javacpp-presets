// Targeted by JavaCPP version 1.5.12: DO NOT EDIT THIS FILE

package org.bytedeco.tensorrt.nvinfer;

import java.nio.*;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

import static org.bytedeco.javacpp.presets.javacpp.*;
import org.bytedeco.cuda.cudart.*;
import static org.bytedeco.cuda.global.cudart.*;
import org.bytedeco.cuda.cublas.*;
import static org.bytedeco.cuda.global.cublas.*;
import org.bytedeco.cuda.cudnn.*;
import static org.bytedeco.cuda.global.cudnn.*;
import org.bytedeco.cuda.nvrtc.*;
import static org.bytedeco.cuda.global.nvrtc.*;

import static org.bytedeco.tensorrt.global.nvinfer.*;


/**
 *  \class IPluginV2IOExt
 * 
 *  \brief Plugin class for user-implemented layers.
 * 
 *  Plugins are a mechanism for applications to implement custom layers. This interface provides additional
 *  capabilities to the IPluginV2Ext interface by extending different I/O data types and tensor formats.
 * 
 *  @see IPluginV2Ext
 * 
 *  @deprecated Deprecated in TensorRT 10.0. Implement IPluginV3 instead.
 *  */
@Namespace("nvinfer1") @Properties(inherit = org.bytedeco.tensorrt.presets.nvinfer.class)
public class IPluginV2IOExt extends IPluginV2Ext {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public IPluginV2IOExt(Pointer p) { super(p); }

    /**
     *  \brief Configure the layer.
     * 
     *  This function is called by the builder prior to initialize(). It provides an opportunity for the layer to make
     *  algorithm choices on the basis of the provided I/O PluginTensorDesc.
     * 
     *  @param in The input tensors attributes that are used for configuration.
     *  @param nbInput Number of input tensors.
     *  @param out The output tensors attributes that are used for configuration.
     *  @param nbOutput Number of output tensors.
     * 
     *  \u005Cusage
     *  - Allowed context for the API call
     *    - Thread-safe: Yes, this method is required to be thread-safe and may be called from multiple threads
     *                   when building networks on multiple devices sharing the same plugin. However, TensorRT
     *                   will not call this method from two threads simultaneously on a given clone of a plugin.
     *  */
    
    
    //!
    //!
    //!
    //!
    //!
    //!
    //!
    //!
    //!
    //!
    //!
    //!
    //!
    public native @NoException(true) void configurePlugin(
            @Const PluginTensorDesc in, int nbInput, @Const PluginTensorDesc out, int nbOutput);

    /**
     *  \brief Return true if plugin supports the format and datatype for the input/output indexed by pos.
     * 
     *  For this method inputs are numbered 0..(nbInputs-1) and outputs are numbered nbInputs..(nbInputs+nbOutputs-1).
     *  Using this numbering, pos is an index into InOut, where 0 <= pos < nbInputs+nbOutputs.
     * 
     *  TensorRT invokes this method to ask if the input/output indexed by pos supports the format/datatype specified
     *  by inOut[pos].format and inOut[pos].type. The override must return true if that format/datatype at inOut[pos]
     *  are supported by the plugin. If support is conditional on other input/output formats/datatypes, the plugin can
     *  make its result conditional on the formats/datatypes in inOut[0..pos-1], which will be set to values
     *  that the plugin supports. The override must not inspect inOut[pos+1..nbInputs+nbOutputs-1],
     *  which will have invalid values.  In other words, the decision for pos must be based on inOut[0..pos] only.
     * 
     *  Some examples:
     * 
     *  * A definition for a plugin that supports only FP16 NCHW:
     * 
     *          return inOut.format[pos] == TensorFormat::kLINEAR && inOut.type[pos] == DataType::kHALF;
     * 
     *  * A definition for a plugin that supports only FP16 NCHW for its two inputs,
     *    and FP32 NCHW for its single output:
     * 
     *          return inOut.format[pos] == TensorFormat::kLINEAR &&
     *                 (inOut.type[pos] == (pos < 2 ?  DataType::kHALF : DataType::kFLOAT));
     * 
     *  * A definition for a "polymorphic" plugin with two inputs and one output that supports
     *    any format or type, but the inputs and output must have the same format and type:
     * 
     *          return pos == 0 || (inOut.format[pos] == inOut.format[0] && inOut.type[pos] == inOut.type[0]);
     * 
     *  Warning: TensorRT will stop asking for formats once it finds kFORMAT_COMBINATION_LIMIT on combinations.
     * 
     *  \u005Cusage
     *  - Allowed context for the API call
     *    - Thread-safe: Yes, this method is required to be thread-safe and may be called from multiple threads
     *                   when building networks on multiple devices sharing the same plugin.
     *  */
    public native @Cast("bool") @NoException(true) boolean supportsFormatCombination(
            int pos, @Const PluginTensorDesc inOut, int nbInputs, int nbOutputs);

    // @cond SuppressDoxyWarnings
}
