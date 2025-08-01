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
 *  \class IPluginV2DynamicExt
 * 
 *  \brief Similar to IPluginV2Ext, but with support for dynamic shapes.
 * 
 *  Clients should override the public methods, including the following inherited methods:
 * 
 *  * virtual int32_t getNbOutputs() const noexcept = 0;
 * 
 *  * virtual DataType getOutputDataType(int32_t index, DataType const* inputTypes,
 *                                       int32_t nbInputs) const noexcept = 0;
 * 
 *  * virtual size_t getSerializationSize() const noexcept = 0;
 * 
 *  * virtual void serialize(void* buffer) const noexcept = 0;
 * 
 *  * virtual void destroy() noexcept = 0;
 * 
 *  * virtual void setPluginNamespace(char const* pluginNamespace) noexcept = 0;
 * 
 *  * virtual char const* getPluginNamespace() const noexcept = 0;
 * 
 *  For weakly typed networks, the inputTypes will always be DataType::kFLOAT or DataType::kINT32,
 *  and the returned type is canonicalized to DataType::kFLOAT if it is DataType::kHALF or DataType:kINT8.
 *  For strongly typed networks, inputTypes are inferred from previous operations, and getOutputDataType
 *  specifies the returned type based on the inputTypes.
 *  Details about the floating-point precision are elicited later by method supportsFormatCombination.
 * 
 *  @deprecated Deprecated in TensorRT 10.0. Please implement IPluginV3 instead.
 *  */
@Namespace("nvinfer1") @NoOffset @Properties(inherit = org.bytedeco.tensorrt.presets.nvinfer.class)
public class IPluginV2DynamicExt extends IPluginV2Ext {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public IPluginV2DynamicExt(Pointer p) { super(p); }

    
    
    //!
    //!
    //!
    //!
    //!
    //!
    //!
    //!
    public native @NoException(true) IPluginV2DynamicExt clone();

    /**
     *  \brief Get expressions for computing dimensions of an output tensor from dimensions of the input tensors.
     * 
     *  @param outputIndex The index of the output tensor
     *  @param inputs Expressions for dimensions of the input tensors
     *  @param nbInputs The number of input tensors
     *  @param exprBuilder Object for generating new expressions
     * 
     *  This function is called by the implementations of IBuilder during analysis of the network.
     * 
     *  Example #1: A plugin has a single output that transposes the last two dimensions of the plugin's single input.
     *  The body of the override of getOutputDimensions can be:
     * 
     *      DimsExprs output(inputs[0]);
     *      std::swap(output.d[output.nbDims-1], output.d[output.nbDims-2]);
     *      return output;
     * 
     *  Example #2: A plugin concatenates its two inputs along the first dimension.
     *  The body of the override of getOutputDimensions can be:
     * 
     *      DimsExprs output(inputs[0]);
     *      output.d[0] = exprBuilder.operation(DimensionOperation::kSUM, *inputs[0].d[0], *inputs[1].d[0]);
     *      return output;
     *  */
    
    
    //!
    //!
    public native @ByVal @NoException(true) DimsExprs getOutputDimensions(
            int outputIndex, @Const DimsExprs inputs, int nbInputs, @ByRef IExprBuilder exprBuilder);

    /**
     *  \brief Limit on number of format combinations accepted.
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
    @MemberGetter public static native int kFORMAT_COMBINATION_LIMIT();
    public static final int kFORMAT_COMBINATION_LIMIT = kFORMAT_COMBINATION_LIMIT();

    /**
     *  \brief Return true if plugin supports the format and datatype for the input/output indexed by pos.
     * 
     *  For this method inputs are numbered 0..(nbInputs-1) and outputs are numbered nbInputs..(nbInputs+nbOutputs-1).
     *  Using this numbering, pos is an index into InOut, where 0 <= pos < nbInputs+nbOutputs.
     * 
     *  TensorRT invokes this method to ask if the input/output indexed by pos supports the format/datatype specified
     *  by inOut[pos].format and inOut[pos].type.  The override should return true if that format/datatype at inOut[pos]
     *  are supported by the plugin.  If support is conditional on other input/output formats/datatypes, the plugin can
     *  make its result conditional on the formats/datatypes in inOut[0..pos-1], which will be set to values
     *  that the plugin supports.  The override should not inspect inOut[pos+1..nbInputs+nbOutputs-1],
     *  which will have invalid values.  In other words, the decision for pos must be based on inOut[0..pos] only.
     * 
     *  Some examples:
     * 
     *  * A definition for a plugin that supports only FP16 NCHW:
     * 
     *          return inOut[pos].format == TensorFormat::kLINEAR && inOut[pos].type == DataType::kHALF;
     * 
     *  * A definition for a plugin that supports only FP16 NCHW for its two inputs,
     *    and FP32 NCHW for its single output:
     * 
     *          return inOut[pos].format == TensorFormat::kLINEAR && (inOut[pos].type == (pos < 2 ? DataType::kHALF :
     *          DataType::kFLOAT));
     * 
     *  * A definition for a "polymorphic" plugin with two inputs and one output that supports
     *    any format or type, but the inputs and output must have the same format and type:
     * 
     *          return pos == 0 || (inOut[pos].format == inOut.format[0] && inOut[pos].type == inOut[0].type);
     * 
     *  Warning: TensorRT will stop asking for formats once it finds kFORMAT_COMBINATION_LIMIT on combinations.
     *  */
    
    
    //!
    //!
    //!
    //!
    //!
    //!
    public native @Cast("bool") @NoException(true) boolean supportsFormatCombination(
            int pos, @Const PluginTensorDesc inOut, int nbInputs, int nbOutputs);

    /**
     *  \brief Configure the plugin.
     * 
     *  configurePlugin() can be called multiple times in both the build and execution phases. The build phase happens
     *  before initialize() is called and only occurs during creation of an engine by IBuilder. The execution phase
     *  happens after initialize() is called and occurs during both creation of an engine by IBuilder and execution
     *  of an engine by IExecutionContext.
     * 
     *  Build phase:
     *  IPluginV2DynamicExt->configurePlugin is called when a plugin is being prepared for profiling but not for any
     *  specific input size. This provides an opportunity for the plugin to make algorithmic choices on the basis of
     *  input and output formats, along with the bound of possible dimensions. The min and max value of the
     *  DynamicPluginTensorDesc correspond to the kMIN and kMAX value of the current profile that the plugin is being
     *  profiled for, with the desc.dims field corresponding to the dimensions of plugin specified at network creation.
     *  Wildcard dimensions will exist during this phase in the desc.dims field.
     * 
     *  Execution phase:
     *  IPluginV2DynamicExt->configurePlugin is called when a plugin is being prepared for executing the plugin for a
     *  specific dimensions. This provides an opportunity for the plugin to change algorithmic choices based on the
     *  explicit input dimensions stored in desc.dims field.
     *   * IBuilder will call this function once per profile, with desc.dims resolved to the values specified by the
     *   kOPT
     *     field of the current profile. Wildcard dimensions will not exist during this phase.
     *   * IExecutionContext will call this during the next subsequent instance enqueue[V2]() or execute[V2]() if:
     *     - The batch size is changed from previous call of execute()/enqueue() if hasImplicitBatchDimension() returns
     *     true.
     *     - The optimization profile is changed via setOptimizationProfileAsync().
     *     - An input execution binding is changed via setInputShape().
     *  \warning The execution phase is timing critical during IExecutionContext but is not part of the timing loop when
     *  called from IBuilder. Performance bottlenecks of configurePlugin won't show up during engine building but will
     *  be visible during execution after calling functions that trigger layer resource updates.
     * 
     *  @param in The input tensors attributes that are used for configuration.
     *  @param nbInputs Number of input tensors.
     *  @param out The output tensors attributes that are used for configuration.
     *  @param nbOutputs Number of output tensors.
     *  */
    
    
    //!
    //!
    //!
    //!
    public native @NoException(true) void configurePlugin(@Const DynamicPluginTensorDesc in, int nbInputs,
            @Const DynamicPluginTensorDesc out, int nbOutputs);

    /**
     *  \brief Find the workspace size required by the layer.
     * 
     *  This function is called after the plugin is configured, and possibly during execution.
     *  The result should be a sufficient workspace size to deal with inputs and outputs of the given size
     *  or any smaller problem.
     * 
     *  @return The workspace size.
     *  */
    
    
    //!
    //!
    //!
    //!
    public native @Cast("size_t") @NoException(true) long getWorkspaceSize(@Const PluginTensorDesc inputs, int nbInputs, @Const PluginTensorDesc outputs,
            int nbOutputs);

    /**
     *  \brief Execute the layer.
     * 
     *  @param inputDesc how to interpret the memory for the input tensors.
     *  @param outputDesc how to interpret the memory for the output tensors.
     *  @param inputs The memory for the input tensors.
     *  @param outputs The memory for the output tensors.
     *  @param workspace Workspace for execution.
     *  @param stream The stream in which to execute the kernels.
     * 
     *  @return 0 for success, else non-zero (which will cause engine termination).
     *  */
    public native @NoException(true) int enqueue(@Const PluginTensorDesc inputDesc, @Const PluginTensorDesc outputDesc,
            @Cast("const void*const*") PointerPointer inputs, @Cast("void*const*") PointerPointer outputs, Pointer workspace, CUstream_st stream);
    public native @NoException(true) int enqueue(@Const PluginTensorDesc inputDesc, @Const PluginTensorDesc outputDesc,
            @Cast("const void*const*") @ByPtrPtr Pointer inputs, @Cast("void*const*") @ByPtrPtr Pointer outputs, Pointer workspace, CUstream_st stream);
}
