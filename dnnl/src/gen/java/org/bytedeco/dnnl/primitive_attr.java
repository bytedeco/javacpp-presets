// Targeted by JavaCPP version 1.5.12: DO NOT EDIT THIS FILE

package org.bytedeco.dnnl;

import java.nio.*;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

import static org.bytedeco.javacpp.presets.javacpp.*;
import org.bytedeco.opencl.*;
import static org.bytedeco.opencl.global.OpenCL.*;

import static org.bytedeco.dnnl.global.dnnl.*;

/** \endcond
 <p>
 *  Primitive attributes.
 * 
 *  @see \ref dev_guide_attributes */
@Namespace("dnnl") @Properties(inherit = org.bytedeco.dnnl.presets.dnnl.class)
public class primitive_attr extends dnnl_primitive_attr_handle {
    static { Loader.load(); }

    
        public primitive_attr() { super((Pointer)null); allocate(); }
        private native void allocate();
        public primitive_attr(@Const @ByRef primitive_attr arg0) { super((Pointer)null); allocate(arg0); }
        private native void allocate(@Const @ByRef primitive_attr arg0);
        
        ///
        public primitive_attr(dnnl_primitive_attr t, @Cast("bool") boolean weak/*=false*/) { super((Pointer)null); allocate(t, weak); }
        private native void allocate(dnnl_primitive_attr t, @Cast("bool") boolean weak/*=false*/);
        public primitive_attr(dnnl_primitive_attr t) { super((Pointer)null); allocate(t); }
        private native void allocate(dnnl_primitive_attr t);
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public primitive_attr(Pointer p) { super(p); }
    /** Native array allocator. Access with {@link Pointer#position(long)}. */
    public primitive_attr(long size) { super((Pointer)null); allocateArray(size); }
    private native void allocateArray(long size);
    @Override public primitive_attr position(long position) {
        return (primitive_attr)super.position(position);
    }
    @Override public primitive_attr getPointer(long i) {
        return new primitive_attr((Pointer)this).offsetAddress(i);
    }


    /** Constructs default (empty) primitive attributes. */

    /** Creates primitive attributes from a C API ::dnnl_primitive_attr_t
     *  handle. The resulting handle is not weak and the C handle will be
     *  destroyed during the destruction of the C++ object.
     * 
     *  @param attr The C API primitive attributes. */
    

    /** Returns the parameters of a dropout attribute.
     * 
     *  @param mask_desc Output memory descriptor of a dropout mask. */
    
    ///
    public native void get_dropout(@ByRef org.bytedeco.dnnl.memory.desc mask_desc);

    /** Sets dropout probability.
     * 
     *  @param mask_desc Output memory descriptor of a dropout mask. */
    public native void set_dropout(@Const @ByRef org.bytedeco.dnnl.memory.desc mask_desc);

    /** Returns the fpmath mode */
    
    ///
    public native fpmath_mode get_fpmath_mode();

    /** Returns the fpmath mode
     * 
     *  @param mode Specified fpmath mode.
     *  @param apply_to_int Use floating-point arithmetic for integer primitives. */
    
    ///
    public native void get_fpmath_mode(@ByRef @Cast("dnnl::fpmath_mode*") IntPointer mode, @Cast("bool*") @ByRef BoolPointer apply_to_int);
    public native void get_fpmath_mode(@ByRef @Cast("dnnl::fpmath_mode*") IntBuffer mode, @Cast("bool*") @ByRef boolean[] apply_to_int);
    public native void get_fpmath_mode(@ByRef @Cast("dnnl::fpmath_mode*") int[] mode, @Cast("bool*") @ByRef BoolPointer apply_to_int);
    public native void get_fpmath_mode(@ByRef @Cast("dnnl::fpmath_mode*") IntPointer mode, @Cast("bool*") @ByRef boolean[] apply_to_int);
    public native void get_fpmath_mode(@ByRef @Cast("dnnl::fpmath_mode*") IntBuffer mode, @Cast("bool*") @ByRef BoolPointer apply_to_int);
    public native void get_fpmath_mode(@ByRef @Cast("dnnl::fpmath_mode*") int[] mode, @Cast("bool*") @ByRef boolean[] apply_to_int);

    /** Sets fpmath mode.
     * 
     *  @param mode Specified fpmath mode.
     *  @param apply_to_int Boolean. Use of floating-point arithmetic for integer primitives. */
    public native void set_fpmath_mode(fpmath_mode mode, @Cast("bool") boolean apply_to_int/*=false*/);
    public native void set_fpmath_mode(fpmath_mode mode);
    public native void set_fpmath_mode(@Cast("dnnl::fpmath_mode") int mode, @Cast("bool") boolean apply_to_int/*=false*/);
    public native void set_fpmath_mode(@Cast("dnnl::fpmath_mode") int mode);

    /** Returns the accumulation mode */
    
    ///
    public native accumulation_mode get_accumulation_mode();

    /** Sets accumulation mode.
     * 
     *  @param mode Specified accumulation mode. */
    public native void set_accumulation_mode(accumulation_mode mode);
    public native void set_accumulation_mode(@Cast("dnnl::accumulation_mode") int mode);

    /** Returns the deterministic attribute value */
    
    ///
    public native @Cast("bool") boolean get_deterministic();

    /** Sets deterministic attribute value
     * 
     *  @param value Specified deterministic mode. */
    
    ///
    public native void set_deterministic(@Cast("bool") boolean value);

    /** Returns the rounding mode attribute value
     * 
     *  @param arg Argument for which rounding mode query applies.
     *  @return The rounding mode applied to the specified argument. */
    
    ///
    public native rounding_mode get_rounding_mode(int arg);

    /** Sets the rounding mode attribute value for a given argument
     * 
     *  @param arg Argument for which to set rounding mode.
     *  @param mode Rounding mode to apply. */
    public native void set_rounding_mode(int arg, rounding_mode mode);
    public native void set_rounding_mode(int arg, @Cast("dnnl::rounding_mode") int mode);

    /** Returns the scratchpad mode. */
    
    ///
    public native scratchpad_mode get_scratchpad_mode();

    /** Sets scratchpad mode.
     * 
     *  @param mode Specified scratchpad mode. */
    
    ///
    ///
    public native void set_scratchpad_mode(scratchpad_mode mode);
    public native void set_scratchpad_mode(@Cast("dnnl::scratchpad_mode") int mode);

    /** Sets scaling factors for primitive operations for a given memory
     *  argument. The scaling factors must be passed at execution time
     *  as an argument with index #DNNL_ARG_ATTR_SCALES | arg.
     * 
     *  @see dnnl_primitive_attr_set_scales_mask
     * 
     *  @param arg Parameter argument index as passed to the
     *      primitive::execute() call.
     *  @param mask Scaling factors correspondence mask that defines the
     *      correspondence between the tensor dimensions and the \p scales
     *      vector. The set i-th bit indicates that a dedicated scaling factor
     *      is used for each index along that dimension. Set the mask to 0 to
     *      use a common scaling factor for the whole output tensor. */
    
    ///
    ///
    public native void set_scales_mask(int arg, int mask);

    /** Sets scaling factors for primitive operations for a given memory
     *  argument. The scaling factors must be passed at execution time
     *  as an argument with index #DNNL_ARG_ATTR_SCALES | arg.
     * 
     *  @see dnnl_primitive_attr_set_scales
     * 
     *  @param arg Parameter argument index as passed to the
     *      primitive::execute() call.
     *  @param mask Scales correspondence mask that defines the
     *      correspondence between the tensor dimensions and the \p
     *      scales vector. The set i-th bit indicates that a dedicated
     *      scale is used for each index along that dimension. Set the
     *      mask to 0 to use a common scale for the whole output tensor.
     *  @param groups Scaling factors correspondence groups that define the
     *      correspondence between the tensor dimensions and the scales array.
     *      The set i-th dimension indicates a number of groups of scaling
     *      factors used for that logical dimension in a memory indicated by \p arg.
     *  @param data_type Scaling factors data_type. */
    
    ///
    ///
    public native void set_scales(int arg, int mask, @Const @Cast({"dnnl_dim_t*", "std::vector<dnnl_dim_t>&"}) @StdVector("dnnl_dim_t") @ByRef LongPointer groups,
                memory.data_type data_type/*=dnnl::memory::data_type::f32*/);
    public native void set_scales(int arg, int mask, @Const @Cast({"dnnl_dim_t*", "std::vector<dnnl_dim_t>&"}) @StdVector("dnnl_dim_t") @ByRef LongPointer groups);
    public native void set_scales(int arg, int mask, @Const @Cast({"dnnl_dim_t*", "std::vector<dnnl_dim_t>&"}) @StdVector("dnnl_dim_t") @ByRef LongBuffer groups,
                memory.data_type data_type/*=dnnl::memory::data_type::f32*/);
    public native void set_scales(int arg, int mask, @Const @Cast({"dnnl_dim_t*", "std::vector<dnnl_dim_t>&"}) @StdVector("dnnl_dim_t") @ByRef LongBuffer groups);
    public native void set_scales(int arg, int mask, @Const @Cast({"dnnl_dim_t*", "std::vector<dnnl_dim_t>&"}) @StdVector("dnnl_dim_t") @ByRef long[] groups,
                memory.data_type data_type/*=dnnl::memory::data_type::f32*/);
    public native void set_scales(int arg, int mask, @Const @Cast({"dnnl_dim_t*", "std::vector<dnnl_dim_t>&"}) @StdVector("dnnl_dim_t") @ByRef long[] groups);

    /** Sets zero points for primitive operations for a given memory argument.
     *  The zero points must be passed at execution time as an argument with
     *  index #DNNL_ARG_ATTR_ZERO_POINTS | arg.
     * 
     *  @see dnnl_primitive_attr_set_zero_points_mask
     * 
     *  @param arg Parameter argument index as passed to the
     *      primitive::execute() call.
     *  @param mask Zero point correspondence mask that defines the
     *      correspondence between the tensor dimensions and the \p
     *      zero_points vector. The set i-th bit indicates that a dedicated
     *      zero point is used for each index along that dimension. Set the
     *      mask to 0 to use a common zero point for the whole output tensor. */
    
    ///
    ///
    public native void set_zero_points_mask(int arg, int mask);

    /** Sets zero points for primitive operations for a given memory argument.
     *  The zero points must be passed at execution time as an argument with
     *  index #DNNL_ARG_ATTR_ZERO_POINTS | arg.
     * 
     *  @see dnnl_primitive_attr_set_zero_points
     * 
     *  @param arg Parameter argument index as passed to the
     *      primitive::execute() call.
     *  @param mask Zero point correspondence mask that defines the
     *      correspondence between the tensor dimensions and the \p
     *      zero_points vector. The set i-th bit indicates that a dedicated
     *      zero point is used for each index along that dimension. Set the
     *      mask to 0 to use a common zero point for the whole output tensor.
     *  @param groups Zero point factors correspondence groups that define the
     *      correspondence between the tensor dimensions and the zero_points array.
     *      The set i-th dimension indicates a number of groups of zero point
     *      factors used for that logical dimension in a memory indicated by \p arg.
     *  @param data_type Zero point factors data_type. */
    
    ///
    public native void set_zero_points(int arg, int mask, @Const @Cast({"dnnl_dim_t*", "std::vector<dnnl_dim_t>&"}) @StdVector("dnnl_dim_t") @ByRef LongPointer groups,
                memory.data_type data_type/*=dnnl::memory::data_type::s32*/);
    public native void set_zero_points(int arg, int mask, @Const @Cast({"dnnl_dim_t*", "std::vector<dnnl_dim_t>&"}) @StdVector("dnnl_dim_t") @ByRef LongPointer groups);
    public native void set_zero_points(int arg, int mask, @Const @Cast({"dnnl_dim_t*", "std::vector<dnnl_dim_t>&"}) @StdVector("dnnl_dim_t") @ByRef LongBuffer groups,
                memory.data_type data_type/*=dnnl::memory::data_type::s32*/);
    public native void set_zero_points(int arg, int mask, @Const @Cast({"dnnl_dim_t*", "std::vector<dnnl_dim_t>&"}) @StdVector("dnnl_dim_t") @ByRef LongBuffer groups);
    public native void set_zero_points(int arg, int mask, @Const @Cast({"dnnl_dim_t*", "std::vector<dnnl_dim_t>&"}) @StdVector("dnnl_dim_t") @ByRef long[] groups,
                memory.data_type data_type/*=dnnl::memory::data_type::s32*/);
    public native void set_zero_points(int arg, int mask, @Const @Cast({"dnnl_dim_t*", "std::vector<dnnl_dim_t>&"}) @StdVector("dnnl_dim_t") @ByRef long[] groups);

    /** Returns post-ops previously set via set_post_ops().
     * 
     *  @return Post-ops. */
    
    ///
    ///
    public native @ByVal post_ops get_post_ops();

    /** Sets post-ops.
     * 
     *  \note
     *      There is no way to check whether the post-ops would be supported
     *      by the target primitive. Any error will be reported
     *      by the respective primitive descriptor constructor.
     * 
     *  @param ops Post-ops object to copy post-ops from. */
    
    ///
    ///
    ///
    ///
    ///
    ///
    ///
    ///
    public native void set_post_ops(@Const @ByRef post_ops ops);

    /** Sets quantization scale and shift parameters for RNN data tensors.
     * 
     *  For performance reasons, the low-precision configuration of the RNN
     *  primitives expect input activations to have the unsigned 8-bit integer
     *  data type. The scale and shift parameters are used to quantize
     *  floating-point data to unsigned integer and must be passed to the RNN
     *  primitive using attributes.
     * 
     *  The quantization formula is {@code scale * data + shift}.
     * 
     *  Example usage:
     *  <pre>{@code
     *      // RNN parameters
     *      int l = 2, t = 2, mb = 32, sic = 32, slc = 32, dic = 32, dlc = 32;
     *      // Activations quantization parameters
     *      float scale = 63.f, shift = 64.f;
     * 
     *      primitive_attr attr;
     * 
     *      // Set scale and shift for int8 quantization of activation
     *      attr.set_rnn_data_qparams(scale, shift);
     * 
     *      // Create an RNN primitive descriptor.
     *      vanilla_rnn_forward::primitive_desc rnn_d(
     *              engine, /* arguments * /, attr);
     *  }</pre>
     * 
     *  \note
     *      Quantization scale and shift are common for src_layer, src_iter,
     *      dst_iter, and dst_layer.
     * 
     *  @param scale The value to scale the data by.
     *  @param shift The value to shift the data by. */
    
    ///
    ///
    public native void set_rnn_data_qparams(float scale, float shift);

    /** Returns the quantization scale and shift parameters for RNN data
     *  tensors.
     * 
     *  \note
     *      Quantization scale and shift are common for src_layer, src_iter,
     *      dst_iter, and dst_layer.
     * 
     *  @param scale The value to scale the data by.
     *  @param shift The value to shift the data by. */
    
    ///
    ///
    ///
    public native void get_rnn_data_qparams(@ByRef FloatPointer scale, @ByRef FloatPointer shift);
    public native void get_rnn_data_qparams(@ByRef FloatBuffer scale, @ByRef FloatBuffer shift);
    public native void get_rnn_data_qparams(@ByRef float[] scale, @ByRef float[] shift);

    /** Sets quantization scaling factors for RNN weights tensors. The
     *  low-precision configuration of the RNN primitives expect input weights
     *  to use the signed 8-bit integer data type. The scaling factors are
     *  used to quantize floating-point data to signed integer and must be
     *  passed to RNN primitives using attributes.
     * 
     *  \note
     *      The dimension order is always native and does not depend on the
     *      actual layout used. For example, five-dimensional weights always
     *      have (l, d, i, g, o) logical dimension ordering.
     * 
     *  \note
     *      Quantization scales are common for weights_layer and
     *      weights_iteration
     * 
     *  @param mask Scaling factors correspondence mask that defines the
     *      correspondence between the output tensor dimensions and the \p
     *      scales vector. The set i-th bit indicates that a dedicated scaling
     *      factor should be used each index along that dimension. Set the
     *      mask to 0 to use a common scaling factor for the whole output
     *      tensor.
     *  @param scales Constant vector of output scaling factors. The following
     *      equality must hold:
     *      {@code scales.size() = \prod\limits_{d \in mask} weights.dims[d].}
     *      Violations can only be detected when the attributes are used to
     *      create a primitive descriptor. */
    
    ///
    ///
    public native void set_rnn_weights_qparams(int mask, @StdVector FloatPointer scales);
    public native void set_rnn_weights_qparams(int mask, @StdVector FloatBuffer scales);
    public native void set_rnn_weights_qparams(int mask, @StdVector float[] scales);

    /** Returns the quantization scaling factors for RNN projection weights
     *  tensors.
     * 
     *  \note
     *      The dimension order is always native and does not depend on the
     *      actual layout used. For example, five-dimensional weights always
     *      have (l, d, i, g, o) logical dimension ordering.
     * 
     *  @param mask Scaling factors correspondence mask that defines the
     *      correspondence between the output tensor dimensions and the \p
     *      scales vector. The set i-th bit indicates that a dedicated scaling
     *      factor should be used each index along that dimension. Set the
     *      mask to 0 to use a common scaling factor for the whole output
     *      tensor.
     *  @param scales Constant vector of output scaling factors. The following
     *      equality must hold:
     *      {@code scales.size() = \prod\limits_{d \in mask} weights.dims[d].}
     *      Violations can only be detected when the attributes are used to
     *      create a primitive descriptor. */
    
    ///
    ///
    ///
    public native void get_rnn_weights_qparams(@ByRef IntPointer mask, @StdVector FloatPointer scales);
    public native void get_rnn_weights_qparams(@ByRef IntBuffer mask, @StdVector FloatBuffer scales);
    public native void get_rnn_weights_qparams(@ByRef int[] mask, @StdVector float[] scales);

    /** Sets quantization scaling factors for RNN projection weights tensors. */
    //  The low-precision configuration of the RNN primitives expect input
    //  weights to use the signed 8-bit integer data type. The scaling factors
    //  are used to quantize floating-point data to signed integer and must be
    /** passed to RNN primitives using attributes.
    /**
    /** \note
    /**     The dimension order is always native and does not depend on the
    /**     actual layout used. For example, five-dimensional weights always
    /**     have (l, d, i, g, o) logical dimension ordering.
    /**
    /** \note
    /**     Quantization scales are common for weights_layer and
    /**     weights_iteration
    /**
    /** @param mask Scaling factors correspondence mask that defines the
    /**     correspondence between the output tensor dimensions and the \p
    /**     scales vector. The set i-th bit indicates that a dedicated scaling
    /**     factor should be used each index along that dimension. Set the
    /**     mask to 0 to use a common scaling factor for the whole output
    /**     tensor.
    /** @param scales Constant vector of output scaling factors. The following
    /**     equality must hold:
    /**     {@code scales.size() = \prod\limits_{d \in mask} weights.dims[d].}
    /**     Violations can only be detected when the attributes are used to
    /**     create a primitive descriptor. */
    
    ///
    ///
    public native void set_rnn_weights_projection_qparams(
                int mask, @StdVector FloatPointer scales);
    public native void set_rnn_weights_projection_qparams(
                int mask, @StdVector FloatBuffer scales);
    public native void set_rnn_weights_projection_qparams(
                int mask, @StdVector float[] scales);

    /** Returns the quantization scaling factors for RNN projection weights
     *  tensors.
     * 
     *  \note
     *      The dimension order is always native and does not depend on the
     *      actual layout used. For example, five-dimensional weights always
     *      have (l, d, i, g, o) logical dimension ordering.
     * 
     *  @param mask Scaling factors correspondence mask that defines the
     *      correspondence between the output tensor dimensions and the \p
     *      scales vector. The set i-th bit indicates that a dedicated scaling
     *      factor should be used each index along that dimension. Set the
     *      mask to 0 to use a common scaling factor for the whole output
     *      tensor.
     *  @param scales Constant vector of output scaling factors. The following
     *      equality must hold:
     *      {@code scales.size() = \prod\limits_{d \in mask} weights.dims[d].}
     *      Violations can only be detected when the attributes are used to
     *      create a primitive descriptor. */
    public native void get_rnn_weights_projection_qparams(
                @ByRef IntPointer mask, @StdVector FloatPointer scales);
    public native void get_rnn_weights_projection_qparams(
                @ByRef IntBuffer mask, @StdVector FloatBuffer scales);
    public native void get_rnn_weights_projection_qparams(
                @ByRef int[] mask, @StdVector float[] scales);
}
