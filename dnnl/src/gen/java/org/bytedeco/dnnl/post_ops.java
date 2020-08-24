// Targeted by JavaCPP version 1.5.4-SNAPSHOT: DO NOT EDIT THIS FILE

package org.bytedeco.dnnl;

import java.nio.*;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

import static org.bytedeco.javacpp.presets.javacpp.*;

import static org.bytedeco.dnnl.global.dnnl.*;

/** \endcond
 <p>
 *  Post-ops.
 * 
 *  Post-ops are computations executed after the main primitive computations
 *  and are attached to the primitive via primitive attributes.
 * 
 *  @see \ref dev_guide_attributes_post_ops
 *  */
@Namespace("dnnl") @Properties(inherit = org.bytedeco.dnnl.presets.dnnl.class)
public class post_ops extends dnnl_post_ops_handle {
    static { Loader.load(); }

    
        public post_ops() { super((Pointer)null); allocate(); }
        private native void allocate();
        public post_ops(@Const @ByRef post_ops arg0) { super((Pointer)null); allocate(arg0); }
        private native void allocate(@Const @ByRef post_ops arg0);
        
        ///
        public post_ops(dnnl_post_ops t, @Cast("bool") boolean weak/*=false*/) { super((Pointer)null); allocate(t, weak); }
        private native void allocate(dnnl_post_ops t, @Cast("bool") boolean weak/*=false*/);
        public post_ops(dnnl_post_ops t) { super((Pointer)null); allocate(t); }
        private native void allocate(dnnl_post_ops t);
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public post_ops(Pointer p) { super(p); }
    /** Native array allocator. Access with {@link Pointer#position(long)}. */
    public post_ops(long size) { super((Pointer)null); allocateArray(size); }
    private native void allocateArray(long size);
    @Override public post_ops position(long position) {
        return (post_ops)super.position(position);
    }
    @Override public post_ops getPointer(long i) {
        return new post_ops(this).position(position + i);
    }


    /** Constructs an empty sequence of post-ops. */

    /** Returns the number of post-ops entries. */
    public native int len();

    /** Returns the primitive kind of post-op at entry with a certain index.
     *  @param index Index of the post-op to return the kind for.
     *  @return Primitive kind of the post-op at the specified index. */
    
    ///
    ///
    ///
    ///
    ///
    ///
    ///
    public native primitive.kind kind(int index);

    /** Appends an accumulation (sum) post-op. Prior to accumulating the
     *  result, the previous value would be multiplied by a scaling factor
     *  \p scale.
     * 
     *  The kind of this post-op is #dnnl::primitive::kind::sum.
     * 
     *  This feature may improve performance for cases like residual learning
     *  blocks, where the result of convolution is accumulated to the
     *  previously computed activations. The parameter \p scale may be used
     *  for the integer-based computations when the result and previous
     *  activations have different logical scaling factors.
     * 
     *  In the simplest case when the accumulation is the only post-op,
     *  the computations would be {@code dst[:] := scale * dst[:] + op(...)}
     *  instead of {@code dst[:] := op(...)}.
     * 
     *  If \p data_type is specified, original dst tensor will be reinterpreted
     *  as a tensor with provided data type. Since it is reinterpretation,
     *  data_type and dst data type should have same size.
     *  As a result, computations would be:
     * 
     *      dst[:] <- scale * as_data_type(dst[:]) + op(...)
     *                                         // instead of dst[:] <- op(...)
     * 
     *  \note
     *      This post-op executes in-place and does not change the
     *      destination layout.
     * 
     *  @param scale Scaling factor.
     *  @param data_type Data type. */
    
    ///
    public native void append_sum(float scale/*=1.f*/,
                memory.data_type data_type/*=dnnl::memory::data_type::undef*/);
    public native void append_sum();

    /** Returns the parameters of an accumulation (sum) post-op.
     * 
     *  @param index Index of the sum post-op.
     *  @param scale Scaling factor of the sum post-op. */
    
    ///
    public native void get_params_sum(int index, @ByRef FloatPointer scale);
    public native void get_params_sum(int index, @ByRef FloatBuffer scale);
    public native void get_params_sum(int index, @ByRef float[] scale);

    /** Returns the parameters of an accumulation (sum) post-op.
     * 
     *  @param index Index of the sum post-op.
     *  @param scale Scaling factor of the sum post-op.
     *  @param data_type Data type of the sum post-op. */
    
    ///
    ///
    ///
    public native void get_params_sum(
                int index, @ByRef FloatPointer scale, memory.data_type data_type);
    public native void get_params_sum(
                int index, @ByRef FloatBuffer scale, memory.data_type data_type);
    public native void get_params_sum(
                int index, @ByRef float[] scale, memory.data_type data_type);

    /** Appends an elementwise post-op.
     * 
     *  The kind of this post-op is #dnnl::primitive::kind::eltwise.
     * 
     *  In the simplest case when the elementwise is the only post-op, the
     *  computations would be {@code dst[:] := scale * eltwise_op (op(...))} instead
     *  of {@code dst[:] <- op(...)}, where eltwise_op is configured with the given
     *  parameters.
     * 
     *  @param scale Scaling factor.
     *  @param aalgorithm Elementwise algorithm.
     *  @param alpha Alpha parameter for the elementwise algorithm.
     *  @param beta Beta parameter for the elementwise algorithm. */
    
    ///
    public native void append_eltwise(
                float scale, algorithm aalgorithm, float alpha, float beta);
    public native void append_eltwise(
                float scale, @Cast("dnnl::algorithm") int aalgorithm, float alpha, float beta);

    /** Returns parameters of an elementwise post-up.
     * 
     *  @param index Index of the post-op.
     *  @param scale Output scaling factor.
     *  @param aalgorithm Output elementwise algorithm kind.
     *  @param alpha Output alpha parameter for the elementwise algorithm.
     *  @param beta Output beta parameter for the elementwise algorithm. */
    
    ///
    ///
    ///
    ///
    ///
    ///
    ///
    public native void get_params_eltwise(int index, @ByRef FloatPointer scale, @ByRef @Cast("dnnl::algorithm*") IntPointer aalgorithm,
                @ByRef FloatPointer alpha, @ByRef FloatPointer beta);
    public native void get_params_eltwise(int index, @ByRef FloatBuffer scale, @ByRef @Cast("dnnl::algorithm*") IntBuffer aalgorithm,
                @ByRef FloatBuffer alpha, @ByRef FloatBuffer beta);
    public native void get_params_eltwise(int index, @ByRef float[] scale, @ByRef @Cast("dnnl::algorithm*") int[] aalgorithm,
                @ByRef float[] alpha, @ByRef float[] beta);

    /** Appends a depthwise post-op convolution with stride 1.
     * 
     *  This post-op can only be fused with a 2D 1x1 convolution (convolution
     *  with weights spatial dimension equal to 1 i.e., kh=kw=1).
     * 
     *  The kind of this post-op is #dnnl_convolution.
     * 
     *  The number of outputs for primitive remain same as before. The output
     *  size remain same as the original primitive due to stride=1.
     * 
     *  The Post-op can be defined as:
     * 
     *       dst[:] <- scales * (conv_dw(conv_1x1))
     * 
     *  See \ref dev_guide_attributes_post_ops_depthwise and
     *  \ref dev_guide_attributes_post_ops_depthwise_fusion for more info.
     * 
     *  @param weights_data_type Weights data type of depthwise post-op
     *  @param bias_data_type Bias data type of depthwise post-op
     *  @param dst_data_type Output data type of depthwise post-op
     *  @param mask Output scaling factors correspondence mask that defines the
     *      correspondence between the output tensor dimensions and the
     *      \p scales array. The set i-th bit indicates that a dedicated output
     *      scaling factor is used for each index along that dimension. The mask
     *      value of 0 implies a common scaling factor for the whole output
     *      tensor.
     *  @param scales Output pointer to a constant array of float scaling
     *      factors. */
    
    ///
    public native void append_dw_k3s1p1(memory.data_type weights_data_type,
                memory.data_type bias_data_type, memory.data_type dst_data_type,
                int mask, @StdVector FloatPointer scales);
    public native void append_dw_k3s1p1(memory.data_type weights_data_type,
                memory.data_type bias_data_type, memory.data_type dst_data_type,
                int mask, @StdVector FloatBuffer scales);
    public native void append_dw_k3s1p1(memory.data_type weights_data_type,
                memory.data_type bias_data_type, memory.data_type dst_data_type,
                int mask, @StdVector float[] scales);

    /** Returns the parameters of an depthwise post-op with stride 1.
     * 
     *  @param index Index of the elementwise post-op.
     *  @param weights_data_type Weights data type of depthwise post-op
     *  @param bias_data_type Bias data type of depthwise post-op
     *  @param dst_data_type Output data type of depthwise post-op
     *  @param mask Output scaling factors correspondence mask that defines the
     *      correspondence between the output tensor dimensions and the
     *      \p scales array. The set i-th bit indicates that a dedicated output
     *      scaling factor is used for each index along that dimension. The mask
     *      value of 0 implies a common scaling factor for the whole output
     *      tensor.
     *  @param scales Output pointer to a constant array of float scaling
     *      factors. */
    
    ///
    ///
    ///
    ///
    ///
    ///
    ///
    ///
    public native void get_params_dw_k3s1p1(int index, memory.data_type weights_data_type,
                memory.data_type bias_data_type, memory.data_type dst_data_type,
                @ByRef IntPointer mask, @StdVector FloatPointer scales);
    public native void get_params_dw_k3s1p1(int index, memory.data_type weights_data_type,
                memory.data_type bias_data_type, memory.data_type dst_data_type,
                @ByRef IntBuffer mask, @StdVector FloatBuffer scales);
    public native void get_params_dw_k3s1p1(int index, memory.data_type weights_data_type,
                memory.data_type bias_data_type, memory.data_type dst_data_type,
                @ByRef int[] mask, @StdVector float[] scales);

    /** Appends a depthwise post-op convolution with stride 2.
     * 
     *  This post-op can only be fused with a 2D 1x1 convolution (convolution
     *  with weights spatial dimension equal to 1 i.e., kh=kw=1).
     * 
     *  The kind of this post-op is #dnnl_convolution.
     * 
     *  The number of outputs for primitive remain same as before. The output
     *  spatial size can be derived as below:
     * 
     *  output_height = ceil(output_height_1x1_convolution, stride)
     *  output_width = ceil(output_width_1x1_convolution, stride)
     * 
     *  The Post-op can be defined as:
     * 
     *       dst[:] <- scales * (conv_dw(conv_1x1))
     * 
     *  See \ref dev_guide_attributes_post_ops_depthwise and
     *  \ref dev_guide_attributes_post_ops_depthwise_fusion for more info.
     * 
     *  @param weights_data_type Weights data type of depthwise post-op
     *  @param bias_data_type Bias data type of depthwise post-op
     *  @param dst_data_type Output data type of depthwise post-op
     *  @param mask Output scaling factors correspondence mask that defines the
     *      correspondence between the output tensor dimensions and the
     *      \p scales array. The set i-th bit indicates that a dedicated output
     *      scaling factor is used for each index along that dimension. The mask
     *      value of 0 implies a common scaling factor for the whole output
     *      tensor.
     *  @param scales Output pointer to a constant array of float scaling
     *      factors.
     *  @return #dnnl_success on success and a status describing the error
     *      otherwise */
    
    ///
    public native void append_dw_k3s2p1(memory.data_type weights_data_type,
                memory.data_type bias_data_type, memory.data_type dst_data_type,
                int mask, @StdVector FloatPointer scales);
    public native void append_dw_k3s2p1(memory.data_type weights_data_type,
                memory.data_type bias_data_type, memory.data_type dst_data_type,
                int mask, @StdVector FloatBuffer scales);
    public native void append_dw_k3s2p1(memory.data_type weights_data_type,
                memory.data_type bias_data_type, memory.data_type dst_data_type,
                int mask, @StdVector float[] scales);

    /** Returns the parameters of an depthwise post-op with stride 2.
     * 
     *  @param index Index of the elementwise post-op.
     *  @param weights_data_type Weights data type of depthwise post-op
     *  @param bias_data_type Bias data type of depthwise post-op
     *  @param dst_data_type Output data type of depthwise post-op
     *  @param mask Output scaling factors correspondence mask that defines the
     *      correspondence between the output tensor dimensions and the
     *      \p scales array. The set i-th bit indicates that a dedicated output
     *      scaling factor is used for each index along that dimension. The mask
     *      value of 0 implies a common scaling factor for the whole output
     *      tensor.
     *  @param scales Output pointer to a constant array of float scaling
     *      factors. */
    public native void get_params_dw_k3s2p1(int index, memory.data_type weights_data_type,
                memory.data_type bias_data_type, memory.data_type dst_data_type,
                @ByRef IntPointer mask, @StdVector FloatPointer scales);
    public native void get_params_dw_k3s2p1(int index, memory.data_type weights_data_type,
                memory.data_type bias_data_type, memory.data_type dst_data_type,
                @ByRef IntBuffer mask, @StdVector FloatBuffer scales);
    public native void get_params_dw_k3s2p1(int index, memory.data_type weights_data_type,
                memory.data_type bias_data_type, memory.data_type dst_data_type,
                @ByRef int[] mask, @StdVector float[] scales);
}