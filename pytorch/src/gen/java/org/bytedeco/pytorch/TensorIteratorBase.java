// Targeted by JavaCPP version 1.5.12: DO NOT EDIT THIS FILE

package org.bytedeco.pytorch;

import org.bytedeco.pytorch.Allocator;
import org.bytedeco.pytorch.Function;
import org.bytedeco.pytorch.Module;
import org.bytedeco.javacpp.annotation.Cast;
import java.nio.*;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

import static org.bytedeco.javacpp.presets.javacpp.*;
import static org.bytedeco.openblas.global.openblas_nolapack.*;
import static org.bytedeco.openblas.global.openblas.*;
import org.bytedeco.javacpp.chrono.*;
import static org.bytedeco.javacpp.global.chrono.*;

import static org.bytedeco.pytorch.global.torch.*;


@Namespace("at") @NoOffset @Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class TensorIteratorBase extends MetaBase {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public TensorIteratorBase(Pointer p) { super(p); }


  public native void build(@ByRef TensorIteratorConfig arg0);

  // The inner-loop function operates on the fastest moving dimension. It
  // implements element-wise operations in terms of 1-d strided tensors.
  //
  // Arguments:
  //  data: data pointers for each operand (length `ntensors`)
  //  strides: stride for each operand (length `ntensors`)
  //  size: size of inner loop
  //
  // The `size` often matches shape[0], but may be smaller due to
  // parallelization of the inner loop.

  

  public native int ndim();
  public native @ByVal LongArrayRef shape();
  public native @Cast("int64_t") long numel();
  public native int ntensors();
  public native int noutputs();
  public native int ninputs();
  public native @ByVal LongArrayRef view_offsets();

  /** number of elements in the output operand. this is the same as numel() for
   *  operations that are not reductions. */
  public native @Cast("int64_t") long num_output_elements();

  /** number of reduced dimensions in a reduction operation */
  public native int num_reduce_dims();

  /** 1-dimensional iteration and no buffering or type conversion */
  public native @Cast("bool") boolean is_trivial_1d();
  /** Reducible to 1-dimensional and all operands are contiguous */
  public native @Cast("bool") boolean is_contiguous();
  public native @Cast("bool") boolean is_dim_reduced(int dim);

  /** Accessors for each operand */
  public native @ByVal LongArrayRef strides(@Cast("int64_t") long arg);
  public native Pointer data_ptr(@Cast("int64_t") long arg);
  public native ScalarType dtype(@Cast("int64_t") long arg/*=0*/);
  public native ScalarType dtype();
  public native ScalarType common_dtype();
  public native ScalarType input_dtype(@Cast("int64_t") long arg/*=0*/);
  public native ScalarType input_dtype();
  public native @ByVal Device device(@Cast("int64_t") long arg/*=0*/);
  public native @ByVal Device device();
  public native DeviceType device_type(@Cast("int64_t") long arg/*=0*/);
  public native DeviceType device_type();
  public native @Cast("int64_t") long element_size(@Cast("int64_t") long arg);
  public native @Cast("bool") boolean is_scalar(@Cast("int64_t") long arg);
  public native @Cast("bool") boolean is_cpu_scalar(@Cast("int64_t") long arg);

  public native @Const @ByRef TensorBase tensor_base(@Cast("int64_t") long arg);
  public native @Const @ByRef Tensor tensor(@Cast("int64_t") long arg);

  public native @Const @ByRef TensorBase output_base(@Cast("int64_t") long arg/*=0*/);
  public native @Const @ByRef TensorBase output_base();

  public native @Const @ByRef Tensor output(@Cast("int64_t") long arg/*=0*/);
  public native @Const @ByRef Tensor output();

  public native @Const @ByRef TensorBase input_base(@Cast("int64_t") long arg/*=0*/);
  public native @Const @ByRef TensorBase input_base();
  public native @Const @ByRef Tensor input(@Cast("int64_t") long arg/*=0*/);
  public native @Const @ByRef Tensor input();

  // Copies from temporary outputs back to the original outputs
  // NOTE: only used on CPU
  public native void cast_outputs();

  /** Removes an operand from this iterator */
  public native void remove_operand(@Cast("int64_t") long arg);
  /** Shrinks an iterated dimension */
  public native void narrow(int dim, @Cast("int64_t") long start, @Cast("int64_t") long size);
  /** Narrows every dim after and including {@code start_dim} to size one. */
  public native void select_all_keeping_dim(int start_dim, @ByVal LongArrayRef starts);
  public native void select_all_keeping_dim(int start_dim, @ByVal @Cast({"int64_t*", "c10::ArrayRef<int64_t>", "std::vector<int64_t>&"}) @StdVector("int64_t") long... starts);
  /** Replaces the data pointer for the operand at index {@code arg}.
   *  The new pointer should have the same sizes, strides and dtype as the
   *  original */
  public native void unsafe_replace_operand(@Cast("int64_t") long arg, Pointer data);

  /** Splits this TensorIterator into two iterators. Together they iterate over
   *  the entire operation. Used by {@code with_32bit_indexing()}. */
  public native @UniquePtr @ByVal TensorIterator split(int dim);

  /** Returns the dimension with the largest extent: (size[dim]-1) * stride[dim] */
  public native int get_dim_to_split();

  /** Return scalar value from original_tensor_base if it is defined. When
   *  common_dtype is Half, casting scalar input to common_dtype might overflow.
   *  If the scalar is aleady given in the type of Half, then return scalar
   *  value from tensor_base. */
  

  

  

  

  

  /** Create a strides array for a Tensor with shape of this iterator. The
   *  parameter {@code element_size} specifies the size of Tensor's data type in
   *  bytes (e.g. {@code 4} for {@code float}) */
  public native @ByVal @Cast("at::TensorIteratorBase::StrideVector*") SymDimVector compatible_stride(@Cast("int64_t") long element_size);

  /** Inverts the re-ordering done by reorder_dimensions. This can only be
   *  called *before* coalesce_dimensions() is called. */
  public native @ByVal DimVector invert_perm(@ByVal LongArrayRef input);
  public native @ByVal DimVector invert_perm(@ByVal @Cast({"int64_t*", "c10::ArrayRef<int64_t>", "std::vector<int64_t>&"}) @StdVector("int64_t") long... input);

  /** Reapply same re-ordering as it is done by reorder_dimensions. This can
   *  only be called *before* coalesce_dimensions() is called. */
  

  /** Helper functions for CPU iteration */
  public native @ByVal @Cast("at::TensorIteratorBase::StrideVector*") SymDimVector get_dim_strides(int dim);
  public native @ByVal @Cast("at::TensorIteratorBase::StrideVector*") SymDimVector get_strides();
  public native @ByVal @Cast("at::TensorIteratorBase::StrideVector*") SymDimVector get_inner_strides();
  public native @ByVal @Cast("at::TensorIteratorBase::PtrVector*") SymDimVector get_base_ptrs();

  // Helper functions for advanced stride manipulations (e.g. torch.flip)
  public native void _unsafe_set_arg_strides(@Cast("const int64_t") long arg, @ByVal LongArrayRef strides);
  public native void _unsafe_set_arg_strides(@Cast("const int64_t") long arg, @ByVal @Cast({"int64_t*", "c10::ArrayRef<int64_t>", "std::vector<int64_t>&"}) @StdVector("int64_t") long... strides);
  public native void _unsafe_set_arg_data(@Cast("const int64_t") long arg, Pointer data);

  // Helper functions for custom device, custom device can get OperandInfo and
  // NameVector in their side.
  public native @ByRef OperandInfo operand(int arg/*=0*/);
  public native @ByRef OperandInfo operand();
  public native @Cast("at::NameVector*") @ByRef SymDimVector get_dim_names();

  /** true if the stride computation can use 32-bit arithmetic. Used by GPU
   *  kernels */
  public native @Cast("bool") boolean can_use_32bit_indexing();

  /** An "iteratable" object that recursively splits this iterator into
   *  sub-iterators that can use 32-bit indexing. */
  public native @ByVal SplitUntil32Bit with_32bit_indexing();

  /** If the kernel should accumulate into the output. Only relevant for CUDA
   *  reductions. */
  public native @Cast("bool") boolean should_accumulate();

  /** Whether this iterator produces the actual output,
   *  as opposed to something that will be accumulated further. Only relevant
   *  for CUDA reductions. */
  public native @Cast("bool") boolean is_final_output();

  public native @Cast("bool") boolean has_contiguous_first_dim();

  public native void set_output_raw_strided(
        @Cast("int64_t") long output_idx,
        @ByVal LongArrayRef sizes,
        @ByVal LongArrayRef strides,
        @ByVal TensorOptions options,
        @ByVal DimnameArrayRef names);
  public native void set_output_raw_strided(
        @Cast("int64_t") long output_idx,
        @ByVal @Cast({"int64_t*", "c10::ArrayRef<int64_t>", "std::vector<int64_t>&"}) @StdVector("int64_t") long[] sizes,
        @ByVal @Cast({"int64_t*", "c10::ArrayRef<int64_t>", "std::vector<int64_t>&"}) @StdVector("int64_t") long[] strides,
        @ByVal TensorOptions options,
        @ByVal DimnameVector names);

// #define TORCH_DISALLOW_TEMPORARIES_IMPL(methodname, maybestatic)
//   maybestatic void methodname(
//       TensorBase&& out, const TensorBase& a, const TensorBase& b) = delete;
//   maybestatic void methodname(
//       const TensorBase& out, TensorBase&& a, const TensorBase& b) = delete;
//   maybestatic void methodname(
//       const TensorBase& out, const TensorBase& a, TensorBase&& b) = delete;
//   maybestatic void methodname(
//       TensorBase&& out, TensorBase&& a, const TensorBase& b) = delete;
//   maybestatic void methodname(
//       TensorBase&& out, const TensorBase& a, TensorBase&& b) = delete;
//   maybestatic void methodname(
//       const TensorBase& out, TensorBase&& a, TensorBase&& b) = delete;
//   maybestatic void methodname(
//       TensorBase&& out, TensorBase&& a, TensorBase&& b) = delete;

// #define TORCH_DISALLOW_TEMPORARIES(methodname)
//   TORCH_DISALLOW_TEMPORARIES_IMPL(methodname, )

  public native void build_binary_float_op(
        @Const @ByRef TensorBase out,
        @Const @ByRef TensorBase a,
        @Const @ByRef TensorBase b);
  public native void build_borrowing_binary_float_op(
        @Const @ByRef TensorBase out,
        @Const @ByRef TensorBase a,
        @Const @ByRef TensorBase b);
  public native void build_binary_op(
        @Const @ByRef TensorBase out,
        @Const @ByRef TensorBase a,
        @Const @ByRef TensorBase b);
  public native void build_borrowing_binary_op(
        @Const @ByRef TensorBase out,
        @Const @ByRef TensorBase a,
        @Const @ByRef TensorBase b);
  public native void build_unary_float_op(@Const @ByRef TensorBase out, @Const @ByRef TensorBase a);
  public native void build_borrowing_unary_float_op(
        @Const @ByRef TensorBase out,
        @Const @ByRef TensorBase a);
  public native void build_unary_op(@Const @ByRef TensorBase out, @Const @ByRef TensorBase a);
  // Odd special case needed for pow. Has to borrow the output because
  // it's a structured kernel, but the argument is potentially a copy.
  public native void build_output_borrowing_argument_owning_unary_op(
        @Const @ByRef TensorBase out,
        @Const @ByRef TensorBase a);
  public native void build_borrowing_unary_op(@Const @ByRef TensorBase out, @Const @ByRef TensorBase a);
  public native void build_borrowing_unary_force_boolean_op(
        @Const @ByRef TensorBase out,
        @Const @ByRef TensorBase a);
  public native void build_comparison_op(
        @Const @ByRef TensorBase out,
        @Const @ByRef TensorBase a,
        @Const @ByRef TensorBase b);
  public native void build_borrowing_comparison_op(
        @Const @ByRef TensorBase out,
        @Const @ByRef TensorBase a,
        @Const @ByRef TensorBase b);
  // Another special case: we need to own the second argument for comparison
  // ops.
  public native void build_borrowing_except_last_argument_comparison_op(
        @Const @ByRef TensorBase out,
        @Const @ByRef TensorBase a,
        @Const @ByRef TensorBase b);
  public native void build_ternary_op(
        @Const @ByRef TensorBase out,
        @Const @ByRef TensorBase a,
        @Const @ByRef TensorBase b,
        @Const @ByRef TensorBase c);

// #undef TORCH_DISALLOW_TEMPORARIES
}
