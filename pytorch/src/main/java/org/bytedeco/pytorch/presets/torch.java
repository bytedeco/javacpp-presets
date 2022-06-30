/*
 * Copyright (C) 2020-2022 Samuel Audet, Eduardo Gonzalez
 *
 * Licensed either under the Apache License, Version 2.0, or (at your option)
 * under the terms of the GNU General Public License as published by
 * the Free Software Foundation (subject to the "Classpath" exception),
 * either version 2, or any later version (collectively, the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *     http://www.gnu.org/licenses/
 *     http://www.gnu.org/software/classpath/license.html
 *
 * or as provided in the LICENSE.txt file that accompanied this code.
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.bytedeco.pytorch.presets;

import java.util.List;
import org.bytedeco.javacpp.ClassProperties;
import org.bytedeco.javacpp.LoadEnabled;
import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.FunctionPointer;
import org.bytedeco.javacpp.annotation.ByRef;
import org.bytedeco.javacpp.annotation.ByVal;
import org.bytedeco.javacpp.annotation.Cast;
import org.bytedeco.javacpp.annotation.Const;
import org.bytedeco.javacpp.annotation.MemberGetter;
import org.bytedeco.javacpp.annotation.Namespace;
import org.bytedeco.javacpp.annotation.Platform;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.javacpp.annotation.StdMove;
import org.bytedeco.javacpp.annotation.StdString;
import org.bytedeco.javacpp.tools.Info;
import org.bytedeco.javacpp.tools.InfoMap;
import org.bytedeco.javacpp.tools.InfoMapper;

import org.bytedeco.openblas.presets.openblas;

/**
 *
 * @author Samuel Audet
 */
@Properties(
    inherit = openblas.class,
    value = {
        @Platform(
            value = {"linux", "macosx", "windows"},
            compiler = "cpp14",
            define = {"SHARED_PTR_NAMESPACE std", "UNIQUE_PTR_NAMESPACE std"},
            include = {
                "c10/macros/cmake_macros.h",
                "c10/macros/Export.h",
                "c10/macros/Macros.h",
                "c10/util/IdWrapper.h",
                "c10/util/MaybeOwned.h",
//                "c10/util/C++17.h",
//                "c10/util/Array.h",
//                "c10/util/ConstexprCrc.h",
//                "c10/util/TypeIndex.h",
//                "c10/util/TypeTraits.h",
//                "c10/util/TypeList.h",
//                "c10/util/TypeSafeSignMath.h",
//                "c10/util/Metaprogramming.h",
//                "c10/util/Optional.h",
//                "c10/util/UniqueVoidPtr.h",
//                "c10/util/accumulate.h",
//                "c10/util/either.h",
//                "c10/util/flat_hash_map.h",
//                "c10/util/intrusive_ptr.h",
//                "c10/util/irange.h",
//                "c10/util/overloaded.h",
//                "c10/util/python_stub.h",
//                "c10/util/reverse_iterator.h",
//                "c10/util/string_view.h",
                "c10/util/typeid.h",
                "c10/util/AlignOf.h",
                "c10/util/Deprecated.h",
                "c10/util/StringUtil.h",
                "c10/util/SmallVector.h",
                "c10/util/Exception.h",
                "c10/util/ArrayRef.h",
                "c10/util/complex.h",
                "c10/util/Half.h",
                "c10/util/qint32.h",
                "c10/util/qint8.h",
                "c10/util/quint8.h",
                "c10/util/BFloat16.h",
                "c10/util/quint2x4.h",
                "c10/util/quint4x2.h",
                "c10/util/ThreadLocalDebugInfo.h",
                "c10/util/Type.h",
                "c10/util/TypeCast.h",
                "c10/util/Registry.h",
                "c10/util/Flags.h",
                "c10/util/Logging.h",
                "c10/util/OptionalArrayRef.h",
                "c10/core/DeviceType.h",
                "c10/core/Device.h",
                "c10/core/DeviceGuard.h",
                "c10/core/DispatchKey.h",
                "c10/core/DispatchKeySet.h",
                "c10/core/Backend.h",
                "c10/core/CopyBytes.h",
                "c10/core/GradMode.h",
                "c10/core/InferenceMode.h",
                "c10/core/Layout.h",
                "c10/core/MemoryFormat.h",
                "c10/core/QEngine.h",
                "c10/core/QScheme.h",
                "c10/core/Stream.h",
                "c10/core/ScalarType.h",
                "c10/core/ScalarTypeToTypeMeta.h",
                "c10/core/Scalar.h",
                "c10/core/SymInt.h",
                "c10/core/SymIntArrayRef.h",
                "c10/core/SymbolicIntNode.h",
                "c10/core/Allocator.h",
                "c10/core/DefaultDtype.h",
                "c10/core/StorageImpl.h",
                "c10/core/Storage.h",
                "c10/core/TensorOptions.h",
                "c10/core/TensorImpl.h",
                "c10/core/UndefinedTensorImpl.h",
                "c10/core/WrapDimMinimal.h",
//                "c10/core/GeneratorImpl.h",
//                "c10/core/impl/LocalDispatchKeySet.h",
//                "c10/core/impl/DeviceGuardImplInterface.h",
//                "caffe2/serialize/read_adapter_interface.h",
//                "caffe2/serialize/istream_adapter.h",
//                "caffe2/serialize/versions.h",
//                "caffe2/serialize/inline_container.h",
//                "ATen/core/op_registration/hacky_wrapper_for_legacy_signatures.h"
//                "ATen/core/custom_class.h",
                "ATen/core/symbol.h",
                "ATen/core/aten_interned_strings.h",
                "ATen/core/interned_strings.h",
                "ATen/core/grad_mode.h",
                "ATen/core/ATenGeneral.h",
                "ATen/core/Dimname.h",
                "ATen/core/DimVector.h",
                "ATen/core/Generator.h",
//                "ATen/core/CheckMemoryFormat.h",
//                "ATen/core/DeprecatedTypeProperties.h",
//                "ATen/core/DeprecatedTypePropertiesRegistry.h",
//                "ATen/core/LegacyTypeDispatch.h",
//                "ATen/core/QuantizerBase.h",
//                "ATen/core/Dict.h",
                "ATen/core/List.h",
                "ATen/core/NamedTensor.h",
                "ATen/core/Reduction.h",
                "ATen/core/Scalar.h",
                "ATen/core/TensorAccessor.h",
                "ATen/core/TensorBase.h",
                "ATen/core/TensorBody.h",
                "ATen/core/Tensor.h",
                "ATen/core/Formatting.h",
                "ATen/core/UnsafeFromTH.h",
                "ATen/core/Variadic.h",
                "ATen/core/blob.h",
                "ATen/core/class_type.h",
//                "ATen/core/dynamic_type.h",
                "ATen/core/enum_type.h",
                "ATen/core/type_ptr.h",
                "ATen/core/functional.h",
                "ATen/core/ivalue.h",
                "ATen/core/ivalue_to.h",
                "ATen/core/operator_name.h",
                "ATen/core/qualified_name.h",
                "ATen/core/stack.h",
                "ATen/core/alias_info.h",
                "ATen/core/jit_type_base.h",
                "ATen/core/jit_type.h",
                "ATen/core/function_schema.h",
                "ATen/core/function.h",
//                "ATen/core/builtin_function.h",
                "ATen/core/boxing/KernelFunction.h",
//                "ATen/core/boxing/impl/boxing.h",
                "ATen/core/dispatch/CppSignature.h",
                "ATen/core/dispatch/DispatchKeyExtractor.h",
                "ATen/core/dispatch/RegistrationHandleRAII.h",
                "ATen/core/dispatch/OperatorOptions.h",
                "ATen/core/dispatch/OperatorEntry.h",
                "ATen/core/dispatch/Dispatcher.h",
                "ATen/core/op_registration/op_allowlist.h",
//                "ATen/core/op_registration/infer_schema.h",
//                "ATen/core/op_registration/op_registration.h",
//                "ATen/detail/CUDAHooksInterface.h",
//                "ATen/detail/HIPHooksInterface.h",
//                "ATen/CPUGeneratorImpl.h",
//                "ATen/FuncTorchTLS.h",
//                "ATen/MethodOperators.h",
                "ATen/record_function.h",
                "ATen/ThreadLocalState.h",
                "ATen/ATen.h",
                "ATen/Config.h",
                "ATen/Device.h",
                "ATen/DeviceGuard.h",
                "ATen/DimVector.h",
                "ATen/Dispatch.h",
                "ATen/EmptyTensor.h",
                "ATen/LinalgBackend.h",
                "ATen/Formatting.h",
                "ATen/Generator.h",
                "ATen/Parallel.h",
                "ATen/Utils.h",
                "ATen/TracerMode.h",
                "ATen/WrapDimUtils.h",
                "ATen/Tensor.h",
                "ATen/TensorGeometry.h",
                "ATen/TensorNames.h",
                "ATen/TensorUtils.h",
                "ATen/Context.h",
                "ATen/ExpandUtils.h",
                "ATen/Functions.h",
                "ATen/NamedTensor.h",
                "ATen/NamedTensorUtils.h",
                "ATen/ScalarOps.h",
                "ATen/SequenceNumber.h",
                "ATen/TensorIndexing.h",
                "ATen/TensorOperators.h",
                "ATen/Version.h",

                "ATen/ops/from_blob.h",
                "ATen/ops/tensor.h",
                "ATen/ops/_adaptive_avg_pool2d.h",
                "ATen/ops/_adaptive_avg_pool2d_backward.h",
                "ATen/ops/_adaptive_avg_pool3d.h",
                "ATen/ops/_adaptive_avg_pool3d_backward.h",
                "ATen/ops/_add_batch_dim.h",
                "ATen/ops/_add_relu.h",
                "ATen/ops/_addmm_activation.h",
                "ATen/ops/_aminmax.h",
                "ATen/ops/_amp_foreach_non_finite_check_and_unscale.h",
                "ATen/ops/_amp_update_scale.h",
                "ATen/ops/_assert_async.h",
                "ATen/ops/_autocast_to_full_precision.h",
                "ATen/ops/_autocast_to_reduced_precision.h",
                "ATen/ops/_backward.h",
                "ATen/ops/_batch_norm_impl_index.h",
                "ATen/ops/_batch_norm_impl_index_backward.h",
                "ATen/ops/_cast_Byte.h",
                "ATen/ops/_cast_Char.h",
                "ATen/ops/_cast_Double.h",
                "ATen/ops/_cast_Float.h",
                "ATen/ops/_cast_Half.h",
                "ATen/ops/_cast_Int.h",
                "ATen/ops/_cast_Long.h",
                "ATen/ops/_cast_Short.h",
//                "ATen/ops/_cat.h",
                "ATen/ops/_cdist_backward.h",
                "ATen/ops/_cdist_forward.h",
                "ATen/ops/_cholesky_solve_helper.h",
                "ATen/ops/_choose_qparams_per_tensor.h",
                "ATen/ops/_coalesce.h",
                "ATen/ops/_coalesced.h",
                "ATen/ops/_compute_linear_combination.h",
                "ATen/ops/_conj.h",
                "ATen/ops/_conj_copy.h",
                "ATen/ops/_conj_physical.h",
                "ATen/ops/_conv_depthwise2d.h",
                "ATen/ops/_convert_indices_from_coo_to_csr.h",
                "ATen/ops/_convert_indices_from_csr_to_coo.h",
                "ATen/ops/_convolution.h",
                "ATen/ops/_convolution_double_backward.h",
                "ATen/ops/_convolution_mode.h",
                "ATen/ops/_copy_from.h",
                "ATen/ops/_copy_from_and_resize.h",
                "ATen/ops/_ctc_loss.h",
                "ATen/ops/_ctc_loss_backward.h",
                "ATen/ops/_cudnn_ctc_loss.h",
                "ATen/ops/_cudnn_init_dropout_state.h",
                "ATen/ops/_cudnn_rnn.h",
                "ATen/ops/_cudnn_rnn_backward.h",
                "ATen/ops/_cudnn_rnn_flatten_weight.h",
                "ATen/ops/_cufft_clear_plan_cache.h",
                "ATen/ops/_cufft_get_plan_cache_max_size.h",
                "ATen/ops/_cufft_get_plan_cache_size.h",
                "ATen/ops/_cufft_set_plan_cache_max_size.h",
                "ATen/ops/_cummax_helper.h",
                "ATen/ops/_cummin_helper.h",
                "ATen/ops/_debug_has_internal_overlap.h",
                "ATen/ops/_det_lu_based_helper.h",
                "ATen/ops/_det_lu_based_helper_backward_helper.h",
                "ATen/ops/_dimI.h",
                "ATen/ops/_dimV.h",
                "ATen/ops/_dim_arange.h",
                "ATen/ops/_dirichlet_grad.h",
                "ATen/ops/_efficientzerotensor.h",
                "ATen/ops/_embedding_bag.h",
                "ATen/ops/_embedding_bag_backward.h",
                "ATen/ops/_embedding_bag_dense_backward.h",
                "ATen/ops/_embedding_bag_forward_only.h",
                "ATen/ops/_embedding_bag_per_sample_weights_backward.h",
                "ATen/ops/_embedding_bag_sparse_backward.h",
                "ATen/ops/_empty_affine_quantized.h",
                "ATen/ops/_empty_per_channel_affine_quantized.h",
                "ATen/ops/_euclidean_dist.h",
                "ATen/ops/_fake_quantize_learnable_per_channel_affine.h",
                "ATen/ops/_fake_quantize_learnable_per_channel_affine_backward.h",
                "ATen/ops/_fake_quantize_learnable_per_tensor_affine.h",
                "ATen/ops/_fake_quantize_learnable_per_tensor_affine_backward.h",
                "ATen/ops/_fake_quantize_per_tensor_affine_cachemask_tensor_qparams.h",
                "ATen/ops/_fft_c2c.h",
                "ATen/ops/_fft_c2r.h",
                "ATen/ops/_fft_r2c.h",
                "ATen/ops/_foreach_abs.h",
                "ATen/ops/_foreach_acos.h",
                "ATen/ops/_foreach_add.h",
                "ATen/ops/_foreach_addcdiv.h",
                "ATen/ops/_foreach_addcmul.h",
                "ATen/ops/_foreach_asin.h",
                "ATen/ops/_foreach_atan.h",
                "ATen/ops/_foreach_ceil.h",
                "ATen/ops/_foreach_cos.h",
                "ATen/ops/_foreach_cosh.h",
                "ATen/ops/_foreach_div.h",
                "ATen/ops/_foreach_erf.h",
                "ATen/ops/_foreach_erfc.h",
                "ATen/ops/_foreach_exp.h",
                "ATen/ops/_foreach_expm1.h",
                "ATen/ops/_foreach_floor.h",
                "ATen/ops/_foreach_frac.h",
                "ATen/ops/_foreach_lgamma.h",
                "ATen/ops/_foreach_log.h",
                "ATen/ops/_foreach_log10.h",
                "ATen/ops/_foreach_log1p.h",
                "ATen/ops/_foreach_log2.h",
                "ATen/ops/_foreach_maximum.h",
                "ATen/ops/_foreach_minimum.h",
                "ATen/ops/_foreach_mul.h",
                "ATen/ops/_foreach_neg.h",
                "ATen/ops/_foreach_norm.h",
                "ATen/ops/_foreach_reciprocal.h",
                "ATen/ops/_foreach_round.h",
                "ATen/ops/_foreach_sigmoid.h",
                "ATen/ops/_foreach_sin.h",
                "ATen/ops/_foreach_sinh.h",
                "ATen/ops/_foreach_sqrt.h",
                "ATen/ops/_foreach_sub.h",
                "ATen/ops/_foreach_tan.h",
                "ATen/ops/_foreach_tanh.h",
                "ATen/ops/_foreach_trunc.h",
                "ATen/ops/_foreach_zero.h",
                "ATen/ops/_fused_dropout.h",
                "ATen/ops/_fused_moving_avg_obs_fq_helper.h",
                "ATen/ops/_fw_primal.h",
                "ATen/ops/_fw_primal_copy.h",
                "ATen/ops/_gather_sparse_backward.h",
                "ATen/ops/_grid_sampler_2d_cpu_fallback.h",
                "ATen/ops/_grid_sampler_2d_cpu_fallback_backward.h",
                "ATen/ops/_has_compatible_shallow_copy_type.h",
                "ATen/ops/_has_same_storage_numel.h",
                "ATen/ops/_histogramdd_bin_edges.h",
                "ATen/ops/_histogramdd_from_bin_cts.h",
                "ATen/ops/_histogramdd_from_bin_tensors.h",
//                "ATen/ops/_index_copy.h",
                "ATen/ops/_index_put_impl.h",
                "ATen/ops/_indices.h",
                "ATen/ops/_indices_copy.h",
                "ATen/ops/_is_zerotensor.h",
                "ATen/ops/_linalg_check_errors.h",
                "ATen/ops/_linalg_inv_out_helper.h",
                "ATen/ops/_linalg_qr_helper.h",
                "ATen/ops/_linalg_svd.h",
                "ATen/ops/_local_scalar_dense.h",
                "ATen/ops/_log_softmax.h",
                "ATen/ops/_log_softmax_backward_data.h",
                "ATen/ops/_logcumsumexp.h",
                "ATen/ops/_lstm_mps.h",
                "ATen/ops/_lu_with_info.h",
                "ATen/ops/_make_dual.h",
                "ATen/ops/_make_dual_copy.h",
                "ATen/ops/_make_per_channel_quantized_tensor.h",
                "ATen/ops/_make_per_tensor_quantized_tensor.h",
                "ATen/ops/_masked_scale.h",
                "ATen/ops/_masked_softmax.h",
                "ATen/ops/_masked_softmax_backward.h",
                "ATen/ops/_mkldnn_reshape.h",
                "ATen/ops/_mkldnn_transpose.h",
                "ATen/ops/_mps_convolution.h",
                "ATen/ops/_mps_convolution_transpose.h",
                "ATen/ops/_mps_linear.h",
                "ATen/ops/_mps_linear_backward_input.h",
                "ATen/ops/_mps_linear_backward_weights.h",
                "ATen/ops/_mps_max_pool2d.h",
                "ATen/ops/_native_multi_head_attention.h",
//                "ATen/ops/_native_multi_head_self_attention.h",
                "ATen/ops/_neg_view.h",
                "ATen/ops/_neg_view_copy.h",
                "ATen/ops/_nested_from_padded.h",
                "ATen/ops/_nested_from_padded_and_nested_example.h",
                "ATen/ops/_nested_tensor_from_mask.h",
                "ATen/ops/_nested_tensor_layer_norm.h",
                "ATen/ops/_new_zeros_with_same_feature_meta.h",
//                "ATen/ops/_nnpack_available.h",
                "ATen/ops/_nnpack_spatial_convolution.h",
                "ATen/ops/_nnz.h",
                "ATen/ops/_pack_padded_sequence.h",
                "ATen/ops/_pack_padded_sequence_backward.h",
                "ATen/ops/_pad_circular.h",
                "ATen/ops/_pad_enum.h",
                "ATen/ops/_pad_packed_sequence.h",
                "ATen/ops/_pdist_backward.h",
                "ATen/ops/_pdist_forward.h",
                "ATen/ops/_pin_memory.h",
                "ATen/ops/_remove_batch_dim.h",
                "ATen/ops/_reshape_alias.h",
                "ATen/ops/_reshape_alias_copy.h",
                "ATen/ops/_reshape_from_tensor.h",
                "ATen/ops/_resize_output.h",
                "ATen/ops/_rowwise_prune.h",
//                "ATen/ops/_s_where.h",
                "ATen/ops/_sample_dirichlet.h",
                "ATen/ops/_saturate_weight_to_fp16.h",
                "ATen/ops/_segment_reduce_backward.h",
                "ATen/ops/_shape_as_tensor.h",
                "ATen/ops/_slow_conv2d_backward.h",
                "ATen/ops/_slow_conv2d_forward.h",
                "ATen/ops/_sobol_engine_draw.h",
                "ATen/ops/_sobol_engine_ff.h",
                "ATen/ops/_sobol_engine_initialize_state.h",
                "ATen/ops/_sobol_engine_scramble.h",
                "ATen/ops/_softmax.h",
                "ATen/ops/_softmax_backward_data.h",
//                "ATen/ops/_solve_helper.h",
                "ATen/ops/_sparse_addmm.h",
                "ATen/ops/_sparse_broadcast_to.h",
                "ATen/ops/_sparse_broadcast_to_copy.h",
                "ATen/ops/_sparse_bsc_tensor_unsafe.h",
                "ATen/ops/_sparse_bsr_tensor_unsafe.h",
                "ATen/ops/_sparse_compressed_tensor_unsafe.h",
                "ATen/ops/_sparse_coo_tensor_unsafe.h",
                "ATen/ops/_sparse_coo_tensor_with_dims.h",
                "ATen/ops/_sparse_coo_tensor_with_dims_and_tensors.h",
                "ATen/ops/_sparse_csc_tensor_unsafe.h",
                "ATen/ops/_sparse_csr_prod.h",
                "ATen/ops/_sparse_csr_sum.h",
                "ATen/ops/_sparse_csr_tensor_unsafe.h",
                "ATen/ops/_sparse_log_softmax.h",
                "ATen/ops/_sparse_log_softmax_backward_data.h",
                "ATen/ops/_sparse_mask_helper.h",
                "ATen/ops/_sparse_mm.h",
                "ATen/ops/_sparse_softmax.h",
                "ATen/ops/_sparse_softmax_backward_data.h",
                "ATen/ops/_sparse_sparse_matmul.h",
                "ATen/ops/_sparse_sum.h",
                "ATen/ops/_sparse_sum_backward.h",
                "ATen/ops/_stack.h",
                "ATen/ops/_standard_gamma.h",
                "ATen/ops/_standard_gamma_grad.h",
                "ATen/ops/_symeig_helper.h",
                "ATen/ops/_test_ambiguous_defaults.h",
                "ATen/ops/_test_optional_filled_intlist.h",
                "ATen/ops/_test_optional_floatlist.h",
                "ATen/ops/_test_optional_intlist.h",
                "ATen/ops/_test_serialization_subcmul.h",
                "ATen/ops/_test_string_default.h",
                "ATen/ops/_test_warn_in_autograd.h",
                "ATen/ops/_thnn_differentiable_gru_cell_backward.h",
                "ATen/ops/_thnn_differentiable_lstm_cell_backward.h",
                "ATen/ops/_thnn_fused_gru_cell.h",
                "ATen/ops/_thnn_fused_gru_cell_backward.h",
                "ATen/ops/_thnn_fused_lstm_cell.h",
                "ATen/ops/_thnn_fused_lstm_cell_backward.h",
                "ATen/ops/_thnn_fused_lstm_cell_backward_impl.h",
                "ATen/ops/_to_copy.h",
                "ATen/ops/_to_cpu.h",
                "ATen/ops/_to_dense.h",
                "ATen/ops/_torch_cuda_cu_linker_symbol_op.h",
                "ATen/ops/_transform_bias_rescale_qkv.h",
                "ATen/ops/_transformer_encoder_layer_fwd.h",
                "ATen/ops/_trilinear.h",
                "ATen/ops/_unique.h",
                "ATen/ops/_unique2.h",
                "ATen/ops/_unpack_dual.h",
                "ATen/ops/_unsafe_view.h",
                "ATen/ops/_upsample_bicubic2d_aa.h",
                "ATen/ops/_upsample_bicubic2d_aa_backward.h",
                "ATen/ops/_upsample_bilinear2d_aa.h",
                "ATen/ops/_upsample_bilinear2d_aa_backward.h",
                "ATen/ops/_upsample_nearest_exact1d.h",
                "ATen/ops/_upsample_nearest_exact1d_backward.h",
                "ATen/ops/_upsample_nearest_exact2d.h",
                "ATen/ops/_upsample_nearest_exact2d_backward.h",
                "ATen/ops/_upsample_nearest_exact3d.h",
                "ATen/ops/_upsample_nearest_exact3d_backward.h",
                "ATen/ops/_use_cudnn_ctc_loss.h",
//                "ATen/ops/_use_cudnn_rnn_flatten_weight.h",
                "ATen/ops/_validate_sparse_bsc_tensor_args.h",
                "ATen/ops/_validate_sparse_bsr_tensor_args.h",
                "ATen/ops/_validate_sparse_compressed_tensor_args.h",
                "ATen/ops/_validate_sparse_coo_tensor_args.h",
                "ATen/ops/_validate_sparse_csc_tensor_args.h",
                "ATen/ops/_validate_sparse_csr_tensor_args.h",
                "ATen/ops/_values.h",
                "ATen/ops/_values_copy.h",
                "ATen/ops/_version.h",
                "ATen/ops/_weight_norm.h",
//                "ATen/ops/_weight_norm_cuda_interface.h",
//                "ATen/ops/_weight_norm_cuda_interface_backward.h",
                "ATen/ops/_weight_norm_differentiable_backward.h",
                "ATen/ops/_weight_norm_interface.h",
                "ATen/ops/_weight_norm_interface_backward.h",
                "ATen/ops/abs.h",
                "ATen/ops/absolute.h",
                "ATen/ops/acos.h",
                "ATen/ops/acosh.h",
                "ATen/ops/adaptive_avg_pool1d.h",
                "ATen/ops/adaptive_avg_pool2d.h",
                "ATen/ops/adaptive_avg_pool3d.h",
                "ATen/ops/adaptive_avg_pool3d_backward.h",
                "ATen/ops/adaptive_max_pool1d.h",
                "ATen/ops/adaptive_max_pool2d.h",
                "ATen/ops/adaptive_max_pool2d_backward.h",
                "ATen/ops/adaptive_max_pool3d.h",
                "ATen/ops/adaptive_max_pool3d_backward.h",
                "ATen/ops/add.h",
                "ATen/ops/addbmm.h",
                "ATen/ops/addcdiv.h",
                "ATen/ops/addcmul.h",
                "ATen/ops/addmm.h",
                "ATen/ops/addmv.h",
                "ATen/ops/addr.h",
                "ATen/ops/adjoint.h",
                "ATen/ops/affine_grid_generator.h",
                "ATen/ops/affine_grid_generator_backward.h",
                "ATen/ops/alias.h",
                "ATen/ops/alias_copy.h",
                "ATen/ops/align_as.h",
                "ATen/ops/align_tensors.h",
                "ATen/ops/align_to.h",
                "ATen/ops/all.h",
                "ATen/ops/allclose.h",
                "ATen/ops/alpha_dropout.h",
                "ATen/ops/amax.h",
                "ATen/ops/amin.h",
                "ATen/ops/aminmax.h",
                "ATen/ops/and.h",
                "ATen/ops/angle.h",
                "ATen/ops/any.h",
                "ATen/ops/arange.h",
                "ATen/ops/arccos.h",
                "ATen/ops/arccosh.h",
                "ATen/ops/arcsin.h",
                "ATen/ops/arcsinh.h",
                "ATen/ops/arctan.h",
                "ATen/ops/arctan2.h",
                "ATen/ops/arctanh.h",
                "ATen/ops/argmax.h",
                "ATen/ops/argmin.h",
                "ATen/ops/argsort.h",
                "ATen/ops/argwhere.h",
                "ATen/ops/as_strided.h",
                "ATen/ops/as_strided_copy.h",
                "ATen/ops/asin.h",
                "ATen/ops/asinh.h",
                "ATen/ops/atan.h",
                "ATen/ops/atan2.h",
                "ATen/ops/atanh.h",
                "ATen/ops/atleast_1d.h",
                "ATen/ops/atleast_2d.h",
                "ATen/ops/atleast_3d.h",
                "ATen/ops/avg_pool1d.h",
                "ATen/ops/avg_pool2d.h",
                "ATen/ops/avg_pool2d_backward.h",
                "ATen/ops/avg_pool3d.h",
                "ATen/ops/avg_pool3d_backward.h",
                "ATen/ops/baddbmm.h",
                "ATen/ops/bartlett_window.h",
                "ATen/ops/batch_norm.h",
                "ATen/ops/batch_norm_backward_elemt.h",
                "ATen/ops/batch_norm_backward_reduce.h",
                "ATen/ops/batch_norm_elemt.h",
                "ATen/ops/batch_norm_gather_stats.h",
                "ATen/ops/batch_norm_gather_stats_with_counts.h",
                "ATen/ops/batch_norm_stats.h",
                "ATen/ops/batch_norm_update_stats.h",
                "ATen/ops/bernoulli.h",
                "ATen/ops/bilinear.h",
                "ATen/ops/binary_cross_entropy.h",
                "ATen/ops/binary_cross_entropy_backward.h",
                "ATen/ops/binary_cross_entropy_with_logits.h",
                "ATen/ops/binary_cross_entropy_with_logits_backward.h",
                "ATen/ops/bincount.h",
                "ATen/ops/binomial.h",
                "ATen/ops/bitwise_and.h",
                "ATen/ops/bitwise_left_shift.h",
                "ATen/ops/bitwise_not.h",
                "ATen/ops/bitwise_or.h",
                "ATen/ops/bitwise_right_shift.h",
                "ATen/ops/bitwise_xor.h",
                "ATen/ops/blackman_window.h",
                "ATen/ops/block_diag.h",
                "ATen/ops/bmm.h",
                "ATen/ops/broadcast_tensors.h",
                "ATen/ops/broadcast_to.h",
                "ATen/ops/bucketize.h",
                "ATen/ops/can_cast.h",
                "ATen/ops/cartesian_prod.h",
                "ATen/ops/cat.h",
                "ATen/ops/cauchy.h",
                "ATen/ops/ccol_indices.h",
                "ATen/ops/ccol_indices_copy.h",
                "ATen/ops/cdist.h",
                "ATen/ops/ceil.h",
                "ATen/ops/celu.h",
                "ATen/ops/chain_matmul.h",
                "ATen/ops/chalf.h",
                "ATen/ops/channel_shuffle.h",
                "ATen/ops/cholesky.h",
                "ATen/ops/cholesky_inverse.h",
                "ATen/ops/cholesky_solve.h",
                "ATen/ops/choose_qparams_optimized.h",
                "ATen/ops/chunk.h",
                "ATen/ops/clamp.h",
                "ATen/ops/clamp_max.h",
                "ATen/ops/clamp_min.h",
                "ATen/ops/clip.h",
                "ATen/ops/clone.h",
                "ATen/ops/coalesce.h",
                "ATen/ops/col2im.h",
                "ATen/ops/col2im_backward.h",
                "ATen/ops/col_indices.h",
                "ATen/ops/col_indices_copy.h",
                "ATen/ops/column_stack.h",
                "ATen/ops/combinations.h",
                "ATen/ops/complex.h",
                "ATen/ops/concat.h",
                "ATen/ops/conj.h",
                "ATen/ops/conj_physical.h",
                "ATen/ops/constant_pad_nd.h",
                "ATen/ops/contiguous.h",
                "ATen/ops/conv1d.h",
                "ATen/ops/conv2d.h",
                "ATen/ops/conv3d.h",
                "ATen/ops/conv_depthwise3d.h",
                "ATen/ops/conv_tbc.h",
                "ATen/ops/conv_tbc_backward.h",
                "ATen/ops/conv_transpose1d.h",
                "ATen/ops/conv_transpose2d.h",
                "ATen/ops/conv_transpose3d.h",
                "ATen/ops/convolution.h",
                "ATen/ops/convolution_backward.h",
                "ATen/ops/convolution_backward_overrideable.h",
                "ATen/ops/convolution_overrideable.h",
                "ATen/ops/copy.h",
                "ATen/ops/copy_sparse_to_sparse.h",
                "ATen/ops/copysign.h",
                "ATen/ops/corrcoef.h",
                "ATen/ops/cos.h",
                "ATen/ops/cosh.h",
                "ATen/ops/cosine_embedding_loss.h",
                "ATen/ops/cosine_similarity.h",
                "ATen/ops/count_nonzero.h",
                "ATen/ops/cov.h",
                "ATen/ops/cross.h",
                "ATen/ops/cross_entropy_loss.h",
                "ATen/ops/crow_indices.h",
                "ATen/ops/crow_indices_copy.h",
                "ATen/ops/ctc_loss.h",
                "ATen/ops/cudnn_affine_grid_generator.h",
                "ATen/ops/cudnn_affine_grid_generator_backward.h",
                "ATen/ops/cudnn_batch_norm.h",
                "ATen/ops/cudnn_batch_norm_backward.h",
                "ATen/ops/cudnn_convolution.h",
                "ATen/ops/cudnn_convolution_add_relu.h",
                "ATen/ops/cudnn_convolution_relu.h",
                "ATen/ops/cudnn_convolution_transpose.h",
                "ATen/ops/cudnn_grid_sampler.h",
                "ATen/ops/cudnn_grid_sampler_backward.h",
                "ATen/ops/cudnn_is_acceptable.h",
                "ATen/ops/cummax.h",
                "ATen/ops/cummaxmin_backward.h",
                "ATen/ops/cummin.h",
                "ATen/ops/cumprod.h",
                "ATen/ops/cumprod_backward.h",
                "ATen/ops/cumsum.h",
                "ATen/ops/cumulative_trapezoid.h",
                "ATen/ops/data.h",
                "ATen/ops/deg2rad.h",
                "ATen/ops/dense_dim.h",
                "ATen/ops/dequantize.h",
                "ATen/ops/det.h",
                "ATen/ops/detach.h",
                "ATen/ops/detach_copy.h",
                "ATen/ops/diag.h",
                "ATen/ops/diag_backward.h",
                "ATen/ops/diag_embed.h",
                "ATen/ops/diagflat.h",
                "ATen/ops/diagonal.h",
                "ATen/ops/diagonal_backward.h",
                "ATen/ops/diagonal_copy.h",
                "ATen/ops/diagonal_scatter.h",
                "ATen/ops/diff.h",
                "ATen/ops/digamma.h",
                "ATen/ops/dist.h",
                "ATen/ops/div.h",
                "ATen/ops/divide.h",
                "ATen/ops/dot.h",
                "ATen/ops/dropout.h",
                "ATen/ops/dsplit.h",
                "ATen/ops/dstack.h",
                "ATen/ops/eig.h",
                "ATen/ops/einsum.h",
                "ATen/ops/elu.h",
                "ATen/ops/elu_backward.h",
                "ATen/ops/embedding.h",
                "ATen/ops/embedding_backward.h",
                "ATen/ops/embedding_bag.h",
                "ATen/ops/embedding_dense_backward.h",
                "ATen/ops/embedding_renorm.h",
                "ATen/ops/embedding_sparse_backward.h",
                "ATen/ops/empty.h",
                "ATen/ops/empty_like.h",
                "ATen/ops/empty_quantized.h",
                "ATen/ops/empty_strided.h",
                "ATen/ops/eq.h",
                "ATen/ops/equal.h",
                "ATen/ops/erf.h",
                "ATen/ops/erfc.h",
                "ATen/ops/erfinv.h",
                "ATen/ops/exp.h",
                "ATen/ops/exp2.h",
                "ATen/ops/expand.h",
                "ATen/ops/expand_as.h",
                "ATen/ops/expand_copy.h",
                "ATen/ops/expm1.h",
                "ATen/ops/exponential.h",
                "ATen/ops/eye.h",
                "ATen/ops/fake_quantize_per_channel_affine.h",
                "ATen/ops/fake_quantize_per_channel_affine_cachemask.h",
                "ATen/ops/fake_quantize_per_channel_affine_cachemask_backward.h",
                "ATen/ops/fake_quantize_per_tensor_affine.h",
                "ATen/ops/fake_quantize_per_tensor_affine_cachemask.h",
                "ATen/ops/fake_quantize_per_tensor_affine_cachemask_backward.h",
                "ATen/ops/fbgemm_linear_fp16_weight.h",
                "ATen/ops/fbgemm_linear_fp16_weight_fp32_activation.h",
                "ATen/ops/fbgemm_linear_int8_weight.h",
                "ATen/ops/fbgemm_linear_int8_weight_fp32_activation.h",
                "ATen/ops/fbgemm_linear_quantize_weight.h",
                "ATen/ops/fbgemm_pack_gemm_matrix_fp16.h",
                "ATen/ops/fbgemm_pack_quantized_matrix.h",
                "ATen/ops/feature_alpha_dropout.h",
                "ATen/ops/feature_dropout.h",
                "ATen/ops/fft_fft.h",
                "ATen/ops/fft_fft2.h",
                "ATen/ops/fft_fftfreq.h",
                "ATen/ops/fft_fftn.h",
                "ATen/ops/fft_fftshift.h",
                "ATen/ops/fft_hfft.h",
                "ATen/ops/fft_hfft2.h",
                "ATen/ops/fft_hfftn.h",
                "ATen/ops/fft_ifft.h",
                "ATen/ops/fft_ifft2.h",
                "ATen/ops/fft_ifftn.h",
                "ATen/ops/fft_ifftshift.h",
                "ATen/ops/fft_ihfft.h",
                "ATen/ops/fft_ihfft2.h",
                "ATen/ops/fft_ihfftn.h",
                "ATen/ops/fft_irfft.h",
                "ATen/ops/fft_irfft2.h",
                "ATen/ops/fft_irfftn.h",
                "ATen/ops/fft_rfft.h",
                "ATen/ops/fft_rfft2.h",
                "ATen/ops/fft_rfftfreq.h",
                "ATen/ops/fft_rfftn.h",
                "ATen/ops/fill.h",
                "ATen/ops/fill_diagonal.h",
                "ATen/ops/fix.h",
                "ATen/ops/flatten.h",
                "ATen/ops/flatten_dense_tensors.h",
                "ATen/ops/flip.h",
                "ATen/ops/fliplr.h",
                "ATen/ops/flipud.h",
                "ATen/ops/float_power.h",
                "ATen/ops/floor.h",
                "ATen/ops/floor_divide.h",
                "ATen/ops/fmax.h",
                "ATen/ops/fmin.h",
                "ATen/ops/fmod.h",
                "ATen/ops/frac.h",
                "ATen/ops/fractional_max_pool2d.h",
                "ATen/ops/fractional_max_pool2d_backward.h",
                "ATen/ops/fractional_max_pool3d.h",
                "ATen/ops/fractional_max_pool3d_backward.h",
                "ATen/ops/frexp.h",
                "ATen/ops/frobenius_norm.h",
                "ATen/ops/from_file.h",
                "ATen/ops/full.h",
                "ATen/ops/full_like.h",
                "ATen/ops/fused_moving_avg_obs_fake_quant.h",
                "ATen/ops/gather.h",
                "ATen/ops/gather_backward.h",
                "ATen/ops/gcd.h",
                "ATen/ops/ge.h",
                "ATen/ops/gelu.h",
                "ATen/ops/gelu_backward.h",
                "ATen/ops/geometric.h",
                "ATen/ops/geqrf.h",
                "ATen/ops/ger.h",
                "ATen/ops/glu.h",
                "ATen/ops/glu_backward.h",
                "ATen/ops/glu_backward_jvp.h",
                "ATen/ops/glu_jvp.h",
                "ATen/ops/gradient.h",
                "ATen/ops/greater.h",
                "ATen/ops/greater_equal.h",
                "ATen/ops/grid_sampler.h",
                "ATen/ops/grid_sampler_2d.h",
                "ATen/ops/grid_sampler_2d_backward.h",
                "ATen/ops/grid_sampler_3d.h",
                "ATen/ops/grid_sampler_3d_backward.h",
                "ATen/ops/group_norm.h",
                "ATen/ops/gru.h",
                "ATen/ops/gru_cell.h",
                "ATen/ops/gt.h",
                "ATen/ops/hamming_window.h",
                "ATen/ops/hann_window.h",
                "ATen/ops/hardshrink.h",
                "ATen/ops/hardshrink_backward.h",
                "ATen/ops/hardsigmoid.h",
                "ATen/ops/hardsigmoid_backward.h",
                "ATen/ops/hardswish.h",
                "ATen/ops/hardswish_backward.h",
                "ATen/ops/hardtanh.h",
                "ATen/ops/hardtanh_backward.h",
                "ATen/ops/heaviside.h",
                "ATen/ops/hinge_embedding_loss.h",
                "ATen/ops/histc.h",
                "ATen/ops/histogram.h",
                "ATen/ops/histogramdd.h",
                "ATen/ops/hsplit.h",
                "ATen/ops/hspmm.h",
                "ATen/ops/hstack.h",
                "ATen/ops/huber_loss.h",
                "ATen/ops/huber_loss_backward.h",
                "ATen/ops/hypot.h",
                "ATen/ops/i0.h",
                "ATen/ops/igamma.h",
                "ATen/ops/igammac.h",
                "ATen/ops/im2col.h",
                "ATen/ops/im2col_backward.h",
                "ATen/ops/imag.h",
                "ATen/ops/index.h",
                "ATen/ops/index_add.h",
                "ATen/ops/index_copy.h",
                "ATen/ops/index_fill.h",
                "ATen/ops/index_put.h",
                "ATen/ops/index_reduce.h",
                "ATen/ops/index_select.h",
                "ATen/ops/index_select_backward.h",
                "ATen/ops/indices.h",
                "ATen/ops/indices_copy.h",
                "ATen/ops/infinitely_differentiable_gelu_backward.h",
                "ATen/ops/inner.h",
                "ATen/ops/instance_norm.h",
                "ATen/ops/int_repr.h",
                "ATen/ops/inverse.h",
                "ATen/ops/is_coalesced.h",
                "ATen/ops/is_complex.h",
                "ATen/ops/is_conj.h",
                "ATen/ops/is_distributed.h",
                "ATen/ops/is_floating_point.h",
                "ATen/ops/is_inference.h",
                "ATen/ops/is_leaf.h",
                "ATen/ops/is_neg.h",
                "ATen/ops/is_nonzero.h",
                "ATen/ops/is_pinned.h",
                "ATen/ops/is_same_size.h",
                "ATen/ops/is_set_to.h",
                "ATen/ops/is_signed.h",
//                "ATen/ops/is_vulkan_available.h",
                "ATen/ops/isclose.h",
                "ATen/ops/isfinite.h",
                "ATen/ops/isin.h",
                "ATen/ops/isinf.h",
                "ATen/ops/isnan.h",
                "ATen/ops/isneginf.h",
                "ATen/ops/isposinf.h",
                "ATen/ops/isreal.h",
                "ATen/ops/istft.h",
                "ATen/ops/item.h",
                "ATen/ops/kaiser_window.h",
                "ATen/ops/kl_div.h",
                "ATen/ops/kl_div_backward.h",
                "ATen/ops/kron.h",
                "ATen/ops/kthvalue.h",
                "ATen/ops/l1_loss.h",
                "ATen/ops/l1_loss_backward.h",
                "ATen/ops/layer_norm.h",
                "ATen/ops/lcm.h",
                "ATen/ops/ldexp.h",
                "ATen/ops/le.h",
                "ATen/ops/leaky_relu.h",
                "ATen/ops/leaky_relu_backward.h",
                "ATen/ops/lerp.h",
                "ATen/ops/less.h",
                "ATen/ops/less_equal.h",
                "ATen/ops/lgamma.h",
                "ATen/ops/lift.h",
                "ATen/ops/linalg_cholesky.h",
                "ATen/ops/linalg_cholesky_ex.h",
                "ATen/ops/linalg_cond.h",
                "ATen/ops/linalg_cross.h",
                "ATen/ops/linalg_det.h",
                "ATen/ops/linalg_diagonal.h",
                "ATen/ops/linalg_eig.h",
                "ATen/ops/linalg_eigh.h",
                "ATen/ops/linalg_eigvals.h",
                "ATen/ops/linalg_eigvalsh.h",
                "ATen/ops/linalg_householder_product.h",
                "ATen/ops/linalg_inv.h",
                "ATen/ops/linalg_inv_ex.h",
                "ATen/ops/linalg_ldl_factor.h",
                "ATen/ops/linalg_ldl_factor_ex.h",
                "ATen/ops/linalg_ldl_solve.h",
                "ATen/ops/linalg_lstsq.h",
                "ATen/ops/linalg_lu.h",
                "ATen/ops/linalg_lu_factor.h",
                "ATen/ops/linalg_lu_factor_ex.h",
                "ATen/ops/linalg_matmul.h",
                "ATen/ops/linalg_matrix_exp.h",
                "ATen/ops/linalg_matrix_norm.h",
                "ATen/ops/linalg_matrix_power.h",
                "ATen/ops/linalg_matrix_rank.h",
                "ATen/ops/linalg_multi_dot.h",
                "ATen/ops/linalg_norm.h",
                "ATen/ops/linalg_pinv.h",
                "ATen/ops/linalg_qr.h",
                "ATen/ops/linalg_slogdet.h",
                "ATen/ops/linalg_solve.h",
                "ATen/ops/linalg_solve_triangular.h",
                "ATen/ops/linalg_svd.h",
                "ATen/ops/linalg_svdvals.h",
                "ATen/ops/linalg_tensorinv.h",
                "ATen/ops/linalg_tensorsolve.h",
                "ATen/ops/linalg_vander.h",
                "ATen/ops/linalg_vector_norm.h",
                "ATen/ops/linear.h",
                "ATen/ops/linspace.h",
                "ATen/ops/log.h",
                "ATen/ops/log10.h",
                "ATen/ops/log1p.h",
                "ATen/ops/log2.h",
                "ATen/ops/log_normal.h",
                "ATen/ops/log_sigmoid.h",
                "ATen/ops/log_sigmoid_backward.h",
                "ATen/ops/log_sigmoid_forward.h",
                "ATen/ops/log_softmax.h",
                "ATen/ops/logaddexp.h",
                "ATen/ops/logaddexp2.h",
                "ATen/ops/logcumsumexp.h",
                "ATen/ops/logdet.h",
                "ATen/ops/logical_and.h",
                "ATen/ops/logical_not.h",
                "ATen/ops/logical_or.h",
                "ATen/ops/logical_xor.h",
                "ATen/ops/logit.h",
                "ATen/ops/logit_backward.h",
                "ATen/ops/logspace.h",
                "ATen/ops/logsumexp.h",
                "ATen/ops/lshift.h",
                "ATen/ops/lstm.h",
                "ATen/ops/lstm_cell.h",
                "ATen/ops/lstm_mps_backward.h",
                "ATen/ops/lstsq.h",
                "ATen/ops/lt.h",
                "ATen/ops/lu_solve.h",
                "ATen/ops/lu_unpack.h",
                "ATen/ops/mH.h",
                "ATen/ops/mT.h",
                "ATen/ops/margin_ranking_loss.h",
                "ATen/ops/masked_fill.h",
                "ATen/ops/masked_scatter.h",
                "ATen/ops/masked_select.h",
                "ATen/ops/masked_select_backward.h",
                "ATen/ops/matmul.h",
                "ATen/ops/matrix_H.h",
                "ATen/ops/matrix_exp.h",
                "ATen/ops/matrix_exp_backward.h",
                "ATen/ops/matrix_power.h",
                "ATen/ops/matrix_rank.h",
                "ATen/ops/max.h",
                "ATen/ops/max_pool1d.h",
                "ATen/ops/max_pool1d_with_indices.h",
                "ATen/ops/max_pool2d.h",
                "ATen/ops/max_pool2d_with_indices.h",
                "ATen/ops/max_pool2d_with_indices_backward.h",
                "ATen/ops/max_pool3d.h",
                "ATen/ops/max_pool3d_with_indices.h",
                "ATen/ops/max_pool3d_with_indices_backward.h",
                "ATen/ops/max_unpool2d.h",
//                "ATen/ops/max_unpool2d_backward.h",
                "ATen/ops/max_unpool3d.h",
//                "ATen/ops/max_unpool3d_backward.h",
                "ATen/ops/maximum.h",
                "ATen/ops/mean.h",
                "ATen/ops/median.h",
                "ATen/ops/meshgrid.h",
                "ATen/ops/min.h",
                "ATen/ops/minimum.h",
                "ATen/ops/miopen_batch_norm.h",
                "ATen/ops/miopen_batch_norm_backward.h",
                "ATen/ops/miopen_convolution.h",
                "ATen/ops/miopen_convolution_transpose.h",
                "ATen/ops/miopen_depthwise_convolution.h",
                "ATen/ops/miopen_rnn.h",
                "ATen/ops/miopen_rnn_backward.h",
                "ATen/ops/mish.h",
                "ATen/ops/mish_backward.h",
                "ATen/ops/mkldnn_adaptive_avg_pool2d.h",
                "ATen/ops/mkldnn_adaptive_avg_pool2d_backward.h",
                "ATen/ops/mkldnn_convolution.h",
                "ATen/ops/mkldnn_linear.h",
                "ATen/ops/mkldnn_linear_backward.h",
                "ATen/ops/mkldnn_linear_backward_input.h",
                "ATen/ops/mkldnn_linear_backward_weights.h",
                "ATen/ops/mkldnn_max_pool2d.h",
                "ATen/ops/mkldnn_max_pool2d_backward.h",
                "ATen/ops/mkldnn_max_pool3d.h",
                "ATen/ops/mkldnn_max_pool3d_backward.h",
                "ATen/ops/mkldnn_reorder_conv2d_weight.h",
                "ATen/ops/mkldnn_reorder_conv3d_weight.h",
                "ATen/ops/mm.h",
                "ATen/ops/mode.h",
                "ATen/ops/moveaxis.h",
                "ATen/ops/movedim.h",
                "ATen/ops/mps_convolution_backward.h",
                "ATen/ops/mps_convolution_transpose_backward.h",
                "ATen/ops/mps_linear_backward.h",
                "ATen/ops/mps_max_pool2d_backward.h",
                "ATen/ops/mse_loss.h",
                "ATen/ops/mse_loss_backward.h",
                "ATen/ops/msort.h",
                "ATen/ops/mul.h",
                "ATen/ops/multi_margin_loss.h",
                "ATen/ops/multi_margin_loss_backward.h",
                "ATen/ops/multilabel_margin_loss.h",
                "ATen/ops/multilabel_margin_loss_backward.h",
                "ATen/ops/multilabel_margin_loss_forward.h",
                "ATen/ops/multinomial.h",
                "ATen/ops/multiply.h",
                "ATen/ops/mv.h",
                "ATen/ops/mvlgamma.h",
                "ATen/ops/nan_to_num.h",
                "ATen/ops/nanmean.h",
                "ATen/ops/nanmedian.h",
                "ATen/ops/nanquantile.h",
                "ATen/ops/nansum.h",
                "ATen/ops/narrow.h",
                "ATen/ops/narrow_copy.h",
                "ATen/ops/native_batch_norm.h",
                "ATen/ops/native_batch_norm_backward.h",
                "ATen/ops/native_channel_shuffle.h",
                "ATen/ops/native_dropout.h",
                "ATen/ops/native_dropout_backward.h",
                "ATen/ops/native_group_norm.h",
                "ATen/ops/native_group_norm_backward.h",
                "ATen/ops/native_layer_norm.h",
                "ATen/ops/native_layer_norm_backward.h",
                "ATen/ops/native_norm.h",
                "ATen/ops/ne.h",
                "ATen/ops/neg.h",
                "ATen/ops/negative.h",
                "ATen/ops/nested_tensor.h",
                "ATen/ops/new_empty.h",
                "ATen/ops/new_empty_strided.h",
                "ATen/ops/new_full.h",
                "ATen/ops/new_ones.h",
                "ATen/ops/new_zeros.h",
                "ATen/ops/nextafter.h",
                "ATen/ops/nll_loss.h",
                "ATen/ops/nll_loss2d.h",
                "ATen/ops/nll_loss2d_backward.h",
                "ATen/ops/nll_loss2d_forward.h",
                "ATen/ops/nll_loss_backward.h",
                "ATen/ops/nll_loss_forward.h",
                "ATen/ops/nll_loss_nd.h",
                "ATen/ops/nonzero.h",
                "ATen/ops/nonzero_numpy.h",
                "ATen/ops/norm.h",
                "ATen/ops/norm_except_dim.h",
                "ATen/ops/normal.h",
                "ATen/ops/not_equal.h",
                "ATen/ops/nuclear_norm.h",
                "ATen/ops/numpy_T.h",
                "ATen/ops/one_hot.h",
                "ATen/ops/ones.h",
                "ATen/ops/ones_like.h",
                "ATen/ops/or.h",
                "ATen/ops/orgqr.h",
                "ATen/ops/ormqr.h",
                "ATen/ops/outer.h",
                "ATen/ops/output_nr.h",
                "ATen/ops/pad.h",
                "ATen/ops/pad_sequence.h",
                "ATen/ops/pairwise_distance.h",
                "ATen/ops/pdist.h",
                "ATen/ops/permute.h",
                "ATen/ops/permute_copy.h",
                "ATen/ops/pin_memory.h",
                "ATen/ops/pinverse.h",
                "ATen/ops/pixel_shuffle.h",
                "ATen/ops/pixel_unshuffle.h",
                "ATen/ops/poisson.h",
                "ATen/ops/poisson_nll_loss.h",
                "ATen/ops/polar.h",
                "ATen/ops/polygamma.h",
                "ATen/ops/positive.h",
                "ATen/ops/pow.h",
                "ATen/ops/prelu.h",
                "ATen/ops/prelu_backward.h",
                "ATen/ops/prod.h",
                "ATen/ops/promote_types.h",
                "ATen/ops/put.h",
                "ATen/ops/q_per_channel_axis.h",
                "ATen/ops/q_per_channel_scales.h",
                "ATen/ops/q_per_channel_zero_points.h",
                "ATen/ops/q_scale.h",
                "ATen/ops/q_zero_point.h",
                "ATen/ops/qr.h",
                "ATen/ops/qscheme.h",
                "ATen/ops/quantile.h",
                "ATen/ops/quantize_per_channel.h",
                "ATen/ops/quantize_per_tensor.h",
                "ATen/ops/quantize_per_tensor_dynamic.h",
                "ATen/ops/quantized_batch_norm.h",
                "ATen/ops/quantized_gru_cell.h",
                "ATen/ops/quantized_lstm_cell.h",
                "ATen/ops/quantized_max_pool1d.h",
                "ATen/ops/quantized_max_pool2d.h",
                "ATen/ops/quantized_rnn_relu_cell.h",
                "ATen/ops/quantized_rnn_tanh_cell.h",
                "ATen/ops/rad2deg.h",
                "ATen/ops/rand.h",
                "ATen/ops/rand_like.h",
                "ATen/ops/randint.h",
                "ATen/ops/randint_like.h",
                "ATen/ops/randn.h",
                "ATen/ops/randn_like.h",
                "ATen/ops/random.h",
                "ATen/ops/randperm.h",
                "ATen/ops/range.h",
                "ATen/ops/ravel.h",
                "ATen/ops/real.h",
                "ATen/ops/reciprocal.h",
                "ATen/ops/record_stream.h",
                "ATen/ops/refine_names.h",
                "ATen/ops/reflection_pad1d.h",
                "ATen/ops/reflection_pad1d_backward.h",
                "ATen/ops/reflection_pad2d.h",
                "ATen/ops/reflection_pad2d_backward.h",
                "ATen/ops/reflection_pad3d.h",
                "ATen/ops/reflection_pad3d_backward.h",
                "ATen/ops/relu.h",
                "ATen/ops/relu6.h",
                "ATen/ops/remainder.h",
                "ATen/ops/rename.h",
                "ATen/ops/renorm.h",
                "ATen/ops/repeat.h",
                "ATen/ops/repeat_interleave.h",
                "ATen/ops/replication_pad1d.h",
                "ATen/ops/replication_pad1d_backward.h",
                "ATen/ops/replication_pad2d.h",
                "ATen/ops/replication_pad2d_backward.h",
                "ATen/ops/replication_pad3d.h",
                "ATen/ops/replication_pad3d_backward.h",
                "ATen/ops/requires_grad.h",
                "ATen/ops/reshape.h",
                "ATen/ops/reshape_as.h",
                "ATen/ops/resize.h",
                "ATen/ops/resize_as.h",
                "ATen/ops/resize_as_sparse.h",
                "ATen/ops/resolve_conj.h",
                "ATen/ops/resolve_neg.h",
                "ATen/ops/result_type.h",
                "ATen/ops/retain_grad.h",
                "ATen/ops/retains_grad.h",
                "ATen/ops/rnn_relu.h",
                "ATen/ops/rnn_relu_cell.h",
                "ATen/ops/rnn_tanh.h",
                "ATen/ops/rnn_tanh_cell.h",
                "ATen/ops/roll.h",
                "ATen/ops/rot90.h",
                "ATen/ops/round.h",
                "ATen/ops/row_indices.h",
                "ATen/ops/row_indices_copy.h",
                "ATen/ops/row_stack.h",
                "ATen/ops/rrelu.h",
                "ATen/ops/rrelu_with_noise.h",
                "ATen/ops/rrelu_with_noise_backward.h",
                "ATen/ops/rshift.h",
                "ATen/ops/rsqrt.h",
                "ATen/ops/rsub.h",
                "ATen/ops/scalar_tensor.h",
                "ATen/ops/scatter.h",
                "ATen/ops/scatter_add.h",
                "ATen/ops/scatter_reduce.h",
                "ATen/ops/searchsorted.h",
                "ATen/ops/segment_reduce.h",
                "ATen/ops/select.h",
                "ATen/ops/select_backward.h",
                "ATen/ops/select_copy.h",
                "ATen/ops/select_scatter.h",
                "ATen/ops/selu.h",
                "ATen/ops/set.h",
                "ATen/ops/set_data.h",
                "ATen/ops/sgn.h",
                "ATen/ops/sigmoid.h",
                "ATen/ops/sigmoid_backward.h",
                "ATen/ops/sign.h",
                "ATen/ops/signbit.h",
                "ATen/ops/silu.h",
                "ATen/ops/silu_backward.h",
                "ATen/ops/sin.h",
                "ATen/ops/sinc.h",
                "ATen/ops/sinh.h",
                "ATen/ops/size.h",
                "ATen/ops/slice.h",
                "ATen/ops/slice_backward.h",
                "ATen/ops/slice_copy.h",
                "ATen/ops/slice_scatter.h",
                "ATen/ops/slogdet.h",
                "ATen/ops/slow_conv3d.h",
                "ATen/ops/slow_conv3d_forward.h",
                "ATen/ops/slow_conv_dilated2d.h",
                "ATen/ops/slow_conv_dilated3d.h",
                "ATen/ops/slow_conv_transpose2d.h",
                "ATen/ops/slow_conv_transpose3d.h",
                "ATen/ops/smm.h",
                "ATen/ops/smooth_l1_loss.h",
                "ATen/ops/smooth_l1_loss_backward.h",
                "ATen/ops/soft_margin_loss.h",
                "ATen/ops/soft_margin_loss_backward.h",
                "ATen/ops/softmax.h",
                "ATen/ops/softplus.h",
                "ATen/ops/softplus_backward.h",
                "ATen/ops/softshrink.h",
                "ATen/ops/softshrink_backward.h",
//                "ATen/ops/solve.h",
                "ATen/ops/sort.h",
                "ATen/ops/sparse_bsc_tensor.h",
                "ATen/ops/sparse_bsr_tensor.h",
                "ATen/ops/sparse_compressed_tensor.h",
                "ATen/ops/sparse_coo_tensor.h",
                "ATen/ops/sparse_csc_tensor.h",
                "ATen/ops/sparse_csr_tensor.h",
                "ATen/ops/sparse_dim.h",
                "ATen/ops/sparse_mask.h",
                "ATen/ops/sparse_resize.h",
                "ATen/ops/sparse_resize_and_clear.h",
                "ATen/ops/sparse_sampled_addmm.h",
                "ATen/ops/special_digamma.h",
                "ATen/ops/special_entr.h",
                "ATen/ops/special_erf.h",
                "ATen/ops/special_erfc.h",
                "ATen/ops/special_erfcx.h",
                "ATen/ops/special_erfinv.h",
                "ATen/ops/special_exp2.h",
                "ATen/ops/special_expit.h",
                "ATen/ops/special_expm1.h",
                "ATen/ops/special_gammainc.h",
                "ATen/ops/special_gammaincc.h",
                "ATen/ops/special_gammaln.h",
                "ATen/ops/special_i0.h",
                "ATen/ops/special_i0e.h",
                "ATen/ops/special_i1.h",
                "ATen/ops/special_i1e.h",
                "ATen/ops/special_log1p.h",
                "ATen/ops/special_log_ndtr.h",
                "ATen/ops/special_log_softmax.h",
                "ATen/ops/special_logit.h",
                "ATen/ops/special_logsumexp.h",
                "ATen/ops/special_multigammaln.h",
                "ATen/ops/special_ndtr.h",
                "ATen/ops/special_ndtri.h",
                "ATen/ops/special_polygamma.h",
                "ATen/ops/special_psi.h",
                "ATen/ops/special_round.h",
                "ATen/ops/special_sinc.h",
                "ATen/ops/special_softmax.h",
                "ATen/ops/special_xlog1py.h",
                "ATen/ops/special_xlogy.h",
                "ATen/ops/special_zeta.h",
                "ATen/ops/split.h",
                "ATen/ops/split_copy.h",
                "ATen/ops/split_with_sizes.h",
                "ATen/ops/split_with_sizes_copy.h",
                "ATen/ops/sqrt.h",
                "ATen/ops/square.h",
                "ATen/ops/squeeze.h",
                "ATen/ops/squeeze_copy.h",
                "ATen/ops/sspaddmm.h",
                "ATen/ops/stack.h",
                "ATen/ops/std.h",
                "ATen/ops/std_mean.h",
                "ATen/ops/stft.h",
                "ATen/ops/stride.h",
                "ATen/ops/sub.h",
                "ATen/ops/subtract.h",
                "ATen/ops/sum.h",
                "ATen/ops/sum_to_size.h",
                "ATen/ops/svd.h",
                "ATen/ops/swapaxes.h",
                "ATen/ops/swapdims.h",
                "ATen/ops/symeig.h",
                "ATen/ops/t.h",
                "ATen/ops/t_copy.h",
                "ATen/ops/take.h",
                "ATen/ops/take_along_dim.h",
                "ATen/ops/tan.h",
                "ATen/ops/tanh.h",
                "ATen/ops/tanh_backward.h",
                "ATen/ops/tensor_split.h",
                "ATen/ops/tensordot.h",
                "ATen/ops/thnn_conv2d.h",
                "ATen/ops/threshold.h",
                "ATen/ops/threshold_backward.h",
                "ATen/ops/tile.h",
                "ATen/ops/to.h",
                "ATen/ops/to_dense.h",
                "ATen/ops/to_dense_backward.h",
                "ATen/ops/to_mkldnn.h",
                "ATen/ops/to_mkldnn_backward.h",
                "ATen/ops/to_padded_tensor.h",
                "ATen/ops/to_sparse.h",
                "ATen/ops/to_sparse_bsc.h",
                "ATen/ops/to_sparse_bsr.h",
                "ATen/ops/to_sparse_csc.h",
                "ATen/ops/to_sparse_csr.h",
                "ATen/ops/topk.h",
                "ATen/ops/trace.h",
                "ATen/ops/trace_backward.h",
                "ATen/ops/transpose.h",
                "ATen/ops/transpose_copy.h",
                "ATen/ops/trapezoid.h",
                "ATen/ops/trapz.h",
                "ATen/ops/triangular_solve.h",
                "ATen/ops/tril.h",
                "ATen/ops/tril_indices.h",
                "ATen/ops/triplet_margin_loss.h",
                "ATen/ops/triu.h",
                "ATen/ops/triu_indices.h",
                "ATen/ops/true_divide.h",
                "ATen/ops/trunc.h",
                "ATen/ops/type_as.h",
                "ATen/ops/unbind.h",
                "ATen/ops/unbind_copy.h",
                "ATen/ops/unflatten.h",
                "ATen/ops/unflatten_dense_tensors.h",
                "ATen/ops/unfold.h",
                "ATen/ops/unfold_backward.h",
                "ATen/ops/unfold_copy.h",
                "ATen/ops/uniform.h",
                "ATen/ops/unique_consecutive.h",
                "ATen/ops/unique_dim.h",
                "ATen/ops/unique_dim_consecutive.h",
                "ATen/ops/unsafe_chunk.h",
                "ATen/ops/unsafe_split.h",
                "ATen/ops/unsafe_split_with_sizes.h",
                "ATen/ops/unsqueeze.h",
                "ATen/ops/unsqueeze_copy.h",
                "ATen/ops/upsample_bicubic2d.h",
                "ATen/ops/upsample_bicubic2d_backward.h",
                "ATen/ops/upsample_bilinear2d.h",
                "ATen/ops/upsample_bilinear2d_backward.h",
                "ATen/ops/upsample_linear1d.h",
                "ATen/ops/upsample_linear1d_backward.h",
                "ATen/ops/upsample_nearest1d.h",
                "ATen/ops/upsample_nearest1d_backward.h",
                "ATen/ops/upsample_nearest2d.h",
                "ATen/ops/upsample_nearest2d_backward.h",
                "ATen/ops/upsample_nearest3d.h",
                "ATen/ops/upsample_nearest3d_backward.h",
                "ATen/ops/upsample_trilinear3d.h",
                "ATen/ops/upsample_trilinear3d_backward.h",
                "ATen/ops/value_selecting_reduction_backward.h",
                "ATen/ops/values.h",
                "ATen/ops/values_copy.h",
                "ATen/ops/vander.h",
                "ATen/ops/var.h",
                "ATen/ops/var_mean.h",
                "ATen/ops/vdot.h",
                "ATen/ops/view.h",
                "ATen/ops/view_as.h",
                "ATen/ops/view_as_complex.h",
                "ATen/ops/view_as_complex_copy.h",
                "ATen/ops/view_as_real.h",
                "ATen/ops/view_as_real_copy.h",
                "ATen/ops/view_copy.h",
                "ATen/ops/vsplit.h",
                "ATen/ops/vstack.h",
                "ATen/ops/where.h",
                "ATen/ops/xlogy.h",
                "ATen/ops/xor.h",
                "ATen/ops/zero.h",
                "ATen/ops/zeros.h",
                "ATen/ops/zeros_like.h",
                "ATen/ops/values.h",
                "ATen/ops/vander.h",
                "ATen/ops/var.h",
                "ATen/ops/var_mean.h",
                "ATen/ops/vdot.h",
                "ATen/ops/view.h",
                "ATen/ops/view_as.h",
                "ATen/ops/view_as_complex.h",
                "ATen/ops/view_as_real.h",
                "ATen/ops/vsplit.h",
                "ATen/ops/vstack.h",
                "ATen/ops/where.h",
                "ATen/ops/xlogy.h",
                "ATen/ops/xor.h",
                "ATen/ops/zero.h",
                "ATen/ops/zeros.h",
                "ATen/ops/zeros_like.h",

                "torch/autograd.h",
//                "torch/library.h",
//                "torch/custom_class.h",
                "torch/script.h",
                "torch/csrc/Export.h",
                "torch/csrc/onnx/onnx.h",
//                "torch/csrc/WindowsTorchApiMacro.h",
                "torch/csrc/api/include/torch/imethod.h",
                "torch/csrc/api/include/torch/types.h",
                "torch/csrc/api/include/torch/cuda.h",
                "torch/csrc/api/include/torch/ordered_dict.h",
//                "torch/csrc/api/include/torch/detail/TensorDataContainer.h",
                "torch/csrc/utils/disallow_copy.h",
                "torch/csrc/utils/memory.h",
                "torch/csrc/utils/python_stub.h",
//                "torch/csrc/utils/object_ptr.h",
                "torch/csrc/utils/variadic.h",
                "torch/csrc/autograd/anomaly_mode.h",
                "torch/csrc/autograd/edge.h",
                "torch/csrc/autograd/grad_mode.h",
                "torch/csrc/autograd/InferenceMode.h",
                "torch/csrc/autograd/input_metadata.h",
                "torch/csrc/autograd/function_hook.h",
                "torch/csrc/autograd/cpp_hook.h",
                "torch/csrc/autograd/profiler.h",
                "torch/csrc/autograd/saved_variable_hooks.h",
                "torch/csrc/autograd/saved_variable.h",
                "torch/csrc/autograd/forward_grad.h",
                "torch/csrc/autograd/variable.h",
                "torch/csrc/autograd/function.h",
                "torch/csrc/autograd/custom_function.h",
                "torch/csrc/autograd/autograd.h",
//                "torch/csrc/autograd/generated/Functions.h",
                "torch/csrc/autograd/generated/VariableType.h",
                "torch/csrc/autograd/generated/variable_factories.h",
                "torch/csrc/jit/frontend/function_schema_parser.h",
                "torch/csrc/jit/frontend/name_mangler.h",
                "torch/csrc/jit/frontend/parser_constants.h",
                "torch/csrc/jit/frontend/source_range.h",
                "torch/csrc/jit/frontend/sugared_value.h",
                "torch/csrc/jit/frontend/resolver.h",
                "torch/csrc/jit/frontend/tracer.h",
                "torch/csrc/jit/frontend/lexer.h",
                "torch/csrc/jit/frontend/strtod.h",
                "torch/csrc/jit/frontend/tree.h",
                "torch/csrc/jit/frontend/error_report.h",
                "torch/csrc/jit/frontend/tree_views.h",
                "torch/csrc/jit/ir/attributes.h",
                "torch/csrc/jit/ir/constants.h",
                "torch/csrc/jit/ir/graph_node_list.h",
                "torch/csrc/jit/ir/named_value.h",
                "torch/csrc/jit/ir/scope.h",
                "torch/csrc/jit/ir/ir.h",
                "torch/csrc/jit/ir/type_hashing.h",
                "torch/csrc/jit/passes/shape_analysis.h",
                "torch/csrc/jit/python/update_graph_executor_opt.h",
                "torch/csrc/jit/runtime/argument_spec.h",
                "torch/csrc/jit/runtime/instruction.h",
                "torch/csrc/jit/runtime/interpreter.h",
//                "torch/csrc/jit/runtime/variable_tensor_list.h",
                "torch/csrc/jit/runtime/graph_executor.h",
                "torch/csrc/jit/runtime/operator_options.h",
                "torch/csrc/jit/runtime/operator.h",
                "torch/csrc/jit/runtime/custom_operator.h",
                "torch/csrc/jit/api/compilation_unit.h",
                "torch/csrc/jit/api/function_impl.h",
                "torch/csrc/jit/api/method.h",
                "torch/csrc/jit/api/object.h",
                "torch/csrc/jit/api/module.h",
                "torch/csrc/jit/serialization/source_range_serialization.h",
                "torch/csrc/jit/serialization/pickler.h",
                "torch/csrc/jit/serialization/unpickler.h",
                "torch/csrc/jit/serialization/import.h",
                "torch/csrc/jit/serialization/pickle.h",
                "torch/csrc/jit/serialization/python_print.h",
                "torch/csrc/jit/serialization/type_name_uniquer.h",
                "torch/csrc/jit/serialization/storage_context.h",
                "torch/csrc/jit/serialization/export.h",

                "torch/arg.h",
                "torch/enum.h",
                "torch/types.h",
                "torch/utils.h",

                "torch/data.h",
                "torch/data/example.h",
                "torch/data/iterator.h",
                "torch/data/worker_exception.h",
                "torch/data/dataloader.h",
                "torch/data/dataloader/base.h",
                "torch/data/dataloader_options.h",
                "torch/data/dataloader/stateful.h",
                "torch/data/dataloader/stateless.h",
                "torch/data/datasets.h",
                "torch/data/datasets/base.h",
//                "torch/data/datasets/chunk.h",
                "torch/data/datasets/map.h",
                "torch/data/datasets/mnist.h",
//                "torch/data/datasets/shared.h",
//                "torch/data/datasets/stateful.h",
//                "torch/data/datasets/tensor.h",
                "torch/data/samplers.h",
                "torch/data/samplers/base.h",
//                "torch/data/samplers/custom_batch_request.h",
//                "torch/data/samplers/distributed.h",
                "torch/data/samplers/random.h",
//                "torch/data/samplers/sequential.h",
//                "torch/data/samplers/serialize.h",
//                "torch/data/samplers/stream.h",
                "torch/data/transforms.h",
                "torch/data/transforms/base.h",
                "torch/data/transforms/collate.h",
                "torch/data/transforms/lambda.h",
                "torch/data/transforms/stack.h",
                "torch/data/transforms/tensor.h",

                "torch/serialize.h",
                "torch/serialize/archive.h",
                "torch/serialize/input-archive.h",
                "torch/serialize/output-archive.h",
                "torch/serialize/tensor.h",

                "torch/nn.h",
                "torch/nn/cloneable.h",
                "torch/nn/init.h",
                "torch/nn/pimpl.h",
                "torch/nn/utils.h",
                "torch/nn/utils/clip_grad.h",
                "torch/nn/utils/convert_parameters.h",
                "torch/nn/utils/rnn.h",

                "torch/nn/options.h",
                "torch/nn/options/activation.h",
                "torch/nn/options/adaptive.h",
                "torch/nn/options/batchnorm.h",
                "torch/nn/options/conv.h",
                "torch/nn/options/distance.h",
                "torch/nn/options/dropout.h",
                "torch/nn/options/embedding.h",
                "torch/nn/options/fold.h",
                "torch/nn/options/linear.h",
                "torch/nn/options/loss.h",
                "torch/nn/options/normalization.h",
                "torch/nn/options/padding.h",
                "torch/nn/options/pixelshuffle.h",
                "torch/nn/options/pooling.h",
                "torch/nn/options/rnn.h",
                "torch/nn/options/upsampling.h",
                "torch/nn/options/vision.h",
                "torch/nn/options/instancenorm.h",
                "torch/nn/options/transformerlayer.h",
                "torch/nn/options/transformercoder.h",
                "torch/nn/options/transformer.h",

                "torch/nn/functional.h",
                "torch/nn/functional/activation.h",
                "torch/nn/functional/batchnorm.h",
                "torch/nn/functional/conv.h",
                "torch/nn/functional/distance.h",
                "torch/nn/functional/dropout.h",
                "torch/nn/functional/embedding.h",
                "torch/nn/functional/fold.h",
                "torch/nn/functional/linear.h",
                "torch/nn/functional/loss.h",
                "torch/nn/functional/normalization.h",
                "torch/nn/functional/padding.h",
                "torch/nn/functional/pixelshuffle.h",
                "torch/nn/functional/pooling.h",
                "torch/nn/functional/upsampling.h",
                "torch/nn/functional/vision.h",
                "torch/nn/functional/instancenorm.h",

                "torch/nn/module.h",
                "torch/nn/modules.h",
                "torch/nn/modules/common.h",

                "torch/nn/modules/container/any.h",
//                "torch/nn/modules/container/functional.h",
                "torch/nn/modules/container/moduledict.h",
                "torch/nn/modules/container/modulelist.h",
                "torch/nn/modules/container/named_any.h",
                "torch/nn/modules/container/sequential.h",
                "torch/nn/modules/container/parameterdict.h",
                "torch/nn/modules/container/parameterlist.h",

                "torch/nn/modules/adaptive.h",
                "torch/nn/modules/batchnorm.h",
                "torch/nn/modules/instancenorm.h",
                "torch/nn/modules/conv.h",
                "torch/nn/modules/dropout.h",
                "torch/nn/modules/distance.h",
                "torch/nn/modules/embedding.h",
                "torch/nn/modules/fold.h",
                "torch/nn/modules/linear.h",
                "torch/nn/modules/loss.h",
                "torch/nn/modules/padding.h",
                "torch/nn/modules/pooling.h",
                "torch/nn/modules/rnn.h",
                "torch/nn/modules/pixelshuffle.h",
                "torch/nn/modules/upsampling.h",
                "torch/nn/modules/activation.h",
                "torch/nn/modules/normalization.h",
                "torch/nn/modules/transformerlayer.h",
                "torch/nn/modules/transformercoder.h",
                "torch/nn/modules/transformer.h",

                "torch/optim.h",
                "torch/optim/optimizer.h",
                "torch/optim/serialize.h",
                "torch/optim/adagrad.h",
                "torch/optim/adam.h",
                "torch/optim/adamw.h",
                "torch/optim/lbfgs.h",
                "torch/optim/rmsprop.h",
                "torch/optim/sgd.h",
                "torch/optim/schedulers/lr_scheduler.h",
                "torch/optim/schedulers/step_lr.h",
            },
            exclude = {
                "ATen/core/UnsafeFromTH.h",
                "torch/csrc/jit/api/method.h",
            },
            link = {"c10", "torch_cpu", "torch"},
            preload = {"gomp@.1", "iomp5", "omp", "tbb@.2", "asmjit", "fbgemm"}
        ),
        @Platform(
            value = {"linux", "macosx", "windows"},
            link = {"c10", "c10_cuda", "torch_cpu", "torch_cuda", "torch"},
            preloadpath = "C:/Program Files/NVIDIA Corporation/NvToolsExt/bin/x64/",
            extension = "-gpu"
        ),
    },
    target = "org.bytedeco.pytorch",
    global = "org.bytedeco.pytorch.global.torch"
)
public class torch implements LoadEnabled, InfoMapper {
    static { Loader.checkVersion("org.bytedeco", "pytorch"); }

    @Override public void init(ClassProperties properties) {
        String platform = properties.getProperty("platform");
        String extension = properties.getProperty("platform.extension");
        List<String> preloads = properties.get("platform.preload");
        List<String> resources = properties.get("platform.preloadresource");

        // Only apply this at load time since we don't want to copy the CUDA libraries here
        if (!Loader.isLoadLibraries() || extension == null || !extension.endsWith("-gpu")) {
            return;
        }
        int i = 0;
        if (platform.startsWith("windows")) {
            preloads.add(i++, "zlibwapi");
        }
        String[] libs = {"cudart", "cublasLt", "cublas", "cufft", "curand", "cusolver", "cusparse", "cudnn", "nccl", "nvrtc", "myelin", "nvinfer",
                         "cudnn_ops_infer", "cudnn_ops_train", "cudnn_adv_infer", "cudnn_adv_train", "cudnn_cnn_infer", "cudnn_cnn_train"};
        for (String lib : libs) {
            if (platform.startsWith("linux")) {
                lib += lib.startsWith("cudnn") ? "@.8"
                     : lib.equals("nccl") ? "@.2"
                     : lib.equals("myelin") ? "@.1"
                     : lib.equals("nvinfer") ? "@.7"
                     : lib.equals("cufft") || lib.equals("curand") ? "@.10"
                     : lib.equals("cudart") ? "@.11.0"
                     : lib.equals("nvrtc") ? "@.11.2"
                     : "@.11";
            } else if (platform.startsWith("windows")) {
                lib += lib.startsWith("cudnn") ? "64_8"
                     : lib.equals("nccl") ? "64_2"
                     : lib.equals("myelin") ? "64_1"
                     : lib.equals("nvinfer") ? "64_7"
                     : lib.equals("cufft") || lib.equals("curand") ? "64_10"
                     : lib.equals("cudart") ? "64_110"
                     : lib.equals("nvrtc") ? "64_112_0"
                     : "64_11";
            } else {
                continue; // no CUDA
            }
            if (!preloads.contains(lib)) {
                preloads.add(i++, lib);
            }
        }
        preloads.add("nvToolsExt@.1");
        preloads.add("nvToolsExt64_1");
        if (i > 0) {
            resources.add("/org/bytedeco/cuda/");
            resources.add("/org/bytedeco/tensorrt/");
        }
    }

    public void mapModule(InfoMap infoMap, String name) {
        mapModule(infoMap, name, false);
    }
    public void mapModule(InfoMap infoMap, String name, String base) {
        mapModule(infoMap, name, base, false);
    }
    public void mapModule(InfoMap infoMap, String name, String base, String baseBase) {
        mapModule(infoMap, name, base, baseBase, false);
    }
    public void mapModule(InfoMap infoMap, String name, boolean hasDefaultConstructor) {
        mapModule(infoMap, name, null, hasDefaultConstructor);
    }
    public void mapModule(InfoMap infoMap, String name, String base, boolean hasDefaultConstructor) {
        mapModule(infoMap, name, base, null, hasDefaultConstructor);
    }
    public void mapModule(InfoMap infoMap, String name, String base, String baseBase, boolean hasDefaultConstructor) {
        if (baseBase != null) {
            infoMap.put(new Info(baseBase).pointerTypes(name + "ImplBaseBase"));
        }

        if (base != null) {
            int template = base.indexOf('<');
            int namespace = base.lastIndexOf("::", template);
            infoMap.put(new Info(base + base.substring(namespace, template)).annotations("@NoDeallocator"))
                   .put(new Info(base, base.replace("torch::nn::" + name + "Impl", name + "Impl")).purify(baseBase != null).pointerTypes(name + "ImplBase"));
        }

        infoMap.put(new Info("torch::nn::" + name + "Impl::" + name + "Impl").annotations("@NoDeallocator"))
               .put(new Info("std::shared_ptr<torch::nn::" + name + "Impl>").annotations("@SharedPtr")
                       .valueTypes("@Cast({\"\", \"std::shared_ptr<torch::nn::" + name + "Impl>\"}) " + name + "Impl").pointerTypes(name + "Impl"))
               .put(new Info("torch::nn::Cloneable<torch::nn::" + name + "Impl>",
                             "torch::nn::Cloneable<" + name + "Impl>").pointerTypes(name + "ImplCloneable"))
               .put(new Info("torch::nn::Cloneable<torch::nn::" + name + "Impl>::reset").javaText("public native void reset();\n"
                     + "@Override public Module asModule() { return asModule(this); }\n"
                     + "@Namespace public static native @Name(\"static_cast<torch::nn::Module*>\") Module asModule(" + name + "ImplCloneable module);\n"))
               .put(new Info("torch::nn::ModuleHolder<torch::nn::" + name + "Impl>").pointerTypes(name + "ImplModuleHolder"))
               .put(new Info("torch::nn::Module::register_module<torch::nn::" + name + "Impl>").javaNames("register_module"));

        if (!hasDefaultConstructor) {
            infoMap.put(new Info("torch::nn::ModuleHolder<torch::nn::" + name + "Impl>()").skip());
        }
    }

    public void map(InfoMap infoMap) {
        infoMap.putFirst(new Info("openblas_config.h", "cblas.h", "lapacke_config.h", "lapacke_mangling.h", "lapack.h", "lapacke.h", "lapacke_utils.h").skip())
               .put(new Info("ordered_dict.h").linePatterns(".*class Item;.*").skip())
               .put(new Info().enumerate())
               .put(new Info().javaText("import org.bytedeco.pytorch.Allocator;"))
               .put(new Info().javaText("import org.bytedeco.pytorch.Function;"))
               .put(new Info().javaText("import org.bytedeco.pytorch.Module;"))

               .put(new Info("basic/containers").cppTypes("c10::optional", "torch::optional", "c10::variant"))
               .put(new Info("std::nullptr_t").cast().pointerTypes("PointerPointer"))
               .put(new Info("auto", "c10::reverse_iterator", "ska::flat_hash_map", "std::atomic", "std::conditional", "std::iterator_traits",
                             "std::initializer_list", "std::integral_constant", "std::mutex", "std::reverse_iterator", "std::weak_ptr").skip())
               .put(new Info("at::CheckedFrom").cast().valueTypes("BytePointer", "String").pointerTypes("PointerPointer"))
               .put(new Info("c10::IValue", "at::IValue").pointerTypes("IValue"))
               .put(new Info("c10::ScalarType", "at::ScalarType", "torch::Dtype").enumerate().valueTypes("ScalarType").pointerTypes("@Cast(\"c10::ScalarType*\") BytePointer"))
               .put(new Info("torch::jit::AttributeKind").enumerate().valueTypes("JitAttributeKind"))
               .put(new Info("torch::jit::PickleOpCode").enumerate().translate(false).valueTypes("PickleOpCode"))
               .put(new Info("std::size_t", "c10::Dict<c10::IValue,c10::IValue>::size_type").cast().valueTypes("long").pointerTypes("SizeTPointer"))
               .put(new Info("std::tuple<int64_t,int64_t>", "std::tuple<double,int64_t>",
                             "torch::ExpandingArray<1>", "torch::ExpandingArray<2>", "torch::ExpandingArray<3>", "torch::ExpandingArray<4>",
                             "torch::ExpandingArray<D*2>", "torch::ExpandingArray<1*2>", "torch::ExpandingArray<2*2>", "torch::ExpandingArray<3*2>",
                             "torch::ExpandingArrayWithOptionalElem<2>", "torch::ExpandingArrayWithOptionalElem<3>").cast().pointerTypes("LongPointer"))
               .put(new Info("torch::ExpandingArray<1,double>", "torch::ExpandingArray<2,double>", "torch::ExpandingArray<3,double>").cast().pointerTypes("DoublePointer"))
               .put(new Info("std::array<bool,2>", "std::array<bool,3>", "std::array<bool,4>").cast().pointerTypes("BoolPointer"))
               .put(new Info("std::pair<std::string,c10::IValue>").pointerTypes("EnumNameValue").define())
               .put(new Info("c10::ClassType::Property").pointerTypes("ClassType.Property"))
               .put(new Info("c10::optional<bool>").pointerTypes("BoolOptional").define())
               .put(new Info("c10::optional<int8_t>").pointerTypes("ByteOptional").define())
               .put(new Info("c10::optional<int>", "c10::optional<int32_t>").pointerTypes("IntOptional").define())
               .put(new Info("c10::optional<int64_t>").pointerTypes("LongOptional").define())
               .put(new Info("c10::optional<double>").pointerTypes("DoubleOptional").define())
               .put(new Info("c10::optional<size_t>").pointerTypes("SizeTOptional").define())
               .put(new Info("c10::optional<std::string>").pointerTypes("StringOptional").define())
               .put(new Info("c10::optional<std::vector<bool> >").pointerTypes("BoolVectorOptional").define())
               .put(new Info("c10::optional<std::vector<int64_t> >").pointerTypes("LongVectorOptional").define())
               .put(new Info("c10::optional<std::vector<double> >").pointerTypes("DoubleVectorOptional").define())
               .put(new Info("c10::optional<std::vector<size_t> >").pointerTypes("SizeTVectorOptional").define())
               .put(new Info("c10::optional<std::vector<std::string> >").pointerTypes("StringVectorOptional").define())
               .put(new Info("c10::optional<std::vector<c10::Stride> >").pointerTypes("StrideVectorOptional").define())
               .put(new Info("c10::optional<std::vector<c10::ShapeSymbol> >").pointerTypes("ShapeSymbolVectorOptional").define())
               .put(new Info("c10::optional<std::vector<at::Tensor> >").pointerTypes("TensorVectorOptional").define())
               .put(new Info("c10::optional<c10::Device>", "c10::optional<at::Device>", "c10::optional<torch::Device>").pointerTypes("DeviceOptional").define())
               .put(new Info("c10::optional<c10::ArrayRef<int64_t> >", "c10::optional<c10::IntArrayRef>", "c10::optional<at::IntArrayRef>").pointerTypes("LongArrayRefOptional").define())
               .put(new Info("c10::optional<c10::ArrayRef<double> >", "c10::optional<at::ArrayRef<double> >").pointerTypes("DoubleArrayRefOptional").define())
               .put(new Info("c10::optional<c10::Layout>", "c10::optional<at::Layout>").pointerTypes("LayoutOptional").define())
               .put(new Info("c10::optional<c10::MemoryFormat>", "c10::optional<at::MemoryFormat>").pointerTypes("MemoryFormatOptional").define())
               .put(new Info("c10::optional<c10::Scalar>", "c10::optional<at::Scalar>").pointerTypes("ScalarOptional").define())
               .put(new Info("c10::optional<c10::ScalarType>", "c10::optional<at::ScalarType>", "c10::optional<torch::Dtype>").pointerTypes("ScalarTypeOptional").define())
               .put(new Info("c10::optional<c10::AliasInfo>").pointerTypes("AliasInfoOptional").define())
               .put(new Info("c10::optional<c10::IValue>").pointerTypes("IValueOptional").define())
               .put(new Info("c10::optional<c10::impl::CppSignature>").pointerTypes("CppSignatureOptional").define())
               .put(new Info("c10::optional<c10::DispatchKey>").pointerTypes("DispatchKeyOptional").define())
               .put(new Info("c10::optional<c10::OperatorHandle>").pointerTypes("OperatorHandleOptional").define())
               .put(new Info("c10::optional<c10::OperatorName>").pointerTypes("OperatorNameOptional").define())
               .put(new Info("c10::optional<c10::QualifiedName>").pointerTypes("QualifiedNameOptional").define())
               .put(new Info("c10::optional<c10::Stream>").pointerTypes("StreamOptional").define())
               .put(new Info("c10::optional<c10::Stride>").pointerTypes("StrideOptional").define())
               .put(new Info("c10::optional<c10::TypePtr>").pointerTypes("TypePtrOptional").define())
               .put(new Info("c10::optional<c10::ClassType::Property>").pointerTypes("ClassTypePropertyOptional").define())
               .put(new Info("c10::optional<at::DimVector>").pointerTypes("DimVectorOptional").define())
               .put(new Info("c10::optional<at::Dimname>").pointerTypes("DimnameOptional").define())
               .put(new Info("c10::optional<at::DimnameList>").pointerTypes("DimnameListOptional").define())
               .put(new Info("c10::optional<at::Generator>").pointerTypes("GeneratorOptional").define())
               .put(new Info("c10::optional<at::Tensor>", "c10::optional<at::TensorBase>", "c10::optional<torch::autograd::Variable>").pointerTypes("TensorOptional").define())
               .put(new Info("c10::optional<at::TensorList>").pointerTypes("TensorListOptional").define())
               .put(new Info("c10::optional<at::ThreadLocalState>").pointerTypes("ThreadLocalStateOptional").define())
               .put(new Info("c10::optional<caffe2::TypeMeta>").pointerTypes("TypeMetaOptional").define())
               .put(new Info("c10::optional<torch::jit::ExecutorExecutionMode>").pointerTypes("ExecutorExecutionModeOptional").define())
               .put(new Info("c10::optional<torch::jit::InlinedCallStack>",
                             "c10::optional<torch::jit::InlinedCallStackPtr>").cast().pointerTypes("InlinedCallStackOptional").define())
               .put(new Info("c10::optional<torch::jit::Scope>",
                             "c10::optional<torch::jit::ScopePtr>").cast().pointerTypes("ScopeOptional").define())
               .put(new Info("c10::optional<torch::jit::ModuleInstanceInfo>").pointerTypes("ModuleInstanceInfoOptional").define())
               .put(new Info("c10::optional<torch::jit::SourceRange>").pointerTypes("SourceRangeOptional").define())
               .put(new Info("c10::optional<torch::jit::Method>").pointerTypes("MethodOptional").define())
               .put(new Info("c10::optional<torch::jit::Operator>").pointerTypes("OperatorOptional").define())
               .put(new Info("c10::optional<torch::jit::NamedValue>", "c10::optional<NamedValue>").pointerTypes("NamedValueOptional").define())
               .put(new Info("c10::optional<torch::jit::Value*>").pointerTypes("ValueOptional").define())
               .put(new Info("c10::optional<torch::ExpandingArray<1> >",
                             "c10::optional<torch::ExpandingArray<2> >",
                             "c10::optional<torch::ExpandingArray<3> >").cast().pointerTypes("LongExpandingArrayOptional").define())
               .put(new Info("c10::optional<torch::ExpandingArray<1,double> >",
                             "c10::optional<torch::ExpandingArray<2,double> >",
                             "c10::optional<torch::ExpandingArray<3,double> >",
                             "c10::optional<torch::nn::FractionalMaxPoolOptions<1>::ExpandingArrayDouble>",
                             "c10::optional<torch::nn::FractionalMaxPoolOptions<2>::ExpandingArrayDouble>",
                             "c10::optional<torch::nn::FractionalMaxPoolOptions<3>::ExpandingArrayDouble>").cast().pointerTypes("DoubleExpandingArrayOptional").define())
               .put(new Info("c10::optional<std::tuple<std::string,size_t,size_t> >").pointerTypes("StringSizeTSizeTTupleOptional").define())
               .put(new Info("torch::optional<std::tuple<at::Tensor,at::Tensor> >").pointerTypes("TensorTensorOptional").define())

               .put(new Info("c10::Type::SingletonOrSharedTypePtr<c10::Type>", "c10::TypePtr", "c10::Type::TypePtr", "decltype(auto)").pointerTypes("Type.TypePtr").define())
               .put(new Info("c10::Type::SingletonOrSharedTypePtr<c10::Type>(c10::SingletonTypePtr<c10::Type>)",
                             "c10::ComplexType::get", "c10::FloatType::get", "c10::IntType::get").skip())
               .put(new Info("c10::SingletonTypePtr<c10::Type>").pointerTypes("SingletonTypePtr").define())
               .put(new Info("c10::SingletonTypePtr<c10::AnyType>").pointerTypes("AnyTypePtr").define())
               .put(new Info("c10::SingletonTypePtr<c10::AnyEnumType>").pointerTypes("AnyEnumTypePtr").define())
               .put(new Info("c10::SingletonTypePtr<c10::NumberType>").pointerTypes("NumberTypePtr").define())
               .put(new Info("c10::SingletonTypePtr<c10::FloatType>").pointerTypes("FloatTypePtr").define())
               .put(new Info("c10::SingletonTypePtr<c10::ComplexType>").pointerTypes("ComplexTypePtr").define())
               .put(new Info("c10::SingletonTypePtr<c10::IntType>").pointerTypes("IntTypePtr").define())
               .put(new Info("c10::SingletonTypePtr<c10::BoolType>").pointerTypes("BoolTypePtr").define())
               .put(new Info("c10::SingletonTypePtr<c10::StringType>").pointerTypes("StringTypePtr").define())
               .put(new Info("c10::SingletonTypePtr<c10::StorageType>").pointerTypes("StorageTypePtr").define())
               .put(new Info("c10::SingletonTypePtr<c10::NoneType>").pointerTypes("NoneTypePtr").define())
               .put(new Info("c10::SingletonTypePtr<c10::GeneratorType>").pointerTypes("GeneratorTypePtr").define())
               .put(new Info("c10::SingletonTypePtr<c10::QuantizerType>").pointerTypes("QuantizerTypePtr").define())
               .put(new Info("c10::SingletonTypePtr<c10::QSchemeType>").pointerTypes("QSchemeTypePtr").define())
               .put(new Info("c10::SingletonTypePtr<c10::DeviceObjType>").pointerTypes("DeviceObjTypePtr").define())
               .put(new Info("c10::SingletonTypePtr<c10::StreamObjType>").pointerTypes("StreamObjTypePtr").define())
               .put(new Info("c10::SingletonTypePtr<c10::CapsuleType>").pointerTypes("CapsuleTypePtr").define())
               .put(new Info("c10::SingletonTypePtr<c10::PyObjectType>").pointerTypes("PyObjectTypePtr").define())
               .put(new Info("c10::SingletonTypePtr<c10::LayoutType>").pointerTypes("LayoutTypePtr").define())
               .put(new Info("c10::SingletonTypePtr<c10::ScalarTypeType>").pointerTypes("ScalarTypeTypePtr").define())
               .put(new Info("c10::SingletonTypePtr<c10::AnyListType>").pointerTypes("AnyListTypePtr").define())
               .put(new Info("c10::SingletonTypePtr<c10::AnyTupleType>").pointerTypes("AnyTupleTypePtr").define())
               .put(new Info("c10::SingletonTypePtr<c10::AnyClassType>").pointerTypes("AnyClassTypePtr").define())

               .put(new Info("c10::variant<torch::enumtype::kLinear,torch::enumtype::kConv1D,torch::enumtype::kConv2D,torch::enumtype::kConv3D,"
                                        + "torch::enumtype::kConvTranspose1D,torch::enumtype::kConvTranspose2D,torch::enumtype::kConvTranspose3D,"
                                        + "torch::enumtype::kSigmoid,torch::enumtype::kTanh,torch::enumtype::kReLU,torch::enumtype::kLeakyReLU>",
                             "torch::nn::init::NonlinearityType").pointerTypes("NonlinearityType").define())
               .put(new Info("c10::variant<torch::enumtype::kFanIn,torch::enumtype::kFanOut>",
                             "torch::nn::init::FanModeType").pointerTypes("FanModeType").define())

               .put(new Info("c10::variant<torch::enumtype::kZeros,torch::enumtype::kReflect,torch::enumtype::kReplicate,torch::enumtype::kCircular>",
                             "torch::nn::ConvOptions<1>::padding_mode_t",
                             "torch::nn::ConvOptions<2>::padding_mode_t",
                             "torch::nn::ConvOptions<3>::padding_mode_t",
                             "torch::nn::ConvTransposeOptions<1>::padding_mode_t",
                             "torch::nn::ConvTransposeOptions<2>::padding_mode_t",
                             "torch::nn::ConvTransposeOptions<3>::padding_mode_t",
                             "torch::nn::detail::conv_padding_mode_t").pointerTypes("conv_padding_mode_t").define())
               .put(new Info("c10::variant<torch::ExpandingArray<1>,torch::enumtype::kValid,torch::enumtype::kSame>",
                             "torch::nn::ConvOptions<1>::padding_t",
                             "torch::nn::detail::ConvNdOptions<1>::padding_t",
                             "torch::nn::functional::ConvFuncOptions<1>::padding_t",
                             "torch::nn::functional::Conv1dFuncOptions::padding_t").purify().pointerTypes("conv_padding_t1").define())
               .put(new Info("c10::variant<torch::ExpandingArray<2>,torch::enumtype::kValid,torch::enumtype::kSame>",
                             "torch::nn::ConvOptions<2>::padding_t",
                             "torch::nn::detail::ConvNdOptions<2>::padding_t",
                             "torch::nn::functional::ConvFuncOptions<2>::padding_t",
                             "torch::nn::functional::Conv2dFuncOptions::padding_t").purify().pointerTypes("conv_padding_t2").define())
               .put(new Info("c10::variant<torch::ExpandingArray<3>,torch::enumtype::kValid,torch::enumtype::kSame>",
                             "torch::nn::ConvOptions<3>::padding_t",
                             "torch::nn::detail::ConvNdOptions<3>::padding_t",
                             "torch::nn::functional::ConvFuncOptions<3>::padding_t",
                             "torch::nn::functional::Conv3dFuncOptions::padding_t").purify().pointerTypes("conv_padding_t3").define())

               .put(new Info("c10::variant<torch::enumtype::kSum,torch::enumtype::kMean,torch::enumtype::kMax>",
                             "torch::nn::EmbeddingBagMode").pointerTypes("EmbeddingBagMode").define())
               .put(new Info("c10::variant<torch::enumtype::kConstant,torch::enumtype::kReflect,torch::enumtype::kReplicate,torch::enumtype::kCircular>",
                             "torch::nn::functional::PadFuncOptions::mode_t").pointerTypes("pad_mode_t").define())

               .put(new Info("c10::variant<torch::enumtype::kNone,torch::enumtype::kMean,torch::enumtype::kSum>",
                             "torch::nn::L1LossOptions::reduction_t", "torch::nn::functional::L1LossFuncOptions::reduction_t",
                             "torch::nn::MSELossOptions::reduction_t", "torch::nn::functional::MSELossFuncOptions::reduction_t",
                             "torch::nn::BCELossOptions::reduction_t", "torch::nn::functional::BinaryCrossEntropyFuncOptions::reduction_t",
                             "torch::nn::HingeEmbeddingLossOptions::reduction_t", "torch::nn::functional::HingeEmbeddingLossFuncOptions::reduction_t",
                             "torch::nn::MultiMarginLossOptions::reduction_t", "torch::nn::functional::MultiMarginLossFuncOptions::reduction_t",
                             "torch::nn::CosineEmbeddingLossOptions::reduction_t", "torch::nn::functional::CosineEmbeddingLossFuncOptions::reduction_t",
                             "torch::nn::MultiLabelMarginLossOptions::reduction_t", "torch::nn::functional::MultilabelMarginLossFuncOptions::reduction_t",
                             "torch::nn::SoftMarginLossOptions::reduction_t", "torch::nn::functional::SoftMarginLossFuncOptions::reduction_t",
                             "torch::nn::MultiLabelSoftMarginLossOptions::reduction_t", "torch::nn::functional::MultilabelSoftMarginLossFuncOptions::reduction_t",
                             "torch::nn::TripletMarginLossOptions::reduction_t", "torch::nn::functional::TripletMarginLossFuncOptions::reduction_t",
                             "torch::nn::TripletMarginWithDistanceLossOptions::reduction_t", "torch::nn::functional::TripletMarginWithDistanceLossFuncOptions::reduction_t",
                             "torch::nn::CTCLossOptions::reduction_t", "torch::nn::functional::CTCLossFuncOptions::reduction_t",
                             "torch::nn::SmoothL1LossOptions::reduction_t", "torch::nn::functional::SmoothL1LossFuncOptions::reduction_t",
                             "torch::nn::HuberLossOptions::reduction_t", "torch::nn::functional::HuberLossFuncOptions::reduction_t",
                             "torch::nn::PoissonNLLLossOptions::reduction_t", "torch::nn::functional::PoissonNLLLossFuncOptions::reduction_t",
                             "torch::nn::MarginRankingLossOptions::reduction_t", "torch::nn::functional::MarginRankingLossFuncOptions::reduction_t",
                             "torch::nn::NLLLossOptions::reduction_t", "torch::nn::functional::NLLLossFuncOptions::reduction_t",
                             "torch::nn::CrossEntropyLossOptions::reduction_t", "torch::nn::functional::CrossEntropyFuncOptions::reduction_t",
                             "torch::nn::BCEWithLogitsLossOptions::reduction_t", "torch::nn::functional::BinaryCrossEntropyWithLogitsFuncOptions::reduction_t").pointerTypes("loss_reduction_t").define())
               .put(new Info("c10::variant<torch::enumtype::kNone,torch::enumtype::kBatchMean,torch::enumtype::kSum,torch::enumtype::kMean>",
                             "torch::nn::KLDivLossOptions::reduction_t", "torch::nn::functional::KLDivFuncOptions::reduction_t").pointerTypes("kldiv_loss_reduction_t").define())

               .put(new Info("c10::variant<torch::enumtype::kBilinear,torch::enumtype::kNearest>",
                             "torch::nn::functional::GridSampleFuncOptions::mode_t").pointerTypes("grid_sample_mode_t").define())
               .put(new Info("c10::variant<torch::enumtype::kZeros,torch::enumtype::kBorder,torch::enumtype::kReflection>",
                             "torch::nn::functional::GridSampleFuncOptions::padding_mode_t").pointerTypes("grid_sample_padding_mode_t").define())

               .put(new Info("c10::variant<torch::enumtype::kLSTM,torch::enumtype::kGRU,torch::enumtype::kRNN_TANH,torch::enumtype::kRNN_RELU>",
                             "torch::nn::detail::RNNOptionsBase::rnn_options_base_mode_t").pointerTypes("rnn_options_base_mode_t").define())
               .put(new Info("c10::variant<torch::enumtype::kTanh,torch::enumtype::kReLU>",
                             "torch::nn::RNNOptions::nonlinearity_t", "torch::nn::RNNCellOptions::nonlinearity_t").pointerTypes("rnn_nonlinearity_t").define())

               .put(new Info("c10::variant<torch::enumtype::kNearest,torch::enumtype::kLinear,torch::enumtype::kBilinear,torch::enumtype::kBicubic,torch::enumtype::kTrilinear>",
                             "torch::nn::UpsampleOptions::mode_t").pointerTypes("upsample_mode_t").define())
               .put(new Info("c10::variant<torch::enumtype::kNearest,torch::enumtype::kLinear,torch::enumtype::kBilinear,torch::enumtype::kBicubic,torch::enumtype::kTrilinear,torch::enumtype::kArea,torch::enumtype::kNearestExact>",
                             "torch::nn::functional::InterpolateFuncOptions::mode_t").pointerTypes("interpolate_mode_t").define())

               .put(new Info("c10::variant<torch::enumtype::kReLU,torch::enumtype::kGELU,std::function<at::Tensor(const at::Tensor&)> >",
                             "torch::nn::TransformerEncoderLayerOptions::activation_t",
                             "torch::nn::TransformerDecoderLayerOptions::activation_t",
                             "torch::nn::TransformerOptions::activation_t").pointerTypes("transformer_activation_t").define())

               .put(new Info("std::vector<std::array<bool,2> >").pointerTypes("Bool2Vector").define())
               .put(new Info("std::vector<bool>").pointerTypes("BoolVector").define())
               .put(new Info("std::vector<const char*>").pointerTypes("BytePointerVector").define())
               .put(new Info("std::vector<int64_t>", "std::tuple<std::vector<int64_t>,std::vector<int64_t> >").cast().pointerTypes("LongVector").define())
               .put(new Info("std::vector<double>").cast().pointerTypes("DoubleVector").define())
               .put(new Info("std::vector<size_t>").cast().pointerTypes("SizeTVector").define())
               .put(new Info("std::vector<std::string>").pointerTypes("StringVector").define())
               .put(new Info("std::vector<std::pair<std::string,int64_t> >").pointerTypes("StringLongVector").define())
               .put(new Info("const std::vector<std::pair<at::RecordFunctionCallback,uint64_t> >",
                             "std::vector<std::pair<at::RecordFunctionCallback,at::CallbackHandle> >").pointerTypes("RecordFunctionCallbackHandleVector").define())
               .put(new Info("std::vector<c10::Argument>").pointerTypes("ArgumentVector").define())
               .put(new Info("std::vector<c10::IValue>", "torch::jit::Stack").pointerTypes("IValueVector").define())
               .put(new Info("std::vector<c10::QEngine>", "std::vector<at::QEngine>").pointerTypes("QEngineVector").define())
               .put(new Info("std::vector<c10::ScalarType>").pointerTypes("ScalarTypeVector").define())
               .put(new Info("std::vector<c10::Symbol>").pointerTypes("SymbolVector").define())
               .put(new Info("c10::Dict<c10::IValue,c10::IValue>").pointerTypes("GenericDict").define())
               .put(new Info("std::map<std::string,std::string>").pointerTypes("StringStringMap").define())
               .put(new Info("std::map<std::string,int>").pointerTypes("StringIntMap").define())
               .put(new Info("std::map<std::string,int64_t>").pointerTypes("StringLongMap").define())
               .put(new Info("std::map<std::string,at::Tensor>").pointerTypes("StringTensorMap").define())
               .put(new Info("std::unordered_set<std::string>").pointerTypes("StringSet").define())
               .put(new Info("std::unordered_set<c10::IValue,c10::IValue::HashAliasedIValue,c10::IValue::CompAliasedIValues>").pointerTypes("HashAliasedIValues").define())
               .put(new Info("std::unordered_set<c10::Symbol>").pointerTypes("SymbolSet").define())
               .put(new Info("std::unordered_set<at::TensorImpl*>").pointerTypes("TensorImplSet").define())
               .put(new Info("std::unordered_set<at::RecordScope,std::hash<at::RecordScope> >").pointerTypes("RecordScopeSet").define())
               .put(new Info("std::unordered_map<c10::IValue,c10::IValue,c10::IValue::HashAliasedIValue,c10::IValue::CompAliasedIValues>").pointerTypes("HashAliasedIValueMap").define())
               .put(new Info("std::unordered_map<int64_t,std::string>").pointerTypes("LongStringMap").define())
               .put(new Info("std::unordered_map<std::string,size_t>").pointerTypes("StringSizeTMap").define())
               .put(new Info("std::unordered_map<std::string,std::string>").pointerTypes("ExtraFilesMap").define())
               .put(new Info("std::unordered_map<std::string,c10::TypePtr>").pointerTypes("TypeEnv").define())
               .put(new Info("std::unordered_map<std::string,c10::IValue>", "std::unordered_map<std::string,at::IValue>").pointerTypes("StringIValueMap").define())
               .put(new Info("std::unordered_map<std::string,std::function<PyObject*(void*)> >").pointerTypes("StringFunctionMap").define())
               .put(new Info("std::unordered_map<std::string,torch::jit::Value*>").pointerTypes("StringValueMap").define())
               .put(new Info("std::unordered_map<std::string,std::unordered_map<int64_t,std::string> >").pointerTypes("StringLongStringMapMap").define())
               .put(new Info("std::unordered_map<torch::jit::ArgumentSpec,torch::jit::ExecutionPlan>").pointerTypes("ArgumentSpecExecutionPlanMap").define())
               .put(new Info("std::unordered_map<torch::jit::Value*,torch::jit::Value*>").pointerTypes("ValueValueMap").define())
               .put(new Info("std::vector<std::shared_ptr<c10::ClassType> >", "std::vector<c10::ClassTypePtr>").pointerTypes("ClassTypeVector").define())
               .put(new Info("std::vector<c10::Type::SingletonOrSharedTypePtr<c10::Type> >", "std::vector<c10::TypePtr>", "std::vector<c10::Type::TypePtr>").pointerTypes("TypeVector").define())
               .put(new Info("const std::vector<at::Dimname>", "std::vector<at::Dimname>").valueTypes("@StdMove DimnameVector").pointerTypes("DimnameVector").define())
               .put(new Info("std::vector<c10::Stride>").pointerTypes("StrideVector").define())
               .put(new Info("std::vector<c10::ShapeSymbol>").pointerTypes("ShapeSymbolVector").define())
               .put(new Info("std::vector<c10::TensorImpl*>").pointerTypes("TensorImplVector").define())
               .put(new Info("std::vector<torch::autograd::Edge>", "torch::autograd::edge_list")
                       .valueTypes("@Cast({\"\", \"std::vector<torch::autograd::Edge>\"}) @StdMove EdgeVector").pointerTypes("EdgeVector").define())
               .put(new Info("std::vector<at::Tensor>", "std::vector<at::Tensor,A>", "std::vector<torch::autograd::Variable>", "torch::autograd::variable_list")
                       .valueTypes("@Cast({\"\", \"std::vector<at::Tensor>\"}) @StdMove TensorVector").pointerTypes("TensorVector").define())
               .put(new Info("std::vector<at::indexing::TensorIndex>", "std::vector<at::indexing::TensorIndex,A>").pointerTypes("TensorIndexVector").define())
               .put(new Info("std::vector<c10::optional<torch::autograd::Variable> >").pointerTypes("TensorOptionalVector").define())
               .put(new Info("std::vector<c10::optional<torch::jit::Operator> >").pointerTypes("OperatorOptionalVector").define())
               .put(new Info("std::vector<std::shared_ptr<torch::autograd::FunctionPreHook> >").pointerTypes("FunctionPreVector").define())
               .put(new Info("const std::vector<std::unique_ptr<torch::autograd::FunctionPreHook> >",
                                   "std::vector<std::unique_ptr<torch::autograd::FunctionPreHook> >").pointerTypes("FunctionPreHookVector").define())
               .put(new Info("const std::vector<std::unique_ptr<torch::autograd::FunctionPostHook> >",
                                   "std::vector<std::unique_ptr<torch::autograd::FunctionPostHook> >").pointerTypes("FunctionPostHookVector").define())
               .put(new Info("const std::vector<std::unique_ptr<torch::jit::TokenTrie> >",
                                   "std::vector<std::unique_ptr<torch::jit::TokenTrie> >").pointerTypes("TokenTrieVector").define())
               .put(new Info("const std::vector<torch::autograd::SavedVariable>", "std::vector<torch::autograd::SavedVariable>").pointerTypes("SavedVariableVector").define())
               .put(new Info("const std::vector<torch::jit::Def>", "std::vector<torch::jit::Def>").pointerTypes("DefVector").define())
               .put(new Info("const std::vector<torch::jit::Property>", "std::vector<torch::jit::Property>").pointerTypes("PropertyVector").define())
               .put(new Info("const std::vector<torch::jit::Instruction>", "std::vector<torch::jit::Instruction>").pointerTypes("InstructionVector").define())
               .put(new Info("const std::vector<torch::jit::CompilationUnit>", "std::vector<torch::jit::CompilationUnit>").pointerTypes("CompilationUnitVector").define())
               .put(new Info("const std::vector<torch::optim::OptimizerParamGroup>", "std::vector<torch::optim::OptimizerParamGroup>").pointerTypes("OptimizerParamGroupVector").define())
               .put(new Info("std::vector<torch::jit::Function*>").pointerTypes("FunctionVector").define())
               .put(new Info("std::vector<std::shared_ptr<torch::jit::Graph> >").pointerTypes("GraphVector").define())
               .put(new Info("std::vector<std::shared_ptr<torch::jit::Operator> >").pointerTypes("OperatorVector").define())
               .put(new Info("std::vector<std::shared_ptr<torch::jit::Resolver> >", "std::vector<torch::jit::ResolverPtr>").pointerTypes("ResolverVector").define())
               .put(new Info("std::vector<std::shared_ptr<torch::jit::SugaredValue> >", "std::vector<torch::jit::SugaredValuePtr>").pointerTypes("SugaredValueVector").define())
               .put(new Info("std::vector<torch::jit::StackEntry>").pointerTypes("StackEntryVector").define())
               .put(new Info("std::vector<torch::jit::Block*>").pointerTypes("BlockVector").define())
               .put(new Info("std::vector<torch::jit::Value*>", "std::vector<Value*>").pointerTypes("ValueVector").define())
               .put(new Info("std::vector<const torch::jit::Node*>").pointerTypes("JitNodeVector").define())
               .put(new Info("std::deque<at::Tensor>").pointerTypes("TensorDeque").define())
               .put(new Info("std::tuple<at::Tensor>").pointerTypes("TensorTuple").define())
               .put(new Info("std::tuple<at::Tensor,at::Tensor>").pointerTypes("TensorTensorTuple").define())
               .put(new Info("std::tuple<at::Tensor,at::Tensor,at::Tensor>").pointerTypes("TensorTensorTensorTuple").define())
               .put(new Info("std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor>").pointerTypes("TensorTensorTensorTensorTuple").define())
               .put(new Info("std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor>").pointerTypes("TensorTensorTensorTensorTensorTuple").define())
               .put(new Info("std::tuple<at::Tensor,at::Tensor,at::Tensor,std::vector<at::Tensor> >").pointerTypes("TensorTensorTensorTensorVectorTuple").define())
               .put(new Info("std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,int64_t>").pointerTypes("TensorTensorTensorTensorLongTuple").define())
               .put(new Info("std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor>").pointerTypes("TensorTensorTensorTensorTensorTensorTuple").define())
               .put(new Info("std::tuple<at::Tensor,at::Tensor,double,int64_t>").pointerTypes("TensorTensorDoubleLongTuple").define())
               .put(new Info("std::tuple<at::Tensor,std::tuple<at::Tensor,at::Tensor> >").pointerTypes("TensorTensorTensorTupleTuple").define())
               .put(new Info("std::tuple<c10::MaybeOwned<at::Tensor>,c10::MaybeOwned<at::Tensor> >")
                       .pointerTypes("TensorMaybeOwnedTensorMaybeOwnedTuple").define())
               .put(new Info("std::tuple<c10::MaybeOwned<at::Tensor>,c10::MaybeOwned<at::Tensor>,c10::MaybeOwned<at::Tensor> >")
                       .pointerTypes("TensorMaybeOwnedTensorMaybeOwnedTensorMaybeOwnedTuple").define())
               .put(new Info("std::tuple<torch::nn::utils::rnn::PackedSequence,at::Tensor>").purify().pointerTypes("PackedSequenceTensorTuple").define())
               .put(new Info("std::tuple<torch::nn::utils::rnn::PackedSequence,std::tuple<at::Tensor,at::Tensor> >").purify().pointerTypes("PackedSequenceTensorTensorTupleTuple").define())
               .put(new Info("std::tuple<at::Tensor&,at::Tensor&>", "std::tuple<at::Tensor&,at::Tensor&,at::Tensor&>",
                             "std::tuple<at::Tensor&,at::Tensor&,at::Tensor&,at::Tensor&>").cast().pointerTypes("PointerPointer<Tensor>"))
               .put(new Info("std::tuple<std::string,size_t,size_t>").pointerTypes("StringSizeTSizeTTuple").define())
               .put(new Info("std::tuple<at::Tensor,std::vector<at::Tensor> >").pointerTypes("TensorTensorVectorTuple").define())
               .put(new Info("std::tuple<std::vector<at::Tensor>,at::Tensor>").pointerTypes("TensorVectorTensorTuple").define())
               .put(new Info("std::tuple<at::Tensor,std::vector<at::Tensor>,std::vector<at::Tensor> >").pointerTypes("TensorTensorVectorTensorVectorTuple").define())
               .put(new Info("torch::OrderedDict<std::string,at::Tensor>", "torch::OrderedDict<std::string,torch::Tensor>").pointerTypes("StringTensorDict").define())
               .put(new Info("torch::OrderedDict<Key,Value>::Item<std::string,at::Tensor>", "torch::OrderedDict<std::string,at::Tensor>::Item",
                             "std::vector<torch::OrderedDict<std::string,at::Tensor>::Item>::iterator").pointerTypes("StringTensorDictItem"))
               .put(new Info("torch::OrderedDict<std::string,torch::nn::Module>").pointerTypes("StringModuleDict").define())
               .put(new Info("torch::OrderedDict<Key,Value>::Item<std::string,torch::nn::Module>", "torch::OrderedDict<std::string,torch::nn::Module>::Item",
                             "std::vector<torch::OrderedDict<std::string,torch::nn::Module>::Item>::iterator").pointerTypes("StringModuleDictItem"))
               .put(new Info("torch::OrderedDict<std::string,torch::nn::AnyModule>")
                       .valueTypes("@Cast({\"\", \"torch::OrderedDict<std::string,torch::nn::AnyModule>&&\"}) @StdMove StringAnyModuleDict").pointerTypes("StringAnyModuleDict").define())
               .put(new Info("torch::OrderedDict<Key,Value>::Item<std::string,torch::nn::AnyModule>", "torch::OrderedDict<std::string,torch::nn::AnyModule>::Item",
                             "std::vector<torch::OrderedDict<std::string,torch::nn::AnyModule>::Item>::iterator").pointerTypes("StringAnyModuleDictItem"))
               .put(new Info("torch::OrderedDict<std::string,std::shared_ptr<torch::nn::Module> >").pointerTypes("StringSharedModuleDict").define())
               .put(new Info("torch::OrderedDict<Key,Value>::Item<std::string,std::shared_ptr<torch::nn::Module> >", "torch::OrderedDict<std::string,std::shared_ptr<torch::nn::Module> >::Item",
                             "std::vector<torch::OrderedDict<std::string,std::shared_ptr<torch::nn::Module> >::Item>::iterator").pointerTypes("StringSharedModuleDictItem"))
               .put(new Info("std::pair<std::string,torch::Tensor>", "std::pair<std::string,at::Tensor>", "torch::OrderedDict<std::string,torch::Tensor>::Item",
                             "std::vector<torch::OrderedDict<std::string,torch::Tensor>::Item>::iterator").cast().pointerTypes("StringTensorPair").define())
               .put(new Info("std::pair<std::string,torch::nn::Module>").pointerTypes("StringModulePair").define())
               .put(new Info("std::pair<std::string,torch::nn::AnyModule>").pointerTypes("StringAnyModulePair").define())
               .put(new Info("std::pair<std::string,std::shared_ptr<torch::nn::Module> >").pointerTypes("StringSharedModulePair").define())
               .put(new Info("std::vector<torch::nn::Module>").pointerTypes("ModuleVector").define())
               .put(new Info("std::vector<torch::nn::Module>::iterator").pointerTypes("ModuleVector.Iterator"))
               .put(new Info("std::vector<torch::nn::AnyModule>").pointerTypes("AnyModuleVector").define())
               .put(new Info("std::vector<torch::nn::AnyModule>::iterator").pointerTypes("AnyModuleVector.Iterator"))
               .put(new Info("std::vector<std::shared_ptr<torch::nn::Module> >").pointerTypes("SharedModuleVector").define())
               .put(new Info("std::vector<std::shared_ptr<torch::nn::Module> >::iterator").pointerTypes("SharedModuleVector.Iterator"))
               .put(new Info("std::vector<std::shared_ptr<torch::nn::AnyModule> >").pointerTypes("SharedAnyModuleVector").define())
               .put(new Info("std::vector<std::shared_ptr<torch::nn::AnyModule> >::iterator").pointerTypes("SharedAnyModuleVector.Iterator"))
               .put(new Info("std::vector<std::pair<std::string,at::Tensor> >").pointerTypes("StringTensorPairVector").define())
               .put(new Info("std::vector<std::pair<std::string,torch::nn::Module> >").pointerTypes("StringModulePairVector").define())
               .put(new Info("std::vector<std::pair<std::string,torch::nn::AnyModule> >").pointerTypes("StringAnyModulePairVector").define())
               .put(new Info("std::vector<std::pair<std::string,std::shared_ptr<torch::nn::Module> > >").pointerTypes("StringSharedModulePairVector").define())
               .put(new Info("std::vector<std::pair<torch::jit::FusionBehavior,size_t> >", "torch::jit::FusionStrategy").pointerTypes("FusionStrategy").define())

               .put(new Info("C10_EXPORT", "C10_HIDDEN", "C10_IMPORT", "C10_API", "C10_API_ENUM", "EXPORT_IF_NOT_GCC",
                             "TORCH_API", "TORCH_CUDA_CU_API", "TORCH_CUDA_CPP_API", "TORCH_HIP_API", "TORCH_PYTHON_API",
                             "__ubsan_ignore_float_divide_by_zero__", "__ubsan_ignore_undefined__", "__ubsan_ignore_signed_int_overflow__", "__ubsan_ignore_function__",
                             "C10_CLANG_DIAGNOSTIC_IGNORE", "C10_CLANG_DIAGNOSTIC_PUSH", "C10_CLANG_DIAGNOSTIC_POP", "C10_ATTR_VISIBILITY_HIDDEN", "C10_ERASE",
                             "C10_UID", "C10_NODISCARD", "C10_UNUSED", "C10_USED", "C10_RESTRICT", "C10_NOINLINE", "C10_ALWAYS_INLINE", "C10_FALLTHROUGH",
                             "C10_HOST_DEVICE", "C10_DEVICE", "C10_HOST", "C10_LAUNCH_BOUNDS_0", "C10_HIP_HOST_DEVICE", "C10_WARP_SIZE",
                             "C10_HOST_CONSTEXPR", "CONSTEXPR_EXCEPT_WIN_CUDA", "C10_HOST_CONSTEXPR_EXCEPT_WIN_CUDA",
                             "alignas", "COMPLEX_INTEGER_OP_TEMPLATE_CONDITION", "C10_DEVICE_HOST_FUNCTION", "FORCE_INLINE_APPLE",
                             "ERROR_UNSUPPORTED_CAST", "LEGACY_CONTIGUOUS_MEMORY_FORMAT", "GFLAGS_DLL_DEFINE_FLAG", "GFLAGS_DLL_DECLARE_FLAG",
                             "AT_X", "DEFINE_KEY", "C10_DISPATCHER_INLINE_UNLESS_MOBILE", "TH_DISALLOW_COPY_AND_ASSIGN").cppTypes().annotations())

               .put(new Info("defined(__CUDACC__) || defined(__HIPCC__)",
                             "defined(_MSC_VER) && _MSC_VER <= 1900",
                             "defined(NDEBUG)",
                             "defined(__ANDROID__)",
                             "defined(__APPLE__)",
                             "defined(__HIP_PLATFORM_HCC__)",
                             "defined(_MSC_VER)", "_WIN32",
                             "defined(USE_ROCM)", "USE_ROCM", "SYCL_LANGUAGE_VERSION",
                             "defined(CUDA_VERSION) && CUDA_VERSION >= 11000",
                             "defined ENABLE_RECORD_KERNEL_FUNCTION_DTYPE").define(false))

               .put(new Info("C10_DEFINE_DEPRECATED_USING").cppText("#define C10_DEFINE_DEPRECATED_USING(TypeName, TypeThingy)").cppTypes())
               .put(new Info("C10_DEPRECATED_MESSAGE").cppText("#define C10_DEPRECATED_MESSAGE() deprecated").cppTypes())
               .put(new Info("C10_DEPRECATED").cppText("#define C10_DEPRECATED deprecated").cppTypes())
               .put(new Info("deprecated").annotations("@Deprecated"))

               .put(new Info("CAFFE2_LOG_THRESHOLD").translate(false))

               .put(new Info("DEFINE_SYMBOL").cppText("#define DEFINE_SYMBOL(ns, s) namespace ns { constexpr Symbol s; }").define())
               .put(new Info("TORCH_ENUM_DECLARE").cppText("#define TORCH_ENUM_DECLARE(name) namespace torch { namespace enumtype { struct k##name { k##name() {} }; } }").define())

               .put(new Info("c10::Error", "c10::IndexError", "c10::LinAlgError", "c10::ValueError", "c10::TypeError", "c10::NotImplementedError", "c10::EnforceFiniteError",
                             "c10::OnnxfiBackendSystemError", "c10::Capsule", "c10::ClassType", "c10::EnumType", "c10::OperatorNameView", "c10::SharedType", "c10::StrongTypePtr",
                             "c10::WeakTypePtr", "c10::NamedType", "torch::autograd::CppFunctionPreHook", "torch::autograd::DifferentiableViewMeta", "torch::autograd::Node",
                             "torch::autograd::NodeGuard", "torch::autograd::TraceableFunction", "torch::jit::Instruction", "torch::jit::Method", "torch::jit::ModuleInstanceInfo",
                             "torch::jit::Object::Property", "torch::jit::Operator", "torch::jit::OperatorSet", "torch::jit::SourceRangePickler", "torch::jit::Suspend", "torch::jit::Unpickler").purify())

               .put(new Info("c10::intrusive_ptr", "c10::weak_intrusive_ptr", "c10::guts::is_fundamental", "c10::operator !=", "c10::operator ==", "c10::operator <<",
                             "c10::detail::CaptureKernelCall", "c10::detail::MultiDispatchKeySet", "c10::ExclusivelyOwnedTraits", "c10::FunctionSchema::dump",
                             "c10::domain_prefix", "c10::C10FlagsRegistry", "c10::enforce_detail::EnforceFailMessage", "c10::impl::build_feature_required_feature_not_available",
                             "c10::complex_literals::operator \"\"_if", "c10::complex_literals::operator \"\"_id", "c10::complex<c10::Half>",
                             "c10::InefficientStdFunctionContext", "c10::DataPtr::move_context", "QuantizerPtr", "c10::IValue::toModule",
                             "c10::List<c10::optional<at::Tensor> >", "c10::optional<THPObjectPtr>", "c10::standardizeVectorForUnion",
                             "c10::impl::ExcludeDispatchKeyGuard", "c10::impl::ScalarTypeToCPPType", "c10::impl::AnnotatedKernel", "c10::impl::OperatorEntry",
                             "c10::StorageImpl(c10::StorageImpl)", "c10::StorageImpl::operator =",
                             "c10::TensorImpl(c10::TensorImpl)", "c10::TensorImpl::operator =",
                             "caffe2::Blob(caffe2::Blob)", "caffe2::Blob::operator =",
                             "torch::serialize::InputArchive(torch::serialize::InputArchive)", "torch::serialize::InputArchive::operator =",
                             "torch::serialize::OutputArchive(torch::serialize::OutputArchive)", "torch::serialize::OutputArchive::operator =",
                             "at::_test_serialization_subcmul", "at::_test_optional_intlist", "at::_test_optional_filled_intlist",
                             "at::_test_optional_floatlist", "at::_test_string_default", "at::_test_ambiguous_defaults",
                             "at::TensorBase::expect_contiguous", "at::Tensor::print", "at::borrow_from_optional_tensor",
                             "at::impl::check_names_valid_for", "at::internal::launch_no_thread_state",
                             "at::checkSameNumel", "at::check_names_valid_for", "at::default_names", "at::get_device", "at::detail::scalar_fill",
                             "at::namedinference::compute_diagonal_outnames", "at::Tensor::packed_accessor", "torch::optim::serialize", "torch::none_of",
                             "torch::CountTensors", "torch::CountVariables", "torch::autograd::ExtractVariables", "torch::autograd::detail::MakeNextFunctionList",
                             "torch::autograd::VariableType::unpack", "torch::autograd::VariableType::unpack_opt", "torch::jit::parseSchemaOrName",
                             "torch::jit::trace", "torch::jit::tracer::TracingState::lookup_var_name_fn", "torch::jit::tracer::ArgumentStash",
                             "torch::jit::constant_not_supported_error", "torch::jit::ObjectAttributeError", "torch::jit::utils::get_module_info",
                             "torch::jit::operator <<(std::ostream&, torch::jit::Instruction)", "torch::jit::toString(torch::jit::OpCode)",
                             "torch::jit::PropertyPropBase::processLoop", "torch::jit::PropertyPropBase::processIf", "torch::jit::PropertyPropBase::propagateBlock",
                             "torch::jit::getMobileInterfaceCallExport", "torch::jit::OperatorSet::getOps", "torch::jit::SourceView::findSourceRangeThatGenerated",

                             "torch::jit::checkHasValidSetGetState", "torch::jit::getTypeTags", "torch::jit::setTypeTags", "torch::jit::getStorageKey",
                             "torch::jit::getUnresolvedClassAttributes", "torch::jit::isOpSupportedInMobile", "torch::jit::restoreAccurateTypeTags",
                             "torch::jit::detail::getDifferentiableGraphOpExecutor","torch::jit::detail::getGradExecutor", "torch::jit::Graph::createPythonOp",
                             "torch::jit::Graph::createDifferentiableSubgraph", "torch::jit::NamedValue::type", "torch::jit::ProfileOp", "torch::jit::Value::isValidName",
                             "torch::jit::EqualType::operator ()", "torch::jit::HashType::operator ()", "torch::jit::InterpreterContinuation::operator ()",
                             "torch::jit::Object(c10::QualifiedName, torch::jit::CompilationUnit*, bool)", "torch::jit::Source::findSourceRangeThatGenerated",
                             "torch::jit::SourceRangeDeserializer::deserialize", "torch::jit::SourceRangePickler::pickle", "torch::jit::Pickler::pushEmptyDict",
                             "torch::jit::PrintDepsTable::add", "torch::jit::printerHasSpecialCaseFor", "ONNX_NAMESPACE::ModelProto", "torch::jit::export_onnx",
                             "torch::jit::Function::call", "torch::jit::GraphFunction::call", "torch::jit::GraphFunction::function_creator", "torch::jit::getOptionsFromGlobal",
                             "torch::jit::serialize_model_proto_to_string", "torch::onnx::IR_VERSION", "torch::onnx::PRODUCER_VERSION").skip())

               .put(new Info("c10::requires_grad", "at::range", "at::bernoulli_out", "at::normal_out", "at::stft").skipDefaults())
               .put(new Info("c10::prim::requires_grad").javaNames("requires_grad"))
               .put(new Info("c10::fetch_and_cast<c10::qint8>").javaNames("fetch_and_cast_qint8"))
               .put(new Info("c10::cast_and_store<c10::qint8>").javaNames("cast_and_store_qint8"))
               .put(new Info("c10::fetch_and_cast<c10::quint8>").javaNames("fetch_and_cast_quint8"))
               .put(new Info("c10::cast_and_store<c10::quint8>").javaNames("cast_and_store_quint8"))
               .put(new Info("c10::fetch_and_cast<c10::qint32>").javaNames("fetch_and_cast_qint32"))
               .put(new Info("c10::cast_and_store<c10::qint32>").javaNames("cast_and_store_qint32"))
               .put(new Info("c10::fetch_and_cast<c10::quint4x2>").javaNames("fetch_and_cast_quint4x2"))
               .put(new Info("c10::cast_and_store<c10::quint4x2>").javaNames("cast_and_store_quint4x2"))
               .put(new Info("c10::aten::clone").javaNames("_clone"))
               .put(new Info("c10::TensorOptions<c10::Device>").javaNames("TensorOptions"))
               .put(new Info("c10::detail::_str<CompileTimeEmptyString>").javaNames("_strCompileTimeEmptyString"))
               .put(new Info("at::TensorBase").base("AbstractTensor").pointerTypes("TensorBase"))
               .put(new Info("at::TensorBase::data_ptr<int8_t>").javaNames("data_ptr_byte"))
               .put(new Info("at::TensorBase::data_ptr<int16_t>").javaNames("data_ptr_short"))
               .put(new Info("at::TensorBase::data_ptr<int>").javaNames("data_ptr_int"))
               .put(new Info("at::TensorBase::data_ptr<int64_t>").javaNames("data_ptr_long"))
               .put(new Info("at::TensorBase::data_ptr<float>").javaNames("data_ptr_float"))
               .put(new Info("at::TensorBase::data_ptr<double>").javaNames("data_ptr_double"))
               .put(new Info("at::Tensor::item<int8_t>").javaNames("item_byte"))
               .put(new Info("at::Tensor::item<int16_t>").javaNames("item_short"))
               .put(new Info("at::Tensor::item<int>").javaNames("item_int"))
               .put(new Info("at::Tensor::item<int64_t>").javaNames("item_long"))
               .put(new Info("at::Tensor::item<float>").javaNames("item_float"))
               .put(new Info("at::Tensor::item<double>").javaNames("item_double"))

//               .put(new Info("c10::complex<double>").pointerTypes("DoubleComplex").define())
//               .put(new Info("c10::complex<float>").pointerTypes("FloatComplex").define())
//               .put(new Info("c10::complex<c10::Half>").pointerTypes("HalfComplex").define())
//               .put(new Info("c10::complex<double>::real", "c10::complex<double>::imag",
//                             "c10::complex<float>::real", "c10::complex<float>::imag",
//                             "c10::complex<c10::Half>::real", "c10::complex<c10::Half>::imag").annotations("@Function"))
               .put(new Info("c10::ArrayRef<jbyte>", "c10::ArrayRef<int8_t>", "c10::ArrayRef<uint8_t>").cast().pointerTypes("ByteArrayRef"))
               .put(new Info("c10::ArrayRef<jbyte>::iterator", "c10::ArrayRef<jbyte>::const_iterator").cast().pointerTypes("BytePointer"))
               .put(new Info("c10::ArrayRef<jshort>", "c10::ArrayRef<int16_t>", "c10::ArrayRef<uint16_t>").cast().pointerTypes("ShortArrayRef"))
               .put(new Info("c10::ArrayRef<jshort>::iterator", "c10::ArrayRef<jshort>::const_iterator").cast().pointerTypes("ShortPointer"))
               .put(new Info("c10::ArrayRef<jint>", "c10::ArrayRef<int>", "c10::ArrayRef<int32_t>", "c10::ArrayRef<uint32_t>").cast().pointerTypes("IntArrayRef"))
               .put(new Info("c10::ArrayRef<jint>::iterator", "c10::ArrayRef<jint>::const_iterator").cast().pointerTypes("IntPointer"))
               .put(new Info("c10::ArrayRef<int64_t>", "c10::IntArrayRef", "at::IntArrayRef")
                       .pointerTypes("@Cast(\"c10::ArrayRef<int64_t>*\") LongArrayRef", "@Cast({\"int64_t*\", \"c10::ArrayRef<int64_t>\", \"std::vector<int64_t>&\"}) @StdVector long..."))
               .put(new Info("c10::ArrayRef<jlong>::iterator", "c10::ArrayRef<jlong>::const_iterator").cast().pointerTypes("LongPointer"))
               .put(new Info("c10::ArrayRef<float>").pointerTypes("FloatArrayRef"))
               .put(new Info("c10::ArrayRef<float>::iterator", "c10::ArrayRef<float>::const_iterator").cast().pointerTypes("FloatPointer"))
               .put(new Info("c10::ArrayRef<double>").pointerTypes("DoubleArrayRef"))
               .put(new Info("c10::ArrayRef<double>::iterator", "c10::ArrayRef<double>::const_iterator").cast().pointerTypes("DoublePointer"))
               .put(new Info("c10::ArrayRef<size_t>", "at::ArrayRef<size_t>").pointerTypes("SizeTArrayRef"))
               .put(new Info("c10::ArrayRef<size_t>::iterator", "c10::ArrayRef<size_t>::const_iterator").cast().pointerTypes("SizeTPointer"))
               .put(new Info("c10::ArrayRef<std::string>", "at::ArrayRef<std::string>").pointerTypes("StringArrayRef").purify())
               .put(new Info("c10::ArrayRef<std::string>::iterator", "c10::ArrayRef<std::string>::const_iterator").pointerTypes("@Cast({\"\", \"std::string*\"}) @StdString BytePointer"))
               .put(new Info("c10::ArrayRef<decltype(::c10::impl::ScalarTypeToCPPType<::c10::ScalarType::Bool>::t)>").pointerTypes("BoolArrayRef"))
               .put(new Info("c10::ArrayRef<decltype(::c10::impl::ScalarTypeToCPPType<::c10::ScalarType::Bool>::t)>::iterator",
                             "c10::ArrayRef<decltype(::c10::impl::ScalarTypeToCPPType<::c10::ScalarType::Bool>::t)>::const_iterator").cast().pointerTypes("BoolPointer"))
               .put(new Info("c10::ArrayRef<decltype(::c10::impl::ScalarTypeToCPPType<::c10::ScalarType::Half>::t)>").pointerTypes("HalfArrayRef"))
               .put(new Info("c10::ArrayRef<decltype(::c10::impl::ScalarTypeToCPPType<::c10::ScalarType::Half>::t)>::iterator",
                             "c10::ArrayRef<decltype(::c10::impl::ScalarTypeToCPPType<::c10::ScalarType::Half>::t)>::const_iterator").cast().pointerTypes("ShortPointer"))
               .put(new Info("c10::ArrayRef<decltype(::c10::impl::ScalarTypeToCPPType<::c10::ScalarType::BFloat16>::t)>").pointerTypes("BFloat16ArrayRef"))
               .put(new Info("c10::ArrayRef<decltype(::c10::impl::ScalarTypeToCPPType<::c10::ScalarType::BFloat16>::t)>::iterator",
                             "c10::ArrayRef<decltype(::c10::impl::ScalarTypeToCPPType<::c10::ScalarType::BFloat16>::t)>::const_iterator").cast().pointerTypes("ShortPointer"))
               .put(new Info("c10::ArrayRef<c10::complex<float> >", "at::ArrayRef<c10::complex<float> >").pointerTypes("FloatComplexrrayRef"))
               .put(new Info("c10::ArrayRef<c10::complex<float> >::iterator", "c10::ArrayRef<c10::complex<float> >::const_iterator").cast().pointerTypes("FloatPointer"))
               .put(new Info("c10::ArrayRef<c10::complex<double> >", "at::ArrayRef<c10::complex<double> >").pointerTypes("DoubleComplexrrayRef"))
               .put(new Info("c10::ArrayRef<c10::complex<double> >::iterator", "c10::ArrayRef<c10::complex<double> >::const_iterator").cast().pointerTypes("DoublePointer"))
               .put(new Info("c10::ArrayRef<c10::ScalarType>", "at::ArrayRef<c10::ScalarType>", "at::ArrayRef<at::ScalarType>").pointerTypes("ScalarTypeArrayRef"))
               .put(new Info("c10::ArrayRef<c10::ScalarType>::iterator", "c10::ArrayRef<c10::ScalarType>::const_iterator").cast().pointerTypes("BytePointer"))
               .put(new Info("c10::ArrayRef<c10::IValue>", "at::ArrayRef<c10::IValue>", "c10::ArrayRef<const at::IValue>").cast().pointerTypes("IValueArrayRef"))
               .put(new Info("c10::ArrayRef<c10::IValue>::iterator", "c10::ArrayRef<c10::IValue>::const_iterator").cast().pointerTypes("IValue"))
               .put(new Info("c10::ArrayRef<c10::EnumNameValue>", "at::ArrayRef<c10::EnumNameValue>").pointerTypes("EnumNameValueArrayRef"))
               .put(new Info("c10::ArrayRef<c10::EnumNameValue>::iterator", "c10::ArrayRef<c10::EnumNameValue>::const_iterator").cast().pointerTypes("EnumNameValue"))
               .put(new Info("c10::ArrayRef<c10::TypePtr>", "at::ArrayRef<c10::TypePtr>", "at::ArrayRef<c10::Type::TypePtr>").pointerTypes("TypeArrayRef"))
               .put(new Info("c10::ArrayRef<c10::TypePtr>::iterator", "c10::ArrayRef<c10::TypePtr>::const_iterator").cast().pointerTypes("Type"))
               .put(new Info("c10::ArrayRef<c10::Symbol>", "at::ArrayRef<c10::Symbol>").pointerTypes("SymbolArrayRef"))
               .put(new Info("c10::ArrayRef<c10::Symbol>::iterator", "c10::ArrayRef<c10::Symbol>::const_iterator").cast().pointerTypes("Symbol"))
               .put(new Info("c10::ArrayRef<c10::Stride>", "at::ArrayRef<c10::Stride>").pointerTypes("StrideArrayRef"))
               .put(new Info("c10::ArrayRef<c10::Stride>::iterator", "c10::ArrayRef<c10::Stride>::const_iterator").cast().pointerTypes("Stride"))
               .put(new Info("c10::ArrayRef<at::Dimname>", "at::DimnameList").pointerTypes("DimnameArrayRef"))
               .put(new Info("c10::ArrayRef<at::Dimname>::iterator", "c10::ArrayRef<at::Dimname>::const_iterator").cast().pointerTypes("Dimname"))
               .put(new Info("c10::ArrayRef<at::Scalar>", "at::ArrayRef<at::Scalar>").pointerTypes("ScalarArrayRef"))
               .put(new Info("c10::ArrayRef<at::Scalar>::iterator", "c10::ArrayRef<at::Scalar>::const_iterator").cast().pointerTypes("Scalar"))
               .put(new Info("c10::ArrayRef<at::Tensor>", "at::ArrayRef<at::Tensor>", "at::TensorList").pointerTypes("TensorArrayRef"))
               .put(new Info("c10::ArrayRef<at::Tensor>(std::vector<at::Tensor,A>&)").javaText(
                       "public TensorArrayRef(@ByRef TensorVector Vec) { super((Pointer)null); allocate(Vec); }\n"
                     + "private native void allocate(@ByRef TensorVector Vec);"))
               .put(new Info("c10::ArrayRef<at::Tensor>::iterator", "c10::ArrayRef<at::Tensor>::const_iterator").cast().pointerTypes("Tensor"))
               .put(new Info("c10::ArrayRef<at::TensorArg>", "at::ArrayRef<at::TensorArg>").pointerTypes("TensorArgArrayRef"))
               .put(new Info("c10::ArrayRef<at::TensorArg>::iterator", "c10::ArrayRef<at::TensorArg>::const_iterator").cast().pointerTypes("TensorArg"))
               .put(new Info("c10::ArrayRef<at::indexing::TensorIndex>").pointerTypes("TensorIndexArrayRef"))
               .put(new Info("c10::ArrayRef<at::indexing::TensorIndex>(std::vector<at::indexing::TensorIndex,A>&)").javaText(
                       "public TensorIndexArrayRef(@ByRef TensorIndexVector Vec) { super((Pointer)null); allocate(Vec); }\n"
                     + "private native void allocate(@ByRef TensorIndexVector Vec);"))
               .put(new Info("c10::ArrayRef<at::indexing::TensorIndex>::iterator", "c10::ArrayRef<at::indexing::TensorIndex>::const_iterator").cast().pointerTypes("TensorIndex"))
               .put(new Info("c10::ArrayRef<c10::optional<at::Tensor> >", "at::ArrayRef<c10::optional<torch::autograd::Variable> >").pointerTypes("TensorOptionalArrayRef"))
               .put(new Info("c10::ArrayRef<c10::optional<at::Tensor> >::iterator", "c10::ArrayRef<c10::optional<at::Tensor> >::const_iterator").cast().pointerTypes("TensorOptional"))
               .put(new Info("c10::ArrayRef<torch::autograd::SavedVariable>", "at::ArrayRef<torch::autograd::SavedVariable>").pointerTypes("SavedVariableArrayRef"))
               .put(new Info("c10::ArrayRef<torch::autograd::SavedVariable>::iterator", "c10::ArrayRef<torch::autograd::SavedVariable>::const_iterator").cast().pointerTypes("SavedVariable"))
               .put(new Info("c10::ArrayRef<torch::jit::SugaredValuePtr>", "at::ArrayRef<torch::jit::SugaredValuePtr>").pointerTypes("SugaredValueArrayRef"))
               .put(new Info("c10::ArrayRef<torch::jit::SugaredValuePtr>::iterator", "c10::ArrayRef<torch::jit::SugaredValuePtr>::const_iterator").annotations("@SharedPtr").pointerTypes("SugaredValue"))
               .put(new Info("c10::ArrayRef<torch::jit::NamedValue>", "at::ArrayRef<torch::jit::NamedValue>", "at::ArrayRef<NamedValue>").pointerTypes("NamedValueArrayRef"))
               .put(new Info("c10::ArrayRef<torch::jit::NamedValue>::iterator", "c10::ArrayRef<torch::jit::NamedValue>::const_iterator").cast().pointerTypes("NamedValue"))
               .put(new Info("c10::ArrayRef<torch::jit::Block*>", "at::ArrayRef<torch::jit::Block*>").purify().pointerTypes("BlockArrayRef"))
               .put(new Info("c10::ArrayRef<torch::jit::Block*>::iterator", "c10::ArrayRef<torch::jit::Block*>::const_iterator").cast().pointerTypes("Block"))
               .put(new Info("c10::ArrayRef<torch::jit::Value*>", "at::ArrayRef<torch::jit::Value*>").purify().pointerTypes("ValueArrayRef"))
               .put(new Info("c10::ArrayRef<torch::jit::Value*>::iterator", "c10::ArrayRef<torch::jit::Value*>::const_iterator").cast().pointerTypes("Value"))
               .put(new Info("c10::ArrayRef<at::Scalar>::equals", "c10::ArrayRef<at::TensorArg>::equals",
                             "c10::ArrayRef<at::Tensor>::equals", "c10::ArrayRef<at::indexing::TensorIndex>::equals",
                             "c10::ArrayRef<c10::optional<at::Tensor> >::equals", "c10::ArrayRef<torch::jit::NamedValue>::equals",
                             "c10::ArrayRef<torch::autograd::SavedVariable>::equals", "c10::ArrayRef<torch::autograd::SavedVariable>::vec",
                             "at::ITensorListRef", "std::array<c10::FunctionalityOffsetAndMask,c10::num_functionality_keys>").skip())
               .put(new Info("c10::OptionalArray<int64_t>").pointerTypes("OptionalLongArray"))
               .put(new Info("c10::OptionalArray<double>").pointerTypes("OptionalDoubleArray"))
               .put(new Info("c10::OptionalArrayRef<int64_t>").pointerTypes("OptionalIntArrayRef"))
               .put(new Info("c10::OptionalArrayRef<double>").pointerTypes("OptionalDoubleArrayRef"))
               .put(new Info("c10::VaryingShape<int64_t>").pointerTypes("LongVaryingShape"))
               .put(new Info("c10::VaryingShape<c10::Stride>").pointerTypes("StrideVaryingShape"))

               .put(new Info("std::hash<c10::DeviceType>").pointerTypes("DeviceTypeHash"))
               .put(new Info("std::hash<c10::Device>").pointerTypes("DeviceHash"))
               .put(new Info("std::hash<c10::Stream>").pointerTypes("StreamHash"))
               .put(new Info("std::hash<c10::Symbol>").pointerTypes("SymbolHash"))
               .put(new Info("decltype(::c10::impl::ScalarTypeToCPPType<::c10::ScalarType::Bool>::t)").cast().valueTypes("boolean").pointerTypes("BoolPointer"))
               .put(new Info("decltype(::c10::impl::ScalarTypeToCPPType<::c10::ScalarType::Half>::t)").pointerTypes("Half"))
               .put(new Info("decltype(::c10::impl::ScalarTypeToCPPType<::c10::ScalarType::BFloat16>::t)").pointerTypes("BFloat16"))
               .put(new Info("c10::DataPtr", "at::DataPtr").valueTypes("@Cast({\"\", \"c10::DataPtr&&\"}) @StdMove DataPtr").pointerTypes("DataPtr"))
               .put(new Info("c10::Storage", "at::Storage").valueTypes("@Cast({\"\", \"c10::Storage&&\"}) @StdMove Storage").pointerTypes("Storage"))
               .put(new Info("std::shared_ptr<c10::ClassType>").annotations("@SharedPtr").pointerTypes("ClassType"))
               .put(new Info("std::shared_ptr<c10::EnumType>", "c10::EnumTypePtr").annotations("@SharedPtr").pointerTypes("EnumType"))
               .put(new Info("std::shared_ptr<c10::NamedType>").annotations("@SharedPtr").pointerTypes("NamedType"))
               .put(new Info("std::shared_ptr<const c10::NamedType>").annotations("@SharedPtr")
                                                                     .valueTypes("@Cast({\"\", \"\", \"std::shared_ptr<c10::NamedType>&&\"}) NamedType")
                                                                     .pointerTypes("NamedType"))
               .put(new Info("std::shared_ptr<c10::Type>").annotations("@SharedPtr").pointerTypes("Type"))
               .put(new Info("std::shared_ptr<c10::TensorType>", "c10::TensorTypePtr", "at::TensorTypePtr").annotations("@SharedPtr").pointerTypes("TensorType"))
               .put(new Info("std::unique_ptr<c10::FunctionSchema>").annotations("@UniquePtr")
                                                                    .valueTypes("@Cast({\"\", \"std::unique_ptr<c10::FunctionSchema>&&\"}) FunctionSchema")
                                                                    .pointerTypes("FunctionSchema"))
               .put(new Info("c10::IdWrapper<TypeIdentifier,c10::util::type_index>", "at::IdWrapper<TypeIdentifier,c10::util::type_index>").pointerTypes("TypeIdentifierIdWrapper"))
               .put(new Info("c10::MaybeOwned<at::Tensor>").valueTypes("@Cast({\"\", \"c10::MaybeOwned<at::Tensor>&&\"}) @StdMove TensorMaybeOwned").pointerTypes("TensorMaybeOwned"))
               .put(new Info("c10::SmallVectorTemplateCommon<int64_t>").pointerTypes("Pointer"))
               .put(new Info("c10::SmallVectorTemplateBase<int64_t>").pointerTypes("SmallVectorBase"))
               .put(new Info("c10::SmallVectorImpl<int64_t>").pointerTypes("DimVectorImpl"))
               .put(new Info("c10::SmallVectorImpl<int64_t>::size_type", "c10::SmallVectorImpl<int64_t>::ValueParamT").valueTypes("long"))
               .put(new Info("c10::SmallVectorImpl<int64_t>::iterator", "c10::SmallVectorImpl<int64_t>::const_iterator").cast().pointerTypes("LongPointer"))
               .put(new Info("c10::SmallVector<int64_t,at::kDimVectorStaticSize>", "at::DimVector").pointerTypes("DimVector"))
               .put(new Info("c10::SmallVector<int64_t,at::kDimVectorStaticSize>(c10::SmallVectorImpl<int64_t>&&)",
                             "c10::SmallVector<int64_t,at::kDimVectorStaticSize>::operator =(c10::SmallVectorImpl<int64_t>&&)").skip())
               .put(new Info("c10::SymIntArrayRef::iterator", "c10::SymIntArrayRef::const_iterator").cast().pointerTypes("SymInt"))
               .put(new Info("c10::EnumerationType<c10::TypeKind::LayoutType>").pointerTypes("LayoutEnumerationType"))
               .put(new Info("c10::EnumerationType<c10::TypeKind::ScalarTypeType>").pointerTypes("ScalarTypeEnumerationType"))
               .put(new Info("c10::EnumerationType<c10::TypeKind::MemoryFormatType>").pointerTypes("MemoryFormattEnumerationType"))
               .put(new Info("c10::SingleElementType<c10::TypeKind::ListType,c10::ListType>").pointerTypes("ListSingleElementType"))
               .put(new Info("c10::SingleElementType<c10::TypeKind::RRefType,c10::RRefType>").pointerTypes("RRefSingleElementType"))
               .put(new Info("c10::SingleElementType<c10::TypeKind::FutureType,c10::FutureType>").pointerTypes("FutureSingleElementType"))
               .put(new Info("c10::SingleElementType<c10::TypeKind::OptionalType,c10::OptionalType>").pointerTypes("OptionalSingleElementType"))
               .put(new Info("at::InferExpandGeometryResult<at::DimVector>").pointerTypes("DimVectorInferExpandGeometryResult"))
               .put(new Info("at::namedinference::TensorName").valueTypes("@Cast({\"\", \"at::namedinference::TensorName&&\"}) @StdMove TensorName").pointerTypes("TensorName"))
               .put(new Info("std::shared_ptr<torch::autograd::FunctionPreHook>").annotations("@SharedPtr").valueTypes("FunctionPreHook").pointerTypes("FunctionPreHook"))
               .put(new Info("std::unique_ptr<torch::autograd::FunctionPreHook>").annotations("@UniquePtr")
                                                                                 .valueTypes("@Cast({\"\", \"std::unique_ptr<torch::autograd::FunctionPreHook>&&\"}) FunctionPreHook")
                                                                                 .pointerTypes("FunctionPreHook"))
               .put(new Info("std::unique_ptr<torch::autograd::FunctionPostHook>").annotations("@UniquePtr")
                                                                                  .valueTypes("@Cast({\"\", \"std::unique_ptr<torch::autograd::FunctionPostHook>&&\"}) FunctionPostHook")
                                                                                  .pointerTypes("FunctionPostHook"))
//               .put(new Info("torch::jit::ScalarAttributeValue<c10::complex<double>,torch::jit::AttributeKind::c>").pointerTypes("ComplexAttr"))
//               .put(new Info("torch::jit::VectorAttributeValue<c10::complex<double>,torch::jit::AttributeKind::cs>").pointerTypes("ComplexValsAttr"))
               .put(new Info("torch::jit::ComplexAttr::ConstructorType", "torch::jit::ComplexAttr::ValueType").cast().pointerTypes("DoublePointer"))
               .put(new Info("torch::jit::ComplexValsAttr::ConstructorType", "torch::jit::ComplexValsAttr::ValueType").cast().pointerTypes("Pointer"))
//               .put(new Info("torch::jit::ScalarAttributeValue<double,torch::jit::AttributeKind::f>").pointerTypes("FloatAttr"))
//               .put(new Info("torch::jit::VectorAttributeValue<double,torch::jit::AttributeKind::fs>").pointerTypes("FloatsAttr"))
               .put(new Info("torch::jit::FloatAttr::ConstructorType", "torch::jit::FloatAttr::ValueType").cast().valueTypes("double").pointerTypes("DoublePointer"))
               .put(new Info("torch::jit::FloatsAttr::ConstructorType", "torch::jit::FloatsAttr::ValueType").cast().pointerTypes("DoubleVector"))
//               .put(new Info("torch::jit::ScalarAttributeValue<int64_t,torch::jit::AttributeKind::i>").pointerTypes("IntAttr"))
//               .put(new Info("torch::jit::VectorAttributeValue<int64_t,torch::jit::AttributeKind::is>").pointerTypes("IntsAttr"))
               .put(new Info("torch::jit::IntAttr::ConstructorType", "torch::jit::IntAttr::ValueType").cast().valueTypes("long").pointerTypes("LongPointer"))
               .put(new Info("torch::jit::IntsAttr::ConstructorType", "torch::jit::IntsAttr::ValueType").cast().pointerTypes("LongVector"))
//               .put(new Info("torch::jit::ScalarAttributeValue<std::string,torch::jit::AttributeKind::s>").pointerTypes("StringAttr"))
//               .put(new Info("torch::jit::VectorAttributeValue<std::string,torch::jit::AttributeKind::ss>").pointerTypes("StringsAttr"))
               .put(new Info("torch::jit::StringAttr::ConstructorType", "torch::jit::StringAttr::ValueType").annotations("@StdString").pointerTypes("BytePointer"))
               .put(new Info("torch::jit::StringsAttr::ConstructorType", "torch::jit::StringsAttr::ValueType").cast().pointerTypes("StringVector"))
//               .put(new Info("torch::jit::ScalarAttributeValue<at::Tensor,torch::jit::AttributeKind::t>").pointerTypes("TensorAttr"))
//               .put(new Info("torch::jit::VectorAttributeValue<at::Tensor,torch::jit::AttributeKind::ts>").pointerTypes("TensorsAttr"))
               .put(new Info("torch::jit::TensorAttr::ConstructorType", "torch::jit::TensorAttr::ValueType").cast().pointerTypes("Tensor"))
               .put(new Info("torch::jit::TensorsAttr::ConstructorType", "torch::jit::TensorsAttr::ValueType").cast().pointerTypes("TensorVector"))
//               .put(new Info("torch::jit::ScalarAttributeValue<c10::TypePtr,torch::jit::AttributeKind::ty>").pointerTypes("TypeAttr"))
//               .put(new Info("torch::jit::VectorAttributeValue<c10::TypePtr,torch::jit::AttributeKind::tys>").pointerTypes("TypesAttr"))
               .put(new Info("torch::jit::TypeAttr::ConstructorType", "torch::jit::TypeAttr::ValueType").cast().pointerTypes("Type.TypePtr"))
               .put(new Info("torch::jit::TypesAttr::ConstructorType", "torch::jit::TypesAttr::ValueType").cast().pointerTypes("TypeVector"))
//               .put(new Info("torch::jit::ScalarAttributeValue<at::IValue,torch::jit::AttributeKind::ival>").pointerTypes("IValueAttr"))
               .put(new Info("torch::jit::IValueAttr::ConstructorType", "torch::jit::IValueAttr::ValueType").cast().pointerTypes("IValue"))
               .put(new Info("std::shared_ptr<torch::jit::Graph>").annotations("@SharedPtr").pointerTypes("Graph"))
               .put(new Info("std::shared_ptr<torch::jit::Operator>").annotations("@SharedPtr").pointerTypes("Operator"))
               .put(new Info("std::shared_ptr<torch::jit::Resolver>", "torch::jit::ResolverPtr").annotations("@SharedPtr").pointerTypes("Resolver"))
               .put(new Info("std::shared_ptr<torch::jit::SugaredValue>", "torch::jit::SugaredValuePtr").annotations("@SharedPtr").pointerTypes("SugaredValue"))
               .put(new Info("std::shared_ptr<const torch::jit::CompilationUnit>").annotations("@SharedPtr")
                                                                                  .valueTypes("@Cast(\"const torch::jit::CompilationUnit*\") CompilationUnit")
                                                                                  .pointerTypes("CompilationUnit"))
               .put(new Info("std::unique_ptr<torch::jit::AttributeValue>", "Ptr").annotations("@UniquePtr").pointerTypes("AttributeValue"))
               .put(new Info("std::unique_ptr<torch::jit::TokenTrie>", "TokenTriePtr").annotations("@UniquePtr").pointerTypes("TokenTrie"))
               .put(new Info("torch::jit::TokenTrie").immutable())
               .put(new Info("torch::cuda::device_count").javaNames("cuda_device_count"))
               .put(new Info("torch::cuda::is_available").javaNames("cuda_is_available"))
               .put(new Info("torch::cuda::manual_seed").javaNames("cuda_manual_seed"))
               .put(new Info("torch::cuda::manual_seed_all").javaNames("cuda_manual_seed_all"))
               .put(new Info("torch::cuda::synchronize").javaNames("cuda_synchronize"))
               .put(new Info("torch::jit::Const").pointerTypes("ConstExpr"))
               .put(new Info("torch::jit::Node").pointerTypes("JitNode"))
               .put(new Info("torch::jit::Module").pointerTypes("JitModule"))
               .put(new Info("torch::jit::Object").pointerTypes("JitObject"))
               .put(new Info("torch::jit::String").pointerTypes("JitString"))
               .put(new Info("torch::jit::generic_graph_node_list<torch::jit::Node>").pointerTypes("graph_node_list"))
               .put(new Info("torch::jit::generic_graph_node_list_iterator<torch::jit::Node>").pointerTypes("graph_node_list_iterator"))

               .put(new Info("torch::jit::slot_list_impl<torch::jit::detail::ModulePolicy>", "torch::jit::module_list").pointerTypes("module_list"))
               .put(new Info("torch::jit::slot_iterator_impl<torch::jit::detail::ModulePolicy>").pointerTypes("module_iterator"))
               .put(new Info("torch::jit::slot_iterator_impl<torch::jit::detail::ModulePolicy>::value_type").pointerTypes("JitModule"))
               .put(new Info("torch::jit::Named<torch::jit::Module>").pointerTypes("NamedJitModule"))
               .put(new Info("torch::jit::detail::NamedPolicy<torch::jit::detail::ModulePolicy>").pointerTypes("NamedModulePolicy"))
               .put(new Info("torch::jit::slot_list_impl<torch::jit::detail::NamedPolicy<torch::jit::detail::ModulePolicy> >", "torch::jit::named_module_list").pointerTypes("named_module_list"))
               .put(new Info("torch::jit::slot_iterator_impl<torch::jit::detail::NamedPolicy<torch::jit::detail::ModulePolicy> >").pointerTypes("named_module_iterator"))
               .put(new Info("torch::jit::slot_iterator_impl<torch::jit::detail::NamedPolicy<torch::jit::detail::ModulePolicy> >::value_type").pointerTypes("NamedJitModule"))

               .put(new Info("torch::jit::slot_list_impl<torch::jit::detail::ParameterPolicy>", "torch::jit::parameter_list").pointerTypes("parameter_list"))
               .put(new Info("torch::jit::slot_iterator_impl<torch::jit::detail::ParameterPolicy>").pointerTypes("parameter_iterator"))
               .put(new Info("torch::jit::slot_iterator_impl<torch::jit::detail::ParameterPolicy>::value_type").pointerTypes("Tensor"))
               .put(new Info("torch::jit::Named<at::Tensor>").pointerTypes("NamedTensor"))
               .put(new Info("torch::jit::detail::NamedPolicy<torch::jit::detail::ParameterPolicy>").pointerTypes("NamedParameterPolicy"))
               .put(new Info("torch::jit::slot_list_impl<torch::jit::detail::NamedPolicy<torch::jit::detail::ParameterPolicy> >", "torch::jit::named_parameter_list").pointerTypes("named_parameter_list"))
               .put(new Info("torch::jit::slot_iterator_impl<torch::jit::detail::NamedPolicy<torch::jit::detail::ParameterPolicy> >").pointerTypes("named_parameter_iterator"))
               .put(new Info("torch::jit::slot_iterator_impl<torch::jit::detail::NamedPolicy<torch::jit::detail::ParameterPolicy> >::value_type").pointerTypes("NamedTensor"))

               .put(new Info("torch::jit::slot_list_impl<torch::jit::detail::AttributePolicy>", "torch::jit::attribute_list").pointerTypes("attribute_list"))
               .put(new Info("torch::jit::slot_iterator_impl<torch::jit::detail::AttributePolicy>").pointerTypes("attribute_iterator"))
               .put(new Info("torch::jit::slot_iterator_impl<torch::jit::detail::AttributePolicy>::value_type").pointerTypes("IValue"))
               .put(new Info("torch::jit::Named<c10::IValue>").pointerTypes("NamedIValue"))
               .put(new Info("torch::jit::detail::NamedPolicy<torch::jit::detail::AttributePolicy>").pointerTypes("NamedAttributePolicy"))
               .put(new Info("torch::jit::slot_list_impl<torch::jit::detail::NamedPolicy<torch::jit::detail::AttributePolicy> >", "torch::jit::named_attribute_list").pointerTypes("named_attribute_list"))
               .put(new Info("torch::jit::slot_iterator_impl<torch::jit::detail::NamedPolicy<torch::jit::detail::AttributePolicy> >").pointerTypes("named_attribute_iterator"))
               .put(new Info("torch::jit::slot_iterator_impl<torch::jit::detail::NamedPolicy<torch::jit::detail::AttributePolicy> >::value_type").pointerTypes("NamedIValue"))

               .put(new Info("torch::jit::slot_list_impl<torch::jit::detail::BufferPolicy>", "torch::jit::buffer_list").pointerTypes("buffer_list"))
               .put(new Info("torch::jit::slot_iterator_impl<torch::jit::detail::BufferPolicy>").pointerTypes("buffer_iterator"))
               .put(new Info("torch::jit::slot_iterator_impl<torch::jit::detail::BufferPolicy>::value_type").pointerTypes("Tensor"))
               .put(new Info("torch::jit::Named<at::Tensor>").pointerTypes("NamedTensor"))
               .put(new Info("torch::jit::detail::NamedPolicy<torch::jit::detail::BufferPolicy>").pointerTypes("NamedBufferPolicy"))
               .put(new Info("torch::jit::slot_list_impl<torch::jit::detail::NamedPolicy<torch::jit::detail::BufferPolicy> >", "torch::jit::named_buffer_list").pointerTypes("named_buffer_list"))
               .put(new Info("torch::jit::slot_iterator_impl<torch::jit::detail::NamedPolicy<torch::jit::detail::BufferPolicy> >").pointerTypes("named_buffer_iterator"))
               .put(new Info("torch::jit::slot_iterator_impl<torch::jit::detail::NamedPolicy<torch::jit::detail::BufferPolicy> >::value_type").pointerTypes("NamedTensor"))

               .put(new Info("torch::jit::tracer::warn_fn_type", "warn_fn_type").cast().pointerTypes("warn_fn_type"))
               .put(new Info("torch::jit::Maybe<torch::jit::Def>").pointerTypes("DefMaybe"))
               .put(new Info("torch::jit::Maybe<torch::jit::Expr>").pointerTypes("ExprMaybe"))
               .put(new Info("torch::jit::Maybe<torch::jit::Var>").pointerTypes("VarMaybe"))
               .put(new Info("torch::jit::Compound::map", "torch::jit::Tree::map", "torch::jit::Maybe<torch::jit::Def>::map",
                             "torch::jit::Maybe<torch::jit::Expr>::map", "torch::jit::Maybe<torch::jit::Var>::map").skip())
               .put(new Info("torch::jit::Wrap<torch::jit::Block>").pointerTypes("BlockWrap"))
               .put(new Info("torch::jit::Wrap<torch::jit::Node>").pointerTypes("JitNodeWrap"))
               .put(new Info("torch::jit::Wrap<torch::jit::Value>").pointerTypes("ValueWrap"))

               .put(new Info("std::vector<torch::data::Example<> >",
                             "std::vector<torch::data::datasets::Dataset<torch::data::datasets::MNIST,torch::data::Example<> >::ExampleType>").pointerTypes("ExampleVector").define())
               .put(new Info("torch::data::Example<torch::Tensor,torch::Tensor>", "torch::data::Example<>").pointerTypes("Example"))
//               .put(new Info("torch::data::Example<torch::Tensor,torch::data::example::NoTarget>").pointerTypes("TensorExample"))
//               .put(new Info("torch::data::detail::SentinelIterator<std::vector<torch::data::Example<> > >").pointerTypes("ExampleSentinelIterator"))
//               .put(new Info("torch::data::detail::ValidIterator<std::vector<torch::data::Example<> > >").pointerTypes("ExampleValidIterator"))
//               .put(new Info("torch::data::detail::IteratorImpl<std::vector<torch::data::Example<> > >").pointerTypes("ExampleIteratorImpl"))
               .put(new Info("torch::data::Iterator<torch::data::Example<> >").purify().pointerTypes("ExampleIterator"))
               .put(new Info("torch::data::Iterator<std::vector<torch::data::Example<> > >").purify().pointerTypes("ExampleVectorIterator"))
               .put(new Info("torch::data::samplers::Sampler<std::vector<size_t> >", "torch::data::samplers::Sampler<>").pointerTypes("Sampler"))
               .put(new Info("torch::data::transforms::BatchTransform<std::vector<torch::data::Example<> >, torch::data::Example<> >",
                             "torch::data::transforms::Collation<torch::data::Example<> >").pointerTypes("ExampleCollation"))
               .put(new Info("torch::data::transforms::Stack<torch::data::Example<> >").pointerTypes("ExampleStack"))

               .put(new Info("torch::data::DataLoaderBase<torch::data::datasets::MapDataset<torch::data::datasets::MNIST,torch::data::transforms::Stack<torch::data::Example<> > >,torch::data::Example<>,std::vector<size_t> >",
                             "torch::data::DataLoaderBase<torch::data::datasets::MapDataset<torch::data::datasets::MNIST,torch::data::transforms::Stack<torch::data::Example<> > >,torch::data::datasets::MapDataset<torch::data::datasets::MNIST,torch::data::transforms::Stack<torch::data::Example<> > >::BatchType,torch::data::samplers::RandomSampler::BatchRequestType>")
                       .purify().pointerTypes("MNISTRandomDataLoaderBase"))
               .put(new Info("torch::data::StatelessDataLoader<torch::data::datasets::MapDataset<torch::data::datasets::MNIST,torch::data::transforms::Stack<torch::data::Example<> > >,torch::data::samplers::RandomSampler>").pointerTypes("MNISTRandomDataLoader"))
               .put(new Info("torch::data::datasets::Dataset<torch::data::datasets::MNIST,torch::data::Example<> >",
                             "torch::data::datasets::Dataset<MNIST>").pointerTypes("MNISTDataSet"))
               .put(new Info("torch::data::datasets::BatchDataset<torch::data::datasets::MNIST,std::vector<torch::data::Example<> >,at::ArrayRef<size_t> >",
                             "torch::data::datasets::BatchDataset<torch::data::datasets::MNIST,std::vector<torch::data::Example<> > >").pointerTypes("MNISTBatchDataset"))
               .put(new Info("torch::data::datasets::BatchDataset<torch::data::datasets::MNIST,std::vector<torch::data::Example<> >,at::ArrayRef<size_t> >::map")
                       .javaText("public native @ByVal MNISTMapDataset map(@ByVal ExampleStack transform);"))
//               .put(new Info("torch::data::datasets::BatchDataset<torch::data::datasets::MNIST,std::vector<torch::data::Example<> >,at::ArrayRef<size_t> >::map<torch::data::transforms::Stack<torch::data::Example<> > >")
//                       .javaNames("map"))
               .put(new Info("torch::data::datasets::MapDataset<torch::data::datasets::MNIST,torch::data::transforms::Stack<torch::data::Example<> > >").pointerTypes("MNISTMapDataset"))
               .put(new Info("torch::data::datasets::MapDataset<torch::data::datasets::MNIST,torch::data::transforms::Stack<torch::data::Example<> > >::reset").skip())
               .put(new Info("torch::data::datasets::MapDataset<torch::data::datasets::MNIST,torch::data::transforms::Stack<torch::data::Example<> > >::DatasetType").pointerTypes("MNIST"))
               .put(new Info("torch::data::datasets::BatchDataset<torch::data::datasets::MapDataset<torch::data::datasets::MNIST,torch::data::transforms::Stack<torch::data::Example<> > >,std::vector<torch::data::Example<> >,at::ArrayRef<size_t> >",
                             "torch::data::datasets::BatchDataset<torch::data::datasets::MapDataset<torch::data::datasets::MNIST,torch::data::transforms::Stack<torch::data::Example<> > >,torch::data::datasets::detail::optional_if_t<torch::data::datasets::MNIST::is_stateful,torch::data::transforms::Stack<torch::data::Example<> >::OutputBatchType>,torch::data::datasets::MNIST::BatchRequestType>")
                       .pointerTypes("MNISTMapBatchDataset"))
//               .put(new Info("torch::data::datasets::MapDataset<torch::data::datasets::MNIST,torch::data::transforms::Stack<torch::data::Example<> > >::BatchRequestType").pointerTypes("SizeTArrayRef"))
//               .put(new Info("torch::data::datasets::MapDataset<torch::data::datasets::MNIST,torch::data::transforms::Stack<torch::data::Example<> > >::OutputBatchType").pointerTypes("Example"))
               .put(new Info("torch::data::datasets::MapDataset<torch::data::datasets::MNIST,torch::data::transforms::Stack<torch::data::Example<> > >::get_batch")
                       .javaText("public native @Name(\"get_batch\") @ByVal Example get_batch_example(@ByVal SizeTArrayRef indices);"))

               .put(new Info("torch::nn::detail::ConvNdOptions<1>").pointerTypes("DetailConv1dOptions"))
               .put(new Info("torch::nn::detail::ConvNdOptions<2>").pointerTypes("DetailConv2dOptions"))
               .put(new Info("torch::nn::detail::ConvNdOptions<3>").pointerTypes("DetailConv3dOptions"))
               .put(new Info("torch::nn::ConvOptions<1>").pointerTypes("Conv1dOptions"))
               .put(new Info("torch::nn::ConvOptions<2>").pointerTypes("Conv2dOptions"))
               .put(new Info("torch::nn::ConvOptions<3>").pointerTypes("Conv3dOptions"))
               .put(new Info("torch::nn::functional::ConvFuncOptions<1>").pointerTypes("Conv1dFuncOptions"))
               .put(new Info("torch::nn::functional::ConvFuncOptions<2>").pointerTypes("Conv2dFuncOptions"))
               .put(new Info("torch::nn::functional::ConvFuncOptions<3>").pointerTypes("Conv3dFuncOptions"))
               .put(new Info("torch::nn::ConvTransposeOptions<1>").pointerTypes("ConvTranspose1dOptions"))
               .put(new Info("torch::nn::ConvTransposeOptions<2>").pointerTypes("ConvTranspose2dOptions"))
               .put(new Info("torch::nn::ConvTransposeOptions<3>").pointerTypes("ConvTranspose3dOptions"))
               .put(new Info("torch::nn::functional::ConvTransposeFuncOptions<1>").pointerTypes("ConvTranspose1dFuncOptions"))
               .put(new Info("torch::nn::functional::ConvTransposeFuncOptions<2>").pointerTypes("ConvTranspose2dFuncOptions"))
               .put(new Info("torch::nn::functional::ConvTransposeFuncOptions<3>").pointerTypes("ConvTranspose3dFuncOptions"))

               .put(new Info("torch::nn::ReflectionPadOptions<1>").pointerTypes("ReflectionPad1dOptions"))
               .put(new Info("torch::nn::ReflectionPadOptions<2>").pointerTypes("ReflectionPad2dOptions"))
               .put(new Info("torch::nn::ReflectionPadOptions<3>").pointerTypes("ReflectionPad3dOptions"))
               .put(new Info("torch::nn::ReplicationPadOptions<1>").pointerTypes("ReplicationPad1dOptions"))
               .put(new Info("torch::nn::ReplicationPadOptions<2>").pointerTypes("ReplicationPad2dOptions"))
               .put(new Info("torch::nn::ReplicationPadOptions<3>").pointerTypes("ReplicationPad3dOptions"))
               .put(new Info("torch::nn::ConstantPadOptions<1>").pointerTypes("ConstantPad1dOptions"))
               .put(new Info("torch::nn::ConstantPadOptions<2>").pointerTypes("ConstantPad2dOptions"))
               .put(new Info("torch::nn::ConstantPadOptions<3>").pointerTypes("ConstantPad3dOptions"))

               .put(new Info("torch::nn::AvgPoolOptions<1>", "torch::nn::functional::AvgPool1dFuncOptions").pointerTypes("AvgPool1dOptions"))
               .put(new Info("torch::nn::AvgPoolOptions<2>", "torch::nn::functional::AvgPool2dFuncOptions").pointerTypes("AvgPool2dOptions"))
               .put(new Info("torch::nn::AvgPoolOptions<3>", "torch::nn::functional::AvgPool3dFuncOptions").pointerTypes("AvgPool3dOptions"))
               .put(new Info("torch::nn::MaxPoolOptions<1>", "torch::nn::functional::MaxPool1dFuncOptions").pointerTypes("MaxPool1dOptions"))
               .put(new Info("torch::nn::MaxPoolOptions<2>", "torch::nn::functional::MaxPool2dFuncOptions").pointerTypes("MaxPool2dOptions"))
               .put(new Info("torch::nn::MaxPoolOptions<3>", "torch::nn::functional::MaxPool3dFuncOptions").pointerTypes("MaxPool3dOptions"))
               .put(new Info("torch::nn::AdaptiveAvgPoolOptions<torch::ExpandingArray<1> >", "torch::nn::functional::AdaptiveAvgPool1dFuncOptions").pointerTypes("AdaptiveAvgPool1dOptions"))
               .put(new Info("torch::nn::AdaptiveAvgPoolOptions<torch::ExpandingArrayWithOptionalElem<2> >", "torch::nn::functional::AdaptiveAvgPool2dFuncOptions").pointerTypes("AdaptiveAvgPool2dOptions"))
               .put(new Info("torch::nn::AdaptiveAvgPoolOptions<torch::ExpandingArrayWithOptionalElem<3> >", "torch::nn::functional::AdaptiveAvgPool3dFuncOptions").pointerTypes("AdaptiveAvgPool3dOptions"))
               .put(new Info("torch::nn::AdaptiveMaxPoolOptions<torch::ExpandingArray<1> >", "torch::nn::functional::AdaptiveMaxPool1dFuncOptions").pointerTypes("AdaptiveMaxPool1dOptions"))
               .put(new Info("torch::nn::AdaptiveMaxPoolOptions<torch::ExpandingArrayWithOptionalElem<2> >", "torch::nn::functional::AdaptiveMaxPool2dFuncOptions").pointerTypes("AdaptiveMaxPool2dOptions"))
               .put(new Info("torch::nn::AdaptiveMaxPoolOptions<torch::ExpandingArrayWithOptionalElem<3> >", "torch::nn::functional::AdaptiveMaxPool3dFuncOptions").pointerTypes("AdaptiveMaxPool3dOptions"))
               .put(new Info("torch::nn::MaxUnpoolOptions<1>").pointerTypes("MaxUnpool1dOptions"))
               .put(new Info("torch::nn::MaxUnpoolOptions<2>").pointerTypes("MaxUnpool2dOptions"))
               .put(new Info("torch::nn::MaxUnpoolOptions<3>").pointerTypes("MaxUnpool3dOptions"))
               .put(new Info("torch::nn::functional::MaxUnpoolFuncOptions<1>").pointerTypes("MaxUnpool1dFuncOptions"))
               .put(new Info("torch::nn::functional::MaxUnpoolFuncOptions<2>").pointerTypes("MaxUnpool2dFuncOptions"))
               .put(new Info("torch::nn::functional::MaxUnpoolFuncOptions<3>").pointerTypes("MaxUnpool3dFuncOptions"))
               .put(new Info("torch::nn::FractionalMaxPoolOptions<1>", "torch::nn::functional::FractionalMaxPool1dFuncOptions").pointerTypes("FractionalMaxPool1dOptions"))
               .put(new Info("torch::nn::FractionalMaxPoolOptions<2>", "torch::nn::functional::FractionalMaxPool2dFuncOptions").pointerTypes("FractionalMaxPool2dOptions"))
               .put(new Info("torch::nn::FractionalMaxPoolOptions<3>", "torch::nn::functional::FractionalMaxPool3dFuncOptions").pointerTypes("FractionalMaxPool3dOptions"))
               .put(new Info("torch::nn::LPPoolOptions<1>", "torch::nn::functional::LPPool1dFuncOptions").pointerTypes("LPPool1dOptions"))
               .put(new Info("torch::nn::LPPoolOptions<2>", "torch::nn::functional::LPPool2dFuncOptions").pointerTypes("LPPool2dOptions"))
               .put(new Info("torch::nn::LPPoolOptions<3>", "torch::nn::functional::LPPool3dFuncOptions").pointerTypes("LPPool3dOptions"))

               .put(new Info("std::shared_ptr<torch::nn::Module>").annotations("@SharedPtr")
                       .valueTypes("@Cast({\"\", \"std::shared_ptr<torch::nn::Module>\"}) Module").pointerTypes("Module"))
               .put(new Info("torch::nn::ModuleHolder<torch::nn::Module>").pointerTypes("ModuleHolder"))
               .put(new Info("torch::nn::Module::as").javaText("public Module asModule() { return this; }"))
               .put(new Info("torch::nn::Module::register_module<torch::nn::Module>").javaNames("register_module"))
               .put(new Info("std::shared_ptr<torch::nn::AnyModule>").annotations("@SharedPtr")
                       .valueTypes("@Cast({\"\", \"std::shared_ptr<torch::nn::AnyModule>\"}) AnyModule").pointerTypes("AnyModule"));

        mapModule(infoMap, "ModuleDict", true);
        mapModule(infoMap, "ModuleList", true);
        mapModule(infoMap, "Sequential", true);
        mapModule(infoMap, "ParameterDict", true);
        mapModule(infoMap, "ParameterList", true);

        mapModule(infoMap, "AdaptiveLogSoftmaxWithLoss", false);

        for (int i = 1; i <= 3; i++) {
            mapModule(infoMap, "BatchNorm" + i + "d", "torch::nn::BatchNormImplBase<" + i + ",torch::nn::BatchNorm" + i + "dImpl>",
                                                      "torch::nn::NormImplBase<" + i + ",torch::nn::BatchNorm" + i + "dImpl,torch::nn::BatchNormOptions>");
            mapModule(infoMap, "InstanceNorm" + i + "d", "torch::nn::InstanceNormImpl<" + i + ",torch::nn::InstanceNorm" + i + "dImpl>",
                                                         "torch::nn::NormImplBase<" + i + ",torch::nn::InstanceNorm" + i + "dImpl,torch::nn::InstanceNormOptions>");

            mapModule(infoMap, "Conv" + i + "d", "torch::nn::ConvNdImpl<" + i + ",torch::nn::Conv" + i + "dImpl>");
            mapModule(infoMap, "ConvTranspose" + i + "d", "torch::nn::ConvTransposeNdImpl<" + i + ",torch::nn::ConvTranspose" + i + "dImpl>",
                                                          "torch::nn::ConvNdImpl<" + i + ",torch::nn::ConvTranspose" + i + "dImpl>");

            mapModule(infoMap, "Dropout" + (i > 1 ? i + "d" : ""), "torch::nn::detail::_DropoutNd<torch::nn::Dropout" + (i > 1 ? i + "d" : "") + "Impl>");
        }
        mapModule(infoMap, "AlphaDropout", "torch::nn::detail::_DropoutNd<torch::nn::AlphaDropoutImpl>");
        mapModule(infoMap, "FeatureAlphaDropout", "torch::nn::detail::_DropoutNd<torch::nn::FeatureAlphaDropoutImpl>");

        mapModule(infoMap, "CosineSimilarity");
        mapModule(infoMap, "PairwiseDistance");

        mapModule(infoMap, "Embedding");
        mapModule(infoMap, "EmbeddingBag");

        mapModule(infoMap, "Fold");
        mapModule(infoMap, "Unfold");

        mapModule(infoMap, "Identity");
        mapModule(infoMap, "Linear");
        mapModule(infoMap, "Bilinear");
        mapModule(infoMap, "Flatten");
        mapModule(infoMap, "Unflatten");

        mapModule(infoMap, "L1Loss");
        mapModule(infoMap, "KLDivLoss");
        mapModule(infoMap, "MSELoss");
        mapModule(infoMap, "BCELoss");
        mapModule(infoMap, "HingeEmbeddingLoss");
        mapModule(infoMap, "MultiMarginLoss");
        mapModule(infoMap, "CosineEmbeddingLoss");
        mapModule(infoMap, "SmoothL1Loss");
        mapModule(infoMap, "HuberLoss");
        mapModule(infoMap, "MultiLabelMarginLoss");
        mapModule(infoMap, "SoftMarginLoss");
        mapModule(infoMap, "MultiLabelSoftMarginLoss");
        mapModule(infoMap, "TripletMarginLoss");
        mapModule(infoMap, "TripletMarginWithDistanceLoss");
        mapModule(infoMap, "CTCLoss");
        mapModule(infoMap, "PoissonNLLLoss");
        mapModule(infoMap, "MarginRankingLoss");
        mapModule(infoMap, "NLLLoss");
        mapModule(infoMap, "CrossEntropyLoss");
        mapModule(infoMap, "BCEWithLogitsLoss");

        for (int i = 1; i <= 3; i++) {
            mapModule(infoMap, "ReflectionPad" + i + "d", "torch::nn::ReflectionPadImpl<" + i + ",torch::nn::ReflectionPad" + i + "dImpl>");
            mapModule(infoMap, "ReplicationPad" + i + "d", "torch::nn::ReplicationPadImpl<" + i + ",torch::nn::ReplicationPad" + i + "dImpl>");
            mapModule(infoMap, "ConstantPad" + i + "d", "torch::nn::ConstantPadImpl<" + i + ",torch::nn::ConstantPad" + i + "dImpl>");
            if (i == 2) {
                mapModule(infoMap, "ZeroPad" + i + "d");
            }

            mapModule(infoMap, "AvgPool" + i + "d", "torch::nn::AvgPoolImpl<" + i + ",torch::nn::AvgPool" + i + "dImpl>");
            mapModule(infoMap, "MaxPool" + i + "d", "torch::nn::MaxPoolImpl<" + i + ",torch::nn::MaxPool" + i + "dImpl>");
            mapModule(infoMap, "AdaptiveAvgPool" + i + "d", "torch::nn::AdaptiveAvgPoolImpl<" + i + ",torch::ExpandingArray" + (i > 1 ? "WithOptionalElem<" : "<") + i + ">,torch::nn::AdaptiveAvgPool" + i + "dImpl>");
            mapModule(infoMap, "AdaptiveMaxPool" + i + "d", "torch::nn::AdaptiveMaxPoolImpl<" + i + ",torch::ExpandingArray" + (i > 1 ? "WithOptionalElem<" : "<") + i + ">,torch::nn::AdaptiveMaxPool" + i + "dImpl>");
            mapModule(infoMap, "MaxUnpool" + i + "d", "torch::nn::MaxUnpoolImpl<" + i + ",torch::nn::MaxUnpool" + i + "dImpl>");
            if (i > 1) {
                mapModule(infoMap, "FractionalMaxPool" + i + "d", "torch::nn::FractionalMaxPoolImpl<" + i + ",torch::nn::FractionalMaxPool" + i + "dImpl>");
            }
            if (i < 3) {
                mapModule(infoMap, "LPPool" + i + "d", "torch::nn::LPPoolImpl<" + i + ",torch::nn::LPPool" + i + "dImpl>");
            }
        }

        mapModule(infoMap, "RNN", "torch::nn::detail::RNNImplBase<torch::nn::RNNImpl>");
        mapModule(infoMap, "LSTM", "torch::nn::detail::RNNImplBase<torch::nn::LSTMImpl>");
        mapModule(infoMap, "GRU", "torch::nn::detail::RNNImplBase<torch::nn::GRUImpl>");
        mapModule(infoMap, "RNNCell", "torch::nn::detail::RNNCellImplBase<torch::nn::RNNCellImpl>");
        mapModule(infoMap, "LSTMCell", "torch::nn::detail::RNNCellImplBase<torch::nn::LSTMCellImpl>");
        mapModule(infoMap, "GRUCell", "torch::nn::detail::RNNCellImplBase<torch::nn::GRUCellImpl>");

        mapModule(infoMap, "PixelShuffle");
        mapModule(infoMap, "PixelUnshuffle");
        mapModule(infoMap, "Upsample");

        mapModule(infoMap, "ELU");
        mapModule(infoMap, "SELU");
        mapModule(infoMap, "Hardshrink");
        mapModule(infoMap, "Hardtanh");
        mapModule(infoMap, "LeakyReLU");
        mapModule(infoMap, "LogSigmoid");
        mapModule(infoMap, "Softmax");
        mapModule(infoMap, "Softmin");
        mapModule(infoMap, "LogSoftmax");
        mapModule(infoMap, "Softmax2d");
        mapModule(infoMap, "PReLU");
        mapModule(infoMap, "ReLU");
        mapModule(infoMap, "ReLU6");
        mapModule(infoMap, "RReLU");
        mapModule(infoMap, "CELU");
        mapModule(infoMap, "GLU");
        mapModule(infoMap, "GELU");
        mapModule(infoMap, "SiLU");
        mapModule(infoMap, "Mish");
        mapModule(infoMap, "Sigmoid");
        mapModule(infoMap, "Softplus");
        mapModule(infoMap, "Softshrink");
        mapModule(infoMap, "Softsign");
        mapModule(infoMap, "Tanh");
        mapModule(infoMap, "Tanhshrink");
        mapModule(infoMap, "Threshold");
        mapModule(infoMap, "MultiheadAttention");

        mapModule(infoMap, "LayerNorm");
        mapModule(infoMap, "LocalResponseNorm");
        mapModule(infoMap, "CrossMapLRN2d");
        mapModule(infoMap, "GroupNorm");

        mapModule(infoMap, "TransformerEncoderLayer");
        mapModule(infoMap, "TransformerDecoderLayer");
        mapModule(infoMap, "TransformerEncoder");
        mapModule(infoMap, "TransformerDecoder");
        mapModule(infoMap, "Transformer");

        infoMap.put(new Info("torch::optim::OptimizerCloneableOptions<torch::optim::AdagradOptions>",
                             "torch::optim::OptimizerCloneableOptions<AdagradOptions>").pointerTypes("OptimizerCloneableAdagradOptions"))
               .put(new Info("torch::optim::OptimizerCloneableParamState<torch::optim::AdagradParamState>",
                             "torch::optim::OptimizerCloneableParamState<AdagradParamState>").pointerTypes("OptimizerCloneableAdagradParamState"))
               .put(new Info("torch::optim::OptimizerCloneableOptions<torch::optim::AdamOptions>",
                             "torch::optim::OptimizerCloneableOptions<AdamOptions>").pointerTypes("OptimizerCloneableAdamOptions"))
               .put(new Info("torch::optim::OptimizerCloneableParamState<torch::optim::AdamParamState>",
                             "torch::optim::OptimizerCloneableParamState<AdamParamState>").pointerTypes("OptimizerCloneableAdamParamState"))
               .put(new Info("torch::optim::OptimizerCloneableOptions<torch::optim::AdamWOptions>",
                             "torch::optim::OptimizerCloneableOptions<AdamWOptions>").pointerTypes("OptimizerCloneableAdamWOptions"))
               .put(new Info("torch::optim::OptimizerCloneableParamState<torch::optim::AdamWParamState>",
                             "torch::optim::OptimizerCloneableParamState<AdamWParamState>").pointerTypes("OptimizerCloneableAdamWParamState"))
               .put(new Info("torch::optim::OptimizerCloneableOptions<torch::optim::LBFGSOptions>",
                             "torch::optim::OptimizerCloneableOptions<LBFGSOptions>").pointerTypes("OptimizerCloneableLBFGSOptions"))
               .put(new Info("torch::optim::OptimizerCloneableParamState<torch::optim::LBFGSParamState>",
                             "torch::optim::OptimizerCloneableParamState<LBFGSParamState>").pointerTypes("OptimizerCloneableLBFGSParamState"))
               .put(new Info("torch::optim::OptimizerCloneableOptions<torch::optim::RMSpropOptions>",
                             "torch::optim::OptimizerCloneableOptions<RMSpropOptions>").pointerTypes("OptimizerCloneableRMSpropOptions"))
               .put(new Info("torch::optim::OptimizerCloneableParamState<torch::optim::RMSpropParamState>",
                             "torch::optim::OptimizerCloneableParamState<RMSpropParamState>").pointerTypes("OptimizerCloneableRMSpropParamState"))
               .put(new Info("torch::optim::OptimizerCloneableOptions<torch::optim::SGDOptions>",
                             "torch::optim::OptimizerCloneableOptions<SGDOptions>").pointerTypes("OptimizerCloneableSGDOptions"))
               .put(new Info("torch::optim::OptimizerCloneableParamState<torch::optim::SGDParamState>",
                             "torch::optim::OptimizerCloneableParamState<SGDParamState>").pointerTypes("OptimizerCloneableSGDParamState"))

               .put(new Info("c10::intrusive_ptr_target", "c10::nullopt", "c10::nullopt_t", "c10::string_view", "c10::GeneratorImpl", "c10::impl::DeviceGuardImplInterface",
                             "PyObject", "std::function<PyObject*(void*)>", "THPObjectPtr", "pyobj_list", "std::chrono::milliseconds", "std::exception_ptr", "std::type_info",
                             "std::is_same<torch::detail::pack<true>,torch::detail::pack<true> >", "at::cuda::NVRTC", "at::RecordFunctionCallback", "at::StepCallbacks", "THCState", "THHState",
                             "torch::autograd::ViewInfo", "torch::jit::InlinedCallStackPtr", "InlinedCallStackPtr", "torch::jit::ScopePtr", "torch::jit::BackendDebugInfoRecorder",
                             "torch::detail::TensorDataContainer", "std::shared_ptr<caffe2::serialize::PyTorchStreamReader>", "caffe2::serialize::PyTorchStreamWriter",
                             "c10::impl::PyInterpreter", "std::function<at::Tensor(const at::Tensor&)>",

                             "c10::optional<PyObject*>", "c10::optional<c10::string_view>", "c10::optional<std::vector<c10::string_view> >", "c10::optional<std::chrono::milliseconds>",
                             "c10::intrusive_ptr<c10::ivalue::Object>", "c10::ArrayRef<c10::intrusive_ptr<c10::ivalue::Object> >", "c10::intrusive_ptr<c10::TensorImpl>",
                             "c10::intrusive_ptr<torch::jit::Tree>", "at::SmallVector<torch::jit::TreeRef,4>", "std::unordered_map<torch::jit::TreeRef,std::string>",
                             "torch::jit::Maybe<c10::List<torch::jit::Property> >", "torch::jit::Maybe<c10::List<torch::jit::Assign> >",
                             "c10::optional<c10::VaryingShape<int64_t>::ListOfOptionalElements>", "c10::optional<c10::VaryingShape<c10::Stride>::ListOfOptionalElements>",
                             "c10::optional<torch::autograd::ViewInfo>", "c10::optional<std::reference_wrapper<const std::string> >",
                             "c10::optional<torch::nn::TripletMarginWithDistanceLossOptions::distance_function_t>",
                             "c10::optional<torch::nn::functional::TripletMarginWithDistanceLossFuncOptions::distance_function_t>",
                             "std::tuple<at::Tensor,c10::optional<std::vector<int64_t> >,c10::optional<std::vector<double> >,c10::optional<bool> >",
                             "c10::optional<std::shared_ptr<torch::jit::CompilationUnit> >", "c10::optional<std::weak_ptr<torch::jit::CompilationUnit> >",
                             "std::vector<std::shared_ptr<std::string> >", "std::reference_wrapper<const c10::FunctionSchema>",

                             "std::enable_shared_from_this<c10::Type>",
                             "std::enable_shared_from_this<c10::SharedType>",
                             "std::enable_shared_from_this<c10::SymbolicIntNode>",
                             "std::enable_shared_from_this<torch::autograd::ForwardGrad>",
                             "std::enable_shared_from_this<torch::autograd::Node>",
                             "std::enable_shared_from_this<torch::jit::SugaredValue>", "std::enable_shared_from_this<SugaredValue>",
                             "std::enable_shared_from_this<torch::jit::tracer::TracingState>", "std::enable_shared_from_this<TracingState>",
                             "std::enable_shared_from_this<torch::nn::Module>", "std::enable_shared_from_this<Module>").cast().pointerTypes("Pointer"))

               .put(new Info("at::Tensor::toString", "at::TensorBase::toString", "at::DeprecatedTypeProperties::toString", "torch::jit::Graph::toString").javaText("public native @StdString String toString();"))
               .put(new Info("torch::jit::tracer::pauseTracing()").javaText("@Namespace(\"torch::jit::tracer\") public static native @ByVal @Cast(\"std::function<void()>*\") Pointer pauseTracing();"))
               .put(new Info("torch::jit::ProfileOp::getCallback()", "torch::jit::ProfileIValueOp::getCallback()").javaText(
                       "public native @ByVal @Cast(\"std::function<void(std::vector<c10::IValue>&)>*\") Pointer getCallback();"))
               .put(new Info("at::indexing::slicePrefix1sSize").javaText(
                       "@Namespace(\"at::indexing\") public static native @ByVal @Cast(\"c10::ArrayRef<int64_t>*\") LongArrayRef slicePrefix1sSize(@ByRef @Cast(\"c10::ArrayRef<int64_t>*\") LongArrayRef sizes);"))
               .put(new Info("torch::optim::AdamOptions::betas", "torch::optim::AdamWOptions::betas").javaText(
                       "public native @Cast(\"std::tuple<double,double>*\") @ByRef @NoException DoublePointer betas();"))
               .put(new Info("torch::optim::Adagrad::step", "torch::optim::Adam::step", "torch::optim::AdamW::step",
                             "torch::optim::LBFG::step", "torch::optim::RMSprop::step", "torch::optim::SGD::step").javaText(
                       "public native @ByVal Tensor step(@ByVal(nullValue = \"torch::optim::Optimizer::LossClosure(nullptr)\") LossClosure closure);\n"
                     + "public native @ByVal Tensor step();\n"))

               .put(new Info("c10::DeleterFnPtr").cast().valueTypes("Deleter", "Pointer", "long"))
               .put(new Info("std::function<void(void*)>").pointerTypes("Deleter", "@Cast(\"void(*)(void*)\") Pointer", "@Cast(\"void(*)(void*)\") long"))
               .put(new Info("std::function<void()>").pointerTypes("Func"))
               .put(new Info("std::function<std::string(void)>").pointerTypes("Fetcher"))
               .put(new Info("std::function<void(const std::string&)>").pointerTypes("Logger"))
               .put(new Info("std::function<void(const c10::DDPLoggingData&)>",
                             "std::function<void(const DDPLoggingData&)>").pointerTypes("DataLogger"))
               .put(new Info("std::function<c10::TypePtr(c10::TypePtr)>",
                             "std::function<c10::TypePtr(TypePtr)>").pointerTypes("TypeMapper"))
               .put(new Info("std::function<torch::jit::Value*(torch::jit::Value*)>",
                             "std::function<torch::jit::Value*(Value*)>").pointerTypes("ValueMapper"))
               .put(new Info("std::function<void(torch::jit::GraphFunction&)>",
                             "std::function<void(GraphFunction&)>").pointerTypes("GraphFunctionCreator"))
               .put(new Info("std::function<void(torch::jit::Module&)>",
                             "std::function<void(Module&)>").pointerTypes("ModuleFunction"))
               .put(new Info("std::function<void(std::vector<c10::IValue>&)>",
                             "std::function<void(std::vector<IValue>&)>").pointerTypes("IValueCallback"))
               .put(new Info("std::function<bool(std::ostream&,const IValue&v)>").pointerTypes("CustomFormatter"))
               .put(new Info("std::function<bool(const IValue&)>").pointerTypes("IValueVisitor"))
               .put(new Info("std::function<size_t(char*,size_t)>").pointerTypes("Reader"))
               .put(new Info("std::function<at::DataPtr(const std::string&)>").pointerTypes("RecordReader"))
               .put(new Info("std::function<void(const char*data_start,size_tdata_len)>",
                             "std::function<void(const char*,size_t)>").pointerTypes("Writer"))
               .put(new Info("std::function<std::string(const at::Tensor&)>").pointerTypes("TensorIdGetter"))
               .put(new Info("std::function<c10::QualifiedName(const c10::ClassTypePtr&)>").pointerTypes("TypeRenamer"))
               .put(new Info("std::function<size_t(uint64_tpos,void*buf,size_tnbytes)>").pointerTypes("ReadFunction"))
               .put(new Info("std::function<size_t(const void*,size_t)>").pointerTypes("WriteFunction"))
               .put(new Info("std::function<size_t(void)>").pointerTypes("SizeFunction"))
               .put(new Info("std::function<Tensor()>").pointerTypes("LossClosure"))
               .put(new Info("std::function<Tensor(const Tensor&,const Tensor&)>",
                             "torch::nn::TripletMarginWithDistanceLossOptions::distance_function_t",
                             "torch::nn::functional::TripletMarginWithDistanceLossFuncOptions::distance_function_t").pointerTypes("DistanceFunction"))
               .put(new Info("c10::TypePtr (*)(const std::string&)", "torch::jit::Unpickler::TypeParserT").pointerTypes("TypeParser").define(false))
        ;
    }

    public static class Deleter extends FunctionPointer {
        static { Loader.load(); }
        /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
        public    Deleter(Pointer p) { super(p); }
        protected Deleter() { allocate(); }
        private native void allocate();
        public native void call(Pointer p);
    }

    public static class Func extends FunctionPointer {
        static { Loader.load(); }
        /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
        public    Func(Pointer p) { super(p); }
        protected Func() { allocate(); }
        private native void allocate();
        public native void call();
    }

    public static class Fetcher extends FunctionPointer {
        static { Loader.load(); }
        /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
        public    Fetcher(Pointer p) { super(p); }
        protected Fetcher() { allocate(); }
        private native void allocate();
        public native @StdString String call();
    }

    public static class Logger extends FunctionPointer {
        static { Loader.load(); }
        /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
        public    Logger(Pointer p) { super(p); }
        protected Logger() { allocate(); }
        private native void allocate();
        public native void call(@Cast({"", "const std::string&"}) @StdString String s);
    }

    public static class DataLogger extends FunctionPointer {
        static { Loader.load(); }
        /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
        public    DataLogger(Pointer p) { super(p); }
        protected DataLogger() { allocate(); }
        private native void allocate();
        public native void call(@ByRef @Cast("const c10::DDPLoggingData*") Pointer d);
    }

    public static class TypeMapper extends FunctionPointer {
        static { Loader.load(); }
        /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
        public    TypeMapper(Pointer p) { super(p); }
        protected TypeMapper() { allocate(); }
        private native void allocate();
        public native @ByVal @Cast("c10::TypePtr*") Pointer call(@ByVal @Cast("c10::TypePtr*") Pointer t);
    }

    public static class ValueMapper extends FunctionPointer {
        static { Loader.load(); }
        /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
        public    ValueMapper(Pointer p) { super(p); }
        protected ValueMapper() { allocate(); }
        private native void allocate();
        public native @Cast("torch::jit::Value*") Pointer call(@Cast("torch::jit::Value*") Pointer v);
    }

    public static class GraphFunctionCreator extends FunctionPointer {
        static { Loader.load(); }
        /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
        public    GraphFunctionCreator(Pointer p) { super(p); }
        protected GraphFunctionCreator() { allocate(); }
        private native void allocate();
        public native void call(@ByRef @Cast("torch::jit::GraphFunction*") Pointer m);
    }

    public static class ModuleFunction extends FunctionPointer {
        static { Loader.load(); }
        /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
        public    ModuleFunction(Pointer p) { super(p); }
        protected ModuleFunction() { allocate(); }
        private native void allocate();
        public native void call(@ByRef @Cast("torch::jit::Module*") Pointer m);
    }

    public static class IValueCallback extends FunctionPointer {
        static { Loader.load(); }
        /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
        public    IValueCallback(Pointer p) { super(p); }
        protected IValueCallback() { allocate(); }
        private native void allocate();
        public native void call(@ByRef @Cast("std::vector<c10::IValue>*") Pointer v);
    }

    public static class CustomFormatter extends FunctionPointer {
        static { Loader.load(); }
        /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
        public    CustomFormatter(Pointer p) { super(p); }
        protected CustomFormatter() { allocate(); }
        private native void allocate();
        public native boolean call(@ByRef @Cast("std::ostream*") Pointer o, @ByRef @Cast("const c10::IValue*") Pointer v);
    }

    public static class IValueVisitor extends FunctionPointer {
        static { Loader.load(); }
        /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
        public    IValueVisitor(Pointer p) { super(p); }
        protected IValueVisitor() { allocate(); }
        private native void allocate();
        public native boolean call(@ByRef @Cast("const c10::IValue*") Pointer v);
    }

    public static class Reader extends FunctionPointer {
        static { Loader.load(); }
        /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
        public    Reader(Pointer p) { super(p); }
        protected Reader() { allocate(); }
        private native void allocate();
        public native @Cast("size_t") long call(@Cast("char*") Pointer data_start, @Cast("size_t") long data_len);
    }

    public static class RecordReader extends FunctionPointer {
        static { Loader.load(); }
        /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
        public    RecordReader(Pointer p) { super(p); }
        protected RecordReader() { allocate(); }
        private native void allocate();
        public native @StdMove @ByVal @Cast("c10::DataPtr*") Pointer call(@ByRef @Cast("const std::string*") Pointer s);
    }

    public static class Writer extends FunctionPointer {
        static { Loader.load(); }
        /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
        public    Writer(Pointer p) { super(p); }
        protected Writer() { allocate(); }
        private native void allocate();
        public native void call(@Cast("const char*") Pointer data_start, @Cast("size_t") long data_len);
    }

    public static class TensorIdGetter extends FunctionPointer {
        static { Loader.load(); }
        /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
        public    TensorIdGetter(Pointer p) { super(p); }
        protected TensorIdGetter() { allocate(); }
        private native void allocate();
        public native @ByVal @Cast("std::string*") Pointer call(@ByRef @Cast("const at::Tensor*") Pointer t);
    }

    public static class TypeRenamer extends FunctionPointer {
        static { Loader.load(); }
        /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
        public    TypeRenamer(Pointer p) { super(p); }
        protected TypeRenamer() { allocate(); }
        private native void allocate();
        public native @ByVal @Cast("c10::QualifiedName*") Pointer call(@ByRef @Cast("const c10::ClassTypePtr*") Pointer t);
    }

    public static class ReadFunction extends FunctionPointer {
        static { Loader.load(); }
        /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
        public    ReadFunction(Pointer p) { super(p); }
        protected ReadFunction() { allocate(); }
        private native void allocate();
        public native @Cast("size_t") long call(@Cast("uint64_t") long pos, Pointer buf, @Cast("size_t") long nbytes);
    }

    public static class WriteFunction extends FunctionPointer {
        static { Loader.load(); }
        /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
        public    WriteFunction(Pointer p) { super(p); }
        protected WriteFunction() { allocate(); }
        private native void allocate();
        public native @Cast("size_t") long call(@Const Pointer buf, @Cast("size_t") long nbytes);
    }

    public static class SizeFunction extends FunctionPointer {
        static { Loader.load(); }
        /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
        public    SizeFunction(Pointer p) { super(p); }
        protected SizeFunction() { allocate(); }
        private native void allocate();
        public native @Cast("size_t") long call();
    }

    public static class LossClosure extends FunctionPointer {
        static { Loader.load(); }
        /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
        public    LossClosure(Pointer p) { super(p); }
        protected LossClosure() { allocate(); }
        private native void allocate();
        public native @ByVal @Cast("at::Tensor*") Pointer call();
    }

    public static class DistanceFunction extends FunctionPointer {
        static { Loader.load(); }
        /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
        public    DistanceFunction(Pointer p) { super(p); }
        protected DistanceFunction() { allocate(); }
        private native void allocate();
        public native @ByVal @Cast("at::Tensor*") Pointer call(@ByRef @Cast("const at::Tensor*") Pointer t1, @ByRef @Cast("const at::Tensor*") Pointer t2);
    }

    public static class TypeParser extends FunctionPointer {
        static { Loader.load(); }
        /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
        public    TypeParser(Pointer p) { super(p); }
        protected TypeParser() { allocate(); }
        private native void allocate();
        public native @ByVal @Cast("c10::TypePtr*") Pointer call(@ByRef @Cast("const std::string*") Pointer s);
    }

    @Namespace("std") public static native @MemberGetter @ByRef @Cast("std::istream*") Pointer cin();
    @Namespace("std") public static native @MemberGetter @ByRef @Cast("std::ostream*") Pointer cout();
    @Namespace("std") public static native @MemberGetter @ByRef @Cast("std::ostream*") Pointer cerr();
    @Namespace("std") public static native @MemberGetter @ByRef @Cast("std::ostream*") Pointer clog();
}
