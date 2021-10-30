/*******************************************************************************
* Copyright 2018-2019 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

/// @example cpu_cnn_inference_int8.cpp
/// @copybrief cpu_cnn_inference_int8_cpp
/// > Annotated version: @ref cpu_cnn_inference_int8_cpp

/// @page cpu_cnn_inference_int8_cpp CNN int8 inference example
/// This C++ API example demonstrates how to run AlexNet's conv3 and relu3
/// with int8 data type.
///
/// > Example code: @ref cpu_cnn_inference_int8.cpp

import org.bytedeco.javacpp.*;

import org.bytedeco.dnnl.*;
import static org.bytedeco.dnnl.global.dnnl.*;

public class CpuCnnInferenceInt8 {

    static long product(long[] dims) {
        long accumulate = 1;
        for (int i = 0; i < dims.length; i++) accumulate *= dims[i];
        return accumulate;
    }

    static void simple_net_int8() throws Exception {
        engine cpu_engine = new engine(engine.kind.cpu, 0);
        stream s = new stream(cpu_engine);

        int batch = 8;

/// Configure tensor shapes
/// @snippet cpu_cnn_inference_int8.cpp Configure tensor shapes
//[Configure tensor shapes]
        // AlexNet: conv3
        // {batch, 256, 13, 13} (x)  {384, 256, 3, 3}; -> {batch, 384, 13, 13}
        // strides: {1, 1}
        long[] conv_src_tz = { batch, 256, 13, 13 };
        long[] conv_weights_tz = { 384, 256, 3, 3 };
        long[] conv_bias_tz = { 384 };
        long[] conv_dst_tz = { batch, 384, 13, 13 };
        long[] conv_strides = { 1, 1 };
        long[] conv_padding = { 1, 1 };
//[Configure tensor shapes]

/// Next, the example configures the scales used to quantize f32 data
/// into int8. For this example, the scaling value is chosen as an
/// arbitrary number, although in a realistic scenario, it should be
/// calculated from a set of precomputed values as previously mentioned.
/// @snippet cpu_cnn_inference_int8.cpp Choose scaling factors
//[Choose scaling factors]
        // Choose scaling factors for input, weight, output and bias quantization
        float[] src_scales = { 1.8f };
        float[] weight_scales = { 2.0f };
        float[] bias_scales = { 1.0f };
        float[] dst_scales = { 0.55f };

        // Choose channel-wise scaling factors for convolution
        float[] conv_scales = new float[384];
        int scales_half = 384 / 2;
        for (int i = 0;               i < scales_half;        i++) conv_scales[i] = 0.3f;
        for (int i = scales_half + 1; i < conv_scales.length; i++) conv_scales[i] = 0.8f;
//[Choose scaling factors]

/// The *source, weights, bias* and *destination* datasets use the single-scale
/// format with mask set to '0', while the *output* from the convolution
/// (conv_scales) will use the array format where mask = 2 corresponding
/// to the output dimension.
/// @snippet cpu_cnn_inference_int8.cpp Set scaling mask
//[Set scaling mask]
        int src_mask = 0;
        int weight_mask = 0;
        int bias_mask = 0;
        int dst_mask = 0;
        int conv_mask = 2; // 1 << output_channel_dim
//[Set scaling mask]

        // Allocate input and output buffers for user data
        float[] user_src = new float[batch * 256 * 13 * 13];
        float[] user_dst = new float[batch * 384 * 13 * 13];

        // Allocate and fill buffers for weights and bias
        float[] conv_weights = new float[(int)product(conv_weights_tz)];
        float[] conv_bias = new float[(int)product(conv_bias_tz)];

/// Create the memory primitives for user data (source, weights, and bias).
/// The user data will be in its original 32-bit floating point format.
/// @snippet cpu_cnn_inference_int8.cpp Allocate buffers
//[Allocate buffers]
        memory user_src_memory = new memory(
                new memory.desc(conv_src_tz, memory.data_type.f32, memory.format_tag.nchw),
                cpu_engine, new FloatPointer(user_src));
        memory user_weights_memory = new memory(
                new memory.desc(conv_weights_tz, memory.data_type.f32, memory.format_tag.oihw),
                cpu_engine, new FloatPointer(conv_weights));
        memory user_bias_memory = new memory(
                new memory.desc(conv_bias_tz, memory.data_type.f32, memory.format_tag.x),
                cpu_engine, new FloatPointer(conv_bias));
//[Allocate buffers]

/// Create a memory descriptor for each convolution parameter.
/// The convolution data uses 8-bit integer values, so the memory
/// descriptors are configured as:
///
/// * 8-bit unsigned (u8) for source and destination.
/// * 8-bit signed (s8) for bias and weights.
///
///  > **Note**
///  > The destination type is chosen as *unsigned* because the
///  > convolution applies a ReLU operation where data results \f$\geq 0\f$.
/// @snippet cpu_cnn_inference_int8.cpp Create convolution memory descriptors
//[Create convolution memory descriptors]
        memory.desc conv_src_md = new memory.desc(conv_src_tz, memory.data_type.u8, memory.format_tag.any);
        memory.desc conv_bias_md = new memory.desc(conv_bias_tz, memory.data_type.s8, memory.format_tag.any);
        memory.desc conv_weights_md = new memory.desc(conv_weights_tz, memory.data_type.s8, memory.format_tag.any);
        memory.desc conv_dst_md = new memory.desc(conv_dst_tz, memory.data_type.u8, memory.format_tag.any);
//[Create convolution memory descriptors]

/// Create a convolution descriptor passing the int8 memory
/// descriptors as parameters.
/// @snippet cpu_cnn_inference_int8.cpp Create convolution descriptor
//[Create convolution descriptor]
        convolution_forward.desc conv_desc = new convolution_forward.desc(prop_kind.forward,
                algorithm.convolution_direct, conv_src_md, conv_weights_md, conv_bias_md,
                conv_dst_md, conv_strides, conv_padding, conv_padding);
//[Create convolution descriptor]

/// Configuring int8-specific parameters in an int8 primitive is done
/// via the Attributes Primitive. Create an attributes object for the
/// convolution and configure it accordingly.
/// @snippet cpu_cnn_inference_int8.cpp Configure scaling
//[Configure scaling]
        primitive_attr conv_attr = new primitive_attr();
        conv_attr.set_output_scales(conv_mask, conv_scales);
//[Configure scaling]

/// The ReLU layer from Alexnet is executed through the PostOps feature. Create
/// a PostOps object and configure it to execute an _eltwise relu_ operation.
/// @snippet cpu_cnn_inference_int8.cpp Configure post-ops
//[Configure post-ops]
        float ops_scale = 1.f;
        float ops_alpha = 0.f; // relu negative slope
        float ops_beta = 0.f;
        post_ops ops = new post_ops();
        ops.append_eltwise(ops_scale, algorithm.eltwise_relu, ops_alpha, ops_beta);
        conv_attr.set_post_ops(ops);
//[Configure post-ops]

        // check if int8 convolution is supported
        try {
            convolution_forward.primitive_desc conv_prim_desc = new convolution_forward.primitive_desc(
                    conv_desc, conv_attr, cpu_engine);
        } catch (Exception e) {
            if (e.getMessage().contains("status = " + dnnl_unimplemented)) {
                System.err.println("Intel DNNL does not have int8 convolution "
                                 + "implementation that supports this system. Please refer to "
                                 + "the developer guide for details.");
            }
            throw e;
        }

/// Create a primitive descriptor using the convolution descriptor
/// and passing along the int8 attributes in the constructor. The primitive
/// descriptor for the convolution will contain the specific memory
/// formats for the computation.
/// @snippet cpu_cnn_inference_int8.cpp Create convolution primitive descriptor
//[Create convolution primitive descriptor]
        convolution_forward.primitive_desc conv_prim_desc = new convolution_forward.primitive_desc(
                conv_desc, conv_attr, cpu_engine);
//[Create convolution primitive descriptor]

/// Create a memory for each of the convolution's data input
/// parameters (source, bias, weights, and destination). Using the convolution
/// primitive descriptor as the creation parameter enables Intel DNNL
/// to configure the memory formats for the convolution.
///
/// Scaling parameters are passed to the reorder primitive via the attributes
/// primitive.
///
/// User memory must be transformed into convolution-friendly memory
/// (for int8 and memory format). A reorder layer performs the data
/// transformation from f32 (the original user data) into int8 format
/// (the data used for the convolution). In addition, the reorder
/// transforms the user data into the required memory format (as explained
/// in the simple_net example).
///
/// @snippet cpu_cnn_inference_int8.cpp Quantize data and weights
//[Quantize data and weights]
        memory conv_src_memory = new memory(conv_prim_desc.src_desc(), cpu_engine);
        primitive_attr src_attr = new primitive_attr();
        src_attr.set_output_scales(src_mask, src_scales);
        reorder.primitive_desc src_reorder_pd = new reorder.primitive_desc(cpu_engine,
                user_src_memory.get_desc(), cpu_engine,
                conv_src_memory.get_desc(), src_attr, false);
        reorder src_reorder = new reorder(src_reorder_pd);
        src_reorder.execute(s, user_src_memory, conv_src_memory);

        memory conv_weights_memory = new memory(conv_prim_desc.weights_desc(), cpu_engine);
        primitive_attr weight_attr = new primitive_attr();
        weight_attr.set_output_scales(weight_mask, weight_scales);
        reorder.primitive_desc weight_reorder_pd = new reorder.primitive_desc(cpu_engine,
                user_weights_memory.get_desc(), cpu_engine,
                conv_weights_memory.get_desc(), weight_attr, false);
        reorder weight_reorder = new reorder(weight_reorder_pd);
        weight_reorder.execute(s, user_weights_memory, conv_weights_memory);

        memory conv_bias_memory = new memory(conv_prim_desc.bias_desc(), cpu_engine);
        primitive_attr bias_attr = new primitive_attr();
        bias_attr.set_output_scales(bias_mask, bias_scales);
        reorder.primitive_desc bias_reorder_pd = new reorder.primitive_desc(cpu_engine,
                user_bias_memory.get_desc(), cpu_engine,
                conv_bias_memory.get_desc(), bias_attr, false);
        reorder bias_reorder = new reorder(bias_reorder_pd);
        bias_reorder.execute(s, user_bias_memory, conv_bias_memory);
//[Quantize data and weights]

        memory conv_dst_memory = new memory(conv_prim_desc.dst_desc(), cpu_engine);

/// Create the convolution primitive and add it to the net. The int8 example
/// computes the same Convolution +ReLU layers from AlexNet simple-net.cpp
/// using the int8 and PostOps approach. Although performance is not
/// measured here, in practice it would require less computation time to achieve
/// similar results.
/// @snippet cpu_cnn_inference_int8.cpp Create convolution primitive
//[Create convolution primitive]
        convolution_forward conv = new convolution_forward(conv_prim_desc);
        conv.execute(s, new IntMemoryMap()
                .put(DNNL_ARG_SRC, conv_src_memory)
                .put(DNNL_ARG_WEIGHTS, conv_weights_memory)
                .put(DNNL_ARG_BIAS, conv_bias_memory)
                .put(DNNL_ARG_DST, conv_dst_memory));
//[Create convolution primitive]

/// @page cpu_cnn_inference_int8_cpp
/// Finally, *dst memory* may be dequantized from int8 into the original
/// f32 format. Create a memory primitive for the user data in the original
/// 32-bit floating point format and then apply a reorder to transform the
/// computation output data.
/// @snippet cpu_cnn_inference_int8.cpp Dequantize the result
//[Dequantize the result]
        memory user_dst_memory = new memory(
                new memory.desc(conv_dst_tz, memory.data_type.f32, memory.format_tag.nchw),
                cpu_engine, new FloatPointer(user_dst));
        primitive_attr dst_attr = new primitive_attr();
        dst_attr.set_output_scales(dst_mask, dst_scales);
        reorder.primitive_desc dst_reorder_pd = new reorder.primitive_desc(cpu_engine,
                conv_dst_memory.get_desc(), cpu_engine,
                user_dst_memory.get_desc(), dst_attr, false);
        reorder dst_reorder = new reorder(dst_reorder_pd);
        dst_reorder.execute(s, conv_dst_memory, user_dst_memory);
//[Dequantize the result]

        s._wait();
    }

    public static void main(String[] args) throws Exception {
        try (PointerScope scope = new PointerScope()) {
            simple_net_int8();
            System.out.println("Simple-net-int8 example passed!");
        } catch (Exception e) {
            System.err.println("exception: " + e);
        }
        System.exit(0);
    }
}
