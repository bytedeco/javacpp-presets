JavaCPP Presets for DNNL 
========================

[![Gitter](https://badges.gitter.im/bytedeco/javacpp.svg)](https://gitter.im/bytedeco/javacpp) [![Maven Central](https://maven-badges.herokuapp.com/maven-central/org.bytedeco/dnnl/badge.svg)](https://maven-badges.herokuapp.com/maven-central/org.bytedeco/dnnl) [![Sonatype Nexus (Snapshots)](https://img.shields.io/nexus/s/https/oss.sonatype.org/org.bytedeco/dnnl.svg)](http://bytedeco.org/builds/)  
<sup>Build status for all platforms:</sup> [![dnnl](https://github.com/bytedeco/javacpp-presets/workflows/dnnl/badge.svg)](https://github.com/bytedeco/javacpp-presets/actions?query=workflow%3Adnnl)  <sup>Commercial support:</sup> [![xscode](https://img.shields.io/badge/Available%20on-xs%3Acode-blue?style=?style=plastic&logo=appveyor&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEAAAABACAMAAACdt4HsAAAAGXRFWHRTb2Z0d2FyZQBBZG9iZSBJbWFnZVJlYWR5ccllPAAAAAZQTFRF////////VXz1bAAAAAJ0Uk5T/wDltzBKAAAAlUlEQVR42uzXSwqAMAwE0Mn9L+3Ggtgkk35QwcnSJo9S+yGwM9DCooCbgn4YrJ4CIPUcQF7/XSBbx2TEz4sAZ2q1RAECBAiYBlCtvwN+KiYAlG7UDGj59MViT9hOwEqAhYCtAsUZvL6I6W8c2wcbd+LIWSCHSTeSAAECngN4xxIDSK9f4B9t377Wd7H5Nt7/Xz8eAgwAvesLRjYYPuUAAAAASUVORK5CYII=)](https://xscode.com/bytedeco/javacpp-presets)


Introduction
------------
This directory contains the JavaCPP Presets module for:

 * DNNL 3.7.1  https://01.org/dnnl

Please refer to the parent README.md file for more detailed information about the JavaCPP Presets.


Documentation
-------------
Java API documentation is available here:

 * http://bytedeco.org/javacpp-presets/dnnl/apidocs/


Sample Usage
------------
Here is a simple example of DNNL ported to Java from this C++ source file:

 * https://github.com/oneapi-src/oneDNN/blob/v3.7.1/examples/cnn_inference_int8.cpp

We can use [Maven 3](http://maven.apache.org/) to download and install automatically all the class files as well as the native binaries. To run this sample code, after creating the `pom.xml` and `CpuCnnInferenceInt8.java` source files below, simply execute on the command line:
```bash
 $ mvn compile exec:java
```

### The `pom.xml` build file
```xml
<project>
    <modelVersion>4.0.0</modelVersion>
    <groupId>org.bytedeco.dnnl</groupId>
    <artifactId>samples</artifactId>
    <version>1.5.12-SNAPSHOT</version>
    <properties>
        <exec.mainClass>CpuCnnInferenceInt8</exec.mainClass>
    </properties>
    <dependencies>
        <dependency>
            <groupId>org.bytedeco</groupId>
            <artifactId>dnnl-platform</artifactId>
            <version>3.7.1-1.5.12-SNAPSHOT</version>
        </dependency>
    </dependencies>
    <build>
        <sourceDirectory>.</sourceDirectory>
    </build>
</project>
```

### The `CpuCnnInferenceInt8.java` source file
```java
/*******************************************************************************
* Copyright 2018-2022 Intel Corporation
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

/// @example cnn_inference_int8.cpp
/// @copybrief cnn_inference_int8_cpp
/// > Annotated version: @ref cnn_inference_int8_cpp

/// @page cnn_inference_int8_cpp CNN int8 inference example
/// This C++ API example demonstrates how to run AlexNet's conv3 and relu3
/// with int8 data type.
///
/// > Example code: @ref cnn_inference_int8.cpp

import org.bytedeco.javacpp.*;

import org.bytedeco.dnnl.*;
import static org.bytedeco.dnnl.global.dnnl.*;

public class CpuCnnInferenceInt8 {

    static long product(long[] dims) {
        long accumulate = 1;
        for (int i = 0; i < dims.length; i++) accumulate *= dims[i];
        return accumulate;
    }

    // Read from handle, write to memory
    static void write_to_dnnl_memory(Pointer handle, memory mem) throws Exception {
        engine eng = mem.get_engine();
        long size = mem.get_desc().get_size();

        if (handle == null || handle.isNull()) throw new Exception("handle is nullptr.");

        if (eng.get_kind() == engine.kind.cpu) {
            BytePointer dst = new BytePointer(mem.get_data_handle());
            if (dst.isNull()) throw new Exception("get_data_handle returned nullptr.");
            dst.capacity(size).put(handle);
            return;
        }

        assert("not expected" == null);
    }

    static void simple_net_int8(engine.kind engine_kind) throws Exception {
        engine eng = new engine(engine_kind, 0);
        stream s = new stream(eng);

        final int batch = 8;

        /// Configure tensor shapes
        /// @snippet cnn_inference_int8.cpp Configure tensor shapes
        //[Configure tensor shapes]
        // AlexNet: conv3
        // {batch, 256, 13, 13} (x)  {384, 256, 3, 3}; -> {batch, 384, 13, 13}
        // strides: {1, 1}
        long[] conv_src_tz = {batch, 256, 13, 13};
        long[] conv_weights_tz = {384, 256, 3, 3};
        long[] conv_bias_tz = {384};
        long[] conv_dst_tz = {batch, 384, 13, 13};
        long[] conv_strides = {1, 1};
        long[] conv_padding = {1, 1};
        //[Configure tensor shapes]

        /// Next, the example configures the scales used to quantize f32 data
        /// into int8. For this example, the scaling value is chosen as an
        /// arbitrary number, although in a realistic scenario, it should be
        /// calculated from a set of precomputed values as previously mentioned.
        /// @snippet cnn_inference_int8.cpp Choose scaling factors
        //[Choose scaling factors]
        // Choose scaling factors for input, weight and output
        float[] src_scales = {1.8f};
        float[] weight_scales = {2.0f};
        float[] dst_scales = {0.55f};

        //[Choose scaling factors]

        /// The *source, weights, bias* and *destination* datasets use the single-scale
        /// format with mask set to '0'.
        /// @snippet cnn_inference_int8.cpp Set scaling mask
        //[Set scaling mask]
        final int src_mask = 0;
        final int weight_mask = 0;
        final int dst_mask = 0;
        //[Set scaling mask]

        // Allocate input and output buffers for user data
        float[] user_src = new float[batch * 256 * 13 * 13];
        float[] user_dst = new float[batch * 384 * 13 * 13];

        // Allocate and fill buffers for weights and bias
        float[] conv_weights = new float[(int)product(conv_weights_tz)];
        float[] conv_bias = new float[(int)product(conv_bias_tz)];

        /// Create the memory primitives for user data (source, weights, and bias).
        /// The user data will be in its original 32-bit floating point format.
        /// @snippet cnn_inference_int8.cpp Allocate buffers
        //[Allocate buffers]
        memory user_src_memory = new memory(new memory.desc(conv_src_tz, memory.data_type.f32, memory.format_tag.nchw), eng);
        write_to_dnnl_memory(new FloatPointer(user_src), user_src_memory);
        memory user_weights_memory
                = new memory(new memory.desc(conv_weights_tz, memory.data_type.f32, memory.format_tag.oihw), eng);
        write_to_dnnl_memory(new FloatPointer(conv_weights), user_weights_memory);
        memory user_bias_memory = new memory(new memory.desc(conv_bias_tz, memory.data_type.f32, memory.format_tag.x), eng);
        write_to_dnnl_memory(new FloatPointer(conv_bias), user_bias_memory);
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
        /// @snippet cnn_inference_int8.cpp Create convolution memory descriptors
        //[Create convolution memory descriptors]
        memory.desc conv_src_md = new memory.desc(conv_src_tz, memory.data_type.u8, memory.format_tag.any);
        memory.desc conv_bias_md = new memory.desc(conv_bias_tz, memory.data_type.s8, memory.format_tag.any);
        memory.desc conv_weights_md = new memory.desc(conv_weights_tz, memory.data_type.s8, memory.format_tag.any);
        memory.desc conv_dst_md = new memory.desc(conv_dst_tz, memory.data_type.u8, memory.format_tag.any);
        //[Create convolution memory descriptors]

        /// Configuring int8-specific parameters in an int8 primitive is done
        /// via the Attributes Primitive. Create an attributes object for the
        /// convolution and configure it accordingly.
        /// @snippet cnn_inference_int8.cpp Configure scaling
        //[Configure scaling]
        primitive_attr conv_attr = new primitive_attr();
        conv_attr.set_scales_mask(DNNL_ARG_SRC, src_mask);
        conv_attr.set_scales_mask(DNNL_ARG_WEIGHTS, weight_mask);
        conv_attr.set_scales_mask(DNNL_ARG_DST, dst_mask);

        // Prepare dst scales
        memory.desc dst_scale_md = new memory.desc(new long[]{1}, memory.data_type.f32, memory.format_tag.x);
        memory dst_scale_memory = new memory(dst_scale_md, eng);
        write_to_dnnl_memory(new FloatPointer(dst_scales), dst_scale_memory);
        //[Configure scaling]

        /// The ReLU layer from Alexnet is executed through the PostOps feature. Create
        /// a PostOps object and configure it to execute an _eltwise relu_ operation.
        /// @snippet cnn_inference_int8.cpp Configure post-ops
        //[Configure post-ops]
        final float ops_alpha = 0.f; // relu negative slope
        final float ops_beta = 0.f;
        post_ops ops = new post_ops();
        ops.append_eltwise(algorithm.eltwise_relu, ops_alpha, ops_beta);
        conv_attr.set_post_ops(ops);
        //[Configure post-ops]

        // check if int8 convolution is supported
        try {
            new convolution_forward.primitive_desc(eng, prop_kind.forward,
                    algorithm.convolution_direct, conv_src_md, conv_weights_md,
                    conv_bias_md, conv_dst_md, conv_strides, conv_padding,
                    conv_padding, conv_attr, false);
        } catch (Exception e) {
            if (e.getMessage().contains("status = " + dnnl_unimplemented)) {
                System.err.println(
                        "No int8 convolution implementation is available for this platform.\n"
                      + "Please refer to the developer guide for details.");
            }
            // on any other error just re-throw
            throw e;
        }

        /// Create a primitive descriptor passing the int8 memory descriptors
        /// and int8 attributes to the constructor. The primitive
        /// descriptor for the convolution will contain the specific memory
        /// formats for the computation.
        /// @snippet cnn_inference_int8.cpp Create convolution primitive descriptor
        //[Create convolution primitive descriptor]
        convolution_forward.primitive_desc conv_prim_desc = new convolution_forward.primitive_desc(eng,
                prop_kind.forward, algorithm.convolution_direct, conv_src_md,
                conv_weights_md, conv_bias_md, conv_dst_md, conv_strides,
                conv_padding, conv_padding, conv_attr, false);
        //[Create convolution primitive descriptor]

        /// Create a memory for each of the convolution's data input
        /// parameters (source, bias, weights, and destination). Using the convolution
        /// primitive descriptor as the creation parameter enables oneDNN
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
        /// @snippet cnn_inference_int8.cpp Quantize data and weights
        //[Quantize data and weights]
        memory conv_src_memory = new memory(conv_prim_desc.src_desc(), eng);
        primitive_attr src_attr = new primitive_attr();
        src_attr.set_scales_mask(DNNL_ARG_DST, src_mask);
        memory.desc src_scale_md = new memory.desc(new long[]{1}, memory.data_type.f32, memory.format_tag.x);
        memory src_scale_memory = new memory(src_scale_md, eng);
        write_to_dnnl_memory(new FloatPointer(src_scales), src_scale_memory);
        reorder.primitive_desc src_reorder_pd
                = new reorder.primitive_desc(eng, user_src_memory.get_desc(), eng,
                        conv_src_memory.get_desc(), src_attr, false);
        reorder src_reorder = new reorder(src_reorder_pd);
        src_reorder.execute(s, new IntMemoryMap()
                .put(DNNL_ARG_FROM, user_src_memory).put(DNNL_ARG_TO, conv_src_memory)
                        .put(DNNL_ARG_ATTR_SCALES | DNNL_ARG_DST, src_scale_memory));

        memory conv_weights_memory = new memory(conv_prim_desc.weights_desc(), eng);
        primitive_attr weight_attr = new primitive_attr();
        weight_attr.set_scales_mask(DNNL_ARG_DST, weight_mask);
        memory.desc wei_scale_md = new memory.desc(new long[]{1}, memory.data_type.f32, memory.format_tag.x);
        memory wei_scale_memory = new memory(wei_scale_md, eng);
        write_to_dnnl_memory(new FloatPointer(weight_scales), wei_scale_memory);
        reorder.primitive_desc weight_reorder_pd
                = new reorder.primitive_desc(eng, user_weights_memory.get_desc(), eng,
                        conv_weights_memory.get_desc(), weight_attr, false);
        reorder weight_reorder = new reorder(weight_reorder_pd);
        weight_reorder.execute(s, new IntMemoryMap()
                .put(DNNL_ARG_FROM, user_weights_memory)
                        .put(DNNL_ARG_TO, conv_weights_memory)
                        .put(DNNL_ARG_ATTR_SCALES | DNNL_ARG_DST, wei_scale_memory));

        memory conv_bias_memory = new memory(conv_prim_desc.bias_desc(), eng);
        write_to_dnnl_memory(new FloatPointer(conv_bias), conv_bias_memory);
        //[Quantize data and weights]

        memory conv_dst_memory = new memory(conv_prim_desc.dst_desc(), eng);

        /// Create the convolution primitive and add it to the net. The int8 example
        /// computes the same Convolution +ReLU layers from AlexNet simple-net.cpp
        /// using the int8 and PostOps approach. Although performance is not
        /// measured here, in practice it would require less computation time to achieve
        /// similar results.
        /// @snippet cnn_inference_int8.cpp Create convolution primitive
        //[Create convolution primitive]
        convolution_forward conv = new convolution_forward(conv_prim_desc);
        conv.execute(s, new IntMemoryMap()
                .put(DNNL_ARG_SRC, conv_src_memory)
                        .put(DNNL_ARG_WEIGHTS, conv_weights_memory)
                        .put(DNNL_ARG_BIAS, conv_bias_memory)
                        .put(DNNL_ARG_DST, conv_dst_memory)
                        .put(DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC, src_scale_memory)
                        .put(DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS, wei_scale_memory)
                        .put(DNNL_ARG_ATTR_SCALES | DNNL_ARG_DST, dst_scale_memory));
        //[Create convolution primitive]

        /// @page cnn_inference_int8_cpp
        /// Finally, *dst memory* may be dequantized from int8 into the original
        /// f32 format. Create a memory primitive for the user data in the original
        /// 32-bit floating point format and then apply a reorder to transform the
        /// computation output data.
        /// @snippet cnn_inference_int8.cpp Dequantize the result
        ///[Dequantize the result]
        memory user_dst_memory = new memory(new memory.desc(conv_dst_tz, memory.data_type.f32, memory.format_tag.nchw), eng);
        write_to_dnnl_memory(new FloatPointer(user_dst), user_dst_memory);
        primitive_attr dst_attr = new primitive_attr();
        dst_attr.set_scales_mask(DNNL_ARG_SRC, dst_mask);
        reorder.primitive_desc dst_reorder_pd
                = new reorder.primitive_desc(eng, conv_dst_memory.get_desc(), eng,
                        user_dst_memory.get_desc(), dst_attr, false);
        reorder dst_reorder = new reorder(dst_reorder_pd);
        dst_reorder.execute(s, new IntMemoryMap()
                .put(DNNL_ARG_FROM, conv_dst_memory).put(DNNL_ARG_TO, user_dst_memory)
                        .put(DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC, dst_scale_memory));
        //[Dequantize the result]

        s._wait();
    }

    public static void main(String[] args) throws Exception {
        try (PointerScope scope = new PointerScope()) {
            simple_net_int8(engine.kind.cpu);
            System.out.println("Simple-net-int8 example passed!");
        } catch (Exception e) {
            System.err.println("exception: " + e);
        }
        System.exit(0);
    }
}
```
