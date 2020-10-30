JavaCPP Presets for MKL-DNN
===========================

Introduction
------------
This directory contains the JavaCPP Presets module for:

 * MKL-DNN 0.21.5  https://01.org/mkl-dnn

Please refer to the parent README.md file for more detailed information about the JavaCPP Presets.


Documentation
-------------
Java API documentation is available here:

 * http://bytedeco.org/javacpp-presets/mkl-dnn/apidocs/

&lowast; Although MKL-DNN comes bundled with a stripped-down version of MKL known as "MKLML", it is sometimes desirable to link instead with the full version of [Intel MKL](https://software.intel.com/intel-mkl). For that, MKL first needs to be installed in its default location, or available in the system PATH or in the "java.library.path", then we can set the "org.bytedeco.mklml.load" system property to `mkl_rt`. We should also set the "org.bytedeco.javacpp.pathsfirst" system property to `true` to ensure that all libraries are actually loaded from the system, unless the `-redist` artifacts listed below are in the class path.


Sample Usage
------------
Here is a simple example of MKL-DNN ported to Java from this C++ source file:

* https://github.com/intel/mkl-dnn/blob/master/examples/simple_net_int8.cpp

We can use [Maven 3](http://maven.apache.org/) to download and install automatically all the class files as well as the native binaries. To run this sample code, after creating the `pom.xml` and `SimpleNetInt8.java` source files below, simply execute on the command line:
```bash
 $ mvn compile exec:java
```

### The `pom.xml` build file
```xml
<project>
    <modelVersion>4.0.0</modelVersion>
    <groupId>org.bytedeco.mkl-dnn</groupId>
    <artifactId>mkl-dnn</artifactId>
    <version>1.5.4</version>
    <properties>
        <exec.mainClass>SimpleNetInt8</exec.mainClass>
    </properties>
    <dependencies>
        <dependency>
            <groupId>org.bytedeco</groupId>
            <artifactId>mkl-dnn-platform</artifactId>
            <version>0.21.5-1.5.4</version>
        </dependency>

        <!-- Additional dependencies to use bundled full version of MKL -->
        <dependency>
            <groupId>org.bytedeco</groupId>
            <artifactId>mkl-platform-redist</artifactId>
            <version>2020.4-1.5.5-SNAPSHOT</version>
        </dependency>

    </dependencies>
    <build>
        <sourceDirectory>.</sourceDirectory>
    </build>
</project>
```

### The `SimpleNetInt8.java` source file
```java
/*******************************************************************************
* Copyright 2018 Intel Corporation
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

import org.bytedeco.javacpp.*;

import org.bytedeco.mkldnn.*;
import static org.bytedeco.mkldnn.global.mkldnn.*;

public class SimpleNetInt8 {

    static void simple_net_int8() throws Exception {
        engine cpu_engine = new engine(engine.cpu, 0);

        /* Create a vector to store the topology primitives */
        primitive_vector net = new primitive_vector();

        int batch = 8;

        /* AlexNet: conv3
         * {batch, 256, 13, 13} (x)  {384, 256, 3, 3}; -> {batch, 384, 13, 13}
         * strides: {1, 1}
         */
        int[] conv_src_tz = { batch, 256, 13, 13 };
        int[] conv_weights_tz = { 384, 256, 3, 3 };
        int[] conv_bias_tz = { 384 };
        int[] conv_dst_tz = { batch, 384, 13, 13 };
        int[] conv_strides = { 1, 1 };
        int[] conv_padding = { 1, 1 };

        /* Set Scaling mode for int8 quantizing */
        float[] src_scales = { 1.8f };
        float[] weight_scales = { 2.0f };
        float[] bias_scales = { 1.0f };
        float[] dst_scales = { 0.55f };
        /* assign halves of vector with arbitrary values */
        float[] conv_scales = new float[384];
        int scales_half = 384 / 2;
        for (int i = 0;               i < scales_half;        i++) conv_scales[i] = 0.3f;
        for (int i = scales_half + 1; i < conv_scales.length; i++) conv_scales[i] = 0.8f;

        int src_mask = 0;
        int weight_mask = 0;
        int bias_mask = 0;
        int dst_mask = 0;
        int conv_mask = 2; // 1 << output_channel_dim

        /* Allocate input and output buffers for user data */
        float[] user_src = new float[batch * 256 * 13 * 13];
        float[] user_dst = new float[batch * 384 * 13 * 13];

        /* Allocate and fill buffers for weights and bias */
        float[] conv_weights = new float[384 * 256 * 3 * 3];
        float[] conv_bias    = new float[384];

        /* create memory for user data */
        memory user_src_memory = new memory(new memory.primitive_desc(
                new memory.desc(conv_src_tz, memory.f32, memory.nchw ),
                        cpu_engine), new FloatPointer(user_src));
        memory user_weights_memory = new memory(new memory.primitive_desc(
                new memory.desc(conv_weights_tz, memory.f32, memory.oihw),
                        cpu_engine), new FloatPointer(conv_weights));
        memory user_bias_memory = new memory(new memory.primitive_desc(
                new memory.desc(conv_bias_tz, memory.f32, memory.x),
                        cpu_engine), new FloatPointer(conv_bias));

        /* create memory descriptors for convolution data w/ no specified format */
        memory.desc conv_src_md = new memory.desc(
                conv_src_tz, memory.u8, memory.any);
        memory.desc conv_bias_md = new memory.desc(
                conv_bias_tz, memory.s8, memory.any);
        memory.desc conv_weights_md = new memory.desc(
                conv_weights_tz, memory.s8, memory.any);
        memory.desc conv_dst_md = new memory.desc(
                conv_dst_tz, memory.u8, memory.any);

        /* create a convolution */
        convolution_forward.desc conv_desc = new convolution_forward.desc(forward,
                convolution_direct, conv_src_md, conv_weights_md, conv_bias_md,
                conv_dst_md, conv_strides, conv_padding, conv_padding, zero);

        /* define the convolution attributes */
        primitive_attr conv_attr = new primitive_attr();
        conv_attr.set_int_output_round_mode(round_nearest);
        conv_attr.set_output_scales(conv_mask, conv_scales);

        /* AlexNet: execute ReLU as PostOps */
        float ops_scale = 1.f;
        float ops_alpha = 0.f; // relu negative slope
        float ops_beta = 0.f;
        post_ops ops = new post_ops();
        ops.append_eltwise(ops_scale, eltwise_relu, ops_alpha, ops_beta);
        conv_attr.set_post_ops(ops);

        /* check if int8 convolution is supported */
        try {
            convolution_forward.primitive_desc conv_prim_desc = new convolution_forward.primitive_desc(
                    conv_desc, conv_attr, cpu_engine);
        } catch (Exception e) {
            if (e.getMessage().contains("status = " + mkldnn_unimplemented)) {
                System.err.println("AVX512-BW support or Intel(R) MKL dependency is "
                                 + "required for int8 convolution");
            }
            throw e;
        }

        convolution_forward.primitive_desc conv_prim_desc = new convolution_forward.primitive_desc(
                conv_desc, conv_attr, cpu_engine);

        /* Next: create memory primitives for the convolution's input data
         * and use reorder to quantize the values into int8 */
        memory conv_src_memory = new memory(conv_prim_desc.src_primitive_desc());
        primitive_attr src_attr = new primitive_attr();
        src_attr.set_int_output_round_mode(round_nearest);
        src_attr.set_output_scales(src_mask, src_scales);
        reorder.primitive_desc src_reorder_pd
                = new reorder.primitive_desc(user_src_memory.get_primitive_desc(),
                        conv_src_memory.get_primitive_desc(), src_attr);
        net.push_back(new reorder(src_reorder_pd, new primitive.at(user_src_memory),
                conv_src_memory));

        memory conv_weights_memory = new memory(conv_prim_desc.weights_primitive_desc());
        primitive_attr weight_attr = new primitive_attr();
        weight_attr.set_int_output_round_mode(round_nearest);
        weight_attr.set_output_scales(weight_mask, weight_scales);
        reorder.primitive_desc weight_reorder_pd
                = new reorder.primitive_desc(user_weights_memory.get_primitive_desc(),
                        conv_weights_memory.get_primitive_desc(), weight_attr);
        net.push_back(new reorder(weight_reorder_pd, new primitive.at(user_weights_memory),
                conv_weights_memory));

        memory conv_bias_memory = new memory(conv_prim_desc.bias_primitive_desc());
        primitive_attr bias_attr = new primitive_attr();
        bias_attr.set_int_output_round_mode(round_nearest);
        bias_attr.set_output_scales(bias_mask, bias_scales);
        reorder.primitive_desc bias_reorder_pd
                = new reorder.primitive_desc(user_bias_memory.get_primitive_desc(),
                        conv_bias_memory.get_primitive_desc(), bias_attr);
        net.push_back(new reorder(bias_reorder_pd, new primitive.at(user_bias_memory),
                conv_bias_memory));

        memory conv_dst_memory = new memory(conv_prim_desc.dst_primitive_desc());

        /* create convolution primitive and add it to net */
        net.push_back(new convolution_forward(conv_prim_desc, new primitive.at(conv_src_memory),
                new primitive.at(conv_weights_memory), new primitive.at(conv_bias_memory),
                conv_dst_memory));

        /* Convert data back into fp32 and compare values with u8.
         * Note: data is unsigned since there are no negative values
         * after ReLU */

        /* Create a memory primitive for user data output */
        memory user_dst_memory = new memory(new memory.primitive_desc(
                new memory.desc(conv_dst_tz, memory.f32, memory.nchw),
                        cpu_engine), new FloatPointer(user_dst));

        primitive_attr dst_attr = new primitive_attr();
        dst_attr.set_int_output_round_mode(round_nearest);
        dst_attr.set_output_scales(dst_mask, dst_scales);
        reorder.primitive_desc dst_reorder_pd
                = new reorder.primitive_desc(conv_dst_memory.get_primitive_desc(),
                        user_dst_memory.get_primitive_desc(), dst_attr);

        /* Convert the destination memory from convolution into user
         * data format if necessary */
        if (conv_dst_memory.notEquals(user_dst_memory)) {
            net.push_back(new reorder(dst_reorder_pd, new primitive.at(conv_dst_memory),
                    user_dst_memory));
        }

        new stream(stream.eager).submit(net)._wait();
    }

    public static void main(String[] args) throws Exception {
        try {
            /* Notes:
             * On convolution creating: check for Intel(R) MKL dependency execution.
             * output: warning if not found. */
            simple_net_int8();
            System.out.println("Simple-net-int8 example passed!");
        } catch (Exception e) {
            System.err.println("exception: " + e);
        }
        System.exit(0);
    }
}
```
