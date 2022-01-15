/*
 * Copyright (C) 2015-2022 Samuel Audet
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

package org.bytedeco.tensorflow.presets;

import org.bytedeco.javacpp.BytePointer;
import org.bytedeco.javacpp.ClassProperties;
import org.bytedeco.javacpp.FunctionPointer;
import org.bytedeco.javacpp.LoadEnabled;
import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.PointerPointer;
import org.bytedeco.javacpp.annotation.Adapter;
import org.bytedeco.javacpp.annotation.ByRef;
import org.bytedeco.javacpp.annotation.ByVal;
import org.bytedeco.javacpp.annotation.Cast;
import org.bytedeco.javacpp.annotation.Platform;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.javacpp.annotation.StdString;
import org.bytedeco.javacpp.presets.javacpp;
import org.bytedeco.javacpp.tools.BuildEnabled;
import org.bytedeco.javacpp.tools.Info;
import org.bytedeco.javacpp.tools.InfoMap;
import org.bytedeco.javacpp.tools.InfoMapper;
import org.bytedeco.javacpp.tools.Logger;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.lang.annotation.Documented;
import java.lang.annotation.ElementType;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;
import java.util.List;

/**
 *
 * @author Samuel Audet
 */
@Properties(inherit = javacpp.class, value = {
        @Platform(
                value = {"linux", "macosx", "windows"},
                compiler = "cpp11",
                define = {"NDEBUG", "UNIQUE_PTR_NAMESPACE std", "SHARED_PTR_NAMESPACE std"},
                exclude = "google/protobuf/port_def.inc",
                include = {
                        "google/protobuf/port_def.inc",
                        "google/protobuf/arena.h",
                        "google/protobuf/message_lite.h",
                        "google/protobuf/unknown_field_set.h",
                        "tensorflow/core/platform/default/integral_types.h",
                        "tensorflow/core/lib/bfloat16/bfloat16.h",
                        "tensorflow/core/framework/numeric_types.h",
                        "tensorflow/core/platform/init_main.h",
                        "tensorflow/core/platform/types.h",
                        "tensorflow/core/platform/mutex.h",
                        "tensorflow/core/platform/macros.h",
                        "tensorflow/core/util/port.h",
                        "tensorflow/core/lib/core/error_codes.pb.h",
                        "tensorflow/core/lib/core/errors.h",
                        "tensorflow/core/platform/logging.h",
                        "tensorflow/core/lib/core/status.h",
                        "tensorflow/core/util/device_name_utils.h",
                        "tensorflow/core/lib/io/zlib_compression_options.h",
                        "tensorflow/core/lib/io/zlib_outputbuffer.h",
                        "tensorflow/core/lib/io/inputstream_interface.h",
                        "tensorflow/core/lib/io/record_reader.h",
                        "tensorflow/core/lib/io/record_writer.h",
                        "tensorflow/core/platform/protobuf.h",
                        "tensorflow/core/platform/file_system.h",
                        "tensorflow/core/platform/file_statistics.h",
                        "tensorflow/core/platform/env.h",
//                        "tensorflow/core/graph/dot.h",
                        "tensorflow/core/example/feature.pb.h",
                        "tensorflow/core/example/example.pb.h",
                        "tensorflow/core/protobuf/debug.pb.h",
                        "tensorflow/core/protobuf/cluster.pb.h",
                        "tensorflow/core/protobuf/verifier_config.pb.h",
                        "tensorflow/core/protobuf/rewriter_config.pb.h",
                        "tensorflow/core/protobuf/config.pb.h",
                        "tensorflow/core/framework/cost_graph.pb.h",
                        "tensorflow/core/framework/step_stats.pb.h",
                        "tensorflow/core/framework/versions.pb.h",
                        "tensorflow/core/public/session_options.h",
                        "tensorflow/core/lib/core/threadpool.h",
                        "tensorflow/core/framework/allocation_description.pb.h",
//                        "tensorflow/core/platform/default/string_coding.h",
                        "tensorflow/core/platform/tensor_coding.h",
                        "tensorflow/core/framework/resource_handle.h",
                        "tensorflow/core/framework/allocator.h",
                        "tensorflow/core/framework/tensor_shape.pb.h",
                        "tensorflow/core/framework/types.pb.h",
                        "tensorflow/core/framework/resource_handle.pb.h",
                        "tensorflow/core/framework/tensor.pb.h",
                        "tensorflow/core/framework/tensor_description.pb.h",
                        "tensorflow/core/framework/tensor_types.h",
                        "tensorflow/core/framework/tensor_shape.h",
                        //        "tensorflow/core/framework/tensor_slice.h",
                        "tensorflow/core/framework/tensor_util.h",
                        "tensorflow/core/framework/tensor_reference.h",
                        "tensorflow/core/framework/tensor.h",
                        "tensorflow/core/framework/attr_value.pb.h",
                        "tensorflow/core/framework/node_def.pb.h",
                        "tensorflow/core/framework/api_def.pb.h",
                        "tensorflow/core/framework/op_def.pb.h",
                        "tensorflow/core/framework/function.pb.h",
                        "tensorflow/core/framework/graph.pb.h",
                        "tensorflow/core/framework/session_state.h",
                        "tensorflow/core/framework/types.h",
                        "tensorflow/core/framework/control_flow.h",
                        "tensorflow/core/framework/kernel_def.pb.h",
                        "tensorflow/core/framework/kernel_def_builder.h",
                        "tensorflow/core/framework/tracking_allocator.h",
                        "tensorflow/core/framework/op_kernel.h",
                        "tensorflow/core/framework/op_segment.h",
                        "tensorflow/core/framework/shape_inference.h",
                        "tensorflow/core/framework/partial_tensor_shape.h",
                        "tensorflow/core/framework/device_attributes.pb.h",
                        "tensorflow/core/public/session.h",
                        "tensorflow/core/framework/tensor_slice.pb.h",
                        "tensorflow/core/framework/tensor_slice.h",
                        "tensorflow/core/util/tensor_slice_set.h",
                        "tensorflow/core/util/tensor_slice_util.h",
                        "tensorflow/core/util/tensor_slice_reader.h",
                        "tensorflow/core/util/tensor_bundle/tensor_bundle.h",
                        "tensorflow/core/framework/summary.pb.h",
                        "tensorflow/core/lib/monitoring/counter.h",
                        "tensorflow/core/lib/monitoring/gauge.h",
                        "tensorflow/core/lib/monitoring/sampler.h",
                        "tensorflow/core/profiler/internal/profiler_interface.h",
                        "tensorflow/core/profiler/lib/profiler_session.h",
                        "tensorflow/c/tf_attrtype.h",
                        "tensorflow/c/tf_datatype.h",
                        "tensorflow/c/tf_status.h",
                        "tensorflow/c/tf_status_helper.h",
                        "tensorflow/c/tf_tensor.h",
                        "tensorflow/c/checkpoint_reader.h",
                        "tensorflow/c/c_api.h",
                        "tensorflow/c/c_api_internal.h",
                        "tensorflow/c/tf_status_internal.h",
                        "tensorflow/c/tf_tensor_internal.h",
                        "tensorflow/c/env.h",
                        "tensorflow/c/kernels.h",
                        "tensorflow/c/ops.h",
                        "tensorflow/core/framework/op_def_builder.h",
                        "tensorflow/core/framework/op_def_util.h",
                        "tensorflow/core/framework/op.h",
                        "tensorflow/core/framework/types.h",
                        "tensorflow/core/graph/edgeset.h",
                        "tensorflow/core/lib/gtl/iterator_range.h",
                        //        "tensorflow/core/lib/gtl/inlined_vector.h",
                        "tensorflow/core/framework/function.h",
                        "tensorflow/core/framework/device_attributes.pb.h",
                        "tensorflow/core/framework/device_base.h",
                        "tensorflow/core/common_runtime/device.h",
                        "tensorflow/core/common_runtime/device_mgr.h",
                        "tensorflow/core/common_runtime/process_function_library_runtime.h",
                        "tensorflow/core/graph/graph.h",
                        "tensorflow/core/graph/tensor_id.h",
                        "tensorflow/core/common_runtime/graph_runner.h",
                        "tensorflow/core/common_runtime/shape_refiner.h",
                        "tensorflow/core/framework/node_def_builder.h",
                        "tensorflow/core/framework/node_def_util.h",
                        "tensorflow/core/framework/selective_registration.h",
                        "tensorflow/core/graph/node_builder.h",
                        "tensorflow/core/graph/graph_def_builder.h",
                        "tensorflow/core/graph/default_device.h",
                        "tensorflow/core/graph/graph_constructor.h",
                        "tensorflow/core/graph/gradients.h",
                        "tensorflow/core/framework/variable.pb.h",
                        "tensorflow/core/protobuf/trackable_object_graph.pb.h",
                        "tensorflow/core/protobuf/struct.pb.h",
                        "tensorflow/core/protobuf/saved_object_graph.pb.h",
                        "tensorflow/core/protobuf/saver.pb.h",
                        "tensorflow/core/protobuf/meta_graph.pb.h",
                        "tensorflow_adapters.h",

                        "tensorflow/cc/framework/scope.h",
                        "tensorflow/cc/framework/ops.h",
                        "tensorflow/core/framework/op_gen_lib.h",
//                        "tensorflow/cc/framework/cc_op_gen.h",
                        "tensorflow/cc/framework/gradients.h",
                        "tensorflow/cc/saved_model/loader.h",
                        "tensorflow/cc/saved_model/tag_constants.h",
                        "tensorflow/cc/saved_model/signature_constants.h",
                        "tensorflow/core/framework/collective.h",
                        "tensorflow/core/platform/fingerprint.h",
                        "tensorflow/core/distributed_runtime/server_lib.h",
                        "tensorflow/core/distributed_runtime/eager/eager_client.h",
                        "tensorflow/core/common_runtime/eager/tensor_handle_data.h",
                        "tensorflow/core/distributed_runtime/eager/remote_tensor_handle_data.h",
                        "tensorflow/core/protobuf/eager_service.pb.h",
                        "tensorflow/core/protobuf/tensorflow_server.pb.h",
                        "tensorflow/core/protobuf/named_tensor.pb.h",
                        "tensorflow/core/protobuf/master.pb.h",
                        "tensorflow/core/protobuf/worker.pb.h",
                        "tensorflow/core/distributed_runtime/call_options.h",
                        "tensorflow/core/distributed_runtime/message_wrappers.h",
                        "tensorflow/core/distributed_runtime/worker_interface.h",
                        "tensorflow/core/distributed_runtime/worker_cache.h",
                        "tensorflow/core/distributed_runtime/worker_env.h",
                        "tensorflow/core/distributed_runtime/worker_session.h",
                        "tensorflow/core/common_runtime/eager/attr_builder.h",
                        "tensorflow/core/common_runtime/eager/context.h",
                        "tensorflow/core/common_runtime/eager/eager_executor.h",
                        "tensorflow/core/common_runtime/eager/eager_operation.h",
                        "tensorflow/core/common_runtime/eager/kernel_and_device.h",
                        "tensorflow/core/common_runtime/eager/tensor_handle.h",
                        "tensorflow/c/eager/c_api.h",
//                        "tensorflow/c/eager/c_api_experimental.h",
                        "tensorflow/c/eager/c_api_internal.h",
                        "tensorflow/c/python_api.h",
                        "tensorflow/cc/ops/standard_ops.h",
                        "tensorflow/cc/ops/const_op.h",
                        "tensorflow/cc/ops/array_ops.h",
                        "tensorflow/cc/ops/audio_ops.h",
                        "tensorflow/cc/ops/candidate_sampling_ops.h",
                        "tensorflow/cc/ops/control_flow_ops.h",
                        "tensorflow/cc/ops/data_flow_ops.h",
                        "tensorflow/cc/ops/image_ops.h",
                        "tensorflow/cc/ops/io_ops.h",
                        "tensorflow/cc/ops/linalg_ops.h",
                        "tensorflow/cc/ops/list_ops.h",
                        "tensorflow/cc/ops/logging_ops.h",
                        "tensorflow/cc/ops/lookup_ops.h",
                        "tensorflow/cc/ops/manip_ops.h",
                        "tensorflow/cc/ops/math_ops.h",
                        "tensorflow/cc/ops/nn_ops.h",
                        "tensorflow/cc/ops/nn_ops_internal.h",
                        "tensorflow/cc/ops/no_op.h",
                        "tensorflow/cc/ops/parsing_ops.h",
                        "tensorflow/cc/ops/random_ops.h",
                        "tensorflow/cc/ops/sparse_ops.h",
                        "tensorflow/cc/ops/state_ops.h",
                        "tensorflow/cc/ops/string_ops.h",
                        "tensorflow/cc/ops/training_ops.h",
                        "tensorflow/cc/ops/user_ops.h"},
                link = "tensorflow_cc@.1", preload = {"iomp5", "mklml", "mklml_intel", "tensorflow_framework"}, preloadresource = "/org/bytedeco/mkldnn/"),
        @Platform(
                value = {"linux-arm64", "linux-ppc64le", "linux-x86_64", "macosx-x86_64"},
                extension = {"-gpu", "-python", "-python-gpu"},
                link = "tensorflow_cc#",
                preload = {"iomp5", "mklml", "mklml_intel", "python3.10@.1.0!", "tensorflow_framework", "tensorflow_cc:python/tensorflow/python/_pywrap_tensorflow_internal.so", "tensorflow_cc:libtensorflow_cc.so.1"},
                resource = "python",
                preloadresource = {"/org/bytedeco/cpython/", "/org/bytedeco/mkldnn/"}),
        @Platform(
                value = "windows",
                link = {"Advapi32#", "mklml"},
// old hacks for the now obsolete CMake build
//                link = {"absl_base", "absl_throw_delegate", "absl_bad_optional_access", "absl_int128", "absl_str_format", "str_format_internal", "absl_strings",
//                        "Advapi32#", "double-conversion", "zlibstatic", "gpr", "grpc_unsecure", "grpc++_unsecure", "farmhash", "fft2d",
//                        "lmdb", "giflib", "libjpeg", "libpng16_static", "nsync", "nsync_cpp", "libprotobuf", "re2", "snappy", "sqlite", "mklml", "mkldnn",
//                        "tensorflow_static", "tf_protos_cc", "tf_cc_op_gen_main", "tf_python_protos_cc", "tf_c_python_api"},
                preload = {"msvcr120", "libiomp5md", "mklml", "python310"}),
        @Platform(
                value = "windows-x86_64",
                extension = {"-gpu", "-python", "-python-gpu"},
                link = {"Advapi32#", "mklml", "cudart", "cudart_static", "cuda", "cublasLt", "cublas", "cudnn", "cufft", "cufftw", "curand", "cusolver", "cusparse", "cupti"},
                resource = "python",
                preloadresource = {"/org/bytedeco/cpython/", "/org/bytedeco/mkldnn/"},
// old hacks for the now obsolete CMake build
//                link = {"absl_base", "absl_throw_delegate", "absl_bad_optional_access", "absl_int128", "absl_str_format", "str_format_internal", "absl_strings",
//                        "Advapi32#", "double-conversion", "zlibstatic", "gpr", "grpc_unsecure", "grpc++_unsecure", "farmhash", "fft2d",
//                        "lmdb", "giflib", "libjpeg", "libpng16_static", "nsync", "nsync_cpp", "libprotobuf", "re2", "snappy", "sqlite", "mklml", "mkldnn",
//                        "cudart", "cudart_static", "cuda", "cublasLt", "cublas", "cudnn", "cufft", "cufftw", "curand", "cusolver", "cusparse", "cupti",
//                        "tf_core_gpu_kernels", "tensorflow_static", "tf_protos_cc", "tf_cc_op_gen_main",  "tf_python_protos_cc", "tf_c_python_api"},
                includepath = {"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.6/include/"},
                linkpath    = {"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.6/lib/x64/",
                               "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.6/extras/CUPTI/lib64/"}),
        @Platform(
                value = {"android"},
                compiler = {"cpp11"},
                define = {"NDEBUG", "UNIQUE_PTR_NAMESPACE std", "SHARED_PTR_NAMESPACE std"},
                exclude = "google/protobuf/port_def.inc",
                include = {
                        "google/protobuf/port_def.inc",
                        "google/protobuf/arena.h",
                        "google/protobuf/message_lite.h",
                        "google/protobuf/unknown_field_set.h",
                        "tensorflow/core/platform/default/integral_types.h",
                        "tensorflow/core/lib/bfloat16/bfloat16.h",
                        "tensorflow/core/framework/numeric_types.h",
                        "tensorflow/core/platform/init_main.h",
                        "tensorflow/core/platform/types.h",
                        "tensorflow/core/platform/mutex.h",
                        "tensorflow/core/platform/macros.h",
                        "tensorflow/core/util/port.h",
                        "tensorflow/core/lib/core/error_codes.pb.h",
                        "tensorflow/core/lib/core/errors.h",
                        "tensorflow/core/platform/logging.h",
                        "tensorflow/core/lib/core/status.h",
                        "tensorflow/core/util/device_name_utils.h",
                        "tensorflow/core/lib/io/zlib_compression_options.h",
                        "tensorflow/core/lib/io/zlib_outputbuffer.h",
                        "tensorflow/core/lib/io/inputstream_interface.h",
                        "tensorflow/core/lib/io/record_reader.h",
                        "tensorflow/core/lib/io/record_writer.h",
                        "tensorflow/core/platform/protobuf.h",
                        "tensorflow/core/platform/file_system.h",
                        "tensorflow/core/platform/file_statistics.h",
                        "tensorflow/core/platform/env.h",
//                        "tensorflow/core/graph/dot.h",
                        "tensorflow/core/example/feature.pb.h",
                        "tensorflow/core/example/example.pb.h",
                        "tensorflow/core/protobuf/debug.pb.h",
                        "tensorflow/core/protobuf/cluster.pb.h",
                        "tensorflow/core/protobuf/verifier_config.pb.h",
                        "tensorflow/core/protobuf/rewriter_config.pb.h",
                        "tensorflow/core/protobuf/config.pb.h",
                        "tensorflow/core/framework/cost_graph.pb.h",
                        "tensorflow/core/framework/step_stats.pb.h",
                        "tensorflow/core/framework/versions.pb.h",
                        "tensorflow/core/public/session_options.h",
                        "tensorflow/core/lib/core/threadpool.h",
                        "tensorflow/core/framework/allocation_description.pb.h",
//                        "tensorflow/core/platform/default/string_coding.h",
                        "tensorflow/core/platform/tensor_coding.h",
                        "tensorflow/core/framework/resource_handle.h",
                        "tensorflow/core/framework/allocator.h",
                        "tensorflow/core/framework/tensor_shape.pb.h",
                        "tensorflow/core/framework/types.pb.h",
                        "tensorflow/core/framework/resource_handle.pb.h",
                        "tensorflow/core/framework/tensor.pb.h",
                        "tensorflow/core/framework/tensor_description.pb.h",
                        "tensorflow/core/framework/tensor_types.h",
                        "tensorflow/core/framework/tensor_shape.h",
                        //        "tensorflow/core/framework/tensor_slice.h",
                        "tensorflow/core/framework/tensor_util.h",
                        "tensorflow/core/framework/tensor_reference.h",
                        "tensorflow/core/framework/tensor.h",
                        "tensorflow/core/framework/attr_value.pb.h",
                        "tensorflow/core/framework/node_def.pb.h",
                        "tensorflow/core/framework/api_def.pb.h",
                        "tensorflow/core/framework/op_def.pb.h",
                        "tensorflow/core/framework/function.pb.h",
                        "tensorflow/core/framework/graph.pb.h",
                        "tensorflow/core/framework/session_state.h",
                        "tensorflow/core/framework/types.h",
                        "tensorflow/core/framework/control_flow.h",
                        "tensorflow/core/framework/kernel_def.pb.h",
                        "tensorflow/core/framework/kernel_def_builder.h",
                        "tensorflow/core/framework/tracking_allocator.h",
                        "tensorflow/core/framework/op_kernel.h",
                        "tensorflow/core/framework/op_segment.h",
                        "tensorflow/core/framework/shape_inference.h",
                        "tensorflow/core/framework/partial_tensor_shape.h",
                        "tensorflow/core/framework/device_attributes.pb.h",
                        "tensorflow/core/public/session.h",
                        "tensorflow/core/framework/tensor_slice.pb.h",
                        "tensorflow/core/framework/tensor_slice.h",
                        "tensorflow/core/util/tensor_slice_set.h",
                        "tensorflow/core/util/tensor_slice_util.h",
                        "tensorflow/core/util/tensor_slice_reader.h",
                        "tensorflow/core/util/tensor_bundle/tensor_bundle.h",
                        "tensorflow/core/framework/summary.pb.h",
                        "tensorflow/core/lib/monitoring/mobile_counter.h",
                        "tensorflow/core/lib/monitoring/mobile_gauge.h",
                        "tensorflow/core/lib/monitoring/mobile_sampler.h",
                        "tensorflow/core/profiler/internal/profiler_interface.h",
                        "tensorflow/core/profiler/lib/profiler_session.h",
                        "tensorflow/c/tf_attrtype.h",
                        "tensorflow/c/tf_datatype.h",
                        "tensorflow/c/tf_status.h",
                        "tensorflow/c/tf_status_helper.h",
                        "tensorflow/c/tf_tensor.h",
                        "tensorflow/c/checkpoint_reader.h",
                        "tensorflow/c/c_api.h",
                        "tensorflow/c/c_api_internal.h",
                        "tensorflow/c/tf_status_internal.h",
                        "tensorflow/c/tf_tensor_internal.h",
                        "tensorflow/c/env.h",
                        "tensorflow/c/kernels.h",
                        "tensorflow/c/ops.h",
                        "tensorflow/core/framework/op_def_builder.h",
                        "tensorflow/core/framework/op_def_util.h",
                        "tensorflow/core/framework/op.h",
                        "tensorflow/core/framework/types.h",
                        "tensorflow/core/graph/edgeset.h",
                        "tensorflow/core/lib/gtl/iterator_range.h",
                        //        "tensorflow/core/lib/gtl/inlined_vector.h",
                        "tensorflow/core/framework/function.h",
                        "tensorflow/core/framework/device_attributes.pb.h",
                        "tensorflow/core/framework/device_base.h",
                        "tensorflow/core/common_runtime/device.h",
                        "tensorflow/core/common_runtime/device_mgr.h",
                        "tensorflow/core/common_runtime/process_function_library_runtime.h",
                        "tensorflow/core/graph/graph.h",
                        "tensorflow/core/graph/tensor_id.h",
                        "tensorflow/core/common_runtime/graph_runner.h",
                        "tensorflow/core/common_runtime/shape_refiner.h",
                        "tensorflow/core/framework/node_def_builder.h",
                        "tensorflow/core/framework/node_def_util.h",
                        "tensorflow/core/framework/selective_registration.h",
                        "tensorflow/core/graph/node_builder.h",
                        "tensorflow/core/graph/graph_def_builder.h",
                        "tensorflow/core/graph/default_device.h",
                        "tensorflow/core/graph/graph_constructor.h",
                        "tensorflow/core/graph/gradients.h",
                        "tensorflow/core/framework/variable.pb.h",
                        "tensorflow/core/protobuf/trackable_object_graph.pb.h",
                        "tensorflow/core/protobuf/struct.pb.h",
                        "tensorflow/core/protobuf/saved_object_graph.pb.h",
                        "tensorflow/core/protobuf/saver.pb.h",
                        "tensorflow/core/protobuf/meta_graph.pb.h",
                        "tensorflow_adapters.h"},
                link = "tensorflow_cc", preload = "tensorflow_framework"),
        },
        target = "org.bytedeco.tensorflow",
        global = "org.bytedeco.tensorflow.global.tensorflow")
public class tensorflow implements BuildEnabled, LoadEnabled, InfoMapper {
    static { Loader.checkVersion("org.bytedeco", "tensorflow"); }

    private static File packageFile = null;

    /** Returns {@code Loader.cacheResource("/org/bytedeco/tensorflow/" + Loader.getPlatform() + extension + "/python/")}. */
    public static synchronized File cachePackage() throws IOException {
        if (packageFile != null) {
            return packageFile;
        }
        Loader.load(org.bytedeco.cpython.global.python.class);
        String path = Loader.load(tensorflow.class);
        if (path != null) {
            path = path.replace(File.separatorChar, '/');
            int i = path.indexOf("/org/bytedeco/tensorflow/" + Loader.getPlatform());
            int j = path.lastIndexOf("/");
            packageFile = Loader.cacheResource(path.substring(i, j) + "/python/");
        }
        return packageFile;
    }

    /** Returns {@code {numpy.cachePackages(), tensorflow.cachePackage()}}. */
    public static File[] cachePackages() throws IOException {
        File[] path = org.bytedeco.numpy.global.numpy.cachePackages();
        path = Arrays.copyOf(path, path.length + 1);
        path[path.length - 1] = cachePackage();
        return path;
    }

    private Logger logger;
    private java.util.Properties properties;
    private String encoding;
    private boolean android;

    @Override
    public void init(Logger logger, java.util.Properties properties, String encoding) {
        this.logger = logger;
        this.properties = properties;
        this.encoding = encoding;
        this.android = properties.getProperty("platform").startsWith("android-");
    }

    @Override public void init(ClassProperties properties) {
        String platform = properties.getProperty("platform");
        String extension = properties.getProperty("platform.extension");
        List<String> preloads = properties.get("platform.preload");
        List<String> resources = properties.get("platform.preloadresource");
        List<String> preloadpaths = properties.get("platform.preloadpath");

        // Only apply this at load time
        if (!Loader.isLoadLibraries()) {
            return;
        }

        // Let users enable loading of the full version of MKL
        String load = System.getProperty("org.bytedeco.openblas.load",
                      System.getProperty("org.bytedeco.mklml.load", "")).toLowerCase();

        int i = 0;
        if (load.equals("mkl") || load.equals("mkl_rt")) {
            String[] libs = {"iomp5", "libiomp5md", "mkl_core@.2", "mkl_avx@.2", "mkl_avx2@.2", "mkl_avx512@.2", "mkl_avx512_mic@.2",
                             "mkl_def@.2", "mkl_mc@.2", "mkl_mc3@.2", "mkl_intel_lp64@.2", "mkl_intel_thread@.2", "mkl_gnu_thread@.2", "mkl_rt@.2"};
            for (i = 0; i < libs.length; i++) {
                preloads.add(i, libs[i] + "#" + libs[i]);
            }
            load = "mkl_rt@.2";
            resources.add("/org/bytedeco/mkl/");
        }

        if (load.length() > 0) {
            if (platform.startsWith("linux")) {
                preloads.add(i, load + "#mklml_intel");
            } else if (platform.startsWith("macosx")) {
                preloads.add(i, load + "#mklml");
            } else if (platform.startsWith("windows")) {
                preloads.add(i, load + "#mklml");
            }
        }

        // Only apply this at load time since we don't want to copy the CUDA libraries here
        if (!Loader.isLoadLibraries() || extension == null || !extension.endsWith("-gpu")) {
            return;
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
        if (i > 0) {
            resources.add("/org/bytedeco/cuda/");
            resources.add("/org/bytedeco/tensorrt/");
        }
    }

    public void map(InfoMap infoMap) {
        infoMap.put(new Info().javaText("import org.bytedeco.tensorflow.Allocator;"))
               .put(new Info("bfloat16.h").linePatterns("struct half;").skip())
               .put(new Info("tensorflow_adapters.h").skip())
               .put(new Info("B16_DEVICE_FUNC", "EIGEN_ALWAYS_INLINE", "EIGEN_DEVICE_FUNC", "EIGEN_STRONG_INLINE", "PROTOBUF_CONSTEXPR", "PROTOBUF_FINAL",
                             "TF_FALLTHROUGH_INTENDED", "TF_ATTRIBUTE_NORETURN", "TF_ATTRIBUTE_NOINLINE", "TF_ATTRIBUTE_UNUSED", "PROTOC_EXPORT",
                             "TF_ATTRIBUTE_COLD", "TF_ATTRIBUTE_WEAK", "TF_PACKED", "TF_MUST_USE_RESULT", "GUARDED_BY", "SHOULD_REGISTER_OP_GRADIENT",
                             "TF_EXPORT", "TF_ATTRIBUTE_ALWAYS_INLINE", "GOOGLE_ATTRIBUTE_ALWAYS_INLINE", "GOOGLE_ATTRIBUTE_FUNC_ALIGN", "GOOGLE_ATTRIBUTE_NOINLINE",
                             "GOOGLE_PROTOBUF_ATTRIBUTE_NOINLINE", "GOOGLE_PREDICT_TRUE", "GOOGLE_PREDICT_FALSE", "GOOGLE_PROTOBUF_ATTRIBUTE_RETURNS_NONNULL",
                             "ACQUIRED_AFTER", "PROTOBUF_EXPORT", "PROTOBUF_ATTRIBUTE_REINITIALIZES", "PROTOBUF_ALWAYS_INLINE", "PROTOBUF_NOINLINE", "PROTOBUF_RETURNS_NONNULL",
                             "PROTOBUF_NAMESPACE_ID", "PROTOBUF_NAMESPACE_OPEN", "PROTOBUF_NAMESPACE_CLOSE", "PROTOBUF_FALLTHROUGH_INTENDED", "PROTOBUF_UNUSED").cppTypes().annotations())
               .put(new Info("TF_CHECK_OK", "TF_QCHECK_OK").cppTypes("void", "tensorflow::Status"))
               .put(new Info("TF_DISALLOW_COPY_AND_ASSIGN").cppText("#define TF_DISALLOW_COPY_AND_ASSIGN(TypeName)"))

               .put(new Info("EIGEN_DEPRECATED").cppText("#define EIGEN_DEPRECATED deprecated").cppTypes())
               .put(new Info("PROTOBUF_DEPRECATED").cppText("#define PROTOBUF_DEPRECATED deprecated").cppTypes())
               .put(new Info("PROTOBUF_RUNTIME_DEPRECATED").cppText("#define PROTOBUF_RUNTIME_DEPRECATED() deprecated").cppTypes())
               .put(new Info("GOOGLE_PROTOBUF_DEPRECATED_ATTR").cppText("#define GOOGLE_PROTOBUF_DEPRECATED_ATTR deprecated").cppTypes())
               .put(new Info("deprecated").annotations("@Deprecated"))

               .put(new Info("__ANDROID__").define(android))
               .put(new Info("!defined(IS_MOBILE_PLATFORM) && !defined(IS_SLIM_BUILD)").define(!android))
               .put(new Info("SWIG", "TENSORFLOW_LITE_PROTOS").define(true))
               .put(new Info("TENSORFLOW_USE_SYCL", "defined(PLATFORM_GOOGLE)", "defined(TENSORFLOW_PROTOBUF_USES_CORD)", "GOOGLE_PROTOBUF_ENABLE_EXPERIMENTAL_PARSER").define(false))
               .put(new Info("std::hash<Eigen::half>").pointerTypes("HalfHash"))
               .put(new Info("GenericNumTraits<tensorflow::bfloat16>").pointerTypes("Pointer"))
               .put(new Info("Eigen::NumTraits<tensorflow::bfloat16>").pointerTypes("bfloat16NumTraits"))
               .put(new Info("Eigen::QInt8", "Eigen::QUInt8").cast().valueTypes("byte").pointerTypes("BytePointer", "ByteBuffer", "byte..."))
               .put(new Info("Eigen::QInt16", "Eigen::QUInt16", "uint16", "tensorflow::uint16", "Eigen::half").cast().valueTypes("short").pointerTypes("ShortPointer", "ShortBuffer", "short..."))
               .put(new Info("Eigen::QInt32", "Eigen::QUInt32", "uint32", "tensorflow::uint32").cast().valueTypes("int").pointerTypes("IntPointer", "IntBuffer", "int..."))
               .put(new Info("short", "tensorflow::int16").valueTypes("short").pointerTypes("ShortPointer", "ShortBuffer", "short..."))
               .put(new Info("int", "int32", "tensorflow::int32").valueTypes("int").pointerTypes("IntPointer", "IntBuffer", "int..."))
               .put(new Info("long long", "tensorflow::int64", "tensorflow::uint64", "std::size_t",
                             "tensorflow::Microseconds", "tensorflow::Nanoseconds", "tensorflow::Bytes").cast().valueTypes("long").pointerTypes("LongPointer", "LongBuffer", "long..."))
               .put(new Info("float").valueTypes("float").pointerTypes("FloatPointer", "FloatBuffer", "float..."))
               .put(new Info("double").valueTypes("double").pointerTypes("DoublePointer", "DoubleBuffer", "double..."))
               .put(new Info("bool").cast().valueTypes("boolean").pointerTypes("BoolPointer", "boolean..."))
               .put(new Info("std::complex<float>").cast().pointerTypes("FloatPointer", "FloatBuffer", "float..."))
               .put(new Info("absl::optional", "absl::Span", "absl::LogSink", "TFLogSink", "std::initializer_list", "std::iterator").skip())
               .put(new Info("absl::string_view", "string", "std::string", "tensorflow::string", "tensorflow::tstring").annotations("@StdString").valueTypes("BytePointer", "String").pointerTypes("@Cast({\"char*\", \"std::string*\"}) BytePointer"))
               .put(new Info("std::set<tensorflow::string>").pointerTypes("StringSet").define())
               .put(new Info("std::list<tensorflow::string>").pointerTypes("StringList").define())
               .put(new Info("std::unordered_map<tensorflow::string,tensorflow::int32>").pointerTypes("StringIntUnorderedMap").define())
               .put(new Info("std::unordered_set<tensorflow::string>").pointerTypes("StringUnorderedSet").define())
               .put(new Info("std::vector<tensorflow::StringPiece>").pointerTypes("StringPieceVector").define())
               .put(new Info("std::vector<std::string>", "std::vector<tensorflow::string>").pointerTypes("StringVector").define())
               .put(new Info("std::vector<std::vector<int> >").pointerTypes("IntIntVector").define())
               .put(new Info("std::vector<std::pair<unsigned,unsigned> >").pointerTypes("IntIntPairVector").define())
               .put(new Info("std::vector<std::pair<tensorflow::string,tensorflow::string> >").pointerTypes("StringStringPairVector").define())
               .put(new Info("std::condition_variable", "std::mutex", "std::type_info", "std::unique_lock<std::mutex>", "Eigen::Allocator",
                             "Eigen::ThreadPoolInterface", "tensorflow::thread::ThreadPoolInterface", "tensorflow::condition_variable",
                             "tensorflow::mutex", "tensorflow::mutex_lock", "tensorflow::tf_shared_lock").cast().pointerTypes("Pointer"))

               .put(new Info("google::protobuf::int8", "google::protobuf::uint8").cast().valueTypes("byte").pointerTypes("BytePointer", "ByteBuffer", "byte[]"))
               .put(new Info("google::protobuf::int16", "google::protobuf::uint16").cast().valueTypes("short").pointerTypes("ShortPointer", "ShortBuffer", "short[]"))
               .put(new Info("google::protobuf::int32", "google::protobuf::uint32").cast().valueTypes("int").pointerTypes("IntPointer", "IntBuffer", "int[]"))
               .put(new Info("google::protobuf::int64", "google::protobuf::uint64").cast().valueTypes("long").pointerTypes("LongPointer", "LongBuffer", "long[]"))
               .put(new Info("std::pair<google::protobuf::uint64,google::protobuf::uint64>").pointerTypes("LongLongPair").define())
               .put(new Info("google::protobuf::Message").cast().pointerTypes("MessageLite"))
               .put(new Info("google::protobuf::Any", "google::protobuf::Descriptor", "google::protobuf::EnumDescriptor", "google::protobuf::Metadata",
                             "google::protobuf::Reflection").cast().pointerTypes("Pointer"))
               .put(new Info("google::protobuf::RepeatedField", "google::protobuf::RepeatedPtrField", "protobuf::RepeatedPtrField",
                             "google::protobuf::internal::ExplicitlyConstructed", "google::protobuf::internal::MapEntry", "google::protobuf::internal::MapField",
                             "google::protobuf::internal::AuxillaryParseTableField", "google::protobuf::internal::ParseTableField", "google::protobuf::internal::ParseTable",
                             "google::protobuf::internal::FieldMetadata", "google::protobuf::internal::SerializationTable", "google::protobuf::internal::proto3_preserve_unknown_",
                             "google::protobuf::internal::MergePartialFromImpl", "google::protobuf::internal::UnknownFieldParse", "google::protobuf::internal::WriteLengthDelimited",
                             "google::protobuf::arena_metrics::EnableArenaMetrics", "google::protobuf::GetEnumDescriptor", "google::quality_webanswers::TempPrivateWorkAround",
                             "google::protobuf::is_proto_enum", "is_proto_enum", "Arena::CreateMaybeMessage", "google::protobuf::internal::DescriptorTable").skip())
               .put(new Info("google::protobuf::Map<std::string,std::string>").pointerTypes("StringStringMap").define())
               .put(new Info("google::protobuf::Map<std::string,google::protobuf::int32>").pointerTypes("StringIntMap").define())
               .put(new Info("google::protobuf::Map<google::protobuf::int32,std::string>").pointerTypes("IntStringMap").define())
               .put(new Info("google::protobuf::Map<google::protobuf::uint32,std::string>").pointerTypes("IntStringMap").cast())
               .put(new Info("google::protobuf::Map<std::string,tensorflow::Feature>").pointerTypes("StringFeatureMap").define())
               .put(new Info("google::protobuf::Map<std::string,tensorflow::FeatureList>").pointerTypes("StringFeatureListMap").define())
               .put(new Info("google::protobuf::Map<std::string,tensorflow::CollectionDef>").pointerTypes("StringCollectionDefMap").define())
               .put(new Info("google::protobuf::Map<std::string,tensorflow::SignatureDef>").pointerTypes("StringSignatureDefMap").define())
               .put(new Info("google::protobuf::Map<std::string,tensorflow::TensorInfo>").pointerTypes("StringTensorInfoMap").define())
               .put(new Info("google::protobuf::Map<google::protobuf::uint32,tensorflow::FunctionDef_ArgAttrs>").pointerTypes("IntFunctionDef_ArgAttrsMap").define())
               .put(new Info("google::protobuf::Map<std::string,tensorflow::StructuredValue>").pointerTypes("StringStructuredValueMap").define())
               .put(new Info("google::protobuf::Map<std::string,tensorflow::SavedConcreteFunction>").pointerTypes("StringSavedConcreteFunctionMap").define())

               .put(new Info("tensorflow::error::protobuf_tensorflow_2fcore_2flib_2fcore_2ferror_5fcodes_2eproto::TableStruct",
                             "tensorflow::protobuf_tensorflow_2fcore_2fprotobuf_2fdebug_2eproto::TableStruct",
                             "tensorflow::protobuf_tensorflow_2fcore_2fprotobuf_2fconfig_2eproto::TableStruct",
                             "tensorflow::protobuf_tensorflow_2fcore_2fframework_2fcost_5fgraph_2eproto::TableStruct",
                             "tensorflow::protobuf_tensorflow_2fcore_2fframework_2fstep_5fstats_2eproto::TableStruct",
                             "tensorflow::protobuf_tensorflow_2fcore_2fframework_2fversions_2eproto::TableStruct",
                             "tensorflow::protobuf_tensorflow_2fcore_2fframework_2fallocation_5fdescription_2eproto::TableStruct",
                             "tensorflow::protobuf_tensorflow_2fcore_2fframework_2ftensor_5fshape_2eproto::TableStruct",
                             "tensorflow::protobuf_tensorflow_2fcore_2fframework_2ftypes_2eproto::TableStruct",
                             "tensorflow::protobuf_tensorflow_2fcore_2fframework_2fresource_5fhandle_2eproto::TableStruct",
                             "tensorflow::protobuf_tensorflow_2fcore_2fframework_2ftensor_2eproto::TableStruct",
                             "tensorflow::protobuf_tensorflow_2fcore_2fframework_2ftensor_5fdescription_2eproto::TableStruct",
                             "tensorflow::protobuf_tensorflow_2fcore_2fframework_2fattr_5fvalue_2eproto::TableStruct",
                             "tensorflow::protobuf_tensorflow_2fcore_2fframework_2fnode_5fdef_2eproto::TableStruct",
                             "tensorflow::protobuf_tensorflow_2fcore_2fframework_2fop_5fdef_2eproto::TableStruct",
                             "tensorflow::protobuf_tensorflow_2fcore_2fframework_2ffunction_2eproto::TableStruct",
                             "tensorflow::protobuf_tensorflow_2fcore_2fframework_2fgraph_2eproto::TableStruct",
                             "tensorflow::protobuf_tensorflow_2fcore_2fframework_2fdevice_5fattributes_2eproto::TableStruct",
                             "TableStruct_tensorflow_2fcore_2flib_2fcore_2ferror_5fcodes_2eproto",
                             "TableStruct_tensorflow_2fcore_2fexample_2ffeature_2eproto",
                             "TableStruct_tensorflow_2fcore_2fexample_2fexample_2eproto",
                             "TableStruct_tensorflow_2fcore_2fprotobuf_2fdebug_2eproto",
                             "TableStruct_tensorflow_2fcore_2fprotobuf_2fcluster_2eproto",
                             "TableStruct_tensorflow_2fcore_2fprotobuf_2fverifier_5fconfig_2eproto",
                             "TableStruct_tensorflow_2fcore_2fprotobuf_2frewriter_5fconfig_2eproto",
                             "TableStruct_tensorflow_2fcore_2fprotobuf_2fconfig_2eproto",
                             "TableStruct_tensorflow_2fcore_2fframework_2fcost_5fgraph_2eproto",
                             "TableStruct_tensorflow_2fcore_2fframework_2fstep_5fstats_2eproto",
                             "TableStruct_tensorflow_2fcore_2fframework_2fversions_2eproto",
                             "TableStruct_tensorflow_2fcore_2fframework_2fallocation_5fdescription_2eproto",
                             "TableStruct_tensorflow_2fcore_2fframework_2ftensor_5fshape_2eproto",
                             "TableStruct_tensorflow_2fcore_2fframework_2ftypes_2eproto",
                             "TableStruct_tensorflow_2fcore_2fframework_2fresource_5fhandle_2eproto",
                             "TableStruct_tensorflow_2fcore_2fframework_2ftensor_2eproto",
                             "TableStruct_tensorflow_2fcore_2fframework_2ftensor_5fdescription_2eproto",
                             "TableStruct_tensorflow_2fcore_2fframework_2fattr_5fvalue_2eproto",
                             "TableStruct_tensorflow_2fcore_2fframework_2fnode_5fdef_2eproto",
                             "TableStruct_tensorflow_2fcore_2fframework_2fapi_5fdef_2eproto",
                             "TableStruct_tensorflow_2fcore_2fframework_2fop_5fdef_2eproto",
                             "TableStruct_tensorflow_2fcore_2fframework_2ffunction_2eproto",
                             "TableStruct_tensorflow_2fcore_2fframework_2fgraph_2eproto",
                             "TableStruct_tensorflow_2fcore_2fframework_2fkernel_5fdef_2eproto",
                             "TableStruct_tensorflow_2fcore_2fframework_2fdevice_5fattributes_2eproto",
                             "TableStruct_tensorflow_2fcore_2fframework_2ftensor_5fslice_2eproto",
                             "TableStruct_tensorflow_2fcore_2fframework_2fsummary_2eproto",
                             "TableStruct_tensorflow_2fcore_2fframework_2fvariable_2eproto",
                             "TableStruct_tensorflow_2fcore_2fprotobuf_2ftrackable_5fobject_5fgraph_2eproto",
                             "TableStruct_tensorflow_2fcore_2fprotobuf_2fstruct_2eproto",
                             "TableStruct_tensorflow_2fcore_2fprotobuf_2fsaved_5fobject_5fgraph_2eproto",
                             "TableStruct_tensorflow_2fcore_2fprotobuf_2fsaver_2eproto",
                             "TableStruct_tensorflow_2fcore_2fprotobuf_2fmeta_5fgraph_2eproto",
                             "TableStruct_tensorflow_2fcore_2fprotobuf_2feager_5fservice_2eproto",
                             "TableStruct_tensorflow_2fcore_2fprotobuf_2ftensorflow_5fserver_2eproto",
                             "TableStruct_tensorflow_2fcore_2fprotobuf_2fnamed_5ftensor_2eproto",
                             "TableStruct_tensorflow_2fcore_2fprotobuf_2fmaster_2eproto",
                             "TableStruct_tensorflow_2fcore_2fprotobuf_2fworker_2eproto",
                             "tensorflow::Features_FeatureEntry_DoNotUse", "tensorflow::FeatureLists_FeatureListEntry_DoNotUse", "tensorflow::JobDef_TasksEntry_DoNotUse",
                             "tensorflow::RewriterConfig_CustomGraphOptimizer_ParameterMapEntry_DoNotUse", "tensorflow::ConfigProto_DeviceCountEntry_DoNotUse", "tensorflow::CallableOptions_FeedDevicesEntry_DoNotUse",
                             "tensorflow::CallableOptions_FetchDevicesEntry_DoNotUse", "tensorflow::DeviceStepStats_ThreadNamesEntry_DoNotUseDefaultTypeInternal", "tensorflow::DeviceStepStats_ThreadNamesEntry_DoNotUse",
                             "tensorflow::NameAttrList_AttrEntry_DoNotUse", "tensorflow::NodeDef_AttrEntry_DoNotUse", "tensorflow::FunctionDef_ArgAttrEntry_DoNotUseDefaultTypeInternal",
                             "tensorflow::FunctionDef_ArgAttrsDefaultTypeInternal", "tensorflow::FunctionDef_ArgAttrs_AttrEntry_DoNotUseDefaultTypeInternal", "tensorflow::FunctionDef_ControlRetEntry_DoNotUseDefaultTypeInternal",
                             "tensorflow::FunctionDef_AttrEntry_DoNotUse", "tensorflow::FunctionDef_ArgAttrs_AttrEntry_DoNotUse", "tensorflow::FunctionDef_ArgAttrEntry_DoNotUse", "tensorflow::FunctionDef_RetEntry_DoNotUse",
                             "tensorflow::FunctionDef_ControlRetEntry_DoNotUse", "tensorflow::MetaGraphDef_CollectionDefEntry_DoNotUse", "tensorflow::MetaGraphDef_SignatureDefEntry_DoNotUse",
                             "tensorflow::SignatureDef_InputsEntry_DoNotUse", "tensorflow::SignatureDef_OutputsEntry_DoNotUse", "tensorflow::eager::Operation_AttrsEntry_DoNotUse",
                             "tensorflow::BytesListDefaultTypeInternal", "tensorflow::FeatureDefaultTypeInternal", "tensorflow::FeatureListDefaultTypeInternal",
                             "tensorflow::Features_FeatureEntryDefaultTypeInternal", "tensorflow::FeatureLists_FeatureListEntryDefaultTypeInternal",
                             "tensorflow::FeatureListsDefaultTypeInternal", "tensorflow::FeatureLists_FeatureListEntry_DoNotUseDefaultTypeInternal", "tensorflow::FeaturesDefaultTypeInternal",
                             "tensorflow::Features_FeatureEntry_DoNotUseDefaultTypeInternal", "tensorflow::FloatListDefaultTypeInternal", "tensorflow::Int64ListDefaultTypeInternal",
                             "tensorflow::ExampleDefaultTypeInternal", "tensorflow::SequenceExampleDefaultTypeInternal", "tensorflow::VariantTensorDataProtoDefaultTypeInternal",
                             "tensorflow::JobDef_TasksEntryDefaultTypeInternal", "tensorflow::ResourceHandleProtoDefaultTypeInternal", "tensorflow::NameAttrList_AttrEntryDefaultTypeInternal",
                             "tensorflow::NodeDef_AttrEntryDefaultTypeInternal", "tensorflow::FunctionDef_AttrEntryDefaultTypeInternal", "tensorflow::FunctionDef_RetEntryDefaultTypeInternal",
                             "tensorflow::DeviceAttributesDefaultTypeInternal", "tensorflow::DeviceLocalityDefaultTypeInternal", "tensorflow::ConfigProto_DeviceCountEntryDefaultTypeInternal",
                             "tensorflow::AutoParallelOptionsDefaultTypeInternal", "tensorflow::ClusterDefDefaultTypeInternal", "tensorflow::JobDefDefaultTypeInternal",
                             "tensorflow::DebugOptionsDefaultTypeInternal", "tensorflow::DebugTensorWatchDefaultTypeInternal", "tensorflow::AllocatorMemoryUsedDefaultTypeInternal",
                             "tensorflow::ConfigProtoDefaultTypeInternal", "tensorflow::CostGraphDefDefaultTypeInternal", "tensorflow::CostGraphDef_NodeDefaultTypeInternal",
                             "tensorflow::CostGraphDef_Node_InputInfoDefaultTypeInternal", "tensorflow::CostGraphDef_Node_OutputInfoDefaultTypeInternal", "tensorflow::DeviceStepStatsDefaultTypeInternal",
                             "tensorflow::GPUOptionsDefaultTypeInternal", "tensorflow::GraphDefDefaultTypeInternal", "tensorflow::GraphOptionsDefaultTypeInternal", "tensorflow::MemoryStatsDefaultTypeInternal",
                             "tensorflow::NodeExecStatsDefaultTypeInternal", "tensorflow::NodeOutputDefaultTypeInternal", "tensorflow::OptimizerOptionsDefaultTypeInternal",
                             "tensorflow::RPCOptionsDefaultTypeInternal", "tensorflow::RewriterConfigDefaultTypeInternal", "tensorflow::RunMetadataDefaultTypeInternal",
                             "tensorflow::RunOptionsDefaultTypeInternal", "tensorflow::StepStatsDefaultTypeInternal", "tensorflow::ThreadPoolOptionProtoDefaultTypeInternal",
                             "tensorflow::TensorShapeProtoDefaultTypeInternal", "tensorflow::TensorShapeProto_DimDefaultTypeInternal", "tensorflow::AllocationDescriptionDefaultTypeInternal",
                             "tensorflow::TensorDescriptionDefaultTypeInternal", "tensorflow::VersionDefDefaultTypeInternal", "tensorflow::ResourceHandleDefaultTypeInternal",
                             "tensorflow::TensorProtoDefaultTypeInternal", "tensorflow::AttrValueDefaultTypeInternal", "tensorflow::AttrValue_ListValueDefaultTypeInternal",
                             "tensorflow::NameAttrListDefaultTypeInternal", "tensorflow::NodeDefDefaultTypeInternal", "tensorflow::OpDefDefaultTypeInternal", "tensorflow::OpDef_ArgDefDefaultTypeInternal",
                             "tensorflow::OpDef_AttrDefDefaultTypeInternal", "tensorflow::OpDeprecationDefaultTypeInternal", "tensorflow::OpListDefaultTypeInternal",
                             "tensorflow::FunctionDefDefaultTypeInternal", "tensorflow::FunctionDefLibraryDefaultTypeInternal", "tensorflow::GradientDefDefaultTypeInternal",
                             "tensorflow::SaverDefDefaultTypeInternal", "tensorflow::AssetFileDefDefaultTypeInternal", "tensorflow::CollectionDefDefaultTypeInternal",
                             "tensorflow::CollectionDef_AnyListDefaultTypeInternal", "tensorflow::CollectionDef_BytesListDefaultTypeInternal", "tensorflow::CollectionDef_FloatListDefaultTypeInternal",
                             "tensorflow::CollectionDef_Int64ListDefaultTypeInternal", "tensorflow::CollectionDef_NodeListDefaultTypeInternal", "tensorflow::MetaGraphDefDefaultTypeInternal",
                             "tensorflow::MetaGraphDef_CollectionDefEntryDefaultTypeInternal", "tensorflow::MetaGraphDef_CollectionDefEntryDefaultTypeInternal",
                             "tensorflow::MetaGraphDef_MetaInfoDefDefaultTypeInternal", "tensorflow::MetaGraphDef_SignatureDefEntryDefaultTypeInternal",
                             "tensorflow::MetaGraphDef_SignatureDefEntryDefaultTypeInternal", "tensorflow::SignatureDefDefaultTypeInternal", "tensorflow::SignatureDef_InputsEntryDefaultTypeInternal",
                             "tensorflow::SignatureDef_OutputsEntryDefaultTypeInternal", "tensorflow::TensorInfoDefaultTypeInternal", "tensorflow::TensorInfo_CooSparseDefaultTypeInternal",
                             "tensorflow::TensorSliceProtoDefaultTypeInternal", "tensorflow::TensorSliceProto_ExtentDefaultTypeInternal", "tensorflow::ApiDefDefaultTypeInternal",
                             "tensorflow::ApiDef_ArgDefaultTypeInternal", "tensorflow::ApiDef_AttrDefaultTypeInternal", "tensorflow::ApiDef_EndpointDefaultTypeInternal",
                             "tensorflow::ApiDefsDefaultTypeInternal", "tensorflow::DebuggedSourceFileDefaultTypeInternal", "tensorflow::DebuggedSourceFilesDefaultTypeInternal",
                             "tensorflow::AllocationRecordDefaultTypeInternal","tensorflow::GPUOptions_ExperimentalDefaultTypeInternal", "tensorflow::GPUOptions_Experimental_VirtualDevicesDefaultTypeInternal",
                             "tensorflow::CallableOptionsDefaultTypeInternal", "tensorflow::TensorConnectionDefaultTypeInternal",
                             "tensorflow::CallableOptions_FeedDevicesEntry_DoNotUseDefaultTypeInternal", "tensorflow::CallableOptions_FetchDevicesEntry_DoNotUseDefaultTypeInternal",
                             "tensorflow::InterconnectLinkDefaultTypeInternal", "tensorflow::LocalLinksDefaultTypeInternal",
                             "tensorflow::JobDef_TasksEntry_DoNotUseDefaultTypeInternal", "tensorflow::ConfigProto_DeviceCountEntry_DoNotUseDefaultTypeInternal",
                             "tensorflow::NameAttrList_AttrEntry_DoNotUseDefaultTypeInternal", "tensorflow::NodeDef_AttrEntry_DoNotUseDefaultTypeInternal",
                             "tensorflow::FunctionDef_AttrEntry_DoNotUseDefaultTypeInternal", "tensorflow::FunctionDef_RetEntry_DoNotUseDefaultTypeInternal",
                             "tensorflow::MetaGraphDef_CollectionDefEntry_DoNotUseDefaultTypeInternal", "tensorflow::MetaGraphDef_SignatureDefEntry_DoNotUseDefaultTypeInternal",
                             "tensorflow::SignatureDef_InputsEntry_DoNotUseDefaultTypeInternal", "tensorflow::SignatureDef_OutputsEntry_DoNotUseDefaultTypeInternal",
                             "tensorflow::RewriterConfig_CustomGraphOptimizerDefaultTypeInternal", "tensorflow::RewriterConfig_CustomGraphOptimizer_ParameterMapEntry_DoNotUseDefaultTypeInternal",
                             "tensorflow::RewriterConfig_CustomGraphOptimizer_ParameterMapEntryDefaultTypeInternal", "tensorflow::ScopedAllocatorOptionsDefaultTypeInternal",
                             "tensorflow::ConfigProto_ExperimentalDefaultTypeInternal", "tensorflow::RunOptions_ExperimentalDefaultTypeInternal",
                             "tensorflow::KernelDefDefaultTypeInternal", "tensorflow::KernelListDefaultTypeInternal", "tensorflow::KernelDef_AttrConstraintDefaultTypeInternal",
                             "tensorflow::NodeDef_ExperimentalDebugInfoDefaultTypeInternal", "tensorflow::ServerDefDefaultTypeInternal", "tensorflow::eager::CloseContextRequestDefaultTypeInternal",
                             "tensorflow::eager::CloseContextResponseDefaultTypeInternal", "tensorflow::eager::CreateContextRequestDefaultTypeInternal", "tensorflow::eager::CreateContextResponseDefaultTypeInternal",
                             "tensorflow::eager::EnqueueRequestDefaultTypeInternal", "tensorflow::eager::EnqueueResponseDefaultTypeInternal", "tensorflow::eager::KeepAliveRequestDefaultTypeInternal",
                             "tensorflow::eager::KeepAliveResponseDefaultTypeInternal", "tensorflow::eager::OperationDefaultTypeInternal", "tensorflow::eager::Operation_AttrsEntry_DoNotUseDefaultTypeInternal",
                             "tensorflow::eager::QueueItemDefaultTypeInternal", "tensorflow::eager::QueueResponseDefaultTypeInternal", "tensorflow::eager::RegisterFunctionRequestDefaultTypeInternal",
                             "tensorflow::eager::RegisterFunctionResponseDefaultTypeInternal", "tensorflow::eager::RemoteTensorHandleDefaultTypeInternal", "tensorflow::eager::SendTensorRequestDefaultTypeInternal",
                             "tensorflow::eager::SendTensorResponseDefaultTypeInternal", "tensorflow::eager::WaitQueueDoneRequestDefaultTypeInternal", "tensorflow::eager::WaitQueueDoneResponseDefaultTypeInternal",
                             "tensorflow::RunMetadata_FunctionGraphsDefaultTypeInternal", "tensorflow::FunctionSpecDefaultTypeInternal", "tensorflow::VerifierConfigDefaultTypeInternal",
                             "tensorflow::HistogramProtoDefaultTypeInternal", "tensorflow::SummaryDefaultTypeInternal", "tensorflow::SummaryDescriptionDefaultTypeInternal",
                             "tensorflow::SummaryMetadataDefaultTypeInternal", "tensorflow::SummaryMetadata_PluginDataDefaultTypeInternal", "tensorflow::Summary_AudioDefaultTypeInternal",
                             "tensorflow::Summary_ImageDefaultTypeInternal", "tensorflow::Summary_ValueDefaultTypeInternal", "tensorflow::SaveSliceInfoDefDefaultTypeInternal",
                             "tensorflow::VariableDefDefaultTypeInternal", "tensorflow::TrackableObjectGraphDefaultTypeInternal", "tensorflow::TrackableObjectGraph_TrackableObjectDefaultTypeInternal",
                             "tensorflow::TrackableObjectGraph_TrackableObject_ObjectReferenceDefaultTypeInternal", "tensorflow::TrackableObjectGraph_TrackableObject_SerializedTensorDefaultTypeInternal",
                             "tensorflow::TrackableObjectGraph_TrackableObject_SlotVariableReferenceDefaultTypeInternal", "tensorflow::DictValue_FieldsEntry_DoNotUse", "tensorflow::DictValueDefaultTypeInternal",
                             "tensorflow::DictValue_FieldsEntry_DoNotUseDefaultTypeInternal", "tensorflow::ListValueDefaultTypeInternal", "tensorflow::NamedTupleValueDefaultTypeInternal",
                             "tensorflow::NoneValueDefaultTypeInternal", "tensorflow::PairValueDefaultTypeInternal", "tensorflow::StructuredValueDefaultTypeInternal", "tensorflow::TensorSpecProtoDefaultTypeInternal",
                             "tensorflow::TupleValueDefaultTypeInternal", "tensorflow::SavedObjectGraph_ConcreteFunctionsEntry_DoNotUse", "tensorflow::SavedAssetDefaultTypeInternal",
                             "tensorflow::SavedBareConcreteFunctionDefaultTypeInternal", "tensorflow::SavedConcreteFunctionDefaultTypeInternal", "tensorflow::SavedConstantDefaultTypeInternal",
                             "tensorflow::SavedFunctionDefaultTypeInternal", "tensorflow::SavedObjectDefaultTypeInternal", "tensorflow::SavedObjectGraphDefaultTypeInternal",
                             "tensorflow::SavedObjectGraph_ConcreteFunctionsEntry_DoNotUseDefaultTypeInternal", "tensorflow::SavedResourceDefaultTypeInternal", "tensorflow::SavedUserObjectDefaultTypeInternal",
                             "tensorflow::SavedVariableDefaultTypeInternal", "tensorflow::NamedTensorProtoDefaultTypeInternal", "tensorflow::CloseSessionRequestDefaultTypeInternal",
                             "tensorflow::CloseSessionResponseDefaultTypeInternal", "tensorflow::CreateSessionRequestDefaultTypeInternal", "tensorflow::CreateSessionResponseDefaultTypeInternal",
                             "tensorflow::ExtendSessionRequestDefaultTypeInternal", "tensorflow::ExtendSessionResponseDefaultTypeInternal", "tensorflow::ListDevicesRequestDefaultTypeInternal",
                             "tensorflow::ListDevicesResponseDefaultTypeInternal", "tensorflow::MakeCallableRequestDefaultTypeInternal", "tensorflow::MakeCallableResponseDefaultTypeInternal",
                             "tensorflow::PartialRunSetupRequestDefaultTypeInternal", "tensorflow::PartialRunSetupResponseDefaultTypeInternal", "tensorflow::ReleaseCallableRequestDefaultTypeInternal",
                             "tensorflow::ReleaseCallableResponseDefaultTypeInternal", "tensorflow::ResetRequestDefaultTypeInternal", "tensorflow::ResetResponseDefaultTypeInternal",
                             "tensorflow::RunCallableRequestDefaultTypeInternal", "tensorflow::RunCallableResponseDefaultTypeInternal", "tensorflow::RunStepRequestDefaultTypeInternal",
                             "tensorflow::RunStepResponseDefaultTypeInternal", "tensorflow::CleanupAllRequestDefaultTypeInternal", "tensorflow::CleanupAllResponseDefaultTypeInternal",
                             "tensorflow::CleanupGraphRequestDefaultTypeInternal", "tensorflow::CleanupGraphResponseDefaultTypeInternal", "tensorflow::CompleteGroupRequestDefaultTypeInternal",
                             "tensorflow::CompleteGroupResponseDefaultTypeInternal", "tensorflow::CompleteInstanceRequestDefaultTypeInternal", "tensorflow::CompleteInstanceResponseDefaultTypeInternal",
                             "tensorflow::CreateWorkerSessionRequestDefaultTypeInternal", "tensorflow::CreateWorkerSessionResponseDefaultTypeInternal", "tensorflow::DeleteWorkerSessionRequestDefaultTypeInternal",
                             "tensorflow::DeleteWorkerSessionResponseDefaultTypeInternal", "tensorflow::DeregisterGraphRequestDefaultTypeInternal", "tensorflow::DeregisterGraphResponseDefaultTypeInternal",
                             "tensorflow::ExecutorOptsDefaultTypeInternal", "tensorflow::GetStatusRequestDefaultTypeInternal", "tensorflow::GetStatusResponseDefaultTypeInternal",
                             "tensorflow::GetStepSequenceRequestDefaultTypeInternal", "tensorflow::GetStepSequenceResponseDefaultTypeInternal", "tensorflow::LabeledStepStatsDefaultTypeInternal",
                             "tensorflow::LoggingRequestDefaultTypeInternal", "tensorflow::LoggingResponseDefaultTypeInternal", "tensorflow::MarkRecvFinishedRequestDefaultTypeInternal",
                             "tensorflow::MarkRecvFinishedResponseDefaultTypeInternal", "tensorflow::RecvBufRequestDefaultTypeInternal", "tensorflow::RecvBufResponseDefaultTypeInternal",
                             "tensorflow::RecvTensorRequestDefaultTypeInternal", "tensorflow::RecvTensorResponseDefaultTypeInternal", "tensorflow::RegisterGraphRequestDefaultTypeInternal",
                             "tensorflow::RegisterGraphResponseDefaultTypeInternal", "tensorflow::RunGraphRequestDefaultTypeInternal", "tensorflow::RunGraphResponseDefaultTypeInternal",
                             "tensorflow::StepSequenceDefaultTypeInternal", "tensorflow::TraceOptsDefaultTypeInternal", "tensorflow::TracingRequestDefaultTypeInternal",
                             "tensorflow::TracingResponseDefaultTypeInternal", "tensorflow::SessionMetadataDefaultTypeInternal", "tensorflow::ResourceHandleProto_DtypeAndShapeDefaultTypeInternal",
                             "tensorflow::TypeSpecProtoDefaultTypeInternal", "tensorflow::TensorInfo_CompositeTensorDefaultTypeInternal", "tensorflow::eager::SendTensorOpDefaultTypeInternal").skip())

               .put(new Info("tensorflow::core::RefCounted").cast().pointerTypes("Pointer"))
               .put(new Info("tensorflow::ConditionResult").cast().valueTypes("int"))
               .put(new Info("tensorflow::protobuf::Message", "tensorflow::protobuf::MessageLite").cast().pointerTypes("MessageLite"))
               .put(new Info("tensorflow::Allocator::is_simple<bfloat16>").skip())

               .put(new Info("basic/containers").cppTypes("tensorflow::gtl::InlinedVector", "google::protobuf::Map", "tensorflow::gtl::FlatMap", "tensorflow::gtl::FlatSet"))
               .put(new Info("tensorflow::TrackingAllocator").purify())
               .put(new Info("tensorflow::DeviceContext").pointerTypes("DeviceContext"))
               .put(new Info("tensorflow::register_kernel::Name").pointerTypes("RegisterKernelName"))
               .put(new Info("tensorflow::register_kernel::system::Name").pointerTypes("RegisterKernelSystemName"))
               .put(new Info("tensorflow::DataType").cast().valueTypes("int").pointerTypes("IntPointer"))
               .put(new Info("std::pair<tensorflow::Allocator*,tensorflow::TrackingAllocator*>").pointerTypes("WrappedAllocator").define())
               .put(new Info("std::tuple<size_t,size_t,size_t>").cast().pointerTypes("SizeTPointer"))
               .put(new Info("std::unique_ptr<tensorflow::Device>").valueTypes("@MoveUniquePtr Device").pointerTypes("@UniquePtr Device"))
               .put(new Info("std::unique_ptr<tensorflow::DeviceMgr>", "std::unique_ptr<const tensorflow::DeviceMgr>").valueTypes("@MoveUniquePtr DeviceMgr").pointerTypes("@UniquePtr DeviceMgr"))
               .put(new Info("std::unique_ptr<tensorflow::Thread>").valueTypes("@MoveUniquePtr Thread").pointerTypes("@UniquePtr Thread"))
               .put(new Info("std::unique_ptr<tensorflow::GraphMgr>").valueTypes("@MoveUniquePtr GraphMgr").pointerTypes("@UniquePtr GraphMgr"))
               .put(new Info("std::unique_ptr<tensorflow::OpKernel>").valueTypes("@MoveUniquePtr OpKernel").pointerTypes("@UniquePtr OpKernel"))
               .put(new Info("std::unique_ptr<tensorflow::TensorShape>").valueTypes("@MoveUniquePtr TensorShape").pointerTypes("@UniquePtr TensorShape"))
               .put(new Info("std::unique_ptr<tensorflow::ServerInterface>").valueTypes("@MoveUniquePtr ServerInterface").pointerTypes("@UniquePtr ServerInterface"))
               .put(new Info("std::unique_ptr<tensorflow::CollectiveExecutor::Handle>").valueTypes("@MoveUniquePtr CollectiveExecutor.Handle").pointerTypes("@UniquePtr CollectiveExecutor.Handle"))
               .put(new Info("std::unique_ptr<tensorflow::EagerNode>").valueTypes("@MoveUniquePtr EagerNode").pointerTypes("@UniquePtr EagerNode"))
               .put(new Info("std::unique_ptr<tensorflow::EagerExecutor>").valueTypes("@MoveUniquePtr EagerExecutor").pointerTypes("@UniquePtr EagerExecutor"))
               .put(new Info("std::unique_ptr<tensorflow::eager::EagerClientCache>").valueTypes("@MoveUniquePtr EagerClientCache").pointerTypes("@UniquePtr EagerClientCache"))
               .put(new Info("std::unique_ptr<tensorflow::eager::RemoteMgr,std::function<void(eager::RemoteMgr*)> >")
                       .valueTypes("@MoveUniquePtr(\"tensorflow::eager::RemoteMgr,std::function<void(tensorflow::eager::RemoteMgr*)>\") RemoteMgr")
                       .pointerTypes("@UniquePtr(\"tensorflow::eager::RemoteMgr,std::function<void(tensorflow::eager::RemoteMgr*)>\") RemoteMgr"))
               .put(new Info("std::unique_ptr<tensorflow::kernel_factory::OpKernelFactory>").valueTypes("@MoveUniquePtr OpKernelFactory").pointerTypes("@UniquePtr OpKernelFactory"))
               .put(new Info("std::unique_ptr<tensorflow::port::StringListDecoder>").valueTypes("@MoveUniquePtr StringListDecoder").pointerTypes("@UniquePtr StringListDecoder"))
               .put(new Info("std::unique_ptr<tensorflow::port::StringListEncoder>").valueTypes("@MoveUniquePtr StringListEncoder").pointerTypes("@UniquePtr StringListEncoder"))
               .put(new Info("std::unique_ptr<tensorflow::monitoring::Buckets>").valueTypes("@MoveUniquePtr Buckets").pointerTypes("@UniquePtr Buckets"))
               .put(new Info("std::unique_ptr<tensorflow::ProfilerSession>").valueTypes("@MoveUniquePtr ProfilerSession").pointerTypes("@UniquePtr ProfilerSession"))
               .put(new Info("std::unique_ptr<tensorflow::profiler::ProfilerInterface>").valueTypes("@MoveUniquePtr ProfilerInterface").pointerTypes("@UniquePtr ProfilerInterface"))
               .put(new Info("std::unique_ptr<tensorflow::WorkerCacheInterface>").valueTypes("@MoveUniquePtr WorkerCacheInterface").pointerTypes("@UniquePtr WorkerCacheInterface"))
               .put(new Info("std::unique_ptr<tensorflow::ClusterFunctionLibraryRuntime>").valueTypes("@MoveUniquePtr ClusterFunctionLibraryRuntime").pointerTypes("@UniquePtr ClusterFunctionLibraryRuntime"))
               .put(new Info("std::unique_ptr<tensorflow::RemoteTensorHandleData>").valueTypes("@MoveUniquePtr RemoteTensorHandleData").pointerTypes("@UniquePtr RemoteTensorHandleData"))
               .put(new Info("std::unique_ptr<tensorflow::UnshapedRemoteTensorHandleData>").valueTypes("@MoveUniquePtr UnshapedRemoteTensorHandleData").pointerTypes("@UniquePtr UnshapedRemoteTensorHandleData"))
               .put(new Info("std::unique_ptr<TFE_OpInferenceContext>").valueTypes("@MoveUniquePtr TFE_OpInferenceContext").pointerTypes("@UniquePtr TFE_OpInferenceContext"))
               .put(new Info("tensorflow::core::RefCountPtr<KernelAndDevice>")
                       .valueTypes("@MoveUniquePtr(\"tensorflow::KernelAndDevice,tensorflow::core::RefCountDeleter\") KernelAndDevice")
                       .pointerTypes("@UniquePtr(\"tensorflow::KernelAndDevice,tensorflow::core::RefCountDeleter\") KernelAndDevice"))
               .put(new Info("tensorflow::ProfilerFactory").pointerTypes("@Cast(\"tensorflow::ProfilerFactory\") ProfilerFactory"))

               .put(new Info("std::vector<std::unique_ptr<tensorflow::Device> >", "std::vector<std::unique_ptr<tensorflow::profiler::ProfilerInterface> >").skip())
               .put(new Info("std::vector<tensorflow::Device*>").pointerTypes("DeviceVector").define())
               .put(new Info("std::vector<tensorflow::DeviceContext*>").pointerTypes("DeviceContextVector").define())
               .put(new Info("tensorflow::gtl::InlinedVector<tensorflow::AllocatorAttributes,4>").pointerTypes("AllocatorAttributesVector").define())
               .put(new Info("tensorflow::gtl::InlinedVector<tensorflow::AllocRecord,4>").pointerTypes("AllocRecordVector").define())
               .put(new Info("tensorflow::gtl::InlinedVector<tensorflow::DeviceContext*,4>").pointerTypes("DeviceContextInlinedVector").define())
               .put(new Info("tensorflow::gtl::InlinedVector<tensorflow::DeviceType,4>").pointerTypes("DeviceTypeVector").define())
               .put(new Info("tensorflow::gtl::InlinedVector<tensorflow::TensorValue,4>", "gtl::InlinedVector<TensorValue,4>").pointerTypes("TensorValueVector").define())
               .put(new Info("tensorflow::gtl::InlinedVector<tensorflow::OpKernelContext::WrappedAllocator,4>").pointerTypes("WrappedAllocatorVector").define())
               .put(new Info("tensorflow::gtl::InlinedVector<tensorflow::int64,4>").pointerTypes("LongVector").define())
               .put(new Info("tensorflow::gtl::InlinedVector<tensorflow::DataType,4>").pointerTypes("DataTypeVector").define())
               .put(new Info("tensorflow::DataTypeSlice").cast().pointerTypes("DataTypeVector"))
               .put(new Info("tensorflow::NumberTypes", "tensorflow::QuantizedTypes", "tensorflow::RealAndQuantizedTypes").skip())

               .put(new Info("tensorflow::OpArgIterator", "tensorflow::OpInputList::Iterator",
                             "tensorflow::OpMutableInputList::Iterator", "tensorflow::OpOutputList::Iterator",
                             "tensorflow::OpKernelContext::inc_num_deferred_ops_function", "tensorflow::OpKernelContext::dec_num_deferred_ops_function").skip())
               .put(new Info("tensorflow::Tensor").base("AbstractTensor").pointerTypes("Tensor"))
               .put(new Info("tensorflow::Tensor::HostScalarTensorBufferBase").skip())
               .put(new Info("tensorflow::TensorBuffer").virtualize())
               .put(new Info("tensorflow::Tensor(tensorflow::DataType, tensorflow::TensorShape&, tensorflow::TensorBuffer*)").javaText(
                       "public Tensor(@Cast(\"tensorflow::DataType\") int type, TensorShape shape, TensorBuffer buf) { super((Pointer)null); allocate(type, shape, buf); this.buffer = buf; }\n"
                     + "private native void allocate(@Cast(\"tensorflow::DataType\") int type, @Const @ByRef TensorShape shape, TensorBuffer buf);\n"
                     + "private TensorBuffer buffer; // a reference to prevent deallocation\n"
                     + "public Tensor(@Cast(\"tensorflow::DataType\") int type, TensorShape shape, final Pointer data) {\n"
                     + "    this(type, shape, new TensorBuffer(data) {\n"
                     + "        @Override public Pointer data() { return data; }\n"
                     + "        @Override public long size() { return data.limit(); }\n"
                     + "        @Override public TensorBuffer root_buffer() { return this; }\n"
                     + "        @Override public void FillAllocationDescription(AllocationDescription proto) { }\n"
                     + "    });\n"
                     + "}\n"))
               .put(new Info("tensorflow::Session").base("AbstractSession"))
               .put(new Info("tensorflow::Session::~Session()").javaText("/** Calls {@link tensorflow#NewSession(SessionOptions)} and registers a deallocator. */\n"
                                                                       + "public Session(SessionOptions options) { super(options); }"))
               .put(new Info("tensorflow::TensorShapeBase<tensorflow::TensorShape>", "tensorflow::TensorShapeBase<tensorflow::PartialTensorShape>").pointerTypes("TensorShapeBase"))
               .put(new Info("tensorflow::TensorShapeIter<tensorflow::TensorShape>").pointerTypes("TensorShapeIter").define())
               .put(new Info("tensorflow::shape_inference::InferenceContext").purify())
               .put(new Info("std::vector<tensorflow::TensorShapeProto*>", "std::vector<const tensorflow::TensorShapeProto*>").cast().pointerTypes("TensorShapeProtoVector").define())
               .put(new Info("std::vector<std::unique_ptr<std::vector<tensorflow::shape_inference::ShapeAndType> > >",
                             "std::vector<std::unique_ptr<std::vector<std::pair<tensorflow::TensorShapeProto,tensorflow::DataType> > > >",
                             "std::vector<std::unique_ptr<std::vector<std::pair<tensorflow::PartialTensorShape,tensorflow::DataType> > > >").skip())
               .put(new Info("std::pair<tensorflow::shape_inference::ShapeHandle,tensorflow::shape_inference::ShapeHandle>").pointerTypes("ShapeHandlePair").define())
               .put(new Info("std::pair<tensorflow::shape_inference::DimensionHandle,tensorflow::shape_inference::DimensionHandle>").pointerTypes("DimensionHandlePair").define())
               .put(new Info("std::vector<tensorflow::Tensor>").pointerTypes("TensorVector").define())
               .put(new Info("std::vector<tensorflow::TensorProto>").pointerTypes("TensorProtoVector").define())
               .put(new Info("std::vector<tensorflow::TensorShape>").pointerTypes("TensorShapeVector").define())
               .put(new Info("std::vector<tensorflow::NodeBuilder::NodeOut>").pointerTypes("NodeOutVector").define())
               .put(new Info("std::vector<tensorflow::Node*>").pointerTypes("NodeVector").define())
               .put(new Info("std::vector<std::pair<tensorflow::Node*,int> >").pointerTypes("NodeIntPairVector").define())

               .put(new Info("tensorflow::tensor::internal::TensorProtoHelper",
                             "tensorflow::tensor::internal::TensorProtoFieldHelper",
                             "tensorflow::errors::internal::PrepareForStrCat",
                             "tensorflow::getTF_OutputDebugString",
                             "tensorflow::AttrValueMap::const_iterator",
                             "google::protobuf::Map<std::string,tensorflow::AttrValue>::const_iterator").skip())
               .put(new Info("google::protobuf::Map<std::string,tensorflow::AttrValue>",
                             "google::protobuf::Map<std::string,::tensorflow::AttrValue>",
                             "protobuf::Map<tensorflow::string,tensorflow::AttrValue>",
                             "tensorflow::protobuf::Map<tensorflow::string,tensorflow::AttrValue>").pointerTypes("StringAttrValueMap").define())
               .put(new Info("tensorflow::FunctionDefHelper::AttrValueWrapper").pointerTypes("FunctionDefHelper.AttrValueWrapper"))
               .put(new Info("std::vector<std::pair<tensorflow::string,tensorflow::FunctionDefHelper::AttrValueWrapper> >",
                             "tensorflow::gtl::ArraySlice<std::pair<tensorflow::string,tensorflow::FunctionDefHelper::AttrValueWrapper> >").cast().pointerTypes("StringAttrPairVector").define())
               .put(new Info("tensorflow::ops::NodeOut").valueTypes("@ByVal NodeBuilder.NodeOut", "Node"))
               .put(new Info("tensorflow::NodeBuilder::NodeOut").pointerTypes("NodeBuilder.NodeOut"))

               .put(new Info("std::function<void(std::function<void()>)>").cast().pointerTypes("Pointer"))
               .put(new Info("std::vector<tensorflow::ops::Input>::iterator").skip())
               .put(new Info("std::vector<tensorflow::ops::Input>::const_iterator").skip())
               .put(new Info("tensorflow::ops::Cast").pointerTypes("CastOp"))
               .put(new Info("tensorflow::ops::Const").pointerTypes("ConstOp"))
               .put(new Info("mode_t").skip())

               .put(new Info("tensorflow::gtl::ArraySlice<tensorflow::StringPiece>").cast().pointerTypes("StringPieceVector"))
               .put(new Info("tensorflow::gtl::ArraySlice<std::string>", "tensorflow::gtl::ArraySlice<tensorflow::string>",
                             "tensorflow::gtl::ArraySlice<tensorflow::tstring>").cast().pointerTypes("StringVector"))
               .put(new Info("tensorflow::gtl::ArraySlice<std::pair<tensorflow::string,tensorflow::string> >").cast().pointerTypes("StringStringPairVector"))
               .put(new Info("tensorflow::gtl::ArraySlice<tensorflow::Tensor>")/*.cast()*/.pointerTypes("TensorVector"))
               .put(new Info("tensorflow::gtl::ArraySlice<tensorflow::TensorProto>")/*.cast()*/.pointerTypes("TensorProtoVector"))
               .put(new Info("tensorflow::gtl::ArraySlice<tensorflow::TensorShape>").cast().pointerTypes("TensorShapeVector"))
               .put(new Info("tensorflow::gtl::ArraySlice<tensorflow::ops::NodeOut>")/*.cast()*/.pointerTypes("NodeOutVector"))
               .put(new Info("tensorflow::gtl::ArraySlice<tensorflow::Node*>")/*.cast()*/.pointerTypes("NodeVector"))
               .put(new Info("tensorflow::gtl::iterator_range<tensorflow::NeighborIter>").pointerTypes("NeighborIterRange").define())
               .put(new Info("tensorflow::gtl::iterator_range<tensorflow::NeighborIter>()").skip())
               .put(new Info("tensorflow::gtl::iterator_range<tensorflow::NodeIter>").pointerTypes("NodeIterRange").define())
               .put(new Info("tensorflow::gtl::iterator_range<tensorflow::NodeIter>()").skip())

//               .put(new Info("std::unordered_map<std::string,std::pair<int,int> >").pointerTypes("NameRangeMap").define())
               .put(new Info("tensorflow::gtl::FlatMap<tensorflow::StringPiece,std::pair<int,int>,tensorflow::hash<tensorflow::StringPiece> >").pointerTypes("NameRangeMap").define())
               .put(new Info("tensorflow::gtl::FlatMap<TF_Session*,tensorflow::string>").pointerTypes("TF_SessionStringMap").define())

                // Skip composite op scopes bc: call to implicitly-deleted default constructor of '::tensorflow::CompositeOpScopes'
               .put(new Info("tensorflow::CompositeOpScopes", "tensorflow::ExtendedInferenceContext").skip())

                // Fixed shape inference
               .put(new Info("std::vector<const tensorflow::Tensor*>").pointerTypes("ConstTensorPtrVector").define())
               .put(new Info("std::vector<const tensorflow::shape_inference::Dimension*>").pointerTypes("ConstDimensionPtrVector").define())

               .put(new Info("std::vector<std::pair<std::string,tensorflow::Tensor> >",
                             "std::vector<std::pair<tensorflow::string,tensorflow::Tensor> >").pointerTypes("StringTensorPairVector").define())
               .put(new Info("std::vector<tensorflow::Edge*>", "std::vector<const tensorflow::Edge*>").cast().pointerTypes("EdgeVector").define())
               .put(new Info("std::pair<tensorflow::EdgeSet::iterator,bool>").pointerTypes("EdgeSetBoolPair").define())
               .put(new Info("tensorflow::EdgeSet::const_iterator", "tensorflow::EdgeSet::iterator").pointerTypes("EdgeSetIterator"))
               .put(new Info("tensorflow::GraphEdgesIterable::const_iterator").purify())

               .put(new Info("tensorflow::register_op::OpDefBuilderWrapper<true>").pointerTypes("TrueOpDefBuilderWrapper"))
               .put(new Info("tensorflow::register_op::OpDefBuilderWrapper<false>").pointerTypes("FalseOpDefBuilderWrapper"))

               .put(new Info("tensorflow::checkpoint::TensorSliceSet::SliceInfo").pointerTypes("TensorSliceSet.SliceInfo"))
               .put(new Info("std::pair<tensorflow::string,int>").pointerTypes("StringIntPair").define())
               .put(new Info("std::pair<tensorflow::StringPiece,int>").pointerTypes("StringPieceIntPair").define())
               .put(new Info("std::pair<tensorflow::TensorSlice,tensorflow::string>").pointerTypes("TensorSlideStringPair").define())
               .put(new Info("std::pair<tensorflow::DataType,tensorflow::TensorShape>").pointerTypes("DataTypeTensorShapePair").define())
               .put(new Info("std::map<tensorflow::TensorId,tensorflow::TensorId>").pointerTypes("TensorIdTensorIdMap").define())
               .put(new Info("std::map<tensorflow::SafeTensorId,tensorflow::SafeTensorId>").pointerTypes("SafeTensorIdTensorIdMap").define())
               .put(new Info("std::unordered_map<std::string,tensorflow::TensorShape>").pointerTypes("VarToShapeMap").define())
               .put(new Info("std::unordered_map<std::string,tensorflow::DataType>").pointerTypes("VarToDataTypeMap").define())
               .put(new Info("std::unordered_map<tensorflow::string,tensorflow::checkpoint::TensorSliceSet*>").pointerTypes("StringTensorSliceSetMap").define())
               .put(new Info("std::unordered_map<tensorflow::string,tensorflow::Node*>").pointerTypes("StringNodeMap").define())
               .put(new Info("std::unordered_map<int,tensorflow::TensorShape>",
                             "std::unordered_map<int,TensorShape>").pointerTypes("IntTensorShapeMap").define())
               .put(new Info("std::unordered_map<int,std::pair<tensorflow::DataType,tensorflow::TensorShape> >",
                             "std::unordered_map<int,std::pair<DataType,TensorShape> >").pointerTypes("IntDataTypeTensorShapePairMap").define())
               .put(new Info("std::unordered_map<int,tensorflow::DtypeAndPartialTensorShape>").pointerTypes("DtypeAndPartialTensorShapeIntMap").define())
               .put(new Info("const std::unordered_map<tensorflow::string,tensorflow::checkpoint::TensorSliceSet::SliceInfo>").pointerTypes("StringSliceInfoMap").define())
               .put(new Info("std::vector<tensorflow::Input>::iterator", "std::vector<tensorflow::Input>::const_iterator").skip())
               .put(new Info("tensorflow::ImportGraphDefResults::Index").cast().valueTypes("int").pointerTypes("IntPointer"))
               .put(new Info("std::pair<tensorflow::Node*,tensorflow::ImportGraphDefResults::Index>").pointerTypes("NodeIndexPair").define())
               .put(new Info("TF_WhileParams").purify())
               .put(new Info("TF_LoadSessionFromSavedModel").annotations("@Platform(not=\"android\")").javaNames("TF_LoadSessionFromSavedModel"))
               .put(new Info("TF_GraphImportGraphDefOptionsRemapControlDependency").annotations("@Platform(not=\"android\")").javaNames("TF_GraphImportGraphDefOptionsRemapControlDependency"))
               .put(new Info("tensorflow::monitoring::GaugeCell<bool>").pointerTypes("BoolGaugeCell"))
               .put(new Info("tensorflow::monitoring::GaugeCell<tensorflow::int64>").pointerTypes("IntGaugeCell"))
               .put(new Info("tensorflow::monitoring::GaugeCell<tensorflow::string>").pointerTypes("StringGaugeCell"))
               .put(new Info("tensorflow::SavedModelBundle::session").javaText("public native @MemberGetter @UniquePtr Session session();"))

               .put(new Info("std::function<void()>").pointerTypes("Fn"))
               .put(new Info("std::function<void(int64,int64)>").pointerTypes("ForFn"))
               .put(new Info("std::function<void(int64,int64,int)>").pointerTypes("ParallelForFn"))
               .put(new Info("std::function<tensorflow::FileSystem*()>").pointerTypes("FactoryFn"))
               .put(new Info("std::function<bool(const KernelDef&)>").pointerTypes("KernelDefPredicateFn"))
               .put(new Info("std::function<tensorflow::uint64()>").cast().valueTypes("FreedByFunc").pointerTypes("Pointer"))
               .put(new Info("std::function<tensorflow::Rendezvous*(const int64)>").pointerTypes("RendezvousCreator"))
               .put(new Info("std::function<tensorflow::Status(const int64,const DeviceMgr*,Rendezvous**r)>").cast().valueTypes("RendezvousCreator").pointerTypes("Pointer"))
               .put(new Info("std::function<std::unique_ptr<tensorflow::monitoring::Buckets>(void)>").pointerTypes("CreateBuckets"))
               .put(new Info("tensorflow::OpKernelContext::Params::inc_num_deferred_ops_function")
                       .javaText("@MemberSetter public native Params inc_num_deferred_ops_function(@ByVal Fn fn);"))
               .put(new Info("tensorflow::OpKernelContext::Params::dec_num_deferred_ops_function")
                       .javaText("@MemberSetter public native Params dec_num_deferred_ops_function(@ByVal Fn fn);"))
               .put(new Info("tensorflow::FunctionLibraryRuntime::InstantiateOptions::optimize_graph_fn")
                       .javaText("@MemberSetter public native InstantiateOptions optimize_graph_fn(@ByVal OptimizeGraphFn optimize_graph_fn);"))
               .put(new Info("TFE_MonitoringBuckets::create_buckets")
                       .javaText("@MemberSetter public native TFE_MonitoringBuckets create_buckets(@ByVal CreateBuckets create_buckets);"))
               .put(new Info("tensorflow::OpRegistrationData::shape_inference_fn")
                       .javaText("@MemberSetter public native OpRegistrationData shape_inference_fn(@ByVal ShapeInferenceFn shape_inference_fn);"))
               .put(new Info("tensorflow::shape_inference::InferenceContext::Run")
                       .javaText("public native @ByVal Status Run(@ByVal ShapeInferenceFn fn);"))
               .put(new Info("tensorflow::ConstantFoldingOptions::consider")
                       .javaText("@MemberSetter public native ConstantFoldingOptions consider(@ByVal ConsiderFunction consider);"))
               .put(new Info("tensorflow::GraphConstructorOptions::cse_consider_function")
                       .javaText("@MemberSetter public native GraphConstructorOptions cse_consider_function(@ByVal ConsiderFunction cse_consider_function);"));

        String[] attrs = {"int", "long long", "float", "double", "bool", "std::string",
                          "tensorflow::Tensor", "tensorflow::TensorProto", "tensorflow::TensorShape",
                          "tensorflow::NameAttrList", "tensorflow::StringPiece"};
        for (int i = 0; i < attrs.length; i++) {
            if ("std::string".equals(attrs[i])) {
                infoMap.put(new Info("tensorflow::GraphDefBuilder::Options::WithAttr<" + attrs[i] + ">").javaText(
                        "public native @ByVal @Name(\"WithAttr<std::string>\") Options WithAttr(@StringPiece BytePointer attr_name, @StdString @Cast({\"char*\", \"std::string&&\"}) BytePointer value);\n"
                      + "public native @ByVal @Name(\"WithAttr<std::string>\") Options WithAttr(@StringPiece String attr_name, @StdString @Cast({\"char*\", \"std::string&&\"}) String value);"));
            } else if (!"tensorflow::StringPiece".equals(attrs[i])) {
                infoMap.put(new Info("tensorflow::GraphDefBuilder::Options::WithAttr<" + attrs[i] + ">").javaNames("WithAttr"));
            }
            if (i < attrs.length - 2) {
                infoMap.put(new Info("tensorflow::GraphDefBuilder::Options::WithAttr<tensorflow::gtl::ArraySlice<" + attrs[i] + "> >").javaNames("WithAttr"));
            }
        }

        infoMap.put(new Info("tensorflow::DotOptions::edge_label")
                       .javaText("@MemberSetter public native DotOptions edge_label(EdgeLabelFunction edge_label_function);"))
               .put(new Info("tensorflow::DotOptions::node_label")
                       .javaText("@MemberSetter public native DotOptions node_label(NodeLabelFunction node_label_function);"))
               .put(new Info("tensorflow::DotOptions::edge_cost")
                        .javaText("@MemberSetter public native DotOptions edge_cost(EdgeCostFunction edge_cost_function);"))
               .put(new Info("tensorflow::DotOptions::node_cost")
                       .javaText("@MemberSetter public native DotOptions node_cost(NodeCostFunction node_cost_function);"))
               .put(new Info("tensorflow::DotOptions::node_color")
                       .javaText("@MemberSetter public native DotOptions node_color(NodeColorFunction node_color_function);"))

               .put(new Info("std::function<double(const *tensorflow::Edge)>").pointerTypes("EdgeCostFunction"))
               .put(new Info("std::function<double(const *tensorflow::Node)>").pointerTypes("NodeCostFunction"))
               .put(new Info("std::function<std::string(const *tensorflow::Node)>").pointerTypes("NodeLabelFunction"))
               .put(new Info("std::function<std::string(const *tensorflow::Edge)>").pointerTypes("EdgeLabelFunction"))
               .put(new Info("std::function<int(const *tensorflow::Node)>").pointerTypes("NodeColorFunction"));

        infoMap.put(new Info("tensorflow::gtl::ArraySlice").annotations("@ArraySlice"))
               .put(new Info("tensorflow::StringPiece").annotations("@StringPiece").valueTypes("BytePointer", "String").pointerTypes("@Cast({\"char*\", \"StringPiece*\"}) BytePointer"))
               .put(new Info("tensorflow::Input::Initializer").pointerTypes("Input.Initializer").valueTypes("@Const @ByRef Input.Initializer",
                             "@ByRef Tensor", "byte", "short", "int", "long", "float", "double", "boolean", "@StdString String", "@StdString BytePointer"));

        infoMap.put(new Info("TF_Buffer::data").javaText("public native @Const Pointer data(); public native TF_Buffer data(Pointer data);"))
               .put(new Info("TF_Status").pointerTypes("TF_Status").base("org.bytedeco.tensorflow.AbstractTF_Status"))
               .put(new Info("TF_Buffer").pointerTypes("TF_Buffer").base("org.bytedeco.tensorflow.AbstractTF_Buffer"))
               .put(new Info("TF_Tensor").pointerTypes("TF_Tensor").base("org.bytedeco.tensorflow.AbstractTF_Tensor"))
               .put(new Info("TF_SessionOptions").pointerTypes("TF_SessionOptions").base("org.bytedeco.tensorflow.AbstractTF_SessionOptions"))
               .put(new Info("TF_Graph").pointerTypes("TF_Graph").base("org.bytedeco.tensorflow.AbstractTF_Graph"))
               .put(new Info("TF_Graph::graph").javaText("public native @MemberGetter @ByRef Graph graph();"))
               .put(new Info("TF_Graph::refiner").javaText("public native @MemberGetter @ByRef ShapeRefiner refiner();"))
               .put(new Info("TF_ImportGraphDefOptions").pointerTypes("TF_ImportGraphDefOptions").base("org.bytedeco.tensorflow.AbstractTF_ImportGraphDefOptions"))
               .put(new Info("TF_Operation", "TFE_MonitoringCounterCell", "TFE_MonitoringSamplerCell",
                             "TFE_MonitoringCounter0", "TFE_MonitoringCounter1", "TFE_MonitoringCounter2",
                             "TFE_MonitoringIntGaugeCell", "TFE_MonitoringStringGaugeCell", "TFE_MonitoringBoolGaugeCell",
                             "TFE_MonitoringIntGauge0", "TFE_MonitoringIntGauge1", "TFE_MonitoringIntGauge2",
                             "TFE_MonitoringStringGauge0", "TFE_MonitoringStringGauge1", "TFE_MonitoringStringGauge2",
                             "TFE_MonitoringBoolGauge0", "TFE_MonitoringBoolGauge1", "TFE_MonitoringBoolGauge2",
                             "TFE_MonitoringSampler0", "TFE_MonitoringSampler1", "TFE_MonitoringSampler2").purify())
               .put(new Info("TFE_MonitoringCounter<0>", "TFE_MonitoringCounter<1>", "TFE_MonitoringCounter<2>",
                             "TFE_MonitoringGauge<bool,0>", "TFE_MonitoringGauge<bool,1>", "TFE_MonitoringGauge<bool,2>",
                             "TFE_MonitoringGauge<tensorflow::int64,0>", "TFE_MonitoringGauge<tensorflow::int64,1>", "TFE_MonitoringGauge<tensorflow::int64,2>",
                             "TFE_MonitoringGauge<tensorflow::string,0>", "TFE_MonitoringGauge<tensorflow::string,1>", "TFE_MonitoringGauge<tensorflow::string,2>",
                             "TFE_MonitoringSampler<0>", "TFE_MonitoringSampler<1>", "TFE_MonitoringSampler<2>").pointerTypes("Pointer"))
               .put(new Info("TF_Operation::node").javaText("public native @MemberGetter @ByRef Node node();"))
               .put(new Info("TFE_MonitoringCounterCell::cell").javaText("public native @MemberGetter @ByRef CounterCell cell();"))
               .put(new Info("TFE_MonitoringSamplerCell::cell").javaText("public native @MemberGetter @ByRef SamplerCell cell();"))
               .put(new Info("TFE_MonitoringIntGaugeCell::cell").javaText("public native @MemberGetter @ByRef IntGaugeCell cell();"))
               .put(new Info("TFE_MonitoringStringGaugeCell::cell").javaText("public native @MemberGetter @ByRef StringGaugeCell cell();"))
               .put(new Info("TFE_MonitoringBoolGaugeCell::cell").javaText("public native @MemberGetter @ByRef BoolGaugeCell cell();"))
               .put(new Info("TFE_CancellationManager::cancellation_manager").javaText("public native @MemberGetter @ByRef CancellationManager cancellation_manager();"))
               .put(new Info("TFE_ContextMirroringPolicy").cast().pointerTypes("Pointer"))
               .put(new Info("TF_ShapeInferenceContextDimValueKnown").skip())
               .put(new Info("TF_Session").pointerTypes("TF_Session").base("org.bytedeco.tensorflow.AbstractTF_Session"))
               .put(new Info("TF_Session::extend_before_run").javaText("public native @MemberGetter @ByRef @Cast(\"std::atomic<bool>*\") Pointer extend_before_run();"));

        infoMap.put(new Info("tensorflow::Scope::WithOpName<std::string>").javaNames("WithOpName").javaText(
                        "public native @ByVal Scope WithOpName(@StdString BytePointer op_name);\n"
                      + "public native @ByVal Scope WithOpName(@StdString String op_name);"));

        infoMap.put(new Info("std::vector<tensorflow::OpDef>").pointerTypes("OpDefVector").define());
        if (!android) {
            infoMap.put(new Info("std::vector<tensorflow::Output>").pointerTypes("OutputVector").define())
                   .put(new Info("tensorflow::gtl::InlinedVector<tensorflow::TensorHandle*,4>").pointerTypes("TensorHandleVector").define())
                   .put(new Info("std::vector<tensorflow::CollectiveImplementationInterface*>").pointerTypes("CollectiveImplementationVector").define())
                   .put(new Info("tensorflow::gtl::FlatMap<tensorflow::string,tensorflow::Device*,StringPieceHasher>").pointerTypes("DeviceMap").define())
                   .put(new Info("tensorflow::gtl::FlatMap<tensorflow::string,tensorflow::uint64>").pointerTypes("RemoteContexts").define())
                   .put(new Info("tensorflow::gtl::FlatSet<std::string>").pointerTypes("StringFlatSet").define())
                   .put(new Info("tensorflow::EagerContext::device_map").javaText("public native DeviceMap device_map();"))
                   .put(new Info("tensorflow::eager::Operation").pointerTypes("Eager_Operation"))
                   .put(new Info("std::map<tensorflow::TensorHandle*,TF_Output>").pointerTypes("TF_OutputTensorHandleMap").define())
                   .put(new Info("std::vector<std::pair<tensorflow::TensorHandle*,TF_Output> >").pointerTypes("TensorHandleTF_OutputPairVector").define())
//                   .put(new Info("TFE_Context::context").javaText("@MemberGetter public native @ByRef EagerContext context();"))
                   .put(new Info("TFE_Op::operation").javaText("@MemberGetter public native @ByRef EagerOperation operation();"));
        }

        String[] consts = {"unsigned char", "short", "int", "long long", "float", "double", "bool", "std::string", "tensorflow::StringPiece"};
        for (int i = 0; i < consts.length; i++) {
            infoMap.put(new Info("tensorflow::ops::Const<" + consts[i] + ">").javaNames("Const"));
        }
    }

    public static class Fn extends FunctionPointer {
        static { Loader.load(); }
        /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
        public    Fn(Pointer p) { super(p); }
        protected Fn() { allocate(); }
        private native void allocate();
        public native void call();
    }

    public static class ForFn extends FunctionPointer {
        static { Loader.load(); }
        /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
        public    ForFn(Pointer p) { super(p); }
        protected ForFn() { allocate(); }
        private native void allocate();
        public native void call(long from, long to);
    }

    public static class ParallelForFn extends FunctionPointer {
        static { Loader.load(); }
        /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
        public    ParallelForFn(Pointer p) { super(p); }
        protected ParallelForFn() { allocate(); }
        private native void allocate();
        public native int call(long from, long to, int i);
    }

    public static class ConsiderFunction extends FunctionPointer {
        static { Loader.load(); }
        /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
        public    ConsiderFunction(Pointer p) { super(p); }
        protected ConsiderFunction() { allocate(); }
        private native void allocate();
        public native @Cast("bool") boolean call(@Cast("const tensorflow::Node*") Pointer node);
    }

    public static class NodeColorFunction extends FunctionPointer {
        static { Loader.load(); }
        /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
        public    NodeColorFunction(Pointer p) { super(p); }
        protected NodeColorFunction() { allocate(); }
        private native void allocate();
        public native @Cast("int") int call(@Cast("const tensorflow::Node*") Pointer node);
    }

    public static class NodeCostFunction extends FunctionPointer {
        static { Loader.load(); }
        /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
        public    NodeCostFunction(Pointer p) { super(p); }
        protected NodeCostFunction() { allocate(); }
        private native void allocate();
        public native @Cast("double") double call(@Cast("const tensorflow::Node*") Pointer node);
    }

   public static class EdgeCostFunction extends FunctionPointer {
        static { Loader.load(); }
        /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
        public    EdgeCostFunction(Pointer p) { super(p); }
        protected EdgeCostFunction() { allocate(); }
        private native void allocate();
        public native @Cast("double") double call(@Cast("const tensorflow::Edge*") Pointer node);
    }

    public static class NodeLabelFunction extends FunctionPointer {
        static { Loader.load(); }
        /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
        public    NodeLabelFunction(Pointer p) { super(p); }
        protected NodeLabelFunction() { allocate(); }
        private native void allocate();
        public native @StdString BytePointer call(@Cast("const tensorflow::Node*") Pointer node);
    }

    public static class EdgeLabelFunction extends FunctionPointer {
        static { Loader.load(); }
        /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
        public    EdgeLabelFunction(Pointer p) { super(p); }
        protected EdgeLabelFunction() { allocate(); }
        private native void allocate();
        public native @StdString BytePointer call(@Cast("const tensorflow::Edge*") Pointer node);
    }

    public static class FactoryFn extends FunctionPointer {
        static { Loader.load(); }
        /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
        public    FactoryFn(Pointer p) { super(p); }
        protected FactoryFn() { allocate(); }
        private native void allocate();
        public native @Cast("tensorflow::FileSystem*") Pointer call();
    }

    public static class FreedByFunc extends FunctionPointer {
        static { Loader.load(); }
        /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
        public    FreedByFunc(Pointer p) { super(p); }
        protected FreedByFunc() { allocate(); }
        private native void allocate();
        public native @Cast("tensorflow::uint64") long call();
    }

    public static class CreateRendezvous extends FunctionPointer {
        static { Loader.load(); }
        /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
        public    CreateRendezvous(Pointer p) { super(p); }
        protected CreateRendezvous() { allocate(); }
        private native void allocate();
        public native @ByVal @Cast("tensorflow::Status*") Pointer call(@Cast("const int64") long i,
                @Cast("const tensorflow::DeviceMgr*") Pointer device, @Cast("tensorflow::Rendezvous**") PointerPointer r);
    }

    public static class RendezvousCreator extends FunctionPointer {
        static { Loader.load(); }
        /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
        public    RendezvousCreator(Pointer p) { super(p); }
        protected RendezvousCreator() { allocate(); }
        private native void allocate();
        public native @Cast("tensorflow::Rendezvous*") Pointer call(@Cast("const int64") long i);
    }

    public static class OptimizeGraphFn extends FunctionPointer {
        static { Loader.load(); }
        /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
        public    OptimizeGraphFn(Pointer p) { super(p); }
        protected OptimizeGraphFn() { allocate(); }
        private native void allocate();
        public native @ByVal @Cast("tensorflow::Status*") Pointer call(@ByVal @Cast("std::vector<string>*") Pointer v, @ByVal @Cast("std::vector<string>*") Pointer v2,
                @Cast("tensorflow::FunctionLibraryDefinition*") Pointer def, @ByRef @Cast("const tensorflow::DeviceSet*") Pointer s, @Cast("tensorflow::Device*") Pointer dev, 
                @Cast("std::unique_ptr<tensorflow::Graph>*") Pointer g);
    }

    public static class CreateBuckets extends FunctionPointer {
        static { Loader.load(); }
        /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
        public    CreateBuckets(Pointer p) { super(p); }
        protected CreateBuckets() { allocate(); }
        private native void allocate();
        public native @MoveUniquePtr @Cast("tensorflow::monitoring::Buckets*") Pointer call();
    }

    public static class KernelDefPredicateFn extends FunctionPointer {
        static { Loader.load(); }
        /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
        public    KernelDefPredicateFn(Pointer p) { super(p); }
        protected KernelDefPredicateFn() { allocate(); }
        private native void allocate();
        public native @Cast("bool") boolean call(@ByRef @Cast("const tensorflow::KernelDef*") Pointer kernelDef);
    }

    public static class ShapeInferenceFn extends FunctionPointer {
        static { Loader.load(); }
        /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
        public    ShapeInferenceFn(Pointer p) { super(p); }
        protected ShapeInferenceFn() { allocate(); }
        private native void allocate();
        public native @ByVal @Cast("tensorflow::Status*") Pointer call(@Cast("shape_inference::InferenceContext*") Pointer node);
    }

    @Documented @Retention(RetentionPolicy.RUNTIME)
    @Target({ElementType.METHOD, ElementType.PARAMETER})
    @Cast({"tensorflow::gtl::ArraySlice", "&"}) @Adapter("ArraySliceAdapter")
    public @interface ArraySlice { String value() default ""; }

    @Documented @Retention(RetentionPolicy.RUNTIME)
    @Target({ElementType.METHOD, ElementType.PARAMETER})
    @Cast("tensorflow::StringPiece&") @Adapter("StringPieceAdapter")
    public @interface StringPiece { }

    @Documented @Retention(RetentionPolicy.RUNTIME)
    @Target({ElementType.METHOD, ElementType.PARAMETER})
    @Cast({"std::unique_ptr", "&&"}) @Adapter("UniquePtrAdapter")
    public @interface MoveUniquePtr {
        String value() default "";
    }
}
