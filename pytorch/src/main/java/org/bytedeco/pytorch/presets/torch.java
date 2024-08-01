/*
 * Copyright (C) 2020-2023 Hervé Guillemet, Samuel Audet, Eduardo Gonzalez
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

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import org.bytedeco.javacpp.ClassProperties;
import org.bytedeco.javacpp.LoadEnabled;
import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.Pointer;

import org.bytedeco.javacpp.annotation.ByRef;
import org.bytedeco.javacpp.annotation.Cast;
import org.bytedeco.javacpp.annotation.MemberGetter;
import org.bytedeco.javacpp.annotation.Namespace;
import org.bytedeco.javacpp.annotation.Platform;
import org.bytedeco.javacpp.annotation.Properties;

import org.bytedeco.javacpp.tools.BuildEnabled;
import org.bytedeco.javacpp.tools.Info;
import org.bytedeco.javacpp.tools.InfoMap;
import org.bytedeco.javacpp.tools.InfoMapper;
import org.bytedeco.javacpp.tools.Logger;

import org.bytedeco.openblas.presets.openblas;

/**
 * @author Samuel Audet, Hervé Guillemet
 */
@Properties(
    inherit = openblas.class,
    value = {
        @Platform(
            value = {"linux", "macosx", "windows"},
            compiler = "cpp17",
	        // __WINSOCKAPI_ fixes compilation error on windows due to
	        // inclusion of both V1 and V2 of winsock API.
            define = {"SHARED_PTR_NAMESPACE std", "UNIQUE_PTR_NAMESPACE std", "USE_C10D_GLOO", "_WINSOCKAPI_"},
            include = {
                "torch/torch.h",
                "torch/script.h",
                "torch/csrc/inductor/aoti_runner/model_container_runner_cpu.h",
                "torch/csrc/distributed/c10d/ProcessGroupGloo.hpp",
                "torch/csrc/distributed/c10d/PrefixStore.hpp",
                "torch/csrc/distributed/c10d/logger.hpp",

                // For inclusion in JNI only, not parsed (compiler needs some complete definitions)
                "torch/csrc/jit/runtime/instruction.h",
                "torch/csrc/jit/serialization/source_range_serialization.h",
                "torch/csrc/jit/frontend/resolver.h",
                "torch/csrc/jit/frontend/tree_views.h",
                "torch/csrc/jit/serialization/storage_context.h",

                "datasets.h",
                "pytorch_adapters.h",

		        // Fix link error on Windows:
		        "gloo/common/logging.cc",

                // Fix compilation error on MacOS-12:
                "<unordered_map>"

            },
            exclude = {"openblas_config.h", "cblas.h", "lapacke_config.h", "lapacke_mangling.h", "lapack.h", "lapacke.h", "lapacke_utils.h"},
            preload = { "asmjit", "fbgemm" }
        ),
        @Platform(
            value = {"linux", "macosx", "windows"},
            includepath = {"/usr/local/cuda/include", "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.3/include/"},
            preloadpath = {
                "/usr/local/cuda-12.3/lib64/",
                "/usr/local/cuda-12.3/extras/CUPTI/lib64/",
                "/usr/local/cuda/lib64/",
                "/usr/local/cuda/extras/CUPTI/lib64/",
                "/usr/lib64/",
                "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.3/lib/x64/",
                "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.3/extras/CUPTI/lib64/",
                "C:/Program Files/NVIDIA Corporation/NvToolsExt/bin/x64/",
            },
            linkpath = {
                "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.3/lib/x64/",
                "/usr/local/cuda-12.3/lib64/",
                "/usr/local/cuda/lib64/",
                "/usr/lib64/"
            },
            extension = "-gpu"
        ),
        @Platform(
            value = {"linux"},
            link = { "c10", "torch", "torch_cpu" }
        ),
        @Platform(
            value = {"macosx"},
            link = { "c10", "torch", "torch_cpu", "omp" }
        ),
        @Platform(
            value = "windows",
            link = { "c10", "torch", "torch_cpu", "uv" }
        ),
        @Platform(
            value = "linux",
            extension = "-gpu",
            link = { "c10", "torch", "torch_cpu", "c10_cuda", "torch_cuda", "cudart", "cusparse", "cudnn" } // cupti@.12 needed ?
        ),
        @Platform(
            value = "windows",
            extension = "-gpu",
            link = { "c10", "torch", "torch_cpu", "uv", "c10_cuda", "torch_cuda", "cudart", "cusparse", "cudnn" }
        )
    },
    target = "org.bytedeco.pytorch",
    global = "org.bytedeco.pytorch.global.torch"
)
public class torch implements LoadEnabled, InfoMapper, BuildEnabled {
    static {
        Loader.checkVersion("org.bytedeco", "pytorch");
    }

    static void initIncludes(Class thisClass, ClassProperties properties) {
        // If we are called from Parser, fetch the list of headers to parse from resources.
        // This check for stack depth 5 also excludes the code path where, because of property inheritance,
        // we are called from torch class while processing torch_cuda. Parser stack depth is 6 in that code path.
        if (Loader.getCallerClass(5).getName().equals("org.bytedeco.javacpp.tools.Parser")) {
            properties.put("platform.include", new ArrayList<String>());
            Class presets = properties.getEffectiveClasses().get(0);
            InputStream includesStream = thisClass.getResourceAsStream(presets.getSimpleName() + "_include.h");
            if (includesStream == null) {
                throw new RuntimeException("Cannot find parse list for " + presets);
            }
            Pattern re = Pattern.compile("^#include\\s+[\"<]([^\">]+)[\">]");
            try (BufferedReader br = new BufferedReader(new InputStreamReader(includesStream))) {
                String line;
                while ((line = br.readLine()) != null) {
                    Matcher m = re.matcher(line);
                    if (m.find())
                        properties.addAll("platform.include", m.group(1));
                }
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        }
    }

    private boolean arm64;

    @Override
    public void init(ClassProperties properties) {
        String platform = properties.getProperty("platform");
        String extension = properties.getProperty("platform.extension");
        List<String> preloads = properties.get("platform.preload");
        List<String> resources = properties.get("platform.preloadresource");

        initIncludes(getClass(), properties);

        // Only apply this at load time since we don't want to copy the CUDA libraries here
        if (!Loader.isLoadLibraries() || extension == null || !extension.endsWith("-gpu")) {
            return;
        }
        int i = 0;
        if (platform.startsWith("windows")) {
            preloads.add(i++, "zlibwapi");
        }
        String[] libs = {"cudart", "cublasLt", "cublas", "cufft", "curand", "nvJitLink", "cusparse", "cusolver",
            "cudnn", "nccl", "nvrtc", "nvrtc-builtins", "myelin", "nvinfer", "cudnn_ops_infer", "cudnn_ops_train",
            "cudnn_adv_infer", "cudnn_adv_train", "cudnn_cnn_infer", "cudnn_cnn_train"};
        for (String lib : libs) {
            if (platform.startsWith("linux")) {
                lib += lib.startsWith("cudnn") ? "@.8"
                    : lib.equals("nccl") ? "@.2"
                    : lib.equals("myelin") ? "@.1"
                    : lib.equals("nvinfer") ? "@.8"
                    : lib.equals("cufft") ? "@.11"
                    : lib.equals("curand") ? "@.10"
                    : lib.equals("cusolver") ? "@.11"
                    : lib.equals("nvrtc-builtins") ? "@.12.3"
                    : "@.12";
            } else if (platform.startsWith("windows")) {
                lib += lib.startsWith("cudnn") ? "64_8"
                    : lib.equals("nccl") ? "64_2"
                    : lib.equals("myelin") ? "64_1"
                    : lib.equals("nvinfer") ? "64_8"
                    : lib.equals("cufft") ? "64_11"
                    : lib.equals("curand") ? "64_10"
                    : lib.equals("cusolver") ? "64_11"
                    : lib.equals("nvrtc") ? "64_120_0"
                    : lib.equals("nvrtc-builtins") ? "64_123"
                    : lib.equals("nvJitLink") ? "_120_0"
                    : "64_12";
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

    @Override
    public void init(Logger logger, java.util.Properties properties, String encoding) {
        arm64 = properties.getProperty("platform").contains("arm64");
    }

    public void mapModule(InfoMap infoMap, String name) {
        mapModule(infoMap, name, null, null, true);
    }

    public void mapModule(InfoMap infoMap, String name, boolean anyModuleCompatible) {
        mapModule(infoMap, name, null, null, anyModuleCompatible);
    }

    public void mapModule(InfoMap infoMap, String name, String base) {
        mapModule(infoMap, name, base, null, true);
    }

    public void mapModule(InfoMap infoMap, String name, String base, String baseBase) {
        mapModule(infoMap, name, base, baseBase, true);
    }

    public void mapModule(InfoMap infoMap, String name, String base, String baseBase, boolean anyModuleCompatible) {
        if (baseBase != null) {
            infoMap.put(new Info(baseBase).pointerTypes(name + "ImplBaseBase"));
        }

        if (base != null) {
            infoMap.put(new Info(base).pointerTypes(name + "ImplBase"));
        }

        infoMap.put(new Info("torch::nn::" + name + "Impl")) // Ensure qualified name is in Info when Cloneable<XImpl> inheritance is parsed (and before class XImpl is finished parsing)
               .put(new Info("torch::nn::" + name + "Impl::" + name + "Impl").annotations("@SharedPtr", "@Name(\"std::make_shared<torch::nn::" + name + "Impl>\")"))
               .put(new Info("torch::nn::Cloneable<torch::nn::" + name + "Impl>").pointerTypes(name + "ImplCloneable").purify())
               .put(new Info("torch::nn::ModuleHolder<torch::nn::" + name + "Impl>").skip())
               .put(new Info("torch::nn::" + name).skip())
               .put(new Info("torch::nn::Module::as<torch::nn::" + name + "Impl,int>").javaNames("as" + name));

        if (anyModuleCompatible) {
            infoMap
                // Parser queries parameter as ModuleType* instead of std::shared_ptr<ModuleType>
                // First cppName is to answer template query, second one to generate instance
                .put(new Info(
                    "torch::nn::AnyModule::AnyModule<torch::nn::" + name + "Impl>(ModuleType*)",
                    "torch::nn::AnyModule::AnyModule<torch::nn::" + name + "Impl>(torch::nn::" + name + "Impl*)"
                ).define().javaText(
                    "public AnyModule(" + name + "Impl module) { super((Pointer)null); allocate(module); }\n" +
                    // We need a @Cast because AnyModule constructor is explicit
                    "private native void allocate(@SharedPtr @Cast({\"\", \"std::shared_ptr<torch::nn::" + name + "Impl>\"}) " + name + "Impl module);\n"))
                .put(new Info("torch::nn::SequentialImpl::push_back<torch::nn::" + name + "Impl>").javaNames("push_back"))
            ;
        }
    }

    public static void sharedMap(InfoMap infoMap) {
        infoMap
            .put(new Info().enumerate().friendly())
            .put(new Info("auto", "c10::reverse_iterator", "ska::flat_hash_map", /*"std::atomic", */"std::conditional", "std::iterator_traits",
                "std::initializer_list", "std::integral_constant", "std::mutex", "std::reverse_iterator" /*, "std::weak_ptr"*/).skip())
            .put(new Info("basic/containers").cppTypes("torch::optional"))
        ;

        //// Macros
        infoMap
            .put(new Info("TORCH_API", "C10_API", "TORCH_XPU_API", "C10_EXPORT", "C10_HIDDEN", "C10_IMPORT", "C10_API_ENUM", "EXPORT_IF_NOT_GCC",
                "TORCH_CUDA_CU_API", "TORCH_CUDA_CPP_API", "TORCH_HIP_API", "TORCH_PYTHON_API",
                "__ubsan_ignore_float_divide_by_zero__", "__ubsan_ignore_undefined__", "__ubsan_ignore_signed_int_overflow__", "__ubsan_ignore_function__",
                "C10_CLANG_DIAGNOSTIC_IGNORE", "C10_CLANG_DIAGNOSTIC_PUSH", "C10_CLANG_DIAGNOSTIC_POP", "C10_ATTR_VISIBILITY_HIDDEN", "C10_ERASE",
                "C10_UID", "C10_NODISCARD", "C10_UNUSED", "C10_USED", "C10_RESTRICT", "C10_NOINLINE", "C10_ALWAYS_INLINE", "C10_FALLTHROUGH",
                "C10_HOST_DEVICE", "C10_DEVICE", "C10_HOST", "C10_LAUNCH_BOUNDS_0", "C10_HIP_HOST_DEVICE", "C10_WARP_SIZE", "C10_IOS", "C10_MOBILE",
                "C10_HOST_CONSTEXPR", "CONSTEXPR_EXCEPT_WIN_CUDA", "C10_HOST_CONSTEXPR_EXCEPT_WIN_CUDA", "C10_ALWAYS_INLINE_UNLESS_MOBILE",
                "alignas", "COMPLEX_INTEGER_OP_TEMPLATE_CONDITION", "C10_DEVICE_HOST_FUNCTION", "FORCE_INLINE_APPLE",
                "ERROR_UNSUPPORTED_CAST", "LEGACY_CONTIGUOUS_MEMORY_FORMAT", "GFLAGS_DLL_DEFINE_FLAG", "GFLAGS_DLL_DECLARE_FLAG",
                "AT_X", "DEFINE_KEY", "C10_DISPATCHER_INLINE_UNLESS_MOBILE", "TH_DISALLOW_COPY_AND_ASSIGN", "__device__",
                "__inline__",
                "TORCH_DSA_KERNEL_ARGS", "TORCH_DSA_KERNEL_ARGS_PASS",
                "C10_CUDA_API", "C10_CUDA_IMPORT", "C10_CUDA_EXPORT",
                "__ubsan_ignore_float_divide_by_zero__", "__ubsan_ignore_undefined__",
                "__ubsan_ignore_signed_int_overflow__", "__ubsan_ignore_pointer_overflow__",
                "__ubsan_ignore_function__").cppTypes().annotations())

            .put(new Info("defined(__CUDACC__) || defined(__HIPCC__)",
                "defined(__CUDACC__) && !defined(USE_ROCM)",
                "defined(SYCL_EXT_ONEAPI_BFLOAT16_MATH_FUNCTIONS)",
                "defined(_MSC_VER) && _MSC_VER <= 1900",
                "defined(NDEBUG)",
                "defined(__ANDROID__)",
                "defined(__APPLE__)",
                "defined(__aarch64__) && !defined(C10_MOBILE) && !defined(__CUDACC__)",
                "defined(__HIP_PLATFORM_HCC__)",
                "defined(_MSC_VER)", "_WIN32",
                "defined(USE_ROCM)", "USE_ROCM", "SYCL_LANGUAGE_VERSION",
                "defined(CUDA_VERSION) && CUDA_VERSION >= 11000",
                "defined ENABLE_RECORD_KERNEL_FUNCTION_DTYPE",
                "__OBJC__").define(false))

            .put(new Info("C10_DEFINE_DEPRECATED_USING").cppText("#define C10_DEFINE_DEPRECATED_USING(TypeName, TypeThingy)").cppTypes())
            .put(new Info("C10_DEPRECATED_MESSAGE").cppText("#define C10_DEPRECATED_MESSAGE() deprecated").cppTypes())
            .put(new Info("C10_DEPRECATED").cppText("#define C10_DEPRECATED deprecated").cppTypes())
            .put(new Info("deprecated").annotations("@Deprecated"))

            .put(new Info("CAFFE2_LOG_THRESHOLD").translate(false))

            .put(new Info("DOXYGEN_SHOULD_SKIP_THIS").define()) // Exclude what the devs decide to not be part of public API

            .put(new Info("TORCH_CHECK").cppText("#define TORCH_CHECK(cond, ...)").define())
            .put(new Info("DEFINE_SYMBOL").cppText("#define DEFINE_SYMBOL(ns, s) namespace ns { constexpr Symbol s; }").define())
            .put(new Info("TORCH_ENUM_DECLARE").cppText("#define TORCH_ENUM_DECLARE(name) namespace torch { namespace enumtype { struct k##name { k##name() {} }; } }").define())
        ;
    }

    public void map(InfoMap infoMap) {
        sharedMap(infoMap);

        infoMap
            .put(new Info("ordered_dict.h").linePatterns(".*class Item;.*").skip())
            .put(new Info("util.h").linePatterns(".*using approx_time_t = decltype.*").skip())

            .put(new Info().javaText("import org.bytedeco.pytorch.Allocator;"))
            .put(new Info().javaText("import org.bytedeco.pytorch.Function;"))
            .put(new Info().javaText("import org.bytedeco.pytorch.chrono.*;"))
            .put(new Info().javaText("import org.bytedeco.pytorch.Module;"))
            .put(new Info().javaText("import org.bytedeco.javacpp.annotation.Cast;"))
            .put(new Info().javaText("import org.bytedeco.pytorch.helper.*;"))

            .put(new Info("std::nullptr_t").cast().pointerTypes("PointerPointer"))

            .put(new Info("at::CheckedFrom").cast().valueTypes("BytePointer", "String").pointerTypes("PointerPointer")) // Alias to const char*
            .put(new Info("c10::IValue", "at::IValue", "decltype(auto)").pointerTypes("IValue"))
            //             .put(new Info("c10::IValue::operator ==").skip()) // Possible name conflict with IValue.equals
            .put(new Info(
                "std::size_t",
                "c10::Dict<c10::IValue,c10::IValue>::size_type",
                "c10::Dict<std::string,c10::impl::GenericList>::size_type",
                "c10::Dict<torch::Tensor,torch::Tensor>::size_type"
            ).cast().valueTypes("long").pointerTypes("SizeTPointer"))
            .put(new Info("c10::approx_time_t").cast().valueTypes("long").pointerTypes("LongPointer"))
            .put(new Info("c10::ClassType::Property").pointerTypes("ClassType.Property"))

            .put(new Info("at::RecordFunctionHandle").valueTypes("long"))
            .put(new Info("operator const std::string&()").javaText( // Hopefully targets the one in ConstantString only
                "public native @Const @ByRef @Name(\"operator const std::string&\") @StdString @Override String toString();"
            ))
            .put(new Info("strong::type<int64_t,_VulkanID,strong::regular,strong::convertible_to<int64_t>,strong::hashable>").pointerTypes("Pointer"))
            .put(new Info("fbgemm::bfloat16", "__nv_bfloat16", "sycl::ext::oneapi::bfloat16").pointerTypes("BFloat16").valueTypes("short", "short", "short"))
            .put(new Info("decltype(::c10::impl::ScalarTypeToCPPType<::c10::ScalarType::Bool>::t)").cast().valueTypes("boolean").pointerTypes("BoolPointer"))
            .put(new Info("decltype(::c10::impl::ScalarTypeToCPPType<::c10::ScalarType::Half>::t)").pointerTypes("Half"))
            .put(new Info("decltype(::c10::impl::ScalarTypeToCPPType<::c10::ScalarType::BFloat16>::t)").pointerTypes("BFloat16"))
            .put(new Info("decltype(::c10::impl::ScalarTypeToCPPType<::c10::ScalarType::Float8_e5m2>::t)").pointerTypes("Float8_e5m2"))
            .put(new Info("decltype(::c10::impl::ScalarTypeToCPPType<::c10::ScalarType::Float8_e4m3fn>::t)").pointerTypes("Float8_e4m3fn"))
            .put(new Info("decltype(::c10::impl::ScalarTypeToCPPType<::c10::ScalarType::Float8_e5m2fnuz>::t)").pointerTypes("Float8_e5m2fnuz"))
            .put(new Info("decltype(::c10::impl::ScalarTypeToCPPType<::c10::ScalarType::Float8_e4m3fnuz>::t)").pointerTypes("Float8_e4m3fnuz"))
            .put(new Info("c10::ClassType").purify().pointerTypes("ClassType")) // Issue #669
            .put(new Info("c10::EnumType").purify().pointerTypes("EnumType")) // Issue #669
            .put(new Info("c10::NamedType").purify().pointerTypes("NamedType")) // Issue #669
            .put(new Info("at::namedinference::TensorName").pointerTypes("TensorName"))
            .put(new Info("c10::remove_symint<c10::SymInt>::type").valueTypes("long"))
            .put(new Info("std::aligned_storage_t<sizeof(IValue),alignof(IValue)>").pointerTypes("Pointer"))
            .put(new Info("c10::requires_grad", "at::range", "at::bernoulli_out", "at::normal_out", "at::stft").skipDefaults())
            .put(new Info("c10::prim::requires_grad").javaNames("requires_grad"))
            .put(new Info("c10::aten::clone").javaNames("_clone"))
            .put(new Info("at::TensorBase").base("AbstractTensor").pointerTypes("TensorBase"))
            .put(new Info("torch::autograd::Variable").pointerTypes("Tensor"))
        ;

        // c10::DataPtr
        // DataPtr::operator= is deleted.
        // So we must move all DataPtr passed by value.
        // UniqueStorageShareExternalPointer has 2 overloads, 1 taking at::DataPtr&& and the other taking void *. StdMove adapter
        // can be cast to both, so we need to disambiguate with a @Cast.
        // No way to define a function pointer taking a moved DataPtr, so skipping GetStorageImplCreate and SetStorageImplCreate,
        // and the automatic generation of the FunctionPointer.
        infoMap
            //.put(new Info("c10::DataPtr&&", "at::DataPtr&&").valueTypes("@Cast({\"\", \"c10::DataPtr&&\"}) @StdMove DataPtr").pointerTypes("DataPtr")) // DataPtr::operator= deleted
            .put(new Info("c10::DataPtr", "at::DataPtr").valueTypes("@StdMove DataPtr").pointerTypes("DataPtr"))
            .put(new Info("c10::StorageImpl::UniqueStorageShareExternalPointer(at::DataPtr&&, size_t)",
                "c10::Storage::UniqueStorageShareExternalPointer(at::DataPtr&&, size_t)").javaText(
                "public native void UniqueStorageShareExternalPointer(@Cast({\"\", \"c10::DataPtr&&\"}) @StdMove DataPtr data_ptr,  @Cast(\"size_t\") long size_bytes);"
            ))
            .put(new Info("c10::GetStorageImplCreate", "c10::SetStorageImplCreate",
                "c10::intrusive_ptr<c10::StorageImpl> (*)(c10::StorageImpl::use_byte_size_t, c10::SymInt, c10::DataPtr, c10::Allocator*, bool)").skip())
        ;
        //// Enumerations
        infoMap
            .put(new Info("c10::ScalarType", "at::ScalarType", "torch::Dtype").enumerate().valueTypes("ScalarType").pointerTypes("@Cast(\"c10::ScalarType*\") BytePointer"))
            .put(new Info("torch::jit::AttributeKind").enumerate().valueTypes("JitAttributeKind"))
            .put(new Info("torch::jit::PickleOpCode").enumerate().translate(false).valueTypes("PickleOpCode"))
        ;

        //// std::optional
        infoMap
            .put(new Info("std::optional<bool>").pointerTypes("BoolOptional").define())
            .put(new Info("std::optional<int8_t>", "std::optional<c10::DeviceIndex>").pointerTypes("ByteOptional").define())
            .put(new Info("std::optional<int>", "std::optional<int32_t>").pointerTypes("IntOptional").define())
            .put(new Info("std::optional<int64_t>", "c10::remove_symint<std::optional<c10::SymInt> >::type").pointerTypes("LongOptional").define())
            .put(new Info("std::optional<float>").pointerTypes("FloatOptional").define())
            .put(new Info("std::optional<double>").pointerTypes("DoubleOptional").define())
            .put(new Info("std::optional<size_t>").pointerTypes("SizeTOptional").define())
            .put(new Info("std::optional<std::string>").pointerTypes("StringOptional").define())
            .put(new Info("std::optional<std::vector<bool> >").pointerTypes("BoolVectorOptional").define())
            .put(new Info("std::optional<std::vector<int64_t> >").pointerTypes("LongVectorOptional").define())
            .put(new Info("std::optional<std::vector<double> >").pointerTypes("DoubleVectorOptional").define())
            .put(new Info("std::optional<std::vector<size_t> >").pointerTypes("SizeTVectorOptional").define())
            .put(new Info("std::optional<std::vector<std::string> >").pointerTypes("StringVectorOptional").define())
            .put(new Info("std::optional<std::vector<c10::Stride> >").pointerTypes("StrideVectorOptional").define())
            .put(new Info("std::optional<std::vector<c10::ShapeSymbol> >").pointerTypes("ShapeSymbolVectorOptional").define())
            .put(new Info("std::optional<std::vector<torch::Tensor> >", "std::optional<std::vector<at::Tensor> >").pointerTypes("TensorVectorOptional").define())
            .put(new Info("std::optional<c10::Device>", "std::optional<at::Device>", "std::optional<torch::Device>", "optional<c10::Device>").pointerTypes("DeviceOptional").define())
            .put(new Info("std::optional<c10::DeviceType>").pointerTypes("DeviceTypeOptional").define())
            .put(new Info("std::optional<c10::ArrayRef<int64_t> >", "std::optional<c10::IntArrayRef>", "std::optional<at::IntArrayRef>",
                "at::OptionalIntArrayRef", "c10::remove_symint<at::OptionalSymIntArrayRef>::type")
                // This second pointer type prevents optional.swap to work. I don't know exactly why. Skipping swap for now.
                .pointerTypes("LongArrayRefOptional", "@Cast({\"int64_t*\", \"c10::ArrayRef<int64_t>\", \"std::vector<int64_t>&\"}) @StdVector long...").define())
            .put(new Info("std::optional<c10::ArrayRef<int64_t> >::swap").skip())
            .put(new Info("std::optional<c10::ArrayRef<double> >", "std::optional<at::ArrayRef<double> >")
                .pointerTypes("DoubleArrayRefOptional", "@Cast({\"double*\", \"c10::ArrayRef<double>\", \"std::vector<double>&\"}) @StdVector double...").define())
            .put(new Info("std::optional<c10::ArrayRef<c10::SymInt> >", "std::optional<at::ArrayRef<c10::SymInt> >",
                "std::optional<c10::SymIntArrayRef>", "at::OptionalSymIntArrayRef").pointerTypes("SymIntArrayRefOptional").define())
            .put(new Info("std::optional<c10::Layout>", "std::optional<at::Layout>", "optional<c10::Layout>").pointerTypes("LayoutOptional").define())
            .put(new Info("std::optional<c10::MemoryFormat>", "std::optional<at::MemoryFormat>").pointerTypes("MemoryFormatOptional").define())
            .put(new Info("std::optional<c10::Scalar>", "std::optional<at::Scalar>").pointerTypes("ScalarOptional").define())
            .put(new Info("std::optional<c10::ScalarType>", "std::optional<at::ScalarType>", "std::optional<torch::Dtype>", "optional<at::ScalarType>", "optional<c10::ScalarType>").pointerTypes("ScalarTypeOptional").define())
            .put(new Info("std::optional<c10::AliasInfo>").pointerTypes("AliasInfoOptional").define())
            .put(new Info("std::optional<c10::IValue>").pointerTypes("IValueOptional").define())
            .put(new Info("std::optional<c10::impl::CppSignature>").pointerTypes("CppSignatureOptional").define())
            .put(new Info("std::optional<c10::DispatchKey>").pointerTypes("DispatchKeyOptional").define())
            .put(new Info("std::optional<c10::OperatorHandle>").pointerTypes("OperatorHandleOptional").define())
            .put(new Info("std::optional<c10::OperatorName>").pointerTypes("OperatorNameOptional").define())
            .put(new Info("std::optional<c10::QualifiedName>").pointerTypes("QualifiedNameOptional").define())
            .put(new Info("std::optional<c10::Stream>", "optional<c10::Stream>").pointerTypes("StreamOptional").define())
            .put(new Info("std::optional<c10::Stride>").pointerTypes("StrideOptional").define())
            .put(new Info("std::optional<c10::TypePtr>").pointerTypes("TypePtrOptional").define())
            .put(new Info("std::optional<c10::ClassType::Property>").pointerTypes("ClassTypePropertyOptional").define())
            .put(new Info("std::optional<c10::AliasTypeSet>").pointerTypes("AliasTypeSetOptional").define())
            .put(new Info("std::optional<c10::FunctionSchema>").pointerTypes("FunctionSchemaOptional").define())
            .put(new Info("std::optional<c10::SymDimVector>", "std::optional<at::SymDimVector>").pointerTypes("SymDimVectorOptional").define())
            .put(new Info("std::optional<c10::SymInt>").pointerTypes("SymIntOptional").define())
            .put(new Info("std::optional<at::IValue>").pointerTypes("IValueOptional").define())
            .put(new Info("std::optional<at::DimVector>").pointerTypes("DimVectorOptional").define())
            .put(new Info("std::optional<at::Dimname>").pointerTypes("DimnameOptional").define())
            .put(new Info("std::optional<at::DimnameList>").pointerTypes("DimnameListOptional").define())
            .put(new Info("std::optional<at::Generator>").pointerTypes("GeneratorOptional").define())
            .put(new Info("std::optional<at::Tensor>", "std::optional<torch::Tensor>", "std::optional<at::Tensor>", "std::optional<torch::TensorBase>", "std::optional<torch::autograd::Variable>").pointerTypes("TensorOptional").define())
            .put(new Info("std::optional<torch::TensorList>", "std::optional<at::TensorList>").pointerTypes("TensorArrayRefOptional").define())
            .put(new Info("std::optional<caffe2::TypeMeta>", "optional<caffe2::TypeMeta>").pointerTypes("TypeMetaOptional").define())
            .put(new Info("std::optional<torch::jit::ExecutorExecutionMode>").pointerTypes("ExecutorExecutionModeOptional").define())
            .put(new Info("std::optional<torch::jit::ExecutorExecutionMode>::operator ->").skip()) // Returns a pointer to ExecutorExecutionMode, which is an enum
            .put(new Info("const std::optional<torch::jit::InlinedCallStack>", "std::optional<torch::jit::InlinedCallStack>",
                "std::optional<torch::jit::InlinedCallStackPtr>").cast().pointerTypes("InlinedCallStackOptional").define())
            .put(new Info("std::optional<torch::jit::Scope>",
                "std::optional<torch::jit::ScopePtr>").cast().pointerTypes("ScopeOptional").define())
            .put(new Info("std::optional<torch::jit::ModuleInstanceInfo>").pointerTypes("ModuleInstanceInfoOptional").define())
            .put(new Info("std::optional<torch::jit::SourceRange>").pointerTypes("SourceRangeOptional").define())
            .put(new Info("std::optional<torch::jit::Method>").pointerTypes("MethodOptional").define())
            .put(new Info("std::optional<torch::jit::NamedValue>", "std::optional<NamedValue>").pointerTypes("NamedValueOptional").define())
            .put(new Info("std::optional<torch::jit::Value*>").pointerTypes("ValueOptional").define())
            .put(new Info("std::optional<torch::ExpandingArray<1> >",
                "std::optional<torch::ExpandingArray<2> >",
                "std::optional<torch::ExpandingArray<3> >").cast().pointerTypes("LongExpandingArrayOptional").define())
            .put(new Info("std::optional<torch::ExpandingArray<1,double> >",
                "std::optional<torch::ExpandingArray<2,double> >",
                "std::optional<torch::ExpandingArray<3,double> >",
                "std::optional<torch::nn::FractionalMaxPoolOptions<1>::ExpandingArrayDouble>",
                "std::optional<torch::nn::FractionalMaxPoolOptions<2>::ExpandingArrayDouble>",
                "std::optional<torch::nn::FractionalMaxPoolOptions<3>::ExpandingArrayDouble>").cast().pointerTypes("DoubleExpandingArrayOptional").define())
            .put(new Info("std::optional<std::tuple<std::string,size_t,size_t> >").pointerTypes("T_StringSizeTSizeT_TOptional").define())
            .put(new Info("torch::optional<std::tuple<torch::Tensor,torch::Tensor> >").pointerTypes("T_TensorTensor_TOptional").define())
            .put(new Info("std::optional<std::tuple<c10::TypePtr,int32_t> >", "std::optional<std::pair<c10::TypePtr,int32_t> >").pointerTypes("T_TypePtrLong_TOptional").cast().define())
            .put(new Info("std::optional<c10::string_view>").pointerTypes("StringViewOptional").define())
            .put(new Info("std::optional<std::vector<c10::string_view> >").pointerTypes("StringViewVectorOptional").define())
            .put(new Info("std::optional<std::pair<void*,void*> >", "std::optional<std::pair<torch::jit::BackendMetaPtr,torch::jit::BackendMetaPtr> >")/*.cast?*/.pointerTypes("PointerPairOptional").define())
            .put(new Info("std::optional<std::vector<c10::weak_intrusive_ptr<c10::StorageImpl> > >", "std::optional<std::vector<c10::ivalue::Future::WeakStorage> >").pointerTypes("WeakStorageVectorOptional").define())
            .put(new Info("std::optional<c10::impl::CppSignature>").pointerTypes("CppSignatureOptional").define())
            .put(new Info("std::optional<std::shared_ptr<c10::SafePyObject> >").pointerTypes("SafePyObjectOptional").define())
            .put(new Info("std::optional<std::pair<const char*,const char*> >").pointerTypes("BytePointerPairOptional").define())
            .put(new Info("std::optional<c10::intrusive_ptr<c10d::Backend> >").pointerTypes("DistributedBackendOptional").define())
            .put(new Info("std::optional<std::weak_ptr<c10d::Logger> >").pointerTypes("LoggerOptional").define())
             //.put(new Info("std::optional<std::function<std::string()> >").pointerTypes("StringSupplierOptional").define()) // .get() of the optional would return a std::function
            .put(new Info("std::optional<std::shared_ptr<c10::SafePyObjectT<c10::impl::TorchDispatchModeKey> > >", "std::optional<std::shared_ptr<c10::impl::PyObject_TorchDispatchMode> >").pointerTypes("PyObject_TorchDispatchModeOptional").define())
        ;


        //// Singleton
        infoMap
            .put(new Info("c10::Type::SingletonOrSharedTypePtr<c10::Type>", "c10::TypePtr", "c10::Type::TypePtr", "at::TypePtr",
                "torch::jit::TypeAttr::ConstructorType", "torch::jit::TypeAttr::ValueType").pointerTypes("Type.TypePtr")) // No way to move it outside Type class
            .put(new Info("c10::SingletonTypePtr<c10::Type>").pointerTypes("SingletonTypePtr"))
            .put(new Info("c10::SingletonTypePtr<c10::AnyType>").pointerTypes("AnyTypePtr"))
            .put(new Info("c10::SingletonTypePtr<c10::AnyEnumType>").pointerTypes("AnyEnumTypePtr"))
            .put(new Info("c10::SingletonTypePtr<c10::NumberType>").pointerTypes("NumberTypePtr"))
            .put(new Info("c10::SingletonTypePtr<c10::FloatType>").pointerTypes("FloatTypePtr"))
            .put(new Info("c10::SingletonTypePtr<c10::ComplexType>").pointerTypes("ComplexTypePtr"))
            .put(new Info("c10::SingletonTypePtr<c10::IntType>").pointerTypes("IntTypePtr"))
            .put(new Info("c10::SingletonTypePtr<c10::BoolType>").pointerTypes("BoolTypePtr"))
            .put(new Info("c10::SingletonTypePtr<c10::StringType>").pointerTypes("StringTypePtr"))
            .put(new Info("c10::SingletonTypePtr<c10::StorageType>").pointerTypes("StorageTypePtr"))
            .put(new Info("c10::SingletonTypePtr<c10::NoneType>").pointerTypes("NoneTypePtr"))
            .put(new Info("c10::SingletonTypePtr<c10::GeneratorType>").pointerTypes("GeneratorTypePtr"))
            .put(new Info("c10::SingletonTypePtr<c10::QuantizerType>").pointerTypes("QuantizerTypePtr"))
            .put(new Info("c10::SingletonTypePtr<c10::QSchemeType>").pointerTypes("QSchemeTypePtr"))
            .put(new Info("c10::SingletonTypePtr<c10::DeviceObjType>").pointerTypes("DeviceObjTypePtr"))
            .put(new Info("c10::SingletonTypePtr<c10::StreamObjType>").pointerTypes("StreamObjTypePtr"))
            .put(new Info("c10::SingletonTypePtr<c10::CapsuleType>").pointerTypes("CapsuleTypePtr"))
            .put(new Info("c10::SingletonTypePtr<c10::PyObjectType>").pointerTypes("PyObjectTypePtr"))
            .put(new Info("c10::SingletonTypePtr<c10::LayoutType>").pointerTypes("LayoutTypePtr"))
            .put(new Info("c10::SingletonTypePtr<c10::ScalarTypeType>").pointerTypes("ScalarTypeTypePtr"))
            .put(new Info("c10::SingletonTypePtr<c10::AnyListType>").pointerTypes("AnyListTypePtr"))
            .put(new Info("c10::SingletonTypePtr<c10::AnyTupleType>").pointerTypes("AnyTupleTypePtr"))
            .put(new Info("c10::SingletonTypePtr<c10::AnyClassType>").pointerTypes("AnyClassTypePtr"))
        ;


        //// std::variant
        infoMap
            .put(new Info("std::variant<torch::enumtype::kLinear,torch::enumtype::kConv1D,torch::enumtype::kConv2D,torch::enumtype::kConv3D,"
                          + "torch::enumtype::kConvTranspose1D,torch::enumtype::kConvTranspose2D,torch::enumtype::kConvTranspose3D,"
                          + "torch::enumtype::kSigmoid,torch::enumtype::kTanh,torch::enumtype::kReLU,torch::enumtype::kLeakyReLU>",
                "torch::nn::init::NonlinearityType").pointerTypes("Nonlinearity").define())
            .put(new Info("std::variant<torch::enumtype::kFanIn,torch::enumtype::kFanOut>",
                "torch::nn::init::FanModeType").pointerTypes("FanModeType").define())

            .put(new Info("std::variant<torch::enumtype::kZeros,torch::enumtype::kReflect,torch::enumtype::kReplicate,torch::enumtype::kCircular>",
                "torch::nn::ConvOptions<1>::padding_mode_t",
                "torch::nn::ConvOptions<2>::padding_mode_t",
                "torch::nn::ConvOptions<3>::padding_mode_t",
                "torch::nn::ConvTransposeOptions<1>::padding_mode_t",
                "torch::nn::ConvTransposeOptions<2>::padding_mode_t",
                "torch::nn::ConvTransposeOptions<3>::padding_mode_t",
                "torch::nn::detail::conv_padding_mode_t").pointerTypes("ConvPaddingMode").define())
            .put(new Info("std::variant<torch::ExpandingArray<1>,torch::enumtype::kValid,torch::enumtype::kSame>",
                "torch::nn::ConvOptions<1>::padding_t",
                "torch::nn::detail::ConvNdOptions<1>::padding_t",
                "torch::nn::functional::ConvFuncOptions<1>::padding_t",
                "torch::nn::functional::Conv1dFuncOptions::padding_t").purify().pointerTypes("Conv1dPadding").define())
            .put(new Info("std::variant<torch::ExpandingArray<2>,torch::enumtype::kValid,torch::enumtype::kSame>",
                "torch::nn::ConvOptions<2>::padding_t",
                "torch::nn::detail::ConvNdOptions<2>::padding_t",
                "torch::nn::functional::ConvFuncOptions<2>::padding_t",
                "torch::nn::functional::Conv2dFuncOptions::padding_t").purify().pointerTypes("Conv2dPadding").define())
            .put(new Info("std::variant<torch::ExpandingArray<3>,torch::enumtype::kValid,torch::enumtype::kSame>",
                "torch::nn::ConvOptions<3>::padding_t",
                "torch::nn::detail::ConvNdOptions<3>::padding_t",
                "torch::nn::functional::ConvFuncOptions<3>::padding_t",
                "torch::nn::functional::Conv3dFuncOptions::padding_t").purify().pointerTypes("Conv3dPadding").define())

            .put(new Info("std::variant<torch::enumtype::kSum,torch::enumtype::kMean,torch::enumtype::kMax>",
                "torch::nn::EmbeddingBagMode").pointerTypes("EmbeddingBagMode").define())
            .put(new Info("std::variant<torch::enumtype::kConstant,torch::enumtype::kReflect,torch::enumtype::kReplicate,torch::enumtype::kCircular>",
                "torch::nn::functional::PadFuncOptions::mode_t").pointerTypes("PaddingMode").define())

            .put(new Info("std::variant<torch::enumtype::kNone,torch::enumtype::kMean,torch::enumtype::kSum>",
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
                "torch::nn::BCEWithLogitsLossOptions::reduction_t", "torch::nn::functional::BinaryCrossEntropyWithLogitsFuncOptions::reduction_t").pointerTypes("LossReduction").define())
            .put(new Info("std::variant<torch::enumtype::kNone,torch::enumtype::kBatchMean,torch::enumtype::kSum,torch::enumtype::kMean>",
                "torch::nn::KLDivLossOptions::reduction_t", "torch::nn::functional::KLDivFuncOptions::reduction_t").pointerTypes("KLDivLossReduction").define())

            .put(new Info("std::variant<torch::enumtype::kBilinear,torch::enumtype::kNearest>",
                "torch::nn::functional::GridSampleFuncOptions::mode_t").pointerTypes("GridSampleMode").define())
            .put(new Info("std::variant<torch::enumtype::kZeros,torch::enumtype::kBorder,torch::enumtype::kReflection>",
                "torch::nn::functional::GridSampleFuncOptions::padding_mode_t").pointerTypes("GridSamplePaddingMode").define())

            .put(new Info("std::variant<torch::enumtype::kLSTM,torch::enumtype::kGRU,torch::enumtype::kRNN_TANH,torch::enumtype::kRNN_RELU>",
                "torch::nn::detail::RNNOptionsBase::rnn_options_base_mode_t").pointerTypes("RNNBaseMode").define())
            .put(new Info("std::variant<torch::enumtype::kTanh,torch::enumtype::kReLU>",
                "torch::nn::RNNOptions::nonlinearity_t", "torch::nn::RNNCellOptions::nonlinearity_t").pointerTypes("RNNNonlinearity").define())

            .put(new Info("std::variant<torch::enumtype::kNearest,torch::enumtype::kLinear,torch::enumtype::kBilinear,torch::enumtype::kBicubic,torch::enumtype::kTrilinear>",
                "torch::nn::UpsampleOptions::mode_t").pointerTypes("UpsampleMode").define())
            .put(new Info("std::variant<torch::enumtype::kNearest,torch::enumtype::kLinear,torch::enumtype::kBilinear,torch::enumtype::kBicubic,torch::enumtype::kTrilinear,torch::enumtype::kArea,torch::enumtype::kNearestExact>",
                "torch::nn::functional::InterpolateFuncOptions::mode_t").pointerTypes("InterpolateMode").define())

            .put(new Info("std::variant<torch::enumtype::kReLU,torch::enumtype::kGELU,std::function<torch::Tensor(const torch::Tensor&)> >",
                "torch::nn::activation_t",
                "torch::nn::TransformerOptions::activation_t").pointerTypes("TransformerActivation")) // Defined explicitly

            .put(new Info("std::variant<c10::Warning::UserWarning,c10::Warning::DeprecationWarning>", "c10::Warning::warning_variant_t").pointerTypes("WarningVariant").define()) // Cannot be defined as inner class of Warning
            .put(new Info("c10::Warning::UserWarning").pointerTypes("Warning.UserWarning"))
            .put(new Info("c10::Warning::DeprecationWarning").pointerTypes("Warning.DeprecationWarning"))
        ;

        /*
         * array of consecutive elements variants:
         * std::array
         *  fixed-size
         *  mapped to raw pointers, with cast()
         * std::vector
         *  variable-size array, re-allocatable
         * c10::ArrayRef<T>, defined in c10/util/ArrayRef.h
         *  not owning ref
         *  iterator is const T* => mapped to T pointer
         *  reverse_iterator is std::reverse_iterator => skipped
         * c10::List, defined in ATen/core/List.h
         *  wrapper around std::vector<IValue>
         *  (using c10::ListImpl::list_type = std::vector<IValue>)
         * SmallVector, defined in c10/util/SmallVector.h
         *  variable-size array, optimized for the case when the array is small, avoiding heap allocation
         *  iterator is T* or const T* => mapped to T pointer
         *  reverse_iterator is std::reverse_iterator => skipped
         */

        //// std::array
        infoMap
            .put(new Info("std::array<bool,2>", "std::array<bool,3>", "std::array<bool,4>").cast().pointerTypes("BoolPointer"))
            .put(new Info("std::array<c10::detail::infer_schema::ArgumentDef,0>").cast().pointerTypes("ArgumentDef"))
            .put(new Info("std::array<const char*,2>").pointerTypes("PointerPointer<BytePointer>"))
            .put(new Info("std::array<c10::FunctionalityOffsetAndMask,c10::num_functionality_keys>").cast().pointerTypes("FunctionalityOffsetAndMask"))
            .put(new Info("std::array<uint32_t,at::MERSENNE_STATE_N>").pointerTypes("IntPointer").cast())
            .put(new Info("std::array<std::optional<std::pair<torch::jit::BackendMetaPtr,torch::jit::BackendMetaPtr> >,at::COMPILE_TIME_MAX_DEVICE_TYPES>").pointerTypes("PointerPairOptional").cast())
            .put(new Info("std::array<uint8_t,c10::NumScalarTypes>").pointerTypes("BytePointer").cast())
        ;


        //// std::vector
        infoMap
            .put(new Info("std::vector<bool>").pointerTypes("BoolVector").define())
            .put(new Info("std::vector<uint8_t>", "std::vector<char>").pointerTypes("ByteVector").define().cast()) // cast to accomodate sign/unsigned
            .put(new Info("std::vector<const char*>").pointerTypes("BytePointerVector").define())
            .put(new Info("std::vector<int64_t>", "std::tuple<std::vector<int64_t>,std::vector<int64_t> >").cast().pointerTypes("LongVector").define())
            .put(new Info("std::vector<double>").cast().pointerTypes("DoubleVector").define())
            .put(new Info("std::vector<size_t>").cast().pointerTypes("SizeTVector").define())
            .put(new Info("std::vector<std::string>").pointerTypes("StringVector").define())
            .put(new Info("std::vector<c10::string_view>").pointerTypes("StringViewVector").define())
            .put(new Info("std::vector<std::pair<std::string,int64_t> >").pointerTypes("StringLongVector").define())
            .put(new Info("std::vector<c10::IValue>", "torch::jit::Stack").pointerTypes("IValueVector").define())
            .put(new Info("std::vector<c10::IValue>::const_iterator", "torch::jit::Stack::const_iterator").pointerTypes("IValueVector.Iterator"))
            .put(new Info("std::vector<c10::QEngine>", "std::vector<at::QEngine>").pointerTypes("QEngineVector").define())
            .put(new Info("std::vector<c10::ScalarType>").pointerTypes("ScalarTypeVector").define())
            .put(new Info("std::vector<c10::Symbol>").pointerTypes("SymbolVector").define())
            .put(new Info("std::vector<std::optional<int64_t> >").pointerTypes("LongOptionalVector").define())
            .put(new Info("std::vector<std::optional<at::IValue> >").pointerTypes("IValueOptionalVector").define())
            .put(new Info("std::vector<std::shared_ptr<c10::ClassType> >", "std::vector<c10::ClassTypePtr>").pointerTypes("SharedClassTypeVector").define())
            .put(new Info("std::vector<c10::Type::SingletonOrSharedTypePtr<c10::Type> >", "std::vector<c10::TypePtr>",
                "std::vector<c10::Type::TypePtr>", "c10::AliasTypeSet").pointerTypes("TypeVector").define())
            .put(new Info("const std::vector<at::Dimname>", "std::vector<at::Dimname>").pointerTypes("DimnameVector").define())
            .put(new Info("std::vector<c10::Stride>").pointerTypes("StrideVector").define())
            .put(new Info("std::vector<c10::ShapeSymbol>").pointerTypes("ShapeSymbolVector").define())
            .put(new Info("std::vector<c10::TensorImpl*>").pointerTypes("TensorImplVector").define())
            .put(new Info("std::vector<torch::autograd::Edge>", "torch::autograd::edge_list").pointerTypes("EdgeVector").define())  // Used in Node constructor
            .put(new Info("std::vector<torch::Tensor>", "std::vector<at::Tensor>", "std::vector<torch::autograd::Variable>", "torch::autograd::variable_list")
                .pointerTypes("TensorVector").define())
            .put(new Info("std::vector<at::indexing::TensorIndex>", "std::vector<at::indexing::TensorIndex,A>").pointerTypes("TensorIndexVector").define())
            .put(new Info("std::vector<std::optional<torch::autograd::Variable> >").pointerTypes("TensorOptionalVector").define())
            .put(new Info("const std::vector<std::unique_ptr<torch::autograd::FunctionPreHook> >",
                "std::vector<std::unique_ptr<torch::autograd::FunctionPreHook> >").pointerTypes("FunctionPreHookVector").define())
            .put(new Info("const std::vector<std::unique_ptr<torch::autograd::FunctionPostHook> >",
                "std::vector<std::unique_ptr<torch::autograd::FunctionPostHook> >").pointerTypes("FunctionPostHookVector").define())
            .put(new Info("const std::vector<torch::jit::Def>", "std::vector<torch::jit::Def>").pointerTypes("DefVector").define())
            .put(new Info("const std::vector<torch::jit::Property>", "std::vector<torch::jit::Property>").pointerTypes("PropertyVector").define())
            .put(new Info("const std::vector<torch::optim::OptimizerParamGroup>", "std::vector<torch::optim::OptimizerParamGroup>").pointerTypes("OptimizerParamGroupVector").define()) // OptimizerParamGroup::operator= erased
            .put(new Info("std::vector<torch::jit::Function*>").pointerTypes("FunctionVector").define())
            .put(new Info("std::vector<std::shared_ptr<torch::jit::Graph> >").pointerTypes("GraphVector").define())
            .put(new Info("std::vector<std::shared_ptr<torch::jit::Operator> >").pointerTypes("OperatorVector").define())
            .put(new Info("std::vector<std::shared_ptr<torch::jit::Resolver> >", "std::vector<torch::jit::ResolverPtr>").pointerTypes("ResolverVector").define())
            .put(new Info("std::vector<torch::jit::Value*>", "std::vector<Value*>").pointerTypes("ValueVector").define()) // Returned by inlineCallTo
            .put(new Info("std::vector<const torch::jit::Node*>").pointerTypes("JitNodeVector").define())
            .put(new Info("std::vector<torch::nn::Module>::iterator").pointerTypes("ModuleVector.Iterator"))
            .put(new Info("std::vector<torch::nn::AnyModule>").pointerTypes("AnyModuleVector").define())
            .put(new Info("std::vector<torch::nn::AnyModule>::iterator").pointerTypes("AnyModuleVector.Iterator"))
            .put(new Info("std::vector<std::shared_ptr<torch::nn::Module> >").pointerTypes("SharedModuleVector").define())
            .put(new Info("std::vector<std::shared_ptr<torch::nn::Module> >::iterator").pointerTypes("SharedModuleVector.Iterator"))
            .put(new Info("std::vector<std::pair<std::string,torch::Tensor> >").pointerTypes("StringTensorVector").define())
            .put(new Info("std::vector<std::pair<std::string,torch::nn::AnyModule> >").pointerTypes("StringAnyModuleVector").define())
            .put(new Info("std::vector<std::pair<std::string,std::shared_ptr<torch::nn::Module> > >").pointerTypes("StringSharedModuleVector").define())
            .put(new Info("std::vector<std::pair<torch::jit::FusionBehavior,size_t> >", "torch::jit::FusionStrategy").pointerTypes("FusionStrategy").define())
            .put(new Info("std::vector<c10::SymInt>").pointerTypes("SymIntVector").define())
            .put(new Info("std::vector<std::shared_ptr<torch::jit::SugaredValue> >").pointerTypes("SharedSugaredValueVector").define())
            .put(new Info("const std::vector<const c10::FunctionSchema*>").pointerTypes("FunctionSchemaVector").define())
            .put(new Info("const std::vector<at::DataPtr>", "std::vector<at::DataPtr>").pointerTypes("DataPtrVector").define()) // Used from cuda only
            .put(new Info("const std::vector<c10::weak_intrusive_ptr<c10::StorageImpl> >", "std::vector<c10::weak_intrusive_ptr<c10::StorageImpl> >").pointerTypes("WeakStorageVector").define())
            .put(new Info("std::vector<at::Tag>").pointerTypes("TagVector").define())
            .put(new Info("std::vector<std::shared_ptr<caffe2::serialize::ReadAdapterInterface> >").pointerTypes("ReadAdapterInterfaceVector").define())
            .put(new Info("std::vector<std::vector<size_t> >").pointerTypes("SizeTVectorVector").define())
            .put(new Info("std::vector<c10::ArrayRef<int64_t> >", "std::vector<c10::IntArrayRef>").pointerTypes("LongArrayRefVector").define())
            .put(new Info("std::vector<c10::intrusive_ptr<c10::ivalue::Future> >").pointerTypes("FutureVector").define())
            .put(new Info("std::vector<c10::intrusive_ptr<c10::SymNodeImpl> >").pointerTypes("SymNodeVector").define())
            .put(new Info("std::vector<std::shared_ptr<::gloo::transport::Device> >").pointerTypes("GlooDeviceVector").define())
        ;


        //// c10::ArrayRef
        /* Transparent cast from variadic java args to ArrayRef is only possible for non-boolean primitives (see mapArrayRef).
         * For Pointer subclasses for which a std::vector has been instantiated, we rely on ArrayRef converting constructor from std::vector and add the vector class as an otherPointerTypes()
         */
        for (ArrayInfo t : new ArrayInfo[]{
            new ArrayInfo("Argument").elementTypes("c10::Argument"),
            new ArrayInfo("ArgumentDef").elementTypes("c10::detail::infer_schema::ArgumentDef"),
            new ArrayInfo("BFloat16") /*.itPointerType("ShortPointer") */.elementTypes("decltype(::c10::impl::ScalarTypeToCPPType<::c10::ScalarType::BFloat16>::t)"),
            new ArrayInfo("Block").elementTypes("torch::jit::Block*").itPointerType("PointerPointer<Block>"),
            new ArrayInfo("Bool").itPointerType("BoolPointer").elementTypes("bool", "decltype(::c10::impl::ScalarTypeToCPPType<::c10::ScalarType::Bool>::t)").elementValueType("boolean"),
            new ArrayInfo("Byte").itPointerType("BytePointer").elementTypes("jbyte", "int8_t", "uint8_t").elementValueType("byte"),
            new ArrayInfo("Dimname").otherCppNames("at::DimnameList").elementTypes("at::Dimname").otherPointerTypes("DimnameVector"),
            new ArrayInfo("Double").itPointerType("DoublePointer").elementTypes("double"),
            new ArrayInfo("DoubleComplex") /*.itPointertype("DoublePointer") */.elementTypes("c10::complex<double>"),
            new ArrayInfo("EnumNameValue").elementTypes("c10::EnumNameValue"),
            new ArrayInfo("Float").itPointerType("FloatPointer").elementTypes("float").elementValueType("float"),
            new ArrayInfo("FloatComplex") /*.itPointerType("FloatPointer") */.elementTypes("c10::complex<float>"),
            new ArrayInfo("Future").elementTypes("c10::intrusive_ptr<c10::ivalue::Future>"),
            new ArrayInfo("Half") /*.itPointerType("ShortPointer") */.elementTypes("decltype(::c10::impl::ScalarTypeToCPPType<::c10::ScalarType::Half>::t)"),
            new ArrayInfo("IValue").elementTypes("c10::IValue", "const at::IValue").otherPointerTypes("IValueVector"),
            new ArrayInfo("Int")
                .itPointerType("IntPointer")
                .elementTypes("jint", "int", "int32_t", "uint32_t")
                .elementValueType("int"),
            new ArrayInfo("Tag").itPointerType("BytePointer").elementTypes("at::Tag"),
            new ArrayInfo("Long") // Warning : c10::IntArrayRef is a Java LongArrayRef and not a Java IntArrayRef
                                  .otherCppNames("c10::IntArrayRef", "torch::IntArrayRef", "at::IntArrayRef", "c10::remove_symint<c10::SymIntArrayRef>::type")
                                  .itPointerType("LongPointer")
                                  .elementTypes("int64_t", "jlong") // Order is important, since ArrayRef<long> and ArrayRef<long long> are incompatible, even though long == long long. And jlong is long long.
                                  .elementValueType("long"),
            new ArrayInfo("LongOptional").elementTypes("std::optional<int64_t>").otherPointerTypes("LongOptionalVector"),
            new ArrayInfo("NamedValue").elementTypes("torch::jit::NamedValue"),
            new ArrayInfo("Scalar").elementTypes("at::Scalar"),
            new ArrayInfo("ScalarType").itPointerType("@Cast(\"c10::ScalarType*\") BytePointer").elementTypes("c10::ScalarType", "at::ScalarType").otherPointerTypes("ScalarTypeVector"),
            new ArrayInfo("Short").itPointerType("ShortPointer").elementTypes("jshort", "int16_t", "uint16_t").elementValueType("short"),
            new ArrayInfo("SizeT").itPointerType("SizeTPointer").elementTypes("size_t").elementValueType("long"),
            new ArrayInfo("Stride").elementTypes("c10::Stride").otherPointerTypes("StrideVector"),
            new ArrayInfo("String").itPointerType("PointerPointer<BytePointer>" /*"@Cast({\"\", \"std::string*\"}) @StdString BytePointer"*/).elementTypes("std::string").otherPointerTypes("StringVector"),
            new ArrayInfo("SymInt").otherCppNames("c10::SymIntArrayRef").elementTypes("c10::SymInt"),
            new ArrayInfo("SymNode").elementTypes("c10::intrusive_ptr<c10::SymNodeImpl>", "c10::SymNode"),
            new ArrayInfo("Symbol").elementTypes("c10::Symbol").otherPointerTypes("SymbolVector"),
            new ArrayInfo("Tensor").otherCppNames("torch::TensorList", "at::TensorList", "at::ITensorListRef").elementTypes("torch::Tensor", "at::Tensor").otherPointerTypes("TensorVector"),  // Warning: not a TensorList (List<Tensor>)
            new ArrayInfo("TensorArg").elementTypes("torch::TensorArg", "at::TensorArg"),
            new ArrayInfo("TensorIndex").elementTypes("at::indexing::TensorIndex").otherPointerTypes("TensorIndexVector"),
            new ArrayInfo("TensorOptional").elementTypes("std::optional<at::Tensor>", "std::optional<torch::Tensor>", "std::optional<torch::autograd::Variable>").otherPointerTypes("TensorOptionalVector"),
            new ArrayInfo("Type").itPointerType("Type.TypePtr").elementTypes("c10::TypePtr", "c10::Type::TypePtr").otherPointerTypes("TypeVector"),
            new ArrayInfo("Value").elementTypes("torch::jit::Value*").otherPointerTypes("ValueVector")

        }) {
            t.mapArrayRef(infoMap);
        }

        // Special case for StringArrayRef: prevent using String or BytePointer and @StdString
        // when arrays or std::string are expected.
        // Any cleaner way to do this ?
        infoMap.put(new Info("c10::ArrayRef<std::string>::begin()").javaText(
            "public native @Const PointerPointer<BytePointer> begin();"
        )).put(new Info("c10::ArrayRef<std::string>::end()").javaText(
            "public native @Const PointerPointer<BytePointer> end();"
        )).put(new Info("c10::ArrayRef<std::string>::cbegin()").javaText(
            "public native @Const PointerPointer<BytePointer> cbegin();"
        )).put(new Info("c10::ArrayRef<std::string>::cend()").javaText(
            "public native @Const PointerPointer<BytePointer> cend();"
        )).put(new Info("c10::ArrayRef<std::string>::data()").javaText(
            "public native @Const PointerPointer<BytePointer> data();"
        )).put(new Info("c10::ArrayRef<std::string>(const std::string*, size_t)").javaText(
            "public StringArrayRef(PointerPointer<BytePointer> data, long length) { super((Pointer)null); allocate(data, length); }\n" +
            "private native void allocate(@Cast(\"const std::string*\") PointerPointer<BytePointer> data, @Cast(\"size_t\") long length);"
        )).put(new Info("c10::ArrayRef<std::string>(const std::string*, const std::string*)").javaText(
            "public StringArrayRef(PointerPointer<BytePointer> begin, PointerPointer<BytePointer> end) { super((Pointer)null); allocate(begin, end); }\n" +
            "private native void allocate(@Cast(\"const std::string*\") PointerPointer<BytePointer> begin, @Cast(\"const std::string*\") PointerPointer<BytePointer> end);"
        ));

        // Special case for TagArrayRef: Tag is an enum and not a Pointer. arrays returned as IntPointer.
        infoMap.put(new Info("c10::ArrayRef<at::Tag>::begin()").javaText(
            "public native @Const IntPointer begin();"
        )).put(new Info("c10::ArrayRef<at::Tag>::end()").javaText(
            "public native @Const IntPointer end();"
        )).put(new Info("c10::ArrayRef<at::Tag>::cbegin()").javaText(
            "public native @Const IntPointer cbegin();"
        )).put(new Info("c10::ArrayRef<at::Tag>::cend()").javaText(
            "public native @Const IntPointer cend();"
        )).put(new Info("c10::ArrayRef<at::Tag>::data()").javaText(
            "public native @Const IntPointer data();"
        )).put(new Info("c10::ArrayRef<at::Tag>(const at::Tag*, size_t)").javaText(
            "public TagArrayRef(IntPointer data, long length) { super((Pointer)null); allocate(data, length); }\n" +
            "private native void allocate(@Cast(\"const at::Tag*\") IntPointer data, @Cast(\"size_t\") long length);"
        )).put(new Info("c10::ArrayRef<at::Tag>(const at::Tag*, const at::Tag*)").javaText(
            "public TagArrayRef(IntPointer begin, IntPointer end) { super((Pointer)null); allocate(begin, end); }\n" +
            "private native void allocate(@Cast(\"const at::Tag*\") IntPointer begin, @Cast(\"const at::Tag*\") IntPointer end);"
        )).put(new Info("c10::ArrayRef<at::Tag>::vec()").skip() // Is there any way to make this work ?
        );


        //// c10::List
        for (ArrayInfo ai : new ArrayInfo[]{
            new ArrayInfo("DoubleComplex").elementTypes("c10::complex<double>"),
            new ArrayInfo("Boolean").elementTypes("bool").elementValueType("boolean"),
            new ArrayInfo("Long").elementTypes("int64_t").elementValueType("long"),
            new ArrayInfo("Double").elementTypes("double").elementValueType("double"),
            new ArrayInfo("TensorOptional").elementTypes("std::optional<at::Tensor>"),
            new ArrayInfo("Tensor").elementTypes("at::Tensor"),
            new ArrayInfo("Future").elementTypes("c10::intrusive_ptr<c10::ivalue::Future>").elementValueType("@IntrusivePtr(\"c10::ivalue::Future\") Future"),
            new ArrayInfo("Generic").elementTypes("c10::IValue").itPointerType("IValue").elementValueType("@ByVal IValue"),
        }) {
            ai.mapList(infoMap);
        }
        // friendly global setting lost + full qualification not resolved by parser
        infoMap.put(new Info("impl::ptr_to_first_element(const c10::List<c10::IValue>&)").javaNames("ptr_to_first_element").annotations("@Name(\"c10::impl::ptr_to_first_element\")").friendly());


        //// Small Vectors
        /* Warning: two classes "Node":
         * torch::autograd::Node, defined in autograd/function.h, referenced in Doxygen, TORCH_API
         * torch::lazy::Node, defined in torch/csrc/lazy/core/ir.h, TORCH_API, not mapped
         */
        infoMap.put(new Info("torch::autograd::Node").pointerTypes("Node").purify()); // Since Node is defined after SmallVector.h
        infoMap.put(new Info("c10::SymInt").pointerTypes("SymInt")); // Since SymInt is defined after SmallVector.h
        for (String[] t : new String[][]{
            {"SymInt", "SymInt", "@ByVal SymInt", "c10::SymInt", "at::kDimVectorStaticSize", "at::SymDimVector", "SymDimVector"},
            {"Long", "LongPointer", "long", "int64_t", "at::kDimVectorStaticSize", "at::DimVector", "DimVector"}
        }) {
            // Assume all have SmallVectorSizeType == uint32_t
            infoMap
                .put(new Info(template("c10::SmallVectorBase", template("c10::SmallVectorSizeType", t[3]))).pointerTypes("IntSizedSmallVectorBase"))
                .put(new Info(template("c10::SmallVectorTemplateCommon", t[3])).pointerTypes(t[0] + "SmallVectorCommon"))
                .put(new Info(template("c10::SmallVectorTemplateCommon", t[3]) + "::size_type",
                    template("c10::SmallVectorImpl", t[3]) + "::size_type").valueTypes("long"))
                .put(new Info(template("c10::SmallVectorTemplateBase", t[3])).pointerTypes(t[0] + "SmallVectorBase"))
                .put(new Info(template("c10::SmallVectorImpl", t[3])).pointerTypes(t[0] + "SmallVectorImpl"))
                .put(new Info(template("c10::SmallVectorImpl", t[3]) + "::iterator",
                    template("c10::SmallVectorImpl", t[3]) + "::const_iterator",
                    template("c10::SmallVectorTemplateCommon", t[3]) + "::iterator",
                    template("c10::SmallVectorTemplateCommon", t[3]) + "::pointer"
                )
                    .cast().pointerTypes(t[1]))
                .put(new Info(
                    template("c10::SmallVector", t[3], t[4]) + "(" + template("c10::SmallVectorImpl", t[3]) + "&&)",
                    template("c10::SmallVector", t[3], t[4]) + "::operator =(" + template("c10::SmallVectorImpl", t[3]) + "&&)")
                    .skip())
                .put(new Info(
                    template("c10::SmallVectorTemplateCommon", t[3]) + "::reference",
                    template("c10::SmallVectorTemplateCommon", t[3]) + "::const_reference")
                    .pointerTypes(t[1]).valueTypes(t[2]))
                .put(new Info(
                    template("c10::SmallVectorTemplateCommon", t[3]) + "::reverse_iterator",
                    template("c10::SmallVectorTemplateCommon", t[3]) + "::const_reverse_iterator")
                    .skip())
                .put(new Info(template("c10::SmallVectorImpl", t[3]) + "::ValueParamT")
                    .valueTypes(t[2]))
            ;
            if (t[5] == null) {
                infoMap.put(new Info(template("c10::SmallVector", t[3], t[4]), template("at::SmallVector", t[3], t[4])).pointerTypes(t[6]));
            } else {
                infoMap.put(new Info(template("c10::SmallVector", t[3], t[4]), template("at::SmallVector", t[3], t[4]), t[5]).pointerTypes(t[6]));
            }
        }


        //// std::map
        infoMap
            .put(new Info("std::map<std::string,std::string>").pointerTypes("StringStringMap").define())
            .put(new Info("std::map<std::string,int64_t>").pointerTypes("StringLongMap").define())
            .put(new Info("std::map<std::string,at::Tensor>").pointerTypes("StringTensorMap").define()) // Used by distributed only
        ;


        //// std::unordered_set
        infoMap
            .put(new Info("std::unordered_set<std::string>").pointerTypes("StringSet").define())
            .put(new Info("std::unordered_set<c10::IValue,c10::IValue::HashAliasedIValue,c10::IValue::CompAliasedIValues>").pointerTypes("HashAliasedIValues").define())
            .put(new Info("std::unordered_set<c10::Symbol>").pointerTypes("SymbolSet").define())
            .put(new Info("std::unordered_set<torch::TensorImpl*>", "std::unordered_set<at::TensorImpl*>").pointerTypes("TensorImplSet").define())
            .put(new Info("std::unordered_set<torch::autograd::Node*>").pointerTypes("NodeSet").define())
            .put(new Info("std::unordered_set<c10::DeviceType>").pointerTypes("DeviceTypeSet").define())
            .put(new Info("std::unordered_set<int16_t>", "std::unordered_set<torch::distributed::rpc::worker_id_t>").pointerTypes("ShortSet").define())
            .put(new Info("std::set<torch::profiler::impl::ActivityType>").pointerTypes("ActivityTypeSet").define())
            .put(new Info("std::unordered_map<size_t,std::string>").pointerTypes("SizeTStringMap").define())
            // .put(new Info("std::unordered_map<int64_t,std::shared_ptr<torch::distributed::autograd::RecvRpcBackward> >").pointerTypes("LongRecvRpcBackwardMap").define()) // Not on windows
            // .put(new Info("std::unordered_map<int64_t,std::shared_ptr<torch::distributed::autograd::SendRpcBackward> >").pointerTypes("LongSendRpcBackwardMap").define())
        ;


        //// std::unordered_map
        infoMap
            .put(new Info("std::unordered_map<c10::IValue,c10::IValue,c10::IValue::HashAliasedIValue,c10::IValue::CompAliasedIValues>").pointerTypes("HashAliasedIValueMap").define())
            .put(new Info("std::unordered_map<std::string,bool>").pointerTypes("StringBoolMap").define())
            .put(new Info("std::unordered_map<std::string,size_t>").pointerTypes("StringSizeTMap").define())
            .put(new Info("std::unordered_map<std::string,std::string>").pointerTypes("ExtraFilesMap").define())
            .put(new Info("std::unordered_map<std::string,c10::TypePtr>").pointerTypes("TypeEnv").define())
            .put(new Info("std::unordered_map<std::string,c10::IValue>", "std::unordered_map<std::string,at::IValue>").pointerTypes("StringIValueMap").define())
            .put(new Info("std::unordered_map<std::string,torch::jit::Value*>").pointerTypes("StringValueMap").define())
            .put(new Info("std::unordered_map<torch::jit::Value*,torch::jit::Value*>").pointerTypes("ValueValueMap").define())
            .put(new Info("std::unordered_map<torch::jit::ArgumentSpec,torch::jit::ExecutionPlan>").pointerTypes("ArgumentSpecExecutionPlanMap").define())
            .put(new Info("std::unordered_map<torch::jit::TreeRef,std::string>", "std::unordered_map<c10::intrusive_ptr<torch::jit::Tree>,std::string>").pointerTypes("TreeStringMap").define())
            .put(new Info("std::unordered_map<std::string,int32_t>").pointerTypes("StringIntMap").define())
            .put(new Info(
                "const std::unordered_map<torch::autograd::Node*,torch::dynamo::autograd::NodeCall>",
                "std::unordered_map<torch::autograd::Node::Node*,torch::dynamo::autograd::NodeCall>" // Fix erroneous ns qualification due to a previous `using Node::Node`
            ).pointerTypes("NodeNodeCallMap").define())
            .put(new Info("std::unordered_map<c10::IValue,c10::IValue,c10::IValue::HashIdentityIValue,c10::IValue::CompIdentityIValues>").pointerTypes("HashIdentityIValueMap").define())
        ;


        //// std::atomic
        infoMap
            .put(new Info("std::atomic_bool", "std::atomic<bool>").cast().valueTypes("boolean").pointerTypes("BoolPointer"))
            .put(new Info("std::atomic_uint64_t", "std::atomic<uint64_t>", "std::atomic<long unsigned int>", "std::atomic_size_t", "std::atomic<size_t>").cast().valueTypes("long").pointerTypes("LongPointer"))
            .put(new Info("std::atomic<const c10::impl::DeviceGuardImplInterface*>").cast().pointerTypes("DeviceGuardImplInterface"))
            .put(new Info("std::atomic<uint32_t>").cast().valueTypes("int").pointerTypes("IntPointer"))
        ;


        //// std::tuple
        infoMap
            .put(new Info("std::tuple<int,int>").pointerTypes("T_IntInt_T").define()) // Needed for CUDAStream
            .put(new Info("std::tuple<int64_t,int64_t>").pointerTypes("T_LongLong_T").define())
            .put(new Info("std::tuple<torch::Tensor,torch::Tensor>", "std::tuple<at::Tensor,at::Tensor>", "std::tuple<torch::Tensor,torch::Tensor>", "std::tuple<at::Tensor&,at::Tensor&>").pointerTypes("T_TensorTensor_T").define())
            .put(new Info("std::tuple<torch::Tensor,torch::Tensor,torch::Tensor>", "std::tuple<at::Tensor,at::Tensor,at::Tensor>", "std::tuple<at::Tensor&,at::Tensor&,at::Tensor&>").pointerTypes("T_TensorTensorTensor_T").define())
            .put(new Info("std::tuple<torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor>", "std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor>", "std::tuple<at::Tensor&,at::Tensor&,at::Tensor&,at::Tensor&>").pointerTypes("T_TensorTensorTensorTensor_T").define())
            .put(new Info("std::tuple<torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor>", "std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor>", "std::tuple<at::Tensor&,at::Tensor&,at::Tensor&,at::Tensor&,at::Tensor&>").pointerTypes("T_TensorTensorTensorTensorTensor_T").define())
            .put(new Info("std::tuple<torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor>", "std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor>", "std::tuple<at::Tensor&,at::Tensor&,at::Tensor&,at::Tensor&,at::Tensor&,at::Tensor&,at::Tensor&>").pointerTypes("T_TensorTensorTensorTensorTensorTensorTensor_T").define())
            .put(new Info("std::tuple<torch::Tensor,torch::Tensor,torch::Tensor,std::vector<torch::Tensor> >", "std::tuple<at::Tensor,at::Tensor,at::Tensor,std::vector<at::Tensor> >").pointerTypes("T_TensorTensorTensorTensorVector_T").define())
            .put(new Info("std::tuple<torch::Tensor,torch::Tensor,double,int64_t>", "std::tuple<at::Tensor,at::Tensor,double,int64_t>").pointerTypes("T_TensorTensorDoubleLong_T").define())
            .put(new Info("std::tuple<torch::Tensor,std::tuple<torch::Tensor,torch::Tensor> >").pointerTypes("T_TensorT_TensorTensor_T_T").define())
            .put(new Info("std::tuple<c10::MaybeOwned<at::Tensor>,c10::MaybeOwned<at::Tensor> >")
                .pointerTypes("T_TensorMaybeOwnedTensorMaybeOwned_T").define())
            .put(new Info("std::tuple<c10::MaybeOwned<at::Tensor>,c10::MaybeOwned<at::Tensor>,c10::MaybeOwned<at::Tensor> >")
                .pointerTypes("T_TensorMaybeOwnedTensorMaybeOwnedTensorMaybeOwned_T").define())
            .put(new Info("std::tuple<torch::nn::utils::rnn::PackedSequence,torch::Tensor>").purify().pointerTypes("T_PackedSequenceTensor_T").define())
            .put(new Info("std::tuple<torch::nn::utils::rnn::PackedSequence,std::tuple<torch::Tensor,torch::Tensor> >").purify().pointerTypes("T_PackedSequenceT_TensorTensor_T_T").define())
            .put(new Info("std::tuple<torch::Tensor&,torch::Tensor&>",
                "std::tuple<torch::Tensor&,torch::Tensor&,torch::Tensor&>",
                "std::tuple<torch::Tensor&,torch::Tensor&,torch::Tensor&,torch::Tensor&>",
                "std::tuple<torch::Tensor&,torch::Tensor&,torch::Tensor&,torch::Tensor&,torch::Tensor&>",
                "std::tuple<torch::Tensor&,torch::Tensor&,torch::Tensor&,torch::Tensor&,torch::Tensor&,torch::Tensor&>",
                "std::tuple<torch::Tensor&,torch::Tensor&,torch::Tensor&,torch::Tensor&,torch::Tensor&,torch::Tensor&,torch::Tensor&>"
            ).cast().pointerTypes("PointerPointer<Tensor>"))
            .put(new Info("std::tuple<std::string,size_t,size_t>").pointerTypes("T_StringSizeTSizeT_T").define())
            .put(new Info("std::tuple<torch::Tensor,std::vector<torch::Tensor> >", "std::tuple<at::Tensor,std::vector<at::Tensor> >").pointerTypes("T_TensorTensorVector_T").define())
            .put(new Info("std::tuple<torch::Tensor,std::vector<torch::Tensor>,std::vector<torch::Tensor> >", "std::tuple<at::Tensor,std::vector<at::Tensor>,std::vector<at::Tensor> >").pointerTypes("T_TensorTensorVectorTensorVector_T").define())
            .put(new Info("const std::tuple<at::DataPtr,size_t>", "std::tuple<at::DataPtr,size_t>").pointerTypes("T_DataPtrSizeT_T").define())
            .put(new Info("std::tuple<c10::TypePtr,int32_t>", "std::pair<c10::TypePtr,int32_t>").pointerTypes("T_TypePtrLong_T").define()) // Parse this pair as tuple because Parser doesn't generate valid code for optional<pair>
            .put(new Info("std::tuple<std::shared_ptr<c10::SafePyObject>,c10::impl::TorchDispatchModeKey>").pointerTypes("T_SafePyObjectTorchDispatchModeKey_T").define())
            //.put(new Info("std::tuple<c10::intrusive_ptr<torch::distributed::rpc::Message>,std::vector<c10::weak_intrusive_ptr<c10::StorageImpl> > >").pointerTypes("T_MessageWeakStorage_T").define()) // Message not on Windows
            .put(new Info("std::tuple<std::vector<std::vector<size_t> >,std::vector<size_t> >").pointerTypes("T_SizeTVectorVectorSizeTVector_T").define())
            .put(new Info("std::tuple<std::shared_ptr<c10::impl::PyObject_TorchDispatchMode>,c10::impl::TorchDispatchModeKey>").pointerTypes("T_PyObject_TorchDispatchModeTorchDispatchModeKey_T").define())
        ;


        //// Other std stuff
        infoMap
            .put(new Info("std::type_index").pointerTypes("@Cast(\"std::type_index*\") Pointer"))
            .put(new Info("std::deque<torch::Tensor>").pointerTypes("TensorDeque").define())
            .put(new Info("std::bitset<64>", "std::bitset<at::kVmapNumLevels>", "std::bitset<dim_bitset_size>",
                "std::bitset<at::kVmapMaxTensorDims>", "std::bitset<at::dim_bitset_size>").valueTypes("long"))
            .put(new Info("std::basic_string<char>").annotations("@StdString").valueTypes("BytePointer").pointerTypes("@Cast({\"char*\", \"std::string\"}) BytePointer"))
        ;


        //// Jit List
        for (String[] t : new String[][]{
            {"ExprList", "torch::jit::Expr", "Expr"},
            {"StmtList", "torch::jit::Stmt", "Stmt"},
            {"WithItemList", "torch::jit::WithItem", "WithItem"},
            {"PropertyList", "torch::jit::Property", "Property"},
            {"AssignList", "torch::jit::Assign", "Assign"},
            {"ParamList", "torch::jit::Param", "Param"},
            {"IdentList", "torch::jit::Ident", "Ident"},
            {"AttributeList", "torch::jit::Attribute", "Attribute"},
        }) {
            infoMap.put(new Info(template("torch::jit::List", t[1])).pointerTypes(t[0]))
                   .put(new Info(template("torch::jit::ListIterator", t[1])).pointerTypes(t[0] + "Iterator"))
                   .put(new Info(template("torch::jit::List", t[1]) + "::map").skip()) // Could map if needed
            ;
        }
        infoMap.put(new Info("torch::jit::TreeList::const_iterator").cast().pointerTypes("Tree"));


        //// c10 Dict
        for (String[] d : new String[][] {
            { "c10::IValue", "c10::IValue", "Generic" },
            { "std::string", "c10::impl::GenericList", "StringGenericList" },
            { "torch::Tensor", "torch::Tensor", "TensorTensor" }
        }) {
            infoMap
                .put(new Info(template("c10::Dict", d[0], d[1])).purify().pointerTypes(d[2] + "Dict"))
                .put(new Info(template("c10::impl::DictEntryRef", d[0], d[1], "c10::detail::DictImpl::dict_map_type::iterator")).pointerTypes("GenericDictEntryRef"))
                .put(new Info(template("c10::impl::DictIterator", d[0], d[1], "c10::detail::DictImpl::dict_map_type::iterator"),
                    template("c10::Dict", d[0], d[1]) + "::iterator").purify().pointerTypes(d[2] + "DictIterator").friendly())
                //.put(new Info("c10::Dict<std::string,c10::impl::GenericList>(c10::TypePtr, c10::TypePtr)").skip())
                // Don't know how to map :difference_type
                .put(new Info(template("c10::impl::DictIterator", d[0], d[1], "c10::detail::DictImpl::dict_map_type::iterator") + "::operator -").skip())
                /* Following operators throw a template error "no match", even in C++. */
                .put(new Info(template("c10::impl::DictIterator", d[0], d[1], "c10::detail::DictImpl::dict_map_type::iterator")
                              + "::operator <(const " + template("c10::impl::DictIterator", d[0], d[1], "c10::detail::DictImpl::dict_map_type::iterator")
                              + "&, const " + template("c10::impl::DictIterator", d[0], d[1], "c10::detail::DictImpl::dict_map_type::iterator") + "&)").skip())
                .put(new Info(template("c10::impl::DictIterator", d[0], d[1], "c10::detail::DictImpl::dict_map_type::iterator")
                              + "::operator <=(const " + template("c10::impl::DictIterator", d[0], d[1], "c10::detail::DictImpl::dict_map_type::iterator")
                              + "&, const " + template("c10::impl::DictIterator", d[0], d[1], "c10::detail::DictImpl::dict_map_type::iterator") + "&)").skip())
                .put(new Info(template("c10::impl::DictIterator", d[0], d[1], "c10::detail::DictImpl::dict_map_type::iterator")
                              + "::operator >=(const " + template("c10::impl::DictIterator", d[0], d[1], "c10::detail::DictImpl::dict_map_type::iterator")
                              + "&, const " + template("c10::impl::DictIterator", d[0], d[1], "c10::detail::DictImpl::dict_map_type::iterator") + "&)").skip())
                .put(new Info(template("c10::impl::DictIterator", d[0], d[1], "c10::detail::DictImpl::dict_map_type::iterator")
                              + "::operator >(const " + template("c10::impl::DictIterator", d[0], d[1], "c10::detail::DictImpl::dict_map_type::iterator")
                              + "&, const " + template("c10::impl::DictIterator", d[0], d[1], "c10::detail::DictImpl::dict_map_type::iterator") + "&)").skip())
            ;
        }
        infoMap
            .put(new Info("c10::impl::DictIterator::operator -(const c10::impl::DictIterator&, const c10::impl::DictIterator&)").skip())
            .put(new Info("c10::Dict::iterator::operator <(const c10::Dict::iterator&, const c10::Dict::iterator&)").skip())
            .put(new Info("c10::Dict::iterator::operator <=(const c10::Dict::iterator&, const c10::Dict::iterator&)").skip())
            .put(new Info("c10::Dict::iterator::operator >=(const c10::Dict::iterator&, const c10::Dict::iterator&)").skip())
            .put(new Info("c10::Dict::iterator::operator >(const c10::Dict::iterator&, const c10::Dict::iterator&)").skip())
        ;



        //// torch::OrderedDict
        for (String[] o: new String[][] {
            { "std::string", "torch::Tensor", "StringTensor" },
            { "std::string", "torch::nn::AnyModule", "StringAnyModule" },
            { "std::string", "std::shared_ptr<torch::nn::Module>", "StringSharedModule" }
        }) {
            infoMap
                .put(new Info(template("torch::OrderedDict", o[0], o[1])).pointerTypes(o[2] + "Dict"))
                .put(new Info(template("torch::OrderedDict<Key,Value>::Item", o[0], o[1]), template("torch::OrderedDict", o[0], o[1]) + "::Item").pointerTypes(o[2] + "DictItem"))
                // Adding const since items don't have no-arg constructors. See PR #664.
                .put(new Info("const " + template("std::vector", template("torch::OrderedDict", o[0], o[1]) + "::Item")).pointerTypes(o[2] + "DictItemVector").define())
            ;
        }


        //// std::pair
        infoMap
            // Parser doesn't generate iterators for vector of pairs, so function returning such iterators, like ParameterListImpl::begin()
            // must be mapped to returning item instead. Issue #673. Change when issue resolved.
            .put(new Info("std::pair<std::string,torch::Tensor>", "std::pair<std::string,torch::Tensor>").cast().pointerTypes("StringTensorPair").define())
            .put(new Info("std::pair<std::string,torch::nn::AnyModule>").pointerTypes("StringAnyModulePair").define())
            .put(new Info("std::pair<std::string,std::shared_ptr<torch::nn::Module> >").pointerTypes("StringSharedModulePair").define())
            .put(new Info("std::pair<at::RecordFunctionHandle,int>").pointerTypes("RecordFunctionHandleIntPair").define())
            .put(new Info("std::pair<void*,void*>", "std::pair<torch::jit::BackendMetaPtr,torch::jit::BackendMetaPtr>").pointerTypes("PointerPair").define())
            .put(new Info("std::pair<size_t,torch::jit::MatchedSchema>").pointerTypes("SizeTMatchedSchemaPair").define())
            .put(new Info("std::pair<const char*,const char*>").pointerTypes("BytePointerPair").define())
            .put(new Info("std::pair<std::string,c10::IValue>").pointerTypes("EnumNameValue").define())
            .put(new Info("std::pair<int,int>").pointerTypes("IntPair").define())
        ;

        //// std::chrono
        infoMap
            .put(new Info("std::chrono::time_point<std::chrono::system_clock>").pointerTypes("TimePoint"))
            .put(new Info("std::chrono::duration<long int, std::ratio<1, 1000> >", "std::chrono::milliseconds").pointerTypes("Milliseconds"))
            .put(new Info("std::chrono::duration<float>").pointerTypes("FloatDuration"))
        ;

        //// c10::intrusive_ptr
        /* We cannot define an adapter working like SharedPtrAdapter since there is no public constructor of
          intrusive_ptr<T> taking a T*. */
        for (PointerInfo pi : new PointerInfo[]{
            new PointerInfo("at::Quantizer"),
            new PointerInfo("c10::GeneratorImpl"),
            new PointerInfo("c10::ivalue::Tuple"),
            new PointerInfo("c10::ivalue::Future", "at::ivalue::Future", "torch::distributed::rpc::JitFuture"),
            new PointerInfo("c10::ivalue::ConstantString"),
            new PointerInfo("c10::ivalue::Await"),
            new PointerInfo("c10::ivalue::Object").javaBaseName("Obj"),
            new PointerInfo("c10::ivalue::PyObjectHolder"),
            new PointerInfo("c10::ivalue::EnumHolder"),
            new PointerInfo("c10::RRefInterface"),
            new PointerInfo("c10::TensorImpl"),
            new PointerInfo("c10::TensorImpl,c10::UndefinedTensorImpl").javaBaseName("TensorImpl"),
            new PointerInfo("c10::StorageImpl", "c10::StorageImpl,NullType"),
            new PointerInfo("c10::SymNodeImpl").javaBaseName("SymNode"),
            new PointerInfo("c10::BackendMeta"), //.javaBaseName("BackendMetaRef"), // Warning: BackendMetaPtr is sth different
            new PointerInfo("torch::jit::Tree").otherCppNames("torch::jit::TreeRef"),

            new PointerInfo("c10d::Store"),
            new PointerInfo("c10d::ProcessGroup::Options"),
            new PointerInfo("c10d::Work"),
            new PointerInfo("c10d::Backend").javaBaseName("DistributedBackend"),
            new PointerInfo("c10d::_SupplementBase"),
            new PointerInfo("c10d::ProcessGroup"),
            new PointerInfo("intra_node_comm::IntraNodeComm"),
            //new PointerInfo("torch::distributed::rpc::Message"), // Not on Windows
            new PointerInfo("c10d::ProcessGroupGloo::AsyncWork"),
            new PointerInfo("c10d::ProcessGroupGloo::Options"),
            new PointerInfo("c10d::ProcessGroupGloo")
        }) {
        pi.makeIntrusive(infoMap);
        }
        infoMap.put(new Info("c10::ivalue::Object").pointerTypes("Obj"));
        infoMap.put(new Info("torch::distributed::rpc::JitFuture").pointerTypes("Future"));
        infoMap.put(new Info("c10::SymNodeImpl").pointerTypes("SymNode"));


        //// Classes that Parser cannot detect as virtual
        infoMap.put(new Info("c10::SharedType", "c10::StrongTypePtr",
            "c10::WeakTypePtr", "torch::autograd::CppFunctionPreHook", "torch::autograd::DifferentiableViewMeta",
            "torch::autograd::TraceableFunction", "torch::jit::Instruction", "torch::jit::Method", "torch::jit::ModuleInstanceInfo",
            "torch::jit::Object::Property", "torch::jit::OperatorSet", "torch::jit::SourceRangePickler", "torch::jit::Unpickler",
            "torch::jit::Operator").purify());


        /// Classes skipped for various non-investigated reasons
        infoMap
            .put(new Info(
                "c10::detail::MultiDispatchKeySet", "c10::ExclusivelyOwnedTraits", "c10::FunctionSchema::dump",
                "c10::domain_prefix", "c10::C10FlagsRegistry", "c10::enforce_detail::EnforceFailMessage", "c10::impl::build_feature_required_feature_not_available",
                "c10::detail::getMaybeFakeTypePtr_", "c10::complex_literals::operator \"\"_if", "c10::complex_literals::operator \"\"_id",
                "decltype(::c10::impl::ScalarTypeToCPPType<::c10::ScalarType::ComplexHalf>::t)", "c10::BoxedKernel", "c10::ExtraMeta", "c10::remove_symint",
                "c10::InefficientStdFunctionContext", "c10::DataPtr::move_context", "c10::detail::UniqueVoidPtr::move_context", "QuantizerPtr", "c10::IValue::toModule", "c10::toBackendComponent",
                "std::optional<THPObjectPtr>", "c10::asIntArrayRefSlow", "c10::standardizeVectorForUnion",
                "c10::impl::ExcludeDispatchKeyGuard", "c10::impl::ScalarTypeToCPPType", "c10::impl::AnnotatedKernel", "c10::impl::OperatorEntry",
                "c10::StorageImpl(c10::StorageImpl)", "c10::StorageImpl::operator =",
                "c10::TensorImpl(c10::TensorImpl)", "c10::TensorImpl::operator =",
                "caffe2::Blob(caffe2::Blob)", "caffe2::Blob::operator =", "c10::detail::infer_schema::bool_t",
                "torch::serialize::InputArchive(torch::serialize::InputArchive)", "torch::serialize::InputArchive::operator =",
                "torch::serialize::OutputArchive(torch::serialize::OutputArchive)", "torch::serialize::OutputArchive::operator =",
                "at::_test_serialization_subcmul", "at::_test_optional_intlist", "at::_test_optional_filled_intlist",
                "at::_test_optional_floatlist", "at::_test_string_default", "at::_test_ambiguous_defaults",
                "at::TensorBase::expect_contiguous", // conflict with returning type of "Tensor::expect_contiguous"
                "torch::Tensor::print", "at::borrow_from_optional_tensor",
                "at::MaterializedITensorListRef", "at::impl::check_names_valid_for", "at::internal::launch_no_thread_state",
                "at::checkSameNumel", "at::check_names_valid_for", "at::default_names", "at::get_device", "at::detail::scalar_fill",
                "at::namedinference::compute_diagonal_outnames", "torch::Tensor::packed_accessor", "torch::optim::serialize", "torch::none_of",
                "torch::CountTensors", "torch::CountVariables", "torch::autograd::ExtractVariables", "torch::autograd::detail::MakeNextFunctionList",
                "torch::autograd::AutogradMeta::hooks_", "torch::autograd::AutogradMeta::cpp_hooks_list_",
                "torch::autograd::VariableType::unpack", "torch::autograd::VariableType::unpack_opt", "torch::jit::parseSchemaOrName",
                "torch::jit::trace", "torch::jit::tracer::TracingState::lookup_var_name_fn", "torch::jit::tracer::ArgumentStash",
                "torch::jit::constant_not_supported_error", "torch::jit::ObjectAttributeError", "torch::jit::utils::get_module_info",
                "torch::jit::operator <<(std::ostream&, torch::jit::Instruction)", "torch::jit::toString(torch::jit::OpCode)",
                "torch::jit::PropertyPropBase::processLoop", "torch::jit::PropertyPropBase::processIf", "torch::jit::PropertyPropBase::propagateBlock",
                "torch::jit::getMobileInterfaceCallExport", "torch::jit::OperatorSet::getOps", "torch::jit::SourceView::findSourceRangeThatGenerated",
                "at::namedinference::propagate_names_if_present_and_nonempty", "torch::jit::_load_jit_module_from_flatbuffer_bytes", "torch::jit::_save_jit_module_to",
                "torch::jit::checkHasValidSetGetState", "torch::jit::getTypeTags", "torch::jit::setTypeTags", "torch::jit::getStorageKey",
                "torch::jit::getUnresolvedClassAttributes", "torch::jit::isOpSupportedInMobile", "torch::jit::restoreAccurateTypeTags",
                "torch::jit::detail::getDifferentiableGraphOpExecutor", "torch::jit::detail::getGradExecutor", "torch::jit::Graph::createPythonOp",
                "torch::jit::Graph::createDifferentiableSubgraph", "torch::jit::NamedValue::type", "torch::jit::ProfileOp", "torch::jit::Value::isValidName",
                "torch::jit::EqualType::operator ()", "torch::jit::HashType::operator ()", "torch::jit::InterpreterContinuation::operator ()",
                "torch::jit::Object(c10::QualifiedName, torch::jit::CompilationUnit*, bool)", "torch::jit::Source::findSourceRangeThatGenerated",
                "torch::jit::SourceRangeDeserializer::deserialize", "torch::jit::SourceRangePickler::pickle", "torch::jit::Pickler::pushEmptyDict",
                "torch::jit::PrintDepsTable::add", "torch::jit::printerHasSpecialCaseFor", "ONNX_NAMESPACE::ModelProto", "torch::jit::export_onnx",
                "torch::jit::Function::call", "torch::jit::GraphFunction::call", "torch::jit::GraphFunction::function_creator", "torch::jit::getOptionsFromGlobal",
                "torch::jit::serialize_model_proto_to_string", "torch::onnx::IR_VERSION", "torch::onnx::PRODUCER_VERSION",
                "TORCH_DISALLOW_TEMPORARIES", "TORCH_DISALLOW_TEMPORARIES_IMPL", // Issue #674
                "DEFINE_CASTING(TAG, ...)", "TORCH_ILISTREF_FORALL_TAGS",
                "torch::autograd::GraphTask::ExecInfo::Capture::DO_NOT_USE_DEPRECATED_get_capture_hooks",
                "torch::autograd::GraphTask::ExecInfo::Capture::DO_NOT_USE_DEPRECATED_register_capture_hook",
                "c10::detail::IListRefTagImplBase<IListRefTag::Unboxed,T,ListElemT>",
                "c10::detail::IListRefTagImpl<IListRefTag::Unboxed,torch::Tensor>",
                "c10::IValue::TagType<c10::Type>",
                "std::conjunction<>",
                "std::disjunction<>",
                "std::numeric_limits<c10::BFloat16>",
                "torch::profiler::impl::ApproximateClockToUnixTimeConverter",
                "basic_string_view<CharT>::npos",
                "c10::impl::boxed_size_one<c10::TensorOptions>",
                "torch::detail::check_not_lvalue_references",
                "c10::guts::false_higher_t"
            ).skip());


        //// Complex
        infoMap
            .put(new Info("c10::complex<double>").pointerTypes("DoubleComplex"))
            .put(new Info("c10::complex<float>").pointerTypes("FloatComplex"))
            .put(new Info("c10::complex<c10::Half>").pointerTypes("HalfComplex"))
            .put(new Info("c10::complex<double>::real", "c10::complex<double>::imag",
                "c10::complex<float>::real", "c10::complex<float>::imag",
                "c10::complex<c10::Half>::real", "c10::complex<c10::Half>::imag").annotations("@org.bytedeco.javacpp.annotation.Function"))
            .put(new Info( // Not implemented in c10::complex<c10::Half> template specialization:
                "c10::complex<c10::Half>::operator =(c10::Half)",
                "c10::complex<c10::Half>::real(c10::Half)",
                "c10::complex<c10::Half>::imag(c10::Half)",
                "c10::complex<c10::Half>::operator const bool()",
                "c10::complex<c10::Half>::operator +=(c10::Half)",
                "c10::complex<c10::Half>::operator -=(c10::Half)",
                "c10::complex<c10::Half>::operator *=(c10::Half)",
                "c10::complex<c10::Half>::operator /=(c10::Half)"
            ).skip())
            .put(new Info("c10::complex<c10::Half>::complex(const c10::Half&, const c10::Half&)").javaText( // Second argument not optional + add specific functions
                    "public HalfComplex(Half re, Half im) { super((Pointer)null); allocate(re, im); }\n" +
                    "private native void allocate(@Const @ByRef Half re, @Const @ByRef(nullValue = \"c10::Half()\") Half im);\n" +
                    "public HalfComplex(@Const @ByRef FloatComplex value) { super((Pointer)null); allocate(value); }\n" +
                    "private native void allocate(@Const @ByRef FloatComplex value);\n" +
                    "\n" +
                    "// Conversion operator\n" +
                    "public native @ByVal @Name(\"operator c10::complex<float>\") FloatComplex asFloatComplex();\n" +
                    "\n" +
                    "public native @ByRef @Name(\"operator +=\") HalfComplex addPut(@Const @ByRef HalfComplex other);\n" +
                    "\n" +
                    "public native @ByRef @Name(\"operator -=\") HalfComplex subtractPut(@Const @ByRef HalfComplex other);\n" +
                    "\n" +
                    "public native @ByRef @Name(\"operator *=\") HalfComplex multiplyPut(@Const @ByRef HalfComplex other);"
                )
            )
        ;


        //// TypeKind
        infoMap
            .put(new Info("c10::EnumerationType<c10::TypeKind::LayoutType>").pointerTypes("LayoutEnumerationType"))
            .put(new Info("c10::EnumerationType<c10::TypeKind::ScalarTypeType>").pointerTypes("ScalarTypeEnumerationType"))
            .put(new Info("c10::EnumerationType<c10::TypeKind::MemoryFormatType>").pointerTypes("MemoryFormattEnumerationType"))
            .put(new Info("c10::SingleElementType<c10::TypeKind::AwaitType,c10::AwaitType>").pointerTypes("AwaitSingleElementType"))
            .put(new Info("c10::SingleElementType<c10::TypeKind::ListType,c10::ListType>").pointerTypes("ListSingleElementType"))
            .put(new Info("c10::SingleElementType<c10::TypeKind::RRefType,c10::RRefType>").pointerTypes("RRefSingleElementType"))
            .put(new Info("c10::SingleElementType<c10::TypeKind::FutureType,c10::FutureType>").pointerTypes("FutureSingleElementType"))
            .put(new Info("c10::SingleElementType<c10::TypeKind::AwaitType,c10::AwaitType>").pointerTypes("AwaitSingleElementType"))
        ;


        //// Jit attributes
        infoMap
            .put(new Info("torch::jit::ComplexAttr::ConstructorType", "torch::jit::ComplexAttr::ValueType").cast().pointerTypes("DoublePointer"))
            .put(new Info("torch::jit::ComplexValsAttr::ConstructorType", "torch::jit::ComplexValsAttr::ValueType").cast().pointerTypes("Pointer"))
            .put(new Info("torch::jit::FloatAttr::ConstructorType", "torch::jit::FloatAttr::ValueType").cast().valueTypes("double").pointerTypes("DoublePointer"))
            .put(new Info("torch::jit::FloatsAttr::ConstructorType", "torch::jit::FloatsAttr::ValueType").cast().pointerTypes("DoubleVector"))
            .put(new Info("torch::jit::IntAttr::ConstructorType", "torch::jit::IntAttr::ValueType").cast().valueTypes("long").pointerTypes("LongPointer"))
            .put(new Info("torch::jit::IntsAttr::ConstructorType", "torch::jit::IntsAttr::ValueType").cast().pointerTypes("LongVector"))
            .put(new Info("torch::jit::StringAttr::ConstructorType", "torch::jit::StringAttr::ValueType").annotations("@StdString").pointerTypes("BytePointer"))
            .put(new Info("torch::jit::StringsAttr::ConstructorType", "torch::jit::StringsAttr::ValueType").cast().pointerTypes("StringVector"))
            .put(new Info("torch::jit::TensorAttr::ConstructorType", "torch::jit::TensorAttr::ValueType").cast().pointerTypes("Tensor"))
            .put(new Info("torch::jit::TensorsAttr::ConstructorType", "torch::jit::TensorsAttr::ValueType").cast().pointerTypes("TensorVector"))
            .put(new Info("torch::jit::TypesAttr::ConstructorType", "torch::jit::TypesAttr::ValueType").cast().pointerTypes("TypeVector"))
            .put(new Info("torch::jit::IValueAttr::ConstructorType", "torch::jit::IValueAttr::ValueType").cast().pointerTypes("IValue"))
        ;


        //// Jit iterators
        for (String[] t : new String[][]{
            {"Module", "JitModule", "torch::jit::Module"},
            {"Parameter", "Tensor", "torch::Tensor"},
            {"Attribute", "IValue", "c10::IValue"},
            {"Buffer", "Tensor", "torch::Tensor"}
        }) {
            infoMap.put(new Info(
                       "torch::jit::slot_list_impl<torch::jit::detail::" + t[0] + "Policy>",
                       "torch::jit::" + t[0].toLowerCase() + "_list").pointerTypes(t[0].toLowerCase() + "_list"))
                   .put(new Info("torch::jit::slot_iterator_impl<torch::jit::detail::" + t[0] + "Policy>").pointerTypes(t[0].toLowerCase() + "_iterator"))
                   .put(new Info("torch::jit::slot_iterator_impl<torch::jit::detail::" + t[0] + "Policy>::value_type").pointerTypes(t[1]))
                   .put(new Info("torch::jit::Named<" + t[2] + ">").pointerTypes("Named" + t[1]))
                   .put(new Info(
                       "torch::jit::slot_list_impl<torch::jit::detail::NamedPolicy<torch::jit::detail::" + t[0] + "Policy> >",
                       "torch::jit::named_" + t[0].toLowerCase() + "_list").pointerTypes("named_" + t[0].toLowerCase() + "_list"))
                   .put(new Info("torch::jit::slot_iterator_impl<torch::jit::detail::NamedPolicy<torch::jit::detail::" + t[0] + "Policy> >").pointerTypes("named_" + t[0].toLowerCase() + "_iterator"))
                   .put(new Info("torch::jit::slot_iterator_impl<torch::jit::detail::NamedPolicy<torch::jit::detail::" + t[0] + "Policy> >::value_type").pointerTypes("Named" + t[1]))
            ;
        }

        infoMap
            .put(new Info("torch::jit::Maybe<torch::jit::Def>").pointerTypes("DefMaybe"))
            .put(new Info("torch::jit::Maybe<torch::jit::Expr>").pointerTypes("ExprMaybe"))
            .put(new Info("torch::jit::Maybe<torch::jit::Var>").pointerTypes("VarMaybe"))
            .put(new Info("torch::jit::Maybe<torch::jit::List<torch::jit::Property> >").pointerTypes("PropertyListMaybe"))
            .put(new Info("torch::jit::Maybe<torch::jit::List<torch::jit::Assign> >").pointerTypes("AssignListMaybe"))
            .put(new Info(
                "torch::jit::Compound::map",
                "torch::jit::Tree::map",
                "torch::jit::Maybe<torch::jit::Def>::map",
                "torch::jit::Maybe<torch::jit::Expr>::map",
                "torch::jit::Maybe<torch::jit::Var>::map",
                "torch::jit::Maybe<torch::jit::List<torch::jit::Assign> >::map",
                "torch::jit::Maybe<torch::jit::List<torch::jit::Property> >::map"
            ).skip()) /* Could be mapped if needed */
            .put(new Info("torch::jit::Wrap<torch::jit::Block>").pointerTypes("BlockWrap"))
            .put(new Info("torch::jit::Wrap<torch::jit::Node>").pointerTypes("JitNodeWrap"))
            .put(new Info("torch::jit::Wrap<torch::jit::Value>").pointerTypes("ValueWrap"))
        ;


        //// Data loader
        infoMap
            .put(new Info("torch::data::example::NoTarget")) // To ensure ns resolution gets it correctly
            .put(new Info(
                "torch::data::Example<torch::Tensor,torch::data::example::NoTarget>::Example"
            ).javaText(
                "public TensorExample(@ByVal Tensor data) { super((Pointer)null); allocate(data); }\n" +
                "private native void allocate(@ByVal Tensor data);\n")) /* or generated constructor will want argument "NoTarget */
            .put(new Info("torch::data::Example<torch::Tensor,torch::data::example::NoTarget>::target").skip())

            .put(new Info(
                "torch::data::samplers::Sampler<std::vector<size_t> >",
                "torch::data::samplers::Sampler<>"
            ).pointerTypes("Sampler"))
            .put(new Info(
                "torch::data::samplers::Sampler<torch::data::samplers::BatchSize>"
            ).pointerTypes("BatchSizeSampler"))
            .put(new Info(
                "torch::data::samplers::RandomSampler"
            ).pointerTypes("RandomSampler"))
            .put(new Info(
                "torch::data::samplers::DistributedSampler<std::vector<size_t> >",
                "torch::data::samplers::DistributedSampler<>"
            ).purify().pointerTypes("DistributedSampler"))
            .put(new Info(
                "const std::optional<torch::data::samplers::BatchSize>", "std::optional<torch::data::samplers::BatchSize>"
            ).pointerTypes("BatchSizeOptional").define())

            .put(new Info("torch::data::DataLoaderBase<torch::data::datasets::MapDataset<torch::data::datasets::MNIST,torch::data::transforms::Stack<torch::data::Example<torch::Tensor,torch::Tensor> > >,torch::data::Example<torch::Tensor,torch::Tensor>,std::vector<size_t> >",
                "torch::data::DataLoaderBase<torch::data::datasets::MapDataset<torch::data::datasets::MNIST,torch::data::transforms::Stack<torch::data::Example<torch::Tensor,torch::Tensor> > >,torch::data::datasets::MapDataset<torch::data::datasets::MNIST,torch::data::transforms::Stack<torch::data::Example<torch::Tensor,torch::Tensor> > >::BatchType,torch::data::samplers::RandomSampler::BatchRequestType>")
                .purify().pointerTypes("MNISTRandomDataLoaderBase"))
            .put(new Info("torch::data::StatelessDataLoader<torch::data::datasets::MapDataset<torch::data::datasets::MNIST,torch::data::transforms::Stack<torch::data::Example<torch::Tensor,torch::Tensor> > >,torch::data::samplers::RandomSampler>").pointerTypes("MNISTRandomDataLoader"))
            .put(new Info("torch::data::datasets::Dataset<torch::data::datasets::MNIST,torch::data::Example<torch::Tensor,torch::Tensor> >",
                "torch::data::datasets::Dataset<MNIST>").pointerTypes("MNISTDataset"))
            .put(new Info("torch::data::datasets::BatchDataset<torch::data::datasets::MNIST,std::vector<torch::data::Example<torch::Tensor,torch::Tensor> >,at::ArrayRef<size_t> >",
                "torch::data::datasets::BatchDataset<torch::data::datasets::MNIST,std::vector<torch::data::Example<torch::Tensor,torch::Tensor> > >").pointerTypes("MNISTBatchDataset"))
            .put(new Info("torch::data::datasets::BatchDataset<torch::data::datasets::MNIST,std::vector<torch::data::Example<torch::Tensor,torch::Tensor> >,at::ArrayRef<size_t> >::map")
                .javaText("public native @ByVal MNISTMapDataset map(@ByVal ExampleStack transform);"))
//               .put(new Info("torch::data::datasets::BatchDataset<torch::data::datasets::MNIST,std::vector<torch::data::Example<> >,at::ArrayRef<size_t> >::map<torch::data::transforms::Stack<torch::data::Example<> > >")
//                       .javaNames("map"))
            .put(new Info("torch::data::datasets::MapDataset<torch::data::datasets::MNIST,torch::data::transforms::Stack<torch::data::Example<torch::Tensor,torch::Tensor> > >").pointerTypes("MNISTMapDataset"))
            .put(new Info("torch::data::datasets::MapDataset<torch::data::datasets::MNIST,torch::data::transforms::Stack<torch::data::Example<torch::Tensor,torch::Tensor> > >::reset").skip())
            .put(new Info("torch::data::datasets::MapDataset<torch::data::datasets::MNIST,torch::data::transforms::Stack<torch::data::Example<torch::Tensor,torch::Tensor> > >::DatasetType").pointerTypes("MNIST"))
            .put(new Info("torch::data::datasets::BatchDataset<torch::data::datasets::MapDataset<torch::data::datasets::MNIST,torch::data::transforms::Stack<torch::data::Example<torch::Tensor,torch::Tensor> > >,std::vector<torch::data::Example<torch::Tensor,torch::Tensor> >,at::ArrayRef<size_t> >",
                "torch::data::datasets::BatchDataset<torch::data::datasets::MapDataset<torch::data::datasets::MNIST,torch::data::transforms::Stack<torch::data::Example<torch::Tensor,torch::Tensor> > >,torch::data::datasets::detail::optional_if_t<torch::data::datasets::MNIST::is_stateful,torch::data::transforms::Stack<torch::data::Example<torch::Tensor,torch::Tensor> >::OutputBatchType>,torch::data::datasets::MNIST::BatchRequestType>")
                .pointerTypes("MNISTMapBatchDataset"))
//               .put(new Info("torch::data::datasets::MapDataset<torch::data::datasets::MNIST,torch::data::transforms::Stack<torch::data::Example<> > >::BatchRequestType").pointerTypes("SizeTArrayRef"))
//               .put(new Info("torch::data::datasets::MapDataset<torch::data::datasets::MNIST,torch::data::transforms::Stack<torch::data::Example<> > >::OutputBatchType").pointerTypes("Example"))
            .put(new Info("torch::data::datasets::MapDataset<torch::data::datasets::MNIST,torch::data::transforms::Stack<torch::data::Example<torch::Tensor,torch::Tensor> > >::get_batch")
                .javaText("public native @Name(\"get_batch\") @ByVal Example get_batch_example(@ByVal SizeTArrayRef indices);\n" +
                          "public native @Name(\"get_batch\") @ByVal Example get_batch_example(@ByVal @Cast({\"size_t*\", \"c10::ArrayRef<size_t>\", \"std::vector<size_t>&\"}) @StdVector long... indices);"))

            // Simple implementation from tensor.h serving a dataset from a single tensor
            .put(new Info("torch::data::datasets::TensorDataset")) // Ensure proper ns resolution
            .put(new Info(
                "torch::data::datasets::Dataset<torch::data::datasets::TensorDataset,torch::data::TensorExample>"
            ).pointerTypes("TensorDatasetBase"))
            .put(new Info(
                "torch::data::datasets::BatchDataset<torch::data::datasets::TensorDataset,std::vector<torch::data::TensorExample> >"
            ).pointerTypes("TensorBatchDataset"))
            .put(new Info("torch::data::datasets::Dataset<torch::data::datasets::TensorDataset,torch::data::TensorExample>::get_batch",
                "torch::data::datasets::BatchDataset<torch::data::datasets::TensorDataset,std::vector<torch::data::TensorExample> >::get_batch")
                .javaText("public native @ByVal TensorExampleVector get_batch(@ByVal SizeTArrayRef request);\n" +
                          "public native @ByVal TensorExampleVector get_batch(@ByVal @Cast({\"size_t*\", \"c10::ArrayRef<size_t>\", \"std::vector<size_t>&\"}) @StdVector(\"size_t\") long... request);"))
        ;

        for (String[] ex : new String[][]{
            /* Prefix, Data, Target */
            {"", "torch::Tensor", "torch::Tensor"},
            {"Tensor", "torch::Tensor", "torch::data::example::NoTarget"}
        }) {
            String example = ex[2] == null ? template("torch::data::Example", ex[1]) : template("torch::data::Example", ex[1], ex[2]);
            String p = ex[0];
            String chunkDataReader = template("torch::data::datasets::ChunkDataReader", example, template("std::vector", example));
            String mangledChunkDataReader = mangle(chunkDataReader);
            String mangledJavaDataset = mangle(template("javacpp::Dataset", ex[1], ex[2]));
            String mangledJavaStreamDataset = mangle(template("javacpp::StreamDataset", ex[1], ex[2]));
            String mangledJavaStatefulDataset = mangle(template("javacpp::StatefulDataset", ex[1], ex[2]));

            infoMap
                .put(new Info(
                    example,
                    template("torch::data::datasets::MapDataset", template("torch::data::datasets::SharedBatchDataset", template("torch::data::datasets::ChunkDataset", mangledChunkDataReader, "torch::data::samplers::RandomSampler", "torch::data::samplers::RandomSampler")), template("torch::data::transforms::Stack", example)) + "::OutputBatchType"
                ).pointerTypes(p + "Example"))
                .put(new Info(
                    template("std::vector", example),
                    template("std::vector", template("torch::data::datasets::Dataset", template("javacpp::Dataset", ex[1], ex[2]), example) + "::ExampleType"),
                    template("std::vector", template("torch::data::datasets::Dataset", template("javacpp::StreamDataset", ex[1], ex[2]), example) + "::ExampleType"),
                    template("std::vector", template("torch::data::datasets::Dataset", template("javacpp::StatefulDataset", ex[1], ex[2]), example) + "::ExampleType"),
                    template("std::vector", template("torch::data::datasets::Dataset", mangledJavaDataset, example) + "::ExampleType"),
                    template("std::vector", template("torch::data::datasets::Dataset", mangledJavaStreamDataset, example) + "::ExampleType"),
                    template("std::vector", template("torch::data::datasets::Dataset", mangledJavaStatefulDataset, example) + "::ExampleType")
                ).pointerTypes(p + "ExampleVector").define())
                .put(new Info(template("std::optional", example)).pointerTypes(p + "ExampleOptional").define())
                .put(new Info(
                    template("std::optional", template("std::vector", example)),
                    template("std::optional", mangledChunkDataReader + "::BatchType"),
                    template("torch::data::datasets::ChunkDataset", mangledChunkDataReader, "torch::data::samplers::RandomSampler", "torch::data::samplers::RandomSampler") + "::BatchType",
                    mangledJavaStreamDataset + "::BatchType"
                ).pointerTypes(p + "ExampleVectorOptional").define())
                .put(new Info(
                    template("torch::data::Iterator", example),
                    template("torch::data::Iterator", mangledJavaDataset + "::BatchType::value_type")
                ).pointerTypes(p + "ExampleIterator").purify())
                .put(new Info(
                    template("torch::data::Iterator", template("std::vector", example)),
                    template("torch::data::Iterator", mangledJavaDataset + "::BatchType"),
                    template("torch::data::Iterator", mangledJavaStreamDataset + "::BatchType"),
                    template("torch::data::Iterator", mangledJavaStatefulDataset + "::BatchType::value_type")
                ).purify().pointerTypes(p + "ExampleVectorIterator"))

                .put(new Info(
                    template("torch::data::transforms::BatchTransform", template("std::vector", example), example),
                    template("torch::data::transforms::Collation", example)
                ).pointerTypes(p + "ExampleCollation"))
                // See explicit definition of ExampleStack and TensorExampleStack.
                .put(new Info(template("torch::data::transforms::Stack", example)).pointerTypes(p + "ExampleStack"))
                .put(new Info(chunkDataReader).pointerTypes("Chunk" + p + "DataReader").virtualize())
                .put(new Info(
                    template("torch::data::datasets::ChunkDataset", mangledChunkDataReader, "torch::data::samplers::RandomSampler", "torch::data::samplers::RandomSampler")
                ).pointerTypes("Chunk" + p + "Dataset"))
                .put(new Info(
                    template("torch::data::datasets::ChunkDataset", mangledChunkDataReader, "torch::data::samplers::RandomSampler", "torch::data::samplers::RandomSampler") + "::ChunkDataset"
                ).javaText(
                    "public Chunk" + p + "Dataset(\n"
                    + "      Chunk" + p + "DataReader chunk_reader,\n"
                    + "      RandomSampler chunk_sampler,\n"
                    + "      RandomSampler example_sampler,\n"
                    + "      ChunkDatasetOptions options) { super((Pointer)null); allocate(chunk_reader, chunk_sampler, example_sampler, options, null); }\n"
                    + "public Chunk" + p + "Dataset(\n"
                    + "      Chunk" + p + "DataReader chunk_reader,\n"
                    + "      RandomSampler chunk_sampler,\n"
                    + "      RandomSampler example_sampler,\n"
                    + "      ChunkDatasetOptions options,\n"
                    + "      Pointer preprocessing_policy) { super((Pointer)null); allocate(chunk_reader, chunk_sampler, example_sampler, options, preprocessing_policy); }\n"
                    + "private native void allocate(\n"
                    + "      @ByVal @Cast(\"" + mangledChunkDataReader + "*\") Chunk" + p + "DataReader chunk_reader,\n"
                    + "      @ByVal RandomSampler chunk_sampler,\n"
                    + "      @ByVal RandomSampler example_sampler,\n"
                    + "      @ByVal ChunkDatasetOptions options,\n"
                    + "      @ByVal(nullValue = \"std::function<void(std::vector<" + example + ">&)>()\") @Cast(\"std::function<void(std::vector<" + example + ">&)>*\") Pointer preprocessing_policy);\n"))
                .put(new Info(
                    template("torch::data::datasets::StatefulDataset", template("torch::data::datasets::ChunkDataset", mangledChunkDataReader, "torch::data::samplers::RandomSampler", "torch::data::samplers::RandomSampler"), mangledChunkDataReader + "::BatchType", "size_t")
                ).pointerTypes("ChunkStateful" + p + "Dataset"))
                .put(new Info(
                    template("torch::data::datasets::BatchDataset", template("torch::data::datasets::ChunkDataset", mangledChunkDataReader, "torch::data::samplers::RandomSampler", "torch::data::samplers::RandomSampler"), template("std::optional", mangledChunkDataReader + "::BatchType"), "size_t"),
                    template("torch::data::datasets::BatchDataset", template("torch::data::datasets::ChunkDataset", mangledChunkDataReader, "torch::data::samplers::RandomSampler", "torch::data::samplers::RandomSampler"), template("std::vector", example))
                ).pointerTypes("Chunk" + p + "BatchDataset"))
                .put(new Info(
                    template("torch::data::datasets::BatchDataset", template("torch::data::datasets::SharedBatchDataset", template("torch::data::datasets::ChunkDataset", mangledChunkDataReader, "torch::data::samplers::RandomSampler", "torch::data::samplers::RandomSampler")), template("std::optional", mangledChunkDataReader + "::BatchType"), "size_t"),
                    template("torch::data::datasets::BatchDataset", template("torch::data::datasets::SharedBatchDataset", template("torch::data::datasets::ChunkDataset", mangledChunkDataReader, "torch::data::samplers::RandomSampler", "torch::data::samplers::RandomSampler")), template("torch::data::datasets::ChunkDataset", mangledChunkDataReader, "torch::data::samplers::RandomSampler", "torch::data::samplers::RandomSampler") + "::BatchType", template("torch::data::datasets::ChunkDataset", mangledChunkDataReader, "torch::data::samplers::RandomSampler", "torch::data::samplers::RandomSampler") + "::BatchRequestType")
                ).pointerTypes("ChunkBatchShared" + p + "BatchDataset"))
                .put(new Info(
                    template("torch::data::datasets::BatchDataset", template("torch::data::datasets::SharedBatchDataset", template("torch::data::datasets::ChunkDataset", mangledChunkDataReader, "torch::data::samplers::RandomSampler", "torch::data::samplers::RandomSampler")), template("std::optional", mangledChunkDataReader + "::BatchType"), "size_t") + "::map"
                ).javaText("public native @ByVal ChunkMap" + p + "Dataset map(@ByVal " + p + "ExampleStack transform);"))
                .put(new Info(
                    template("torch::data::datasets::SharedBatchDataset", template("torch::data::datasets::ChunkDataset", mangledChunkDataReader, "torch::data::samplers::RandomSampler", "torch::data::samplers::RandomSampler"))
                ).pointerTypes("ChunkShared" + p + "BatchDataset"))
                .put(new Info(
                    template("torch::data::datasets::MapDataset", template("torch::data::datasets::SharedBatchDataset", template("torch::data::datasets::ChunkDataset", mangledChunkDataReader, "torch::data::samplers::RandomSampler", "torch::data::samplers::RandomSampler")), template("torch::data::transforms::Stack", example))
                ).pointerTypes("ChunkMap" + p + "Dataset"))
                .put(new Info(
                    template("torch::data::datasets::MapDataset", template("torch::data::datasets::SharedBatchDataset", template("torch::data::datasets::ChunkDataset", mangledChunkDataReader, "torch::data::samplers::RandomSampler", "torch::data::samplers::RandomSampler")), template("torch::data::transforms::Stack", example)) + "::reset"
                ).skip())
                .put(new Info(
                    template("torch::data::datasets::MapDataset", template("torch::data::datasets::SharedBatchDataset", template("torch::data::datasets::ChunkDataset", mangledChunkDataReader, "torch::data::samplers::RandomSampler", "torch::data::samplers::RandomSampler")), template("torch::data::transforms::Stack", example)) + "::DatasetType"
                ).pointerTypes("ChunkShared" + p + "BatchDataset"))
                .put(new Info(
                    template("torch::data::datasets::BatchDataset", template("torch::data::datasets::MapDataset", template("torch::data::datasets::SharedBatchDataset", template("torch::data::datasets::ChunkDataset", mangledChunkDataReader, "torch::data::samplers::RandomSampler", "torch::data::samplers::RandomSampler")), template("torch::data::transforms::Stack", example)), template("std::vector", example), "at::ArrayRef<size_t>"),
                    template("torch::data::datasets::BatchDataset", template("torch::data::datasets::MapDataset", template("torch::data::datasets::SharedBatchDataset", template("torch::data::datasets::ChunkDataset", mangledChunkDataReader, "torch::data::samplers::RandomSampler", "torch::data::samplers::RandomSampler")), template("torch::data::transforms::Stack", example)), template("torch::data::datasets::detail::optional_if_t", template("torch::data::datasets::SharedBatchDataset", template("torch::data::datasets::ChunkDataset", mangledChunkDataReader, "torch::data::samplers::RandomSampler", "torch::data::samplers::RandomSampler")) + "::is_stateful", template("torch::data::transforms::Stack", example) + "::OutputBatchType"), template("torch::data::datasets::SharedBatchDataset", template("torch::data::datasets::ChunkDataset", mangledChunkDataReader, "torch::data::samplers::RandomSampler", "torch::data::samplers::RandomSampler")) + "::BatchRequestType")
                ).pointerTypes("ChunkMap" + p + "BatchDataset"))
                .put(new Info(
                    template("torch::data::datasets::MapDataset", template("torch::data::datasets::SharedBatchDataset", template("torch::data::datasets::ChunkDataset", mangledChunkDataReader, "torch::data::samplers::RandomSampler", "torch::data::samplers::RandomSampler")), template("torch::data::transforms::Stack", example)) + "::BatchRequestType",
                    template("torch::data::datasets::BatchDataset", mangledJavaDataset, template("std::vector", example)) + "::BatchRequest",
                    template("torch::data::datasets::BatchDataset", template("javacpp::Dataset", ex[1], ex[2]), template("std::vector", example)) + "::BatchRequest"
                ).pointerTypes("SizeTArrayRef", "@Cast({\"size_t*\", \"c10::ArrayRef<size_t>\", \"std::vector<size_t>&\"}) @StdVector(\"size_t\") long..."))
                .put(new Info(
                    template("torch::data::datasets::MapDataset", template("torch::data::datasets::SharedBatchDataset", template("torch::data::datasets::ChunkDataset", mangledChunkDataReader, "torch::data::samplers::RandomSampler", "torch::data::samplers::RandomSampler")), template("torch::data::transforms::Stack", example)) + "::get_batch"
                ).javaText("public native @Name(\"get_batch\") @ByVal " + p + "ExampleOptional get_batch_example(@Cast(\"size_t\") long indices);"))
                .put(new Info(
                    template("torch::data::DataLoaderBase", template("torch::data::datasets::MapDataset", template("torch::data::datasets::SharedBatchDataset", template("torch::data::datasets::ChunkDataset", mangledChunkDataReader, "torch::data::samplers::RandomSampler", "torch::data::samplers::RandomSampler")), template("torch::data::transforms::Stack", example)), example, "size_t"),
                    template("torch::data::DataLoaderBase", template("torch::data::datasets::MapDataset", template("torch::data::datasets::SharedBatchDataset", template("torch::data::datasets::ChunkDataset", mangledChunkDataReader, "torch::data::samplers::RandomSampler", "torch::data::samplers::RandomSampler")), template("torch::data::transforms::Stack", example)), template("torch::data::datasets::MapDataset", template("torch::data::datasets::SharedBatchDataset", template("torch::data::datasets::ChunkDataset", mangledChunkDataReader, "torch::data::samplers::RandomSampler", "torch::data::samplers::RandomSampler")), template("torch::data::transforms::Stack", example)) + "::BatchType::value_type", template("torch::data::datasets::MapDataset", template("torch::data::datasets::SharedBatchDataset", template("torch::data::datasets::ChunkDataset", mangledChunkDataReader, "torch::data::samplers::RandomSampler", "torch::data::samplers::RandomSampler")), template("torch::data::transforms::Stack", example)) + "::BatchRequestType")
                ).purify().pointerTypes("ChunkRandom" + p + "DataLoaderBase"))
                .put(new Info(
                    template("torch::data::StatefulDataLoader", template("torch::data::datasets::MapDataset", template("torch::data::datasets::SharedBatchDataset", template("torch::data::datasets::ChunkDataset", mangledChunkDataReader, "torch::data::samplers::RandomSampler", "torch::data::samplers::RandomSampler")), template("torch::data::transforms::Stack", example)))
                ).pointerTypes("ChunkRandom" + p + "DataLoader"))

                .put(new Info(
                    template("torch::data::datasets::BatchDataset", template("javacpp::Dataset", ex[1], ex[2]), template("std::vector", example))
                ).pointerTypes("Java" + p + "BatchDataset"))
                .put(new Info(
                    template("torch::data::datasets::Dataset", template("javacpp::Dataset", ex[1], ex[2]), example)
                ).pointerTypes("Java" + p + "DatasetBase").purify())
                .put(new Info(
                    template("torch::data::StatelessDataLoader", mangledJavaDataset, "torch::data::samplers::RandomSampler")
                ).pointerTypes("JavaRandom" + p + "DataLoader"))
                .put(new Info(
                    template("torch::data::DataLoaderBase", mangledJavaDataset, mangledJavaDataset + "::BatchType", "torch::data::samplers::RandomSampler::BatchRequestType")
                ).pointerTypes("JavaRandom" + p + "DataLoaderBase").purify())
                .put(new Info(
                    template("torch::data::StatelessDataLoader", mangledJavaDataset, "torch::data::samplers::DistributedRandomSampler")
                ).pointerTypes("JavaDistributedRandom" + p + "DataLoader"))
                .put(new Info(
                    template("torch::data::DataLoaderBase", mangledJavaDataset, mangledJavaDataset + "::BatchType", "torch::data::samplers::DistributedRandomSampler::BatchRequestType")
                ).pointerTypes("JavaDistributedRandom" + p + "DataLoaderBase").purify())
                .put(new Info(
                    template("torch::data::StatelessDataLoader", mangledJavaDataset, "torch::data::samplers::DistributedSequentialSampler")
                ).pointerTypes("JavaDistributedSequential" + p + "DataLoader"))
                .put(new Info(
                    template("torch::data::DataLoaderBase", mangledJavaDataset, mangledJavaDataset + "::BatchType", "torch::data::samplers::DistributedSequentialSampler::BatchRequestType")
                ).pointerTypes("JavaDistributedSequential" + p + "DataLoaderBase").purify())
                .put(new Info(
                    template("torch::data::StatelessDataLoader", mangledJavaDataset, "torch::data::samplers::SequentialSampler")
                ).pointerTypes("JavaSequential" + p + "DataLoader"))
                .put(new Info(
                    template("torch::data::DataLoaderBase", mangledJavaDataset, mangledJavaDataset + "::BatchType", "torch::data::samplers::SequentialSampler::BatchRequestType")
                ).pointerTypes("JavaSequential" + p + "DataLoaderBase").purify())
                .put(new Info(
                    template("torch::data::datasets::BatchDataset", template("javacpp::StreamDataset", ex[1], ex[2]), template("std::vector", example), "size_t")
                ).pointerTypes("JavaStream" + p + "BatchDataset"))
                .put(new Info(
                    template("torch::data::StatelessDataLoader", mangledJavaStreamDataset, "torch::data::samplers::StreamSampler")
                ).pointerTypes("JavaStream" + p + "DataLoader"))
                .put(new Info(
                    template("torch::data::DataLoaderBase", mangledJavaStreamDataset, mangledJavaStreamDataset + "::BatchType", "torch::data::samplers::StreamSampler::BatchRequestType")
                ).pointerTypes("JavaStream" + p + "DataLoaderBase").purify())

                .put(new Info(
                    template("javacpp::Dataset", ex[1], ex[2])
                ).pointerTypes("Java" + p + "Dataset").virtualize())
                .put(new Info(
                    mangledJavaDataset
                ).pointerTypes("@Cast(\"" + mangledJavaDataset + "*\") Java" + p + "Dataset"))
                .put(new Info(
                    template("javacpp::StreamDataset", ex[1], ex[2])
                ).pointerTypes("JavaStream" + p + "Dataset").virtualize())
                .put(new Info(
                    mangledJavaStreamDataset
                ).pointerTypes("@Cast(\"" + mangledJavaStreamDataset + "*\") JavaStream" + p + "Dataset"))
                .put(new Info(
                    template("javacpp::StatefulDataset", ex[1], ex[2])
                ).pointerTypes("JavaStateful" + p + "Dataset").virtualize())
                .put(new Info(
                    mangledJavaStatefulDataset
                ).pointerTypes("@Cast(\"" + mangledJavaStatefulDataset + "*\") JavaStateful" + p + "Dataset"))
                .put(new Info(
                    template("torch::data::datasets::StatefulDataset", template("javacpp::StatefulDataset", ex[1], ex[2]), template("std::vector", example), "size_t")
                ).pointerTypes("JavaStateful" + p + "DatasetBase").purify())
                .put(new Info(
                    template("torch::data::StatefulDataLoader", mangledJavaStatefulDataset)
                ).pointerTypes("JavaStateful" + p + "DataLoader"))
                .put(new Info(
                    template("torch::data::DataLoaderBase", mangledJavaStatefulDataset, mangledJavaStatefulDataset + "::BatchType::value_type", mangledJavaStatefulDataset + "::BatchRequestType")
                ).pointerTypes("JavaStateful" + p + "DataLoaderBase").purify())
                .put(new Info(
                    template("torch::data::datasets::BatchDataset", template("javacpp::StatefulDataset", ex[1], ex[2]), template("std::optional", template("std::vector", example)), "size_t")
                ).pointerTypes("JavaStateful" + p + "BatchDataset").purify())
            ;
        }
        addCppName(infoMap,
            "std::vector<torch::data::Example<torch::Tensor,torch::Tensor> >",
            "std::vector<torch::data::datasets::Dataset<torch::data::datasets::MNIST,torch::data::Example<torch::Tensor,torch::Tensor> >::ExampleType>");

        // Because explicitly defined in stack.h
        addCppName(infoMap,
            "torch::data::Example<torch::Tensor,torch::Tensor>",
            "torch::data::Example<>");
        addCppName(infoMap,
            "torch::data::transforms::Stack<torch::data::Example<torch::Tensor,torch::Tensor> >",
            "torch::data::transforms::Stack<torch::data::Example<> >");
        addCppName(infoMap,
            "torch::data::transforms::Stack<torch::data::Example<torch::Tensor,torch::data::example::NoTarget> >",
            "torch::data::transforms::Stack<torch::data::TensorExample>");
        addCppName(infoMap,
            "torch::data::transforms::Collation<torch::data::Example<torch::Tensor,torch::Tensor>,std::vector<torch::data::Example<torch::Tensor,torch::Tensor> > >",
            "torch::data::transforms::Collation<torch::data::Example<> >");


        //// Tensor factories
        String[] factories = {"_cudnn_init_dropout_state", "arange", "bartlett_window", "blackman_window", "empty", "_empty_affine_quantized",
            "_empty_per_channel_affine_quantized", "empty_quantized", "empty_like", "empty_strided", "eye", "full", "full_like", "from_file",
            "hann_window", "hamming_window", "kaiser_window", "linspace", "logspace", "ones", "ones_like", "scalar_tensor", "rand", "rand_like",
            "randint", "randint_like", "randn", "randn_like", "randperm", "range", "zeros", "_efficientzerotensor", "zeros_like",
            "sparse_compressed_tensor", "sparse_csr_tensor", "sparse_csc_tensor", "sparse_bsr_tensor", "sparse_bsc_tensor",
            "_sparse_compressed_tensor_unsafe", "_sparse_csr_tensor_unsafe", "_sparse_csc_tensor_unsafe", "_sparse_bsr_tensor_unsafe", "_sparse_bsc_tensor_unsafe",
            "sparse_coo_tensor", "_sparse_coo_tensor_unsafe", "_sparse_coo_tensor_with_dims", "_sparse_coo_tensor_with_dims_and_tensors",
            "_to_copy", "tril_indices", "triu_indices", "normal", "fft_fftfreq", "fft_rfftfreq"};
        for (String factory : factories) {
            infoMap.put(new Info("torch::" + factory).javaNames("torch_" + factory).skipDefaults(factory.equals("range")))
                   .put(new Info("torch::autograd::" + factory))
                   .put(new Info("torch::jit::" + factory))
                   .put(new Info("torch::nn::" + factory));
        }


        //// Module options
        infoMap
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
            .put(new Info("torch::nn::ZeroPadOptions<1>").pointerTypes("ZeroPad1dOptions"))
            .put(new Info("torch::nn::ZeroPadOptions<2>").pointerTypes("ZeroPad2dOptions"))
            .put(new Info("torch::nn::ZeroPadOptions<3>").pointerTypes("ZeroPad3dOptions"))
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
        ;

        //// Modules
        infoMap
            // Mimic C++ register_module and return the subclass instance. Also keep API compatibility with
            // presets before 2.0.1 where register_module template was instantiated for all known
            // native subclasses.
            .put(new Info("torch::nn::Module::register_module<torch::nn::Module>").javaText(
                "private native @Name(\"register_module<torch::nn::Module>\") void _register_module(@StdString BytePointer name, @SharedPtr @ByVal Module module);\n" +
                "public <M extends Module> M register_module(BytePointer name, M module) { _register_module(name, module); return module; }\n" +
                "private native @Name(\"register_module<torch::nn::Module>\") void _register_module(@StdString String name, @SharedPtr @ByVal Module module);\n" +
                "public <M extends Module> M register_module(String name, M module) { _register_module(name, module); return module; }"
            ))
        ;
        String[] virtuals = {"train", "is_training", "to", "zero_grad", "save", "load", "pretty_print", "is_serializable"};
        for (String m : virtuals)
            infoMap.put(new Info("torch::nn::Module::" + m).virtualize().annotations("@Virtual(subclasses=false, method=\"" + m + "\")"));

        // clone returns a std::shared_ptr<Module> and not a Module.
        // This cast is normally added automatically by Parser but the info on shared_ptr<Module> prevents this (issue #670)
        // The second value of @Cast is used for the return type
        infoMap.put(new Info("torch::nn::Module::clone")
            .virtualize()
            .annotations("@Virtual(subclasses=false, method=\"clone\")", "@Cast({\"\", \"std::shared_ptr<torch::nn::Module>\"})"));

        mapModule(infoMap, "ModuleDict", false);
        mapModule(infoMap, "ModuleList", false);
        mapModule(infoMap, "Sequential", false);
        mapModule(infoMap, "ParameterDict", false);
        mapModule(infoMap, "ParameterList", false);

        mapModule(infoMap, "AdaptiveLogSoftmaxWithLoss");

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
            mapModule(infoMap, "ZeroPad" + i + "d", "torch::nn::ZeroPadImpl<" + i + ",torch::nn::ZeroPad" + i + "dImpl>");
            mapModule(infoMap, "AvgPool" + i + "d", "torch::nn::AvgPoolImpl<" + i + ",torch::nn::AvgPool" + i + "dImpl>");
            mapModule(infoMap, "MaxPool" + i + "d", "torch::nn::MaxPoolImpl<" + i + ",torch::nn::MaxPool" + i + "dImpl>");
            mapModule(infoMap, "AdaptiveAvgPool" + i + "d", "torch::nn::AdaptiveAvgPoolImpl<" + i + ",torch::ExpandingArray" + (i > 1 ? "WithOptionalElem<" : "<") + i + ">,torch::nn::AdaptiveAvgPool" + i + "dImpl>");
            mapModule(infoMap, "AdaptiveMaxPool" + i + "d", "torch::nn::AdaptiveMaxPoolImpl<" + i + ",torch::ExpandingArray" + (i > 1 ? "WithOptionalElem<" : "<") + i + ">,torch::nn::AdaptiveMaxPool" + i + "dImpl>");
            mapModule(infoMap, "MaxUnpool" + i + "d", "torch::nn::MaxUnpoolImpl<" + i + ",torch::nn::MaxUnpool" + i + "dImpl>");
            if (i > 1) {
                mapModule(infoMap, "FractionalMaxPool" + i + "d", "torch::nn::FractionalMaxPoolImpl<" + i + ",torch::nn::FractionalMaxPool" + i + "dImpl>");
            }
            mapModule(infoMap, "LPPool" + i + "d", "torch::nn::LPPoolImpl<" + i + ",torch::nn::LPPool" + i + "dImpl>");
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


        //// AnyModule, AnyValue and Sequential
        infoMap
            // All forward variants of native modules
            .put(new Info("torch::nn::AnyModule::any_forward").javaText(
                "public native @ByVal AnyValue any_forward(@Const @ByRef AnyValue input);\n" +
                "public native @ByVal AnyValue any_forward(@Const @ByRef Tensor input);\n" +
                "public native @ByVal AnyValue any_forward(@Const @ByRef Tensor input1, @Const @ByRef Tensor input2);\n" +
                "public native @ByVal AnyValue any_forward(@Const @ByRef Tensor input1, @Const @ByRef Tensor input2, @Const @ByRef Tensor input3);\n" +
                "public native @ByVal AnyValue any_forward(@Const @ByRef Tensor input1, @Const @ByRef Tensor input2, @Const @ByRef Tensor input3, @Const @ByRef Tensor input4);\n" +
                "public native @ByVal AnyValue any_forward(@Const @ByRef Tensor input1, @Const @ByRef Tensor input2, @Const @ByRef Tensor input3, @Const @ByRef Tensor input4, @Const @ByRef Tensor input5, @Const @ByRef Tensor input6);\n" +
                "public native @ByVal AnyValue any_forward(@Const @ByRef Tensor input1, @Const @ByRef Tensor input2, @Const @ByRef Tensor input3, @Const @ByRef Tensor input4, @Const @ByRef Tensor input5, @Const @ByRef Tensor input6, @Const @ByRef Tensor input7, @Const @ByRef Tensor input8);\n" +
                "public native @ByVal AnyValue any_forward(@Const @ByRef Tensor input, @ByRef(nullValue = \"std::optional<at::IntArrayRef>(c10::nullopt)\") @Cast({\"int64_t*\", \"c10::ArrayRef<int64_t>\", \"std::vector<int64_t>&\"}) @StdVector long... output_size);\n" +
                "public native @ByVal AnyValue any_forward(@Const @ByRef Tensor input, @Const @ByRef(nullValue = \"std::optional<at::IntArrayRef>(c10::nullopt)\") LongArrayRefOptional output_size);\n" +
                "public native @ByVal AnyValue any_forward(@Const @ByRef Tensor input, @Const @ByRef Tensor indices, @Const @ByRef(nullValue = \"std::optional<std::vector<int64_t> >(c10::nullopt)\") LongVectorOptional output_size);\n" +
                "public native @ByVal AnyValue any_forward(@Const @ByRef Tensor input, @ByVal(nullValue = \"torch::optional<std::tuple<torch::Tensor,torch::Tensor> >{}\") T_TensorTensor_TOptional hx_opt);\n" +
                "public native @ByVal AnyValue any_forward(@Const @ByRef Tensor query, @Const @ByRef Tensor key, @Const @ByRef Tensor value, @Const @ByRef(nullValue = \"torch::Tensor{}\") Tensor key_padding_mask, @Cast(\"bool\") boolean need_weights/*=true*/, @Const @ByRef(nullValue = \"torch::Tensor{}\") Tensor attn_mask, @Cast(\"bool\") boolean average_attn_weights/*=true*/);\n"
            ))
            .put(new Info("torch::nn::AnyModule::forward", "torch::nn::SequentialImpl::forward").javaText(
                "public native @ByVal Tensor forward(@Const @ByRef Tensor input);\n" +
                "public native @ByVal Tensor forward(@Const @ByRef Tensor input1, @Const @ByRef Tensor input2);\n" +
                "public native @ByVal Tensor forward(@Const @ByRef Tensor input1, @Const @ByRef Tensor input2, @Const @ByRef Tensor input3);\n" +
                "public native @ByVal Tensor forward(@Const @ByRef Tensor input1, @Const @ByRef Tensor input2, @Const @ByRef Tensor input3, @Const @ByRef Tensor input4);\n" +
                "public native @ByVal Tensor forward(@Const @ByRef Tensor input1, @Const @ByRef Tensor input2, @Const @ByRef Tensor input3, @Const @ByRef Tensor input4, @Const @ByRef Tensor input5, @Const @ByRef Tensor input6);\n" +
                "public native @ByVal Tensor forward(@Const @ByRef Tensor input1, @Const @ByRef Tensor input2, @Const @ByRef Tensor input3, @Const @ByRef Tensor input4, @Const @ByRef Tensor input5, @Const @ByRef Tensor input6, @Const @ByRef Tensor input7, @Const @ByRef Tensor input8);\n" +
                "public native @ByVal Tensor forward(@Const @ByRef Tensor input, @ByRef(nullValue = \"std::optional<at::IntArrayRef>(c10::nullopt)\") @Cast({\"int64_t*\", \"c10::ArrayRef<int64_t>\", \"std::vector<int64_t>&\"}) @StdVector long... output_size);\n" +
                "public native @ByVal Tensor forward(@Const @ByRef Tensor input, @Const @ByRef(nullValue = \"std::optional<at::IntArrayRef>(c10::nullopt)\") LongArrayRefOptional output_size);\n" +
                "public native @ByVal Tensor forward(@Const @ByRef Tensor input, @Const @ByRef Tensor indices, @Const @ByRef(nullValue = \"std::optional<std::vector<int64_t> >(c10::nullopt)\") LongVectorOptional output_size);\n" +
                "public native @ByVal @Name(\"forward<std::tuple<torch::Tensor,std::tuple<torch::Tensor,torch::Tensor>>>\") T_TensorT_TensorTensor_T_T forwardT_TensorT_TensorTensor_T_T(@Const @ByRef Tensor input);\n" +
                "public native @ByVal @Name(\"forward<std::tuple<torch::Tensor,std::tuple<torch::Tensor,torch::Tensor>>>\") T_TensorT_TensorTensor_T_T forwardT_TensorT_TensorTensor_T_T(@Const @ByRef Tensor input, @ByVal(nullValue = \"torch::optional<std::tuple<torch::Tensor,torch::Tensor> >{}\") T_TensorTensor_TOptional hx_opt);\n" +
                "public native @ByVal @Name(\"forward<std::tuple<torch::Tensor,torch::Tensor>>\") T_TensorTensor_T forwardT_TensorTensor_T(@Const @ByRef Tensor input);\n" +
                "public native @ByVal @Name(\"forward<std::tuple<torch::Tensor,torch::Tensor>>\") T_TensorTensor_T forwardT_TensorTensor_T(@Const @ByRef Tensor input1, @Const @ByRef Tensor input2);\n" +
                "public native @ByVal @Name(\"forward<std::tuple<torch::Tensor,torch::Tensor>>\") T_TensorTensor_T forwardT_TensorTensor_T(@Const @ByRef Tensor input1, @Const @ByRef Tensor input2, @Const @ByRef Tensor input3);\n" +
                "public native @ByVal @Name(\"forward<std::tuple<torch::Tensor,torch::Tensor>>\") T_TensorTensor_T forwardT_TensorTensor_T(@Const @ByRef Tensor input, @ByVal(nullValue = \"torch::optional<std::tuple<torch::Tensor,torch::Tensor> >{}\") T_TensorTensor_TOptional hx_opt);\n" +
                "public native @ByVal @Name(\"forward<std::tuple<torch::Tensor,torch::Tensor>>\") T_TensorTensor_T forwardT_TensorTensor_T(@Const @ByRef Tensor query, @Const @ByRef Tensor key, @Const @ByRef Tensor value, @Const @ByRef(nullValue = \"torch::Tensor{}\") Tensor key_padding_mask, @Cast(\"bool\") boolean need_weights/*=true*/, @Const @ByRef(nullValue = \"torch::Tensor{}\") Tensor attn_mask, @Cast(\"bool\") boolean average_attn_weights/*=true*/);\n" +
                "public native @ByVal @Name(\"forward<torch::nn::ASMoutput>\") ASMoutput forwardASMoutput(@Const @ByRef Tensor input, @Const @ByRef Tensor target);\n"
            ))
        ;

        for (String[] outputType : new String[][]{
            {"at::Tensor", "Tensor"},
            {"torch::nn::ASMoutput", "ASMoutput"},
            {"std::tuple<at::Tensor,at::Tensor>", "T_TensorTensor_T"},
            {"std::tuple<torch::Tensor,std::tuple<torch::Tensor,torch::Tensor> >", "T_TensorT_TensorTensor_T_T"}
        }) {
            infoMap
                .put(new Info(template("torch::nn::AnyValue::get", outputType[0])).javaNames("get" + outputType[1]))
                .put(new Info(template("torch::nn::AnyValue::try_get", outputType[0])).javaNames("try_get" + outputType[1]))
            ;
        }


        //// Classes handled with @SharedPtr
        // Annotating the constructor is normally needed for all classes for which
        // at least an API call takes a shared pointer of this class AND
        // if instances of this class can be created from a Java constructor.
        for (PointerInfo pi : new PointerInfo[]{
            new PointerInfo("torch::jit::Graph"),
            new PointerInfo("torch::jit::Operator"),
            new PointerInfo("torch::jit::Resolver"),
            new PointerInfo("c10::ClassType"),
            new PointerInfo("c10::TensorType").otherCppNames("c10::TensorTypePtr", "at::TensorTypePtr", "torch::TensorTypePtr"),
            new PointerInfo("torch::nn::Module"),
            new PointerInfo("const at::functorch::FuncTorchTLSBase"),
            new PointerInfo("const torch::jit::CompilationUnit"),
            new PointerInfo("torch::jit::SugaredValue"),
            new PointerInfo("caffe2::serialize::ReadAdapterInterface"),
            new PointerInfo("c10::SafePyObject"),
            //new PointerInfo("torch::distributed::autograd::SendRpcBackward"), // Not on Windows
            //new PointerInfo("torch::distributed::autograd::RecvRpcBackward"),
            new PointerInfo("c10d::Logger"), // Not sure if this class (and c10d::Reducer) has any use,
            new PointerInfo("torch::distributed::autograd::DistAutogradContext"),
            new PointerInfo("torch::jit::CompilationUnit"),
            new PointerInfo("c10d::WorkInfo"),
            new PointerInfo("c10::impl::PyObject_TorchDispatchMode"),
            new PointerInfo("c10::LazyValue<std::string>", "const c10::LazyValue<std::string>").javaBaseName("Backtrace"),
            new PointerInfo("c10::SafePyObjectT<c10::impl::TorchDispatchModeKey>").javaBaseName("PyObject_TorchDispatchMode")
        }) {
            pi.makeShared(infoMap);
        }
        // Disambiguate between candidate functions
        infoMap.put(new Info("torch::dynamo::autograd::CompiledNodeArgs::collect(torch::autograd::Node::Node*)") // Really collect(const std::shared_ptr<torch::autograd::Node>&)
                .javaText("public native void collect(@Cast({\"\", \"const std::shared_ptr<torch::autograd::Node>\"}) @SharedPtr Node t);"))
               ;


        //// Classes handled with @UniquePtr
        for (String opt: new String[] { "Adagrad", "Adam", "AdamW", "LBFGS", "RMSprop", "SGD" }) {
            infoMap
                .put(new Info("torch::optim::" + opt + "Options", "torch::optim::" + opt + "ParamState")) // Help qualification
                .put(new Info("torch::optim::OptimizerCloneableOptions<torch::optim::" + opt + "Options>").pointerTypes("OptimizerCloneable" + opt + "Options"))
                .put(new Info("torch::optim::OptimizerCloneableParamState<torch::optim::" + opt + "ParamState>").pointerTypes("OptimizerCloneable" + opt + "ParamState"))
            ;
            new PointerInfo("torch::optim::" + opt + "Options").makeUnique(infoMap);
            new PointerInfo("torch::optim::OptimizerCloneableParamState<torch::optim::" + opt + "ParamState>").javaBaseName("OptimizerCloneable" + opt + "AdagradParamState").makeUnique(infoMap);
            new PointerInfo("torch::optim::OptimizerCloneableOptions<torch::optim::" + opt + "Options>").javaBaseName("OptimizerCloneable" + opt + "Options").makeUnique(infoMap);
            new PointerInfo("torch::optim::" + opt + "Options").makeUnique(infoMap);
            new PointerInfo("torch::optim::" + opt + "ParamState").makeUnique(infoMap);
        }
        for (PointerInfo pi : new PointerInfo[]{
            new PointerInfo("torch::optim::OptimizerOptions"),
            new PointerInfo("torch::optim::OptimizerParamState"),
            new PointerInfo("torch::autograd::AutogradMeta"),
            new PointerInfo("torch::jit::GraphAttr"),
            new PointerInfo("torch::jit::Graph"),
            new PointerInfo("c10::NamedTensorMeta"),
            new PointerInfo("c10::FunctionSchema"),
            new PointerInfo("at::CPUGeneratorImpl"),
            new PointerInfo("at::TensorIterator"),
            new PointerInfo("caffe2::serialize::IStreamAdapter"),
            new PointerInfo("torch::autograd::FunctionPreHook").virtualize(),
            new PointerInfo("torch::autograd::FunctionPostHook").virtualize(),
            // Other classes passed as unique ptr are abstract, so not instantiated from Java:
            // ReadAdapterInterface, PostAccumulateGradHook, FuncTorchTLSBase, AutogradMetaInterface,
            // GeneratorImpl, OpRegistrationListener, AttributeValue
        }) {
            pi.makeUnique(infoMap);
        }
        infoMap
            .put(new Info("std::unique_ptr<torch::jit::AttributeValue>", "torch::jit::GraphAttr::Ptr").annotations("@UniquePtr").pointerTypes("AttributeValue")) // Ptr is really defined in AttributeValue (superclass of GraphAttr). But Parser doesn't find it.
            .put(new Info("torch::autograd::AutogradMeta::post_acc_grad_hooks_").annotations("@UniquePtr", "@Cast({\"\", \"\", \"std::unique_ptr<torch::autograd::PostAccumulateGradHook>&&\"})")) // See JavaCPP Issue #717

            .put(new Info("std::unique_ptr<c10::SafePyObject>").skip()) // A class cannot be handled by both shared and unique ptr
        ;

        // Already defined in gloo
        infoMap
            .put(new Info("std::shared_ptr<::gloo::transport::Device>").annotations("@SharedPtr").pointerTypes("org.bytedeco.pytorch.gloo.Device"))
            .put(new Info("::gloo::transport::UnboundBuffer").pointerTypes("org.bytedeco.pytorch.gloo.UnboundBuffer"))
            .put(new Info("::gloo::rendezvous::Store").pointerTypes("org.bytedeco.pytorch.gloo.Store"))
            .put(new Info("::gloo::Context").pointerTypes("org.bytedeco.pytorch.gloo.Context"))
        ;

        // See https://github.com/pytorch/pytorch/issues/127873
        infoMap
            .put(new Info("c10d::AllReduceCommHook", "c10d::FP16CompressCommHook").skip())
        ;

        infoMap.put(new Info("torch::distributed::rpc::SerializedPyObj::SerializedPyObj").javaText(
            "  public SerializedPyObj(BytePointer payload, TensorVector tensors) { super((Pointer)null); allocate(payload, tensors); }\n" +
            "  private native void allocate(@Cast({\"\",\"std::string&&\"}) @StdString BytePointer payload, @ByRef(true) TensorVector tensors);\n" +
            "  public SerializedPyObj(String payload, TensorVector tensors) { super((Pointer)null); allocate(payload, tensors); }\n" +
            "  private native void allocate(@Cast({\"\",\"std::string&&\"}) @StdString String payload, @ByRef(true) TensorVector tensors);")
        ); // Parser doesn't add the @Cast

        /* TODO: see how to map these, if needed and meant to be part of API */
        infoMap.put(new Info("c10::MaybeOwnedTraitsGenericImpl<std::shared_ptr<at::Tensor> >::assignBorrow",
            "c10::MaybeOwnedTraitsGenericImpl<std::shared_ptr<at::Tensor> >::destroyBorrow",
            "torch::autograd::profiler::ProfilerResult", "torch::profiler::impl::ProfilerEventStub",
            "torch::autograd::profiler::enableProfiler", "torch::autograd::profiler::enableProfilerWithEventPostProcess",
            "torch::profiler::impl::ProfilerStateBase", "torch::profiler::impl::ProfilerStubs", "torch::autograd::profiler::KinetoEvent",
            "at::Tensor::wrap_tensor_impl(c10::TensorImpl*)",
            "c10::impl::list_element_to_const_ref",
            "c10::unpackSymInt(at::OptionalSymIntArrayRef)",
            "c10::detail::infer_schema::make_function_schema(std::string&&, std::string&&, c10::ArrayRef<c10::detail::infer_schema::ArgumentDef>, c10::ArrayRef<c10::detail::infer_schema::ArgumentDef>)",
            "torch::autograd::_wrap_outputs",
            "torch::autograd::Node::retains_grad_hooks", // IntFunctionPreHookMap cannot be instantiated because static_assert errors due to unique_ptr copying
            "c10::impl::GPUTrace", "torch::jit::IterableTree",
            "c10::cuda::CaptureStatus",

            // Ignore for now, takes a callback.
            "c10::IValue::repr", "c10::IValue::visit",
            "at::TensorIteratorBase::foreach_reduced_elt",
            "at::TensorIteratorBase::parallel_reduce",
            "at::TensorIteratorBase::serial_for_each",
            "at::TensorIteratorBase::for_each",

            "torch::autograd::get_current_graph_task_exec_info", // Would need to map GraphTask, NodeExec...too much burden

            "torch::Library::def",

            // Could not figure out how to map shared_ptr of std::function
            "torch::distributed::rpc::RpcAgent::getTypeResolver", "torch::distributed::rpc::RpcAgent::setTypeResolver",

            // The unique constructor takes a std::shared_ptr<DistAutogradContext>&&
            // How to pass a shared_ptr as an r-value with the adapter ?
            "torch::distributed::autograd::ThreadLocalDistAutogradContext"

            ).skip())
        ;

        //// Prevents compiler to croak about "non-standard-layout type".
        /* We cannot add an Info annotation("@NoOffset") on the class, or the parser will also add the annotation on method argument,
           which is not supported and has no sense.
           We need either to put an annotation info on each member, or javaName("@NoOffset XXX") on the whole class.
           If an info exists on the member, it must not have annotations, or they will be replaced.
         */
        for (String n : new String[]{
            "c10::DDPLoggingData::strs_map",
            "c10::DDPLoggingData::ints_map",
            "torch::dynamo::autograd::TensorArgs::inputs",
            "torch::dynamo::autograd::AutogradCompilerCall::all_size_inputs",
            "torch::dynamo::autograd::AutogradCompilerCall::dyn_size_inputs",
            "torch::dynamo::autograd::AutogradCompilerCall::node_calls",
            "torch::dynamo::autograd::AutogradCompilerCall::default_dyn_type",
            "torch::jit::Object::Property::setter_func",
            "torch::jit::Object::Property::getter_func",
            "torch::jit::Object::Property::name",
            "torch::jit::Named<torch::jit::Module>::name",
            "torch::jit::Named<torch::jit::Module>::value",
            "torch::jit::detail::SlotCursor::i_",
            "torch::jit::detail::SlotCursor::module_",
            "torch::jit::StackEntry::filename",
            "torch::jit::StackEntry::range",
            "torch::jit::Call::fn_name",
            "torch::jit::Call::caller_range",
            "c10::SymbolicShapeMeta::sizes_",
            "c10::SymbolicShapeMeta::strides_",
            "c10::SymbolicShapeMeta::numel_",
            "c10::SymbolicShapeMeta::storage_offset_",
            "c10::SymbolicShapeMeta::is_contiguous_",
            "c10::SymbolicShapeMeta::is_channels_last_contiguous_",
            "c10::SymbolicShapeMeta::is_channels_last_3d_contiguous_",
            "c10::SymbolicShapeMeta::is_channels_last_",
            "c10::SymbolicShapeMeta::is_channels_last_3d_",
            "c10::SymbolicShapeMeta::is_non_overlapping_and_dense_",
            "c10d::AllreduceOptions::timeout",
            "c10d::AllreduceOptions::reduceOp",
            "c10d::AllreduceOptions::sparseIndices",
            "c10d::C10dLoggingData::strings",
            "c10d::C10dLoggingData::integers",
            "c10d::ReduceOptions::timeout",
            "c10d::ReduceOptions::reduceOp",
            "c10d::ReduceOptions::rootRank",
            "c10d::ReduceOptions::rootTensor",
            "c10d::ReduceScatterOptions::reduceOp",
            "c10d::ReduceScatterOptions::timeout",
            "c10d::ReduceScatterOptions::asyncOp"
        }) {
            Info i = infoMap.getFirst(n, false);
            if (i == null) {
                i = new Info(n);
                infoMap.put(i);
            }
            i.annotations("@NoOffset");
        }


        //// Classes whose parent are useless for us
        infoMap.put(new Info(
            "caffe2::TypeIdentifier", "c10::util::crc64_t", "c10::util::type_index"
        ).base("Pointer"));


        //// Pytorch "internal only"
        infoMap.put(new Info(
            "at::RecordFunction::_setAsync", "at::RecordFunction::_setStaticRuntimeOutVariant",
            "at::Tensor::Tensor(c10::TensorImpl*)", // "should not be used by end users". Really at::Tensor(c10::intrusive_ptr<at::TensorImpl,c10::UndefinedTensorImpl> but the Parser gets the wrong fullname
            "at::Tensor::_set_fw_grad", "at::Tensor::_fw_grad",
            "at::TensorBase::_set_fw_grad", "at::TensorBase::_fw_grad",
            "at::TensorImpl::_set_fw_grad", "at::TensorImpl::_fw_grad",
            "c10::KernelFunction::_equalsBoxedAndUnboxed",
            "c10::RegisterOperators::Options::catchAllKernel()",
            "c10::RegisterOperators::Options::kernel(c10::DispatchKey)",
            "c10::RegisterOperators::Options::schema(c10::FunctionSchema&&)",
            "c10::RegisterOperators::op(c10::FunctionSchema,c10::Options&&)",
            "c10::ThreadLocalDebugInfo::_forceCurrentDebugInfo",
            "c10::impl::_force_tls_local_dispatch_key_set",
            "torch::jit::CompilationUnit::_clear_python_cu",
            "torch::jit::GraphFunction::_set_initial_executor_execution_mode", "torch::jit::GraphFunction::_set_ignore_amp",
            "c10::detail::_str",
            "torch::jit::kJitOnlyOperatorTags",
            "c10::IValue::Tag", // 2.2.0 make IValue::tag public, while IValue::Tag is supposed to be private. Bug ? Check if fixed in next release
            "c10d::_AllReduceBySumCommHook", //  "Only used internally and not released as a public built-in communication hook."

            // Optional args of AOTModelContainerRun.run. Opaque types without apparent use in 2.2.0.
            "AOTInductorStreamOpaque",
            "AOTInductorStreamHandle",
            "AOTIProxyExecutorOpaque",
            "AOTIProxyExecutorHandle"
        ).skip());


        //// Deprecated
        infoMap.put(new Info(
            "c10::detail::deprecated_AT_ERROR",
            "c10::detail::deprecated_AT_ASSERT",
            "c10::detail::deprecated_AT_ASSERTM",
            "detail::deprecated_AT_DISPATCH_ALL_TYPES_AND_HALF",
            "detail::deprecated_AT_DISPATCH_ALL_TYPES_AND_HALF_AND_COMPLEX",
            "detail::scalar_type(const at::DeprecatedTypeProperties&)",
            "at::DeprecatedTypeProperties",
            "c10::Scalar::isIntegral()",
            "c10::isIntegralType(c10::ScalarType)",
            "at::Tensor::type()",
            "at::Tensor::is_variable()",
            "c10d::Store::watchKey"
        ).skip());

        //// Function returning object by value, and copy constructor was deleted. Any way to get around this ?
        infoMap.put(new Info(
            "c10::RegisterOperators::Options", //All methods of Options return Options&&
            "c10::impl::device_guard_impl_registry",
            "torch::autograd::graph_task_id",
            "c10::getLessThanComparator", "c10::getGreaterThanComparator"
        ).skip());


        //// Deleted operator= or related errors. Any way to skip setter only ?
        infoMap.put(new Info(
            "at::native::RNNDescriptor::dropout_desc_",
            "torch::dynamo::autograd::AutogradCompilerCall::hooks"
        ).skip());


        //// ifdef'd out
        infoMap.put(new Info(
            "c10_complex_math::_detail::sqrt",
            "c10_complex_math::_detail::acos",
            "c10::__ldg",
            "c10::impl::raw_local_dispatch_key_set" // non-windows, non-android only
        ).skip());


        //// Function not compiling because failing some static_assert
        infoMap.put(new Info("at::SplitUntil32Bit::iterator::vec",
            //"std::vector<std::unique_ptr<at::TensorIterator> >::put(std::vector<std::unique_ptr<at::TensorIterator> >)",
            "c10::ArrayRef<c10::detail::infer_schema::ArgumentDef>::equals",
            "c10::ArrayRef<torch::jit::NamedValue>::equals",
            "c10::ArrayRef<at::Scalar>::equals",
            "c10::ArrayRef<torch::TensorArg>::equals",
            "c10::ArrayRef<torch::Tensor>::equals",
            "c10::ArrayRef<at::indexing::TensorIndex>::equals",
            "c10::ArrayRef<std::optional<at::Tensor> >::equals"
        ).skip());

        infoMap
            .put(new Info("torch::distributed::rpc::worker_id_t").valueTypes("short").pointerTypes("ShortPointer"))
            .put(new Info("torch::distributed::rpc::local_id_t").valueTypes("long").pointerTypes("LongPointer"))
        ;
        infoMap
            .put(new Info("torch::distributed::rpc::MessageTypeFlags").enumerate(false))
        ;


        //// Avoiding name clashes by skipping or renaming
        infoMap.put(new Info("c10::ComplexType::get").javaNames("getComplexTypePtr"))
               .put(new Info("c10::FloatType::get").javaNames("getFloatTypePtr"))
               .put(new Info("c10::IntType::get").javaNames("getIntTypePtr"))
               .put(new Info("c10::NumberType::get").javaNames("getNumberIntTypePtr"))
               .put(new Info("c10::GeneratorImpl::clone").javaNames("clonePtr"))
               .put(new Info("c10::IValue::toString", "at::IValue::toString").javaNames("toConstantString"))
               .put(new Info("torch::jit::TreeView::get").skip()) // Prevents override of get() in subclasses, and tree is available as tree() anyway
               .put(new Info("torch::cuda::device_count").javaNames("cuda_device_count"))
               .put(new Info("torch::cuda::is_available").javaNames("cuda_is_available"))
               .put(new Info("torch::cuda::manual_seed").javaNames("cuda_manual_seed"))
               .put(new Info("torch::cuda::manual_seed_all").javaNames("cuda_manual_seed_all"))
               .put(new Info("torch::cuda::synchronize").javaNames("cuda_synchronize"))
               .put(new Info("torch::xpu::device_count").javaNames("xpu_device_count"))
               .put(new Info("torch::xpu::is_available").javaNames("xpu_is_available"))
               .put(new Info("torch::xpu::manual_seed").javaNames("xpu_manual_seed"))
               .put(new Info("torch::xpu::manual_seed_all").javaNames("xpu_manual_seed_all"))
               .put(new Info("torch::xpu::synchronize").javaNames("xpu_synchronize"))
               .put(new Info("torch::jit::Const").pointerTypes("ConstExpr"))
               .put(new Info("torch::jit::Node").pointerTypes("JitNode"))
               .put(new Info("torch::jit::Module").pointerTypes("JitModule"))
               .put(new Info("torch::jit::Object").pointerTypes("JitObject"))
               .put(new Info("torch::jit::String").pointerTypes("JitString"))
               .put(new Info("torch::autograd::Error").pointerTypes("AutogradError")) // Clash with c10::Error or Java Error
               .put(new Info("c10d::Backend").pointerTypes("DistributedBackend").purify())
               .put(new Info("torch::dynamo::autograd::TensorArg").pointerTypes("DynamoTensorArg")) // Clash with at::TensorArg
        ;


        //// Instantiation of misc class templates.
        infoMap
            .put(new Info("std::list<std::pair<at::RecordFunctionHandle,int> >").pointerTypes("RecordFunctionHandleIntList").define())
            .put(new Info(
                "torch::ExpandingArray<1>", "torch::ExpandingArray<2>", "torch::ExpandingArray<3>", "torch::ExpandingArray<4>",
                "torch::ExpandingArray<D*2>", "torch::ExpandingArray<1*2>", "torch::ExpandingArray<2*2>", "torch::ExpandingArray<3*2>").cast().pointerTypes("LongPointer"))
            .put(new Info("torch::ExpandingArray<1,double>", "torch::ExpandingArray<2,double>", "torch::ExpandingArray<3,double>").cast().pointerTypes("DoublePointer"))
            .put(new Info("torch::ExpandingArrayWithOptionalElem<2>", "torch::ExpandingArrayWithOptionalElem<3>").cast().pointerTypes("LongOptional"))
            .put(new Info("c10::VaryingShape<int64_t>").pointerTypes("LongVaryingShape"))
            .put(new Info("c10::VaryingShape<c10::Stride>").pointerTypes("StrideVaryingShape"))
            .put(new Info("c10::MaybeOwned<at::Tensor>").pointerTypes("TensorMaybeOwned"))
            .put(new Info("c10::MaybeOwned<at::TensorBase>").pointerTypes("TensorBaseMaybeOwned"))
            .put(new Info("at::InferExpandGeometryResult<at::DimVector>").pointerTypes("DimVectorInferExpandGeometryResult"))
            .put(new Info("c10::TensorImpl::identity<c10::SymInt>").pointerTypes("SymIntIdentity"))
            .put(new Info("c10::TensorImpl::identity<int64_t>").pointerTypes("LongIdentity"))
            .put(new Info("torch::detail::SelectiveStr<false>").pointerTypes("DisabledStr"))
            .put(new Info("torch::detail::SelectiveStr<true>").pointerTypes("EnabledStr"))
            .put(new Info("torch::detail::SelectiveStr<false>::operator const char*",
                "torch::detail::SelectiveStr<true>::operator const char*").
                javaText("public native @Name(\"operator const char*\") @Cast(\"const char*\") BytePointer asBytePointer();"))// Fixes bug where constexpr prevents addition of const in @Name

            .put(new Info("torch::monitor::Stat<double>").pointerTypes("DoubleStat"))
            .put(new Info("torch::monitor::Stat<int64_t>").pointerTypes("LongStat"))
            .put(new Info("torch::jit::generic_graph_node_list<torch::jit::Node>").pointerTypes("graph_node_list"))
            .put(new Info("torch::jit::generic_graph_node_list_iterator<torch::jit::Node>").pointerTypes("graph_node_list_iterator"))
            .put(new Info("torch::autograd::Function<torch::nn::CrossMapLRN2d>").pointerTypes("FunctionCrossMapLRN2d"))
            .put(new Info("c10d::CppCommHookInterface<c10::intrusive_ptr<c10d::ProcessGroup> >").pointerTypes("ProcessGroupCppCommHookInterface").purify())
            .put(new Info("c10::SafePyObjectT<c10::impl::TorchDispatchModeKey>").pointerTypes("PyObject_TorchDispatchMode"))
            .put(new Info("c10::SafePyObjectT<c10::impl::TorchDispatchModeKey>::SafePyObjectT(c10::SafePyObjectT<c10::impl::TorchDispatchModeKey>&&)").skip()) // As of 2.4.0, this constructor doesn't compile because a std::move is missing in SafePyObject move constructor
            .put(new Info("c10::LazyValue<std::string>", "const c10::LazyValue<std::string>").pointerTypes("Backtrace"))
            .put(new Info("c10::Backtrace").annotations("@SharedPtr(\"const c10::LazyValue<std::string>\")"))
        ;

        //// Instantiation of function templates.
        for (String op : new String[]{"exp", "log", "log10", "log2", "sqrt", "pow", "sin", "cos", "tan",
            "asin", "acos", "atan", "sinh", "cosh", "tanh", "asinh", "acosh", "atanh", "log1p" }) {
            infoMap.put(new Info("c10_complex_math::" + op + "<float>").javaNames(op))
                   .put(new Info("c10_complex_math::" + op + "<double>").javaNames(op))
                   .put(new Info("at::" + op).javaNames(op)); // Needed because "ATen/ops/*.h"
            // are parsed after complex_math.h and Parser would set the qualified names to the first
            // matching cppName it finds in infoMap.
        }
        infoMap.put(new Info("ska::detailv3::log2").javaNames("log2")) // Same reason
               .put(new Info("c10_complex_math::pow(c10::complex<T>&, c10::complex<U>&)").javaText(
                   "@Namespace(\"c10_complex_math\") public static native @ByVal @Name(\"pow<double,float>\") DoubleComplex pow(@Const @ByRef DoubleComplex x, @Const @ByRef FloatComplex y);\n"
                   + "@Namespace(\"c10_complex_math\") public static native @ByVal @Name(\"pow<float,double>\") DoubleComplex pow(@Const @ByRef FloatComplex x, @Const @ByRef DoubleComplex y);\n"
               ))
               .put(new Info("c10_complex_math::pow(c10::complex<T>&, U&)").javaText(
                   "@Namespace(\"c10_complex_math\") public static native @ByVal @Name(\"pow<double,float>\") DoubleComplex pow(@Const @ByRef DoubleComplex x, @Const @ByRef float y);\n"
                   + "@Namespace(\"c10_complex_math\") public static native @ByVal @Name(\"pow<float,double>\") DoubleComplex pow(@Const @ByRef FloatComplex x, @Const @ByRef double y);\n"
               ))
               .put(new Info("c10_complex_math::pow(T&, c10::complex<U>&)").javaText(
                   "@Namespace(\"c10_complex_math\") public static native @ByVal @Name(\"pow<double,float>\") DoubleComplex pow(@Const @ByRef double x, @Const @ByRef FloatComplex y);\n"
                   + "@Namespace(\"c10_complex_math\") public static native @ByVal @Name(\"pow<float,double>\") DoubleComplex pow(@Const @ByRef float x, @Const @ByRef DoubleComplex y);\n"
               ))
               .put(new Info("c10::util::get_type_index<std::string>").javaNames("get_type_index_string"))
               .put(new Info("at::TensorBase::data_ptr<bool>").javaNames("data_ptr_bool"))
               .put(new Info("at::TensorBase::data_ptr<int8_t>").javaNames("data_ptr_char"))
               .put(new Info("at::TensorBase::data_ptr<uint8_t>").javaNames("data_ptr_byte"))
               .put(new Info("at::TensorBase::data_ptr<int16_t>").javaNames("data_ptr_short"))
               .put(new Info("at::TensorBase::data_ptr<int>").javaNames("data_ptr_int"))
               .put(new Info("at::TensorBase::data_ptr<int64_t>").javaNames("data_ptr_long"))
               .put(new Info("at::TensorBase::data_ptr<float>").javaNames("data_ptr_float"))
               .put(new Info("at::TensorBase::data_ptr<double>").javaNames("data_ptr_double"))
               .put(new Info("at::Tensor::item<bool>").javaNames("item_bool"))
               .put(new Info("at::Tensor::item<int8_t>").javaNames("item_char"))
               .put(new Info("at::Tensor::item<uint8_t>").javaNames("item_byte"))
               .put(new Info("at::Tensor::item<int16_t>").javaNames("item_short"))
               .put(new Info("at::Tensor::item<int>").javaNames("item_int"))
               .put(new Info("at::Tensor::item<int64_t>").javaNames("item_long"))
               .put(new Info("at::Tensor::item<float>").javaNames("item_float"))
               .put(new Info("at::Tensor::item<double>").javaNames("item_double"))
               .put(new Info("at::make_generator").javaText(
                   "@Namespace(\"at\") public static native @ByVal @Name(\"make_generator<at::CPUGeneratorImpl>\") Generator make_generator_cpu();\n" +
                   "@Namespace(\"at\") public static native @ByVal @Name(\"make_generator<at::CPUGeneratorImpl,uint64_t>\") Generator make_generator_cpu(@Cast(\"uint64_t&&\") long seed_in);"
               ))
               .put(new Info("c10::TensorOptions::TensorOptions<c10::Device>").javaNames("XXX"))
        ;

        for (String[] t : new String[][]{
            {"c10::qint8", "qint8"},
            {"c10::quint8", "quint8"},
            {"c10::qint32", "quint32"},
            {"c10::quint4x2", "quint4x2"},
            {"c10::quint2x4", "quint2x4"},
            {"int8_t", "byte"},
            {"int16_t", "short"},
            {"int", "int"},
            {"int64_t", "long"},
            {"at::Half", "Half"},
            {"float", "float"},
            {"double", "double"},
            {"c10::complex<float>", "ComplexFloat"},
            {"c10::complex<double>", "ComplexDouble"},
            {"bool", "boolean"},
            {"at::BFloat16", "BFloat16"},
            {"at::Float8_e4m3fn", "Float8_e4m3fn"},
            {"at::Float8_e5m2", "Float8_e5m2"},
            {"at::Float8_e5m2fnuz", "Float8_e5m2fnuz"},
            {"at::Float8_e4m3fnuz", "Float8_e4m3fnuz"}
        }) {
            infoMap.put(new Info(template("c10::fetch_and_cast", t[0])).javaNames("fetch_and_cast_to_" + t[1]))
                   .put(new Info(template("c10::cast_and_store", t[0])).javaNames("cast_and_store_from_" + t[1]));
        }


        //// c10::string_view
        infoMap.put(new Info("c10::basic_string_view<char>", "c10::string_view").annotations("@StringView").valueTypes("BytePointer", "String"));

        // Registries.
        // Skipped them for now. Much burden with variadic args and creator function pointers.
        // We cannot map ThreadPoolRegistry because it takes 3 arguments in the variadic Args Registry template arguments

        /*
                       .put(new Info("c10::Registry<std::string,std::unique_ptr<at::MPSHooksInterface>,at::MPSHooksArgs>").pointerTypes("MPSHooksRegistry"))
                .put(new Info("c10::Registry<std::string,std::unique_ptr<at::MPSHooksInterface>,at::MPSHooksArgs>::Create").javaText(
                        "public native @UniquePtr MPSHooksInterface Create(@StdString BytePointer key, @ByRef MPSHooksArgs args);\n" +
                                "public native @UniquePtr MPSHooksInterface Create(@StdString String key, @ByRef MPSHooksArgs args);"))   // Handle pack extension

                .put(new Info("c10::Registry<std::string,std::shared_ptr<c10::TaskThreadPoolBase>,int,int,bool>",
                     "c10::Registry<std::string,std::shared_ptr<c10::TaskThreadPoolBase>,int>" // JavaCPP doesn't really support variadic templates argument.
                     // We must provide this truncated list of arguments so that Context.qualify can find this Info. Issue #81.
                 ).pointerTypes("ThreadPoolRegistry").javaNames("ThreadPoolRegistry"))
                .put(new Info("c10::Registry<std::string,std::shared_ptr<c10::TaskThreadPoolBase>,int>::Create").javaText(
                        "public native @SharedPtr TaskThreadPoolBase Create(@StdString BytePointer key, int i1, int i2, boolean b);\n" +
                        "public native @SharedPtr TaskThreadPoolBase Create(@StdString String key, int i1, int i2, boolean b);"))   // Handle pack extension
                .put(new Info("std::shared_ptr<c10::TaskThreadPoolBase>").pointerTypes("TaskThreadPoolBase").annotations("@SharedPtr"))

                .put(new Info("c10::Registry<std::string,std::unique_ptr<at::CUDAHooksInterface>,at::CUDAHooksArgs>").pointerTypes("CUDAHooksRegistry"))
                .put(new Info("c10::Registry<std::string,std::unique_ptr<at::CUDAHooksInterface>,at::CUDAHooksArgs>::Create").javaText(
                        "public native @UniquePtr CUDAHooksInterface Create(@StdString BytePointer key, @ByRef CUDAHooksArgs args);\n" +
                                "public native @UniquePtr CUDAHooksInterface Create(@StdString String key, @ByRef CUDAHooksArgs args);"))   // Handle pack extension

                .put(new Info("c10::Registry<std::string,std::unique_ptr<at::HIPHooksInterface>,at::HIPHooksArgs>").pointerTypes("HIPHooksRegistry"))
                .put(new Info("c10::Registry<std::string,std::unique_ptr<at::HIPHooksInterface>,at::HIPHooksArgs>::Create").javaText(
                        "public native @UniquePtr HIPHooksInterface Create(@StdString BytePointer key, @ByRef HIPHooksArgs args);\n" +
                                "public native @UniquePtr HIPHooksInterface Create(@StdString String key, @ByRef HIPHooksArgs args);"))   // Handle pack extension

                .put(new Info("c10::Registry<std::string,std::unique_ptr<at::ORTHooksInterface>,at::ORTHooksArgs>").pointerTypes("ORTHooksRegistry"))
                .put(new Info("c10::Registry<std::string,std::unique_ptr<at::ORTHooksInterface>,at::ORTHooksArgs>::Create").javaText(
                        "public native @UniquePtr ORTHooksInterface Create(@StdString BytePointer key, @ByRef ORTHooksArgs args);\n" +
                                "public native @UniquePtr ORTHooksInterface Create(@StdString String key, @ByRef ORTHooksArgs args);"))   // Handle pack extension

                                ,
                .put(new Info("c10::Registry<std::string,std::unique_ptr<at::ORTHooksInterface>,at::ORTHooksArgs>::Creator",
                       "c10::Registry<std::string,std::unique_ptr<at::CUDAHooksInterface>,at::CUDAHooksArgs>::Creator",
                       "c10::Registry<std::string,std::unique_ptr<at::HIPHooksInterface>,at::HIPHooksArgs>::Creator",
                       "c10::Registry<std::string,std::unique_ptr<at::MPSHooksInterface>,at::MPSHooksArgs>::Creator").pointerTypes("Pointer"))
        */

        infoMap.put(new Info("c10::ThreadPoolRegistry()",
            "c10::CUDAHooksRegistry()").skip());


        //// Not mapping all custom pytorch errors since there is currently no way to catch them as objects from Java
        infoMap.put(new Info(
                "c10::Error",
                "c10::ivalue::Future::FutureError",
                "c10::ThrowEnforceNotMet",
                "torch::jit::ErrorReport",
                "c10::DistError",
                "c10::DistBackendError",
                "c10::DistStoreError",
                "c10::DistNetworkError",
                "c10::EnforceFiniteError",
                "c10::ErrorAlwaysShowCppStacktrace",
                "c10::IndexError",
                "c10::LinAlgError",
                "c10::NotImplementedError",
                "c10::OnnxfiBackendSystemError",
                "c10::OutOfMemoryError",
                "c10::TypeError",
                "c10::ValueError"
            ).skip()
        );


        //// Forward references and opaque classes
        infoMap
            .put(new Info("c10::Argument").pointerTypes("Argument")) // Ref in function_schema_inl.h, defined in function_schema.h
            .put(new Info("c10::impl::CppSignature"))
        ;

        /* Classes that are not part of the API (no TORCH_API nor C10_API) and are not argument nor return type of API methods.
         * Consider manual exclusion of all at::meta, at::native and caffe2 namespaces (but TypeMeta, that should
         * be moved to c10 one day). */
        infoMap.put(new Info(
            "CUevent_st",
            "mz_zip_archive",
            "ModuleHolderIndicator",
            "at::MTIAHooksArgs",
            "at::ObserverContext",
            "at::Range",
            "at::StepCallbacks::StartEndPair",
            "at::TensorBase::unsafe_borrow_t",
            "at::internal::OpaqueOptionalTensorRef",
            "at::impl::VariableHooksRegisterer", // TORCH_API but unused ?
            "at::TensorRef",
            "at::OptionalTensorRef",
            //"at::mt19937_data_pod",
            //"at::mt19937_engine",
            "at::tracer::impl::NoTracerDispatchMode",
            "c10::_CopyBytesFunctionRegisterer",
            "c10::AlignedCharArray<1,Size>::",
            "c10::AlignedCharArray<2,Size>::",
            "c10::AlignedCharArray<4,Size>::",
            "c10::AlignedCharArray<8,Size>::",
            "c10::Capsule",
            "c10::DeviceGuard",
            "c10::DispatchTraceNestingGuard",
            "c10::Dispatcher::OperatorDef",
            "c10::DynamicType",
            "c10::DynamicType::",
            "c10::DynamicType::Arguments",
            "c10::DynamicType::LabeledDynamicType",
            "c10::DynamicTypeTrait<c10::TensorType>",
            "c10::Event",
            "c10::ExclusivelyOwned::",
            "c10::IListRef::Payload",
            "c10::IListRefIterator::Payload",
            "c10::IValue::CompAliasedIValues",
            "c10::IValue::HashAliasedIValue",
            "c10::IValue::Payload",
            "c10::IValue::Payload::TriviallyCopyablePayload",
            "c10::IValue::Payload::TriviallyCopyablePayload::",
            "c10::MaybeOwnedTraits",
            "c10::MultiStreamGuard",
            "c10::OpTableOffsetAndMask",
            "c10::OperatorNameView",
            "c10::PyHandleCache",
            "c10::RegisterOperators::Options::KernelRegistrationConfig",
            "c10::Registry<std::string,std::shared_ptr<c10::TaskThreadPoolBase>,int>",
            "c10::Registry<std::string,std::unique_ptr<at::CUDAHooksInterface>,at::CUDAHooksArgs>",
            "c10::Registry<std::string,std::unique_ptr<at::HIPHooksInterface>,at::HIPHooksArgs>",
            "c10::Registry<std::string,std::unique_ptr<at::MPSHooksInterface>,at::MPSHooksArgs>",
            "c10::Registry<std::string,std::unique_ptr<at::ORTHooksInterface>,at::ORTHooksArgs>",
            "c10::SchemaRegistrationHandleRAII",
            "c10::Scalar::v_t",
            "c10::StreamGuard",
            "c10::Type::SingletonOrSharedTypePtr::Repr",
            "c10::Type::SingletonOrSharedTypePtr::Repr::RawRepr",
            "c10::Type::SingletonOrSharedTypePtr::Repr::SingletonRepr",
            "c10::Type::SingletonOrSharedTypePtr::SharedPtrWrapper",
            "c10::Type::SingletonOrSharedTypePtr<c10::Type>::Repr",
            "c10::Type::SingletonOrSharedTypePtr<c10::Type>::Repr::RawRepr",
            "c10::Type::SingletonOrSharedTypePtr<c10::Type>::Repr::SingletonRepr",
            "c10::Type::SingletonOrSharedTypePtr<c10::Type>::SharedPtrWrapper",
            "c10::TypeFactoryBase<c10::DynamicType>",
            "c10::VarType",
            "c10::VariableVersion::VersionCounter",
            "c10::arrayref_optional_base::storage",
            "c10::arrayref_optional_base::storage::raw",
            "c10::bad_optional_access",
            "c10::basic_string_view::charIsEqual_",
            "c10::basic_string_view::charIsNotEqual_",
            "c10::basic_string_view::stringViewContainsChar_",
            "c10::basic_string_view::stringViewDoesNotContainChar_",
            "c10::basic_string_view<char>",
            "c10::basic_string_view<char>::charIsEqual_",
            "c10::basic_string_view<char>::charIsNotEqual_",
            "c10::basic_string_view<char>::stringViewContainsChar_",
            "c10::basic_string_view<char>::stringViewDoesNotContainChar_",
            "c10::detail::DictImpl",
            "c10::detail::DictKeyEqualTo",
            "c10::detail::DictKeyHash",
            "c10::detail::ListElementFrom<c10::IValue>",
            "c10::detail::ListImpl",
            "c10::detail::LoadImpl<bool>",
            "c10::detail::_guarded_unsigned_long_unique_dummy",
            "c10::detail::_str_wrapper<std::string>",
            "c10::detail::getTypePtr_<at::IValue>",
            "c10::detail::infer_schema::createReturns<void,void>",
            "c10::detail::infer_schema::createReturns<std::tuple<>,void><void>", // Parsing error ?
            "c10::detail::ivalue_to_const_ref_overload_return<at::Tensor>",
            "c10::detail::RegistrationListenerList",
            "c10::either::",
            "c10::either<c10::OperatorName,c10::FunctionSchema>",
            "c10::either<c10::OperatorName,c10::FunctionSchema>::",
            "c10::guts::conjunction",
            "c10::guts::detail::DummyClassForToString",
            "c10::guts::detail::__array_traits<_Tp,0>::_Type",
            "c10::guts::detail::_identity",
            "c10::guts::detail::_if_constexpr<true>",
            "c10::guts::disjunction",
            "c10::guts::typelist::concat<>",
            "c10::guts::typelist::concat<c10::guts::typelist::typelist<> >",
            "c10::guts::typelist::concat<c10::guts::typelist::typelist<> ><>", // Parsing error ?
            "c10::guts::typelist::reverse<c10::guts::typelist::typelist<> >",
            "c10::guts::typelist::concat<c10::guts::typelist::typelist<>,c10::guts::typelist::typelist<> >",
            "c10::guts::typelist::concat<c10::guts::typelist::typelist<>,c10::guts::typelist::typelist<> ><>", // Persing error ?
            "c10::hash<std::tuple<> >::tuple_hash<0><std::tuple<> >",
            "c10::hash<std::tuple<> >::tuple_hash<std::tuple<> >",
            "c10::impl::AnnotatedSchema",
            "c10::impl::ListElementConstReferenceTraits<std::optional<std::string> >",
            "c10::impl::SizesAndStrides::",
            "c10::impl::VirtualGuardImpl",
            "c10::impl::decay_if_not_tensor<at::Tensor&>",
            "c10::impl::is_mutable_tensor_ref<at::Tensor&>",
            "c10::in_place_t",
            "c10::ivalue::ComplexHolder",
            "c10::ivalue::StreamData3Holder",
            "c10::ivalue::TupleElements::",
            "c10::ivalue::TupleTypeFactory<c10::TupleType>",
            "c10::once_flag",
            "c10::sha1",
            "c10::static_cast_with_inter_type<c10::complex<c10::Half>,c10::BFloat16>",
            "c10::trivial_init_t",
            "caffe2::detail::_Uninitialized",
            "caffe2::detail::TypeMetaData",
            "ska::detailv3::sherwood_v3_entry::",
            "ska::detailv3::sherwood_v3_table::convertible_to_iterator",
            "ska::fibonacci_hash_policy",
            "ska::power_of_two_hash_policy",
            "ska::prime_number_hash_policy",
            "ska_ordered::fibonacci_hash_policy",
            "ska_ordered::prime_number_hash_policy",
            "ska_ordered::detailv3::sherwood_v3_entry::",
            "ska_ordered::detailv3::sherwood_v3_table::convertible_to_iterator",
            "ska_ordered::order_preserving_flat_hash_map::convertible_to_value",
            "ska_ordered::power_of_two_hash_policy",
            "std::hash<c10::Device>",
            "std::hash<c10::DeviceType>",
            "std::hash<c10::Stream>",
            "std::hash<c10::Symbol>",
            "torch::Indices",
            "torch::MakeIndices<0>",
            "torch::NoInferSchemaTag",
            "torch::all_of",
            "torch::any_of<>",
            "torch::autograd::CheckpointValidGuard",
            "torch::autograd::NodeTask",
            "torch::autograd::ReadyQueue",
            "torch::autograd::CppFunctionSingleTensorPreHook",
            "torch::autograd::CppFunctionTensorPreHook",
            "torch::autograd::GraphTask",
            "torch::autograd::GraphTask::ExecInfo", // returned by an API function get_current_graph_task_exec_info, finally excluding get_current_graph_task_exec_info
            "torch::autograd::GraphTask::ExecInfo::Capture",
            "torch::autograd::GraphTask::ExecInfo::Capture::GradCaptureHook",
            "torch::autograd::GraphTaskGuard",
            "torch::autograd::InputBuffer",
            "torch::autograd::InputMetadata",
            "torch::autograd::NodeGuard",
            "torch::autograd::ReadyQueue",
            "torch::autograd::TraceableFunction",
            "torch::autograd::TypeAndSize",
            "torch::autograd::SavedVariable",
            "torch::autograd::VariableHooks",
            "torch::data::DataLoaderBase::Job",
            "torch::data::DataLoaderBase::QuitWorker",
            "torch::data::DataLoaderBase::Result",
            "torch::data::DataLoaderBase::Sequenced",
            "torch::data::WorkerException",
            "torch::data::datasets::detail::BatchDataBuffer::UnwrappedBatchData",
            "torch::detail::ClassNotSelected",
            "torch::detail::TorchLibraryInit",
            "torch::enumtype::_compute_enum_name",
            "torch::jit::CompleteArgumentInfo",
            "torch::jit::CompleteArgumentInfoPOD",
            "torch::jit::CompleteArgumentSpec",
            "torch::jit::IRAttributeError",
            "torch::jit::InterpreterContinuation",
            "torch::jit::InterpreterState",
            "torch::jit::InterpreterStateImpl",
            "torch::jit::Lexer",
            "torch::jit::Operator::C10Operator",
            "torch::jit::Operator::JitOnlyOperator",
            "torch::jit::Operator::UnparsedFunctionSchema",
            "torch::jit::OwnedSourceRange",
            "torch::jit::RecursiveMethodCallError",
            "torch::jit::StrongFunctionPtr",
            "torch::jit::Suspend",
            "torch::jit::TokenTrie",
            "torch::jit::TaggedRange",
            "torch::jit::VectorReader",
            "torch::jit::WithCurrentScope",
            "torch::jit::WithInsertPoint",
            "torch::jit::variable_tensor_list",
            "torch::nn::AnyModuleHolder::CheckedGetter",
            "torch::nn::AnyModuleHolder::InvokeForward",
            "torch::nn::AnyModulePlaceholder",
            "torch::nn::AnyValue::Placeholder",
            "torch::nn::NamedAnyModule",
            "torch::nn::functions::CrossMapLRN2d",
            "torch::profiler::impl::HashCombine",
            "torch::profiler::impl::kineto::ActivityTraceWrapper",
            "torch::profiler::impl::ProfilerVoidEventStub",

            "torch::autograd::_jvp_fn_t", "torch::autograd::profiler::post_process_t",
            "at::StringView" // Confusion with string_view and @StringView, and doesn't seem to be of any use in API

        ).skip())
        ;


        //// Functions not part of the API
        //// TORCH_API and the like are not honored on Linux but are on Windows. We must skip all public
        //// functions not marked as part of API.
        infoMap.put(new Info(
            "at::TensorIteratorBase::apply_perm_and_mul",
            "at::assert_no_partial_overlap(c10::TensorImpl*, c10::TensorImpl*)",
            "at::impl::VariableHooksInterface::_register_hook",
            "at::native::get_numel_from_nested_size_tensor",
            "at::operator <<(std::ostream&, at::Range&)",
            "c10::cuda::CUDACachingAllocator::format_size",
            "c10::detail::makeBaseType",
            "c10::ivalue::Await::operator <<",
            "c10::ivalue::ConstantString::operator <<", // No idea why these are not exported. TODO: dig
            "c10::ivalue::EnumHolder::is", // Calls ==, which is not exported
            "c10::ivalue::EnumHolder::operator <<",
            "c10::ivalue::EnumHolder::operator ==", // The friend operator is truly a member of c10::ivalue and not c10::ivalue::EnumHolder
            "c10::ivalue::EnumHolder::unqualifiedClassName",
            "c10::ivalue::Future::operator <<",
            "c10::operator <<(std::ostream&, c10::SourceLocation&)",
            "caffe2::serialize::detail::getPadding",
            "torch::autograd::add_node_to_current_graph_task_exec_info",
            "torch::detail::constructSchemaOrName",
            "torch::jit::ClassDef::create",
            "torch::jit::Code::operator <<(std::ostream&, const torch::jit::Code&)", // The friend operator is truly a member of torch::jit and not torch::jit::Code
            "torch::profiler::impl::getNvtxStr",
            "torch::profiler::impl::shapeToStr",
            "c10::merge_primitive", // templated function with some specializations. Will have to figure what instances to create if needed.
            "at::TensorBase::TensorBase(c10::intrusive_ptr<c10::TensorImpl,c10::UndefinedTensorImpl>)", // "should not be used by end users"
            "torch::jit::Object::Object(c10::QualifiedName, std::shared_ptr<torch::jit::CompilationUnit>, bool)", // No definition
            "c10d::Logger::operator <<(std::ostream&, const c10d::Logger&)", // No definition
            "torch::distributed::rpc::Message:isShutdown", // No definition
            "torch::distributed::rpm::getAllowJitRRefPickle",
            "c10d::ProcessGroupGloo::createProcessGroupGloo", // No definition
            "torch::autograd::set_device(int)",
            "torch::distributed::rpc::Message::isShutdown", // No definition
            "torch::distributed::rpc::getAllowJitRRefPickle"
        ).skip());

        //// Aliases necessary because of Parser limited namespace resolution
        infoMap.put(new Info("at::Device", "torch::Device"))
               .put(new Info("torch::Tensor", "at::Tensor"))


        //// Classes kept but passed as generic pointer
               .put(new Info("c10::intrusive_ptr_target", "c10::nullopt", "c10::nullopt_t", "c10::impl::PyObjectSlot",
                   "_object",
                   "PyObject", "THPObjectPtr", "pyobj_list", "std::chrono::milliseconds", "std::exception_ptr", "std::type_info",
                   "std::pair<PyObject*,PyObject*>", "std::stack<std::pair<PyObject*,PyObject*> >", "torch::autograd::utils::DelayWarningHandler",
                   "std::is_same<torch::detail::pack<true>,torch::detail::pack<true> >", "at::cuda::NVRTC", "at::RecordFunctionCallback", "at::StepCallbacks", "THCState", "THHState",
                   "torch::jit::InlinedCallStackPtr", "InlinedCallStackPtr", "torch::jit::ScopePtr", "torch::jit::BackendDebugInfoRecorder",
                   "torch::detail::TensorDataContainer", "at::ArrayRef<torch::detail::TensorDataContainer>",
                   "std::shared_ptr<caffe2::serialize::PyTorchStreamReader>", "caffe2::serialize::PyTorchStreamWriter",
                   "c10::detail::DictImpl::dict_map_type::iterator",
                   "std::iterator<std::forward_iterator_tag,c10::impl::DictEntryRef<c10::IValue,c10::IValue,c10::detail::DictImpl::dict_map_type::iterator> >",
                   "std::optional<PyObject*>", "std::optional<std::chrono::milliseconds>",
                   "c10::intrusive_ptr<torch::CustomClassHolder>", "c10::intrusive_ptr<caffe2::Blob>",
                   "c10::ArrayRef<c10::intrusive_ptr<c10::ivalue::Object> >",
                   "torch::jit::DetachedBuffer::UniqueDetachedBuffer", "std::optional<at::StepCallbacks>",
                   "std::optional<c10::VaryingShape<int64_t>::ListOfOptionalElements>", "std::optional<c10::VaryingShape<c10::Stride>::ListOfOptionalElements>",
                   "std::optional<std::reference_wrapper<const std::string> >",
                   "std::optional<torch::nn::TripletMarginWithDistanceLossOptions::distance_function_t>",
                   "std::optional<torch::nn::functional::TripletMarginWithDistanceLossFuncOptions::distance_function_t>",
                   "std::tuple<torch::Tensor,std::optional<std::vector<int64_t> >,std::optional<std::vector<double> >,std::optional<bool> >",
                   "std::optional<std::shared_ptr<torch::jit::CompilationUnit> >", "std::optional<std::weak_ptr<torch::jit::CompilationUnit> >",
                   "std::vector<std::shared_ptr<std::string> >", "std::reference_wrapper<const c10::FunctionSchema>",
                   "std::enable_shared_from_this<torch::jit::tensorexpr::Expr>",
                   "std::enable_shared_from_this<c10::Type>",
                   "std::enable_shared_from_this<c10::SharedType>",
                   "std::enable_shared_from_this<c10::SymbolicIntNode>",
                   "std::enable_shared_from_this<torch::autograd::ForwardGrad>",
                   "std::enable_shared_from_this<torch::autograd::GraphTask>",
                   "std::enable_shared_from_this<GraphTask>",
                   "std::enable_shared_from_this<torch::jit::Graph>",
                   "std::enable_shared_from_this<torch::autograd::Node>",
                   "std::enable_shared_from_this<torch::jit::Graph>",
                   "std::enable_shared_from_this<torch::jit::SugaredValue>", "std::enable_shared_from_this<SugaredValue>",
                   "std::enable_shared_from_this<torch::jit::tracer::TracingState>", "std::enable_shared_from_this<TracingState>",
                   "std::enable_shared_from_this<torch::nn::Module>", "std::enable_shared_from_this<Module>"
               ).pointerTypes("Pointer").cast())
               .put(new Info("MTLCommandBuffer_t", "DispatchQueue_t").valueTypes("Pointer").pointerTypes("PointerPointer").skip());


        ///// Special cases needing javaText
        infoMap
            .put(new Info("at::Tensor::toString", "at::TensorBase::toString", "torch::Tensor::toString", "torch::TensorBase::toString", "torch::jit::Graph::toString").javaText("public native @StdString String toString();"))
            .put(new Info("torch::jit::ProfileOp::getCallback()", "torch::jit::ProfileIValueOp::getCallback()").javaText(
                "public native @ByVal @Cast(\"std::function<void(std::vector<c10::IValue>&)>*\") Pointer getCallback();"))
            .put(new Info("torch::optim::AdamOptions::betas", "torch::optim::AdamWOptions::betas").javaText(
                "public native @Cast(\"std::tuple<double,double>*\") @ByRef @NoException DoublePointer betas();"))
            .put(new Info("torch::optim::Adagrad::step", "torch::optim::Adam::step", "torch::optim::AdamW::step",
                "torch::optim::LBFG::step", "torch::optim::RMSprop::step", "torch::optim::SGD::step").javaText(
                "public native @ByVal Tensor step(@ByVal(nullValue = \"torch::optim::Optimizer::LossClosure(nullptr)\") LossClosure closure);\n"
                + "public native @ByVal Tensor step();\n"));


        // Abstract classes not detected as such by Parser (eg because parent class is abstract).
        String[] abstracts = new String[]{
            "torch::nn::InstanceNormImpl<1,torch::nn::InstanceNorm1dImpl>",
            "torch::nn::InstanceNormImpl<2,torch::nn::InstanceNorm2dImpl>",
            "torch::nn::InstanceNormImpl<3,torch::nn::InstanceNorm3dImpl>",
            "torch::nn::InstanceNormImpl<3,torch::nn::InstanceNorm3dImpl>",
            "torch::nn::BatchNormImplBase<1,torch::nn::BatchNorm1dImpl>",
            "torch::nn::BatchNormImplBase<2,torch::nn::BatchNorm2dImpl>",
            "torch::nn::BatchNormImplBase<3,torch::nn::BatchNorm3dImpl>"
        };
        for (String a : abstracts) {
            infoMap.getFirst(a, false).purify();
        }
        infoMap.put(new Info(
            "at::TensorIteratorBase",
            "c10::NamedTensorMetaInterface"
        ).purify());


        //// Function pointers
        // skip() is added when function pointer are parsed instead of std::function to use the class in package
        // functions and prevent the creation of an automatic class in main package.
        // If a native function returns a std::function, no way to map it. So either cast to pointer or skip.
        infoMap
            .put(new Info("void (*)(void*)", "c10::DeleterFnPtr", "torch::Deleter", "at::ContextDeleter",
                "caffe2::TypeMeta::Delete", "std::function<void(void*)>").pointerTypes("PointerConsumer").valueTypes("PointerConsumer").skip())
            .put(new Info("void* (*)()", "caffe2::TypeMeta::New").pointerTypes("PointerSupplier").valueTypes("PointerSupplier").skip())
            .put(new Info("std::function<void()>").pointerTypes("Func"))
            .put(new Info("std::function<std::string()>", "std::function<std::string(void)>").pointerTypes("StringSupplier"))
            .put(new Info("std::function<void(const std::string&)>").pointerTypes("StringConsumer"))
            .put(new Info("std::function<std::string(const std::string&)>").pointerTypes("StringMapper"))
            .put(new Info("std::function<void(const c10::DDPLoggingData&)>",
                "std::function<void(const DDPLoggingData&)>").pointerTypes("DDPLogger"))
            .put(new Info("std::function<c10::TypePtr(c10::TypePtr)>").pointerTypes("TypeMapper"))
            .put(new Info("c10::detail::infer_schema::ArgumentDef::GetTypeFn").pointerTypes("TypeSupplier").skip())
            .put(new Info("c10::TypePtr (*)()", "c10::detail::infer_schema::ArgumentDef::GetTypeFn*").pointerTypes("TypeSupplier").valueTypes("TypeSupplier").skip())
            .put(new Info("std::function<torch::jit::Value*(torch::jit::Value*)>").pointerTypes("ValueMapper"))
            .put(new Info("std::function<void(torch::jit::GraphFunction&)>").pointerTypes("GraphFunctionCreator"))
            .put(new Info("torch::nn::Module::ModuleApplyFunction", "torch::nn::Module::ConstModuleApplyFunction", "std::function<void(const torch::nn::Module&)>", "std::function<void(torch::nn::Module&)>").pointerTypes("ModuleApplyFunction"))
            .put(new Info("std::function<void(const torch::jit::Module&)>", "std::function<void(torch::jit::Module&)>").pointerTypes("JitModuleApplyFunction"))
            .put(new Info("torch::nn::NamedModuleApplyFunction", "torch::nn::ConstNamedModuleApplyFunction", "std::function<void(const std::string&,const torch::nn::Module&)>", "std::function<void(const std::string&,torch::nn::Module&)>").pointerTypes("NamedModuleApplyFunction"))
            .put(new Info("torch::nn::ModulePointerApplyFunction", "std::function<void(const std::shared_ptr<torch::nn::Module>&)>").pointerTypes("SharedModuleApplyFunction"))
            .put(new Info("torch::nn::Module::NamedModulePointerApplyFunction", "std::function<void(const std::string&,const std::shared_ptr<torch::nn::Module>&)>").pointerTypes("NamedSharedModuleApplyFunction"))
            .put(new Info("std::function<void(std::vector<c10::IValue>&)>").pointerTypes("IValueVectorConsumer"))
            .put(new Info("std::function<c10::IValue()>").pointerTypes("IValueSupplier"))
            .put(new Info("std::function<size_t(uint64_t,void*,size_t)>").pointerTypes("Reader"))
            .put(new Info("std::function<size_t(const void*,size_t)>").pointerTypes("ArchiveWriter"))
            .put(new Info("std::function<void(const char*,size_t)>").pointerTypes("PickleWriter"))
            .put(new Info("std::function<size_t(char*,size_t)>").pointerTypes("PickleReader"))
            .put(new Info("std::function<c10::QualifiedName(const std::shared_ptr<c10::ClassType>&)>").pointerTypes("TypeRenamer"))
            .put(new Info("std::function<std::string(const at::Tensor&)>").pointerTypes("TensorIdGetter"))
            .put(new Info("std::function<size_t(void)>").pointerTypes("SizeTSupplier"))
            .put(new Info("std::function<torch::Tensor()>").pointerTypes("LossClosure"))
            .put(new Info("std::function<torch::Tensor(const torch::Tensor&,const torch::Tensor&)>",
                "torch::nn::TripletMarginWithDistanceLossOptions::distance_function_t",
                "torch::nn::functional::TripletMarginWithDistanceLossFuncOptions::distance_function_t").pointerTypes("DistanceFunction"))
            .put(new Info("std::function<void(void*,const void*,size_t)>").pointerTypes("MemCopyFunction"))
            .put(new Info("std::function<void(std::function<void()>)>").pointerTypes("Pointer"))

            .put(new Info("at::TensorBase::register_hook<std::function<void(at::TensorBase)> >").javaNames("register_hook"))
            .put(new Info("at::TensorBase::register_hook<std::function<at::TensorBase(at::TensorBase)> >").javaNames("register_hook"))
            .put(new Info("std::function<void(at::TensorBase)>").pointerTypes("VoidTensorHook"))
            .put(new Info("std::function<at::TensorBase(at::TensorBase)>").pointerTypes("TensorTensorHook"))
            .put(new Info("std::function<at::TensorBase(const at::TensorBase&)>").pointerTypes("TensorTensorRefHook"))
            .put(new Info("std::function<torch::Tensor(const torch::Tensor&)>").pointerTypes("TensorMapper"))
            .put(new Info("at::TensorBase::hook_return_void_t<std::function<void(at::TensorBase)> > ",
                "at::TensorBase::hook_return_void_t<std::function<at::TensorBase(at::TensorBase)> >").valueTypes("int"))
            .put(new Info("std::function<void(const std::string&,const std::map<std::string,std::string>&)>").pointerTypes("MetadataLogger"))
            .put(new Info("std::function<c10::StrongTypePtr(const c10::QualifiedName&)>", "torch::jit::TypeResolver").pointerTypes("TypeResolver"))
            .put(new Info("std::function<c10::TypePtr(const std::string&)>", "torch::jit::TypeParserT",
                "c10::TypePtr (*)(const std::string&)",
                "c10::Type::SingletonOrSharedTypePtr<c10::Type> (*)(const std::string&)"
            ).pointerTypes("TypeParser").skip())
            .put(new Info("std::function<std::optional<std::string>(const c10::Type&)>").pointerTypes("TypePrinter"))
            .put(new Info("void (*)(void*, size_t)", "c10::PlacementDtor", "caffe2::TypeMeta::PlacementNew", "caffe2::TypeMeta::PlacementDelete").pointerTypes("PlacementConsumer").valueTypes("PlacementConsumer").skip())
            .put(new Info("void (*)(const void*, void*, size_t)", "caffe2::TypeMeta::Copy").pointerTypes("PlacementCopier").valueTypes("PlacementCopier").skip())
            .put(new Info("torch::jit::Operation (*)(const torch::jit::Node*)", "torch::jit::OperationCreator").pointerTypes("OperationCreator").valueTypes("OperationCreator").skip())
            .put(new Info("c10::ApproximateClockToUnixTimeConverter::makeConverter").skip()) // Function returning a std::function
            .put(new Info("std::function<c10::intrusive_ptr<c10::ivalue::Object>(const at::StrongTypePtr&,c10::IValue)>", "torch::jit::ObjLoader").pointerTypes("ObjLoader"))
            .put(new Info("std::function<void(std::shared_ptr<c10d::WorkInfo>)>", "std::function<void(c10d::WorkInfo*)").pointerTypes("WorkInfoConsumer"))
            .put(new Info("std::function<bool(torch::Tensor&)>", "torch::distributed::autograd::DistAutogradContext::GradCallback").pointerTypes("GradCallback"))
            .put(new Info("std::function<std::shared_ptr<const c10::LazyValue<std::string> >()>", "std::function<c10::Backtrace()>").pointerTypes("StackTraceFetcher"))

            //// std::function passed as generic pointer because are returned by some methods.
            .put(new Info("std::function<PyObject*(void*)>", "torch::jit::BackendMetaPtr", "std::function<void(const at::Tensor&, std::unordered_map<std::string, bool>&)>")
                .pointerTypes("Pointer").cast())
        ;

        infoMap.put(new Info("caffe2::TypeMeta::deleteFn").javaText("public native @NoException(true) PointerConsumer deleteFn();")); // Parser picks up the wrong Delete

        infoMap.put(new Info("c10::VaryingShape<c10::Stride>::merge").skip()); // https://github.com/pytorch/pytorch/issues/123248, waiting for the fix in 2.3.1 or 2.4

        //// Different C++ API between platforms
        // This will produce different Java codes, but as long as the differences only concern
        // JavaCPP annotations, we don't care.
        if (arm64) {
            infoMap
                .put(new Info("c10::Half::Half(float)").javaText(
                    "public Half(float value) { super((Pointer)null); allocate(value); }\n" +
                    "private native void allocate(@Cast(\"float16_t\") float value);"
                ))
                .put(new Info("c10::Half::operator float()").javaText(
                    "public native @Name(\"operator float16_t\") @Cast(\"float\") float asFloat();"
                ));
        }
    }

    static String template(String t, String... args) {
        StringBuilder sb = new StringBuilder(t);
        sb.append('<');
        for (int i = 0; i < args.length; i++) {
            if (i > 0) sb.append(',');
            sb.append(args[i]);
        }
        if (args[args.length - 1].endsWith(">")) sb.append(' ');
        sb.append('>');
        return sb.toString();
    }

    // Copy from Generator
    private static String mangle(String name) {
        StringBuilder mangledName = new StringBuilder(2 * name.length());
        mangledName.append("JavaCPP_");
        for (int i = 0; i < name.length(); i++) {
            char c = name.charAt(i);
            if ((c >= '0' && c <= '9') || (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z')) {
                mangledName.append(c);
            } else if (c == '_') {
                mangledName.append("_1");
            } else if (c == ';') {
                mangledName.append("_2");
            } else if (c == '[') {
                mangledName.append("_3");
            } else if (c == '.' || c == '/') {
                mangledName.append("_");
            } else {
                String code = Integer.toHexString(c);
                mangledName.append("_0");
                switch (code.length()) {
                    case 1:
                        mangledName.append("0");
                    case 2:
                        mangledName.append("0");
                    case 3:
                        mangledName.append("0");
                    default:
                        mangledName.append(code);
                }
            }
        }
        return mangledName.toString();
    }

    // We cannot add a cppName to an existing info, we must clone the info and change the cpp name
    // keeping the first (main) cppName.
    static private void addCppName(InfoMap infoMap, String... n) {
        Info i = new Info(infoMap.getFirst(n[0]));
        i.cppNames(n);
        infoMap.put(i);
    }

    static class ArrayInfo {
        String baseJavaName;
        String[] elementTypes = new String[0];
        String[] otherCppNames = new String[0];
        String itPointerType;
        String[] otherPointerTypes = new String[0];
        String elementValueType;

        ArrayInfo(String b) {
            baseJavaName = b;
            itPointerType = "@ByPtr " + b;
            elementValueType = "@ByVal " + b;
        }

        ArrayInfo elementTypes(String... vt) {
            elementTypes = vt;
            return this;
        }

        ArrayInfo otherCppNames(String... jn) {
            otherCppNames = jn;
            return this;
        }

        ArrayInfo itPointerType(String p) {
            itPointerType = p;
            return this;
        }

        ArrayInfo elementValueType(String t) {
            elementValueType = t;
            return this;
        }

        ArrayInfo otherPointerTypes(String... p) {
            otherPointerTypes = p;
            return this;
        }

        void mapArrayRef(InfoMap infoMap) {
            String[] cppNames = new String[elementTypes.length * 3 + otherCppNames.length];
            String[] cppNamesIterator = new String[cppNames.length * 2];
            String[] cppNamesRIterator = new String[cppNames.length * 2];
            int n = 0;
            for (String vt : elementTypes) {
                String mainName = cppNames[n++] = template("c10::ArrayRef", vt);
                cppNames[n++] = template("at::ArrayRef", vt);
                cppNames[n++] = template("torch::ArrayRef", vt);
                infoMap.put(new Info(mainName + "(const " + vt + "&)").skip())// Causes SIGSEGV since it just make a pointer to the value
                       .put(new Info(mainName + "(" + vt + "&)").skip());// Parser removes const for non-investigated reasons for some elementTypes (eg Block*)
                // With the following info, any operator<<
                //infoMap.put(new Info(template("c10::operator <<", vt)).javaNames("shiftLeft"));
            }
            for (String on : otherCppNames)
                cppNames[n++] = on;
            n = 0;
            for (String cn : cppNames) {
                cppNamesIterator[n++] = cn + "::iterator";
                cppNamesIterator[n++] = cn + "::const_iterator";
                /*
                infoMap.put(new Info(cn + "::at").javaText(
                    //"@Index(function = \"at\") public native @Const " + elementValueType + "get(@Cast(\"size_t\") long i);\n" +
                    "@ValueSetter @Index(function = \"at\") public native " + baseJavaName + "ArrayRef put(@Cast(\"size_t\") long i, " + elementValueType + " value);"
                ));
                 */
            }
            n = 0;
            for (String cn : cppNames) {
                cppNamesRIterator[n++] = cn + "::reverse_iterator";
                cppNamesRIterator[n++] = cn + "::const_reverse_iterator";
            }

            // Use converting constructor from std::vector when it works to allow passing java array literals.
            // Generator doesn't support passing arrays of Pointers as argument, so elementType must be primitive
            // and not boolean, since ArrayRef<bool> cannot be constructed from a std::vector<bool> bitfield.
            boolean variadicPointerType = elementValueType.equals("byte") || elementValueType.equals("short") ||
                                          elementValueType.equals("int") || elementValueType.equals("long") ||
                                          elementValueType.equals("float") || elementValueType.equals("double");

            int numPt = otherPointerTypes.length + (variadicPointerType ? 2 : 1);
            String[] pt = new String[numPt * numPt]; // List numPt times to help generating all possible combinations
            // when a method takes other arguments having multiple pointerTypes
            pt[0] = baseJavaName + "ArrayRef";
            System.arraycopy(otherPointerTypes, 0, pt, 1, otherPointerTypes.length);
            if (variadicPointerType)
                pt[otherPointerTypes.length + 1] = "@Cast({\"" + elementTypes[0] + "*\", \"" + cppNames[0] + "\", \"std::vector<" + elementTypes[0] + ">&\"}) @StdVector(\"" + elementTypes[0] + "\") " + elementValueType + "...";
            for (int i = 1; i < numPt; i++) {
                pt[i * numPt] = pt[i * numPt - 1];
                System.arraycopy(pt, (i - 1) * numPt, pt, i * numPt + 1, numPt - 1);
            }
            Info info = new Info(cppNames).pointerTypes(pt);
            if (baseJavaName.contains("@Cast")) info.cast();
            infoMap.put(info);
            info = new Info(cppNamesIterator).valueTypes("@Const " + itPointerType);
            infoMap.put(info);
            infoMap.put(new Info(cppNamesRIterator).skip());

            // Add templated constructor taking a std::vector, if the vector class has been mapped.
            // Relies on the fact that std::vector info are created before.
            Info vectorInfo = infoMap.getFirst(template("std::vector", elementTypes[0]), false);
            if (vectorInfo != null && !elementTypes[0].equals("bool"))
                infoMap.put(new Info(template(cppNames[0] + "::ArrayRef", template("std::allocator", elementTypes[0])) + "(" + elementTypes[0] + "*)").javaNames("XXX")
                    .javaText(
                        "public " + baseJavaName + "ArrayRef(@ByRef " + baseJavaName + "Vector vec) { super((Pointer)null); allocate(vec); }\n"
                        + "private native void allocate(@ByRef " + baseJavaName + "Vector vec);"));
        }

        void mapList(InfoMap infoMap) {
            String t = elementTypes[0];
            infoMap.put(new Info(template("c10::List", t)).pointerTypes(baseJavaName + "List"))
                   .put(new Info(
                       template("c10::impl::ListElementReference", t, "typename c10::detail::ListImpl::list_type::iterator"),
                       template("c10::impl::ListElementReference", t, "c10::detail::ListImpl::list_type::iterator"),
                       template("c10::impl::ListElementReference", t, template("std::vector", t) + "::iterator"))
                       .pointerTypes(baseJavaName + "ElementReference"))
                   .put(new Info(template("c10::impl::ListIterator", t, "typename c10::detail::ListImpl::list_type::iterator"),
                       template("c10::impl::ListIterator", t, "c10::detail::ListImpl::list_type::iterator"))
                       .pointerTypes(baseJavaName + "ListIterator"))
                   .put(new Info(template("c10::List", t) + "::value_type").valueTypes(elementValueType))
                   .put(new Info(template("operator std::conditional_t", template("std::is_reference_v", template("c10::detail::ivalue_to_const_ref_overload_return", t) + "::type"), "const " + t + "&", t) + "()")
                       .javaNames("get" + baseJavaName))
                   .put(new Info(template("c10::List", t) + "::size_type").valueTypes("long"))
                   .put(new Info(template("c10::impl::ListElementReference", t, "c10::detail::ListImpl::list_type::iterator") + "::" + template("swap", t, "c10::detail::ListImpl::list_type::iterator"))
                       .javaNames("swap").friendly())
                   .put(new Info(template("c10::List", t) + "::get(" + template("c10::List", t) + "::size_type)").javaText("public native " + elementValueType +" get(long pos);"))
            ;
            Info listElementRefInfo = new Info(template("std::conditional_t", template("std::is_reference_v", template("c10::detail::ivalue_to_const_ref_overload_return", t) + "::type"), "const " + t + "&", t));
            listElementRefInfo.pointerTypes(itPointerType).valueTypes(elementValueType);
            infoMap.put(new Info(template("c10::List", t) + "::operator []").skip()) // Returns an internal_reference_type by value, which is a ListElementReference, whose copy constructor is disabled.
                   .put(new Info(
                       template("c10::impl::ListIterator", t, "c10::detail::ListImpl::list_type::iterator") + "::operator []",
                       template("c10::impl::ListIterator", t, "c10::detail::ListImpl::list_type::iterator") + "::operator *")
                       .skip()) // Returns ListElementReference by value, and ListElementReference has copy constructor disabled.
                   .put(listElementRefInfo)
                   .put(new Info(template("c10::impl::swap", t, "typename c10::detail::ListImpl::list_type::iterator")).javaNames("swap"))
            ;

            // Some List constructors are only for specific instances
            if (baseJavaName.equals("Generic"))
                infoMap.put(new Info(
                    template("c10::List", t) + "(" + template("std::initializer_list", t) + ")",
                    template("c10::List", t) + "(" + template("c10::ArrayRef", t) + ")",
                    template("c10::List", t) + "()"
                ).skip());
            else if (!baseJavaName.equals("Future"))
                infoMap.put(new Info(template("c10::List", t) + "(c10::TypePtr)").skip());
        }
    }

    static class PointerInfo {
        String javaBaseName;
        String javaName;
        final String[] argumentNames;
        String[] otherCppNames = new String[0];
        boolean virtualize = false;

        PointerInfo(String... an) {
            argumentNames = an;
            javaBaseName = an[0].substring(an[0].lastIndexOf(':') + 1);
        }

        PointerInfo otherCppNames(String... n) {
            otherCppNames = n;
            return this;
        }

        PointerInfo javaBaseName(String jn) {
            javaBaseName = jn;
            return this;
        }

        PointerInfo javaName(String jn) {
            javaName = jn;
            return this;
        }

        PointerInfo virtualize() {
            virtualize = true;
            return this;
        }

        void makeShared(InfoMap infoMap) {
            // See issue #670
            String[] cppNamesStrong = new String[argumentNames.length + otherCppNames.length];
            String[] cppNamesWeak = new String[argumentNames.length];
            int i = 0;
            int j = 0;
            for (String n : argumentNames) {
                cppNamesStrong[i++] = template("std::shared_ptr", n);
                cppNamesWeak[j++] = template("std::weak_ptr", n);
            }
            for (String n : otherCppNames) cppNamesStrong[i++] = n;
            // Specifying the parameter of the annotation allows to disambiguate cases where a class can store either a
            // std::shared_ptr<const X> or std::shared_ptr<X> (like CompilationUnit)
            // .valueTypes("@Cast(\"const torch::jit::CompilationUnit*\") CompilationUnit") seems to work too but for obscure reason
            infoMap.put(new Info(cppNamesStrong).annotations("@SharedPtr(\"" + argumentNames[0] + "\")").pointerTypes(javaBaseName));
            infoMap.put(new Info(cppNamesWeak).annotations("@WeakPtr(\"" + argumentNames[0] + "\")").pointerTypes(javaBaseName));


            // Also annotate constructor of target class to ensure only one shared_ptr exists for each instance
            String n = argumentNames[0].substring(argumentNames[0].lastIndexOf(' ') + 1); // Remove possible const
            String n2 = n;
            if (virtualize) {
                n2 = mangle(n2);
                infoMap.put(new Info(n).virtualize());
            } else if (n.equals("torch::nn::Module")) {
                // We don't set virtualize on Module since we don't want all virtual
                // member functions to be annotated @Virtual (clone_, ...)
                n2 = mangle(n2);
            }
            infoMap.put(new Info(n + n.substring(n.lastIndexOf("::"))).annotations("@SharedPtr", "@Name(\"std::make_shared<" + n2 + ">\")"));
        }

        void makeIntrusive(InfoMap infoMap) {
            // See issue #670
            String[] cppNames = new String[argumentNames.length*2 + otherCppNames.length];
            int i = 0;
            for (String n : argumentNames) {
                cppNames[i++] = template("c10::intrusive_ptr", n);
                cppNames[i++] = template("c10::weak_intrusive_ptr", n);
            }
            for (String n : otherCppNames) cppNames[i++] = n;
            // Specifying the parameter of the annotation allows to disambiguate cases where a class can store either a
            // std::shared_ptr<const X> or std::shared_ptr<X> (like CompilationUnit)
            // .valueTypes("@Cast(\"const torch::jit::CompilationUnit*\") CompilationUnit") seems to work too but for obscure reason
            Info info = new Info(cppNames).annotations("@IntrusivePtr(\"" + argumentNames[0] + "\")").pointerTypes(javaBaseName);
            info.valueTypes("@Cast({\"\", \"" + cppNames[0] + "&\"}) " + javaBaseName); // Disambiguate between & and * cast operator for IValue constructors and others
            infoMap.put(info);

            // Also annotate constructor of target class to ensure only one shared_ptr exists for each instance
            String n = argumentNames[0].substring(argumentNames[0].lastIndexOf(' ') + 1); // Remove possible const
            String n2 = n;
            if (virtualize) {
                n2 = mangle(n2);
                infoMap.put(new Info(n).virtualize());
            }
            infoMap.put(new Info(n + n.substring(n.lastIndexOf("::"))).annotations("@IntrusivePtr", "@Name(\"c10::make_intrusive<" + n2 + ">\")"));
        }

        void makeUnique(InfoMap infoMap) {
            // The default info in infoMap is not enough for classes that are elements for containers like vector<unique_ptr<...>>
            String[] cppNames = new String[argumentNames.length + otherCppNames.length];
            int i = 0;
            for (String n : argumentNames) cppNames[i++] = template("std::unique_ptr", n);
            for (String n : otherCppNames) cppNames[i++] = n;
            infoMap.put(new Info(cppNames).annotations("@UniquePtr").pointerTypes(javaBaseName));

            String n = argumentNames[0].substring(argumentNames[0].lastIndexOf(' ') + 1); // Remove possible const
            String n2 = n;
            if (virtualize) {
                n2 = mangle(n2);
                infoMap.put(new Info(n).virtualize());
            }
            infoMap.put(new Info(n + n.substring(n.lastIndexOf("::"))).annotations("@UniquePtr", "@Name(\"std::make_unique<" + n2 + ">\")"));
        }
    }

    @Namespace("std") public static native @MemberGetter @ByRef @Cast("std::istream*") Pointer cin();

    @Namespace("std") public static native @MemberGetter @ByRef @Cast("std::ostream*") Pointer cout();

    @Namespace("std") public static native @MemberGetter @ByRef @Cast("std::ostream*") Pointer cerr();

    @Namespace("std") public static native @MemberGetter @ByRef @Cast("std::ostream*") Pointer clog();

}
