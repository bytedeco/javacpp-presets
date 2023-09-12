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

import org.bytedeco.javacpp.tools.Info;
import org.bytedeco.javacpp.tools.InfoMap;
import org.bytedeco.javacpp.tools.InfoMapper;

import org.bytedeco.openblas.presets.openblas;

/**
 * @author Samuel Audet, Hervé Guillemet
 */
@Properties(
    inherit = openblas.class,
    value = {
        @Platform(
            value = {"linux", "macosx", "windows"},
            compiler = "cpp14",
            define = {"SHARED_PTR_NAMESPACE std", "UNIQUE_PTR_NAMESPACE std"},
            include = {
                "torch/torch.h",
                "ATen/native/TensorShape.h",
                "torch/csrc/jit/serialization/storage_context.h",
                "torch/csrc/jit/serialization/import.h",

                // For inclusion in JNI only, not parsed (compiler needs some complete definitions)
                "torch/csrc/jit/runtime/instruction.h",
                "torch/csrc/jit/serialization/source_range_serialization.h",

                "pytorch_adapters.h"
            },
            link = {"c10", "torch_cpu", "torch"},
            preload = {"gomp@.1", "iomp5", "omp", "tbb@.2", "asmjit", "fbgemm"}
        ),
        @Platform(
            value = {"linux", "macosx", "windows"},
            link = { "c10", "c10_cuda", "torch_cpu", "torch_cuda", "torch" },
            // If nvfuser_codegen is linked and not preloaded, and javacpp cache is empty, we get:
            // Loading nvfuser library failed with: Error in dlopen: libtorch.so: Cannot open...  (function LoadingNvfuserLibrary)
            // The warning disappears once the cache is filled. Probably some obscure race condition.
            preload = {"gomp@.1", "iomp5", "omp", "tbb@.2", "asmjit", "fbgemm", "cupti@.12", "nvfuser_codegen"},
            includepath = {"/usr/local/cuda/include", "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.1/include/"},
            preloadpath = {
                "/usr/local/cuda-12.1/lib64/",
                "/usr/local/cuda-12.1/extras/CUPTI/lib64/",
                "/usr/local/cuda/lib64/",
                "/usr/local/cuda/extras/CUPTI/lib64/",
                "/usr/lib64/",
                "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.1/lib/x64/",
                "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.1/extras/CUPTI/lib64/",
                "C:/Program Files/NVIDIA Corporation/NvToolsExt/bin/x64/",
            },

            extension = "-gpu"
        ),
    },
    target = "org.bytedeco.pytorch",
    global = "org.bytedeco.pytorch.global.torch"
)
public class torch implements LoadEnabled, InfoMapper {
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
        String[] libs = {"cudart", "cublasLt", "cublas", "cufft", "curand", "cusolver", "nvJitLink", "cusparse", "cudnn", "nccl", "nvrtc", "myelin", "nvinfer",
            "cudnn_ops_infer", "cudnn_ops_train", "cudnn_adv_infer", "cudnn_adv_train", "cudnn_cnn_infer", "cudnn_cnn_train"};
        for (String lib : libs) {
            if (platform.startsWith("linux")) {
                lib += lib.startsWith("cudnn") ? "@.8"
                    : lib.equals("nccl") ? "@.2"
                    : lib.equals("myelin") ? "@.1"
                    : lib.equals("nvinfer") ? "@.8"
                    : lib.equals("cufft") ? "@.11"
                    : lib.equals("curand") ? "@.10"
                    : lib.equals("cusolver") ? "@.11"
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
                    : lib.equals("nvJitLink") ? "64_120_0"
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

    String anyModuleConstructors = "";

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
               .put(new Info("torch::nn::" + name).skip());

        if (anyModuleCompatible) {
            anyModuleConstructors +=
                "public AnyModule(" + name + "Impl module) { super((Pointer)null); allocate(module); }\n" +
                // We need a @Cast because AnyModule constructor is explicit
                "private native void allocate(@SharedPtr @Cast({\"\", \"std::shared_ptr<torch::nn::" + name + "Impl>\"}) " + name + "Impl module);\n";
            infoMap.put(new Info("torch::nn::SequentialImpl::push_back<torch::nn::" + name + "Impl>").javaNames("push_back"));
        }
    }

    public static void sharedMap(InfoMap infoMap) {
        infoMap
            .put(new Info().enumerate().friendly())
            .put(new Info("auto", "c10::reverse_iterator", "ska::flat_hash_map", /*"std::atomic", */"std::conditional", "std::iterator_traits",
                "std::initializer_list", "std::integral_constant", "std::mutex", "std::reverse_iterator", "std::weak_ptr").skip())
        ;

        //// Macros
        infoMap
            .put(new Info("TORCH_API", "C10_API", "C10_EXPORT", "C10_HIDDEN", "C10_IMPORT", "C10_API_ENUM", "EXPORT_IF_NOT_GCC",
                "TORCH_CUDA_CU_API", "TORCH_CUDA_CPP_API", "TORCH_HIP_API", "TORCH_PYTHON_API",
                "__ubsan_ignore_float_divide_by_zero__", "__ubsan_ignore_undefined__", "__ubsan_ignore_signed_int_overflow__", "__ubsan_ignore_function__",
                "C10_CLANG_DIAGNOSTIC_IGNORE", "C10_CLANG_DIAGNOSTIC_PUSH", "C10_CLANG_DIAGNOSTIC_POP", "C10_ATTR_VISIBILITY_HIDDEN", "C10_ERASE",
                "C10_UID", "C10_NODISCARD", "C10_UNUSED", "C10_USED", "C10_RESTRICT", "C10_NOINLINE", "C10_ALWAYS_INLINE", "C10_FALLTHROUGH",
                "C10_HOST_DEVICE", "C10_DEVICE", "C10_HOST", "C10_LAUNCH_BOUNDS_0", "C10_HIP_HOST_DEVICE", "C10_WARP_SIZE", "C10_IOS", "C10_MOBILE",
                "C10_HOST_CONSTEXPR", "CONSTEXPR_EXCEPT_WIN_CUDA", "C10_HOST_CONSTEXPR_EXCEPT_WIN_CUDA", "C10_ALWAYS_INLINE_UNLESS_MOBILE",
                "alignas", "COMPLEX_INTEGER_OP_TEMPLATE_CONDITION", "C10_DEVICE_HOST_FUNCTION", "FORCE_INLINE_APPLE",
                "ERROR_UNSUPPORTED_CAST", "LEGACY_CONTIGUOUS_MEMORY_FORMAT", "GFLAGS_DLL_DEFINE_FLAG", "GFLAGS_DLL_DECLARE_FLAG",
                "AT_X", "DEFINE_KEY", "C10_DISPATCHER_INLINE_UNLESS_MOBILE", "TH_DISALLOW_COPY_AND_ASSIGN", "__device__",
                "TORCH_DSA_KERNEL_ARGS", "TORCH_DSA_KERNEL_ARGS_PASS",
                "C10_CUDA_API", "C10_CUDA_IMPORT", "C10_CUDA_EXPORT").cppTypes().annotations())

            .put(new Info("defined(__CUDACC__) || defined(__HIPCC__)",
                "defined(__CUDACC__) && !defined(USE_ROCM)",
                "defined(SYCL_EXT_ONEAPI_BFLOAT16_MATH_FUNCTIONS)",
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
            .put(new Info().javaText("import org.bytedeco.pytorch.functions.*;"))
            .put(new Info().javaText("import org.bytedeco.pytorch.Module;"))
            .put(new Info().javaText("import org.bytedeco.javacpp.annotation.Cast;"))

            .put(new Info("basic/containers").cppTypes("c10::optional", "torch::optional", "c10::variant"))
            .put(new Info("std::nullptr_t").cast().pointerTypes("PointerPointer"))

            .put(new Info("at::CheckedFrom").cast().valueTypes("BytePointer", "String").pointerTypes("PointerPointer")) // Alias to const char*
            .put(new Info("c10::IValue", "at::IValue", "decltype(auto)").pointerTypes("IValue"))
            //             .put(new Info("c10::IValue::operator ==").skip()) // Possible name conflict with IValue.equals
            .put(new Info("std::size_t", "c10::Dict<c10::IValue,c10::IValue>::size_type",
                "c10::Dict<std::string,c10::impl::GenericList>::size_type").cast().valueTypes("long").pointerTypes("SizeTPointer"))
            .put(new Info("approx_time_t").cast().valueTypes("long").pointerTypes("LongPointer"))
            .put(new Info(
                "torch::ExpandingArray<1>", "torch::ExpandingArray<2>", "torch::ExpandingArray<3>", "torch::ExpandingArray<4>",
                "torch::ExpandingArray<D*2>", "torch::ExpandingArray<1*2>", "torch::ExpandingArray<2*2>", "torch::ExpandingArray<3*2>").cast().pointerTypes("LongPointer"))
            .put(new Info("torch::ExpandingArray<1,double>", "torch::ExpandingArray<2,double>", "torch::ExpandingArray<3,double>").cast().pointerTypes("DoublePointer"))
            .put(new Info("torch::ExpandingArrayWithOptionalElem<2>", "torch::ExpandingArrayWithOptionalElem<3>").cast().pointerTypes("LongOptional"))
            .put(new Info("std::pair<std::string,c10::IValue>").pointerTypes("EnumNameValue").define())
            .put(new Info("c10::ClassType::Property").pointerTypes("ClassType.Property"))

            .put(new Info("std::list<std::pair<at::RecordFunctionHandle,int> >").pointerTypes("RecordFunctionHandleIntList").define())
            .put(new Info("at::RecordFunctionHandle").valueTypes("long"))
            .put(new Info("c10::ivalue::Future::FutureError::FutureError").skip()) // This constructor takes a std::string&&  but parser sends a std::string&
            .put(new Info("operator const std::string&()").javaText( // Hopefully targets the one in ConstantString only
                "public native @Const @ByRef @Name(\"operator const std::string&\") @StdString @Override String toString();"
            ))
            .put(new Info("c10::weak_intrusive_ptr<c10::StorageImpl>").pointerTypes("WeakStorage"))

            .put(new Info("torch::monitor::Stat<double>").pointerTypes("DoubleStat"))
            .put(new Info("torch::monitor::Stat<int64_t>").pointerTypes("LongStat"))
            .put(new Info("torch::jit::generic_graph_node_list<torch::jit::Node>").pointerTypes("graph_node_list"))
            .put(new Info("torch::jit::generic_graph_node_list_iterator<torch::jit::Node>").pointerTypes("graph_node_list_iterator"))
            .put(new Info("torch::autograd::Function<torch::nn::CrossMapLRN2d>").pointerTypes("FunctionCrossMapLRN2d"))

            .put(new Info("strong::type<int64_t,_VulkanID,strong::regular,strong::convertible_to<int64_t>,strong::hashable>").pointerTypes("Pointer"))

            .put(new Info("c10::VaryingShape<int64_t>").pointerTypes("LongVaryingShape"))
            .put(new Info("c10::VaryingShape<c10::Stride>").pointerTypes("StrideVaryingShape"))
            .put(new Info("torch::detail::SelectiveStr<false>").pointerTypes("DisabledStr"))
            .put(new Info("torch::detail::SelectiveStr<true>").pointerTypes("EnabledStr"))
            .put(new Info("torch::detail::SelectiveStr<false>::operator const char*",
                "torch::detail::SelectiveStr<true>::operator const char*").
                javaText("public native @Name(\"operator const char*\") @Cast(\"const char*\") BytePointer asBytePointer();"))// Fixes bug where constexpr prevents addition of const in @Name
            .put(new Info("fbgemm::bfloat16", "__nv_bfloat16", "sycl::ext::oneapi::bfloat16").pointerTypes("BFloat16").valueTypes("short", "short", "short"))
            .put(new Info("decltype(::c10::impl::ScalarTypeToCPPType<::c10::ScalarType::Bool>::t)").cast().valueTypes("boolean").pointerTypes("BoolPointer"))
            .put(new Info("decltype(::c10::impl::ScalarTypeToCPPType<::c10::ScalarType::Half>::t)").pointerTypes("Half"))
            .put(new Info("decltype(::c10::impl::ScalarTypeToCPPType<::c10::ScalarType::BFloat16>::t)").pointerTypes("BFloat16"))
            .put(new Info("c10::DataPtr", "at::DataPtr").valueTypes("@Cast({\"\", \"c10::DataPtr&&\"}) @StdMove DataPtr").pointerTypes("DataPtr"))
            .put(new Info("c10::Storage", "at::Storage").valueTypes("@Cast({\"\", \"c10::Storage&&\"}) @StdMove Storage").pointerTypes("Storage"))
            .put(new Info("c10::ClassType").purify().pointerTypes("ClassType")) // Issue #669
            .put(new Info("c10::EnumType").purify().pointerTypes("EnumType")) // Issue #669
            .put(new Info("c10::NamedType").purify().pointerTypes("NamedType")) // Issue #669
            // See comments in PR#668 about a const-agnostic adapter
            .put(new Info("std::unique_ptr<c10::FunctionSchema>").annotations("@UniquePtr")
                                                                 .valueTypes("@Cast({\"\", \"std::unique_ptr<c10::FunctionSchema>&&\"}) FunctionSchema")
                                                                 .pointerTypes("FunctionSchema"))
            .put(new Info("c10::MaybeOwned<at::Tensor>").valueTypes("@Cast({\"\", \"c10::MaybeOwned<at::Tensor>&&\"}) @StdMove TensorMaybeOwned").pointerTypes("TensorMaybeOwned"))
            .put(new Info("c10::MaybeOwned<at::TensorBase>").valueTypes("@Cast({\"\", \"c10::MaybeOwned<at::TensorBase>&&\"}) @StdMove TensorBaseMaybeOwned").pointerTypes("TensorBaseMaybeOwned"))
            .put(new Info("c10::MaybeOwnedTraits<at::Tensor>").pointerTypes("MaybeOwnedTraitsTensor"))
            .put(new Info("at::InferExpandGeometryResult<at::DimVector>").pointerTypes("DimVectorInferExpandGeometryResult"))
            .put(new Info("at::namedinference::TensorName").valueTypes("@Cast({\"\", \"at::namedinference::TensorName&&\"}) @StdMove TensorName").pointerTypes("TensorName"))
            .put(new Info("c10::remove_symint<c10::SymInt>::type").valueTypes("long"))
            .put(new Info("std::aligned_storage_t<sizeof(IValue),alignof(IValue)>").pointerTypes("Pointer"))
            .put(new Info("c10::TensorImpl::identity<c10::SymInt>").pointerTypes("SymIntIdentity"))
            .put(new Info("c10::TensorImpl::identity<int64_t>").pointerTypes("LongIdentity"))
            .put(new Info("c10::requires_grad", "at::range", "at::bernoulli_out", "at::normal_out", "at::stft").skipDefaults())
            .put(new Info("c10::prim::requires_grad").javaNames("requires_grad"))
            .put(new Info("c10::aten::clone").javaNames("_clone"))
            .put(new Info("c10::TensorOptions<c10::Device>").javaNames("TensorOptions"))
            .put(new Info("c10::detail::_str<CompileTimeEmptyString>").javaNames("_strCompileTimeEmptyString"))
            .put(new Info("at::TensorBase").base("AbstractTensor").pointerTypes("TensorBase"))
        ;

        //// Enumerations
        infoMap
            .put(new Info("c10::ScalarType", "at::ScalarType", "torch::Dtype").enumerate().valueTypes("ScalarType").pointerTypes("@Cast(\"c10::ScalarType*\") BytePointer"))
            .put(new Info("torch::jit::AttributeKind").enumerate().valueTypes("JitAttributeKind"))
            .put(new Info("torch::jit::PickleOpCode").enumerate().translate(false).valueTypes("PickleOpCode"))
        ;

        //// c10::optional
        infoMap
            .put(new Info("c10::optional<bool>").pointerTypes("BoolOptional").define())
            .put(new Info("c10::optional<int8_t>", "c10::optional<c10::DeviceIndex>").pointerTypes("ByteOptional").define())
            .put(new Info("c10::optional<int>", "c10::optional<int32_t>").pointerTypes("IntOptional").define())
            .put(new Info("c10::optional<int64_t>", "c10::remove_symint<c10::optional<c10::SymInt> >::type").pointerTypes("LongOptional").define())
            .put(new Info("c10::optional<float>").pointerTypes("FloatOptional").define())
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
            .put(new Info("c10::optional<std::vector<torch::Tensor> >").pointerTypes("TensorVectorOptional").define())
            .put(new Info("c10::optional<c10::Device>", "c10::optional<at::Device>", "c10::optional<torch::Device>").pointerTypes("DeviceOptional").define())
            .put(new Info("c10::optional<c10::ArrayRef<int64_t> >", "c10::optional<c10::IntArrayRef>", "c10::optional<at::IntArrayRef>",
                "c10::OptionalArrayRef<int64_t>", "c10::OptionalIntArrayRef", "at::OptionalIntArrayRef", "c10::remove_symint<at::OptionalSymIntArrayRef>::type")
                // This second pointer type prevents optional.swap to work. I don't know exactly why. Skipping swap for now.
                .pointerTypes("LongArrayRefOptional", "@Cast({\"int64_t*\", \"c10::ArrayRef<int64_t>\", \"std::vector<int64_t>&\"}) @StdVector long...").define())
            .put(new Info("c10::optional<c10::ArrayRef<int64_t> >::swap").skip())
            .put(new Info("c10::optional<c10::ArrayRef<double> >", "c10::optional<at::ArrayRef<double> >",
                "c10::OptionalArrayRef<double>").pointerTypes("DoubleArrayRefOptional").define())
            .put(new Info("c10::optional<c10::ArrayRef<c10::SymInt> >", "c10::optional<at::ArrayRef<c10::SymInt> >",
                "c10::OptionalArrayRef<c10::SymInt>", "c10::OptionalSymIntArrayRef", "at::OptionalSymIntArrayRef", "c10::optional<c10::SymIntArrayRef>").pointerTypes("SymIntArrayRefOptional").define())
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
            .put(new Info("c10::optional<c10::AliasTypeSet>").pointerTypes("AliasTypeSetOptional").define())
            .put(new Info("c10::optional<c10::FunctionSchema>").pointerTypes("FunctionSchemaOptional").define())
            .put(new Info("c10::optional<c10::SymDimVector>", "c10::optional<at::SymDimVector>").pointerTypes("SymDimVectorOptional").define())
            .put(new Info("c10::optional<c10::SymInt>").pointerTypes("SymIntOptional").define())
            .put(new Info("c10::optional<at::IValue>").pointerTypes("IValueOptional").define())
            .put(new Info("c10::optional<at::DimVector>").pointerTypes("DimVectorOptional").define())
            .put(new Info("c10::optional<at::Dimname>").pointerTypes("DimnameOptional").define())
            .put(new Info("c10::optional<at::DimnameList>").pointerTypes("DimnameListOptional").define())
            .put(new Info("c10::optional<at::Generator>").pointerTypes("GeneratorOptional").define())
            .put(new Info("c10::optional<at::Tensor>", "c10::optional<torch::Tensor>", "c10::optional<at::Tensor>", "c10::optional<torch::TensorBase>", "c10::optional<torch::autograd::Variable>").pointerTypes("TensorOptional").define())
            .put(new Info("c10::optional<torch::TensorList>", "c10::optional<at::TensorList>").pointerTypes("TensorArrayRefOptional").define())
            .put(new Info("c10::optional<at::ThreadLocalState>").pointerTypes("ThreadLocalStateOptional").define())
            .put(new Info("c10::optional<caffe2::TypeMeta>").pointerTypes("TypeMetaOptional").define())
            .put(new Info("c10::optional<torch::jit::ExecutorExecutionMode>").pointerTypes("ExecutorExecutionModeOptional").define())
            .put(new Info("c10::optional<torch::jit::ExecutorExecutionMode>::operator ->").skip()) // Returns a pointer to ExecutorExecutionMode, which is an enum
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
            .put(new Info("c10::optional<std::tuple<std::string,size_t,size_t> >").pointerTypes("T_StringSizeTSizeT_TOptional").define())
            .put(new Info("torch::optional<std::tuple<torch::Tensor,torch::Tensor> >").pointerTypes("T_TensorTensor_TOptional").define())
            .put(new Info("c10::optional<std::tuple<c10::TypePtr,int32_t> >", "c10::optional<std::pair<c10::TypePtr,int32_t> >").pointerTypes("T_TypePtrLong_TOptional").cast().define())
            .put(new Info("c10::optional<c10::string_view>").pointerTypes("StringViewOptional").define())
            .put(new Info("c10::optional<std::vector<c10::string_view> >").pointerTypes("StringViewVectorOptional").define())
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


        //// c10::variant
        infoMap
            .put(new Info("c10::variant<torch::enumtype::kLinear,torch::enumtype::kConv1D,torch::enumtype::kConv2D,torch::enumtype::kConv3D,"
                          + "torch::enumtype::kConvTranspose1D,torch::enumtype::kConvTranspose2D,torch::enumtype::kConvTranspose3D,"
                          + "torch::enumtype::kSigmoid,torch::enumtype::kTanh,torch::enumtype::kReLU,torch::enumtype::kLeakyReLU>",
                "torch::nn::init::NonlinearityType").pointerTypes("Nonlinearity").define())
            .put(new Info("c10::variant<torch::enumtype::kFanIn,torch::enumtype::kFanOut>",
                "torch::nn::init::FanModeType").pointerTypes("FanModeType").define())

            .put(new Info("c10::variant<torch::enumtype::kZeros,torch::enumtype::kReflect,torch::enumtype::kReplicate,torch::enumtype::kCircular>",
                "torch::nn::ConvOptions<1>::padding_mode_t",
                "torch::nn::ConvOptions<2>::padding_mode_t",
                "torch::nn::ConvOptions<3>::padding_mode_t",
                "torch::nn::ConvTransposeOptions<1>::padding_mode_t",
                "torch::nn::ConvTransposeOptions<2>::padding_mode_t",
                "torch::nn::ConvTransposeOptions<3>::padding_mode_t",
                "torch::nn::detail::conv_padding_mode_t").pointerTypes("ConvPaddingMode").define())
            .put(new Info("c10::variant<torch::ExpandingArray<1>,torch::enumtype::kValid,torch::enumtype::kSame>",
                "torch::nn::ConvOptions<1>::padding_t",
                "torch::nn::detail::ConvNdOptions<1>::padding_t",
                "torch::nn::functional::ConvFuncOptions<1>::padding_t",
                "torch::nn::functional::Conv1dFuncOptions::padding_t").purify().pointerTypes("Conv1dPadding").define())
            .put(new Info("c10::variant<torch::ExpandingArray<2>,torch::enumtype::kValid,torch::enumtype::kSame>",
                "torch::nn::ConvOptions<2>::padding_t",
                "torch::nn::detail::ConvNdOptions<2>::padding_t",
                "torch::nn::functional::ConvFuncOptions<2>::padding_t",
                "torch::nn::functional::Conv2dFuncOptions::padding_t").purify().pointerTypes("Conv2dPadding").define())
            .put(new Info("c10::variant<torch::ExpandingArray<3>,torch::enumtype::kValid,torch::enumtype::kSame>",
                "torch::nn::ConvOptions<3>::padding_t",
                "torch::nn::detail::ConvNdOptions<3>::padding_t",
                "torch::nn::functional::ConvFuncOptions<3>::padding_t",
                "torch::nn::functional::Conv3dFuncOptions::padding_t").purify().pointerTypes("Conv3dPadding").define())

            .put(new Info("c10::variant<torch::enumtype::kSum,torch::enumtype::kMean,torch::enumtype::kMax>",
                "torch::nn::EmbeddingBagMode").pointerTypes("EmbeddingBagMode").define())
            .put(new Info("c10::variant<torch::enumtype::kConstant,torch::enumtype::kReflect,torch::enumtype::kReplicate,torch::enumtype::kCircular>",
                "torch::nn::functional::PadFuncOptions::mode_t").pointerTypes("PaddingMode").define())

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
                "torch::nn::BCEWithLogitsLossOptions::reduction_t", "torch::nn::functional::BinaryCrossEntropyWithLogitsFuncOptions::reduction_t").pointerTypes("LossReduction").define())
            .put(new Info("c10::variant<torch::enumtype::kNone,torch::enumtype::kBatchMean,torch::enumtype::kSum,torch::enumtype::kMean>",
                "torch::nn::KLDivLossOptions::reduction_t", "torch::nn::functional::KLDivFuncOptions::reduction_t").pointerTypes("KLDivLossReduction").define())

            .put(new Info("c10::variant<torch::enumtype::kBilinear,torch::enumtype::kNearest>",
                "torch::nn::functional::GridSampleFuncOptions::mode_t").pointerTypes("GridSampleMode").define())
            .put(new Info("c10::variant<torch::enumtype::kZeros,torch::enumtype::kBorder,torch::enumtype::kReflection>",
                "torch::nn::functional::GridSampleFuncOptions::padding_mode_t").pointerTypes("GridSamplePaddingMode").define())

            .put(new Info("c10::variant<torch::enumtype::kLSTM,torch::enumtype::kGRU,torch::enumtype::kRNN_TANH,torch::enumtype::kRNN_RELU>",
                "torch::nn::detail::RNNOptionsBase::rnn_options_base_mode_t").pointerTypes("RNNBaseMode").define())
            .put(new Info("c10::variant<torch::enumtype::kTanh,torch::enumtype::kReLU>",
                "torch::nn::RNNOptions::nonlinearity_t", "torch::nn::RNNCellOptions::nonlinearity_t").pointerTypes("RNNNonlinearity").define())

            .put(new Info("c10::variant<torch::enumtype::kNearest,torch::enumtype::kLinear,torch::enumtype::kBilinear,torch::enumtype::kBicubic,torch::enumtype::kTrilinear>",
                "torch::nn::UpsampleOptions::mode_t").pointerTypes("UpsampleMode").define())
            .put(new Info("c10::variant<torch::enumtype::kNearest,torch::enumtype::kLinear,torch::enumtype::kBilinear,torch::enumtype::kBicubic,torch::enumtype::kTrilinear,torch::enumtype::kArea,torch::enumtype::kNearestExact>",
                "torch::nn::functional::InterpolateFuncOptions::mode_t").pointerTypes("InterpolateMode").define())

            .put(new Info("c10::variant<torch::enumtype::kReLU,torch::enumtype::kGELU,std::function<torch::Tensor(const torch::Tensor&)> >",
                "torch::nn::activation_t",
                "torch::nn::TransformerOptions::activation_t").pointerTypes("TransformerActivation")) // Defined explicitly
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
        ;


        //// std::vector
        infoMap
            .put(new Info("std::vector<std::array<bool,2> >").pointerTypes("Bool2Vector").define())
            .put(new Info("std::vector<bool>").pointerTypes("BoolVector").define())
            .put(new Info("std::vector<const char*>").pointerTypes("BytePointerVector").define())
            .put(new Info("std::vector<int64_t>", "std::tuple<std::vector<int64_t>,std::vector<int64_t> >").cast().pointerTypes("LongVector").define())
            .put(new Info("std::vector<double>").cast().pointerTypes("DoubleVector").define())
            .put(new Info("std::vector<size_t>").cast().pointerTypes("SizeTVector").define())
            .put(new Info("std::vector<std::string>").pointerTypes("StringVector").define())
            .put(new Info("std::vector<c10::string_view>").pointerTypes("StringViewVector").define())
            .put(new Info("std::vector<std::pair<std::string,int64_t> >").pointerTypes("StringLongVector").define())
            .put(new Info("const std::vector<std::pair<at::RecordFunctionCallback,uint64_t> >",
                "std::vector<std::pair<at::RecordFunctionCallback,at::CallbackHandle> >").pointerTypes("RecordFunctionCallbackHandleVector").define())
            .put(new Info("std::vector<c10::IValue>", "torch::jit::Stack").pointerTypes("IValueVector").define())
            .put(new Info("std::vector<c10::IValue>::const_iterator", "torch::jit::Stack::const_iterator").pointerTypes("IValueVector.Iterator"))
            .put(new Info("std::vector<c10::QEngine>", "std::vector<at::QEngine>").pointerTypes("QEngineVector").define())
            .put(new Info("std::vector<c10::ScalarType>").pointerTypes("ScalarTypeVector").define())
            .put(new Info("std::vector<c10::Symbol>").pointerTypes("SymbolVector").define())
            .put(new Info("std::vector<c10::optional<int64_t> >").pointerTypes("LongOptionalVector").define())
            .put(new Info("std::vector<c10::optional<at::IValue> >").pointerTypes("IValueOptionalVector").define())
            .put(new Info("std::vector<std::shared_ptr<c10::ClassType> >", "std::vector<c10::ClassTypePtr>").pointerTypes("SharedClassTypeVector").define())
            .put(new Info("std::vector<c10::Type::SingletonOrSharedTypePtr<c10::Type> >", "std::vector<c10::TypePtr>",
                "std::vector<c10::Type::TypePtr>", "c10::AliasTypeSet").pointerTypes("TypeVector").define())
            .put(new Info("const std::vector<at::Dimname>", "std::vector<at::Dimname>").valueTypes("@StdMove DimnameVector").pointerTypes("DimnameVector").define())
            .put(new Info("std::vector<c10::Stride>").pointerTypes("StrideVector").define())
            .put(new Info("std::vector<c10::ShapeSymbol>").pointerTypes("ShapeSymbolVector").define())
            .put(new Info("std::vector<c10::TensorImpl*>").pointerTypes("TensorImplVector").define())
            .put(new Info("std::vector<torch::autograd::Edge>", "torch::autograd::edge_list") // Used in Node constructor
                                                                                              .valueTypes("@Cast({\"\", \"std::vector<torch::autograd::Edge>\"}) @StdMove EdgeVector").pointerTypes("EdgeVector").define())
            .put(new Info("std::vector<torch::Tensor>", "std::vector<at::Tensor>", "std::vector<torch::autograd::Variable>", "torch::autograd::variable_list")
                .valueTypes("@Cast({\"\", \"std::vector<torch::Tensor>\"}) @StdMove TensorVector").pointerTypes("TensorVector").define())
            .put(new Info("std::vector<at::indexing::TensorIndex>", "std::vector<at::indexing::TensorIndex,A>").pointerTypes("TensorIndexVector").define())
            .put(new Info("std::vector<c10::optional<torch::autograd::Variable> >").pointerTypes("TensorOptionalVector").define())
            .put(new Info("std::vector<c10::optional<torch::jit::Operator> >").pointerTypes("OperatorOptionalVector").define())
            .put(new Info("std::vector<std::shared_ptr<torch::autograd::FunctionPreHook> >").pointerTypes("SharedFunctionPreVector").define())
            .put(new Info("const std::vector<std::unique_ptr<torch::autograd::FunctionPreHook> >",
                "std::vector<std::unique_ptr<torch::autograd::FunctionPreHook> >").pointerTypes("FunctionPreHookVector").define())
            .put(new Info("const std::vector<std::unique_ptr<torch::autograd::FunctionPostHook> >",
                "std::vector<std::unique_ptr<torch::autograd::FunctionPostHook> >").pointerTypes("FunctionPostHookVector").define())
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
            .put(new Info("std::vector<torch::jit::StackEntry>").pointerTypes("StackEntryVector").define())
            .put(new Info("std::vector<torch::jit::Value*>", "std::vector<Value*>").pointerTypes("ValueVector").define()) // Returned by inlineCallTo
            .put(new Info("std::vector<const torch::jit::Node*>").pointerTypes("JitNodeVector").define())
            .put(new Info("std::vector<torch::nn::Module>").pointerTypes("ModuleVector").define())
            .put(new Info("std::vector<torch::nn::Module>::iterator").pointerTypes("ModuleVector.Iterator"))
            .put(new Info("std::vector<torch::nn::AnyModule>").pointerTypes("AnyModuleVector").define())
            .put(new Info("std::vector<torch::nn::AnyModule>::iterator").pointerTypes("AnyModuleVector.Iterator"))
            .put(new Info("std::vector<std::shared_ptr<torch::nn::Module> >").pointerTypes("SharedModuleVector").define())
            .put(new Info("std::vector<std::shared_ptr<torch::nn::Module> >::iterator").pointerTypes("SharedModuleVector.Iterator"))
            .put(new Info("std::vector<std::pair<std::string,torch::Tensor> >").pointerTypes("StringTensorVector").define())
            .put(new Info("std::vector<std::pair<std::string,torch::nn::Module> >").pointerTypes("StringModuleVector").define())
            .put(new Info("std::vector<std::pair<std::string,torch::nn::AnyModule> >").pointerTypes("StringAnyModuleVector").define())
            .put(new Info("std::vector<std::pair<std::string,std::shared_ptr<torch::nn::Module> > >").pointerTypes("StringSharedModuleVector").define())
            .put(new Info("std::vector<std::pair<torch::jit::FusionBehavior,size_t> >", "torch::jit::FusionStrategy").pointerTypes("FusionStrategy").define())
            .put(new Info("std::vector<c10::SymInt>").pointerTypes("SymIntVector").define())
            .put(new Info("std::vector<std::shared_ptr<torch::jit::SugaredValue> >").pointerTypes("SharedSugaredValueVector").define())
            .put(new Info("const std::vector<const c10::FunctionSchema*>").pointerTypes("FunctionSchemaVector").define())
        ;


        //// c10::ArrayRef
        for (ArrayInfo t : new ArrayInfo[]{
            new ArrayInfo("Argument").elementTypes("c10::Argument"),
            new ArrayInfo("ArgumentDef").elementTypes("c10::detail::infer_schema::ArgumentDef"),
            new ArrayInfo("BFloat16") /*.itPointerType("ShortPointer") */.elementTypes("decltype(::c10::impl::ScalarTypeToCPPType<::c10::ScalarType::BFloat16>::t)"),
            new ArrayInfo("Block").elementTypes("torch::jit::Block*").itPointerType("PointerPointer<Block>"),
            new ArrayInfo("Bool").itPointerType("BoolPointer").elementTypes("bool", "decltype(::c10::impl::ScalarTypeToCPPType<::c10::ScalarType::Bool>::t)").elementValueType("boolean"),
            new ArrayInfo("Byte").itPointerType("BytePointer").elementTypes("jbyte", "int8_t", "uint8_t").elementValueType("byte"),
            new ArrayInfo("Dimname").otherCppNames("at::DimnameList").elementTypes("at::Dimname"),
            new ArrayInfo("Double").itPointerType("DoublePointer").elementTypes("double"),
            new ArrayInfo("DoubleComplex") /*.itPointertype("DoublePointer") */.elementTypes("c10::complex<double>"),
            new ArrayInfo("EnumNameValue").elementTypes("c10::EnumNameValue"),
            new ArrayInfo("Float").itPointerType("FloatPointer").elementTypes("float").elementValueType("float"),
            new ArrayInfo("FloatComplex") /*.itPointerType("FloatPointer") */.elementTypes("c10::complex<float>"),
            new ArrayInfo("FuturePtr").elementTypes("c10::intrusive_ptr<c10::ivalue::Future>"),
            new ArrayInfo("Half") /*.itPointerType("ShortPointer") */.elementTypes("decltype(::c10::impl::ScalarTypeToCPPType<::c10::ScalarType::Half>::t)"),
            new ArrayInfo("IValue").elementTypes("c10::IValue", "const at::IValue"),
            new ArrayInfo("Int")
                .itPointerType("IntPointer")
                .elementTypes("jint", "int", "int32_t", "uint32_t")
                .elementValueType("int"),
            new ArrayInfo("Tag").itPointerType("BytePointer").elementTypes("at::Tag"),
            new ArrayInfo("Long") // Warning : c10::IntArrayRef is a Java LongArrayRef and not a Java IntArrayRef
                                  .otherCppNames("c10::IntArrayRef", "torch::IntArrayRef", "at::IntArrayRef", "c10::OptionalArray<int64_t>", "c10::remove_symint<c10::SymIntArrayRef>::type")
                                  .itPointerType("LongPointer")
                                  .otherPointerTypes("@Cast({\"int64_t*\", \"c10::ArrayRef<int64_t>\", \"std::vector<int64_t>&\"}) @StdVector long...")
                                  .elementTypes("int64_t", "jlong") // Order is important, since ArrayRef<long> and ArrayRef<long long> are incompatible, even though long == long long. And jlong is long long.
                                  .elementValueType("long"),
            new ArrayInfo("LongOptional").elementTypes("c10::optional<int64_t>"),
            new ArrayInfo("LongVector").elementTypes("std::vector<int64_t>"),
            new ArrayInfo("NamedValue").elementTypes("torch::jit::NamedValue"),
            new ArrayInfo("SavedVariable").elementTypes("torch::autograd::SavedVariable"),
            new ArrayInfo("Scalar").elementTypes("at::Scalar"),
            new ArrayInfo("ScalarType").itPointerType("@Cast(\"c10::ScalarType*\") BytePointer").elementTypes("c10::ScalarType", "at::ScalarType"),
            new ArrayInfo("Short").itPointerType("ShortPointer").elementTypes("jshort", "int16_t", "uint16_t").elementValueType("short"),
            new ArrayInfo("SizeT").itPointerType("SizeTPointer").elementTypes("size_t").elementValueType("long"),
            new ArrayInfo("Stride").elementTypes("c10::Stride"),
            new ArrayInfo("String").itPointerType("PointerPointer<BytePointer>" /*"@Cast({\"\", \"std::string*\"}) @StdString BytePointer"*/).elementTypes("std::string"),
            new ArrayInfo("SymInt").otherCppNames("c10::SymIntArrayRef").elementTypes("c10::SymInt"),
            new ArrayInfo("SymNode").elementTypes("c10::SymNode", "c10::intrusive_ptr<c10::SymNodeImpl>"),
            new ArrayInfo("Symbol").elementTypes("c10::Symbol"),
            new ArrayInfo("Tensor").otherCppNames("torch::TensorList", "at::ITensorListRef").elementTypes("torch::Tensor", "at::Tensor"),  // Warning: not a TensorList (List<Tensor>)
            new ArrayInfo("TensorArg").elementTypes("torch::TensorArg", "at::TensorArg"),
            new ArrayInfo("TensorIndex").elementTypes("at::indexing::TensorIndex"),
            new ArrayInfo("TensorOptional").elementTypes("c10::optional<at::Tensor>", "c10::optional<torch::Tensor>", "c10::optional<torch::autograd::Variable>"),
            new ArrayInfo("Type").itPointerType("Type.TypePtr").elementTypes("c10::TypePtr", "c10::Type::TypePtr"),
            new ArrayInfo("Value").elementTypes("torch::jit::Value*")

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
            new ArrayInfo("TensorOptional").elementTypes("c10::optional<at::Tensor>"),
            new ArrayInfo("Tensor").elementTypes("at::Tensor"),
            new ArrayInfo("FuturePtr").elementTypes("c10::intrusive_ptr<c10::ivalue::Future>"),
            new ArrayInfo("Generic").elementTypes("c10::IValue").itPointerType("IValue").elementValueType("@ByVal IValue"),
        }) {
            ai.mapList(infoMap);
        }
        // swap is a friend templated function. Parser fails to perform template substitution in this case.
        infoMap.put(new Info("c10::impl::ListElementReference::swap<T,Iterator>").skip());
        // friendly global setting lost
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
            {"Long", "LongPointer", "long", "int64_t", "at::kDimVectorStaticSize", "at::DimVector", "DimVector"},
            {"Node", "Node", "@ByPtr Node", "torch::autograd::Node*", "4", null, "SmallNodeVector"},
            {"TreeRef", "TreeRef", "@ByVal TreeRef", "c10::intrusive_ptr<torch::jit::Tree>", "4", null, "TreeList"}

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
            .put(new Info("std::map<std::string,int>").pointerTypes("StringIntMap").define())
            .put(new Info("std::map<std::string,int64_t>").pointerTypes("StringLongMap").define())
            .put(new Info("std::map<std::string,torch::Tensor>").pointerTypes("StringTensorMap").define())
        ;


        //// std::unordered_set
        infoMap
            .put(new Info("std::unordered_set<std::string>").pointerTypes("StringSet").define())
            .put(new Info("std::unordered_set<c10::IValue,c10::IValue::HashAliasedIValue,c10::IValue::CompAliasedIValues>").pointerTypes("HashAliasedIValues").define())
            .put(new Info("std::unordered_set<c10::Symbol>").pointerTypes("SymbolSet").define())
            .put(new Info("std::unordered_set<torch::TensorImpl*>", "std::unordered_set<at::TensorImpl*>").pointerTypes("TensorImplSet").define())
            .put(new Info("std::unordered_set<at::RecordScope,std::hash<at::RecordScope> >").pointerTypes("RecordScopeSet").define())
            .put(new Info("std::unordered_set<torch::autograd::Node*>").pointerTypes("NodeSet").define())
            .put(new Info("std::unordered_set<c10::Stream>").pointerTypes("StreamSet").define())
            .put(new Info("std::unordered_set<at::RecordScope>").pointerTypes("RecordScopeSet").define())
            .put(new Info("std::set<torch::profiler::impl::ActivityType>").pointerTypes("ActivityTypeSet").define())
        ;


        //// std::unordered_map
        infoMap
            .put(new Info("std::unordered_map<torch::autograd::Node*,int>").pointerTypes("NodeIntMap").define())
            .put(new Info("std::unordered_map<c10::IValue,c10::IValue,c10::IValue::HashAliasedIValue,c10::IValue::CompAliasedIValues>").pointerTypes("HashAliasedIValueMap").define())
            .put(new Info("std::unordered_map<int64_t,std::string>").pointerTypes("LongStringMap").define())
            .put(new Info("std::unordered_map<std::string,bool>").pointerTypes("StringBoolMap").define())
            .put(new Info("std::unordered_map<std::string,size_t>").pointerTypes("StringSizeTMap").define())
            .put(new Info("std::unordered_map<std::string,std::string>").pointerTypes("ExtraFilesMap").define())
            .put(new Info("std::unordered_map<std::string,c10::TypePtr>").pointerTypes("TypeEnv").define())
            .put(new Info("std::unordered_map<std::string,c10::IValue>", "std::unordered_map<std::string,at::IValue>").pointerTypes("StringIValueMap").define())
            .put(new Info("std::unordered_map<std::string,std::function<PyObject*(void*)> >").pointerTypes("StringFunctionMap").define())
            .put(new Info("std::unordered_map<std::string,torch::jit::Value*>").pointerTypes("StringValueMap").define())
            .put(new Info("std::unordered_map<std::string,std::unordered_map<int64_t,std::string> >").pointerTypes("StringLongStringMapMap").define())
            .put(new Info("std::unordered_map<torch::jit::Value*,torch::jit::Value*>").pointerTypes("ValueValueMap").define())
            .put(new Info("std::unordered_map<torch::jit::ArgumentSpec,torch::jit::ExecutionPlan>").pointerTypes("ArgumentSpecExecutionPlanMap").define())
            .put(new Info("std::unordered_map<torch::jit::TreeRef,std::string>").pointerTypes("TreeRefStringMap").define())
        ;


        //// std::atomic
        infoMap
            .put(new Info("std::atomic_bool", "std::atomic<bool>").cast().valueTypes("boolean").pointerTypes("BoolPointer"))
            .put(new Info("std::atomic_uint64_t", "std::atomic<uint64_t>", "std::atomic<long unsigned int>", "std::atomic_size_t", "std::atomic<size_t>").cast().valueTypes("long").pointerTypes("LongPointer"))
            .put(new Info("std::atomic<const c10::impl::DeviceGuardImplInterface*>").cast().pointerTypes("DeviceGuardImplInterface"))
        ;


        //// std::tuple
        infoMap
            .put(new Info("std::tuple<int,int>").pointerTypes("T_IntInt_T").define())
            .put(new Info("std::tuple<int64_t,int64_t>").pointerTypes("T_LongLong_T").define())
            .put(new Info("std::tuple<double,int64_t>").pointerTypes("T_DoubleLong_T").define())
            //.put(new Info("std::tuple<torch::Tensor>").pointerTypes("TensorTuple").define())
            .put(new Info("std::tuple<torch::Tensor,torch::Tensor>", "std::tuple<at::Tensor,at::Tensor>", "std::tuple<torch::Tensor,torch::Tensor>", "std::tuple<at::Tensor&,at::Tensor&>").pointerTypes("T_TensorTensor_T").define())
            .put(new Info("std::tuple<torch::Tensor,torch::Tensor,torch::Tensor>", "std::tuple<at::Tensor,at::Tensor,at::Tensor>", "std::tuple<at::Tensor&,at::Tensor&,at::Tensor&>").pointerTypes("T_TensorTensorTensor_T").define())
            .put(new Info("std::tuple<torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor>", "std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor>", "std::tuple<at::Tensor&,at::Tensor&,at::Tensor&,at::Tensor&>").pointerTypes("T_TensorTensorTensorTensor_T").define())
            .put(new Info("std::tuple<torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor>", "std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor>", "std::tuple<at::Tensor&,at::Tensor&,at::Tensor&,at::Tensor&,at::Tensor&>").pointerTypes("T_TensorTensorTensorTensorTensor_T").define())
            .put(new Info("std::tuple<torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor>", "std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor>", "std::tuple<at::Tensor&,at::Tensor&,at::Tensor&,at::Tensor&,at::Tensor&,at::Tensor&>").pointerTypes("T_TensorTensorTensorTensorTensorTensor_T").define())
            .put(new Info("std::tuple<torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor>", "std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor>", "std::tuple<at::Tensor&,at::Tensor&,at::Tensor&,at::Tensor&,at::Tensor&,at::Tensor&,at::Tensor&>").pointerTypes("T_TensorTensorTensorTensorTensorTensorTensor_T").define())
            .put(new Info("std::tuple<torch::Tensor,torch::Tensor,torch::Tensor,std::vector<torch::Tensor> >", "std::tuple<at::Tensor,at::Tensor,at::Tensor,std::vector<at::Tensor> >").pointerTypes("T_TensorTensorTensorTensorVector_T").define())
            .put(new Info("std::tuple<torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,int64_t>", "std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,int64_t>").pointerTypes("T_TensorTensorTensorTensorLong_T").define())
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
            .put(new Info("std::tuple<std::string,uint64_t>").pointerTypes("T_StringLong_T").define())
            .put(new Info("std::tuple<torch::Tensor,std::vector<torch::Tensor> >", "std::tuple<at::Tensor,std::vector<at::Tensor> >").pointerTypes("T_TensorTensorVector_T").define())
            .put(new Info("std::tuple<std::vector<torch::Tensor>,torch::Tensor>", "std::tuple<std::vector<at::Tensor>,at::Tensor>").pointerTypes("T_TensorVectorTensor_T").define())
            .put(new Info(
                "std::tuple<std::vector<torch::Tensor>,std::vector<torch::Tensor>,std::vector<torch::Tensor>,std::vector<torch::Tensor>,std::vector<torch::Tensor> >",
                "std::tuple<std::vector<at::Tensor>,std::vector<at::Tensor>,std::vector<at::Tensor>,std::vector<at::Tensor>,std::vector<at::Tensor> >")
                .pointerTypes("T_TensorVectorTensorVectorTensorVectorTensorVectorTensorVector_T").define())
            .put(new Info("std::tuple<torch::Tensor,std::vector<torch::Tensor>,std::vector<torch::Tensor> >", "std::tuple<at::Tensor,std::vector<at::Tensor>,std::vector<at::Tensor> >").pointerTypes("T_TensorTensorVectorTensorVector_T").define())
            .put(new Info("std::tuple<torch::Tensor,torch::Tensor,int64_t,int64_t,torch::Tensor>", "std::tuple<at::Tensor,at::Tensor,int64_t,int64_t,at::Tensor>").pointerTypes("T_TensorTensorLongLongTensor_T").define())
            .put(new Info("std::tuple<torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,int64_t,int64_t,int64_t,int64_t,torch::Tensor>", "std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,int64_t,int64_t,int64_t,int64_t,at::Tensor>").pointerTypes("T_TensorTensorTensorTensorsLongLongLongLongTensor_T").define())
            .put(new Info("const std::tuple<at::DataPtr,size_t>", "std::tuple<at::DataPtr,size_t>").pointerTypes("T_DataPtrSizeT_T").define())
            .put(new Info("std::tuple<c10::TypePtr,int32_t>", "std::pair<c10::TypePtr,int32_t>").pointerTypes("T_TypePtrLong_T").define()) // Parse this pair as tuple because Parser doesn't generate valid code for optional<pair>
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
        infoMap.put(new Info("torch::jit::TreeList::const_iterator").cast().pointerTypes("TreeRef"));


        /* Not parsed anymore
        List<String> binaryOps = Arrays.asList("Add", "Sub", "Div", "Max", "Min", "Mul", "Mod", "Xor", "And", "Or", "Rshift", "Lshift");
        List<String> exprOps = new ArrayList<>();
        exprOps.addAll(Arrays.asList("CharImm", "FloatImm", "BitCast", "Intrinsics", "Broadcast", "Cast"));
        exprOps.addAll(binaryOps);
        List<String> bitwiseOps = Arrays.asList("Xor", "And", "Or", "Rshift", "Lshift");

        for (String op : binaryOps)
            infoMap.put(new Info("torch::jit::tensorexpr::BinaryOpNode<torch::jit::tensorexpr::" + op + ">").pointerTypes("BinaryOpNode" + op));
        for (String op : exprOps)
            infoMap.put(new Info("torch::jit::tensorexpr::ExprNode<torch::jit::tensorexpr::" + op + ">").pointerTypes("ExprNode" + op));
        for (String op : bitwiseOps)
            infoMap.put(new Info("torch::jit::tensorexpr::BitwiseOpNode<torch::jit::tensorexpr::" + op + ">").pointerTypes("BitwiseOpNode" + op));
        */


        //// c10 Dict
        infoMap
            .put(new Info("c10::Dict<c10::IValue,c10::IValue>").purify().pointerTypes("GenericDict"))
            .put(new Info("c10::impl::DictEntryRef<c10::IValue,c10::IValue,c10::detail::DictImpl::dict_map_type::iterator>").pointerTypes("GenericDictEntryRef"))
            .put(new Info("c10::impl::DictIterator<c10::IValue,c10::IValue,c10::detail::DictImpl::dict_map_type::iterator>",
                "c10::Dict<c10::IValue,c10::IValue>::iterator").purify().pointerTypes("GenericDictIterator").friendly())
            .put(new Info("c10::Dict<std::string,c10::impl::GenericList>").pointerTypes("StringGenericListDict"))
            .put(new Info("c10::Dict<std::string,c10::impl::GenericList>(c10::TypePtr, c10::TypePtr)").skip())
            .put(new Info(
                "c10::impl::DictIterator::operator -(const c10::impl::DictIterator&, const c10::impl::DictIterator&)",
                "c10::impl::DictIterator<c10::IValue,c10::IValue,c10::detail::DictImpl::dict_map_type::iterator>::operator -").skip()) // Don't know how to map :difference_type

            /* Following operators throw a template error "no match", even in C++. */
            .put(new Info("c10::Dict::iterator::operator <(const c10::Dict::iterator&, const c10::Dict::iterator&)").skip())
            .put(new Info("c10::impl::DictIterator<c10::IValue,c10::IValue,c10::detail::DictImpl::dict_map_type::iterator>::operator <(const c10::impl::DictIterator<c10::IValue,c10::IValue,c10::detail::DictImpl::dict_map_type::iterator>&, const c10::impl::DictIterator<c10::IValue,c10::IValue,c10::detail::DictImpl::dict_map_type::iterator>&)").skip())
            .put(new Info("c10::Dict::iterator::operator <=(const c10::Dict::iterator&, const c10::Dict::iterator&)").skip())
            .put(new Info("c10::impl::DictIterator<c10::IValue,c10::IValue,c10::detail::DictImpl::dict_map_type::iterator>::operator <=(const c10::impl::DictIterator<c10::IValue,c10::IValue,c10::detail::DictImpl::dict_map_type::iterator>&, const c10::impl::DictIterator<c10::IValue,c10::IValue,c10::detail::DictImpl::dict_map_type::iterator>&)").skip())
            .put(new Info("c10::Dict::iterator::operator >=(const c10::Dict::iterator&, const c10::Dict::iterator&)").skip())
            .put(new Info("c10::impl::DictIterator<c10::IValue,c10::IValue,c10::detail::DictImpl::dict_map_type::iterator>::operator >=(const c10::impl::DictIterator<c10::IValue,c10::IValue,c10::detail::DictImpl::dict_map_type::iterator>&, const c10::impl::DictIterator<c10::IValue,c10::IValue,c10::detail::DictImpl::dict_map_type::iterator>&)").skip())
            .put(new Info("c10::Dict::iterator::operator >(const c10::Dict::iterator&, const c10::Dict::iterator&)").skip())
            .put(new Info("c10::impl::DictIterator<c10::IValue,c10::IValue,c10::detail::DictImpl::dict_map_type::iterator>::operator >(const c10::impl::DictIterator<c10::IValue,c10::IValue,c10::detail::DictImpl::dict_map_type::iterator>&, const c10::impl::DictIterator<c10::IValue,c10::IValue,c10::detail::DictImpl::dict_map_type::iterator>&)").skip())
        ;


        //// torch::OrderedDict
        for (String[] o: new String[][] {
            { "std::string", "torch::Tensor", "StringTensor" },
            { "std::string", "torch::nn::Module", "StringModule" },
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

        // What is the use for this ?
        //.put(new Info("torch::OrderedDict<std::string,torch::nn::AnyModule>")
        //        .valueTypes("@Cast({\"\", \"torch::OrderedDict<std::string,torch::nn::AnyModule>&&\"}) @StdMove StringAnyModuleDict"))

        //// std::pair
        infoMap
            // Parser doesn't generate iterators for vector of pairs, so function returning such iterators, like ParameterListImpl::begin()
            // must be mapped to returning item instead. Issue #673. Change when issue resolved.
            .put(new Info("std::pair<std::string,torch::Tensor>", "std::pair<std::string,torch::Tensor>").cast().pointerTypes("StringTensorPair").define())
            .put(new Info("std::pair<std::string,torch::nn::Module>").pointerTypes("StringModulePair").define())
            .put(new Info("std::pair<std::string,torch::nn::AnyModule>").pointerTypes("StringAnyModulePair").define())
            .put(new Info("std::pair<std::string,std::shared_ptr<torch::nn::Module> >").pointerTypes("StringSharedModulePair").define())
            .put(new Info("std::pair<at::RecordFunctionHandle,int>").pointerTypes("RecordFunctionHandleIntPair").define())
            .put(new Info("std::pair<size_t,torch::jit::MatchedSchema>").pointerTypes("SizeTMatchedSchemaPair").define())
        ;

        //// Intrusive pointers
        /* We cannot define an adapter working like SharedPtrAdapter since there is no public constructor of
          intrusive_ptr<T> taking a T*. */
        for (PointerInfo pi : new PointerInfo[]{
            new PointerInfo("c10::ivalue::Tuple"),
            new PointerInfo("c10::ivalue::Future", "at::ivalue::Future"),
            new PointerInfo("c10::ivalue::ConstantString"),
            new PointerInfo("c10::GeneratorImpl"),
            new PointerInfo("at::Quantizer"),
            new PointerInfo("c10::ivalue::Await"),
            new PointerInfo("c10::RRefInterface"),
            new PointerInfo("c10::ivalue::PyObjectHolder"),
            new PointerInfo("c10::ivalue::EnumHolder"),
            new PointerInfo("c10::TensorImpl"),
            new PointerInfo("c10::TensorImpl,c10::UndefinedTensorImpl").javaBaseName("TensorImpl"),
            new PointerInfo("torch::jit::Tree").javaName("TreeRef"),
            new PointerInfo("c10::StorageImpl", "c10::StorageImpl,NullType"),
            new PointerInfo("c10::SymNodeImpl").javaName("SymNode")
        }) {
            String[] cppNames = new String[pi.argumentNames.length + pi.otherCppNames.length];
            int i = 0;
            for (String n : pi.argumentNames) {
                String ipn = template("c10::intrusive_ptr", n);
                cppNames[i++] = ipn;
                // Skipping constructor taking a unique_ptr
                infoMap.put(new Info(ipn + "(" + n + "*)").skip());
                /* If we need to map a unique_ptr with this type, we need to disambiguate constructor
                with something like:
                infoMap.put(new Info(ipn + "(" + upn + ")").javaText(
                        "public " + pi.javaName + "(" + xxx + " rhs) { super((Pointer)null); allocate(rhs); }\n" +
                        "@NoException(true) private native void allocate(@Cast({\"\", \"" + upn + "\"}) @UniquePtr " + xxx + " rhs);"));
                 */
            }
            for (String n : pi.otherCppNames)
                cppNames[i++] = n;
            infoMap.put(new Info(cppNames).pointerTypes(pi.javaName == null ? (pi.javaBaseName + "Ptr") : pi.javaName));

        }


        //// Classes that Parser cannot detect as virtual
        infoMap.put(new Info("c10::Error", "c10::IndexError", "c10::LinAlgError", "c10::ValueError", "c10::TypeError", "c10::NotImplementedError", "c10::EnforceFiniteError", "c10::OutOfMemoryError",
            "c10::OnnxfiBackendSystemError", "c10::DistBackendError", "c10::SharedType", "c10::StrongTypePtr",
            "c10::WeakTypePtr", "torch::autograd::CppFunctionPreHook", "torch::autograd::DifferentiableViewMeta",
            "torch::autograd::TraceableFunction", "torch::jit::Instruction", "torch::jit::Method", "torch::jit::ModuleInstanceInfo",
            "torch::jit::Object::Property", "torch::jit::OperatorSet", "torch::jit::SourceRangePickler", "torch::jit::Unpickler",
            "torch::jit::Operator", "c10::CuDNNError").purify());


        /// Classes skipped for various non-investigated reasons
        infoMap
            .put(new Info(/*"c10::intrusive_ptr", "c10::weak_intrusive_ptr", */"c10::guts::is_fundamental",
                "c10::detail::CaptureKernelCall", "c10::detail::DictImpl", "c10::detail::MultiDispatchKeySet", "c10::ExclusivelyOwnedTraits", "c10::FunctionSchema::dump",
                "c10::domain_prefix", "c10::C10FlagsRegistry", "c10::enforce_detail::EnforceFailMessage", "c10::impl::build_feature_required_feature_not_available",
                "c10::detail::getMaybeFakeTypePtr_", "c10::complex_literals::operator \"\"_if", "c10::complex_literals::operator \"\"_id",
                "decltype(::c10::impl::ScalarTypeToCPPType<::c10::ScalarType::ComplexHalf>::t)", "c10::BoxedKernel", "c10::ExtraMeta", "c10::remove_symint",
                "c10::InefficientStdFunctionContext", "c10::DataPtr::move_context", "c10::detail::UniqueVoidPtr::move_context", "QuantizerPtr", "c10::IValue::toModule", "c10::toBackendComponent",
                "c10::optional<THPObjectPtr>", "c10::asIntArrayRefSlow", "c10::standardizeVectorForUnion",
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
            .put(new Info("c10::SingleElementType<c10::TypeKind::OptionalType,c10::OptionalType>").pointerTypes("OptionalSingleElementType"))
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
                   .put(new Info("torch::jit::detail::NamedPolicy<torch::jit::detail::" + t[0] + "Policy>").pointerTypes("Named" + t[1] + "Policy"))
                   .put(new Info(
                       "torch::jit::slot_list_impl<torch::jit::detail::NamedPolicy<torch::jit::detail::" + t[0] + "Policy> >",
                       "torch::jit::named_" + t[0].toLowerCase() + "_list").pointerTypes("named_" + t[0].toLowerCase() + "_list"))
                   .put(new Info("torch::jit::slot_iterator_impl<torch::jit::detail::NamedPolicy<torch::jit::detail::" + t[0] + "Policy> >").pointerTypes("named_" + t[0].toLowerCase() + "_iterator"))
                   .put(new Info("torch::jit::slot_iterator_impl<torch::jit::detail::NamedPolicy<torch::jit::detail::" + t[0] + "Policy> >::value_type").pointerTypes("Named" + t[1]))
            ;
        }

        infoMap
            .put(new Info("torch::jit::tracer::warn_fn_type", "warn_fn_type").cast().pointerTypes("warn_fn_type"))
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
                "torch::jit::Maybe<torch::jit::List<torch::jit::Property> >::map").skip())
            .put(new Info("torch::jit::Wrap<torch::jit::Block>").pointerTypes("BlockWrap"))
            .put(new Info("torch::jit::Wrap<torch::jit::Node>").pointerTypes("JitNodeWrap"))
            .put(new Info("torch::jit::Wrap<torch::jit::Value>").pointerTypes("ValueWrap"));


        //// Datasets
        String VirtualChunkDataReader = "JavaCPP_torch_0003a_0003adata_0003a_0003adatasets_0003a_0003aChunkDataReader_0003ctorch_0003a_0003adata_0003a_0003aExample_0003c_0003e_0002cstd_0003a_0003avector_0003ctorch_0003a_0003adata_0003a_0003aExample_0003c_0003e_00020_0003e_00020_0003e";

        infoMap.put(new Info("std::vector<torch::data::Example<> >", // "UnwrappedBatchType",
                   "std::vector<torch::data::datasets::Dataset<torch::data::datasets::MNIST,torch::data::Example<> >::ExampleType>").pointerTypes("ExampleVector").define())
               .put(new Info("std::vector<torch::data::Example<torch::Tensor,torch::data::example::NoTarget> >").pointerTypes("TensorExampleVector").define())
               .put(new Info("c10::optional<std::vector<torch::data::Example<> > >", "c10::optional<" + VirtualChunkDataReader + "::BatchType>",
                   "torch::data::datasets::ChunkDataset<" + VirtualChunkDataReader + ",torch::data::samplers::RandomSampler,torch::data::samplers::RandomSampler>::BatchType")
                   .pointerTypes("ExampleVectorOptional").define())

               .put(new Info("torch::data::Example<torch::Tensor,torch::Tensor>", "torch::data::Example<>").pointerTypes("Example"))
               .put(new Info("c10::optional<torch::data::Example<torch::Tensor,torch::Tensor> >", "c10::optional<torch::data::Example<> >").pointerTypes("ExampleOptional").define())
               .put(new Info("torch::data::Example<torch::Tensor,torch::data::example::NoTarget>").pointerTypes("TensorExample"))
               .put(new Info("torch::data::Example<torch::Tensor,torch::data::example::NoTarget>::Example").javaText(
                   "public TensorExample(@ByVal Tensor data) { super((Pointer)null); allocate(data); }\n"
                   + "private native void allocate(@ByVal Tensor data);\n"))
               .put(new Info("torch::data::Example<torch::Tensor,torch::data::example::NoTarget>::target").skip())
//               .put(new Info("torch::data::detail::SentinelIterator<std::vector<torch::data::Example<> > >").pointerTypes("ExampleSentinelIterator"))
//               .put(new Info("torch::data::detail::ValidIterator<std::vector<torch::data::Example<> > >").pointerTypes("ExampleValidIterator"))
//               .put(new Info("torch::data::detail::IteratorImpl<std::vector<torch::data::Example<> > >").pointerTypes("ExampleIteratorImpl"))
               .put(new Info("torch::data::Iterator<torch::data::Example<> >").purify().pointerTypes("ExampleIterator"))
               //.put(new Info("torch::data::Iterator<std::vector<torch::data::Example<> > >").purify().pointerTypes("ExampleVectorIterator"))
               .put(new Info("torch::data::Iterator<c10::optional<std::vector<torch::data::Example<> > > >").purify().pointerTypes("ExampleVectorOptionalIterator"))
               .put(new Info("torch::data::samplers::Sampler<std::vector<size_t> >", "torch::data::samplers::Sampler<>").pointerTypes("Sampler"))
               .put(new Info("torch::data::samplers::Sampler<torch::data::samplers::BatchSize>").pointerTypes("BatchSizeSampler"))
               .put(new Info("torch::data::samplers::RandomSampler").pointerTypes("RandomSampler"))
               .put(new Info("torch::data::samplers::DistributedSampler<std::vector<size_t> >", "torch::data::samplers::DistributedSampler<>").purify().pointerTypes("DistributedSampler"))
               .put(new Info("c10::optional<torch::data::samplers::BatchSize>").pointerTypes("BatchSizeOptional").define())
               .put(new Info("torch::data::transforms::BatchTransform<std::vector<torch::data::Example<> >, torch::data::Example<> >",
                   "torch::data::transforms::Collation<torch::data::Example<> >").pointerTypes("ExampleCollation"))
               .put(new Info("torch::data::transforms::Stack<torch::data::Example<> >").pointerTypes("ExampleStack"))
               .put(new Info("c10::optional<std::vector<c10::ivalue::Future::WeakStorage> >").pointerTypes("WeakStorageVectorOptional").define())
               .put(new Info("const std::vector<c10::ivalue::Future::WeakStorage>", "std::vector<c10::ivalue::Future::WeakStorage>").pointerTypes("WeakStorageVector").define())
               .put(new Info("std::vector<torch::autograd::GraphTask::ExecInfo::Capture>").pointerTypes("CaptureVector"))


               .put(new Info("torch::data::datasets::ChunkDataReader<torch::data::Example<>,std::vector<torch::data::Example<> > >", VirtualChunkDataReader).pointerTypes("ChunkDataReader").virtualize())
               .put(new Info("torch::data::datasets::ChunkDataset<" + VirtualChunkDataReader + ",torch::data::samplers::RandomSampler,torch::data::samplers::RandomSampler>").pointerTypes("ChunkDataset"))
               .put(new Info("torch::data::datasets::ChunkDataset<" + VirtualChunkDataReader + ",torch::data::samplers::RandomSampler,torch::data::samplers::RandomSampler>::ChunkDataset").javaText(
                   "public ChunkDataset(\n"
                   + "      ChunkDataReader chunk_reader,\n"
                   + "      RandomSampler chunk_sampler,\n"
                   + "      RandomSampler example_sampler,\n"
                   + "      ChunkDatasetOptions options) { super((Pointer)null); allocate(chunk_reader, chunk_sampler, example_sampler, options, null); }\n"
                   + "public ChunkDataset(\n"
                   + "      ChunkDataReader chunk_reader,\n"
                   + "      RandomSampler chunk_sampler,\n"
                   + "      RandomSampler example_sampler,\n"
                   + "      ChunkDatasetOptions options,\n"
                   + "      Pointer preprocessing_policy) { super((Pointer)null); allocate(chunk_reader, chunk_sampler, example_sampler, options, preprocessing_policy); }\n"
                   + "private native void allocate(\n"
                   + "      @ByVal @Cast(\"" + VirtualChunkDataReader + "*\") ChunkDataReader chunk_reader,\n"
                   + "      @ByVal RandomSampler chunk_sampler,\n"
                   + "      @ByVal RandomSampler example_sampler,\n"
                   + "      @ByVal ChunkDatasetOptions options,\n"
                   + "      @ByVal(nullValue = \"std::function<void(std::vector<torch::data::Example<>>&)>()\") @Cast(\"std::function<void(std::vector<torch::data::Example<>>&)>*\") Pointer preprocessing_policy);\n"))
               .put(new Info("torch::data::datasets::StatefulDataset<torch::data::datasets::ChunkDataset<" + VirtualChunkDataReader + ",torch::data::samplers::RandomSampler,torch::data::samplers::RandomSampler>," + VirtualChunkDataReader + "::BatchType,size_t>")
                   .pointerTypes("ChunkStatefulDataset"))
               .put(new Info("torch::data::datasets::BatchDataset<torch::data::datasets::ChunkDataset<" + VirtualChunkDataReader + ",torch::data::samplers::RandomSampler,torch::data::samplers::RandomSampler>,c10::optional<" + VirtualChunkDataReader + "::BatchType>,size_t>",
                   "torch::data::datasets::BatchDataset<torch::data::datasets::ChunkDataset<" + VirtualChunkDataReader + ",torch::data::samplers::RandomSampler,torch::data::samplers::RandomSampler>,std::vector<torch::data::Example<> > >")
                   .pointerTypes("ChunkBatchDataset"))
               .put(new Info("torch::data::datasets::BatchDataset<torch::data::datasets::SharedBatchDataset<torch::data::datasets::ChunkDataset<" + VirtualChunkDataReader + ",torch::data::samplers::RandomSampler,torch::data::samplers::RandomSampler> >,c10::optional<" + VirtualChunkDataReader + "::BatchType>,size_t>",
                   "torch::data::datasets::BatchDataset<torch::data::datasets::SharedBatchDataset<torch::data::datasets::ChunkDataset<" + VirtualChunkDataReader + ",torch::data::samplers::RandomSampler,torch::data::samplers::RandomSampler> >,torch::data::datasets::ChunkDataset<" + VirtualChunkDataReader + ",torch::data::samplers::RandomSampler,torch::data::samplers::RandomSampler>::BatchType,torch::data::datasets::ChunkDataset<" + VirtualChunkDataReader + ",torch::data::samplers::RandomSampler,torch::data::samplers::RandomSampler>::BatchRequestType>")
                   .pointerTypes("ChunkBatchSharedBatchDataset"))
               .put(new Info("torch::data::datasets::BatchDataset<torch::data::datasets::SharedBatchDataset<torch::data::datasets::ChunkDataset<" + VirtualChunkDataReader + ",torch::data::samplers::RandomSampler,torch::data::samplers::RandomSampler> >,c10::optional<" + VirtualChunkDataReader + "::BatchType>,size_t>::map")
                   .javaText("public native @ByVal ChunkMapDataset map(@ByVal ExampleStack transform);"))
               .put(new Info("torch::data::datasets::SharedBatchDataset<torch::data::datasets::ChunkDataset<" + VirtualChunkDataReader + ",torch::data::samplers::RandomSampler,torch::data::samplers::RandomSampler> >")
                   .pointerTypes("ChunkSharedBatchDataset"))
               .put(new Info("torch::data::datasets::MapDataset<torch::data::datasets::SharedBatchDataset<torch::data::datasets::ChunkDataset<" + VirtualChunkDataReader + ",torch::data::samplers::RandomSampler,torch::data::samplers::RandomSampler> >,torch::data::transforms::Stack<torch::data::Example<> > >")
                   .pointerTypes("ChunkMapDataset"))
               .put(new Info("torch::data::datasets::MapDataset<torch::data::datasets::SharedBatchDataset<torch::data::datasets::ChunkDataset<" + VirtualChunkDataReader + ",torch::data::samplers::RandomSampler,torch::data::samplers::RandomSampler> >,torch::data::transforms::Stack<torch::data::Example<> > >::reset")
                   .skip())
               .put(new Info("torch::data::datasets::MapDataset<torch::data::datasets::SharedBatchDataset<torch::data::datasets::ChunkDataset<" + VirtualChunkDataReader + ",torch::data::samplers::RandomSampler,torch::data::samplers::RandomSampler> >,torch::data::transforms::Stack<torch::data::Example<> > >::DatasetType")
                   .pointerTypes("ChunkSharedBatchDataset"))
               .put(new Info("torch::data::datasets::BatchDataset<torch::data::datasets::MapDataset<torch::data::datasets::SharedBatchDataset<torch::data::datasets::ChunkDataset<" + VirtualChunkDataReader + ",torch::data::samplers::RandomSampler,torch::data::samplers::RandomSampler> >,torch::data::transforms::Stack<torch::data::Example<> > >,std::vector<torch::data::Example<> >,at::ArrayRef<size_t> >",
                   "torch::data::datasets::BatchDataset<torch::data::datasets::MapDataset<torch::data::datasets::SharedBatchDataset<torch::data::datasets::ChunkDataset<" + VirtualChunkDataReader + ",torch::data::samplers::RandomSampler,torch::data::samplers::RandomSampler> >,torch::data::transforms::Stack<torch::data::Example<> > >,torch::data::datasets::detail::optional_if_t<torch::data::datasets::SharedBatchDataset<torch::data::datasets::ChunkDataset<" + VirtualChunkDataReader + ",torch::data::samplers::RandomSampler,torch::data::samplers::RandomSampler> >::is_stateful,torch::data::transforms::Stack<torch::data::Example<> >::OutputBatchType>,torch::data::datasets::SharedBatchDataset<torch::data::datasets::ChunkDataset<" + VirtualChunkDataReader + ",torch::data::samplers::RandomSampler,torch::data::samplers::RandomSampler> >::BatchRequestType>")
                   .pointerTypes("ChunkMapBatchDataset"))
               .put(new Info("torch::data::datasets::MapDataset<torch::data::datasets::SharedBatchDataset<torch::data::datasets::ChunkDataset<" + VirtualChunkDataReader + ",torch::data::samplers::RandomSampler,torch::data::samplers::RandomSampler> >,torch::data::transforms::Stack<torch::data::Example<> > >::BatchRequestType").pointerTypes("SizeTArrayRef"))
               .put(new Info("torch::data::datasets::MapDataset<torch::data::datasets::SharedBatchDataset<torch::data::datasets::ChunkDataset<" + VirtualChunkDataReader + ",torch::data::samplers::RandomSampler,torch::data::samplers::RandomSampler> >,torch::data::transforms::Stack<torch::data::Example<> > >::OutputBatchType").pointerTypes("Example"))
               .put(new Info("torch::data::datasets::MapDataset<torch::data::datasets::SharedBatchDataset<torch::data::datasets::ChunkDataset<" + VirtualChunkDataReader + ",torch::data::samplers::RandomSampler,torch::data::samplers::RandomSampler> >,torch::data::transforms::Stack<torch::data::Example<> > >::get_batch")
                   .javaText("public native @Name(\"get_batch\") @ByVal ExampleOptional get_batch_example(@Cast(\"size_t\") long indices);"))
               .put(new Info("torch::data::DataLoaderBase<torch::data::datasets::MapDataset<torch::data::datasets::SharedBatchDataset<torch::data::datasets::ChunkDataset<" + VirtualChunkDataReader + ",torch::data::samplers::RandomSampler,torch::data::samplers::RandomSampler> >,torch::data::transforms::Stack<torch::data::Example<> > >,torch::data::Example<>,size_t>",
                   "torch::data::DataLoaderBase<torch::data::datasets::MapDataset<torch::data::datasets::SharedBatchDataset<torch::data::datasets::ChunkDataset<" + VirtualChunkDataReader + ",torch::data::samplers::RandomSampler,torch::data::samplers::RandomSampler> >,torch::data::transforms::Stack<torch::data::Example<> > >,torch::data::datasets::MapDataset<torch::data::datasets::SharedBatchDataset<torch::data::datasets::ChunkDataset<" + VirtualChunkDataReader + ",torch::data::samplers::RandomSampler,torch::data::samplers::RandomSampler> >,torch::data::transforms::Stack<torch::data::Example<> > >::BatchType::value_type,torch::data::datasets::MapDataset<torch::data::datasets::SharedBatchDataset<torch::data::datasets::ChunkDataset<" + VirtualChunkDataReader + ",torch::data::samplers::RandomSampler,torch::data::samplers::RandomSampler> >,torch::data::transforms::Stack<torch::data::Example<> > >::BatchRequestType>")
                   .purify().pointerTypes("ChunkRandomDataLoaderBase"))
               .put(new Info("torch::data::StatefulDataLoader<torch::data::datasets::MapDataset<torch::data::datasets::SharedBatchDataset<torch::data::datasets::ChunkDataset<" + VirtualChunkDataReader + ",torch::data::samplers::RandomSampler,torch::data::samplers::RandomSampler> >,torch::data::transforms::Stack<torch::data::Example<> > > >")
                   .pointerTypes("ChunkRandomDataLoader"))

               .put(new Info("torch::data::DataLoaderBase<torch::data::datasets::MapDataset<torch::data::datasets::MNIST,torch::data::transforms::Stack<torch::data::Example<> > >,torch::data::Example<>,std::vector<size_t> >",
                   "torch::data::DataLoaderBase<torch::data::datasets::MapDataset<torch::data::datasets::MNIST,torch::data::transforms::Stack<torch::data::Example<> > >,torch::data::datasets::MapDataset<torch::data::datasets::MNIST,torch::data::transforms::Stack<torch::data::Example<> > >::BatchType,torch::data::samplers::RandomSampler::BatchRequestType>")
                   .purify().pointerTypes("MNISTRandomDataLoaderBase"))
               .put(new Info("torch::data::StatelessDataLoader<torch::data::datasets::MapDataset<torch::data::datasets::MNIST,torch::data::transforms::Stack<torch::data::Example<> > >,torch::data::samplers::RandomSampler>").pointerTypes("MNISTRandomDataLoader"))
               .put(new Info("torch::data::datasets::Dataset<torch::data::datasets::MNIST,torch::data::Example<> >",
                   "torch::data::datasets::Dataset<MNIST>").pointerTypes("MNISTDataset"))
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

               .put(new Info("torch::data::datasets::Dataset<torch::data::datasets::TensorDataset,torch::data::TensorExample>",
                   "torch::data::datasets::Dataset<TensorDataset,torch::data::TensorExample>").pointerTypes("TensorExampleDataset"))
               .put(new Info("torch::data::datasets::BatchDataset<torch::data::datasets::TensorDataset,std::vector<torch::data::TensorExample> >",
                   "torch::data::datasets::BatchDataset<TensorDataset,std::vector<torch::data::TensorExample> >").pointerTypes("TensorExampleBatchDataset"))
               .put(new Info("torch::data::datasets::Dataset<torch::data::datasets::TensorDataset,torch::data::TensorExample>::get_batch",
                   "torch::data::datasets::BatchDataset<torch::data::datasets::TensorDataset,std::vector<torch::data::TensorExample> >::get_batch")
                   .javaText("public native @ByVal TensorExampleVector get_batch(@ByVal SizeTArrayRef request);"))
        ;


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
            .put(new Info("torch::nn::Module::register_module<torch::nn::Module>").javaNames("register_module"))
            .put(new Info("torch::nn::Module").upcast())
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
        ;

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
                "public native @ByVal AnyValue any_forward(@Const @ByRef Tensor input, @ByRef(nullValue = \"c10::optional<at::IntArrayRef>(c10::nullopt)\") @Cast({\"int64_t*\", \"c10::ArrayRef<int64_t>\", \"std::vector<int64_t>&\"}) @StdVector long... output_size);\n" +
                "public native @ByVal AnyValue any_forward(@Const @ByRef Tensor input, @Const @ByRef(nullValue = \"c10::optional<at::IntArrayRef>(c10::nullopt)\") LongArrayRefOptional output_size);\n" +
                "public native @ByVal AnyValue any_forward(@Const @ByRef Tensor input, @Const @ByRef Tensor indices, @Const @ByRef(nullValue = \"c10::optional<std::vector<int64_t> >(c10::nullopt)\") LongVectorOptional output_size);\n" +
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
                "public native @ByVal Tensor forward(@Const @ByRef Tensor input, @ByRef(nullValue = \"c10::optional<at::IntArrayRef>(c10::nullopt)\") @Cast({\"int64_t*\", \"c10::ArrayRef<int64_t>\", \"std::vector<int64_t>&\"}) @StdVector long... output_size);\n" +
                "public native @ByVal Tensor forward(@Const @ByRef Tensor input, @Const @ByRef(nullValue = \"c10::optional<at::IntArrayRef>(c10::nullopt)\") LongArrayRefOptional output_size);\n" +
                "public native @ByVal Tensor forward(@Const @ByRef Tensor input, @Const @ByRef Tensor indices, @Const @ByRef(nullValue = \"c10::optional<std::vector<int64_t> >(c10::nullopt)\") LongVectorOptional output_size);\n" +
                "public native @ByVal @Name(\"forward<std::tuple<torch::Tensor,std::tuple<torch::Tensor,torch::Tensor>>>\") T_TensorT_TensorTensor_T_T forwardT_TensorT_TensorTensor_T_T(@Const @ByRef Tensor input);\n" +
                "public native @ByVal @Name(\"forward<std::tuple<torch::Tensor,std::tuple<torch::Tensor,torch::Tensor>>>\") T_TensorT_TensorTensor_T_T forwardT_TensorT_TensorTensor_T_T(@Const @ByRef Tensor input, @ByVal(nullValue = \"torch::optional<std::tuple<torch::Tensor,torch::Tensor> >{}\") T_TensorTensor_TOptional hx_opt);\n" +
                "public native @ByVal @Name(\"forward<std::tuple<torch::Tensor,torch::Tensor>>\") T_TensorTensor_T forwardT_TensorTensor_T(@Const @ByRef Tensor input);\n" +
                "public native @ByVal @Name(\"forward<std::tuple<torch::Tensor,torch::Tensor>>\") T_TensorTensor_T forwardT_TensorTensor_T(@Const @ByRef Tensor input1, @Const @ByRef Tensor input2);\n" +
                "public native @ByVal @Name(\"forward<std::tuple<torch::Tensor,torch::Tensor>>\") T_TensorTensor_T forwardT_TensorTensor_T(@Const @ByRef Tensor input1, @Const @ByRef Tensor input2, @Const @ByRef Tensor input3);\n" +
                "public native @ByVal @Name(\"forward<std::tuple<torch::Tensor,torch::Tensor>>\") T_TensorTensor_T forwardT_TensorTensor_T(@Const @ByRef Tensor input, @ByVal(nullValue = \"torch::optional<std::tuple<torch::Tensor,torch::Tensor> >{}\") T_TensorTensor_TOptional hx_opt);\n" +
                "public native @ByVal @Name(\"forward<std::tuple<torch::Tensor,torch::Tensor>>\") T_TensorTensor_T forwardT_TensorTensor_T(@Const @ByRef Tensor query, @Const @ByRef Tensor key, @Const @ByRef Tensor value, @Const @ByRef(nullValue = \"torch::Tensor{}\") Tensor key_padding_mask, @Cast(\"bool\") boolean need_weights/*=true*/, @Const @ByRef(nullValue = \"torch::Tensor{}\") Tensor attn_mask, @Cast(\"bool\") boolean average_attn_weights/*=true*/);\n" +
                "public native @ByVal @Name(\"forward<torch::nn::ASMoutput>\") ASMoutput forwardASMoutput(@Const @ByRef Tensor input, @Const @ByRef Tensor target);\n"
            ))
            .put(new Info("torch::nn::AnyModule(ModuleType*)")
                // We cannot use template instantiation mechanism in Parser with something like
                // new Info("torch::nn::AnyModule<torch::nn::" + name + "Impl>(ModuleType*)")
                // because it doesn't work with javaText. And we need javaText because of @Cast.
                .javaText(anyModuleConstructors));

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
            new PointerInfo("torch::jit::tensorexpr::analysis::AccessInfo"),
            new PointerInfo("c10::ClassType"),
            new PointerInfo("c10::TensorType").otherCppNames("c10::TensorTypePtr", "at::TensorTypePtr", "torch::TensorTypePtr"),
            new PointerInfo("torch::autograd::FunctionPreHook"),
            new PointerInfo("torch::nn::Module"),
            new PointerInfo("const at::functorch::FuncTorchTLSBase"),
            new PointerInfo("const torch::jit::CompilationUnit"),
            new PointerInfo("torch::jit::SugaredValue")
        }) {
            // See issue #670
            String[] cppNames = new String[pi.argumentNames.length + pi.otherCppNames.length];
            int i = 0;
            for (String n : pi.argumentNames) cppNames[i++] = template("std::shared_ptr", n);
            for (String n : pi.otherCppNames) cppNames[i++] = n;
            // Specifying the parameter of the annotation allows to disambiguate cases where a class can store either a
            // std::shared_ptr<const X> or std::shared_ptr<X> (like CompilationUnit)
            // .valueTypes("@Cast(\"const torch::jit::CompilationUnit*\") CompilationUnit") seems to work too but for obscure reason
            infoMap.put(new Info(cppNames).annotations("@SharedPtr(\"" + pi.argumentNames[0] + "\")").pointerTypes(pi.javaBaseName));

            // Also annotate constructor of target class to ensure only one shared_ptr exists for each instance
            String n = pi.argumentNames[0].substring(pi.argumentNames[0].lastIndexOf(' ') + 1); // Remove possible const
            String n2 = n.equals("torch::nn::Module") ? "JavaCPP_torch_0003a_0003ann_0003a_0003aModule" : n;
            infoMap.put(new Info(n + n.substring(n.lastIndexOf("::"))).annotations("@SharedPtr", "@Name(\"std::make_shared<" + n2 + ">\")"));
        }


        //// @UniquePtr
        infoMap
            .put(new Info("std::unique_ptr<torch::autograd::FunctionPreHook>").annotations("@UniquePtr")
                                                                              .valueTypes("@Cast({\"\", \"std::unique_ptr<torch::autograd::FunctionPreHook>&&\"}) FunctionPreHook")
                                                                              .pointerTypes("FunctionPreHook"))
            .put(new Info("std::unique_ptr<torch::autograd::FunctionPostHook>").annotations("@UniquePtr")
                                                                               .valueTypes("@Cast({\"\", \"std::unique_ptr<torch::autograd::FunctionPostHook>&&\"}) FunctionPostHook")
                                                                               .pointerTypes("FunctionPostHook"))
            .put(new Info("std::unique_ptr<torch::jit::AttributeValue>", "Ptr").annotations("@UniquePtr").pointerTypes("AttributeValue"))

        ;

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

            "torch::autograd::get_current_graph_task_exec_info" // Would need to map GraphTask, NodeExec...too much burden

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
            "torch::jit::Call::caller_range"
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
            "at::Tensor(c10::TensorImpl*)", // Really at::Tensor(c10::intrusive_ptr<at::TensorImpl,c10::UndefinedTensorImpl> but the Parser gets the wrong fullname
            "at::Tensor::_set_fw_grad", "at::Tensor::_fw_grad",
            "at::TensorBase(c10::intrusive_ptr<at::TensorImpl,c10::UndefinedTensorImpl>",
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
            "torch::jit::GraphFunction::_set_initial_executor_execution_mode", "torch::jit::GraphFunction::_set_ignore_amp"
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
            "at::Tensor::is_variable()"
        ).skip());

        //// Function returning object by value, and copy constructor was deleted. Any way to get around this ?
        infoMap.put(new Info(
            "c10::RegisterOperators::Options", //All methods of Options return Options&&
            "c10::impl::device_guard_impl_registry",
            "torch::autograd::graph_task_id",
            "c10::getLessThanComparator", "c10::getGreaterThanComparator"
        ).skip());


        //// Deleted operator=. Any way to skip setter only ?
        infoMap.put(new Info("at::native::RNNDescriptor::dropout_desc_").skip());


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
            "c10::ArrayRef<torch::autograd::SavedVariable>::equals",
            "c10::ArrayRef<torch::autograd::SavedVariable>::vec",
            "c10::ArrayRef<at::Scalar>::equals",
            "c10::ArrayRef<torch::TensorArg>::equals",
            "c10::ArrayRef<torch::Tensor>::equals",
            "c10::ArrayRef<at::indexing::TensorIndex>::equals",
            "c10::ArrayRef<c10::optional<at::Tensor> >::equals"
        ).skip());


        //// Avoiding name clashes or making them more explicit.
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
               .put(new Info("torch::jit::Const").pointerTypes("ConstExpr"))
               .put(new Info("torch::jit::Node").pointerTypes("JitNode"))
               .put(new Info("torch::jit::Module").pointerTypes("JitModule"))
               .put(new Info("torch::jit::Object").pointerTypes("JitObject"))
               .put(new Info("torch::jit::String").pointerTypes("JitString"))
        ;


        //// Instantiation of templated functions.
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
               .put(new Info("at::TensorBase::data_ptr<int8_t>").javaNames("data_ptr_char"))
               .put(new Info("at::TensorBase::data_ptr<int16_t>").javaNames("data_ptr_short"))
               .put(new Info("at::TensorBase::data_ptr<int>").javaNames("data_ptr_int"))
               .put(new Info("at::TensorBase::data_ptr<int64_t>").javaNames("data_ptr_long"))
               .put(new Info("at::TensorBase::data_ptr<float>").javaNames("data_ptr_float"))
               .put(new Info("at::TensorBase::data_ptr<double>").javaNames("data_ptr_double"))
               .put(new Info("at::Tensor::item<int8_t>").javaNames("item_char"))
               .put(new Info("at::Tensor::item<int16_t>").javaNames("item_short"))
               .put(new Info("at::Tensor::item<int>").javaNames("item_int"))
               .put(new Info("at::Tensor::item<int64_t>").javaNames("item_long"))
               .put(new Info("at::Tensor::item<float>").javaNames("item_float"))
               .put(new Info("at::Tensor::item<double>").javaNames("item_double"))
               .put(new Info("at::make_generator").javaText(
                   "@Namespace(\"at\") public static native @ByVal @Name(\"make_generator<at::CPUGeneratorImpl>\") Generator make_generator_cpu();\n" +
                   "@Namespace(\"at\") public static native @ByVal @Name(\"make_generator<at::CPUGeneratorImpl,uint64_t>\") Generator make_generator_cpu(@Cast(\"uint64_t&&\") long seed_in);"
               ))
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
            {"at::BFloat16", "BFload16"}
        }) {
            infoMap.put(new Info("c10::fetch_and_cast<" + t[0] + ">").javaNames("fetch_and_cast_to_" + t[1]))
                   .put(new Info("c10::cast_and_store<" + t[0] + ">").javaNames("cast_and_store_from_" + t[1]));
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


        /* Classes that are not part of API (no TORCH_API nor C10_API) and are not argument nor return type of API methods.
         * Consider manual exclusion of all at::meta, at::native and caffe2 namespaces (but TypeMeta, that should
         * be moved to c10 one day). */
        infoMap.put(new Info(
            "ModuleHolderIndicator",
            "at::ObserverContext",
            "at::Range",
            "at::StepCallbacks::StartEndPair",
            "at::TensorBase::unsafe_borrow_t",
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
            "c10::MultiStreamGuard",
            "c10::OpTableOffsetAndMask",
            "c10::OperatorNameView",
            "c10::OptionalStreamGuard",
            "c10::PyHandleCache",
            "c10::RegisterOperators::Options::KernelRegistrationConfig",
            "c10::Registry<std::string,std::shared_ptr<c10::TaskThreadPoolBase>,int>",
            "c10::Registry<std::string,std::unique_ptr<at::CUDAHooksInterface>,at::CUDAHooksArgs>",
            "c10::Registry<std::string,std::unique_ptr<at::HIPHooksInterface>,at::HIPHooksArgs>",
            "c10::Registry<std::string,std::unique_ptr<at::MPSHooksInterface>,at::MPSHooksArgs>",
            "c10::Registry<std::string,std::unique_ptr<at::ORTHooksInterface>,at::ORTHooksArgs>",
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
            "c10::impl::ListElementConstReferenceTraits<c10::optional<std::string> >",
            "c10::impl::SizesAndStrides::",
            "c10::impl::VirtualGuardImpl",
            "c10::impl::decay_if_not_tensor<at::Tensor&>",
            "c10::impl::is_mutable_tensor_ref<at::Tensor&>",
            "c10::in_place_t",
            "c10::ivalue::ComplexHolder",
            "c10::ivalue::Object",
            "c10::ivalue::StreamData3Holder",
            "c10::ivalue::TupleElements::",
            "c10::ivalue::TupleTypeFactory<c10::TupleType>",
            "c10::once_flag",
            "c10::sha1",
            "c10::static_cast_with_inter_type<c10::complex<c10::Half>,c10::BFloat16>",
            "c10::trivial_init_t",
            "caffe2::detail::_Uninitialized",
            "ska::detailv3::sherwood_v3_entry::",
            "ska::detailv3::sherwood_v3_table::convertible_to_iterator",
            "ska::fibonacci_hash_policy",
            "ska::power_of_two_hash_policy",
            "ska::prime_number_hash_policy",
            "ska_ordered::detailv3::sherwood_v3_entry::",
            "ska_ordered::detailv3::sherwood_v3_table::convertible_to_iterator",
            "ska_ordered::order_preserving_flat_hash_map::convertible_to_value",
            "std::hash<c10::Device>",
            "std::hash<c10::DeviceType>",
            "std::hash<c10::Stream>",
            "std::hash<c10::Symbol>",
            "torch::Indices",
            "torch::MakeIndices<0>",
            "torch::NoInferSchemaTag",
            "torch::all_of",
            "torch::any_of<>",
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
            "torch::autograd::TraceableFunction",
            "torch::data::DataLoaderBase::Job",
            "torch::data::DataLoaderBase::QuitWorker",
            "torch::data::DataLoaderBase::Result",
            "torch::data::DataLoaderBase::Sequenced",
            "torch::data::FullDataLoaderOptions",
            "torch::data::Iterator<c10::optional<std::vector<torch::data::Example<> > > >",
            "torch::data::Iterator<std::vector<torch::data::Example<> > >",
            "torch::data::WorkerException",
            "torch::data::datasets::TensorDataset",
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
            "torch::jit::Operator::C10Operator",
            "torch::jit::Operator::JitOnlyOperator",
            "torch::jit::Operator::UnparsedFunctionSchema",
            "torch::jit::OwnedSourceRange",
            "torch::jit::RecursiveMethodCallError",
            "torch::jit::StrongFunctionPtr",
            "torch::jit::Suspend",
            "torch::jit::TokenTrie",
            "torch::jit::TaggedRange",
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

            "torch::autograd::_jvp_fn_t", "torch::autograd::profiler::post_process_t",
            "at::StringView" // Confusion with string_view and @StringView, and doesn't seem to be of any use in API

        ).skip())
        ;

        //// Functions not part of the API
        //// TORCH_API and the like are not honored on Linux but are on Windows. We must skip all public
        //// functions not marked as part of API.
        infoMap.put(new Info(
            "c10::detail::makeBaseType",
            "torch::detail::constructSchemaOrName",
            "at::operator <<(std::ostream&, at::Range&)",
            "caffe2::serialize::detail::getPadding",
            "at::assert_no_partial_overlap(c10::TensorImpl*, c10::TensorImpl*)",
            "at::TensorIteratorBase::apply_perm_and_mul",
            "c10::ivalue::ConstantString::operator <<", // No idea why these are not exported. TODO: dig
            "c10::ivalue::Future::operator <<",
            "c10::ivalue::EnumHolder::operator <<",
            "c10::ivalue::Await::operator <<",
            "c10::ivalue::EnumHolder::operator ==", // The friend operator is truly a member of c10::ivalue and not c10::ivalue::EnumHolder
            "c10::ivalue::EnumHolder::is", // Calls ==, which is not exported
            "c10::ivalue::EnumHolder::unqualifiedClassName",
            "c10::operator <<(std::ostream&, c10::SourceLocation&)",
            "torch::jit::Code::operator <<(std::ostream&, const torch::jit::Code&)", // The friend operator is truly a member of torch::jit and not torch::jit::Code
            "torch::jit::ClassDef::create",
            "torch::profiler::impl::getNvtxStr",
            "torch::autograd::add_node_to_current_graph_task_exec_info"
        ).skip());

        //// Aliases necessary because of Parser limited namespace resolution
        infoMap.put(new Info("at::Device", "torch::Device"))
               .put(new Info("torch::Tensor", "at::Tensor"))


               //// Classes kept but passed as generic pointer
               .put(new Info("c10::intrusive_ptr_target", "c10::nullopt", "c10::nullopt_t", "c10::impl::PyObjectSlot",
                   "_object",
                   "PyObject", "std::function<PyObject*(void*)>", "THPObjectPtr", "pyobj_list", "std::chrono::milliseconds", "std::exception_ptr", "std::type_info",
                   "std::pair<PyObject*,PyObject*>", "std::stack<std::pair<PyObject*,PyObject*> >", "torch::autograd::utils::DelayWarningHandler",
                   "std::is_same<torch::detail::pack<true>,torch::detail::pack<true> >", "at::cuda::NVRTC", "at::RecordFunctionCallback", "at::StepCallbacks", "THCState", "THHState",
                   "torch::autograd::ViewInfo", "torch::jit::InlinedCallStackPtr", "InlinedCallStackPtr", "torch::jit::ScopePtr", "torch::jit::BackendDebugInfoRecorder",
                   "torch::detail::TensorDataContainer", "at::ArrayRef<torch::detail::TensorDataContainer>",
                   "std::shared_ptr<caffe2::serialize::PyTorchStreamReader>", "caffe2::serialize::PyTorchStreamWriter",
                   "c10::detail::DictImpl::dict_map_type::iterator",
                   "std::iterator<std::forward_iterator_tag,c10::impl::DictEntryRef<c10::IValue,c10::IValue,c10::detail::DictImpl::dict_map_type::iterator> >",
                   "c10::optional<PyObject*>", "c10::optional<std::chrono::milliseconds>",
                   "c10::intrusive_ptr<torch::CustomClassHolder>", "c10::intrusive_ptr<caffe2::Blob>",
                   "c10::intrusive_ptr<c10::ivalue::Object>", "c10::ArrayRef<c10::intrusive_ptr<c10::ivalue::Object> >",
                   "torch::jit::DetachedBuffer::UniqueDetachedBuffer", "c10::optional<at::StepCallbacks>",
                   "c10::optional<c10::VaryingShape<int64_t>::ListOfOptionalElements>", "c10::optional<c10::VaryingShape<c10::Stride>::ListOfOptionalElements>",
                   "c10::optional<torch::autograd::ViewInfo>", "c10::optional<std::reference_wrapper<const std::string> >",
                   "c10::optional<torch::nn::TripletMarginWithDistanceLossOptions::distance_function_t>",
                   "c10::optional<torch::nn::functional::TripletMarginWithDistanceLossFuncOptions::distance_function_t>",
                   "std::tuple<torch::Tensor,c10::optional<std::vector<int64_t> >,c10::optional<std::vector<double> >,c10::optional<bool> >",
                   "c10::optional<std::shared_ptr<torch::jit::CompilationUnit> >", "c10::optional<std::weak_ptr<torch::jit::CompilationUnit> >",
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
               ).pointerTypes("Pointer").cast());


        ///// Special cases needing javaText
        infoMap
            .put(new Info("at::Tensor::toString", "at::TensorBase::toString", "torch::Tensor::toString", "torch::TensorBase::toString", "torch::jit::Graph::toString").javaText("public native @StdString String toString();"))
            .put(new Info("torch::jit::tracer::pauseTracing()").javaText("@Namespace(\"torch::jit::tracer\") public static native @ByVal @Cast(\"std::function<void()>*\") Pointer pauseTracing();"))
            .put(new Info("torch::jit::ProfileOp::getCallback()", "torch::jit::ProfileIValueOp::getCallback()").javaText(
                "public native @ByVal @Cast(\"std::function<void(std::vector<c10::IValue>&)>*\") Pointer getCallback();"))
            .put(new Info("torch::optim::AdamOptions::betas", "torch::optim::AdamWOptions::betas").javaText(
                "public native @Cast(\"std::tuple<double,double>*\") @ByRef @NoException DoublePointer betas();"))
            .put(new Info("torch::optim::Adagrad::step", "torch::optim::Adam::step", "torch::optim::AdamW::step",
                "torch::optim::LBFG::step", "torch::optim::RMSprop::step", "torch::optim::SGD::step").javaText(
                "public native @ByVal Tensor step(@ByVal(nullValue = \"torch::optim::Optimizer::LossClosure(nullptr)\") LossClosure closure);\n"
                + "public native @ByVal Tensor step();\n"));


        // Abstract classes because parent class is abstract, and not detected as such by Parser.
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
        infoMap.put(new Info("at::TensorIteratorBase").purify());


        //// Callback functions
        infoMap
            .put(new Info("c10::DeleterFnPtr").cast().valueTypes("PointerConsumer", "Pointer", "long"))
            .put(new Info("torch::Deleter", "std::function<void(void*)>").pointerTypes("PointerConsumer", "@Cast(\"void(*)(void*)\") Pointer", "@Cast(\"void(*)(void*)\") long"))
            .put(new Info("std::function<void()>").pointerTypes("Func"))
            .put(new Info("std::function<std::string(void)>").pointerTypes("StringSupplier"))
            .put(new Info("std::function<void(const std::string&)>").pointerTypes("StringConsumer"))
            .put(new Info("std::function<void(const c10::DDPLoggingData&)>",
                "std::function<void(const DDPLoggingData&)>").pointerTypes("DDPLogger"))
            .put(new Info("std::function<c10::TypePtr(c10::TypePtr)>").pointerTypes("TypeMapper"))
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
            .put(new Info("std::function<c10::QualifiedName(const std::shared_ptr<c10::ClassType>&)>").pointerTypes("TypeRenamer"))
            .put(new Info("std::function<std::string(const at::Tensor&)>").pointerTypes("TensorIdGetter"))
            .put(new Info("std::function<size_t(void)>").pointerTypes("SizeTSupplier"))
            .put(new Info("std::function<torch::Tensor()>").pointerTypes("LossClosure"))
            .put(new Info("std::function<torch::Tensor(const torch::Tensor&,const torch::Tensor&)>",
                "torch::nn::TripletMarginWithDistanceLossOptions::distance_function_t",
                "torch::nn::functional::TripletMarginWithDistanceLossFuncOptions::distance_function_t").pointerTypes("DistanceFunction"))
            .put(new Info("std::function<void(std::function<void()>)>").pointerTypes("Pointer"))

            .put(new Info("at::TensorBase::register_hook<std::function<void(at::TensorBase)> >").javaNames("register_hook"))
            .put(new Info("at::TensorBase::register_hook<std::function<at::TensorBase(at::TensorBase)> >").javaNames("register_hook"))
            .put(new Info("std::function<void(at::TensorBase)>").pointerTypes("VoidTensorHook"))
            .put(new Info("std::function<at::TensorBase(at::TensorBase)>").pointerTypes("TensorTensorHook"))
            .put(new Info("std::function<torch::Tensor(const torch::Tensor&)>").pointerTypes("TensorMapper"))
            .put(new Info("at::TensorBase::hook_return_void_t<std::function<void(at::TensorBase)> > ",
                "at::TensorBase::hook_return_void_t<std::function<at::TensorBase(at::TensorBase)> >").valueTypes("int"))
        ;
    }

    private static String template(String t, String... args) {
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
            String[] pt = new String[otherPointerTypes.length + 1];
            pt[0] = baseJavaName + "ArrayRef";
            System.arraycopy(otherPointerTypes, 0, pt, 1, otherPointerTypes.length);
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
                infoMap.put(new Info(template(cppNames[0], template("std::allocator", elementTypes[0])) + "(" + elementTypes[0] + "*)")
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
                   .put(new Info(template("operator std::conditional_t", template("std::is_reference", template("c10::detail::ivalue_to_const_ref_overload_return", t) + "::type") + "::value", "const " + t + "&", t) + "()")
                       .javaNames("get" + baseJavaName))
                   .put(new Info(template("c10::List", t) + "::size_type").valueTypes("long"))
                   .put(new Info(
                       template("c10::impl::ListElementReference", t, "typename c10::detail::ListImpl::list_type::iterator") + "::swap<T,Iterator>",
                       template("c10::impl::ListElementReference", t, "c10::detail::ListImpl::list_type::iterator") + "::swap<T,Iterator>",
                       template("c10::impl::ListElementReference", t, template("std::vector", t) + "::iterator") + "::swap<T,Iterator>")
                       .skip());
            infoMap.put(new Info(template("c10::List", t) + "::operator []").skip()) // Returns an internal_reference_type by value, which is a ListElementReference, whose copy constructor is disabled.
                   .put(new Info(
                       template("c10::impl::ListIterator", t, "c10::detail::ListImpl::list_type::iterator") + "::operator []",
                       template("c10::impl::ListIterator", t, "c10::detail::ListImpl::list_type::iterator") + "::operator *")
                       .skip()) // Returns ListElementReference by value, and ListElementReference has copy constructor disabled.
                   .put(new Info(template("std::conditional_t", template("std::is_reference", template("c10::detail::ivalue_to_const_ref_overload_return", t) + "::type") + "::value", "const " + t + "&", t))
                       .pointerTypes(itPointerType).valueTypes(elementValueType))

                   .put(new Info(template("c10::impl::swap", t, "typename c10::detail::ListImpl::list_type::iterator")).javaNames("swap").friendly());

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

    private static class PointerInfo {
        String javaBaseName;
        String javaName;
        final String[] argumentNames;
        String[] otherCppNames = new String[0];

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
    }

    @Namespace("std") public static native @MemberGetter @ByRef @Cast("std::istream*") Pointer cin();

    @Namespace("std") public static native @MemberGetter @ByRef @Cast("std::ostream*") Pointer cout();

    @Namespace("std") public static native @MemberGetter @ByRef @Cast("std::ostream*") Pointer cerr();

    @Namespace("std") public static native @MemberGetter @ByRef @Cast("std::ostream*") Pointer clog();

}
