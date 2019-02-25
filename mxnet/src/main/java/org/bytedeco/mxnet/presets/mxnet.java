/*
 * Copyright (C) 2016-2018 Samuel Audet
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

package org.bytedeco.mxnet.presets;

import java.util.Arrays;
import java.util.List;
import org.bytedeco.javacpp.ClassProperties;
import org.bytedeco.javacpp.FunctionPointer;
import org.bytedeco.javacpp.LoadEnabled;
import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.annotation.Cast;
import org.bytedeco.javacpp.annotation.Platform;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.javacpp.tools.Info;
import org.bytedeco.javacpp.tools.InfoMap;
import org.bytedeco.javacpp.tools.InfoMapper;

import org.bytedeco.openblas.presets.*;
import org.bytedeco.opencv.presets.*;

/**
 *
 * @author Samuel Audet
 */
@Properties(inherit = {openblas.class, opencv_imgcodecs.class, opencv_highgui.class}, target = "org.bytedeco.mxnet", global = "org.bytedeco.mxnet.global.mxnet", value = {
    @Platform(value = {"linux-x86", "macosx", "windows"}, compiler = {"cpp11", "fastfpu"},
        define = {"DMLC_USE_CXX11 1", "MSHADOW_USE_CBLAS 1", "MSHADOW_IN_CXX11 1", "MSHADOW_USE_CUDA 0", "MSHADOW_USE_F16C 0"},
        include = {"mxnet/c_api.h", "mxnet/c_predict_api.h", /*"dmlc/base.h", "dmlc/io.h", "dmlc/logging.h", "dmlc/type_traits.h",
                   "dmlc/parameter.h", "mshadow/base.h", "mshadow/expression.h", "mshadow/tensor.h", "mxnet/base.h",*/
                   "org_apache_mxnet_init_native_c_api.cc", "org_apache_mxnet_native_c_api.cc"},
        link = "mxnet", preload = {"mkldnn@.0", "libmxnet"}, /*resource = {"include", "lib"},*/
        includepath = {"/System/Library/Frameworks/vecLib.framework/", "/System/Library/Frameworks/Accelerate.framework/"}),
    @Platform(value = {"linux-x86_64", "macosx-x86_64", "windows-x86_64"},
        define = {"DMLC_USE_CXX11 1", "MSHADOW_USE_CBLAS 1", "MSHADOW_IN_CXX11 1", "MSHADOW_USE_CUDA 1", "MSHADOW_USE_F16C 0"},
        link = {"cudart@.10.0#", "cuda@.10.0#", "mxnet"}, preload = {"mkldnn@.0", "libmxnet"},
        includepath = {"/usr/local/cuda/include/", "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.0/include/"},
        linkpath = {"/usr/local/cuda/lib/", "/usr/local/cuda/lib64/", "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.0/lib/x64/"},
        preloadpath = {"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.0/bin/"}, extension = "-gpu") })
public class mxnet implements LoadEnabled, InfoMapper {

    @Override public void init(ClassProperties properties) {
        String platform = properties.getProperty("platform");
        String extension = properties.getProperty("platform.extension");
        List<String> preloads = properties.get("platform.preload");
        List<String> resources = properties.get("platform.preloadresource");

        // Only apply this at load time since we don't want to copy the MKL or CUDA libraries here
        if (Loader.isLoadLibraries()) {
            preloads.addAll(0, Arrays.asList("iomp5", "libiomp5md", "mklml", "mklml_intel"));
            resources.add("/org/bytedeco/mkldnn/");
        }
        if (!Loader.isLoadLibraries() || extension == null || !extension.equals("-gpu")) {
            return;
        }
        int i = 0;
        String[] libs = {"cudart", "cublas", "cufft", "curand", "cusolver", "cudnn", "nccl", "nvrtc"};
        for (String lib : libs) {
            switch (platform) {
                case "linux-x86_64":
                case "macosx-x86_64":
                    lib += lib.equals("cudnn") ? "@.7" : lib.equals("nccl") ? "@.2" : "@.10.0";
                    break;
                case "windows-x86_64":
                    lib += lib.equals("cudnn") ? "64_7" : lib.equals("nvrtc") ? "64_100_0" : "64_100";
                    break;
                default:
                    continue; // no CUDA
            }
            if (!preloads.contains(lib)) {
                preloads.add(i++, lib);
            }
        }
        if (i > 0) {
            resources.add("/org/bytedeco/cuda/");
        }
    }

    public void map(InfoMap infoMap) {
        infoMap.put(new Info("org_apache_mxnet_init_native_c_api.cc", "org_apache_mxnet_native_c_api.cc").skip())
               .put(new Info("MXNET_EXTERN_C", "MXNET_DLL").cppTypes().annotations())
               .put(new Info("NDArrayHandle").valueTypes("NDArrayHandle").pointerTypes("PointerPointer", "@Cast(\"NDArrayHandle*\") @ByPtrPtr NDArrayHandle"))
               .put(new Info("const NDArrayHandle").valueTypes("NDArrayHandle").pointerTypes("@Cast(\"NDArrayHandle*\") PointerPointer", "@Cast(\"NDArrayHandle*\") @ByPtrPtr NDArrayHandle"))
               .put(new Info("FunctionHandle").annotations("@Const").valueTypes("FunctionHandle").pointerTypes("PointerPointer", "@Cast(\"FunctionHandle*\") @ByPtrPtr FunctionHandle"))
               .put(new Info("AtomicSymbolCreator").valueTypes("AtomicSymbolCreator").pointerTypes("PointerPointer", "@Cast(\"AtomicSymbolCreator*\") @ByPtrPtr AtomicSymbolCreator"))
               .put(new Info("SymbolHandle").valueTypes("SymbolHandle").pointerTypes("PointerPointer", "@Cast(\"SymbolHandle*\") @ByPtrPtr SymbolHandle"))
               .put(new Info("const SymbolHandle").valueTypes("SymbolHandle").pointerTypes("@Cast(\"SymbolHandle*\") PointerPointer", "@Cast(\"SymbolHandle*\") @ByPtrPtr SymbolHandle"))
               .put(new Info("AtomicSymbolHandle").valueTypes("AtomicSymbolHandle").pointerTypes("PointerPointer", "@Cast(\"AtomicSymbolHandle*\") @ByPtrPtr AtomicSymbolHandle"))
               .put(new Info("ExecutorHandle").valueTypes("ExecutorHandle").pointerTypes("PointerPointer", "@Cast(\"ExecutorHandle*\") @ByPtrPtr ExecutorHandle"))
               .put(new Info("DataIterCreator").valueTypes("DataIterCreator").pointerTypes("PointerPointer", "@Cast(\"DataIterCreator*\") @ByPtrPtr DataIterCreator"))
               .put(new Info("DataIterHandle").valueTypes("DataIterHandle").pointerTypes("PointerPointer", "@Cast(\"DataIterHandle*\") @ByPtrPtr DataIterHandle"))
               .put(new Info("KVStoreHandle").valueTypes("KVStoreHandle").pointerTypes("PointerPointer", "@Cast(\"KVStoreHandle*\") @ByPtrPtr KVStoreHandle"))
               .put(new Info("RecordIOHandle").valueTypes("RecordIOHandle").pointerTypes("PointerPointer", "@Cast(\"RecordIOHandle*\") @ByPtrPtr RecordIOHandle"))
               .put(new Info("RtcHandle").valueTypes("RtcHandle").pointerTypes("PointerPointer", "@Cast(\"RtcHandle*\") @ByPtrPtr RtcHandle"))
               .put(new Info("OptimizerCreator").valueTypes("OptimizerCreator").pointerTypes("PointerPointer", "@Cast(\"OptimizerCreator*\") @ByPtrPtr OptimizerCreator"))
               .put(new Info("OptimizerHandle").valueTypes("OptimizerHandle").pointerTypes("PointerPointer", "@Cast(\"OptimizerHandle*\") @ByPtrPtr OptimizerHandle"));
/*
        infoMap.put(new Info("DMLC_USE_REGEX", "DMLC_USE_CXX11", "DMLC_ENABLE_STD_THREAD").define())
               .put(new Info("!defined(__GNUC__)", "_MSC_VER < 1900", "__APPLE__", "defined(_MSC_VER) && _MSC_VER < 1900").define(false))
               .put(new Info("std::basic_ostream<char>", "std::basic_istream<char>", "std::runtime_error").cast().pointerTypes("Pointer"))
               .put(new Info("LOG_INFO", "LOG_ERROR", "LOG_WARNING", "LOG_FATAL", "LOG_QFATAL", "LG", "LOG_DFATAL", "DFATAL").cppTypes().annotations())
               .put(new Info("type_name<float>").javaNames("type_name_float"))
               .put(new Info("type_name<double>").javaNames("type_name_double"))
               .put(new Info("type_name<int>").javaNames("type_name_int"))
               .put(new Info("type_name<uint32_t>").javaNames("type_name_uint32_t"))
               .put(new Info("type_name<uint64_t>").javaNames("type_name_uint64_t"))
               .put(new Info("type_name<bool>").javaNames("type_name_bool"))
               .put(new Info("std::pair<std::string,std::string>").pointerTypes("StringStringPair").define())
               .put(new Info("IfThenElseType<dmlc::is_arithmetic<int>::value,dmlc::parameter::FieldEntryNumeric<dmlc::parameter::FieldEntry<int>,int>,"
                           + "dmlc::parameter::FieldEntryBase<dmlc::parameter::FieldEntry<int>,int> >::Type").pointerTypes("Pointer"))
               .put(new Info("dmlc::parameter::FieldEntry<int>").pointerTypes("IntFieldEntry").define())
               .put(new Info("dmlc::parameter::FieldEntryBase<dmlc::parameter::FieldEntry<int>,int>").pointerTypes("IntIntFieldEntryBase").define())
               .put(new Info("dmlc::parameter::FieldEntryNumeric<dmlc::parameter::FieldEntry<int>,int>").pointerTypes("IntIntFieldEntryNumeric").define())

               .put(new Info("MSHADOW_FORCE_INLINE", "MSHADOW_XINLINE", "MSHADOW_CINLINE", "MSHADOW_CONSTEXPR", "MSHADOW_DEFAULT_DTYPE",
                             "MSHADOW_USE_GLOG", "MSHADOW_ALLOC_PAD", "MSHADOW_SCALAR_").cppTypes().annotations())
               .put(new Info("mshadow::red::limits::MinValue<float>").javaNames("MinValueFloat"))
               .put(new Info("mshadow::red::limits::MinValue<double>").javaNames("MinValueDouble"))
               .put(new Info("mshadow::red::limits::MinValue<int>").javaNames("MinValueInt"));

        for (int i = 1; i <= 5; i++) {
            infoMap.put(new Info("mshadow::Shape<" + i + ">").pointerTypes("Shape" + i).define())
                   .put(new Info("mshadow::Shape<mshadow::Shape<" + i + ">::kDimension>").pointerTypes("Shape" + i));
            if (i > 1) {
                infoMap.put(new Info("mshadow::Shape<mshadow::Shape<" + i + ">::kSubdim>").pointerTypes("Shape" + (i - 1)));
            } else {
                infoMap.put(new Info("mshadow::Shape<1>::SubShape").skip());
            }
        }
*/
    }
}
