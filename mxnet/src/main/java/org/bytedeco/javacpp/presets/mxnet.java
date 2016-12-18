/*
 * Copyright (C) 2016 Samuel Audet
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

package org.bytedeco.javacpp.presets;

import org.bytedeco.javacpp.FunctionPointer;
import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.annotation.Cast;
import org.bytedeco.javacpp.annotation.Platform;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.javacpp.tools.Info;
import org.bytedeco.javacpp.tools.InfoMap;
import org.bytedeco.javacpp.tools.InfoMapper;

/**
 *
 * @author Samuel Audet
 */
@Properties(inherit = {openblas.class, opencv_imgcodecs.class, opencv_highgui.class}, target = "org.bytedeco.javacpp.mxnet", value = {
    @Platform(value = {"linux-x86", "macosx"}, compiler = "cpp11", define = {"DMLC_USE_CXX11 1", "MSHADOW_USE_CBLAS 1", "MSHADOW_IN_CXX11 1"},
        include = {"mxnet/c_api.h", "mxnet/c_predict_api.h", /*"dmlc/base.h", "dmlc/io.h", "dmlc/logging.h", "dmlc/type_traits.h",
                   "dmlc/parameter.h", "mshadow/base.h", "mshadow/expression.h", "mshadow/tensor.h", "mxnet/base.h",*/},
        link = "mxnet", includepath = {"/usr/local/cuda/include/",
        "/System/Library/Frameworks/vecLib.framework/", "/System/Library/Frameworks/Accelerate.framework/"}, linkpath = "/usr/local/cuda/lib/") })
public class mxnet implements InfoMapper {
    public void map(InfoMap infoMap) {
        infoMap.put(new Info("MXNET_EXTERN_C", "MXNET_DLL").cppTypes().annotations())
               .put(new Info("NDArrayHandle").valueTypes("NDArrayHandle").pointerTypes("PointerPointer", "@Cast(\"NDArrayHandle*\") @ByPtrPtr NDArrayHandle"))
               .put(new Info("FunctionHandle").annotations("@Const").valueTypes("FunctionHandle").pointerTypes("PointerPointer", "@Cast(\"FunctionHandle*\") @ByPtrPtr FunctionHandle"))
               .put(new Info("AtomicSymbolCreator").valueTypes("AtomicSymbolCreator").pointerTypes("PointerPointer", "@Cast(\"AtomicSymbolCreator*\") @ByPtrPtr AtomicSymbolCreator"))
               .put(new Info("SymbolHandle").valueTypes("SymbolHandle").pointerTypes("PointerPointer", "@Cast(\"SymbolHandle*\") @ByPtrPtr SymbolHandle"))
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
