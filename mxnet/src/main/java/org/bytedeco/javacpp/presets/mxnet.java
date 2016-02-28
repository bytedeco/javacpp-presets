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
@Properties(inherit = opencv_highgui.class, target = "org.bytedeco.javacpp.mxnet", value = {
    @Platform(not = "android", compiler = "cpp11", define = {"DMLC_USE_CXX11 1", "MSHADOW_USE_CBLAS 1", "MSHADOW_IN_CXX11 1"},
        include = {"mxnet/c_api.h", "mxnet/c_predict_api.h"}, link = "mxnet", includepath = {"/usr/local/cuda/include/",
        "/System/Library/Frameworks/vecLib.framework/", "/System/Library/Frameworks/Accelerate.framework/"}, linkpath = "/usr/local/cuda/lib/") })
public class mxnet implements InfoMapper {
    public void map(InfoMap infoMap) {
        infoMap.put(new Info("MXNET_EXTERN_C", "MXNET_DLL").cppTypes().annotations())
               .put(new Info("NDArrayHandle").valueTypes("NDArrayHandle").pointerTypes("PointerPointer"))
               .put(new Info("FunctionHandle").annotations("@Const").valueTypes("FunctionHandle").pointerTypes("PointerPointer"))
               .put(new Info("AtomicSymbolCreator").valueTypes("AtomicSymbolCreator").pointerTypes("PointerPointer"))
               .put(new Info("SymbolHandle").valueTypes("SymbolHandle").pointerTypes("PointerPointer"))
               .put(new Info("AtomicSymbolHandle").valueTypes("AtomicSymbolHandle").pointerTypes("PointerPointer"))
               .put(new Info("ExecutorHandle").valueTypes("ExecutorHandle").pointerTypes("PointerPointer"))
               .put(new Info("DataIterCreator").valueTypes("DataIterCreator").pointerTypes("PointerPointer"))
               .put(new Info("DataIterHandle").valueTypes("DataIterHandle").pointerTypes("PointerPointer"))
               .put(new Info("KVStoreHandle").valueTypes("KVStoreHandle").pointerTypes("PointerPointer"))
               .put(new Info("RecordIOHandle").valueTypes("RecordIOHandle").pointerTypes("PointerPointer"))
               .put(new Info("RtcHandle").valueTypes("RtcHandle").pointerTypes("PointerPointer"))
               .put(new Info("OptimizerCreator").valueTypes("OptimizerCreator").pointerTypes("PointerPointer"))
               .put(new Info("OptimizerHandle").valueTypes("OptimizerHandle").pointerTypes("PointerPointer"));
    }
}
