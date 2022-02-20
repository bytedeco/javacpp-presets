/*
 * Copyright (C) 2019-2022 Samuel Audet
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

package org.bytedeco.tensorrt.presets;

import org.bytedeco.javacpp.annotation.Platform;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.javacpp.tools.Info;
import org.bytedeco.javacpp.tools.InfoMap;
import org.bytedeco.javacpp.tools.InfoMapper;

/**
 *
 * @author Samuel Audet
 */
@Properties(
    inherit = nvinfer_plugin.class,
    value = @Platform(
        include = "NvOnnxParser.h",
        link = "nvonnxparser@.8.2.3"),
    target = "org.bytedeco.tensorrt.nvonnxparser",
    global = "org.bytedeco.tensorrt.global.nvonnxparser")
public class nvonnxparser implements InfoMapper {
    public void map(InfoMap infoMap) {
        infoMap.put(new Info("SWIG").define(false))
               .put(new Info("std::pair<std::vector<size_t>,bool>").pointerTypes("SubGraph_t").define())
               .put(new Info("std::vector<SubGraph_t>").pointerTypes("SubGraphCollection_t").define())
               .put(new Info("std::vector<size_t>").annotations("@StdVector").pointerTypes("SizeTPointer"))
               .put(new Info("nvonnxparser::IPluginFactory").pointerTypes("OnnxPluginFactory"))
               .put(new Info("nvonnxparser::IPluginFactoryExt").pointerTypes("OnnxPluginFactoryExt"))
               .put(new Info("nvonnxparser::ErrorCode").valueTypes("org.bytedeco.tensorrt.global.nvonnxparser.ErrorCode").enumerate())
               .put(new Info("nvonnxparser::EnumMax<nvonnxparser::ErrorCode>", "nvonnxparser::EnumMax<ErrorCode>").javaNames("ErrorCodeEnumMax"));
    }
}
