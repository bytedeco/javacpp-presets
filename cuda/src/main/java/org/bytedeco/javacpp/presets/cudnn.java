/*
 * Copyright (C) 2015-2016 Samuel Audet
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

import org.bytedeco.javacpp.annotation.Platform;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.javacpp.tools.Info;
import org.bytedeco.javacpp.tools.InfoMap;
import org.bytedeco.javacpp.tools.InfoMapper;

/**
 *
 * @author Samuel Audet
 */
@Properties(inherit = cuda.class, value = {
    @Platform(include = "<cudnn.h>", link = "cudnn@.5")},
        target = "org.bytedeco.javacpp.cudnn")
public class cudnn implements InfoMapper {
    public void map(InfoMap infoMap) {
        infoMap.put(new Info("CUDNNWINAPI").cppTypes().annotations().cppText(""))
               .put(new Info("cudnnHandle_t").valueTypes("cudnnContext").pointerTypes("@ByPtrPtr cudnnContext"))
               .put(new Info("cudnnTensorDescriptor_t").valueTypes("cudnnTensorStruct").pointerTypes("@Cast(\"cudnnTensorStruct**\") @ByPtrPtr cudnnTensorStruct"))
               .put(new Info("cudnnFilterDescriptor_t").valueTypes("cudnnFilterStruct").pointerTypes("@ByPtrPtr cudnnFilterStruct"))
               .put(new Info("cudnnConvolutionDescriptor_t").valueTypes("cudnnConvolutionStruct").pointerTypes("@ByPtrPtr cudnnConvolutionStruct"))
               .put(new Info("cudnnPoolingDescriptor_t").valueTypes("cudnnPoolingStruct").pointerTypes("@ByPtrPtr cudnnPoolingStruct"))
               .put(new Info("cudnnLRNDescriptor_t").valueTypes("cudnnLRNStruct").pointerTypes("@ByPtrPtr cudnnLRNStruct"))
               .put(new Info("cudnnActivationDescriptor_t").valueTypes("cudnnActivationStruct").pointerTypes("@ByPtrPtr cudnnActivationStruct"))
               .put(new Info("cudnnSpatialTransformerDescriptor_t").valueTypes("cudnnSpatialTransformerStruct").pointerTypes("@ByPtrPtr cudnnSpatialTransformerStruct"))
               .put(new Info("cudnnOpTensorDescriptor_t").valueTypes("cudnnOpTensorStruct").pointerTypes("@ByPtrPtr cudnnOpTensorStruct"))
               .put(new Info("cudnnRNNDescriptor_t").valueTypes("cudnnRNNStruct").pointerTypes("@ByPtrPtr cudnnRNNStruct"))
               .put(new Info("cudnnDropoutDescriptor_t").valueTypes("cudnnDropoutStruct").pointerTypes("@ByPtrPtr cudnnDropoutStruct"));
    }
}
