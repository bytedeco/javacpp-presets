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

package org.bytedeco.cuda.presets;

import org.bytedeco.javacpp.annotation.NoException;
import org.bytedeco.javacpp.annotation.Platform;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.javacpp.tools.Info;
import org.bytedeco.javacpp.tools.InfoMap;
import org.bytedeco.javacpp.tools.InfoMapper;

/**
 *
 * @author Samuel Audet
 */
@Properties(inherit = cublas.class, value = {
    @Platform(include = {"<cudnn.h>", "<cudnn_version.h>", "<cudnn_ops_infer.h>", "<cudnn_ops_train.h>", "<cudnn_adv_infer.h>", "<cudnn_adv_train.h>", "<cudnn_cnn_infer.h>", "<cudnn_cnn_train.h>"},
        link = {"cudnn@.8", "cudnn_ops_infer@.8", "cudnn_ops_train@.8", "cudnn_adv_infer@.8", "cudnn_adv_train@.8", "cudnn_cnn_infer@.8", "cudnn_cnn_train@.8"}),
    @Platform(value = "windows-x86_64", preload = {"cudnn64_8", "cudnn_ops_infer64_8", "cudnn_ops_train64_8", "cudnn_adv_infer64_8", "cudnn_adv_train64_8", "cudnn_cnn_infer64_8", "cudnn_cnn_train64_8"})},
        target = "org.bytedeco.cuda.cudnn", global = "org.bytedeco.cuda.global.cudnn")
@NoException
public class cudnn implements InfoMapper {
    public void map(InfoMap infoMap) {
        infoMap.put(new Info("CUDNNWINAPI").cppTypes().annotations().cppText(""))
               .put(new Info("cudnnHandle_t").valueTypes("cudnnContext").pointerTypes("@ByPtrPtr cudnnContext"))
               .put(new Info("cudnnTensorDescriptor_t").valueTypes("cudnnTensorStruct")
                       .pointerTypes("@Cast(\"cudnnTensorStruct**\") @ByPtrPtr cudnnTensorStruct", "@Cast(\"cudnnTensorStruct**\") PointerPointer"))
               .put(new Info("cudnnFilterDescriptor_t").valueTypes("cudnnFilterStruct").pointerTypes("@ByPtrPtr cudnnFilterStruct"))
               .put(new Info("cudnnConvolutionDescriptor_t").valueTypes("cudnnConvolutionStruct").pointerTypes("@ByPtrPtr cudnnConvolutionStruct"))
               .put(new Info("cudnnPoolingDescriptor_t").valueTypes("cudnnPoolingStruct").pointerTypes("@ByPtrPtr cudnnPoolingStruct"))
               .put(new Info("cudnnLRNDescriptor_t").valueTypes("cudnnLRNStruct").pointerTypes("@ByPtrPtr cudnnLRNStruct"))
               .put(new Info("cudnnActivationDescriptor_t").valueTypes("cudnnActivationStruct").pointerTypes("@ByPtrPtr cudnnActivationStruct"))
               .put(new Info("cudnnSpatialTransformerDescriptor_t").valueTypes("cudnnSpatialTransformerStruct").pointerTypes("@ByPtrPtr cudnnSpatialTransformerStruct"))
               .put(new Info("cudnnOpTensorDescriptor_t").valueTypes("cudnnOpTensorStruct").pointerTypes("@ByPtrPtr cudnnOpTensorStruct"))
               .put(new Info("cudnnReduceTensorDescriptor_t").valueTypes("cudnnReduceTensorStruct").pointerTypes("@ByPtrPtr cudnnReduceTensorStruct"))
               .put(new Info("cudnnAlgorithmDescriptor_t").valueTypes("cudnnAlgorithmStruct").pointerTypes("@ByPtrPtr cudnnAlgorithmStruct"))
               .put(new Info("cudnnAlgorithmPerformance_t").valueTypes("cudnnAlgorithmPerformanceStruct").pointerTypes("@ByPtrPtr cudnnAlgorithmPerformanceStruct"))
               .put(new Info("cudnnRNNDescriptor_t").valueTypes("cudnnRNNStruct").pointerTypes("@ByPtrPtr cudnnRNNStruct"))
               .put(new Info("cudnnRNNDataDescriptor_t").valueTypes("cudnnRNNDataStruct").pointerTypes("@ByPtrPtr cudnnRNNDataStruct"))
               .put(new Info("cudnnPersistentRNNPlan_t").valueTypes("cudnnPersistentRNNPlan").pointerTypes("@ByPtrPtr cudnnPersistentRNNPlan"))
               .put(new Info("cudnnDropoutDescriptor_t").valueTypes("cudnnDropoutStruct").pointerTypes("@ByPtrPtr cudnnDropoutStruct"))
               .put(new Info("cudnnCTCLossDescriptor_t").valueTypes("cudnnCTCLossStruct").pointerTypes("@ByPtrPtr cudnnCTCLossStruct"))
               .put(new Info("cudnnTensorTransformDescriptor_t").valueTypes("cudnnTensorTransformStruct").pointerTypes("@ByPtrPtr cudnnTensorTransformStruct"))
               .put(new Info("cudnnSeqDataDescriptor_t").valueTypes("cudnnSeqDataStruct").pointerTypes("@ByPtrPtr cudnnSeqDataStruct"))
               .put(new Info("cudnnAttnDescriptor_t").valueTypes("cudnnAttnStruct").pointerTypes("@ByPtrPtr cudnnAttnStruct"))
               .put(new Info("cudnnFusedOpsConstParamPack_t").valueTypes("cudnnFusedOpsConstParamStruct").pointerTypes("@ByPtrPtr cudnnFusedOpsConstParamStruct"))
               .put(new Info("cudnnFusedOpsVariantParamPack_t").valueTypes("cudnnFusedOpsVariantParamStruct").pointerTypes("@ByPtrPtr cudnnFusedOpsVariantParamStruct"))
               .put(new Info("cudnnFusedOpsPlan_t").valueTypes("cudnnFusedOpsPlanStruct").pointerTypes("@ByPtrPtr cudnnFusedOpsPlanStruct"))

               .put(new Info("cudnnSetConvolution2dDescriptor_v4").javaText(
                          " public static int cudnnSetConvolution2dDescriptor(cudnnConvolutionStruct convDesc,\n"
                        + "        int pad_h, int pad_w, int u, int v, int dilation_h, int dilation_w, int mode) {\n"
                        + "    return cudnnSetConvolution2dDescriptor_v4(convDesc, pad_h, pad_w, u, v, dilation_h, dilation_w, mode);\n"
                        + "}\n"
                        + "public static native @Cast(\"cudnnStatus_t\") int cudnnSetConvolution2dDescriptor_v4(\n"
                        + "                                cudnnConvolutionStruct convDesc,\n"
                        + "                                int pad_h,\n"
                        + "                                int pad_w,\n"
                        + "                                int u,\n"
                        + "                                int v,\n"
                        + "                                int dilation_h,\n"
                        + "                                int dilation_w,\n"
                        + "                                @Cast(\"cudnnConvolutionMode_t\") int mode );\n"));
    }
}
