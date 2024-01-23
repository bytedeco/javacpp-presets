/*
 * Copyright (C) 2016-2022 Samuel Audet
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

package org.bytedeco.opencv.presets;

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
    inherit = opencv_imgproc.class,
    value = {
        @Platform(include = {"<opencv2/dnn.hpp>", "<opencv2/dnn/version.hpp>", "<opencv2/dnn/dict.hpp>","<opencv2/dnn/all_layers.hpp>",
            "<opencv2/dnn/dnn.hpp>", "<opencv2/dnn/layer.hpp>", "<opencv2/dnn/shape_utils.hpp>"},
            link = "opencv_dnn@.409"),
        @Platform(value = "ios", preload = "libopencv_dnn"),
        @Platform(value = "windows", link = "opencv_dnn490")},
    target = "org.bytedeco.opencv.opencv_dnn",
    global = "org.bytedeco.opencv.global.opencv_dnn"
)
public class opencv_dnn implements InfoMapper {
    public void map(InfoMap infoMap) {
        infoMap.put(new Info("CV__DNN_EXPERIMENTAL_NS_BEGIN", "CV__DNN_EXPERIMENTAL_NS_END",
                             "CV__DNN_INLINE_NS_BEGIN", "CV__DNN_INLINE_NS_END", "CV__DNN_INLINE_NS").cppTypes().annotations())
               .put(new Info("!defined CV_DOXYGEN && !defined CV_DNN_DONT_ADD_INLINE_NS").define(false))
               .put(new Info("cv::dnn::MatShape").annotations("@StdVector").pointerTypes("IntPointer"))
               .put(new Info("std::pair<int,float>").pointerTypes("IntFloatPair").define())
               .put(new Info("std::vector<cv::dnn::MatShape>").pointerTypes("MatShapeVector").define())
               .put(new Info("std::vector<std::vector<cv::dnn::MatShape> >").pointerTypes("MatShapeVectorVector").define())
               .put(new Info("std::vector<std::vector<cv::Range> >").pointerTypes("RangeVectorVector").define())
               .put(new Info("std::vector<std::pair<cv::dnn::Backend,cv::dnn::Target> >").pointerTypes("IntIntPairVector").cast())
               .put(new Info("cv::dnn::LRNLayer::type").javaNames("lrnType"))
               .put(new Info("cv::dnn::PoolingLayer::type").javaNames("poolingType"))
               .put(new Info("cv::dnn::BlankLayer", "cv::dnn::LSTMLayer", "cv::dnn::RNNLayer", "cv::dnn::BaseConvolutionLayer",
                             "cv::dnn::ConvolutionLayer", "cv::dnn::DeconvolutionLayer", "cv::dnn::LRNLayer", "cv::dnn::PoolingLayer",
                             "cv::dnn::SoftmaxLayer", "cv::dnn::InnerProductLayer", "cv::dnn::MVNLayer", "cv::dnn::ReshapeLayer",
                             "cv::dnn::FlattenLayer", "cv::dnn::ConcatLayer", "cv::dnn::SplitLayer", "cv::dnn::SliceLayer",
                             "cv::dnn::PermuteLayer", "cv::dnn::PaddingLayer", "cv::dnn::ActivationLayer", "cv::dnn::ReLULayer",
                             "cv::dnn::ChannelsPReLULayer", "cv::dnn::ELULayer", "cv::dnn::TanHLayer", "cv::dnn::SigmoidLayer",
                             "cv::dnn::BNLLLayer", "cv::dnn::AbsLayer", "cv::dnn::PowerLayer", "cv::dnn::CropLayer", "cv::dnn::ExpLayer", "cv::dnn::EltwiseLayer",
                             "cv::dnn::BatchNormLayer", "cv::dnn::MaxUnpoolLayer", "cv::dnn::ScaleLayer", "cv::dnn::ShiftLayer",
                             "cv::dnn::PriorBoxLayer", "cv::dnn::DetectionOutputLayer", "cv::dnn::NormalizeBBoxLayer", "cv::dnn::ProposalLayer",
                             "cv::dnn::ReLU6Layer", "cv::dnn::ReorgLayer", "cv::dnn::RegionLayer", "cv::dnn::ResizeNearestNeighborLayer",
                             "cv::dnn::CropAndResizeLayer", "cv::dnn::InterpLayer", "cv::dnn::ResizeLayer", "cv::dnn::ShuffleChannelLayer",
                             "cv::dnn::experimental_dnn_v5::ShuffleChannelLayer", "cv::dnn::SwishLayer", "cv::dnn::MishLayer").purify())
               .put(new Info("cv::dnn::Net::forward(cv::dnn::Net::LayerId, cv::dnn::Net::LayerId)",
                             "cv::dnn::Net::forward(cv::dnn::Net::LayerId*, cv::dnn::Net::LayerId*)",
                             "cv::dnn::Net::forwardOpt(cv::dnn::Net::LayerId)",
                             "cv::dnn::Net::forwardOpt(cv::dnn::Net::LayerId*)",
                             "std::map<cv::String,cv::dnn::DictValue>::const_iterator").skip())
               .put(new Info("std::vector<cv::Mat*>").pointerTypes("MatPointerVector").define())
               .put(new Info("cv::dnn::Layer* (*)(cv::dnn::LayerParams&)").javaText(
                       "@Convention(value=\"\", extern=\"C++\") public static class Constructor extends FunctionPointer {\n"
                     + "    static { Loader.load(); }\n"
                     + "    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */\n"
                     + "    public    Constructor(Pointer p) { super(p); }\n"
                     + "    protected Constructor() { allocate(); }\n"
                     + "    private native void allocate();\n"
                     + "    public native @Cast({\"\", \"cv::Ptr<cv::dnn::Layer>\"}) @Ptr Layer call(@ByRef LayerParams params);\n"
                     + "}\n"));
    }
}
