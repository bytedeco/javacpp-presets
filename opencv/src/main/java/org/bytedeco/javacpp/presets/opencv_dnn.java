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

import org.bytedeco.javacpp.annotation.Platform;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.javacpp.tools.Info;
import org.bytedeco.javacpp.tools.InfoMap;
import org.bytedeco.javacpp.tools.InfoMapper;

/**
 *
 * @author Samuel Audet
 */
@Properties(inherit = opencv_imgproc.class, value = {
    @Platform(include = {"<opencv2/dnn.hpp>", "<opencv2/dnn/dict.hpp>","<opencv2/dnn/blob.hpp>",
                         "<opencv2/dnn/dnn.hpp>", "<opencv2/dnn/layer.hpp>"}, link = "opencv_dnn@.3.2"),
    @Platform(value = "windows", link = "opencv_dnn320")},
        target = "org.bytedeco.javacpp.opencv_dnn")
public class opencv_dnn implements InfoMapper {
    public void map(InfoMap infoMap) {
        infoMap.put(new Info("std::vector<cv::dnn::Blob>").pointerTypes("BlobVector").define())
               .put(new Info("std::vector<cv::dnn::Blob*>").pointerTypes("BlobPointerVector").define())
               .put(new Info("cv::dnn::Net::forward(cv::dnn::Net::LayerId, cv::dnn::Net::LayerId)",
                             "cv::dnn::Net::forward(cv::dnn::Net::LayerId*, cv::dnn::Net::LayerId*)",
                             "cv::dnn::Net::forwardOpt(cv::dnn::Net::LayerId)",
                             "cv::dnn::Net::forwardOpt(cv::dnn::Net::LayerId*)",
                             "cv::dnn::Net::setParam(cv::dnn::Net::LayerId, int, cv::dnn::Blob&)",
                             "cv::dnn::readTorchBlob(cv::String&, bool)",
                             "cv::dnn::Blob::fill(cv::InputArray)").skip())
               .put(new Info("cv::dnn::Layer* (*)(cv::dnn::LayerParams&)").annotations("@Convention(value=\"\", extern=\"C++\")"));
    }
}
