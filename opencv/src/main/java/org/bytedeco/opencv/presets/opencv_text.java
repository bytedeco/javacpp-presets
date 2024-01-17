/*
 * Copyright (C) 2016-2022 Bram Biesbrouck, Samuel Audet
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
 * Wrapper for OpenCV module text, part of OpenCV_Contrib.
 *
 * @author Bram Biesbrouck
 */
@Properties(
    inherit = {opencv_dnn.class, opencv_features2d.class, opencv_ml.class},
    value = {
        @Platform(include = {"<opencv2/text.hpp>", "<opencv2/text/erfilter.hpp>", "<opencv2/text/ocr.hpp>", "opencv2/text/textDetector.hpp"},
            link = "opencv_text@.409"),
        @Platform(value = "ios", preload = "libopencv_text"),
        @Platform(value = "windows", link = "opencv_text490")},
    target = "org.bytedeco.opencv.opencv_text",
    global = "org.bytedeco.opencv.global.opencv_text"
)
public class opencv_text implements InfoMapper {
    public void map(InfoMap infoMap) {
        infoMap.put(new Info().javaText("import org.bytedeco.javacpp.annotation.Index;"))
               .put(new Info("std::deque<int>").pointerTypes("IntDeque").define())
               .put(new Info("std::vector<cv::text::ERStat>").pointerTypes("ERStatVector").define())
               .put(new Info("std::vector<std::vector<cv::text::ERStat> >").pointerTypes("ERStatVectorVector").define())
               .put(new Info("std::vector<int>").pointerTypes("IntVector").define())
               .put(new Info("std::vector<float>").pointerTypes("FloatVector").define())
               .put(new Info("std::vector<double>").pointerTypes("DoubleVector").define())
               .put(new Info("std::vector<cv::Vec2i>").pointerTypes("PointVector").cast())
               .put(new Info("std::vector<std::vector<cv::Vec2i> >").pointerTypes("PointVectorVector").cast())
               .put(new Info("cv::text::OCRBeamSearchDecoder::create").skipDefaults());
    }
}
