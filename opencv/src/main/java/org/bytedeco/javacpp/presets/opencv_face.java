/*
 * Copyright (C) 2015-2017 Samuel Audet
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
@Properties(inherit = {opencv_dnn.class, opencv_objdetect.class, opencv_photo.class, opencv_video.class}, value = {
    @Platform(include = {"<opencv2/face/predict_collector.hpp>", "<opencv2/face.hpp>", "<opencv2/face/facerec.hpp>",
                         "<opencv2/face/facemark.hpp>", "<opencv2/face/facemarkLBF.hpp>", "<opencv2/face/facemarkAAM.hpp>",
                         "<opencv2/face/face_alignment.hpp>"},
              link = "opencv_face@.3.4", preload = {"opencv_plot@.3.4", "opencv_tracking@.3.4"}),
    @Platform(value = "ios", preload = "libopencv_face"),
    @Platform(value = "windows", link = "opencv_face341", preload = {"opencv_plot341", "opencv_tracking341"})},
        target = "org.bytedeco.javacpp.opencv_face")
public class opencv_face implements InfoMapper {
    public void map(InfoMap infoMap) {
        infoMap.put(new Info("cv::face::BasicFaceRecognizer", "cv::face::EigenFaceRecognizer",
                             "cv::face::FisherFaceRecognizer", "cv::face::LBPHFaceRecognizer",
                             "cv::face::FacemarkLBF", "cv::face::FacemarkAAM").purify())
               .put(new Info("cv::face::FN_FaceDetector").cast().valueTypes("Pointer"))
               .put(new Info("bool (*)(cv::InputArray, cv::OutputArray, void*)").cast().pointerTypes("Pointer"));
    }
}
