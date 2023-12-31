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
    inherit = {opencv_objdetect.class, opencv_photo.class},
    value = {
        @Platform(include = {"<opencv2/face/predict_collector.hpp>", "<opencv2/face.hpp>", "<opencv2/face/facerec.hpp>",
            "<opencv2/face/facemark.hpp>", "<opencv2/face/facemark_train.hpp>", "<opencv2/face/facemarkLBF.hpp>",
            "<opencv2/face/facemarkAAM.hpp>", "<opencv2/face/face_alignment.hpp>"},
            link = "opencv_face@.409"),
        @Platform(value = "ios", preload = "libopencv_face"),
        @Platform(value = "windows", link = "opencv_face490")},
    target = "org.bytedeco.opencv.opencv_face",
    global = "org.bytedeco.opencv.global.opencv_face"
)
public class opencv_face implements InfoMapper {
    public void map(InfoMap infoMap) {
        infoMap.put(new Info("cv::face::BasicFaceRecognizer", "cv::face::EigenFaceRecognizer",
                             "cv::face::FisherFaceRecognizer", "cv::face::LBPHFaceRecognizer",
                             "cv::face::FacemarkLBF", "cv::face::FacemarkAAM").purify())
               .put(new Info("cv::face::FN_FaceDetector").cast().valueTypes("Pointer"))
               .put(new Info("bool (*)(cv::InputArray, cv::OutputArray, void*)").cast().pointerTypes("Pointer"))
               .put(new Info("cv::face::getFaces").javaText("@Namespace(\"cv::face\") public static native @Cast(\"bool\") boolean getFaces(@ByVal Mat image, @ByRef RectVector faces, CParams params);"))
               .put(new Info("cv::face::getFacesHAAR").javaText("@Namespace(\"cv::face\") public static native @Cast(\"bool\") boolean getFacesHAAR(@ByVal Mat image, @ByRef RectVector faces, @Str String face_cascade_name);"))
               .put(new Info("cv::face::loadTrainingData(cv::String, std::vector<cv::String>&, cv::OutputArray, char, float)").javaText(
                       "@Namespace(\"cv::face\") public static native @Cast(\"bool\") boolean loadTrainingData( @Str String filename, @ByRef StringVector images,\n"
                     + "                                    @ByRef Point2fVectorVector facePoints,\n"
                     + "                                    @Cast(\"char\") byte delim/*=' '*/, float offset/*=0.0f*/);"))
               .put(new Info("cv::face::loadTrainingData(cv::String, cv::String, std::vector<cv::String>&, cv::OutputArray, float)").javaText(
                       "@Namespace(\"cv::face\") public static native @Cast(\"bool\") boolean loadTrainingData( @Str String imageList, @Str String groundTruth,\n"
                     + "                                    @ByRef StringVector images,\n"
                     + "                                    @ByRef Point2fVectorVector facePoints,\n"
                     + "                                    float offset/*=0.0f*/);"))
               .put(new Info("cv::face::loadFacePoints(cv::String, cv::OutputArray, float)").javaText(
                       "@Namespace(\"cv::face\") public static native @Cast(\"bool\") boolean loadFacePoints( @Str String filename, @ByRef Point2fVectorVector points,\n"
                     + "                                  float offset/*=0.0f*/);"))
               .put(new Info("cv::face::drawFacemarks(cv::InputOutputArray, cv::InputArray, cv::Scalar)").javaText(
                       "@Namespace(\"cv::face\") public static native void drawFacemarks( @ByVal Mat image, @ByRef Point2fVector points,\n"
                     + "                                 @ByVal(nullValue = \"cv::Scalar(255,0,0)\") Scalar color);"))
               .put(new Info("cv::face::FacemarkTrain::addTrainingSample").javaText("public native @Cast(\"bool\") boolean addTrainingSample(@ByVal Mat image, @ByRef Point2fVector landmarks);"))
               .put(new Info("cv::face::Facemark::fit").javaText(
                       "public native @Cast(\"bool\") boolean fit( @ByVal Mat image,\n"
                     + "                      @ByRef RectVector faces,\n"
                     + "                      @ByRef Point2fVectorVector landmarks);"))
               .put(new Info("cv::face::FacemarkTrain::getFaces").javaText("public native @Cast(\"bool\") boolean getFaces(@ByVal Mat image, @ByRef RectVector faces);"))
               .put(new Info("cv::face::FacemarkAAM::fitConfig").javaText("public native @Cast(\"bool\") boolean fitConfig( @ByVal Mat image, @ByRef RectVector roi, @ByRef Point2fVectorVector _landmarks, @StdVector Config runtime_params );"))
               .put(new Info("cv::face::FacemarkKazemi::fit").javaText("public native @Cast(\"bool\") boolean fit( @ByVal Mat image, @ByRef RectVector faces, @ByRef Point2fVectorVector landmarks );"))
               .put(new Info("cv::face::FacemarkKazemi::getFaces").javaText("public native @Cast(\"bool\") boolean getFaces(@ByVal Mat image, @ByRef RectVector faces);"))
        ;
    }
}
