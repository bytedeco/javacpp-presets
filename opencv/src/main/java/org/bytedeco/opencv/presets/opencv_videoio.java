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
    inherit = opencv_imgcodecs.class,
    value = {
        @Platform(include = {/*"<opencv2/videoio/videoio_c.h>",*/ "<opencv2/videoio.hpp>"}, link = "opencv_videoio@.409"),
        @Platform(value = "android", preload = {
            "native_camera_r2.2.0", "native_camera_r2.3.4", "native_camera_r3.0.1", "native_camera_r4.0.0", "native_camera_r4.0.3",
            "native_camera_r4.1.1", "native_camera_r4.2.0", "native_camera_r4.3.0", "native_camera_r4.4.0"}),
        @Platform(value = "ios", preload = "libopencv_videoio"),
        @Platform(value = "windows", link = "opencv_videoio490", preload = {"opencv_ffmpeg490", "opencv_ffmpeg490_64"})},
    target = "org.bytedeco.opencv.opencv_videoio",
    global = "org.bytedeco.opencv.global.opencv_videoio"
)
public class opencv_videoio implements InfoMapper {
    public void map(InfoMap infoMap) {
        infoMap.put(new Info("CV_FOURCC_DEFAULT").cppTypes("int"))
               .put(new Info("cvCaptureFromFile", "cvCaptureFromAVI").cppTypes("CvCapture*", "const char*"))
               .put(new Info("cvCaptureFromCAM").cppTypes("CvCapture*", "int"))
               .put(new Info("cvCreateAVIWriter").cppTypes("CvVideoWriter*", "const char*", "int", "double", "CvSize", "int"))
               .put(new Info("cvWriteToAVI").cppTypes("int", "CvVideoWriter*", "IplImage*"))
               .put(new Info("std::vector<int>").annotations("@StdVector").valueTypes(
                        "@Cast({\"int*\", \"std::vector<int>&\"}) IntPointer",
                        "@Cast({\"int*\", \"std::vector<int>&\"}) IntBuffer",
                        "@Cast({\"int*\", \"std::vector<int>&\"}) int[]").pointerTypes("IntPointer", "IntBuffer", "int[]"))
               .put(new Info("cv::DefaultDeleter<CvCapture>").pointerTypes("CvCaptureDefaultDeleter"))
               .put(new Info("cv::DefaultDeleter<CvVideoWriter>").pointerTypes("CvVideoWriterDefaultDeleter"));
    }
}

