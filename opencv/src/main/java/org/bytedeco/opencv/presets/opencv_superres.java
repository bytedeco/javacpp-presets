/*
 * Copyright (C) 2014-2022 Samuel Audet
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
    inherit = {opencv_objdetect.class, opencv_optflow.class},
    value = {
        @Platform(not = "ios", include = {"<opencv2/superres.hpp>", "<opencv2/superres/optical_flow.hpp>"}, link = "opencv_superres@.412",
            preload = {"opencv_cuda@.412", "opencv_cudacodec@.412", "opencv_cudaarithm@.412", "opencv_cudafilters@.412",
                "opencv_cudaimgproc@.412", "opencv_cudafeatures2d@.412", "opencv_cudalegacy@.412",
                "opencv_cudaoptflow@.412", "opencv_cudawarping@.412"}),
        @Platform(value = "windows", link = "opencv_superres4120",
            preload = {"opencv_cuda4120", "opencv_cudacodec4120", "opencv_cudaarithm4120", "opencv_cudafilters4120",
                "opencv_cudaimgproc4120", "opencv_cudafeatures2d4120", "opencv_cudalegacy4120",
                "opencv_cudaoptflow4120", "opencv_cudawarping4120"})},
    target = "org.bytedeco.opencv.opencv_superres",
    global = "org.bytedeco.opencv.global.opencv_superres"
)
public class opencv_superres implements InfoMapper {
    public void map(InfoMap infoMap) {
        infoMap.put(new Info("override").annotations()) // classes with multiple parents cannot inherit all functions, so disable override annotation
               .put(new Info("cv::superres::FarnebackOpticalFlow").pointerTypes("SuperResFarnebackOpticalFlow"))
               .put(new Info("cv::superres::DualTVL1OpticalFlow").pointerTypes("SuperResDualTVL1OpticalFlow"));
    }
}

