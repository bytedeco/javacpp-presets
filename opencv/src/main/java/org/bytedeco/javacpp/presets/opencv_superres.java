/*
 * Copyright (C) 2014-2017 Samuel Audet
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
@Properties(inherit = {opencv_video.class, opencv_videoio.class}, value = {
    @Platform(not = "ios", include = {"<opencv2/superres.hpp>", "<opencv2/superres/optical_flow.hpp>"}, link = "opencv_superres@.3.4",
              preload = {"opencv_cuda@.3.4", "opencv_cudacodec@.3.4", "opencv_cudaarithm@.3.4", "opencv_cudafilters@.3.4",
                         "opencv_cudaimgproc@.3.4", "opencv_cudafeatures2d@.3.4", "opencv_cudalegacy@.3.4",
                         "opencv_cudaoptflow@.3.4", "opencv_cudawarping@.3.4"}),
    @Platform(value = "windows", link = "opencv_superres343",
              preload = {"opencv_cuda343", "opencv_cudacodec343", "opencv_cudaarithm343", "opencv_cudafilters343",
                         "opencv_cudaimgproc343", "opencv_cudafeatures2d343", "opencv_cudalegacy343",
                         "opencv_cudaoptflow343", "opencv_cudawarping343"})},
        target = "org.bytedeco.javacpp.opencv_superres")
public class opencv_superres implements InfoMapper {
    public void map(InfoMap infoMap) {
        infoMap.put(new Info("cv::superres::FarnebackOpticalFlow").pointerTypes("SuperResFarnebackOpticalFlow"))
               .put(new Info("cv::superres::DualTVL1OpticalFlow").pointerTypes("SuperResDualTVL1OpticalFlow"));
    }
}

