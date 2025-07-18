/*
 * Copyright (C) 2017-2022 Sam Carlberg, Samuel Audet
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

@Properties(
    inherit = opencv_imgproc.class,
    value = {
        @Platform(
            include = "<opencv2/cudawarping.hpp>",
            link = "opencv_cudawarping@.412",
            extension = "-gpu"
        ),
        @Platform(
            value = "windows",
            link = "opencv_cudawarping4120",
            extension = "-gpu"
        )
    },
    global = "org.bytedeco.opencv.global.opencv_cudawarping"
)
public class opencv_cudawarping implements InfoMapper {

    @Override
    public void map(InfoMap infoMap) {
        infoMap.put(new Info("cv::cuda::warpAffine").javaText(
                        "@Namespace(\"cv::cuda\") public static native void warpAffine(@ByVal GpuMat src, @ByVal GpuMat dst, @ByVal Mat M, @ByVal Size dsize, int flags/*=cv::INTER_LINEAR*/,\n"
                      + "    int borderMode/*=cv::BORDER_CONSTANT*/, @ByVal(nullValue = \"cv::Scalar()\") Scalar borderValue, @ByRef(nullValue = \"cv::cuda::Stream::Null()\") Stream stream);\n"
                      + "@Namespace(\"cv::cuda\") public static native void warpAffine(@ByVal GpuMat src, @ByVal GpuMat dst, @ByVal Mat M, @ByVal Size dsize);\n"))
               .put(new Info("cv::cuda::warpPerspective").javaText(
                        "@Namespace(\"cv::cuda\") public static native void warpPerspective(@ByVal GpuMat src, @ByVal GpuMat dst, @ByVal Mat M, @ByVal Size dsize, int flags/*=cv::INTER_LINEAR*/,\n"
                      + "    int borderMode/*=cv::BORDER_CONSTANT*/, @ByVal(nullValue = \"cv::Scalar()\") Scalar borderValue, @ByRef(nullValue = \"cv::cuda::Stream::Null()\") Stream stream);\n"
                      + "@Namespace(\"cv::cuda\") public static native void warpPerspective(@ByVal GpuMat src, @ByVal GpuMat dst, @ByVal Mat M, @ByVal Size dsize);\n"));
    }
}
