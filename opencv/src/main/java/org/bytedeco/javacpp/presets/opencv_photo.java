/*
 * Copyright (C) 2013-2018 Samuel Audet
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
import org.bytedeco.javacpp.tools.InfoMap;
import org.bytedeco.javacpp.tools.InfoMapper;

/**
 *
 * @author Samuel Audet
 */
@Properties(inherit = opencv_imgproc.class, value = {
    @Platform(include = {/*"<opencv2/photo/photo_c.h>",*/ "<opencv2/photo.hpp>", "<opencv2/photo/cuda.hpp>"},
              link = "opencv_photo@.4.0",
              preload = {"opencv_cuda@.4.0", "opencv_cudaarithm@.4.0", "opencv_cudafilters@.4.0", "opencv_cudaimgproc@.4.0"}),
    @Platform(value = "ios", preload = "libopencv_photo"),
    @Platform(value = "windows", link = "opencv_photo401",
              preload = {"opencv_cuda401", "opencv_cudaarithm401", "opencv_cudafilters401", "opencv_cudaimgproc401"})},
        target = "org.bytedeco.javacpp.opencv_photo")
public class opencv_photo implements InfoMapper {
    public void map(InfoMap infoMap) {
    }
}

