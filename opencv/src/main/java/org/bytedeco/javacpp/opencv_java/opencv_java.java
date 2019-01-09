/*
 * Copyright (C) 2018 Samuel Audet
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

package org.bytedeco.javacpp.opencv_java;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Platform;
import org.bytedeco.javacpp.annotation.Properties;


/**
 * This is only a placeholder to facilitate loading the {@code opencv_java} module with JavaCPP.
 * <p>
 * Call {@code Loader.load(opencv_java.class)} before using the API in the {@code org.opencv} namespace.
 *
 * @author Samuel Audet
 */
@Properties(
    inherit = {
        org.bytedeco.javacpp.opencv_aruco.opencv_aruco.class,
        org.bytedeco.javacpp.opencv_bgsegm.opencv_bgsegm.class,
        org.bytedeco.javacpp.opencv_bioinspired.opencv_bioinspired.class,
        org.bytedeco.javacpp.opencv_face.opencv_face.class,
        org.bytedeco.javacpp.opencv_img_hash.opencv_img_hash.class,
        org.bytedeco.javacpp.opencv_structured_light.opencv_structured_light.class,
        org.bytedeco.javacpp.opencv_text.opencv_text.class,
        org.bytedeco.javacpp.opencv_tracking.opencv_tracking.class,
        org.bytedeco.javacpp.opencv_xfeatures2d.opencv_xfeatures2d.class,
        org.bytedeco.javacpp.opencv_ximgproc.opencv_ximgproc.class,
        org.bytedeco.javacpp.opencv_xphoto.opencv_xphoto.class,
    },
    value = {
        @Platform(preload = {"opencv_cuda@.4.0", "opencv_cudaarithm@.4.0", "opencv_cudafilters@.4.0", "opencv_cudaimgproc@.4.0", "opencv_java"}),
        @Platform(value = "ios", preload = "libopencv_java"),
        @Platform(value = "windows", preload = {"opencv_cuda400", "opencv_cudaarithm400", "opencv_cudafilters400", "opencv_cudaimgproc400", "opencv_java"}),
    }
)
public class opencv_java {
    static { Loader.load(); }
}
