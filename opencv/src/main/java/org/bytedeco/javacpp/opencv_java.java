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

package org.bytedeco.javacpp;

import org.bytedeco.javacpp.annotation.Platform;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.javacpp.tools.Info;
import org.bytedeco.javacpp.tools.InfoMap;
import org.bytedeco.javacpp.tools.InfoMapper;

/**
 * This is only a placeholder to facilitate loading the {@code opencv_java} module with JavaCPP.
 * <p>
 * Call {@code Loader.load(opencv_java.class)} before using the API in the {@code org.opencv} namespace.
 *
 * @author Samuel Audet
 */
@Properties(
    inherit = {
        opencv_aruco.class,
        opencv_bgsegm.class,
        opencv_bioinspired.class,
        opencv_face.class,
        opencv_img_hash.class,
        opencv_structured_light.class,
        opencv_text.class,
        opencv_tracking.class,
        opencv_xfeatures2d.class,
        opencv_ximgproc.class,
        opencv_xphoto.class,
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
