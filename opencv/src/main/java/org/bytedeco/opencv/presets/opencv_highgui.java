/*
 * Copyright (C) 2013-2022 Samuel Audet
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
    inherit = opencv_videoio.class,
    value = {
        @Platform(include = {"<opencv2/highgui/highgui_c.h>", "<opencv2/highgui.hpp>"}, link = "opencv_highgui@.409"),
        @Platform(value = "ios", preload = "libopencv_highgui"),
        @Platform(value = "windows", link = "opencv_highgui490")},
    target = "org.bytedeco.opencv.opencv_highgui",
    global = "org.bytedeco.opencv.global.opencv_highgui"
)
public class opencv_highgui implements InfoMapper {
    public void map(InfoMap infoMap) {
        infoMap.remove("std::vector<int>"); // disable hack from opencv_videoio.class
        infoMap.put(new Info("defined _WIN32").define(false))
               .put(new Info("cvFontQt").annotations("@Platform(\"linux\")").javaNames("cvFontQt"))
               .put(new Info("cvAddText").annotations("@Platform(\"linux\")").javaNames("cvAddText"))
               .put(new Info("cvDisplayOverlay").annotations("@Platform(\"linux\")").javaNames("cvDisplayOverlay"))
               .put(new Info("cvDisplayStatusBar").annotations("@Platform(\"linux\")").javaNames("cvDisplayStatusBar"))
               .put(new Info("cvSaveWindowParameters").annotations("@Platform(\"linux\")").javaNames("cvSaveWindowParameters"))
               .put(new Info("cvLoadWindowParameters").annotations("@Platform(\"linux\")").javaNames("cvLoadWindowParameters"))
               .put(new Info("cvStartLoop").annotations("@Platform(\"linux\")").javaNames("cvStartLoop"))
               .put(new Info("cvStopLoop").annotations("@Platform(\"linux\")").javaNames("cvStopLoop"))
               .put(new Info("cvCreateButton").annotations("@Platform(\"linux\")").javaNames("cvCreateButton"))
               .put(new Info("cvvInitSystem").cppTypes("int", "int", "char**"))
               .put(new Info("cvvNamedWindow").cppTypes("void", "const char*", "int"))
               .put(new Info("cvvShowImage").cppTypes("void", "const char*", "CvArr*"))
               .put(new Info("cvvResizeWindow").cppTypes("void", "const char*", "int", "int"))
               .put(new Info("cvvDestroyWindow").cppTypes("void", "const char*"))
               .put(new Info("cvvCreateTrackbar").cppTypes("int", "const char*", "const char*", "int*", "int", "CvTrackbarCallback"))
               .put(new Info("cvvAddSearchPath", "cvAddSearchPath").cppTypes("void", "const char*"))
               .put(new Info("cvvWaitKey").cppTypes("int", "const char*"))
               .put(new Info("cvvWaitKeyEx").cppTypes("int", "const char*", "int"))
               .put(new Info("set_preprocess_func", "set_postprocess_func").cppTypes());
    }
}
