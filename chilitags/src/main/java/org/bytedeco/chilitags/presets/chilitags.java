/*
 * Copyright (C) 2015 Samuel Audet
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

package org.bytedeco.chilitags.presets;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Platform;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.javacpp.tools.Info;
import org.bytedeco.javacpp.tools.InfoMap;
import org.bytedeco.javacpp.tools.InfoMapper;

import org.bytedeco.opencv.presets.opencv_calib3d;
import org.bytedeco.opencv.presets.opencv_video;

/**
 * @author Samuel Audet
 */
@Properties(value = @Platform(define = "CHILITAGS_STATIC_DEFINE", include = "chilitags/chilitags.hpp",
                              link = "chilitags_static"/*, resource = {"include", "lib"}*/, compiler = "cpp11"),
    inherit = {opencv_calib3d.class, opencv_video.class}, target = "org.bytedeco.chilitags",
                                                          global = "org.bytedeco.chilitags.global.chilitags")
public class chilitags implements InfoMapper {
    static { Loader.checkVersion("org.bytedeco", "chilitags"); }

    public void map(InfoMap infoMap) {
        infoMap.put(new Info().javaText("import org.bytedeco.javacpp.annotation.Index;"))

               .put(new Info("chilitags::Quad").annotations("@ByRef @Cast(\"chilitags::Quad*\")").pointerTypes("FloatPointer"))
               .put(new Info("std::map<int,chilitags::Quad>").pointerTypes("TagCornerMap").define())

               .put(new Info("chilitags::Chilitags3D_<float>").pointerTypes("Chilitags3D"))
               .put(new Info("chilitags::Chilitags3D_<float>::TransformMatrix").
                       annotations("@ByRef @Cast(\"chilitags::Chilitags3D_<float>::TransformMatrix*\")").pointerTypes("FloatPointer"))
               .put(new Info("std::map<std::string,chilitags::Chilitags3D_<float>::TransformMatrix>").pointerTypes("TagPoseMap").define())

               .put(new Info("chilitags::Chilitags3D_<double>").pointerTypes("Chilitags3Dd"))
               .put(new Info("chilitags::Chilitags3D_<double>::TransformMatrix").
                       annotations("@ByRef @Cast(\"chilitags::Chilitags3D_<double>::TransformMatrix*\")").pointerTypes("DoublePointer"))
               .put(new Info("std::map<std::string,chilitags::Chilitags3D_<double>::TransformMatrix>").pointerTypes("TagPoseMapd").define());
    }
}
