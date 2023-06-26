/*
 * Copyright (C) 2021 Jack He, Samuel Audet
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

package org.bytedeco.tritonserver.presets;

import java.util.List;
import org.bytedeco.javacpp.ClassProperties;
import org.bytedeco.javacpp.LoadEnabled;
import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Platform;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.javacpp.tools.Info;
import org.bytedeco.javacpp.tools.InfoMap;
import org.bytedeco.javacpp.tools.InfoMapper;

/**
 *
 * @author Jack He
 */
@Properties(
    value = {
        @Platform(
            value = {"linux-arm64", "linux-ppc64le", "linux-x86_64", "windows-x86_64"},
            include = {"tritonserver.h", "tritonbackend.h", "tritonrepoagent.h"},
            link = "tritonserver",
            includepath = {"/opt/tritonserver/include/triton/core/", "/opt/tritonserver/include/", "/usr/include"},
            linkpath = {"/opt/tritonserver/lib/"}
        ),
        @Platform(
            value = "windows-x86_64",
            includepath = "C:/Program Files/NVIDIA GPU Computing Toolkit/TritonServer/include/triton/core/",
            linkpath = "C:/Program Files/NVIDIA GPU Computing Toolkit/TritonServer/lib/",
            preloadpath = "C:/Program Files/NVIDIA GPU Computing Toolkit/TritonServer/bin/"
        )
    },
    target = "org.bytedeco.tritonserver.tritonserver",
    global = "org.bytedeco.tritonserver.global.tritonserver"
)
public class tritonserver implements InfoMapper {
    static { Loader.checkVersion("org.bytedeco", "tritonserver"); }
    public void map(InfoMap infoMap) {
        infoMap.putFirst(new Info().enumerate(false))
               .put(new Info("bool").cast().valueTypes("boolean").pointerTypes("boolean[]", "BoolPointer"))
               .put(new Info("const char").pointerTypes("String", "@Cast(\"const char*\") BytePointer"))
               .put(new Info("std::size_t").cast().valueTypes("long").pointerTypes("LongPointer", "LongBuffer", "long[]"))
               .put(new Info("TRITONSERVER_EXPORT", "TRITONSERVER_DECLSPEC",
                             "TRITONBACKEND_DECLSPEC", "TRITONBACKEND_ISPEC",
                             "TRITONREPOAGENT_DECLSPEC", "TRITONREPOAGENT_ISPEC").cppTypes().annotations())
        ;
    }
}
