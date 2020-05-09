/*
 * Copyright (C) 2020 Dmytro Maksutov
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

package org.bytedeco.hyperscan.presets;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Platform;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.javacpp.presets.javacpp;
import org.bytedeco.javacpp.tools.Info;
import org.bytedeco.javacpp.tools.InfoMap;
import org.bytedeco.javacpp.tools.InfoMapper;

@Properties(
    inherit = javacpp.class,
    value = {
        @Platform(
            value = {"linux-x86_64", "macosx-x86_64", "windows-x86_64"},
            compiler = "cpp11",
            include = {"hs/hs_common.h", "hs/hs_compile.h", "hs/hs_runtime.h", "hs/hs.h"},
            link = {"hs@.5", "hs_runtime@.5"}
        )
    },
    target = "org.bytedeco.hyperscan",
    global = "org.bytedeco.hyperscan.global.hyperscan"
)
public class hyperscan implements InfoMapper {
    static { Loader.checkVersion("org.bytedeco", "hyperscan"); }

    public void map(InfoMap infoMap) {
        infoMap.put(new Info("HS_CDECL").cppTypes().annotations());
    }
}
