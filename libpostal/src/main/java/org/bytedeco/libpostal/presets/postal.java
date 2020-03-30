/*
 * Copyright (C) 2018-2020 Maurice Betzel, Samuel Audet
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

package org.bytedeco.libpostal.presets;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Platform;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.javacpp.presets.javacpp;
import org.bytedeco.javacpp.tools.Info;
import org.bytedeco.javacpp.tools.InfoMap;
import org.bytedeco.javacpp.tools.InfoMapper;

/**
 * @author Maurice Betzel, Samuel Audet
 */
@Properties(
    inherit = javacpp.class,
    value = {
        @Platform(
            value = {"linux-x86_64", "macosx-x86_64", "windows-x86_64"},
            cinclude = "libpostal/libpostal.h",
            link = "postal@.1",
            preload = "libpostal-1"
        )
    },
    target = "org.bytedeco.libpostal",
    global = "org.bytedeco.libpostal.global.postal"
)
public class postal implements InfoMapper {
    static { Loader.checkVersion("org.bytedeco", "libpostal"); }

    public void map(InfoMap infoMap) {
        infoMap.put(new Info("LIBPOSTAL_EXPORT").cppTypes().annotations())
               .put(new Info("libpostal_normalized_tokens").skip())
               .put(new Info("char").cast().valueTypes("byte").pointerTypes("BytePointer", "String"));
    }
}
