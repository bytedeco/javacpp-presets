/*
 * Copyright (C) 2018-2019 Samuel Audet
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

package org.bytedeco.arpackng.presets;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.NoException;
import org.bytedeco.javacpp.annotation.Platform;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.javacpp.tools.Info;
import org.bytedeco.javacpp.tools.InfoMap;
import org.bytedeco.javacpp.tools.InfoMapper;

import org.bytedeco.openblas.presets.openblas;

/**
 *
 * @author Samuel Audet
 */
@Properties(
    inherit = openblas.class,
    value = @Platform(
        include = {"arpack/arpackdef.h", "arpack/arpack.h", "arpack/arpack.hpp", "arpack/debug_c.hpp", "arpack/stat_c.hpp"},
        link = "arpack@.2",
        preload = "libarpack-2"),
    global = "org.bytedeco.arpackng.global.arpack")
@NoException
public class arpack implements InfoMapper {
    static { Loader.checkVersion("org.bytedeco", "arpack-ng"); }

    public void map(InfoMap infoMap) {
        infoMap.put(new Info("INTERFACE64").define(false))
               .put(new Info("a_int", "a_uint").cppTypes())
               .put(new Info("const char").cast().valueTypes("byte").pointerTypes("BytePointer", "ByteBuffer", "byte[]"));
    }
}
