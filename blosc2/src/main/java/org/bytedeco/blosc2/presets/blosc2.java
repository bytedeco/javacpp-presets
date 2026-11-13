/*
 * Copyright (C) 2022 Mark Kittisopikul
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

package org.bytedeco.blosc2.presets;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Platform;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.javacpp.presets.javacpp;
import org.bytedeco.javacpp.tools.Info;
import org.bytedeco.javacpp.tools.InfoMap;
import org.bytedeco.javacpp.tools.InfoMapper;

/**
 * Wrapper for the C-Blosc2 compression library.
 * <p>
 * C-Blosc2 reads and writes Blosc1-compatible chunks through {@code blosc1_compress()} and
 * {@code blosc1_decompress()}, as well as native Blosc2 chunks, super-chunks, frames and N-dimensional arrays.
 *
 * @see <a href="https://github.com/Blosc/c-blosc2">https://github.com/Blosc/c-blosc2</a>
 * @author Mark Kittisopikul
 */
@Properties(inherit = javacpp.class, value = {
    @Platform(include = {"blosc2.h", "blosc2/blosc2-stdio.h", "b2nd.h"},
              link = "blosc2@.10"),
    // the import library is blosc2.lib, but the DLL is named libblosc2.dll
    @Platform(value = "windows", link = "blosc2", preload = "libblosc2")},
    target = "org.bytedeco.blosc2", global = "org.bytedeco.blosc2.global.blosc2")
public class blosc2 implements InfoMapper {
    static { Loader.checkVersion("org.bytedeco", "blosc2"); }

    public void map(InfoMap infoMap) {
        infoMap.put(new Info("BLOSC_EXPORT", "BLOSC_NO_EXPORT", "BLOSC_DEPRECATED", "BLOSC_PRIVATE", "B2ND_DEPRECATED").cppTypes().annotations())
               .put(new Info("BLOSC_ATTRIBUTE_UNUSED", "print_error").skip())
               .put(new Info("BLOSC1_COMPAT", "_WIN32", "defined(_WIN32)", "defined(_WIN32) && !defined(__MINGW32__)", "defined(_MSC_VER)", "__cplusplus").define(false))
               .put(new Info("BLOSC2_MAX_BUFFERSIZE").javaText("public static final int BLOSC2_MAX_BUFFERSIZE = Integer.MAX_VALUE - BLOSC2_MAX_OVERHEAD;"))
               .put(new Info("BLOSC_MAX_TYPESIZE").javaText("public static final int BLOSC_MAX_TYPESIZE = 255;"))
               .put(new Info("BLOSC2_VERSION_STRING", "BLOSC2_VERSION_DATE").cppTypes("const char*"));
    }
}
