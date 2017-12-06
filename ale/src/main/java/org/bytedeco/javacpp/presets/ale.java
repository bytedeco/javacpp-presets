/*
 * Copyright (C) 2017 Samuel Audet
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

import java.nio.ByteBuffer;
import org.bytedeco.javacpp.BytePointer;
import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.annotation.Cast;
import org.bytedeco.javacpp.annotation.Platform;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.javacpp.tools.Info;
import org.bytedeco.javacpp.tools.InfoMap;
import org.bytedeco.javacpp.tools.InfoMapper;

/**
 *
 * @author Samuel Audet
 */
@Properties(value = @Platform(value = {"linux-x86", "macosx"}, include = {"emucore/m6502/src/bspf/src/bspf.hxx", "emucore/Event.hxx",
    "common/Constants.h", "common/ColourPalette.hpp", "common/ScreenExporter.hpp", "environment/ale_ram.hpp", "environment/ale_screen.hpp",
    "environment/ale_state.hpp", "ale_interface.hpp"}, link = "ale"), target = "org.bytedeco.javacpp.ale")
public class ale implements InfoMapper {
    public void map(InfoMap infoMap) {
        infoMap.put(new Info("BSPF_strcasecmp", "BSPF_strncasecmp", "BSPF_snprintf", "BSPF_vsnprintf").cppTypes())
               .put(new Info("ALEInterface::theOSystem", "ALEInterface::theSettings", "ALEInterface::romSettings",
                             "ALEInterface::environment", "ALEInterface::createOSystem", "ALEInterface::loadSettings").skip());
    }
}
