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

package org.bytedeco.ale.presets;

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
@Properties(
    value = {
        @Platform(value = {"linux-x86", "macosx", "windows"}, compiler = "cpp11", define = "UNIQUE_PTR_NAMESPACE std", link = "ale",
            include = {"emucore/m6502/src/bspf/src/bspf.hxx", "emucore/m6502/src/Device.hxx", "emucore/Control.hxx", "emucore/Event.hxx",
                       "emucore/Random.hxx", "common/Constants.h", "common/Array.hxx", "common/display_screen.h", "emucore/M6532.hxx",
                       "emucore/Cart.hxx", "emucore/Console.hxx", "emucore/Sound.hxx", "emucore/Settings.hxx", "emucore/OSystem.hxx",
                       "common/ColourPalette.hpp", "common/ScreenExporter.hpp", "environment/ale_ram.hpp", "environment/ale_screen.hpp",
                       "environment/ale_state.hpp", "environment/stella_environment_wrapper.hpp", "environment/stella_environment.hpp", "ale_interface.hpp"}),
        @Platform(value = "linux-x86",     preload = "SDL-1.2@.0", preloadpath = {"/usr/lib32/", "/usr/lib/"}),
        @Platform(value = "linux-x86_64",  preload = "SDL-1.2@.0", preloadpath = {"/usr/lib64/", "/usr/lib/"}),
        @Platform(value = "macosx-x86_64", preload = "SDL-1.2@.0", preloadpath = "/usr/local/lib/"),
        @Platform(value = "windows-x86",    preload = {"SDL", "libale"}, preloadpath = "/mingw32/bin/"),
        @Platform(value = "windows-x86_64", preload = {"SDL", "libale"}, preloadpath = "/mingw64/bin")},
    target = "org.bytedeco.ale", global = "org.bytedeco.ale.global.ale")
public class ale implements InfoMapper {
    public void map(InfoMap infoMap) {
        infoMap.put(new Info("DEBUGGER_SUPPORT", "CHEATCODE_SUPPORT").define(false))
               .put(new Info("BSPF_strcasecmp", "BSPF_strncasecmp", "BSPF_snprintf", "BSPF_vsnprintf").cppTypes())
               .put(new Info("Common::Array<Resolution>").pointerTypes("ResolutionList").define())
               .put(new Info("StellaEnvironmentWrapper::m_environment").javaText("public native @MemberGetter @ByRef StellaEnvironment m_environment();"))
               .put(new Info("StellaEnvironment::getWrapper").javaText("public native @Name(\"getWrapper().get\") StellaEnvironmentWrapper getWrapper();"))
               .put(new Info("ALEInterface::theOSystem").javaText("public native @Name(\"theOSystem.get\") OSystem theOSystem();"))
               .put(new Info("ALEInterface::theSettings").javaText("public native @Name(\"theSettings.get\") Settings theSettings();"))
               .put(new Info("ALEInterface::romSettings").javaText("public native @Name(\"romSettings.get\") RomSettings romSettings();"))
               .put(new Info("ALEInterface::environment").javaText("public native @Name(\"environment.get\") StellaEnvironment environment();"))
               .put(new Info("AtariVox", "Common::Array<Resolution>::contains", "ALEState::reset", "CheatManager", "CommandMenu", "Debugger",
                             "GameController", "Launcher", "Menu", "Properties", "PropertiesSet", "VideoDialog").skip());
    }
}
