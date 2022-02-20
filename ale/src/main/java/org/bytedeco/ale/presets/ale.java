/*
 * Copyright (C) 2017-2022 Samuel Audet
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

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Platform;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.javacpp.presets.javacpp;
import org.bytedeco.javacpp.tools.Info;
import org.bytedeco.javacpp.tools.InfoMap;
import org.bytedeco.javacpp.tools.InfoMapper;

/**
 *
 * @author Samuel Audet
 */
@Properties(inherit = javacpp.class,
    value = {
        @Platform(value = {"linux-x86", "macosx", "windows"}, compiler = "cpp17", define = "UNIQUE_PTR_NAMESPACE std", link = "ale",
            include = {"emucore/Device.hxx", "emucore/Control.hxx", "emucore/Event.hxx", "emucore/Random.hxx", "common/Constants.h", "emucore/M6532.hxx",
                       "emucore/Cart.hxx", "emucore/Console.hxx", "emucore/Screen.hxx", "emucore/Sound.hxx", "emucore/Settings.hxx", "emucore/OSystem.hxx",
                       "emucore/Props.hxx", "emucore/PropsSet.hxx", "common/ColourPalette.hpp", "common/ScreenExporter.hpp", "environment/ale_ram.hpp",
                       "environment/ale_screen.hpp", "environment/ale_state.hpp", "environment/stella_environment_wrapper.hpp", "environment/stella_environment.hpp",
                       "ale_interface.hpp"}),
        @Platform(value = "linux-x86",     preload = "SDL2-2.0@.0", preloadpath = {"/usr/lib32/", "/usr/lib/"}),
        @Platform(value = "linux-x86_64",  preload = "SDL2-2.0@.0", preloadpath = {"/usr/lib64/", "/usr/lib/"}),
        @Platform(value = "macosx-x86_64", preload = "SDL2-2.0@.0", preloadpath = "/usr/local/lib/"),
        @Platform(value = "windows",        preload = {"libwinpthread-1", "libgcc_s_dw2-1", "libgcc_s_seh-1", "libgomp-1", "libstdc++-6", "SDL2", "zlib1", "libale"}),
        @Platform(value = "windows-x86",    preloadpath = "C:/msys64/mingw32/bin/"),
        @Platform(value = "windows-x86_64", preloadpath = "C:/msys64/mingw64/bin/")},
    target = "org.bytedeco.ale", global = "org.bytedeco.ale.global.ale")
public class ale implements InfoMapper {
    static { Loader.checkVersion("org.bytedeco", "ale"); }

    public void map(InfoMap infoMap) {
        infoMap.put(new Info("DEBUGGER_SUPPORT", "CHEATCODE_SUPPORT").define(false))
               .put(new Info("BSPF_strcasecmp", "BSPF_strncasecmp", "BSPF_snprintf", "BSPF_vsnprintf").cppTypes())
               .put(new Info("Common::Array<Resolution>").pointerTypes("ResolutionList").define())
               .put(new Info("fs::path").annotations("@StdString").valueTypes("@Cast(\"const char*\") BytePointer", "String").pointerTypes("BytePointer"))
               .put(new Info("std::size_t").cast().valueTypes("long").pointerTypes("SizeTPointer"))
               .put(new Info("std::optional<std::string>").pointerTypes("StringOptional").define())
               .put(new Info("ale::stella::Properties").pointerTypes("StellaProperties"))
               .put(new Info("ale::stella::PropertiesSet").pointerTypes("StellaPropertiesSet"))
               .put(new Info("ale::StellaEnvironmentWrapper::m_environment").javaText("public native @MemberGetter @ByRef StellaEnvironment m_environment();"))
               .put(new Info("ale::StellaEnvironment::getWrapper").javaText("public native @Name(\"getWrapper().get\") StellaEnvironmentWrapper getWrapper();"))
               .put(new Info("ale::ALEInterface::theOSystem").javaText("public native @Name(\"theOSystem.get\") OSystem theOSystem();"))
               .put(new Info("ale::ALEInterface::theSettings").javaText("public native @Name(\"theSettings.get\") Settings theSettings();"))
               .put(new Info("ale::ALEInterface::romSettings").javaText("public native @Name(\"romSettings.get\") RomSettings romSettings();"))
               .put(new Info("ale::ALEInterface::environment").javaText("public native @Name(\"environment.get\") StellaEnvironment environment();"))
               .put(new Info("AtariVox", "Common::Array<Resolution>::contains", "ale::ALEInterface::disableBufferedIO", "ale::ALEState::reset",
                             "CheatManager", "CommandMenu", "Debugger", "GameController", "Launcher", "Menu", "Properties", "PropertiesSet", "VideoDialog").skip());
    }
}
