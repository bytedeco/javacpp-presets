/*
 * Copyright (C) 2013-2020 Samuel Audet
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

package org.bytedeco.videoinput.presets;

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
@Properties(inherit=javacpp.class, target="org.bytedeco.videoinput", global="org.bytedeco.videoinput.global.videoInputLib", value={
    @Platform(value="windows", include={"<videoInput.h>", "<videoInput.cpp>"}, link={"ole32", "oleaut32", "amstrmid", "strmiids", "uuid"}) })
public class videoInputLib implements InfoMapper {
    static { Loader.checkVersion("org.bytedeco", "videoinput"); }

    public void map(InfoMap infoMap) {
          infoMap.put(new Info("videoInput.cpp").skip())
                 .put(new Info("_WIN32_WINNT").cppTypes().define(false))
                 .put(new Info("std::vector<std::string>").pointerTypes("StringVector").define())
                 .put(new Info("GUID").cast().pointerTypes("Pointer"))
                 .put(new Info("long", "unsigned long").cast().valueTypes("int").pointerTypes("IntPointer", "IntBuffer", "int[]"));
    }
}
