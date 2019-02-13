/*
 * Copyright (C) 2014-2017 Samuel Audet, Jarek Sacha
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
 *
 *
 * Permission was received from Point Grey Research, Inc. to disclose the
 * information released by the application of this preset under the GPL,
 * as long as it is distributed as part of a substantially larger package,
 * and not as a standalone wrapper to the FlyCapture library.
 *
 */

package org.bytedeco.flycapture.presets;

import org.bytedeco.javacpp.annotation.Platform;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.javacpp.tools.Info;
import org.bytedeco.javacpp.tools.InfoMap;
import org.bytedeco.javacpp.tools.InfoMapper;

/**
 * Wrapper for Point Grey FlyCapture2_C library (the C API v.2).
 *
 * @author Jarek Sacha
 */
@Properties(inherit = FlyCapture2.class,
            target = "org.bytedeco.flycapture.FlyCapture2_C",
            global = "org.bytedeco.flycapture.global.FlyCapture2_C", value = {
        @Platform(value = {"linux-x86", "linux-arm", "windows"},
                include = {"<FlyCapture2Defs_C.h>", "<FlyCapture2_C.h>",
                        "MultiSyncLibraryDefs_C.h", "MultiSyncLibrary_C.h"},
                link = {"flycapture-c@.2", "multisync-c@.2", "flycapturevideo-c@.2"},
                includepath = "/usr/include/flycapture/C/"),
        @Platform(value = "linux-arm",
                include = {"<FlyCapture2Defs_C.h>", "<FlyCapture2_C.h>"},
                link = {"flycapture-c@.2", "flycapturevideo-c@.2"}),
        @Platform(value = "windows",
                link = {"FlyCapture2_C_v140", "MultiSyncLibrary_C_v140", "FlyCapture2Video_C_v140"},
                preload = {"libiomp5md", "FlyCapture2"},
                includepath = {"C:/Program Files/Point Grey Research/FlyCapture2/include/C/",
                               "C:/Program Files (x86)/Point Grey Research/FlyCapture2/include/C/"}),
        @Platform(value = "windows-x86",
                linkpath    = {"C:/Program Files/Point Grey Research/FlyCapture2/lib/vs2015/C/",
                               "C:/Program Files (x86)/Point Grey Research/FlyCapture2/lib/vs2015/C/"},
                preloadpath = {"C:/Program Files/Point Grey Research/FlyCapture2/bin/vs2015/",
                               "C:/Program Files/Point Grey Research/FlyCapture2/bin/vs2015/C/",
                               "C:/Program Files (x86)/Point Grey Research/FlyCapture2/bin/vs2015/",
                               "C:/Program Files (x86)/Point Grey Research/FlyCapture2/bin/vs2015/C/"}),
        @Platform(value = "windows-x86_64",
                linkpath    =  "C:/Program Files/Point Grey Research/FlyCapture2/lib64/vs2015/C/",
                preloadpath = {"C:/Program Files/Point Grey Research/FlyCapture2/bin64/vs2015/",
                               "C:/Program Files/Point Grey Research/FlyCapture2/bin64/vs2015/C/"}) })
public class FlyCapture2_C implements InfoMapper {
    public void map(InfoMap infoMap) {
        infoMap.put(new Info("FLYCAPTURE2_C_API", "FLYCAPTURE2_C_CALL_CONVEN",
                "MULTISYNCLIBRARY_C_API", "MULTISYNCLIBRARY_C_CALL_CONVEN").cppTypes().annotations().cppText(""))
               .put(new Info("fc2TriggerDelayInfo").cast().pointerTypes("fc2PropertyInfo"))
               .put(new Info("fc2TriggerDelay").cast().pointerTypes("fc2Property"))
               .put(new Info("fc2ImageEventCallback").valueTypes("fc2ImageEventCallback")
                       .pointerTypes("@Cast(\"fc2ImageEventCallback*\") @ByPtrPtr fc2ImageEventCallback"))
               .put(new Info("fc2Context").valueTypes("fc2Context")
                       .pointerTypes("@Cast(\"fc2Context*\") @ByPtrPtr fc2Context"))
                // To avoid linking error on Windows 64: "unresolved external symbol __imp_ResetStats"
               .put(new Info("ResetStats").skip());
    }
}
