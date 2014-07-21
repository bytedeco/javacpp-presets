/*
 * Copyright (C) 2014 Samuel Audet, Jarek Sacha
 *
 * This file is part of JavaCPP.
 *
 * JavaCPP is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 2 of the License, or
 * (at your option) any later version (subject to the "Classpath" exception
 * as provided in the LICENSE.txt file that accompanied this code).
 *
 * JavaCPP is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with JavaCPP.  If not, see <http://www.gnu.org/licenses/>.
 *
 *
 * Permission was received from Point Grey Research, Inc. to disclose the
 * information released by the application of this preset under the GPL,
 * as long as it is distributed as part of a substantially larger package,
 * and not as a standalone wrapper to the FlyCapture library.
 *
 */

package org.bytedeco.javacpp.presets;

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
@Properties(target = "org.bytedeco.javacpp.FlyCapture2_C", value = {
        @Platform(value = {"linux", "windows"}, include = {"<FlyCapture2Defs_C.h>", "<FlyCapture2_C.h>"}),
        @Platform(value = "linux", link = "flycapture-c@.2", includepath = "/usr/include/flycapture/C/"),
        @Platform(value = "windows", link = "FlyCapture2_C", preload = {"libiomp5md", "FlyCapture2"},
                includepath =  "C:/Program Files/Point Grey Research/FlyCapture2/include/C/"),
        @Platform(value = "windows-x86",
                linkpath    =  "C:/Program Files/Point Grey Research/FlyCapture2/lib/C/",
                preloadpath = {"C:/Program Files/Point Grey Research/FlyCapture2/bin/",
                               "C:/Program Files/Point Grey Research/FlyCapture2/bin/C/"}),
        @Platform(value = "windows-x86_64",
                linkpath    =  "C:/Program Files/Point Grey Research/FlyCapture2/lib64/C/",
                preloadpath = {"C:/Program Files/Point Grey Research/FlyCapture2/bin64/",
                               "C:/Program Files/Point Grey Research/FlyCapture2/bin64/C/"}) })
public class FlyCapture2_C implements InfoMapper {
    public void map(InfoMap infoMap) {
        infoMap.put(new Info("FLYCAPTURE2_C_API", "FLYCAPTURE2_C_CALL_CONVEN").cppTypes().annotations().cppText(""))
               .put(new Info("fc2TriggerDelayInfo").cast().pointerTypes("fc2PropertyInfo"))
               .put(new Info("fc2TriggerDelay").cast().pointerTypes("fc2Property"))
               .put(new Info("fc2ImageEventCallback").valueTypes("fc2ImageEventCallback")
                       .pointerTypes("@Cast(\"fc2ImageEventCallback*\") @ByPtrPtr fc2ImageEventCallback"))
               .put(new Info("fc2Context").valueTypes("fc2Context")
                       .pointerTypes("@Cast(\"fc2Context*\") @ByPtrPtr fc2Context"));
    }
}
