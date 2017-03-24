/*
 * Copyright (C) 2014 Samuel Audet, Jarek Sacha
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

package org.bytedeco.javacpp.presets;

import org.bytedeco.javacpp.annotation.Platform;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.javacpp.tools.Info;
import org.bytedeco.javacpp.tools.InfoMap;
import org.bytedeco.javacpp.tools.InfoMapper;

/**
 * Wrapper for Point Grey FlyCapture2 library (the C++ API v.2).
 *
 * @author Jarek Sacha
 */
@Properties(target = "org.bytedeco.javacpp.FlyCapture2", value = {
        @Platform(value = {"linux-x86", "linux-arm", "windows"}, include = {"<FlyCapture2Platform.h>", "<FlyCapture2Defs.h>",
                "<Error.h>", "<BusManager.h>", "<CameraBase.h>", "<Camera.h>", "<GigECamera.h>", "<Image.h>",
                "<Utilities.h>", "<AVIRecorder.h>", "<TopologyNode.h>", "<ImageStatistics.h>",
                "<MultiSyncLibraryPlatform.h>", "<MultiSyncLibraryDefs.h>", "<MultiSyncLibrary.h>"}),
        @Platform(value = {"linux-x86", "linux-arm"}, link = "flycapture@.2", includepath = "/usr/include/flycapture/"),
        @Platform(value = "windows", link = {"FlyCapture2", "MultiSyncLibrary"},
                includepath = "C:/Program Files/Point Grey Research/FlyCapture2/include/"),
        @Platform(value = "windows-x86",
                linkpath    = {"C:/Program Files/Point Grey Research/FlyCapture2/lib/",
                               "C:/Program Files (x86)/Point Grey Research/FlyCapture2/lib/"},
                preloadpath = {"C:/Program Files/Point Grey Research/FlyCapture2/bin/",
                               "C:/Program Files (x86)/Point Grey Research/FlyCapture2/bin/"}),
        @Platform(value = "windows-x86_64",
                linkpath    = "C:/Program Files/Point Grey Research/FlyCapture2/lib64/",
                preloadpath = "C:/Program Files/Point Grey Research/FlyCapture2/bin64/") })
public class FlyCapture2 implements InfoMapper {
    public void map(InfoMap infoMap) {
        infoMap.put(new Info("FLYCAPTURE2_API", "FLYCAPTURE2_LOCAL",
                "MULTISYNCLIBRARY_API", "MULTISYNCLIBRARY_LOCAL").cppTypes().annotations().cppText(""))
               .put(new Info("defined(WIN32) || defined(WIN64)").define())
               .put(new Info("FlyCapture2::ImageEventCallback").valueTypes("ImageEventCallback")
                       .pointerTypes("@Cast(\"FlyCapture2::ImageEventCallback*\") @ByPtrPtr ImageEventCallback"))
               .put(new Info("FlyCapture2::CameraBase::GetRegisterString", "FlyCapture2::CameraBase::StartSyncCapture",
                             "FlyCapture2::TopologyNode::AddPort").skip());
    }
}
