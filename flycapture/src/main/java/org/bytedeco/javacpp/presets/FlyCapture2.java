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
 * Wrapper for Point Grey FlyCapture2 library (the C++ API v.2).
 *
 * @author Jarek Sacha
 */
@Properties(target = "org.bytedeco.javacpp.FlyCapture2", value = {
        @Platform(value = {"linux", "windows"}, include = {"<FlyCapture2Platform.h>", "<FlyCapture2Defs.h>",
                "<Error.h>", "<BusManager.h>", "<CameraBase.h>", "<Camera.h>", "<GigECamera.h>", "<Image.h>",
                "<Utilities.h>", "<AVIRecorder.h>", "<TopologyNode.h>", "<ImageStatistics.h>"}),
        @Platform(value = "linux", link = "flycapture@.2", includepath = "/usr/include/flycapture/"),
        @Platform(value = "windows", link = "FlyCapture2",
                includepath = "C:/Program Files/Point Grey Research/FlyCapture2/include/"),
        @Platform(value = "windows-x86",    define = {"WIN32", "AddPort AddPortA"},
                linkpath    = {"C:/Program Files/Point Grey Research/FlyCapture2/lib/",
                               "C:/Program Files (x86)/Point Grey Research/FlyCapture2/lib/"},
                preloadpath = {"C:/Program Files/Point Grey Research/FlyCapture2/bin/",
                               "C:/Program Files (x86)/Point Grey Research/FlyCapture2/bin/"}),
        @Platform(value = "windows-x86_64", define = {"WIN64", "AddPort AddPortA"},
                linkpath    = "C:/Program Files/Point Grey Research/FlyCapture2/lib64/",
                preloadpath = "C:/Program Files/Point Grey Research/FlyCapture2/bin64/") })
public class FlyCapture2 implements InfoMapper {
    public void map(InfoMap infoMap) {
        infoMap.put(new Info("FLYCAPTURE2_API", "FLYCAPTURE2_LOCAL").cppTypes().annotations().cppText(""))
               .put(new Info("defined(WIN32) || defined(WIN64)").define())
               .put(new Info("FlyCapture2::ImageEventCallback").valueTypes("ImageEventCallback")
                       .pointerTypes("@Cast(\"FlyCapture2::ImageEventCallback*\") @ByPtrPtr ImageEventCallback"))
               .put(new Info("FlyCapture2::CameraBase::GetRegisterString", "FlyCapture2::CameraBase::StartSyncCapture").skip());
    }
}
