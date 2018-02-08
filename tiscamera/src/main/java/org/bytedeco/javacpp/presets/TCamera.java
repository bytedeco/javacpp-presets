///*
// * Copyright (C) 2014-2017 Samuel Audet, Bram Biesbrouck
// *
// * Licensed either under the Apache License, Version 2.0, or (at your option)
// * under the terms of the GNU General Public License as published by
// * the Free Software Foundation (subject to the "Classpath" exception),
// * either version 2, or any later version (collectively, the "License");
// * you may not use this file except in compliance with the License.
// * You may obtain a copy of the License at
// *
// *     http://www.apache.org/licenses/LICENSE-2.0
// *     http://www.gnu.org/licenses/
// *     http://www.gnu.org/software/classpath/license.html
// *
// * or as provided in the LICENSE.txt file that accompanied this code.
// * Unless required by applicable law or agreed to in writing, software
// * distributed under the License is distributed on an "AS IS" BASIS,
// * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// * See the License for the specific language governing permissions and
// * limitations under the License.
// *
// *
// * Permission was received from Point Grey Research, Inc. to disclose the
// * information released by the application of this preset under the GPL,
// * as long as it is distributed as part of a substantially larger package,
// * and not as a standalone wrapper to the FlyCapture library.
// *
// */
//
//package org.bytedeco.javacpp.presets;
//
//import org.bytedeco.javacpp.annotation.Platform;
//import org.bytedeco.javacpp.annotation.Properties;
//import org.bytedeco.javacpp.tools.Info;
//import org.bytedeco.javacpp.tools.InfoMap;
//import org.bytedeco.javacpp.tools.InfoMapper;
//
///**
// * Wrapper for Allied Vision Vimba library (the C++ API).
// *
// * @author Bram Biesbrouck
// */
//@Properties(target = "org.bytedeco.javacpp.TCamera", value = {
//                @Platform(value = { "linux-x86_64" },
//                                compiler = "cpp11",
//                                define = { "SHARED_PTR_NAMESPACE std" },
//                                include = {
//                                                "property_identifications.h",
//                                                "base_types.h",
//
//                                                "CaptureDevice.h",
//
//                                                "PropertyImpl.h",
//                                                "DeviceInfo.h",
//                                                "DeviceInterface.h",
//                                                "DeviceInterface.cpp",
//                                                "BackendLoader.h",
//                                                "BackendLoader.cpp",
//                                },
//                                link = { "tcam", "tcamprop" /*, "tcamgstbase", "gsttcamautoexposure", "gsttcamautofocus", "gsttcamwhitebalance", "gsttcamsrc", "gsttcambin", "tcam-v4l2", "tcam-libusb"*/ }
//                )
//}
//                //,helper = "org.bytedeco.javacpp.helper.GObject"
//)
//public class TCamera implements InfoMapper
//{
//    public void map(InfoMap infoMap)
//    {
//        infoMap.put(new Info("std::shared_ptr<tcam::Property>").annotations("@SharedPtr").pointerTypes("Property"));
//        infoMap.put(new Info("std::vector<std::shared_ptr<tcam::Property> >").pointerTypes("SharedPropertyVector").define());
//
//        infoMap.put(new Info("DeviceInterface.h").linePatterns(".*video_format.*", ".*sink.*", ".*buffer.*").skip());
//        infoMap.put(new Info("CaptureDevice.h").linePatterns(".*video_format.*", ".*start_stream.*").skip());
//    }
//}
