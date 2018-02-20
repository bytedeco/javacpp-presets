/*
 * Copyright (C) 2014-2017 Samuel Audet, Bram Biesbrouck
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
 * Wrapper for Allied Vision Vimba library (the C++ API).
 *
 * @author Bram Biesbrouck
 */
@Properties(target = "org.bytedeco.javacpp.TISCamera", value = {
                @Platform(value = { "linux-x86_64" },
                                compiler = "cpp11",
                                define = { "SHARED_PTR_NAMESPACE std",
                                           //"NO_JNI_DETACH_THREAD"
                                },
                                include = {
                                                //                                                "algorithms/bayer.h",
                                                //                                                "algorithms/auto_focus.h",
                                                //                                                "algorithms/debayer.h",
                                                //                                                "algorithms/image_sampling.h",
                                                //                                                "algorithms/image_transform_base.h",
                                                //                                                "algorithms/tcam-algorithm.h",
                                                //                                                "algorithms/whitebalance.h",
                                                //                                                "algorithms/AutoFocus.h",

                                                //                                                "gstreamer-1.0/gsttcamautoexposure.h",
                                                //                                                "gstreamer-1.0/gsttcamautofocus.h",
                                                //                                                "gstreamer-1.0/gsttcambin.h",
                                                //                                                "gstreamer-1.0/gsttcamsrc.h",
                                                //                                                "gstreamer-1.0/gsttcamwhitebalance.h",
                                                //                                                "gstreamer-1.0/tcamgstbase.h",
                                                //                                                "gstreamer-1.0/tcamgststrings.h",

                                                "property_identifications.h",
                                                "base_types.h",
                                                //                                                "base.h",
                                                //                                                "compiler_defines.h",
                                                //"devicelibrary.h",
                                                //                                                "format.h",
                                                //                                                "image_base_defines.h",
                                                //                                                "image_fourcc.h",
                                                //                                                "image_transform_base.h",
                                                //                                                //"internal.h",
                                                //                                                "logging.h",
                                                "public_utils.h",
                                                //                                                //                                                "serialization.h",
                                                //                                                "standard_properties.h",
                                                //                                                //"tcam.h",
                                                //                                                //                                                "tcam-semaphores.h",
                                                //                                                "utils.h",
                                                //                                                "version.h",

                                                "PropertyImpl.h",
                                                "Property.h",
                                                "DeviceInfo.h",
                                                "BackendLoader.h",
                                                "VideoFormat.h",
                                                "VideoFormatDescription.h",
                                                "MemoryBuffer.h",
                                                "SinkInterface.h",
                                                "DeviceInterface.h",
                                                "DeviceInterface.cpp",
                                                "BackendLoader.h",
                                                "BackendLoader.cpp",
                                                "CaptureDevice.h",
                                                "ImageSink.h",
                                                //"ImageSource.h",
                                                //                                                "CaptureDeviceImpl.h",
                                                //                                                "DeviceIndex.h",
                                                //                                                "FilterBase.h",
                                                "FormatHandlerInterface.h",
                                                //"PipelineManager.h",
                                                //                                                "Properties.h",
                                                //                                                "PropertyGeneration.h",
                                                //                                                "PropertyHandler.h",

                                //#include "UsbSession.h"
                                //#include "LibusbDevice.h"

//                                                "libusb/AFU420Device.h",
//                                                "libusb/AFU420Device.cpp",
                                },
                                link = { "tcam", "tcam-libusb", "tcamprop", "tcamgstbase", /*"gsttcamautoexposure", "gsttcamautofocus", "gsttcamwhitebalance", "gsttcamsrc", "gsttcambin", "tcam-v4l2",*/ /*"tcam-libusb"*/ }
                )
}
                //,helper = "org.bytedeco.javacpp.helper.GObject"
)
public class TISCamera implements InfoMapper
{
    public void map(InfoMap infoMap)
    {
        infoMap.put(new Info("std::map<std::string,int>").pointerTypes("StringIntMap").define());
        infoMap.put(new Info("std::vector<tcam::Property*>").pointerTypes("PropertyVector").define());

        infoMap.put(new Info("std::shared_ptr<tcam::MemoryBuffer>").annotations("@SharedPtr").pointerTypes("MemoryBuffer"));
        infoMap.put(new Info("std::vector<std::shared_ptr<tcam::MemoryBuffer> >").pointerTypes("SharedMemoryBufferVector").define());

        infoMap.put(new Info("std::shared_ptr<tcam::Property>").annotations("@SharedPtr").pointerTypes("Property"));
        infoMap.put(new Info("std::vector<std::shared_ptr<tcam::Property> >").pointerTypes("SharedPropertyVector").define());

        infoMap.put(new Info("tcam::SinkInterface::set_source").skip());
        infoMap.put(new Info("tcam::ImageSink::set_source").skip());

        //infoMap.put(new Info("tcam::SinkInterface::get_buffer_collection").skip());
        infoMap.put(new Info("SinkInterface.h").linePatterns(".*get_buffer_collection.*").skip());
        //infoMap.put(new Info("tcam::SinkInterface::set_buffer_collection").skip());
        infoMap.put(new Info("SinkInterface.h").linePatterns(".*set_buffer_collection.*").skip());
        infoMap.put(new Info("tcam::ImageSink::get_buffer_collection").skip());
        infoMap.put(new Info("tcam::ImageSink::set_buffer_collection").skip());

        infoMap.put(new Info("tcam::DeviceInterface::requeue_buffer()").skip());
        infoMap.put(new Info("tcam::DeviceInterface::requeue_buffer_ptr").javaText("public native void requeue_buffer_ptr(@ByPtr MemoryBuffer arg0);"));

        //we'll define our own callbacks, see below
        infoMap.put(new Info("ImageSink.h").linePatterns(
                        ".*typedef void \\(\\*shared_callback\\).*",
                        ".*typedef void \\(\\*sink_callback\\).*",
                        ".*typedef void \\(\\*c_callback\\).*")
                                           .skip());

        infoMap.put(new Info().javaText("\n" +
                                        "public static class sink_callback extends FunctionPointer {\n" +
                                        "    static { Loader.load(); }\n" +
                                        "    public    sink_callback(Pointer p) { super(p); }\n" +
                                        "    protected sink_callback() { allocate(); }\n" +
                                        "    private native void allocate();\n" +
                                        //see the ImageSink C++ code, the callback is called with a pointer ref: this->callback(&*buffer, user_data);
                                        "    public native void call(@ByRef @Cast(\"tcam::MemoryBuffer*\") MemoryBuffer arg0, Pointer arg1);\n" +
                                        "}").define());

        infoMap.put(new Info("tcam::ImageSink::registerCallback(sink_callback, void*)")
                                    .javaText("public native @Cast(\"bool\") boolean registerCallback(@Cast(\"sink_callback\") sink_callback arg0, Pointer arg1);"));
        infoMap.put(new Info("tcam::ImageSink::registerCallback(shared_callback, void*)").skip());
        infoMap.put(new Info("tcam::ImageSink::registerCallback(c_callback, void*)").skip());
    }
}
