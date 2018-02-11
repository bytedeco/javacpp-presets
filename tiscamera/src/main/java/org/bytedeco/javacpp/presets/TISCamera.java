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
                                define = { "SHARED_PTR_NAMESPACE std", "NO_JNI_DETACH_THREAD" },
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

        //infoMap.put(new Info("std::shared_ptr<tcam::MemoryBuffer>").annotations("@SharedPtr").pointerTypes("MemoryBuffer"));
        //infoMap.put(new Info("std::shared_ptr<tcam::MemoryBuffer>").pointerTypes("SharedMemoryBuffer").define());
        //infoMap.put(new Info("std::vector<std::shared_ptr<tcam::MemoryBuffer> >").pointerTypes("SharedMemoryBufferVector").define());

        infoMap.put(new Info("std::shared_ptr<tcam::Property>").annotations("@SharedPtr").pointerTypes("Property"));
        infoMap.put(new Info("std::vector<std::shared_ptr<tcam::Property> >").pointerTypes("SharedPropertyVector").define());

        //note: we need that cast
        //infoMap.put(new Info("std::weak_ptr<tcam::SinkInterface>").cast().pointerTypes("SinkInterface"));
        infoMap.put(new Info("tcam::SinkInterface::set_source").skip());
        infoMap.put(new Info("tcam::ImageSink::set_source").skip());

        //infoMap.put(new Info("tcam::SinkInterface::get_buffer_collection").skip());
        infoMap.put(new Info("SinkInterface.h").linePatterns(".*get_buffer_collection.*").skip());
        //infoMap.put(new Info("tcam::SinkInterface::set_buffer_collection").skip());
        infoMap.put(new Info("SinkInterface.h").linePatterns(".*set_buffer_collection.*").skip());
        infoMap.put(new Info("tcam::ImageSink::get_buffer_collection").skip());
        infoMap.put(new Info("tcam::ImageSink::set_buffer_collection").skip());
        infoMap.put(new Info("DeviceInterface.h").linePatterns(".*initialize_buffers .*").skip());
        infoMap.put(new Info("std::vector<tcam::MemoryBuffer*>").pointerTypes("MemoryBufferVector").define());

        //infoMap.put(new Info("tcam::DeviceInterface::requeue_buffer_ptr").javaText("public native void requeue_buffer_ptr(@ByRef MemoryBuffer arg0);"));

        infoMap.put(new Info().javaText("\n" +
                                        "public static class sink_callback extends FunctionPointer {\n" +
                                        "    static { Loader.load(); }\n" +
                                        "    public    sink_callback(Pointer p) { super(p); }\n" +
                                        "    protected sink_callback() { allocate(); }\n" +
                                        "    private native void allocate();\n" +
                                        "    public native void call(@ByPtr @Cast(\"tcam::MemoryBuffer*\") MemoryBuffer arg0, Pointer arg1);\n" +
                                        "}").define());
        infoMap.put(new Info("ImageSink.h").linePatterns(".*typedef void \\(\\*sink_callback\\)\\(tcam::MemoryBuffer\\*, void\\*\\).*").skip());

        infoMap.put(new Info("tcam::ImageSink::registerCallback")
                                    .javaText("public native @Cast(\"bool\") boolean registerCallback(@Cast(\"sink_callback\") sink_callback arg0, Pointer arg1);"));
                infoMap.put(new Info("tcam::ImageSink::registerCallback").skip());


        //std::shared_ptr<tcam::MemoryBuffer>

        //.annotations("@Cast({\"std::shared_ptr<tcam::MemoryBuffer>\", \"\"}) @SharedPtr")
        //this one seems to be a difficult one to parse...
//        infoMap.put(new Info("shared_callback").skip());
//        infoMap.put(new Info("tcam::ImageSink::shared_callback").skip());

//        infoMap.put(new Info("sink_callback").skip());
//        infoMap.put(new Info("tcam::ImageSink::sink_callback").skip());
//        infoMap.put(new Info("sink_callback").javaText("public static class sink_callback extends FunctionPointer {\n" +
//                                                       "    static { Loader.load(); }\n" +
//                                                       "    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */\n" +
//                                                       "    public    sink_callback(Pointer p) { super(p); }\n" +
//                                                       "    protected sink_callback() { allocate(); }\n" +
//                                                       "    private native void allocate();\n" +
//                                                       "    public native void call( MemoryBuffer arg0, Pointer arg1);\n" +
//                                                       "}"));
//        infoMap.put(new Info("sink_callback").javaText("public static class sink_callback extends FunctionPointer {\n" +
//                                                       "    static { Loader.load(); }\n" +
//                                                       "    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */\n" +
//                                                       "    public    sink_callback(Pointer p) { super(p); }\n" +
//                                                       "    protected sink_callback() { allocate(); }\n" +
//                                                       "    private native void allocate();\n" +
//                                                       "    public native void call(MemoryBuffer arg0, Pointer arg1);\n" +
//                                                       "}").define());
//
        //public native @Cast("bool") boolean registerCallback(sink_callback arg0, Pointer arg1);
//        infoMap.put(new Info("tcam::ImageSink::sink_callback").javaText("public native @Cast(\"bool\") boolean registerCallback(sink_callback arg0, Pointer arg1);"));

//        infoMap.put(new Info("c_callback").skip());
//        infoMap.put(new Info("tcam::ImageSink::c_callback").skip());

        //        infoMap.put(new Info("tcam::ImageSink::sink_callback").skip());
        //        infoMap.put(new Info("tcam::ImageSink::c_callback").skip());

        //infoMap.put(new Info("tcam::ImageSink::shared_callback").pointerTypes("my_shared_callback").cppTypes("void*").translate());
        //infoMap.put(new Info("sink_callback::sink_callback").pointerTypes("SinkCallbackFunction"));
        //infoMap.put(new Info("sink_callback::sink_callback").annotations("@SharedPtr").pointerTypes("MemoryBuffer"));

//        infoMap.put(new Info("tcam::ImageSource")
//                                    .javaText("@Namespace(\"tcam\") @NoOffset public static class ImageSource extends SinkInterface {\n" +
//                                              "    static { Loader.load(); }\n" +
//                                              "    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */\n" +
//                                              "    public ImageSource(Pointer p) { super(p); }\n" +
//                                              "    /** Native array allocator. Access with {@link Pointer#position(long)}. */\n" +
//                                              "    public ImageSource(long size) { super((Pointer)null); allocateArray(size); }\n" +
//                                              "    private native void allocateArray(long size);\n" +
//                                              "    @Override public ImageSource position(long position) {\n" +
//                                              "        return (ImageSource)super.position(position);\n" +
//                                              "    }\n" +
//                                              "    \n" +
//                                              "    public ImageSource() { super((Pointer)null); allocate(); }\n" +
//                                              "    private native void allocate();\n" +
//                                              "\n" +
//                                              "    public native @Cast(\"bool\") boolean set_status(@Cast(\"TCAM_PIPELINE_STATUS\") int arg0);\n" +
//                                              "\n" +
//                                              "    public native @Cast(\"TCAM_PIPELINE_STATUS\") int get_status();\n" +
//                                              "\n" +
//                                              "    public native @Cast(\"bool\") boolean setDevice(@SharedPtr DeviceInterface arg0);\n" +
//                                              "\n" +
//                                              "    public native @Cast(\"bool\") boolean setVideoFormat(@Const @ByRef VideoFormat arg0);\n" +
//                                              "\n" +
//                                              "    public native @ByVal VideoFormat getVideoFormat();\n" +
//                                              "\n" +
//                                              "    public native void push_image(@SharedPtr @ByVal MemoryBuffer arg0);\n" +
//                                              "\n" +
//                                              "    public native void requeue_buffer(@SharedPtr @ByVal MemoryBuffer arg0);\n" +
//                                              "\n" +
//                                              "    public native @Cast(\"bool\") boolean setSink(@SharedPtr SinkInterface arg0);\n" +
//                                              "\n" +
//                                              "    public native @Cast(\"bool\") boolean set_buffer_collection(@ByVal SharedMemoryBufferVector new_buffers);\n" +
//                                              "\n" +
//                                              "    public native @ByVal SharedMemoryBufferVector get_buffer_collection();\n" +
//                                              "\n" +
//                                              "}"));
        //        infoMap.put(new Info("std::enable_shared_from_this<ImageSource>()").javaText("").skip());
        //        infoMap.put(new Info("asstd::enable_shared_from_this<ImageSource>").javaText("").skip());
        //        infoMap.put(new Info("asstd::enable_shared_from_this<ImageSource>()").javaText("").skip());

        //                infoMap.put(new Info("tcam::openDeviceInterface").skip());

//        infoMap.put(new Info("tcam::AFU420Device::set_framerate").skip());
        //        infoMap.put(new Info("tcam::AFU420Device::get_framerate").skip());
    }

//    public static class sink_callback extends FunctionPointer
//    {
//        static { Loader.load(); }
//        /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
//        public    sink_callback(Pointer p) { super(p); }
//        protected sink_callback() { allocate(); }
//        private native void allocate();
//        public native void call(@ByVal @Cast("tcam::MemoryBuffer*") Pointer arg0, Pointer arg1);
//    }
}
