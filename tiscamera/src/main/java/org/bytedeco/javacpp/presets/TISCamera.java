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
                                compiler = { "cpp11" },
                                define = { "USER_SHARED_POINTER", "SHARED_PTR_NAMESPACE std" },
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
                                                ////                                                "devicelibrary.h",
                                                //                                                "format.h",
                                                //                                                "image_base_defines.h",
                                                //                                                "image_fourcc.h",
                                                //                                                "image_transform_base.h",
                                                //                                                //"internal.h",
                                                //                                                "logging.h",
                                                //                                                "public_utils.h",
                                                //                                                //                                                "serialization.h",
                                                //                                                "standard_properties.h",
                                                //                                                //"tcam.h",
                                                //                                                //                                                "tcam-semaphores.h",
                                                //                                                "utils.h",
                                                //                                                "version.h",

//                                                "</usr/include/glib-2.0/glib-object.h>",
//                                                "gobject/tcamprop.h",

                                                "PropertyImpl.h",
                                                "Property.h",
                                                "DeviceInfo.h",
                                                //                                                "BackendLoader.h",
                                                "VideoFormat.h",
                                                "VideoFormatDescription.h",
                                                "MemoryBuffer.h",
//                                                "SinkInterface.h",
                                                "CaptureDevice.h",
                                                //                                                "CaptureDeviceImpl.h",
                                                //                                                "DeviceIndex.h",
                                                //                                                "DeviceInterface.h",
                                                //                                                "FilterBase.h",
                                                "FormatHandlerInterface.h",
                                                //                                                "ImageSink.h",
                                                //                                                "ImageSource.h",
                                                //                                                "PipelineManager.h",
                                                //                                                "Properties.h",
                                                //                                                "PropertyGeneration.h",
                                                //                                                "PropertyHandler.h",
                                },
                                link = { "tcam" }
                )
}
                //,helper = "org.bytedeco.javacpp.helper.GObject"
)
public class TISCamera implements InfoMapper
{
    public void map(InfoMap infoMap)
    {
//        infoMap.put(new Info().javaText("public static class GSList extends Pointer {\n" +
//                                        "    static { Loader.load(); }\n" +
//                                        "    /** Default native constructor. */\n" +
//                                        "    public GSList() { super((Pointer)null); allocate(); }\n" +
//                                        "    /** Native array allocator. Access with {@link Pointer#position(long)}. */\n" +
//                                        "    public GSList(long size) { super((Pointer)null); allocateArray(size); }\n" +
//                                        "    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */\n" +
//                                        "    public GSList(Pointer p) { super(p); }\n" +
//                                        "    private native void allocate();\n" +
//                                        "    private native void allocateArray(long size);\n" +
//                                        "    @Override public GSList position(long position) {\n" +
//                                        "        return (GSList)super.position(position);\n" +
//                                        "    }\n" +
//                                        "\n" +
//                                        "  public native Pointer data(); public native GSList data(Pointer data);\n" +
//                                        "  public native GSList next(); public native GSList next(GSList next);\n" +
//                                        "}"));

        infoMap.put(new Info("G_TYPE_FUNDAMENTAL_MAX").skip());
        infoMap.put(new Info("G_DECLARE_INTERFACE").define(false));

        infoMap.put(new Info("TcamProp").cast().pointerTypes("Pointer"));

        infoMap.put(new Info("gdouble").cast().valueTypes("double").pointerTypes("DoublePointer", "DoubleBuffer", "double[]"));
        infoMap.put(new Info("gboolean").cast().valueTypes("boolean").pointerTypes("BooleanPointer", "BooleanBuffer", "boolean[]"));
        infoMap.put(new Info("gint", "guint").cast().valueTypes("int").pointerTypes("IntPointer", "IntBuffer", "int[]"));
        infoMap.put(new Info("guint16").cast().valueTypes("int").pointerTypes("ShortPointer", "ShortBuffer", "short[]"));
        infoMap.put(new Info("gpointer", "gconstpointer").cppTypes().valueTypes("void").pointerTypes("Pointer"));
        infoMap.put(new Info("gchar").cast().valueTypes("char").pointerTypes("CharPointer", "CharBuffer", "char[]"));
        infoMap.put(new Info("gsize").cast().valueTypes("long").pointerTypes("LongPointer", "LongBuffer", "long[]"));

        infoMap.put(new Info("GQuark",
                             //"GSList",
                             "GType",
                             "GTypeInterface",
                             "GValue").cast().pointerTypes("Pointer"));

        infoMap.put(new Info("GLIB_DEPRECATED_IN_2_36",
                             "GLIB_AVAILABLE_IN_ALL",
                             "GLIB_AVAILABLE_IN_2_38",
                             "GLIB_AVAILABLE_IN_2_34",
                             "GLIB_AVAILABLE_IN_2_36",
                             "GLIB_AVAILABLE_IN_2_44",
                             "GLIB_AVAILABLE_IN_2_42").cppTypes().annotations());

        infoMap.put(new Info("G_TYPE_FLAG_RESERVED_ID_BIT").skip());

        //infoMap.put(new Info("std::vector<std::vector<cv::Vec2i> >").pointerTypes("PointVectorVector").cast());
//        infoMap.put(new Info("std::shared_ptr<tcam::MemoryBuffer>").annotations("@SharedPtr").pointerTypes("MemoryBuffer"));
//        infoMap.put(new Info("SinkInterface").cppTypes("set_source").skip());
        //infoMap.put(new Info("std::vector<std::shared_ptr<MemoryBuffer>>").annotations("@SharedPtr").pointerTypes("MemoryBuffer"));

        infoMap.put(new Info("tcam::CaptureDevice::start_stream").skip());
        infoMap.put(new Info("std::map<std::string,int>").pointerTypes("StringIntMap").define());
//        //infoMap.put(new Info("Property*").cast().pointerTypes("Property").define());
//        infoMap.put(new Info("std::vector<tcam::Property>").pointerTypes("PropertyVector").cast());
        //infoMap.put(new Info("std::vector<Property*>").annotations("@StdVector").pointerTypes("Property"));
        infoMap.put(new Info("std::vector<tcam::Property*>").cast().pointerTypes("PropertyVector").define());
//        infoMap.put(new Info("std::vector<tensorflow::Edge*>", "std::vector<const tensorflow::Edge*>").cast().pointerTypes("EdgeVector").define())
//        //infoMap.put(new Info("std::vector<Property*>").pointerTypes("Property"));
//        //infoMap.put(new Info("tcam::CaptureDevice::get_available_properties").skip());
    }
}
