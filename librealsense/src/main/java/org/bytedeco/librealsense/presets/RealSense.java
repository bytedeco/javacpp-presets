/*
 * Copyright (C) 2016-2020 Jérémy Laviole, Samuel Audet
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
package org.bytedeco.librealsense.presets;

import java.util.List;
import org.bytedeco.javacpp.FunctionPointer;
import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.annotation.ByVal;
import org.bytedeco.javacpp.annotation.Cast;
import org.bytedeco.javacpp.annotation.Platform;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.javacpp.presets.javacpp;
import org.bytedeco.javacpp.tools.Info;
import org.bytedeco.javacpp.tools.InfoMap;
import org.bytedeco.javacpp.tools.InfoMapper;

/**
 *
 * @author Jérémy Laviole
 */
@Properties(inherit = javacpp.class, target = "org.bytedeco.librealsense", global = "org.bytedeco.librealsense.global.RealSense", value = {
    @Platform(value = {"linux-armhf", "linux-arm64", "linux-x86", "macosx-x86", "windows-x86"}, compiler = "cpp11",
         include = {"<librealsense/rs.h>", "<librealsense/rs.hpp>", "<librealsense/rscore.hpp>", "<librealsense/rsutil.h>"},
         link = "realsense@.1"),
    @Platform(value = "macosx", preload = "usb-1.0@.0", preloadpath = "/usr/local/lib/") })
public class RealSense implements InfoMapper {
    static { Loader.checkVersion("org.bytedeco", "librealsense"); }

    public void map(InfoMap infoMap) {
        infoMap.put(new Info("std::runtime_error").cast().pointerTypes("Pointer"))
               .put(new Info("std::timed_mutex").cast().pointerTypes("Pointer"))
               .put(new Info("RS_API_VERSION_STR").cppTypes("const char*").pointerTypes("String").translate(false))
               .put(new Info("rs::log_severity::none").javaNames("log_none"))
               .put(new Info("rs::output_buffer_format::native").javaNames("output_native"))
               .put(new Info("rs::stream::depth", "rs::stream::color", "rs::stream::infrared", "rs::stream::infrared2").skip())
               .put(new Info("rs::device").purify())
               .put(new Info("std::function<void(motion_data)>").pointerTypes("MotionFunction"))
               .put(new Info("std::function<void(frame)>").pointerTypes("FrameFunction"))
               .put(new Info("std::function<void(timestamp_data)>").pointerTypes("TimestampFunction"))
               .put(new Info("std::function<void(log_severity,const char*)>").pointerTypes("LogCallbackFunction"));
    }

    public static class MotionFunction extends FunctionPointer {
        static { Loader.load(); }
        /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
        public    MotionFunction(Pointer p) { super(p); }
        protected MotionFunction() { allocate(); }
        private native void allocate();
        public native void call(@ByVal @Cast("rs::motion_data*") Pointer motion);
    }

    public static class FrameFunction extends FunctionPointer {
        static { Loader.load(); }
        /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
        public    FrameFunction(Pointer p) { super(p); }
        protected FrameFunction() { allocate(); }
        private native void allocate();
        public native void call(@ByVal @Cast("rs::frame*") Pointer frame);
    }

    public static class TimestampFunction extends FunctionPointer {
        static { Loader.load(); }
        /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
        public    TimestampFunction(Pointer p) { super(p); }
        protected TimestampFunction() { allocate(); }
        private native void allocate();
        public native void call(@ByVal @Cast("rs::timestamp_data*") Pointer timestamp);
    }

    public static class LogCallbackFunction extends FunctionPointer {
        static { Loader.load(); }
        /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
        public    LogCallbackFunction(Pointer p) { super(p); }
        protected LogCallbackFunction() { allocate(); }
        private native void allocate();
        public native void call(@Cast("rs::log_severity") int severity, @Cast("const char*") String message);
    }
}
