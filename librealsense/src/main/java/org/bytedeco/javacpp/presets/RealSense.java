/*
 * Copyright notice to add here.
 */
package org.bytedeco.javacpp.presets;

import org.bytedeco.javacpp.FunctionPointer;
import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.RealSense.context;
import org.bytedeco.javacpp.RealSense.device;
import org.bytedeco.javacpp.RealSense.frame;
import org.bytedeco.javacpp.RealSense.motion_data;
import org.bytedeco.javacpp.RealSense.timestamp_data;
import org.bytedeco.javacpp.annotation.Namespace;
//import org.bytedeco.javacpp.RealSense.context;
//import org.bytedeco.javacpp.RealSense.device;
import org.bytedeco.javacpp.annotation.Platform;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.javacpp.tools.Info;
import org.bytedeco.javacpp.tools.InfoMap;
import org.bytedeco.javacpp.tools.InfoMapper;

/**
 *
 * @author Jérémy Laviole
 */
@Properties(target = "org.bytedeco.javacpp.RealSense",
        value = {
            @Platform(include = {"<librealsense/rs.h>",
        "<librealsense/rs.hpp>",
        "<librealsense/rscore.hpp>",
        "<librealsense/rsutil.h>"},
            link = {"realsense@", "stdc++"},
            preload = "libstdc++")
        })

@Platform(include = {"<stdexcept>", "<mutex>"})

public class RealSense implements InfoMapper {

    public void map(InfoMap infoMap) {

        infoMap.put(new Info("std::runtime_error").cast().pointerTypes("Pointer"));

        infoMap
//                .put(new Info("rs::motion_callback::motion_callback").virtualize())
                .put(new Info("log_severity").cast().valueTypes("int").pointerTypes("IntPointer"))
                
                .put(new Info("std::timed_mutex").cast().pointerTypes("Pointer"))
                
                .put(new Info("rs_device::start_fw_logger").virtualize())
                .put(new Info("rs_device::stop_wf_logger").virtualize())
                // Not working
                .put(new Info("std::function<void()>").cast().pointerTypes("Fn"))
//                .put(new Info("std::function<void(motion_data)>").cast().pointerTypes("MotionFn"))
                .put(new Info("std::function<void(rs::motion_data)>").cast().pointerTypes("Pointer"))
//                .put(new Info("std::function<void(frame)>").cast().pointerTypes("FrameFn"))
                .put(new Info("std::function<void(rs::frame)>").cast().pointerTypes("Pointer"))
                
//                .put(new Info("std::function<void(log_severity,const char*>").cast().pointerTypes("LogFn"))
//                .put(new Info("std::function<void(log_severity,const char*>").cast().pointerTypes("Pointer"))
                .put(new Info("std::function<void(rs::timestamp_data)>").cast().pointerTypes("Pointer"));
//                .put(new Info("std::function<void(timestamp_data)>").cast().pointerTypes("TimestampFn"));

        // Not working
//        infoMap.put(new Info("std::timed_mutex").pointerTypes("stdMutex").define());

        // infoMap.put(new Info("AR_EXPORT").cppTypes().annotations())
        //        .put(new Info("defined(_MSC_VER) || defined(_WIN32_WCE)").define(false))
        //        .put(new Info("ARToolKitPlus::IDPATTERN").cast().valueTypes("long").pointerTypes("LongPointer", "LongBuffer", "long[]"))
        //        .put(new Info("ARFloat").cast().valueTypes("float").pointerTypes("FloatPointer", "FloatBuffer", "float[]"))
        //        .put(new Info("ARToolKitPlus::_64bits").cast().valueTypes("long").pointerTypes("LongPointer", "LongBuffer", "long[]"))
        //        .put(new Info("rpp_vec").cast().valueTypes("DoublePointer").pointerTypes("PointerPointer"))
        //        .put(new Info("rpp_mat").valueTypes("@Cast(\"double(*)[3]\") DoublePointer").pointerTypes("@Cast(\"double(*)[3][3]\") PointerPointer"));
    }

    public static class MotionFn extends FunctionPointer {
        static {
            Loader.load();
        }
        public MotionFn(Pointer p) {
            super(p);
        }
        protected MotionFn() {
            allocate();
        }
        private native void allocate();
        public native void call(motion_data motion);
    }

    public static class FrameFn extends FunctionPointer {
        static {
            Loader.load();
        }
        /**
         * Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}.
         */
        public FrameFn(Pointer p) {
            super(p);
        }

        protected FrameFn() {
            allocate();
        }
        private native void allocate();
        public native void call(frame fr);
    }

    public static class LogFn extends FunctionPointer {
        static {
            Loader.load();
        }
        /**
         * Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}.
         */
        public LogFn(Pointer p) {
            super(p);
        }
        protected LogFn() {
            allocate();
        }
        private native void allocate();
        public native void call();
    }

    public static class TimestampFn extends FunctionPointer {
        static {
            Loader.load();
        }
        /**
         * Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}.
         */
        public TimestampFn(Pointer p) {
            super(p);
        }
        protected TimestampFn() {
            allocate();
        }
        private native void allocate();
        public native void call(timestamp_data timestamp);
    }

    public static void main(String[] args) {

        context context = new context();

        System.out.println("Devices found: " + context.get_device_count());

        device device = context.get_device(0);

        System.out.println("Using device 0, an " + device.get_name());
        System.out.println(" Serial number: " + device.get_serial());
    }
    
    

//    // Create a context object. This object owns the handles to all connected realsense devices.
//    rs::context ctx;
//    printf("There are %d connected RealSense devices.\n", ctx.get_device_count());
//    if(ctx.get_device_count() == 0) return EXIT_FAILURE;
//
//    // This tutorial will access only a single device, but it is trivial to extend to multiple devices
//    rs::device * dev = ctx.get_device(0);
//    printf("\nUsing device 0, an %s\n", dev->get_name());
//    printf("    Serial number: %s\n", dev->get_serial());
//    printf("    Firmware version: %s\n", dev->get_firmware_version());
//
//    // Configure all streams to run at VGA resolution at 60 frames per second
//    dev->enable_stream(rs::stream::depth, 640, 480, rs::format::z16, 60);
//    dev->enable_stream(rs::stream::color, 640, 480, rs::format::rgb8, 60);
//    dev->enable_stream(rs::stream::infrared, 640, 480, rs::format::y8, 60);
//    try { dev->enable_stream(rs::stream::infrared2, 640, 480, rs::format::y8, 60); }
//    catch(...) { printf("Device does not provide infrared2 stream.\n"); }
//    dev->start();
//
//    // Open a GLFW window to display our output
//    glfwInit();
//    GLFWwindow * win = glfwCreateWindow(1280, 960, "librealsense tutorial #2", nullptr, nullptr);
//    glfwMakeContextCurrent(win);
//    while(!glfwWindowShouldClose(win))
//    {
//        // Wait for new frame data
//        glfwPollEvents();
//        dev->wait_for_frames();
//
//        glClear(GL_COLOR_BUFFER_BIT);
//        glPixelZoom(1, -1);
//
//        // Display depth data by linearly mapping depth between 0 and 2 meters to the red channel
//        glRasterPos2f(-1, 1);
//        glPixelTransferf(GL_RED_SCALE, 0xFFFF * dev->get_depth_scale() / 2.0f);
//        glDrawPixels(640, 480, GL_RED, GL_UNSIGNED_SHORT, dev->get_frame_data(rs::stream::depth));
//        glPixelTransferf(GL_RED_SCALE, 1.0f);
//
//        // Display color image as RGB triples
//        glRasterPos2f(0, 1);
//        glDrawPixels(640, 480, GL_RGB, GL_UNSIGNED_BYTE, dev->get_frame_data(rs::stream::color));
//
//        // Display infrared image by mapping IR intensity to visible luminance
//        glRasterPos2f(-1, 0);
//        glDrawPixels(640, 480, GL_LUMINANCE, GL_UNSIGNED_BYTE, dev->get_frame_data(rs::stream::infrared));
//
//        // Display second infrared image by mapping IR intensity to visible luminance
//        if(dev->is_stream_enabled(rs::stream::infrared2))
//        {
//            glRasterPos2f(0, 0);
//            glDrawPixels(640, 480, GL_LUMINANCE, GL_UNSIGNED_BYTE, dev->get_frame_data(rs::stream::infrared2));
//        }
//
//        glfwSwapBuffers(win);
//    }
}
