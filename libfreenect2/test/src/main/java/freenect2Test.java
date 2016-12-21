/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
import org.bytedeco.javacpp.freenect2.CpuPacketPipeline;
import org.bytedeco.javacpp.freenect2.Freenect2;
import org.bytedeco.javacpp.freenect2.Freenect2Device;
import org.bytedeco.javacpp.freenect2.PacketPipeline;

/**
 *
 * @author jiii
 */
public class freenect2Test {

    public static void main(String[] args) {

        Freenect2 freenect2 = new Freenect2();
        Freenect2Device device = null;
        PacketPipeline pipeline = null;
/// [context]
//  libfreenect2::Freenect2 freenect2;
//  libfreenect2::Freenect2Device *dev = 0;
//  libfreenect2::PacketPipeline *pipeline = 0;
/// [context]
        String serial = "";

        boolean viewer_enabled = true;
        boolean enable_rgb = true;
        boolean enable_depth = true;
        int deviceId = -1;
        int framemax = -1;

        pipeline = new CpuPacketPipeline();
//        pipeline = new libfreenect2::OpenGLPacketPipeline();
//        pipeline = new libfreenect2::OpenCLPacketPipeline(deviceId);
//        pipeline = new libfreenect2::CudaPacketPipeline(deviceId);

        if (freenect2.enumerateDevices() == 0) {
            System.out.println("no device connected!");
            return;
        }

        if (serial == "") {
            serial = freenect2.getDefaultDeviceSerialNumber().getString();
            System.out.println("Serial:" + serial);
        }
/// [discovery]

//        if (pipeline) {
/// [open]
        device = freenect2.openDevice(serial, pipeline);
/// [open]
//        } else {
//            dev = freenect2.openDevice(serial);
//        }

        if (device == null) {
            System.out.println("failure opening device!");
        } 

///// [listeners]
//        int types = 0;
//        if (enable_rgb) {
//            types |= libfreenect2::Frame::Color;
//        }
//        if (enable_depth) {
//            types |= libfreenect2::Frame::Ir | libfreenect2::Frame::Depth;
//        }
//        libfreenect2::SyncMultiFrameListener listener(types);
//        libfreenect2::FrameMap frames;
//        dev -> setColorFrameListener( & listener);
//        dev -> setIrAndDepthFrameListener( & listener);
///// [listeners]
//
///// [start]
//        if (enable_rgb && enable_depth) {
//            if (!dev -> start()) {
//                return -1;
//            }
//        } else {
//            if (!dev -> startStreams(enable_rgb, enable_depth)) {
//                return -1;
//            }
//        }
//        std::cout << "device serial: " << dev -> getSerialNumber() << std::endl;
//        std::cout << "device firmware: " << dev -> getFirmwareVersion() << std::endl;
///// [start]
//
///// [registration setup]
//        libfreenect2::Registration * registration = new libfreenect2::Registration(dev -> getIrCameraParams(), dev -> getColorCameraParams());
//  libfreenect2::Frame undistorted(512, 424, 4)
//        , registered(512, 424, 4);
///// [registration setup]
//
//        size_t framecount = 0;
//        #
//        ifdef EXAMPLES_WITH_OPENGL_SUPPORT
//        Viewer viewer;
//        if (viewer_enabled) {
//            viewer.initialize();
//        }
//        #else
//  viewer_enabled = false;
//        #endif /// [loop start]
//        while (!protonect_shutdown && (framemax == (size_t) - 1 || framecount < framemax)) {
//            if (!listener.waitForNewFrame(frames, 10 * 1000)) // 10 sconds
//            {
//                std::cout << "timeout!" << std::endl;
//                return -1;
//            }
//            libfreenect2::Frame * rgb = frames[libfreenect2::Frame::Color];
//            libfreenect2::Frame * ir = frames[libfreenect2::Frame::Ir];
//            libfreenect2::Frame * depth = frames[libfreenect2::Frame::Depth];
///// [loop start]
//
//            if (enable_rgb && enable_depth) {
//                /// [registration]
//                registration -> apply(rgb, depth,  & undistorted,  & registered);
///// [registration]
//            }
//
//            framecount++;
//            if (!viewer_enabled) {
//                if (framecount % 100 == 0) {
//                    std::cout << "The viewer is turned off. Received " << framecount << " frames. Ctrl-C to stop." << std::endl;
//                }
//                listener.release(frames);
//                continue;
//            }
//            #
//            ifdef EXAMPLES_WITH_OPENGL_SUPPORT
//            if (enable_rgb) {
//                viewer.addFrame("RGB", rgb);
//            }
//            if (enable_depth) {
//                viewer.addFrame("ir", ir);
//                viewer.addFrame("depth", depth);
//            }
//            if (enable_rgb && enable_depth) {
//                viewer.addFrame("registered",  & registered);
//            }
//
//            protonect_shutdown = protonect_shutdown || viewer.render();
//            #
//            endif /// [loop end]
//                    listener
//            .release(frames);
//            /**
//             * libfreenect2::this_thread::sleep_for(libfreenect2::chrono::milliseconds(100));
//             */
//        }
//        /// [loop end]
//
//        // TODO: restarting ir stream doesn't work!
//        // TODO: bad things will happen, if frame listeners are freed before dev->stop() :(
//        /// [stop]
        device.stop();
        device.close();
//        delete registration;

        return;
    }

}
