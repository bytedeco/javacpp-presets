// Targeted by JavaCPP version 0.8-2-SNAPSHOT

package org.bytedeco.javacpp;

import java.nio.*;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

import static org.bytedeco.javacpp.avutil.*;
import static org.bytedeco.javacpp.avcodec.*;
import static org.bytedeco.javacpp.avformat.*;
import static org.bytedeco.javacpp.postproc.*;
import static org.bytedeco.javacpp.swresample.*;
import static org.bytedeco.javacpp.swscale.*;
import static org.bytedeco.javacpp.avfilter.*;

public class avdevice extends org.bytedeco.javacpp.presets.avdevice {
    static { Loader.load(); }

// Parsed from <libavdevice/avdevice.h>

/*
 * This file is part of FFmpeg.
 *
 * FFmpeg is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * FFmpeg is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with FFmpeg; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
 */

// #ifndef AVDEVICE_AVDEVICE_H
// #define AVDEVICE_AVDEVICE_H

// #include "version.h"

/**
 * @file
 * @ingroup lavd
 * Main libavdevice API header
 */

/**
 * @defgroup lavd Special devices muxing/demuxing library
 * @{
 * Libavdevice is a complementary library to @ref libavf "libavformat". It
 * provides various "special" platform-specific muxers and demuxers, e.g. for
 * grabbing devices, audio capture and playback etc. As a consequence, the
 * (de)muxers in libavdevice are of the AVFMT_NOFILE type (they use their own
 * I/O functions). The filename passed to avformat_open_input() often does not
 * refer to an actually existing file, but has some special device-specific
 * meaning - e.g. for x11grab it is the display name.
 *
 * To use libavdevice, simply call avdevice_register_all() to register all
 * compiled muxers and demuxers. They all use standard libavformat API.
 * @}
 */

// #include "libavformat/avformat.h"

/**
 * Return the LIBAVDEVICE_VERSION_INT constant.
 */
public static native @Cast("unsigned") int avdevice_version();

/**
 * Return the libavdevice build-time configuration.
 */
public static native @Cast("const char*") BytePointer avdevice_configuration();

/**
 * Return the libavdevice license.
 */
public static native @Cast("const char*") BytePointer avdevice_license();

/**
 * Initialize libavdevice and register all the input and output devices.
 * @warning This function is not thread safe.
 */
public static native void avdevice_register_all();

public static class AVDeviceRect extends Pointer {
    static { Loader.load(); }
    public AVDeviceRect() { allocate(); }
    public AVDeviceRect(int size) { allocateArray(size); }
    public AVDeviceRect(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public AVDeviceRect position(int position) {
        return (AVDeviceRect)super.position(position);
    }

    /** x coordinate of top left corner */
    public native int x(); public native AVDeviceRect x(int x);
    /** y coordinate of top left corner */
    public native int y(); public native AVDeviceRect y(int y);
    /** width */
    public native int width(); public native AVDeviceRect width(int width);
    /** height */
    public native int height(); public native AVDeviceRect height(int height);
}

/**
 * Message types used by avdevice_app_to_dev_control_message().
 */
/** enum AVAppToDevMessageType */

public static native @MemberGetter int AV_APP_TO_DEV_NONE();
public static final int
    /**
     * Dummy message.
     */
    AV_APP_TO_DEV_NONE = AV_APP_TO_DEV_NONE();
public static native @MemberGetter int AV_APP_TO_DEV_WINDOW_SIZE();
public static final int

    /**
     * Window size change message.
     *
     * Message is sent to the device every time the application changes the size
     * of the window device renders to.
     * Message should also be sent right after window is created.
     *
     * data: AVDeviceRect: new window size.
     */
    AV_APP_TO_DEV_WINDOW_SIZE = AV_APP_TO_DEV_WINDOW_SIZE();
public static native @MemberGetter int AV_APP_TO_DEV_WINDOW_REPAINT();
public static final int

    /**
     * Repaint request message.
     *
     * Message is sent to the device when window have to be rapainted.
     *
     * data: AVDeviceRect: area required to be repainted.
     *       NULL: whole area is required to be repainted.
     */
    AV_APP_TO_DEV_WINDOW_REPAINT = AV_APP_TO_DEV_WINDOW_REPAINT();

/**
 * Message types used by avdevice_dev_to_app_control_message().
 */
/** enum AVDevToAppMessageType */

public static native @MemberGetter int AV_DEV_TO_APP_NONE();
public static final int
    /**
     * Dummy message.
     */
    AV_DEV_TO_APP_NONE = AV_DEV_TO_APP_NONE();
public static native @MemberGetter int AV_DEV_TO_APP_CREATE_WINDOW_BUFFER();
public static final int

    /**
     * Create window buffer message.
     *
     * Device requests to create a window buffer. Exact meaning is device-
     * and application-dependent. Message is sent before rendering first
     * frame and all one-shot initializations should be done here.
     * Application is allowed to ignore preferred window buffer size.
     *
     * @note: Application is obligated to inform about window buffer size
     *        with AV_APP_TO_DEV_WINDOW_SIZE message.
     *
     * data: AVDeviceRect: preferred size of the window buffer.
     *       NULL: no preferred size of the window buffer.
     */
    AV_DEV_TO_APP_CREATE_WINDOW_BUFFER = AV_DEV_TO_APP_CREATE_WINDOW_BUFFER();
public static native @MemberGetter int AV_DEV_TO_APP_PREPARE_WINDOW_BUFFER();
public static final int

    /**
     * Prepare window buffer message.
     *
     * Device requests to prepare a window buffer for rendering.
     * Exact meaning is device- and application-dependent.
     * Message is sent before rendering of each frame.
     *
     * data: NULL.
     */
    AV_DEV_TO_APP_PREPARE_WINDOW_BUFFER = AV_DEV_TO_APP_PREPARE_WINDOW_BUFFER();
public static native @MemberGetter int AV_DEV_TO_APP_DISPLAY_WINDOW_BUFFER();
public static final int

    /**
     * Display window buffer message.
     *
     * Device requests to display a window buffer.
     * Message is sent when new frame is ready to be displyed.
     * Usually buffers need to be swapped in handler of this message.
     *
     * data: NULL.
     */
    AV_DEV_TO_APP_DISPLAY_WINDOW_BUFFER = AV_DEV_TO_APP_DISPLAY_WINDOW_BUFFER();
public static native @MemberGetter int AV_DEV_TO_APP_DESTROY_WINDOW_BUFFER();
public static final int

    /**
     * Destroy window buffer message.
     *
     * Device requests to destroy a window buffer.
     * Message is sent when device is about to be destroyed and window
     * buffer is not required anymore.
     *
     * data: NULL.
     */
    AV_DEV_TO_APP_DESTROY_WINDOW_BUFFER = AV_DEV_TO_APP_DESTROY_WINDOW_BUFFER();

/**
 * Send control message from application to device.
 *
 * @param s         device context.
 * @param type      message type.
 * @param data      message data. Exact type depends on message type.
 * @param data_size size of message data.
 * @return >= 0 on success, negative on error.
 *         AVERROR(ENOSYS) when device doesn't implement handler of the message.
 */
public static native int avdevice_app_to_dev_control_message(AVFormatContext s,
                                        @Cast("AVAppToDevMessageType") int type,
                                        Pointer data, @Cast("size_t") long data_size);

/**
 * Send control message from device to application.
 *
 * @param s         device context.
 * @param type      message type.
 * @param data      message data. Can be NULL.
 * @param data_size size of message data.
 * @return >= 0 on success, negative on error.
 *         AVERROR(ENOSYS) when application doesn't implement handler of the message.
 */
public static native int avdevice_dev_to_app_control_message(AVFormatContext s,
                                        @Cast("AVDevToAppMessageType") int type,
                                        Pointer data, @Cast("size_t") long data_size);

/**
 * Structure describes basic parameters of the device.
 */
public static class AVDeviceInfo extends Pointer {
    static { Loader.load(); }
    public AVDeviceInfo() { allocate(); }
    public AVDeviceInfo(int size) { allocateArray(size); }
    public AVDeviceInfo(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public AVDeviceInfo position(int position) {
        return (AVDeviceInfo)super.position(position);
    }

    /** device name, format depends on device */
    public native @Cast("char*") BytePointer device_name(); public native AVDeviceInfo device_name(BytePointer device_name);
    /** human friendly name */
    public native @Cast("char*") BytePointer device_description(); public native AVDeviceInfo device_description(BytePointer device_description);
}

/**
 * List of devices.
 */
public static class AVDeviceInfoList extends Pointer {
    static { Loader.load(); }
    public AVDeviceInfoList() { allocate(); }
    public AVDeviceInfoList(int size) { allocateArray(size); }
    public AVDeviceInfoList(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public AVDeviceInfoList position(int position) {
        return (AVDeviceInfoList)super.position(position);
    }

    /** list of autodetected devices */
    public native AVDeviceInfo devices(int i); public native AVDeviceInfoList devices(int i, AVDeviceInfo devices);
    @MemberGetter public native @Cast("AVDeviceInfo**") PointerPointer devices();
    /** number of autodetected devices */
    public native int nb_devices(); public native AVDeviceInfoList nb_devices(int nb_devices);
    /** index of default device or -1 if no default */
    public native int default_device(); public native AVDeviceInfoList default_device(int default_device);
}

/**
 * List devices.
 *
 * Returns available device names and their parameters.
 *
 * @note: Some devices may accept system-dependent device names that cannot be
 *        autodetected. The list returned by this function cannot be assumed to
 *        be always completed.
 *
 * @param s                device context.
 * @param[out] device_list list of autodetected devices.
 * @return count of autodetected devices, negative on error.
 */
public static native int avdevice_list_devices(AVFormatContext s, @Cast("AVDeviceInfoList**") PointerPointer device_list);
public static native int avdevice_list_devices(AVFormatContext s, @ByPtrPtr AVDeviceInfoList device_list);

/**
 * Convinient function to free result of avdevice_list_devices().
 *
 * @param devices device list to be freed.
 */
public static native void avdevice_free_list_devices(@Cast("AVDeviceInfoList**") PointerPointer device_list);
public static native void avdevice_free_list_devices(@ByPtrPtr AVDeviceInfoList device_list);

// #endif /* AVDEVICE_AVDEVICE_H */


}
