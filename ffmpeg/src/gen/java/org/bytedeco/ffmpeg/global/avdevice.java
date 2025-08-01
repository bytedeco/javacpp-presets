// Targeted by JavaCPP version 1.5.12: DO NOT EDIT THIS FILE

package org.bytedeco.ffmpeg.global;

import org.bytedeco.ffmpeg.avdevice.*;

import java.nio.*;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

import static org.bytedeco.javacpp.presets.javacpp.*;
import org.bytedeco.ffmpeg.avutil.*;
import static org.bytedeco.ffmpeg.global.avutil.*;
import org.bytedeco.ffmpeg.swresample.*;
import static org.bytedeco.ffmpeg.global.swresample.*;
import org.bytedeco.ffmpeg.avcodec.*;
import static org.bytedeco.ffmpeg.global.avcodec.*;
import org.bytedeco.ffmpeg.avformat.*;
import static org.bytedeco.ffmpeg.global.avformat.*;
import org.bytedeco.ffmpeg.postproc.*;
import static org.bytedeco.ffmpeg.global.postproc.*;
import org.bytedeco.ffmpeg.swscale.*;
import static org.bytedeco.ffmpeg.global.swscale.*;
import org.bytedeco.ffmpeg.avfilter.*;
import static org.bytedeco.ffmpeg.global.avfilter.*;

public class avdevice extends org.bytedeco.ffmpeg.presets.avdevice {
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

// #include "version_major.h"
// #ifndef HAVE_AV_CONFIG_H
/* When included as part of the ffmpeg build, only include the major version
 * to avoid unnecessary rebuilds. When included externally, keep including
 * the full version information. */
// #include "version.h"
// #endif

/**
 * \file
 * \ingroup lavd
 * Main libavdevice API header
 */

/**
 * \defgroup lavd libavdevice
 * Special devices muxing/demuxing library.
 *
 * Libavdevice is a complementary library to \ref libavf "libavformat". It
 * provides various "special" platform-specific muxers and demuxers, e.g. for
 * grabbing devices, audio capture and playback etc. As a consequence, the
 * (de)muxers in libavdevice are of the AVFMT_NOFILE type (they use their own
 * I/O functions). The filename passed to avformat_open_input() often does not
 * refer to an actually existing file, but has some special device-specific
 * meaning - e.g. for xcbgrab it is the display name.
 *
 * To use libavdevice, simply call avdevice_register_all() to register all
 * compiled muxers and demuxers. They all use standard libavformat API.
 *
 * \{
 */

// #include "libavutil/log.h"
// #include "libavutil/opt.h"
// #include "libavutil/dict.h"
// #include "libavformat/avformat.h"

/**
 * Return the LIBAVDEVICE_VERSION_INT constant.
 */
@NoException public static native @Cast("unsigned") int avdevice_version();

/**
 * Return the libavdevice build-time configuration.
 */
@NoException public static native @Cast("const char*") BytePointer avdevice_configuration();

/**
 * Return the libavdevice license.
 */
@NoException public static native @Cast("const char*") BytePointer avdevice_license();

/**
 * Initialize libavdevice and register all the input and output devices.
 */
@NoException public static native void avdevice_register_all();

/**
 * Audio input devices iterator.
 *
 * If d is NULL, returns the first registered input audio/video device,
 * if d is non-NULL, returns the next registered input audio/video device after d
 * or NULL if d is the last one.
 */
@NoException public static native @Const AVInputFormat av_input_audio_device_next(@Const AVInputFormat d);

/**
 * Video input devices iterator.
 *
 * If d is NULL, returns the first registered input audio/video device,
 * if d is non-NULL, returns the next registered input audio/video device after d
 * or NULL if d is the last one.
 */
@NoException public static native @Const AVInputFormat av_input_video_device_next(@Const AVInputFormat d);

/**
 * Audio output devices iterator.
 *
 * If d is NULL, returns the first registered output audio/video device,
 * if d is non-NULL, returns the next registered output audio/video device after d
 * or NULL if d is the last one.
 */
@NoException public static native @Const AVOutputFormat av_output_audio_device_next(@Const AVOutputFormat d);

/**
 * Video output devices iterator.
 *
 * If d is NULL, returns the first registered output audio/video device,
 * if d is non-NULL, returns the next registered output audio/video device after d
 * or NULL if d is the last one.
 */
@NoException public static native @Const AVOutputFormat av_output_video_device_next(@Const AVOutputFormat d);
// Targeting ../avdevice/AVDeviceRect.java



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
     * Message is sent to the device when window has to be repainted.
     *
     * data: AVDeviceRect: area required to be repainted.
     *       NULL: whole area is required to be repainted.
     */
    AV_APP_TO_DEV_WINDOW_REPAINT = AV_APP_TO_DEV_WINDOW_REPAINT();
public static native @MemberGetter int AV_APP_TO_DEV_PAUSE();
public static final int

    /**
     * Request pause/play.
     *
     * Application requests pause/unpause playback.
     * Mostly usable with devices that have internal buffer.
     * By default devices are not paused.
     *
     * data: NULL
     */
    AV_APP_TO_DEV_PAUSE        = AV_APP_TO_DEV_PAUSE();
public static native @MemberGetter int AV_APP_TO_DEV_PLAY();
public static final int
    AV_APP_TO_DEV_PLAY         = AV_APP_TO_DEV_PLAY();
public static native @MemberGetter int AV_APP_TO_DEV_TOGGLE_PAUSE();
public static final int
    AV_APP_TO_DEV_TOGGLE_PAUSE = AV_APP_TO_DEV_TOGGLE_PAUSE();
public static native @MemberGetter int AV_APP_TO_DEV_SET_VOLUME();
public static final int

    /**
     * Volume control message.
     *
     * Set volume level. It may be device-dependent if volume
     * is changed per stream or system wide. Per stream volume
     * change is expected when possible.
     *
     * data: double: new volume with range of 0.0 - 1.0.
     */
    AV_APP_TO_DEV_SET_VOLUME = AV_APP_TO_DEV_SET_VOLUME();
public static native @MemberGetter int AV_APP_TO_DEV_MUTE();
public static final int

    /**
     * Mute control messages.
     *
     * Change mute state. It may be device-dependent if mute status
     * is changed per stream or system wide. Per stream mute status
     * change is expected when possible.
     *
     * data: NULL.
     */
    AV_APP_TO_DEV_MUTE        = AV_APP_TO_DEV_MUTE();
public static native @MemberGetter int AV_APP_TO_DEV_UNMUTE();
public static final int
    AV_APP_TO_DEV_UNMUTE      = AV_APP_TO_DEV_UNMUTE();
public static native @MemberGetter int AV_APP_TO_DEV_TOGGLE_MUTE();
public static final int
    AV_APP_TO_DEV_TOGGLE_MUTE = AV_APP_TO_DEV_TOGGLE_MUTE();
public static native @MemberGetter int AV_APP_TO_DEV_GET_VOLUME();
public static final int

    /**
     * Get volume/mute messages.
     *
     * Force the device to send AV_DEV_TO_APP_VOLUME_LEVEL_CHANGED or
     * AV_DEV_TO_APP_MUTE_STATE_CHANGED command respectively.
     *
     * data: NULL.
     */
    AV_APP_TO_DEV_GET_VOLUME = AV_APP_TO_DEV_GET_VOLUME();
public static native @MemberGetter int AV_APP_TO_DEV_GET_MUTE();
public static final int
    AV_APP_TO_DEV_GET_MUTE   = AV_APP_TO_DEV_GET_MUTE();

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
     * \note: Application is obligated to inform about window buffer size
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
     * Message is sent when new frame is ready to be displayed.
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
public static native @MemberGetter int AV_DEV_TO_APP_BUFFER_OVERFLOW();
public static final int

    /**
     * Buffer fullness status messages.
     *
     * Device signals buffer overflow/underflow.
     *
     * data: NULL.
     */
    AV_DEV_TO_APP_BUFFER_OVERFLOW = AV_DEV_TO_APP_BUFFER_OVERFLOW();
public static native @MemberGetter int AV_DEV_TO_APP_BUFFER_UNDERFLOW();
public static final int
    AV_DEV_TO_APP_BUFFER_UNDERFLOW = AV_DEV_TO_APP_BUFFER_UNDERFLOW();
public static native @MemberGetter int AV_DEV_TO_APP_BUFFER_READABLE();
public static final int

    /**
     * Buffer readable/writable.
     *
     * Device informs that buffer is readable/writable.
     * When possible, device informs how many bytes can be read/write.
     *
     * \warning Device may not inform when number of bytes than can be read/write changes.
     *
     * data: int64_t: amount of bytes available to read/write.
     *       NULL: amount of bytes available to read/write is not known.
     */
    AV_DEV_TO_APP_BUFFER_READABLE = AV_DEV_TO_APP_BUFFER_READABLE();
public static native @MemberGetter int AV_DEV_TO_APP_BUFFER_WRITABLE();
public static final int
    AV_DEV_TO_APP_BUFFER_WRITABLE = AV_DEV_TO_APP_BUFFER_WRITABLE();
public static native @MemberGetter int AV_DEV_TO_APP_MUTE_STATE_CHANGED();
public static final int

    /**
     * Mute state change message.
     *
     * Device informs that mute state has changed.
     *
     * data: int: 0 for not muted state, non-zero for muted state.
     */
    AV_DEV_TO_APP_MUTE_STATE_CHANGED = AV_DEV_TO_APP_MUTE_STATE_CHANGED();
public static native @MemberGetter int AV_DEV_TO_APP_VOLUME_LEVEL_CHANGED();
public static final int

    /**
     * Volume level change message.
     *
     * Device informs that volume level has changed.
     *
     * data: double: new volume with range of 0.0 - 1.0.
     */
    AV_DEV_TO_APP_VOLUME_LEVEL_CHANGED = AV_DEV_TO_APP_VOLUME_LEVEL_CHANGED();

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
@NoException public static native int avdevice_app_to_dev_control_message(AVFormatContext s,
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
@NoException public static native int avdevice_dev_to_app_control_message(AVFormatContext s,
                                        @Cast("AVDevToAppMessageType") int type,
                                        Pointer data, @Cast("size_t") long data_size);
// Targeting ../avdevice/AVDeviceInfo.java


// Targeting ../avdevice/AVDeviceInfoList.java



/**
 * List devices.
 *
 * Returns available device names and their parameters.
 *
 * \note: Some devices may accept system-dependent device names that cannot be
 *        autodetected. The list returned by this function cannot be assumed to
 *        be always completed.
 *
 * @param s                device context.
 * @param device_list [out] list of autodetected devices.
 * @return count of autodetected devices, negative on error.
 */
@NoException public static native int avdevice_list_devices(AVFormatContext s, @Cast("AVDeviceInfoList**") PointerPointer device_list);
@NoException public static native int avdevice_list_devices(AVFormatContext s, @ByPtrPtr AVDeviceInfoList device_list);

/**
 * Convenient function to free result of avdevice_list_devices().
 *
 * @param device_list device list to be freed.
 */
@NoException public static native void avdevice_free_list_devices(@Cast("AVDeviceInfoList**") PointerPointer device_list);
@NoException public static native void avdevice_free_list_devices(@ByPtrPtr AVDeviceInfoList device_list);

/**
 * List devices.
 *
 * Returns available device names and their parameters.
 * These are convinient wrappers for avdevice_list_devices().
 * Device context is allocated and deallocated internally.
 *
 * @param device           device format. May be NULL if device name is set.
 * @param device_name      device name. May be NULL if device format is set.
 * @param device_options   An AVDictionary filled with device-private options. May be NULL.
 *                         The same options must be passed later to avformat_write_header() for output
 *                         devices or avformat_open_input() for input devices, or at any other place
 *                         that affects device-private options.
 * @param device_list [out] list of autodetected devices
 * @return count of autodetected devices, negative on error.
 * \note device argument takes precedence over device_name when both are set.
 */
@NoException public static native int avdevice_list_input_sources(@Const AVInputFormat device, @Cast("const char*") BytePointer device_name,
                                AVDictionary device_options, @Cast("AVDeviceInfoList**") PointerPointer device_list);
@NoException public static native int avdevice_list_input_sources(@Const AVInputFormat device, @Cast("const char*") BytePointer device_name,
                                AVDictionary device_options, @ByPtrPtr AVDeviceInfoList device_list);
@NoException public static native int avdevice_list_input_sources(@Const AVInputFormat device, String device_name,
                                AVDictionary device_options, @ByPtrPtr AVDeviceInfoList device_list);
@NoException public static native int avdevice_list_output_sinks(@Const AVOutputFormat device, @Cast("const char*") BytePointer device_name,
                               AVDictionary device_options, @Cast("AVDeviceInfoList**") PointerPointer device_list);
@NoException public static native int avdevice_list_output_sinks(@Const AVOutputFormat device, @Cast("const char*") BytePointer device_name,
                               AVDictionary device_options, @ByPtrPtr AVDeviceInfoList device_list);
@NoException public static native int avdevice_list_output_sinks(@Const AVOutputFormat device, String device_name,
                               AVDictionary device_options, @ByPtrPtr AVDeviceInfoList device_list);

/**
 * \}
 */

// #endif /* AVDEVICE_AVDEVICE_H */


// Parsed from <libavdevice/version_major.h>

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

// #ifndef AVDEVICE_VERSION_MAJOR_H
// #define AVDEVICE_VERSION_MAJOR_H

/**
 * \file
 * \ingroup lavd
 * Libavdevice version macros
 */

public static final int LIBAVDEVICE_VERSION_MAJOR =  61;

/**
 * FF_API_* defines may be placed below to indicate public API that will be
 * dropped at a future version bump. The defines themselves are not part of
 * the public API and may change, break or disappear at any time.
 */

// reminder to remove the bktr device on next major bump
public static final boolean FF_API_BKTR_DEVICE = (LIBAVDEVICE_VERSION_MAJOR < 62);
// reminder to remove the opengl device on next major bump
public static final boolean FF_API_OPENGL_DEVICE = (LIBAVDEVICE_VERSION_MAJOR < 62);
// reminder to remove the sdl2 device on next major bump
public static final boolean FF_API_SDL2_DEVICE = (LIBAVDEVICE_VERSION_MAJOR < 62);

// #endif /* AVDEVICE_VERSION_MAJOR_H */


// Parsed from <libavdevice/version.h>

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

// #ifndef AVDEVICE_VERSION_H
// #define AVDEVICE_VERSION_H

/**
 * \file
 * \ingroup lavd
 * Libavdevice version macros
 */

// #include "libavutil/version.h"

// #include "version_major.h"

public static final int LIBAVDEVICE_VERSION_MINOR =   3;
public static final int LIBAVDEVICE_VERSION_MICRO = 100;

public static native @MemberGetter int LIBAVDEVICE_VERSION_INT();
public static final int LIBAVDEVICE_VERSION_INT = LIBAVDEVICE_VERSION_INT();
// #define LIBAVDEVICE_VERSION     AV_VERSION(LIBAVDEVICE_VERSION_MAJOR,
//                                            LIBAVDEVICE_VERSION_MINOR,
//                                            LIBAVDEVICE_VERSION_MICRO)
public static final int LIBAVDEVICE_BUILD =       LIBAVDEVICE_VERSION_INT;

public static native @MemberGetter String LIBAVDEVICE_IDENT();
public static final String LIBAVDEVICE_IDENT = LIBAVDEVICE_IDENT();

// #endif /* AVDEVICE_VERSION_H */


}
