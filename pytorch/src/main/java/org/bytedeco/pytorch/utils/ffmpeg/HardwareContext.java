/*
 * Copyright (C) 2020-2026 Eduardo Gonzalez, Hervé Guillemet, Samuel Audet
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
package org.bytedeco.pytorch.utils.ffmpeg;

import org.bytedeco.ffmpeg.avcodec.AVCodecContext;
import org.bytedeco.ffmpeg.avutil.AVBufferRef;
import org.bytedeco.ffmpeg.avutil.AVDictionary;

import static org.bytedeco.ffmpeg.global.avutil.av_buffer_unref;
import static org.bytedeco.ffmpeg.global.avutil.av_hwdevice_ctx_create;
import static org.bytedeco.ffmpeg.global.avutil.av_hwdevice_find_type_by_name;
import static org.bytedeco.ffmpeg.global.avutil.av_hwdevice_get_type_name;

/**
 * Hardware device context — PyAV {@code av.codec.hwaccel.HardwareContext}.
 *
 * <pre>{@code
 * HardwareContext hw = HardwareContext.create("videotoolbox"); // or "cuda", "qsv", ...
 * stream.hwaccel(hw);
 * }</pre>
 */
public final class HardwareContext implements AutoCloseable {

    private final String typeName;
    private final int type;
    private AVBufferRef deviceCtx;
    private boolean closed;

    private HardwareContext(String typeName, int type, AVBufferRef deviceCtx) {
        this.typeName = typeName;
        this.type = type;
        this.deviceCtx = deviceCtx;
    }

    /**
     * PyAV {@code av.HardwareContext.create("cuda")}.
     *
     * @param type device type name: cuda, videotoolbox, qsv, vaapi, d3d11va, ...
     */
    public static HardwareContext create(String type) {
        FFmpegNative.load();
        if (type == null || type.isEmpty()) throw new FFmpegException("hardware type required");
        int t = av_hwdevice_find_type_by_name(type);
        if (t < 0) {
            throw new FFmpegException("unknown hardware type: " + type);
        }
        // @ByPtrPtr AVBufferRef holder
        AVBufferRef ref = new AVBufferRef(null);
        int ret = av_hwdevice_ctx_create(ref, t, (String) null, (AVDictionary) null, 0);
        if (ret < 0 || ref.isNull()) {
            throw new FFmpegException("av_hwdevice_ctx_create(" + type + ") failed: "
                    + FFmpegNative.errorString(ret), ret);
        }
        String name = FFmpegNative.ptrToString(av_hwdevice_get_type_name(t));
        return new HardwareContext(name != null ? name : type, t, ref);
    }

    public String typeName() { return typeName; }
    public int type() { return type; }

    public AVBufferRef deviceContext() {
        if (closed) throw new FFmpegException("HardwareContext closed");
        return deviceCtx;
    }

    /** Attach as {@code codecCtx.hw_device_ctx} for decoding. */
    void attachTo(AVCodecContext codecCtx) {
        if (closed || deviceCtx == null) return;
        try {
            codecCtx.hw_device_ctx(deviceCtx);
        } catch (Throwable t) {
            // best-effort
        }
    }

    @Override
    public void close() {
        if (closed) return;
        closed = true;
        if (deviceCtx != null && !deviceCtx.isNull()) {
            av_buffer_unref(deviceCtx);
            deviceCtx = null;
        }
    }

    @Override
    public String toString() {
        return "HardwareContext(" + typeName + ")";
    }
}
