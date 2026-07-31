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
package org.bytedeco.pytorch.vision.ffmpeg;

import org.bytedeco.ffmpeg.avutil.AVFrame;
import org.bytedeco.ffmpeg.swscale.SwsContext;
import org.bytedeco.javacpp.BytePointer;
import org.bytedeco.javacpp.DoublePointer;
import org.bytedeco.javacpp.IntPointer;
import org.bytedeco.javacpp.PointerPointer;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.data.numpy.DType;
import org.bytedeco.pytorch.data.numpy.NDArray;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.global.torch.ScalarType;

import static org.bytedeco.ffmpeg.global.avutil.*;
import static org.bytedeco.ffmpeg.global.swscale.*;

/**
 * Video frame glue over {@link AVFrame} — PyAV {@code av.video.frame.VideoFrame}.
 *
 * <pre>{@code
 * for (Frame f : container.decode(0)) {          // or decodeVideo(0)
 *     VideoFrame vf = (VideoFrame) f;
 *     NDArray rgb = vf.toNdarray("rgb24");       // H,W,C uint8
 *     Tensor t = vf.toTensor("rgb24");           // [H,W,C] or use toTensorChw()
 * }
 * VideoFrame.fromNdarray(arr, "rgb24");
 * }</pre>
 */
public final class VideoFrame extends Frame {

    VideoFrame(AVFrame frame, Rational timeBase) {
        super(frame, timeBase);
    }

    public int width() {
        ensureOpen();
        return frame.width();
    }

    public int height() {
        ensureOpen();
        return frame.height();
    }

    public int format() {
        ensureOpen();
        return frame.format();
    }

    public String formatName() {
        BytePointer p = av_get_pix_fmt_name(format());
        String n = FFmpegNative.ptrToString(p);
        return n != null ? n : ("pix_fmt_" + format());
    }

    /**
     * Reformat to another pixel format / size (new owned frame).
     * PyAV: {@code frame.reformat(format="rgb24")}.
     */
    public VideoFrame reformat(String pixFmt) {
        return reformat(-1, -1, pixFmt);
    }

    public VideoFrame reformat(int width, int height, String pixFmt) {
        ensureOpen();
        int dstFmt = pixFmt != null ? av_get_pix_fmt(pixFmt) : format();
        if (dstFmt < 0) throw new FFmpegException("unknown pixel format: " + pixFmt);
        int dw = width > 0 ? width : this.width();
        int dh = height > 0 ? height : this.height();
        if (dw == this.width() && dh == this.height() && dstFmt == format()) {
            return new VideoFrame(cloneNative(), timeBase);
        }

        SwsContext sws = sws_getContext(
                this.width(), this.height(), format(),
                dw, dh, dstFmt,
                SWS_BILINEAR, null, null, (DoublePointer) null);
        if (sws == null || sws.isNull()) {
            throw new FFmpegException("sws_getContext failed " + this.width() + "x" + this.height()
                    + " -> " + dw + "x" + dh + " fmt=" + pixFmt);
        }
        try {
            AVFrame dst = allocFrame();
            dst.width(dw);
            dst.height(dh);
            dst.format(dstFmt);
            FFmpegNative.check(av_frame_get_buffer(dst, 32), "av_frame_get_buffer");

            PointerPointer<BytePointer> srcPP = new PointerPointer<>(8);
            IntPointer srcLS = new IntPointer(8);
            for (int i = 0; i < 8; i++) {
                srcPP.put(i, frame.data(i));
                srcLS.put(i, frame.linesize(i));
            }
            PointerPointer<BytePointer> dstPP = new PointerPointer<>(8);
            IntPointer dstLS = new IntPointer(8);
            for (int i = 0; i < 8; i++) {
                dstPP.put(i, dst.data(i));
                dstLS.put(i, dst.linesize(i));
            }
            int ret = sws_scale(sws, srcPP, srcLS, 0, this.height(), dstPP, dstLS);
            if (ret < 0) {
                av_frame_free(dst);
                throw new FFmpegException("sws_scale failed", ret);
            }
            dst.pts(frame.pts());
            VideoFrame out = new VideoFrame(dst, timeBase);
            out.stream = this.stream;
            return out;
        } finally {
            sws_freeContext(sws);
        }
    }

    /**
     * PyAV {@code frame.to_ndarray(format="rgb24")} → shape {@code [H, W, C]} uint8.
     */
    public NDArray toNdarray() {
        return toNdarray("rgb24");
    }

    public NDArray toNdarray(String format) {
        String fmt = format != null ? format : "rgb24";
        VideoFrame rgb = needsConvert(fmt) ? reformat(fmt) : this;
        try {
            int w = rgb.width();
            int h = rgb.height();
            int c = channelsFor(fmt);
            long[] idata = new long[(int) ((long) h * w * c)];
            BytePointer data = rgb.frame.data(0);
            int linesize = rgb.frame.linesize(0);
            int rowBytes = w * c;
            byte[] row = new byte[rowBytes];
            int idx = 0;
            for (int y = 0; y < h; y++) {
                data.position((long) y * linesize).get(row);
                for (int i = 0; i < rowBytes; i++) {
                    idata[idx++] = row[i] & 0xFF;
                }
            }
            return new NDArray(idata, DType.UINT8, h, w, c);
        } finally {
            if (rgb != this) rgb.close();
        }
    }

    /**
     * Tensor {@code [H, W, C]} uint8 (PyAV ndarray layout).
     */
    public Tensor toTensor() {
        return toTensor("rgb24");
    }

    public Tensor toTensor(String format) {
        NDArray arr = toNdarray(format);
        // build uint8 tensor HWC from long[] values
        int h = (int) arr.shape[0];
        int w = (int) arr.shape[1];
        int c = (int) arr.shape[2];
        byte[] bytes = new byte[h * w * c];
        for (int i = 0; i < bytes.length; i++) bytes[i] = (byte) arr.getLong(i);
        // float path more portable across tensor bindings: use float [0,255] HWC
        float[] flat = new float[bytes.length];
        for (int i = 0; i < bytes.length; i++) flat[i] = bytes[i] & 0xFF;
        Tensor t = torch.tensor(flat).reshape(new long[]{h, w, c});
        return t;
    }

    /** CHW float32 [0,255] — matches existing {@link VideoTensors} layout. */
    public Tensor toTensorChw() {
        return toTensorChw("rgb24");
    }

    public Tensor toTensorChw(String format) {
        String fmt = format != null ? format : "rgb24";
        VideoFrame rgb = needsConvert(fmt) ? reformat(fmt) : this;
        try {
            int w = rgb.width();
            int h = rgb.height();
            int c = channelsFor(fmt);
            BytePointer data = rgb.frame.data(0);
            int linesize = rgb.frame.linesize(0);
            float[] flat = new float[c * h * w];
            byte[] row = new byte[w * c];
            for (int y = 0; y < h; y++) {
                data.position((long) y * linesize).get(row);
                for (int x = 0; x < w; x++) {
                    for (int ch = 0; ch < c; ch++) {
                        flat[ch * h * w + y * w + x] = row[x * c + ch] & 0xFF;
                    }
                }
            }
            return torch.tensor(flat).reshape(new long[]{c, h, w});
        } finally {
            if (rgb != this) rgb.close();
        }
    }

    /**
     * PyAV {@code VideoFrame.from_ndarray(array, format="rgb24")}.
     * Accepts HWC uint8-like NDArray (values in idata 0..255).
     */
    public static VideoFrame fromNdarray(NDArray array, String format) {
        if (array == null) throw new FFmpegException("array is null");
        if (array.shape.length < 2) throw new FFmpegException("expected HWC or HW array, got rank " + array.shape.length);
        int h = (int) array.shape[0];
        int w = (int) array.shape[1];
        int c = array.shape.length >= 3 ? (int) array.shape[2] : 1;
        String fmt = format != null ? format : (c == 1 ? "gray" : "rgb24");
        int pix = av_get_pix_fmt(fmt);
        if (pix < 0) throw new FFmpegException("unknown pixel format: " + fmt);

        FFmpegNative.load();
        AVFrame fr = allocFrame();
        fr.width(w);
        fr.height(h);
        fr.format(pix);
        FFmpegNative.check(av_frame_get_buffer(fr, 32), "av_frame_get_buffer");
        FFmpegNative.check(av_frame_make_writable(fr), "av_frame_make_writable");

        int channels = channelsFor(fmt);
        BytePointer data = fr.data(0);
        int linesize = fr.linesize(0);
        byte[] row = new byte[w * channels];
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                for (int ch = 0; ch < channels; ch++) {
                    long v;
                    if (array.shape.length >= 3) {
                        v = array.getLong((int) ((long) y * w * c + (long) x * c + Math.min(ch, c - 1)));
                    } else {
                        v = array.getLong(y * w + x);
                    }
                    row[x * channels + ch] = (byte) (v & 0xFF);
                }
            }
            data.position((long) y * linesize).put(row);
        }
        return new VideoFrame(fr, new Rational(1, 1));
    }

    /**
     * From HWC or CHW float/byte Tensor. HWC assumed if last dim is 1/3/4; else CHW.
     */
    public static VideoFrame fromTensor(Tensor tensor, String format) {
        if (tensor == null || tensor.isNull()) throw new FFmpegException("tensor is null");
        long[] shape = new long[(int) tensor.dim()];
        for (int i = 0; i < shape.length; i++) shape[i] = tensor.size(i);
        if (shape.length != 3) throw new FFmpegException("expected 3D tensor HWC or CHW, got " + java.util.Arrays.toString(shape));

        boolean chw = shape[0] <= 4 && shape[0] < shape[2]; // heuristic: C small
        int h, w, c;
        if (chw) {
            c = (int) shape[0];
            h = (int) shape[1];
            w = (int) shape[2];
        } else {
            h = (int) shape[0];
            w = (int) shape[1];
            c = (int) shape[2];
        }
        Tensor flat = tensor.reshape(new long[]{tensor.numel()}).contiguous();
        // read as float then to uint (Tensor.to(ScalarType) — not TensorOptions)
        float[] vals;
        try {
            Tensor asFloat = flat.to(ScalarType.Float);
            NDArray tmp = NDArray.fromTensor(asFloat);
            vals = new float[(int) tmp.size];
            for (int i = 0; i < vals.length; i++) vals[i] = (float) tmp.getDouble(i);
        } catch (Throwable t) {
            throw new FFmpegException("cannot read tensor data: " + t.getMessage(), t);
        }
        long[] idata = new long[h * w * c];
        for (int i = 0; i < idata.length; i++) {
            float v = i < vals.length ? vals[i] : 0;
            if (v > 0 && v <= 1.0f) v *= 255f; // accept [0,1] normalized
            idata[i] = Math.max(0, Math.min(255, Math.round(v)));
        }
        // store as HWC in NDArray
        long[] hwc = new long[h * w * c];
        if (chw) {
            for (int y = 0; y < h; y++) {
                for (int x = 0; x < w; x++) {
                    for (int ch = 0; ch < c; ch++) {
                        hwc[(y * w + x) * c + ch] = idata[ch * h * w + y * w + x];
                    }
                }
            }
        } else {
            System.arraycopy(idata, 0, hwc, 0, hwc.length);
        }
        return fromNdarray(new NDArray(hwc, DType.UINT8, h, w, c), format);
    }

    private boolean needsConvert(String fmt) {
        int pix = av_get_pix_fmt(fmt);
        return pix < 0 || pix != format();
    }

    private static int channelsFor(String fmt) {
        if (fmt == null) return 3;
        switch (fmt.toLowerCase()) {
            case "gray":
            case "gray8":
            case "yuv420p": // planar — we only dump first plane if used directly; prefer rgb path
                return 1;
            case "rgb24":
            case "bgr24":
                return 3;
            case "rgba":
            case "bgra":
            case "rgb32":
                return 4;
            default:
                return 3;
        }
    }

    @Override
    public String toString() {
        if (closed) return "VideoFrame(closed)";
        return "VideoFrame(" + width() + "x" + height() + ", " + formatName() + ", pts=" + pts() + ")";
    }
}
