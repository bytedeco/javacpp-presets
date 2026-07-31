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

import org.bytedeco.ffmpeg.avutil.AVChannelLayout;
import org.bytedeco.ffmpeg.avutil.AVFrame;
import org.bytedeco.javacpp.BytePointer;
import org.bytedeco.javacpp.FloatPointer;
import org.bytedeco.javacpp.ShortPointer;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.data.numpy.NDArray;
import org.bytedeco.pytorch.global.torch;

import static org.bytedeco.ffmpeg.global.avutil.*;

/**
 * Audio frame glue over {@link AVFrame} — PyAV {@code av.audio.frame.AudioFrame}.
 *
 * <p>{@link #toNdarray()} / {@link #toTensor()} return float32 planar samples as
 * shape {@code [channels, samples]} (PyAV-style). Conversion reuses the same
 * format table as {@link AudioTensorsFFmpeg} (no extra native stack).
 */
public final class AudioFrame extends Frame {

    AudioFrame(AVFrame frame, Rational timeBase) {
        super(frame, timeBase);
    }

    public int sampleRate() {
        ensureOpen();
        return frame.sample_rate();
    }

    public int samples() {
        ensureOpen();
        return frame.nb_samples();
    }

    public int channels() {
        ensureOpen();
        try {
            int ch = (int) frame.ch_layout().nb_channels();
            if (ch > 0) return ch;
        } catch (Throwable ignored) {}
        return 1;
    }

    public int format() {
        ensureOpen();
        return frame.format();
    }

    /** Float32 planar {@code [C, T]} — PyAV {@code frame.to_ndarray()}. */
    public NDArray toNdarray() {
        float[] planar = toFloatPlanar();
        int ch = Math.max(1, channels());
        int n = planar.length / ch;
        return new NDArray(planar, ch, n);
    }

    /** Tensor {@code [C, T]} float32. */
    public Tensor toTensor() {
        float[] planar = toFloatPlanar();
        int ch = Math.max(1, channels());
        int n = planar.length / ch;
        return torch.tensor(planar).reshape(new long[]{ch, n});
    }

    private float[] toFloatPlanar() {
        ensureOpen();
        int ch = Math.max(1, channels());
        int n = samples();
        if (n <= 0) return new float[0];
        return decodeFrameToPlanarFloat(frame, ch, n, format());
    }

    /** Same format table as {@link AudioTensorsFFmpeg}. */
    static float[] decodeFrameToPlanarFloat(AVFrame frame, int ch, int nSamples, int fmt) {
        float[] planar = new float[ch * nSamples];
        final int U8 = 0, S16 = 1, S32 = 2, FLT = 3, DBL = 4;
        final int U8P = 5, S16P = 6, S32P = 7, FLTP = 8, DBLP = 9;
        switch (fmt) {
            case FLTP: {
                for (int c = 0; c < ch; c++) {
                    FloatPointer plane = new FloatPointer(frame.data(c));
                    for (int s = 0; s < nSamples; s++) planar[c * nSamples + s] = plane.get(s);
                }
                break;
            }
            case FLT: {
                FloatPointer inter = new FloatPointer(frame.data(0));
                for (int s = 0; s < nSamples; s++)
                    for (int c = 0; c < ch; c++)
                        planar[c * nSamples + s] = inter.get((long) s * ch + c);
                break;
            }
            case S16P: {
                for (int c = 0; c < ch; c++) {
                    ShortPointer plane = new ShortPointer(frame.data(c));
                    for (int s = 0; s < nSamples; s++) planar[c * nSamples + s] = plane.get(s) / 32768.0f;
                }
                break;
            }
            case S16: {
                ShortPointer inter = new ShortPointer(frame.data(0));
                for (int s = 0; s < nSamples; s++)
                    for (int c = 0; c < ch; c++)
                        planar[c * nSamples + s] = inter.get((long) s * ch + c) / 32768.0f;
                break;
            }
            case S32P: {
                for (int c = 0; c < ch; c++) {
                    BytePointer plane = frame.data(c);
                    for (int s = 0; s < nSamples; s++)
                        planar[c * nSamples + s] = plane.getInt((long) s * 4L) / 2147483648.0f;
                }
                break;
            }
            case S32: {
                BytePointer inter = frame.data(0);
                for (int s = 0; s < nSamples; s++)
                    for (int c = 0; c < ch; c++)
                        planar[c * nSamples + s] =
                                inter.getInt(((long) s * ch + c) * 4L) / 2147483648.0f;
                break;
            }
            case U8P: {
                for (int c = 0; c < ch; c++) {
                    BytePointer plane = frame.data(c);
                    for (int s = 0; s < nSamples; s++)
                        planar[c * nSamples + s] = ((plane.get(s) & 0xFF) - 128) / 128.0f;
                }
                break;
            }
            case U8: {
                BytePointer inter = frame.data(0);
                for (int s = 0; s < nSamples; s++)
                    for (int c = 0; c < ch; c++)
                        planar[c * nSamples + s] =
                                ((inter.get((long) s * ch + c) & 0xFF) - 128) / 128.0f;
                break;
            }
            case DBLP: {
                for (int c = 0; c < ch; c++) {
                    BytePointer plane = frame.data(c);
                    for (int s = 0; s < nSamples; s++)
                        planar[c * nSamples + s] = (float) Double.longBitsToDouble(plane.getLong((long) s * 8L));
                }
                break;
            }
            case DBL: {
                BytePointer inter = frame.data(0);
                for (int s = 0; s < nSamples; s++)
                    for (int c = 0; c < ch; c++)
                        planar[c * nSamples + s] = (float) Double.longBitsToDouble(
                                inter.getLong(((long) s * ch + c) * 8L));
                break;
            }
            default: {
                // best-effort: treat as S16 interleaved
                try {
                    ShortPointer inter = new ShortPointer(frame.data(0));
                    for (int s = 0; s < nSamples; s++)
                        for (int c = 0; c < ch; c++)
                            planar[c * nSamples + s] = inter.get((long) s * ch + c) / 32768.0f;
                } catch (Throwable t) {
                    // leave zeros
                }
            }
        }
        return planar;
    }

    /**
     * Build an audio frame from planar float {@code [C, T]} NDArray.
     * PyAV: {@code AudioFrame.from_ndarray}.
     */
    public static AudioFrame fromNdarray(NDArray array, int sampleRate) {
        if (array == null || array.shape.length != 2) {
            throw new FFmpegException("expected [channels, samples] array");
        }
        int ch = (int) array.shape[0];
        int n = (int) array.shape[1];
        FFmpegNative.load();
        AVFrame fr = allocFrame();
        fr.nb_samples(n);
        fr.format(AV_SAMPLE_FMT_FLTP);
        fr.sample_rate(sampleRate > 0 ? sampleRate : 48000);
        try {
            AVChannelLayout layout = fr.ch_layout();
            av_channel_layout_default(layout, ch);
        } catch (Throwable ignored) {}
        FFmpegNative.check(av_frame_get_buffer(fr, 0), "av_frame_get_buffer");
        FFmpegNative.check(av_frame_make_writable(fr), "av_frame_make_writable");
        for (int c = 0; c < ch; c++) {
            FloatPointer plane = new FloatPointer(fr.data(c));
            for (int i = 0; i < n; i++) {
                plane.put(i, (float) array.getDouble(c * n + i));
            }
        }
        int sr = sampleRate > 0 ? sampleRate : 1;
        return new AudioFrame(fr, new Rational(1, sr));
    }

    @Override
    public String toString() {
        if (closed) return "AudioFrame(closed)";
        return "AudioFrame(ch=" + channels() + ", n=" + samples() + ", sr=" + sampleRate() + ", pts=" + pts() + ")";
    }
}
