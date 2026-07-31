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
package org.bytedeco.pytorch.audio.utils;

import org.bytedeco.javacpp.FloatPointer;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.dataframe.dtype.AudioData;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.global.torch.ScalarType;

import java.util.Objects;

/**
 * Converters between float PCM samples / {@link AudioData} and torchaudio-style
 * waveform {@link Tensor}s. Layout is channel-first: {@code [C, T]} for multi-channel,
 * {@code [T]} when mono is requested as 1-D.
 */
public final class AudioTensors {
    private AudioTensors() {}

    /**
     * Pack interleaved (or mono) samples into a waveform tensor.
     * <ul>
     *   <li>{@code channels == 1} → shape {@code [time]}</li>
     *   <li>{@code channels > 1} → shape {@code [channels, time]} (de-interleaved)</li>
     * </ul>
     */
    public static Tensor toTensor(float[] samples, int channels) {
        Objects.requireNonNull(samples, "samples");
        if (channels < 1) {
            throw new IllegalArgumentException("channels must be >= 1, got " + channels);
        }
        if (samples.length % channels != 0) {
            throw new IllegalArgumentException(
                    "samples.length (" + samples.length + ") not divisible by channels (" + channels + ")");
        }
        int time = samples.length / channels;
        if (channels == 1) {
            // torchaudio-style mono is still channel-first: [1, T]
            return torch.tensor(samples).reshape(1, time);
        }
        // de-interleave LRLR... → [C, T]
        float[] planar = new float[samples.length];
        for (int t = 0; t < time; t++) {
            for (int c = 0; c < channels; c++) {
                planar[c * time + t] = samples[t * channels + c];
            }
        }
        return torch.tensor(planar).reshape(channels, time);
    }

    /** Channel-first tensor from {@link AudioData} samples. */
    public static Tensor toTensor(AudioData audio) {
        Objects.requireNonNull(audio, "audio");
        float[] samples = audio.getSamples();
        if (samples == null) {
            throw new IllegalArgumentException("AudioData has no samples");
        }
        int ch = Math.max(1, audio.getChannels());
        return toTensor(samples, ch);
    }

    /**
     * Convert a waveform tensor back to interleaved float samples.
     * Accepts {@code [T]}, {@code [C,T]}, or {@code [B,C,T]} (first batch item).
     */
    public static float[] fromTensor(Tensor waveform) {
        Objects.requireNonNull(waveform, "waveform");
        Tensor cpu = waveform.contiguous().cpu().to(ScalarType.Float);
        long[] shape = sizes(cpu);
        float[] data = toFloatArray(cpu);

        if (shape.length == 1) {
            return data; // [T] mono
        }
        if (shape.length == 3) {
            // [B,C,T] → take first batch
            int c = (int) shape[1];
            int t = (int) shape[2];
            float[] first = new float[c * t];
            System.arraycopy(data, 0, first, 0, first.length);
            return planarToInterleaved(first, c, t);
        }
        if (shape.length == 2) {
            // Could be [C,T] or [T,C]. Prefer torchaudio convention [C,T] when C is small.
            int d0 = (int) shape[0];
            int d1 = (int) shape[1];
            if (d0 <= 16 && d1 >= d0) {
                return planarToInterleaved(data, d0, d1); // [C,T]
            }
            // treat as [T,C] already interleaved-ish row-major
            return data;
        }
        throw new IllegalArgumentException("expected [T], [C,T] or [B,C,T], got rank " + shape.length);
    }

    /** Infer channel count from waveform shape. */
    public static int inferChannels(Tensor waveform) {
        Objects.requireNonNull(waveform, "waveform");
        long[] shape = sizes(waveform);
        if (shape.length == 1) {
            return 1;
        }
        if (shape.length == 2) {
            return (int) (shape[0] <= 16 ? shape[0] : 1);
        }
        if (shape.length == 3) {
            return (int) shape[1];
        }
        throw new IllegalArgumentException("expected [T], [C,T] or [B,C,T], got rank " + shape.length);
    }

    /** Infer number of time samples from waveform shape. */
    public static int inferTime(Tensor waveform) {
        Objects.requireNonNull(waveform, "waveform");
        long[] shape = sizes(waveform);
        if (shape.length == 1) {
            return (int) shape[0];
        }
        if (shape.length == 2) {
            return (int) (shape[0] <= 16 ? shape[1] : shape[0]);
        }
        if (shape.length == 3) {
            return (int) shape[2];
        }
        throw new IllegalArgumentException("expected [T], [C,T] or [B,C,T], got rank " + shape.length);
    }

    /** Build {@link AudioData} from a waveform tensor. */
    public static AudioData toAudioData(Tensor waveform, int sampleRate) {
        float[] samples = fromTensor(waveform);
        int channels = inferChannels(waveform);
        return new AudioData(samples, sampleRate, channels);
    }

    /** Samples from {@link AudioData} (may be null if not loaded). */
    public static float[] samples(AudioData audio) {
        Objects.requireNonNull(audio, "audio");
        return audio.getSamples();
    }

    /** Flatten feature matrix {@code [n_features, n_frames]} to a 2-D tensor. */
    public static Tensor featureToTensor(float[][] feature) {
        Objects.requireNonNull(feature, "feature");
        if (feature.length == 0) {
            return torch.tensor(new float[0]).reshape(0, 0);
        }
        int rows = feature.length;
        int cols = feature[0].length;
        float[] flat = new float[rows * cols];
        for (int r = 0; r < rows; r++) {
            if (feature[r].length != cols) {
                throw new IllegalArgumentException("ragged feature matrix at row " + r);
            }
            System.arraycopy(feature[r], 0, flat, r * cols, cols);
        }
        return torch.tensor(flat).reshape(rows, cols);
    }

    /** Convert 2-D feature tensor back to {@code float[][]}. */
    public static float[][] tensorToFeature(Tensor t) {
        Objects.requireNonNull(t, "tensor");
        Tensor cpu = t.contiguous().cpu().to(ScalarType.Float);
        long[] shape = sizes(cpu);
        if (shape.length != 2) {
            throw new IllegalArgumentException("expected 2-D feature tensor, got rank " + shape.length);
        }
        int rows = (int) shape[0];
        int cols = (int) shape[1];
        float[] data = toFloatArray(cpu);
        float[][] out = new float[rows][cols];
        for (int r = 0; r < rows; r++) {
            System.arraycopy(data, r * cols, out[r], 0, cols);
        }
        return out;
    }

    public static float[] toFloatArray(Tensor t) {
        Tensor cpu = t.contiguous().cpu().to(ScalarType.Float);
        long n = cpu.numel();
        float[] data = new float[(int) n];
        FloatPointer ptr = cpu.data_ptr_float();
        for (int i = 0; i < n; i++) {
            data[i] = ptr.get(i);
        }
        return data;
    }

    public static long[] sizes(Tensor t) {
        long ndim = t.dim();
        long[] out = new long[(int) ndim];
        for (int i = 0; i < ndim; i++) {
            out[i] = t.size(i);
        }
        return out;
    }

    private static float[] planarToInterleaved(float[] planar, int channels, int time) {
        float[] interleaved = new float[channels * time];
        for (int t = 0; t < time; t++) {
            for (int c = 0; c < channels; c++) {
                interleaved[t * channels + c] = planar[c * time + t];
            }
        }
        return interleaved;
    }
}
