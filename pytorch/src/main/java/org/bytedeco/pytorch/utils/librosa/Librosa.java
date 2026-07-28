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
package org.bytedeco.pytorch.utils.librosa;

import org.bytedeco.pytorch.dataframe.dtype.AudioData;
import org.bytedeco.pytorch.utils.audio.functional.F;

import java.util.Objects;

/**
 * Top-level librosa-style facade. Methods mirror common {@code librosa.*} entry points
 * and delegate DSP to {@link AudioData}.
 */
public final class Librosa {
    public static final int DEFAULT_SR = 22050;

    private Librosa() {}

    /**
     * Load audio: {@code y, sr = librosa.load(path, sr, mono)}.
     */
    public static AudioLoad load(String path) {
        return load(path, DEFAULT_SR, true);
    }

    public static AudioLoad load(String path, int sr) {
        return load(path, sr, true);
    }

    public static AudioLoad load(String path, int sr, boolean mono) {
        Objects.requireNonNull(path, "path");
        int target = sr > 0 ? sr : DEFAULT_SR;
        AudioData ad = AudioData.load(path, target, mono);
        float[] y = ad.getSamples();
        if (y == null) {
            y = new float[0];
        }
        // ensure mono if requested and multi-channel interleaved
        if (mono && ad.getChannels() > 1) {
            y = to_mono(y, ad.getChannels());
        }
        return new AudioLoad(y, ad.getSampleRate());
    }

    /** Downmix interleaved multi-channel to mono. */
    public static float[] to_mono(float[] y) {
        return to_mono(y, 2);
    }

    public static float[] to_mono(float[] y, int channels) {
        Objects.requireNonNull(y, "y");
        int ch = Math.max(1, channels);
        if (ch == 1 || y.length < ch) {
            return y.clone();
        }
        int frames = y.length / ch;
        float[] mono = new float[frames];
        for (int t = 0; t < frames; t++) {
            double sum = 0;
            for (int c = 0; c < ch; c++) {
                sum += y[t * ch + c];
            }
            mono[t] = (float) (sum / ch);
        }
        return mono;
    }

    public static float[] toMono(float[] y, int channels) {
        return to_mono(y, channels);
    }

    /** Linear resample. */
    public static float[] resample(float[] y, int origSr, int targetSr) {
        Objects.requireNonNull(y, "y");
        return F.resampleSamples(y, 1, origSr, targetSr);
    }

    public static float[] resample(float[] y, int origSr, int targetSr, int channels) {
        return F.resampleSamples(y, channels, origSr, targetSr);
    }

    /** Duration in seconds. */
    public static double get_duration(float[] y, int sr) {
        return get_duration(y, sr, 1);
    }

    public static double get_duration(float[] y, int sr, int channels) {
        Objects.requireNonNull(y, "y");
        if (sr <= 0) {
            throw new IllegalArgumentException("sr must be > 0");
        }
        int ch = Math.max(1, channels);
        return (double) y.length / ch / (double) sr;
    }

    public static double getDuration(float[] y, int sr) {
        return get_duration(y, sr);
    }

    public static double get_duration(AudioData audio) {
        Objects.requireNonNull(audio, "audio");
        return audio.getDuration();
    }

    /** Helper: wrap samples as {@link AudioData}. */
    public static AudioData asAudioData(float[] y, int sr) {
        return asAudioData(y, sr, 1);
    }

    public static AudioData asAudioData(float[] y, int sr, int channels) {
        Objects.requireNonNull(y, "y");
        return new AudioData(y, sr, Math.max(1, channels));
    }

    /** Result pair of {@link #load}. */
    public static final class AudioLoad {
        public final float[] y;
        public final int sr;

        public AudioLoad(float[] y, int sr) {
            this.y = Objects.requireNonNull(y, "y");
            this.sr = sr;
        }

        public float[] y() {
            return y;
        }

        public int sr() {
            return sr;
        }

        public int sampleRate() {
            return sr;
        }

        public AudioData toAudioData() {
            return new AudioData(y, sr, 1);
        }

        public double duration() {
            return get_duration(y, sr);
        }

        @Override
        public String toString() {
            return "AudioLoad{sr=" + sr + ", samples=" + y.length + "}";
        }
    }
}
