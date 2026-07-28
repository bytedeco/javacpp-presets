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
package org.bytedeco.pytorch.utils.audio.io;

import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.dataframe.dtype.AudioData;
import org.bytedeco.pytorch.utils.audio.utils.AudioTensors;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Objects;

/**
 * torchaudio-style audio I/O. Delegates decode/encode to {@link AudioData}.
 */
public final class AudioIO {
    public static final int DEFAULT_SAMPLE_RATE = 16000;

    private AudioIO() {}

    /** Load waveform + sample rate (mono @ {@link #DEFAULT_SAMPLE_RATE}). */
    public static AudioLoadResult load(String path) {
        return load(path, DEFAULT_SAMPLE_RATE, true);
    }

    public static AudioLoadResult load(Path path) {
        return load(path.toString(), DEFAULT_SAMPLE_RATE, true);
    }

    /**
     * Load audio file.
     *
     * @param path file path
     * @param sr   target sample rate (0 keeps source rate when possible)
     * @param mono if true, downmix to mono
     */
    public static AudioLoadResult load(String path, int sr, boolean mono) {
        Objects.requireNonNull(path, "path");
        int targetSr = sr > 0 ? sr : DEFAULT_SAMPLE_RATE;
        AudioData ad = AudioData.load(path, targetSr, mono);
        Tensor waveform = AudioTensors.toTensor(ad);
        return new AudioLoadResult(waveform, ad.getSampleRate());
    }

    public static AudioLoadResult load(Path path, int sr, boolean mono) {
        return load(path.toString(), sr, mono);
    }

    /** Snake_case alias. */
    public static AudioLoadResult load_audio(String path, int sr, boolean mono) {
        return load(path, sr, mono);
    }

    /** Save waveform tensor as WAV. */
    public static void save(String path, Tensor waveform, int sampleRate) throws IOException {
        Objects.requireNonNull(path, "path");
        Objects.requireNonNull(waveform, "waveform");
        if (sampleRate <= 0) {
            throw new IllegalArgumentException("sampleRate must be > 0");
        }
        AudioData ad = AudioTensors.toAudioData(waveform, sampleRate);
        ad.saveAsWav(path);
    }

    public static void save(Path path, Tensor waveform, int sampleRate) throws IOException {
        save(path.toString(), waveform, sampleRate);
    }

    /** Snake_case alias. */
    public static void save_audio(String path, Tensor waveform, int sampleRate) throws IOException {
        save(path, waveform, sampleRate);
    }

    /** Load as {@link AudioData} without tensor conversion. */
    public static AudioData loadAudioData(String path, int sr, boolean mono) {
        return AudioData.load(path, sr, mono);
    }

    /**
     * Result of {@link #load}: waveform tensor + sample rate.
     * Waveform layout is {@code [C, T]} or {@code [T]} for mono.
     */
    public static final class AudioLoadResult {
        public final Tensor waveform;
        public final int sampleRate;

        public AudioLoadResult(Tensor waveform, int sampleRate) {
            this.waveform = Objects.requireNonNull(waveform, "waveform");
            this.sampleRate = sampleRate;
        }

        public Tensor waveform() {
            return waveform;
        }

        public int sampleRate() {
            return sampleRate;
        }

        public int sample_rate() {
            return sampleRate;
        }

        public AudioData toAudioData() {
            return AudioTensors.toAudioData(waveform, sampleRate);
        }

        @Override
        public String toString() {
            return "AudioLoadResult{sampleRate=" + sampleRate + ", waveform=" + waveform + "}";
        }
    }
}
