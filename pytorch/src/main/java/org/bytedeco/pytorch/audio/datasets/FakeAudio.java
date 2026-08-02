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
package org.bytedeco.pytorch.audio.datasets;

import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.audio.transforms.AudioTransform;
import org.bytedeco.pytorch.audio.utils.AudioTensors;

import java.util.Random;

/**
 * Synthetic audio dataset of labeled sine waves (torchaudio FakeData-like).
 * Class {@code k} is a pure tone at frequency {@code (k+1) * baseFreqHz}.
 */
public final class FakeAudio extends AudioDataset {
    private final int size;
    private final int sampleRate;
    private final int numSamples;
    private final int numClasses;
    private final int channels;
    private final double baseFreqHz;
    private final long seed;

    public FakeAudio(int size, int sampleRate, int numSamples, int numClasses) {
        this(size, sampleRate, numSamples, numClasses, 1, 220.0, 0L);
    }

    public FakeAudio(int size, int sampleRate, int numSamples, int numClasses,
                     int channels, double baseFreqHz, long seed) {
        super((String) null);
        this.size = Math.max(0, size);
        this.sampleRate = sampleRate > 0 ? sampleRate : 16000;
        this.numSamples = Math.max(1, numSamples);
        this.numClasses = Math.max(1, numClasses);
        this.channels = Math.max(1, channels);
        this.baseFreqHz = baseFreqHz > 0 ? baseFreqHz : 220.0;
        this.seed = seed;
    }

    public FakeAudio setTransform(AudioTransform<?, ?> audioTransform) {
        super.setTransform(audioTransform);
        return this;
    }

    public int sampleRate() {
        return sampleRate;
    }

    public int numClasses() {
        return numClasses;
    }

    public int numSamples() {
        return numSamples;
    }

    @Override
    public int size() {
        return size;
    }

    @Override
    public Sample get(int index) {
        if (index < 0 || index >= size) {
            throw new IndexOutOfBoundsException("index=" + index + " size=" + size);
        }
        Random rng = new Random(seed + index * 9973L);
        int label = rng.nextInt(numClasses);
        double freq = baseFreqHz * (label + 1);
        double phase = rng.nextDouble() * 2.0 * Math.PI;
        float amp = 0.3f + 0.4f * rng.nextFloat();
        float[] samples = new float[numSamples * channels];
        for (int t = 0; t < numSamples; t++) {
            double s = amp * Math.sin(2.0 * Math.PI * freq * t / sampleRate + phase);
            // slight harmonic + noise for non-trivial features
            s += 0.15 * amp * Math.sin(4.0 * Math.PI * freq * t / sampleRate + phase);
            s += 0.02 * (rng.nextDouble() * 2.0 - 1.0);
            float v = (float) Math.max(-1.0, Math.min(1.0, s));
            for (int c = 0; c < channels; c++) {
                samples[t * channels + c] = v;
            }
        }
        Tensor waveform = AudioTensors.toTensor(samples, channels);
        Object data = applyTransform(waveform);
        Object target = applyTargetTransform(label);
        return new Sample(data, target);
    }

    /** Convenience: random float waveform batch {@code [N, C, T]}. */
    public static Tensor randomBatch(int n, int channels, int time) {
        float[] data = new float[n * channels * time];
        Random r = new Random(42);
        for (int i = 0; i < data.length; i++) {
            data[i] = (r.nextFloat() * 2f) - 1f;
        }
        return torch.tensor(data).reshape(n, channels, time);
    }

    /** Generate a mono sine tone. */
    public static float[] sine(double freqHz, int sampleRate, int numSamples, double amplitude) {
        float[] out = new float[numSamples];
        for (int t = 0; t < numSamples; t++) {
            out[t] = (float) (amplitude * Math.sin(2.0 * Math.PI * freqHz * t / sampleRate));
        }
        return out;
    }
}
