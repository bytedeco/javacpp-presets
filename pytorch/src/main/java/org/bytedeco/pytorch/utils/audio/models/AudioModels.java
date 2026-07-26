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
package org.bytedeco.pytorch.utils.audio.models;
import org.bytedeco.pytorch.nn.options.*;
import org.bytedeco.pytorch.nn.modules.*;

import org.bytedeco.javacpp.LongPointer;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.modules.Conv1dImpl;
import org.bytedeco.pytorch.nn.modules.LinearImpl;
import org.bytedeco.pytorch.nn.options.Conv1dOptions;

import static org.bytedeco.pytorch.global.torch.adaptive_avg_pool1d;
import static org.bytedeco.pytorch.global.torch.flatten;
import static org.bytedeco.pytorch.global.torch.relu;

/**
 * Small audio / speech classifier factories (torchaudio-models style).
 */
public final class AudioModels {
    private AudioModels() {}

    private static LongPointer k1(long v) {
        // ExpandingArray<1> must hold the value, NOT allocate `v` empty slots.
        // new LongPointer(v) allocates capacity=v (bug); use a 1-element array.
        return new LongPointer(new long[]{v});
    }

    private static Conv1dImpl conv1d(long in, long out, long k, long stride, boolean bias) {
        Conv1dOptions opt = new Conv1dOptions(in, out, k1(k));
        opt.stride(k1(stride));
        opt.bias(bias);
        // padding defaults to 0; same-padding is approximated via adaptive pooling later
        return new Conv1dImpl(opt);
    }

    // -------------------------------------------------------------------------
    // Simple mel-feature MLP
    // -------------------------------------------------------------------------

    /**
     * Tiny MLP classifier on flattened mel / MFCC features — always safe for mini-train benches.
     * Input: {@code [N, F]} or {@code [N, C, T]} (flattened).
     */
    public static final class SimpleAudioClassifier extends Module {
        final LinearImpl fc1, fc2, fc3;
        final long inFeatures;

        public SimpleAudioClassifier(long inFeatures, long numClasses) {
            this(inFeatures, 256, numClasses);
        }

        public SimpleAudioClassifier(long inFeatures, long hidden, long numClasses) {
            super("SimpleAudioClassifier");
            this.inFeatures = inFeatures;
            fc1 = register_module("fc1", new LinearImpl(inFeatures, hidden));
            fc2 = register_module("fc2", new LinearImpl(hidden, hidden / 2));
            fc3 = register_module("fc3", new LinearImpl(hidden / 2, numClasses));
        }

        @Override
        public Tensor forward(Tensor x) {
            Tensor flat = x.reshape(x.size(0), -1);
            Tensor h = relu(fc1.forward(flat));
            h = relu(fc2.forward(h));
            return fc3.forward(h);
        }
    }

    // -------------------------------------------------------------------------
    // M5-style 1-D conv speech classifier
    // -------------------------------------------------------------------------

    /**
     * M5-inspired raw-waveform CNN (Dai et al. / torchaudio tutorials).
     * Expects input {@code [N, 1, T]} (mono waveform) or {@code [N, C, T]}.
     */
    public static final class M5 extends Module {
        final Conv1dImpl conv1, conv2, conv3, conv4;
        final LinearImpl fc;
        final long numClasses;

        public M5(long nInputChannel, long numClasses) {
            super("M5");
            this.numClasses = numClasses;
            // kernel 80 stride 16 ≈ first stage of M5
            conv1 = register_module("conv1", conv1d(nInputChannel, 32, 80, 16, true));
            conv2 = register_module("conv2", conv1d(32, 64, 3, 1, true));
            conv3 = register_module("conv3", conv1d(64, 128, 3, 1, true));
            conv4 = register_module("conv4", conv1d(128, 256, 3, 1, true));
            fc = register_module("fc", new LinearImpl(256, numClasses));
        }

        /** Backbone features before classifier — shape {@code [N, 256]}. */
        public Tensor features(Tensor x) {
            if (x.dim() == 2) {
                x = x.unsqueeze(1);
            }
            Tensor h = relu(conv1.forward(x));
            h = adaptive_avg_pool1d(h, Math.max(1L, h.size(2) / 4));
            h = relu(conv2.forward(h));
            h = adaptive_avg_pool1d(h, Math.max(1L, h.size(2) / 4));
            h = relu(conv3.forward(h));
            h = adaptive_avg_pool1d(h, Math.max(1L, h.size(2) / 4));
            h = relu(conv4.forward(h));
            h = adaptive_avg_pool1d(h, 1L);
            return flatten(h, 1L, -1L);
        }

        public long featureDim() { return 256L; }

        @Override
        public Tensor forward(Tensor x) {
            return fc.forward(features(x));
        }
    }

    /**
     * Wav2Letter-lite: stack of 1-D convs + linear classifier on time-pooled features.
     * Input {@code [N, C, T]} raw waveform or spectrogram frames.
     */
    public static final class Wav2LetterLite extends Module {
        final Conv1dImpl conv1, conv2, conv3;
        final LinearImpl fc1, fc2;
        final long numClasses;

        public Wav2LetterLite(long inChannels, long numClasses) {
            super("Wav2LetterLite");
            this.numClasses = numClasses;
            conv1 = register_module("conv1", conv1d(inChannels, 128, 11, 2, true));
            conv2 = register_module("conv2", conv1d(128, 128, 11, 1, true));
            conv3 = register_module("conv3", conv1d(128, 256, 11, 1, true));
            fc1 = register_module("fc1", new LinearImpl(256, 128));
            fc2 = register_module("fc2", new LinearImpl(128, numClasses));
        }

        /** Backbone features (after fc1) — shape {@code [N, 128]}. */
        public Tensor features(Tensor x) {
            if (x.dim() == 2) {
                x = x.unsqueeze(1);
            }
            Tensor h = relu(conv1.forward(x));
            h = relu(conv2.forward(h));
            h = relu(conv3.forward(h));
            h = adaptive_avg_pool1d(h, 1L);
            h = flatten(h, 1L, -1L);
            return relu(fc1.forward(h));
        }

        public long featureDim() { return 128L; }

        @Override
        public Tensor forward(Tensor x) {
            return fc2.forward(features(x));
        }
    }

    // -------------------------------------------------------------------------
    // Factories
    // -------------------------------------------------------------------------

    public static SimpleAudioClassifier simple_audio_classifier(long inFeatures, long numClasses) {
        return new SimpleAudioClassifier(inFeatures, numClasses);
    }

    public static SimpleAudioClassifier simpleAudioClassifier(long inFeatures, long numClasses) {
        return simple_audio_classifier(inFeatures, numClasses);
    }

    public static M5 m5(long nInputChannel, long numClasses) {
        return new M5(nInputChannel, numClasses);
    }

    public static Wav2LetterLite wav2letter_lite(long inChannels, long numClasses) {
        return new Wav2LetterLite(inChannels, numClasses);
    }

    public static Wav2LetterLite wav2letterLite(long inChannels, long numClasses) {
        return wav2letter_lite(inChannels, numClasses);
    }
}
