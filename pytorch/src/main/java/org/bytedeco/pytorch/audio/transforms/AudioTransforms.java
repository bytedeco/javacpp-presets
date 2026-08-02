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
package org.bytedeco.pytorch.audio.transforms;

import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.audio.functional.AudioF;

/**
 * torchaudio.transforms factories and callable classes.
 * Inputs/outputs are waveform or spectrogram {@link Tensor}s.
 */
public final class AudioTransforms {
    private AudioTransforms() {}

    // -------------------------------------------------------------------------
    // Spectrogram / Mel / MFCC
    // -------------------------------------------------------------------------

    public static final class Spectrogram implements AudioTransform<Tensor, Tensor> {
        private final int sampleRate;
        private final int nFft;
        private final int hopLength;
        private final int winLength;
        private final boolean power;

        public Spectrogram(int sampleRate) {
            this(sampleRate, 2048, 512, 0, true);
        }

        public Spectrogram(int sampleRate, int nFft, int hopLength, int winLength, boolean power) {
            this.sampleRate = sampleRate;
            this.nFft = nFft;
            this.hopLength = hopLength;
            this.winLength = winLength;
            this.power = power;
        }

        @Override
        public Tensor forward(Tensor waveform) {
            return AudioF.spectrogram(waveform, sampleRate, nFft, hopLength, winLength, power);
        }
    }

    public static final class MelSpectrogram implements AudioTransform<Tensor, Tensor> {
        private final int sampleRate;
        private final int nMels;
        private final double fMin;
        private final double fMax;
        private final int nFft;
        private final int hopLength;

        public MelSpectrogram(int sampleRate) {
            this(sampleRate, 128, 0.0, sampleRate / 2.0, 2048, 512);
        }

        public MelSpectrogram(int sampleRate, int nMels) {
            this(sampleRate, nMels, 0.0, sampleRate / 2.0, 2048, 512);
        }

        public MelSpectrogram(int sampleRate, int nMels, double fMin, double fMax, int nFft, int hopLength) {
            this.sampleRate = sampleRate;
            this.nMels = nMels;
            this.fMin = fMin;
            this.fMax = fMax;
            this.nFft = nFft;
            this.hopLength = hopLength;
        }

        @Override
        public Tensor forward(Tensor waveform) {
            return AudioF.mel_spectrogram(waveform, sampleRate, nMels, fMin, fMax, nFft, hopLength);
        }
    }

    public static final class MFCC implements AudioTransform<Tensor, Tensor> {
        private final int sampleRate;
        private final int nMfcc;
        private final int nMels;
        private final double fMin;
        private final double fMax;
        private final int nFft;
        private final int hopLength;

        public MFCC(int sampleRate) {
            this(sampleRate, 13, 128, 0.0, sampleRate / 2.0, 2048, 512);
        }

        public MFCC(int sampleRate, int nMfcc) {
            this(sampleRate, nMfcc, 128, 0.0, sampleRate / 2.0, 2048, 512);
        }

        public MFCC(int sampleRate, int nMfcc, int nMels, double fMin, double fMax, int nFft, int hopLength) {
            this.sampleRate = sampleRate;
            this.nMfcc = nMfcc;
            this.nMels = nMels;
            this.fMin = fMin;
            this.fMax = fMax;
            this.nFft = nFft;
            this.hopLength = hopLength;
        }

        @Override
        public Tensor forward(Tensor waveform) {
            return AudioF.mfcc(waveform, sampleRate, nMfcc, nMels, fMin, fMax, nFft, hopLength);
        }
    }

    // -------------------------------------------------------------------------
    // Resample / Vol / Fade
    // -------------------------------------------------------------------------

    public static final class Resample implements AudioTransform<Tensor, Tensor> {
        private final int origFreq;
        private final int newFreq;

        public Resample(int origFreq, int newFreq) {
            this.origFreq = origFreq;
            this.newFreq = newFreq;
        }

        @Override
        public Tensor forward(Tensor waveform) {
            return AudioF.resample(waveform, origFreq, newFreq);
        }
    }

    public static final class Vol implements AudioTransform<Tensor, Tensor> {
        private final double gain;
        private final String gainType;

        public Vol(double gain) {
            this(gain, "amplitude");
        }

        public Vol(double gain, String gainType) {
            this.gain = gain;
            this.gainType = gainType;
        }

        @Override
        public Tensor forward(Tensor waveform) {
            return AudioF.vol(waveform, gain, gainType);
        }
    }

    public static final class Fade implements AudioTransform<Tensor, Tensor> {
        private final int fadeInLen;
        private final int fadeOutLen;
        private final String shape;

        public Fade(int fadeInLen, int fadeOutLen) {
            this(fadeInLen, fadeOutLen, "linear");
        }

        public Fade(int fadeInLen, int fadeOutLen, String shape) {
            this.fadeInLen = fadeInLen;
            this.fadeOutLen = fadeOutLen;
            this.shape = shape;
        }

        @Override
        public Tensor forward(Tensor waveform) {
            return AudioF.fade(waveform, fadeInLen, fadeOutLen, shape);
        }
    }

    // -------------------------------------------------------------------------
    // AmplitudeToDB / SpecAugment masks
    // -------------------------------------------------------------------------

    public static final class AmplitudeToDB implements AudioTransform<Tensor, Tensor> {
        private final double multiplier;
        private final double amin;
        private final double topDb;

        public AmplitudeToDB() {
            this(10.0, 1e-10, 80.0);
        }

        /** @param stype {@code "power"} → multiplier 10, {@code "amplitude"} → 20 */
        public AmplitudeToDB(String stype) {
            this("amplitude".equalsIgnoreCase(stype) ? 20.0 : 10.0, 1e-10, 80.0);
        }

        public AmplitudeToDB(double multiplier, double amin, double topDb) {
            this.multiplier = multiplier;
            this.amin = amin;
            this.topDb = topDb;
        }

        @Override
        public Tensor forward(Tensor x) {
            return AudioF.amplitudeToDB(x, multiplier, amin, 0.0, topDb);
        }
    }

    public static class FrequencyMasking implements AudioTransform<Tensor, Tensor> {
        private final int freqMaskParam;
        private final boolean iidMasks;

        public FrequencyMasking(int freqMaskParam) {
            this(freqMaskParam, false);
        }

        public FrequencyMasking(int freqMaskParam, boolean iidMasks) {
            this.freqMaskParam = freqMaskParam;
            this.iidMasks = iidMasks;
        }

        @Override
        public Tensor forward(Tensor specgram) {
            // iidMasks currently ignored (single mask); kept for torchaudio API parity
            return AudioF.frequency_masking(specgram, freqMaskParam);
        }

        public boolean iidMasks() {
            return iidMasks;
        }
    }

    public static class TimeMasking implements AudioTransform<Tensor, Tensor> {
        private final int timeMaskParam;
        private final boolean iidMasks;

        public TimeMasking(int timeMaskParam) {
            this(timeMaskParam, false);
        }

        public TimeMasking(int timeMaskParam, boolean iidMasks) {
            this.timeMaskParam = timeMaskParam;
            this.iidMasks = iidMasks;
        }

        @Override
        public Tensor forward(Tensor specgram) {
            return AudioF.time_masking(specgram, timeMaskParam);
        }

        public boolean iidMasks() {
            return iidMasks;
        }
    }

    public static final class MuLawEncoding implements AudioTransform<Tensor, Tensor> {
        private final int quantizationChannels;

        public MuLawEncoding() {
            this(256);
        }

        public MuLawEncoding(int quantizationChannels) {
            this.quantizationChannels = quantizationChannels;
        }

        @Override
        public Tensor forward(Tensor x) {
            return AudioF.mu_law_encoding(x, quantizationChannels);
        }
    }

    public static final class MuLawDecoding implements AudioTransform<Tensor, Tensor> {
        private final int quantizationChannels;

        public MuLawDecoding() {
            this(256);
        }

        public MuLawDecoding(int quantizationChannels) {
            this.quantizationChannels = quantizationChannels;
        }

        @Override
        public Tensor forward(Tensor x) {
            return AudioF.mu_law_decoding(x, quantizationChannels);
        }
    }

    // -------------------------------------------------------------------------
    // TimeStretch / PitchShift / LFCC / Deltas / Pad / Trim / DBToAmplitude
    // -------------------------------------------------------------------------

    /**
     * Approximate inverse of {@link Spectrogram} (zero-phase overlap-add).
     * See {@link AudioF#inverseSpectrogram}.
     */
    public static final class InverseSpectrogram implements AudioTransform<Tensor, Tensor> {
        private final int nFft;
        private final int hopLength;
        private final int winLength;
        private final boolean power;

        public InverseSpectrogram() {
            this(2048, 512, 0, true);
        }

        public InverseSpectrogram(int nFft, int hopLength) {
            this(nFft, hopLength, 0, true);
        }

        public InverseSpectrogram(int nFft, int hopLength, int winLength, boolean power) {
            this.nFft = nFft;
            this.hopLength = hopLength;
            this.winLength = winLength;
            this.power = power;
        }

        @Override
        public Tensor forward(Tensor specgram) {
            return AudioF.inverse_spectrogram(specgram, nFft, hopLength, winLength, power);
        }
    }

    /**
     * Time-stretch a spectrogram (magnitude) without changing pitch content layout.
     * {@code rate > 1} shortens time axis.
     */
    public static final class TimeStretch implements AudioTransform<Tensor, Tensor> {
        private final double fixedRate;
        private final int nFreq; // kept for torchaudio API parity (unused on magnitude path)

        public TimeStretch() {
            this(1.0, 0);
        }

        public TimeStretch(double fixedRate) {
            this(fixedRate, 0);
        }

        public TimeStretch(int nFreq, double fixedRate) {
            this(fixedRate, nFreq);
        }

        public TimeStretch(double fixedRate, int nFreq) {
            this.fixedRate = fixedRate <= 0 ? 1.0 : fixedRate;
            this.nFreq = nFreq;
        }

        @Override
        public Tensor forward(Tensor specgram) {
            return AudioF.time_stretch(specgram, fixedRate);
        }

        public int nFreq() {
            return nFreq;
        }
    }

    /** Pitch shift waveform by {@code nSteps} semitones (resample round-trip). */
    public static final class PitchShift implements AudioTransform<Tensor, Tensor> {
        private final int sampleRate;
        private final double nSteps;

        public PitchShift(int sampleRate, double nSteps) {
            this.sampleRate = sampleRate;
            this.nSteps = nSteps;
        }

        @Override
        public Tensor forward(Tensor waveform) {
            return AudioF.pitch_shift(waveform, sampleRate, nSteps);
        }
    }

    /** Linear-Frequency Cepstral Coefficients. */
    public static final class LFCC implements AudioTransform<Tensor, Tensor> {
        private final int sampleRate;
        private final int nLfcc;
        private final int nFilter;
        private final int nFft;
        private final int hopLength;

        public LFCC(int sampleRate) {
            this(sampleRate, 13, 128, 2048, 512);
        }

        public LFCC(int sampleRate, int nLfcc) {
            this(sampleRate, nLfcc, 128, 2048, 512);
        }

        public LFCC(int sampleRate, int nLfcc, int nFilter, int nFft, int hopLength) {
            this.sampleRate = sampleRate;
            this.nLfcc = nLfcc;
            this.nFilter = nFilter;
            this.nFft = nFft;
            this.hopLength = hopLength;
        }

        @Override
        public Tensor forward(Tensor waveform) {
            return AudioF.lfcc(waveform, sampleRate, nLfcc, nFilter, nFft, hopLength);
        }
    }

    /** First-order temporal deltas on spectrogram / feature last dim. */
    public static final class ComputeDeltas implements AudioTransform<Tensor, Tensor> {
        private final int winLength;

        public ComputeDeltas() {
            this(5);
        }

        public ComputeDeltas(int winLength) {
            this.winLength = winLength;
        }

        @Override
        public Tensor forward(Tensor specgram) {
            return AudioF.compute_deltas(specgram, winLength);
        }
    }

    /** Second-order deltas (acceleration). */
    public static final class Compute2DDeltas implements AudioTransform<Tensor, Tensor> {
        private final int winLength;

        public Compute2DDeltas() {
            this(5);
        }

        public Compute2DDeltas(int winLength) {
            this.winLength = winLength;
        }

        @Override
        public Tensor forward(Tensor specgram) {
            return AudioF.compute_2d_deltas(specgram, winLength);
        }
    }

    /** Inverse of {@link AmplitudeToDB}. */
    public static final class DBToAmplitude implements AudioTransform<Tensor, Tensor> {
        private final double ref;
        private final double power;

        public DBToAmplitude() {
            this(1.0, 0.5);
        }

        public DBToAmplitude(double ref, double power) {
            this.ref = ref;
            this.power = power;
        }

        @Override
        public Tensor forward(Tensor x) {
            return AudioF.dbToAmplitude(x, ref, power);
        }
    }

    /** Waveform pad along time. */
    public static final class Pad implements AudioTransform<Tensor, Tensor> {
        private final int left;
        private final int right;
        private final String mode;
        private final float value;

        public Pad(int padding) {
            this(padding, padding, "constant", 0f);
        }

        public Pad(int left, int right) {
            this(left, right, "constant", 0f);
        }

        public Pad(int left, int right, String mode) {
            this(left, right, mode, 0f);
        }

        public Pad(int left, int right, String mode, float value) {
            this.left = Math.max(0, left);
            this.right = Math.max(0, right);
            this.mode = mode;
            this.value = value;
        }

        @Override
        public Tensor forward(Tensor waveform) {
            return AudioF.pad(waveform, new int[]{left, right}, mode, value);
        }
    }

    /** Silence trim by energy threshold. */
    public static final class Trim implements AudioTransform<Tensor, Tensor> {
        private final int sampleRate;
        private final float topDb;

        public Trim(int sampleRate) {
            this(sampleRate, 60f);
        }

        public Trim(int sampleRate, float topDb) {
            this.sampleRate = sampleRate;
            this.topDb = topDb;
        }

        @Override
        public Tensor forward(Tensor waveform) {
            return AudioF.trim(waveform, sampleRate, topDb);
        }
    }

    /** Alias kept for checklist naming {@code FrequencyMask}. */
    public static final class FrequencyMask extends FrequencyMasking {
        public FrequencyMask(int freqMaskParam) {
            super(freqMaskParam);
        }

        public FrequencyMask(int freqMaskParam, boolean iidMasks) {
            super(freqMaskParam, iidMasks);
        }
    }

    /** Alias kept for checklist naming {@code TimeMask}. */
    public static final class TimeMask extends TimeMasking {
        public TimeMask(int timeMaskParam) {
            super(timeMaskParam);
        }

        public TimeMask(int timeMaskParam, boolean iidMasks) {
            super(timeMaskParam, iidMasks);
        }
    }

    // -------------------------------------------------------------------------
    // Factories (snake_case + camelCase)
    // -------------------------------------------------------------------------

    public static Spectrogram spectrogram(int sampleRate) {
        return new Spectrogram(sampleRate);
    }

    public static InverseSpectrogram inverse_spectrogram() {
        return new InverseSpectrogram();
    }

    public static InverseSpectrogram inverseSpectrogram() {
        return inverse_spectrogram();
    }

    public static InverseSpectrogram inverse_spectrogram(int nFft, int hopLength) {
        return new InverseSpectrogram(nFft, hopLength);
    }

    public static MelSpectrogram mel_spectrogram(int sampleRate) {
        return new MelSpectrogram(sampleRate);
    }

    public static MelSpectrogram melSpectrogram(int sampleRate) {
        return mel_spectrogram(sampleRate);
    }

    public static MelSpectrogram mel_spectrogram(int sampleRate, int nMels) {
        return new MelSpectrogram(sampleRate, nMels);
    }

    public static MFCC mfcc(int sampleRate) {
        return new MFCC(sampleRate);
    }

    public static MFCC mfcc(int sampleRate, int nMfcc) {
        return new MFCC(sampleRate, nMfcc);
    }

    public static Resample resample(int origFreq, int newFreq) {
        return new Resample(origFreq, newFreq);
    }

    public static Vol vol(double gain) {
        return new Vol(gain);
    }

    public static Fade fade(int fadeInLen, int fadeOutLen) {
        return new Fade(fadeInLen, fadeOutLen);
    }

    public static AmplitudeToDB amplitude_to_DB() {
        return new AmplitudeToDB();
    }

    public static AmplitudeToDB amplitudeToDB() {
        return amplitude_to_DB();
    }

    public static FrequencyMasking frequency_masking(int freqMaskParam) {
        return new FrequencyMasking(freqMaskParam);
    }

    public static FrequencyMasking frequencyMasking(int freqMaskParam) {
        return frequency_masking(freqMaskParam);
    }

    public static TimeMasking time_masking(int timeMaskParam) {
        return new TimeMasking(timeMaskParam);
    }

    public static TimeMasking timeMasking(int timeMaskParam) {
        return time_masking(timeMaskParam);
    }

    public static TimeStretch time_stretch(double fixedRate) {
        return new TimeStretch(fixedRate);
    }

    public static TimeStretch timeStretch(double fixedRate) {
        return time_stretch(fixedRate);
    }

    public static PitchShift pitch_shift(int sampleRate, double nSteps) {
        return new PitchShift(sampleRate, nSteps);
    }

    public static PitchShift pitchShift(int sampleRate, double nSteps) {
        return pitch_shift(sampleRate, nSteps);
    }

    public static LFCC lfcc(int sampleRate) {
        return new LFCC(sampleRate);
    }

    public static LFCC lfcc(int sampleRate, int nLfcc) {
        return new LFCC(sampleRate, nLfcc);
    }

    public static ComputeDeltas compute_deltas() {
        return new ComputeDeltas();
    }

    public static ComputeDeltas computeDeltas() {
        return compute_deltas();
    }

    public static Compute2DDeltas compute_2d_deltas() {
        return new Compute2DDeltas();
    }

    public static Compute2DDeltas compute2DDeltas() {
        return compute_2d_deltas();
    }

    public static DBToAmplitude db_to_amplitude() {
        return new DBToAmplitude();
    }

    public static DBToAmplitude dbToAmplitude() {
        return db_to_amplitude();
    }

    public static Pad pad(int padding) {
        return new Pad(padding);
    }

    public static Trim trim(int sampleRate) {
        return new Trim(sampleRate);
    }
}
