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
package org.bytedeco.pytorch.audio.functional;

import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.audio.librosa.effects.Effects;
import org.bytedeco.pytorch.dataframe.dtype.AudioData;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.audio.utils.AudioTensors;

import java.util.Objects;

/**
 * torchaudio.functional-style static helpers. Heavy DSP is delegated to {@link AudioData}.
 */
public final class F {
    private F() {}

    // -------------------------------------------------------------------------
    // Spectrogram family
    // -------------------------------------------------------------------------

    public static Tensor spectrogram(Tensor waveform, int sampleRate) {
        return spectrogram(waveform, sampleRate, 2048, 512, 0, true);
    }

    /**
     * Power spectrogram from STFT magnitude squared (or magnitude if {@code power == 1}).
     *
     * @param power 1 = magnitude, 2 = power (default-like torchaudio)
     */
    public static Tensor spectrogram(Tensor waveform, int sampleRate, int nFft, int hopLength,
                                     int winLength, boolean power) {
        AudioData ad = AudioTensors.toAudioData(waveform, sampleRate);
        AudioData.ComplexMatrix stft = ad.stft(nFft, hopLength > 0 ? hopLength : nFft / 4, "hann");
        float[][] spec = stft.powerSpectrum();
        if (!power) {
            // convert power → magnitude
            for (int i = 0; i < spec.length; i++) {
                for (int j = 0; j < spec[i].length; j++) {
                    spec[i][j] = (float) Math.sqrt(Math.max(spec[i][j], 0f));
                }
            }
        }
        return AudioTensors.featureToTensor(spec);
    }

    public static Tensor mel_spectrogram(Tensor waveform, int sampleRate) {
        return melSpectrogram(waveform, sampleRate, 128, 0.0, sampleRate / 2.0, 2048, 512);
    }

    public static Tensor melSpectrogram(Tensor waveform, int sampleRate) {
        return mel_spectrogram(waveform, sampleRate);
    }

    public static Tensor mel_spectrogram(Tensor waveform, int sampleRate, int nMels,
                                         double fMin, double fMax, int nFft, int hopLength) {
        AudioData ad = AudioTensors.toAudioData(waveform, sampleRate);
        float[][] mel = ad.melSpectrogram(nFft, hopLength, nMels, fMin, fMax);
        return AudioTensors.featureToTensor(mel);
    }

    public static Tensor melSpectrogram(Tensor waveform, int sampleRate, int nMels,
                                        double fMin, double fMax, int nFft, int hopLength) {
        return mel_spectrogram(waveform, sampleRate, nMels, fMin, fMax, nFft, hopLength);
    }

    public static Tensor mfcc(Tensor waveform, int sampleRate) {
        return mfcc(waveform, sampleRate, 13, 128, 0.0, sampleRate / 2.0, 2048, 512);
    }

    public static Tensor mfcc(Tensor waveform, int sampleRate, int nMfcc, int nMels,
                              double fMin, double fMax, int nFft, int hopLength) {
        AudioData ad = AudioTensors.toAudioData(waveform, sampleRate);
        float[][] coeffs = ad.mfcc(nMfcc, nFft, hopLength, nMels, fMin, fMax);
        return AudioTensors.featureToTensor(coeffs);
    }

    // -------------------------------------------------------------------------
    // Amplitude / dB
    // -------------------------------------------------------------------------

    /** Power/amplitude → decibels (torchaudio.functional.amplitude_to_DB). */
    public static Tensor amplitude_to_DB(Tensor x, double multiplier, double amin, double db_multiplier) {
        return amplitudeToDB(x, multiplier, amin, db_multiplier, 80.0);
    }

    public static Tensor amplitudeToDB(Tensor x) {
        return amplitudeToDB(x, 10.0, 1e-10, 0.0, 80.0);
    }

    /**
     * @param multiplier typically 10 for power, 20 for amplitude
     * @param amin       clamp floor
     * @param topDb      dynamic range cap (≤0 disables)
     */
    public static Tensor amplitudeToDB(Tensor x, double multiplier, double amin,
                                       double dbMultiplier, double topDb) {
        Objects.requireNonNull(x, "x");
        float[] data = AudioTensors.toFloatArray(x);
        float maxDb = Float.NEGATIVE_INFINITY;
        for (int i = 0; i < data.length; i++) {
            double v = Math.max(Math.abs(data[i]), amin);
            float db = (float) (multiplier * Math.log10(v) - dbMultiplier);
            data[i] = db;
            if (db > maxDb) maxDb = db;
        }
        if (topDb > 0) {
            float floor = maxDb - (float) topDb;
            for (int i = 0; i < data.length; i++) {
                if (data[i] < floor) data[i] = floor;
            }
        }
        return reshapeLike(data, x);
    }

    public static Tensor DB_to_amplitude(Tensor x, double ref, double power) {
        return dbToAmplitude(x, ref, power);
    }

    public static Tensor dbToAmplitude(Tensor x) {
        return dbToAmplitude(x, 1.0, 0.5);
    }

    /**
     * @param power 1 → power, 0.5 → amplitude (torchaudio convention)
     */
    public static Tensor dbToAmplitude(Tensor x, double ref, double power) {
        Objects.requireNonNull(x, "x");
        float[] data = AudioTensors.toFloatArray(x);
        for (int i = 0; i < data.length; i++) {
            data[i] = (float) (Math.pow(10.0, data[i] * power / 10.0) * ref);
        }
        return reshapeLike(data, x);
    }

    // -------------------------------------------------------------------------
    // Resample
    // -------------------------------------------------------------------------

    /** Linear-interpolation resample (AudioData has no dedicated resample). */
    public static Tensor resample(Tensor waveform, int origFreq, int newFreq) {
        if (origFreq <= 0 || newFreq <= 0) {
            throw new IllegalArgumentException("sample rates must be > 0");
        }
        if (origFreq == newFreq) {
            return waveform;
        }
        int channels = AudioTensors.inferChannels(waveform);
        float[] samples = AudioTensors.fromTensor(waveform);
        float[] out = resampleSamples(samples, channels, origFreq, newFreq);
        return AudioTensors.toTensor(out, channels);
    }

    public static float[] resampleSamples(float[] samples, int channels, int origFreq, int newFreq) {
        Objects.requireNonNull(samples, "samples");
        int ch = Math.max(1, channels);
        int inFrames = samples.length / ch;
        if (inFrames == 0) {
            return new float[0];
        }
        long outFramesLong = Math.max(1L, Math.round((double) inFrames * newFreq / (double) origFreq));
        int outFrames = (int) Math.min(outFramesLong, Integer.MAX_VALUE / ch);
        float[] out = new float[outFrames * ch];
        double ratio = (double) origFreq / (double) newFreq;
        for (int t = 0; t < outFrames; t++) {
            double src = t * ratio;
            int i0 = (int) Math.floor(src);
            int i1 = Math.min(i0 + 1, inFrames - 1);
            double frac = src - i0;
            for (int c = 0; c < ch; c++) {
                float a = samples[i0 * ch + c];
                float b = samples[i1 * ch + c];
                out[t * ch + c] = (float) (a + frac * (b - a));
            }
        }
        return out;
    }

    // -------------------------------------------------------------------------
    // Mu-law
    // -------------------------------------------------------------------------

    public static Tensor mu_law_encoding(Tensor x, int quantizationChannels) {
        return muLawEncoding(x, quantizationChannels);
    }

    public static Tensor muLawEncoding(Tensor x, int quantizationChannels) {
        Objects.requireNonNull(x, "x");
        int mu = Math.max(2, quantizationChannels) - 1;
        float[] data = AudioTensors.toFloatArray(x);
        for (int i = 0; i < data.length; i++) {
            float v = clamp(data[i], -1f, 1f);
            double mag = Math.log1p(mu * Math.abs(v)) / Math.log1p(mu);
            double signal = Math.signum(v) * mag;
            // map [-1,1] → [0, mu]
            data[i] = (float) Math.floor((signal + 1.0) / 2.0 * mu + 0.5);
        }
        return reshapeLike(data, x);
    }

    public static Tensor mu_law_decoding(Tensor x, int quantizationChannels) {
        return muLawDecoding(x, quantizationChannels);
    }

    public static Tensor muLawDecoding(Tensor x, int quantizationChannels) {
        Objects.requireNonNull(x, "x");
        int mu = Math.max(2, quantizationChannels) - 1;
        float[] data = AudioTensors.toFloatArray(x);
        for (int i = 0; i < data.length; i++) {
            double y = clamp(data[i], 0f, mu) / mu * 2.0 - 1.0;
            data[i] = (float) (Math.signum(y) * (1.0 / mu) * (Math.pow(1.0 + mu, Math.abs(y)) - 1.0));
        }
        return reshapeLike(data, x);
    }

    // -------------------------------------------------------------------------
    // Fade / volume
    // -------------------------------------------------------------------------

    public static Tensor fade(Tensor waveform, int fadeInLen, int fadeOutLen) {
        return fade(waveform, fadeInLen, fadeOutLen, "linear");
    }

    /**
     * Apply fade-in / fade-out along the time axis.
     *
     * @param shape {@code "linear"}, {@code "exponential"}, {@code "logarithmic"}, {@code "quarter_sine"}, {@code "half_sine"}
     */
    public static Tensor fade(Tensor waveform, int fadeInLen, int fadeOutLen, String shape) {
        Objects.requireNonNull(waveform, "waveform");
        int channels = AudioTensors.inferChannels(waveform);
        int time = AudioTensors.inferTime(waveform);
        float[] samples = AudioTensors.fromTensor(waveform);
        int fi = Math.max(0, Math.min(fadeInLen, time));
        int fo = Math.max(0, Math.min(fadeOutLen, time));
        String kind = shape == null ? "linear" : shape.toLowerCase();

        for (int t = 0; t < fi; t++) {
            float g = fadeGain(t, fi, kind, true);
            for (int c = 0; c < channels; c++) {
                samples[t * channels + c] *= g;
            }
        }
        for (int k = 0; k < fo; k++) {
            int t = time - fo + k;
            float g = fadeGain(k, fo, kind, false);
            for (int c = 0; c < channels; c++) {
                samples[t * channels + c] *= g;
            }
        }
        return AudioTensors.toTensor(samples, channels);
    }

    public static Tensor vol(Tensor waveform, double gain, String gainType) {
        Objects.requireNonNull(waveform, "waveform");
        String type = gainType == null ? "amplitude" : gainType.toLowerCase();
        double factor;
        switch (type) {
            case "db":
            case "power_db":
                factor = Math.pow(10.0, gain / 20.0);
                break;
            case "power":
                factor = Math.sqrt(Math.max(0.0, gain));
                break;
            case "amplitude":
            default:
                factor = gain;
                break;
        }
        float[] data = AudioTensors.toFloatArray(waveform);
        for (int i = 0; i < data.length; i++) {
            data[i] = (float) (data[i] * factor);
        }
        return reshapeLike(data, waveform);
    }

    public static Tensor vol(Tensor waveform, double gain) {
        return vol(waveform, gain, "amplitude");
    }

    /** Peak normalize to target peak (default 1.0). */
    public static Tensor normalize(Tensor waveform) {
        return normalize(waveform, 1.0);
    }

    public static Tensor normalize(Tensor waveform, double peak) {
        float[] data = AudioTensors.toFloatArray(waveform);
        float max = 0f;
        for (float v : data) {
            float a = Math.abs(v);
            if (a > max) max = a;
        }
        if (max > 1e-12f) {
            float scale = (float) (peak / max);
            for (int i = 0; i < data.length; i++) {
                data[i] *= scale;
            }
        }
        return reshapeLike(data, waveform);
    }

    // -------------------------------------------------------------------------
    // SpecAugment-style masking (on 2-D spectrograms [F,T] or [B,F,T])
    // -------------------------------------------------------------------------

    public static Tensor mask_along_axis(Tensor specgram, int maskParam, int maskValue, int axis) {
        float[] data = AudioTensors.toFloatArray(specgram);
        long[] shape = AudioTensors.sizes(specgram);
        if (shape.length < 2) {
            return specgram;
        }
        // work on last two dims as [F, T]
        int f = (int) shape[shape.length - 2];
        int t = (int) shape[shape.length - 1];
        int planes = data.length / (f * t);
        int dimSize = axis == 1 || axis == -1 ? t : f;
        int maskWidth = maskParam <= 0 ? 0 : (int) (Math.random() * maskParam);
        if (maskWidth <= 0 || dimSize <= 0) {
            return specgram;
        }
        int start = (int) (Math.random() * Math.max(1, dimSize - maskWidth + 1));
        for (int p = 0; p < planes; p++) {
            int base = p * f * t;
            if (axis == 1 || axis == -1) {
                // time mask
                for (int fi = 0; fi < f; fi++) {
                    for (int ti = start; ti < start + maskWidth && ti < t; ti++) {
                        data[base + fi * t + ti] = maskValue;
                    }
                }
            } else {
                // frequency mask
                for (int fi = start; fi < start + maskWidth && fi < f; fi++) {
                    for (int ti = 0; ti < t; ti++) {
                        data[base + fi * t + ti] = maskValue;
                    }
                }
            }
        }
        return reshapeLike(data, specgram);
    }

    public static Tensor frequency_masking(Tensor specgram, int freqMaskParam) {
        return mask_along_axis(specgram, freqMaskParam, 0, 0);
    }

    public static Tensor time_masking(Tensor specgram, int timeMaskParam) {
        return mask_along_axis(specgram, timeMaskParam, 0, 1);
    }

    // -------------------------------------------------------------------------
    // Pad / Trim / Deltas / TimeStretch / PitchShift / LFCC
    // -------------------------------------------------------------------------

    /**
     * Pad waveform along time axis.
     *
     * @param padding {left, right} samples, or single value for both sides
     * @param mode    {@code "constant"}, {@code "reflect"}, {@code "replicate"}
     */
    public static Tensor pad(Tensor waveform, int[] padding, String mode, float value) {
        Objects.requireNonNull(waveform, "waveform");
        int left;
        int right;
        if (padding == null || padding.length == 0) {
            left = right = 0;
        } else if (padding.length == 1) {
            left = right = Math.max(0, padding[0]);
        } else {
            left = Math.max(0, padding[0]);
            right = Math.max(0, padding[1]);
        }
        if (left == 0 && right == 0) {
            return waveform;
        }
        int channels = AudioTensors.inferChannels(waveform);
        int time = AudioTensors.inferTime(waveform);
        float[] samples = AudioTensors.fromTensor(waveform); // interleaved
        int outTime = time + left + right;
        float[] out = new float[outTime * channels];
        String m = mode == null ? "constant" : mode.toLowerCase();
        boolean constant = "constant".equals(m);
        for (int t = 0; t < outTime; t++) {
            int idx = t - left;
            if (constant && (idx < 0 || idx >= time)) {
                for (int c = 0; c < channels; c++) {
                    out[t * channels + c] = value;
                }
            } else {
                int src = mapPadIndex(idx, time, m);
                for (int c = 0; c < channels; c++) {
                    out[t * channels + c] = samples[src * channels + c];
                }
            }
        }
        return AudioTensors.toTensor(out, channels);
    }

    public static Tensor pad(Tensor waveform, int padding) {
        return pad(waveform, new int[]{padding, padding}, "constant", 0f);
    }

    public static Tensor pad(Tensor waveform, int left, int right) {
        return pad(waveform, new int[]{left, right}, "constant", 0f);
    }

    /**
     * Silence trim by energy threshold (librosa/torchaudio style).
     *
     * @param topDb frames quieter than peak-topDb are treated as silence
     */
    public static Tensor trim(Tensor waveform, int sampleRate, float topDb) {
        Objects.requireNonNull(waveform, "waveform");
        int channels = AudioTensors.inferChannels(waveform);
        float[] samples = AudioTensors.fromTensor(waveform);
        // mixdown for detection
        int frames = samples.length / channels;
        float[] mono = new float[frames];
        for (int t = 0; t < frames; t++) {
            float s = 0;
            for (int c = 0; c < channels; c++) s += samples[t * channels + c];
            mono[t] = s / channels;
        }
        Effects.TrimResult tr =
                Effects.trim(mono, sampleRate, topDb, 0.02f);
        int start = tr.start;
        int end = tr.end;
        if (end <= start) {
            return AudioTensors.toTensor(new float[0], channels);
        }
        float[] out = new float[(end - start) * channels];
        System.arraycopy(samples, start * channels, out, 0, out.length);
        return AudioTensors.toTensor(out, channels);
    }

    public static Tensor trim(Tensor waveform, int sampleRate) {
        return trim(waveform, sampleRate, 60f);
    }

    /**
     * First-order temporal differences (torchaudio.functional.compute_deltas) on last dim.
     * Spec shape [..., freq, time] or [..., time].
     */
    public static Tensor compute_deltas(Tensor specgram, int winLength) {
        return computeDeltas(specgram, winLength);
    }

    public static Tensor computeDeltas(Tensor specgram) {
        return computeDeltas(specgram, 5);
    }

    public static Tensor computeDeltas(Tensor specgram, int winLength) {
        Objects.requireNonNull(specgram, "specgram");
        int wl = winLength < 3 ? 3 : (winLength % 2 == 0 ? winLength + 1 : winLength);
        int half = wl / 2;
        float[] data = AudioTensors.toFloatArray(specgram);
        long[] shape = AudioTensors.sizes(specgram);
        int time = (int) shape[shape.length - 1];
        int rows = data.length / Math.max(1, time);
        // regressor denominators: sum_{n=-H..H} n^2
        double denom = 0;
        for (int n = -half; n <= half; n++) {
            denom += n * n;
        }
        if (denom < 1e-12) denom = 1;
        float[] out = new float[data.length];
        for (int r = 0; r < rows; r++) {
            int base = r * time;
            for (int t = 0; t < time; t++) {
                double num = 0;
                for (int n = -half; n <= half; n++) {
                    int idx = t + n;
                    if (idx < 0) idx = 0;
                    if (idx >= time) idx = time - 1;
                    num += n * data[base + idx];
                }
                out[base + t] = (float) (num / denom);
            }
        }
        return reshapeLike(out, specgram);
    }

    /** Second-order deltas = deltas of deltas. */
    public static Tensor compute_2d_deltas(Tensor specgram, int winLength) {
        return compute2DDeltas(specgram, winLength);
    }

    public static Tensor compute2DDeltas(Tensor specgram) {
        return compute2DDeltas(specgram, 5);
    }

    public static Tensor compute2DDeltas(Tensor specgram, int winLength) {
        return computeDeltas(computeDeltas(specgram, winLength), winLength);
    }

    /**
     * Time-stretch a spectrogram along the time axis by {@code rate}
     * (rate &gt; 1 → shorter / faster). Phase is not modified (magnitude-only).
     * Input shape [F, T] or [..., F, T].
     */
    public static Tensor time_stretch(Tensor specgram, double rate) {
        return timeStretch(specgram, rate);
    }

    public static Tensor timeStretch(Tensor specgram, double rate) {
        Objects.requireNonNull(specgram, "specgram");
        if (rate <= 0) {
            throw new IllegalArgumentException("rate must be > 0");
        }
        if (Math.abs(rate - 1.0) < 1e-9) {
            return specgram;
        }
        float[] data = AudioTensors.toFloatArray(specgram);
        long[] shape = AudioTensors.sizes(specgram);
        int tDim = (int) shape[shape.length - 1];
        int fDim = (int) shape[shape.length - 2];
        int planes = data.length / (fDim * tDim);
        int newT = Math.max(1, (int) Math.round(tDim / rate));
        float[] out = new float[planes * fDim * newT];
        for (int p = 0; p < planes; p++) {
            for (int f = 0; f < fDim; f++) {
                for (int t = 0; t < newT; t++) {
                    double src = t * rate;
                    int i0 = (int) Math.floor(src);
                    int i1 = Math.min(i0 + 1, tDim - 1);
                    double frac = src - i0;
                    if (i0 >= tDim) i0 = tDim - 1;
                    float a = data[p * fDim * tDim + f * tDim + i0];
                    float b = data[p * fDim * tDim + f * tDim + i1];
                    out[p * fDim * newT + f * newT + t] = (float) (a + frac * (b - a));
                }
            }
        }
        // rebuild shape with new time
        long[] newShape = shape.clone();
        newShape[newShape.length - 1] = newT;
        Tensor t = torch.tensor(out);
        return t.reshape(newShape);
    }

    /**
     * Crude pitch shift via resample round-trip (change rate then resample back to original length rate).
     * Good enough for augmentation; not phase-vocoder quality.
     *
     * @param nSteps semitones (positive → higher pitch)
     */
    public static Tensor pitch_shift(Tensor waveform, int sampleRate, double nSteps) {
        return pitchShift(waveform, sampleRate, nSteps);
    }

    public static Tensor pitchShift(Tensor waveform, int sampleRate, double nSteps) {
        Objects.requireNonNull(waveform, "waveform");
        if (Math.abs(nSteps) < 1e-9) {
            return waveform;
        }
        double factor = Math.pow(2.0, nSteps / 12.0);
        // speed up/down by factor (changes pitch+duration), then resample time back
        int channels = AudioTensors.inferChannels(waveform);
        float[] samples = AudioTensors.fromTensor(waveform);
        int inFrames = samples.length / channels;
        // first: resample as if rate changed by factor (shorter if factor>1)
        int midFreq = Math.max(1, (int) Math.round(sampleRate * factor));
        float[] mid = resampleSamples(samples, channels, sampleRate, midFreq);
        // then resample back to original sample count roughly: mid at midFreq → sampleRate
        // but we want original duration: stretch mid length to inFrames
        int midFrames = mid.length / channels;
        float[] out = new float[inFrames * channels];
        if (midFrames == 0) {
            return AudioTensors.toTensor(out, channels);
        }
        double ratio = (double) midFrames / (double) inFrames;
        for (int t = 0; t < inFrames; t++) {
            double src = t * ratio;
            int i0 = (int) Math.floor(src);
            int i1 = Math.min(i0 + 1, midFrames - 1);
            double frac = src - i0;
            for (int c = 0; c < channels; c++) {
                float a = mid[i0 * channels + c];
                float b = mid[i1 * channels + c];
                out[t * channels + c] = (float) (a + frac * (b - a));
            }
        }
        return AudioTensors.toTensor(out, channels);
    }

    /**
     * LFCC (Linear-Frequency Cepstral Coefficients) — DCT of log linear spectrogram.
     * Shape [nLfcc, time].
     */
    public static Tensor lfcc(Tensor waveform, int sampleRate, int nLfcc, int nFilter,
                              int nFft, int hopLength) {
        Objects.requireNonNull(waveform, "waveform");
        int nL = Math.max(1, nLfcc);
        int nFilt = Math.max(nL, nFilter);
        // power spectrogram [F, T]
        Tensor spec = spectrogram(waveform, sampleRate, nFft, hopLength, 0, true);
        float[] data = AudioTensors.toFloatArray(spec);
        long[] shape = AudioTensors.sizes(spec);
        int nFreq = (int) shape[shape.length - 2];
        int nTime = (int) shape[shape.length - 1];
        // triangular filter bank on linear freq axis
        float[][] fb = linearFilterBank(nFreq, nFilt);
        float[][] filtered = new float[nFilt][nTime];
        for (int m = 0; m < nFilt; m++) {
            for (int t = 0; t < nTime; t++) {
                double sum = 0;
                for (int f = 0; f < nFreq; f++) {
                    sum += fb[m][f] * data[f * nTime + t];
                }
                filtered[m][t] = (float) Math.log(Math.max(sum, 1e-10));
            }
        }
        // DCT-II
        float[][] cep = new float[nL][nTime];
        for (int k = 0; k < nL; k++) {
            double norm = k == 0 ? Math.sqrt(1.0 / nFilt) : Math.sqrt(2.0 / nFilt);
            for (int t = 0; t < nTime; t++) {
                double s = 0;
                for (int m = 0; m < nFilt; m++) {
                    s += filtered[m][t] * Math.cos(Math.PI * k * (m + 0.5) / nFilt);
                }
                cep[k][t] = (float) (norm * s);
            }
        }
        return AudioTensors.featureToTensor(cep);
    }

    public static Tensor lfcc(Tensor waveform, int sampleRate) {
        return lfcc(waveform, sampleRate, 13, 128, 2048, 512);
    }

    /** Inverse of amplitude_to_DB transform wrapper (power=1 → amplitude scale). */
    public static Tensor db_to_amplitude(Tensor x) {
        return dbToAmplitude(x);
    }

    /**
     * Approximate inverse spectrogram (Griffin-Lim style, single iteration / zero-phase).
     * Accepts power or magnitude spectrogram {@code [F, T]} / {@code [..., F, T]} and
     * reconstructs a mono waveform via overlap-add of real IFFT frames (zero phase).
     * Not bit-exact vs torchaudio; suitable for augmentation pipelines / previews.
     */
    public static Tensor inverse_spectrogram(Tensor specgram, int nFft, int hopLength,
                                             int winLength, boolean power) {
        return inverseSpectrogram(specgram, nFft, hopLength, winLength, power);
    }

    public static Tensor inverseSpectrogram(Tensor specgram, int nFft, int hopLength) {
        return inverseSpectrogram(specgram, nFft, hopLength, 0, true);
    }

    public static Tensor inverseSpectrogram(Tensor specgram, int nFft, int hopLength,
                                            int winLength, boolean power) {
        Objects.requireNonNull(specgram, "specgram");
        int nfft = nFft > 0 ? nFft : 400;
        int hop = hopLength > 0 ? hopLength : nfft / 4;
        int win = winLength > 0 ? winLength : nfft;
        float[] data = AudioTensors.toFloatArray(specgram);
        long[] shape = AudioTensors.sizes(specgram);
        int nFreq = (int) shape[shape.length - 2];
        int nTime = (int) shape[shape.length - 1];
        // use last plane if batched
        int planeSize = nFreq * nTime;
        int base = Math.max(0, data.length - planeSize);
        int outLen = hop * (nTime - 1) + win;
        if (outLen <= 0) {
            return AudioTensors.toTensor(new float[0], 1);
        }
        float[] out = new float[outLen];
        float[] windowSum = new float[outLen];
        float[] frame = new float[nfft];
        float[] hann = hannWindow(win);
        for (int t = 0; t < nTime; t++) {
            // build real-even spectrum (zero phase): bins 0..nFreq-1
            java.util.Arrays.fill(frame, 0f);
            for (int f = 0; f < nFreq && f < nfft; f++) {
                float v = data[base + f * nTime + t];
                if (power) {
                    v = (float) Math.sqrt(Math.max(v, 0f));
                }
                frame[f] = v;
                if (f > 0 && f < nfft - f) {
                    // conjugate symmetric for real signal
                    int mirror = nfft - f;
                    if (mirror < nfft) {
                        frame[mirror] = v;
                    }
                }
            }
            float[] timeFrame = realIfft(frame);
            int offset = t * hop;
            for (int i = 0; i < win && offset + i < outLen; i++) {
                float w = i < hann.length ? hann[i] : 1f;
                float sample = i < timeFrame.length ? timeFrame[i] * w : 0f;
                out[offset + i] += sample;
                windowSum[offset + i] += w * w;
            }
        }
        for (int i = 0; i < outLen; i++) {
            if (windowSum[i] > 1e-8f) {
                out[i] /= windowSum[i];
            }
        }
        return AudioTensors.toTensor(out, 1);
    }

    private static float[] hannWindow(int n) {
        float[] w = new float[Math.max(1, n)];
        if (n <= 1) {
            w[0] = 1f;
            return w;
        }
        for (int i = 0; i < n; i++) {
            w[i] = (float) (0.5 - 0.5 * Math.cos(2.0 * Math.PI * i / (n - 1)));
        }
        return w;
    }

    /** Naive O(n²) real IFFT assuming conjugate-symmetric real spectrum packed in {@code re}. */
    private static float[] realIfft(float[] re) {
        int n = re.length;
        float[] out = new float[n];
        for (int t = 0; t < n; t++) {
            double sum = 0;
            for (int k = 0; k < n; k++) {
                sum += re[k] * Math.cos(2.0 * Math.PI * k * t / n);
            }
            out[t] = (float) (sum / n);
        }
        return out;
    }

    // -------------------------------------------------------------------------
    // helpers
    // -------------------------------------------------------------------------

    private static float[][] linearFilterBank(int nFreq, int nFilt) {
        float[][] fb = new float[nFilt][nFreq];
        double step = (nFreq - 1.0) / (nFilt + 1.0);
        for (int m = 1; m <= nFilt; m++) {
            double left = (m - 1) * step;
            double center = m * step;
            double right = (m + 1) * step;
            for (int f = 0; f < nFreq; f++) {
                if (f >= left && f <= center && center > left) {
                    fb[m - 1][f] = (float) ((f - left) / (center - left));
                } else if (f > center && f <= right && right > center) {
                    fb[m - 1][f] = (float) ((right - f) / (right - center));
                }
            }
        }
        return fb;
    }

    /** Map out-of-range index for pad modes. */
    private static int mapPadIndex(int idx, int len, String mode) {
        if (len <= 0) return 0;
        if (idx >= 0 && idx < len) return idx;
        switch (mode) {
            case "replicate":
            case "edge":
                return idx < 0 ? 0 : len - 1;
            case "reflect": {
                if (len == 1) return 0;
                // mirror without repeating edge twice
                int period = 2 * (len - 1);
                int x = idx % period;
                if (x < 0) x += period;
                if (x >= len) x = period - x;
                return x;
            }
            case "circular":
            case "wrap": {
                int x = idx % len;
                if (x < 0) x += len;
                return x;
            }
            case "constant":
            default:
                return idx < 0 ? -1 : (idx >= len ? -1 : idx);
        }
    }

    

    private static Tensor reshapeLike(float[] data, Tensor ref) {
        long[] shape = AudioTensors.sizes(ref);
        Tensor t = torch.tensor(data);
        if (shape.length == 0) {
            return t;
        }
        long[] args = new long[shape.length];
        System.arraycopy(shape, 0, args, 0, shape.length);
        return t.reshape(args);
    }

    private static float clamp(float v, float lo, float hi) {
        if (v < lo) return lo;
        if (v > hi) return hi;
        return v;
    }

    private static double clamp(double v, double lo, double hi) {
        if (v < lo) return lo;
        if (v > hi) return hi;
        return v;
    }

    private static float fadeGain(int i, int len, String kind, boolean fadeIn) {
        if (len <= 1) {
            return 1f;
        }
        double x = fadeIn ? (double) i / (len - 1) : 1.0 - (double) i / (len - 1);
        switch (kind) {
            case "exponential":
                return (float) (Math.pow(2.0, x) - 1.0);
            case "logarithmic":
                return (float) Math.log1p(x * (Math.E - 1.0));
            case "quarter_sine":
                return (float) Math.sin(x * Math.PI / 2.0);
            case "half_sine":
                return (float) (0.5 * (1.0 - Math.cos(x * Math.PI)));
            case "linear":
            default:
                return (float) x;
        }
    }
}
