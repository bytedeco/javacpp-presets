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
package org.bytedeco.pytorch.utils.librosa.feature;

import org.bytedeco.pytorch.dataframe.dtype.AudioData;
import org.bytedeco.pytorch.utils.librosa.Librosa;

import java.util.Objects;

/**
 * librosa.feature-style extractors. All methods take {@code float[] y, int sr}
 * and return {@code float[][]} ({@code [n_features, n_frames]}) or {@code float[]}
 * time series. DSP is delegated to {@link AudioData}.
 */
public final class Feature {
    private static final int DEFAULT_N_FFT = 2048;
    private static final int DEFAULT_HOP = 512;
    private static final int DEFAULT_N_MELS = 128;
    private static final int DEFAULT_N_MFCC = 13;

    private Feature() {}

    public static float[][] mfcc(float[] y, int sr) {
        return mfcc(y, sr, DEFAULT_N_MFCC);
    }

    public static float[][] mfcc(float[] y, int sr, int nMfcc) {
        return mfcc(y, sr, nMfcc, DEFAULT_N_FFT, DEFAULT_HOP, DEFAULT_N_MELS);
    }

    public static float[][] mfcc(float[] y, int sr, int nMfcc, int nFft, int hopLength, int nMels) {
        AudioData ad = audio(y, sr);
        return ad.mfcc(nMfcc, nFft, hopLength, nMels, 0.0, sr / 2.0);
    }

    public static float[][] melspectrogram(float[] y, int sr) {
        return melspectrogram(y, sr, DEFAULT_N_MELS);
    }

    public static float[][] melspectrogram(float[] y, int sr, int nMels) {
        return melspectrogram(y, sr, nMels, DEFAULT_N_FFT, DEFAULT_HOP, 0.0, sr / 2.0);
    }

    public static float[][] melspectrogram(float[] y, int sr, int nMels, int nFft, int hopLength,
                                          double fMin, double fMax) {
        AudioData ad = audio(y, sr);
        return ad.melSpectrogram(nFft, hopLength, nMels, fMin, fMax);
    }

    public static float[][] mel_spectrogram(float[] y, int sr) {
        return melspectrogram(y, sr);
    }

    public static float[][] chroma_stft(float[] y, int sr) {
        return chroma_stft(y, sr, DEFAULT_N_FFT, DEFAULT_HOP);
    }

    public static float[][] chroma_stft(float[] y, int sr, int nFft, int hopLength) {
        AudioData ad = audio(y, sr);
        return ad.chroma(nFft, hopLength);
    }

    public static float[][] chromaStft(float[] y, int sr) {
        return chroma_stft(y, sr);
    }

    public static float[] spectral_centroid(float[] y, int sr) {
        return spectral_centroid(y, sr, DEFAULT_N_FFT, DEFAULT_HOP);
    }

    public static float[] spectral_centroid(float[] y, int sr, int nFft, int hopLength) {
        AudioData ad = audio(y, sr);
        return ad.spectralCentroid(nFft, hopLength);
    }

    public static float[] spectralCentroid(float[] y, int sr) {
        return spectral_centroid(y, sr);
    }

    public static float[] spectral_bandwidth(float[] y, int sr) {
        AudioData ad = audio(y, sr);
        return ad.spectralBandwidth();
    }

    public static float[] spectralBandwidth(float[] y, int sr) {
        return spectral_bandwidth(y, sr);
    }

    public static float[][] spectral_contrast(float[] y, int sr) {
        return spectral_contrast(y, sr, DEFAULT_N_FFT, DEFAULT_HOP, 6);
    }

    public static float[][] spectral_contrast(float[] y, int sr, int nFft, int hopLength, int nBands) {
        AudioData ad = audio(y, sr);
        return ad.spectralContrast(nFft, hopLength, nBands);
    }

    public static float[][] spectralContrast(float[] y, int sr) {
        return spectral_contrast(y, sr);
    }

    public static float[] spectral_rolloff(float[] y, int sr) {
        return spectral_rolloff(y, sr, 0.85f);
    }

    public static float[] spectral_rolloff(float[] y, int sr, float rollPercent) {
        AudioData ad = audio(y, sr);
        return ad.spectralRolloff(rollPercent);
    }

    public static float[] spectralRolloff(float[] y, int sr) {
        return spectral_rolloff(y, sr);
    }

    public static float[] zero_crossing_rate(float[] y) {
        return zero_crossing_rate(y, 2048);
    }

    public static float[] zero_crossing_rate(float[] y, int frameLength) {
        AudioData ad = audio(y, Librosa.DEFAULT_SR);
        return ad.zeroCrossingRate(frameLength);
    }

    public static float[] zeroCrossingRate(float[] y) {
        return zero_crossing_rate(y);
    }

    public static float[] rms(float[] y) {
        return rms(y, 2048);
    }

    public static float[] rms(float[] y, int frameLength) {
        AudioData ad = audio(y, Librosa.DEFAULT_SR);
        return ad.rmsEnergy(frameLength);
    }

    /** Alias for {@link #rms(float[])}. */
    public static float[] rms_energy(float[] y) {
        return rms(y);
    }

    private static AudioData audio(float[] y, int sr) {
        Objects.requireNonNull(y, "y");
        if (sr <= 0) {
            throw new IllegalArgumentException("sr must be > 0");
        }
        return new AudioData(y, sr, 1);
    }
}
