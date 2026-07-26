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
package org.bytedeco.pytorch.utils.librosa.core;

import org.bytedeco.pytorch.data.dataframe.dtype.AudioData;

import java.util.Objects;

/**
 * librosa.core spectrum helpers: STFT, magphase, power_to_db / db_to_power.
 */
public final class Spectrum {
    private static final int DEFAULT_N_FFT = 2048;
    private static final int DEFAULT_HOP = 512;

    private Spectrum() {}

    /**
     * Short-time Fourier transform. Returns complex STFT as magnitude and phase
     * packed into {@link StftResult}.
     */
    public static StftResult stft(float[] y, int sr) {
        return stft(y, sr, DEFAULT_N_FFT, DEFAULT_HOP, "hann");
    }

    public static StftResult stft(float[] y, int sr, int nFft, int hopLength, String window) {
        Objects.requireNonNull(y, "y");
        AudioData ad = new AudioData(y, sr > 0 ? sr : 22050, 1);
        AudioData.ComplexMatrix cm = ad.stft(nFft, hopLength, window == null ? "hann" : window);
        float[][] power = cm.powerSpectrum();
        int bins = power.length;
        int frames = bins > 0 ? power[0].length : 0;
        float[][] magnitude = new float[bins][frames];
        float[][] phase = new float[bins][frames];
        for (int b = 0; b < bins; b++) {
            for (int t = 0; t < frames; t++) {
                AudioData.Complex c = cm.get(b, t);
                magnitude[b][t] = (float) c.magnitude();
                phase[b][t] = (float) c.phase();
            }
        }
        return new StftResult(magnitude, phase, power, nFft, hopLength);
    }

    /** Split complex STFT into magnitude and phase (from power spectrum + zeros phase). */
    public static MagPhase magphase(float[][] D) {
        return magphase(D, 1.0);
    }

    /**
     * @param power 1 = treat D as magnitude, 2 = treat as power
     */
    public static MagPhase magphase(float[][] D, double power) {
        Objects.requireNonNull(D, "D");
        if (D.length == 0) {
            return new MagPhase(new float[0][0], new float[0][0]);
        }
        int bins = D.length;
        int frames = D[0].length;
        float[][] mag = new float[bins][frames];
        float[][] phase = new float[bins][frames];
        for (int b = 0; b < bins; b++) {
            for (int t = 0; t < frames; t++) {
                float v = D[b][t];
                if (power == 2.0) {
                    mag[b][t] = (float) Math.sqrt(Math.max(v, 0f));
                } else {
                    mag[b][t] = Math.abs(v);
                }
                phase[b][t] = 0f;
            }
        }
        return new MagPhase(mag, phase);
    }

    public static float[][] power_to_db(float[][] S) {
        return power_to_db(S, 1.0, 1e-10, 80.0);
    }

    public static float[][] power_to_db(float[][] S, double ref, double amin, double topDb) {
        Objects.requireNonNull(S, "S");
        if (S.length == 0) {
            return new float[0][0];
        }
        int bins = S.length;
        int frames = S[0].length;
        float[][] out = new float[bins][frames];
        double refVal = ref <= 0 ? 1.0 : ref;
        float maxDb = Float.NEGATIVE_INFINITY;
        for (int b = 0; b < bins; b++) {
            for (int t = 0; t < frames; t++) {
                double v = Math.max(S[b][t], amin);
                float db = (float) (10.0 * Math.log10(v / refVal));
                out[b][t] = db;
                if (db > maxDb) maxDb = db;
            }
        }
        if (topDb > 0) {
            float floor = maxDb - (float) topDb;
            for (int b = 0; b < bins; b++) {
                for (int t = 0; t < frames; t++) {
                    if (out[b][t] < floor) out[b][t] = floor;
                }
            }
        }
        return out;
    }

    public static float[][] powerToDb(float[][] S) {
        return power_to_db(S);
    }

    public static float[][] db_to_power(float[][] S_db) {
        return db_to_power(S_db, 1.0);
    }

    public static float[][] db_to_power(float[][] S_db, double ref) {
        Objects.requireNonNull(S_db, "S_db");
        if (S_db.length == 0) {
            return new float[0][0];
        }
        int bins = S_db.length;
        int frames = S_db[0].length;
        float[][] out = new float[bins][frames];
        for (int b = 0; b < bins; b++) {
            for (int t = 0; t < frames; t++) {
                out[b][t] = (float) (ref * Math.pow(10.0, S_db[b][t] / 10.0));
            }
        }
        return out;
    }

    public static float[][] dbToPower(float[][] S_db) {
        return db_to_power(S_db);
    }

    /** Amplitude spectrogram → dB (20 * log10). */
    public static float[][] amplitude_to_db(float[][] S) {
        return amplitude_to_db(S, 1.0, 1e-5, 80.0);
    }

    public static float[][] amplitude_to_db(float[][] S, double ref, double amin, double topDb) {
        Objects.requireNonNull(S, "S");
        if (S.length == 0) {
            return new float[0][0];
        }
        int bins = S.length;
        int frames = S[0].length;
        float[][] out = new float[bins][frames];
        double refVal = ref <= 0 ? 1.0 : ref;
        float maxDb = Float.NEGATIVE_INFINITY;
        for (int b = 0; b < bins; b++) {
            for (int t = 0; t < frames; t++) {
                double v = Math.max(Math.abs(S[b][t]), amin);
                float db = (float) (20.0 * Math.log10(v / refVal));
                out[b][t] = db;
                if (db > maxDb) maxDb = db;
            }
        }
        if (topDb > 0) {
            float floor = maxDb - (float) topDb;
            for (int b = 0; b < bins; b++) {
                for (int t = 0; t < frames; t++) {
                    if (out[b][t] < floor) out[b][t] = floor;
                }
            }
        }
        return out;
    }

    public static final class StftResult {
        public final float[][] magnitude;
        public final float[][] phase;
        public final float[][] power;
        public final int nFft;
        public final int hopLength;

        public StftResult(float[][] magnitude, float[][] phase, float[][] power, int nFft, int hopLength) {
            this.magnitude = magnitude;
            this.phase = phase;
            this.power = power;
            this.nFft = nFft;
            this.hopLength = hopLength;
        }

        public float[][] magnitude() {
            return magnitude;
        }

        public float[][] phase() {
            return phase;
        }

        public float[][] power() {
            return power;
        }
    }

    public static final class MagPhase {
        public final float[][] magnitude;
        public final float[][] phase;

        public MagPhase(float[][] magnitude, float[][] phase) {
            this.magnitude = magnitude;
            this.phase = phase;
        }

        public float[][] magnitude() {
            return magnitude;
        }

        public float[][] phase() {
            return phase;
        }
    }
}
