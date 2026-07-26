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
package org.bytedeco.pytorch.utils.librosa.util;
import org.bytedeco.pytorch.data.transforms.*;

import java.util.Objects;

/**
 * librosa.util helpers: frame, normalize, softmask.
 */
public final class Utils {
    private Utils() {}

    /**
     * Slice a 1-D signal into overlapping frames.
     * Returns {@code [frameLength, n_frames]} (column-major frames like librosa).
     */
    public static float[][] frame(float[] x, int frameLength, int hopLength) {
        Objects.requireNonNull(x, "x");
        if (frameLength <= 0 || hopLength <= 0) {
            throw new IllegalArgumentException("frameLength and hopLength must be > 0");
        }
        if (x.length < frameLength) {
            float[][] one = new float[frameLength][1];
            System.arraycopy(x, 0, one[0], 0, 0); // no-op clarity
            for (int i = 0; i < x.length; i++) {
                one[i][0] = x[i];
            }
            return one;
        }
        int nFrames = 1 + (x.length - frameLength) / hopLength;
        float[][] frames = new float[frameLength][nFrames];
        for (int f = 0; f < nFrames; f++) {
            int start = f * hopLength;
            for (int i = 0; i < frameLength; i++) {
                frames[i][f] = x[start + i];
            }
        }
        return frames;
    }

    /**
     * Normalize array.
     *
     * @param norm 1 = L1, 2 = L2, {@link Double#POSITIVE_INFINITY} = max abs
     * @param axis ignored for 1-D; for 2-D, 0 = across rows, 1 = across cols
     */
    public static float[] normalize(float[] x, double norm) {
        Objects.requireNonNull(x, "x");
        float[] out = x.clone();
        double n = computeNorm(out, norm);
        if (n > 1e-12) {
            for (int i = 0; i < out.length; i++) {
                out[i] = (float) (out[i] / n);
            }
        }
        return out;
    }

    public static float[][] normalize(float[][] X, double norm, int axis) {
        Objects.requireNonNull(X, "X");
        if (X.length == 0) {
            return new float[0][0];
        }
        int rows = X.length;
        int cols = X[0].length;
        float[][] out = new float[rows][cols];
        for (int r = 0; r < rows; r++) {
            System.arraycopy(X[r], 0, out[r], 0, cols);
        }
        if (axis == 0) {
            // normalize each column
            for (int c = 0; c < cols; c++) {
                float[] col = new float[rows];
                for (int r = 0; r < rows; r++) col[r] = out[r][c];
                double n = computeNorm(col, norm);
                if (n > 1e-12) {
                    for (int r = 0; r < rows; r++) {
                        out[r][c] = (float) (out[r][c] / n);
                    }
                }
            }
        } else {
            // normalize each row
            for (int r = 0; r < rows; r++) {
                double n = computeNorm(out[r], norm);
                if (n > 1e-12) {
                    for (int c = 0; c < cols; c++) {
                        out[r][c] = (float) (out[r][c] / n);
                    }
                }
            }
        }
        return out;
    }

    /**
     * Soft mask: {@code M = X^power / (X^power + X_ref^power)}.
     */
    public static float[][] softmask(float[][] X, float[][] X_ref) {
        return softmask(X, X_ref, 1.0, true);
    }

    public static float[][] softmask(float[][] X, float[][] X_ref, double power, boolean splitZeros) {
        Objects.requireNonNull(X, "X");
        Objects.requireNonNull(X_ref, "X_ref");
        if (X.length != X_ref.length) {
            throw new IllegalArgumentException("X and X_ref row mismatch");
        }
        if (X.length == 0) {
            return new float[0][0];
        }
        int rows = X.length;
        int cols = X[0].length;
        float[][] M = new float[rows][cols];
        for (int r = 0; r < rows; r++) {
            if (X[r].length != cols || X_ref[r].length != cols) {
                throw new IllegalArgumentException("ragged matrix at row " + r);
            }
            for (int c = 0; c < cols; c++) {
                double a = Math.pow(Math.max(X[r][c], 0f), power);
                double b = Math.pow(Math.max(X_ref[r][c], 0f), power);
                double den = a + b;
                if (den <= 1e-20) {
                    M[r][c] = splitZeros ? 0.5f : 0f;
                } else {
                    M[r][c] = (float) (a / den);
                }
            }
        }
        return M;
    }

    /** Pad 1-D array (constant mode). */
    public static float[] pad_center(float[] x, int size) {
        Objects.requireNonNull(x, "x");
        if (size <= x.length) {
            float[] out = new float[size];
            System.arraycopy(x, 0, out, 0, size);
            return out;
        }
        float[] out = new float[size];
        int start = (size - x.length) / 2;
        System.arraycopy(x, 0, out, start, x.length);
        return out;
    }

    public static float[] padCenter(float[] x, int size) {
        return pad_center(x, size);
    }

    /** Valid finite check (librosa.util.valid_audio simplified). */
    public static boolean valid_audio(float[] y) {
        if (y == null || y.length == 0) return false;
        for (float v : y) {
            if (Float.isNaN(v) || Float.isInfinite(v)) return false;
        }
        return true;
    }

    public static boolean validAudio(float[] y) {
        return valid_audio(y);
    }

    private static double computeNorm(float[] x, double norm) {
        if (Double.isInfinite(norm)) {
            float m = 0f;
            for (float v : x) {
                float a = Math.abs(v);
                if (a > m) m = a;
            }
            return m;
        }
        if (norm == 1.0) {
            double s = 0;
            for (float v : x) s += Math.abs(v);
            return s;
        }
        // default L2
        double s = 0;
        for (float v : x) s += (double) v * v;
        return Math.sqrt(s);
    }
}
