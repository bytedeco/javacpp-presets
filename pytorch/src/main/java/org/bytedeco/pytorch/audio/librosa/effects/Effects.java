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
package org.bytedeco.pytorch.audio.librosa.effects;

import org.bytedeco.pytorch.dataframe.dtype.AudioData;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * librosa.effects-style audio effects: trim, split, preemphasis.
 */
public final class Effects {
    private Effects() {}

    /**
     * Trim leading/trailing silence by energy threshold.
     * Returns {@code [trimmed, (start, end)]} sample indices.
     */
    public static TrimResult trim(float[] y) {
        return trim(y, 22050, 60.0f, 0.1f);
    }

    /**
     * @param topDb  threshold in dB below reference peak
     * @param frameLength seconds of analysis window (approx)
     */
    public static TrimResult trim(float[] y, int sr, float topDb, float frameSec) {
        Objects.requireNonNull(y, "y");
        if (y.length == 0) {
            return new TrimResult(new float[0], 0, 0);
        }
        int frame = Math.max(1, (int) (Math.max(0.01f, frameSec) * Math.max(1, sr)));
        float peak = 0f;
        for (float v : y) {
            float a = Math.abs(v);
            if (a > peak) peak = a;
        }
        if (peak < 1e-12f) {
            return new TrimResult(new float[0], 0, 0);
        }
        double thresh = peak * Math.pow(10.0, -topDb / 20.0);

        int start = 0;
        while (start < y.length) {
            int end = Math.min(y.length, start + frame);
            if (frameEnergy(y, start, end) >= thresh) break;
            start = end;
        }
        int stop = y.length;
        while (stop > start) {
            int begin = Math.max(start, stop - frame);
            if (frameEnergy(y, begin, stop) >= thresh) break;
            stop = begin;
        }
        if (start >= stop) {
            return new TrimResult(new float[0], start, stop);
        }
        float[] out = new float[stop - start];
        System.arraycopy(y, start, out, 0, out.length);
        return new TrimResult(out, start, stop);
    }

    /** Time-range trim via {@link AudioData#trim(float, float)}. */
    public static float[] trim_time(float[] y, int sr, float startSec, float endSec) {
        Objects.requireNonNull(y, "y");
        AudioData ad = new AudioData(y, sr > 0 ? sr : 22050, 1);
        AudioData trimmed = ad.trim(startSec, endSec);
        float[] s = trimmed.getSamples();
        return s == null ? new float[0] : s;
    }

    /**
     * Split non-silent intervals (energy VAD).
     * Returns list of {@code [start, end)} sample index pairs.
     */
    public static int[][] split(float[] y, int sr) {
        return split(y, sr, 60.0f, 0.1f, 0.1f);
    }

    /**
     * @param topDb        silence threshold relative to peak (dB)
     * @param frameSec     analysis frame length
     * @param hopSec       hop between frames
     */
    public static int[][] split(float[] y, int sr, float topDb, float frameSec, float hopSec) {
        Objects.requireNonNull(y, "y");
        if (y.length == 0) {
            return new int[0][];
        }
        int rate = sr > 0 ? sr : 22050;
        int frame = Math.max(1, (int) (frameSec * rate));
        int hop = Math.max(1, (int) (hopSec * rate));

        float peak = 0f;
        for (float v : y) {
            float a = Math.abs(v);
            if (a > peak) peak = a;
        }
        double thresh = peak * Math.pow(10.0, -topDb / 20.0);

        List<int[]> intervals = new ArrayList<>();
        boolean inSpeech = false;
        int segStart = 0;
        for (int i = 0; i < y.length; i += hop) {
            int end = Math.min(y.length, i + frame);
            boolean active = frameEnergy(y, i, end) >= thresh;
            if (active && !inSpeech) {
                inSpeech = true;
                segStart = i;
            } else if (!active && inSpeech) {
                inSpeech = false;
                intervals.add(new int[]{segStart, i});
            }
        }
        if (inSpeech) {
            intervals.add(new int[]{segStart, y.length});
        }
        return intervals.toArray(new int[0][]);
    }

    /** First-order pre-emphasis filter: {@code y[n] - coef * y[n-1]}. */
    public static float[] preemphasis(float[] y) {
        return preemphasis(y, 0.97f);
    }

    public static float[] preemphasis(float[] y, float coef) {
        Objects.requireNonNull(y, "y");
        if (y.length == 0) {
            return new float[0];
        }
        float[] out = new float[y.length];
        out[0] = y[0];
        for (int i = 1; i < y.length; i++) {
            out[i] = y[i] - coef * y[i - 1];
        }
        return out;
    }

    /** Peak normalize via {@link AudioData#normalize()}. */
    public static float[] normalize(float[] y, int sr) {
        Objects.requireNonNull(y, "y");
        AudioData ad = new AudioData(y, sr > 0 ? sr : 22050, 1);
        AudioData n = ad.normalize();
        float[] s = n.getSamples();
        return s == null ? new float[0] : s;
    }

    private static double frameEnergy(float[] y, int start, int end) {
        double e = 0;
        int n = Math.max(0, end - start);
        for (int i = start; i < end; i++) {
            e += Math.abs(y[i]);
        }
        return n == 0 ? 0 : e / n;
    }

    public static final class TrimResult {
        public final float[] y;
        public final int start;
        public final int end;

        public TrimResult(float[] y, int start, int end) {
            this.y = y;
            this.start = start;
            this.end = end;
        }

        public float[] y() {
            return y;
        }

        public int[] index() {
            return new int[]{start, end};
        }

        @Override
        public String toString() {
            return "TrimResult{start=" + start + ", end=" + end + ", len=" + y.length + "}";
        }
    }
}
