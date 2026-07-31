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
package org.bytedeco.pytorch.audio.librosa.onset;

import org.bytedeco.pytorch.dataframe.dtype.AudioData;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * librosa.onset-style onset strength and peak picking.
 */
public final class Onset {
    private static final int DEFAULT_HOP = 512;

    private Onset() {}

    /** Spectral-flux onset strength envelope. */
    public static float[] onset_strength(float[] y, int sr) {
        Objects.requireNonNull(y, "y");
        AudioData ad = new AudioData(y, sr > 0 ? sr : 22050, 1);
        return ad.onsetStrength();
    }

    public static float[] onsetStrength(float[] y, int sr) {
        return onset_strength(y, sr);
    }

    /**
     * Detect onset frame indices by peak-picking the onset strength envelope.
     *
     * @param wait      minimum frames between onsets
     * @param delta     threshold above local mean
     * @param preMax    pre-window for local max (frames)
     * @param postMax   post-window for local max (frames)
     */
    public static int[] onset_detect(float[] y, int sr, int hopLength,
                                     int wait, float delta, int preMax, int postMax) {
        float[] oenv = onset_strength(y, sr);
        return peakPick(oenv, wait, delta, preMax, postMax);
    }

    public static int[] onset_detect(float[] y, int sr) {
        return onset_detect(y, sr, DEFAULT_HOP, 1, 0.07f, 1, 1);
    }

    public static int[] onsetDetect(float[] y, int sr) {
        return onset_detect(y, sr);
    }

    /** Convert onset frames to times (seconds). */
    public static float[] frames_to_time(int[] frames, int sr, int hopLength) {
        Objects.requireNonNull(frames, "frames");
        int hop = hopLength > 0 ? hopLength : DEFAULT_HOP;
        int rate = sr > 0 ? sr : 22050;
        float[] times = new float[frames.length];
        for (int i = 0; i < frames.length; i++) {
            times[i] = (float) frames[i] * hop / rate;
        }
        return times;
    }

    public static float[] framesToTime(int[] frames, int sr) {
        return frames_to_time(frames, sr, DEFAULT_HOP);
    }

    /**
     * Simple peak picker: local maxima above mean + delta, with wait spacing.
     */
    public static int[] peakPick(float[] x, int wait, float delta, int preMax, int postMax) {
        Objects.requireNonNull(x, "x");
        if (x.length == 0) {
            return new int[0];
        }
        int pre = Math.max(0, preMax);
        int post = Math.max(0, postMax);
        int minWait = Math.max(1, wait);

        // global mean for threshold
        double sum = 0;
        for (float v : x) sum += v;
        float mean = (float) (sum / x.length);
        float thresh = mean + delta;

        List<Integer> peaks = new ArrayList<>();
        int last = -minWait;
        for (int i = 0; i < x.length; i++) {
            if (x[i] < thresh) continue;
            if (i - last < minWait) continue;
            boolean isMax = true;
            int lo = Math.max(0, i - pre);
            int hi = Math.min(x.length - 1, i + post);
            for (int j = lo; j <= hi; j++) {
                if (j != i && x[j] > x[i]) {
                    isMax = false;
                    break;
                }
            }
            if (isMax) {
                peaks.add(i);
                last = i;
            }
        }
        int[] out = new int[peaks.size()];
        for (int i = 0; i < peaks.size(); i++) {
            out[i] = peaks.get(i);
        }
        return out;
    }
}
