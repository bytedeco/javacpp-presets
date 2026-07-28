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
package org.bytedeco.pytorch.utils.librosa.beat;

import org.bytedeco.pytorch.dataframe.dtype.AudioData;

import java.util.Objects;

/**
 * librosa.beat-style tempo and beat tracking.
 */
public final class Beat {
    private Beat() {}

    /**
     * Track beats: returns tempo (BPM) and beat frame indices / times.
     * {@code tempo, beats = librosa.beat.beat_track(y=y, sr=sr)}
     */
    public static BeatTrackResult beat_track(float[] y, int sr) {
        Objects.requireNonNull(y, "y");
        AudioData ad = new AudioData(y, sr > 0 ? sr : 22050, 1);
        AudioData.BeatTrackResult r = ad.beatTrack();
        // AudioData returns beat positions; treat as times in seconds when values look continuous,
        // otherwise as frame indices converted with hop=512.
        float[] beats = r.beats == null ? new float[0] : r.beats;
        float[] times = new float[beats.length];
        float hopSec = 512f / Math.max(1, ad.getSampleRate());
        boolean looksLikeFrames = true;
        for (float b : beats) {
            if (b > 0 && b < 1.0f) {
                looksLikeFrames = false;
                break;
            }
            if (b > ad.getSamples().length) {
                looksLikeFrames = false;
                break;
            }
        }
        for (int i = 0; i < beats.length; i++) {
            times[i] = looksLikeFrames ? beats[i] * hopSec : beats[i];
        }
        return new BeatTrackResult(r.tempo, beats, times);
    }

    public static BeatTrackResult beatTrack(float[] y, int sr) {
        return beat_track(y, sr);
    }

    /** Estimate global tempo in BPM. */
    public static float tempo(float[] y, int sr) {
        return beat_track(y, sr).tempo;
    }

    public static final class BeatTrackResult {
        public final float tempo;
        public final float[] beats;
        public final float[] beatTimes;

        public BeatTrackResult(float tempo, float[] beats, float[] beatTimes) {
            this.tempo = tempo;
            this.beats = beats == null ? new float[0] : beats;
            this.beatTimes = beatTimes == null ? new float[0] : beatTimes;
        }

        public float tempo() {
            return tempo;
        }

        public float[] beats() {
            return beats;
        }

        public float[] beat_times() {
            return beatTimes;
        }

        @Override
        public String toString() {
            return String.format("BeatTrackResult[tempo=%.1f BPM, beats=%d]", tempo, beats.length);
        }
    }
}
