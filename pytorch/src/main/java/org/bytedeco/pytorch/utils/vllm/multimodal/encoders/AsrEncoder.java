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
package org.bytedeco.pytorch.utils.vllm.multimodal.encoders;

import org.bytedeco.pytorch.utils.vllm.multimodal.MediaInput;
import org.bytedeco.pytorch.utils.vllm.multimodal.MediaType;

import java.util.Objects;

/**
 * ASR-oriented audio encoder wrapper around Whisper (or any audio MediaEncoder).
 *
 * <p>Full Whisper decoder CTC/seq2seq transcription is not wired into the text
 * CausalLM path; this encoder exposes real continuous encoder features that the
 * multimodal processor projects into discrete token ids for the LM, and keeps a
 * short "pseudo transcript cue" derived from feature energy for human-readable
 * logs / result dumps.
 *
 * <p>Use for ASR stress path: wav → Whisper encoder → feature-hash tokens → text LM.
 */
public final class AsrEncoder implements MediaEncoder {

    public static final int DEFAULT_TOKEN_BUDGET_HINT = 128;

    private final MediaEncoder audio;
    private final String name;
    private String lastCue = "";

    public AsrEncoder(MediaEncoder audio) {
        this.audio = Objects.requireNonNull(audio, "audio");
        this.name = "asr/" + audio.encoderName();
    }

    public static AsrEncoder wrap(MediaEncoder audio) {
        return new AsrEncoder(audio);
    }

    public MediaEncoder audioEncoder() {
        return audio;
    }

    /** Last human-readable energy cue (not true transcript). */
    public String lastCue() {
        return lastCue;
    }

    @Override
    public MediaType modality() {
        return MediaType.AUDIO;
    }

    @Override
    public String encoderName() {
        return name;
    }

    @Override
    public int featureDim() {
        return audio.featureDim();
    }

    @Override
    public boolean supports(MediaInput input) {
        return input != null && input.type == MediaType.AUDIO;
    }

    @Override
    public EncoderFeatures encode(MediaInput input) {
        long t0 = System.nanoTime();
        if (input == null) {
            lastCue = "";
            return EncoderFeatures.empty(encoderName());
        }
        try {
            EncoderFeatures base = audio.encode(input);
            if (base == null || base.isEmpty()) {
                lastCue = "(empty)";
                return EncoderFeatures.empty(encoderName());
            }
            lastCue = energyCue(base);
            // Keep sequence denser for ASR (more temporal tokens in hash)
            float[] pooled = base.pooled;
            float[][] seq = base.sequence;
            if (seq == null || seq.length == 0) {
                seq = new float[][]{pooled};
            }
            double ms = (System.nanoTime() - t0) / 1e6;
            return new EncoderFeatures(pooled, seq, encoderName() + " cue=" + lastCue, ms);
        } catch (Exception e) {
            System.out.println("[AsrEncoder] encode failed: " + e.getMessage());
            lastCue = "(error)";
            return EncoderFeatures.empty(encoderName());
        }
    }

    /**
     * Build a short energy/activity cue from pooled features for logging and
     * human verification dumps (not a real transcript).
     */
    public static String energyCue(EncoderFeatures feat) {
        if (feat == null || feat.isEmpty()) return "silence";
        float[] p = feat.pooled;
        if (p == null || p.length == 0) {
            if (feat.sequence != null && feat.sequence.length > 0) p = feat.sequence[0];
        }
        if (p == null || p.length == 0) return "silence";
        double energy = 0;
        double absMax = 0;
        int peaks = 0;
        for (float v : p) {
            double a = Math.abs(v);
            energy += a * a;
            if (a > absMax) absMax = a;
            if (a > 0.15) peaks++;
        }
        energy = Math.sqrt(energy / p.length);
        String activity;
        if (energy < 0.02) activity = "silence";
        else if (energy < 0.08) activity = "soft";
        else if (energy < 0.25) activity = "speech-like";
        else activity = "loud";
        int bins = Math.min(8, p.length);
        StringBuilder hash = new StringBuilder();
        int chunk = Math.max(1, p.length / bins);
        for (int b = 0; b < bins; b++) {
            double s = 0;
            int n = 0;
            for (int i = b * chunk; i < Math.min(p.length, (b + 1) * chunk); i++) {
                s += p[i];
                n++;
            }
            int q = n == 0 ? 0 : (int) Math.floorMod(Math.round((s / n) * 10), 16L);
            hash.append(Integer.toHexString(q));
        }
        return activity + "/" + hash + "/pk=" + peaks;
    }

    @Override
    public void close() {
        // audio owned by registry
    }
}
