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
package org.bytedeco.pytorch.llm.vllm.multimodal.encoders;

import org.bytedeco.pytorch.llm.vllm.multimodal.MediaInput;
import org.bytedeco.pytorch.llm.vllm.multimodal.MediaType;

import java.nio.file.Path;

/**
 * Real multimodal encoder contract (DINOv2 / CLIP / Whisper / SmolVLM).
 *
 * <p>{@link #encode(MediaInput)} returns continuous features; callers may also
 * project them to discrete token ids for the text LM path via
 * {@link EncoderFeatures#toTokenIds(int, int)}.
 */
public interface MediaEncoder extends AutoCloseable {

    MediaType modality();

    /** Human-readable model id / path (not Module.name which returns BytePointer). */
    String encoderName();

    /** Feature dimension of the pooled embedding (or per-token dim). */
    int featureDim();

    /**
     * Encode one media input into continuous features.
     *
     * @return non-null features; empty tensor if encode fails and soft-fail is enabled
     */
    EncoderFeatures encode(MediaInput input);

    /** Whether this encoder can handle the given input. */
    default boolean supports(MediaInput input) {
        return input != null && input.type == modality();
    }

    @Override
    default void close() {}

    /** Result of a real encoder forward. */
    final class EncoderFeatures {
        /** Pooled embedding {@code [D]} (may be empty). */
        public final float[] pooled;
        /** Optional sequence features {@code [T][D]} (patch / frame tokens). */
        public final float[][] sequence;
        /** Encoder that produced this (for logging). */
        public final String source;
        /** Wall-clock encode time in ms. */
        public final double encodeMs;

        public EncoderFeatures(float[] pooled, float[][] sequence, String source, double encodeMs) {
            this.pooled = pooled == null ? new float[0] : pooled;
            this.sequence = sequence == null ? new float[0][] : sequence;
            this.source = source == null ? "" : source;
            this.encodeMs = encodeMs;
        }

        public static EncoderFeatures empty(String source) {
            return new EncoderFeatures(new float[0], new float[0][], source, 0);
        }

        public boolean isEmpty() {
            return pooled.length == 0 && sequence.length == 0;
        }

        public int dim() {
            if (pooled.length > 0) return pooled.length;
            if (sequence.length > 0 && sequence[0] != null) return sequence[0].length;
            return 0;
        }

        public int seqLen() {
            return sequence.length;
        }

        /**
         * Project continuous features into {@code nTokens} discrete token ids so the
         * text LM path sees media-dependent tokens (not fixed placeholders).
         *
         * <p>IDs are restricted to a <b>safe band</b> that avoids special/EOS tokens
         * (Qwen: {@code <|endoftext|>=151643}, {@code <|im_end|>=151645}, vision pads…).
         * Hashing into the full {@code [0, vocab)} range previously produced EOS in the
         * prompt, causing immediate empty generations after strip.
         */
        public int[] toTokenIds(int nTokens, int vocab) {
            if (nTokens <= 0 || vocab <= 1) return new int[0];
            float[] feat = pooled;
            if (feat.length == 0 && sequence.length > 0) {
                // mean-pool sequence
                int d = sequence[0] == null ? 0 : sequence[0].length;
                feat = new float[d];
                int n = 0;
                for (float[] row : sequence) {
                    if (row == null) continue;
                    for (int i = 0; i < d && i < row.length; i++) feat[i] += row[i];
                    n++;
                }
                if (n > 0) for (int i = 0; i < d; i++) feat[i] /= n;
            }
            // Safe content band: skip low control ids and high special-token range.
            // Qwen2/3 specials start ~151643; keep headroom for smaller vocabs too.
            int specialFloor = Math.min(vocab - 1, 151_000);
            int lo = Math.min(100, Math.max(1, vocab / 100));
            int hi = Math.max(lo + 1, Math.min(specialFloor, vocab - 1));
            int span = Math.max(1, hi - lo);
            if (feat.length == 0) {
                int[] zeros = new int[nTokens];
                for (int i = 0; i < nTokens; i++) zeros[i] = lo + (i % span);
                return zeros;
            }
            // L2 normalize
            double norm = 0;
            for (float v : feat) norm += (double) v * v;
            norm = Math.sqrt(norm);
            if (norm < 1e-8) norm = 1;
            int[] ids = new int[nTokens];
            int chunk = Math.max(1, feat.length / nTokens);
            for (int t = 0; t < nTokens; t++) {
                int start = Math.min(t * chunk, feat.length - 1);
                int end = Math.min(start + chunk, feat.length);
                long h = 1125899906842597L ^ t;
                for (int i = start; i < end; i++) {
                    // quantize to keep stable across tiny float noise
                    int q = Math.round(feat[i] / (float) norm * 1000f);
                    h = 31 * h + q;
                }
                int id = lo + (int) Math.floorMod(h, (long) span);
                if (id < lo) id = lo;
                if (id >= hi) id = hi - 1;
                ids[t] = id;
            }
            return ids;
        }

        @Override
        public String toString() {
            return "EncoderFeatures{src=" + source + ", dim=" + dim()
                    + ", seq=" + seqLen() + ", ms=" + String.format("%.1f", encodeMs) + "}";
        }
    }

    /** Optional factory from a local HF snapshot directory. */
    interface Factory {
        MediaEncoder load(Path dir) throws Exception;
    }
}
