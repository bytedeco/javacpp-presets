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

package org.bytedeco.pytorch.llm.unsloth.studio.train;

import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Long-context training policy: gradient checkpointing, sequence slicing, RoPE hints.
 * Aligns with Studio claims of faster long-context FT via configuration — measured locally.
 */
public final class LongContextPolicy {

    public static final class Advice {
        public final boolean gradientCheckpointing;
        public final int microSeqLen;
        public final int chunks;
        public final boolean ropeScaling;
        public final double ropeFactor;
        public final String notes;

        public Advice(boolean gradientCheckpointing, int microSeqLen, int chunks,
                      boolean ropeScaling, double ropeFactor, String notes) {
            this.gradientCheckpointing = gradientCheckpointing;
            this.microSeqLen = microSeqLen;
            this.chunks = chunks;
            this.ropeScaling = ropeScaling;
            this.ropeFactor = ropeFactor;
            this.notes = notes;
        }

        public Map<String, Object> toMap() {
            Map<String, Object> m = new LinkedHashMap<>();
            m.put("gradient_checkpointing", gradientCheckpointing);
            m.put("micro_seq_len", microSeqLen);
            m.put("chunks", chunks);
            m.put("rope_scaling", ropeScaling);
            m.put("rope_factor", ropeFactor);
            m.put("notes", notes);
            return m;
        }
    }

    private LongContextPolicy() {}

    public static Advice advise(int maxSeqLength, long vramMb) {
        boolean ckpt = maxSeqLength >= 2048 || vramMb > 0 && vramMb < 24000;
        int micro = maxSeqLength;
        int chunks = 1;
        if (maxSeqLength > 8192) {
            micro = 4096;
            chunks = (int) Math.ceil(maxSeqLength / (double) micro);
            ckpt = true;
        } else if (maxSeqLength > 4096) {
            micro = 2048;
            chunks = (int) Math.ceil(maxSeqLength / (double) micro);
            ckpt = true;
        }
        boolean rope = maxSeqLength > 8192;
        double factor = rope ? Math.max(2.0, maxSeqLength / 4096.0) : 1.0;
        String notes = "Enable grad checkpointing for seq>2k; slice when seq>4k; RoPE scale when seq>8k.";
        return new Advice(ckpt, micro, chunks, rope, factor, notes);
    }
}
