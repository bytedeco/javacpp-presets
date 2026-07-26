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
package org.bytedeco.pytorch.llm.trl.config;

/** Supervised fine-tuning config (Hugging Face TRL {@code SFTConfig} subset). */
public final class SFTConfig extends TrainerConfig {
    private final int maxSeqLength;
    private final long ignoreIndex;
    private final boolean packing;

    private SFTConfig(Builder b) {
        super(b);
        this.maxSeqLength = b.maxSeqLength;
        this.ignoreIndex = b.ignoreIndex;
        this.packing = b.packing;
    }

    public int maxSeqLength() { return maxSeqLength; }
    public long ignoreIndex() { return ignoreIndex; }
    public boolean packing() { return packing; }

    public static Builder builder() { return new Builder(); }

    public static final class Builder extends TrainerConfig.Builder<Builder> {
        private int maxSeqLength = 2048;
        private long ignoreIndex = -100L;
        private boolean packing = false;

        public Builder maxSeqLength(int v) { this.maxSeqLength = v; return this; }
        public Builder ignoreIndex(long v) { this.ignoreIndex = v; return this; }
        public Builder packing(boolean v) { this.packing = v; return this; }

        @Override
        public SFTConfig build() { return new SFTConfig(this); }
    }
}
