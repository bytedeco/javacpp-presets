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
package org.bytedeco.pytorch.llm.ktransformers.config;

import java.nio.file.Path;
import java.util.Objects;

/**
 * Three-tier (GPU–CPU–Disk) prefix-cache configuration.
 *
 * <p>Corresponds to upstream "3-layer prefix cache reuse". Builds on this
 * repository's {@code llm.kvcache.HierarchicalKvCache} and
 * {@code PrefixRadixCache} by adding an explicit disk tier.
 */
public final class KtCacheConfig {

    private final int gpuBlocks;
    private final int cpuBlocks;
    private final int diskBlocks;
    private final int blockSize;
    private final int numLayers;
    private final int numHeads;
    private final int headDim;
    private final boolean prefixEnable;
    private final Path diskPath;
    private final boolean asyncPrefetch;
    private final double gpuWatermark;
    private final double cpuWatermark;
    private final boolean quantizeCold;

    private KtCacheConfig(Builder b) {
        if (b.gpuBlocks < 1 || b.cpuBlocks < 1) {
            throw new IllegalArgumentException("gpuBlocks/cpuBlocks must be >= 1");
        }
        if (b.diskBlocks < 0) {
            throw new IllegalArgumentException("diskBlocks must be >= 0");
        }
        if (b.blockSize < 1 || b.numLayers < 1 || b.numHeads < 1 || b.headDim < 1) {
            throw new IllegalArgumentException("blockSize/numLayers/numHeads/headDim must be >= 1");
        }
        this.gpuBlocks = b.gpuBlocks;
        this.cpuBlocks = b.cpuBlocks;
        this.diskBlocks = b.diskBlocks;
        this.blockSize = b.blockSize;
        this.numLayers = b.numLayers;
        this.numHeads = b.numHeads;
        this.headDim = b.headDim;
        this.prefixEnable = b.prefixEnable;
        this.diskPath = b.diskPath;
        this.asyncPrefetch = b.asyncPrefetch;
        this.gpuWatermark = b.gpuWatermark;
        this.cpuWatermark = b.cpuWatermark;
        this.quantizeCold = b.quantizeCold;
    }

    public int gpuBlocks() { return gpuBlocks; }
    public int cpuBlocks() { return cpuBlocks; }
    public int diskBlocks() { return diskBlocks; }
    public int blockSize() { return blockSize; }
    public int numLayers() { return numLayers; }
    public int numHeads() { return numHeads; }
    public int headDim() { return headDim; }
    public boolean prefixEnable() { return prefixEnable; }
    public Path diskPath() { return diskPath; }
    public boolean asyncPrefetch() { return asyncPrefetch; }
    public double gpuWatermark() { return gpuWatermark; }
    public double cpuWatermark() { return cpuWatermark; }
    public boolean quantizeCold() { return quantizeCold; }

    public boolean diskEnabled() {
        return diskBlocks > 0 && diskPath != null;
    }

    public static Builder builder() { return new Builder(); }

    /** Small defaults suitable for unit tests / mini models. */
    public static KtCacheConfig mini() {
        return builder()
                .gpuBlocks(32).cpuBlocks(64).diskBlocks(128)
                .blockSize(16).numLayers(4).numHeads(4).headDim(32)
                .prefixEnable(true)
                .diskPath(Path.of(System.getProperty("java.io.tmpdir"), "kt-prefix-cache"))
                .build();
    }

    public static final class Builder {
        private int gpuBlocks = 256;
        private int cpuBlocks = 1024;
        private int diskBlocks = 4096;
        private int blockSize = 16;
        private int numLayers = 32;
        private int numHeads = 32;
        private int headDim = 128;
        private boolean prefixEnable = true;
        private Path diskPath = Path.of(System.getProperty("java.io.tmpdir"), "kt-prefix-cache");
        private boolean asyncPrefetch = true;
        private double gpuWatermark = 0.90;
        private double cpuWatermark = 0.85;
        private boolean quantizeCold = false;

        public Builder gpuBlocks(int v) { this.gpuBlocks = v; return this; }
        public Builder cpuBlocks(int v) { this.cpuBlocks = v; return this; }
        public Builder diskBlocks(int v) { this.diskBlocks = v; return this; }
        public Builder blockSize(int v) { this.blockSize = v; return this; }
        public Builder numLayers(int v) { this.numLayers = v; return this; }
        public Builder numHeads(int v) { this.numHeads = v; return this; }
        public Builder headDim(int v) { this.headDim = v; return this; }
        public Builder prefixEnable(boolean v) { this.prefixEnable = v; return this; }
        public Builder diskPath(Path v) { this.diskPath = v; return this; }
        public Builder asyncPrefetch(boolean v) { this.asyncPrefetch = v; return this; }
        public Builder gpuWatermark(double v) { this.gpuWatermark = v; return this; }
        public Builder cpuWatermark(double v) { this.cpuWatermark = v; return this; }
        public Builder quantizeCold(boolean v) { this.quantizeCold = v; return this; }

        public KtCacheConfig build() { return new KtCacheConfig(this); }
    }
}
