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
package org.bytedeco.pytorch.llm.llamafactory.hparams;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;

/** Dataset / template / packing args (LLaMA-Factory data section). */
public final class DataArgs {
    private final String dataset;
    private final String datasetDir;
    private final String template;
    private final int cutoffLen;
    private final boolean trainOnPrompt;
    private final boolean maskHistory;
    private final boolean streaming;
    private final int bufferSize;
    private final String mixStrategy;
    private final String interleaveProbs;
    private final boolean overwriteCache;
    private final int preprocessingNumWorkers;
    private final int maxSamples;
    private final String evalDataset;
    private final boolean evalOnEachDataset;
    private final boolean packing;
    private final boolean neatPacking;
    private final String toolFormat;
    private final String tokenizedPath;
    private final String mediaDir;
    private final int imageMaxPixels;
    private final int videoMaxPixels;
    private final double videoFps;
    private final int videoMaxFrames;
    private final boolean ignorePadTokenForLoss;
    private final String defaultSystem;

    private DataArgs(Builder b) {
        this.dataset = Objects.requireNonNull(b.dataset, "dataset");
        this.datasetDir = b.datasetDir == null ? "data" : b.datasetDir;
        this.template = b.template == null ? "default" : b.template;
        this.cutoffLen = b.cutoffLen;
        this.trainOnPrompt = b.trainOnPrompt;
        this.maskHistory = b.maskHistory;
        this.streaming = b.streaming;
        this.bufferSize = b.bufferSize;
        this.mixStrategy = b.mixStrategy == null ? "concat" : b.mixStrategy;
        this.interleaveProbs = b.interleaveProbs;
        this.overwriteCache = b.overwriteCache;
        this.preprocessingNumWorkers = b.preprocessingNumWorkers;
        this.maxSamples = b.maxSamples;
        this.evalDataset = b.evalDataset;
        this.evalOnEachDataset = b.evalOnEachDataset;
        this.packing = b.packing;
        this.neatPacking = b.neatPacking;
        this.toolFormat = b.toolFormat == null ? "default" : b.toolFormat;
        this.tokenizedPath = b.tokenizedPath;
        this.mediaDir = b.mediaDir;
        this.imageMaxPixels = b.imageMaxPixels;
        this.videoMaxPixels = b.videoMaxPixels;
        this.videoFps = b.videoFps;
        this.videoMaxFrames = b.videoMaxFrames;
        this.ignorePadTokenForLoss = b.ignorePadTokenForLoss;
        this.defaultSystem = b.defaultSystem;
    }

    public String dataset() { return dataset; }
    public String datasetDir() { return datasetDir; }
    public String template() { return template; }
    public int cutoffLen() { return cutoffLen; }
    public boolean trainOnPrompt() { return trainOnPrompt; }
    public boolean maskHistory() { return maskHistory; }
    public boolean streaming() { return streaming; }
    public int bufferSize() { return bufferSize; }
    public String mixStrategy() { return mixStrategy; }
    public String interleaveProbs() { return interleaveProbs; }
    public boolean overwriteCache() { return overwriteCache; }
    public int preprocessingNumWorkers() { return preprocessingNumWorkers; }
    public int maxSamples() { return maxSamples; }
    public String evalDataset() { return evalDataset; }
    public boolean evalOnEachDataset() { return evalOnEachDataset; }
    public boolean packing() { return packing; }
    public boolean neatPacking() { return neatPacking; }
    public String toolFormat() { return toolFormat; }
    public String tokenizedPath() { return tokenizedPath; }
    public String mediaDir() { return mediaDir; }
    public int imageMaxPixels() { return imageMaxPixels; }
    public int videoMaxPixels() { return videoMaxPixels; }
    public double videoFps() { return videoFps; }
    public int videoMaxFrames() { return videoMaxFrames; }
    public boolean ignorePadTokenForLoss() { return ignorePadTokenForLoss; }
    public String defaultSystem() { return defaultSystem; }

    public Map<String, Object> toMap() {
        Map<String, Object> m = new LinkedHashMap<>();
        HparamsMaps.put(m, "dataset", dataset);
        HparamsMaps.put(m, "dataset_dir", datasetDir);
        HparamsMaps.put(m, "template", template);
        HparamsMaps.put(m, "cutoff_len", cutoffLen);
        HparamsMaps.put(m, "train_on_prompt", trainOnPrompt);
        HparamsMaps.put(m, "mask_history", maskHistory);
        HparamsMaps.put(m, "streaming", streaming);
        HparamsMaps.put(m, "buffer_size", bufferSize);
        HparamsMaps.put(m, "mix_strategy", mixStrategy);
        HparamsMaps.put(m, "interleave_probs", interleaveProbs);
        HparamsMaps.put(m, "overwrite_cache", overwriteCache);
        HparamsMaps.put(m, "preprocessing_num_workers", preprocessingNumWorkers);
        HparamsMaps.put(m, "max_samples", maxSamples);
        HparamsMaps.put(m, "eval_dataset", evalDataset);
        HparamsMaps.put(m, "eval_on_each_dataset", evalOnEachDataset);
        HparamsMaps.put(m, "packing", packing);
        HparamsMaps.put(m, "neat_packing", neatPacking);
        HparamsMaps.put(m, "tool_format", toolFormat);
        HparamsMaps.put(m, "tokenized_path", tokenizedPath);
        HparamsMaps.put(m, "media_dir", mediaDir);
        HparamsMaps.put(m, "image_max_pixels", imageMaxPixels);
        HparamsMaps.put(m, "video_max_pixels", videoMaxPixels);
        HparamsMaps.put(m, "video_fps", videoFps);
        HparamsMaps.put(m, "video_maxlen", videoMaxFrames);
        HparamsMaps.put(m, "ignore_pad_token_for_loss", ignorePadTokenForLoss);
        HparamsMaps.put(m, "default_system", defaultSystem);
        return m;
    }

    public static DataArgs defaults() { return builder().build(); }

    public static DataArgs fromMap(Map<String, ?> m) {
        if (m == null || m.isEmpty()) return defaults();
        Builder b = builder();
        b.dataset(HparamsMaps.str(m, b.dataset, "dataset", "data"));
        b.datasetDir(HparamsMaps.str(m, b.datasetDir, "dataset_dir", "data_dir"));
        b.template(HparamsMaps.str(m, b.template, "template"));
        b.cutoffLen(HparamsMaps.integer(m, b.cutoffLen, "cutoff_len", "max_length", "max_seq_length"));
        b.trainOnPrompt(HparamsMaps.bool(m, b.trainOnPrompt, "train_on_prompt"));
        b.maskHistory(HparamsMaps.bool(m, b.maskHistory, "mask_history"));
        b.streaming(HparamsMaps.bool(m, b.streaming, "streaming"));
        b.bufferSize(HparamsMaps.integer(m, b.bufferSize, "buffer_size"));
        b.mixStrategy(HparamsMaps.str(m, b.mixStrategy, "mix_strategy"));
        b.interleaveProbs(HparamsMaps.strOrNull(m, "interleave_probs"));
        b.overwriteCache(HparamsMaps.bool(m, b.overwriteCache, "overwrite_cache"));
        b.preprocessingNumWorkers(HparamsMaps.integer(m, b.preprocessingNumWorkers, "preprocessing_num_workers"));
        b.maxSamples(HparamsMaps.integer(m, b.maxSamples, "max_samples"));
        b.evalDataset(HparamsMaps.strOrNull(m, "eval_dataset"));
        b.evalOnEachDataset(HparamsMaps.bool(m, b.evalOnEachDataset, "eval_on_each_dataset"));
        b.packing(HparamsMaps.bool(m, b.packing, "packing"));
        b.neatPacking(HparamsMaps.bool(m, b.neatPacking, "neat_packing"));
        b.toolFormat(HparamsMaps.str(m, b.toolFormat, "tool_format"));
        b.tokenizedPath(HparamsMaps.strOrNull(m, "tokenized_path"));
        b.mediaDir(HparamsMaps.strOrNull(m, "media_dir"));
        b.imageMaxPixels(HparamsMaps.integer(m, b.imageMaxPixels, "image_max_pixels"));
        b.videoMaxPixels(HparamsMaps.integer(m, b.videoMaxPixels, "video_max_pixels"));
        b.videoFps(HparamsMaps.dbl(m, b.videoFps, "video_fps"));
        b.videoMaxFrames(HparamsMaps.integer(m, b.videoMaxFrames, "video_maxlen", "video_max_frames"));
        b.ignorePadTokenForLoss(HparamsMaps.bool(m, b.ignorePadTokenForLoss, "ignore_pad_token_for_loss"));
        b.defaultSystem(HparamsMaps.strOrNull(m, "default_system", "system"));
        return b.build();
    }

    public static Builder builder() { return new Builder(); }

    public static final class Builder {
        private String dataset = "alpaca_en_demo";
        private String datasetDir = "data";
        private String template = "default";
        private int cutoffLen = 2048;
        private boolean trainOnPrompt;
        private boolean maskHistory;
        private boolean streaming;
        private int bufferSize = 16384;
        private String mixStrategy = "concat";
        private String interleaveProbs;
        private boolean overwriteCache;
        private int preprocessingNumWorkers = 1;
        private int maxSamples = -1;
        private String evalDataset;
        private boolean evalOnEachDataset;
        private boolean packing;
        private boolean neatPacking;
        private String toolFormat = "default";
        private String tokenizedPath;
        private String mediaDir;
        private int imageMaxPixels = 768 * 768;
        private int videoMaxPixels = 256 * 256;
        private double videoFps = 2.0;
        private int videoMaxFrames = 128;
        private boolean ignorePadTokenForLoss = true;
        private String defaultSystem;

        public Builder dataset(String v) { this.dataset = v; return this; }
        public Builder datasetDir(String v) { this.datasetDir = v; return this; }
        public Builder template(String v) { this.template = v; return this; }
        public Builder cutoffLen(int v) { this.cutoffLen = v; return this; }
        public Builder trainOnPrompt(boolean v) { this.trainOnPrompt = v; return this; }
        public Builder maskHistory(boolean v) { this.maskHistory = v; return this; }
        public Builder streaming(boolean v) { this.streaming = v; return this; }
        public Builder bufferSize(int v) { this.bufferSize = v; return this; }
        public Builder mixStrategy(String v) { this.mixStrategy = v; return this; }
        public Builder interleaveProbs(String v) { this.interleaveProbs = v; return this; }
        public Builder overwriteCache(boolean v) { this.overwriteCache = v; return this; }
        public Builder preprocessingNumWorkers(int v) { this.preprocessingNumWorkers = v; return this; }
        public Builder maxSamples(int v) { this.maxSamples = v; return this; }
        public Builder evalDataset(String v) { this.evalDataset = v; return this; }
        public Builder evalOnEachDataset(boolean v) { this.evalOnEachDataset = v; return this; }
        public Builder packing(boolean v) { this.packing = v; return this; }
        public Builder neatPacking(boolean v) { this.neatPacking = v; return this; }
        public Builder toolFormat(String v) { this.toolFormat = v; return this; }
        public Builder tokenizedPath(String v) { this.tokenizedPath = v; return this; }
        public Builder mediaDir(String v) { this.mediaDir = v; return this; }
        public Builder imageMaxPixels(int v) { this.imageMaxPixels = v; return this; }
        public Builder videoMaxPixels(int v) { this.videoMaxPixels = v; return this; }
        public Builder videoFps(double v) { this.videoFps = v; return this; }
        public Builder videoMaxFrames(int v) { this.videoMaxFrames = v; return this; }
        public Builder ignorePadTokenForLoss(boolean v) { this.ignorePadTokenForLoss = v; return this; }
        public Builder defaultSystem(String v) { this.defaultSystem = v; return this; }
        public DataArgs build() { return new DataArgs(this); }
    }
}
