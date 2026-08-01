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
package org.bytedeco.pytorch.llm.ktransformers.attention;

import org.bytedeco.pytorch.llm.ktransformers.cache.ThreeTierPrefixCache;
import org.bytedeco.pytorch.llm.ktransformers.config.KtCacheConfig;
import org.bytedeco.pytorch.llm.ktransformers.config.KtConfig;
import org.bytedeco.pytorch.llm.ktransformers.config.KtInferenceConfig;
import org.bytedeco.pytorch.llm.ktransformers.util.DeviceBudget;
import org.bytedeco.pytorch.llm.ktransformers.util.KtPreconditions;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;

/**
 * Long-context budget policy for KT inference paths.
 *
 * <p>Mirrors the engineering strategy behind upstream "longer Context @ 24GB"
 * demos: compress KV via MLA when possible, keep a sliding active window on GPU,
 * and demote cold prefix blocks GPU→CPU→Disk under {@link DeviceBudget} pressure.
 * This is a <strong>configurable planner</strong>, not a hard-coded single-SKU
 * recipe.
 */
public final class LongContextPolicy {

    /** Prefer MLA compressed KV over full dense KV pages. */
    private final boolean preferMla;
    /** Soft max tokens kept on the hot (GPU) tier before demote. */
    private final int gpuActiveTokens;
    /** Tokens kept as attention sink (never demoted). */
    private final int sinkTokens;
    /** Trigger demote when GPU util ≥ this watermark. */
    private final double demoteWatermark;
    /** Soft max sequence the planner will accept before refusing. */
    private final int maxSeqLen;
    /** Estimated bytes per token for dense KV (all layers). */
    private final long denseBytesPerToken;
    /** Estimated bytes per token for MLA compressed KV. */
    private final long mlaBytesPerToken;
    private final DeviceBudget budget;

    private long demoteActions;
    private long compressActions;
    private long refuseActions;

    public LongContextPolicy(boolean preferMla, int gpuActiveTokens, int sinkTokens,
                             double demoteWatermark, int maxSeqLen,
                             long denseBytesPerToken, long mlaBytesPerToken,
                             DeviceBudget budget) {
        this.preferMla = preferMla;
        this.gpuActiveTokens = KtPreconditions.checkPositive(gpuActiveTokens, "gpuActiveTokens");
        this.sinkTokens = Math.max(0, sinkTokens);
        this.demoteWatermark = demoteWatermark <= 0 || demoteWatermark > 1
                ? 0.85 : demoteWatermark;
        this.maxSeqLen = KtPreconditions.checkPositive(maxSeqLen, "maxSeqLen");
        this.denseBytesPerToken = Math.max(1L, denseBytesPerToken);
        this.mlaBytesPerToken = Math.max(1L, mlaBytesPerToken);
        this.budget = budget != null ? budget : DeviceBudget.mini();
    }

    /**
     * Build from KT configs: uses cache layout + 24GB-class defaults when
     * {@link DeviceBudget#consumer24g()} is supplied by the caller.
     */
    public static LongContextPolicy from(KtConfig config, DeviceBudget budget) {
        Objects.requireNonNull(config, "config");
        KtCacheConfig cache = config.cache();
        KtInferenceConfig inf = config.inference();
        int layers = Math.max(1, cache.numLayers() > 0 ? cache.numLayers() : config.numLayers());
        int heads = Math.max(1, cache.numHeads());
        int headDim = Math.max(1, cache.headDim());
        // dense: 2 (K+V) * layers * heads * headDim * 2 bytes (fp16-ish accounting)
        long dense = 2L * layers * heads * headDim * 2L;
        // MLA compressed: layers * (kv_lora + rope) * 2 bytes — assume kv_lora≈headDim
        long mlaBytes = layers * (long) headDim * 2L * 2L;
        int active = Math.max(cache.blockSize() * 4, Math.min(inf.maxSeqLen(), 4096));
        boolean preferMla = config.modelFamily() != null
                && config.modelFamily().name().toUpperCase().contains("DEEPSEEK");
        return new LongContextPolicy(preferMla, active, 4, cache.gpuWatermark(),
                inf.maxSeqLen(), dense, mlaBytes, budget);
    }

    /** Mini CI defaults. */
    public static LongContextPolicy mini() {
        return new LongContextPolicy(true, 64, 4, 0.80, 256,
                4L * 4 * 32 * 2 * 2, // small dense estimate
                4L * 32 * 2 * 2,
                DeviceBudget.mini());
    }

    /** Consumer 24GB VRAM style defaults (planning only). */
    public static LongContextPolicy consumer24g() {
        // ~DeepSeek-style: prefer MLA, keep 8k active on GPU, sink 16
        long dense = 2L * 60 * 128 * 128 * 2; // rough large-model dense
        long mla = 60L * 512 * 2 * 2;         // compressed latent
        return new LongContextPolicy(true, 8192, 16, 0.85, 128_000,
                dense, mla, DeviceBudget.consumer24g());
    }

    public boolean preferMla() { return preferMla; }
    public int gpuActiveTokens() { return gpuActiveTokens; }
    public int sinkTokens() { return sinkTokens; }
    public double demoteWatermark() { return demoteWatermark; }
    public int maxSeqLen() { return maxSeqLen; }
    public long denseBytesPerToken() { return denseBytesPerToken; }
    public long mlaBytesPerToken() { return mlaBytesPerToken; }
    public DeviceBudget budget() { return budget; }
    public long demoteActions() { return demoteActions; }
    public long compressActions() { return compressActions; }
    public long refuseActions() { return refuseActions; }

    /** Bytes needed for {@code tokens} under current compression preference. */
    public long estimateBytes(int tokens) {
        long per = preferMla ? mlaBytesPerToken : denseBytesPerToken;
        return Math.max(0, tokens) * per;
    }

    /**
     * Decide whether a sequence of {@code seqLen} tokens is admissible under
     * the budget; if not, recommend demote / refuse.
     */
    public Decision plan(int seqLen) {
        KtPreconditions.checkArgument(seqLen >= 0, "seqLen must be >= 0");
        if (seqLen > maxSeqLen) {
            refuseActions++;
            return Decision.refuse("seqLen " + seqLen + " > maxSeqLen " + maxSeqLen);
        }
        long need = estimateBytes(seqLen);
        long free = budget.gpuFreeBytes();
        boolean pressure = budget.gpuPressure(demoteWatermark) || need > free;
        int keepHot = Math.min(seqLen, gpuActiveTokens);
        int demoteTokens = Math.max(0, seqLen - keepHot);
        if (pressure && demoteTokens > 0) {
            demoteActions++;
            if (preferMla) compressActions++;
            return Decision.demote(keepHot, demoteTokens, preferMla,
                    "gpu pressure util=" + String.format("%.2f", budget.gpuUtilization())
                            + " need=" + need + " free=" + free);
        }
        if (preferMla) {
            compressActions++;
            return Decision.ok(keepHot, 0, true, "mla preferred, within budget");
        }
        return Decision.ok(keepHot, 0, false, "dense within budget");
    }

    /**
     * Apply demote pressure on a live {@link ThreeTierPrefixCache}: force demotes
     * until GPU util is under watermark or no more GPU blocks remain.
     *
     * @return number of demote operations performed
     */
    public int enforceOnCache(ThreeTierPrefixCache cache) {
        Objects.requireNonNull(cache, "cache");
        int n = 0;
        // Soft: if over watermark or over active token heuristic, demote
        while (budget.gpuPressure(demoteWatermark) || cache.gpuSize() > Math.max(1, gpuActiveTokens / Math.max(1, cache.blockSize()))) {
            if (!cache.forceDemoteOneFromGpu()) break;
            demoteActions++;
            n++;
            if (n > 10_000) break; // safety
        }
        return n;
    }

    public Map<String, Double> toMetricMap() {
        Map<String, Double> m = new LinkedHashMap<>();
        m.put("kt/longctx/prefer_mla", preferMla ? 1.0 : 0.0);
        m.put("kt/longctx/gpu_active_tokens", (double) gpuActiveTokens);
        m.put("kt/longctx/sink_tokens", (double) sinkTokens);
        m.put("kt/longctx/demote_watermark", demoteWatermark);
        m.put("kt/longctx/max_seq", (double) maxSeqLen);
        m.put("kt/longctx/demote_actions", (double) demoteActions);
        m.put("kt/longctx/compress_actions", (double) compressActions);
        m.put("kt/longctx/refuse_actions", (double) refuseActions);
        m.put("kt/longctx/dense_bpt", (double) denseBytesPerToken);
        m.put("kt/longctx/mla_bpt", (double) mlaBytesPerToken);
        return m;
    }

    /** Planner outcome for one sequence. */
    public static final class Decision {
        public enum Action { OK, DEMOTE, REFUSE }

        public final Action action;
        public final int keepHotTokens;
        public final int demoteTokens;
        public final boolean useMla;
        public final String reason;

        private Decision(Action action, int keepHotTokens, int demoteTokens,
                         boolean useMla, String reason) {
            this.action = action;
            this.keepHotTokens = keepHotTokens;
            this.demoteTokens = demoteTokens;
            this.useMla = useMla;
            this.reason = reason != null ? reason : "";
        }

        public static Decision ok(int keep, int demote, boolean mla, String reason) {
            return new Decision(Action.OK, keep, demote, mla, reason);
        }

        public static Decision demote(int keep, int demote, boolean mla, String reason) {
            return new Decision(Action.DEMOTE, keep, demote, mla, reason);
        }

        public static Decision refuse(String reason) {
            return new Decision(Action.REFUSE, 0, 0, false, reason);
        }

        public boolean allowed() { return action != Action.REFUSE; }

        @Override
        public String toString() {
            return "Decision{" + action + " keep=" + keepHotTokens
                    + " demote=" + demoteTokens + " mla=" + useMla
                    + " reason=" + reason + "}";
        }
    }
}
