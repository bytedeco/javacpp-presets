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
package org.bytedeco.pytorch.utils.transformers;
import org.bytedeco.pytorch.nn.modules.*;
import org.bytedeco.pytorch.jit.*;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.LongOptional;
import org.bytedeco.pytorch.LongVector;
import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorOptions;
import org.bytedeco.pytorch.TensorVector;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.modules.EmbeddingImpl;
import org.bytedeco.pytorch.nn.modules.LayerNormImpl;
import org.bytedeco.pytorch.nn.modules.LinearImpl;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Objects;

import static org.bytedeco.pytorch.global.torch.ScalarType;
import static org.bytedeco.pytorch.global.torch.argmax;
import static org.bytedeco.pytorch.global.torch.arange;
import static org.bytedeco.pytorch.global.torch.cat;
import static org.bytedeco.pytorch.global.torch.cross_entropy;
import static org.bytedeco.pytorch.global.torch.exp;
import static org.bytedeco.pytorch.global.torch.full;
import static org.bytedeco.pytorch.global.torch.gelu;
import static org.bytedeco.pytorch.global.torch.matmul;
import static org.bytedeco.pytorch.global.torch.multinomial;
import static org.bytedeco.pytorch.global.torch.ones;
import static org.bytedeco.pytorch.global.torch.softmax;
import static org.bytedeco.pytorch.global.torch.tensor;
import static org.bytedeco.pytorch.global.torch.topk;
import static org.bytedeco.pytorch.global.torch.triu;

/**
 * Minimal causal language model (GPT/Llama/Qwen/Mistral-style backbone).
 *
 * <p>Architecture (per layer):
 * <pre>
 *   x → LN → Multi-Head Self-Attn (causal) → +residual
 *     → LN → MLP (Linear→GELU→Linear)       → +residual
 *   logits = LM-head(LN(x))
 * </pre>
 *
 * <pre>{@code
 * CausalLM model = CausalLM.fromConfig(PretrainedConfig.tinyGpt2());
 * Tensor logits = model.forward(inputIds);          // [B, T, V]
 * int[] out = model.generate(promptIds, 16);
 * }</pre>
 */
@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class CausalLM extends Module {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    private final PretrainedConfig config;
    private final EmbeddingImpl tokEmbed;
    private final EmbeddingImpl posEmbed; // null when RoPE-style
    private final List<Block> blocks = new ArrayList<>();
    private final LayerNormImpl finalNorm;
    private final LinearImpl lmHead;
    private final boolean useRope;
    /** Optional LoRA adapters keyed by quantizable linear path (e.g. h/0/attn/c_attn). */
    private final java.util.LinkedHashMap<String, org.bytedeco.pytorch.llm.peft.LoraLinear> loraAdapters =
            new java.util.LinkedHashMap<>();

    public CausalLM(PretrainedConfig config) {
        super("CausalLM");
        this.config = Objects.requireNonNull(config, "config");
        this.useRope = config.modelType() == PretrainedConfig.ModelType.LLAMA
                || config.modelType() == PretrainedConfig.ModelType.QWEN
                || config.modelType() == PretrainedConfig.ModelType.MISTRAL;

        this.tokEmbed = register_module("wte",
                new EmbeddingImpl(config.vocabSize(), config.hiddenSize()));
        if (!useRope) {
            this.posEmbed = register_module("wpe",
                    new EmbeddingImpl(config.maxPositionEmbeddings(), config.hiddenSize()));
        } else {
            this.posEmbed = null;
        }

        // LongVector(long) is a SIZE ctor (n zeros) — must put the single normalized dim.
        LongVector normShape = new LongVector().put((long) config.hiddenSize());
        for (int i = 0; i < config.numHiddenLayers(); i++) {
            Block b = new Block(this, config, useRope, i);
            blocks.add(register_module("h/" + i, b));
        }
        this.finalNorm = register_module("ln_f", new LayerNormImpl(normShape));
        this.lmHead = register_module("lm_head",
                new LinearImpl(config.hiddenSize(), config.vocabSize()));
    }

    public static CausalLM fromConfig(PretrainedConfig config) {
        return new CausalLM(config);
    }

    public PretrainedConfig config() {
        return config;
    }

    public int vocabSize() {
        return config.vocabSize();
    }

    public int hiddenSize() {
        return config.hiddenSize();
    }

    public List<Block> blocks() {
        return Collections.unmodifiableList(blocks);
    }

    public LinearImpl lmHead() {
        return lmHead;
    }

    /**
     * Collect named linear layers for BitsAndBytes / QLoRA quantization.
     * Keys use HF-ish paths: {@code h/0/attn/c_attn}, {@code h/0/mlp/fc_in}, {@code lm_head}.
     */
    public java.util.Map<String, LinearImpl> namedLinears() {
        java.util.LinkedHashMap<String, LinearImpl> m = new java.util.LinkedHashMap<>();
        for (int i = 0; i < blocks.size(); i++) {
            Block b = blocks.get(i);
            String prefix = "h/" + i;
            m.put(prefix + "/attn/c_attn", b.attn.qkv);
            m.put(prefix + "/attn/c_proj", b.attn.proj);
            m.put(prefix + "/mlp/fc_in", b.mlp.fc1);
            m.put(prefix + "/mlp/fc_out", b.mlp.fc2);
        }
        m.put("lm_head", lmHead);
        return m;
    }

    /** Target modules typically quantized / LoRA-adapted (excludes lm_head). */
    public java.util.Map<String, LinearImpl> quantizableLinears() {
        java.util.LinkedHashMap<String, LinearImpl> m = new java.util.LinkedHashMap<>();
        for (java.util.Map.Entry<String, LinearImpl> e : namedLinears().entrySet()) {
            if (!"lm_head".equals(e.getKey())) {
                m.put(e.getKey(), e.getValue());
            }
        }
        return m;
    }

    /**
     * Attach a LoRA adapter for a named linear so {@link #forward} / {@link #loss}
     * apply {@code y = base(x) + scale·(x Aᵀ Bᵀ)} end-to-end (ΔW in the LM path).
     *
     * <p>Uses {@link org.bytedeco.pytorch.llm.peft.LoraLinear#borrowBase} so the
     * base {@link LinearImpl} is not double-registered under the CausalLM tree.
     *
     * @return the adapter, or {@code null} if {@code name} is unknown
     */
    public org.bytedeco.pytorch.llm.peft.LoraLinear attachLora(
            String name, org.bytedeco.pytorch.llm.peft.LoraConfig cfg) {
        Objects.requireNonNull(name, "name");
        Objects.requireNonNull(cfg, "cfg");
        LinearImpl lin = namedLinears().get(name);
        if (lin == null) return null;
        org.bytedeco.pytorch.llm.peft.LoraLinear adapter =
                org.bytedeco.pytorch.llm.peft.LoraLinear.borrowBase(lin, cfg);
        loraAdapters.put(name, adapter);
        // Keep Module.parameters() aware of A/B without re-parenting the base linear.
        try {
            register_module("lora/" + name.replace('/', '_'), adapter);
        } catch (Exception ignored) {
            // Adapter still usable via explicit map even if register_module fails.
        }
        return adapter;
    }

    /**
     * Attach LoRA on every name matching {@code cfg.targetModules()} (leaf match).
     * @return number of adapters attached
     */
    public int attachLora(org.bytedeco.pytorch.llm.peft.LoraConfig cfg) {
        Objects.requireNonNull(cfg, "cfg");
        int n = 0;
        for (String name : quantizableLinears().keySet()) {
            String leaf = name;
            int slash = Math.max(name.lastIndexOf('/'), name.lastIndexOf('.'));
            if (slash >= 0) leaf = name.substring(slash + 1);
            if (org.bytedeco.pytorch.llm.peft.PeftModelHelper.matchesTarget(leaf, cfg)
                    || org.bytedeco.pytorch.llm.peft.PeftModelHelper.matchesTarget(name, cfg)) {
                if (attachLora(name, cfg) != null) n++;
            }
        }
        // Fallback: if nothing matched (e.g. llama targets on gpt2 names), attach all.
        if (n == 0) {
            for (String name : quantizableLinears().keySet()) {
                if (attachLora(name, cfg) != null) n++;
            }
        }
        return n;
    }

    public java.util.Map<String, org.bytedeco.pytorch.llm.peft.LoraLinear> loraAdapters() {
        return java.util.Collections.unmodifiableMap(loraAdapters);
    }

    public boolean hasLora() {
        return !loraAdapters.isEmpty();
    }

    /** Apply linear or its LoRA wrapper when present. */
    Tensor applyLinear(String name, LinearImpl lin, Tensor x) {
        org.bytedeco.pytorch.llm.peft.LoraLinear adapter = loraAdapters.get(name);
        if (adapter != null) return adapter.forward(x);
        return lin.forward(x);
    }

    /**
     * Forward pass.
     *
     * @param inputIds Long tensor {@code [B, T]} or {@code [T]}
     * @return Float logits {@code [B, T, V]}
     */
    @Override
    public Tensor forward(Tensor inputIds) {
        Tensor ids = inputIds;
        if (ids.dim() == 1) {
            ids = ids.unsqueeze(0);
        }
        long B = ids.size(0);
        long T = ids.size(1);
        if (T > config.maxPositionEmbeddings()) {
            throw new IllegalArgumentException("Sequence length " + T
                    + " exceeds max_position_embeddings=" + config.maxPositionEmbeddings());
        }

        Tensor x = tokEmbed.forward(ids); // [B, T, C]
        if (posEmbed != null) {
            Tensor pos = arange(new Scalar(0L), new Scalar(T), new Scalar(1L),
                    new TensorOptions(ScalarType.Long));
            pos = pos.unsqueeze(0); // [1, T]
            x = x.add(posEmbed.forward(pos));
        }

        for (Block block : blocks) {
            x = block.forward(x);
        }
        x = finalNorm.forward(x);
        return lmHead.forward(x); // [B, T, V]
    }

    /** Cross-entropy LM loss on next-token prediction (shift by 1). */
    public Tensor loss(Tensor inputIds) {
        Tensor ids = inputIds.dim() == 1 ? inputIds.unsqueeze(0) : inputIds;
        // cross_entropy requires Long/Byte targets — promote Int token ids if needed.
        // JavaCPP: scalar_type() returns a non-canonical proxy — intern() before compare.
        ScalarType idSt = ids.scalar_type().intern();
        if (idSt == ScalarType.Int || idSt == ScalarType.Short || idSt == ScalarType.Byte
                || idSt == ScalarType.Char) {
            ids = ids.to(ScalarType.Long);
        }
        Tensor logits = forward(ids); // [B, T, V]
        Tensor shiftLogits = logits.slice(1, new LongOptional(0), new LongOptional(logits.size(1) - 1), 1)
                .contiguous();
        Tensor shiftLabels = ids.slice(1, new LongOptional(1), new LongOptional(ids.size(1)), 1)
                .contiguous();
        long V = logits.size(2);
        Tensor flatLogits = shiftLogits.reshape(-1, V);
        Tensor flatLabels = shiftLabels.reshape(-1).to(ScalarType.Long);
        return cross_entropy(flatLogits, flatLabels);
    }

    public int[] generate(int[] promptIds, int maxNewTokens) {
        return generate(promptIds, maxNewTokens, GenerationConfig.greedy());
    }

    public int[] generate(int[] promptIds, int maxNewTokens, GenerationConfig gen) {
        Objects.requireNonNull(promptIds, "promptIds");
        if (gen == null) gen = GenerationConfig.greedy();
        List<Integer> seq = new ArrayList<>(promptIds.length + maxNewTokens);
        for (int id : promptIds) seq.add(id);

        boolean wasTraining = this.is_training();
        this.eval();
        try {
            for (int step = 0; step < maxNewTokens; step++) {
                int start = Math.max(0, seq.size() - config.maxPositionEmbeddings());
                long[] cur = new long[seq.size() - start];
                for (int i = 0; i < cur.length; i++) cur[i] = seq.get(start + i);
                Tensor ids = tensor(cur).unsqueeze(0); // [1, T]
                Tensor logits = forward(ids); // [1, T, V]
                Tensor last = logits
                        .slice(1, new LongOptional(logits.size(1) - 1), new LongOptional(logits.size(1)), 1)
                        .squeeze(0).squeeze(0); // [V]
                if (gen.temperature > 0 && Math.abs(gen.temperature - 1.0) > 1e-6) {
                    last = last.div(new Scalar(gen.temperature));
                }
                int next;
                if (gen.doSample && gen.temperature > 0) {
                    if (gen.topK > 0) {
                        last = topKFilter(last, gen.topK);
                    }
                    Tensor probs = softmax(last, 0L);
                    Tensor sampled = multinomial(probs, 1L);
                    next = (int) sampled.item_long();
                } else {
                    next = (int) argmax(last).item_long();
                }
                seq.add(next);
                if (next == config.eosTokenId() && gen.eosStop) {
                    break;
                }
            }
        } finally {
            if (wasTraining) this.train(true);
        }
        int[] out = new int[seq.size()];
        for (int i = 0; i < seq.size(); i++) out[i] = seq.get(i);
        return out;
    }

    private static Tensor topKFilter(Tensor logits, int k) {
        long V = logits.size(0);
        if (k <= 0 || k >= V) return logits;
        var top = topk(logits, k);
        Tensor values = top.get0();
        float threshold = values
                .slice(0, new LongOptional(values.size(0) - 1), new LongOptional(values.size(0)), 1)
                .squeeze()
                .item_float();
        Tensor negInf = full(new long[]{V}, new Scalar(-1e9f));
        Tensor mask = logits.gt(new Scalar(threshold - 1e-6f)).to(ScalarType.Float);
        Tensor ones = full(new long[]{V}, new Scalar(1.0f));
        return logits.mul(mask).add(negInf.mul(ones.sub(mask)));
    }

    // ---- Block -----------------------------------------------------------

    public static final class Block extends Module {
        final CausalLM parent;
        final int layerIdx;
        final LayerNormImpl ln1;
        final CausalAttention attn;
        final LayerNormImpl ln2;
        final Mlp mlp;

        Block(CausalLM parent, PretrainedConfig cfg, boolean useRope, int layerIdx) {
            super("Block" + layerIdx);
            this.parent = parent;
            this.layerIdx = layerIdx;
            // LongVector(long) is SIZE ctor — put the single normalized dim instead.
            LongVector shape = new LongVector().put((long) cfg.hiddenSize());
            this.ln1 = register_module("ln_1", new LayerNormImpl(shape));
            this.attn = register_module("attn", new CausalAttention(parent, cfg, useRope, layerIdx));
            this.ln2 = register_module("ln_2", new LayerNormImpl(shape));
            this.mlp = register_module("mlp", new Mlp(parent, cfg, layerIdx));
        }

        @Override
        public Tensor forward(Tensor x) {
            x = x.add(attn.forward(ln1.forward(x)));
            x = x.add(mlp.forward(ln2.forward(x)));
            return x;
        }
    }

    public static final class CausalAttention extends Module {
        final CausalLM parent;
        final int layerIdx;
        final PretrainedConfig cfg;
        final LinearImpl qkv;
        final LinearImpl proj;
        final boolean useRope;
        final int nHead;
        final int headDim;

        CausalAttention(CausalLM parent, PretrainedConfig cfg, boolean useRope, int layerIdx) {
            super("CausalAttention");
            this.parent = parent;
            this.layerIdx = layerIdx;
            this.cfg = cfg;
            this.useRope = useRope;
            this.nHead = cfg.numAttentionHeads();
            this.headDim = cfg.headDim();
            this.qkv = register_module("c_attn", new LinearImpl(cfg.hiddenSize(), 3L * cfg.hiddenSize()));
            this.proj = register_module("c_proj", new LinearImpl(cfg.hiddenSize(), cfg.hiddenSize()));
        }

        @Override
        public Tensor forward(Tensor x) {
            long B = x.size(0);
            long T = x.size(1);
            long C = x.size(2);
            String prefix = "h/" + layerIdx + "/attn";
            // LoRA-aware linear: y = base(x) + ΔW when adapter attached
            Tensor mixed = parent.applyLinear(prefix + "/c_attn", qkv, x); // [B, T, 3C]
            Tensor q = mixed.slice(2, new LongOptional(0), new LongOptional(C), 1);
            Tensor k = mixed.slice(2, new LongOptional(C), new LongOptional(2 * C), 1);
            Tensor v = mixed.slice(2, new LongOptional(2 * C), new LongOptional(3 * C), 1);

            q = q.view(B, T, nHead, headDim).transpose(1, 2);
            k = k.view(B, T, nHead, headDim).transpose(1, 2);
            v = v.view(B, T, nHead, headDim).transpose(1, 2);

            if (useRope) {
                q = applyRope(q, cfg.ropeTheta());
                k = applyRope(k, cfg.ropeTheta());
            }

            double scale = 1.0 / Math.sqrt(headDim);
            Tensor att = matmul(q, k.transpose(-2, -1)).mul(new Scalar(scale));
            att = att.add(causalMask(T));
            att = softmax(att, -1L);
            Tensor y = matmul(att, v);
            y = y.transpose(1, 2).contiguous().view(B, T, C);
            return parent.applyLinear(prefix + "/c_proj", proj, y);
        }

        private static Tensor causalMask(long T) {
            Tensor onesT = ones(new long[]{T, T});
            Tensor tri = triu(onesT, 1L);
            return tri.mul(new Scalar(-1e9f));
        }

        private static Tensor applyRope(Tensor x, double theta) {
            long T = x.size(2);
            long D = x.size(3);
            if (D % 2 != 0) return x;
            long half = D / 2;
            Tensor pos = arange(new Scalar(0L), new Scalar(T), new Scalar(1L),
                    new TensorOptions(ScalarType.Float));
            Tensor idx = arange(new Scalar(0L), new Scalar(half), new Scalar(1L),
                    new TensorOptions(ScalarType.Float));
            Tensor freq = idx.mul(new Scalar(2.0)).div(new Scalar((double) D));
            Tensor invFreq = exp(freq.neg().mul(new Scalar(Math.log(theta))));
            Tensor angles = pos.unsqueeze(1).mul(invFreq.unsqueeze(0));
            Tensor cos = angles.cos();
            Tensor sin = angles.sin();
            Tensor x1 = x.slice(3, new LongOptional(0), new LongOptional(half), 1);
            Tensor x2 = x.slice(3, new LongOptional(half), new LongOptional(D), 1);
            cos = cos.unsqueeze(0).unsqueeze(0);
            sin = sin.unsqueeze(0).unsqueeze(0);
            Tensor r1 = x1.mul(cos).sub(x2.mul(sin));
            Tensor r2 = x1.mul(sin).add(x2.mul(cos));
            return cat(new TensorVector(r1, r2), 3);
        }
    }

    public static final class Mlp extends Module {
        final CausalLM parent;
        final int layerIdx;
        final LinearImpl fc1;
        final LinearImpl fc2;

        Mlp(CausalLM parent, PretrainedConfig cfg, int layerIdx) {
            super("Mlp");
            this.parent = parent;
            this.layerIdx = layerIdx;
            this.fc1 = register_module("fc_in", new LinearImpl(cfg.hiddenSize(), cfg.intermediateSize()));
            this.fc2 = register_module("fc_out", new LinearImpl(cfg.intermediateSize(), cfg.hiddenSize()));
        }

        @Override
        public Tensor forward(Tensor x) {
            String prefix = "h/" + layerIdx + "/mlp";
            Tensor h = parent.applyLinear(prefix + "/fc_in", fc1, x);
            return parent.applyLinear(prefix + "/fc_out", fc2, gelu(h));
        }
    }

    public static final class GenerationConfig {
        public final boolean doSample;
        public final double temperature;
        public final int topK;
        public final double topP;
        public final boolean eosStop;

        public GenerationConfig(boolean doSample, double temperature, int topK, double topP, boolean eosStop) {
            this.doSample = doSample;
            this.temperature = temperature;
            this.topK = topK;
            this.topP = topP;
            this.eosStop = eosStop;
        }

        public static GenerationConfig greedy() {
            return new GenerationConfig(false, 1.0, 0, 1.0, true);
        }

        public static GenerationConfig sample(double temperature, int topK) {
            return new GenerationConfig(true, temperature, topK, 1.0, true);
        }

        public static Builder builder() {
            return new Builder();
        }

        public static final class Builder {
            private boolean doSample;
            private double temperature = 1.0;
            private int topK;
            private double topP = 1.0;
            private boolean eosStop = true;

            public Builder doSample(boolean v) { this.doSample = v; return this; }
            public Builder temperature(double v) { this.temperature = v; return this; }
            public Builder topK(int v) { this.topK = v; return this; }
            public Builder topP(double v) { this.topP = v; return this; }
            public Builder eosStop(boolean v) { this.eosStop = v; return this; }
            public GenerationConfig build() {
                return new GenerationConfig(doSample, temperature, topK, topP, eosStop);
            }
        }
    }
}
