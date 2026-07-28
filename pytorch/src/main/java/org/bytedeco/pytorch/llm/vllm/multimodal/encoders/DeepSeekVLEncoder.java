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

import org.bytedeco.pytorch.NoGradGuard;
import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorOptional;
import org.bytedeco.pytorch.TensorVector;
import org.bytedeco.pytorch.llm.vllm.multimodal.MediaInput;
import org.bytedeco.pytorch.llm.vllm.multimodal.MediaType;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.Map;

import static org.bytedeco.pytorch.global.torch.ScalarType;
import static org.bytedeco.pytorch.global.torch.cat;
import static org.bytedeco.pytorch.global.torch.conv2d;
import static org.bytedeco.pytorch.global.torch.gelu;
import static org.bytedeco.pytorch.global.torch.layer_norm;
import static org.bytedeco.pytorch.global.torch.matmul;
import static org.bytedeco.pytorch.global.torch.softmax;

/**
 * DeepSeek-VL vision tower (SigLIP-style ViT).
 *
 * <p>DeepSeek-VL-1.3B uses {@code siglip_large_patch16_384} (hidden=1024). On Mac we prefer
 * the smaller {@code google/siglip-base-patch16-224} (~800MB) as a drop-in encoder with the
 * same key layout:
 * <pre>
 *   vision_model.embeddings.patch_embedding.{weight,bias}
 *   vision_model.embeddings.position_embedding.weight
 *   vision_model.encoder.layers.{i}.self_attn.{q,k,v,out}_proj
 *   vision_model.encoder.layers.{i}.mlp.{fc1,fc2}
 *   vision_model.encoder.layers.{i}.layer_norm{1,2}
 *   vision_model.post_layernorm
 * </pre>
 *
 * <p>Also accepts DeepSeek-VL full checkpoints with {@code vision_model.*} keys
 * (or {@code model.vision_model.*}).
 */
public final class DeepSeekVLEncoder implements MediaEncoder {

    public static final float[] MEAN = {0.5f, 0.5f, 0.5f};
    public static final float[] STD = {0.5f, 0.5f, 0.5f};

    private final String encoderName;
    private final int imageSize;
    private final int patchSize;
    private final int hidden;
    private final int heads;
    private final int layers;
    private final Map<String, Tensor> w;
    private final String keyPrefix; // "vision_model." or "model.vision_model." etc.

    private DeepSeekVLEncoder(String name, int imageSize, int patchSize, int hidden, int heads,
                              int layers, Map<String, Tensor> w, String keyPrefix) {
        this.encoderName = name;
        this.imageSize = imageSize;
        this.patchSize = patchSize;
        this.hidden = hidden;
        this.heads = heads;
        this.layers = layers;
        this.w = w;
        this.keyPrefix = keyPrefix;
    }

    public static DeepSeekVLEncoder fromDirectory(Path dir) throws Exception {
        Map<String, Tensor> raw = loadWeights(dir);
        Map<String, Tensor> w = toFloatMap(raw);
        String prefix = detectPrefix(w);
        if (prefix == null) {
            throw new IllegalStateException("No SigLIP/DeepSeek vision keys in " + dir);
        }
        Tensor pw = w.get(prefix + "embeddings.patch_embedding.weight");
        int hidden = 768, patch = 16, img = 224, heads = 12, layers = 12;
        if (pw != null && pw.defined()) {
            hidden = (int) pw.size(0);
            patch = (int) pw.size(2);
        }
        if (hidden >= 1024) {
            heads = 16;
            img = 384; // siglip-large default
        } else {
            heads = 12;
            img = 224;
        }
        // Cap image size for Mac speed
        if (img > 256) img = 256;
        // ensure divisible by patch
        img = (img / patch) * patch;
        int matched = 0;
        while (w.containsKey(prefix + "encoder.layers." + matched + ".self_attn.q_proj.weight")
                || w.containsKey(prefix + "encoder.layers." + matched + ".self_attn.qkv_proj.weight")) {
            matched++;
        }
        if (matched > 0) layers = matched;
        System.out.println("[DeepSeekVLEncoder] tensors=" + w.size()
                + " prefix=" + prefix + " layers=" + layers
                + " hidden=" + hidden + " img=" + img);
        return new DeepSeekVLEncoder("deepseek-vl:" + dir, img, patch, hidden, heads, layers, w, prefix);
    }

    private static Map<String, Tensor> loadWeights(Path dir) throws Exception {
        Path vision = dir.resolve("vision_weights.safetensors");
        if (Files.isRegularFile(vision)) return WeightBinder.loadSafetensors(vision);
        Path model = dir.resolve("model.safetensors");
        if (Files.isRegularFile(model)) return WeightBinder.loadSafetensors(model);
        return WeightBinder.loadSafetensors(dir);
    }

    private static String detectPrefix(Map<String, Tensor> w) {
        String[] candidates = {
                "vision_model.",
                "model.vision_model.",
                "visual.",
                "model.visual.",
                ""
        };
        for (String p : candidates) {
            if (w.containsKey(p + "embeddings.patch_embedding.weight")
                    || w.containsKey(p + "embeddings.patch_embeddings.projection.weight")) {
                return p;
            }
        }
        // DeepSeek-VL may store under vision_model with different patch name
        for (String k : w.keySet()) {
            if (k.contains("patch_embedding.weight") || k.contains("patch_embed")) {
                int idx = k.indexOf("embeddings.");
                if (idx > 0) return k.substring(0, idx);
                if (k.startsWith("vision_model")) return "vision_model.";
            }
        }
        return null;
    }

    private static Map<String, Tensor> toFloatMap(Map<String, Tensor> in) {
        Map<String, Tensor> out = new LinkedHashMap<>();
        for (var e : in.entrySet()) {
            Tensor t = e.getValue();
            if (t == null || !t.defined()) continue;
            // keep only vision-ish keys when full LM present
            String k = e.getKey();
            if (k.contains("language") || k.contains("aligner") || k.startsWith("model.layers")
                    || k.contains("lm_head") || k.contains("embed_tokens")) {
                // skip LM
                if (!k.contains("vision")) continue;
            }
            try {
                out.put(k, t.to(ScalarType.Float).contiguous());
            } catch (Throwable ex) {
                out.put(k, t);
            }
        }
        return out;
    }

    private Tensor req(String key) {
        Tensor t = w.get(key);
        if (t == null || !t.defined()) {
            throw new IllegalStateException("missing weight: " + key);
        }
        return t;
    }

    private Tensor opt(String key) {
        return w.get(key);
    }

    private String k(String suffix) {
        return keyPrefix + suffix;
    }

    @Override public MediaType modality() { return MediaType.IMAGE; }
    @Override public String encoderName() { return encoderName; }
    @Override public int featureDim() { return hidden; }

    @Override
    public EncoderFeatures encode(MediaInput input) {
        long t0 = System.nanoTime();
        try (NoGradGuard g = new NoGradGuard()) {
            Tensor pixels = ImagePreprocess.loadNormalized(input, imageSize, MEAN, STD);
            Tensor tokens = forwardTokens(pixels);
            // CLS = token 0 for SigLIP/CLIP-style
            Tensor pooled = tokens.select(1, 0);
            Tensor postW = opt(k("post_layernorm.weight"));
            Tensor postB = opt(k("post_layernorm.bias"));
            if (postW != null) {
                pooled = layerNorm(pooled, postW, postB);
            }
            float[] pool = ImagePreprocess.toFloatArray(pooled.reshape(-1));
            // L2 normalize
            double n = 0;
            for (float v : pool) n += v * v;
            n = Math.sqrt(n);
            if (n > 1e-8) for (int i = 0; i < pool.length; i++) pool[i] /= (float) n;

            int seqN = (int) Math.min(8, tokens.size(1));
            float[][] seq = new float[seqN][];
            for (int i = 0; i < seqN; i++) {
                seq[i] = ImagePreprocess.toFloatArray(tokens.select(1, i).reshape(-1));
            }
            double ms = (System.nanoTime() - t0) / 1e6;
            return new EncoderFeatures(pool, seq, encoderName, ms);
        } catch (Exception e) {
            System.out.println("[" + encoderName + "] encode failed: " + e.getMessage());
            e.printStackTrace(System.out);
            return EncoderFeatures.empty(encoderName);
        }
    }

    private Tensor forwardTokens(Tensor pixels) {
        Tensor pw = firstPresent(
                k("embeddings.patch_embedding.weight"),
                k("embeddings.patch_embeddings.projection.weight"));
        Tensor pb = firstPresentOpt(
                k("embeddings.patch_embedding.bias"),
                k("embeddings.patch_embeddings.projection.bias"));
        long[] stride = new long[]{patchSize, patchSize};
        long[] padding = new long[]{0, 0};
        long[] dilation = new long[]{1, 1};
        TensorOptional biasOpt = (pb != null && pb.defined())
                ? new TensorOptional(pb) : new TensorOptional();
        Tensor x = conv2d(pixels, pw, biasOpt, stride, padding, dilation, 1L);
        long B = x.size(0), C = x.size(1), Gh = x.size(2), Gw = x.size(3);
        x = x.reshape(B, C, Gh * Gw).transpose(1, 2).contiguous(); // [1,N,D]

        // SigLIP has no class token in some variants; HF SigLIP uses position emb covering patches only
        // or includes CLS. Detect via position emb length.
        Tensor pos = firstPresent(
                k("embeddings.position_embedding.weight"),
                k("embeddings.position_embeddings"));
        if (pos.dim() == 3) pos = pos.squeeze(0);
        boolean hasCls = pos.size(0) == x.size(1) + 1;
        if (hasCls) {
            // invent CLS as mean of patches (SigLIP often has no dedicated CLS weight in vision_model)
            Tensor cls = x.mean(1L).unsqueeze(1);
            Tensor clsW = opt(k("embeddings.class_embedding"));
            if (clsW != null) {
                if (clsW.dim() == 1) cls = clsW.reshape(1, 1, hidden);
                else if (clsW.dim() == 2) cls = clsW.unsqueeze(0);
                else cls = clsW;
            }
            x = cat(new TensorVector(cls, x), 1);
        }
        if (pos.size(0) != x.size(1)) {
            if (pos.size(0) > x.size(1)) {
                pos = pos.slice(0, new org.bytedeco.pytorch.LongOptional(0),
                        new org.bytedeco.pytorch.LongOptional(x.size(1)), 1);
            } else {
                long need = x.size(1) - pos.size(0);
                Tensor pad = org.bytedeco.pytorch.global.torch.zeros(need, pos.size(1));
                pos = cat(new TensorVector(pos, pad), 0);
            }
        }
        x = x.add(pos.unsqueeze(0));

        for (int i = 0; i < layers; i++) {
            x = block(x, i);
        }
        return x;
    }

    private Tensor firstPresent(String... keys) {
        for (String key : keys) {
            Tensor t = w.get(key);
            if (t != null && t.defined()) return t;
        }
        throw new IllegalStateException("missing any of " + String.join(",", keys));
    }

    private Tensor firstPresentOpt(String... keys) {
        for (String key : keys) {
            Tensor t = w.get(key);
            if (t != null && t.defined()) return t;
        }
        return null;
    }

    private Tensor block(Tensor x, int i) {
        String p = keyPrefix + "encoder.layers." + i + ".";
        Tensor h = layerNorm(x, req(p + "layer_norm1.weight"), req(p + "layer_norm1.bias"));
        Tensor a;
        if (w.containsKey(p + "self_attn.q_proj.weight")) {
            a = attentionSeparate(h,
                    req(p + "self_attn.q_proj.weight"), opt(p + "self_attn.q_proj.bias"),
                    req(p + "self_attn.k_proj.weight"), opt(p + "self_attn.k_proj.bias"),
                    req(p + "self_attn.v_proj.weight"), opt(p + "self_attn.v_proj.bias"),
                    req(p + "self_attn.out_proj.weight"), opt(p + "self_attn.out_proj.bias"));
        } else {
            // fused qkv
            a = attentionFused(h,
                    req(p + "self_attn.qkv_proj.weight"), opt(p + "self_attn.qkv_proj.bias"),
                    req(p + "self_attn.out_proj.weight"), opt(p + "self_attn.out_proj.bias"));
        }
        x = x.add(a);
        Tensor h2 = layerNorm(x, req(p + "layer_norm2.weight"), req(p + "layer_norm2.bias"));
        Tensor m = mlp(h2,
                req(p + "mlp.fc1.weight"), opt(p + "mlp.fc1.bias"),
                req(p + "mlp.fc2.weight"), opt(p + "mlp.fc2.bias"));
        return x.add(m);
    }

    private Tensor attentionSeparate(Tensor x, Tensor qw, Tensor qb, Tensor kw, Tensor kb,
                                     Tensor vw, Tensor vb, Tensor ow, Tensor ob) {
        long B = x.size(0), N = x.size(1), C = x.size(2);
        int headDim = (int) (C / heads);
        Tensor q = linear(x, qw, qb).reshape(B, N, heads, headDim).transpose(1, 2);
        Tensor k = linear(x, kw, kb).reshape(B, N, heads, headDim).transpose(1, 2);
        Tensor v = linear(x, vw, vb).reshape(B, N, heads, headDim).transpose(1, 2);
        double scale = 1.0 / Math.sqrt(headDim);
        Tensor attn = softmax(matmul(q, k.transpose(-2, -1)).mul(new Scalar(scale)), -1L);
        Tensor out = matmul(attn, v).transpose(1, 2).contiguous().reshape(B, N, C);
        return linear(out, ow, ob);
    }

    private Tensor attentionFused(Tensor x, Tensor qkvW, Tensor qkvB, Tensor ow, Tensor ob) {
        long B = x.size(0), N = x.size(1), C = x.size(2);
        int headDim = (int) (C / heads);
        Tensor qkv = linear(x, qkvW, qkvB);
        Tensor q = qkv.slice(2, new org.bytedeco.pytorch.LongOptional(0),
                new org.bytedeco.pytorch.LongOptional(C), 1);
        Tensor k = qkv.slice(2, new org.bytedeco.pytorch.LongOptional(C),
                new org.bytedeco.pytorch.LongOptional(2 * C), 1);
        Tensor v = qkv.slice(2, new org.bytedeco.pytorch.LongOptional(2 * C),
                new org.bytedeco.pytorch.LongOptional(3 * C), 1);
        q = q.reshape(B, N, heads, headDim).transpose(1, 2);
        k = k.reshape(B, N, heads, headDim).transpose(1, 2);
        v = v.reshape(B, N, heads, headDim).transpose(1, 2);
        double scale = 1.0 / Math.sqrt(headDim);
        Tensor attn = softmax(matmul(q, k.transpose(-2, -1)).mul(new Scalar(scale)), -1L);
        Tensor out = matmul(attn, v).transpose(1, 2).contiguous().reshape(B, N, C);
        return linear(out, ow, ob);
    }

    private Tensor mlp(Tensor x, Tensor w1, Tensor b1, Tensor w2, Tensor b2) {
        return linear(gelu(linear(x, w1, b1)), w2, b2);
    }

    private static Tensor layerNorm(Tensor x, Tensor weight, Tensor bias) {
        long H = weight.numel();
        long[] shape = new long[]{H};
        TensorOptional wOpt = new TensorOptional(weight);
        TensorOptional bOpt = (bias != null && bias.defined())
                ? new TensorOptional(bias) : new TensorOptional();
        try {
            return layer_norm(x, shape, wOpt, bOpt, 1e-6, true);
        } catch (Throwable t) {
            Tensor mean = x.mean(new long[]{-1L}, true, new org.bytedeco.pytorch.ScalarTypeOptional());
            Tensor var = x.var(new long[]{-1L}, false, true);
            Tensor y = x.sub(mean).div(var.add(new Scalar(1e-6)).sqrt());
            if (bias != null && bias.defined()) return y.mul(weight).add(bias);
            return y.mul(weight);
        }
    }

    private static Tensor linear(Tensor x, Tensor weight, Tensor bias) {
        Tensor y = matmul(x, weight.transpose(0, 1));
        if (bias != null && bias.defined()) y = y.add(bias);
        return y;
    }
}
