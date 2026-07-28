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

import java.nio.file.Path;
import java.util.Map;

import static org.bytedeco.pytorch.global.torch.ScalarType;
import static org.bytedeco.pytorch.global.torch.cat;
import static org.bytedeco.pytorch.global.torch.conv2d;
import static org.bytedeco.pytorch.global.torch.gelu;
import static org.bytedeco.pytorch.global.torch.layer_norm;
import static org.bytedeco.pytorch.global.torch.matmul;
import static org.bytedeco.pytorch.global.torch.softmax;

/**
 * Functional ViT-style forward over a raw HF weight map (no nested Module registration).
 * Avoids JavaCPP Module peer / ExpandingArray GC issues seen with register_module Conv2d.
 *
 * <p>Supports DINOv2 / CLIP-vision / SmolVLM-vision key layouts via {@link KeyStyle}.
 */
public final class FunctionalVisionEncoder implements MediaEncoder {

    public enum KeyStyle { DINO, CLIP, SMOL_VISION }

    private final String encoderName;
    private final KeyStyle style;
    private final int imageSize;
    private final int patchSize;
    private final int hidden;
    private final int heads;
    private final int layers;
    private final int intermediate;
    private final float[] mean;
    private final float[] std;
    private final Map<String, Tensor> w;
    private final int matchedLayers;

    private FunctionalVisionEncoder(String name, KeyStyle style, int imageSize, int patchSize,
                                    int hidden, int heads, int layers, int intermediate,
                                    float[] mean, float[] std, Map<String, Tensor> w,
                                    int matchedLayers) {
        this.encoderName = name;
        this.style = style;
        this.imageSize = imageSize;
        this.patchSize = patchSize;
        this.hidden = hidden;
        this.heads = heads;
        this.layers = layers;
        this.intermediate = intermediate;
        this.mean = mean;
        this.std = std;
        this.w = w;
        this.matchedLayers = matchedLayers;
    }

    public static FunctionalVisionEncoder dinov2(Path dir) throws Exception {
        Map<String, Tensor> w = WeightBinder.loadSafetensors(dir);
        // force float32
        w = toFloatMap(w);
        int img = 518, patch = 14, hidden = 384, heads = 6, layers = 12, inter = 1536;
        int matched = countLayers(w, "encoder.layer.", ".attention.attention.query.weight");
        if (matched > 0) layers = matched;
        System.out.println("[FunctionalVision/DINO] tensors=" + w.size()
                + " layers=" + layers + " hasPatch=" + w.containsKey("embeddings.patch_embeddings.projection.weight"));
        return new FunctionalVisionEncoder("dinov2:" + dir, KeyStyle.DINO, img, patch,
                hidden, heads, layers, inter,
                ImagePreprocess.IMAGENET_MEAN, ImagePreprocess.IMAGENET_STD, w, layers);
    }

    public static FunctionalVisionEncoder clip(Path dir) throws Exception {
        Map<String, Tensor> w;
        Path vs = dir.resolve("vision_weights.safetensors");
        if (java.nio.file.Files.isRegularFile(vs)) {
            w = WeightBinder.loadSafetensors(vs);
        } else {
            w = ClipEncoder.loadClipWeights(dir);
        }
        w = toFloatMap(w);
        int img = 224, patch = 32, hidden = 768, heads = 12, layers = 12, inter = 3072;
        int matched = countLayers(w, "vision_model.encoder.layers.", ".self_attn.q_proj.weight");
        if (matched > 0) layers = matched;
        System.out.println("[FunctionalVision/CLIP] tensors=" + w.size() + " layers=" + layers);
        return new FunctionalVisionEncoder("clip:" + dir, KeyStyle.CLIP, img, patch,
                hidden, heads, layers, inter,
                ImagePreprocess.CLIP_MEAN, ImagePreprocess.CLIP_STD, w, layers);
    }

    public static FunctionalVisionEncoder smolVision(Path dir) throws Exception {
        Map<String, Tensor> w = WeightBinder.loadSafetensors(dir);
        w = toFloatMap(w);
        // Use 224 for speed; pos emb has 1024 entries for 512/16
        int img = 224, patch = 16, hidden = 768, heads = 12, layers = 12, inter = 3072;
        int matched = countLayers(w, "model.vision_model.encoder.layers.", ".self_attn.q_proj.weight");
        if (matched > 0) layers = matched;
        System.out.println("[FunctionalVision/Smol] tensors=" + w.size() + " layers=" + layers);
        return new FunctionalVisionEncoder("smolvlm:" + dir, KeyStyle.SMOL_VISION, img, patch,
                hidden, heads, layers, inter,
                ImagePreprocess.IMAGENET_MEAN, ImagePreprocess.IMAGENET_STD, w, layers);
    }

    private static Map<String, Tensor> toFloatMap(Map<String, Tensor> in) {
        Map<String, Tensor> out = new java.util.LinkedHashMap<>();
        for (var e : in.entrySet()) {
            Tensor t = e.getValue();
            if (t == null || !t.defined()) continue;
            try {
                out.put(e.getKey(), t.to(ScalarType.Float).contiguous());
            } catch (Throwable ex) {
                out.put(e.getKey(), t);
            }
        }
        return out;
    }

    private static int countLayers(Map<String, Tensor> w, String prefix, String suffix) {
        int n = 0;
        while (w.containsKey(prefix + n + suffix)) n++;
        return n;
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

    @Override public MediaType modality() { return MediaType.IMAGE; }
    @Override public String encoderName() { return encoderName; }
    @Override public int featureDim() {
        if (style == KeyStyle.CLIP) {
            Tensor p = opt("visual_projection.weight");
            if (p != null) return (int) p.size(0);
        }
        if (style == KeyStyle.SMOL_VISION) {
            Tensor p = opt("model.connector.modality_projection.proj.weight");
            if (p != null) return (int) p.size(0);
        }
        return hidden;
    }

    @Override
    public EncoderFeatures encode(MediaInput input) {
        long t0 = System.nanoTime();
        try (NoGradGuard g = new NoGradGuard()) {
            Tensor pixels = ImagePreprocess.loadNormalized(input, imageSize, mean, std);
            Tensor tokens = forwardTokens(pixels); // [1,N,D]
            Tensor pooled = tokens.select(1, 0); // CLS or first token
            if (style == KeyStyle.CLIP) {
                Tensor post = layerNorm(pooled, req("vision_model.post_layernorm.weight"),
                        req("vision_model.post_layernorm.bias"));
                Tensor projW = req("visual_projection.weight"); // [512,768]
                pooled = linear(post, projW, null);
            } else if (style == KeyStyle.SMOL_VISION) {
                // mean pool then connector
                Tensor meanTok = tokens.mean(1L);
                Tensor projW = opt("model.connector.modality_projection.proj.weight");
                if (projW != null) {
                    // pack: tile to vision_h * 16
                    float[] pv = ImagePreprocess.toFloatArray(meanTok.reshape(-1));
                    int pack = 16;
                    float[] packed = new float[pv.length * pack];
                    for (int i = 0; i < pack; i++) System.arraycopy(pv, 0, packed, i * pv.length, pv.length);
                    Tensor packedT = ImagePreprocess.fromFloatArray(packed, 1, packed.length);
                    pooled = linear(packedT, projW, null);
                } else {
                    pooled = meanTok;
                }
            } else if (style == KeyStyle.DINO) {
                // CLS already selected; optional final layernorm already applied in forward
            }
            float[] pool = ImagePreprocess.toFloatArray(pooled.reshape(-1));
            // L2 for CLIP-like
            if (style == KeyStyle.CLIP) {
                double n = 0;
                for (float v : pool) n += v * v;
                n = Math.sqrt(n);
                if (n > 1e-8) for (int i = 0; i < pool.length; i++) pool[i] /= (float) n;
            }
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
        // pixels [1,3,H,W]
        Tensor pw, pb = null, cls, pos;
        switch (style) {
            case DINO -> {
                pw = req("embeddings.patch_embeddings.projection.weight");
                pb = req("embeddings.patch_embeddings.projection.bias");
                cls = req("embeddings.cls_token"); // [1,1,H]
                pos = req("embeddings.position_embeddings"); // [1,N,H]
            }
            case CLIP -> {
                pw = req("vision_model.embeddings.patch_embedding.weight");
                cls = req("vision_model.embeddings.class_embedding").reshape(1, 1, hidden);
                pos = req("vision_model.embeddings.position_embedding.weight"); // [N,H]
                if (pos.dim() == 2) pos = pos.unsqueeze(0);
            }
            case SMOL_VISION -> {
                pw = req("model.vision_model.embeddings.patch_embedding.weight");
                pb = req("model.vision_model.embeddings.patch_embedding.bias");
                // no cls — use first token as pooled later via mean
                cls = null;
                pos = req("model.vision_model.embeddings.position_embedding.weight");
                if (pos.dim() == 2) pos = pos.unsqueeze(0);
            }
            default -> throw new IllegalStateException("style");
        }

        // conv2d: weight [out,in,k,k], stride=patch
        // Keep arrays as locals so GC cannot free them while the native call runs.
        long[] stride = new long[]{patchSize, patchSize};
        long[] padding = new long[]{0, 0};
        long[] dilation = new long[]{1, 1};
        TensorOptional biasOpt = (pb != null && pb.defined())
                ? new TensorOptional(pb) : new TensorOptional();
        Tensor x = conv2d(pixels, pw, biasOpt, stride, padding, dilation, 1L); // [1,D,Gh,Gw]
        long B = x.size(0);
        long C = x.size(1);
        long Gh = x.size(2);
        long Gw = x.size(3);
        x = x.reshape(B, C, Gh * Gw).transpose(1, 2).contiguous(); // [1,N,D]

        if (cls != null) {
            x = cat(new TensorVector(cls, x), 1);
        }
        // position
        if (pos.size(1) != x.size(1)) {
            if (pos.size(1) > x.size(1)) {
                pos = pos.slice(1, new org.bytedeco.pytorch.LongOptional(0),
                        new org.bytedeco.pytorch.LongOptional(x.size(1)), 1);
            } else {
                // pad with zeros
                long need = x.size(1) - pos.size(1);
                Tensor pad = zerosLikePos(pos, need);
                pos = cat(new TensorVector(pos, pad), 1);
            }
        }
        x = x.add(pos);

        // optional pre-ln for CLIP
        if (style == KeyStyle.CLIP) {
            Tensor lnw = opt("vision_model.pre_layrnorm.weight");
            Tensor lnb = opt("vision_model.pre_layrnorm.bias");
            if (lnw == null) {
                lnw = opt("vision_model.pre_layernorm.weight");
                lnb = opt("vision_model.pre_layernorm.bias");
            }
            if (lnw != null) x = layerNorm(x, lnw, lnb);
        }

        for (int i = 0; i < layers; i++) {
            x = block(x, i);
        }

        // final ln
        switch (style) {
            case DINO -> x = layerNorm(x, req("layernorm.weight"), req("layernorm.bias"));
            case CLIP -> { /* post ln applied on CLS in encode */ }
            case SMOL_VISION -> x = layerNorm(x,
                    req("model.vision_model.post_layernorm.weight"),
                    req("model.vision_model.post_layernorm.bias"));
        }
        return x;
    }

    private Tensor block(Tensor x, int i) {
        return switch (style) {
            case DINO -> dinoBlock(x, i);
            case CLIP -> clipBlock(x, i);
            case SMOL_VISION -> smolBlock(x, i);
        };
    }

    private Tensor dinoBlock(Tensor x, int i) {
        String p = "encoder.layer." + i + ".";
        Tensor n1w = req(p + "norm1.weight");
        Tensor n1b = req(p + "norm1.bias");
        Tensor h = layerNorm(x, n1w, n1b);
        Tensor a = attentionSeparate(h,
                req(p + "attention.attention.query.weight"), req(p + "attention.attention.query.bias"),
                req(p + "attention.attention.key.weight"), req(p + "attention.attention.key.bias"),
                req(p + "attention.attention.value.weight"), req(p + "attention.attention.value.bias"),
                req(p + "attention.output.dense.weight"), req(p + "attention.output.dense.bias"));
        // layer scale
        Tensor ls1 = opt(p + "layer_scale1.lambda1");
        if (ls1 != null) a = a.mul(ls1);
        x = x.add(a);
        Tensor n2w = req(p + "norm2.weight");
        Tensor n2b = req(p + "norm2.bias");
        Tensor h2 = layerNorm(x, n2w, n2b);
        Tensor m = mlp(h2, req(p + "mlp.fc1.weight"), req(p + "mlp.fc1.bias"),
                req(p + "mlp.fc2.weight"), req(p + "mlp.fc2.bias"));
        Tensor ls2 = opt(p + "layer_scale2.lambda1");
        if (ls2 != null) m = m.mul(ls2);
        return x.add(m);
    }

    private Tensor clipBlock(Tensor x, int i) {
        String p = "vision_model.encoder.layers." + i + ".";
        Tensor h = layerNorm(x, req(p + "layer_norm1.weight"), req(p + "layer_norm1.bias"));
        Tensor a = attentionSeparate(h,
                req(p + "self_attn.q_proj.weight"), req(p + "self_attn.q_proj.bias"),
                req(p + "self_attn.k_proj.weight"), req(p + "self_attn.k_proj.bias"),
                req(p + "self_attn.v_proj.weight"), req(p + "self_attn.v_proj.bias"),
                req(p + "self_attn.out_proj.weight"), req(p + "self_attn.out_proj.bias"));
        x = x.add(a);
        Tensor h2 = layerNorm(x, req(p + "layer_norm2.weight"), req(p + "layer_norm2.bias"));
        Tensor m = mlp(h2, req(p + "mlp.fc1.weight"), req(p + "mlp.fc1.bias"),
                req(p + "mlp.fc2.weight"), req(p + "mlp.fc2.bias"));
        return x.add(m);
    }

    private Tensor smolBlock(Tensor x, int i) {
        String p = "model.vision_model.encoder.layers." + i + ".";
        Tensor h = layerNorm(x, req(p + "layer_norm1.weight"), req(p + "layer_norm1.bias"));
        Tensor a = attentionSeparate(h,
                req(p + "self_attn.q_proj.weight"), req(p + "self_attn.q_proj.bias"),
                req(p + "self_attn.k_proj.weight"), req(p + "self_attn.k_proj.bias"),
                req(p + "self_attn.v_proj.weight"), req(p + "self_attn.v_proj.bias"),
                req(p + "self_attn.out_proj.weight"), req(p + "self_attn.out_proj.bias"));
        x = x.add(a);
        Tensor h2 = layerNorm(x, req(p + "layer_norm2.weight"), req(p + "layer_norm2.bias"));
        Tensor m = mlp(h2, req(p + "mlp.fc1.weight"), req(p + "mlp.fc1.bias"),
                req(p + "mlp.fc2.weight"), req(p + "mlp.fc2.bias"));
        return x.add(m);
    }

    private Tensor attentionSeparate(Tensor x, Tensor qw, Tensor qb, Tensor kw, Tensor kb,
                                     Tensor vw, Tensor vb, Tensor ow, Tensor ob) {
        // x [B,N,C]
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

    private Tensor mlp(Tensor x, Tensor w1, Tensor b1, Tensor w2, Tensor b2) {
        return linear(gelu(linear(x, w1, b1)), w2, b2);
    }

    private static Tensor layerNorm(Tensor x, Tensor weight, Tensor bias) {
        // x [..., H], weight/bias [H]
        long H = weight.numel();
        long[] shape = new long[]{H};
        TensorOptional wOpt = new TensorOptional(weight);
        TensorOptional bOpt = (bias != null && bias.defined())
                ? new TensorOptional(bias) : new TensorOptional();
        try {
            return layer_norm(x, shape, wOpt, bOpt, 1e-6, true);
        } catch (Throwable t) {
            // manual fallback
            Tensor mean = x.mean(new long[]{-1L}, true, new org.bytedeco.pytorch.ScalarTypeOptional());
            Tensor var = x.var(new long[]{-1L}, false, true);
            Tensor y = x.sub(mean).div(var.add(new Scalar(1e-6)).sqrt());
            return y.mul(weight).add(bias);
        }
    }

    private static Tensor linear(Tensor x, Tensor weight, Tensor bias) {
        // weight [out, in]
        Tensor y = matmul(x, weight.transpose(0, 1));
        if (bias != null && bias.defined()) y = y.add(bias);
        return y;
    }

    private static Tensor zerosLikePos(Tensor pos, long need) {
        return org.bytedeco.pytorch.global.torch.zeros(1, need, pos.size(2));
    }
}
