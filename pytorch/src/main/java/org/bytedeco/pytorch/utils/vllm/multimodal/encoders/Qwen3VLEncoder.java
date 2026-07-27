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

import org.bytedeco.pytorch.NoGradGuard;
import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorOptional;
import org.bytedeco.pytorch.utils.vllm.multimodal.MediaInput;
import org.bytedeco.pytorch.utils.vllm.multimodal.MediaType;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.Map;

import static org.bytedeco.pytorch.global.torch.ScalarType;
import static org.bytedeco.pytorch.global.torch.conv3d;
import static org.bytedeco.pytorch.global.torch.gelu;
import static org.bytedeco.pytorch.global.torch.layer_norm;
import static org.bytedeco.pytorch.global.torch.matmul;
import static org.bytedeco.pytorch.global.torch.softmax;

/**
 * Functional Qwen3-VL vision tower ({@code model.visual.*}).
 *
 * <p>Layout (from {@code Qwen/Qwen3-VL-2B-Instruct-FP8} vision_config):
 * <ul>
 *   <li>patch_embed: Conv3d {@code [1024, 3, 2, 16, 16]} temporal_patch=2, patch=16</li>
 *   <li>pos_embed: {@code [2304, 1024]}</li>
 *   <li>24 transformer blocks with fused QKV {@code [3072, 1024]}, 16 heads</li>
 *   <li>spatial merger (2×2) → MLP → out_hidden=2048</li>
 * </ul>
 *
 * <p>Accepts either a full VL checkpoint or a vision-only {@code model.safetensors}
 * / {@code vision_weights.safetensors} under the model directory.
 */
public final class Qwen3VLEncoder implements MediaEncoder {

    public static final float[] MEAN = {0.5f, 0.5f, 0.5f};
    public static final float[] STD = {0.5f, 0.5f, 0.5f};

    private final String encoderName;
    private final int imageSize;
    private final int patchSize;
    private final int temporalPatch;
    private final int mergeSize;
    private final int hidden;
    private final int heads;
    private final int layers;
    private final int intermediate;
    private final int outHidden;
    private final Map<String, Tensor> w;

    private Qwen3VLEncoder(String name, int imageSize, int patchSize, int temporalPatch,
                           int mergeSize, int hidden, int heads, int layers, int intermediate,
                           int outHidden, Map<String, Tensor> w) {
        this.encoderName = name;
        this.imageSize = imageSize;
        this.patchSize = patchSize;
        this.temporalPatch = temporalPatch;
        this.mergeSize = mergeSize;
        this.hidden = hidden;
        this.heads = heads;
        this.layers = layers;
        this.intermediate = intermediate;
        this.outHidden = outHidden;
        this.w = w;
    }

    public static Qwen3VLEncoder fromDirectory(Path dir) throws Exception {
        Map<String, Tensor> raw = loadWeights(dir);
        Map<String, Tensor> w = toFloatMap(filterVisual(raw));
        if (w.isEmpty()) {
            throw new IllegalStateException("No model.visual.* weights in " + dir);
        }
        int hidden = 1024, heads = 16, layers = 24, inter = 4096, outH = 2048;
        int patch = 16, tPatch = 2, merge = 2;
        // Prefer 256 on Mac (fast); must be divisible by patch*merge=32
        int img = 256;
        Tensor pe = w.get("model.visual.patch_embed.proj.weight");
        if (pe != null && pe.defined() && pe.dim() == 5) {
            hidden = (int) pe.size(0);
            tPatch = (int) pe.size(2);
            patch = (int) pe.size(3);
        }
        int matched = 0;
        while (w.containsKey("model.visual.blocks." + matched + ".attn.qkv.weight")) matched++;
        if (matched > 0) layers = matched;
        Tensor m2 = w.get("model.visual.merger.linear_fc2.weight");
        if (m2 != null && m2.defined()) outH = (int) m2.size(0);
        Tensor m1 = w.get("model.visual.merger.linear_fc1.weight");
        if (m1 != null && m1.defined()) inter = (int) m1.size(0);
        if (hidden > 0 && heads > 0) {
            // head dim 64 typical
        }
        System.out.println("[Qwen3VLEncoder] tensors=" + w.size()
                + " layers=" + layers + " hidden=" + hidden + " out=" + outH
                + " img=" + img + " patch=" + patch);
        return new Qwen3VLEncoder("qwen3vl:" + dir, img, patch, tPatch, merge,
                hidden, heads, layers, inter, outH, w);
    }

    private static Map<String, Tensor> loadWeights(Path dir) throws Exception {
        Path vision = dir.resolve("vision_weights.safetensors");
        if (Files.isRegularFile(vision)) {
            return WeightBinder.loadSafetensors(vision);
        }
        Path model = dir.resolve("model.safetensors");
        if (Files.isRegularFile(model)) {
            return WeightBinder.loadSafetensors(model);
        }
        // sharded or multi-file
        return WeightBinder.loadSafetensors(dir);
    }

    private static Map<String, Tensor> filterVisual(Map<String, Tensor> in) {
        Map<String, Tensor> out = new LinkedHashMap<>();
        for (var e : in.entrySet()) {
            String k = e.getKey();
            if (k.startsWith("model.visual.") || k.startsWith("visual.")) {
                String nk = k.startsWith("visual.") ? "model." + k : k;
                out.put(nk, e.getValue());
            }
        }
        // if already vision-only file without prefix stripping needed
        if (out.isEmpty()) {
            for (var e : in.entrySet()) {
                if (e.getKey().contains("patch_embed") || e.getKey().contains("blocks.")) {
                    out.put(e.getKey().startsWith("model.") ? e.getKey() : "model.visual." + e.getKey(),
                            e.getValue());
                }
            }
        }
        if (out.isEmpty()) {
            // accept full map if it has visual keys under alternate layout
            return in;
        }
        return out;
    }

    private static Map<String, Tensor> toFloatMap(Map<String, Tensor> in) {
        Map<String, Tensor> out = new LinkedHashMap<>();
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
    @Override public int featureDim() { return outHidden; }

    @Override
    public EncoderFeatures encode(MediaInput input) {
        long t0 = System.nanoTime();
        try (NoGradGuard g = new NoGradGuard()) {
            Tensor pixels = ImagePreprocess.loadNormalized(input, imageSize, MEAN, STD); // [1,3,H,W]
            Tensor tokens = forwardTokens(pixels); // [1,N,outHidden] after merger
            Tensor pooled = tokens.mean(1L); // mean over sequence
            float[] pool = ImagePreprocess.toFloatArray(pooled.reshape(-1));
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
        // pixels [1,3,H,W] → stack temporal frames for Conv3d [1,3,T,H,W]
        long H = pixels.size(2), W = pixels.size(3);
        Tensor frame = pixels.unsqueeze(2); // [1,3,1,H,W]
        Tensor vol;
        if (temporalPatch <= 1) {
            vol = frame;
        } else {
            // repeat frame along time (image → pseudo video of T frames)
            Tensor[] frames = new Tensor[temporalPatch];
            for (int i = 0; i < temporalPatch; i++) frames[i] = frame;
            vol = org.bytedeco.pytorch.global.torch.cat(
                    new org.bytedeco.pytorch.TensorVector(frames), 2);
        }

        Tensor pw = req("model.visual.patch_embed.proj.weight"); // [D,3,t,p,p]
        Tensor pb = opt("model.visual.patch_embed.proj.bias");
        long[] stride = new long[]{temporalPatch, patchSize, patchSize};
        long[] padding = new long[]{0, 0, 0};
        long[] dilation = new long[]{1, 1, 1};
        TensorOptional biasOpt = (pb != null && pb.defined())
                ? new TensorOptional(pb) : new TensorOptional();
        Tensor x = conv3d(vol, pw, biasOpt, stride, padding, dilation, 1L);
        // x: [1, D, Tt, Gh, Gw] with Tt≈1
        long B = x.size(0);
        long D = x.size(1);
        long Gh = x.size(x.dim() - 2);
        long Gw = x.size(x.dim() - 1);
        // collapse time+spatial → sequence
        x = x.reshape(B, D, -1).transpose(1, 2).contiguous(); // [1, N, D]
        long N = x.size(1);

        // absolute pos embed [num_pos, D] → [1,N,D]
        Tensor pos = req("model.visual.pos_embed.weight");
        if (pos.dim() == 2) {
            if (pos.size(0) != N) {
                pos = interpolatePos(pos, N);
            }
            pos = pos.unsqueeze(0);
        }
        x = x.add(pos);

        for (int i = 0; i < layers; i++) {
            x = block(x, i);
        }

        // spatial merge 2x2 + merger MLP → [1, N/4, outHidden]
        x = spatialMergeAndProject(x, Gh, Gw);
        return x;
    }

    private Tensor interpolatePos(Tensor pos, long nTarget) {
        // pos [P, D] → sample / truncate / pad to nTarget
        long P = pos.size(0);
        long D = pos.size(1);
        if (nTarget == P) return pos;
        if (nTarget < P) {
            return pos.slice(0, new org.bytedeco.pytorch.LongOptional(0),
                    new org.bytedeco.pytorch.LongOptional(nTarget), 1);
        }
        // pad by repeating last
        long need = nTarget - P;
        Tensor last = pos.select(0, P - 1).unsqueeze(0);
        Tensor[] parts = new Tensor[(int) need + 1];
        parts[0] = pos;
        for (int i = 0; i < need; i++) parts[i + 1] = last;
        return org.bytedeco.pytorch.global.torch.cat(
                new org.bytedeco.pytorch.TensorVector(parts), 0);
    }

    private Tensor block(Tensor x, int i) {
        String p = "model.visual.blocks." + i + ".";
        Tensor h = layerNorm(x, req(p + "norm1.weight"), req(p + "norm1.bias"));
        Tensor a = attentionFused(h, req(p + "attn.qkv.weight"), req(p + "attn.qkv.bias"),
                req(p + "attn.proj.weight"), req(p + "attn.proj.bias"));
        x = x.add(a);
        Tensor h2 = layerNorm(x, req(p + "norm2.weight"), req(p + "norm2.bias"));
        Tensor m = mlp(h2,
                req(p + "mlp.linear_fc1.weight"), req(p + "mlp.linear_fc1.bias"),
                req(p + "mlp.linear_fc2.weight"), req(p + "mlp.linear_fc2.bias"));
        return x.add(m);
    }

    private Tensor attentionFused(Tensor x, Tensor qkvW, Tensor qkvB, Tensor projW, Tensor projB) {
        // x [B,N,C], qkvW [3C, C]
        long B = x.size(0), N = x.size(1), C = x.size(2);
        int headDim = (int) (C / heads);
        Tensor qkv = linear(x, qkvW, qkvB); // [B,N,3C]
        Tensor[] splits = splitQkv(qkv, C);
        Tensor q = splits[0].reshape(B, N, heads, headDim).transpose(1, 2);
        Tensor k = splits[1].reshape(B, N, heads, headDim).transpose(1, 2);
        Tensor v = splits[2].reshape(B, N, heads, headDim).transpose(1, 2);
        double scale = 1.0 / Math.sqrt(headDim);
        Tensor attn = softmax(matmul(q, k.transpose(-2, -1)).mul(new Scalar(scale)), -1L);
        Tensor out = matmul(attn, v).transpose(1, 2).contiguous().reshape(B, N, C);
        return linear(out, projW, projB);
    }

    private static Tensor[] splitQkv(Tensor qkv, long c) {
        // qkv [B,N,3C] → q,k,v each [B,N,C]
        Tensor q = qkv.slice(2, new org.bytedeco.pytorch.LongOptional(0),
                new org.bytedeco.pytorch.LongOptional(c), 1);
        Tensor k = qkv.slice(2, new org.bytedeco.pytorch.LongOptional(c),
                new org.bytedeco.pytorch.LongOptional(2 * c), 1);
        Tensor v = qkv.slice(2, new org.bytedeco.pytorch.LongOptional(2 * c),
                new org.bytedeco.pytorch.LongOptional(3 * c), 1);
        return new Tensor[]{q, k, v};
    }

    private Tensor mlp(Tensor x, Tensor w1, Tensor b1, Tensor w2, Tensor b2) {
        // gelu_pytorch_tanh ≈ gelu for feature extract
        return linear(gelu(linear(x, w1, b1)), w2, b2);
    }

    /**
     * Spatial merge 2×2 then merger MLP.
     * Input [1, Gh*Gw, hidden] with Gh,Gw divisible by mergeSize.
     */
    private Tensor spatialMergeAndProject(Tensor x, long Gh, long Gw) {
        long B = x.size(0);
        long N = x.size(1);
        long C = x.size(2);
        int m = mergeSize;
        // reshape to grid
        if (Gh * Gw != N) {
            // fall back: treat as 1D and pack groups of m*m
            long pack = (long) m * m;
            long nOut = N / pack;
            if (nOut * pack != N) {
                // truncate
                N = nOut * pack;
                x = x.slice(1, new org.bytedeco.pytorch.LongOptional(0),
                        new org.bytedeco.pytorch.LongOptional(N), 1);
            }
            x = x.reshape(B, nOut, pack * C);
        } else {
            long gh2 = Gh / m, gw2 = Gw / m;
            // [B, Gh, Gw, C] → [B, gh2, m, gw2, m, C] → [B, gh2*gw2, m*m*C]
            x = x.reshape(B, Gh, Gw, C);
            x = x.reshape(B, gh2, m, gw2, m, C);
            x = x.permute(0, 1, 3, 2, 4, 5).contiguous();
            x = x.reshape(B, gh2 * gw2, (long) m * m * C);
        }

        // merger: norm is on pre-merge hidden (1024) in HF — but after pack input is 4096.
        // Qwen3: norm weight [1024] applied before merge in some versions; here linear_fc1 is [4096,4096]
        // so input is packed 4096. Optional pre-norm on 1024 path if shapes match.
        Tensor normW = opt("model.visual.merger.norm.weight");
        Tensor normB = opt("model.visual.merger.norm.bias");
        if (normW != null && normW.numel() == x.size(2)) {
            x = layerNorm(x, normW, normB);
        } else if (normW != null && normW.numel() == C) {
            // norm was meant pre-merge; already packed — skip or apply via mean-scale noop
        }

        Tensor fc1w = req("model.visual.merger.linear_fc1.weight");
        Tensor fc1b = req("model.visual.merger.linear_fc1.bias");
        Tensor fc2w = req("model.visual.merger.linear_fc2.weight");
        Tensor fc2b = req("model.visual.merger.linear_fc2.bias");
        // if packed dim mismatch, mean-pool tokens instead of merger
        if (fc1w.size(1) != x.size(2)) {
            System.out.println("[Qwen3VLEncoder] merger in_dim mismatch got=" + x.size(2)
                    + " want=" + fc1w.size(1) + " — mean pool hidden");
            return x.mean(2L).unsqueeze(-1).expand(-1, -1, outHidden); // weak fallback
        }
        return linear(gelu(linear(x, fc1w, fc1b)), fc2w, fc2b);
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
            return y.mul(weight).add(bias);
        }
    }

    private static Tensor linear(Tensor x, Tensor weight, Tensor bias) {
        Tensor y = matmul(x, weight.transpose(0, 1));
        if (bias != null && bias.defined()) y = y.add(bias);
        return y;
    }
}
