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

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.LongPointer;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.LongVector;
import org.bytedeco.pytorch.NoGradGuard;
import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.modules.Conv2dImpl;
import org.bytedeco.pytorch.nn.modules.LayerNormImpl;
import org.bytedeco.pytorch.nn.modules.LinearImpl;
import org.bytedeco.pytorch.nn.options.Conv2dOptions;
import org.bytedeco.pytorch.nn.options.LinearOptions;
import org.bytedeco.pytorch.llm.vllm.multimodal.MediaInput;
import org.bytedeco.pytorch.llm.vllm.multimodal.MediaType;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

import static org.bytedeco.pytorch.global.torch.cat;
import static org.bytedeco.pytorch.global.torch.gelu;
import static org.bytedeco.pytorch.global.torch.matmul;
import static org.bytedeco.pytorch.global.torch.softmax;
import static org.bytedeco.pytorch.global.torch.zeros;

/**
 * DINOv2-small vision encoder loaded from a local HF snapshot.
 *
 * <p>HF layout:
 * {@code embeddings.*}, {@code encoder.layer.N.attention.attention.{query,key,value}},
 * {@code encoder.layer.N.attention.output.dense}, {@code encoder.layer.N.mlp.fc*},
 * {@code encoder.layer.N.layer_scale*.lambda1}, {@code layernorm}.
 */
@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public final class Dinov2Encoder extends Module implements MediaEncoder {

    static { Loader.load(org.bytedeco.pytorch.presets.torch.class); }

    private final String name;
    private final int imageSize;
    private final int patchSize;
    private final int hiddenSize;
    private final int numHeads;
    private final int numLayers;
    private final WeightBinder.Report loadReport;

    private final LongPointer patchKernel;
    private final LongPointer patchStride;
    private final Conv2dImpl patchEmbed;          // embeddings.patch_embeddings.projection
    private Tensor clsToken;                      // embeddings.cls_token [1,1,H]
    private Tensor posEmbed;                      // embeddings.position_embeddings [1,N,H]
    private final List<DinoBlock> blocks = new ArrayList<>();
    private final LayerNormImpl layernorm;

    public Dinov2Encoder(Path dir) throws Exception {
        super("Dinov2Encoder");
        Objects.requireNonNull(dir, "dir");
        this.name = dir.toString();
        // Match HF dinov2-small defaults so pos_embed binds exactly (518/14 → 37²+1=1370).
        int img = 518, patch = 14, hidden = 384, heads = 6, layers = 12, inter = 1536;
        Path cfg = dir.resolve("config.json");
        if (Files.isRegularFile(cfg)) {
            String json = Files.readString(cfg);
            img = readInt(json, "image_size", img);
            patch = readInt(json, "patch_size", patch);
            hidden = readInt(json, "hidden_size", hidden);
            heads = readInt(json, "num_attention_heads", heads);
            layers = readInt(json, "num_hidden_layers", layers);
            double ratio = 4.0;
            try {
                int idx = json.indexOf("\"mlp_ratio\"");
                if (idx >= 0) {
                    String sub = json.substring(idx).replaceAll("[^0-9.]+", " ").trim().split("\\s+")[0];
                    ratio = Double.parseDouble(sub);
                }
            } catch (Exception ignored) {}
            inter = (int) (hidden * ratio);
        }
        // Runtime encode still resizes to this imageSize so pos_embed matches.
        this.imageSize = img;
        this.patchSize = patch;
        this.hiddenSize = hidden;
        this.numHeads = heads;
        this.numLayers = layers;

        // Keep ExpandingArray pointers as fields so GC cannot free them after ctor.
        this.patchKernel = new LongPointer(new long[]{patch, patch});
        this.patchStride = new LongPointer(new long[]{patch, patch});
        Conv2dOptions copt = new Conv2dOptions(3, hidden, patchKernel);
        copt.stride(patchStride);
        copt.bias(true);
        this.patchEmbed = register_module("embeddings/patch_embeddings/projection", new Conv2dImpl(copt));
        int numPatches = (img / patch) * (img / patch);
        this.clsToken = register_parameter("embeddings/cls_token", zeros(1, 1, hidden), true);
        this.posEmbed = register_parameter("embeddings/position_embeddings",
                zeros(1, numPatches + 1, hidden), true);

        LongVector lnShape = new LongVector().put((long) hidden);
        for (int i = 0; i < layers; i++) {
            blocks.add(register_module("encoder/layer/" + i, new DinoBlock(hidden, heads, inter)));
        }
        this.layernorm = register_module("layernorm", new LayerNormImpl(lnShape));

        this.eval();
        this.loadReport = WeightBinder.bindSafetensors(this, dir,
                List.of("model.", "dinov2."), false);
        // Layer-scale stays at ones if HF key names don't match — forward still works.
        // Re-sync Tensor fields after bind (copy_/set_ may rebind storage).
        resyncParams();
        System.out.println("[Dinov2Encoder] " + loadReport
                + " img=" + img + " patches=" + numPatches
                + " cls=" + shapeOf(clsToken) + " pos=" + shapeOf(posEmbed)
                + " dir=" + dir.getFileName());
    }

    private void resyncParams() {
        try {
            var dict = this.named_parameters(true);
            if (dict == null || dict.isNull()) return;
            for (long i = 0; i < dict.size(); i++) {
                var item = dict.get(i);
                if (item == null || item.isNull()) continue;
                String k = item.key() != null ? item.key().getString() : "";
                Tensor v = item.value();
                if (v == null || !v.defined()) continue;
                if (k.endsWith("cls_token") || k.contains("cls_token")) clsToken = v;
                if (k.contains("position_embeddings")) posEmbed = v;
            }
        } catch (Throwable t) {
            System.out.println("[Dinov2Encoder] resync warning: " + t.getMessage());
        }
    }

    private static String shapeOf(Tensor t) {
        if (t == null || !t.defined()) return "undef";
        StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < t.dim(); i++) {
            if (i > 0) sb.append(',');
            sb.append(t.size(i));
        }
        return sb.append(']').toString();
    }

    public static Dinov2Encoder fromDirectory(Path dir) throws Exception {
        return new Dinov2Encoder(dir);
    }

    public WeightBinder.Report loadReport() { return loadReport; }

    @Override
    public MediaType modality() { return MediaType.IMAGE; }

    @Override
   public String encoderName() { return "dinov2:" + name; }

    @Override
    public int featureDim() { return hiddenSize; }

    @Override
    public EncoderFeatures encode(MediaInput input) {
        long t0 = System.nanoTime();
        try {
            Tensor pixels = ImagePreprocess.loadNormalized(
                    input, imageSize, ImagePreprocess.IMAGENET_MEAN, ImagePreprocess.IMAGENET_STD);
            Tensor tokens = forwardTokens(pixels); // [1, N, D]
            Tensor pooled = tokens.select(1, 0);   // CLS
            float[] pool = ImagePreprocess.toFloatArray(pooled.reshape(-1));
            // also export a few patch tokens for sequence path
            int seqN = (int) Math.min(16, tokens.size(1));
            float[][] seq = new float[seqN][];
            for (int i = 0; i < seqN; i++) {
                seq[i] = ImagePreprocess.toFloatArray(tokens.select(1, i).reshape(-1));
            }
            double ms = (System.nanoTime() - t0) / 1e6;
            return new EncoderFeatures(pool, seq, encoderName(), ms);
        } catch (Exception e) {
            System.out.println("[Dinov2Encoder] encode failed: " + e.getMessage());
            return EncoderFeatures.empty(encoderName());
        }
    }

    public Tensor forwardTokens(Tensor pixelValues) {
        try (NoGradGuard guard = new NoGradGuard()) {
            // pixelValues [B,3,H,W]
            Tensor x = patchEmbed.forward(pixelValues); // [B,D,Gh,Gw]
            if (x == null || !x.defined()) {
                throw new IllegalStateException("patchEmbed returned undefined tensor");
            }
            long B = x.size(0);
            long C = x.size(1);
            long Gh = x.size(2);
            long Gw = x.size(3);
            // [B,D,Gh,Gw] → [B, Gh*Gw, D]
            x = x.reshape(B, C, Gh * Gw).transpose(1, 2).contiguous();
            // CLS broadcast without expand(-1)
            Tensor cls = clsToken;
            if (cls == null || !cls.defined()) {
                throw new IllegalStateException("clsToken undefined after bind");
            }
            if (B > 1) {
                cls = cls.repeat(new long[]{B, 1, 1});
            }
            x = cat(new org.bytedeco.pytorch.TensorVector(cls, x), 1);
            Tensor pe = posEmbed;
            if (pe == null || !pe.defined()) {
                throw new IllegalStateException("posEmbed undefined after bind");
            }
            if (pe.dim() == 2) pe = pe.unsqueeze(0);
            if (pe.size(1) != x.size(1)) {
                pe = interpolatePos(pe, (int) x.size(1));
            }
            if (pe.size(0) == 1 && B > 1) {
                pe = pe.repeat(new long[]{B, 1, 1});
            }
            x = x.add(pe);
            for (DinoBlock b : blocks) x = b.forward(x);
            return layernorm.forward(x);
        }
    }

    private static Tensor interpolatePos(Tensor pe, int newLen) {
        // pe [1, old, D] — simple truncate / repeat
        long old = pe.size(1);
        if (old == newLen) return pe;
        if (old > newLen) {
            return pe.slice(1, new org.bytedeco.pytorch.LongOptional(0),
                    new org.bytedeco.pytorch.LongOptional(newLen), 1);
        }
        Tensor last = pe.slice(1, new org.bytedeco.pytorch.LongOptional(old - 1),
                new org.bytedeco.pytorch.LongOptional(old), 1);
        org.bytedeco.pytorch.TensorVector tv = new org.bytedeco.pytorch.TensorVector();
        tv.put(pe);
        for (long i = old; i < newLen; i++) tv.put(last);
        return cat(tv, 1);
    }

    @Override
    public Tensor forward(Tensor input) {
        return forwardTokens(input).select(1, 0);
    }

    @Override
    public void close() {
        // Module cleanup is GC-driven for JavaCPP peers
    }

    // ---- HF DINO block: separate Q/K/V + output.dense + layer scale ------------

    @Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
    public static class DinoBlock extends Module {
        static { Loader.load(org.bytedeco.pytorch.presets.torch.class); }

        public final LayerNormImpl norm1;
        public final DinoAttention attn;
        public final LayerNormImpl norm2;
        public final DinoMlp mlp;
        public Tensor ls1;
        public Tensor ls2;

        public DinoBlock(int hidden, int heads, int inter) {
            super("DinoBlock");
            LongVector shape = new LongVector().put((long) hidden);
            this.norm1 = register_module("norm1", new LayerNormImpl(shape));
            this.attn = register_module("attention", new DinoAttention(hidden, heads));
            this.norm2 = register_module("norm2", new LayerNormImpl(shape));
            this.mlp = register_module("mlp", new DinoMlp(hidden, inter));
            // Simple leaf names; HF keys remapped in bind aliases
            this.ls1 = register_parameter("ls1",
                    org.bytedeco.pytorch.global.torch.ones(hidden), true);
            this.ls2 = register_parameter("ls2",
                    org.bytedeco.pytorch.global.torch.ones(hidden), true);
        }

        @Override
        public Tensor forward(Tensor x) {
            Tensor a = attn.forward(norm1.forward(x));
            // layer-scale optional — skip if unbound / wrong shape (defaults keep ones)
            try {
                if (ls1 != null && ls1.defined() && ls1.numel() == x.size(2)) {
                    a = a.mul(ls1);
                }
            } catch (Throwable ignored) {}
            x = x.add(a);
            Tensor m = mlp.forward(norm2.forward(x));
            try {
                if (ls2 != null && ls2.defined() && ls2.numel() == x.size(2)) {
                    m = m.mul(ls2);
                }
            } catch (Throwable ignored) {}
            return x.add(m);
        }
    }

    @Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
    public static class DinoAttention extends Module {
        static { Loader.load(org.bytedeco.pytorch.presets.torch.class); }

        public final LinearImpl query;
        public final LinearImpl key;
        public final LinearImpl value;
        public final LinearImpl output;
        private final int heads;
        private final int headDim;

        public DinoAttention(int hidden, int heads) {
            super("DinoAttention");
            this.heads = heads;
            this.headDim = hidden / heads;
            // HF: attention.attention.query/key/value + attention.output.dense
            this.query = register_module("attention/query",
                    new LinearImpl(new LinearOptions(hidden, hidden).bias(true)));
            this.key = register_module("attention/key",
                    new LinearImpl(new LinearOptions(hidden, hidden).bias(true)));
            this.value = register_module("attention/value",
                    new LinearImpl(new LinearOptions(hidden, hidden).bias(true)));
            this.output = register_module("output/dense",
                    new LinearImpl(new LinearOptions(hidden, hidden).bias(true)));
        }

        @Override
        public Tensor forward(Tensor x) {
            long B = x.size(0), N = x.size(1), C = x.size(2);
            Tensor q = query.forward(x).reshape(B, N, heads, headDim).transpose(1, 2);
            Tensor k = key.forward(x).reshape(B, N, heads, headDim).transpose(1, 2);
            Tensor v = value.forward(x).reshape(B, N, heads, headDim).transpose(1, 2);
            double scale = 1.0 / Math.sqrt(headDim);
            Tensor attn = softmax(matmul(q, k.transpose(-2, -1)).mul(new Scalar(scale)), -1L);
            Tensor out = matmul(attn, v).transpose(1, 2).contiguous().reshape(B, N, C);
            return output.forward(out);
        }
    }

    @Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
    public static class DinoMlp extends Module {
        static { Loader.load(org.bytedeco.pytorch.presets.torch.class); }
        public final LinearImpl fc1;
        public final LinearImpl fc2;
        public DinoMlp(int hidden, int inter) {
            super("DinoMlp");
            this.fc1 = register_module("fc1", new LinearImpl(new LinearOptions(hidden, inter).bias(true)));
            this.fc2 = register_module("fc2", new LinearImpl(new LinearOptions(inter, hidden).bias(true)));
        }
        @Override
        public Tensor forward(Tensor x) {
            return fc2.forward(gelu(fc1.forward(x)));
        }
    }

    private static int readInt(String json, String key, int def) {
        try {
            String pat = "\"" + key + "\"";
            int i = json.indexOf(pat);
            if (i < 0) return def;
            String rest = json.substring(i + pat.length());
            rest = rest.replaceAll("^[^0-9-]+", "");
            int end = 0;
            while (end < rest.length() && (Character.isDigit(rest.charAt(end)) || rest.charAt(end) == '-')) end++;
            return Integer.parseInt(rest.substring(0, end));
        } catch (Exception e) {
            return def;
        }
    }
}
