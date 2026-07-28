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
import java.util.Map;
import java.util.Objects;

import static org.bytedeco.pytorch.global.torch.ScalarType;
import static org.bytedeco.pytorch.global.torch.cat;
import static org.bytedeco.pytorch.global.torch.gelu;
import static org.bytedeco.pytorch.global.torch.matmul;
import static org.bytedeco.pytorch.global.torch.softmax;
import static org.bytedeco.pytorch.global.torch.tensor;
import static org.bytedeco.pytorch.global.torch.zeros;

/**
 * OpenAI CLIP ViT-B/32 vision tower (+ optional text projection dim 512).
 *
 * <p>Loads {@code pytorch_model.bin} via a Python helper dump to safetensors when
 * needed, or binds vision_* keys from an already-converted map.
 */
@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public final class ClipEncoder extends Module implements MediaEncoder {

    static { Loader.load(org.bytedeco.pytorch.presets.torch.class); }

    private final String name;
    private final int imageSize;
    private final int patchSize;
    private final int hiddenSize;
    private final int projDim;
    private final int numHeads;
    private final int numLayers;
    private final WeightBinder.Report loadReport;

    private final LongPointer patchKernel;
    private final LongPointer patchStride;
    private final Conv2dImpl patchEmbed;
    private Tensor classEmbedding;   // [H]
    private Tensor posEmbed;         // [N+1, H] or [1,N+1,H]
    private final List<ClipBlock> blocks = new ArrayList<>();
    private final LayerNormImpl preLn;
    private final LayerNormImpl postLn;
    private final LinearImpl visualProjection; // [proj, hidden]

    public ClipEncoder(Path dir) throws Exception {
        super("ClipEncoder");
        Objects.requireNonNull(dir, "dir");
        this.name = dir.toString();
        int img = 224, patch = 32, hidden = 768, heads = 12, layers = 12, proj = 512, inter = 3072;
        Path cfg = dir.resolve("config.json");
        if (Files.isRegularFile(cfg)) {
            String json = Files.readString(cfg);
            // nested vision_config
            img = readNestedInt(json, "image_size", img);
            patch = readNestedInt(json, "patch_size", patch);
            hidden = readNestedInt(json, "hidden_size", hidden);
            // prefer vision block values: crude — last occurrence often vision
            heads = readNestedInt(json, "num_attention_heads", heads);
            layers = readNestedInt(json, "num_hidden_layers", layers);
            inter = readNestedInt(json, "intermediate_size", inter);
            proj = readInt(json, "projection_dim", proj);
        }
        this.imageSize = img;
        this.patchSize = patch;
        this.hiddenSize = hidden;
        this.projDim = proj;
        this.numHeads = heads;
        this.numLayers = layers;

        this.patchKernel = new LongPointer(new long[]{patch, patch});
        this.patchStride = new LongPointer(new long[]{patch, patch});
        Conv2dOptions copt = new Conv2dOptions(3, hidden, patchKernel);
        copt.stride(patchStride);
        copt.bias(false); // CLIP patch embed has no bias
        this.patchEmbed = register_module("vision_model/embeddings/patch_embedding", new Conv2dImpl(copt));
        int numPatches = (img / patch) * (img / patch);
        this.classEmbedding = register_parameter("vision_model/embeddings/class_embedding",
                zeros(hidden), true);
        this.posEmbed = register_parameter("vision_model/embeddings/position_embedding/weight",
                zeros(numPatches + 1, hidden), true);
        LongVector lnShape = new LongVector().put((long) hidden);
        this.preLn = register_module("vision_model/pre_layrnorm", new LayerNormImpl(lnShape)); // HF typo "pre_layrnorm"
        // also alias name pre_layernorm for matching
        for (int i = 0; i < layers; i++) {
            blocks.add(register_module("vision_model/encoder/layers/" + i,
                    new ClipBlock(hidden, heads, inter)));
        }
        this.postLn = register_module("vision_model/post_layernorm", new LayerNormImpl(lnShape));
        this.visualProjection = register_module("visual_projection",
                new LinearImpl(new LinearOptions(hidden, proj).bias(false)));

        this.eval();
        Map<String, Tensor> weights = loadClipWeights(dir);
        this.loadReport = WeightBinder.bind(this, weights,
                List.of("module."), false);
        System.out.println("[ClipEncoder] " + loadReport + " dir=" + dir.getFileName());
    }

    public static ClipEncoder fromDirectory(Path dir) throws Exception {
        return new ClipEncoder(dir);
    }

    public WeightBinder.Report loadReport() { return loadReport; }

    @Override public MediaType modality() { return MediaType.IMAGE; }
    @Override
    public String encoderName() { return "clip:" + name; }
    @Override public int featureDim() { return projDim; }

    @Override
    public EncoderFeatures encode(MediaInput input) {
        long t0 = System.nanoTime();
        try {
            Tensor pixels = ImagePreprocess.loadNormalized(
                    input, imageSize, ImagePreprocess.CLIP_MEAN, ImagePreprocess.CLIP_STD);
            Tensor pooled = forwardVision(pixels); // [1, proj]
            float[] pool = ImagePreprocess.toFloatArray(pooled.reshape(-1));
            // L2 normalize (CLIP convention)
            double n = 0;
            for (float v : pool) n += v * v;
            n = Math.sqrt(n);
            if (n > 1e-8) for (int i = 0; i < pool.length; i++) pool[i] /= (float) n;
            double ms = (System.nanoTime() - t0) / 1e6;
            return new EncoderFeatures(pool, new float[][]{pool}, encoderName(), ms);
        } catch (Exception e) {
            System.out.println("[ClipEncoder] encode failed: " + e.getMessage());
            return EncoderFeatures.empty(encoderName());
        }
    }

    public Tensor forwardVision(Tensor pixelValues) {
        // [B,3,H,W] → patch tokens
        Tensor x = patchEmbed.forward(pixelValues); // [B,D,Gh,Gw]
        long B = x.size(0);
        long Gh = x.size(2), Gw = x.size(3);
        x = x.reshape(B, x.size(1), Gh * Gw).transpose(1, 2).contiguous(); // [B,N,D]
        Tensor cls = classEmbedding.reshape(1, 1, hiddenSize);
        if (B > 1) cls = cls.repeat(new long[]{B, 1, 1});
        x = cat(new org.bytedeco.pytorch.TensorVector(cls, x), 1);
        Tensor pe = posEmbed.dim() == 2 ? posEmbed.unsqueeze(0) : posEmbed;
        if (pe.size(1) != x.size(1)) {
            pe = pe.slice(1, new org.bytedeco.pytorch.LongOptional(0),
                    new org.bytedeco.pytorch.LongOptional(x.size(1)), 1);
        }
        if (pe.size(0) == 1 && B > 1) pe = pe.repeat(new long[]{B, 1, 1});
        x = x.add(pe);
        try {
            x = preLn.forward(x);
        } catch (Throwable ignored) {
            // some dumps only have post_layernorm
        }
        for (ClipBlock b : blocks) x = b.forward(x);
        Tensor clsOut = postLn.forward(x.select(1, 0)); // [B,D]
        return visualProjection.forward(clsOut); // [B, proj]
    }

    @Override
    public Tensor forward(Tensor input) {
        return forwardVision(input);
    }

    /** Load pytorch_model.bin via torch/python dump or safetensors if present. */
    static Map<String, Tensor> loadClipWeights(Path dir) throws Exception {
        Path st = dir.resolve("model.safetensors");
        if (Files.isRegularFile(st) && Files.size(st) > 1_000_000L) {
            return WeightBinder.loadSafetensors(st);
        }
        Path visionSt = dir.resolve("vision_weights.safetensors");
        if (Files.isRegularFile(visionSt) && Files.size(visionSt) > 1_000_000L) {
            System.out.println("[ClipEncoder] using vision_weights.safetensors");
            return WeightBinder.loadSafetensors(visionSt);
        }
        Path bin = dir.resolve("pytorch_model.bin");
        if (!Files.isRegularFile(bin)) {
            throw new IllegalStateException("No CLIP weights in " + dir);
        }
        // Convert bin → temp safetensors via python torch (available on this machine)
        Path out = dir.resolve("vision_weights.safetensors");
        if (!Files.isRegularFile(out) || Files.size(out) < 1_000_000) {
            ProcessBuilder pb = new ProcessBuilder(
                    "python3", "-c",
                    "import torch; from safetensors.torch import save_file; "
                            + "sd=torch.load(r'" + bin.toAbsolutePath() + "', map_location='cpu', weights_only=True); "
                            + "out={}; "
                            + "for k,v in sd.items():\n"
                            + "  if k.startswith('vision_model') or k.startswith('visual_projection') or k=='logit_scale':\n"
                            + "    if v.dtype==torch.float16: v=v.float()\n"
                            + "    if v.is_floating_point(): out[k]=v.contiguous()\n"
                            + "save_file(out, r'" + out.toAbsolutePath() + "'); print(len(out))"
            );
            pb.redirectErrorStream(true);
            Process p = pb.start();
            String log = new String(p.getInputStream().readAllBytes());
            int code = p.waitFor();
            if (code != 0 || !Files.isRegularFile(out)) {
                throw new IllegalStateException("CLIP bin→safetensors failed: " + log);
            }
            System.out.println("[ClipEncoder] converted pytorch_model.bin → vision_weights.safetensors " + log.trim());
        }
        return WeightBinder.loadSafetensors(out);
    }

    // ---- blocks: separate q/k/v like CLIP HF ---------------------------------

    @Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
    public static class ClipBlock extends Module {
        static { Loader.load(org.bytedeco.pytorch.presets.torch.class); }
        public final LayerNormImpl ln1;
        public final ClipAttn attn;
        public final LayerNormImpl ln2;
        public final ClipMlp mlp;
        public ClipBlock(int hidden, int heads, int inter) {
            super("ClipBlock");
            LongVector s = new LongVector().put((long) hidden);
            this.ln1 = register_module("layer_norm1", new LayerNormImpl(s));
            this.attn = register_module("self_attn", new ClipAttn(hidden, heads));
            this.ln2 = register_module("layer_norm2", new LayerNormImpl(s));
            this.mlp = register_module("mlp", new ClipMlp(hidden, inter));
        }
        @Override
        public Tensor forward(Tensor x) {
            x = x.add(attn.forward(ln1.forward(x)));
            return x.add(mlp.forward(ln2.forward(x)));
        }
    }

    @Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
    public static class ClipAttn extends Module {
        static { Loader.load(org.bytedeco.pytorch.presets.torch.class); }
        public final LinearImpl q_proj, k_proj, v_proj, out_proj;
        private final int heads, headDim;
        public ClipAttn(int hidden, int heads) {
            super("ClipAttn");
            this.heads = heads;
            this.headDim = hidden / heads;
            this.q_proj = register_module("q_proj", new LinearImpl(new LinearOptions(hidden, hidden).bias(true)));
            this.k_proj = register_module("k_proj", new LinearImpl(new LinearOptions(hidden, hidden).bias(true)));
            this.v_proj = register_module("v_proj", new LinearImpl(new LinearOptions(hidden, hidden).bias(true)));
            this.out_proj = register_module("out_proj", new LinearImpl(new LinearOptions(hidden, hidden).bias(true)));
        }
        @Override
        public Tensor forward(Tensor x) {
            long B = x.size(0), N = x.size(1), C = x.size(2);
            Tensor q = q_proj.forward(x).reshape(B, N, heads, headDim).transpose(1, 2);
            Tensor k = k_proj.forward(x).reshape(B, N, heads, headDim).transpose(1, 2);
            Tensor v = v_proj.forward(x).reshape(B, N, heads, headDim).transpose(1, 2);
            double scale = 1.0 / Math.sqrt(headDim);
            Tensor attn = softmax(matmul(q, k.transpose(-2, -1)).mul(new Scalar(scale)), -1L);
            Tensor out = matmul(attn, v).transpose(1, 2).contiguous().reshape(B, N, C);
            return out_proj.forward(out);
        }
    }

    @Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
    public static class ClipMlp extends Module {
        static { Loader.load(org.bytedeco.pytorch.presets.torch.class); }
        public final LinearImpl fc1, fc2;
        public ClipMlp(int hidden, int inter) {
            super("ClipMlp");
            this.fc1 = register_module("fc1", new LinearImpl(new LinearOptions(hidden, inter).bias(true)));
            this.fc2 = register_module("fc2", new LinearImpl(new LinearOptions(inter, hidden).bias(true)));
        }
        @Override
        public Tensor forward(Tensor x) {
            // CLIP uses quick_gelu ≈ x * sigmoid(1.702x); gelu is close enough for features
            return fc2.forward(gelu(fc1.forward(x)));
        }
    }

    private static int readInt(String json, String key, int def) {
        try {
            String pat = "\"" + key + "\"";
            int i = json.indexOf(pat);
            if (i < 0) return def;
            String rest = json.substring(i + pat.length()).replaceAll("^[^0-9-]+", "");
            int end = 0;
            while (end < rest.length() && (Character.isDigit(rest.charAt(end)) || rest.charAt(end) == '-')) end++;
            return Integer.parseInt(rest.substring(0, end));
        } catch (Exception e) { return def; }
    }

    /** Prefer value inside vision_config block when present. */
    private static int readNestedInt(String json, String key, int def) {
        int vis = json.indexOf("\"vision_config\"");
        if (vis >= 0) {
            String sub = json.substring(vis, Math.min(json.length(), vis + 2500));
            int v = readInt(sub, key, Integer.MIN_VALUE);
            if (v != Integer.MIN_VALUE) return v;
        }
        return readInt(json, key, def);
    }
}
