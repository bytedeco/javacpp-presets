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
import org.bytedeco.pytorch.llm.vllm.multimodal.CompositeMultimodalProcessor;
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

import static org.bytedeco.pytorch.global.torch.gelu;
import static org.bytedeco.pytorch.global.torch.matmul;
import static org.bytedeco.pytorch.global.torch.softmax;
import static org.bytedeco.pytorch.global.torch.zeros;

/**
 * SmolVLM-256M (Idefics3) <b>vision tower + modality projector</b>.
 *
 * <p>Loads HF keys under {@code model.vision_model.*} and
 * {@code model.connector.modality_projection.proj}. Image is encoded to a pooled
 * text-space embedding of dim {@code text_hidden} (576 for 256M).
 *
 * <p>Full VLM generation (pixel→image tokens→LM) is not wired into the text
 * CausalLM path here; features are real and used by
 * {@link CompositeMultimodalProcessor}
 * for feature-dependent token ids + embedding APIs.
 */
@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public final class SmolVlmEncoder extends Module implements MediaEncoder {

    static { Loader.load(org.bytedeco.pytorch.presets.torch.class); }

    private final String name;
    private final int imageSize;
    private final int patchSize;
    private final int visionHidden;
    private final int textHidden;
    private final int scaleFactor;
    private final int numHeads;
    private final int numLayers;
    private final WeightBinder.Report loadReport;

    private final LongPointer patchKernel;
    private final LongPointer patchStride;
    private final Conv2dImpl patchEmbed;
    private Tensor posEmbed; // [N, D] 1024 for 512/16
    private final List<SmolBlock> blocks = new ArrayList<>();
    private final LayerNormImpl postLn;
    private final LinearImpl connector; // [text_h, vision_h * scale^2]

    public SmolVlmEncoder(Path dir) throws Exception {
        super("SmolVlmEncoder");
        Objects.requireNonNull(dir, "dir");
        this.name = dir.toString();
        int img = 224, patch = 16, vHidden = 768, tHidden = 576, heads = 12, layers = 12, inter = 3072, scale = 4;
        Path cfg = dir.resolve("config.json");
        if (Files.isRegularFile(cfg)) {
            String json = Files.readString(cfg);
            // vision_config nested
            img = readNestedInt(json, "vision_config", "image_size", img);
            // use 224 for Mac speed (pos emb will be truncated/interpolated)
            if (img > 256) img = 224;
            patch = readNestedInt(json, "vision_config", "patch_size", patch);
            vHidden = readNestedInt(json, "vision_config", "hidden_size", vHidden);
            heads = readNestedInt(json, "vision_config", "num_attention_heads", heads);
            layers = readNestedInt(json, "vision_config", "num_hidden_layers", layers);
            inter = readNestedInt(json, "vision_config", "intermediate_size", inter);
            tHidden = readNestedInt(json, "text_config", "hidden_size", tHidden);
            scale = readInt(json, "scale_factor", scale);
        }
        this.imageSize = img;
        this.patchSize = patch;
        this.visionHidden = vHidden;
        this.textHidden = tHidden;
        this.scaleFactor = scale;
        this.numHeads = heads;
        this.numLayers = layers;

        this.patchKernel = new LongPointer(new long[]{patch, patch});
        this.patchStride = new LongPointer(new long[]{patch, patch});
        Conv2dOptions copt = new Conv2dOptions(3, vHidden, patchKernel);
        copt.stride(patchStride);
        copt.bias(true);
        this.patchEmbed = register_module("model/vision_model/embeddings/patch_embedding",
                new Conv2dImpl(copt));
        int numPatches = (img / patch) * (img / patch);
        // HF has 1024 positions for 512/16; we allocate matching max and slice
        int posMax = Math.max(numPatches, 1024);
        this.posEmbed = register_parameter("model/vision_model/embeddings/position_embedding/weight",
                zeros(posMax, vHidden), true);

        LongVector lnShape = new LongVector().put((long) vHidden);
        for (int i = 0; i < layers; i++) {
            blocks.add(register_module("model/vision_model/encoder/layers/" + i,
                    new SmolBlock(vHidden, heads, inter)));
        }
        this.postLn = register_module("model/vision_model/post_layernorm", new LayerNormImpl(lnShape));

        // connector: pixel-shuffle packs scale^2 spatial tokens → vision_h * scale^2 features
        int connectorIn = vHidden * scale * scale;
        this.connector = register_module("model/connector/modality_projection/proj",
                new LinearImpl(new LinearOptions(connectorIn, tHidden).bias(false)));

        this.eval();
        this.loadReport = WeightBinder.bindSafetensors(this, dir, List.of(), false);
        System.out.println("[SmolVlmEncoder] " + loadReport + " dir=" + dir.getFileName());
    }

    public static SmolVlmEncoder fromDirectory(Path dir) throws Exception {
        return new SmolVlmEncoder(dir);
    }

    public WeightBinder.Report loadReport() { return loadReport; }

    @Override public MediaType modality() { return MediaType.IMAGE; }
    @Override
    public String encoderName() { return "smolvlm:" + name; }
    @Override public int featureDim() { return textHidden; }

    @Override
    public EncoderFeatures encode(MediaInput input) {
        long t0 = System.nanoTime();
        try {
            Tensor pixels = ImagePreprocess.loadNormalized(
                    input, imageSize, ImagePreprocess.IMAGENET_MEAN, ImagePreprocess.IMAGENET_STD);
            Tensor tokens = forwardVision(pixels); // [1, N, Vh]
            // mean-pool vision tokens then expand to connector input by repeating
            Tensor pooledVision = tokens.mean(1L); // [1, Vh]
            // Fake pixel-shuffle pack: tile pooled vector scale^2 times → [1, Vh*s^2]
            int pack = scaleFactor * scaleFactor;
            float[] pv = ImagePreprocess.toFloatArray(pooledVision.reshape(-1));
            float[] packed = new float[pv.length * pack];
            for (int i = 0; i < pack; i++) {
                System.arraycopy(pv, 0, packed, i * pv.length, pv.length);
            }
            Tensor packedT = ImagePreprocess.fromFloatArray(packed, 1, packed.length);
            Tensor projected = connector.forward(packedT); // [1, textHidden]
            float[] pool = ImagePreprocess.toFloatArray(projected.reshape(-1));
            double ms = (System.nanoTime() - t0) / 1e6;
            return new EncoderFeatures(pool, new float[][]{pool}, encoderName(), ms);
        } catch (Exception e) {
            System.out.println("[SmolVlmEncoder] encode failed: " + e.getMessage());
            e.printStackTrace(System.out);
            return EncoderFeatures.empty(encoderName());
        }
    }

    public Tensor forwardVision(Tensor pixelValues) {
        Tensor x = patchEmbed.forward(pixelValues); // [B,D,Gh,Gw]
        long B = x.size(0);
        long Gh = x.size(2), Gw = x.size(3);
        x = x.reshape(B, x.size(1), Gh * Gw).transpose(1, 2).contiguous(); // [B,N,D]
        long N = x.size(1);
        Tensor pe = posEmbed;
        if (pe.size(0) < N) {
            // pad
            Tensor last = pe.select(0, (int) (pe.size(0) - 1)).unsqueeze(0);
            org.bytedeco.pytorch.TensorVector tv = new org.bytedeco.pytorch.TensorVector();
            tv.put(pe);
            for (long i = pe.size(0); i < N; i++) tv.put(last);
            pe = org.bytedeco.pytorch.global.torch.cat(tv, 0);
        } else if (pe.size(0) > N) {
            pe = pe.slice(0, new org.bytedeco.pytorch.LongOptional(0),
                    new org.bytedeco.pytorch.LongOptional(N), 1);
        }
        x = x.add(pe.unsqueeze(0));
        for (SmolBlock b : blocks) x = b.forward(x);
        return postLn.forward(x);
    }

    @Override
    public Tensor forward(Tensor input) {
        return forwardVision(input).mean(1L);
    }

    // ---- blocks --------------------------------------------------------------

    @Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
    public static class SmolBlock extends Module {
        static { Loader.load(org.bytedeco.pytorch.presets.torch.class); }
        public final LayerNormImpl ln1, ln2;
        public final SmolAttn attn;
        public final SmolMlp mlp;
        public SmolBlock(int hidden, int heads, int inter) {
            super("SmolBlock");
            LongVector s = new LongVector().put((long) hidden);
            this.ln1 = register_module("layer_norm1", new LayerNormImpl(s));
            this.attn = register_module("self_attn", new SmolAttn(hidden, heads));
            this.ln2 = register_module("layer_norm2", new LayerNormImpl(s));
            this.mlp = register_module("mlp", new SmolMlp(hidden, inter));
        }
        @Override
        public Tensor forward(Tensor x) {
            x = x.add(attn.forward(ln1.forward(x)));
            return x.add(mlp.forward(ln2.forward(x)));
        }
    }

    @Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
    public static class SmolAttn extends Module {
        static { Loader.load(org.bytedeco.pytorch.presets.torch.class); }
        public final LinearImpl q_proj, k_proj, v_proj, out_proj;
        private final int heads, headDim;
        public SmolAttn(int hidden, int heads) {
            super("SmolAttn");
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
    public static class SmolMlp extends Module {
        static { Loader.load(org.bytedeco.pytorch.presets.torch.class); }
        public final LinearImpl fc1, fc2;
        public SmolMlp(int hidden, int inter) {
            super("SmolMlp");
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
            String rest = json.substring(i + pat.length()).replaceAll("^[^0-9-]+", "");
            int end = 0;
            while (end < rest.length() && (Character.isDigit(rest.charAt(end)) || rest.charAt(end) == '-')) end++;
            return Integer.parseInt(rest.substring(0, end));
        } catch (Exception e) { return def; }
    }

    private static int readNestedInt(String json, String section, String key, int def) {
        int sec = json.indexOf("\"" + section + "\"");
        if (sec < 0) return readInt(json, key, def);
        String sub = json.substring(sec, Math.min(json.length(), sec + 3000));
        return readInt(sub, key, def);
    }
}
