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

import java.util.ArrayList;
import java.util.List;

import static org.bytedeco.pytorch.global.torch.cat;
import static org.bytedeco.pytorch.global.torch.gelu;
import static org.bytedeco.pytorch.global.torch.matmul;
import static org.bytedeco.pytorch.global.torch.softmax;
import static org.bytedeco.pytorch.global.torch.zeros;

/**
 * Shared Vision Transformer used by DINOv2 / CLIP-ViT / SmolVLM vision towers.
 *
 * <p>Layout is intentionally close to HF so weight binding can map:
 * <ul>
 *   <li>{@code embeddings.patch_embeddings.projection} / {@code embeddings.patch_embedding}</li>
 *   <li>{@code embeddings.cls_token} / {@code embeddings.class_embedding}</li>
 *   <li>{@code embeddings.position_embeddings} / {@code embeddings.position_embedding.weight}</li>
 *   <li>{@code encoder.layer.N.*} or {@code encoder.layers.N.*}</li>
 *   <li>{@code layernorm} / {@code post_layernorm}</li>
 * </ul>
 */
@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class VisionTransformer extends Module {

    static { Loader.load(org.bytedeco.pytorch.presets.torch.class); }

    public final int imageSize;
    public final int patchSize;
    public final int hiddenSize;
    public final int numLayers;
    public final int numHeads;
    public final int numPatches;
    public final boolean useCls;
    public final boolean useLayerScale;

    public final Conv2dImpl patchEmbed;
    public Tensor clsToken;           // [1,1,H] or null
    public Tensor posEmbed;           // [1, N(+1), H] or [N(+1), H]
    public final List<ViTBlock> blocks = new ArrayList<>();
    public final LayerNormImpl finalNorm;

    public VisionTransformer(int imageSize, int patchSize, int hiddenSize,
                             int numLayers, int numHeads, int intermediateSize,
                             boolean useCls, boolean useLayerScale, double layerNormEps) {
        super("VisionTransformer");
        this.imageSize = imageSize;
        this.patchSize = patchSize;
        this.hiddenSize = hiddenSize;
        this.numLayers = numLayers;
        this.numHeads = numHeads;
        this.useCls = useCls;
        this.useLayerScale = useLayerScale;
        this.numPatches = (imageSize / patchSize) * (imageSize / patchSize);

        LongPointer k = new LongPointer(new long[]{patchSize, patchSize});
        Conv2dOptions copt = new Conv2dOptions(3, hiddenSize, k);
        copt.stride(k);
        copt.bias(true);
        this.patchEmbed = register_module("patch_embed", new Conv2dImpl(copt));

        int seq = numPatches + (useCls ? 1 : 0);
        if (useCls) {
            this.clsToken = register_parameter("cls_token",
                    zeros(1, 1, hiddenSize), true);
        } else {
            this.clsToken = null;
        }
        this.posEmbed = register_parameter("pos_embed",
                zeros(1, seq, hiddenSize), true);

        LongVector lnShape = new LongVector().put((long) hiddenSize);
        for (int i = 0; i < numLayers; i++) {
            blocks.add(register_module("blocks/" + i,
                    new ViTBlock(hiddenSize, numHeads, intermediateSize, useLayerScale, layerNormEps)));
        }
        this.finalNorm = register_module("norm", new LayerNormImpl(lnShape));
    }

    /**
     * Forward image batch {@code [B,3,H,W]} → sequence {@code [B, N(+1), D]}.
     */
    public Tensor forwardTokens(Tensor pixelValues) {
        // patch embed: [B, D, Gh, Gw] → [B, Gh*Gw, D]
        Tensor x = patchEmbed.forward(pixelValues);
        long B = x.size(0);
        long D = x.size(1);
        long Gh = x.size(2);
        long Gw = x.size(3);
        x = x.flatten(2L, 3L).transpose(1, 2); // [B, N, D]

        if (useCls && clsToken != null) {
            Tensor cls = clsToken.expand(B, 1L, clsToken.size(2));
            x = cat(new org.bytedeco.pytorch.TensorVector(cls, x), 1);
        }

        // add position embeddings (interpolate if needed for size mismatch)
        Tensor pe = posEmbed;
        if (pe.dim() == 2) pe = pe.unsqueeze(0);
        if (pe.size(1) != x.size(1)) {
            // truncate or tile as fallback
            if (pe.size(1) > x.size(1)) {
                pe = pe.slice(1, new org.bytedeco.pytorch.LongOptional(0),
                        new org.bytedeco.pytorch.LongOptional(x.size(1)), 1);
            } else {
                // pad with last pos
                long need = x.size(1) - pe.size(1);
                Tensor last = pe.slice(1, new org.bytedeco.pytorch.LongOptional(pe.size(1) - 1),
                        new org.bytedeco.pytorch.LongOptional(pe.size(1)), 1);
                Tensor[] parts = new Tensor[(int) need + 1];
                parts[0] = pe;
                for (int i = 0; i < need; i++) parts[i + 1] = last;
                org.bytedeco.pytorch.TensorVector tv = new org.bytedeco.pytorch.TensorVector();
                for (Tensor p : parts) tv.put(p);
                pe = cat(tv, 1);
            }
        }
        x = x.add(pe);

        for (ViTBlock block : blocks) {
            x = block.forward(x);
        }
        return finalNorm.forward(x);
    }

    /** CLS (or mean) pooled embedding {@code [B, D]}. */
    public Tensor forwardPooled(Tensor pixelValues) {
        Tensor tokens = forwardTokens(pixelValues);
        if (useCls) {
            return tokens.select(1, 0); // [B, D]
        }
        return tokens.mean(1L);
    }

    @Override
    public Tensor forward(Tensor input) {
        return forwardPooled(input);
    }

    // ---- block ----------------------------------------------------------------

    @Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
    public static class ViTBlock extends Module {
        static { Loader.load(org.bytedeco.pytorch.presets.torch.class); }

        public final LayerNormImpl norm1;
        public final Attention attn;
        public final LayerNormImpl norm2;
        public final Mlp mlp;
        public Tensor ls1; // optional layer scale
        public Tensor ls2;

        public ViTBlock(int hidden, int heads, int intermediate, boolean layerScale, double eps) {
            super("ViTBlock");
            LongVector shape = new LongVector().put((long) hidden);
            this.norm1 = register_module("norm1", new LayerNormImpl(shape));
            this.attn = register_module("attn", new Attention(hidden, heads));
            this.norm2 = register_module("norm2", new LayerNormImpl(shape));
            this.mlp = register_module("mlp", new Mlp(hidden, intermediate));
            if (layerScale) {
                this.ls1 = register_parameter("ls1",
                        org.bytedeco.pytorch.global.torch.ones(hidden), true);
                this.ls2 = register_parameter("ls2",
                        org.bytedeco.pytorch.global.torch.ones(hidden), true);
            } else {
                this.ls1 = null;
                this.ls2 = null;
            }
        }

        @Override
        public Tensor forward(Tensor x) {
            Tensor a = attn.forward(norm1.forward(x));
            if (ls1 != null) a = a.mul(ls1);
            x = x.add(a);
            Tensor m = mlp.forward(norm2.forward(x));
            if (ls2 != null) m = m.mul(ls2);
            return x.add(m);
        }
    }

    @Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
    public static class Attention extends Module {
        static { Loader.load(org.bytedeco.pytorch.presets.torch.class); }

        public final LinearImpl qkv;
        public final LinearImpl proj;
        private final int heads;
        private final int headDim;

        public Attention(int hidden, int heads) {
            super("Attention");
            this.heads = heads;
            this.headDim = hidden / heads;
            this.qkv = register_module("qkv",
                    new LinearImpl(new LinearOptions(hidden, 3L * hidden).bias(true)));
            this.proj = register_module("proj",
                    new LinearImpl(new LinearOptions(hidden, hidden).bias(true)));
        }

        @Override
        public Tensor forward(Tensor x) {
            // x: [B, N, C]
            long B = x.size(0);
            long N = x.size(1);
            long C = x.size(2);
            Tensor mixed = qkv.forward(x); // [B,N,3C]
            // reshape to [3, B, heads, N, headDim]
            Tensor qkvT = mixed.reshape(B, N, 3, heads, headDim).permute(2, 0, 3, 1, 4);
            Tensor q = qkvT.select(0, 0); // [B, heads, N, headDim]
            Tensor k = qkvT.select(0, 1);
            Tensor v = qkvT.select(0, 2);
            double scale = 1.0 / Math.sqrt(headDim);
            Tensor attn = matmul(q, k.transpose(-2, -1)).mul(new Scalar(scale));
            attn = softmax(attn, -1L);
            Tensor out = matmul(attn, v); // [B, heads, N, headDim]
            out = out.transpose(1, 2).contiguous().reshape(B, N, C);
            return proj.forward(out);
        }
    }

    @Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
    public static class Mlp extends Module {
        static { Loader.load(org.bytedeco.pytorch.presets.torch.class); }

        public final LinearImpl fc1;
        public final LinearImpl fc2;

        public Mlp(int hidden, int intermediate) {
            super("Mlp");
            this.fc1 = register_module("fc1",
                    new LinearImpl(new LinearOptions(hidden, intermediate).bias(true)));
            this.fc2 = register_module("fc2",
                    new LinearImpl(new LinearOptions(intermediate, hidden).bias(true)));
        }

        @Override
        public Tensor forward(Tensor x) {
            return fc2.forward(gelu(fc1.forward(x)));
        }
    }
}
