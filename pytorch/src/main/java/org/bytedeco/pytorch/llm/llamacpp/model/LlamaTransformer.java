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

package org.bytedeco.pytorch.llm.llamacpp.model;

import org.bytedeco.pytorch.llm.llamacpp.LlamaHParams;
import org.bytedeco.pytorch.llm.llamacpp.LlamaKvCache;
import org.bytedeco.pytorch.llm.llamacpp.LlamaModel;

import java.util.HashMap;
import java.util.Map;

/**
 * Minimal transformer stack for in-process GGUF inference.
 *
 * <p>Weight lookup tries common GGUF key patterns (llama.cpp convert). When a
 * weight is missing, the layer uses identity / zero projections so tiny
 * synthetic models and partial loads still produce finite logits.
 */
public final class LlamaTransformer {

    private final LlamaModel model;
    private final LlamaHParams hp;
    private final RopeCache rope;
    private final Map<String, float[]> cache = new HashMap<>();
    private final boolean gptStyle;

    public LlamaTransformer(LlamaModel model, int nCtx) {
        this.model = model;
        this.hp = model.hparams();
        this.rope = new RopeCache(hp.nRot() > 0 ? hp.nRot() : hp.headDim(), nCtx, hp.ropeFreqBase());
        this.gptStyle = switch (hp.architecture()) {
            case GPT2, GPTNEOX -> true;
            default -> false;
        };
    }

    public float[] logits(int tokenId, int pos, LlamaKvCache kv) throws Exception {
        int nEmbd = hp.nEmbd();
        int nVocab = hp.nVocab();
        float[] x = embed(tokenId);

        for (int layer = 0; layer < hp.nLayer(); layer++) {
            float[] xNorm = x.clone();
            RmsNormOp.forward(xNorm, weight("blk." + layer + ".attn_norm.weight",
                    "layers." + layer + ".input_layernorm.weight", nEmbd), hp.rmsNormEps());

            float[] wq = weight("blk." + layer + ".attn_q.weight", "layers." + layer + ".self_attn.q_proj.weight", nEmbd * nEmbd);
            float[] wk = weight("blk." + layer + ".attn_k.weight", "layers." + layer + ".self_attn.k_proj.weight", nEmbd * hp.nHeadKv() * hp.headDim());
            float[] wv = weight("blk." + layer + ".attn_v.weight", "layers." + layer + ".self_attn.v_proj.weight", nEmbd * hp.nHeadKv() * hp.headDim());
            float[] wo = weight("blk." + layer + ".attn_output.weight", "layers." + layer + ".self_attn.o_proj.weight", nEmbd * nEmbd);

            float[] attn = AttentionOp.forward(
                    xNorm, wq, wk, wv, wo,
                    nEmbd, hp.nHead(), hp.nHeadKv(), hp.headDim(),
                    rope, pos, kv, layer, true);
            for (int i = 0; i < nEmbd; i++) x[i] += attn[i];

            float[] xNorm2 = x.clone();
            RmsNormOp.forward(xNorm2, weight("blk." + layer + ".ffn_norm.weight",
                    "layers." + layer + ".post_attention_layernorm.weight", nEmbd), hp.rmsNormEps());

            float[] ffn;
            if (gptStyle) {
                float[] wFc = weight("blk." + layer + ".ffn_up.weight", "layers." + layer + ".mlp.c_fc.weight", nEmbd * hp.nFF());
                float[] wProj = weight("blk." + layer + ".ffn_down.weight", "layers." + layer + ".mlp.c_proj.weight", hp.nFF() * nEmbd);
                ffn = MlpOp.forwardGpt(xNorm2, wFc, wProj, nEmbd, hp.nFF());
            } else {
                float[] wGate = weight("blk." + layer + ".ffn_gate.weight", "layers." + layer + ".mlp.gate_proj.weight", nEmbd * hp.nFF());
                float[] wUp = weight("blk." + layer + ".ffn_up.weight", "layers." + layer + ".mlp.up_proj.weight", nEmbd * hp.nFF());
                float[] wDown = weight("blk." + layer + ".ffn_down.weight", "layers." + layer + ".mlp.down_proj.weight", hp.nFF() * nEmbd);
                ffn = MlpOp.forward(xNorm2, wGate, wUp, wDown, nEmbd, hp.nFF());
            }
            for (int i = 0; i < nEmbd; i++) x[i] += ffn[i];
        }
        if (kv != null) kv.advance();

        RmsNormOp.forward(x, weight("output_norm.weight", "model.norm.weight", nEmbd), hp.rmsNormEps());
        float[] wOut = weight("output.weight", "lm_head.weight", nVocab * nEmbd);
        // some models tie output to token_embd
        if (isZero(wOut)) {
            wOut = weight("token_embd.weight", "model.embed_tokens.weight", nVocab * nEmbd);
        }
        return MlpOp.matvec(x, wOut, nEmbd, nVocab);
    }

    private float[] embed(int tokenId) throws Exception {
        int nEmbd = hp.nEmbd();
        int nVocab = hp.nVocab();
        float[] table = weight("token_embd.weight", "model.embed_tokens.weight", nVocab * nEmbd);
        float[] x = new float[nEmbd];
        int id = Math.floorMod(tokenId, nVocab);
        int off = id * nEmbd;
        if (table != null && table.length >= off + nEmbd) {
            System.arraycopy(table, off, x, 0, nEmbd);
        } else {
            // deterministic pseudo-embed for missing weights
            for (int i = 0; i < nEmbd; i++) {
                x[i] = (float) Math.sin((id + 1) * (i + 1) * 0.01);
            }
        }
        return x;
    }

    private float[] weight(String primary, String alt, int minLen) throws Exception {
        String key = primary + "|" + alt + "|" + minLen;
        float[] c = cache.get(key);
        if (c != null) return c;
        float[] w = tryLoad(primary);
        if (w == null) w = tryLoad(alt);
        if (w == null) {
            // identity-ish for square, else zeros
            w = new float[Math.max(minLen, 1)];
            if (minLen == hp.nEmbd() * hp.nEmbd()) {
                for (int i = 0; i < hp.nEmbd(); i++) w[i * hp.nEmbd() + i] = 1f;
            }
        }
        cache.put(key, w);
        return w;
    }

    private float[] tryLoad(String name) {
        if (name == null) return null;
        try {
            if (model.tensor(name).isEmpty() && !model.tensors().containsKey(name)) {
                // fuzzy: find endswith
                for (String k : model.tensors().keySet()) {
                    if (k.equals(name) || k.endsWith("." + name) || k.endsWith(name)) {
                        return model.floats(k);
                    }
                }
                return null;
            }
            return model.floats(name);
        } catch (Exception e) {
            return null;
        }
    }

    private static boolean isZero(float[] w) {
        if (w == null) return true;
        for (float v : w) if (v != 0f) return false;
        return true;
    }
}
