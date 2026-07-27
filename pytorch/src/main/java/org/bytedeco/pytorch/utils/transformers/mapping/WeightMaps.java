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
package org.bytedeco.pytorch.utils.transformers.mapping;

/**
 * Built-in weight maps. Qwen2 / Llama modules use HF-identical parameter names,
 * so their maps are identity.
 *
 * <p>GPT-2 needs Conv1D-style transpose ({@code [in,out]} → {@code [out,in]}) on
 * {@code c_attn}/{@code c_proj}/{@code c_fc} weights.
 */
public final class WeightMaps {

    private WeightMaps() {}

    public static WeightMap identity() {
        return WeightMap.identity();
    }

    public static WeightMap qwen2() {
        return WeightMap.identity();
    }

    public static WeightMap llama() {
        return WeightMap.identity();
    }

    public static WeightMap mistral() {
        return WeightMap.identity();
    }

    /**
     * GPT-2 HF → CausalLM module keys.
     *
     * <p>HF uses Conv1D layout {@code [in, out]} and names
     * {@code h.N.attn.c_attn} / {@code h.N.mlp.c_fc} / {@code h.N.mlp.c_proj}.
     * Our {@code CausalLM} registers {@code h/N.attn.c_attn},
     * {@code h/N.mlp.fc_in}, {@code h/N.mlp.fc_out} as Linear {@code [out, in]}.
     *
     * <p>Rules must rewrite both the layer-index slash and the MLP names;
     * a bare {@code $1} keep-HF-key rule leaves dots and never matches modules.
     */
    public static WeightMap gpt2() {
        return WeightMap.builder()
                // attention qkv / out (Conv1D → Linear transpose)
                .rule(new WeightMap.Rule(
                        "^h\\.(\\d+)\\.attn\\.c_attn\\.weight$",
                        "h/$1.attn.c_attn.weight", true, WeightMap.Transform.TRANSPOSE))
                .rule(new WeightMap.Rule(
                        "^h\\.(\\d+)\\.attn\\.c_attn\\.bias$",
                        "h/$1.attn.c_attn.bias", true, WeightMap.Transform.NONE))
                .rule(new WeightMap.Rule(
                        "^h\\.(\\d+)\\.attn\\.c_proj\\.weight$",
                        "h/$1.attn.c_proj.weight", true, WeightMap.Transform.TRANSPOSE))
                .rule(new WeightMap.Rule(
                        "^h\\.(\\d+)\\.attn\\.c_proj\\.bias$",
                        "h/$1.attn.c_proj.bias", true, WeightMap.Transform.NONE))
                // MLP: HF c_fc/c_proj → module fc_in/fc_out
                .rule(new WeightMap.Rule(
                        "^h\\.(\\d+)\\.mlp\\.c_fc\\.weight$",
                        "h/$1.mlp.fc_in.weight", true, WeightMap.Transform.TRANSPOSE))
                .rule(new WeightMap.Rule(
                        "^h\\.(\\d+)\\.mlp\\.c_fc\\.bias$",
                        "h/$1.mlp.fc_in.bias", true, WeightMap.Transform.NONE))
                .rule(new WeightMap.Rule(
                        "^h\\.(\\d+)\\.mlp\\.c_proj\\.weight$",
                        "h/$1.mlp.fc_out.weight", true, WeightMap.Transform.TRANSPOSE))
                .rule(new WeightMap.Rule(
                        "^h\\.(\\d+)\\.mlp\\.c_proj\\.bias$",
                        "h/$1.mlp.fc_out.bias", true, WeightMap.Transform.NONE))
                .build();
    }

    /** Prefer classpath resource; fall back to built-in. */
    public static WeightMap forModelType(String modelType) {
        if (modelType == null) return identity();
        String t = modelType.toLowerCase();
        String res = "org/bytedeco/pytorch/transformers/weight_maps/" + t + "_hf.json";
        try {
            return WeightMap.fromResource(res);
        } catch (Exception ignored) {
            // fall through
        }
        return switch (t) {
            case "qwen2", "qwen" -> qwen2();
            case "llama", "llama2", "llama3" -> llama();
            case "mistral" -> mistral();
            case "gpt2", "gpt" -> gpt2();
            default -> identity();
        };
    }
}
