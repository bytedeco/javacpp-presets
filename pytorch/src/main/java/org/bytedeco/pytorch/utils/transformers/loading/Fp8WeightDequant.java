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
package org.bytedeco.pytorch.utils.transformers.loading;

import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.global.torch.ScalarType;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

/**
 * Dequantize HuggingFace FP8 (E4M3) checkpoints that store:
 * <ul>
 *   <li>{@code *.weight} as {@code F8_E4M3} / {@code Float8_e4m3fn}</li>
 *   <li>{@code *.weight_scale_inv} as float32 block scales (often 128×128 blocks)</li>
 * </ul>
 *
 * <p>Used by Qwen3-VL-*-FP8 and similar compressed-tensors style dumps. Vision
 * towers in those dumps are usually BF16 and left untouched.
 */
public final class Fp8WeightDequant {

    private Fp8WeightDequant() {}

    /**
     * Replace FP8 weight tensors with BF16/float dequantized values and drop scale keys.
     * Non-FP8 tensors pass through unchanged.
     */
    public static Map<String, Tensor> dequantizeInPlace(Map<String, Tensor> weights) {
        if (weights == null || weights.isEmpty()) return weights;
        List<String> scaleKeys = new ArrayList<>();
        List<String> fp8Weights = new ArrayList<>();
        for (Map.Entry<String, Tensor> e : weights.entrySet()) {
            String k = e.getKey();
            if (k.endsWith("weight_scale_inv") || k.endsWith("weight_scale")
                    || k.endsWith("input_scale") || k.endsWith("input_scale_inv")) {
                scaleKeys.add(k);
                continue;
            }
            Tensor t = e.getValue();
            if (t != null && t.defined() && isFloat8(t)) {
                fp8Weights.add(k);
            }
        }
        if (fp8Weights.isEmpty()) {
            return weights;
        }
        System.out.println("[Fp8WeightDequant] dequantizing " + fp8Weights.size()
                + " FP8 weights (" + scaleKeys.size() + " scale tensors)");
        Map<String, Tensor> out = new LinkedHashMap<>(weights.size());
        int ok = 0, fail = 0;
        for (Map.Entry<String, Tensor> e : weights.entrySet()) {
            String k = e.getKey();
            if (scaleKeys.contains(k)) {
                continue; // drop scales after use
            }
            Tensor t = e.getValue();
            if (t == null || !t.defined() || !isFloat8(t)) {
                out.put(k, t);
                continue;
            }
            try {
                Tensor scale = findScale(weights, k);
                Tensor deq = dequantOne(t, scale);
                out.put(k, deq);
                ok++;
            } catch (Throwable ex) {
                System.out.println("[Fp8WeightDequant] FAIL " + k + ": " + ex.getMessage());
                // keep as float cast without scale as last resort
                try {
                    out.put(k, t.to(ScalarType.BFloat16).contiguous());
                } catch (Throwable ex2) {
                    out.put(k, t.to(ScalarType.Float).contiguous());
                }
                fail++;
            }
        }
        System.out.println("[Fp8WeightDequant] done ok=" + ok + " fail=" + fail
                + " out_tensors=" + out.size());
        return out;
    }

    private static boolean isFloat8(Tensor t) {
        try {
            ScalarType st = t.scalar_type().intern();
            return st == ScalarType.Float8_e4m3fn || st == ScalarType.Float8_e5m2;
        } catch (Throwable e) {
            return false;
        }
    }

    private static Tensor findScale(Map<String, Tensor> weights, String weightKey) {
        // foo.weight → foo.weight_scale_inv (compressed-tensors / Qwen FP8)
        String base = weightKey;
        if (base.endsWith(".weight")) {
            base = base.substring(0, base.length() - ".weight".length());
        }
        String[] candidates = {
                base + ".weight_scale_inv",
                weightKey + "_scale_inv",
                base + ".weight_scale",
                weightKey + "_scale",
                base + ".scale_inv",
                base + ".scale",
        };
        for (String c : candidates) {
            Tensor s = weights.get(c);
            if (s != null && s.defined()) return s;
        }
        // loose endsWith search
        String needle = base.endsWith(".") ? base : base + ".";
        for (Map.Entry<String, Tensor> e : weights.entrySet()) {
            String k = e.getKey();
            if (k.startsWith(needle) && k.toLowerCase(Locale.ROOT).contains("scale")) {
                return e.getValue();
            }
        }
        return null;
    }

    /**
     * Dequant: cast FP8 → float, multiply by block-wise scale_inv.
     * Scale is often [ceil(O/128), ceil(I/128)] for weight [O,I].
     */
    private static Tensor dequantOne(Tensor fp8, Tensor scaleInv) {
        Tensor w = fp8.to(ScalarType.Float).contiguous();
        if (scaleInv == null || !scaleInv.defined()) {
            return w.to(ScalarType.BFloat16).contiguous();
        }
        Tensor s = scaleInv.to(ScalarType.Float).contiguous();
        if (s.dim() == 0 || s.numel() == 1) {
            // per-tensor
            float inv = s.item_float();
            return w.mul(new Scalar(inv)).to(ScalarType.BFloat16).contiguous();
        }
        if (w.dim() == 2 && s.dim() == 2) {
            long outF = w.size(0), inF = w.size(1);
            long nBo = s.size(0), nBi = s.size(1);
            // infer block size
            long blockO = Math.max(1, (outF + nBo - 1) / nBo);
            long blockI = Math.max(1, (inF + nBi - 1) / nBi);
            // Expand scale to full weight shape via repeat_interleave-like indexing
            Tensor expanded = expandBlockScale(s, outF, inF, blockO, blockI);
            return w.mul(expanded).to(ScalarType.BFloat16).contiguous();
        }
        // fallback: broadcast if shapes align, else ignore scale
        try {
            return w.mul(s).to(ScalarType.BFloat16).contiguous();
        } catch (Throwable t) {
            System.out.println("[Fp8WeightDequant] scale broadcast failed shape w="
                    + shapeStr(w) + " s=" + shapeStr(s) + " — cast only");
            return w.to(ScalarType.BFloat16).contiguous();
        }
    }

    /**
     * Expand block scale [nBo, nBi] to [outF, inF] by repeating each scale over its block.
     */
    private static Tensor expandBlockScale(Tensor scale, long outF, long inF, long blockO, long blockI) {
        // scale[i,j] applies to w[i*blockO:(i+1)*blockO, j*blockI:(j+1)*blockI]
        // Use repeat_interleave if available; else manual via index
        try {
            // unsqueeze and expand: [nBo,1,nBi,1] → [nBo,blockO,nBi,blockI] → reshape
            Tensor s = scale.unsqueeze(1).unsqueeze(3); // [nBo,1,nBi,1]
            Tensor exp = s.expand(scale.size(0), blockO, scale.size(1), blockI).contiguous();
            Tensor flat = exp.reshape(scale.size(0) * blockO, scale.size(1) * blockI);
            // crop to exact outF x inF
            if (flat.size(0) != outF || flat.size(1) != inF) {
                flat = flat.slice(0, new org.bytedeco.pytorch.LongOptional(0),
                                new org.bytedeco.pytorch.LongOptional(outF), 1)
                        .slice(1, new org.bytedeco.pytorch.LongOptional(0),
                                new org.bytedeco.pytorch.LongOptional(inF), 1)
                        .contiguous();
            }
            return flat;
        } catch (Throwable t) {
            // slower path: build float array
            float[] sc = toFloatArray(scale);
            long nBo = scale.size(0), nBi = scale.size(1);
            float[] full = new float[(int) (outF * inF)];
            for (long o = 0; o < outF; o++) {
                long bo = Math.min(o / blockO, nBo - 1);
                for (long i = 0; i < inF; i++) {
                    long bi = Math.min(i / blockI, nBi - 1);
                    full[(int) (o * inF + i)] = sc[(int) (bo * nBi + bi)];
                }
            }
            return org.bytedeco.pytorch.global.torch.tensor(full).reshape(outF, inF);
        }
    }

    private static float[] toFloatArray(Tensor t) {
        Tensor c = t.to(ScalarType.Float).contiguous().cpu().reshape(-1);
        long n = c.numel();
        float[] out = new float[(int) n];
        // item-by-item is slow but safe without FloatIndexer dependency here
        for (int i = 0; i < n; i++) {
            out[i] = c.get(i).item_float();
        }
        return out;
    }

    private static String shapeStr(Tensor t) {
        StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < t.dim(); i++) {
            if (i > 0) sb.append(',');
            sb.append(t.size(i));
        }
        return sb.append(']').toString();
    }
}
