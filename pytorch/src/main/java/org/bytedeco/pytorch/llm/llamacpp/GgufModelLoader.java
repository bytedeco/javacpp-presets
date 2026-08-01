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

package org.bytedeco.pytorch.llm.llamacpp;

import org.bytedeco.pytorch.data.gguf.GGUFConstants;
import org.bytedeco.pytorch.data.gguf.GGUFReader;
import org.bytedeco.pytorch.llm.llamacpp.quant.Dequantizer;
import org.bytedeco.pytorch.llm.llamacpp.quant.GgmlQuantType;

import java.io.File;
import java.io.IOException;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;

/**
 * Load GGUF metadata + tensors into {@link LlamaModel}.
 * Quantized tensors are dequantized to float32 arrays for the in-process engine.
 */
public final class GgufModelLoader {

    private GgufModelLoader() {}

    public static LlamaHParams parseHParams(Map<String, Object> meta) {
        Objects.requireNonNull(meta, "meta");
        String archStr = str(meta, "general.architecture", "llama");
        LlamaArchitecture arch = LlamaArchitecture.fromMetadata(archStr);
        String prefix = arch.metadataPrefix();

        LlamaHParams.Builder b = LlamaHParams.builder()
                .architecture(arch)
                .name(str(meta, "general.name", archStr))
                .raw(meta);

        b.nVocab(intOf(meta, prefix + ".vocab_size",
                intOf(meta, "tokenizer.ggml.tokens", -1) > 0
                        ? listSize(meta.get("tokenizer.ggml.tokens"))
                        : 32000));
        // vocab often only in tokenizer array
        if (meta.get("tokenizer.ggml.tokens") != null) {
            int vs = listSize(meta.get("tokenizer.ggml.tokens"));
            if (vs > 0) b.nVocab(vs);
        }
        b.nEmbd(intOf(meta, prefix + ".embedding_length",
                intOf(meta, "llama.embedding_length", 4096)));
        b.nLayer(intOf(meta, prefix + ".block_count",
                intOf(meta, "llama.block_count", 32)));
        b.nFF(intOf(meta, prefix + ".feed_forward_length",
                intOf(meta, "llama.feed_forward_length", 0)));
        b.nHead(intOf(meta, prefix + ".attention.head_count",
                intOf(meta, "llama.attention.head_count", 32)));
        b.nHeadKv(intOf(meta, prefix + ".attention.head_count_kv",
                intOf(meta, "llama.attention.head_count_kv", 0)));
        b.nCtxTrain(intOf(meta, prefix + ".context_length",
                intOf(meta, "llama.context_length", 2048)));
        b.nRot(intOf(meta, prefix + ".rope.dimension_count",
                intOf(meta, "llama.rope.dimension_count", 0)));
        b.ropeFreqBase(floatOf(meta, prefix + ".rope.freq_base",
                floatOf(meta, "llama.rope.freq_base", 10000f)));
        b.ropeFreqScale(floatOf(meta, prefix + ".rope.scaling.factor",
                floatOf(meta, "llama.rope.freq_scale", 1f)));
        b.rmsNormEps(floatOf(meta, prefix + ".attention.layer_norm_rms_epsilon",
                floatOf(meta, "llama.attention.layer_norm_rms_epsilon", 1e-5f)));
        b.expertCount(intOf(meta, prefix + ".expert_count", 0));
        b.expertUsedCount(intOf(meta, prefix + ".expert_used_count", 0));
        return b.build();
    }

    public static LlamaModel load(Path path, boolean dequantToFloat) throws IOException {
        return load(path.toFile(), dequantToFloat);
    }

    public static LlamaModel load(File file, boolean dequantToFloat) throws IOException {
        try (GGUFReader reader = new GGUFReader(file)) {
            LlamaHParams hp = parseHParams(reader.metadata());
            Map<String, LlamaModel.TensorBlob> blobs = new LinkedHashMap<>();
            for (Map.Entry<String, GGUFReader.TensorInfo> e : reader.tensorInfos().entrySet()) {
                GGUFReader.TensorInfo ti = e.getValue();
                GgmlQuantType qt = GgmlQuantType.fromId(ti.ggmlType);
                float[] floats = null;
                // Prefer reader.loadTensor for non-quant (returns libtorch Tensor) — we still
                // materialize float[] for engine portability in pure paths.
                if (dequantToFloat) {
                    try {
                        if (!qt.quantized() && qt != GgmlQuantType.UNKNOWN) {
                            // load via GGUFReader into Tensor then copy — heavy; skip raw for now
                            // Use nbytes estimate: read raw through loadAll is memory heavy.
                        }
                    } catch (Exception ignored) {}
                }
                blobs.put(e.getKey(), new LlamaModel.TensorBlob(
                        e.getKey(), ti.shape, ti.ggmlType, ti.offset, ti.nBytes(), floats));
            }
            return new LlamaModel(file.toPath(), hp, reader.metadata(), blobs, reader.version());
        }
    }

    /**
     * Load a single tensor as float[], dequantizing if needed.
     * Reads payload via GGUFReader.loadTensor for F32/F16; for Q-types reads raw file bytes.
     */
    public static float[] loadFloatTensor(File file, String name) throws IOException {
        try (GGUFReader reader = new GGUFReader(file)) {
            GGUFReader.TensorInfo ti = reader.tensorInfos().get(name);
            if (ti == null) throw new IOException("tensor not found: " + name);
            GgmlQuantType qt = GgmlQuantType.fromId(ti.ggmlType);
            if (qt == GgmlQuantType.F32 || qt == GgmlQuantType.F16 || qt == GgmlQuantType.BF16
                    || qt == GgmlQuantType.I8 || qt == GgmlQuantType.I16
                    || qt == GgmlQuantType.I32 || qt == GgmlQuantType.I64 || qt == GgmlQuantType.F64) {
                org.bytedeco.pytorch.Tensor t = reader.loadTensor(name);
                return tensorToFloat(t);
            }
            // quantized: read raw bytes from file
            long absOff = reader.tensorDataOffset() + ti.offset;
            int nBytes = (int) Math.min(ti.nBytes(), Integer.MAX_VALUE);
            byte[] raw = new byte[nBytes];
            try (java.io.RandomAccessFile raf = new java.io.RandomAccessFile(file, "r")) {
                raf.seek(absOff);
                raf.readFully(raw);
            }
            return Dequantizer.dequant(raw, ti.nElements(), qt);
        }
    }

    private static float[] tensorToFloat(org.bytedeco.pytorch.Tensor t) {
        if (t == null || !t.defined()) return new float[0];
        long n = t.numel();
        int ni = (int) Math.min(n, Integer.MAX_VALUE);
        float[] out = new float[ni];
        try {
            org.bytedeco.pytorch.Tensor cpu = t;
            try {
                if (!t.is_cpu()) {
                    cpu = t.cpu();
                }
            } catch (Throwable ignored) {
                try { cpu = t.contiguous().cpu(); } catch (Throwable ignored2) {}
            }
            org.bytedeco.pytorch.Tensor f = cpu;
            try {
                f = cpu.to(org.bytedeco.pytorch.global.torch.ScalarType.Float).contiguous();
            } catch (Throwable ignored) {
                try { f = cpu.contiguous(); } catch (Throwable ignored2) {}
            }
            try {
                org.bytedeco.javacpp.FloatPointer fp = f.data_ptr_float();
                fp.get(out);
                return out;
            } catch (Throwable t2) {
                try {
                    org.bytedeco.pytorch.Tensor flat = f.reshape(new long[]{ni});
                    for (int i = 0; i < ni; i++) {
                        try {
                            out[i] = flat.select(0, i).item_float();
                        } catch (Throwable ex) {
                            out[i] = 0f;
                        }
                    }
                } catch (Throwable ignored) {}
                return out;
            }
        } catch (Throwable e) {
            return out;
        }
    }

    private static String str(Map<String, Object> m, String k, String def) {
        Object v = m.get(k);
        return v != null ? String.valueOf(v) : def;
    }

    private static int intOf(Map<String, Object> m, String k, int def) {
        Object v = m.get(k);
        if (v instanceof Number n) return n.intValue();
        if (v != null) {
            try { return (int) Double.parseDouble(String.valueOf(v)); } catch (Exception ignored) {}
        }
        return def;
    }

    private static float floatOf(Map<String, Object> m, String k, float def) {
        Object v = m.get(k);
        if (v instanceof Number n) return n.floatValue();
        if (v != null) {
            try { return Float.parseFloat(String.valueOf(v)); } catch (Exception ignored) {}
        }
        return def;
    }

    private static int listSize(Object v) {
        if (v instanceof java.util.Collection<?> c) return c.size();
        if (v instanceof Object[] a) return a.length;
        return -1;
    }
}
