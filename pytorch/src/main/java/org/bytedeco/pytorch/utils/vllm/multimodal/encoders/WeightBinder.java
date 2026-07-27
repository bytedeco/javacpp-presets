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

import org.bytedeco.pytorch.StringTensorDict;
import org.bytedeco.pytorch.StringTensorDictItem;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.data.safetensors.SafeTensors;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.utils.transformers.loading.WeightLoader;
import org.bytedeco.pytorch.utils.transformers.mapping.WeightMap;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;

/**
 * Load HF safetensors / pytorch_model.bin into a {@link Module} with flexible key matching.
 *
 * <p>Handles:
 * <ul>
 *   <li>dot ↔ slash conversion for layer indices ({@code layers.0} ↔ {@code layers/0})</li>
 *   <li>optional prefix strip ({@code model.}, {@code vision_model.}, …)</li>
 *   <li>strict=false for partial encoder loads</li>
 * </ul>
 */
public final class WeightBinder {

    private WeightBinder() {}

    public static final class Report {
        public final int matched;
        public final int missing;
        public final int unexpected;
        public final int shapeMismatch;
        public final List<String> missingKeys;
        public final List<String> shapeKeys;

        public Report(int matched, int missing, int unexpected, int shapeMismatch,
                      List<String> missingKeys, List<String> shapeKeys) {
            this.matched = matched;
            this.missing = missing;
            this.unexpected = unexpected;
            this.shapeMismatch = shapeMismatch;
            this.missingKeys = missingKeys == null ? List.of() : List.copyOf(missingKeys);
            this.shapeKeys = shapeKeys == null ? List.of() : List.copyOf(shapeKeys);
        }

        public boolean ok() {
            return matched > 0 && shapeMismatch == 0;
        }

        @Override
        public String toString() {
            return String.format(Locale.ROOT,
                    "WeightBinder.Report{matched=%d missing=%d unexpected=%d shapeMismatch=%d}",
                    matched, missing, unexpected, shapeMismatch);
        }
    }

    /** Load all {@code *.safetensors} under dir (or single file). */
    public static Map<String, Tensor> loadSafetensors(Path dirOrFile) throws IOException {
        Objects.requireNonNull(dirOrFile, "dirOrFile");
        if (Files.isRegularFile(dirOrFile) && dirOrFile.getFileName().toString().endsWith(".safetensors")) {
            return SafeTensors.loadAsTensors(dirOrFile.toFile(), true);
        }
        if (!Files.isDirectory(dirOrFile)) {
            throw new IOException("Not a directory or safetensors file: " + dirOrFile);
        }
        Path single = dirOrFile.resolve("model.safetensors");
        if (Files.isRegularFile(single)) {
            return SafeTensors.loadAsTensors(single.toFile(), true);
        }
        Map<String, Tensor> all = new LinkedHashMap<>();
        try (var stream = Files.newDirectoryStream(dirOrFile, "*.safetensors")) {
            for (Path p : stream) {
                if (!Files.isRegularFile(p)) continue;
                String name = p.getFileName().toString();
                if (name.contains(".partial")) continue;
                Map<String, Tensor> part = SafeTensors.loadAsTensors(p.toFile(), true);
                all.putAll(part);
            }
        }
        if (all.isEmpty()) {
            throw new IOException("No safetensors weights in " + dirOrFile);
        }
        return all;
    }

    /**
     * Bind weights into module. Keys are matched after:
     * <ol>
     *   <li>optional strip of each prefix in {@code stripPrefixes}</li>
     *   <li>{@code layers.0} → {@code layers/0} conversion</li>
     *   <li>full-dot → slash fallback</li>
     *   <li>suffix match</li>
     * </ol>
     */
    public static Report bind(Module module, Map<String, Tensor> weights,
                              List<String> stripPrefixes, boolean strict) {
        Objects.requireNonNull(module, "module");
        Objects.requireNonNull(weights, "weights");
        Map<String, Tensor> params = collectParams(module);
        Map<String, Tensor> remapped = expandAliases(weights, stripPrefixes);

        int matched = 0, rebound = 0, copied = 0;
        List<String> missing = new ArrayList<>();
        List<String> shapeMismatch = new ArrayList<>();
        List<String> used = new ArrayList<>();

        for (Map.Entry<String, Tensor> pe : params.entrySet()) {
            String key = pe.getKey();
            Tensor dest = pe.getValue();
            if (dest == null || !dest.defined()) {
                missing.add(key);
                continue;
            }
            Tensor src = remapped.get(key);
            if (src == null) {
                // try alternate forms
                src = remapped.get(key.replace('/', '.'));
            }
            if (src == null) {
                for (Map.Entry<String, Tensor> e : remapped.entrySet()) {
                    String k = e.getKey();
                    if (k.equals(key) || k.endsWith("." + key) || k.endsWith("/" + key)
                            || key.endsWith("." + k) || key.endsWith("/" + k)
                            || k.replace('.', '/').equals(key)
                            || k.replace('/', '.').equals(key)) {
                        src = e.getValue();
                        break;
                    }
                }
            }
            if (src == null || !src.defined()) {
                missing.add(key);
                continue;
            }
            if (!shapesEqual(src, dest)) {
                // try transpose for Linear-like 2D
                if (src.dim() == 2 && dest.dim() == 2
                        && src.size(0) == dest.size(1) && src.size(1) == dest.size(0)) {
                    src = src.transpose(0, 1).contiguous();
                } else {
                    shapeMismatch.add(key + " src=" + shapeStr(src) + " dest=" + shapeStr(dest));
                    continue;
                }
            }
            try {
                dest.requires_grad_(false);
            } catch (Throwable ignored) {}
            try {
                dest.copy_(src);
                copied++;
                matched++;
                used.add(key);
            } catch (Throwable t) {
                try {
                    dest.set_(src);
                    rebound++;
                    matched++;
                    used.add(key);
                } catch (Throwable t2) {
                    shapeMismatch.add(key + " bind-fail:" + t2.getMessage());
                }
            }
        }

        int unexpected = 0;
        for (String k : remapped.keySet()) {
            boolean usedK = false;
            for (String u : used) {
                if (k.equals(u) || k.endsWith(u) || u.endsWith(k)) {
                    usedK = true;
                    break;
                }
            }
            if (!usedK) unexpected++;
        }

        Report report = new Report(matched, missing.size(), unexpected, shapeMismatch.size(),
                missing.size() > 8 ? missing.subList(0, 8) : missing,
                shapeMismatch.size() > 8 ? shapeMismatch.subList(0, 8) : shapeMismatch);
        if (strict && (matched == 0 || !shapeMismatch.isEmpty())) {
            throw new IllegalStateException("Weight bind failed: " + report
                    + " missingSample=" + report.missingKeys
                    + " shapeSample=" + report.shapeKeys);
        }
        return report;
    }

    public static Report bindSafetensors(Module module, Path dir, List<String> stripPrefixes,
                                         boolean strict) throws IOException {
        return bind(module, loadSafetensors(dir), stripPrefixes, strict);
    }

    /** Also try WeightLoader path for modules that already match HF names. */
    public static WeightLoader.LoadReport bindViaLoader(Module module, Path dir,
                                                        boolean strict) throws IOException {
        return WeightLoader.loadAndBind(module, dir, WeightMap.identity(),
                WeightLoader.BindMode.COPY, strict, true);
    }

    private static Map<String, Tensor> expandAliases(Map<String, Tensor> weights,
                                                     List<String> stripPrefixes) {
        Map<String, Tensor> out = new LinkedHashMap<>();
        for (Map.Entry<String, Tensor> e : weights.entrySet()) {
            String k = e.getKey();
            Tensor t = e.getValue();
            putAlias(out, k, t);
            putAlias(out, dotBeforeDigitToSlash(k), t);
            putAlias(out, k.replace('.', '/'), t);
            if (k.contains("layer_scale1.lambda1")) {
                putAlias(out, k.replace("layer_scale1.lambda1", "ls1").replace('.', '/'), t);
            }
            if (k.contains("layer_scale2.lambda1")) {
                putAlias(out, k.replace("layer_scale2.lambda1", "ls2").replace('.', '/'), t);
            }
            if (stripPrefixes != null) {
                for (String p : stripPrefixes) {
                    if (p != null && !p.isEmpty() && k.startsWith(p)) {
                        String stripped = k.substring(p.length());
                        putAlias(out, stripped, t);
                        putAlias(out, dotBeforeDigitToSlash(stripped), t);
                        putAlias(out, stripped.replace('.', '/'), t);
                        // DINOv2 layer scale: encoder.layer.N.layer_scale1.lambda1
                        // → encoder/layer/N/ls1 (module leaf name)
                        if (stripped.contains("layer_scale1.lambda1")) {
                            putAlias(out, stripped.replace("layer_scale1.lambda1", "ls1")
                                    .replace('.', '/'), t);
                        }
                        if (stripped.contains("layer_scale2.lambda1")) {
                            putAlias(out, stripped.replace("layer_scale2.lambda1", "ls2")
                                    .replace('.', '/'), t);
                        }
                        if (k.contains("layer_scale1.lambda1")) {
                            putAlias(out, k.replace("layer_scale1.lambda1", "ls1")
                                    .replace('.', '/'), t);
                        }
                        if (k.contains("layer_scale2.lambda1")) {
                            putAlias(out, k.replace("layer_scale2.lambda1", "ls2")
                                    .replace('.', '/'), t);
                        }
                    }
                }
            }
        }
        return out;
    }

    /** {@code layers.0.xxx} → {@code layers/0.xxx} (HF → JavaCPP module keys). */
    static String dotBeforeDigitToSlash(String hfKey) {
        if (hfKey == null) return null;
        return hfKey.replaceAll("\\.(\\d)", "/$1");
    }

    private static void putAlias(Map<String, Tensor> m, String k, Tensor t) {
        if (k == null || k.isEmpty()) return;
        m.putIfAbsent(k, t);
    }

    private static Map<String, Tensor> collectParams(Module module) {
        Map<String, Tensor> out = new LinkedHashMap<>();
        StringTensorDict dict = module.named_parameters(/*recurse=*/true);
        if (dict == null || dict.isNull()) return out;
        long n = dict.size();
        for (long i = 0; i < n; i++) {
            StringTensorDictItem item = dict.get(i);
            if (item == null || item.isNull()) continue;
            String key = item.key() != null ? item.key().getString() : null;
            Tensor val = item.value();
            if (key != null && val != null) out.put(key, val);
        }
        return out;
    }

    private static boolean shapesEqual(Tensor a, Tensor b) {
        if (a.dim() != b.dim()) return false;
        for (int i = 0; i < a.dim(); i++) {
            if (a.size(i) != b.size(i)) return false;
        }
        return true;
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
