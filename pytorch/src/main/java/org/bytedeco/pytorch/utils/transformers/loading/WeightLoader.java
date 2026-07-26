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

import org.bytedeco.pytorch.StringTensorDict;
import org.bytedeco.pytorch.StringTensorDictItem;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.data.safetensors.SafeTensors;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.utils.json.Json;
import org.bytedeco.pytorch.utils.transformers.mapping.WeightMap;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;

/**
 * Bind HuggingFace safetensors weights into a live {@link Module}.
 *
 * <p>{@link BindMode#ZERO_COPY} rebinds each parameter via {@link Tensor#set_(Tensor)}
 * onto mmap/{@code from_blob} tensors (no data copy). {@link BindMode#COPY} uses
 * {@link Tensor#copy_} for training / dtype conversion paths.
 *
 * <pre>{@code
 * LoadReport r = WeightLoader.bind(model, weights, WeightMap.identity(), BindMode.ZERO_COPY, true);
 * System.out.println(r);
 * }</pre>
 */
public final class WeightLoader {

    public enum BindMode {
        /** Rebind parameter storage to the source tensor (mmap-friendly). */
        ZERO_COPY,
        /** Copy values into existing parameter storage. */
        COPY
    }

    private WeightLoader() {}

    // ---- public API ---------------------------------------------------------

    public static LoadReport bind(Module module,
                                  Map<String, Tensor> weights,
                                  WeightMap map,
                                  BindMode mode,
                                  boolean strict) {
        Objects.requireNonNull(module, "module");
        Objects.requireNonNull(weights, "weights");
        if (map == null) map = WeightMap.identity();
        if (mode == null) mode = BindMode.ZERO_COPY;

        Map<String, Tensor> params = collectNamedParameters(module);
        Map<String, Tensor> remapped = map.apply(weights);

        List<String> matched = new ArrayList<>();
        List<String> missing = new ArrayList<>();
        List<String> unexpected = new ArrayList<>();
        List<String> shapeMismatch = new ArrayList<>();
        int rebound = 0;
        int copied = 0;

        Set<String> usedSources = new LinkedHashSet<>();

        for (Map.Entry<String, Tensor> pe : params.entrySet()) {
            String key = pe.getKey();
            Tensor dest = pe.getValue();
            if (dest == null || !dest.defined()) {
                missing.add(key);
                continue;
            }
            Tensor src = remapped.get(key);
            if (src == null) {
                // try common HF prefixes already stripped by WeightMap; also bare suffix match
                src = findLoose(remapped, key);
            }
            if (src == null || !src.defined()) {
                missing.add(key);
                continue;
            }
            if (!shapesEqual(src, dest)) {
                shapeMismatch.add(key + " src=" + shapeStr(src) + " dest=" + shapeStr(dest));
                continue;
            }
            if (mode == BindMode.ZERO_COPY) {
                // Disable grad BEFORE in-place set_ to avoid autograd leaf variable error
                try { dest.requires_grad_(false); } catch (Throwable ignored) {}
                dest.set_(src);
                rebound++;
            } else {
                dest.copy_(src);
                copied++;
            }
            matched.add(key);
            usedSources.add(key);
            // also mark original HF key if map rewrote
            for (Map.Entry<String, Tensor> we : remapped.entrySet()) {
                if (we.getValue() == src) usedSources.add(we.getKey());
            }
        }

        for (String k : remapped.keySet()) {
            if (!usedSources.contains(k) && !params.containsKey(k)) {
                // only report tops that look like tensors we care about
                unexpected.add(k);
            }
        }

        LoadReport report = new LoadReport(matched, missing, unexpected, shapeMismatch, rebound, copied, mode);
        if (strict) {
            if (!missing.isEmpty() || !shapeMismatch.isEmpty()) {
                throw new IllegalStateException("Weight load strict failure:\n" + report);
            }
        }
        return report;
    }

    /**
     * Load all {@code *.safetensors} (or sharded index) under {@code dir} and bind.
     */
    public static LoadReport loadAndBind(Module module, Path dir, WeightMap map,
                                         BindMode mode, boolean strict, boolean zeroCopyMmap)
            throws IOException {
        Map<String, Tensor> weights = SnapshotFiles.loadAllWeights(dir, zeroCopyMmap);
        return bind(module, weights, map, mode, strict);
    }

    public static LoadReport loadAndBind(Module module, Path dir) throws IOException {
        return loadAndBind(module, dir, WeightMap.identity(), BindMode.ZERO_COPY, true, true);
    }

    // ---- helpers ------------------------------------------------------------

    public static Map<String, Tensor> collectNamedParameters(Module module) {
        Map<String, Tensor> out = new LinkedHashMap<>();
        StringTensorDict dict = module.named_parameters(/*recurse=*/true);
        if (dict == null || dict.isNull()) return out;
        long n = dict.size();
        for (long i = 0; i < n; i++) {
            StringTensorDictItem item = dict.get(i);
            if (item == null || item.isNull()) continue;
            String key = item.key() != null ? item.key().getString() : null;
            Tensor val = item.value();
            if (key == null || val == null) continue;
            out.put(key, val);
        }
        return out;
    }

    private static Tensor findLoose(Map<String, Tensor> remapped, String key) {
        // strip leading "module." etc.
        if (remapped.containsKey(key)) return remapped.get(key);
        for (Map.Entry<String, Tensor> e : remapped.entrySet()) {
            String k = e.getKey();
            if (k.equals(key) || k.endsWith("." + key) || key.endsWith("." + k)) {
                return e.getValue();
            }
        }
        return null;
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

    // ---- report -------------------------------------------------------------

    public static final class LoadReport {
        public final List<String> matched;
        public final List<String> missing;
        public final List<String> unexpected;
        public final List<String> shapeMismatch;
        public final int rebound;
        public final int copied;
        public final BindMode mode;

        public LoadReport(List<String> matched, List<String> missing, List<String> unexpected,
                          List<String> shapeMismatch, int rebound, int copied, BindMode mode) {
            this.matched = Collections.unmodifiableList(new ArrayList<>(matched));
            this.missing = Collections.unmodifiableList(new ArrayList<>(missing));
            this.unexpected = Collections.unmodifiableList(new ArrayList<>(unexpected));
            this.shapeMismatch = Collections.unmodifiableList(new ArrayList<>(shapeMismatch));
            this.rebound = rebound;
            this.copied = copied;
            this.mode = mode;
        }

        public int matchedCount() { return matched.size(); }
        public boolean ok() { return missing.isEmpty() && shapeMismatch.isEmpty(); }

        @Override
        public String toString() {
            return "LoadReport{mode=" + mode
                    + ", matched=" + matched.size()
                    + ", missing=" + missing.size()
                    + ", unexpected=" + unexpected.size()
                    + ", shapeMismatch=" + shapeMismatch.size()
                    + ", rebound=" + rebound
                    + ", copied=" + copied
                    + (missing.isEmpty() ? "" : ", missingKeys=" + preview(missing))
                    + (shapeMismatch.isEmpty() ? "" : ", shapeMismatch=" + preview(shapeMismatch))
                    + "}";
        }

        private static String preview(List<String> xs) {
            if (xs.size() <= 8) return xs.toString();
            return xs.subList(0, 8) + "…(+" + (xs.size() - 8) + ")";
        }
    }
}
