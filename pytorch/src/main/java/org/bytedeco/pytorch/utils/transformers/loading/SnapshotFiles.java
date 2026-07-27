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

import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.StringTensorDict;
import org.bytedeco.pytorch.StringTensorDictItem;
import org.bytedeco.pytorch.data.safetensors.SafeTensors;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.utils.json.Json;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.DirectoryStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;

/**
 * Resolve weight files inside a HuggingFace model snapshot directory.
 *
 * <p>Supports:
 * <ul>
 *   <li>single {@code model.safetensors} / dtype-tagged alternates</li>
 *   <li>sharded {@code model.safetensors.index.json} + listed shards</li>
 *   <li>numbered shards {@code model-00001-of-000NN.safetensors} without index</li>
 *   <li>files larger than ~2 GiB via per-tensor mmap in {@link SafeTensors}</li>
 *   <li>any remaining {@code *.safetensors} fallback scan (skips {@code *.partial})</li>
 * </ul>
 */
public final class SnapshotFiles {

    private SnapshotFiles() {}

    public static Path configJson(Path dir) {
        return dir.resolve("config.json");
    }

    public static Path generationConfigJson(Path dir) {
        return dir.resolve("generation_config.json");
    }

    public static Path tokenizerJson(Path dir) {
        return dir.resolve("tokenizer.json");
    }

    public static Path tokenizerConfigJson(Path dir) {
        return dir.resolve("tokenizer_config.json");
    }

    public static Path specialTokensMapJson(Path dir) {
        return dir.resolve("special_tokens_map.json");
    }

    public static Path vocabJson(Path dir) {
        return dir.resolve("vocab.json");
    }

    public static Path mergesTxt(Path dir) {
        return dir.resolve("merges.txt");
    }

    public static Path tokenizerModel(Path dir) {
        Path spm = dir.resolve("tokenizer.model");
        if (Files.isRegularFile(spm)) return spm;
        return dir.resolve("spiece.model");
    }

    /** True when the directory has enough artifacts to load a real tokenizer. */
    public static boolean hasTokenizerArtifacts(Path dir) {
        if (dir == null || !Files.isDirectory(dir)) return false;
        if (Files.isRegularFile(tokenizerJson(dir))) return true;
        return Files.isRegularFile(vocabJson(dir)) && Files.isRegularFile(mergesTxt(dir));
    }

    /**
     * Ordered list of safetensors files to open for this snapshot.
     *
     * <p>Resolution order:
     * <ol>
     *   <li>{@code model.safetensors.index.json} weight_map (HF multi-shard)</li>
     *   <li>legacy {@code pytorch_model.bin.index.json} when it points at {@code *.safetensors}</li>
     *   <li>single {@code model.safetensors}</li>
     *   <li>HF-style shards {@code model-00001-of-000NN.safetensors} (sorted), even without index</li>
     *   <li>other named single-file alternates</li>
     *   <li>any remaining {@code *.safetensors} (ignores {@code *.partial})</li>
     * </ol>
     */
    public static List<Path> weightFiles(Path dir) throws IOException {
        Objects.requireNonNull(dir, "dir");
        if (!Files.isDirectory(dir)) {
            throw new IOException("Not a directory: " + dir);
        }

        // 1) Official HF index (preferred for multi-shard)
        Path index = dir.resolve("model.safetensors.index.json");
        if (Files.isRegularFile(index)) {
            return shardsFromIndex(dir, index);
        }
        // some older dumps use pytorch_model.bin.index.json but store safetensors shards
        Path legacyIndex = dir.resolve("pytorch_model.bin.index.json");
        if (Files.isRegularFile(legacyIndex)) {
            List<Path> fromLegacy = shardsFromIndex(dir, legacyIndex);
            if (!fromLegacy.isEmpty() && fromLegacy.stream().allMatch(p ->
                    p.getFileName().toString().endsWith(".safetensors"))) {
                return fromLegacy;
            }
        }

        // 2) Single-file checkpoint
        Path single = dir.resolve("model.safetensors");
        if (Files.isRegularFile(single) && isUsableWeightFile(single)) {
            return List.of(single);
        }

        // 3) HF multi-shard naming without index: model-00001-of-00010.safetensors
        List<Path> numbered = discoverNumberedShards(dir);
        if (!numbered.isEmpty()) {
            return numbered;
        }

        // 4) common alternates
        for (String name : new String[]{
                "model.fp16.safetensors", "model.bf16.safetensors",
                "pytorch_model.safetensors", "adapter_model.safetensors"
        }) {
            Path p = dir.resolve(name);
            if (Files.isRegularFile(p) && isUsableWeightFile(p)) return List.of(p);
        }

        // 5) fallback scan — skip incomplete downloads / partials
        List<Path> found = new ArrayList<>();
        try (DirectoryStream<Path> ds = Files.newDirectoryStream(dir, "*.safetensors")) {
            for (Path p : ds) {
                if (Files.isRegularFile(p) && isUsableWeightFile(p)) found.add(p);
            }
        }
        found.sort(Path::compareTo);
        return found;
    }

    /** Skip incomplete downloads and empty stubs. */
    static boolean isUsableWeightFile(Path p) {
        if (p == null) return false;
        String name = p.getFileName().toString();
        if (name.endsWith(".partial") || name.contains(".partial.") || name.endsWith(".chunk")) {
            return false;
        }
        try {
            return Files.size(p) > 1024L; // ignore empty/tiny stubs
        } catch (IOException e) {
            return false;
        }
    }

    /**
     * Discover {@code model-0000N-of-0000M.safetensors} (or {@code pytorch_model-...}) shards
     * when the index json is missing. Sorted lexicographically so 00001..0000N stay ordered.
     */
    static List<Path> discoverNumberedShards(Path dir) throws IOException {
        List<Path> found = new ArrayList<>();
        try (DirectoryStream<Path> ds = Files.newDirectoryStream(dir, "*.safetensors")) {
            for (Path p : ds) {
                if (!Files.isRegularFile(p) || !isUsableWeightFile(p)) continue;
                String n = p.getFileName().toString();
                // model-00001-of-00010.safetensors  OR  pytorch_model-00001-of-00010.safetensors
                if (n.matches("(?i)(model|pytorch_model)-\\d{5}-of-\\d{5}\\.safetensors")) {
                    found.add(p);
                }
            }
        }
        found.sort(Path::compareTo);
        return found;
    }

    @SuppressWarnings("unchecked")
    private static List<Path> shardsFromIndex(Path dir, Path index) throws IOException {
        String raw = Files.readString(index, StandardCharsets.UTF_8);
        Map<String, Object> root = Json.decodeObject(raw);
        Object wm = root.get("weight_map");
        Set<String> files = new LinkedHashSet<>();
        if (wm instanceof Map<?, ?> map) {
            for (Object v : map.values()) {
                if (v != null) files.add(String.valueOf(v));
            }
        }
        List<Path> out = new ArrayList<>();
        for (String f : files) {
            Path p = dir.resolve(f);
            if (!Files.isRegularFile(p)) {
                throw new IOException("Shard listed in index but missing: " + p);
            }
            if (!isUsableWeightFile(p)) {
                throw new IOException("Shard listed in index is incomplete/unusable: " + p);
            }
            out.add(p);
        }
        if (out.isEmpty()) {
            throw new IOException("Empty weight_map in " + index);
        }
        // Stable order: prefer numbered shard order, else insertion order from weight_map
        out.sort(Path::compareTo);
        return out;
    }

    /**
     * Load and merge all weight tensors from the snapshot (mmap when {@code zeroCopy}).
     *
     * <p>Multi-shard: each {@code model-XXXXX-of-YYYYY.safetensors} (or index-listed
     * shard) is opened via {@link SafeTensors#loadAsTensors}; keys are merged.
     * Files larger than ~2 GiB use per-tensor mmap inside SafeTensors.
     */
    public static Map<String, Tensor> loadAllWeights(Path dir, boolean zeroCopy) throws IOException {
        List<Path> files = weightFiles(dir);
        if (files.isEmpty()) {
            throw new IOException("No .safetensors weight files in " + dir);
        }
        Map<String, Tensor> all = new LinkedHashMap<>();
        int shardIdx = 0;
        for (Path f : files) {
            shardIdx++;
            long sz = Files.size(f);
            System.out.println("[SnapshotFiles] loading shard " + shardIdx + "/" + files.size()
                    + " " + f.getFileName() + " (" + String.format(java.util.Locale.ROOT, "%.2f", sz / 1e9)
                    + " GB) zeroCopy=" + zeroCopy);
            Map<String, Tensor> part = SafeTensors.loadAsTensors(f.toFile(), zeroCopy);
            for (Map.Entry<String, Tensor> e : part.entrySet()) {
                if (all.containsKey(e.getKey())) {
                    throw new IOException("Duplicate tensor key across shards: " + e.getKey()
                            + " (file=" + f.getFileName() + ")");
                }
                all.put(e.getKey(), e.getValue());
            }
        }
        System.out.println("[SnapshotFiles] merged " + all.size() + " tensors from "
                + files.size() + " shard(s)");
        return all;
    }

    /** True if the snapshot looks like a loadable HF model dir. */
    public static boolean isModelDir(Path dir) {
        if (dir == null || !Files.isDirectory(dir)) return false;
        if (Files.isRegularFile(dir.resolve("config.json"))) return true;
        try {
            return !weightFiles(dir).isEmpty();
        } catch (IOException e) {
            return false;
        }
    }

    /**
     * Convert all parameters in a module to the target dtype.
     * This must be called BEFORE loading weights to ensure storage sizes match.
     * Uses set_() to rebind storage in-place.
     */
    public static void toDtype(Module module, org.bytedeco.pytorch.global.torch.ScalarType scalarType) {
        if (module == null || scalarType == null) return;
        StringTensorDict dict = module.named_parameters(/*recurse=*/true);
        if (dict == null || dict.isNull()) return;
        long n = dict.size();
        for (long i = 0; i < n; i++) {
            StringTensorDictItem item = dict.get(i);
            if (item == null || item.isNull()) continue;
            Tensor val = item.value();
            if (val == null || !val.defined()) continue;
            try {
                // Convert dtype and rebind storage to match
                val.requires_grad_(false);
                Tensor converted = val.to(scalarType);
                val.set_(converted);
            } catch (Throwable ignored) {
                // Some tensors may not support dtype conversion
            }
        }
    }
}
