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
 *   <li>single {@code model.safetensors} / {@code model.fp32.safetensors}</li>
 *   <li>sharded {@code model.safetensors.index.json} + listed shards</li>
 *   <li>any {@code *.safetensors} fallback scan</li>
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
     */
    public static List<Path> weightFiles(Path dir) throws IOException {
        Objects.requireNonNull(dir, "dir");
        if (!Files.isDirectory(dir)) {
            throw new IOException("Not a directory: " + dir);
        }
        Path index = dir.resolve("model.safetensors.index.json");
        if (Files.isRegularFile(index)) {
            return shardsFromIndex(dir, index);
        }
        Path single = dir.resolve("model.safetensors");
        if (Files.isRegularFile(single)) {
            return List.of(single);
        }
        // common alternates
        for (String name : new String[]{
                "model.fp16.safetensors", "model.bf16.safetensors",
                "pytorch_model.safetensors", "adapter_model.safetensors"
        }) {
            Path p = dir.resolve(name);
            if (Files.isRegularFile(p)) return List.of(p);
        }
        List<Path> found = new ArrayList<>();
        try (DirectoryStream<Path> ds = Files.newDirectoryStream(dir, "*.safetensors")) {
            for (Path p : ds) {
                if (Files.isRegularFile(p)) found.add(p);
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
            out.add(p);
        }
        if (out.isEmpty()) {
            throw new IOException("Empty weight_map in " + index);
        }
        return out;
    }

    /**
     * Load and merge all weight tensors from the snapshot (mmap when {@code zeroCopy}).
     */
    public static Map<String, Tensor> loadAllWeights(Path dir, boolean zeroCopy) throws IOException {
        List<Path> files = weightFiles(dir);
        if (files.isEmpty()) {
            throw new IOException("No .safetensors weight files in " + dir);
        }
        Map<String, Tensor> all = new LinkedHashMap<>();
        for (Path f : files) {
            Map<String, Tensor> part = SafeTensors.loadAsTensors(f.toFile(), zeroCopy);
            for (Map.Entry<String, Tensor> e : part.entrySet()) {
                if (all.containsKey(e.getKey())) {
                    throw new IOException("Duplicate tensor key across shards: " + e.getKey());
                }
                all.put(e.getKey(), e.getValue());
            }
        }
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
