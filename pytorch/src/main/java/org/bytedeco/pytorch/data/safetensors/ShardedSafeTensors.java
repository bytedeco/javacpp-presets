package org.bytedeco.pytorch.data.safetensors;

import org.bytedeco.pytorch.Device;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.utils.json.Json;

import java.io.File;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.DirectoryStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.Set;

/**
 * HuggingFace multi-shard safetensors loader — pure-Java counterpart of
 * {@code safetensors.torch.load_model} / HF {@code model.safetensors.index.json}.
 *
 * <p>Resolution order for a directory:
 * <ol>
 *   <li>{@code model.safetensors.index.json} {@code weight_map}</li>
 *   <li>legacy {@code pytorch_model.bin.index.json} pointing at {@code *.safetensors}</li>
 *   <li>single {@code model.safetensors}</li>
 *   <li>numbered shards {@code model-00001-of-000NN.safetensors}</li>
 *   <li>any remaining usable {@code *.safetensors}</li>
 * </ol>
 *
 * <p>Lives in {@code data.safetensors} (not only under {@code llm.transformers})
 * so {@link SafeTensors#loadFile} / {@link org.bytedeco.pytorch.data.serialize.ModelWeights}
 * can load enterprise LLM and recsys checkpoints without a transformers dependency edge.
 */
public final class ShardedSafeTensors {

    private ShardedSafeTensors() {}

    // ---- public entry points ------------------------------------------------

    /** Load + merge all shards under {@code dir} with default options. */
    public static Map<String, Tensor> loadDirectory(Path dir) throws IOException {
        return loadDirectory(dir, LoadOptions.defaults());
    }

    public static Map<String, Tensor> loadDirectory(File dir) throws IOException {
        return loadDirectory(dir.toPath(), LoadOptions.defaults());
    }

    public static Map<String, Tensor> loadDirectory(String dir) throws IOException {
        return loadDirectory(Path.of(dir), LoadOptions.defaults());
    }

    /**
     * Load + merge all shards. Honours {@link LoadOptions#zeroCopy},
     * {@link LoadOptions#mapLocation}, {@link LoadOptions#dtype}.
     * FP8 dequant is applied when {@link LoadOptions#dequantFp8} is true and the
     * optional dequant helper is on the classpath.
     */
    public static Map<String, Tensor> loadDirectory(Path dir, LoadOptions opts) throws IOException {
        Objects.requireNonNull(dir, "dir");
        if (opts == null) opts = LoadOptions.defaults();
        if (!Files.isDirectory(dir)) {
            throw new IOException("not a directory: " + dir);
        }
        List<Path> shards = resolveShards(dir);
        if (shards.isEmpty()) {
            throw new IOException("No loadable .safetensors shards in " + dir);
        }
        Map<String, Tensor> all = new LinkedHashMap<>();
        int i = 0;
        for (Path shard : shards) {
            i++;
            long sz = Files.size(shard);
            System.out.println("[ShardedSafeTensors] shard " + i + "/" + shards.size()
                    + " " + shard.getFileName()
                    + " (" + String.format(Locale.ROOT, "%.2f", sz / 1e9) + " GB)"
                    + " zeroCopy=" + opts.zeroCopy);
            Map<String, Tensor> part = SafeTensors.loadAsTensors(shard.toFile(), opts.zeroCopy);
            for (Map.Entry<String, Tensor> e : part.entrySet()) {
                if (all.containsKey(e.getKey())) {
                    throw new IOException("Duplicate tensor key across shards: " + e.getKey()
                            + " (file=" + shard.getFileName() + ")");
                }
                all.put(e.getKey(), e.getValue());
            }
        }
        System.out.println("[ShardedSafeTensors] merged " + all.size()
                + " tensors from " + shards.size() + " shard(s)");
        if (opts.dequantFp8) {
            all = tryDequantFp8(all);
        }
        return SafeTensors.applyMapLocation(all, opts);
    }

    /**
     * Load from an explicit index JSON ({@code model.safetensors.index.json}).
     * {@code indexParent} is the directory that contains the shard files listed
     * in {@code weight_map}.
     */
    public static Map<String, Tensor> loadIndex(Path indexFile, LoadOptions opts) throws IOException {
        Objects.requireNonNull(indexFile, "indexFile");
        if (opts == null) opts = LoadOptions.defaults();
        if (!Files.isRegularFile(indexFile)) {
            throw new IOException("index not a file: " + indexFile);
        }
        Path parent = indexFile.getParent() != null ? indexFile.getParent() : Path.of(".");
        List<Path> shards = shardsFromIndex(parent, indexFile);
        Map<String, Tensor> all = new LinkedHashMap<>();
        for (Path shard : shards) {
            Map<String, Tensor> part = SafeTensors.loadAsTensors(shard.toFile(), opts.zeroCopy);
            for (Map.Entry<String, Tensor> e : part.entrySet()) {
                if (all.containsKey(e.getKey())) {
                    throw new IOException("Duplicate tensor key across shards: " + e.getKey());
                }
                all.put(e.getKey(), e.getValue());
            }
        }
        if (opts.dequantFp8) all = tryDequantFp8(all);
        return SafeTensors.applyMapLocation(all, opts);
    }

    public static Map<String, Tensor> loadIndex(File indexFile) throws IOException {
        return loadIndex(indexFile.toPath(), LoadOptions.defaults());
    }

    /**
     * Read the {@code weight_map} (tensor name → shard file name) without loading
     * any tensor payloads. Useful for selective / layer-wise loads with {@link SafeOpen}.
     */
    @SuppressWarnings("unchecked")
    public static Map<String, String> readWeightMap(Path indexFile) throws IOException {
        String raw = Files.readString(indexFile, StandardCharsets.UTF_8);
        Map<String, Object> root = Json.decodeObject(raw);
        Object wm = root.get("weight_map");
        Map<String, String> out = new LinkedHashMap<>();
        if (wm instanceof Map<?, ?> map) {
            for (Map.Entry<?, ?> e : map.entrySet()) {
                if (e.getKey() != null && e.getValue() != null) {
                    out.put(String.valueOf(e.getKey()), String.valueOf(e.getValue()));
                }
            }
        }
        return out;
    }

    /** Metadata block from index.json ({@code metadata: {total_size, ...}}). */
    @SuppressWarnings("unchecked")
    public static Map<String, Object> readIndexMetadata(Path indexFile) throws IOException {
        String raw = Files.readString(indexFile, StandardCharsets.UTF_8);
        Map<String, Object> root = Json.decodeObject(raw);
        Object meta = root.get("metadata");
        if (meta instanceof Map<?, ?> m) {
            Map<String, Object> out = new LinkedHashMap<>();
            for (Map.Entry<?, ?> e : m.entrySet()) {
                if (e.getKey() != null) out.put(String.valueOf(e.getKey()), e.getValue());
            }
            return out;
        }
        return Collections.emptyMap();
    }

    /**
     * Ordered list of shard files for a model directory (same resolution as load).
     */
    public static List<Path> resolveShards(Path dir) throws IOException {
        Objects.requireNonNull(dir, "dir");
        if (!Files.isDirectory(dir)) {
            throw new IOException("not a directory: " + dir);
        }

        Path index = dir.resolve("model.safetensors.index.json");
        if (Files.isRegularFile(index)) {
            return shardsFromIndex(dir, index);
        }
        Path legacyIndex = dir.resolve("pytorch_model.bin.index.json");
        if (Files.isRegularFile(legacyIndex)) {
            List<Path> fromLegacy = shardsFromIndex(dir, legacyIndex);
            if (!fromLegacy.isEmpty() && fromLegacy.stream().allMatch(p ->
                    p.getFileName().toString().endsWith(".safetensors"))) {
                return fromLegacy;
            }
        }

        Path single = dir.resolve("model.safetensors");
        if (Files.isRegularFile(single) && isUsable(single)) {
            return List.of(single);
        }

        List<Path> numbered = discoverNumberedShards(dir);
        if (!numbered.isEmpty()) return numbered;

        for (String name : new String[]{
                "model.fp16.safetensors", "model.bf16.safetensors",
                "pytorch_model.safetensors", "adapter_model.safetensors"
        }) {
            Path p = dir.resolve(name);
            if (Files.isRegularFile(p) && isUsable(p)) return List.of(p);
        }

        List<Path> found = new ArrayList<>();
        try (DirectoryStream<Path> ds = Files.newDirectoryStream(dir, "*.safetensors")) {
            for (Path p : ds) {
                if (Files.isRegularFile(p) && isUsable(p)) found.add(p);
            }
        }
        found.sort(Path::compareTo);
        return found;
    }

    /**
     * Selectively load tensors whose names match any of {@code keyPrefixes}
     * (or exact keys). Opens only the shards that contain matching keys when an
     * index is present; otherwise falls back to full-directory load + filter.
     */
    public static Map<String, Tensor> loadKeys(Path dir, Iterable<String> keysOrPrefixes,
                                               LoadOptions opts) throws IOException {
        if (opts == null) opts = LoadOptions.defaults();
        Path index = dir.resolve("model.safetensors.index.json");
        if (!Files.isRegularFile(index)) {
            index = dir.resolve("pytorch_model.bin.index.json");
        }
        if (!Files.isRegularFile(index)) {
            // no index — load all then filter
            Map<String, Tensor> all = loadDirectory(dir, opts);
            return filterKeys(all, keysOrPrefixes);
        }
        Map<String, String> weightMap = readWeightMap(index);
        Set<String> wantedKeys = new LinkedHashSet<>();
        Set<String> neededShards = new LinkedHashSet<>();
        List<String> patterns = new ArrayList<>();
        for (String k : keysOrPrefixes) patterns.add(k);
        for (Map.Entry<String, String> e : weightMap.entrySet()) {
            if (matchesAny(e.getKey(), patterns)) {
                wantedKeys.add(e.getKey());
                neededShards.add(e.getValue());
            }
        }
        Map<String, Tensor> out = new LinkedHashMap<>();
        for (String shardName : neededShards) {
            Path shard = dir.resolve(shardName);
            if (!Files.isRegularFile(shard)) {
                throw new IOException("Shard missing: " + shard);
            }
            // Prefer SafeOpen slice when possible to avoid materialising whole shard
            try (SafeOpen so = SafeOpen.open(shard.toFile(), opts.zeroCopy)) {
                List<String> inThis = new ArrayList<>();
                for (String k : wantedKeys) {
                    if (shardName.equals(weightMap.get(k)) && so.contains(k)) inThis.add(k);
                }
                Map<String, Tensor> part = so.loadSlice(inThis, /*strict=*/false);
                out.putAll(part);
            }
        }
        if (opts.dequantFp8) out = tryDequantFp8(out);
        return SafeTensors.applyMapLocation(out, opts);
    }

    // ---- internals ----------------------------------------------------------

    static boolean isUsable(Path p) {
        if (p == null) return false;
        String name = p.getFileName().toString();
        if (name.endsWith(".partial") || name.contains(".partial.") || name.endsWith(".chunk")) {
            return false;
        }
        try {
            return Files.size(p) > 1024L;
        } catch (IOException e) {
            return false;
        }
    }

    static List<Path> discoverNumberedShards(Path dir) throws IOException {
        List<Path> found = new ArrayList<>();
        try (DirectoryStream<Path> ds = Files.newDirectoryStream(dir, "*.safetensors")) {
            for (Path p : ds) {
                if (!Files.isRegularFile(p) || !isUsable(p)) continue;
                String n = p.getFileName().toString();
                if (n.matches("(?i)(model|pytorch_model)-\\d{5}-of-\\d{5}\\.safetensors")) {
                    found.add(p);
                }
            }
        }
        found.sort(Path::compareTo);
        return found;
    }

    @SuppressWarnings("unchecked")
    static List<Path> shardsFromIndex(Path dir, Path index) throws IOException {
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
            if (!isUsable(p)) {
                throw new IOException("Shard listed in index is incomplete/unusable: " + p);
            }
            out.add(p);
        }
        if (out.isEmpty()) {
            throw new IOException("Empty weight_map in " + index);
        }
        out.sort(Path::compareTo);
        return out;
    }

    private static boolean matchesAny(String key, List<String> patterns) {
        for (String p : patterns) {
            if (p == null) continue;
            if (key.equals(p) || key.startsWith(p) || key.endsWith(p) || key.contains(p)) {
                return true;
            }
        }
        return false;
    }

    private static Map<String, Tensor> filterKeys(Map<String, Tensor> all, Iterable<String> patterns) {
        List<String> ps = new ArrayList<>();
        for (String p : patterns) ps.add(p);
        Map<String, Tensor> out = new LinkedHashMap<>();
        for (Map.Entry<String, Tensor> e : all.entrySet()) {
            if (matchesAny(e.getKey(), ps)) out.put(e.getKey(), e.getValue());
        }
        return out;
    }

    /**
     * Optional FP8 dequant via reflection so {@code data.safetensors} does not
     * hard-depend on {@code llm.transformers.loading.Fp8WeightDequant}.
     */
    @SuppressWarnings("unchecked")
    public static Map<String, Tensor> tryDequantFp8(Map<String, Tensor> weights) {
        try {
            Class<?> clz = Class.forName(
                    "org.bytedeco.pytorch.llm.transformers.loading.Fp8WeightDequant");
            Object r = clz.getMethod("dequantizeInPlace", Map.class).invoke(null, weights);
            if (r instanceof Map<?, ?> m) {
                return (Map<String, Tensor>) m;
            }
        } catch (ClassNotFoundException ignored) {
            // optional
        } catch (Throwable t) {
            System.err.println("[ShardedSafeTensors] FP8 dequant skipped: " + t.getMessage());
        }
        return weights;
    }
}
