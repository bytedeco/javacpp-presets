package org.bytedeco.pytorch.data.serialize;

import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.data.safetensors.LoadOptions;
import org.bytedeco.pytorch.data.safetensors.SafeTensors;
import org.bytedeco.pytorch.data.safetensors.ShardedSafeTensors;

import java.io.File;
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
 * Build a trainable {@link WeightBagModule} from Recsys checkpoints
 * ({@code config.json} + single/sharded {@code .safetensors}).
 *
 * <p>Supports DeepFM / DIN / DSSM / WideDeep / NFM / XDeepFM / HoFM etc.
 * Heuristics infer tower architecture from key patterns (embed, linear, mlp, fm).
 */
public final class RecsysModuleBuilder {

    private static final String[] TOWER_PATTERNS = {
            "user_tower", "item_tower", "deep", "fm", "linear", "mlp", "embedding",
            "wide_deep", "deepfm", "din", "dssm", "nfm", "xdeepfm", "hofm", "afm"
    };

    private RecsysModuleBuilder() {}

    // ---- public entry points ------------------------------------------------

    /** HF-style or simple recsys directory. */
    public static WeightBagModule fromDirectory(Path dir) throws IOException {
        return fromDirectory(dir, true, LoadOptions.defaults());
    }

    public static WeightBagModule fromDirectory(Path dir, boolean requiresGrad) throws IOException {
        return fromDirectory(dir, requiresGrad, LoadOptions.defaults());
    }

    public static WeightBagModule fromDirectory(Path dir, boolean requiresGrad, LoadOptions opts)
            throws IOException {
        Objects.requireNonNull(dir, "dir");
        if (opts == null) opts = LoadOptions.defaults();

        if (Files.isDirectory(dir)) {
            Map<String, Tensor> weights = ShardedSafeTensors.loadDirectory(dir, opts);
            Map<String, String> structure = inferRecsysStructureMeta(weights);
            return new WeightBagModule(weights, requiresGrad, true, true, structure);
        }
        // single file
        return fromSafetensorsWithConfig(dir, findSiblingConfig(dir), requiresGrad, opts);
    }

    public static WeightBagModule fromSafetensorsWithConfig(File weightsFile, File configFile)
            throws IOException {
        return fromSafetensorsWithConfig(weightsFile.toPath(), configFile == null ? null : configFile.toPath(),
                true, LoadOptions.defaults());
    }

    public static WeightBagModule fromSafetensorsWithConfig(Path weightsPath, Path configPath,
                                                             boolean requiresGrad, LoadOptions opts)
            throws IOException {
        Objects.requireNonNull(weightsPath, "weightsPath");
        if (opts == null) opts = LoadOptions.defaults();

        Map<String, String> meta = null;
        Map<String, Tensor> weights;
        File wf = weightsPath.toFile();

        if (Files.isDirectory(weightsPath)) {
            weights = ShardedSafeTensors.loadDirectory(weightsPath, opts);
        } else if (weightsPath.getFileName().toString().toLowerCase(Locale.ROOT).endsWith("index.json")) {
            weights = ShardedSafeTensors.loadIndex(weightsPath, opts);
        } else {
            weights = SafeTensors.loadAsTensors(wf, opts.zeroCopy);
            if (opts.dequantFp8) weights = ShardedSafeTensors.tryDequantFp8(weights);
            weights = SafeTensors.applyMapLocation(weights, opts);
            meta = SafeTensors.readMetadata(wf);
        }

        Map<String, String> structure = inferRecsysStructureMeta(weights);
        mergeEncodedStructure(structure, meta);

        return new WeightBagModule(weights, requiresGrad, true, true, structure);
    }

    public static Map<String, Tensor> loadWeightsOnly(Path path) throws IOException {
        return loadWeightsOnly(path, LoadOptions.weightsOnly());
    }

    public static Map<String, Tensor> loadWeightsOnly(Path path, LoadOptions opts) throws IOException {
        if (opts == null) opts = LoadOptions.weightsOnly();
        else if (!opts.weightsOnly) {
            opts = opts.toBuilder().weightsOnly(true).build();
        }
        return SafeTensors.loadFile(path.toFile(), opts);
    }

    // ---- internals ----------------------------------------------------------

    static void mergeEncodedStructure(Map<String, String> into, Map<String, String> fileMeta) {
        if (into == null || fileMeta == null) return;
        String enc = fileMeta.get("module_structure");
        if (enc == null) enc = fileMeta.get("structure");
        if (enc == null || enc.isEmpty()) return;
        try {
            Map<String, String> decoded = StateDictModuleBuilder.decodeStructureMeta(enc);
            for (Map.Entry<String, String> e : decoded.entrySet()) {
                into.putIfAbsent(e.getKey(), e.getValue());
            }
        } catch (Throwable ignored) {}
    }

    static Map<String, String> inferRecsysStructureMeta(Map<String, Tensor> weights) {
        Map<String, String> meta = new LinkedHashMap<>();
        if (weights == null || weights.isEmpty()) return meta;

        for (String k : weights.keySet()) {
            String path = parentPath(k);
            if (path == null) continue;
            String leaf = leafName(k).toLowerCase(Locale.ROOT);
            String pl = path.toLowerCase(Locale.ROOT);

            if ("embedding".equals(leaf) || pl.contains("embedding") || pl.contains("embed")) {
                meta.putIfAbsent(path, "EMBEDDING");
            }
            if ("linear".equals(leaf) || pl.contains("linear") || pl.contains("fc")) {
                meta.putIfAbsent(path, "LINEAR");
            }
            if (pl.contains("mlp") || pl.contains("tower") || pl.contains("deep")) {
                meta.putIfAbsent(path, "MLP");
            }
            if (pl.contains("fm") || pl.contains("wide") || pl.contains("wide_deep")) {
                meta.putIfAbsent(path, "FM");
            }
            if (pl.contains("din") || pl.contains("sequence")) {
                meta.putIfAbsent(path, "DIN");
            }
            if (pl.contains("dssm")) {
                meta.putIfAbsent(path, "DSSM");
            }
            if (pl.contains("nfm")) {
                meta.putIfAbsent(path, "NFM");
            }
            if (pl.contains("xdeepfm")) {
                meta.putIfAbsent(path, "XDEEPFM");
            }
            if (pl.contains("hofm")) {
                meta.putIfAbsent(path, "HOFM");
            }
            if (pl.contains("afm")) {
                meta.putIfAbsent(path, "AFM");
            }
            if (pl.endsWith("logit") || pl.endsWith("output") || pl.endsWith("pred")) {
                meta.putIfAbsent(path, "LINEAR");
            }
        }
        return meta;
    }

    static Path findSiblingConfig(Path weightsOrDir) {
        if (weightsOrDir == null) return null;
        if (Files.isDirectory(weightsOrDir)) {
            Path c = weightsOrDir.resolve("config.json");
            return Files.isRegularFile(c) ? c : null;
        }
        Path parent = weightsOrDir.getParent();
        if (parent == null) return null;
        Path c = parent.resolve("config.json");
        return Files.isRegularFile(c) ? c : null;
    }

    static String parentPath(String key) {
        if (key == null) return null;
        int dot = key.lastIndexOf('.');
        if (dot <= 0) return null;
        return key.substring(0, dot);
    }

    static String leafName(String key) {
        if (key == null) return null;
        int dot = key.lastIndexOf('.');
        return dot < 0 ? key : key.substring(dot + 1);
    }
}
