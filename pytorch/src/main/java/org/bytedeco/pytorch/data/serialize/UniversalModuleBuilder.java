package org.bytedeco.pytorch.data.serialize;

import org.bytedeco.pytorch.Device;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.data.safetensors.LoadOptions;
import org.bytedeco.pytorch.data.safetensors.SafeTensors;
import org.bytedeco.pytorch.data.safetensors.ShardedSafeTensors;
import org.bytedeco.pytorch.llm.transformers.PretrainedConfig;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;

/**
 * Single entry point that auto-detects checkpoint domain and rebuilds a trainable
 * {@link WeightBagModule}.
 *
 * <p>Dispatch order:
 * <ol>
 *   <li>{@link Domain#LLM} — HF {@code config.json} + transformer key patterns
 *       → {@link LLMModuleBuilder}</li>
 *   <li>{@link Domain#RECSYS} — DeepFM / DIN / DSSM / tower key patterns
 *       → {@link RecsysModuleBuilder}</li>
 *   <li>{@link Domain#GENERIC} — format auto-detect
 *       → {@link WeightBagModule#fromFile} / {@link ModelWeights#toModule}</li>
 * </ol>
 *
 * <pre>{@code
 *   // One call for HF LLM dirs, recsys safetensors, or plain .pth:
 *   WeightBagModule bag = UniversalModuleBuilder.fromPath("Llama-3.2-1B");
 *   // or force domain:
 *   WeightBagModule bag = UniversalModuleBuilder.fromPath(path, true, opts, Domain.LLM);
 * }</pre>
 */
public final class UniversalModuleBuilder {

    /** High-level checkpoint family used for structure-meta dispatch. */
    public enum Domain {
        /** Llama / Qwen / Mistral / Gemma / GPT-2 style transformers. */
        LLM,
        /** DeepFM / DIN / DSSM / WideDeep / NFM / XDeepFM / … */
        RECSYS,
        /** No domain-specific structure meta — typed leaf inference only. */
        GENERIC,
        /** Auto-detect from path layout + keys / config.json. */
        AUTO
    }

    private UniversalModuleBuilder() {}

    // ---- public entry points ------------------------------------------------

    public static WeightBagModule fromPath(Path path) throws IOException {
        return fromPath(path, true, LoadOptions.defaults(), Domain.AUTO);
    }

    public static WeightBagModule fromPath(String path) throws IOException {
        return fromPath(Path.of(path));
    }

    public static WeightBagModule fromPath(File path) throws IOException {
        return fromPath(path.toPath());
    }

    public static WeightBagModule fromPath(Path path, boolean requiresGrad) throws IOException {
        return fromPath(path, requiresGrad, LoadOptions.defaults(), Domain.AUTO);
    }

    public static WeightBagModule fromPath(Path path, boolean requiresGrad, LoadOptions opts)
            throws IOException {
        return fromPath(path, requiresGrad, opts, Domain.AUTO);
    }

    /**
     * Load any supported checkpoint and rebuild a typed Module.
     *
     * @param domain {@link Domain#AUTO} to sniff, or force LLM / RECSYS / GENERIC
     */
    public static WeightBagModule fromPath(Path path, boolean requiresGrad, LoadOptions opts,
                                             Domain domain) throws IOException {
        Objects.requireNonNull(path, "path");
        if (opts == null) opts = LoadOptions.defaults();
        if (domain == null) domain = Domain.AUTO;

        Domain resolved = domain == Domain.AUTO ? detectDomain(path) : domain;
        switch (resolved) {
            case LLM:
                return LLMModuleBuilder.fromHuggingFace(path, requiresGrad, opts);
            case RECSYS:
                return RecsysModuleBuilder.fromDirectory(path, requiresGrad, opts);
            case GENERIC:
            default:
                return loadGeneric(path, requiresGrad, opts);
        }
    }

    public static WeightBagModule fromPath(Path path, boolean requiresGrad,
                                             Device mapLocation, boolean strict) throws IOException {
        return fromPath(path, requiresGrad, LoadOptions.builder()
                .mapLocation(mapLocation)
                .strict(strict)
                .build(), Domain.AUTO);
    }

    /**
     * Single safetensors (or dir) + optional config.json, with domain auto-detect.
     * When config is a HF transformer config → LLM path; otherwise recsys/generic.
     */
    public static WeightBagModule fromSafetensorsWithConfig(Path weightsPath, Path configPath)
            throws IOException {
        return fromSafetensorsWithConfig(weightsPath, configPath, true, LoadOptions.defaults(),
                Domain.AUTO);
    }

    public static WeightBagModule fromSafetensorsWithConfig(File weightsFile, File configFile)
            throws IOException {
        return fromSafetensorsWithConfig(
                weightsFile.toPath(),
                configFile == null ? null : configFile.toPath(),
                true, LoadOptions.defaults(), Domain.AUTO);
    }

    public static WeightBagModule fromSafetensorsWithConfig(Path weightsPath, Path configPath,
                                                             boolean requiresGrad, LoadOptions opts)
            throws IOException {
        return fromSafetensorsWithConfig(weightsPath, configPath, requiresGrad, opts, Domain.AUTO);
    }

    public static WeightBagModule fromSafetensorsWithConfig(Path weightsPath, Path configPath,
                                                             boolean requiresGrad, LoadOptions opts,
                                                             Domain domain) throws IOException {
        Objects.requireNonNull(weightsPath, "weightsPath");
        if (opts == null) opts = LoadOptions.defaults();
        if (domain == null) domain = Domain.AUTO;

        Path cfg = configPath;
        if (cfg == null || !Files.isRegularFile(cfg)) {
            cfg = findSiblingConfig(weightsPath);
        }

        Domain resolved = domain;
        if (resolved == Domain.AUTO) {
            resolved = detectDomainFromConfigAndKeys(weightsPath, cfg);
        }

        switch (resolved) {
            case LLM:
                return LLMModuleBuilder.fromSafetensorsWithConfig(weightsPath, cfg, requiresGrad, opts);
            case RECSYS:
                return RecsysModuleBuilder.fromSafetensorsWithConfig(weightsPath, cfg, requiresGrad, opts);
            case GENERIC:
            default: {
                // Load tensors + optional structure meta, then typed bag
                Map<String, Tensor> weights = loadTensors(weightsPath, opts);
                Map<String, String> structure = new LinkedHashMap<>();
                Map<String, String> fileMeta = null;
                if (Files.isRegularFile(weightsPath)
                        && weightsPath.getFileName().toString().toLowerCase(Locale.ROOT)
                        .endsWith(".safetensors")) {
                    try {
                        fileMeta = SafeTensors.readMetadata(weightsPath.toFile());
                    } catch (Throwable ignored) {}
                }
                mergeEncodedStructure(structure, fileMeta);
                // If we have a config but it wasn't LLM, still try mild LLM structure
                // only when keys look transformer-like
                if (structure.isEmpty()) {
                    String fam = LLMModuleBuilder.detectFamily(weights);
                    if (!"generic".equals(fam) && !"unknown".equals(fam)) {
                        PretrainedConfig pc = null;
                        if (cfg != null && Files.isRegularFile(cfg)) {
                            try { pc = PretrainedConfig.fromFile(cfg); } catch (Throwable ignored) {}
                        }
                        structure.putAll(LLMModuleBuilder.buildLlmStructureMeta(weights, pc));
                    } else {
                        structure.putAll(RecsysModuleBuilder.inferRecsysStructureMeta(weights));
                        // If recsys heuristics found nothing useful, leave empty (pure typed)
                        if (structure.isEmpty() || onlyWeakRecsysTags(structure)) {
                            structure.clear();
                        }
                    }
                }
                return new WeightBagModule(weights, requiresGrad, /*clone=*/true, /*typed=*/true,
                        structure.isEmpty() ? null : structure);
            }
        }
    }

    /**
     * Python {@code torch.load(..., weights_only=True)} for any path
     * (file / dir / index.json). Never builds a Module.
     */
    public static Map<String, Tensor> loadWeightsOnly(Path path) throws IOException {
        return loadWeightsOnly(path, LoadOptions.weightsOnly());
    }

    public static Map<String, Tensor> loadWeightsOnly(Path path, LoadOptions opts) throws IOException {
        if (opts == null) opts = LoadOptions.weightsOnly();
        else if (!opts.weightsOnly) {
            opts = opts.toBuilder().weightsOnly(true).build();
        }
        return loadTensors(path, opts);
    }

    public static Map<String, Tensor> loadWeightsOnly(File path, Device device) throws IOException {
        return loadWeightsOnly(path.toPath(), LoadOptions.weightsOnly(device));
    }

    // ---- domain detection ---------------------------------------------------

    /**
     * Sniff path layout + optional config.json + (cheap) key samples.
     * Does <b>not</b> fully materialize a Module.
     */
    public static Domain detectDomain(Path path) {
        if (path == null) return Domain.GENERIC;
        try {
            Path cfg = null;
            if (Files.isDirectory(path)) {
                cfg = path.resolve("config.json");
                if (!Files.isRegularFile(cfg)) cfg = path.resolve("model_config.json");
                if (!Files.isRegularFile(cfg)) cfg = null;

                // HF transformer config is strong signal
                if (cfg != null) {
                    Domain fromCfg = domainFromConfigFile(cfg);
                    if (fromCfg != Domain.GENERIC) return fromCfg;
                }

                // Peek keys via first shard / single file without full merge when possible
                try {
                    List<Path> shards = ShardedSafeTensors.resolveShards(path);
                    if (!shards.isEmpty()) {
                        Map<String, Tensor> sample = SafeTensors.loadAsTensors(shards.get(0).toFile(), true);
                        return detectDomainFromWeights(sample, cfg);
                    }
                } catch (Throwable ignored) {}

                // Directory with config but no shards yet — still prefer LLM if config says so
                if (cfg != null) return Domain.LLM;
                return Domain.GENERIC;
            }

            String name = path.getFileName() == null ? ""
                    : path.getFileName().toString().toLowerCase(Locale.ROOT);
            cfg = findSiblingConfig(path);

            if (name.endsWith("index.json")) {
                if (cfg != null) {
                    Domain d = domainFromConfigFile(cfg);
                    if (d != Domain.GENERIC) return d;
                }
                return Domain.LLM; // index.json is almost always HF
            }

            if (name.endsWith(".safetensors")) {
                try {
                    Map<String, String> meta = SafeTensors.readMetadata(path.toFile());
                    if (meta != null) {
                        if (meta.containsKey("__llm.family") || meta.containsKey("__llm.model_type")
                                || (meta.get("format") != null
                                && meta.get("format").toLowerCase(Locale.ROOT).contains("pt"))) {
                            // still need keys for recsys vs llm; fall through
                        }
                    }
                    // Sample tensors for key heuristics (mmap ok)
                    Map<String, Tensor> sample = SafeTensors.loadAsTensors(path.toFile(), true);
                    return detectDomainFromWeights(sample, cfg);
                } catch (Throwable t) {
                    if (cfg != null) {
                        Domain d = domainFromConfigFile(cfg);
                        if (d != Domain.GENERIC) return d;
                    }
                    return Domain.GENERIC;
                }
            }

            // .pth / .pt / unknown — generic (Python path or format sniff)
            if (cfg != null) {
                Domain d = domainFromConfigFile(cfg);
                if (d != Domain.GENERIC) return d;
            }
            return Domain.GENERIC;
        } catch (Throwable t) {
            return Domain.GENERIC;
        }
    }

    public static Domain detectDomain(File path) {
        return detectDomain(path == null ? null : path.toPath());
    }

    public static Domain detectDomainFromWeights(Map<String, Tensor> weights) {
        return detectDomainFromWeights(weights, null);
    }

    public static Domain detectDomainFromWeights(Map<String, Tensor> weights, Path configOrNull) {
        if (configOrNull != null && Files.isRegularFile(configOrNull)) {
            Domain d = domainFromConfigFile(configOrNull);
            if (d == Domain.LLM) return Domain.LLM;
            // RECSYS config is rare; key heuristics dominate
        }
        if (weights == null || weights.isEmpty()) return Domain.GENERIC;

        String family = LLMModuleBuilder.detectFamily(weights);
        if ("llama_qwen_mistral".equals(family) || "gemma".equals(family)
                || "gpt2".equals(family) || "transformer".equals(family)) {
            return Domain.LLM;
        }

        if (looksLikeRecsys(weights)) return Domain.RECSYS;
        return Domain.GENERIC;
    }

    // ---- internals ----------------------------------------------------------

    private static WeightBagModule loadGeneric(Path path, boolean requiresGrad, LoadOptions opts)
            throws IOException {
        if (Files.isDirectory(path)
                || (path.getFileName() != null
                && path.getFileName().toString().toLowerCase(Locale.ROOT).endsWith("index.json"))) {
            return WeightBagModule.fromSafetensors(path.toFile(), requiresGrad, opts);
        }
        String lower = path.getFileName() == null ? ""
                : path.getFileName().toString().toLowerCase(Locale.ROOT);
        if (lower.endsWith(".safetensors")) {
            return WeightBagModule.fromSafetensors(path.toFile(), requiresGrad, opts);
        }
        // .pth / .pt / magic sniff
        return WeightBagModule.fromFile(path.toFile(), requiresGrad, opts);
    }

    private static Map<String, Tensor> loadTensors(Path path, LoadOptions opts) throws IOException {
        if (Files.isDirectory(path)) {
            return ShardedSafeTensors.loadDirectory(path, opts);
        }
        String lower = path.getFileName() == null ? ""
                : path.getFileName().toString().toLowerCase(Locale.ROOT);
        if (lower.endsWith("index.json")) {
            return ShardedSafeTensors.loadIndex(path, opts);
        }
        if (lower.endsWith(".safetensors") || SafeTensorsLooksLike(path)) {
            return SafeTensors.loadFile(path.toFile(), opts);
        }
        // pth / auto
        return ModelWeights.load(path.toFile(), true, opts);
    }

    private static boolean SafeTensorsLooksLike(Path path) {
        try {
            return ModelWeights.detect(path.toFile()) == ModelWeights.Format.SAFETENSORS;
        } catch (Throwable t) {
            return false;
        }
    }

    private static Domain detectDomainFromConfigAndKeys(Path weightsPath, Path cfg) {
        if (cfg != null && Files.isRegularFile(cfg)) {
            Domain d = domainFromConfigFile(cfg);
            if (d == Domain.LLM) return Domain.LLM;
        }
        try {
            Map<String, Tensor> w = loadTensors(weightsPath,
                    LoadOptions.builder().zeroCopy(true).dequantFp8(false).build());
            return detectDomainFromWeights(w, cfg);
        } catch (Throwable t) {
            return cfg != null ? Domain.LLM : Domain.GENERIC;
        }
    }

    private static Domain domainFromConfigFile(Path cfg) {
        try {
            // Fast path: read model_type without full parse failure cost
            String text = Files.readString(cfg);
            String lower = text.toLowerCase(Locale.ROOT);
            if (lower.contains("\"model_type\"")
                    || lower.contains("num_hidden_layers")
                    || lower.contains("num_attention_heads")
                    || lower.contains("rms_norm_eps")
                    || lower.contains("rope_theta")
                    || lower.contains("architectures")) {
                // Try real parse for known transformer types
                try {
                    PretrainedConfig pc = PretrainedConfig.fromFile(cfg);
                    PretrainedConfig.ModelType mt = pc.modelType();
                    if (mt != null
                            && mt != PretrainedConfig.ModelType.GENERIC) {
                        return Domain.LLM;
                    }
                    // Even GENERIC model_type with transformer hypers → LLM
                    if (pc.numHiddenLayers() > 0 && pc.hiddenSize() > 0) return Domain.LLM;
                } catch (Throwable ignored) {
                    // string heuristics
                    if (lower.contains("llama") || lower.contains("qwen")
                            || lower.contains("mistral") || lower.contains("gemma")
                            || lower.contains("gpt2") || lower.contains("gpt_neox")
                            || lower.contains("phi") || lower.contains("falcon")
                            || lower.contains("bloom") || lower.contains("opt")) {
                        return Domain.LLM;
                    }
                }
            }
            // Recsys-ish config keys
            if (lower.contains("embedding_dim") || lower.contains("field_num")
                    || lower.contains("deepfm") || lower.contains("xdeepfm")
                    || lower.contains("\"din\"") || lower.contains("dssm")
                    || lower.contains("wide_deep") || lower.contains("feature_columns")) {
                return Domain.RECSYS;
            }
        } catch (Throwable ignored) {}
        return Domain.GENERIC;
    }

    private static boolean looksLikeRecsys(Map<String, Tensor> weights) {
        int hits = 0;
        boolean hasEmbed = false, hasTower = false, hasFm = false, hasMlp = false;
        for (String k : weights.keySet()) {
            String lk = k.toLowerCase(Locale.ROOT);
            if (lk.contains("embedding") || lk.contains("embed_dict")
                    || lk.contains("sparse_emb") || lk.contains("feature_emb")) {
                hasEmbed = true; hits++;
            }
            if (lk.contains("user_tower") || lk.contains("item_tower")
                    || lk.contains("user_mlp") || lk.contains("item_mlp")
                    || lk.contains("user_gmf") || lk.contains("item_gmf")) {
                hasTower = true; hits += 2;
            }
            if (lk.contains(".fm.") || lk.contains("fm_linear") || lk.contains("wide")
                    || lk.contains("deepfm") || lk.contains("xdeepfm") || lk.contains("cin.")) {
                hasFm = true; hits += 2;
            }
            if (lk.contains("mlp") || lk.contains("dnn") || lk.contains("deep_layers")) {
                hasMlp = true; hits++;
            }
            if (lk.contains("attention_mlp") || lk.contains("din") || lk.contains("history_emb")) {
                hits += 2;
            }
            // Strong LLM countersignals
            if (lk.contains("gate_proj") || lk.contains("q_proj") || lk.contains("layers.")
                    || lk.contains("embed_tokens") || lk.contains("lm_head")
                    || lk.contains("input_layernorm") || lk.contains("c_attn")) {
                return false;
            }
        }
        return hits >= 3 && (hasEmbed || hasTower || hasFm) && (hasMlp || hasTower || hasFm);
    }

    /** Recsys meta that only tagged paths as MLP/FM without real structure value. */
    private static boolean onlyWeakRecsysTags(Map<String, String> structure) {
        if (structure == null || structure.isEmpty()) return true;
        for (String v : structure.values()) {
            if (v == null) continue;
            String u = v.toUpperCase(Locale.ROOT);
            if ("EMBEDDING".equals(u) || "LINEAR".equals(u)
                    || "RELU".equals(u) || "DROPOUT".equals(u)
                    || u.startsWith("DROPOUT:")) {
                return false; // useful for StateDictModuleBuilder
            }
        }
        // Only MLP/FM/DIN/… tokens — StateDictModuleBuilder treats unknown as CONTAINER/RELU
        return true;
    }

    static Path findSiblingConfig(Path weightsOrDir) {
        if (weightsOrDir == null) return null;
        if (Files.isDirectory(weightsOrDir)) {
            Path c = weightsOrDir.resolve("config.json");
            if (Files.isRegularFile(c)) return c;
            Path alt = weightsOrDir.resolve("model_config.json");
            return Files.isRegularFile(alt) ? alt : null;
        }
        Path parent = weightsOrDir.getParent();
        if (parent == null) return null;
        Path c = parent.resolve("config.json");
        if (Files.isRegularFile(c)) return c;
        Path alt = parent.resolve("model_config.json");
        return Files.isRegularFile(alt) ? alt : null;
    }

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
}
