package org.bytedeco.pytorch.data.serialize;

import org.bytedeco.pytorch.Device;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.data.safetensors.LoadOptions;
import org.bytedeco.pytorch.data.safetensors.SafeTensors;
import org.bytedeco.pytorch.data.safetensors.ShardedSafeTensors;
import org.bytedeco.pytorch.llm.transformers.PretrainedConfig;
import org.bytedeco.pytorch.utils.json.Json;

import java.io.File;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Build a trainable {@link WeightBagModule} from HuggingFace-style LLM checkpoints
 * ({@code config.json} + single/sharded {@code .safetensors}).
 *
 * <p>Pure Java — no CPython. Complements generic {@link StateDictModuleBuilder} with:
 * <ul>
 *   <li>config.json → {@link PretrainedConfig} hypers (hidden, heads, layers, rope, rms eps)</li>
 *   <li>Llama/Qwen/Mistral/Gemma key heuristics → structure meta for RMSNorm / SiLU gaps</li>
 *   <li>directory + index.json multi-shard load via {@link ShardedSafeTensors}</li>
 *   <li>{@link LoadOptions} ({@code map_location}, zero-copy, dtype, dequantFp8)</li>
 * </ul>
 *
 * <pre>{@code
 *   WeightBagModule bag = LLMModuleBuilder.fromHuggingFace(Path.of("Llama-3.2-1B"));
 *   // or weights_only:
 *   Map<String,Tensor> sd = LLMModuleBuilder.loadWeightsOnly(dir, LoadOptions.weightsOnly());
 * }</pre>
 */
public final class LLMModuleBuilder {

    private static final Pattern LAYER_IDX =
            Pattern.compile("(?:^|\\.)(?:model\\.)?layers\\.(\\d+)\\.");
    private static final Pattern DECODER_IDX =
            Pattern.compile("(?:^|\\.)(?:transformer\\.)?h\\.(\\d+)\\.");

    private LLMModuleBuilder() {}

    // ---- public entry points ------------------------------------------------

    /** Load HF model directory ({@code config.json} + safetensors shards). */
    public static WeightBagModule fromHuggingFace(Path dir) throws IOException {
        return fromHuggingFace(dir, true, LoadOptions.defaults());
    }

    public static WeightBagModule fromHuggingFace(File dir) throws IOException {
        return fromHuggingFace(dir.toPath(), true, LoadOptions.defaults());
    }

    public static WeightBagModule fromHuggingFace(String dir) throws IOException {
        return fromHuggingFace(Path.of(dir), true, LoadOptions.defaults());
    }

    public static WeightBagModule fromHuggingFace(Path dir, boolean requiresGrad) throws IOException {
        return fromHuggingFace(dir, requiresGrad, LoadOptions.defaults());
    }

    public static WeightBagModule fromHuggingFace(Path dir, boolean requiresGrad, LoadOptions opts)
            throws IOException {
        Objects.requireNonNull(dir, "dir");
        if (opts == null) opts = LoadOptions.defaults();
        if (!Files.isDirectory(dir)) {
            // single file → treat as safetensors next to optional sibling config.json
            return fromSafetensorsWithConfig(dir, findSiblingConfig(dir), requiresGrad, opts);
        }

        PretrainedConfig cfg = loadConfig(dir);
        Map<String, Tensor> weights = ShardedSafeTensors.loadDirectory(dir, opts);
        if (weights.isEmpty()) {
            throw new IOException("No tensors loaded from HF directory: " + dir);
        }
        Map<String, String> structure = buildLlmStructureMeta(weights, cfg);
        // merge any module_structure already in shard metadata
        try {
            List<Path> shards = ShardedSafeTensors.resolveShards(dir);
            if (!shards.isEmpty()) {
                Map<String, String> meta = SafeTensors.readMetadata(shards.get(0).toFile());
                mergeEncodedStructure(structure, meta);
            }
        } catch (Throwable ignored) {}

        WeightBagModule bag = new WeightBagModule(weights, requiresGrad, true, true, structure);
        stampConfigMeta(bag, cfg);
        return bag;
    }

    public static WeightBagModule fromHuggingFace(Path dir, boolean requiresGrad,
                                                    Device mapLocation, boolean strict)
            throws IOException {
        return fromHuggingFace(dir, requiresGrad, LoadOptions.builder()
                .mapLocation(mapLocation)
                .strict(strict)
                .build());
    }

    /**
     * Single safetensors + optional {@code config.json}.
     * When config is null, hypers are inferred from tensor shapes / key patterns only.
     */
    public static WeightBagModule fromSafetensorsWithConfig(File weightsFile, File configFile)
            throws IOException {
        return fromSafetensorsWithConfig(weightsFile.toPath(),
                configFile == null ? null : configFile.toPath(),
                true, LoadOptions.defaults());
    }

    public static WeightBagModule fromSafetensorsWithConfig(Path weightsPath, Path configPath,
                                                             boolean requiresGrad, LoadOptions opts)
            throws IOException {
        Objects.requireNonNull(weightsPath, "weightsPath");
        if (opts == null) opts = LoadOptions.defaults();

        PretrainedConfig cfg = null;
        if (configPath != null && Files.isRegularFile(configPath)) {
            cfg = PretrainedConfig.fromFile(configPath);
        } else {
            Path sibling = findSiblingConfig(weightsPath);
            if (sibling != null) cfg = PretrainedConfig.fromFile(sibling);
        }

        Map<String, Tensor> weights;
        Map<String, String> fileMeta = null;
        File wf = weightsPath.toFile();
        if (Files.isDirectory(weightsPath)) {
            weights = ShardedSafeTensors.loadDirectory(weightsPath, opts);
        } else if (weightsPath.getFileName().toString().toLowerCase(Locale.ROOT).endsWith("index.json")) {
            weights = ShardedSafeTensors.loadIndex(weightsPath, opts);
        } else {
            weights = SafeTensors.loadAsTensors(wf, opts.zeroCopy);
            if (opts.dequantFp8) weights = ShardedSafeTensors.tryDequantFp8(weights);
            weights = SafeTensors.applyMapLocation(weights, opts);
            fileMeta = SafeTensors.readMetadata(wf);
        }

        Map<String, String> structure = buildLlmStructureMeta(weights, cfg);
        mergeEncodedStructure(structure, fileMeta);

        WeightBagModule bag = new WeightBagModule(weights, requiresGrad, true, true, structure);
        if (cfg != null) stampConfigMeta(bag, cfg);
        return bag;
    }

    /**
     * Python {@code torch.load(..., weights_only=True)} for HF LLM dirs / files.
     */
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

    public static Map<String, Tensor> loadWeightsOnly(File path, Device device) throws IOException {
        return loadWeightsOnly(path.toPath(), LoadOptions.weightsOnly(device));
    }

    /** Read {@code config.json} from a HF model directory (required). */
    public static PretrainedConfig loadConfig(Path dir) throws IOException {
        Path cfg = dir.resolve("config.json");
        if (!Files.isRegularFile(cfg)) {
            // some exports use model_config.json
            Path alt = dir.resolve("model_config.json");
            if (Files.isRegularFile(alt)) cfg = alt;
            else throw new IOException("config.json not found in " + dir);
        }
        return PretrainedConfig.fromFile(cfg);
    }

    public static PretrainedConfig loadConfigOrNull(Path dir) {
        try {
            Path cfg = dir.resolve("config.json");
            if (!Files.isRegularFile(cfg)) return null;
            return PretrainedConfig.fromFile(cfg);
        } catch (Throwable t) {
            return null;
        }
    }

    /**
     * Infer LLM family from state-dict keys (no config required).
     */
    public static String detectFamily(Map<String, Tensor> weights) {
        if (weights == null || weights.isEmpty()) return "unknown";
        boolean hasGate = false, hasCAttn = false, hasQProj = false, hasGemma = false;
        boolean hasLayers = false, hasH = false;
        int maxLayer = -1;
        for (String k : weights.keySet()) {
            String lk = k.toLowerCase(Locale.ROOT);
            if (lk.contains("gate_proj")) hasGate = true;
            if (lk.contains("c_attn")) hasCAttn = true;
            if (lk.contains("q_proj")) hasQProj = true;
            if (lk.contains("gemma") || lk.contains("mlp.gate_up_proj")) hasGemma = true;
            Matcher m = LAYER_IDX.matcher(k);
            if (m.find()) {
                hasLayers = true;
                try { maxLayer = Math.max(maxLayer, Integer.parseInt(m.group(1))); } catch (NumberFormatException ignored) {}
            }
            m = DECODER_IDX.matcher(k);
            if (m.find()) {
                hasH = true;
                try { maxLayer = Math.max(maxLayer, Integer.parseInt(m.group(1))); } catch (NumberFormatException ignored) {}
            }
        }
        if (hasGemma) return "gemma";
        if (hasGate && hasQProj) return "llama_qwen_mistral"; // SwiGLU + GQA family
        if (hasCAttn) return "gpt2";
        if (hasLayers || hasH) return "transformer";
        return "generic";
    }

    /**
     * Build compact structure metadata so {@link StateDictModuleBuilder} inserts
     * RMSNorm / SiLU / identity gaps for Llama-style decoder blocks.
     *
     * <p>Tokens follow the existing encode convention:
     * {@code path → "SILU"}, {@code path → "RMS_NORM"}, etc.
     */
    public static Map<String, String> buildLlmStructureMeta(Map<String, Tensor> weights,
                                                             PretrainedConfig cfg) {
        Map<String, String> meta = new LinkedHashMap<>();
        if (weights == null || weights.isEmpty()) return meta;

        String family = detectFamily(weights);
        int nLayers = cfg != null ? cfg.numHiddenLayers() : inferNumLayers(weights);

        // Mark known RMSNorm leaves (1D weight, no bias) by path
        for (String key : weights.keySet()) {
            String path = parentPath(key);
            if (path == null) continue;
            String leaf = leafName(key);
            String pl = path.toLowerCase(Locale.ROOT);
            String ll = leaf.toLowerCase(Locale.ROOT);

            if ("weight".equals(ll) && (pl.endsWith("input_layernorm")
                    || pl.endsWith("post_attention_layernorm")
                    || pl.endsWith("norm")
                    || pl.contains("rms_norm")
                    || pl.endsWith("ln_f")
                    || pl.endsWith("final_layernorm")
                    || pl.endsWith("model.norm"))) {
                // RMS for llama family; LayerNorm for gpt2
                if ("gpt2".equals(family) || pl.contains("ln_")) {
                    meta.putIfAbsent(path, "LAYER_NORM");
                } else {
                    meta.putIfAbsent(path, "RMS_NORM");
                }
            }

            // SwiGLU: gate_proj / up_proj / down_proj — act is SiLU between gate*up
            if (pl.endsWith("mlp") || pl.contains(".mlp.")) {
                // ensure SiLU token for synthetic act child if Sequential gap-fill runs
                String actPath = path.contains(".mlp") ? path : path;
                // common export: mlp is ModuleDict of projs, act is implicit
            }
            if (ll.equals("weight") && (pl.endsWith("gate_proj") || pl.endsWith("up_proj"))) {
                // parent mlp may host an act sibling — tag mlp path with SILU hint
                String mlp = parentPath(path);
                if (mlp != null) meta.putIfAbsent(mlp + ".act", "SILU");
            }
        }

        // Decoder layer shells: mark residual style via config family
        if (nLayers > 0 && ("llama_qwen_mistral".equals(family) || "gemma".equals(family)
                || (cfg != null && cfg.modelType() != PretrainedConfig.ModelType.GPT2))) {
            String prefix = inferLayerPrefix(weights);
            for (int i = 0; i < nLayers; i++) {
                String layer = prefix + i;
                meta.putIfAbsent(layer, "DECODER_LAYER");
                meta.putIfAbsent(layer + ".self_attn", "ATTENTION");
                meta.putIfAbsent(layer + ".mlp", "SWIGLU_MLP");
                meta.putIfAbsent(layer + ".input_layernorm", "RMS_NORM");
                meta.putIfAbsent(layer + ".post_attention_layernorm", "RMS_NORM");
                meta.putIfAbsent(layer + ".mlp.act", "SILU");
            }
            meta.putIfAbsent(inferModelNormPath(weights, prefix), "RMS_NORM");
        } else if ("gpt2".equals(family)) {
            String prefix = hasKeyPrefix(weights, "transformer.h.") ? "transformer.h." : "h.";
            for (int i = 0; i <= Math.max(0, nLayers - 1); i++) {
                String layer = prefix + i;
                meta.putIfAbsent(layer, "DECODER_LAYER");
                meta.putIfAbsent(layer + ".attn", "ATTENTION");
                meta.putIfAbsent(layer + ".mlp", "GPT2_MLP");
                meta.putIfAbsent(layer + ".ln_1", "LAYER_NORM");
                meta.putIfAbsent(layer + ".ln_2", "LAYER_NORM");
                meta.putIfAbsent(layer + ".mlp.act", "GELU");
            }
        }

        // Embeddings / lm_head
        for (String key : weights.keySet()) {
            String path = parentPath(key);
            if (path == null) continue;
            String pl = path.toLowerCase(Locale.ROOT);
            if (pl.endsWith("embed_tokens") || pl.endsWith("wte") || pl.endsWith("word_embeddings")) {
                meta.putIfAbsent(path, "EMBEDDING");
            }
            if (pl.endsWith("lm_head") || pl.endsWith("embed_out")) {
                meta.putIfAbsent(path, "LINEAR");
            }
        }

        if (cfg != null) {
            meta.putIfAbsent("__llm.model_type", cfg.modelType().name());
            meta.putIfAbsent("__llm.hidden_size", String.valueOf(cfg.hiddenSize()));
            meta.putIfAbsent("__llm.num_hidden_layers", String.valueOf(cfg.numHiddenLayers()));
            meta.putIfAbsent("__llm.num_attention_heads", String.valueOf(cfg.numAttentionHeads()));
            meta.putIfAbsent("__llm.num_key_value_heads", String.valueOf(cfg.numKeyValueHeads()));
            meta.putIfAbsent("__llm.intermediate_size", String.valueOf(cfg.intermediateSize()));
            meta.putIfAbsent("__llm.rms_norm_eps", String.valueOf(cfg.rmsNormEps()));
            meta.putIfAbsent("__llm.rope_theta", String.valueOf(cfg.ropeTheta()));
            meta.putIfAbsent("__llm.vocab_size", String.valueOf(cfg.vocabSize()));
            meta.putIfAbsent("__llm.head_dim", String.valueOf(cfg.headDim()));
        }
        meta.putIfAbsent("__llm.family", family);
        return meta;
    }

    // ---- internals ----------------------------------------------------------

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

    static void stampConfigMeta(WeightBagModule bag, PretrainedConfig cfg) {
        if (bag == null || cfg == null) return;
        try {
            // structureMeta is unmodifiable view; re-save embeds via saveSafetensors path.
            // Put hypers into bag by re-merging through a soft put on internal map is not public;
            // callers can read via bag.structureMeta() which already has __llm.* from build.
        } catch (Throwable ignored) {}
    }

    static int inferNumLayers(Map<String, Tensor> weights) {
        int max = -1;
        for (String k : weights.keySet()) {
            Matcher m = LAYER_IDX.matcher(k);
            if (m.find()) {
                try { max = Math.max(max, Integer.parseInt(m.group(1))); } catch (NumberFormatException ignored) {}
            }
            m = DECODER_IDX.matcher(k);
            if (m.find()) {
                try { max = Math.max(max, Integer.parseInt(m.group(1))); } catch (NumberFormatException ignored) {}
            }
        }
        return max >= 0 ? max + 1 : 0;
    }

    static String inferLayerPrefix(Map<String, Tensor> weights) {
        if (hasKeyPrefix(weights, "model.layers.")) return "model.layers.";
        if (hasKeyPrefix(weights, "layers.")) return "layers.";
        if (hasKeyPrefix(weights, "transformer.h.")) return "transformer.h.";
        if (hasKeyPrefix(weights, "h.")) return "h.";
        if (hasKeyPrefix(weights, "model.decoder.layers.")) return "model.decoder.layers.";
        return "model.layers.";
    }

    static String inferModelNormPath(Map<String, Tensor> weights, String layerPrefix) {
        for (String cand : new String[]{
                "model.norm", "model.model.norm", "transformer.ln_f", "ln_f", "model.final_layernorm"
        }) {
            if (weights.containsKey(cand + ".weight") || hasKeyPrefix(weights, cand + ".")) {
                return cand;
            }
        }
        // default beside layers
        if (layerPrefix.startsWith("model.layers")) return "model.norm";
        if (layerPrefix.startsWith("transformer")) return "transformer.ln_f";
        return "model.norm";
    }

    static boolean hasKeyPrefix(Map<String, Tensor> weights, String prefix) {
        for (String k : weights.keySet()) {
            if (k.startsWith(prefix)) return true;
        }
        return false;
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

    /** Dump a minimal config.json skeleton next to a bag (round-trip helper). */
    public static void saveConfigSkeleton(WeightBagModule bag, Path outConfig) throws IOException {
        Objects.requireNonNull(outConfig, "outConfig");
        Map<String, Object> m = new LinkedHashMap<>();
        Map<String, String> sm = bag != null ? bag.structureMeta() : Map.of();
        m.put("model_type", sm.getOrDefault("__llm.model_type", "generic").toLowerCase(Locale.ROOT));
        putInt(m, "hidden_size", sm, "__llm.hidden_size", 0);
        putInt(m, "num_hidden_layers", sm, "__llm.num_hidden_layers", 0);
        putInt(m, "num_attention_heads", sm, "__llm.num_attention_heads", 0);
        putInt(m, "num_key_value_heads", sm, "__llm.num_key_value_heads", 0);
        putInt(m, "intermediate_size", sm, "__llm.intermediate_size", 0);
        putInt(m, "vocab_size", sm, "__llm.vocab_size", 0);
        putInt(m, "head_dim", sm, "__llm.head_dim", 0);
        if (sm.containsKey("__llm.rms_norm_eps")) {
            try { m.put("rms_norm_eps", Double.parseDouble(sm.get("__llm.rms_norm_eps"))); }
            catch (NumberFormatException ignored) {}
        }
        if (sm.containsKey("__llm.rope_theta")) {
            try { m.put("rope_theta", Double.parseDouble(sm.get("__llm.rope_theta"))); }
            catch (NumberFormatException ignored) {}
        }
        m.put("architectures", List.of("WeightBagModule"));
        m.put("converted_by", "org.bytedeco.pytorch.data.serialize.LLMModuleBuilder");
        Files.writeString(outConfig, Json.encode(m), StandardCharsets.UTF_8);
    }

    private static void putInt(Map<String, Object> m, String jsonKey,
                               Map<String, String> sm, String metaKey, int dflt) {
        String v = sm.get(metaKey);
        if (v == null) {
            if (dflt != 0) m.put(jsonKey, dflt);
            return;
        }
        try { m.put(jsonKey, Integer.parseInt(v)); }
        catch (NumberFormatException e) { if (dflt != 0) m.put(jsonKey, dflt); }
    }
}
