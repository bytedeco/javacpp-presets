package distribute;/*
 * Copyright (C) 2020-2026 Eduardo Gonzalez, Hervé Guillemet, Samuel Audet
 *
 * Offline smoke for transformers WeightLoader + Qwen2 structure.
 *
 *   1. Build tiny Qwen2 from config (random init)
 *   2. Dump named_parameters → safetensors
 *   3. Build a fresh model and ZERO_COPY-bind the dump
 *   4. Assert LoadReport.ok() and forward runs
 *
 * No network required.
 */

import org.bytedeco.pytorch.StringTensorDict;
import org.bytedeco.pytorch.StringTensorDictItem;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.data.safetensors.SafeTensors;
import org.bytedeco.pytorch.llm.transformers.AutoModelForCausalLM;
import org.bytedeco.pytorch.llm.transformers.PretrainedConfig;
import org.bytedeco.pytorch.llm.transformers.generation.GenerationConfig;
import org.bytedeco.pytorch.llm.transformers.loading.WeightLoader;
import org.bytedeco.pytorch.llm.transformers.mapping.ModelRegistry;
import org.bytedeco.pytorch.llm.transformers.mapping.WeightMaps;
import org.bytedeco.pytorch.llm.transformers.modeling.Qwen2ForCausalLM;
import org.bytedeco.pytorch.llm.transformers.tokenization.ChatTemplate;

import java.io.File;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import static org.bytedeco.pytorch.global.torch.tensor;

public final class TransformersOfflineSmoke {

    public static void main(String[] args) throws Exception {
        Path tmp = Files.createTempDirectory("tf-smoke-");
        System.out.println("tmp = " + tmp);

        // ---- 1. config + model ------------------------------------------------
        PretrainedConfig cfg = PretrainedConfig.tinyQwen();
        System.out.println("config: " + cfg);
        String cfgJson = cfg.toJson();
        Files.writeString(tmp.resolve("config.json"), cfgJson, StandardCharsets.UTF_8);
        // round-trip
        PretrainedConfig cfg2 = PretrainedConfig.fromJson(cfgJson);
        if (cfg2.hiddenSize() != cfg.hiddenSize() || cfg2.numHiddenLayers() != cfg.numHiddenLayers()) {
            throw new IllegalStateException("config round-trip failed: " + cfg2);
        }
        System.out.println("config round-trip OK");

        Qwen2ForCausalLM model = Qwen2ForCausalLM.fromConfig(cfg);
        model.eval();
        Map<String, Tensor> state = collect(model);
        System.out.println("params: " + state.size());
        if (state.isEmpty()) {
            throw new IllegalStateException("named_parameters empty — Module registration broken");
        }
        // print a few keys for HF-name sanity
        int shown = 0;
        for (String k : state.keySet()) {
            System.out.println("  key: " + k + " " + shape(state.get(k)));
            if (++shown >= 8) break;
        }

        // ---- 2. save safetensors ---------------------------------------------
        File st = tmp.resolve("model.safetensors").toFile();
        SafeTensors.save(state, st);
        System.out.println("wrote " + st + " (" + st.length() + " bytes)");

        // minimal tokenizer.json so fromDirectory can pair one
        Files.writeString(tmp.resolve("tokenizer.json"),
                org.bytedeco.pytorch.llm.tokenizers.FastTokenizer.whitespace()
                        .modelMaxLength(cfg.maxPositionEmbeddings())
                        .build()
                        .toTokenizerJson(),
                StandardCharsets.UTF_8);

        // ---- 3. reload ZERO_COPY ---------------------------------------------
        Qwen2ForCausalLM model2 = Qwen2ForCausalLM.fromConfig(cfg);
        model2.eval();
        WeightLoader.LoadReport report = WeightLoader.loadAndBind(
                model2, tmp, WeightMaps.qwen2(),
                WeightLoader.BindMode.ZERO_COPY, /*strict=*/true, /*mmap=*/true);
        System.out.println("load: " + report);
        if (!report.ok()) {
            throw new IllegalStateException("strict load failed: " + report);
        }
        if (report.rebound + report.copied != report.matchedCount()) {
            throw new IllegalStateException("rebound/copied mismatch: " + report);
        }
        System.out.println("ZERO_COPY bind OK matched=" + report.matchedCount()
                + " rebound=" + report.rebound);

        // ---- 4. forward + generate -------------------------------------------
        Tensor ids = tensor(new long[]{1, 2, 3, 4}).unsqueeze(0);
        Tensor logits = model2.forward(ids);
        System.out.println("logits shape: [" + logits.size(0) + "," + logits.size(1) + "," + logits.size(2) + "]");
        if (logits.size(2) != cfg.vocabSize()) {
            throw new IllegalStateException("vocab dim mismatch");
        }

        int[] out = model2.generate(new int[]{1, 2, 3},
                GenerationConfig.builder().maxNewTokens(4).eosTokenId(cfg.eosTokenId()).build());
        System.out.println("generate ids: " + java.util.Arrays.toString(out));

        // ---- 5. chat template ------------------------------------------------
        String prompt = ChatTemplate.qwen().apply(List.of(
                Map.of("role", "user", "content", "hi")
        ), true);
        if (!prompt.contains("<|im_start|>user") || !prompt.contains("<|im_start|>assistant")) {
            throw new IllegalStateException("bad chat template: " + prompt);
        }
        System.out.println("chat template OK: " + prompt.replace("\n", "\\n"));

        // ---- 6. AutoModelForCausalLM.fromDirectory ---------------------------
        AutoModelForCausalLM.Bundle bundle = AutoModelForCausalLM.fromDirectory(tmp);
        System.out.println("bundle load: " + bundle.loadReport());
        if (bundle.loadReport() == null || !bundle.loadReport().ok()) {
            throw new IllegalStateException("bundle load not ok");
        }
        String text = bundle.generate("hello", 4);
        System.out.println("bundle.generate => " + text);

        // registry resolve
        var entry = ModelRegistry.resolve(cfg);
        System.out.println("registry: " + entry.modelType());

        System.out.println("\nALL OFFLINE SMOKES PASSED");
    }

    private static Map<String, Tensor> collect(org.bytedeco.pytorch.nn.Module m) {
        Map<String, Tensor> out = new LinkedHashMap<>();
        StringTensorDict dict = m.named_parameters(true);
        if (dict == null || dict.isNull()) return out;
        for (long i = 0; i < dict.size(); i++) {
            StringTensorDictItem item = dict.get(i);
            if (item == null || item.isNull()) continue;
            String key = item.key() != null ? item.key().getString() : null;
            Tensor val = item.value();
            if (key != null && val != null && val.defined()) {
                // detach clone so save doesn't alias live params oddly
                out.put(key, val.contiguous().clone());
            }
        }
        return out;
    }

    private static String shape(Tensor t) {
        StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < t.dim(); i++) {
            if (i > 0) sb.append(',');
            sb.append(t.size(i));
        }
        return sb.append(']').toString();
    }
}
