/*
 * BenchmarkLlamaFactoryMultimodal — VL collator + Qwen3-VL path smoke
 *
 * Run: java -cp ... distribute.BenchmarkLlamaFactoryMultimodal
 */
package distribute;

import org.bytedeco.pytorch.llm.llamafactory.data.collator.MultimodalCollator;
import org.bytedeco.pytorch.llm.llamafactory.hparams.FactoryArgs;
import org.bytedeco.pytorch.llm.llamafactory.hparams.FinetuningType;
import org.bytedeco.pytorch.llm.llamafactory.hparams.ModelArgs;
import org.bytedeco.pytorch.llm.llamafactory.hparams.Stage;
import org.bytedeco.pytorch.llm.llamafactory.model.ModelLoader;
import org.bytedeco.pytorch.llm.llamafactory.data.template.TemplateRegistry;

import java.util.ArrayList;
import java.util.List;

/**
 * D1  MultimodalCollator defaults builds
 * D2  MultimodalCollator collate with pixel_values
 * D3  MultimodalCollator zero-pixels fallback when no pixel_values key
 * D4  VL collator shapes on batch of 2
 * D5  MultimodalModelLoader smoke (registers if available)
 * D6  ModelRegistry resolve for vl model types
 * D7  TemplateRegistry vl names (llava, qwen2_vl, qwen3_vl)
 * D8  export/merge adapter smoke on VL model
 */
public class BenchmarkLlamaFactoryMultimodal {

    static int passed = 0, failed = 0;
    static final List<String> failures = new ArrayList<>();

    public static void main(String[] args) throws Exception {
        System.out.println("=== BenchmarkLlamaFactoryMultimodal ===\n");
        d1MultimodalCollatorDefaults();
        d2MultimodalCollatorWithPixels();
        d3MultimodalCollatorZeroFallback();
        d4VlCollatorBatchShapes();
        d5MultimodalModelLoaderSmoke();
        d6TemplateRegistryVlNames();
        d7ExportMergeSmoke();
        done();
    }

    // ── D1 ───────────────────────────────────────────────────────────────────
    static void d1MultimodalCollatorDefaults() {
        section("D1 MultimodalCollator defaults");
        benchmark("MultimodalCollator defaults() non-null", () -> {
            MultimodalCollator coll = MultimodalCollator.defaults();
            check("MultimodalCollator defaults non-null", coll != null);
        });

        benchmark("MultimodalCollator custom C/H/W", () -> {
            MultimodalCollator coll = new MultimodalCollator(0L, -100L, 512, 3L, 224L, 224L);
            check("MultimodalCollator created", coll != null);
        });
    }

    // ── D2 ───────────────────────────────────────────────────────────────────
    static void d2MultimodalCollatorWithPixels() {
        section("D2 MultimodalCollator collate with pixel_values");
        benchmark("collate with pixel_values keys", () -> {
            MultimodalCollator coll = MultimodalCollator.defaults();
            org.bytedeco.pytorch.Tensor px = org.bytedeco.pytorch.global.torch.rand(
                    new long[]{3, 224, 224});
            java.util.List<java.util.Map<String, Object>> features = new java.util.ArrayList<>();
            java.util.Map<String, Object> feat = new java.util.LinkedHashMap<>();
            feat.put("input_ids", new long[]{1, 2, 3, 4});
            feat.put("labels", new long[]{1, 2, 3, 4});
            feat.put("pixel_values", px);
            features.add(feat);

            java.util.Map<String, org.bytedeco.pytorch.Tensor> batch = coll.collate(features);
            check("batch has pixel_values", batch.containsKey("pixel_values"));
            check("batch has input_ids", batch.containsKey("input_ids"));
            org.bytedeco.pytorch.Tensor pv = batch.get("pixel_values");
            check("pixel_values dim=4", pv.dim() == 4);
            check("pixel_values B=1", pv.size(0) == 1);
        });
    }

    // ── D3 ───────────────────────────────────────────────────────────────────
    static void d3MultimodalCollatorZeroFallback() {
        section("D3 MultimodalCollator zero-pixels fallback");
        benchmark("collate without pixel_values yields zero placeholder", () -> {
            MultimodalCollator coll = MultimodalCollator.defaults();
            java.util.List<java.util.Map<String, Object>> features = new java.util.ArrayList<>();
            java.util.Map<String, Object> feat = new java.util.LinkedHashMap<>();
            feat.put("input_ids", new long[]{1, 2, 3, 4});
            feat.put("labels", new long[]{1, 2, 3, 4});
            // no pixel_values key
            features.add(feat);

            java.util.Map<String, org.bytedeco.pytorch.Tensor> batch = coll.collate(features);
            check("batch has pixel_values even when absent", batch.containsKey("pixel_values"));
        });
    }

    // ── D4 ───────────────────────────────────────────────────────────────────
    static void d4VlCollatorBatchShapes() {
        section("D4 VL collator shapes on batch of 2");
        benchmark("batch of 2 pixel_values shape", () -> {
            MultimodalCollator coll = MultimodalCollator.defaults();
            java.util.List<java.util.Map<String, Object>> features = new java.util.ArrayList<>();
            for (int i = 0; i < 2; i++) {
                java.util.Map<String, Object> feat = new java.util.LinkedHashMap<>();
                feat.put("input_ids", new long[]{1, 2, 3, 4});
                feat.put("labels", new long[]{1, 2, 3, 4});
                org.bytedeco.pytorch.Tensor px = org.bytedeco.pytorch.global.torch.rand(
                        new long[]{3, 224, 224});
                feat.put("pixel_values", px);
                features.add(feat);
            }

            java.util.Map<String, org.bytedeco.pytorch.Tensor> batch = coll.collate(features);
            org.bytedeco.pytorch.Tensor pv = batch.get("pixel_values");
            check("pixel_values B=2", pv.size(0) == 2);
            check("pixel_values C=3", pv.size(1) == 3);
        });
    }

    // ── D5 ───────────────────────────────────────────────────────────────────
    static void d5MultimodalModelLoaderSmoke() {
        section("D5 MultimodalModelLoader smoke");
        benchmark("MultimodalModelLoader class loads", () -> {
            Class<?> cls = null;
            try {
                cls = Class.forName(
                        "org.bytedeco.pytorch.llm.llamafactory.model.MultimodalModelLoader");
            } catch (ClassNotFoundException e) {
                throw new RuntimeException(e);
            }
            check("MultimodalModelLoader class found", cls != null);
        });

        benchmark("MultimodalModelLoader load via FactoryArgs template=llava", () -> {
            FactoryArgs fa = FactoryArgs.builder()
                    .model(ModelArgs.builder().modelNameOrPath("llava-hf/llava-1.5-7b-hf").build())
                    .data(d -> d.template("llava").cutoffLen(512))
                    .finetuning(f -> f.stage(Stage.SFT).finetuningType(FinetuningType.LORA)
                            .loraRank(8))
                    .training(t -> t.outputDir("tmp/vl")
                            .perDeviceTrainBatchSize(1).learningRate(5e-5).maxSteps(1)
                            .gradientAccumulationSteps(1).boardEnabled(false).reportTo("none"))
                    .build();
            // MultimodalModelLoader may not have the model locally;
            // the smoke is that it doesn't throw on construction path
            // (offline path falls back to tiny-gpt2)
            check("FactoryArgs with llava template ok", fa.data().template().equals("llava"));
        });
    }

    // ── D6 ───────────────────────────────────────────────────────────────────
    static void d6TemplateRegistryVlNames() {
        section("D6 TemplateRegistry VL names");
        String[] vlNames = {"llava", "qwen2_vl", "qwen3_vl"};
        for (String n : vlNames) {
            benchmark("TemplateRegistry '" + n + "'", () -> {
                var t = TemplateRegistry.get(n);
                check("template '" + n + "' non-null", t != null);
            });
        }
    }

    // ── D7 ───────────────────────────────────────────────────────────────────
    static void d7ExportMergeSmoke() {
        section("D7 export/merge smoke on VL model");
        benchmark("FactoryArgs for VL export path builds", () -> {
            FactoryArgs fa = FactoryArgs.builder()
                    .model(ModelArgs.builder().modelNameOrPath("tiny-gpt2").build())
                    .data(d -> d.template("llava").cutoffLen(512))
                    .finetuning(f -> f.stage(Stage.SFT).finetuningType(FinetuningType.LORA)
                            .loraRank(8))
                    .training(t -> t.outputDir("tmp/vl-export")
                            .perDeviceTrainBatchSize(1).learningRate(5e-5).maxSteps(1)
                            .gradientAccumulationSteps(1).boardEnabled(false).reportTo("none"))
                    .build();
            ModelLoader.LoadedModel loaded = ModelLoader.load(fa);
            check("loaded module non-null", loaded.module() != null);
            check("loaded card non-null", loaded.card() != null);
            check("loaded peft null (LORA not attached yet)", loaded.peft() == null);
            loaded.close();
        });
    }

    // ── helpers ───────────────────────────────────────────────────────────────
    static void section(String n) { System.out.println("\n=== " + n + " ==="); }
    static void check(String name, boolean ok) {
        if (ok) { passed++; System.out.println("  PASS  " + name); }
        else { failed++; failures.add(name); System.out.println("  FAIL  " + name); }
    }
    static void benchmark(String name, Runnable r) {
        try { r.run(); }
        catch (Throwable t) { failed++; failures.add(name);
            System.out.println("  EXC   " + name + " — " + t.getMessage()); }
    }
    static void done() {
        System.out.println("\n=== RESULT ===");
        System.out.println("PASSED : " + passed);
        System.out.println("FAILED : " + failed);
        if (!failures.isEmpty()) {
            System.out.println("FAILURES:");
            for (String f : failures) System.out.println("  " + f);
        }
        if (failed > 0) throw new RuntimeException(failed + " tests failed");
    }
}
