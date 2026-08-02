/*
 * BenchmarkLlamaFactoryCore — factory core plane: hparams validate / data template collator
 *
 * Run: java -cp ... distribute.BenchmarkLlamaFactoryCore
 */
package distribute;

import org.bytedeco.pytorch.llm.llamafactory.data.DataLoaderFactory;
import org.bytedeco.pytorch.llm.llamafactory.data.DatasetBuilder;
import org.bytedeco.pytorch.llm.llamafactory.data.template.Template;
import org.bytedeco.pytorch.llm.llamafactory.data.template.TemplateRegistry;
import org.bytedeco.pytorch.llm.llamafactory.hparams.DataArgs;
import org.bytedeco.pytorch.llm.llamafactory.hparams.FactoryArgs;
import org.bytedeco.pytorch.llm.llamafactory.hparams.FinetuningType;
import org.bytedeco.pytorch.llm.llamafactory.hparams.ModelArgs;
import org.bytedeco.pytorch.llm.llamafactory.hparams.QuantizationMethod;
import org.bytedeco.pytorch.llm.llamafactory.hparams.Stage;
import org.bytedeco.pytorch.llm.llamafactory.train.ParallelLauncher;
import org.bytedeco.pytorch.llm.llamafactory.data.collator.KtoCollator;
import org.bytedeco.pytorch.llm.llamafactory.data.collator.PairwiseCollator;
import org.bytedeco.pytorch.llm.llamafactory.data.collator.SupervisedCollator;

import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Map;

/**
 * D1  hparams validate
 * D2  Templates / collators (alpaca / sharegpt / llama3 / qwen / empty)
 * D3  SupervisedCollator → collated Map<String,Tensor> batch with labels
 * D4  PairwiseCollator   → chosen/rejected batch
 * D5  DatasetBuilder → features from demo rows
 * D6  DataLoaderFactory → BatchSupplier cycling
 * D7  ParallelLauncher resolve (single / DDP / FSDP / DeepSpeed)
 */
public class BenchmarkLlamaFactoryCore {

    static int passed = 0, failed = 0;
    static final List<String> failures = new ArrayList<>();

    public static void main(String[] args) throws Exception {
        System.out.println("=== BenchmarkLlamaFactoryCore ===\n");
        d1HparamsValidate();
        d2Templates();
        d3SupervisedCollator();
        d4PairwiseCollator();
        d5DatasetBuilder();
        d6DataLoaderFactory();
        d7ParallelLauncher();
        done();
    }

    // ── D1 ───────────────────────────────────────────────────────────────────
    static void d1HparamsValidate() {
        section("D1 hparams validate");
        benchmark("defaults build", () -> {
            FactoryArgs fa = FactoryArgs.builder()
                    .model(ModelArgs.builder().modelNameOrPath("tiny-gpt2").build())
                    .data(d -> d.dataset("alpaca_en_demo").maxSamples(4).cutoffLen(256))
                    .finetuning(f -> f.stage(Stage.SFT).finetuningType(FinetuningType.LORA).loraRank(8))
                    .training(t -> t.outputDir("tmp/core-d1")
                            .perDeviceTrainBatchSize(1).learningRate(5e-5).maxSteps(2)
                            .gradientAccumulationSteps(1).saveSteps(100).loggingSteps(1))
                    .build();
            check("defaults ok", fa.model().modelNameOrPath().equals("tiny-gpt2"));
            check("SFT stage", fa.finetuning().stage() == Stage.SFT);
            check("LORA type", fa.finetuning().finetuningType() == FinetuningType.LORA);
            check("loraRank=8", fa.finetuning().loraRank() == 8);
        });

        benchmark("QLORA requires quant", () -> {
            try {
                FactoryArgs fa = FactoryArgs.builder()
                        .model(m -> m.modelNameOrPath("tiny-gpt2")
                                .quantizationMethod(QuantizationMethod.BNB))
                        .data(DataArgs.builder().cutoffLen(256).build())
                        .finetuning(f -> f.stage(Stage.SFT).finetuningType(FinetuningType.QLORA).loraRank(8))
                        .training(t -> t.outputDir("tmp/d1-qlora")
                                .perDeviceTrainBatchSize(1).learningRate(5e-5).maxSteps(1)
                                .gradientAccumulationSteps(1))
                        .build();
                fa.validate();
                check("QLORA+BnB ok", true);
            } catch (IllegalArgumentException e) {
                check("QLORA+BnB throws", e.getMessage().contains("quantization"));
            }
        });

        benchmark("useDora+useOft mutually exclusive", () -> {
            try {
                FactoryArgs fa = FactoryArgs.builder()
                        .model(ModelArgs.builder().modelNameOrPath("tiny-gpt2").build())
                        .data(DataArgs.builder().cutoffLen(256).build())
                        .finetuning(f -> f.stage(Stage.SFT).finetuningType(FinetuningType.LORA)
                                .loraRank(8).useDora(true).useOft(true))
                        .training(t -> t.outputDir("tmp/d1-doraoft")
                                .perDeviceTrainBatchSize(1).learningRate(5e-5).maxSteps(1)
                                .gradientAccumulationSteps(1))
                        .build();
                fa.validate();
                check("dora+oft throws", false);
            } catch (IllegalArgumentException e) {
                check("dora+oft throws", e.getMessage().toLowerCase(Locale.ROOT).contains("dora"));
            }
        });

        benchmark("loraRank > 0 for LORA", () -> {
            try {
                FactoryArgs fa = FactoryArgs.builder()
                        .model(ModelArgs.builder().modelNameOrPath("tiny-gpt2").build())
                        .data(DataArgs.builder().cutoffLen(256).build())
                        .finetuning(f -> f.stage(Stage.SFT).finetuningType(FinetuningType.LORA).loraRank(0))
                        .training(t -> t.outputDir("tmp/d1-lora0")
                                .perDeviceTrainBatchSize(1).learningRate(5e-5).maxSteps(1)
                                .gradientAccumulationSteps(1))
                        .build();
                fa.validate();
                check("loraRank=0 throws", false);
            } catch (IllegalArgumentException e) {
                check("loraRank=0 throws", true);
            }
        });

        benchmark("flat map parse", () -> {
            // Map.of supports ≤10 pairs; use LinkedHashMap for 11+
            Map<String, Object> flat = new java.util.LinkedHashMap<>();
            flat.put("model_name_or_path", "tiny-gpt2");
            flat.put("stage", "sft");
            flat.put("finetuning_type", "lora");
            flat.put("lora_rank", 8);
            flat.put("dataset", "alpaca_en_demo");
            flat.put("cutoff_len", 256);
            flat.put("per_device_train_batch_size", 1);
            flat.put("learning_rate", 5e-5);
            flat.put("max_steps", 2);
            flat.put("output_dir", "tmp/d1-flatmap");
            flat.put("report_to", "none");
            FactoryArgs fa = FactoryArgs.parse(flat);
            check("flat map stage", fa.finetuning().stage() == Stage.SFT);
            check("flat map loraRank", fa.finetuning().loraRank() == 8);
            check("flat map dataset", fa.data().dataset().equals("alpaca_en_demo"));
        });

        benchmark("toFlatMap roundtrip", () -> {
            FactoryArgs fa = FactoryArgs.builder()
                    .model(m -> m.modelNameOrPath("tiny-gpt2").quantizationMethod(QuantizationMethod.NONE))
                    .data(d -> d.dataset("alpaca_en_demo").cutoffLen(512))
                    .finetuning(f -> f.stage(Stage.SFT).finetuningType(FinetuningType.LORA).loraRank(16))
                    .training(t -> t.outputDir("tmp/d1-rt")
                            .perDeviceTrainBatchSize(2).learningRate(1e-4).maxSteps(3)
                            .gradientAccumulationSteps(2).saveSteps(100).loggingSteps(1))
                    .build();
            Map<String, Object> flat = fa.toFlatMap();
            FactoryArgs re = FactoryArgs.parse(flat);
            check("roundtrip stage", re.finetuning().stage() == Stage.SFT);
            check("roundtrip loraRank", re.finetuning().loraRank() == 16);
            check("roundtrip cutoff", re.data().cutoffLen() == 512);
        });
    }

    // ── D2 ───────────────────────────────────────────────────────────────────
    static void d2Templates() {
        section("D2 Templates");
        benchmark("TemplateRegistry defaults", () -> {
            Template t0 = TemplateRegistry.getOrDefault("unknown");
            check("unknown → default", t0 != null);
            check("empty name → default", TemplateRegistry.getOrDefault(null) != null);
            check("empty name → default", TemplateRegistry.getOrDefault("") != null);
        });

        benchmark("TemplateRegistry registered names", () -> {
            String[] names = {"alpaca", "sharegpt", "llama3", "qwen", "chatml", "glm4", "empty", "default"};
            for (String n : names) {
                Template t = TemplateRegistry.get(n);
                check("template '" + n + "'", t != null);
            }
        });

        benchmark("Template encodePrompt / encodeOneline", () -> {
            Template t = TemplateRegistry.get("alpaca");
            List<Template.Message> msgs = List.of(
                    Template.Message.user("What is 2+2?"),
                    Template.Message.assistant("4"));
            String oneline = t.encodeOneline(msgs);
            check("encodeOneline non-empty", oneline != null && !oneline.isEmpty());
            check("encodeOneline contains user text", oneline.contains("2+2"));
            String prompt = t.encodePrompt(msgs, null);
            check("encodePrompt non-empty", prompt != null && !prompt.isEmpty());
        });

        benchmark("System override in prompt", () -> {
            Template t = TemplateRegistry.get("default");
            List<Template.Message> msgs = List.of(Template.Message.user("hello"));
            String withSys = t.encodePrompt(msgs, "You are helpful");
            check("system override prompt", withSys != null && withSys.length() > 0);
        });
    }

    // ── D3 ───────────────────────────────────────────────────────────────────
    static void d3SupervisedCollator() {
        section("D3 SupervisedCollator");
        benchmark("collate 4 supervised features", () -> {
            DatasetBuilder builder = DatasetBuilder.from(
                    DataArgs.builder().dataset("alpaca_en_demo").cutoffLen(128).build(),
                    Stage.SFT);
            List<Map<String, Object>> features = builder.buildFeatures(DatasetBuilder.demoAlpacaRows());
            check("features non-empty", features.size() > 0);
            check("features <= 4", features.size() <= 4);
            Map<String, org.bytedeco.pytorch.Tensor> batch = builder.collate(features);
            check("batch has input_ids", batch.containsKey("input_ids"));
            check("batch has labels", batch.containsKey("labels"));
            check("batch has attention_mask", batch.containsKey("attention_mask"));
            org.bytedeco.pytorch.Tensor ids = batch.get("input_ids");
            check("input_ids rank=2", ids.dim() == 2);
            check("input_ids dtype long", ids.dtype().toString().contains("Long"));
            org.bytedeco.pytorch.Tensor lbls = batch.get("labels");
            check("labels rank=2", lbls.dim() == 2);
            check("labels same seq as ids", lbls.size(1) == ids.size(1));
        });
    }

    // ── D4 ───────────────────────────────────────────────────────────────────
    static void d4PairwiseCollator() {
        section("D4 PairwiseCollator");
        benchmark("collate 2 pairwise features", () -> {
            DatasetBuilder builder = DatasetBuilder.from(
                    DataArgs.builder().dataset("preference_demo").cutoffLen(128).build(),
                    Stage.DPO);
            List<Map<String, Object>> features = builder.buildFeatures(DatasetBuilder.demoPreferenceRows());
            check("pairwise features", features.size() > 0);
            Map<String, org.bytedeco.pytorch.Tensor> batch = builder.collate(features);
            check("batch has chosen_input_ids", batch.containsKey("chosen_input_ids"));
            check("batch has rejected_input_ids", batch.containsKey("rejected_input_ids"));
            check("batch has chosen_labels", batch.containsKey("chosen_labels"));
            check("batch has rejected_labels", batch.containsKey("rejected_labels"));
        });

        benchmark("KTO collator", () -> {
            DatasetBuilder builder = DatasetBuilder.from(
                    DataArgs.builder().cutoffLen(128).build(),
                    Stage.KTO);
            List<Map<String, Object>> features = builder.buildFeatures(DatasetBuilder.demoKtoRows());
            check("kto features", features.size() > 0);
            Map<String, org.bytedeco.pytorch.Tensor> batch = builder.collate(features);
            check("kto has input_ids", batch.containsKey("input_ids"));
        });
    }

    // ── D5 ───────────────────────────────────────────────────────────────────
    static void d5DatasetBuilder() {
        section("D5 DatasetBuilder");
        benchmark("PT pretrain rows", () -> {
            DatasetBuilder builder = DatasetBuilder.from(
                    DataArgs.builder().cutoffLen(128).build(),
                    Stage.PT);
            List<Map<String, Object>> features = builder.buildFeatures(DatasetBuilder.demoPretrainRows());
            check("pt features non-empty", features.size() > 0);
            check("pt features same count", features.size() == DatasetBuilder.demoPretrainRows().size());
            check("PT collator returns batch", builder.collate(features) != null);
        });

        benchmark("SFT alpaca rows", () -> {
            DatasetBuilder builder = DatasetBuilder.from(
                    DataArgs.builder().cutoffLen(128).build(),
                    Stage.SFT);
            List<Map<String, Object>> features = builder.buildFeatures(DatasetBuilder.demoAlpacaRows());
            check("sft features", features.size() > 0);
        });

        benchmark("DatasetBuilder collator() selection", () -> {
            DatasetBuilder sft = DatasetBuilder.from(
                    DataArgs.builder().cutoffLen(128).build(), Stage.SFT);
            DatasetBuilder dpo = DatasetBuilder.from(
                    DataArgs.builder().cutoffLen(128).build(), Stage.DPO);
            DatasetBuilder kto = DatasetBuilder.from(
                    DataArgs.builder().cutoffLen(128).build(), Stage.KTO);
            check("SFT has SupervisedCollator",
                    sft.collator() instanceof SupervisedCollator);
            check("DPO has PairwiseCollator",
                    dpo.collator() instanceof PairwiseCollator);
            check("KTO has KtoCollator",
                    kto.collator() instanceof KtoCollator);
        });
    }

    // ── D6 ───────────────────────────────────────────────────────────────────
    static void d6DataLoaderFactory() {
        section("D6 DataLoaderFactory");
        benchmark("cycling supplier", () -> {
            DatasetBuilder builder = DatasetBuilder.from(
                    DataArgs.builder().cutoffLen(128).build(),
                    Stage.SFT);
            List<Map<String, Object>> features = builder.buildFeatures(DatasetBuilder.demoAlpacaRows());
            DataLoaderFactory loader = new DataLoaderFactory(
                    features, builder.collator(), 2, true, false, 42L);
            check("batches per epoch", loader.batchesPerEpoch() > 0);
            org.bytedeco.pytorch.llm.trl.BaseTrainer.BatchSupplier s = loader.cycling(3);
            org.bytedeco.pytorch.llm.trl.BaseTrainer.BatchSupplier finalS = s;
            int count = 0;
            while (finalS.next() != null && count < 10) count++;
            check("cycling yields 3 batches", count == 3);
        });

        benchmark("cancellable stops", () -> {
            java.util.concurrent.atomic.AtomicBoolean stop = new java.util.concurrent.atomic.AtomicBoolean(false);
            DatasetBuilder builder = DatasetBuilder.from(
                    DataArgs.builder().cutoffLen(128).build(), Stage.SFT);
            List<Map<String, Object>> features = builder.buildFeatures(DatasetBuilder.demoAlpacaRows());
            DataLoaderFactory loader = new DataLoaderFactory(
                    features, builder.collator(), 1, false, false, 0L);
            org.bytedeco.pytorch.llm.trl.BaseTrainer.BatchSupplier inner = loader.cycling(100);
            org.bytedeco.pytorch.llm.trl.BaseTrainer.BatchSupplier gated =
                    DataLoaderFactory.cancellable(inner, stop);
            gated.next();
            stop.set(true);
            // next call should return null immediately
            boolean gotNull = (gated.next() == null);
            check("cancellable returns null after stop", gotNull);
        });
    }

    // ── D7 ───────────────────────────────────────────────────────────────────
    static void d7ParallelLauncher() {
        section("D7 ParallelLauncher");
        benchmark("single resolve", () -> {
            FactoryArgs fa = FactoryArgs.builder()
                    .model(ModelArgs.builder().modelNameOrPath("tiny-gpt2").build())
                    .data(DataArgs.builder().cutoffLen(128).build())
                    .finetuning(f -> f.stage(Stage.SFT).finetuningType(FinetuningType.LORA).loraRank(8))
                    .training(t -> t.outputDir("tmp/d7-single")
                            .perDeviceTrainBatchSize(1).learningRate(5e-5).maxSteps(1)
                            .gradientAccumulationSteps(1))
                    .build();
            ParallelLauncher.Plan plan = ParallelLauncher.resolve(fa);
            check("single backend", plan.backend() == ParallelLauncher.Backend.SINGLE);
            check("world=1", plan.worldSize() == 1);
            check("not distributed", !plan.distributed());
            check("isMain", ParallelLauncher.isMain(plan));
        });

        benchmark("plan toMap", () -> {
            FactoryArgs fa = FactoryArgs.builder()
                    .model(ModelArgs.builder().modelNameOrPath("tiny-gpt2").build())
                    .data(DataArgs.builder().cutoffLen(128).build())
                    .finetuning(f -> f.stage(Stage.SFT).finetuningType(FinetuningType.LORA).loraRank(8))
                    .training(t -> t.outputDir("tmp/d7-map")
                            .perDeviceTrainBatchSize(1).learningRate(5e-5).maxSteps(1)
                            .gradientAccumulationSteps(1).fsdp(false))
                    .build();
            ParallelLauncher.Plan plan = ParallelLauncher.resolve(fa);
            Map<String, Object> m = plan.toMap();
            check("toMap has backend", m.containsKey("backend"));
            check("toMap has world_size", m.containsKey("world_size"));
        });
    }

    // ── helpers ───────────────────────────────────────────────────────────────
    static void section(String name) { System.out.println("\n=== " + name + " ==="); }

    static void check(String name, boolean ok) {
        if (ok) { passed++; System.out.println("  PASS  " + name); }
        else { failed++; failures.add(name); System.out.println("  FAIL  " + name); }
    }

    static void checkFinite(String name, double v) {
        check(name + "=" + String.format(Locale.US, "%.4g", v), !Double.isNaN(v) && !Double.isInfinite(v));
    }

    static void benchmark(String name, Runnable r) {
        try {
            r.run();
        } catch (Throwable t) {
            failed++; failures.add(name);
            System.out.println("  EXC   " + name + " — " + t.getMessage());
        }
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
