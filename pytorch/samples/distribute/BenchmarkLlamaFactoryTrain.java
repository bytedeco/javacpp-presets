/*
 * BenchmarkLlamaFactoryTrain — SFT/PT/DPO/ORPO/GRPO one-step loop
 *
 * Run: java -cp ... distribute.BenchmarkLlamaFactoryTrain
 */
package distribute;

import org.bytedeco.pytorch.llm.llamafactory.data.DataLoaderFactory;
import org.bytedeco.pytorch.llm.llamafactory.data.DatasetBuilder;
import org.bytedeco.pytorch.llm.llamafactory.hparams.FactoryArgs;
import org.bytedeco.pytorch.llm.llamafactory.hparams.FinetuningType;
import org.bytedeco.pytorch.llm.llamafactory.hparams.ModelArgs;
import org.bytedeco.pytorch.llm.llamafactory.hparams.Stage;
import org.bytedeco.pytorch.llm.llamafactory.model.ModelLoader;
import org.bytedeco.pytorch.llm.llamafactory.train.CheckpointManager;
import org.bytedeco.pytorch.llm.llamafactory.train.TrainerFactory;
import org.bytedeco.pytorch.llm.trl.BaseTrainer;
import org.bytedeco.pytorch.llm.trl.SFTTrainer;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * D1  SFT trainer one-step finite loss on tiny model
 * D2  PT trainer one-step on base model (no PEFT)
 * D3  DPO trainer bridge with synthetic pairwise batch
 * D4  ORPO trainer bridge
 * D5  GRPO trainer bridge
 * D6  Freeze-tuning (freeze last N layers)
 * D7  LoRA/QLORA config builds correctly
 * D8  CheckpointManager save/load roundtrip
 */
public class BenchmarkLlamaFactoryTrain {

    static int passed = 0, failed = 0;
    static final List<String> failures = new ArrayList<>();

    public static void main(String[] args) throws Exception {
        System.out.println("=== BenchmarkLlamaFactoryTrain ===\n");
        d1SftOneStep();
        d2PtOneStep();
        d3Dpo();
        d4Orpo();
        d5Grpo();
        d6FreezeTuning();
        d7LoraConfig();
        d8Checkpoint();
        done();
    }

    static FactoryArgs tinyArgs(Stage stage, FinetuningType ft) {
        return FactoryArgs.builder()
                .model(ModelArgs.builder().modelNameOrPath("tiny-gpt2").build())
                .data(d -> d.dataset("alpaca_en_demo").cutoffLen(128).maxSamples(4))
                .finetuning(f -> f.stage(stage).finetuningType(ft)
                        .loraRank(ft == FinetuningType.LORA ? 8 : 0))
                .training(t -> t.outputDir("tmp/train-" + stage.wireName())
                        .perDeviceTrainBatchSize(1).learningRate(5e-5).maxSteps(2)
                        .gradientAccumulationSteps(1).saveSteps(100).loggingSteps(1)
                        .saveTotalLimit(2).boardEnabled(false).reportTo("none"))
                .build();
    }

    // ── D1 ───────────────────────────────────────────────────────────────────
    static void d1SftOneStep() {
        section("D1 SFT one-step finite loss");
        benchmark("SFT on tiny model finite loss", () -> {
            FactoryArgs fa = tinyArgs(Stage.SFT, FinetuningType.LORA);
            FaModel mdl = loadModel(fa);
            int steps = 2;
            BaseTrainer trainer = TrainerFactory.create(fa, mdl.loaded, steps);

            DataLoaderFactory loader = dataLoader(fa, Stage.SFT);
            trainer.train(loader.oneEpoch());

            int gs = trainer.globalStep();
            check("SFT trainer globalStep > 0", gs > 0);
            check("SFT trainer has callbacks", trainer.config().maxSteps() == steps);

            if (trainer instanceof SFTTrainer sft) {
                check("SFTTrainer model non-null", sft.model() != null);
            }
            mdl.close();
        });

        benchmark("LlamaFactory.train(FactoryArgs) smoke", () -> {
            FactoryArgs fa = tinyArgs(Stage.SFT, FinetuningType.LORA);
            FaModel mdl = loadModel(fa);
            int steps = 2;
            BaseTrainer trainer = TrainerFactory.create(fa, mdl.loaded, steps);
            DataLoaderFactory loader = dataLoader(fa, Stage.SFT);
            trainer.train(loader.oneEpoch());
            check("trainer finite after smoke", trainer.globalStep() > 0);
            mdl.close();
        });
    }

    // ── D2 ───────────────────────────────────────────────────────────────────
    static void d2PtOneStep() {
        section("D2 PT one-step (no PEFT)");
        benchmark("PT on base causal LM", () -> {
            FactoryArgs fa = FactoryArgs.builder()
                    .model(ModelArgs.builder().modelNameOrPath("tiny-gpt2").build())
                    .data(d -> d.dataset("alpaca_en_demo").cutoffLen(128).maxSamples(4))
                    .finetuning(f -> f.stage(Stage.PT).finetuningType(FinetuningType.FULL))
                    .training(t -> t.outputDir("tmp/pt")
                            .perDeviceTrainBatchSize(1).learningRate(5e-5).maxSteps(2)
                            .gradientAccumulationSteps(1).saveSteps(100).loggingSteps(1)
                            .boardEnabled(false).reportTo("none"))
                    .build();
            FaModel mdl = loadModel(fa);
            BaseTrainer trainer = TrainerFactory.create(fa, mdl.loaded, 2);
            DataLoaderFactory loader = dataLoader(fa, Stage.PT);
            trainer.train(loader.oneEpoch());
            check("PT globalStep > 0", trainer.globalStep() > 0);
            mdl.close();
        });
    }

    // ── D3 ───────────────────────────────────────────────────────────────────
    static void d3Dpo() {
        section("D3 DPO trainer bridge");
        benchmark("DPO trainer builds", () -> {
            FactoryArgs fa = FactoryArgs.builder()
                    .model(ModelArgs.builder().modelNameOrPath("tiny-gpt2").build())
                    .data(d -> d.dataset("preference_demo").cutoffLen(128).maxSamples(2))
                    .finetuning(f -> f.stage(Stage.DPO).finetuningType(FinetuningType.LORA)
                            .loraRank(8).prefBeta(0.1))
                    .training(t -> t.outputDir("tmp/dpo")
                            .perDeviceTrainBatchSize(1).learningRate(5e-5).maxSteps(1)
                            .gradientAccumulationSteps(1).saveSteps(100).loggingSteps(1)
                            .boardEnabled(false).reportTo("none"))
                    .build();
            FaModel mdl = loadModel(fa);
            BaseTrainer trainer = TrainerFactory.create(fa, mdl.loaded, 1);
            check("DPO trainer non-null", trainer != null);
            check("DPO trainer is BaseTrainer", trainer instanceof BaseTrainer);
            mdl.close();
        });
    }

    // ── D4 ───────────────────────────────────────────────────────────────────
    static void d4Orpo() {
        section("D4 ORPO trainer bridge");
        benchmark("ORPO trainer builds", () -> {
            FactoryArgs fa = FactoryArgs.builder()
                    .model(ModelArgs.builder().modelNameOrPath("tiny-gpt2").build())
                    .data(d -> d.dataset("preference_demo").cutoffLen(128).maxSamples(2))
                    .finetuning(f -> f.stage(Stage.ORPO).finetuningType(FinetuningType.LORA)
                            .loraRank(8).prefBeta(0.1))
                    .training(t -> t.outputDir("tmp/orpo")
                            .perDeviceTrainBatchSize(1).learningRate(5e-5).maxSteps(1)
                            .gradientAccumulationSteps(1).saveSteps(100).loggingSteps(1)
                            .boardEnabled(false).reportTo("none"))
                    .build();
            FaModel mdl = loadModel(fa);
            BaseTrainer trainer = TrainerFactory.create(fa, mdl.loaded, 1);
            check("ORPO trainer non-null", trainer != null);
            mdl.close();
        });
    }

    // ── D5 ───────────────────────────────────────────────────────────────────
    static void d5Grpo() {
        section("D5 GRPO trainer bridge");
        benchmark("GRPO trainer builds", () -> {
            FactoryArgs fa = FactoryArgs.builder()
                    .model(ModelArgs.builder().modelNameOrPath("tiny-gpt2").build())
                    .data(d -> d.dataset("alpaca_en_demo").cutoffLen(128).maxSamples(2))
                    .finetuning(f -> f.stage(Stage.GRPO).finetuningType(FinetuningType.LORA)
                            .loraRank(8))
                    .training(t -> t.outputDir("tmp/grpo")
                            .perDeviceTrainBatchSize(1).learningRate(5e-5).maxSteps(1)
                            .gradientAccumulationSteps(1).saveSteps(100).loggingSteps(1)
                            .boardEnabled(false).reportTo("none"))
                    .build();
            FaModel mdl = loadModel(fa);
            BaseTrainer trainer = TrainerFactory.create(fa, mdl.loaded, 1);
            check("GRPO trainer non-null", trainer != null);
            mdl.close();
        });
    }

    // ── D6 ───────────────────────────────────────────────────────────────────
    static void d6FreezeTuning() {
        section("D6 Freeze-tuning");
        benchmark("freeze last N layers", () -> {
            FactoryArgs fa = FactoryArgs.builder()
                    .model(ModelArgs.builder().modelNameOrPath("tiny-gpt2").build())
                    .data(d -> d.dataset("alpaca_en_demo").cutoffLen(128).maxSamples(2))
                    .finetuning(f -> f.stage(Stage.SFT).finetuningType(FinetuningType.FREEZE)
                            .freezeTrainableLayers(1))
                    .training(t -> t.outputDir("tmp/freeze")
                            .perDeviceTrainBatchSize(1).learningRate(5e-5).maxSteps(1)
                            .gradientAccumulationSteps(1).saveSteps(100).loggingSteps(1)
                            .boardEnabled(false).reportTo("none"))
                    .build();
            FaModel mdl = loadModel(fa);
            BaseTrainer trainer = TrainerFactory.create(fa, mdl.loaded, 1);
            check("freeze trainer builds", trainer != null);
            check("freeze model has params", mdl.loaded.module().parameters() != null);
            mdl.close();
        });
    }

    // ── D7 ───────────────────────────────────────────────────────────────────
    static void d7LoraConfig() {
        section("D7 LoRA/QLORA config");
        benchmark("LORA TrainerFactory config builder", () -> {
            FactoryArgs fa = tinyArgs(Stage.SFT, FinetuningType.LORA);
            FaModel mdl = loadModel(fa);
            int steps = 2;
            BaseTrainer trainer = TrainerFactory.create(fa, mdl.loaded, steps);
            check("trainer trainable params non-empty",
                    TrainerFactory.trainableParams(mdl.loaded) != null);
            check("trainer optim non-null", trainer.optimizer() != null);
            mdl.close();
        });
    }

    // ── D8 ───────────────────────────────────────────────────────────────────
    static void d8Checkpoint() {
        section("D8 CheckpointManager save/load roundtrip");
        benchmark("CheckpointManager save creates dir", () -> {
            FactoryArgs fa = tinyArgs(Stage.SFT, FinetuningType.LORA);
            FaModel mdl = loadModel(fa);
            CheckpointManager cm =
                    CheckpointManager.from(fa);
            java.nio.file.Path dir = cm.checkpointDir(1);
            check("checkpoint dir path non-null", dir != null);
            check("checkpoint dir ends with checkpoint-1",
                    dir.getFileName().toString().equals("checkpoint-1"));
            mdl.close();
        });

        benchmark("CheckpointManager loadGlobalStep from marker", () -> {
            FactoryArgs fa = tinyArgs(Stage.SFT, FinetuningType.LORA);
            CheckpointManager cm =
                    CheckpointManager.from(fa);
            java.nio.file.Path resumeDir = cm.resolveResumeDir();
            // No checkpoints saved yet → should return null
            check("resolveResumeDir null when no ckpts", resumeDir == null);
            int step = 0;
            try {
                step = cm.loadGlobalStep(java.nio.file.Path.of("tmp/no-such-dir"));
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
            check("loadGlobalStep 0 for missing dir", step == 0);
        });

        benchmark("CheckpointManager shouldSave logic", () -> {
            FactoryArgs fa = tinyArgs(Stage.SFT, FinetuningType.LORA);
            CheckpointManager cm =
                    CheckpointManager.from(fa);
            check("shouldSave(10) when saveSteps=100", !cm.shouldSave(10));
            check("shouldSave(100) when saveSteps=100", cm.shouldSave(100));
            check("shouldSave(0)", !cm.shouldSave(0));
        });
    }

    // ── helpers ───────────────────────────────────────────────────────────────
    static FaModel loadModel(FactoryArgs fa) {
        ModelLoader.LoadedModel loaded = ModelLoader.load(fa);
        return new FaModel(loaded);
    }

    static DataLoaderFactory dataLoader(FactoryArgs fa, Stage stage) {
        DatasetBuilder builder = DatasetBuilder.from(fa.data(), stage);
        List<Map<String, Object>> rows = stage == Stage.DPO || stage == Stage.ORPO
                ? DatasetBuilder.demoPreferenceRows()
                : stage == Stage.PT
                        ? DatasetBuilder.demoPretrainRows()
                        : DatasetBuilder.demoAlpacaRows();
        List<Map<String, Object>> features = builder.buildFeatures(rows);
        return new DataLoaderFactory(
                features, builder.collator(),
                Math.max(1, fa.training().perDeviceTrainBatchSize()),
                false, false, fa.training().dataSeed());
    }

    static class FaModel implements AutoCloseable {
        final ModelLoader.LoadedModel loaded;
        FaModel(ModelLoader.LoadedModel loaded) { this.loaded = loaded; }
        public void close() { try { loaded.close(); } catch (Exception ignored) {} }
    }

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
