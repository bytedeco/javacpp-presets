/*
 * BenchmarkLlamaFactoryE2E — full stage chain: SFT LoRA tiny → export → chat
 *
 * Run: java -cp ... distribute.BenchmarkLlamaFactoryE2E
 */
package distribute;

import org.bytedeco.pytorch.llm.llamafactory.FinetuneAdapter;
import org.bytedeco.pytorch.llm.llamafactory.DefaultFinetuneJob;
import org.bytedeco.pytorch.llm.llamafactory.LlamaFactory;
import org.bytedeco.pytorch.llm.llamafactory.chat.ChatEngine;
import org.bytedeco.pytorch.llm.llamafactory.eval.EvalResult;
import org.bytedeco.pytorch.llm.llamafactory.export.ModelExporter;
import org.bytedeco.pytorch.llm.llamafactory.hparams.*;
import org.bytedeco.pytorch.llm.llamafactory.model.ModelLoader;
import org.bytedeco.pytorch.llm.llamafactory.webui.BoardState;
import org.bytedeco.pytorch.llm.llamafactory.extras.monitor.MonitorBundle;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

/**
 * D1  LlamaFactory.open + train (SFT LoRA tiny, 2 steps) → globalStep > 0
 * D2  LlamaFactory.train(FactoryArgs) smoke
 * D3  FinetuneAdapter.export saves adapter + config
 * D4  FinetuneAdapter.chat returns ChatEngine
 * D5  ChatEngine.chat returns non-empty string
 * D6  LlamaFactory.eval(EvaluationArgs) returns EvalResult
 * D7  EvalResult.accuracy in [0,1]
 * D8  LlamaFactory.version() non-null
 * D9  BoardState attached to FinetuneAdapter via board()
 * D10 MonitorBundle board-only factory
 */
public class BenchmarkLlamaFactoryE2E {

    static int passed = 0, failed = 0;
    static final List<String> failures = new ArrayList<>();
    static Path tmpDir;

    public static void main(String[] args) throws Exception {
        tmpDir = Path.of("tmp/e2e-" + System.nanoTime());
        Files.createDirectories(tmpDir);
        System.out.println("=== BenchmarkLlamaFactoryE2E ===\n");
        System.out.println("tmpDir: " + tmpDir + "\n");
        d1FactoryOpenTrain();
        d2FactoryTrainSmoke();
        d3ExportSaves();
        d4ChatEngine();
        d5ChatOneReply();
        d6EvalResult();
        d7EvalAccuracy();
        d8FactoryVersion();
        d9BoardAttached();
        d10MonitorBundleBoardOnly();
        done();
    }

    static FactoryArgs tinyFactoryArgs(String outDir) {
        return FactoryArgs.builder()
                .model(ModelArgs.builder().modelNameOrPath("tiny-gpt2").build())
                .data(d -> d.dataset("alpaca_en_demo").cutoffLen(128).maxSamples(4))
                .finetuning(f -> f.stage(Stage.SFT).finetuningType(FinetuningType.LORA)
                        .loraRank(8))
                .training(t -> t.outputDir(outDir)
                        .perDeviceTrainBatchSize(1).learningRate(5e-5).maxSteps(2)
                        .gradientAccumulationSteps(1).saveSteps(100).loggingSteps(1)
                        .saveTotalLimit(2).boardEnabled(false).reportTo("none"))
                .build();
    }

    // ── D1 ───────────────────────────────────────────────────────────────────
    static void d1FactoryOpenTrain() {
        section("D1 LlamaFactory.open + train");
        benchmark("open + train → globalStep > 0", () -> {
            String out = tmpDir.resolve("d1").toString();
            FactoryArgs fa = tinyFactoryArgs(out);
            try (FinetuneAdapter job = LlamaFactory.open(fa)) {
                job.train();
                int gs = job.globalStep();
                check("globalStep > 0", gs > 0);
                check("lastMetrics non-empty", !job.lastMetrics().isEmpty());
            }
        });
    }

    // ── D2 ───────────────────────────────────────────────────────────────────
    static void d2FactoryTrainSmoke() {
        section("D2 LlamaFactory.train(FactoryArgs) smoke");
        benchmark("LlamaFactory.train no-throw", () -> {
            String out = tmpDir.resolve("d2").toString();
            FactoryArgs fa = tinyFactoryArgs(out);
            LlamaFactory.train(fa);
            check("train completed", true);
        });
    }

    // ── D3 ───────────────────────────────────────────────────────────────────
    static void d3ExportSaves() {
        section("D3 FinetuneAdapter.export");
        benchmark("export saves adapter directory", () -> {
            String out = tmpDir.resolve("d3").toString();
            FactoryArgs fa = tinyFactoryArgs(out);
            Path exportDir = tmpDir.resolve("export-d3");
            try (FinetuneAdapter job = LlamaFactory.open(fa)) {
                job.train();
                Path saved = job.export(exportDir, ExportArgs.builder()
                        .exportDir(exportDir.toString())
                        .mergeAdapters(false)
                        .build());
                check("export dir returned", saved != null);
                check("export dir exists", Files.exists(saved));
            }
        });

        benchmark("ModelExporter.export saves report", () -> {
            FactoryArgs fa = tinyFactoryArgs(tmpDir.resolve("d3b").toString());
            try (FinetuneAdapter job = LlamaFactory.open(fa)) {
                ModelLoader.LoadedModel loaded = ((DefaultFinetuneJob) job).loaded();
                Path exportDir = tmpDir.resolve("export-d3b");
                Path saved = null;
                try {
                    saved = ModelExporter.export(fa, loaded,
                            ExportArgs.builder().exportDir(exportDir.toString()).build());
                } catch (IOException e) {
                    throw new RuntimeException(e);
                }
                check("export dir exists", Files.exists(saved));
                check("export_report.json exists",
                        Files.exists(saved.resolve("export_report.json")));
            }
        });
    }

    // ── D4 ───────────────────────────────────────────────────────────────────
    static void d4ChatEngine() {
        section("D4 FinetuneAdapter.chat");
        benchmark("chat() returns ChatEngine", () -> {
            String out = tmpDir.resolve("d4").toString();
            FactoryArgs fa = tinyFactoryArgs(out);
            try (FinetuneAdapter job = LlamaFactory.open(fa)) {
                ChatEngine chat = job.chat();
                check("chat engine non-null", chat != null);
            }
        });
    }

    // ── D5 ───────────────────────────────────────────────────────────────────
    static void d5ChatOneReply() {
        section("D5 ChatEngine.chat returns non-empty string");
        benchmark("chat returns non-empty reply", () -> {
            String out = tmpDir.resolve("d5").toString();
            FactoryArgs fa = tinyFactoryArgs(out);
            try (FinetuneAdapter job = LlamaFactory.open(fa)) {
                ChatEngine chat = job.chat();
                String reply = chat.chat("Say hello in one word.");
                check("reply non-null", reply != null);
                check("reply non-empty", !reply.isEmpty());
                check("reply length > 0", reply.length() > 0);
            }
        });
    }

    // ── D6 ───────────────────────────────────────────────────────────────────
    static void d6EvalResult() {
        section("D6 LlamaFactory.eval");
        benchmark("eval returns EvalResult", () -> {
            EvaluationArgs ea = EvaluationArgs.builder()
                    .task("mmlu_test").lang("en").nShot(0)
                    .saveDir(tmpDir.resolve("eval-d6").toString())
                    .build();
            EvalResult r = LlamaFactory.eval(ea);
            check("eval result non-null", r != null);
            check("eval result task non-null", r.task() != null);
            check("eval result total >= 0", r.total() >= 0);
        });
    }

    // ── D7 ───────────────────────────────────────────────────────────────────
    static void d7EvalAccuracy() {
        section("D7 EvalResult.accuracy in [0,1]");
        benchmark("accuracy in [0,1]", () -> {
            EvaluationArgs ea = EvaluationArgs.builder()
                    .task("mmlu_test").lang("en").nShot(0).build();
            EvalResult r = LlamaFactory.eval(ea);
            double acc = r.accuracy();
            check("accuracy >= 0", acc >= 0.0);
            check("accuracy <= 1", acc <= 1.0);
        });
    }

    // ── D8 ───────────────────────────────────────────────────────────────────
    static void d8FactoryVersion() {
        section("D8 LlamaFactory.version()");
        benchmark("version non-null", () -> {
            String v = LlamaFactory.version();
            check("version non-null", v != null);
            check("version non-empty", !v.isEmpty());
            System.out.println("  INFO  factory version = " + v);
        });
    }

    // ── D9 ───────────────────────────────────────────────────────────────────
    static void d9BoardAttached() {
        section("D9 FinetuneAdapter.board()");
        benchmark("board() returns BoardState when enabled", () -> {
            FactoryArgs fa = FactoryArgs.builder()
                    .model(ModelArgs.builder().modelNameOrPath("tiny-gpt2").build())
                    .data(d -> d.dataset("alpaca_en_demo").cutoffLen(128).maxSamples(2))
                    .finetuning(f -> f.stage(Stage.SFT).finetuningType(FinetuningType.LORA)
                            .loraRank(8))
                    .training(t -> t.outputDir(tmpDir.resolve("d9").toString())
                            .perDeviceTrainBatchSize(1).learningRate(5e-5).maxSteps(1)
                            .gradientAccumulationSteps(1).saveSteps(100).loggingSteps(1)
                            .boardEnabled(true).reportTo("none"))
                    .build();
            try (FinetuneAdapter job = LlamaFactory.open(fa)) {
                BoardState b = job.board();
                check("board non-null when enabled", b != null);
                check("board status IDLE", b.status() == BoardState.Status.IDLE);
            }
        });
    }

    // ── D10 ───────────────────────────────────────────────────────────────────
    static void d10MonitorBundleBoardOnly() {
        section("D10 MonitorBundle board-only");
        benchmark("MonitorBundle.boardOnly", () -> {
            BoardState s = new BoardState();
            var bundle = MonitorBundle.boardOnly(s);
            check("boardOnly bundle non-null", bundle != null);
            check("boardOnly board() == s", bundle.board() == s);
            check("boardOnly lastMetrics non-null", bundle.lastMetrics() != null);
        });
    }

    // ── helpers ───────────────────────────────────────────────────────────────
    static void deleteRecursive(Path p) {
        if (!Files.exists(p)) return;
        try {
            Files.walk(p).sorted(java.util.Comparator.reverseOrder())
                    .forEach(ap -> { try { Files.deleteIfExists(ap); } catch (Exception ignored) {} });
        } catch (Exception ignored) {}
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
        // cleanup
        deleteRecursive(tmpDir);
        if (failed > 0) throw new RuntimeException(failed + " tests failed");
    }
}
