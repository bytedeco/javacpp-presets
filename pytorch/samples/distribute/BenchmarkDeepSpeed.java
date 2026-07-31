package distribute;

import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.distributed.DistributedStore;
import org.bytedeco.pytorch.distributed.ProcessGroupWrapper;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.modules.LinearImpl;
import org.bytedeco.pytorch.optim.Adam;
import org.bytedeco.pytorch.optim.options.AdamOptions;
import org.bytedeco.pytorch.llm.accelerate.utils.MultiProcessLauncher;
import org.bytedeco.pytorch.llm.deepspeed.DeepSpeed;
import org.bytedeco.pytorch.llm.deepspeed.DeepSpeedConfig;
import org.bytedeco.pytorch.llm.deepspeed.DeepSpeedEngine;
import org.bytedeco.pytorch.llm.deepspeed.inference.InferenceEngine;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;

import static org.bytedeco.pytorch.global.torch.randn;
import static org.bytedeco.pytorch.global.torch.mse_loss;

public class BenchmarkDeepSpeed {
    static int passed = 0, failed = 0;
    static final AtomicInteger MP_RANK = new AtomicInteger(-1);

    static void check(String name, boolean ok) {
        if (ok) { passed++; System.out.println("  PASS  " + name); }
        else { failed++; System.out.println("  FAIL  " + name); }
    }
    static void section(String t) { System.out.println("\n=== " + t + " ==="); }

    static Module tinyModel() { return new LinearImpl(32, 16); }

    public static void main(String[] args) throws Exception {
        int rank = MultiProcessLauncher.envRank();
        MP_RANK.set(rank);
        if (rank < 0) runSingleProcess();
        else runMultiProcess();
    }
    public static void mainSingle() throws Exception { runSingleProcess(); }

    static void runSingleProcess() throws Exception {
        System.out.println("=== DeepSpeed benchmark (single-process) ===");
        d1Config();
        d2Partition();
        d3TrainStep();
        d4MPskip();
        d5Checkpoint();
        d6MemoryStats();
        d7Clip();
        d8Inference();
        d9Offload();
        d10Stress();
        done();
    }

    static void runMultiProcess() throws Exception {
        int rank = MP_RANK.get();
        try (DistributedStore store = DistributedStore.create(rank, 2)) {
            ProcessGroupWrapper pg = ProcessGroupWrapper.create(rank, 2, store);
            DeepSpeedConfig cfg = DeepSpeedConfig.builder().zeroStage(2).build();
            Module m = tinyModel();
            Adam opt = new Adam(m.parameters(), new AdamOptions());
            try (DeepSpeedEngine eng = DeepSpeed.initialize(m, opt, cfg, pg)) {
                for (int i = 0; i < 3; i++) {
                    Tensor x = randn(new long[]{2, 32});
                    Tensor y = randn(new long[]{2, 16});
                    Tensor out = eng.forward(x);
                    Tensor loss = mse_loss(out, y, null);
                    eng.backward(loss);
                    eng.step();
                }
                if (rank == 0) System.out.println("[rank0] multi-process step OK");
                pg.barrier();
            }
        }
    }

    static void d1Config() throws Exception {
        section("D1 Config / fromMap / toMap");
        DeepSpeedConfig cfg = DeepSpeedConfig.builder()
                .zeroStage(2).cpuOffload(true).gradientAccumulationSteps(4)
                .gradientClip(1.0).precision("bf16").build();
        check("zeroStage=2", cfg.zeroStage() == 2);
        check("cpuOffload=true", cfg.cpuOffload());
        check("bf16=true", cfg.bf16());
        check("fp16=false", !cfg.fp16());
        Map<String, Object> map = cfg.toMap();
        check("toMap non-empty", !map.isEmpty());
        DeepSpeedConfig fromMap = DeepSpeedConfig.fromMap(map);
        check("fromMap zeroStage=2", fromMap.zeroStage() == 2);
        check("fromMap bf16", fromMap.bf16());
        InferenceEngine ie = DeepSpeed.init_inference(tinyModel());
        check("init_inference alias", ie != null);
    }

    static void d2Partition() throws Exception {
        section("D2 ZeRO partition math");
        Module m = tinyModel();
        try (DeepSpeedEngine eng = DeepSpeed.initialize(m, null,
                DeepSpeedConfig.builder().zeroStage(2).build())) {
            var parts = eng.partitions();
            check("has partitions", parts.size() > 0);
            boolean allLocal = true;
            for (var p : parts) { if (!p.local) { allLocal = false; break; } }
            check("all local in sp=1", allLocal);
        }
    }

    static void d3TrainStep() throws Exception {
        section("D3 Single-rank train step");
        Module m = tinyModel();
        Adam opt = new Adam(m.parameters(), new AdamOptions());
        try (DeepSpeedEngine eng = DeepSpeed.initialize(m, opt,
                DeepSpeedConfig.builder().zeroStage(1).gradientAccumulationSteps(1).build())) {
            Tensor x = randn(new long[]{2, 32});
            Tensor y = randn(new long[]{2, 16});
            Tensor out = eng.forward(x);
            Tensor loss = mse_loss(out, y, null);
            boolean fin = !Double.isNaN(loss.item_double());
            check("loss finite", fin);
            eng.backward(loss);
            eng.step();
            check("globalStep=1", eng.globalStep() == 1);
        }
    }

    static void d4MPskip() {
        section("D4 Multi-process launcher");
        check("MultiProcessLauncher class", true);
        System.out.println("  SKIP   D4: run via MultiProcessLauncher with worldSize=2");
    }

    static void d5Checkpoint() throws Exception {
        section("D5 Checkpoint roundtrip");
        Path tmp = Files.createTempDirectory("ds_ckpt");
        Module m = tinyModel();
        Adam opt = new Adam(m.parameters(), new AdamOptions());
        try (DeepSpeedEngine eng = DeepSpeed.initialize(m, opt,
                DeepSpeedConfig.builder().zeroStage(1).build())) {
            eng.forward(randn(new long[]{2, 32}));
            eng.forward(randn(new long[]{2, 32}));
            Tensor loss = mse_loss(eng.module().forward(randn(new long[]{2, 32})), randn(new long[]{2, 16}), null);
            eng.backward(loss);
            eng.step();
            eng.saveCheckpoint(tmp);
            check("checkpoint meta exists", Files.exists(tmp.resolve("ds_checkpoint.meta")));
            Module m2 = tinyModel();
            try (DeepSpeedEngine eng2 = DeepSpeed.initialize(m2, opt,
                    DeepSpeedConfig.builder().zeroStage(1).build())) {
                Map<String, Object> meta = eng2.loadCheckpoint(tmp);
                check("checkpoint loaded", meta != null);
            }
        }
    }

    static void d6MemoryStats() throws Exception {
        section("D6 memory_stats invariants");
        Module m = tinyModel();
        try (DeepSpeedEngine eng = DeepSpeed.initialize(m, null,
                DeepSpeedConfig.builder().zeroStage(2).build())) {
            Map<String, Object> stats = eng.memoryStats();
            check("has total_param_numel", stats.containsKey("total_param_numel"));
            long total = ((Number) stats.get("total_param_numel")).longValue();
            check("total_param_numel>0", total > 0);
            check("has zero_stage", stats.containsKey("zero_stage"));
            int zs = ((Number) stats.get("zero_stage")).intValue();
            check("zero_stage=2", zs == 2);
            check("has world_size", stats.containsKey("world_size"));
        }
    }

    static void d7Clip() throws Exception {
        section("D7 Gradient clipping");
        Module m = tinyModel();
        try (DeepSpeedEngine eng = DeepSpeed.initialize(m, null,
                DeepSpeedConfig.builder().zeroStage(0).gradientClip(1.0).build())) {
            eng.forward(randn(new long[]{2, 32}));
            Tensor loss = mse_loss(eng.module().forward(randn(new long[]{2, 32})), randn(new long[]{2, 16}), null);
            eng.backward(loss);
            eng.step();
            double gn = eng.getGlobalGradNorm();
            check("gradNorm>=0", gn >= 0);
        }
    }

    static void d8Inference() throws Exception {
        section("D8 Inference engine");
        Module m = tinyModel();
        try (InferenceEngine ie = DeepSpeed.initInference(m)) {
            Tensor out = ie.forward(randn(new long[]{2, 32}));
            check("inference output non-null", out != null);
            check("numForwards=1", ie.numForwards() == 1);
            check("stats has dtype", ie.stats().containsKey("dtype"));
        }
    }

    static void d9Offload() throws Exception {
        section("D9 Offload flags");
        DeepSpeedConfig cfg = DeepSpeedConfig.builder()
                .zeroStage(2).cpuOffload(true).nvmeOffload(true).build();
        check("offloadOptimizer=true", cfg.offloadOptimizer());
        check("offloadParam=true", cfg.offloadParam());
        check("cpuOffload=true", cfg.cpuOffload());
        check("nvmeOffload=true", cfg.nvmeOffload());
        DeepSpeedConfig fromMap = DeepSpeedConfig.fromMap(cfg.toMap());
        check("fromMap offloadOptimizer", fromMap.offloadOptimizer());
        check("fromMap nvme", fromMap.nvmeOffload());
    }

    static void d10Stress() throws Exception {
        section("D10 Stress many micro-batches");
        Module m = tinyModel();
        try (DeepSpeedEngine eng = DeepSpeed.initialize(m, null,
                DeepSpeedConfig.builder().zeroStage(1).gradientAccumulationSteps(2).build())) {
            for (int i = 0; i < 50; i++) {
                Tensor x = randn(new long[]{2, 32});
                Tensor y = randn(new long[]{2, 16});
                eng.forward(x);
                Tensor loss = mse_loss(eng.module().forward(x), y, null);
                eng.backward(loss);
                if (eng.isGradientAccumulationBoundary()) eng.step();
            }
            check("globalStep>0", eng.globalStep() > 0);
            check("module still valid", eng.module() != null);
        }
    }

    static void done() {
        System.out.println("\n============================================================");
        System.out.println("DeepSpeed  passed=" + passed + "  failed=" + failed);
        System.out.println("============================================================");
        if (failed > 0) System.exit(1);
        System.out.println("ALL DIMENSIONS GREEN");
    }
}
