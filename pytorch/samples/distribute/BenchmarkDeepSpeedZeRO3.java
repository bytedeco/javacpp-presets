package distribute;
import org.bytedeco.pytorch.optim.*;
import org.bytedeco.pytorch.optim.options.*;

import org.bytedeco.pytorch.distributed.*;
import org.bytedeco.pytorch.distributed.examples.MockLLM;
import org.bytedeco.pytorch.llm.accelerate.utils.MultiProcessLauncher;
import org.bytedeco.pytorch.llm.deepspeed.*;
import org.bytedeco.pytorch.optim.Adam;
import org.bytedeco.pytorch.optim.options.AdamOptions;
import org.bytedeco.pytorch.Tensor;

import java.util.concurrent.atomic.AtomicInteger;

import static org.bytedeco.pytorch.global.torch.*;
import org.bytedeco.pytorch.distributed.DistributedLoss;

/**
 * Benchmark 5: DeepSpeed ZeRO-3 — extends existing DeepSpeed benchmark.
 * Corresponds to ddp.md instance 5 (DeepSpeed ZeRO3 stage=3).
 */
public class BenchmarkDeepSpeedZeRO3 {
    static int passed = 0, failed = 0;
    static final AtomicInteger MP_RANK = new AtomicInteger(-1);

    static void check(String name, boolean ok) {
        if (ok) { passed++; System.out.println("  PASS  " + name); }
        else { failed++; System.out.println("  FAIL  " + name); }
    }
    static void section(String t) { System.out.println("\n=== " + t + " ==="); }

    public static void main(String[] args) throws Exception {
        // isLaunched() checks RANK env is actually set. envRank() defaults to 0,
        // so "rank < 0" NEVER selects single-process and caused 11min Gloo hangs.
        if (!MultiProcessLauncher.isLaunched()) {
            MP_RANK.set(-1);
            runSingleProcess();
        } else {
            int rank = MultiProcessLauncher.envRank();
            MP_RANK.set(rank);
            runMultiProcess();
        }
    }
    public static void mainSingle() throws Exception { runSingleProcess(); }

    static void runSingleProcess() throws Exception {
        System.out.println("=== DeepSpeed ZeRO3 benchmark (single-process) ===");
        section("Init ZeRO3");
        DeepSpeedConfig cfg = DeepSpeedConfig.builder()
                .zeroStage(3)
                .trainBatchSize(4)
                .gradientAccumulationSteps(1)
                .gradientClip(1.0)
                .precision("fp32")  // Mac CPU — no bf16
                .build();
        try (DistributedStore store = DistributedStore.create(0, 1);
             ProcessGroupWrapper pg = ProcessGroupWrapper.create(0, 1, store);
             MockLLM model = MockLLM.tiny();
             DeepSpeedEngine eng = DeepSpeed.initialize(model, new Adam(model.parameters(), new AdamOptions().lr(1e-4)), cfg, pg)) {
            check("zeroStage=3", eng.zeroStage() == 3);
            check("trainBatchSize=4", eng.getTrainBatchSize() == 4);
            check("gathered=false at start", !eng.isGathered());
            section("Training");
            for (int i = 0; i < 3; i++) {
                Tensor x = randint(1024, new long[]{2, 16});
                Tensor y = randint(1024, new long[]{2, 16});
                Tensor out = eng.forward(x);
                Tensor loss = DistributedLoss.crossEntropy(out, y);
                eng.backward(loss);
                eng.step();
                eng.zeroGrad();
                check("loss finite step " + i, loss != null && !loss.isNull() && isFinite(loss));
            }
            check("step count > 0", eng.globalStep() > 0);
            section("Memory stats");
            var mem = eng.memoryStats();
            check("memory stats non-null", mem != null);
            System.out.println(eng);
        }
        done();
    }

    static void runMultiProcess() throws Exception {
        int rank = MP_RANK.get();
        int world = 2;
        DeepSpeedConfig cfg = DeepSpeedConfig.builder()
                .zeroStage(3).trainBatchSize(4)
                .gradientAccumulationSteps(1).precision("fp32").build();
        try (DistributedStore store = DistributedStore.create(rank, world);
             ProcessGroupWrapper pg = ProcessGroupWrapper.create(rank, world, store);
             MockLLM model = MockLLM.tiny();
             DeepSpeedEngine eng = DeepSpeed.initialize(model,
                     new Adam(model.parameters(), new AdamOptions().lr(1e-4)), cfg, pg)) {
            for (int i = 0; i < 3; i++) {
                Tensor x = randint(1024, new long[]{2, 16});
                Tensor y = randint(1024, new long[]{2, 16});
                Tensor out = eng.forward(x);
                Tensor loss = DistributedLoss.crossEntropy(out, y);
                eng.backward(loss);
                eng.step();
                eng.zeroGrad();
                check("loss step " + i + " rank " + rank, loss != null && !loss.isNull() && isFinite(loss));
            }
            if (rank == 0) System.out.println("[rank0] ZeRO3 OK: " + eng);
            pg.barrierWait();
        }
    }

    static boolean isFinite(Tensor t) {
        try { double v = t.reshape(-1).get(0).item().toDouble(); return !Double.isNaN(v) && !Double.isInfinite(v); } catch (Throwable e) { return false; }
    }
    static void done() {
        System.out.println("\n=== Results: " + passed + " passed, " + failed + " failed ===");
        if (failed > 0) throw new RuntimeException(failed + " checks failed");
    }
}
