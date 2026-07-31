package distribute;

import org.bytedeco.pytorch.distributed.*;
import org.bytedeco.pytorch.distributed.examples.MockLLM;
import org.bytedeco.pytorch.llm.accelerate.*;
import org.bytedeco.pytorch.llm.accelerate.plugins.*;
import org.bytedeco.pytorch.llm.accelerate.utils.MultiProcessLauncher;
import org.bytedeco.pytorch.optim.Adam;
import org.bytedeco.pytorch.optim.options.AdamOptions;
import org.bytedeco.pytorch.Tensor;

import java.util.concurrent.atomic.AtomicInteger;

import static org.bytedeco.pytorch.global.torch.*;
import org.bytedeco.pytorch.distributed.DistributedLoss;

/**
 * Benchmark 6: Accelerator + FsdpPlugin → NativeFSDPTrainer.
 * Corresponds to ddp.md instance 6 (HF Accelerate FSDP).
 */
public class BenchmarkAccelerateFsdp {
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
        System.out.println("=== Accelerator FSDP (single-process) ===");
        section("Prepare without PG (world=1 → no FSDP wrap, plain device place)");
        // FSDP only activates when processGroup.worldSize > 1. Single-process
        // still exercises Accelerator prepare/backward/step + plugin construction.
        FullyShardedDataParallelPlugin fsdpPlugin = FullyShardedDataParallelPlugin.fullShard();
        check("fsdpPlugin useNative", fsdpPlugin.useNative());
        Accelerator acc = Accelerator.builder()
                .cpu(true)
                .mixedPrecision("no")
                .gradientAccumulationSteps(1)
                .fsdpPlugin(fsdpPlugin)
                .build();
        try (MockLLM model = MockLLM.tiny()) {
            acc.prepare(model);
            check("Accelerator prepared", acc.isPrepared());
            // world=1: native FSDP not wrapped (by design)
            check("native FSDP null on world=1", acc.nativeFsdpTrainer() == null);
            Adam opt = new Adam(model.parameters(), new AdamOptions().lr(1e-3));
            acc.prepare(model, opt);
            section("Training");
            for (int i = 0; i < 3; i++) {
                Tensor x = randint(1024, new long[]{2, 16});
                Tensor y = randint(1024, new long[]{2, 16});
                Tensor out = model.forward(x);
                Tensor loss = DistributedLoss.crossEntropy(out, y);
                acc.backward(loss);
                acc.step();
                acc.zeroGrad();
                check("loss finite step " + i, loss != null && !loss.isNull() && isFinite(loss));
            }
            // Also smoke NativeFSDPTrainer directly on single-process HashStore
            section("Direct NativeFSDPTrainer world=1");
            try (DistributedStore store = DistributedStore.createSingleProcess();
                 ProcessGroupWrapper pg = ProcessGroupWrapper.create(0, 1, store);
                 NativeFSDPTrainer fsdp = NativeFSDPTrainer.create(model, pg)) {
                check("direct fsdp shardSize>0", fsdp.getShardSize() > 0);
                Tensor x = randint(1024, new long[]{2, 16});
                Tensor y = randint(1024, new long[]{2, 16});
                Tensor loss = fsdp.step(x, y, opt);
                check("direct fsdp loss", loss != null && !loss.isNull() && isFinite(loss));
            }
            System.out.println(acc);
        }
        done();
    }

    static void runMultiProcess() throws Exception {
        int rank = MP_RANK.get();
        int world = 2;
        FullyShardedDataParallelPlugin fsdpPlugin = FullyShardedDataParallelPlugin.fullShard();
        try (DistributedStore store = DistributedStore.create(rank, world);
             ProcessGroupWrapper pg = ProcessGroupWrapper.create(rank, world, store)) {
            Accelerator acc = Accelerator.builder()
                    .processGroup(pg)
                    .fsdpPlugin(fsdpPlugin)
                    .mixedPrecision("no")
                    .gradientAccumulationSteps(1)
                    .build();
            try (MockLLM model = MockLLM.tiny()) {
                Adam opt = new Adam(model.parameters(), new AdamOptions(1e-3));
                acc.prepare(model, opt);
                check("rank " + rank + " prepared", acc.isPrepared());
                check("native FSDP trainer", acc.nativeFsdpTrainer() != null);
                for (int i = 0; i < 3; i++) {
                    Tensor x = randint(1024, new long[]{2, 16});
                    Tensor y = randint(1024, new long[]{2, 16});
                    Tensor out = model.forward(x);
                    Tensor loss = DistributedLoss.crossEntropy(out, y);
                    acc.backward(loss);
                    acc.step();
                    acc.zeroGrad();
                    check("loss step " + i + " rank " + rank, loss != null && !loss.isNull() && isFinite(loss));
                }
                if (rank == 0) System.out.println("[rank0] Accelerate FSDP OK");
            }
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
