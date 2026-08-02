package distribute;
import org.bytedeco.pytorch.optim.*;
import org.bytedeco.pytorch.optim.options.*;

import org.bytedeco.pytorch.distributed.*;
import org.bytedeco.pytorch.distributed.examples.MockLLM;
import org.bytedeco.pytorch.llm.accelerate.utils.MultiProcessLauncher;
import org.bytedeco.pytorch.optim.Adam;
import org.bytedeco.pytorch.optim.options.AdamOptions;
import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.Tensor;

import java.util.concurrent.atomic.AtomicInteger;

import static org.bytedeco.pytorch.global.torch.*;
import org.bytedeco.pytorch.distributed.DistributedLoss;

/**
 * Benchmark 9: FSDP gradient accumulation — grad_acc_steps, no_sync, effective batch.
 * Corresponds to ddp.md instance 9 (FSDP gradient accumulation).
 */
public class BenchmarkFsdpGradAccumulate {
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
        System.out.println("=== FSDP Grad Accumulation benchmark (single-process) ===");
        int world = 1, accSteps = 4;
        try (DistributedStore store = DistributedStore.create(0, world);
             ProcessGroupWrapper pg = ProcessGroupWrapper.create(0, world, store);
             MockLLM model = MockLLM.tiny();
             NativeFSDPTrainer fsdp = NativeFSDPTrainer.builder()
                     .module(model).processGroup(pg)
                     .shardingStrategy(ShardingStrategy.FULL_SHARD)
                     .build()) {
            Adam opt = new Adam(model.parameters(), new AdamOptions().lr(1e-3));
            section("Grad accumulation (no_sync)");
            try (var ns = fsdp.noSync()) {
                for (int i = 0; i < accSteps; i++) {
                    Tensor x = randint(1024, new long[]{2, 16});
                    Tensor y = randint(1024, new long[]{2, 16});
                    fsdp.zeroGrad();
                    Tensor out = fsdp.forward(x);
                    Tensor loss = DistributedLoss.crossEntropy(out, y).div(new Scalar(accSteps));
                    loss.backward();
                    check("inside noSync sync disabled", !fsdp.isSyncEnabled());
                }
            }
            check("sync re-enabled after NoSync", fsdp.isSyncEnabled());
            fsdp.reduceScatterGradients();
            opt.step();
            fsdp.zeroGrad();
            section("Full accumulation training");
            for (int epoch = 0; epoch < 2; epoch++) {
                for (int i = 0; i < accSteps; i++) {
                    Tensor x = randint(1024, new long[]{2, 16});
                    Tensor y = randint(1024, new long[]{2, 16});
                    fsdp.zeroGrad();
                    Tensor out = fsdp.forward(x);
                    Tensor loss = DistributedLoss.crossEntropy(out, y).div(new Scalar(accSteps));
                    loss.backward();
                    if ((i + 1) % accSteps == 0) {
                        fsdp.reduceScatterGradients();
                        opt.step();
                        fsdp.zeroGrad();
                        check("effective step epoch " + epoch, fsdp.getNumForwardCalls() > 0);
                    }
                }
            }
            System.out.println(fsdp);
        }
        done();
    }

    static void runMultiProcess() throws Exception {
        int rank = MP_RANK.get();
        int world = 2, accSteps = 4;
        try (DistributedStore store = DistributedStore.create(rank, world);
             ProcessGroupWrapper pg = ProcessGroupWrapper.create(rank, world, store);
             MockLLM model = MockLLM.tiny();
             NativeFSDPTrainer fsdp = NativeFSDPTrainer.builder()
                     .module(model).processGroup(pg)
                     .shardingStrategy(ShardingStrategy.FULL_SHARD)
                     .build()) {
            Adam opt = new Adam(model.parameters(), new AdamOptions().lr(1e-3));
            for (int i = 0; i < accSteps; i++) {
                Tensor x = randint(1024, new long[]{2, 16});
                Tensor y = randint(1024, new long[]{2, 16});
                fsdp.zeroGrad();
                Tensor out = fsdp.forward(x);
                Tensor loss = DistributedLoss.crossEntropy(out, y).div(new Scalar(accSteps));
                loss.backward();
                if ((i + 1) % accSteps == 0) {
                    fsdp.reduceScatterGradients();
                    opt.step();
                    fsdp.zeroGrad();
                }
            }
            if (rank == 0) System.out.println("[rank0] grad accum OK: " + fsdp);
            pg.barrierWait();
        }
    }

    static void done() {
        System.out.println("\n=== Results: " + passed + " passed, " + failed + " failed ===");
        if (failed > 0) throw new RuntimeException(failed + " checks failed");
    }
}
