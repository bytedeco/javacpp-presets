package distribute;
import org.bytedeco.pytorch.optim.*;
import org.bytedeco.pytorch.optim.options.*;

import org.bytedeco.pytorch.distributed.*;
import org.bytedeco.pytorch.distributed.examples.MockLLM;
import org.bytedeco.pytorch.llm.accelerate.utils.MultiProcessLauncher;
import org.bytedeco.pytorch.optim.Adam;
import org.bytedeco.pytorch.optim.options.AdamOptions;
import org.bytedeco.pytorch.Tensor;

import java.util.concurrent.atomic.AtomicInteger;
import java.nio.file.*;

import static org.bytedeco.pytorch.global.torch.*;

/**
 * Benchmark 2: NativeFSDPTrainer — FULL_SHARD FSDP with real allgather/reduce-scatter.
 * Corresponds to ddp.md instance 2 (FSDP1 ShardingStrategy.FULL_SHARD).
 */
public class BenchmarkNativeFSDP1 {

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
        System.out.println("=== NativeFSDP1 benchmark (single-process) ===");
        section("Init");
        try (DistributedStore store = DistributedStore.create(0, 1);
             ProcessGroupWrapper pg = ProcessGroupWrapper.create(0, 1, store);
             MockLLM model = MockLLM.tiny();
             NativeFSDPTrainer fsdp = NativeFSDPTrainer.builder()
                     .module(model)
                     .processGroup(pg)
                     .shardingStrategy(ShardingStrategy.FULL_SHARD)
                     .reshardAfterForward(true)
                     .mixedPrecision(MixedPrecisionConfig.fp32())
                     .build()) {
            check("shardSize > 0", fsdp.getShardSize() > 0);
            check("totalParamNumel > 0", fsdp.getTotalParamSize() > 0);
            check("strategy = FULL_SHARD", fsdp.getShardingStrategy() == ShardingStrategy.FULL_SHARD);
            check("world=1", fsdp.getWorldSize() == 1);
            section("Training steps");
            Adam opt = new Adam(model.parameters(), new AdamOptions().lr(1e-3));
            for (int i = 0; i < 3; i++) {
                Tensor x = randint(1024, new long[]{2, 16});
                Tensor y = randint(1024, new long[]{2, 16});
                Tensor loss = fsdp.step(x, y, opt);
                check("loss finite step " + i, loss != null && !loss.isNull() && isFinite(loss));
            }
            check("allgather calls > 0", fsdp.getNumAllGatherCalls() > 0);
            check("reduceScatter calls > 0", fsdp.getNumReduceScatterCalls() > 0);
            section("Checkpoint save/load");
            Path ckpt = Files.createTempDirectory("fsdp_ckpt");
            fsdp.saveSharded(ckpt);
            check("checkpoint saved", Files.exists(ckpt.resolve("shard_rank0.f32")));
            System.out.println(fsdp);
        }
        done();
    }

    static void runMultiProcess() throws Exception {
        int rank = MP_RANK.get();
        int world = 2;
        try (DistributedStore store = DistributedStore.create(rank, world);
             ProcessGroupWrapper pg = ProcessGroupWrapper.create(rank, world, store);
             MockLLM model = MockLLM.tiny();
             NativeFSDPTrainer fsdp = NativeFSDPTrainer.builder()
                     .module(model)
                     .processGroup(pg)
                     .shardingStrategy(ShardingStrategy.FULL_SHARD)
                     .build()) {
            Adam opt = new Adam(model.parameters(), new AdamOptions().lr(1e-3));
            section("Multi-process FSDP");
            for (int i = 0; i < 4; i++) {
                Tensor x = randint(1024, new long[]{2, 16});
                Tensor y = randint(1024, new long[]{2, 16});
                Tensor loss = fsdp.step(x, y, opt);
                check("loss finite step " + i + " rank " + rank, loss != null && !loss.isNull() && isFinite(loss));
                if (rank == 0 && i % 2 == 0) System.out.println("  [rank0] step=" + i + " loss=" + fmt(loss));
            }
            pg.barrierWait();
            if (rank == 0) System.out.println("[rank0] multi-proc FSDP OK");
            System.out.println("  [rank" + rank + "] shardsize=" + fsdp.getShardSize()
                    + " ag=" + fsdp.getNumAllGatherCalls()
                    + " rs=" + fsdp.getNumReduceScatterCalls());
        }
    }

    static boolean isFinite(Tensor t) {
        try { double v = t.reshape(-1).get(0).item().toDouble(); return !Double.isNaN(v) && !Double.isInfinite(v); } catch (Throwable e) { return false; }
    }

    static String fmt(Tensor t) {
        try { return String.format("%.4f", t.item().toDouble()); } catch (Throwable e) { return "N/A"; }
    }

    static void done() {
        System.out.println("\n=== Results: " + passed + " passed, " + failed + " failed ===");
        if (failed > 0) throw new RuntimeException(failed + " checks failed");
    }
}
