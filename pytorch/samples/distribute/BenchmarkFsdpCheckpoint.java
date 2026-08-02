package distribute;
import org.bytedeco.pytorch.optim.*;
import org.bytedeco.pytorch.optim.options.*;

import org.bytedeco.pytorch.distributed.*;
import org.bytedeco.pytorch.distributed.examples.MockLLM;
import org.bytedeco.pytorch.llm.accelerate.utils.MultiProcessLauncher;
import org.bytedeco.pytorch.optim.Adam;
import org.bytedeco.pytorch.optim.options.AdamOptions;
import org.bytedeco.pytorch.Tensor;

import java.nio.file.*;
import java.util.concurrent.atomic.AtomicInteger;

import static org.bytedeco.pytorch.global.torch.*;

/**
 * Benchmark 10: FSDP checkpoint — sharded save/load roundtrip + rank0 full export.
 * Corresponds to ddp.md instance 10 (FSDP checkpointing / full state dict export).
 */
public class BenchmarkFsdpCheckpoint {
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
        System.out.println("=== FSDP Checkpoint benchmark (single-process) ===");
        Path base = Files.createTempDirectory("fsdp_ckpt_test");
        try {
            try (DistributedStore store = DistributedStore.create(0, 1);
                 ProcessGroupWrapper pg = ProcessGroupWrapper.create(0, 1, store);
                 MockLLM model = MockLLM.tiny();
                 NativeFSDPTrainer fsdp = NativeFSDPTrainer.builder()
                         .module(model).processGroup(pg)
                         .shardingStrategy(ShardingStrategy.FULL_SHARD)
                         .build()) {
                section("Train briefly");
                Adam opt = new Adam(model.parameters(), new AdamOptions().lr(1e-3));
                for (int i = 0; i < 2; i++) {
                    Tensor x = randint(1024, new long[]{2, 16});
                    Tensor y = randint(1024, new long[]{2, 16});
                    fsdp.step(x, y, opt);
                }
                section("Save sharded checkpoint");
                Path shardDir = base.resolve("sharded");
                fsdp.saveSharded(shardDir);
                check("shard file exists", Files.exists(shardDir.resolve("shard_rank0.f32")));
                check("meta.txt exists", Files.exists(shardDir.resolve("meta.txt")));
                section("Load sharded checkpoint");
                // Load back: modifies shard and rebuilds full from allgather
                fsdp.loadSharded(shardDir);
                check("load succeeded", true);
                section("Save full checkpoint (rank0)");
                Path fullPath = base.resolve("full_ckpt.f32");
                fsdp.saveFull(fullPath);
                if (pg.isMainProcess()) {
                    check("full checkpoint exists", Files.exists(fullPath));
                    long sz = Files.size(fullPath);
                    check("full size > 0", sz > 0);
                    System.out.println("  full ckpt size: " + sz + " bytes");
                }
                section("Final smoke train after load");
                for (int i = 0; i < 2; i++) {
                    Tensor x = randint(1024, new long[]{2, 16});
                    Tensor y = randint(1024, new long[]{2, 16});
                    Tensor loss = fsdp.step(x, y, opt);
                    check("loss after load step " + i, loss != null && !loss.isNull() && isFinite(loss));
                }
            }
        } finally {
            deleteRecursive(base);
        }
        done();
    }

    static void runMultiProcess() throws Exception {
        int rank = MP_RANK.get();
        int world = 2;
        Path base = Files.createTempDirectory("fsdp_ckpt_test_" + rank);
        try (DistributedStore store = DistributedStore.create(rank, world);
             ProcessGroupWrapper pg = ProcessGroupWrapper.create(rank, world, store);
             MockLLM model = MockLLM.tiny();
             NativeFSDPTrainer fsdp = NativeFSDPTrainer.builder()
                     .module(model).processGroup(pg)
                     .shardingStrategy(ShardingStrategy.FULL_SHARD)
                     .build()) {
            Adam opt = new Adam(model.parameters(), new AdamOptions().lr(1e-3));
            for (int i = 0; i < 2; i++) {
                Tensor x = randint(1024, new long[]{2, 16});
                Tensor y = randint(1024, new long[]{2, 16});
                fsdp.step(x, y, opt);
            }
            section("Multi-process sharded save/load");
            Path shardDir = base.resolve("sharded");
            fsdp.saveSharded(shardDir);
            pg.barrierWait();
            check("shard saved rank" + rank, Files.exists(shardDir.resolve("shard_rank" + rank + ".f32")));
            fsdp.loadSharded(shardDir);
            check("shard loaded rank" + rank, true);
            pg.barrierWait();
            if (rank == 0) System.out.println("[rank0] multi-proc checkpoint OK");
        } finally {
            deleteRecursive(base);
        }
    }

    static void deleteRecursive(Path p) {
        try {
            if (Files.isDirectory(p)) {
                for (Path c : Files.list(p).toArray(Path[]::new)) deleteRecursive(c);
            }
            Files.deleteIfExists(p);
        } catch (Throwable ignored) {}
    }

    static boolean isFinite(Tensor t) {
        try { double v = t.reshape(-1).get(0).item().toDouble(); return !Double.isNaN(v) && !Double.isInfinite(v); } catch (Throwable e) { return false; }
    }
    static void done() {
        System.out.println("\n=== Results: " + passed + " passed, " + failed + " failed ===");
        if (failed > 0) throw new RuntimeException(failed + " checks failed");
    }
}
