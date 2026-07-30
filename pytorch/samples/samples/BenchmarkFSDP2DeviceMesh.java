package samples;
import org.bytedeco.pytorch.optim.options.*;
import org.bytedeco.pytorch.optim.*;

import org.bytedeco.pytorch.distributed.*;
import org.bytedeco.pytorch.distributed.examples.MockLLM;
import org.bytedeco.pytorch.llm.accelerate.utils.MultiProcessLauncher;
import org.bytedeco.pytorch.optim.Adam;
import org.bytedeco.pytorch.optim.options.AdamOptions;
import org.bytedeco.pytorch.Tensor;

import java.util.concurrent.atomic.AtomicInteger;

import static org.bytedeco.pytorch.global.torch.*;

/**
 * Benchmark 3: FSDP2 + DeviceMesh — 1D mesh, native FSDP on mesh PG.
 * Corresponds to ddp.md instance 3 (FSDP2 + init_device_mesh).
 */
public class BenchmarkFSDP2DeviceMesh {
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
        System.out.println("=== FSDP2 + DeviceMesh benchmark (single-process) ===");
        section("Init");
        int world = 1;
        try (DistributedStore store = DistributedStore.create(0, world);
             ProcessGroupWrapper pg = ProcessGroupWrapper.create(0, world, store);
             DeviceMesh mesh = DeviceMesh.init1d(pg);
             MockLLM model = MockLLM.tiny();
             NativeFSDPTrainer fsdp = NativeFSDPTrainer.builder()
                     .module(model).processGroup(pg)
                     .deviceMesh(mesh)
                     .shardingStrategy(ShardingStrategy.FULL_SHARD)
                     .build()) {
            check("mesh ndim=1", mesh.ndim() == 1);
            check("mesh size=world", mesh.size() == world);
            check("mesh has dp dim", mesh.size("dp") == world);
            check("fsdp shardSize > 0", fsdp.getShardSize() > 0);
            section("Training");
            Adam opt = new Adam(model.parameters(), new AdamOptions().lr(1e-4));
            for (int i = 0; i < 3; i++) {
                Tensor x = randint(1024, new long[]{2, 16});
                Tensor y = randint(1024, new long[]{2, 16});
                Tensor loss = fsdp.step(x, y, opt);
                check("loss finite step " + i, loss != null && !loss.isNull() && isFinite(loss));
            }
            System.out.println(fsdp);
            System.out.println(mesh);
        }
        done();
    }

    static void runMultiProcess() throws Exception {
        int rank = MP_RANK.get();
        int world = 2;
        try (DistributedStore store = DistributedStore.create(rank, world);
             ProcessGroupWrapper pg = ProcessGroupWrapper.create(rank, world, store);
             DeviceMesh mesh = DeviceMesh.init1d(pg, "dp");
             MockLLM model = MockLLM.tiny();
             NativeFSDPTrainer fsdp = NativeFSDPTrainer.builder()
                     .module(model).processGroup(pg)
                     .deviceMesh(mesh)
                     .shardingStrategy(ShardingStrategy.FULL_SHARD)
                     .build()) {
            Adam opt = new Adam(model.parameters(), new AdamOptions().lr(1e-4));
            for (int i = 0; i < 3; i++) {
                Tensor x = randint(1024, new long[]{2, 16});
                Tensor y = randint(1024, new long[]{2, 16});
                Tensor loss = fsdp.step(x, y, opt);
                check("loss step " + i + " rank " + rank, loss != null && !loss.isNull() && isFinite(loss));
            }
            if (rank == 0) System.out.println("[rank0] mesh OK: " + mesh);
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
