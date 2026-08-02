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

import static org.bytedeco.pytorch.global.torch.*;

/**
 * Benchmark 4: 2D DeviceMesh (dp, tp) — DP FSDP + TP ColumnParallel.
 * Corresponds to ddp.md instance 4 (DP+TP hybrid).
 * worldSize must be divisible by tpSize.
 */
public class BenchmarkMeshTpFsdp {
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
        System.out.println("=== DP+TP hybrid (single-process smoke) ===");
        int world = 1, tpSize = 1;
        try (DistributedStore store = DistributedStore.create(0, world);
             ProcessGroupWrapper pg = ProcessGroupWrapper.create(0, world, store);
             DeviceMesh mesh = DeviceMesh.initDpTp(pg, tpSize);
             MockLLM model = MockLLM.tiny();
             TensorParallel.TPTrainer tp = TensorParallel.TPTrainer.create(model, pg, mesh)) {
            check("mesh 2D", mesh.ndim() == 2);
            check("dp size", mesh.size("dp") == 1);
            check("tp size", mesh.size("tp") == 1);
            section("Training");
            Adam opt = new Adam(model.parameters(), new AdamOptions().lr(1e-4));
            for (int i = 0; i < 3; i++) {
                Tensor x = randint(1024, new long[]{2, 16});
                Tensor y = randint(1024, new long[]{2, 16});
                Tensor loss = tp.step(x, y, opt);
                check("loss step " + i, loss != null && !loss.isNull() && isFinite(loss));
            }
            System.out.println(mesh);
        }
        done();
    }

    static void runMultiProcess() throws Exception {
        int rank = MP_RANK.get();
        int world = 2;
        int tpSize = 2; // world divisible by tpSize
        try (DistributedStore store = DistributedStore.create(rank, world);
             ProcessGroupWrapper pg = ProcessGroupWrapper.create(rank, world, store);
             DeviceMesh mesh = DeviceMesh.initDpTp(pg, tpSize);
             DeviceMesh dpMesh = mesh.get("dp");
             DeviceMesh tpMesh = mesh.get("tp");
             MockLLM model = MockLLM.tiny();
             NativeFSDPTrainer fsdp = NativeFSDPTrainer.builder()
                     .module(model).processGroup(pg).deviceMesh(dpMesh)
                     .shardingStrategy(ShardingStrategy.FULL_SHARD).build()) {
            Adam opt = new Adam(model.parameters(), new AdamOptions().lr(1e-4));
            for (int i = 0; i < 3; i++) {
                Tensor x = randint(1024, new long[]{2, 16});
                Tensor y = randint(1024, new long[]{2, 16});
                Tensor loss = fsdp.step(x, y, opt);
                check("loss step " + i + " rank " + rank, loss != null && !loss.isNull() && isFinite(loss));
            }
            if (rank == 0) {
                System.out.println("  mesh=" + mesh);
                System.out.println("  dpMesh=" + dpMesh);
                System.out.println("  tpMesh=" + tpMesh);
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
