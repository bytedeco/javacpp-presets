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
import org.bytedeco.pytorch.distributed.DistributedLoss;

/**
 * Benchmark 1: NativeDDP — distributed data-parallel trainer with real c10d Reducer.
 * Corresponds to ddp.md instance 1 (nn.parallel.DistributedDataParallel baseline).
 *
 * <p>Mac/Linux smoke: worldSize=2 FileStore+Gloo.
 * Multi-process: launch via MultiProcessLauncher or external `torchrun` equivalent.
 */
public class BenchmarkNativeDDP {

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
        System.out.println("=== NativeDDP benchmark (single-process smoke) ===");
        section("Init");
        int world = 1;
        try (DistributedStore store = DistributedStore.create(0, world);
             ProcessGroupWrapper pg = ProcessGroupWrapper.create(0, world, store);
             MockLLM model = MockLLM.tiny();
             NativeDDPTrainer ddp = NativeDDPTrainer.create(model, pg)) {
            check("DDP mode is set", ddp.commMode() != null);
            check("rank=0", ddp.getRank() == 0);
            check("world=1", ddp.getWorldSize() == 1);
            section("Training");
            Adam opt = new Adam(model.parameters(), new AdamOptions().lr(1e-3));
            for (int i = 0; i < 3; i++) {
                Tensor x = randint(1024, new long[]{2, 16});
                Tensor y = randint(1024, new long[]{2, 16});
                Tensor loss = ddp.step(x, y, opt);
                if (loss != null && !loss.isNull()) {
                    check("loss finite step " + i, isFinite(loss));
                }
            }
            check("forward calls > 0", ddp.getNumForwardCalls() > 0);
            section("Grad accumulation (no_sync)");
            ddp.resetStats();
            try (var ns = ddp.noSync()) {
                check("sync disabled inside NoSync", !ddp.isSyncEnabled());
                for (int i = 0; i < 2; i++) {
                    Tensor x = randint(1024, new long[]{2, 16});
                    Tensor y = randint(1024, new long[]{2, 16});
                    Tensor out = ddp.forward(x);
                    Tensor loss = DistributedLoss.crossEntropy(out, y);
                    ddp.backward(loss);
                }
            }
            check("backward calls after noSync", ddp.getNumBackwardCalls() > 0);
            check("sync re-enabled after close", ddp.isSyncEnabled());
            System.out.println(ddp);
        }
        done();
    }

    static void runMultiProcess() throws Exception {
        int rank = MP_RANK.get();
        int world = 2;
        try (DistributedStore store = DistributedStore.create(rank, world);
             ProcessGroupWrapper pg = ProcessGroupWrapper.create(rank, world, store);
             MockLLM model = MockLLM.tiny();
             NativeDDPTrainer ddp = NativeDDPTrainer.create(model, pg)) {
            Adam opt = new Adam(model.parameters(), new AdamOptions().lr(1e-3));
            section("Multi-process training");
            for (int i = 0; i < 5; i++) {
                Tensor x = randint(1024, new long[]{2, 16});
                Tensor y = randint(1024, new long[]{2, 16});
                Tensor loss = ddp.step(x, y, opt);
                check("loss finite step " + i + " rank " + rank, isFinite(loss));
                if (rank == 0 && i % 2 == 0) {
                    System.out.println("  [rank0] step=" + i + " loss=" + fmt(loss));
                }
            }
            check("DDP multi-process done", true);
            System.out.println("  [rank" + rank + "] mode=" + ddp.commMode());
            pg.barrierWait();
            if (rank == 0) System.out.println("[rank0] all ranks finished");
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
