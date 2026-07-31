package distribute;

import org.bytedeco.pytorch.distributed.*;
import org.bytedeco.pytorch.llm.accelerate.utils.MultiProcessLauncher;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.optim.Adam;
import org.bytedeco.pytorch.optim.options.AdamOptions;
import org.bytedeco.pytorch.Tensor;

import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;

import static org.bytedeco.pytorch.global.torch.*;

/**
 * Benchmark 7: PipelineParallel — 2-stage send/recv, microbatch GPipe.
 * Corresponds to ddp.md instance 7 (torch.distributed.pipeline.sync.Pipe).
 */
public class BenchmarkPipelineParallel {
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

    static Module makeStage0(long hidden) {
        return new Module("Stage0") {
            private final org.bytedeco.pytorch.nn.modules.LinearImpl l1 =
                    register_module("l1", new org.bytedeco.pytorch.nn.modules.LinearImpl(hidden, hidden * 2));
            private final org.bytedeco.pytorch.nn.modules.ReLUImpl relu = register_module("relu", new org.bytedeco.pytorch.nn.modules.ReLUImpl());
            public Tensor forward(Tensor x) { return relu.forward(l1.forward(x)); }
        };
    }

    static Module makeStage1(long hidden, long vocab) {
        return new Module("Stage1") {
            private final org.bytedeco.pytorch.nn.modules.LinearImpl l2 =
                    register_module("l2", new org.bytedeco.pytorch.nn.modules.LinearImpl(hidden * 2, hidden));
            private final org.bytedeco.pytorch.nn.modules.LinearImpl head =
                    register_module("head", new org.bytedeco.pytorch.nn.modules.LinearImpl(hidden, vocab));
            public Tensor forward(Tensor x) { return head.forward(l2.forward(x)); }
        };
    }

    static void runSingleProcess() throws Exception {
        System.out.println("=== PipelineParallel benchmark (single-process) ===");
        int world = 1, chunks = 2;
        List<Module> stages = List.of(makeStage0(128), makeStage1(128, 1024));
        try (DistributedStore store = DistributedStore.create(0, world);
             ProcessGroupWrapper pg = ProcessGroupWrapper.create(0, world, store);
             PipelineParallelTrainer pp = PipelineParallelTrainer.create(stages, pg, chunks)) {
            check("pp first stage", pp.isFirstStage());
            check("pp last stage", pp.isLastStage());
            check("chunks=" + chunks, pp.getChunks() == chunks);
            section("Training");
            Adam opt = new Adam(stages.get(0).parameters(), new AdamOptions().lr(1e-3));
            for (int i = 0; i < 3; i++) {
                Tensor x = randn(new long[]{4, 128});
                Tensor y = randint(1024, new long[]{4});
                Tensor loss = pp.step(x, y, opt);
                check("loss step " + i, loss != null && !loss.isNull() && isFinite(loss));
            }
            check("steps > 0", pp.getNumSteps() > 0);
            check("microbatches > 0", pp.getNumMicroBatches() > 0);
            System.out.println(pp);
        }
        done();
    }

    static void runMultiProcess() throws Exception {
        int rank = MP_RANK.get();
        int world = 2, chunks = 2;
        List<Module> stages = world == 1
                ? List.of(makeStage0(128), makeStage1(128, 1024))
                : rank == 0 ? List.of(makeStage0(128)) : List.of(makeStage1(128, 1024));
        try (DistributedStore store = DistributedStore.create(rank, world);
             ProcessGroupWrapper pg = ProcessGroupWrapper.create(rank, world, store);
             PipelineParallelTrainer pp = PipelineParallelTrainer.create(stages, pg, chunks)) {
            Adam opt = new Adam(pp.localStage().parameters(), new AdamOptions().lr(1e-3));
            for (int i = 0; i < 3; i++) {
                Tensor x = randn(new long[]{4, 128});
                Tensor y = randint(1024, new long[]{4});
                Tensor loss = pp.step(x, y, opt);
                check("loss step " + i + " rank " + rank, loss == null || !loss.isNull() && isFinite(loss));
            }
            if (rank == 0) System.out.println("[rank0] PP OK: " + pp);
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
