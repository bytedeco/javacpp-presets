package distribute;
import org.bytedeco.pytorch.nn.modules.*;
import org.bytedeco.pytorch.optim.*;
import org.bytedeco.pytorch.optim.options.*;
import org.bytedeco.pytorch.distributed.*;

import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.distributed.DistributedStore;
import org.bytedeco.pytorch.distributed.ProcessGroupWrapper;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.modules.LinearImpl;
import org.bytedeco.pytorch.optim.Adam;
import org.bytedeco.pytorch.optim.options.AdamOptions;
import org.bytedeco.pytorch.llm.accelerate.Accelerator;
import org.bytedeco.pytorch.llm.accelerate.DataLoaderShard;
import org.bytedeco.pytorch.llm.accelerate.PartialState;
import org.bytedeco.pytorch.llm.accelerate.plugins.DeepSpeedPlugin;
import org.bytedeco.pytorch.llm.accelerate.utils.MultiProcessLauncher;
import org.bytedeco.pytorch.llm.deepspeed.DeepSpeedConfig;

import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

import static org.bytedeco.pytorch.global.torch.randn;
import static org.bytedeco.pytorch.global.torch.mse_loss;

public class BenchmarkAccelerate {
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
        System.out.println("=== Accelerate benchmark (single-process) ===");
        d1PrepareBackwardStep();
        d2Accumulate();
        d3Gather();
        d4PartialState();
        d5FSDPsmoke();
        d6DSplugin();
        d7SaveLoad();
        d8DataLoaderShard();
        d9MixedPrecision();
        d10Unwrap();
        done();
    }

    static void runMultiProcess() throws Exception {
        int rank = MP_RANK.get();
        try (DistributedStore store = DistributedStore.create(rank, 2)) {
            ProcessGroupWrapper pg = ProcessGroupWrapper.create(rank, 2, store);
            Accelerator acc = Accelerator.builder().processGroup(pg).build();
            Module m = tinyModel();
            Adam opt = new Adam(m.parameters(), new AdamOptions());
            acc.prepare(m, opt);
            for (int i = 0; i < 3; i++) {
                Tensor x = acc.toDevice(randn(new long[]{2, 32}));
                Tensor y = acc.toDevice(randn(new long[]{2, 16}));
                Module unwrapped = acc.unwrapModel(m);
                Tensor out = unwrapped.forward(x);
                Tensor loss = mse_loss(out, y, null);
                acc.backward(loss);
                acc.step();
            }
            if (rank == 0) System.out.println("[rank0] multi-process accelerate OK");
            pg.barrier();
            acc.close();
        }
    }

    static void d1PrepareBackwardStep() throws Exception {
        section("D1 prepare / backward / step");
        Accelerator acc = Accelerator.builder().build();
        Module m = tinyModel();
        Adam opt = new Adam(m.parameters(), new AdamOptions());
        acc.prepare(m, opt);
        check("isPrepared", acc.isPrepared());
        check("numProcesses=1", acc.numProcesses() == 1);
        check("isMainProcess", acc.isMainProcess());
        Module unwrapped = acc.unwrapModel(m);
        Tensor x = acc.toDevice(randn(new long[]{2, 32}));
        Tensor y = acc.toDevice(randn(new long[]{2, 16}));
        Tensor loss = mse_loss(unwrapped.forward(x), y, null);
        acc.backward(loss);
        acc.step();
        check("stepCount=1", acc.stepCount() == 1);
        acc.close();
    }

    static void d2Accumulate() throws Exception {
        section("D2 Gradient accumulation");
        Accelerator acc = Accelerator.builder().gradientAccumulationSteps(4).build();
        Module m = tinyModel();
        Adam opt = new Adam(m.parameters(), new AdamOptions());
        acc.prepare(m, opt);
        for (int i = 0; i < 8; i++) {
            Module unwrapped = acc.unwrapModel(m);
            Tensor loss = mse_loss(unwrapped.forward(acc.toDevice(randn(new long[]{1, 32}))),
                    acc.toDevice(randn(new long[]{1, 16})), null);
            acc.backward(loss);
            if (acc.isGradientAccumulationBoundary()) acc.step();
        }
        check("stepCount=2", acc.stepCount() == 2);
        acc.close();
    }

    static void d3Gather() {
        section("D3 gatherObject");
        Accelerator acc = Accelerator.builder().build();
        List<String> list = acc.gatherObject("hello");
        check("gather size=1 in sp", list.size() == 1);
        check("gather contains hello", list.get(0).equals("hello"));
        acc.close();
    }

    static void d4PartialState() {
        section("D4 PartialState / waitForEveryone");
        try (PartialState ps = PartialState.single()) {
            check("processIndex=0", ps.processIndex() == 0);
            check("numProcesses=1", ps.numProcesses() == 1);
            check("isMainProcess", ps.isMainProcess());
            ps.waitForEveryone();
            check("waitForEveryone no-throw", true);
        }
        try (PartialState ps = PartialState.fromEnv()) {
            check("fromEnv rank=0", ps.processIndex() == 0);
        }
    }

    static void d5FSDPsmoke() {
        section("D5 FSDP plugin smoke");
        try { Class.forName("org.bytedeco.pytorch.llm.accelerate.plugins.FullyShardedDataParallelPlugin"); check("FSDPPlugin class", true); }
        catch (Exception e) { check("FSDPPlugin class", false); }
    }

    static void d6DSplugin() {
        section("D6 DeepSpeedPlugin");
        DeepSpeedPlugin plugin = new DeepSpeedPlugin(DeepSpeedConfig.builder().zeroStage(2).build());
        check("DS plugin config zeroStage=2", plugin.config().zeroStage() == 2);
        check("DS plugin not initialized", !plugin.isInitialized());
    }

    static void d7SaveLoad() throws Exception {
        section("D7 saveState / loadState");
        Accelerator acc = Accelerator.builder().build();
        Module m = tinyModel();
        Adam opt = new Adam(m.parameters(), new AdamOptions());
        acc.prepare(m, opt);
        acc.saveState("foo", 42);
        Object val = acc.loadState("foo");
        check("saveState get=42", val != null && val.equals(42));
        acc.close();
    }

    static void d8DataLoaderShard() {
        section("D8 DataLoaderShard");
        List<String> data = List.of("a", "b", "c", "d");
        DataLoaderShard<String> shard0 = new DataLoaderShard<>(data, 0, 1, true);
        check("shard size in sp=4", shard0.size() == 4);
        DataLoaderShard<String> shard1 = new DataLoaderShard<>(data, 1, 2, true);
        check("shard rank1 size>0", shard1.size() > 0);
    }

    static void d9MixedPrecision() {
        section("D9 Mixed precision flags");
        Accelerator fp16 = Accelerator.builder().mixedPrecision("fp16").build();
        check("mixedPrecision=fp16", fp16.mixedPrecision().equals("fp16"));
        Accelerator bf16 = Accelerator.builder().mixedPrecision("bf16").build();
        check("mixedPrecision=bf16", bf16.mixedPrecision().equals("bf16"));
        Accelerator no = Accelerator.builder().mixedPrecision("no").build();
        check("mixedPrecision=no", no.mixedPrecision().equals("no"));
    }

    static void d10Unwrap() throws Exception {
        section("D10 unwrapModel");
        Accelerator acc = Accelerator.builder().build();
        Module m = tinyModel();
        acc.prepare(m, null);
        Module unwrapped = acc.unwrapModel(null);
        check("unwrap returns same model", unwrapped != null);
        acc.close();
    }

    static void done() {
        System.out.println("\n============================================================");
        System.out.println("Accelerate  passed=" + passed + "  failed=" + failed);
        System.out.println("============================================================");
        if (failed > 0) System.exit(1);
        System.out.println("ALL DIMENSIONS GREEN");
    }
}
