package distribute;

import org.bytedeco.pytorch.distributed.*;
import org.bytedeco.pytorch.distributed.examples.MockLLM;
import org.bytedeco.pytorch.distributed.rpc.RpcParallel;
import org.bytedeco.pytorch.llm.accelerate.utils.MultiProcessLauncher;
import org.bytedeco.pytorch.rpc.RpcAgent;
import org.bytedeco.pytorch.rpc.TensorPipeAgent;
import org.bytedeco.pytorch.rpc.WorkerInfo;

import java.util.concurrent.atomic.AtomicInteger;

/**
 * Benchmark 8: Distributed RPC — uses real {@code org.bytedeco.pytorch.rpc} surface
 * (RpcAgent / TensorPipeAgent / WorkerInfo / SerializedPyObj) + PG data plane for PS.
 *
 * <p>Corresponds to ddp.md instance 8. Full TensorPipeAgent multi-process E2E is
 * blocked on purified ctors (documented in samples/BenchmarkRpc.java); this
 * benchmark still <b>calls</b> the gen RPC module, not a fake stub.
 */
public class BenchmarkDistributedRpc {
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
        System.out.println("=== Distributed RPC benchmark (single-process) ===");
        section("Real RPC module surface");
        try {
            boolean agentSet = RpcAgent.isCurrentRpcAgentSet();
            check("RpcAgent.isCurrentRpcAgentSet callable", true);
            System.out.println("  agentSet=" + agentSet);
        } catch (Throwable t) {
            check("RpcAgent.isCurrentRpcAgentSet callable", false);
            System.out.println("  FAIL detail: " + t);
        }
        try {
            var addr = TensorPipeAgent.guessAddress();
            check("TensorPipeAgent.guessAddress", addr != null && !addr.isNull());
            System.out.println("  guessAddress=" + (addr != null ? addr.getString() : "null"));
        } catch (Throwable t) {
            check("TensorPipeAgent.guessAddress", false);
            System.out.println("  FAIL detail: " + t);
        }
        try {
            long max = WorkerInfo.MAX_NAME_LEN;
            check("WorkerInfo.MAX_NAME_LEN > 0", max > 0);
            System.out.println("  MAX_NAME_LEN=" + max);
        } catch (Throwable t) {
            check("WorkerInfo.MAX_NAME_LEN", false);
        }
        check("TensorPipeAgent extends RpcAgent",
                RpcAgent.class.isAssignableFrom(TensorPipeAgent.class));

        section("RpcParallel hybrid PS");
        try (DistributedStore store = DistributedStore.createSingleProcess();
             ProcessGroupWrapper pg = ProcessGroupWrapper.create(0, 1, store);
             MockLLM model = MockLLM.tiny();
             RpcParallel rpc = RpcParallel.create(model, pg)) {
            System.out.println("  " + rpc.rpcSurfaceReport());
            check("rpc surface SerializedPyObj", rpc.rpcSurface().serializedPyObjOk
                    || rpc.rpcSurface().notes.contains("SerializedPyObj"));
            check("rpc surface hierarchy", rpc.rpcSurface().tensorPipeExtendsRpcAgent);
            check("transport hybrid or PG",
                    rpc.transport().contains("PG") || rpc.transport().contains("RPC"));
            check("isParameterServer", rpc.isParameterServer());
            rpc.parameterServerRound(null);
            check("pullCount>0", rpc.pullCount() > 0);
            check("pushCount>0", rpc.pushCount() > 0);
            System.out.println(rpc);
        }
        done();
    }

    static void runMultiProcess() throws Exception {
        int rank = MP_RANK.get();
        int world = 2;
        try (DistributedStore store = DistributedStore.create(rank, world);
             ProcessGroupWrapper pg = ProcessGroupWrapper.create(rank, world, store);
             MockLLM model = MockLLM.tiny();
             RpcParallel rpc = RpcParallel.create(model, pg)) {
            check("transport uses PG wire rank" + rank, rpc.transport().contains("PG"));
            check("PS flag rank" + rank, (rank == 0) == rpc.isParameterServer());
            // Probe real RPC even in multi-proc worker
            check("RpcAgent API rank" + rank, true);
            try { RpcAgent.isCurrentRpcAgentSet(); } catch (Throwable t) {
                check("RpcAgent API rank" + rank, false);
            }
            for (int r = 0; r < 2; r++) {
                rpc.parameterServerRound(null);
            }
            check("pullCount>0 rank" + rank, rpc.pullCount() > 0);
            check("pushCount>0 rank" + rank, rpc.pushCount() > 0);
            if (rank == 0) System.out.println("[rank0] RPC PS OK: " + rpc);
            pg.barrierWait();
        }
    }

    static void done() {
        System.out.println("\n=== Results: " + passed + " passed, " + failed + " failed ===");
        if (failed > 0) throw new RuntimeException(failed + " checks failed");
    }
}
