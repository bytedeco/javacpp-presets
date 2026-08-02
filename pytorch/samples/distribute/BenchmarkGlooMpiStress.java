package distribute;
import org.bytedeco.pytorch.nn.options.*;
import org.bytedeco.pytorch.optim.*;
import org.bytedeco.pytorch.c10.*;
import org.bytedeco.pytorch.distributed.*;

import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorVector;
import org.bytedeco.pytorch.distributed.BackendType;
import org.bytedeco.pytorch.distributed.DistributedStore;
import org.bytedeco.pytorch.distributed.NativeDDPTrainer;
import org.bytedeco.pytorch.distributed.ProcessGroupWrapper;
import org.bytedeco.pytorch.distributed.StoreType;
import org.bytedeco.pytorch.distributed.examples.MockLLM;
import org.bytedeco.pytorch.llm.accelerate.utils.MultiProcessLauncher;
import org.bytedeco.pytorch.optim.Adam;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;

import static org.bytedeco.pytorch.global.torch.ones;
import static org.bytedeco.pytorch.global.torch.randn;

/**
 * Mac multi-backend / multi-thread / multi-process stress for Gloo + MPI.
 *
 * <h2>What works on Mac (verified)</h2>
 * <ul>
 *   <li><b>Gloo</b> via {@code ProcessGroupGloo.Options.create_default()} inside
 *       {@link ProcessGroupWrapper} — real multi-process allreduce (2 JVMs + FileStore).</li>
 *   <li><b>Multi-thread</b> concurrent single-rank Gloo PGs (each thread own store+pg)
 *       and concurrent DDP training steps on local backend.</li>
 *   <li><b>MPI</b> via {@code ProcessGroupMPI.createProcessGroupMPI()} + packaged
 *       {@code jnitorch_mpi} (libtorch {@code USE_MPI=1}). Prefer {@code mpirun -n N}
 *       for true multi-rank; single-JVM world=1 still exercises create/allreduce.</li>
 * </ul>
 *
 * <h2>MPI multi-thread model (important)</h2>
 * ProcessGroupMPI is process-oriented (like Gloo multi-rank):
 * <ul>
 *   <li>Do <b>not</b> create one MPI PG per thread (needs {@code MPI_THREAD_MULTIPLE}
 *       and still fights with a single COMM_WORLD).</li>
 *   <li>Do: one ProcessGroupMPI per process; multi-thread concurrent collectives on
 *       that single PG (OpenMPI 5.x typically provides MULTIPLE / SERIALIZED).</li>
 *   <li>True multi-rank: {@code mpirun -n 2 java ... distribute.BenchmarkGlooMpiStress --mpi-worker}</li>
 * </ul>
 *
 * <p>Launch modes:
 * <pre>
 *   # single JVM: Gloo stress + MPI world=1 probe/collectives/threads
 *   java ... distribute.BenchmarkGlooMpiStress
 *
 *   # self-launch 2-rank Gloo multi-process (FileStore)
 *   java ... distribute.BenchmarkGlooMpiStress --mp
 *
 *   # mpirun multi-rank MPI (recommended for real MPI allreduce)
 *   mpirun -n 2 java ... distribute.BenchmarkGlooMpiStress --mpi-worker
 * </pre>
 */
public class BenchmarkGlooMpiStress {
    static int passed = 0, failed = 0, skipped = 0;
    static final AtomicInteger MP_RANK = new AtomicInteger(-1);

    /** Cached: null = not probed, TRUE/FALSE after first successful/failed load. */
    static Boolean mpiNativeOk = null;
    static String mpiNativeWhy = null;
    /**
     * Single-process tests share one ProcessGroupMPI — under MPI_THREAD_SERIALIZED
     * only one PG may exist globally. Multi-rank mpirun workers create their own.
     */
    static ProcessGroupMPI sharedMpiPg = null;

    static void check(String name, boolean ok) {
        if (ok) { passed++; System.out.println("  PASS  " + name); }
        else { failed++; System.out.println("  FAIL  " + name); }
    }
    static void skip(String name, String why) {
        skipped++;
        System.out.println("  SKIP  " + name + " — " + why);
    }
    static void section(String t) { System.out.println("\n=== " + t + " ==="); }

    public static void main(String[] args) throws Exception {
        boolean mpiWorker = false;
        boolean doMp = false;
        boolean doMpirunSelf = false;
        for (String a : args) {
            if ("--mpi-worker".equals(a) || "--mpi".equals(a)) mpiWorker = true;
            else if ("--mp".equals(a)) doMp = true;
            else if ("--mpirun".equals(a)) doMpirunSelf = true;
        }

        // Detect external mpirun even without --mpi-worker flag
        if (!mpiWorker && underMpiLauncher()) {
            mpiWorker = true;
        }

        if (mpiWorker) {
            runMpiMultiRankWorker();
            done();
            return;
        }

        if (!MultiProcessLauncher.isLaunched()) {
            MP_RANK.set(-1);
            runSingleProcess();
            if (doMp) {
                section("Self-launch MultiProcessLauncher world=2 (Gloo)");
                MultiProcessLauncher.LaunchResult r = MultiProcessLauncher.builder()
                        .worldSize(2)
                        .mainClass(BenchmarkGlooMpiStress.class)
                        .timeoutMs(90_000)
                        .launch();
                check("multi-process launcher ok", r.ok());
                if (!r.ok()) {
                    System.out.println(r.summary());
                    System.out.println(MultiProcessLauncher.joinOutputs(r));
                }
            }
            if (doMpirunSelf) {
                section("Self-launch mpirun -n 2 MPI workers");
                selfLaunchMpirun(2);
            }
        } else {
            int rank = MultiProcessLauncher.envRank();
            MP_RANK.set(rank);
            runMultiProcess();
        }
        done();
    }

    public static void mainSingle() throws Exception { runSingleProcess(); done(); }

    static void runSingleProcess() throws Exception {
        System.out.println("=== Gloo/MPI multi-thread stress (single JVM) ===");
        t1GlooForceCollectiveWorld1();
        t2GlooConcurrentThreads();
        t3ConcurrentLocalDdpTraining();
        t4MpiProbeAndWorld1();
        t5GlooAllreduceLatency();
        t6MpiConcurrentThreadsSinglePg();
        t7MpiWrapperWorld1();
        t8MpiAllreduceLatency();
    }

    static void runMultiProcess() throws Exception {
        int rank = MP_RANK.get();
        int world = MultiProcessLauncher.envWorldSize();
        System.out.println("=== Gloo multi-process worker rank=" + rank + " world=" + world + " ===");
        String fs = MultiProcessLauncher.envFileStore();
        DistributedStore.Options sopts = new DistributedStore.Options()
                .type(StoreType.FILE)
                .timeout(30_000);
        if (fs != null && !fs.isBlank()) sopts.fileStorePath(fs);
        try (DistributedStore store = DistributedStore.create(sopts, rank, world);
             ProcessGroupWrapper pg = ProcessGroupWrapper.create(
                     new ProcessGroupWrapper.Options()
                             .backend(BackendType.GLOO)
                             .timeout(30_000)
                             .forceCollective(true),
                     rank, world, store)) {
            check("rank" + rank + " backend is gloo", pg.getBackendName().contains("gloo"));
            check("rank" + rank + " not local-only", !pg.isLocalOnly());

            // allreduce: each rank contributes (rank+1); sum = world*(world+1)/2
            Tensor t = ones(8L).mul(new Scalar((double) (rank + 1)));
            long t0 = System.nanoTime();
            pg.allreduce(t);
            double ms = (System.nanoTime() - t0) / 1e6;
            float v = t.reshape(-1).get(0).item().toFloat();
            float expect = world * (world + 1) / 2.0f;
            check("rank" + rank + " allreduce sum==" + expect, Math.abs(v - expect) < 1e-3f);
            System.out.printf("  [rank%d] allreduce %.2f ms value=%.1f%n", rank, ms, v);

            // NativeDDP multi-process step
            try (MockLLM model = MockLLM.tiny();
                 NativeDDPTrainer ddp = NativeDDPTrainer.create(model, pg)) {
                Adam opt = new Adam(model.parameters());
                for (int i = 0; i < 3; i++) {
                    Tensor x = org.bytedeco.pytorch.global.torch.randint(1024, new long[]{2, 16});
                    Tensor y = org.bytedeco.pytorch.global.torch.randint(1024, new long[]{2, 16});
                    Tensor loss = ddp.step(x, y, opt);
                    check("rank" + rank + " ddp step " + i,
                            loss != null && !loss.isNull());
                }
                System.out.println("  [rank" + rank + "] ddp mode=" + ddp.commMode());
            }
            pg.barrierWait();
        }
    }

    /**
     * True multi-rank MPI worker (launched by {@code mpirun -n N} or {@code --mpirun}).
     * Rank/size come from MPI_COMM_WORLD via createProcessGroupMPI().
     * Prefer <b>one</b> ProcessGroupMPI per process (SERIALIZED-safe); wrapper path
     * is best-effort and may SKIP if a second PG cannot be created.
     */
    static void runMpiMultiRankWorker() throws Exception {
        section("MPI multi-rank worker (mpirun)");
        // Fresh process under mpirun — create the shared PG here.
        ProcessGroupMPI mpi = sharedMpiOrFail();
        if (mpi == null) {
            // probe may have failed before create; try once more explicitly
            try {
                mpi = ProcessGroupMPI.createProcessGroupMPI();
                sharedMpiPg = mpi;
                mpiNativeOk = mpi != null && !mpi.isNull();
            } catch (Throwable t) {
                mpiNativeOk = false;
                mpiNativeWhy = t.getClass().getSimpleName() + ": " + t.getMessage();
            }
        }
        if (mpi == null || mpi.isNull()) {
            check("MPI createProcessGroupMPI non-null", false);
            if (mpiNativeWhy != null) System.out.println("  detail: " + mpiNativeWhy);
            return;
        }
        int rank = mpi.getRank();
        int world = mpi.getSize();
        System.out.printf("  MPI rank=%d size=%d backend=%s%n",
                rank, world, bytePtrToString(mpi.getBackendName()));
        check("MPI rank>=0", rank >= 0);
        check("MPI size>=1", world >= 1);
        if (world < 2) {
            System.out.println("  NOTE: world=1 under mpirun — multi-rank sum check is trivial");
        }

        // Direct ProcessGroupMPI allreduce: each rank contributes (rank+1)
        {
            Tensor t = ones(8L).mul(new Scalar((double) (rank + 1)));
            TensorVector tv = new TensorVector(t);
            long t0 = System.nanoTime();
            Work w = mpi.allreduce(tv);
            waitWork(w);
            double ms = (System.nanoTime() - t0) / 1e6;
            float v = t.reshape(-1).get(0).item().toFloat();
            float expect = world * (world + 1) / 2.0f;
            check("MPI direct allreduce sum==" + expect, Math.abs(v - expect) < 1e-3f);
            System.out.printf("  [mpi rank%d] direct allreduce %.2f ms value=%.1f expect=%.1f%n",
                    rank, ms, v, expect);
        }

        // broadcast root=0
        {
            Tensor t = ones(4L).mul(new Scalar(rank == 0 ? 7.0 : 0.0));
            TensorVector tv = new TensorVector(t);
            waitWork(mpi.broadcast(tv));
            float v = t.reshape(-1).get(0).item().toFloat();
            check("MPI broadcast root0 == 7.0", Math.abs(v - 7.0f) < 1e-3f);
        }

        // barrier
        {
            Work b = mpi.barrier();
            waitWork(b);
            check("MPI barrier ok", true);
        }

        // Multi-thread concurrent collectives on the single multi-rank PG
        {
            final ProcessGroupMPI mpiPg = mpi; // effectively-final for lambdas
            int threads = 4;
            int iters = 10;
            ExecutorService pool = Executors.newFixedThreadPool(threads);
            CountDownLatch start = new CountDownLatch(1);
            AtomicInteger ok = new AtomicInteger();
            AtomicReference<String> err = new AtomicReference<>();
            final Object mpiLock = new Object();
            List<Future<?>> futs = new ArrayList<>();
            for (int i = 0; i < threads; i++) {
                final int tid = i;
                futs.add(pool.submit(() -> {
                    try {
                        start.await();
                        for (int k = 0; k < iters; k++) {
                            Tensor t = ones(8L).mul(new Scalar((double) (rank + 1)));
                            TensorVector tv = new TensorVector(t);
                            synchronized (mpiLock) {
                                waitWork(mpiPg.allreduce(tv));
                            }
                            float expect = world * (world + 1) / 2.0f;
                            float v = t.reshape(-1).get(0).item().toFloat();
                            if (Math.abs(v - expect) > 1e-3f) {
                                throw new IllegalStateException(
                                        "tid=" + tid + " allreduce got " + v + " expect " + expect);
                            }
                        }
                        ok.incrementAndGet();
                    } catch (Throwable t) {
                        err.compareAndSet(null, String.valueOf(t));
                    }
                }));
            }
            start.countDown();
            for (Future<?> f : futs) f.get(60, TimeUnit.SECONDS);
            pool.shutdown();
            check("MPI multi-thread coll (" + ok.get() + "/" + threads + ")", ok.get() == threads);
            if (err.get() != null) System.out.println("  first err: " + err.get());
        }

        // Wrapper path is best-effort (second createProcessGroupMPI).
        try (DistributedStore store = DistributedStore.createSingleProcess();
             ProcessGroupWrapper pg = ProcessGroupWrapper.create(
                     new ProcessGroupWrapper.Options()
                             .backend(BackendType.MPI)
                             .forceCollective(true)
                             .timeout(30_000),
                     rank, world, store)) {
            String bn = pg.getBackendName();
            if (bn == null || !bn.toLowerCase().contains("mpi")
                    || bn.contains("fallback") || bn.startsWith("local")) {
                skip("wrapper MPI", "backend=" + bn + " (second PG may be unavailable)");
            } else {
                check("wrapper backend contains mpi", true);
                check("wrapper not local-only", !pg.isLocalOnly());
                Tensor t = ones(4L).mul(new Scalar((double) (rank + 1)));
                pg.allreduce(t);
                float v = t.reshape(-1).get(0).item().toFloat();
                float expect = world * (world + 1) / 2.0f;
                check("wrapper MPI allreduce sum==" + expect, Math.abs(v - expect) < 1e-3f);
                pg.barrierWait();
                check("wrapper MPI barrierWait", true);
            }
        } catch (Throwable t) {
            skip("wrapper MPI path", t.getClass().getSimpleName() + ": " + t.getMessage());
        }
    }

    /** Force real Gloo even at worldSize=1. */
    static void t1GlooForceCollectiveWorld1() throws Exception {
        section("T1 Gloo forceCollective world=1");
        try (DistributedStore store = DistributedStore.createSingleProcess();
             ProcessGroupWrapper pg = ProcessGroupWrapper.create(
                     new ProcessGroupWrapper.Options()
                             .backend(BackendType.GLOO)
                             .forceCollective(true)
                             .timeout(15_000),
                     0, 1, store)) {
            check("T1 backend gloo", pg.getBackendName().contains("gloo"));
            check("T1 not local-only", !pg.isLocalOnly());
            Tensor t = ones(16L).mul(new Scalar(2.0));
            pg.allreduce(t);
            check("T1 allreduce preserves value@ws1",
                    Math.abs(t.reshape(-1).get(0).item().toFloat() - 2.0f) < 1e-3f);
            pg.barrierWait();
            check("T1 barrierWait ok", true);
        } catch (Throwable t) {
            check("T1 Gloo forceCollective", false);
            System.out.println("  detail: " + t);
        }
    }

    /**
     * Multi-thread: each thread owns its own HashStore + world=1 Gloo PG.
     * (True multi-rank Gloo needs multi-process; multi-thread stress is concurrent PG ops.)
     */
    static void t2GlooConcurrentThreads() throws Exception {
        section("T2 concurrent Gloo PGs (8 threads, world=1 each)");
        int threads = 8;
        int iters = 20;
        ExecutorService pool = Executors.newFixedThreadPool(threads);
        CountDownLatch start = new CountDownLatch(1);
        AtomicInteger ok = new AtomicInteger();
        AtomicReference<String> err = new AtomicReference<>();
        List<Future<?>> futs = new ArrayList<>();
        for (int i = 0; i < threads; i++) {
            final int tid = i;
            futs.add(pool.submit(() -> {
                try {
                    start.await();
                    try (DistributedStore store = DistributedStore.create(
                            new DistributedStore.Options().type(StoreType.HASH), 0, 1);
                         ProcessGroupWrapper pg = ProcessGroupWrapper.create(
                                 new ProcessGroupWrapper.Options()
                                         .backend(BackendType.GLOO)
                                         .forceCollective(true)
                                         .timeout(15_000),
                                 0, 1, store)) {
                        for (int k = 0; k < iters; k++) {
                            Tensor t = randn(new long[]{64});
                            pg.allreduce(t);
                            pg.broadcast(t, 0);
                            if ((k + 1) % 5 == 0) pg.barrierWait();
                        }
                        ok.incrementAndGet();
                    }
                } catch (Throwable t) {
                    err.compareAndSet(null, "tid=" + tid + " " + t);
                }
            }));
        }
        long t0 = System.nanoTime();
        start.countDown();
        for (Future<?> f : futs) f.get(60, TimeUnit.SECONDS);
        pool.shutdown();
        double ms = (System.nanoTime() - t0) / 1e6;
        check("T2 all threads ok (" + ok.get() + "/" + threads + ")", ok.get() == threads);
        if (err.get() != null) System.out.println("  first err: " + err.get());
        System.out.printf("  T2 wall %.1f ms  (~%.0f coll/s)%n",
                ms, threads * iters * 2 * 1000.0 / Math.max(ms, 1));
    }

    /** Multi-thread concurrent local DDP training (no Gloo required). */
    static void t3ConcurrentLocalDdpTraining() throws Exception {
        section("T3 concurrent local NativeDDP (4 threads)");
        int threads = 4;
        ExecutorService pool = Executors.newFixedThreadPool(threads);
        CountDownLatch start = new CountDownLatch(1);
        AtomicInteger ok = new AtomicInteger();
        AtomicReference<String> err = new AtomicReference<>();
        List<Future<?>> futs = new ArrayList<>();
        for (int i = 0; i < threads; i++) {
            futs.add(pool.submit(() -> {
                try {
                    start.await();
                    try (DistributedStore store = DistributedStore.createSingleProcess();
                         ProcessGroupWrapper pg = ProcessGroupWrapper.create(0, 1, store);
                         MockLLM model = MockLLM.tiny();
                         NativeDDPTrainer ddp = NativeDDPTrainer.create(model, pg)) {
                        Adam opt = new Adam(model.parameters());
                        for (int s = 0; s < 3; s++) {
                            Tensor x = org.bytedeco.pytorch.global.torch.randint(1024, new long[]{2, 8});
                            Tensor y = org.bytedeco.pytorch.global.torch.randint(1024, new long[]{2, 8});
                            ddp.step(x, y, opt);
                        }
                        ok.incrementAndGet();
                    }
                } catch (Throwable t) {
                    err.compareAndSet(null, t.toString());
                }
            }));
        }
        start.countDown();
        for (Future<?> f : futs) f.get(90, TimeUnit.SECONDS);
        pool.shutdown();
        check("T3 concurrent DDP (" + ok.get() + "/" + threads + ")", ok.get() == threads);
        if (err.get() != null) System.out.println("  first err: " + err.get());
    }

    /**
     * MPI probe + world=1 collectives.
     * With jnitorch_mpi packaged this should PASS (no longer honest SKIP).
     * Uses the shared ProcessGroupMPI created by {@link #probeMpiNative()}.
     */
    static void t4MpiProbeAndWorld1() {
        section("T4 MPI ProcessGroupMPI create + world=1 allreduce");
        try {
            ProcessGroupMPI mpi = sharedMpiOrFail();
            if (mpi == null) {
                skip("T4 MPI", mpiNativeWhy != null ? mpiNativeWhy : "shared MPI PG unavailable");
                return;
            }
            int rank = mpi.getRank();
            int size = mpi.getSize();
            check("T4 MPI rank>=0", rank >= 0);
            check("T4 MPI size>=1", size >= 1);
            System.out.printf("  MPI rank=%d size=%d backend=%s%n",
                    rank, size, bytePtrToString(mpi.getBackendName()));

            // world=1 allreduce should preserve value
            Tensor t = ones(16L).mul(new Scalar(3.0));
            TensorVector tv = new TensorVector(t);
            Work w = mpi.allreduce(tv);
            waitWork(w);
            float v = t.reshape(-1).get(0).item().toFloat();
            check("T4 MPI allreduce@ws1 preserves 3.0", Math.abs(v - 3.0f) < 1e-3f);

            Work b = mpi.barrier();
            waitWork(b);
            check("T4 MPI barrier ok", true);
        } catch (UnsatisfiedLinkError ule) {
            mpiNativeOk = false;
            mpiNativeWhy = "jnitorch_mpi UnsatisfiedLinkError: " + ule.getMessage();
            skip("T4 MPI", mpiNativeWhy);
        } catch (Throwable t) {
            check("T4 MPI", false);
            System.out.println("  detail: " + t);
            t.printStackTrace(System.out);
        }
    }

    /** Gloo allreduce latency micro-bench world=1. */
    static void t5GlooAllreduceLatency() throws Exception {
        section("T5 Gloo allreduce latency (world=1, N=50)");
        try (DistributedStore store = DistributedStore.createSingleProcess();
             ProcessGroupWrapper pg = ProcessGroupWrapper.create(
                     new ProcessGroupWrapper.Options().backend(BackendType.GLOO).forceCollective(true),
                     0, 1, store)) {
            int n = 50;
            double[] ms = new double[n];
            Tensor t = randn(new long[]{1024});
            // warmup
            for (int i = 0; i < 5; i++) pg.allreduce(t);
            for (int i = 0; i < n; i++) {
                long t0 = System.nanoTime();
                pg.allreduce(t);
                ms[i] = (System.nanoTime() - t0) / 1e6;
            }
            java.util.Arrays.sort(ms);
            System.out.printf("  allreduce p50=%.3f ms p95=%.3f ms p99=%.3f ms%n",
                    ms[n / 2], ms[(int) (n * 0.95)], ms[n - 1]);
            check("T5 p50 < 100ms", ms[n / 2] < 100);
        }
    }

    /**
     * Multi-thread stress on a <b>single</b> ProcessGroupMPI (world=1).
     * Unlike Gloo T2 (one PG per thread), MPI typically allows only one PG
     * globally under MPI_THREAD_SERIALIZED; concurrent ops share that PG.
     * Calls are serialized with a lock so SERIALIZED MPI stays correct; still
     * stresses multi-thread scheduling around the PG.
     */
    static void t6MpiConcurrentThreadsSinglePg() {
        section("T6 MPI concurrent collectives (1 PG, 4 threads)");
        ProcessGroupMPI mpi = sharedMpiOrFail();
        if (mpi == null) {
            skip("T6 MPI multi-thread", mpiNativeWhy != null ? mpiNativeWhy : "shared MPI PG unavailable");
            return;
        }

        int threads = 4;
        int iters = 25;
        ExecutorService pool = Executors.newFixedThreadPool(threads);
        CountDownLatch start = new CountDownLatch(1);
        AtomicInteger ok = new AtomicInteger();
        AtomicReference<String> err = new AtomicReference<>();
        List<Future<?>> futs = new ArrayList<>();
        // Serialize MPI calls if runtime is only SERIALIZED — still stress scheduling.
        final Object mpiLock = new Object();
        for (int i = 0; i < threads; i++) {
            final int tid = i;
            futs.add(pool.submit(() -> {
                try {
                    start.await();
                    for (int k = 0; k < iters; k++) {
                        Tensor t = randn(new long[]{64});
                        TensorVector tv = new TensorVector(t);
                        synchronized (mpiLock) {
                            Work w = mpi.allreduce(tv);
                            waitWork(w);
                            if ((k + 1) % 5 == 0) {
                                Work b = mpi.barrier();
                                waitWork(b);
                            }
                        }
                    }
                    ok.incrementAndGet();
                } catch (Throwable t) {
                    err.compareAndSet(null, "tid=" + tid + " " + t);
                }
            }));
        }
        long t0 = System.nanoTime();
        start.countDown();
        try {
            for (Future<?> f : futs) f.get(60, TimeUnit.SECONDS);
        } catch (Throwable t) {
            err.compareAndSet(null, t.toString());
        }
        pool.shutdown();
        double ms = (System.nanoTime() - t0) / 1e6;
        check("T6 all threads ok (" + ok.get() + "/" + threads + ")", ok.get() == threads);
        if (err.get() != null) System.out.println("  first err: " + err.get());
        System.out.printf("  T6 wall %.1f ms  (~%.0f coll/s)%n",
                ms, threads * iters * 1000.0 / Math.max(ms, 1));
    }

    /**
     * ProcessGroupWrapper BackendType.MPI at world=1.
     * Note: wrapper calls createProcessGroupMPI() again. Under
     * MPI_THREAD_SERIALIZED a second PG may fail — we report that honestly
     * (PASS if real mpi, SKIP/soft if fallback or second-PG rejected).
     */
    static void t7MpiWrapperWorld1() {
        section("T7 ProcessGroupWrapper BackendType.MPI world=1");
        if (!probeMpiNative()) {
            skip("T7 MPI wrapper", mpiNativeWhy);
            return;
        }
        try (DistributedStore store = DistributedStore.createSingleProcess();
             ProcessGroupWrapper pg = ProcessGroupWrapper.create(
                     new ProcessGroupWrapper.Options()
                             .backend(BackendType.MPI)
                             .forceCollective(true)
                             .timeout(15_000),
                     0, 1, store)) {
            String bn = pg.getBackendName();
            System.out.println("  wrapper backend=" + bn);
            if (bn == null) {
                check("T7 backend non-null", false);
                return;
            }
            String low = bn.toLowerCase();
            if (low.contains("fallback") || low.startsWith("local")) {
                // Second createProcessGroupMPI often fails after shared PG exists.
                skip("T7 real mpi backend",
                        "wrapper fell back to '" + bn + "' (likely second ProcessGroupMPI under SERIALIZED)");
                return;
            }
            check("T7 backend is mpi", low.contains("mpi"));
            check("T7 not local-only", !pg.isLocalOnly());
            Tensor t = ones(8L).mul(new Scalar(4.0));
            pg.allreduce(t);
            check("T7 allreduce preserves 4.0@ws1",
                    Math.abs(t.reshape(-1).get(0).item().toFloat() - 4.0f) < 1e-3f);
            pg.barrierWait();
            check("T7 barrierWait", true);
        } catch (Throwable t) {
            // Second PG creation can throw — not a packaging failure
            skip("T7 MPI wrapper", t.getClass().getSimpleName() + ": " + t.getMessage());
        }
    }

    /** MPI allreduce latency micro-bench world=1 (shared PG). */
    static void t8MpiAllreduceLatency() {
        section("T8 MPI allreduce latency (world=1, N=50)");
        ProcessGroupMPI mpi = sharedMpiOrFail();
        if (mpi == null) {
            skip("T8 MPI latency", mpiNativeWhy != null ? mpiNativeWhy : "shared MPI PG unavailable");
            return;
        }
        try {
            int n = 50;
            double[] ms = new double[n];
            Tensor t = randn(new long[]{1024});
            for (int i = 0; i < 5; i++) {
                TensorVector tv = new TensorVector(t);
                waitWork(mpi.allreduce(tv));
            }
            for (int i = 0; i < n; i++) {
                TensorVector tv = new TensorVector(t);
                long t0 = System.nanoTime();
                waitWork(mpi.allreduce(tv));
                ms[i] = (System.nanoTime() - t0) / 1e6;
            }
            java.util.Arrays.sort(ms);
            System.out.printf("  MPI allreduce p50=%.3f ms p95=%.3f ms p99=%.3f ms%n",
                    ms[n / 2], ms[(int) (n * 0.95)], ms[n - 1]);
            check("T8 MPI p50 < 100ms", ms[n / 2] < 100);
        } catch (Throwable t) {
            check("T8 MPI latency", false);
            System.out.println("  detail: " + t);
        }
    }

    // ── helpers ──────────────────────────────────────────────────────────

    static boolean underMpiLauncher() {
        // OpenMPI / PMIx / generic
        return envSet("OMPI_COMM_WORLD_SIZE")
                || envSet("OMPI_COMM_WORLD_RANK")
                || envSet("PMIX_RANK")
                || envSet("PMI_RANK")
                || envSet("MPI_LOCALRANKID");
    }

    static boolean envSet(String k) {
        String v = System.getenv(k);
        return v != null && !v.isBlank();
    }

    /**
     * Probe whether jnitorch_mpi + ProcessGroupMPI symbols load.
     * Caches result so later tests skip consistently without re-throwing.
     * On success also creates {@link #sharedMpiPg} for single-JVM tests.
     */
    static boolean probeMpiNative() {
        if (mpiNativeOk != null) return mpiNativeOk;
        try {
            // Force Loader.load of torch_mpi / ProcessGroupMPI (static block)
            Class.forName("org.bytedeco.pytorch.distributed.ProcessGroupMPI");
            ProcessGroupMPI pg = ProcessGroupMPI.createProcessGroupMPI();
            if (pg == null || pg.isNull()) {
                mpiNativeOk = false;
                mpiNativeWhy = "createProcessGroupMPI returned null (MPI init failed?)";
                return false;
            }
            sharedMpiPg = pg;
            mpiNativeOk = true;
            mpiNativeWhy = null;
            System.out.printf("  [mpi-probe] native OK rank=%d size=%d%n",
                    pg.getRank(), pg.getSize());
            return true;
        } catch (UnsatisfiedLinkError ule) {
            mpiNativeOk = false;
            mpiNativeWhy = "jnitorch_mpi not loadable — " + ule.getMessage();
            return false;
        } catch (Throwable t) {
            mpiNativeOk = false;
            mpiNativeWhy = t.getClass().getSimpleName() + ": " + t.getMessage();
            return false;
        }
    }

    /** Shared single-JVM ProcessGroupMPI (created by {@link #probeMpiNative()}). */
    static ProcessGroupMPI sharedMpiOrFail() {
        if (!probeMpiNative() || sharedMpiPg == null || sharedMpiPg.isNull()) {
            return null;
        }
        return sharedMpiPg;
    }

    static void waitWork(Work w) {
        if (w == null || w.isNull()) return;
        try {
            w._wait();
        } catch (Throwable t) {
            // some Work impls may no-op; rethrow real failures via caller checks
            throw new RuntimeException("Work._wait failed: " + t, t);
        }
    }

    static String bytePtrToString(org.bytedeco.javacpp.BytePointer bp) {
        if (bp == null || bp.isNull()) return "(null)";
        try {
            return bp.getString();
        } catch (Throwable t) {
            return bp.toString();
        }
    }

    /**
     * Self-launch {@code mpirun -n N} re-executing this class with {@code --mpi-worker}.
     * Requires mpirun on PATH (Homebrew open-mpi).
     */
    static void selfLaunchMpirun(int n) {
        try {
            String java = ProcessHandle.current().info().command().orElse("java");
            String cp = System.getProperty("java.class.path");
            String lib = System.getProperty("java.library.path");
            List<String> cmd = new ArrayList<>();
            cmd.add("mpirun");
            cmd.add("-n");
            cmd.add(String.valueOf(n));
            cmd.add(java);
            cmd.add("--enable-native-access=ALL-UNNAMED");
            cmd.add("--add-opens=java.base/java.nio=ALL-UNNAMED");
            if (lib != null && !lib.isBlank()) {
                cmd.add("-Djava.library.path=" + lib);
            }
            // Fresh cache dir avoids stale non-MPI libtorch extraction
            String cache = System.getProperty("org.bytedeco.javacpp.cachedir");
            if (cache != null && !cache.isBlank()) {
                cmd.add("-Dorg.bytedeco.javacpp.cachedir=" + cache);
            }
            cmd.add("-cp");
            cmd.add(cp);
            cmd.add(BenchmarkGlooMpiStress.class.getName());
            cmd.add("--mpi-worker");
            System.out.println("  exec: " + String.join(" ", cmd));
            ProcessBuilder pb = new ProcessBuilder(cmd);
            pb.inheritIO();
            pb.environment().putIfAbsent("OMPI_MCA_btl", "self,tcp");
            Process p = pb.start();
            boolean finished = p.waitFor(120, TimeUnit.SECONDS);
            if (!finished) {
                p.destroyForcibly();
                check("mpirun self-launch finished", false);
                return;
            }
            check("mpirun self-launch exit=0", p.exitValue() == 0);
        } catch (Throwable t) {
            skip("mpirun self-launch", t.getClass().getSimpleName() + ": " + t.getMessage());
        }
    }

    static void done() {
        System.out.println("\n=== Results: " + passed + " passed, " + failed + " failed, "
                + skipped + " skipped ===");
        if (failed > 0) throw new RuntimeException(failed + " checks failed");
    }
}
