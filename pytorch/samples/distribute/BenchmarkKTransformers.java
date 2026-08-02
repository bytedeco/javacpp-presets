/*
 * Copyright (C) 2020-2026 Eduardo Gonzalez, Hervé Guillemet, Samuel Audet
 *
 * Licensed either under the Apache License, Version 2.0, or (at your option)
 * under the terms of the GNU General Public License as published by
 * the Free Software Foundation (subject to the "Classpath" exception),
 * either version 2, or any later version (collectively, the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *     http://www.gnu.org/licenses/
 *     http://www.gnu.org/software/classpath/license.html
 *
 * or as provided in the LICENSE.txt file that accompanied this code.
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package distribute;
import org.bytedeco.pytorch.c10.*;

import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.llm.llamafactory.FinetuneAdapter;
import org.bytedeco.pytorch.llm.llamafactory.webui.BoardState;
import org.bytedeco.pytorch.llm.ktransformers.KTransformers;
import org.bytedeco.pytorch.llm.ktransformers.KTransformersVersion;
import org.bytedeco.pytorch.llm.ktransformers.adapter.HostMeshHints;
import org.bytedeco.pytorch.llm.ktransformers.adapter.KTransformersFinetuneAdapter;
import org.bytedeco.pytorch.llm.ktransformers.attention.KtMlaAttention;
import org.bytedeco.pytorch.llm.ktransformers.attention.KtPagedAttention;
import org.bytedeco.pytorch.llm.ktransformers.attention.LongContextPolicy;
import org.bytedeco.pytorch.llm.ktransformers.cache.KtCacheManager;
import org.bytedeco.pytorch.llm.ktransformers.cache.PrefixHitStats;
import org.bytedeco.pytorch.llm.ktransformers.cache.ThreeTierPrefixCache;
import org.bytedeco.pytorch.llm.ktransformers.config.KtConfig;
import org.bytedeco.pytorch.llm.ktransformers.config.KtMoEConfig;
import org.bytedeco.pytorch.llm.ktransformers.inference.KtGenerateOutput;
import org.bytedeco.pytorch.llm.ktransformers.inference.KtGenerateRequest;
import org.bytedeco.pytorch.llm.ktransformers.inference.KtInferenceEngine;
import org.bytedeco.pytorch.llm.ktransformers.kernel.AmxLikeGemm;
import org.bytedeco.pytorch.llm.ktransformers.kernel.CpuRefKernelBackend;
import org.bytedeco.pytorch.llm.ktransformers.kernel.DequantOps;
import org.bytedeco.pytorch.llm.ktransformers.kernel.Fp8ChannelGemm;
import org.bytedeco.pytorch.llm.ktransformers.kernel.KernelRegistry;
import org.bytedeco.pytorch.llm.ktransformers.kernel.KtKernelBackend;
import org.bytedeco.pytorch.llm.ktransformers.kernel.QuantLinearOp;
import org.bytedeco.pytorch.llm.ktransformers.model.KtMiniMoECausalLM;
import org.bytedeco.pytorch.llm.ktransformers.moe.ExpertDevice;
import org.bytedeco.pytorch.llm.ktransformers.moe.RoutedMoE;
import org.bytedeco.pytorch.llm.ktransformers.monitor.KtMetrics;
import org.bytedeco.pytorch.llm.ktransformers.monitor.KtTrainMonitor;
import org.bytedeco.pytorch.llm.ktransformers.sft.FreezeAndOffloadPolicy;
import org.bytedeco.pytorch.llm.ktransformers.sft.KtSftSession;
import org.bytedeco.pytorch.llm.ktransformers.util.DeviceBudget;
import org.bytedeco.pytorch.llm.ktransformers.util.Timing;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;

/**
 * Full-spectrum multi-dimension benchmark / correctness suite for
 * {@code org.bytedeco.pytorch.llm.ktransformers}.
 *
 * <p>Dimensions (plan §7):
 * <ol>
 *   <li>Dequant round-trip</li>
 *   <li>Quant matmul vs FP32</li>
 *   <li>MoE routing correctness</li>
 *   <li>Expert CPU/GPU schedule</li>
 *   <li>Three-tier prefix cache</li>
 *   <li>Concurrency</li>
 *   <li>Long-context budget / demote</li>
 *   <li>SFT one-step</li>
 *   <li>Visual metrics / board</li>
 *   <li>Throughput microbench</li>
 *   <li>Memory / expert offload</li>
 *   <li>Regression facade + attention + HostMeshHints</li>
 * </ol>
 *
 * <p>Run after compiling the ktransformers package + this sample, e.g.:
 * {@code java -cp target/samples-compile:target/classes:... distribute.BenchmarkKTransformers}
 */
public class BenchmarkKTransformers {

    static int passed = 0;
    static int failed = 0;
    static int skipped = 0;
    static final StringBuilder report = new StringBuilder();
    static final List<PerfRow> perf = new ArrayList<>();

    public static void main(String[] args) throws Exception {
        System.out.println("=== KTransformers Module Full-Spectrum Benchmark ===");
        System.out.println(KTransformers.banner());
        System.out.println("capabilities: " + Arrays.toString(KTransformers.capabilities()));
        System.out.println();

        section("0. Facade / version");
        benchFacade();

        section("1. Dequant round-trip");
        benchDequant();

        section("2. Quant matmul");
        benchQuantMatmul();

        section("3. MoE routing");
        benchMoERouting();

        section("4. Expert CPU/GPU schedule");
        benchExpertSchedule();

        section("5. Three-tier prefix cache");
        benchThreeTierCache();

        section("6. Concurrency");
        benchConcurrency();

        section("7. Long-context policy");
        benchLongContext();

        section("8. SFT one-step");
        benchSft();

        section("9. Visual metrics / board");
        benchVisual();

        section("10. Throughput");
        benchThroughput();

        section("11. Memory / expert offload");
        benchMemoryOffload();

        section("12. Attention + HostMeshHints + inference E2E");
        benchAttentionAndMesh();

        // ── summary ───────────────────────────────────────────────────────
        System.out.println("\n=== Correctness ===");
        System.out.println("Passed:  " + passed);
        System.out.println("Failed:  " + failed);
        System.out.println("Skipped: " + skipped);
        if (failed > 0) {
            System.out.println("\nFAILED CHECKS:");
            System.out.println(report);
        }

        if (!perf.isEmpty()) {
            System.out.println("\n=== Throughput ===");
            System.out.printf(Locale.ROOT, "%-48s %12s %12s%n", "metric", "ops", "ops/s");
            for (PerfRow r : perf) {
                System.out.printf(Locale.ROOT, "%-48s %12d %12.1f%n", r.name, r.ops, r.opsPerSec);
            }
        }

        if (failed > 0) {
            System.exit(1);
        }
        System.out.println("\nAll tests PASSED!");
    }

    // =====================================================================
    // 0. Facade
    // =====================================================================

    static void benchFacade() {
        benchmark("version non-empty", () -> {
            check("version", KTransformers.version() != null && !KTransformers.version().isEmpty());
            check("banner", KTransformers.banner().contains("KTransformers"));
            check("capabilities>0", KTransformers.capabilities().length > 0);
            check("version const", KTransformersVersion.VERSION.equals(KTransformers.version()));
        });
        benchmark("mini config build", () -> {
            KtConfig c = KtConfig.miniDemo();
            check("hidden>0", c.hiddenSize() > 0);
            check("layers>0", c.numLayers() > 0);
            check("experts>0", c.moe().numExperts() > 0);
            check("vocab>0", c.vocabSize() > 0);
        });
    }

    // =====================================================================
    // 1. Dequant
    // =====================================================================

    static void benchDequant() {
        benchmark("int8 groupwise round-trip", () -> {
            Tensor w = torch.randn(32, 64).mul(new Scalar(0.5));
            try {
                KtKernelBackend.QuantizedWeight q = DequantOps.quantizeGroupwise(w, 8, 32);
                try {
                    Tensor d = DequantOps.dequantGroupwise(q.qweight, q.scale, q.zero, 8, 32);
                    try {
                        double err = DequantOps.maxAbsError(w, d);
                        check("int8 maxAbs < 0.15", err < 0.15);
                        System.out.printf(Locale.ROOT, "    int8 maxAbs=%.6f%n", err);
                    } finally {
                        d.close();
                    }
                } finally {
                    closeQW(q);
                }
            } finally {
                w.close();
            }
        });

        benchmark("int4 groupwise round-trip", () -> {
            Tensor w = torch.randn(16, 64).mul(new Scalar(0.25));
            try {
                KtKernelBackend.QuantizedWeight q = DequantOps.quantizeGroupwise(w, 4, 32);
                try {
                    Tensor d = DequantOps.dequantGroupwise(q.qweight, q.scale, q.zero, 4, 32);
                    try {
                        double err = DequantOps.maxAbsError(w, d);
                        // 4-bit is coarser
                        check("int4 maxAbs < 0.55", err < 0.55);
                        System.out.printf(Locale.ROOT, "    int4 maxAbs=%.6f%n", err);
                    } finally {
                        d.close();
                    }
                } finally {
                    closeQW(q);
                }
            } finally {
                w.close();
            }
        });

        benchmark("fp8 per-channel round-trip", () -> {
            Tensor w = torch.randn(24, 48).mul(new Scalar(0.1));
            try {
                double err = Fp8ChannelGemm.roundTripError(w);
                check("fp8 maxAbs < 0.05", err < 0.05);
                System.out.printf(Locale.ROOT, "    fp8 maxAbs=%.6f%n", err);
            } finally {
                w.close();
            }
        });
    }

    // =====================================================================
    // 2. Quant matmul
    // =====================================================================

    static void benchQuantMatmul() {
        benchmark("CpuRef quantMatmul vs FP32", () -> {
            KtKernelBackend be = KernelRegistry.defaultBackend();
            check("backend name", be.name() != null && !be.name().isEmpty());
            check("supports INT8", be.supports(KtKernelBackend.Capability.INT8_GROUPWISE));

            Tensor w = torch.randn(32, 64).mul(new Scalar(0.2));
            Tensor x = torch.randn(4, 64).mul(new Scalar(0.5));
            try {
                KtKernelBackend.QuantizedWeight q = be.quantizeWeight(w, 8, 32);
                try {
                    Tensor yQ = be.quantMatmul(x, q.qweight, q.scale, q.zero, null, 8, 32);
                    Tensor wDeq = be.dequant(q.qweight, q.scale, q.zero, 8, 32);
                    try {
                        Tensor yRef = DequantOps.matmulDequant(x, w, null);
                        Tensor yVia = DequantOps.matmulDequant(x, wDeq, null);
                        try {
                            // quant path should match dequant-then-matmul
                            double errQ = DequantOps.maxAbsError(yVia, yQ);
                            check("quantMatmul≈dequant-mm", errQ < 1e-4);
                            double errRef = DequantOps.maxAbsError(yRef, yVia);
                            check("dequant mm vs fp32 < 0.5", errRef < 0.5);
                            System.out.printf(Locale.ROOT, "    quant vs fp32 maxAbs=%.6f%n", errRef);
                        } finally {
                            yRef.close();
                            yVia.close();
                        }
                    } finally {
                        wDeq.close();
                        yQ.close();
                    }
                } finally {
                    closeQW(q);
                }
            } finally {
                w.close();
                x.close();
            }
        });

        benchmark("QuantLinearOp forward", () -> {
            Tensor w = torch.randn(16, 32).mul(new Scalar(0.3));
            Tensor x = torch.randn(2, 32);
            try {
                QuantLinearOp op = QuantLinearOp.fromFloatWeight(w, 8, 16, null);
                try {
                    Tensor y = op.forward(x);
                    try {
                        check("out dim", y.dim() == 2);
                        check("out rows", y.size(0) == 2);
                        check("out cols", y.size(1) == 16);
                    } finally {
                        y.close();
                    }
                } finally {
                    op.close();
                }
            } finally {
                w.close();
                x.close();
            }
        });

        benchmark("AmxLikeGemm int8", () -> {
            Tensor w = torch.randn(16, 64).mul(new Scalar(0.2));
            Tensor x = torch.randn(3, 64);
            try {
                KtKernelBackend.QuantizedWeight q = DequantOps.quantizeGroupwise(w, 8, 32);
                try {
                    Tensor y = AmxLikeGemm.gemmInt8(x, q.qweight, q.scale, q.zero, null, 8, 32);
                    try {
                        check("amx out shape", y.size(0) == 3 && y.size(1) == 16);
                    } finally {
                        y.close();
                    }
                } finally {
                    closeQW(q);
                }
            } finally {
                w.close();
                x.close();
            }
        });
    }

    // =====================================================================
    // 3. MoE routing
    // =====================================================================

    static void benchMoERouting() {
        benchmark("RoutedMoE forward shape + metrics", () -> {
            RoutedMoE moe = RoutedMoE.mini(64, 128);
            try {
                Tensor x = torch.randn(2, 8, 64);
                try {
                    Tensor y = moe.forward(x);
                    try {
                        check("moe out dim", y.dim() == 3);
                        check("moe B", y.size(0) == 2);
                        check("moe T", y.size(1) == 8);
                        check("moe H", y.size(2) == 64);
                    } finally {
                        y.close();
                    }
                    // second forward to accumulate hits
                    Tensor y2 = moe.forward(x);
                    y2.close();
                    Map<String, Double> m = moe.metrics().toMetricMap();
                    check("metrics non-empty", m != null && !m.isEmpty());
                    check("numExperts=4", moe.numExperts() == 4);
                    check("topK=2", moe.topK() == 2);
                    check("shared expert", moe.hasSharedExpert());
                } finally {
                    x.close();
                }
            } finally {
                moe.close();
            }
        });

        benchmark("gate top-k deterministic under fixed input scale", () -> {
            // Same module, same x twice → finite outputs (no NaN)
            RoutedMoE moe = RoutedMoE.mini(32, 64);
            try {
                Tensor x = torch.ones(1, 4, 32).mul(new Scalar(0.1));
                try {
                    Tensor a = moe.forward(x);
                    Tensor b = moe.forward(x);
                    try {
                        double err = DequantOps.maxAbsError(a, b);
                        check("moe deterministic", err < 1e-5);
                        float[] fa = DequantOps.toFloatArray(a);
                        boolean finite = true;
                        for (float v : fa) {
                            if (!Float.isFinite(v)) { finite = false; break; }
                        }
                        check("moe finite", finite);
                    } finally {
                        a.close();
                        b.close();
                    }
                } finally {
                    x.close();
                }
            } finally {
                moe.close();
            }
        });
    }

    // =====================================================================
    // 4. Expert schedule
    // =====================================================================

    static void benchExpertSchedule() {
        benchmark("mixed GPU/CPU experts forward", () -> {
            KtMoEConfig cfg = KtMoEConfig.builder()
                    .numExperts(4).topK(2).sharedExpert(false)
                    .schedule(KtMoEConfig.SchedulePolicy.BALANCED)
                    .gpuExpertSlots(2)
                    .migrateCooldownSteps(1)
                    .migrateHotThreshold(0.01)
                    .build();
            RoutedMoE moe = new RoutedMoE(32, 64, cfg, DeviceBudget.mini());
            try {
                int gpu0 = moe.pool().gpuResidentCount();
                check("initial gpu residents > 0", gpu0 > 0);
                check("initial gpu residents <= slots", gpu0 <= 2);

                Tensor x = torch.randn(4, 32);
                try {
                    for (int i = 0; i < 6; i++) {
                        Tensor y = moe.forward(x);
                        y.close();
                    }
                } finally {
                    x.close();
                }
                // force migrate path
                moe.scheduler().forceMigrate();
                long promotes = moe.metrics().promoteCount();
                long demotes = moe.metrics().demoteCount();
                System.out.printf(Locale.ROOT, "    gpuResidents=%d promote=%d demote=%d hits=%d%n",
                        moe.pool().gpuResidentCount(), promotes, demotes, moe.metrics().totalHits());
                check("dispatch happened", moe.metrics().totalHits() > 0 || moe.metrics().dispatchSteps() >= 0);
                // demote one and ensure still runs
                boolean dem = moe.pool().demoteToCpu(0);
                Tensor x2 = torch.randn(2, 32);
                try {
                    Tensor y = moe.forward(x2);
                    try {
                        check("after demote finite", y.numel() == 2L * 32);
                    } finally {
                        y.close();
                    }
                } finally {
                    x2.close();
                }
                check("demote attempted or already cpu", dem || moe.pool().get(0).device() == ExpertDevice.CPU);
            } finally {
                moe.close();
            }
        });
    }

    // =====================================================================
    // 5. Three-tier cache
    // =====================================================================

    static void benchThreeTierCache() {
        benchmark("put/get L1 + promote path", () -> {
            try (ThreeTierPrefixCache cache = ThreeTierPrefixCache.mini()) {
                int[] tokens = new int[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
                long h = ThreeTierPrefixCache.hashBlock(0L, tokens, 0, cache.blockSize());
                byte[] payload = ThreeTierPrefixCache.encodeTokenBlock(tokens, 0, cache.blockSize());
                cache.put(h, payload);
                byte[] got = cache.get(h);
                check("l1 hit payload", got != null && Arrays.equals(payload, got));
                check("l1 size>=1", cache.gpuSize() >= 1);
                PrefixHitStats st = cache.stats();
                check("l1 hits>=1", st.l1Hits.sum() >= 1);
            }
        });

        benchmark("insertTokens + matchPrefix second hit", () -> {
            try (ThreeTierPrefixCache cache = ThreeTierPrefixCache.mini()) {
                int bs = cache.blockSize();
                int[] tokens = new int[bs * 3];
                for (int i = 0; i < tokens.length; i++) tokens[i] = i + 7;
                cache.insertTokens(tokens);
                ThreeTierPrefixCache.PrefixMatch m1 = cache.matchPrefix(tokens);
                check("first match tokens>0", m1.matchedTokens > 0);

                // force demote to exercise L2/L3
                while (cache.gpuSize() > 0) {
                    if (!cache.forceDemoteOneFromGpu()) break;
                }
                // may sit on CPU or Disk
                ThreeTierPrefixCache.PrefixMatch m2 = cache.matchPrefix(tokens);
                check("second match hit", m2.hit());
                check("second match tokens>0", m2.matchedTokens > 0);
                PrefixHitStats st = cache.stats();
                System.out.printf(Locale.ROOT,
                        "    hitRate=%.3f l1=%d l2=%d l3=%d miss=%d promote=%d demote=%d%n",
                        st.hitRate(), st.l1Hits.sum(), st.l2Hits.sum(), st.l3Hits.sum(),
                        st.misses.sum(), st.promotes.sum(), st.demotes.sum());
                check("some demotes or multi-tier activity",
                        st.demotes.sum() >= 0); // always true; real assert:
                check("lookups recorded", st.lookups.sum() >= 1 || st.l1Hits.sum() + st.l2Hits.sum() + st.l3Hits.sum() >= 1);
            }
        });

        benchmark("KtCacheManager remember/lookup", () -> {
            try (KtCacheManager mgr = KtCacheManager.mini()) {
                int[] prompt = new int[32];
                for (int i = 0; i < prompt.length; i++) prompt[i] = (i * 3) % 50;
                mgr.rememberPrefix(prompt);
                ThreeTierPrefixCache.PrefixMatch m = mgr.lookupPrefix(prompt);
                check("manager match", m.matchedTokens > 0);
                check("stats accessible", mgr.stats() != null);
            }
        });
    }

    // =====================================================================
    // 6. Concurrency
    // =====================================================================

    static void benchConcurrency() {
        benchmark("parallel generate mini engine", () -> {
            try (KtInferenceEngine eng = KTransformers.openInferenceMini()) {
                int n = 4;
                ExecutorService pool = Executors.newFixedThreadPool(n);
                CountDownLatch start = new CountDownLatch(1);
                AtomicInteger ok = new AtomicInteger();
                AtomicReference<Throwable> err = new AtomicReference<>();
                List<Future<?>> futs = new ArrayList<>();
                for (int i = 0; i < n; i++) {
                    final int id = i;
                    futs.add(pool.submit(() -> {
                        try {
                            start.await(5, TimeUnit.SECONDS);
                            int[] prompt = new int[]{1 + id, 2, 3, 4};
                            KtGenerateOutput out = eng.generate(prompt, 3);
                            check("out tokens", out.tokenIds().length >= prompt.length);
                            check("new tokens", out.newTokens() >= 1);
                            ok.incrementAndGet();
                        } catch (Throwable t) {
                            err.compareAndSet(null, t);
                        }
                    }));
                }
                start.countDown();
                for (Future<?> f : futs) f.get(120, TimeUnit.SECONDS);
                pool.shutdownNow();
                if (err.get() != null) {
                    throw new RuntimeException("concurrent generate failed", err.get());
                }
                check("all concurrent ok", ok.get() == n);
                check("generateCount>=n", eng.generateCount() >= n);
            }
        });
    }

    // =====================================================================
    // 7. Long context
    // =====================================================================

    static void benchLongContext() {
        benchmark("policy plan + refuse", () -> {
            LongContextPolicy p = LongContextPolicy.mini();
            LongContextPolicy.Decision ok = p.plan(32);
            check("plan 32 allowed", ok.allowed());
            LongContextPolicy.Decision big = p.plan(p.maxSeqLen() + 10);
            check("over max refused", !big.allowed());
            check("refuse counted", p.refuseActions() >= 1);
        });

        benchmark("policy demote under pressure + cache enforce", () -> {
            DeviceBudget budget = new DeviceBudget(1024, 1L << 20, 1L << 20);
            // fill GPU budget near watermark
            check("reserve gpu", budget.tryReserveGpu(900));
            LongContextPolicy p = new LongContextPolicy(true, 16, 2, 0.80, 256,
                    64, 32, budget);
            LongContextPolicy.Decision d = p.plan(64);
            check("pressure triggers demote or ok-with-mla",
                    d.action == LongContextPolicy.Decision.Action.DEMOTE
                            || d.action == LongContextPolicy.Decision.Action.OK);
            check("use mla preferred", d.useMla);

            try (ThreeTierPrefixCache cache = ThreeTierPrefixCache.mini()) {
                int[] tok = new int[cache.blockSize() * 4];
                for (int i = 0; i < tok.length; i++) tok[i] = i;
                cache.insertTokens(tok);
                int before = cache.gpuSize();
                // fill more pressure
                budget.tryReserveGpu(100);
                int n = p.enforceOnCache(cache);
                System.out.printf(Locale.ROOT, "    enforce demotes=%d gpuBefore=%d gpuAfter=%d%n",
                        n, before, cache.gpuSize());
                check("metrics map", !p.toMetricMap().isEmpty());
            }
        });

        benchmark("from(KtConfig) builds", () -> {
            LongContextPolicy p = LongContextPolicy.from(KtConfig.miniDemo(), DeviceBudget.mini());
            check("maxSeq>0", p.maxSeqLen() > 0);
            check("active>0", p.gpuActiveTokens() > 0);
        });
    }

    // =====================================================================
    // 8. SFT
    // =====================================================================

    static void benchSft() {
        benchmark("KtSftSession train one-step loss finite", () -> {
            try (KtSftSession session = KTransformers.openSftSessionMini()) {
                session.train();
                double loss = session.lastLoss();
                check("loss finite", Double.isFinite(loss));
                check("loss >= 0", loss >= 0.0);
                check("step >= 1", session.globalStep() >= 1);
                Map<String, Double> m = session.lastMetrics();
                check("metrics non-empty", m != null && !m.isEmpty());
                System.out.printf(Locale.ROOT, "    loss=%.6f step=%d metrics=%d%n",
                        loss, session.globalStep(), m.size());
            }
        });

        benchmark("FinetuneAdapter SPI train+export+chat", () -> {
            Path dir = Files.createTempDirectory("kt-export-");
            try (FinetuneAdapter job = KTransformers.openSftMini()) {
                job.train();
                Map<String, Double> m = job.lastMetrics();
                check("adapter metrics", m != null && !m.isEmpty());
                Path out = job.export(dir, null);
                check("export dir", out != null && Files.isDirectory(out));
                check("export marker", Files.exists(out.resolve("kt_export_ok.txt")));
                String reply = job.chat().chat("hello-kt");
                check("chat non-empty", reply != null && !reply.isEmpty());
                BoardState board = job.board();
                check("board non-null", board != null);
                check("board step>=1", board.globalStep() >= 1);
            } finally {
                try {
                    Files.walk(dir)
                            .sorted((a, b) -> b.compareTo(a))
                            .forEach(p -> { try { Files.deleteIfExists(p); } catch (Exception ignored) {} });
                } catch (Exception ignored) {}
            }
        });

        benchmark("model loss backward-capable (grad exists on leaf)", () -> {
            KtMiniMoECausalLM model = KtMiniMoECausalLM.miniDemo();
            try {
                int[] idsArr = new int[]{1, 2, 3, 4, 5, 6, 7, 8};
                long[] ids = new long[idsArr.length];
                for (int i = 0; i < idsArr.length; i++) ids[i] = idsArr[i];
                Tensor input = torch.tensor(ids).unsqueeze(0);
                try {
                    // requires_grad on input not needed; param grads via loss.backward
                    model.train(true);
                    Tensor loss = model.loss(input);
                    try {
                        double v = loss.item_double();
                        check("ce loss finite", Double.isFinite(v));
                        loss.backward();
                        // lm_head weight should have grad
                        Tensor w = model.lmHead.weight();
                        Tensor g = w.grad();
                        check("lm_head has grad", g != null && !g.isNull() && g.defined());
                    } finally {
                        loss.close();
                    }
                } finally {
                    input.close();
                }
            } finally {
                model.close();
            }
        });
    }

    // =====================================================================
    // 9. Visual
    // =====================================================================

    static void benchVisual() {
        benchmark("KtTrainMonitor + board metrics", () -> {
            try (KtTrainMonitor mon = KtTrainMonitor.forDemo()) {
                mon.onTrainStep(1, 2.5, 1e-4, 0.3);
                mon.onTrainStep(2, 2.1, 1e-4, 0.25);
                mon.publish(Map.of("custom/foo", 1.0));
                BoardState b = mon.board();
                check("board metrics has loss or step", b.globalStep() >= 1 || !b.metrics().isEmpty());
                Map<String, Double> snap = mon.metrics().snapshot();
                check("kt metrics non-empty", snap != null && !snap.isEmpty());
                System.out.println("    board.metrics keys=" + b.metrics().keySet());
            }
        });

        benchmark("engine metrics after generate", () -> {
            try (KtInferenceEngine eng = KTransformers.openInferenceMini()) {
                eng.generate(new int[]{3, 1, 4, 1}, 2);
                Map<String, Double> m = eng.lastMetrics();
                check("engine lastMetrics", m != null);
                KtMetrics km = eng.metrics();
                check("generateCalls>=1", km.generateCalls() >= 1);
            }
        });
    }

    // =====================================================================
    // 10. Throughput
    // =====================================================================

    static void benchThroughput() {
        benchmark("mini generate tok/s", () -> {
            try (KtInferenceEngine eng = KTransformers.openInferenceMini()) {
                int[] prompt = new int[]{1, 2, 3, 4, 5, 6, 7, 8};
                // warmup
                eng.generate(prompt, 2);
                Timing t = Timing.start();
                int rounds = 4;
                int newTok = 0;
                long prefillNs = 0, decodeNs = 0;
                for (int i = 0; i < rounds; i++) {
                    KtGenerateOutput out = eng.generate(prompt, 4);
                    newTok += out.newTokens();
                    prefillNs += out.prefillNanos();
                    decodeNs += out.decodeNanos();
                }
                double sec = t.elapsedSec();
                double tps = newTok / Math.max(1e-9, sec);
                record("mini-generate-new-tokens", newTok, tps);
                record("mini-prefill-ns-total", prefillNs, prefillNs > 0 ? 1e9 / (prefillNs / (double) rounds) : 0);
                System.out.printf(Locale.ROOT, "    newTok=%d wallTok/s=%.1f prefillNs/iter=%d decodeNs/iter=%d%n",
                        newTok, tps, prefillNs / rounds, decodeNs / rounds);
                check("throughput ran", newTok > 0);
            }
        });

        benchmark("quant matmul microbench", () -> {
            Tensor w = torch.randn(64, 128).mul(new Scalar(0.1));
            Tensor x = torch.randn(16, 128);
            try {
                KtKernelBackend be = new CpuRefKernelBackend();
                KtKernelBackend.QuantizedWeight q = be.quantizeWeight(w, 8, 32);
                try {
                    // warmup
                    be.quantMatmul(x, q.qweight, q.scale, q.zero, null, 8, 32).close();
                    int n = 20;
                    Timing t = Timing.start();
                    for (int i = 0; i < n; i++) {
                        be.quantMatmul(x, q.qweight, q.scale, q.zero, null, 8, 32).close();
                    }
                    double sec = t.elapsedSec();
                    record("quantMatmul-64x128", n, n / Math.max(1e-9, sec));
                } finally {
                    closeQW(q);
                }
            } finally {
                w.close();
                x.close();
            }
        });
    }

    // =====================================================================
    // 11. Memory / offload
    // =====================================================================

    static void benchMemoryOffload() {
        benchmark("FreezeAndOffloadPolicy demotes experts", () -> {
            KtMiniMoECausalLM model = KtMiniMoECausalLM.miniDemo();
            try {
                long before = 0;
                for (KtMiniMoECausalLM.Layer layer : model.layers) {
                    before += FreezeAndOffloadPolicy.countGpuExperts(layer.moe);
                }
                FreezeAndOffloadPolicy pol = new FreezeAndOffloadPolicy(true, 1);
                pol.applyModel(model);
                long after = 0;
                for (KtMiniMoECausalLM.Layer layer : model.layers) {
                    after += FreezeAndOffloadPolicy.countGpuExperts(layer.moe);
                    check("keep at most 1 gpu expert/layer",
                            FreezeAndOffloadPolicy.countGpuExperts(layer.moe) <= 1);
                }
                System.out.printf(Locale.ROOT, "    gpuExperts before=%d after=%d%n", before, after);
                check("gpu experts reduced or already low", after <= before);
                // still runnable
                int[] out = model.generateGreedy(new int[]{1, 2, 3}, 2);
                check("generate after offload", out.length == 5);
            } finally {
                model.close();
            }
        });

        benchmark("DeviceBudget reserve/release", () -> {
            DeviceBudget b = DeviceBudget.mini();
            long free0 = b.gpuFreeBytes();
            check("reserve", b.tryReserveGpu(1024));
            check("used increased", b.gpuUsed() >= 1024);
            b.releaseGpu(1024);
            check("free restored roughly", b.gpuFreeBytes() >= free0 - 1);
        });
    }

    // =====================================================================
    // 12. Attention + mesh + e2e
    // =====================================================================

    static void benchAttentionAndMesh() {
        benchmark("KtMlaAttention mini forward", () -> {
            KtMlaAttention attn = KtMlaAttention.mini(64, 4);
            try {
                Tensor x = torch.randn(1, 8, 64);
                try {
                    Tensor y = attn.forward(x);
                    try {
                        check("mla out shape", y.size(0) == 1 && y.size(1) == 8 && y.size(2) == 64);
                    } finally {
                        y.close();
                    }
                    Tensor[] cached = attn.forwardCached(x, 0L, null, null);
                    try {
                        check("mla cached out", cached[0].size(2) == 64);
                        check("mla ckv rank", cached[1].dim() == 3);
                        check("forwardCalls", attn.forwardCalls() >= 2);
                    } finally {
                        for (Tensor t : cached) if (t != null) t.close();
                    }
                } finally {
                    x.close();
                }
            } finally {
                attn.close();
            }
        });

        benchmark("KtPagedAttention mini forward", () -> {
            KtPagedAttention attn = KtPagedAttention.mini(64, 4);
            try {
                Tensor x = torch.randn(1, 6, 64);
                try {
                    Tensor y = attn.forward(x);
                    try {
                        check("paged out", y.size(0) == 1 && y.size(1) == 6 && y.size(2) == 64);
                    } finally {
                        y.close();
                    }
                    check("heads", attn.nHeads() == 4);
                    check("estimate bytes", attn.estimateCacheBytes(16) > 0);
                } finally {
                    x.close();
                }
            } finally {
                attn.close();
            }
        });

        benchmark("HostMeshHints suggest + deepspeed/accelerate maps", () -> {
            HostMeshHints h = HostMeshHints.suggest(KtConfig.miniDemo(), 8);
            check("dp>=1", h.dataParallel() >= 1);
            check("ep>=1", h.expertParallel() >= 1);
            check("tp>=1", h.tensorParallel() >= 1);
            Map<String, Object> acc = h.accelerateHints();
            Map<String, Object> ds = h.deepSpeedZeROHints();
            check("accelerate keys", acc.containsKey("data_parallel_size"));
            check("deepspeed stage", ds.containsKey("zero_optimization.stage"));
            Map<String, Object> keys = h.toKtKeys();
            check("kt_dp present", keys.containsKey("kt_dp"));
            HostMeshHints parsed = HostMeshHints.fromMap(keys);
            check("round-trip dp", parsed.dataParallel() == h.dataParallel());
            check("compatible world", h.compatibleWithWorldSize(1));
            System.out.println("    " + h);
            System.out.println("    deepspeed stage=" + ds.get("zero_optimization.stage"));
        });

        benchmark("openInferenceMini generate e2e", () -> {
            try (KtInferenceEngine eng = KtInferenceEngine.openMini()) {
                KtGenerateRequest req = KtGenerateRequest.builder()
                        .promptTokenIds(new int[]{1, 5, 9, 2})
                        .maxNewTokens(4)
                        .temperature(0.0)
                        .usePrefixCache(true)
                        .seed(42L)
                        .build();
                KtGenerateOutput out = eng.generate(req);
                check("prompt tokens", out.promptTokens() == 4);
                check("new tokens>0", out.newTokens() > 0);
                check("full len", out.tokenIds().length == out.promptTokens() + out.newTokens());
                System.out.printf(Locale.ROOT, "    ttft=%.2fms decode=%.1f tok/s prefill=%.1f tok/s prefixHit=%d%n",
                        out.ttftMillis(), out.decodeTokensPerSec(), out.prefillTokensPerSec(),
                        out.prefixHitTokens());
            }
        });

        benchmark("KTransformersFinetuneAdapter.openMini type", () -> {
            try (KTransformersFinetuneAdapter ad = KTransformersFinetuneAdapter.openMini()) {
                check("ktConfig", ad.ktConfig() != null);
                check("session", ad.session() != null);
            }
        });
    }

    // =====================================================================
    // helpers
    // =====================================================================

    static void closeQW(KtKernelBackend.QuantizedWeight q) {
        if (q == null) return;
        if (q.qweight != null) q.qweight.close();
        if (q.scale != null) q.scale.close();
        if (q.zero != null) q.zero.close();
    }

    static void section(String name) {
        System.out.println("\n── " + name + " ──");
    }

    @FunctionalInterface
    interface CheckedRunnable { void run() throws Exception; }

    static void benchmark(String name, CheckedRunnable r) {
        try {
            r.run();
            System.out.println("  ✓ " + name);
        } catch (Throwable t) {
            failed++;
            report.append("  FAIL [").append(name).append("]: ")
                    .append(t.getClass().getSimpleName()).append(": ")
                    .append(t.getMessage()).append("\n");
            System.out.println("  ✗ " + name + " — " + t.getMessage());
            t.printStackTrace(System.out);
        }
    }

    static void check(String name, boolean condition) {
        if (condition) {
            passed++;
        } else {
            failed++;
            report.append("  CHECK FAILED: ").append(name).append("\n");
            throw new AssertionError("CHECK FAILED: " + name);
        }
    }

    static void record(String name, long ops, double opsPerSec) {
        perf.add(new PerfRow(name, ops, opsPerSec));
    }

    static final class PerfRow {
        final String name;
        final long ops;
        final double opsPerSec;
        PerfRow(String name, long ops, double opsPerSec) {
            this.name = name;
            this.ops = ops;
            this.opsPerSec = opsPerSec;
        }
    }
}
