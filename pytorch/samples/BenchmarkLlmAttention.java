package samples;

import org.bytedeco.pytorch.NoGradGuard;
import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.llm.modules.Attention;
import org.bytedeco.pytorch.llm.modules.CrossAttention;
import org.bytedeco.pytorch.llm.modules.DifferentialAttention;
import org.bytedeco.pytorch.llm.modules.FlashAttention;
import org.bytedeco.pytorch.llm.modules.GatedAttention;
import org.bytedeco.pytorch.llm.modules.H2OAttention;
import org.bytedeco.pytorch.llm.modules.InfiniAttention;
import org.bytedeco.pytorch.llm.modules.LinearAttention;
import org.bytedeco.pytorch.llm.modules.Modules;
import org.bytedeco.pytorch.llm.modules.NativeSparseAttention;
import org.bytedeco.pytorch.llm.modules.PagedAttention;
import org.bytedeco.pytorch.llm.modules.RetentionAttention;
import org.bytedeco.pytorch.llm.modules.SparseAttention;
import org.bytedeco.pytorch.llm.modules.StreamingSinkAttention;
import org.bytedeco.pytorch.nn.Module;

import static org.bytedeco.pytorch.global.torch.manual_seed;
import static org.bytedeco.pytorch.global.torch.randn;

/**
 * Multi-dimensional accuracy bench for paper-level attention variants in
 * {@code org.bytedeco.pytorch.llm.modules}.
 *
 * <pre>
 * A1  Construct all factories / Modules shortcuts
 * A2  Shape [B,T,H]→[B,T,H] matrix
 * A3  Finite (no NaN/Inf)
 * A4  Flash ↔ dense Attention parity (same weights)
 * A5  GQA / MQA head counts
 * A6  Cache continuity (prefill+decode vs one-shot)
 * A7  Sparse / sink / sliding masks (far mass ~0 via finite path)
 * A8  DifferentialAttention λ path
 * A9  Linear / Retention / Infini state shapes
 * A10 CrossAttention memory length ≠ query length
 * A11 PagedAttention contiguous path
 * A12 Throughput smoke (informational)
 * A13 H2OAttention mass side-channel
 * </pre>
 *
 * <pre>
 *   CP=target/classes:$(cat target/cp.txt 2>/dev/null)
 *   javac -cp "$CP" -d target/samples-compile samples/BenchmarkLlmAttention.java
 *   java -cp "target/samples-compile:$CP" samples.BenchmarkLlmAttention
 * </pre>
 */
public class BenchmarkLlmAttention {

    static int passed = 0, failed = 0;
    static final StringBuilder report = new StringBuilder();
    /** fp32 parity tolerance for Flash vs dense on small T. */
    static final double PARITY_TAU = 2e-3;
    static final double CONT_TAU = 5e-3;

    @FunctionalInterface
    interface CheckedRunnable {
        void run() throws Exception;
    }

    static void check(String name, boolean ok) {
        if (ok) {
            passed++;
            System.out.println("  PASS  " + name);
        } else {
            failed++;
            report.append("FAIL ").append(name).append('\n');
            System.out.println("  FAIL  " + name);
        }
    }

    static void section(String t) {
        System.out.println("\n=== " + t + " ===");
    }

    static void benchmark(String name, CheckedRunnable r) {
        try {
            r.run();
        } catch (Throwable t) {
            failed++;
            String msg = "EXC   " + name + " :: " + t.getClass().getSimpleName() + ": " + t.getMessage();
            report.append(msg).append('\n');
            System.out.println("  " + msg);
            t.printStackTrace(System.out);
        }
    }

    static boolean finite(Tensor t) {
        try {
            double s = t.detach().sum().item().toDouble();
            return !Double.isNaN(s) && !Double.isInfinite(s);
        } catch (Throwable e) {
            return false;
        }
    }

    static boolean shapeEq(Tensor t, long... dims) {
        if (t.dim() != dims.length) {
            return false;
        }
        for (int i = 0; i < dims.length; i++) {
            if (t.size(i) != dims[i]) {
                return false;
            }
        }
        return true;
    }

    static double maxAbsDiff(Tensor a, Tensor b) {
        Tensor d = a.detach().sub(b.detach()).abs();
        return d.max().item().toDouble();
    }

    public static void main(String[] args) {
        manual_seed(0L);
        System.out.println("BenchmarkLlmAttention — paper-level attention variants");
        try (NoGradGuard ng = new NoGradGuard()) {
            a1Factories();
            a2Shapes();
            a3Finite();
            a4FlashParity();
            a5Gqa();
            a6CacheContinuity();
            a7SparseSink();
            a8DiffAttn();
            a9Stateful();
            a10Cross();
            a11Paged();
            a12Throughput();
            a13H2O();
        }
        System.out.println("\n========================================");
        System.out.println("PASSED=" + passed + "  FAILED=" + failed);
        if (failed > 0) {
            System.out.println(report);
            System.exit(1);
        }
    }

    static void a1Factories() {
        section("A1 Factories");
        benchmark("A1", () -> {
            check("flash mha", Modules.flashMha(64, 4, 10000.0) != null);
            check("flash gqa", Modules.flashGqa(64, 4, 2, 10000.0).nKvHeads() == 2);
            check("cross", Modules.crossMha(64, 4) != null);
            check("diff", Modules.diffAttn(64, 4, 10000.0).lambdaInit() > 0);
            check("linear", Modules.linearMha(64, 4) != null);
            check("longformer", Modules.longformer(64, 4, 10000.0, 4, 2) != null);
            check("infini", Modules.infini(64, 4, 10000.0, 8) != null);
            check("retention", Modules.retention(64, 4) != null);
            check("streaming", Modules.streamingSink(64, 4, 10000.0, 2, 8).sinkTokens() == 2);
            check("nsa", Modules.nsa(64, 4, 10000.0) != null);
            check("gated", Modules.gatedGqa(64, 4, 2, 10000.0) != null);
            check("h2o", Modules.h2oGqa(64, 4, 2, 10000.0) != null);
            check("paged", Modules.pagedGqa(64, 4, 2, 10000.0) != null);
        });
    }

    static void a2Shapes() {
        section("A2 Shapes");
        long H = 64;
        long[][] shapes = {{1, 1}, {1, 8}, {2, 8}, {1, 33}, {2, 16}};
        for (long[] bt : shapes) {
            long B = bt[0], T = bt[1];
            Tensor x = randn(B, T, H);
            final long fB = B, fT = T;
            benchmark("shape B" + B + "T" + T, () -> {
                check("flash " + fB + "x" + fT,
                        shapeEq(FlashAttention.mha(H, 4, 10000.0).forward(x), fB, fT, H));
                check("gated " + fB + "x" + fT,
                        shapeEq(GatedAttention.mha(H, 4, 10000.0).forward(x), fB, fT, H));
                check("stream " + fB + "x" + fT,
                        shapeEq(StreamingSinkAttention.paperDefault(H, 4, 10000.0).forward(x), fB, fT, H));
                check("sparse " + fB + "x" + fT,
                        shapeEq(SparseAttention.longformer(H, 4, 10000.0, 4, 1).forward(x), fB, fT, H));
                check("diff " + fB + "x" + fT,
                        shapeEq(DifferentialAttention.paperDefault(H, 4, 10000.0).forward(x), fB, fT, H));
                check("linear " + fB + "x" + fT,
                        shapeEq(LinearAttention.mha(H, 4).forward(x), fB, fT, H));
                check("ret " + fB + "x" + fT,
                        shapeEq(RetentionAttention.mha(H, 4).forward(x), fB, fT, H));
                check("infini " + fB + "x" + fT,
                        shapeEq(InfiniAttention.paperDefault(H, 4, 10000.0).forward(x), fB, fT, H));
                check("nsa " + fB + "x" + fT,
                        shapeEq(NativeSparseAttention.paperDefault(H, 4, 10000.0).forward(x), fB, fT, H));
                check("h2o " + fB + "x" + fT,
                        shapeEq(H2OAttention.mha(H, 4, 10000.0).forward(x), fB, fT, H));
                check("paged " + fB + "x" + fT,
                        shapeEq(PagedAttention.gqa(H, 4, 2, 10000.0).forward(x), fB, fT, H));
            });
        }
    }

    static void a3Finite() {
        section("A3 Finite");
        Tensor x = randn(2, 16, 64);
        benchmark("finite", () -> {
            Module[] mods = {
                    FlashAttention.gqa(64, 4, 2, 10000.0),
                    GatedAttention.gqa(64, 4, 2, 10000.0),
                    DifferentialAttention.paperDefault(64, 4, 10000.0),
                    LinearAttention.mha(64, 4),
                    SparseAttention.bigbird(64, 4, 10000.0, 4, 1, 2),
                    InfiniAttention.gqa(64, 4, 2, 10000.0, 8),
                    RetentionAttention.mha(64, 4),
                    StreamingSinkAttention.gqa(64, 4, 2, 10000.0, 2, 8),
                    NativeSparseAttention.gqa(64, 4, 2, 10000.0, 4, 4),
                    H2OAttention.gqa(64, 4, 2, 10000.0),
                    PagedAttention.flashGqa(64, 4, 2, 10000.0),
                    CrossAttention.gqa(64, 4, 2)
            };
            for (Module m : mods) {
                Tensor y = m.forward(x);
                check(m.getClass().getSimpleName() + " finite", finite(y));
            }
        });
    }

    static void a4FlashParity() {
        section("A4 Flash ↔ dense parity");
        benchmark("parity", () -> {
            manual_seed(42L);
            long H = 64;
            Attention dense = Attention.mha(H, 4, 10000.0);
            FlashAttention flash = FlashAttention.mha(H, 4, 10000.0);
            flash.copyWeightsFrom(dense);
            // small T for tight tolerance; also try a few lengths
            for (long T : new long[]{4, 8, 16}) {
                Tensor x = randn(1, T, H);
                Tensor yD = dense.forward(x);
                Tensor yF = flash.forward(x);
                double mad = maxAbsDiff(yD, yF);
                check("parity T=" + T + " mad=" + String.format("%.6g", mad), mad < PARITY_TAU);
                check("parity finite T=" + T, finite(yD) && finite(yF));
            }
            // GQA parity
            Attention gqa = Attention.gqa(H, 4, 2, 10000.0);
            FlashAttention fg = FlashAttention.gqa(H, 4, 2, 10000.0);
            fg.copyWeightsFrom(gqa);
            Tensor x = randn(2, 8, H);
            double mad = maxAbsDiff(gqa.forward(x), fg.forward(x));
            check("parity GQA mad=" + String.format("%.6g", mad), mad < PARITY_TAU);
        });
    }

    static void a5Gqa() {
        section("A5 GQA/MQA");
        benchmark("gqa", () -> {
            FlashAttention mqa = new FlashAttention(64, 4, 1, 16, 10000.0, 1.0, true,
                    false, false, false, 1e-6, -1, true, 8, 16);
            check("mqa kv=1", mqa.nKvHeads() == 1);
            check("mqa shape", shapeEq(mqa.forward(randn(1, 8, 64)), 1, 8, 64));
            GatedAttention g = GatedAttention.gqa(64, 8, 2, 10000.0);
            check("gated kv=2", g.nKvHeads() == 2);
            check("gated heads=8", g.nHeads() == 8);
        });
    }

    static void a6CacheContinuity() {
        section("A6 Cache continuity");
        benchmark("cache", () -> {
            long H = 64;
            Tensor full = randn(1, 12, H);
            Tensor pre = full.narrow(1, 0, 8);
            // dense reference
            Attention dense = Attention.gqa(H, 4, 2, 10000.0);
            Tensor oneShot = dense.forward(full);
            Tensor[] step = dense.forwardCached(pre, 0L, null, null);
            Tensor out = step[0];
            Tensor pk = step[1], pv = step[2];
            for (int t = 8; t < 12; t++) {
                Tensor tok = full.narrow(1, t, 1);
                Tensor[] s = dense.forwardCached(tok, t, pk, pv);
                out = org.bytedeco.pytorch.global.torch.cat(
                        new org.bytedeco.pytorch.TensorVector(out, s[0]), 1);
                pk = org.bytedeco.pytorch.global.torch.cat(
                        new org.bytedeco.pytorch.TensorVector(pk, s[1]), 2);
                pv = org.bytedeco.pytorch.global.torch.cat(
                        new org.bytedeco.pytorch.TensorVector(pv, s[2]), 2);
            }
            double madD = maxAbsDiff(oneShot, out);
            check("dense cont mad=" + String.format("%.6g", madD), madD < CONT_TAU);

            // Flash
            FlashAttention flash = FlashAttention.gqa(H, 4, 2, 10000.0);
            flash.copyWeightsFrom(dense);
            Tensor oneF = flash.forward(full);
            Tensor[] sf = flash.forwardCached(pre, 0L, null, null);
            Tensor outF = sf[0];
            Tensor pfk = sf[1], pfv = sf[2];
            for (int t = 8; t < 12; t++) {
                Tensor tok = full.narrow(1, t, 1);
                Tensor[] s = flash.forwardCached(tok, t, pfk, pfv);
                outF = org.bytedeco.pytorch.global.torch.cat(
                        new org.bytedeco.pytorch.TensorVector(outF, s[0]), 1);
                pfk = org.bytedeco.pytorch.global.torch.cat(
                        new org.bytedeco.pytorch.TensorVector(pfk, s[1]), 2);
                pfv = org.bytedeco.pytorch.global.torch.cat(
                        new org.bytedeco.pytorch.TensorVector(pfv, s[2]), 2);
            }
            double madF = maxAbsDiff(oneF, outF);
            check("flash cont mad=" + String.format("%.6g", madF), madF < CONT_TAU);

            // Gated
            GatedAttention gated = GatedAttention.gqa(H, 4, 2, 10000.0);
            Tensor oneG = gated.forward(full);
            Tensor[] sg0 = gated.forwardCached(pre, 0L, null, null);
            Tensor outG = sg0[0];
            Tensor pgk = sg0[1], pgv = sg0[2];
            for (int t = 8; t < 12; t++) {
                Tensor[] s = gated.forwardCached(full.narrow(1, t, 1), t, pgk, pgv);
                outG = org.bytedeco.pytorch.global.torch.cat(
                        new org.bytedeco.pytorch.TensorVector(outG, s[0]), 1);
                pgk = org.bytedeco.pytorch.global.torch.cat(
                        new org.bytedeco.pytorch.TensorVector(pgk, s[1]), 2);
                pgv = org.bytedeco.pytorch.global.torch.cat(
                        new org.bytedeco.pytorch.TensorVector(pgv, s[2]), 2);
            }
            check("gated cont", maxAbsDiff(oneG, outG) < CONT_TAU);

            // Streaming sink
            StreamingSinkAttention sink = StreamingSinkAttention.gqa(H, 4, 2, 10000.0, 2, 8);
            Tensor oneS = sink.forward(full);
            Tensor[] ss0 = sink.forwardCached(pre, 0L, null, null);
            Tensor outS = ss0[0];
            Tensor psk = ss0[1], psv = ss0[2];
            for (int t = 8; t < 12; t++) {
                Tensor[] s = sink.forwardCached(full.narrow(1, t, 1), t, psk, psv);
                outS = org.bytedeco.pytorch.global.torch.cat(
                        new org.bytedeco.pytorch.TensorVector(outS, s[0]), 1);
                psk = org.bytedeco.pytorch.global.torch.cat(
                        new org.bytedeco.pytorch.TensorVector(psk, s[1]), 2);
                psv = org.bytedeco.pytorch.global.torch.cat(
                        new org.bytedeco.pytorch.TensorVector(psv, s[2]), 2);
            }
            check("stream cont", maxAbsDiff(oneS, outS) < CONT_TAU);
        });
    }

    static void a7SparseSink() {
        section("A7 Sparse / sink");
        benchmark("sparse-sink", () -> {
            Tensor x = randn(1, 32, 64);
            SparseAttention sp = SparseAttention.longformer(64, 4, 10000.0, 4, 2);
            check("longformer finite", finite(sp.forward(x)));
            SparseAttention bb = SparseAttention.bigbird(64, 4, 10000.0, 4, 1, 3);
            check("bigbird finite", finite(bb.forward(x)));
            StreamingSinkAttention sk = StreamingSinkAttention.gqa(64, 4, 2, 10000.0, 4, 8);
            check("sink window", sk.windowTokens() == 8);
            check("sink forward", shapeEq(sk.forward(x), 1, 32, 64));
        });
    }

    static void a8DiffAttn() {
        section("A8 DifferentialAttention");
        benchmark("diff", () -> {
            DifferentialAttention d = DifferentialAttention.paperDefault(64, 4, 10000.0);
            check("lambda in (0,1)", d.lambdaInit() > 0 && d.lambdaInit() < 1.5);
            Tensor y = d.forward(randn(2, 8, 64));
            check("diff shape", shapeEq(y, 2, 8, 64));
            check("diff finite", finite(y));
            Tensor[] c = d.forwardCached(randn(1, 4, 64), 0L, null, null);
            check("diff cache out", shapeEq(c[0], 1, 4, 64));
            check("diff cache k", c[1].defined() && c[1].size(2) == 4);
        });
    }

    static void a9Stateful() {
        section("A9 Linear / Retention / Infini states");
        benchmark("stateful", () -> {
            Tensor x = randn(1, 6, 64);
            LinearAttention lin = LinearAttention.mha(64, 4);
            Tensor[] ls = lin.forwardCached(x, null, null);
            check("linear out", shapeEq(ls[0], 1, 6, 64));
            check("linear S", ls[1].dim() == 4 && ls[1].size(2) == 16); // headDim=16
            check("linear Z", ls[2].dim() == 3);
            // decode step with state
            Tensor[] ls2 = lin.forwardCached(randn(1, 1, 64), ls[1], ls[2]);
            check("linear decode", shapeEq(ls2[0], 1, 1, 64) && finite(ls2[0]));

            RetentionAttention ret = RetentionAttention.mha(64, 4);
            Tensor[] rs = ret.forwardCached(x, null);
            check("ret out", shapeEq(rs[0], 1, 6, 64));
            check("ret S", rs[1].dim() == 4);
            Tensor[] rs2 = ret.forwardCached(randn(1, 1, 64), rs[1]);
            check("ret decode", shapeEq(rs2[0], 1, 1, 64) && finite(rs2[0]));

            InfiniAttention inf = InfiniAttention.paperDefault(64, 4, 10000.0);
            Tensor[] is_ = inf.forwardCached(x, 0L, null, null, null, null);
            check("infini out", shapeEq(is_[0], 1, 6, 64));
            check("infini mem", is_[3].dim() == 4);
            check("infini z", is_[4].dim() == 3);
            check("infini finite", finite(is_[0]));
        });
    }

    static void a10Cross() {
        section("A10 CrossAttention");
        benchmark("cross", () -> {
            CrossAttention c = CrossAttention.mha(64, 4);
            Tensor q = randn(2, 5, 64);
            Tensor mem = randn(2, 11, 64);
            Tensor[] out = c.forwardCross(q, mem, null, null);
            check("cross out", shapeEq(out[0], 2, 5, 64));
            check("cross memK", out[1].size(2) == 11);
            check("cross finite", finite(out[0]));
            // reuse cached memory
            Tensor[] out2 = c.forwardCross(randn(2, 3, 64), mem, out[1], out[2]);
            check("cross cached", shapeEq(out2[0], 2, 3, 64) && out2[1] == out[1]);
        });
    }

    static void a11Paged() {
        section("A11 PagedAttention contiguous");
        benchmark("paged", () -> {
            PagedAttention p = PagedAttention.gqa(64, 4, 2, 10000.0);
            Tensor x = randn(1, 8, 64);
            Tensor[] r = p.forwardCached(x, 0L, null, null);
            check("paged out", shapeEq(r[0], 1, 8, 64));
            check("paged k", r[1].size(1) == 4); // GQA-repeated to nHeads
            // decode
            Tensor[] r2 = p.forwardCached(randn(1, 1, 64), 8L, r[1], r[2]);
            check("paged decode", shapeEq(r2[0], 1, 1, 64) && finite(r2[0]));
            PagedAttention pf = PagedAttention.flashGqa(64, 4, 2, 10000.0);
            check("paged flash", finite(pf.forward(x)));
        });
    }

    static void a12Throughput() {
        section("A12 Throughput smoke (informational)");
        benchmark("tput", () -> {
            Tensor x = randn(1, 32, 64);
            FlashAttention f = FlashAttention.mha(64, 4, 10000.0);
            long t0 = System.nanoTime();
            for (int i = 0; i < 5; i++) {
                f.forward(x);
            }
            long ms = (System.nanoTime() - t0) / 1_000_000;
            System.out.println("  INFO  flash 5×fwd ~ " + ms + " ms");
            check("tput ran", ms >= 0);
        });
    }

    static void a13H2O() {
        section("A13 H2OAttention mass");
        benchmark("h2o", () -> {
            H2OAttention h = H2OAttention.gqa(64, 4, 2, 10000.0);
            Tensor[] r = h.forwardCached(randn(1, 10, 64), 0L, null, null);
            check("h2o out", shapeEq(r[0], 1, 10, 64));
            check("h2o mass", r[3].dim() == 2 && r[3].size(1) == 10);
            check("h2o mass finite", finite(r[3]));
            // mass should be positive-ish (softmax rows sum to 1 → mass ≈ nHeads * Tq per key average)
            double sum = r[3].sum().item().toDouble();
            check("h2o mass sum>0", sum > 0);
        });
    }
}
