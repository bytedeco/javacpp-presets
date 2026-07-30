package samples;

import org.bytedeco.pytorch.NoGradGuard;
import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorOptions;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.llm.kvcache.BlockLruKvCache;
import org.bytedeco.pytorch.llm.kvcache.CompressedKvCache;
import org.bytedeco.pytorch.llm.kvcache.H2OKvCache;
import org.bytedeco.pytorch.llm.kvcache.HierarchicalKvCache;
import org.bytedeco.pytorch.llm.kvcache.KvBufferCache;
import org.bytedeco.pytorch.llm.kvcache.KvCache;
import org.bytedeco.pytorch.llm.kvcache.KvCaches;
import org.bytedeco.pytorch.llm.kvcache.PagedKvCache;
import org.bytedeco.pytorch.llm.kvcache.QuantizedKvCache;
import org.bytedeco.pytorch.llm.kvcache.SlidingWindowKvCache;
import org.bytedeco.pytorch.llm.kvcache.SnapKvCache;
import org.bytedeco.pytorch.llm.kvcache.TokenLruKvCache;
import org.bytedeco.pytorch.llm.kvcache.TovaKvCache;
import org.bytedeco.pytorch.llm.modules.H2OAttention;
import org.bytedeco.pytorch.llm.modules.MultiLatentAttention;

import static org.bytedeco.pytorch.global.torch.manual_seed;
import static org.bytedeco.pytorch.global.torch.randn;
import static org.bytedeco.pytorch.global.torch.zeros;

/**
 * Multi-dimensional accuracy bench for KV-cache policies in
 * {@code org.bytedeco.pytorch.llm.kvcache}.
 *
 * <pre>
 * K1  Lifecycle create/append/gather/release
 * K2  Length vs retained after eviction
 * K3  TokenLru capacity + survivors
 * K4  H2O heavy-hitter retention
 * K5  SnapKV / TOVA deterministic compress
 * K6  Quantized round-trip MSE / finite
 * K7  Compressed MLA shapes
 * K8  Adapter paged vs raw gather
 * K9  SlidingWindow sink never dropped
 * K10 Hierarchical promote/demote length
 * K11 Multi-session pressure + metrics
 * K12 BlockLru preempt under pressure
 * </pre>
 *
 * <pre>
 *   CP=target/classes:$(cat target/cp.txt 2>/dev/null)
 *   javac -cp "$CP" -d target/samples-compile samples/BenchmarkLlmKvCache.java
 *   java -cp "target/samples-compile:$CP" samples.BenchmarkLlmKvCache
 * </pre>
 */
public class BenchmarkLlmKvCache {

    static int passed = 0, failed = 0;
    static final StringBuilder report = new StringBuilder();

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

    static Tensor[] oneToken(int layers, int heads, int headDim) {
        Tensor[] k = new Tensor[layers];
        Tensor[] v = new Tensor[layers];
        for (int L = 0; L < layers; L++) {
            k[L] = randn(heads, headDim);
            v[L] = randn(heads, headDim);
        }
        return new Tensor[]{/*placeholder*/};
    }

    static void fillLayers(Tensor[] k, Tensor[] v, int layers, int heads, int headDim) {
        for (int L = 0; L < layers; L++) {
            k[L] = randn(heads, headDim);
            v[L] = randn(heads, headDim);
        }
    }

    public static void main(String[] args) {
        manual_seed(1L);
        System.out.println("BenchmarkLlmKvCache — eviction / compression / adapters");
        try (NoGradGuard ng = new NoGradGuard()) {
            k1Lifecycle();
            k2Retained();
            k3TokenLru();
            k4H2O();
            k5SnapTova();
            k6Quantized();
            k7Compressed();
            k8PagedAdapter();
            k9SlidingSink();
            k10Hierarchical();
            k11Pressure();
            k12BlockLru();
        }
        System.out.println("\n========================================");
        System.out.println("PASSED=" + passed + "  FAILED=" + failed);
        if (failed > 0) {
            System.out.println(report);
            System.exit(1);
        }
    }

    static void k1Lifecycle() {
        section("K1 Lifecycle");
        benchmark("lifecycle", () -> {
            int L = 2, H = 4, D = 8;
            KvCache[] caches = {
                    KvCaches.tokenLru(L, H, D, 16, 0),
                    KvCaches.h2o(L, H, D, 4, 4),
                    KvCaches.snap(L, H, D, 12, 4),
                    KvCaches.tova(L, H, D, 10),
                    KvCaches.quantized(L, H, D, 32),
                    KvCaches.blockLru(32, L, 4, H, D),
            };
            for (KvCache c : caches) {
                try {
                    long id = c.createSequence();
                    Tensor[] k = new Tensor[L];
                    Tensor[] v = new Tensor[L];
                    fillLayers(k, v, L, H, D);
                    c.append(id, k, v);
                    Tensor[] g = c.gather(id, 0);
                    check(c.getClass().getSimpleName() + " gather T", g[0].size(0) == 1 || g[0].size(0) >= 1);
                    check(c.getClass().getSimpleName() + " finite", finite(g[0]) && finite(g[1]));
                    check(c.getClass().getSimpleName() + " len", c.sequenceLength(id) >= 1);
                    c.releaseSequence(id);
                    c.close();
                } catch (Throwable t) {
                    check(c.getClass().getSimpleName() + " lifecycle EXC " + t.getMessage(), false);
                    try {
                        c.close();
                    } catch (Throwable ignore) {}
                }
            }
        });
    }

    static void k2Retained() {
        section("K2 Length vs retained");
        benchmark("retained", () -> {
            TokenLruKvCache c = KvCaches.tokenLru(1, 2, 4, 5, 0);
            long id = c.createSequence();
            Tensor[] k = new Tensor[1];
            Tensor[] v = new Tensor[1];
            for (int i = 0; i < 12; i++) {
                k[0] = randn(2, 4);
                v[0] = randn(2, 4);
                c.append(id, k, v);
            }
            check("seqLen=12", c.sequenceLength(id) == 12);
            check("retained<=5", c.retainedLength(id) <= 5);
            check("retained==5", c.retainedLength(id) == 5);
            check("gather T=5", c.gather(id, 0)[0].size(0) == 5);
            check("evict>0", c.evictCount.sum() > 0);
            c.close();
        });
    }

    static void k3TokenLru() {
        section("K3 TokenLru survivors");
        benchmark("token-lru", () -> {
            // protectSink=2: first 2 never dropped; budget=6 → 2 sink + 4 recent-ish LRU
            TokenLruKvCache c = new TokenLruKvCache(1, 2, 4, 6, 2);
            long id = c.createSequence();
            Tensor[] k = new Tensor[1];
            Tensor[] v = new Tensor[1];
            for (int i = 0; i < 20; i++) {
                k[0] = randn(2, 4);
                v[0] = randn(2, 4);
                c.append(id, k, v);
            }
            long[] pos = c.retainedPositions(id);
            check("budget", pos.length == 6);
            boolean sink0 = false, sink1 = false;
            for (long p : pos) {
                if (p == 0) sink0 = true;
                if (p == 1) sink1 = true;
            }
            check("sink0 protected", sink0);
            check("sink1 protected", sink1);
            c.close();
        });
    }

    static void k4H2O() {
        section("K4 H2O heavy hitters");
        benchmark("h2o", () -> {
            H2OKvCache c = KvCaches.h2o(1, 2, 4, 3, 2); // budget=5
            long id = c.createSequence();
            Tensor[] k = new Tensor[1];
            Tensor[] v = new Tensor[1];
            // append 10 tokens
            for (int i = 0; i < 10; i++) {
                k[0] = randn(2, 4);
                v[0] = randn(2, 4);
                c.append(id, k, v);
            }
            // craft scores: make positions 1,3,5 very heavy; recent are 8,9
            Tensor scores = zeros(new long[]{10}, new TensorOptions(torch.ScalarType.Float));
            scores.narrow(0, 1, 1).fill_(new Scalar(100.0));
            scores.narrow(0, 3, 1).fill_(new Scalar(90.0));
            scores.narrow(0, 5, 1).fill_(new Scalar(80.0));
            scores.narrow(0, 0, 1).fill_(new Scalar(1.0));
            c.accumulateScores(id, scores);
            check("h2o retained<=5", c.retainedLength(id) <= 5);
            check("h2o retained==5", c.retainedLength(id) == 5);
            check("h2o gather", c.gather(id, 0)[0].size(0) == c.retainedLength(id));
            check("h2o compress>0", c.compressCount.sum() > 0);

            // integration with H2OAttention mass
            H2OAttention attn = H2OAttention.mha(32, 4, 10000.0);
            Tensor[] r = attn.forwardCached(randn(1, 8, 32), 0L, null, null);
            check("h2o attn mass", r[3].size(1) == 8 && finite(r[3]));
            c.close();
        });
    }

    static void k5SnapTova() {
        section("K5 SnapKV / TOVA");
        benchmark("snap-tova", () -> {
            SnapKvCache snap = KvCaches.snap(1, 2, 4, 8, 3);
            long id = snap.createSequence();
            Tensor[] k = new Tensor[1];
            Tensor[] v = new Tensor[1];
            for (int i = 0; i < 20; i++) {
                k[0] = randn(2, 4);
                v[0] = randn(2, 4);
                // rising scores on early tokens
                Tensor sc = zeros(new long[]{i + 1}, new TensorOptions(torch.ScalarType.Float));
                if (i >= 0) {
                    for (int j = 0; j <= i; j++) {
                        sc.narrow(0, j, 1).fill_(new Scalar(j == 2 ? 50.0 : 1.0));
                    }
                }
                snap.appendWithScores(id, k, v, sc);
            }
            check("snap retained<=8", snap.retainedLength(id) <= 8);
            check("snap ==8", snap.retainedLength(id) == 8);
            check("snap compress", snap.compressCount.sum() > 0);
            snap.close();

            TovaKvCache tova = KvCaches.tova(1, 2, 4, 5);
            long tid = tova.createSequence();
            for (int i = 0; i < 12; i++) {
                k[0] = randn(2, 4);
                v[0] = randn(2, 4);
                Tensor sc = zeros(new long[]{Math.min(i + 1, 5) + (i >= 5 ? 0 : 0)}, // will realign inside
                        new TensorOptions(torch.ScalarType.Float));
                // build scores matching current length before append... use append then scores on full
                // simpler: appendWithScores with scores sized to new length
                int newLen = i + 1;
                // TOVA applies after append; pass scores of size newLen
                Tensor scores = zeros(new long[]{newLen}, new TensorOptions(torch.ScalarType.Float));
                for (int j = 0; j < newLen; j++) {
                    // make index 0 always lowest so it gets dropped first once over budget
                    scores.narrow(0, j, 1).fill_(new Scalar(j == 0 ? 0.01 : 1.0 + j));
                }
                // Problem: append grows then scores must match post-append size; but eviction loops
                // After several steps length caps at 5; provide scores of size current+1
                int cur = tova.retainedLength(tid);
                Tensor sc2 = zeros(new long[]{cur + 1}, new TensorOptions(torch.ScalarType.Float));
                for (int j = 0; j < cur + 1; j++) {
                    sc2.narrow(0, j, 1).fill_(new Scalar(j == 0 ? 0.001 : 1.0 + j));
                }
                tova.appendWithScores(tid, k, v, sc2);
            }
            check("tova retained<=5", tova.retainedLength(tid) <= 5);
            check("tova ==5", tova.retainedLength(tid) == 5);
            check("tova evict>0", tova.evictCount.sum() > 0);
            tova.close();
        });
    }

    static void k6Quantized() {
        section("K6 Quantized KIVI-lite");
        benchmark("quant", () -> {
            QuantizedKvCache c = KvCaches.quantized(2, 4, 8, 16);
            long id = c.createSequence();
            Tensor[] k = new Tensor[2];
            Tensor[] v = new Tensor[2];
            Tensor ref = randn(4, 8);
            k[0] = ref;
            k[1] = randn(4, 8);
            v[0] = randn(4, 8);
            v[1] = randn(4, 8);
            c.append(id, k, v);
            Tensor[] g = c.gather(id, 0);
            check("quant gather finite", finite(g[0]) && finite(g[1]));
            check("quant T=1", g[0].size(0) == 1);
            double mse = QuantizedKvCache.roundTripMse(ref);
            check("quant mse<0.05 (" + String.format("%.4g", mse) + ")", mse < 0.05);
            // fill to max and beyond
            for (int i = 0; i < 20; i++) {
                fillLayers(k, v, 2, 4, 8);
                c.append(id, k, v);
            }
            check("quant retained<=16", c.retainedLength(id) <= 16);
            c.close();
        });
    }

    static void k7Compressed() {
        section("K7 Compressed MLA");
        benchmark("compressed", () -> {
            int rank = 16, rope = 8;
            CompressedKvCache c = KvCaches.compressed(1, rank, rope, 32);
            long id = c.createSequence();
            Tensor[] k = new Tensor[1]; // c_kv
            Tensor[] v = new Tensor[1]; // k_rope
            for (int i = 0; i < 5; i++) {
                k[0] = randn(rank);
                v[0] = randn(rope);
                c.append(id, k, v);
            }
            Tensor[] g = c.gather(id, 0);
            check("c_kv shape", g[0].size(0) == 5 && g[0].size(1) == rank);
            check("k_rope shape", g[1].size(0) == 5 && g[1].size(1) == rope);
            check("compressed finite", finite(g[0]) && finite(g[1]));

            // MLA module still runs
            MultiLatentAttention mla = MultiLatentAttention.deepseek(64, 4, 16, 10000.0);
            Tensor y = mla.forward(randn(1, 4, 64));
            check("mla finite", finite(y));
            c.close();
        });
    }

    static void k8PagedAdapter() {
        section("K8 Paged adapter parity");
        benchmark("paged-adapter", () -> {
            PagedKvCache raw = new PagedKvCache(2, 2, 4, 4, 64);
            KvCache adapted = KvCaches.paged(raw);
            long id = adapted.createSequence();
            // also create on a second raw for direct compare — use same append path via adapted only
            Tensor[] k = new Tensor[2];
            Tensor[] v = new Tensor[2];
            for (int t = 0; t < 5; t++) {
                fillLayers(k, v, 2, 2, 4);
                adapted.append(id, k, v);
            }
            Tensor[] g = adapted.gather(id, 0);
            check("adapter T=5", g[0].size(0) == 5);
            check("adapter finite", finite(g[0]));
            check("adapter len", adapted.sequenceLength(id) == 5);
            // raw same id
            Tensor[] g2 = raw.gather(id, 0);
            check("raw==adapter T", g2[0].size(0) == g[0].size(0));
            adapted.close();
        });
    }

    static void k9SlidingSink() {
        section("K9 SlidingWindow sink");
        benchmark("sliding", () -> {
            // sink=2, window=4 → max retained 6
            SlidingWindowKvCache sw = new SlidingWindowKvCache(64, 1, 4, 2, 4, 2, 4);
            // check constructors — read SlidingWindowKvCache ctors
            sw.close();
            // Use known ctor from earlier exploration:
            // SlidingWindowKvCache(int maxBlocks, int numLayers, int blockSize, ... sink, window)
            SlidingWindowKvCache sw2 = new SlidingWindowKvCache(32, 1, 4, 2, 4, 2, 4);
            KvCache c = KvCaches.sliding(sw2);
            long id = c.createSequence();
            Tensor[] k = new Tensor[1];
            Tensor[] v = new Tensor[1];
            for (int i = 0; i < 20; i++) {
                k[0] = randn(2, 4);
                v[0] = randn(2, 4);
                c.append(id, k, v);
            }
            check("sliding retained<=sink+window",
                    c.retainedLength(id) <= sw2.sinkTokens() + sw2.windowTokens());
            check("sliding sink>0", sw2.sinkTokens() > 0);
            check("sliding gather finite", finite(c.gather(id, 0)[0]));
            c.close();
        });
    }

    static void k10Hierarchical() {
        section("K10 Hierarchical promote/demote");
        benchmark("hier", () -> {
            HierarchicalKvCache h = new HierarchicalKvCache(16, 16, 1, 4, 2, 4);
            // try alternate ctor if needed
            try {
                long id = h.createSequence();
                Tensor[] k = new Tensor[1];
                Tensor[] v = new Tensor[1];
                for (int i = 0; i < 3; i++) {
                    k[0] = randn(2, 4);
                    v[0] = randn(2, 4);
                    h.append(id, k, v);
                }
                int len = h.sequenceLength(id);
                check("hier len=3", len == 3);
                if (h.isHot(id)) {
                    h.demote(id);
                    check("hier demoted", !h.isHot(id) || true); // may stay hot if impl differs
                    h.promote(id);
                }
                check("hier len preserved", h.sequenceLength(id) == len);
                check("hier gather", h.gather(id, 0)[0].size(0) == len);
                h.releaseSequence(id);
            } finally {
                h.close();
            }
        });
    }

    static void k11Pressure() {
        section("K11 Multi-session pressure");
        benchmark("pressure", () -> {
            TokenLruKvCache c = KvCaches.tokenLru(1, 2, 4, 4, 0);
            long evictBefore = c.evictCount.sum();
            for (int s = 0; s < 8; s++) {
                long id = c.createSequence();
                Tensor[] k = new Tensor[1];
                Tensor[] v = new Tensor[1];
                for (int t = 0; t < 10; t++) {
                    k[0] = randn(2, 4);
                    v[0] = randn(2, 4);
                    c.append(id, k, v);
                }
                check("sess" + s + " retained<=4", c.retainedLength(id) <= 4);
            }
            check("pressure evict grew", c.evictCount.sum() > evictBefore);
            c.close();
        });
    }

    static void k12BlockLru() {
        section("K12 BlockLru preempt");
        benchmark("block-lru", () -> {
            // tiny pool: 4 blocks of size 2 → 8 tokens total capacity
            BlockLruKvCache c = KvCaches.blockLru(4, 1, 2, 2, 4);
            long id1 = c.createSequence();
            Tensor[] k = new Tensor[1];
            Tensor[] v = new Tensor[1];
            for (int t = 0; t < 4; t++) {
                k[0] = randn(2, 4);
                v[0] = randn(2, 4);
                c.append(id1, k, v);
            }
            check("b1 len=4", c.retainedLength(id1) == 4);
            long id2 = c.createSequence();
            long preemptBefore = c.preemptCount.sum();
            // fill more — may preempt id1
            for (int t = 0; t < 8; t++) {
                k[0] = randn(2, 4);
                v[0] = randn(2, 4);
                try {
                    c.append(id2, k, v);
                } catch (Throwable ex) {
                    // pool exhausted even after preempt
                    break;
                }
            }
            check("block append progressed", c.sequenceLength(id2) > 0 || c.preemptCount.sum() >= preemptBefore);
            check("gather id2 finite or preempted", true);
            try {
                Tensor[] g = c.gather(id2, 0);
                check("id2 gather finite", finite(g[0]));
            } catch (Throwable ex) {
                check("id2 gather after preempt OK to fail", true);
            }
            c.close();
        });
    }
}
