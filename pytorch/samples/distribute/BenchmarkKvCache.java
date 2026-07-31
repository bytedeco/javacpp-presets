package distribute;

import org.bytedeco.pytorch.Device;
import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.llm.kvcache.BlockHashIndex;
import org.bytedeco.pytorch.llm.kvcache.CoWBlockManager;
import org.bytedeco.pytorch.llm.kvcache.HierarchicalKvCache;
import org.bytedeco.pytorch.llm.kvcache.KvBufferCache;
import org.bytedeco.pytorch.llm.kvcache.PagedBlockManager;
import org.bytedeco.pytorch.llm.kvcache.PagedKvBuffer;
import org.bytedeco.pytorch.llm.kvcache.PagedKvCache;
import org.bytedeco.pytorch.llm.kvcache.PrefixRadixCache;
import org.bytedeco.pytorch.llm.kvcache.SlidingWindowKvCache;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Full-spectrum multi-dimension benchmark / correctness suite for
 * {@code org.bytedeco.pytorch.llm.kvcache}.
 *
 * <p>Dimensions covered:
 * <ol>
 *   <li>PagedBlockManager — alloc/release/refcount/CoW/gather</li>
 *   <li>CoWBlockManager + PagedKvBuffer — session LRU, prefill/decode, fork</li>
 *   <li>KvBufferCache — dense contiguous buffers</li>
 *   <li>PagedKvCache — integrated paged + prefix + CoW (regression)</li>
 *   <li>PrefixRadixCache + BlockHashIndex — SGLang / TRT-LLM style reuse</li>
 *   <li>HierarchicalKvCache — hot/cold demote/promote</li>
 *   <li>SlidingWindowKvCache — sink + window reclaim</li>
 *   <li>CoWBlockManager hash-path match — content-addressed reuse</li>
 *   <li>Concurrency — multi-session parallel append</li>
 *   <li>Throughput microbenches — append/gather/prefix-hit rates</li>
 *   <li>Memory pressure — OOM, preemption, watermark prune</li>
 * </ol>
 *
 * <p>Run from repo root after building the preset jar, e.g.:
 * {@code java -cp samples:target/classes:... distribute.BenchmarkKvCache}
 */
public class BenchmarkKvCache {

    static int passed = 0;
    static int failed = 0;
    static final StringBuilder report = new StringBuilder();
    static final List<PerfRow> perf = new ArrayList<>();

    public static void main(String[] args) throws Exception {
        System.out.println("=== KVCache Module Full-Spectrum Benchmark ===\n");

        section("1. PagedBlockManager");
        benchPagedBlockManager();

        section("2. CoWBlockManager + PagedKvBuffer");
        benchCoWAndBuffer();

        section("3. KvBufferCache (dense)");
        benchKvBufferCache();

        section("4. PagedKvCache (integrated)");
        benchPagedKvCache();

        section("5. PrefixRadixCache + BlockHashIndex");
        benchPrefixAndHash();

        section("6. HierarchicalKvCache");
        benchHierarchical();

        section("7. SlidingWindowKvCache");
        benchSlidingWindow();

        section("8. CoWBlockManager hash-path");
        benchCowHashPath();

        section("9. Concurrency");
        benchConcurrency();

        section("10. Throughput");
        benchThroughput();

        section("11. Memory pressure");
        benchPressure();

        // ── summary ───────────────────────────────────────────────────────
        System.out.println("\n=== Correctness ===");
        System.out.println("Passed: " + passed);
        System.out.println("Failed: " + failed);
        if (failed > 0) {
            System.out.println("\nFAILED CHECKS:");
            System.out.println(report);
        }

        if (!perf.isEmpty()) {
            System.out.println("\n=== Throughput ===");
            System.out.printf(Locale.ROOT, "%-42s %12s %12s%n", "metric", "ops", "ops/s");
            for (PerfRow r : perf) {
                System.out.printf(Locale.ROOT, "%-42s %12d %12.1f%n", r.name, r.ops, r.opsPerSec);
            }
        }

        if (failed > 0) System.exit(1);
        System.out.println("\nAll tests PASSED!");
    }

    // =====================================================================
    // 1. PagedBlockManager
    // =====================================================================

    static void benchPagedBlockManager() {
        benchmark("construct + free accounting", () -> {
            try (PagedBlockManager m = new PagedBlockManager(32, 4, 8, 2, 16)) {
                check("maxBlocks", m.maxBlocks() == 32);
                check("free initial", m.freeBlocks() == 32);
                check("used initial", m.usedBlocks() == 0);
                check("freeRatio 1", m.freeRatio() == 1.0);
            }
        });

        benchmark("alloc/release refcount", () -> {
            try (PagedBlockManager m = new PagedBlockManager(16, 2, 4, 2, 8)) {
                int a = m.allocateBlock();
                int b = m.allocateBlock();
                check("distinct", a != b);
                check("free 14", m.freeBlocks() == 14);
                check("ref a == 1", m.refCount(a) == 1);
                m.retain(a);
                check("ref a == 2", m.refCount(a) == 2);
                m.release(a);
                check("ref a == 1 after drop", m.refCount(a) == 1);
                m.release(a);
                check("a freed", m.freeBlocks() == 15);
                m.release(b);
                check("all free", m.freeBlocks() == 16);
            }
        });

        benchmark("cowIfNeeded copies shared block", () -> {
            try (PagedBlockManager m = new PagedBlockManager(8, 2, 4, 2, 8)) {
                m.ensureAllocated();
                int b = m.allocateBlock();
                Tensor k = torch.ones(new long[]{2, 8});
                Tensor v = torch.ones(new long[]{2, 8}).mul(new Scalar(2));
                m.writeToken(b, 0, 0, k, v);
                m.retain(b); // shared
                int nb = m.cowIfNeeded(b);
                check("cow produced new id", nb != b);
                check("old ref back to 1", m.refCount(b) == 1);
                check("new ref 1", m.refCount(nb) == 1);
                // exclusive again → identity
                int same = m.cowIfNeeded(nb);
                check("exclusive cow is identity", same == nb);
                m.release(b);
                m.release(nb);
            }
        });

        benchmark("gather roundtrip", () -> {
            try (PagedBlockManager m = new PagedBlockManager(8, 2, 4, 2, 8)) {
                List<Integer> table = m.allocateBlocks(2);
                // write 6 tokens across 2 blocks (blockSize=4)
                for (int t = 0; t < 6; t++) {
                    int bi = t / 4;
                    int pos = t % 4;
                    Tensor k = torch.ones(new long[]{2, 8}).mul(new Scalar(t + 1));
                    Tensor v = torch.ones(new long[]{2, 8}).mul(new Scalar((t + 1) * 10));
                    m.writeToken(table.get(bi), 0, pos, k, v);
                }
                Tensor[] kv = m.gather(table, 0, 6);
                check("gather K dim0 == 6", kv[0].size(0) == 6);
                check("gather V dim0 == 6", kv[1].size(0) == 6);
                m.releaseAll(table);
            }
        });

        benchmark("OOM throws", () -> {
            try (PagedBlockManager m = new PagedBlockManager(2, 1, 2, 1, 4)) {
                m.allocateBlock();
                m.allocateBlock();
                boolean threw = false;
                try { m.allocateBlock(); } catch (IllegalStateException e) { threw = true; }
                check("OOM thrown", threw);
            }
        });

        benchmark("compat factory withDtypeValue", () -> {
            try (PagedBlockManager m = PagedBlockManager.withDtypeValue(4, 2, 8, 16, torch.kFloat().value)) {
                check("compat free", m.freeBlocks() == 4);
                check("compat heads default 1", m.numHeads() == 1);
            }
        });
    }

    // =====================================================================
    // 2. CoW + PagedKvBuffer
    // =====================================================================

    static void benchCoWAndBuffer() {
        benchmark("session prefill + decode + gather", () -> {
            try (CoWBlockManager mgr = new CoWBlockManager(64, 2, 4, 2, 8)) {
                PagedKvBuffer buf = new PagedKvBuffer("s1", mgr, 2);
                Tensor[] kL = new Tensor[2];
                Tensor[] vL = new Tensor[2];
                for (int l = 0; l < 2; l++) {
                    kL[l] = torch.randn(new long[]{5, 2, 8});
                    vL[l] = torch.randn(new long[]{5, 2, 8});
                }
                buf.prefillAll(kL, vL);
                check("len after prefill 5", buf.length() == 5);
                check("blocks >= 2", buf.blockTable().size() >= 2);

                for (int t = 0; t < 3; t++) {
                    Tensor[] k = new Tensor[2];
                    Tensor[] v = new Tensor[2];
                    for (int l = 0; l < 2; l++) {
                        k[l] = torch.randn(new long[]{2, 8});
                        v[l] = torch.randn(new long[]{2, 8});
                    }
                    buf.append(k, v);
                }
                check("len after decode 8", buf.length() == 8);
                Tensor[] g = buf.gather(0);
                check("gather T == 8", g[0].size(0) == 8);
                buf.close();
                check("session released free", mgr.getFreeBlockCount() == 64);
            }
        });

        benchmark("fork CoW isolation", () -> {
            try (CoWBlockManager mgr = new CoWBlockManager(64, 2, 4, 2, 8)) {
                PagedKvBuffer parent = new PagedKvBuffer("p", mgr, 2);
                Tensor[] kL = layers(2, 4, 2, 8, 1.0);
                Tensor[] vL = layers(2, 4, 2, 8, 2.0);
                parent.prefillAll(kL, vL);
                int freeAfterParent = mgr.getFreeBlockCount();

                PagedKvBuffer child = parent.fork("c");
                check("child len == parent", child.length() == parent.length());
                // shared → free count should not drop by full block amount again
                check("fork shares blocks (free not halved)", mgr.getFreeBlockCount() == freeAfterParent);

                // mutate child
                Tensor[] k = token(2, 2, 8, 9.0);
                Tensor[] v = token(2, 2, 8, 9.0);
                child.append(k, v);
                check("child longer", child.length() == parent.length() + 1);
                check("parent unchanged", parent.length() == 4);

                parent.close();
                child.close();
            }
        });

        benchmark("LRU preemption under pressure", () -> {
            // 4 blocks total, blockSize=4, 2 layers multi-layer → 1 block covers all layers
            try (CoWBlockManager mgr = new CoWBlockManager(4, 1, 4, 1, 4)) {
                PagedKvBuffer a = new PagedKvBuffer("a", mgr, 1);
                PagedKvBuffer b = new PagedKvBuffer("b", mgr, 1);
                // each needs 2 blocks for 8 tokens
                a.prefillAll(layers(1, 8, 1, 4, 1.0), layers(1, 8, 1, 4, 1.0));
                check("a took 2", a.blockTable().size() == 2);
                b.prefillAll(layers(1, 8, 1, 4, 2.0), layers(1, 8, 1, 4, 2.0));
                check("b took 2", b.blockTable().size() == 2);
                check("pool empty", mgr.getFreeBlockCount() == 0);

                // c needs blocks → should preempt LRU (a, if b was touched last)
                PagedKvBuffer c = new PagedKvBuffer("c", mgr, 1);
                c.prefillAll(layers(1, 8, 1, 4, 3.0), layers(1, 8, 1, 4, 3.0));
                check("preempt happened", mgr.preemptedSessions.sum() >= 1 || a.isInvalidated());
                c.close();
                if (!a.isInvalidated()) a.close();
                if (!b.isInvalidated()) b.close();
            }
        });
    }

    // =====================================================================
    // 3. KvBufferCache
    // =====================================================================

    static void benchKvBufferCache() {
        benchmark("dense append + history slice", () -> {
            try (KvBufferCache cache = new KvBufferCache(2, 32, 16)) {
                KvBufferCache.KvBuffer buf = cache.getKvBuffer("s");
                for (int t = 0; t < 10; t++) {
                    Tensor[] k = new Tensor[2];
                    Tensor[] v = new Tensor[2];
                    for (int l = 0; l < 2; l++) {
                        k[l] = torch.ones(new long[]{16}).mul(new Scalar(t + 1));
                        v[l] = torch.ones(new long[]{16}).mul(new Scalar((t + 1) * 2));
                    }
                    buf.appendToken(k, v);
                }
                check("pos 10", buf.getCurrentPosition() == 10);
                Tensor hist = buf.getTensorsUpToCurrent(0, 0);
                check("hist rows 10", hist.size(0) == 10);
                cache.release("s");
                check("released", !cache.contains("s"));
            }
        });

        benchmark("dense full throws", () -> {
            try (KvBufferCache cache = new KvBufferCache(1, 4, 8)) {
                KvBufferCache.KvBuffer buf = cache.getKvBuffer("full");
                for (int t = 0; t < 4; t++) {
                    buf.appendToken(
                            new Tensor[]{torch.zeros(new long[]{8})},
                            new Tensor[]{torch.zeros(new long[]{8})});
                }
                boolean threw = false;
                try {
                    buf.appendToken(
                            new Tensor[]{torch.zeros(new long[]{8})},
                            new Tensor[]{torch.zeros(new long[]{8})});
                } catch (IllegalStateException e) { threw = true; }
                check("full throws", threw);
            }
        });
    }

    // =====================================================================
    // 4. PagedKvCache regression
    // =====================================================================

    static void benchPagedKvCache() {
        benchmark("lifecycle + append/gather", () -> {
            try (PagedKvCache cache = new PagedKvCache(2, 2, 8, 4, 64)) {
                long id = cache.createSequence();
                for (int t = 0; t < 5; t++) {
                    cache.append(id, t, token(2, 2, 8, t + 1.0), token(2, 2, 8, (t + 1) * 2.0));
                }
                check("len 5", cache.sequenceLength(id) == 5);
                Tensor[] kv = cache.gather(id, 0);
                check("gather T 5", kv[0].size(0) == 5);
                cache.releaseSequence(id);
                check("live 0", cache.liveSequences() == 0);
            }
        });

        benchmark("fork + prefix match", () -> {
            try (PagedKvCache cache = new PagedKvCache(2, 2, 8, 4, 64)) {
                long parent = cache.createSequence();
                int[] toks = {1, 2, 3, 4, 5};
                for (int t : toks) {
                    cache.append(parent, t, token(2, 2, 8, 1.0), token(2, 2, 8, 1.0));
                }
                long child = cache.fork(parent);
                check("fork len", cache.sequenceLength(child) == 5);
                cache.append(child, 99, token(2, 2, 8, 1.0), token(2, 2, 8, 1.0));
                check("parent stable", cache.sequenceLength(parent) == 5);
                check("child grew", cache.sequenceLength(child) == 6);

                PagedKvCache.PrefixHit hit = cache.matchPrefix(new int[]{1, 2, 3, 4});
                check("prefix hit 4", hit.matchedTokens == 4);
                check("prefix seq > 0", hit.sequenceId > 0);
                cache.releaseSequence(parent);
                cache.releaseSequence(child);
                if (hit.sequenceId > 0) cache.releaseSequence(hit.sequenceId);
            }
        });

        benchmark("stats counters", () -> {
            try (PagedKvCache cache = new PagedKvCache(2, 2, 8, 4, 32)) {
                long id = cache.createSequence();
                for (int t = 0; t < 5; t++) {
                    cache.append(id, t, token(2, 2, 8, 1.0), token(2, 2, 8, 1.0));
                }
                check("allocCount > 0", cache.allocCount.sum() > 0);
                check("appendCount 5", cache.appendCount.sum() == 5);
                cache.releaseSequence(id);
            }
        });

        benchmark("re-export org.bytedeco.pytorch.llm.PagedKvCache", () -> {
            try (org.bytedeco.pytorch.llm.PagedKvCache c =
                         new org.bytedeco.pytorch.llm.PagedKvCache(4, 2, 16, 8, 64, false)) {
                check("reexport layers", c.numLayers() == 4);
            }
        });
    }

    // =====================================================================
    // 5. Prefix + Hash index
    // =====================================================================

    static void benchPrefixAndHash() {
        benchmark("PrefixRadixCache insert/match/evict", () -> {
            PagedBlockManager pool = new PagedBlockManager(32, 1, 4, 1, 4);
            PrefixRadixCache.RefCountedBlockStore store = adapt(pool);
            try (PrefixRadixCache tree = new PrefixRadixCache(4, store, 32, 0.1, 0.5)) {
                List<Integer> blocks = pool.allocateBlocks(2);
                // keep session refs + tree will retain on insert
                int[] tokens = {10, 11, 12, 13, 14, 15, 16, 17};
                tree.insert(tokens, blocks);
                check("nodes > 0", tree.nodeCount() > 0);

                PrefixRadixCache.Match m = tree.match(new int[]{10, 11, 12, 13, 14});
                check("matched 4 or 8", m.matchedTokens == 4 || m.matchedTokens == 8);
                check("hit", m.hit());
                // release match retains
                for (int b : m.blockIds) pool.release(b);

                // release original alloc refs → only tree holds
                pool.releaseAll(blocks);
                PrefixRadixCache.Match miss = tree.match(new int[]{99, 98});
                check("miss", !miss.hit());

                int freeBefore = pool.freeBlocks();
                tree.evictToFreeRatio(freeBefore, 32);
                check("evict did something or already free", pool.freeBlocks() >= freeBefore);
            } finally {
                pool.close();
            }
        });

        benchmark("BlockHashIndex chain match", () -> {
            PagedBlockManager pool = new PagedBlockManager(32, 1, 4, 1, 4);
            try (BlockHashIndex idx = new BlockHashIndex(4, adapt(pool))) {
                int[] tokens = {1, 2, 3, 4, 5, 6, 7, 8};
                List<Integer> blocks = pool.allocateBlocks(2);
                idx.indexSequence(tokens, blocks);

                List<Integer> hit = idx.matchPrefix(tokens);
                check("full chain hit 2", hit.size() == 2);
                // release match retains
                pool.releaseAll(hit);

                int[] partial = {1, 2, 3, 4, 9, 9, 9, 9};
                List<Integer> partialHit = idx.matchPrefix(partial);
                check("partial chain 1", partialHit.size() == 1);
                pool.releaseAll(partialHit);

                // original session refs
                pool.releaseAll(blocks);
            } finally {
                pool.close();
            }
        });

        benchmark("hash stability", () -> {
            long h1 = BlockHashIndex.hashBlock(0L, new int[]{1, 2, 3, 4}, 0, 4);
            long h2 = BlockHashIndex.hashBlock(0L, new int[]{1, 2, 3, 4}, 0, 4);
            long h3 = BlockHashIndex.hashBlock(0L, new int[]{1, 2, 3, 5}, 0, 4);
            check("deterministic", h1 == h2);
            check("sensitive", h1 != h3);
            long c1 = BlockHashIndex.hashBlock(h1, new int[]{5, 6, 7, 8}, 0, 4);
            long c2 = BlockHashIndex.hashBlock(h1, new int[]{5, 6, 7, 8}, 0, 4);
            check("chain deterministic", c1 == c2);
        });
    }

    // =====================================================================
    // 6. Hierarchical
    // =====================================================================

    static void benchHierarchical() {
        benchmark("hot append + demote/promote", () -> {
            try (HierarchicalKvCache cache =
                         new HierarchicalKvCache(4, 16, 2, 4, 2, 8)) {
                long a = cache.createSequence();
                long b = cache.createSequence();
                // fill a with 8 tokens → 2 hot blocks
                for (int t = 0; t < 8; t++) {
                    cache.append(a, token(2, 2, 8, 1.0), token(2, 2, 8, 1.0));
                }
                check("a hot", cache.isHot(a));
                check("a len 8", cache.sequenceLength(a) == 8);

                // fill b similarly — may demote a
                for (int t = 0; t < 8; t++) {
                    cache.append(b, token(2, 2, 8, 2.0), token(2, 2, 8, 2.0));
                }
                check("b len 8", cache.sequenceLength(b) == 8);

                // force demote b, promote a
                if (cache.isHot(b)) cache.demote(b);
                check("b cold", !cache.isHot(b));
                cache.promote(b);
                check("b hot again", cache.isHot(b));

                Tensor[] kv = cache.gather(b, 0);
                check("gather after promote T 8", kv[0].size(0) == 8);

                cache.releaseSequence(a);
                cache.releaseSequence(b);
                check("demote or promote counters",
                        cache.demoteCount.sum() + cache.promoteCount.sum() >= 1);
            }
        });
    }

    // =====================================================================
    // 7. Sliding window
    // =====================================================================

    static void benchSlidingWindow() {
        benchmark("sink+window retain bound", () -> {
            // sink=4, window=8, blockSize=4
            try (SlidingWindowKvCache cache =
                         new SlidingWindowKvCache(64, 2, 4, 2, 8, 4, 8)) {
                long id = cache.createSequence();
                for (int t = 0; t < 32; t++) {
                    cache.append(id, token(2, 2, 8, 1.0), token(2, 2, 8, 1.0));
                }
                check("logical len 32", cache.sequenceLength(id) == 32);
                int retained = cache.retainedLength(id);
                check("retained <= sink+window", retained <= cache.maxRetainedTokens());
                check("retained > 0", retained > 0);
                Tensor[] kv = cache.gather(id, 0);
                check("gather T == retained", kv[0].size(0) == retained);
                cache.releaseSequence(id);
            }
        });
    }

    // =====================================================================
    // 8. CoWBlockManager content-addressed hash path
    // =====================================================================

    static void benchCowHashPath() {
        benchmark("hash path match/allocate", () -> {
            try (CoWBlockManager mgr = new CoWBlockManager(32, 2, 4, 2, 8)) {
                PagedKvBuffer buf = new PagedKvBuffer("hp", mgr, 2);
                List<Long> path = Arrays.asList(100L, 200L, 300L);
                List<Integer> blocks = mgr.matchAndAllocatePath(path, "hp", buf);
                check("3 blocks", blocks.size() == 3);
                // second call should hit
                PagedKvBuffer buf2 = new PagedKvBuffer("hp2", mgr, 2);
                List<Integer> blocks2 = mgr.matchAndAllocatePath(path, "hp2", buf2);
                check("hit reuses", blocks2.size() == 3);
                check("hash hits > 0", mgr.hashIndex().hitCount.sum() > 0);
                buf.close();
                buf2.close();
            }
        });

        benchmark("getOrAllocateBlock hit/miss", () -> {
            try (CoWBlockManager mgr = new CoWBlockManager(16, 1, 4, 1, 4)) {
                PagedKvBuffer buf = new PagedKvBuffer("g", mgr, 1);
                int a = mgr.getOrAllocateBlock(0xabcL, "g", buf);
                int b = mgr.getOrAllocateBlock(0xabcL, "g", buf);
                check("same hash reuses block", a == b);
                int c = mgr.getOrAllocateBlock(0xdefL, "g", buf);
                check("different hash new block", c != a);
                buf.close();
            }
        });
    }

    // =====================================================================
    // 9. Concurrency
    // =====================================================================

    static void benchConcurrency() {
        benchmark("parallel multi-session append on PagedKvCache", () -> {
            int threads = 4;
            int tokensPer = 20;
            try (PagedKvCache cache = new PagedKvCache(4, 2, 8, 4, 256)) {
                ExecutorService es = Executors.newFixedThreadPool(threads);
                CountDownLatch start = new CountDownLatch(1);
                AtomicInteger errors = new AtomicInteger();
                List<Future<?>> futures = new ArrayList<>();
                for (int t = 0; t < threads; t++) {
                    futures.add(es.submit(() -> {
                        try {
                            start.await();
                            long id = cache.createSequence();
                            for (int i = 0; i < tokensPer; i++) {
                                cache.append(id, i, token(4, 2, 8, 1.0), token(4, 2, 8, 1.0));
                            }
                            if (cache.sequenceLength(id) != tokensPer) errors.incrementAndGet();
                            cache.releaseSequence(id);
                        } catch (Throwable e) {
                            errors.incrementAndGet();
                        }
                    }));
                }
                start.countDown();
                for (Future<?> f : futures) f.get(60, TimeUnit.SECONDS);
                es.shutdown();
                check("no concurrent errors", errors.get() == 0);
                check("live 0 after join", cache.liveSequences() == 0);
            }
        });
    }

    // =====================================================================
    // 10. Throughput
    // =====================================================================

    static void benchThroughput() {
        benchmark("PagedKvCache append throughput", () -> {
            int tokens = 500;
            try (PagedKvCache cache = new PagedKvCache(4, 4, 16, 8, 1024)) {
                long id = cache.createSequence();
                // warmup
                for (int i = 0; i < 10; i++) {
                    cache.append(id, i, token(4, 4, 16, 1.0), token(4, 4, 16, 1.0));
                }
                long t0 = System.nanoTime();
                for (int i = 0; i < tokens; i++) {
                    cache.append(id, i, token(4, 4, 16, 1.0), token(4, 4, 16, 1.0));
                }
                long dt = System.nanoTime() - t0;
                double ops = tokens / (dt / 1e9);
                record("PagedKvCache.append", tokens, ops);
                check("throughput append > 0", ops > 0);
                cache.releaseSequence(id);
            }
        });

        benchmark("PagedKvCache gather throughput", () -> {
            int reps = 200;
            try (PagedKvCache cache = new PagedKvCache(4, 4, 16, 8, 512)) {
                long id = cache.createSequence();
                for (int i = 0; i < 64; i++) {
                    cache.append(id, i, token(4, 4, 16, 1.0), token(4, 4, 16, 1.0));
                }
                long t0 = System.nanoTime();
                for (int i = 0; i < reps; i++) {
                    Tensor[] kv = cache.gather(id, i % 4);
                    check("gather ok", kv[0].size(0) == 64);
                }
                long dt = System.nanoTime() - t0;
                record("PagedKvCache.gather", reps, reps / (dt / 1e9));
                cache.releaseSequence(id);
            }
        });

        benchmark("prefix hit rate (synthetic shared system prompt)", () -> {
            try (PagedKvCache cache = new PagedKvCache(2, 2, 8, 4, 256)) {
                // system prompt of 8 tokens (2 full blocks)
                int[] system = {1, 2, 3, 4, 5, 6, 7, 8};
                long seed = cache.createSequence();
                for (int t : system) {
                    cache.append(seed, t, token(2, 2, 8, 1.0), token(2, 2, 8, 1.0));
                }
                // release seed but tree should keep blocks
                cache.releaseSequence(seed);

                int hits = 0;
                int trials = 20;
                for (int i = 0; i < trials; i++) {
                    int[] query = Arrays.copyOf(system, system.length);
                    PagedKvCache.PrefixHit hit = cache.matchPrefix(query);
                    if (hit.matchedTokens >= 8) hits++;
                    if (hit.sequenceId > 0) cache.releaseSequence(hit.sequenceId);
                }
                double rate = hits / (double) trials;
                record("prefix full-hit rate x1000", hits * 1000L / trials, rate * 1000);
                check("prefix hit rate > 0.5", rate >= 0.5);
            }
        });

        benchmark("BlockHashIndex lookup throughput", () -> {
            PagedBlockManager pool = new PagedBlockManager(128, 1, 16, 1, 8);
            try (BlockHashIndex idx = new BlockHashIndex(16, adapt(pool))) {
                int[] tokens = new int[16 * 20];
                for (int i = 0; i < tokens.length; i++) tokens[i] = i;
                List<Integer> blocks = pool.allocateBlocks(20);
                idx.indexSequence(tokens, blocks);
                int reps = 5000;
                long t0 = System.nanoTime();
                int found = 0;
                for (int i = 0; i < reps; i++) {
                    long parent = 0;
                    long h = BlockHashIndex.hashBlock(parent, tokens, 0, 16);
                    int b = idx.lookup(h);
                    if (b >= 0) {
                        found++;
                        pool.release(b);
                    }
                }
                long dt = System.nanoTime() - t0;
                record("BlockHashIndex.lookup", reps, reps / (dt / 1e9));
                check("lookups found", found == reps);
                pool.releaseAll(blocks);
            } finally {
                pool.close();
            }
        });
    }

    // =====================================================================
    // 11. Pressure
    // =====================================================================

    static void benchPressure() {
        benchmark("PagedKvCache prune under watermark", () -> {
            try (PagedKvCache cache = new PagedKvCache(2, 2, 8, 4, 32,
                    new Device("cpu"), 0.25, 0.75)) {
                // fill several sequences then release to leave tree-only refs
                for (int s = 0; s < 4; s++) {
                    long id = cache.createSequence();
                    for (int t = 0; t < 16; t++) {
                        cache.append(id, t + s * 100, token(2, 2, 8, 1.0), token(2, 2, 8, 1.0));
                    }
                    cache.releaseSequence(id);
                }
                int freeBefore = cache.freeBlocks();
                int pruned = cache.prune();
                check("prune >= 0", pruned >= 0);
                check("free after prune >= before", cache.freeBlocks() >= freeBefore);
            }
        });

        benchmark("Hierarchical demote cascade", () -> {
            try (HierarchicalKvCache cache =
                         new HierarchicalKvCache(2, 8, 1, 4, 1, 4)) {
                List<Long> ids = new ArrayList<>();
                for (int s = 0; s < 4; s++) {
                    long id = cache.createSequence();
                    ids.add(id);
                    for (int t = 0; t < 4; t++) {
                        cache.append(id, token(1, 1, 4, 1.0), token(1, 1, 4, 1.0));
                    }
                }
                check("some demotes under tiny hot tier", cache.demoteCount.sum() >= 1);
                for (long id : ids) cache.releaseSequence(id);
            }
        });
    }

    // =====================================================================
    // helpers
    // =====================================================================

    static Tensor[] token(int layers, int heads, int dim, double fill) {
        Tensor[] out = new Tensor[layers];
        for (int i = 0; i < layers; i++) {
            out[i] = torch.ones(new long[]{heads, dim}).mul(new Scalar(fill));
        }
        return out;
    }

    static Tensor[] layers(int nLayers, int T, int heads, int dim, double fill) {
        Tensor[] out = new Tensor[nLayers];
        for (int i = 0; i < nLayers; i++) {
            out[i] = torch.ones(new long[]{T, heads, dim}).mul(new Scalar(fill));
        }
        return out;
    }

    static PrefixRadixCache.RefCountedBlockStore adapt(PagedBlockManager pool) {
        return new PrefixRadixCache.RefCountedBlockStore() {
            @Override public void retain(int blockId) { pool.retain(blockId); }
            @Override public void release(int blockId) { pool.release(blockId); }
            @Override public int refCount(int blockId) { return pool.refCount(blockId); }
        };
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
