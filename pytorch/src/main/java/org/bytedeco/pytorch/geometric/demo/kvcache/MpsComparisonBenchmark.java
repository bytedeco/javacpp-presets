package org.bytedeco.pytorch.geometric.demo.kvcache;

import org.bytedeco.pytorch.*;
import static org.bytedeco.pytorch.global.torch.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.LongAdder;

public class MpsComparisonBenchmark {
    public static void main(String[] args) throws InterruptedException {
        int layers = 32;
        int headDim = 128;
        int prefillBatch = 512; // 模拟一次 Prefill 的长度
        int testRequests = 50;  // 总请求数
        int threads = 8;        // 针对 Mac M 芯片优化线程数

        System.out.println("=== Starting MPS KV Cache Comparison (Mac M-Series) ===");

        // 1. 压测 PagedKV (你的 OptimizedPagedAttentionTest 逻辑)
        runPagedBenchmark(layers, headDim, prefillBatch, testRequests, threads);

        System.out.println("\n---------------------------------------------------\n");

        // 2. 压测 NaiveKV (你的 KvCacheLoadTest 逻辑)
        runNaiveBenchmark(layers, headDim, prefillBatch, testRequests, threads);
    }

    static void runPagedBenchmark(int layers, int headDim, int batch, int reqs, int threads) throws InterruptedException {
        CoWBlockManager manager = new CoWBlockManager(20000, layers, 16, headDim, kFloat().value);
        ExecutorService executor = Executors.newFixedThreadPool(threads);
        LongAdder tokenCounter = new LongAdder();
        TensorOptions options =new TensorOptions().dtype(new ScalarTypeOptional(kFloat())).device(new DeviceOptional(new Device(kMPS())));

        long startTime = System.currentTimeMillis();
        for (int i = 0; i < reqs; i++) {
            executor.submit(() -> {
                try (PagedKvBuffer2 kv = new PagedKvBuffer2(manager)) {
                    Tensor inputK = randn(new long[]{batch, headDim}, options);
                    Tensor inputV = randn(new long[]{batch, headDim}, options);

                    for (int l = 0; l < layers; l++) {
                        kv.prefillUltra(l, 0, inputK);
                        kv.prefillUltra(l, 1, inputV);
                    }
                    tokenCounter.add(batch);
                    inputK.deallocate(); inputV.deallocate();
                }
            });
        }
        executor.shutdown();
        executor.awaitTermination(1, TimeUnit.HOURS);

        long duration = System.currentTimeMillis() - startTime;
        printResult("Paged (prefillUltra)", tokenCounter.sum(), duration);
        manager.close();
    }

    static void runNaiveBenchmark(int layers, int headDim, int batch, int reqs, int threads) throws InterruptedException {
        // 模拟 Naive：每个请求需要连续内存，且读取时需要 cat
        KvBufferCache cache = new KvBufferCache(kFloat().value, layers, 2048, headDim);
        ExecutorService executor = Executors.newFixedThreadPool(threads);
        LongAdder opCounter = new LongAdder();
        TensorOptions options =new TensorOptions().dtype(new ScalarTypeOptional(kFloat())).device(new DeviceOptional(new Device(kMPS())));

        long startTime = System.currentTimeMillis();
        for (int i = 0; i < reqs; i++) {
            final int id = i;
            executor.submit(() -> {
                String sid = "session-" + id;
                KvBufferCache.KvBuffer kv = cache.getKvBuffer(sid);
                Tensor input = randn(new long[]{headDim}, options);
                System.out.println("Naive Benchmark - Request " + id + " started.");

                // 模拟 batch 次 append
                for (int p = 0; p < batch; p++) {
                    for (int l = 0; l < layers; l++) {
                        kv.append(l, 0, input);
                        kv.append(l, 1, input);
                        // 模拟计算时的 Cat 操作
                        try (Tensor hist = kv.getTensorsUpTo(l, 0, p + 1)) {
                            opCounter.add(1);
                        }
                    }
                    kv.incrementPosition();
                }
                input.deallocate();
            });
        }
        executor.shutdown();
        executor.awaitTermination(1, TimeUnit.HOURS);

        long duration = System.currentTimeMillis() - startTime;
        printResult("Naive (append + cat)", opCounter.sum(), duration);
        cache.close();
    }

    static void printResult(String label, long total, long duration) {
        double tps = total / (duration / 1000.0);
        System.out.println("Result for [" + label + "]:");
        System.out.println("Total Ops/Tokens: " + total);
        System.out.println("Throughput: " + String.format("%.2f", tps) + " ops/sec");
    }
}


/***
 Result for [Paged (prefillUltra)]:
 Total Ops/Tokens: 25600
 Throughput: 7219.40 ops/sec

 ---------------------------------------------------

 Naive Benchmark - Request 2 started.
 Naive Benchmark - Request 3 started.
 Naive Benchmark - Request 0 started.
 Naive Benchmark - Request 5 started.
 Naive Benchmark - Request 6 started.
 Naive Benchmark - Request 4 started.
 Naive Benchmark - Request 1 started.
 Naive Benchmark - Request 7 started.
 Naive Benchmark - Request 8 started.
 Naive Benchmark - Request 9 started.
 Naive Benchmark - Request 13 started.
 Naive Benchmark - Request 11 started.
 Naive Benchmark - Request 15 started.
 Naive Benchmark - Request 12 started.
 Naive Benchmark - Request 10 started.
 Naive Benchmark - Request 14 started.
 Naive Benchmark - Request 16 started.
 Naive Benchmark - Request 17 started.
 Naive Benchmark - Request 18 started.
 Naive Benchmark - Request 19 started.
 Naive Benchmark - Request 20 started.
 Naive Benchmark - Request 21 started.
 Naive Benchmark - Request 22 started.
 Naive Benchmark - Request 23 started.
 Naive Benchmark - Request 24 started.
 Naive Benchmark - Request 25 started.
 Naive Benchmark - Request 26 started.
 Naive Benchmark - Request 27 started.
 Naive Benchmark - Request 28 started.
 Naive Benchmark - Request 29 started.
 Naive Benchmark - Request 30 started.
 Naive Benchmark - Request 31 started.
 Naive Benchmark - Request 32 started.
 Naive Benchmark - Request 33 started.
 Naive Benchmark - Request 34 started.
 Naive Benchmark - Request 35 started.
 Naive Benchmark - Request 36 started.
 Naive Benchmark - Request 37 started.
 Naive Benchmark - Request 39 started.
 Naive Benchmark - Request 38 started.
 Naive Benchmark - Request 40 started.
 Naive Benchmark - Request 41 started.
 Naive Benchmark - Request 42 started.
 Naive Benchmark - Request 43 started.
 Naive Benchmark - Request 44 started.
 Naive Benchmark - Request 45 started.
 Naive Benchmark - Request 46 started.
 Naive Benchmark - Request 47 started.
 Naive Benchmark - Request 48 started.
 Naive Benchmark - Request 49 started.
 Result for [Naive (append + cat)]:
 Total Ops/Tokens: 819200
 Throughput: 1182.70 ops/sec***/