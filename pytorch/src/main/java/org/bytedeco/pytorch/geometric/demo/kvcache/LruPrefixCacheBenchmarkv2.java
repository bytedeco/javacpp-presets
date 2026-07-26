package org.bytedeco.pytorch.geometric.demo.kvcache;
import org.bytedeco.pytorch.jit.*;


import org.bytedeco.pytorch.*;
import static org.bytedeco.pytorch.global.torch.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.LongAdder;

public class LruPrefixCacheBenchmarkv2 {

    public static void main(String[] args) throws InterruptedException {
        // 配置参数
        int numLayers = 32;
        int headDim = 128;
        int blockSize = 16;

        // FIX: 增加物理 Block 以支持 8 线程并发
        // 8 threads * 1025 blocks/req = 8200 blocks needed.
        // 设置 12000 留有余量，避免死锁
        int totalBlocks = 12000;
        CoWBlockManagerV8 manager = new CoWBlockManagerV8(totalBlocks, numLayers, blockSize, headDim, kFloat().value);

        int totalRequests = 1000;
        // FIX: 将线程数从 10 降为 8，确保 totalBlocks (12000) > threads * req_blocks (8 * 1025)
        int threads = 8;
        ExecutorService executor = Executors.newFixedThreadPool(threads);

        LongAdder totalTokens = new LongAdder();
        LongAdder cacheHits = new LongAdder();

        long SYSTEM_PROMPT_HASH = 7777777L;
        TensorOptions options = new TensorOptions().dtype(new ScalarTypeOptional(kFloat())).device(new DeviceOptional(new Device(kMPS())));

        System.out.println("=== Starting V5 Prefix Cache Benchmark (O(1) Invalidation) ===");
        System.out.println("Threads: " + threads + " | Physical Blocks: " + totalBlocks);

        long startTime = System.currentTimeMillis();
        int maxSimultaneousSessions = threads; // 保证 Session 数与线程数一致

        for (int i = 0; i < totalRequests; i++) {
            final int requestId = i;
            final String sid = "session-" + (requestId % maxSimultaneousSessions);

            executor.submit(() -> {
                try (PagedKvBufferV3 kv = new PagedKvBufferV3(sid, manager, numLayers)) {
                    // 1. 模拟前缀匹配
                    long currentHash = (requestId % 5 == 0) ? SYSTEM_PROMPT_HASH : (requestId + 10000);
                    int pBlock = manager.getOrAllocateBlock(currentHash, sid, kv);

                    if (currentHash == SYSTEM_PROMPT_HASH && pBlock != -1) {
                        cacheHits.increment();
                    }

                    // 2. 模拟后续数据的 Prefill
                    int payloadLen = 512;
                    Tensor input = randn(new long[]{payloadLen, headDim}, options);

                    try {
                        for (int l = 0; l < numLayers; l++) {
                            kv.prefillUltra(l, 0, input);
                            System.out.println("Request " + requestId + " prefill layer " + l);

                        }
                    } finally {
                        input.deallocate();
                    }

                    totalTokens.add(payloadLen + (pBlock != -1 ? blockSize : 0));

                    if (requestId % 50 == 0) {
                        System.out.printf("[Progress] Req %d completed.%n", requestId);
                    }
                } catch (Exception e) {
                    // FIX: 打印完整堆栈以诊断 NullPointerException
                    System.err.println("Request " + requestId + " failed: " + e.getMessage());
                    e.printStackTrace();
                }
            });

            if (i % 20 == 0) Thread.sleep(10);
        }

        executor.shutdown();
        boolean finished = executor.awaitTermination(5, TimeUnit.MINUTES);

        long duration = System.currentTimeMillis() - startTime;
        double tps = totalTokens.sum() / (duration / 1000.0);

        System.out.println("\n--------------------------------------------");
        System.out.println("Benchmark Results (V5):");
        System.out.println("Completion: " + (finished ? "Success" : "Timed Out"));
        System.out.println("Total Processed Tokens: " + totalTokens.sum());
        System.out.println("System Prompt Hits: " + cacheHits.sum());
        System.out.println("Final Throughput: " + String.format("%.2f", tps) + " tokens/sec");
        System.out.println("--------------------------------------------");

        manager.close();
    }
}

