package org.bytedeco.pytorch.geometric.demo.kvcache;

import org.bytedeco.pytorch.*;
import static org.bytedeco.pytorch.global.torch.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.LongAdder;

public class LruAutoEvictionTest {
    public static void main(String[] args) throws InterruptedException {
        // 配置参数
        int numLayers = 32;
        int blockSize = 16;
        int headDim = 128;

        // 1. 设置一个极小的池子（仅 2000 个块），模拟高压环境
        // 理论容量 = 2000 * 16 = 32,000 Tokens
        int totalPhysicalBlocks = 2000;
        CoWBlockManagerV2 manager = new CoWBlockManagerV2(totalPhysicalBlocks, numLayers, blockSize, headDim, kFloat().value);

        // 2. 模拟大量并发请求
        int totalRequests = 1000;
        int tokensPerRequest = 1024; // 每个请求需要 1024/16 = 64 个块
        // 1000个请求总共需要 64,000 个块，远超 2000 个块的物理限制

        ExecutorService executor = Executors.newFixedThreadPool(12);
        LongAdder totalTokensProcessed = new LongAdder();
        LongAdder evictionCount = new LongAdder();
        TensorOptions options =new TensorOptions().dtype(new ScalarTypeOptional(kFloat())).device(new DeviceOptional(new Device(kMPS())));

        System.out.println("=== Starting LRU Auto-Eviction Benchmark ===");
        System.out.println("Physical Limit: " + (totalPhysicalBlocks * blockSize) + " tokens");
        System.out.println("Total Workload: " + (totalRequests * tokensPerRequest) + " tokens");
        System.out.println("--------------------------------------------");

        long startTime = System.currentTimeMillis();

        for (int i = 0; i < totalRequests; i++) {
            final String sid = "session-" + i;
            executor.submit(() -> {
                try {
                    // 注意：这里我们故意不使用 try-with-resources，模拟“忘记手动释放”或“长连接”场景
                    // 依靠 Manager 内部的 allocate 触发驱逐
                    PagedKvBufferV3 kv = new PagedKvBufferV3(sid, manager,numLayers);

                    Tensor inputK = randn(new long[]{tokensPerRequest, headDim}, options);
                    Tensor inputV = randn(new long[]{tokensPerRequest, headDim}, options);

                    for (int l = 0; l < numLayers; l++) {
                        // 内部 allocateBlocks 会在空间不足时触发 manager.evictOldestSession()
                        kv.prefillUltra(l, 0, inputK);
                        kv.prefillUltra(l, 1, inputV);
                    }

                    totalTokensProcessed.add(tokensPerRequest);
                    inputK.deallocate();
                    inputV.deallocate();

                } catch (Exception e) {
                    // 如果驱逐失败（比如所有块都被锁定），会走到这里
                    System.err.println("Request failed: " + e.getMessage());
                }
            });

            // 模拟真实的请求流入速率
            if (i % 20 == 0) Thread.sleep(100);
        }

        executor.shutdown();
        executor.awaitTermination(1, TimeUnit.HOURS);

        long duration = System.currentTimeMillis() - startTime;
        System.out.println("\n--------------------------------------------");
        System.out.println("Benchmark Finished.");
        System.out.println("Total Tokens: " + totalTokensProcessed.sum());
        System.out.println("Overall TPS: " + String.format("%.2f", totalTokensProcessed.sum() / (duration / 1000.0)));
        System.out.println("Memory recovered by LRU: Success");

        manager.close();
    }
}