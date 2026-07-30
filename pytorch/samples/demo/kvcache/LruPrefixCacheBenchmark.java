package samples.demo.kvcache;
import org.bytedeco.pytorch.jit.*;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.llm.kvcache.CoWBlockManager;
import org.bytedeco.pytorch.llm.kvcache.PagedKvBuffer;

import java.util.concurrent.*;
import java.util.concurrent.atomic.LongAdder;

import static org.bytedeco.pytorch.global.torch.*;

public class LruPrefixCacheBenchmark {

    public static void main(String[] args) throws InterruptedException {
        // 配置参数
        int numLayers = 32;
        int headDim = 128; // 增加 Head Dim 模拟真实负载
        int blockSize = 16;

        // 设置物理池大小
        // 6000 个 Block 大约能容纳 5-6 个并发请求 (每个请求约 1025 Block)
        int totalBlocks = 160000;
        CoWBlockManager manager = new CoWBlockManager(totalBlocks, numLayers, blockSize, headDim, kFloat().value);

        int totalRequests = 1000;
        int threads = 10;
        ExecutorService executor = Executors.newFixedThreadPool(threads);

        LongAdder totalTokens = new LongAdder();
        LongAdder cacheHits = new LongAdder();

        // 模拟一个公共前缀
        long SYSTEM_PROMPT_HASH = 7777777L;
        TensorOptions options = new TensorOptions().dtype(new ScalarTypeOptional(kFloat())).device(new DeviceOptional(new Device(kMPS())));

        System.out.println("=== Starting V5 Prefix Cache Benchmark (O(1) Invalidation) ===");
        System.out.println("Threads: " + threads + " | Physical Blocks: " + totalBlocks);

        long startTime = System.currentTimeMillis();

        // FIX: 限制并发 Session 的总数，模拟有限用户池
        // 这样可以强制 Manager 在有限的 Session 集合中进行 LRU 淘汰
        int maxSimultaneousSessions = 10;

        for (int i = 0; i < totalRequests; i++) {
            final int requestId = i;

            // FIX: 使用取模生成循环的 Session ID
            // 之前的 `session-` + requestId 会导致生成无限个唯一 Session，
            // 使得 Manager 的 LRU 列表无限增长，或导致旧 Session 无法被作为“受害者”正确找到
            final String sid = "session-" + (requestId % maxSimultaneousSessions);

            executor.submit(() -> {
                try (PagedKvBuffer kv = new PagedKvBuffer(sid, manager, numLayers)) {
                    // 1. 模拟前缀匹配：20% 的请求命中公共前缀
                    long currentHash = (requestId % 5 == 0) ? SYSTEM_PROMPT_HASH : (requestId + 10000);

                    // 获取第一个 Block（模拟前缀缓存命中）
                    // 可能会阻塞等待 eviction
                    int pBlock = manager.getOrAllocateBlock(currentHash, sid, kv);

                    if (currentHash == SYSTEM_PROMPT_HASH && pBlock != -1) {
                        cacheHits.increment();
                    }

                    // 2. 模拟后续数据的 Prefill
                    int payloadLen = 512;
                    Tensor input = randn(new long[]{payloadLen, headDim}, options);

                    try {
                        for (int l = 0; l < numLayers; l++) {
                            // 这里会大量申请 Block，如果满了会触发 allocateWithRetry -> evictOldestSession
                           
                            System.out.println("Request " + requestId + " prefill layer " + l);
                            kv.prefillUltra(l, 0, input);
                        }
                    } finally {
                        // 及时释放 Tensor 内存
                        input.deallocate();
                    }

                    totalTokens.add(payloadLen + (pBlock != -1 ? blockSize : 0));

                    if (requestId % 50 == 0) {
                        System.out.printf("[Progress] Req %d completed. Active Blocks: %d%n",
                                requestId, manager.getActiveBlockCount());
                    }
                } catch (Exception e) {
                    System.err.println("Request " + requestId + " failed: " + e.getMessage());
                    // e.printStackTrace(); 
                }
            });

            // 稍微控制一下提交速度，避免队列堆积太长
            if (i % 20 == 0) Thread.sleep(10);
        }

        executor.shutdown();
        boolean finished = executor.awaitTermination(2, TimeUnit.MINUTES);

        long duration = System.currentTimeMillis() - startTime;
        double tps = totalTokens.sum() / (duration / 1000.0);

        System.out.println("\n--------------------------------------------");
        System.out.println("Benchmark Results (V5):");
        System.out.println("Threads: " + threads);
        System.out.println("Completion: " + (finished ? "Success" : "Timed Out"));
        System.out.println("Total Processed Tokens: " + totalTokens.sum());
        System.out.println("System Prompt Hits: " + cacheHits.sum());
        System.out.println("Final Throughput: " + String.format("%.2f", tps) + " tokens/sec");
        System.out.println("--------------------------------------------");

        manager.close();
    }
}

//public class LruPrefixCacheBenchmark {
//    
//    public static void main(String[] args) throws InterruptedException {
//        // 配置参数
//        int numLayers = 32;
//        int headDim = 128;
//        int blockSize = 16;
//
//        // 设置一个较小的物理池，强制触发 V5 的双向索引清理逻辑
//        int totalBlocks = 6000;
//        CoWBlockManagerV6 manager = new CoWBlockManagerV6(totalBlocks, numLayers, blockSize, headDim, kFloat().value);
//
//        int totalRequests = 1000;
//        int threads =1;// Runtime.getRuntime().availableProcessors(); // 适配 Mac 核心数
//        ExecutorService executor = Executors.newFixedThreadPool(threads);
//
//        LongAdder totalTokens = new LongAdder();
//        LongAdder cacheHits = new LongAdder();
//
//        // 模拟一个公共前缀（8个 Blocks）
//        long SYSTEM_PROMPT_HASH = 7777777L;
//        int prefixLen = 128;
//        TensorOptions options =new TensorOptions().dtype(new ScalarTypeOptional(kFloat())).device(new DeviceOptional(new Device(kMPS())));
//
//        System.out.println("=== Starting V5 Prefix Cache Benchmark (O(1) Invalidation) ===");
//        System.out.println("Threads: " + threads + " | Physical Blocks: " + totalBlocks);
//
//        long startTime = System.currentTimeMillis();
//
//        for (int i = 0; i < totalRequests; i++) {
//            final int requestId = i;
//            final String sid = "session-" + requestId;
//
//            executor.submit(() -> {
//                try (PagedKvBufferV3 kv = new PagedKvBufferV3(sid, manager, numLayers)) {
//                    // 1. 模拟前缀匹配：20% 的请求命中公共前缀
//                    long currentHash = (requestId % 5 == 0) ? SYSTEM_PROMPT_HASH : (requestId + 10000);
//
//                    // 获取第一个 Block（模拟前缀缓存命中）
//                    int pBlock = manager.getOrAllocateBlock(currentHash, sid, kv);
//                    if (currentHash == SYSTEM_PROMPT_HASH && pBlock != -1) {
//                        cacheHits.increment();
//                    }
//
//                    // 2. 模拟后续数据的 Prefill
//                    int payloadLen = 512;
//                    Tensor input = randn(new long[]{payloadLen, headDim}, options);
//
//                    for (int l = 0; l < numLayers; l++) {
//                        kv.prefillUltra(l, 0, input);
//                    }
//
//                    totalTokens.add(payloadLen + (pBlock != -1 ? blockSize : 0));
//                    input.deallocate();
//
//                    if (requestId % 100 == 0) {
//                        System.out.printf("[Progress] Req %d completed. Active Blocks: %d%n",
//                                requestId, manager.getActiveBlockCount());
//                    }
//                } catch (Exception e) {
//                    e.printStackTrace();
//                }
//            });
//
//            // 匀速流式请求
//            if (i % 20 == 0) Thread.sleep(10);
//        }
//
//        executor.shutdown();
//        executor.awaitTermination(1, TimeUnit.HOURS);
//
//        long duration = System.currentTimeMillis() - startTime;
//        double tps = totalTokens.sum() / (duration / 1000.0);
//
//        System.out.println("\n--------------------------------------------");
//        System.out.println("Benchmark Results (V5):");
//        System.out.println("Total Processed Tokens: " + totalTokens.sum());
//        System.out.println("System Prompt Hits: " + cacheHits.sum());
//        System.out.println("Final Throughput: " + String.format("%.2f", tps) + " tokens/sec");
//        System.out.println("--------------------------------------------");
//
//        manager.close();
//    }
//}