package samples.demo.kvcache;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.llm.kvcache.CoWBlockManager;
import org.bytedeco.pytorch.llm.kvcache.PagedKvBuffer;

import static org.bytedeco.pytorch.global.torch.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.LongAdder;

public class PrefixCacheBenchmark {
    public static void main(String[] args) throws InterruptedException {
        int numLayers = 32;
        int headDim = 128;
        int totalBlocks = 5000;
        CoWBlockManager manager = new CoWBlockManager(totalBlocks, numLayers, 16, headDim, kFloat().value);

        ExecutorService executor = Executors.newFixedThreadPool(12);
        LongAdder totalTokens = new LongAdder();
        LongAdder cacheHits = new LongAdder();
        TensorOptions options =new TensorOptions().dtype(new ScalarTypeOptional(kFloat())).device(new DeviceOptional(new Device(kMPS())));

        // 模拟一个高价值的公共前缀哈希
        long SYSTEM_PROMPT_HASH = 88888888L;
        int prefixLen = 128; // 占用 8 个 blocks

        System.out.println("=== Starting Prefix Cache Benchmark (MPS) ===");
        long startTime = System.currentTimeMillis();

        for (int i = 0; i < 1000; i++) {
            final int requestId = i;
            executor.submit(() -> {
                String sid = "session-" + requestId;
                try (PagedKvBuffer kv = new PagedKvBuffer(sid, manager, numLayers)) {
                    // 1. 模拟处理前缀 (Prefix Matching)
                    // 实际应用中，这里会先算一遍前缀的 Hash
                    int prefixBlock = manager.getOrAllocateBlock(SYSTEM_PROMPT_HASH, sid, kv);
                    if (prefixBlock != -1) cacheHits.add(1);

                    // 2. 模拟正常的生成数据写入
                    Tensor input = randn(new long[]{512, headDim}, options);
                    for (int l = 0; l < numLayers; l++) {
                        kv.prefillUltra(l, 0, input);
                    }

                    totalTokens.add(512 + prefixLen);
                    input.deallocate();
                }
            });
            if (i % 50 == 0) Thread.sleep(20);
        }

        executor.shutdown();
        executor.awaitTermination(1, TimeUnit.HOURS);

        long duration = System.currentTimeMillis() - startTime;
        System.out.println("--------------------------------------------");
        System.out.println("Total Tokens: " + totalTokens.sum());
        System.out.println("Cache Hit Count: " + cacheHits.sum() + " (Efficiency: " + (cacheHits.sum()/10.0) + "%)");
        System.out.println("Throughput with Prefix Cache: " + String.format("%.2f", totalTokens.sum() / (duration / 1000.0)) + " tokens/sec");

        manager.close();
    }
}