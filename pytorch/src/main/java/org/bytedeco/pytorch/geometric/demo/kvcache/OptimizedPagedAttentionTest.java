package org.bytedeco.pytorch.geometric.demo.kvcache;

import org.bytedeco.pytorch.ScalarTypeOptional;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorOptions;
import org.bytedeco.pytorch.global.torch;
import java.util.concurrent.*;
import java.util.concurrent.atomic.LongAdder;

public class OptimizedPagedAttentionTest {
    public static void main(String[] args) throws InterruptedException {
        int numLayers = 32;
        int blockSize = 16;
        int headDim = 128;
        int prefillBatch = 1024; // 增加批处理大小
        int threads = 1000;// Runtime.getRuntime().availableProcessors();

        CoWBlockManager manager = new CoWBlockManager(20000, numLayers, blockSize, headDim, torch.kFloat().value);
        ExecutorService executor = Executors.newFixedThreadPool(threads);
        LongAdder tokenCounter = new LongAdder();

        System.out.println("Starting Optimized Prefill Benchmark (Threads: " + threads + ")...");
        long startTime = System.currentTimeMillis();

        for (int i = 0; i < 100; i++) { // 总共 100 个大请求
            executor.submit(() -> {
                try (PagedKvBuffer2 kv = new PagedKvBuffer2(manager)) {
                    // 模拟输入 Tensor [1024, 128]
                    Tensor inputK = torch.randn(new long[]{prefillBatch, headDim}, new TensorOptions().dtype(new ScalarTypeOptional(torch.kFloat())));
                    Tensor inputV = torch.randn(new long[]{prefillBatch, headDim}, new TensorOptions().dtype(new ScalarTypeOptional(torch.kFloat())));

                    for (int l = 0; l < numLayers; l++) {
//                        kv.prefillOptimized(l, 0, inputK);
//                        kv.prefillOptimized(l, 1, inputV);
                        kv.prefillUltra(l, 0, inputK);
                        kv.prefillUltra(l, 1, inputV);
                    }
                    kv.advance(prefillBatch);
                    tokenCounter.add(prefillBatch);

                    inputK.deallocate(); inputV.deallocate();
                } catch (Exception e) { e.printStackTrace(); }
            });
        }

        executor.shutdown();
        executor.awaitTermination(1, TimeUnit.HOURS);

        long duration = System.currentTimeMillis() - startTime;
        System.out.println("Benchmark Result:");
        System.out.println("Total Tokens: " + tokenCounter.sum());
        System.out.println("TPS: " + (tokenCounter.sum() / (duration / 1000.0)) + " tokens/sec");

        manager.close();
    }
}


/***
 * Starting Optimized Prefill Benchmark (Threads: 1000)...
 * Benchmark Result:
 * Total Tokens: 102400
 * TPS: 7649.783355744808 tokens/sec
 * ***/