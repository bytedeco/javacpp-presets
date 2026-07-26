package org.bytedeco.pytorch.geometric.demo.kvcache;
import org.bytedeco.pytorch.jit.*;

import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorOptions;
import org.bytedeco.pytorch.ScalarTypeOptional;
import org.bytedeco.pytorch.global.torch;
import java.util.concurrent.*;
import java.util.concurrent.atomic.LongAdder;

public class PagedAttentionLoadTest {
    public static void main(String[] args) throws InterruptedException {
        int numLayers = 32;
        int blockSize = 16;
        int headDim = 128;
        int prefillLen = 512;
        int decodeLen = 128;
        int concurrentUsers = 16;

        PagedBlockManager manager = new PagedBlockManager(
                10000, numLayers, blockSize, headDim, torch.kFloat().value);

        ExecutorService executor = Executors.newFixedThreadPool(concurrentUsers);
        LongAdder totalTokensProcessed = new LongAdder();
        long startTime = System.currentTimeMillis();

        for (int i = 0; i < concurrentUsers; i++) {
            executor.submit(() -> {
                try (PagedKvBuffer kv = new PagedKvBuffer("user-" + Thread.currentThread().getId(), manager)) {
                    TensorOptions options = new TensorOptions().dtype(new ScalarTypeOptional(torch.kFloat()));

                    // 1. 模拟 Prefill 阶段 [512, 128]
                    Tensor prefillK = torch.randn(new long[]{prefillLen, headDim}, options);
                    Tensor prefillV = torch.randn(new long[]{prefillLen, headDim}, options);

                    for (int l = 0; l < numLayers; l++) {
                        kv.prefill(l, 0, prefillK);
                        kv.prefill(l, 1, prefillV);
                    }
                    kv.advanceTokens(prefillLen);
                    totalTokensProcessed.add(prefillLen);

                    // 2. 模拟 Decode 阶段 (逐个 Token)
                    Tensor decodeK = torch.randn(new long[]{headDim}, options);
                    Tensor decodeV = torch.randn(new long[]{headDim}, options);

                    for (int d = 0; d < decodeLen; d++) {
                        for (int l = 0; l < numLayers; l++) {
                            // 这里复用逻辑，可以使用优化过的 append 或单 token prefill
                            kv.prefill(l, 0, decodeK.reshape(1, headDim));
                            kv.prefill(l, 1, decodeV.reshape(1, headDim));
                        }
                        kv.advanceTokens(1);
                        totalTokensProcessed.add(1);
                    }

                    prefillK.deallocate(); prefillV.deallocate();
                    decodeK.deallocate(); decodeV.deallocate();
                }
            });
        }

        executor.shutdown();
        executor.awaitTermination(1, TimeUnit.HOURS);

        long endTime = System.currentTimeMillis();
        double seconds = (endTime - startTime) / 1000.0;

        System.out.println("Paged Attention Benchmark Finished.");
        System.out.println("Total Tokens Ingested: " + totalTokensProcessed.sum());
        System.out.println("Throughput: " + (totalTokensProcessed.sum() / seconds) + " tokens/sec");

        manager.close();
    }
}