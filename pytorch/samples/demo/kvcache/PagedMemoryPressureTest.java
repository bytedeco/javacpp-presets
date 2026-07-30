package samples.demo.kvcache;
import org.bytedeco.pytorch.jit.*;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.llm.kvcache.CoWBlockManager;

import static org.bytedeco.pytorch.global.torch.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.LongAdder;

public class PagedMemoryPressureTest {
    public static void main(String[] args) throws InterruptedException {
        int numLayers = 32;
        int blockSize = 16;
        int headDim = 128;

        // 1. 故意缩小物理块总量，模拟“显存几乎填满”的极限情况
        // 假设我们只有 5000 个块，很快就会被填满
        int totalPhysicalBlocks = 5000;
//        CoWBlockManager manager = CoWBlockManager.withDtypeValue(totalPhysicalBlocks, numLayers, blockSize, headDim, kFloat().value);
        CoWBlockManager manager = CoWBlockManager.withDtypeValue(totalPhysicalBlocks, numLayers, blockSize, headDim, kFloat().value);

        int threads = 12; // 匹配 Mac M 芯片的核心数
        ExecutorService executor = Executors.newFixedThreadPool(threads);
        LongAdder tokenCounter = new LongAdder();

        System.out.println("=== Starting Paged Memory Pressure Test (MPS) ===");
        System.out.println("Total Physical Blocks: " + totalPhysicalBlocks);
        System.out.println("Max Capacity: ~" + (totalPhysicalBlocks * blockSize / 1000) + "k tokens");
        TensorOptions options =new TensorOptions().dtype(new ScalarTypeOptional(kFloat())).device(new DeviceOptional(new Device(kMPS())));

        long globalStart = System.currentTimeMillis();

        // 2. 持续模拟 500 个请求，远超总块容量，强制触发回收
        for (int i = 0; i < 500; i++) {
            final int requestId = i;
            executor.submit(() -> {
                // 每个请求处理 512 个 token
                int prefillBatch = 512;

                try (PagedKvBuffer2 kv = new PagedKvBuffer2(manager)) {
                    Tensor inputK = randn(new long[]{prefillBatch, headDim}, options);
                    Tensor inputV = randn(new long[]{prefillBatch, headDim}, options);

                    long reqStart = System.currentTimeMillis();

                    for (int l = 0; l < numLayers; l++) {
                        kv.prefillUltra(l, 0, inputK);
                        kv.prefillUltra(l, 1, inputV);
                    }
                    kv.advance(prefillBatch);
                    tokenCounter.add(prefillBatch);

                    long reqDuration = System.currentTimeMillis() - reqStart;

                    // 打印关键指标：当前的块使用率和处理时间
                    if (requestId % 50 == 0) {
                        Integer usage = (Integer) manager.getActiveBlockCount() / totalPhysicalBlocks * 100;
                        System.out.printf("[Req %d] Block Usage: %.2f%% | Req Time: %dms | Current TPS: %.2f\n",
                                requestId, usage, reqDuration,
                                (tokenCounter.sum() / ((System.currentTimeMillis() - globalStart) / 1000.0)));
                    }

                    inputK.deallocate(); inputV.deallocate();
                } catch (Exception e) {
                    System.err.println("Request " + requestId + " failed: " + e.getMessage());
                }
            });

            // 稍微放慢发射速度，观察平滑回收过程
            Thread.sleep(50);
        }

        executor.shutdown();
        executor.awaitTermination(1, TimeUnit.HOURS);

        System.out.println("\nFinal Result:");
        System.out.println("Total Tokens Processed: " + tokenCounter.sum());
        System.out.println("Average Overall TPS: " + (tokenCounter.sum() / ((System.currentTimeMillis() - globalStart) / 1000.0)));

        manager.close();
    }
}