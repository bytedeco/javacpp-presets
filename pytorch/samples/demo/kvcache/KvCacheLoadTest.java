package samples.demo.kvcache;
import org.bytedeco.pytorch.jit.*;


import org.bytedeco.pytorch.ScalarTypeOptional;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorOptions;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.llm.kvcache.KvBufferCache;

import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.LongAdder;

//import static org.bytedeco.pytorch.global.torch.kFloat32;

public class KvCacheLoadTest {

    public static void main(String[] args) throws InterruptedException {
        int threads = 16;
        int sessions = 100;
        int layers = 32;
        int ctxLen = 2048;
        int kvDim = 128;

        System.out.println("Starting LibTorch KV Cache Load Test...");
        KvBufferCache cache = new KvBufferCache(torch.kFloat().value, layers, ctxLen, kvDim);

        ExecutorService executor = Executors.newFixedThreadPool(threads);
        LongAdder opCount = new LongAdder();
        long startTime = System.currentTimeMillis();

        for (int i = 0; i < threads; i++) {
            executor.submit(() -> {
                try {
                    // 每个线程模拟处理不同的 Session
                    for (int s = 0; s < sessions / threads; s++) {
                        String sid = "session-" + Thread.currentThread().getId() + "-" + s;
                        KvBufferCache.KvBuffer kv = cache.getKvBuffer(sid);
//                        torch.ScalarType st = torch.ScalarType.Undefined;
//                        for (torch.ScalarType e : torch.ScalarType.values()) {
//                            if (e.value == scalarType) {
//                                st = e;
//                                break;
//                            }
//                        }
                        // 构造模拟数据 [kv_dim]
                        Tensor mockToken = torch.rand(new long[]{kvDim}, new TensorOptions().dtype(new ScalarTypeOptional(torch.kFloat())));

                        // 模拟生成过程：逐个 Token 写入
                        for (int p = 0; p < 100; p++) {
                            for (int l = 0; l < layers; l++) {
                                kv.append(l, 0, mockToken); // 写入 Key
                                kv.append(l, 1, mockToken); // 写入 Value

                                // 模拟 Attention 读取：获取历史所有 K
//                                Tensor history = kv.getTensorsUpTo(l, 0, p + 1);
//                                opCount.add(1);
                                // 关键：history 对象如果不 deallocate，会造成短时间内 Java 对象堆积
                                try (Tensor history = kv.getTensorsUpTo(l, 0, p + 1)) {
                                    opCount.add(1);
                                } //
                            }
                            kv.incrementPosition();
                        }
                    }
                } catch (Exception e) {
                    e.printStackTrace();
                }
            });
        }

        executor.shutdown();
        executor.awaitTermination(1, TimeUnit.HOURS);

        long duration = System.currentTimeMillis() - startTime;
        System.out.println("Test Finished.");
        System.out.println("Total Slice/Append Ops: " + opCount.sum());
        System.out.println("Throughput: " + (opCount.sum() / (duration / 1000.0)) + " ops/sec");

        cache.close();
    }
}