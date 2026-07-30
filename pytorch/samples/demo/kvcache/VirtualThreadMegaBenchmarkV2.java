import org.bytedeco.pytorch.jit.*;

//package samples.demo.kvcache;
//
//import org.bytedeco.pytorch.llm.kvcache.CoWBlockManager;
//import org.bytedeco.pytorch.llm.kvcache.PagedKvBuffer;
//
//import java.time.Duration;
//import java.time.Instant;
//import java.util.*;
//import java.util.concurrent.*;
//import java.util.concurrent.atomic.LongAdder;
//
//public class VirtualThreadMegaBenchmarkV2 {
//    public static void main(String[] args) throws InterruptedException {
//        // 降低池容量，故意制造“拥堵”
//        int totalBlocks = 200000;
//        int numLayers = 32;
//        int blockSize = 16;
//        CoWBlockManager manager = new CoWBlockManager(totalBlocks, numLayers, blockSize, 128, 0);
//
//        int totalUsers = 1000;
//        LongAdder successCount = new LongAdder();
//        LongAdder totalTokens = new LongAdder();
//
//        List<Long> commonPrefix = Arrays.asList(101L, 102L, 103L, 104L, 105L);
//
//        System.out.println("🔥 启动重量级压测：模拟真实 GPU 任务流水线...");
//        Instant startTime = Instant.now();
//
//        // 使用信号量限制“进入 GPU”的物理线程数，模拟真实的算力瓶颈
//        Semaphore gpuCores = new Semaphore(16);
//
//        try (var executor = Executors.newVirtualThreadPerTaskExecutor()) {
//            for (int i = 0; i < totalUsers; i++) {
//                final int userId = i;
//                executor.submit(() -> {
//                    String sid = "user-" + userId;
//                    try {
//                        // 1. 模拟前置处理（不占 GPU）
//                        List<Long> userPath = new ArrayList<>(commonPrefix);
//                        for (int j = 0; j < 5; j++) userPath.add((long) (userId * 100 + j));
//
//                        // 2. 模拟 KV Cache 逐层加载（这是最耗时且占内存的部分）
//                        try (PagedKvBuffer kv = new PagedKvBuffer(sid, manager, numLayers)) {
//
//                            // 分层请求 Block
//                            for (int layer = 0; layer < numLayers; layer++) {
//                                // 模拟每一层都需要申请内存并等待 GPU 计算
//                                gpuCores.acquire();
//                                try {
//                                    manager.matchAndAllocatePath(userPath, sid, kv);
//                                    // 模拟 GPU 算子执行耗时
////                                    Thread.sleep(Duration.ofMillis(2));
//                                } finally {
//                                    gpuCores.release();
//                                }
//                            }
//
//                            successCount.increment();
//                            totalTokens.add(userPath.size() * blockSize);
//                        }
//                    } catch (Exception e) {
//                        // 如果内存满了且没法驱逐，会抛出 RuntimeException
//                    } finally {
//                        manager.releaseSession(sid);
//                    }
//                });
//            }
//        }
//
//        Instant endTime = Instant.now();
//        long durationMs = Duration.between(startTime, endTime).toMillis();
//
//        System.out.println("\n--------------------------------------------");
//        System.out.println("真实负载压测总结:");
//        System.out.println("运行时间: " + (durationMs / 1000.0) + " 秒");
//        System.out.println("TPS: " + String.format("%.2f", totalTokens.sum() / (durationMs / 1000.0)));
//        System.out.println("当前空闲块: " + manager.getFreeBlockCount());
//        System.out.println("--------------------------------------------");
//    }
//}
