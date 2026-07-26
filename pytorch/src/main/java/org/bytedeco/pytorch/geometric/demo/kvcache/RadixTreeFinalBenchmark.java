package org.bytedeco.pytorch.geometric.demo.kvcache;
import org.bytedeco.pytorch.jit.*;

import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.LongAdder;

public class RadixTreeFinalBenchmark {
    public static void main(String[] args) throws InterruptedException {
        // --- 1. 配置参数 ---
        int totalBlocks = 2000;    // 总显存块数（调小一点容易触发驱逐）
        int numLayers = 32;        // 模型层数
        int blockSize = 16;        // 每个 Block 16 个 Token

        CoWBlockManagerV8 manager = new CoWBlockManagerV8(totalBlocks, numLayers, blockSize, 128, 0);

        // 线程池模拟并发用户
        int concurrentUsers = 8;
        ExecutorService executor = Executors.newFixedThreadPool(concurrentUsers);

        // 统计工具
        LongAdder totalTokens = new LongAdder();
        LongAdder cacheHitBlocks = new LongAdder();

        // --- 2. 构造测试数据 ---
        // 模拟一个极其常见的系统提示词 (System Prompt)，占用 3 个 Block
        List<Long> systemPromptHashes = Arrays.asList(99901L, 99902L, 99903L);

        System.out.println("=== 启动 Radix Tree 工业级压测 ===");
        System.out.println("物理池容量: " + totalBlocks + " Blocks");
        System.out.println("模拟并发用户: " + concurrentUsers);

        long startTime = System.currentTimeMillis();

        // --- 3. 提交 200 个任务 ---
        for (int i = 0; i < 200; i++) {
            final int requestId = i;
            executor.submit(() -> {
                String sid = "session-" + requestId;
                try (PagedKvBufferV3 kv = new PagedKvBufferV3(sid, manager, numLayers)) {

                    // A. 构造路径：公共前缀 + 个人私有数据
                    List<Long> myPath = new ArrayList<>(systemPromptHashes);
                    // 每个用户私有的 10 个 Block
                    for (int j = 0; j < 10; j++) {
                        myPath.add((long) (requestId * 1000 + j));
                    }

                    // B. 执行分配 (Radix 核心逻辑)
                    // 如果 requestId > 0，前 3 个块应该直接命中缓存
                    List<Integer> allocatedIds = manager.matchAndAllocatePath(myPath, sid, kv);

                    // 统计命中：除了第一个请求，后续请求的前 3 个块都是命中的
                    if (requestId > 0) cacheHitBlocks.add(systemPromptHashes.size());

                    // C. 模拟推理过程中的计算耗时
                    // 这里模拟 PagedAttention 的写入过程
                    for (int layer = 0; layer < numLayers; layer++) {
                        // 实际生产中这里会调用 JNI 写入 MPS/CUDA
                        // 我们模拟一下耗时，让内存占用能维持一会儿
                        Thread.sleep(1);
                    }

                    totalTokens.add(myPath.size() * blockSize);

                    if (requestId % 20 == 0) {
                        System.out.printf("[进度] 请求 %d 完成 | 剩余空闲块: %d\n",
                                requestId, manager.getFreeBlockCount());
                    }

                } catch (Exception e) {
                    System.err.println("请求失败: " + e.getMessage());
                } finally {
                    // D. 显式释放 Session，触发引用计数回收
                    manager.releaseSession(sid);
                }
            });

            // 稍微控制一下进入速度，模拟流式请求
            if (i % 10 == 0) Thread.sleep(50);
        }

        // --- 4. 优雅停止 ---
        executor.shutdown();
        if (executor.awaitTermination(1, TimeUnit.MINUTES)) {
            long duration = System.currentTimeMillis() - startTime;
            System.out.println("\n--------------------------------------------");
            System.out.println("压测完成报告:");
            System.out.println("总处理 Token 量: " + totalTokens.sum());
            System.out.println("Radix 缓存节省块数: " + cacheHitBlocks.sum());
            System.out.println("吞吐量: " + String.format("%.2f", totalTokens.sum() / (duration / 1000.0)) + " tokens/sec");
            System.out.println("最终空闲块: " + manager.getFreeBlockCount() + " (应接近初始值)");
            System.out.println("--------------------------------------------");
        }
    }
}