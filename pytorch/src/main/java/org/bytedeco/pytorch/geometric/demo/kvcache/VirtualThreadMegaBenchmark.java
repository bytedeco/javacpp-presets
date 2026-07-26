package org.bytedeco.pytorch.geometric.demo.kvcache;
import org.bytedeco.pytorch.jit.*;

import java.time.Duration;
import java.time.Instant;
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.LongAdder;

public class VirtualThreadMegaBenchmark {
    public static void main(String[] args) throws InterruptedException {
        // --- 1. 初始化管理器 ---
        int totalBlocks = 5000;     // 物理池容量（5000块支持万级用户必须要靠 Radix Tree 共享）
        int numLayers = 32;
        int blockSize = 16;
        CoWBlockManagerV8 manager = new CoWBlockManagerV8(totalBlocks, numLayers, blockSize, 128, 0);

        // --- 2. 压测配置 ---
        int totalUsers = 10000;    // 一万个虚拟线程用户
        LongAdder successCount = new LongAdder();
        LongAdder failCount = new LongAdder();
        LongAdder totalTokens = new LongAdder();

        // 模拟三个不同层级的公共前缀（模拟不同的热门话题）
        List<Long> commonPrefixA = Arrays.asList(101L, 102L, 103L, 104L, 105L); // 深度前缀
        List<Long> commonPrefixB = Arrays.asList(201L, 202L);                // 浅层前缀

        System.out.println("🚀 启动 JDK 25 虚拟线程万级并发压测...");
        System.out.println("用户总数: " + totalUsers + " | 物理块限制: " + totalBlocks);

        Instant startTime = Instant.now();

        // --- 3. 使用虚拟线程执行器 ---
        try (var executor = Executors.newVirtualThreadPerTaskExecutor()) {
            for (int i = 0; i < totalUsers; i++) {
                final int userId = i;
                executor.submit(() -> {
                    String sid = "user-" + userId;
                    try (PagedKvBufferV3 kv = new PagedKvBufferV3(sid, manager, numLayers)) {

                        // 模拟用户行为：有的用户共享 A 话题，有的共享 B，有的完全独立
                        List<Long> userPath = new ArrayList<>();
                        if (userId % 2 == 0) userPath.addAll(commonPrefixA);
                        else if (userId % 3 == 0) userPath.addAll(commonPrefixB);

                        // 加上用户自己的 5 个独占 Block
                        for (int j = 0; j < 5; j++) {
                            userPath.add((long) (userId * 100 + j));
                        }

                        // Radix Tree 路径分配
                        // 这一步会发生激烈的锁竞争和 Condition 等待
                        manager.matchAndAllocatePath(userPath, sid, kv);

                        // 模拟推理耗时（虚拟线程在 sleep 时会挂起，不占用真实 CPU 线程）
                        Thread.sleep(Duration.ofMillis(10 + (userId % 50)));

                        successCount.increment();
                        totalTokens.add(userPath.size() * blockSize);

                    } catch (Exception e) {
                        failCount.increment();
                        // 内存实在不够或者超时会跑到这里
                    } finally {
                        manager.releaseSession(sid);
                    }
                });
            }
        } // executor.close() 会等待所有虚拟线程执行完毕

        // --- 4. 统计结果 ---
        Instant endTime = Instant.now();
        long durationMs = Duration.between(startTime, endTime).toMillis();

        System.out.println("\n--------------------------------------------");
        System.out.println("JDK 25 虚拟线程压测总结:");
        System.out.println("耗时: " + (durationMs / 1000.0) + " 秒");
        System.out.println("成功请求数: " + successCount.sum());
        System.out.println("失败请求数: " + failCount.sum());
        System.out.println("总处理 Token: " + totalTokens.sum());
        System.out.println("每秒吞吐量 (TPS): " + String.format("%.2f", totalTokens.sum() / (durationMs / 1000.0)));
        System.out.println("最终空闲块: " + manager.getFreeBlockCount());
        System.out.println("--------------------------------------------");
    }
}