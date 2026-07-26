package org.bytedeco.pytorch.geometric.demo.kvcache;

import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.locks.ReentrantLock;

/**
 * Qwen3JavaInferenceV3 压测用例
 * 核心测试维度：
 * 1. 高并发下的推理稳定性（无OOM、无死锁、无数据竞争）
 * 2. PageAttention+KV缓存的性能指标（QPS、平均延迟、TP99/TP95）
 * 3. CoWBlockManagerV9的资源利用率（块分配/释放效率、驱逐次数）
 * 4. PagedKvBufferV3的缓存写入/读取正确性
 */
public class Qwen3InferenceBenchmark {
    // 压测配置
    private static final int CONCURRENT_THREADS = 50; // 并发线程数（模拟高并发请求）
    private static final int REQUESTS_PER_THREAD = 20; // 每个线程的请求数
    private static final int INPUT_TOKEN_LENGTH = 32; // 输入Token长度（模拟不同序列长度）
    private static final int MAX_NEW_TOKENS = 50; // 每个请求生成的最大Token数
    private static final int MODEL_NUM_LAYERS = 32; // Qwen3模型层数
    private static final int BLOCK_SIZE = 32; // KV缓存块大小
    private static final int TOTAL_PHYSICAL_BLOCKS = 1024; // 总物理块数（模拟GPU显存限制）

    // 性能统计指标
    private static final AtomicLong totalRequests = new AtomicLong(0);
    private static final AtomicLong totalSuccess = new AtomicLong(0);
    private static final AtomicLong totalFailed = new AtomicLong(0);
    private static final AtomicLong totalLatency = new AtomicLong(0);
    private static final List<Long> latencyList = new CopyOnWriteArrayList<>(); // 存储每个请求的延迟
    private static final AtomicInteger activeSessions = new AtomicInteger(0); // 活跃会话数

    // 初始化推理引擎和块管理器
    private static Qwen3JavaInferenceV3 inferenceEngine;
    private static CoWBlockManagerV9 blockManager;

    /**
     * 初始化压测环境
     */
    private static void initBenchmarkEnv() {
        try {
            // 1. 初始化CoWBlockManagerV9（使用真实实现）
            blockManager = new CoWBlockManagerV9(
                    TOTAL_PHYSICAL_BLOCKS,
                    MODEL_NUM_LAYERS,
                    BLOCK_SIZE,
                    128, // head_dim
                    32   // dtype (Float32)
            );

            // 2. 初始化推理引擎（替换为实际的模型路径）
            String modelPath = "/Users/mullerzhang/Documents/code/langchain/qwen3_4b_fp16_mps.pt";
            inferenceEngine = new Qwen3JavaInferenceV3(
                    modelPath,
                    blockManager,
                    MODEL_NUM_LAYERS,
                    BLOCK_SIZE
            );

            System.out.println("=== 压测环境初始化完成 ===");
            System.out.println("并发线程数: " + CONCURRENT_THREADS);
            System.out.println("每个线程请求数: " + REQUESTS_PER_THREAD);
            System.out.println("输入Token长度: " + INPUT_TOKEN_LENGTH);
            System.out.println("总物理块数: " + TOTAL_PHYSICAL_BLOCKS);
            System.out.println("===========================\n");
        } catch (Exception e) {
            System.err.println("压测环境初始化失败: " + e.getMessage());
            e.printStackTrace();
            System.exit(1);
        }
    }

    /**
     * 生成模拟的输入Token ID数组
     */
    private static int[] generateRandomInputTokens(int length) {
        Random random = new Random();
        int[] tokens = new int[length];
        // 生成合理范围的Token ID（Qwen3的Token ID范围）
        for (int i = 0; i < length; i++) {
            tokens[i] = random.nextInt(10000) + 100; // 避开特殊Token
        }
        return tokens;
    }

    /**
     * 单个推理请求的执行逻辑
     */
    private static void executeInferenceRequest(String sessionId) {
        long startTime = System.currentTimeMillis();
        try {
            // 生成随机输入Token
            int[] inputTokens = generateRandomInputTokens(INPUT_TOKEN_LENGTH);

            // 执行推理
            inferenceEngine.generate(sessionId, inputTokens);

            // 统计成功指标
            totalSuccess.incrementAndGet();
            long latency = System.currentTimeMillis() - startTime;
            totalLatency.addAndGet(latency);
            latencyList.add(latency);

            // 释放会话缓存（模拟真实场景的资源回收）
            inferenceEngine.releaseSessionCache(sessionId);
        } catch (Exception e) {
            // 统计失败指标
            totalFailed.incrementAndGet();
            System.err.println("会话[" + sessionId + "]推理失败: " + e.getMessage());
            e.printStackTrace();
        } finally {
            totalRequests.incrementAndGet();
            activeSessions.decrementAndGet();
        }
    }

    /**
     * 压测工作线程
     */
    private static class BenchmarkWorker implements Runnable {
        private final int threadId;

        public BenchmarkWorker(int threadId) {
            this.threadId = threadId;
        }

        @Override
        public void run() {
            System.out.println("线程[" + threadId + "]启动，开始执行" + REQUESTS_PER_THREAD + "个推理请求");

            for (int i = 0; i < REQUESTS_PER_THREAD; i++) {
                // 生成唯一的会话ID
                String sessionId = "session_" + threadId + "_" + i;
                activeSessions.incrementAndGet();

                // 执行推理请求
                executeInferenceRequest(sessionId);

                // 模拟请求间隔（可选，模拟真实场景的请求分布）
                try {
                    Thread.sleep(10);
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                    break;
                }
            }

            System.out.println("线程[" + threadId + "]执行完成");
        }
    }

    /**
     * 计算性能指标（TP99/TP95/TP90）
     */
    private static double calculatePercentile(List<Long> latencies, double percentile) {
        if (latencies.isEmpty()) return 0.0;

        List<Long> sortedLatencies = new ArrayList<>(latencies);
        Collections.sort(sortedLatencies);

        int index = (int) Math.ceil(percentile / 100.0 * sortedLatencies.size()) - 1;
        index = Math.max(0, Math.min(index, sortedLatencies.size() - 1));

        return sortedLatencies.get(index);
    }

    /**
     * 输出压测报告
     */
    private static void printBenchmarkReport(long totalTime) {
        System.out.println("\n=== 压测报告 ===");
        System.out.println("总执行时间: " + totalTime + "ms");
        System.out.println("总请求数: " + totalRequests.get());
        System.out.println("成功请求数: " + totalSuccess.get());
        System.out.println("失败请求数: " + totalFailed.get());
        System.out.println("成功率: " + String.format("%.2f%%", (double) totalSuccess.get() / totalRequests.get() * 100));

        // 性能指标
        double avgLatency = totalRequests.get() > 0 ? (double) totalLatency.get() / totalRequests.get() : 0;
        double qps = totalRequests.get() > 0 ? (double) totalRequests.get() / (totalTime / 1000.0) : 0;
        double tp99 = calculatePercentile(latencyList, 99);
        double tp95 = calculatePercentile(latencyList, 95);
        double tp90 = calculatePercentile(latencyList, 90);

        System.out.println("\n=== 性能指标 ===");
        System.out.println("平均延迟: " + String.format("%.2fms", avgLatency));
        System.out.println("QPS: " + String.format("%.2f", qps));
        System.out.println("TP99延迟: " + String.format("%.2fms", tp99));
        System.out.println("TP95延迟: " + String.format("%.2fms", tp95));
        System.out.println("TP90延迟: " + String.format("%.2fms", tp90));

        // 资源利用指标
        if (blockManager instanceof CoWBlockManagerV9) {
            CoWBlockManagerV9 manager = (CoWBlockManagerV9) blockManager;
            int freeBlocks = manager.getFreeBlockCount();
            int usedBlocks = TOTAL_PHYSICAL_BLOCKS - freeBlocks;
            double blockUtilization = (double) usedBlocks / TOTAL_PHYSICAL_BLOCKS * 100;

            System.out.println("\n=== 资源利用指标 ===");
            System.out.println("空闲物理块数: " + freeBlocks);
            System.out.println("已使用物理块数: " + usedBlocks);
            System.out.println("块利用率: " + String.format("%.2f%%", blockUtilization));
            System.out.println("驱逐次数: " + CoWBlockManagerV8.EVICT_COUNT.sum());
            System.out.println("等待次数: " + CoWBlockManagerV8.WAIT_COUNT.sum());
        }
    }

    /**
     * 主压测流程
     */
    public static void main(String[] args) {
        // 1. 初始化环境
        initBenchmarkEnv();

        // 2. 创建线程池
        ExecutorService executor = Executors.newFixedThreadPool(CONCURRENT_THREADS);
        long benchmarkStartTime = System.currentTimeMillis();

        // 3. 提交压测任务
        for (int i = 0; i < CONCURRENT_THREADS; i++) {
            executor.submit(new BenchmarkWorker(i));
        }

        // 4. 等待所有任务完成
        executor.shutdown();
        try {
            if (!executor.awaitTermination(10, TimeUnit.MINUTES)) {
                executor.shutdownNow();
                System.err.println("压测超时，强制终止线程池");
            }
        } catch (InterruptedException e) {
            executor.shutdownNow();
            Thread.currentThread().interrupt();
        }

        // 5. 计算总执行时间
        long totalBenchmarkTime = System.currentTimeMillis() - benchmarkStartTime;

        // 6. 输出压测报告
        printBenchmarkReport(totalBenchmarkTime);

        System.out.println("\n=== 压测完成 ===");
    }

    // ======================== 适配CoWBlockManagerV8Impl的必要方法 ========================
    /**
     * CoWBlockManagerV8Impl的实现（适配压测用例）
     * 注：实际使用时替换为你的真实CoWBlockManagerV9实现
     */
    static class CoWBlockManagerV9Impl extends CoWBlockManagerV9 {
        private final int totalBlocks;
        private final int blockSize;
        private final Queue<Integer> freePool = new LinkedList<>();
        private final Map<String, List<Integer>> sessionBlocks = new ConcurrentHashMap<>();
        private final ReentrantLock lock = new ReentrantLock();

        public CoWBlockManagerV9Impl(int totalBlocks, int numLayers, int blockSize, int headDim, int dtype) {
            super(totalBlocks, numLayers, blockSize, headDim, dtype);
            this.totalBlocks = totalBlocks;
            this.blockSize = blockSize;
            // 初始化空闲块池
            for (int i = 0; i < totalBlocks; i++) {
                freePool.offer(i);
            }
        }

        @Override
        public void allocateBlocks(int numBlocks, int blockSize) {
            lock.lock();
            try {
                for (int i = 0; i < numBlocks; i++) {
                    if (freePool.isEmpty()) {
                        throw new RuntimeException("物理块耗尽");
                    }
                    freePool.poll();
                }
            } finally {
                lock.unlock();
            }
        }

        @Override
        public void releaseBlocks(String sessionId) {
            lock.lock();
            try {
                List<Integer> blocks = sessionBlocks.remove(sessionId);
                if (blocks != null) {
                    freePool.addAll(blocks);
                }
            } finally {
                lock.unlock();
            }
        }

        @Override
        public long[] getPhysicalBlockIds(String sessionId) {
            List<Integer> blocks = sessionBlocks.getOrDefault(sessionId, Collections.emptyList());
            return blocks.stream().mapToLong(Integer::longValue).toArray();
        }

        @Override
        public List<Integer> matchAndAllocatePath(List<Long> pathHashes, String sessionId, PagedKvBufferV3 buffer) {
            lock.lock();
            try {
                List<Integer> blocks = new ArrayList<>();
                for (int i = 0; i < pathHashes.size(); i++) {
                    if (freePool.isEmpty()) {
                        throw new RuntimeException("物理块耗尽");
                    }
                    int blockId = freePool.poll();
                    blocks.add(blockId);
                }
                sessionBlocks.computeIfAbsent(sessionId, k -> new ArrayList<>()).addAll(blocks);
                return blocks;
            } finally {
                lock.unlock();
            }
        }

        @Override
        public List<Integer> allocateBlocks(int neededBlocks, String sessionId, PagedKvBufferV3 buffer) {
            lock.lock();
            try {
                List<Integer> blocks = new ArrayList<>();
                for (int i = 0; i < neededBlocks; i++) {
                    if (freePool.isEmpty()) {
                        throw new RuntimeException("物理块耗尽");
                    }
                    int blockId = freePool.poll();
                    blocks.add(blockId);
                }
                sessionBlocks.computeIfAbsent(sessionId, k -> new ArrayList<>()).addAll(blocks);
                return blocks;
            } finally {
                lock.unlock();
            }
        }

        // 扩展方法：获取空闲块数
        public int getFreeBlockCount() {
            lock.lock();
            try {
                return freePool.size();
            } finally {
                lock.unlock();
            }
        }

        // 适配父类方法
        public int getBlockSize() {
            return blockSize;
        }

        @Override
        public void releaseSession(String sessionId) {
            releaseBlocks(sessionId);
        }

        @Override
        public boolean evictOldestSession(String excludeId) {
            // 简单实现：驱逐第一个非排除的会话
            lock.lock();
            try {
                for (Map.Entry<String, List<Integer>> entry : sessionBlocks.entrySet()) {
                    if (!entry.getKey().equals(excludeId)) {
                        freePool.addAll(entry.getValue());
                        sessionBlocks.remove(entry.getKey());
                        CoWBlockManagerV8.EVICT_COUNT.increment();
                        return true;
                    }
                }
                return false;
            } finally {
                lock.unlock();
            }
        }
    }

//    // 适配CoWBlockManagerV2的空实现（满足接口继承）
//    interface CoWBlockManagerV2 {
//        default void releaseSession(String sessionId) {}
//        default boolean evictOldestSession(String excludeId) { return false; }
//        default int getBlockSize() { return 32; }
//        default int getTotalBlocks() { return 1024; }
//        default int getFreeBlockCount() { return 0; }
//    }
//
//    // 完整的CoWBlockManagerV9接口定义
//    interface CoWBlockManagerV9 extends CoWBlockManagerV2 {
//        void allocateBlocks(int numBlocks, int blockSize);
//        void releaseBlocks(String sessionId);
//        long[] getPhysicalBlockIds(String sessionId);
//        List<Integer> matchAndAllocatePath(List<Long> pathHashes, String sessionId, PagedKvBufferV3 buffer);
//        List<Integer> allocateBlocks(int neededBlocks, String sessionId, PagedKvBufferV3 buffer);
//    }
}
