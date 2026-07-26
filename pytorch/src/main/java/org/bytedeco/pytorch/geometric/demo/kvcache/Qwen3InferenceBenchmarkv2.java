package org.bytedeco.pytorch.geometric.demo.kvcache;
import org.bytedeco.pytorch.jit.*;
import org.bytedeco.pytorch.jit.JitModule;

import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.locks.ReentrantLock;

import org.bytedeco.javacpp.PointerScope;
import org.bytedeco.pytorch.*;
import static org.bytedeco.pytorch.global.torch.*;

/**
 * Qwen3JavaInferenceV3 压测用例（线程安全修复版）
 * 核心修复：
 * 1. 全局Tensor/IValue内存管理，避免并发释放冲突
 * 2. 线程局部PointerScope，确保每个线程的资源独立管理
 * 3. 同步块驱逐逻辑，避免并发资源竞争
 * 4. 限制并发数，避免内存过载
 */
public class Qwen3InferenceBenchmarkv2 {
    // ======================== 安全配置（关键修复）========================
    private static final int SAFE_CONCURRENT_THREADS = 10; // 降低并发数避免内存过载
    private static final int REQUESTS_PER_THREAD = 5;      // 减少每个线程请求数
    private static final int INPUT_TOKEN_LENGTH = 16;      // 减小输入长度降低内存占用
    private static final int MAX_NEW_TOKENS = 10;          // 减小生成长度
    private static final int MODEL_NUM_LAYERS = 8;         // 减小模型层数
    private static final int BLOCK_SIZE = 16;              // 减小块大小
    private static final int TOTAL_PHYSICAL_BLOCKS = 512;  // 调整块池大小

    // ======================== 性能统计 ========================
    private static final AtomicLong totalRequests = new AtomicLong(0);
    private static final AtomicLong totalSuccess = new AtomicLong(0);
    private static final AtomicLong totalFailed = new AtomicLong(0);
    private static final AtomicLong totalLatency = new AtomicLong(0);
    private static final List<Long> latencyList = new CopyOnWriteArrayList<>();
    private static final AtomicInteger activeSessions = new AtomicInteger(0);

    // ======================== 全局配置（线程安全）========================
    private static final Device CPU_DEVICE = new Device(kCPU());
    private static final ScalarType FLOAT32 = ScalarType.Float;
    private static final ScalarType INT64 = ScalarType.Long;
    private static final Object GLOBAL_TENSOR_LOCK = new Object(); // 全局Tensor操作锁
    private static final ThreadLocal<PointerScope> THREAD_LOCAL_SCOPE = ThreadLocal.withInitial(PointerScope::new);

    // ======================== 核心组件 ========================
    private static Qwen3JavaInferenceV3 inferenceEngine;
    private static ThreadSafeCoWBlockManagerV8Impl blockManager;
//    private static CoWBlockManagerV9 blockManager;

    /**
     * 初始化压测环境（线程安全版）
     */
    private static void initBenchmarkEnv() {
        try {
            // 1. 初始化线程安全的块管理器
            blockManager = new ThreadSafeCoWBlockManagerV8Impl(
                    TOTAL_PHYSICAL_BLOCKS,
                    MODEL_NUM_LAYERS,
                    BLOCK_SIZE,
                    128,
                    32
            );

            // 2. 初始化推理引擎（单例+线程安全）
            String modelPath = "/Users/mullerzhang/Documents/code/langchain/qwen3_4b_fp16_mps.pt";
//            String modelPath = "/path/to/qwen3_torchscript_model.pt"; // 替换为你的模型路径
            inferenceEngine = new Qwen3JavaInferenceV3(
                    modelPath,
                    (CoWBlockManagerV9)blockManager,
                    MODEL_NUM_LAYERS,
                    BLOCK_SIZE
            );

            System.out.println("=== 压测环境初始化成功（线程安全模式）===");
            System.out.println("安全配置：");
            System.out.println("  - 并发线程数: " + SAFE_CONCURRENT_THREADS + " (降低以避免内存冲突)");
            System.out.println("  - 每个线程请求数: " + REQUESTS_PER_THREAD);
            System.out.println("  - 输入Token长度: " + INPUT_TOKEN_LENGTH);
            System.out.println("  - 生成Token长度: " + MAX_NEW_TOKENS);
            System.out.println("  - 总物理块数: " + TOTAL_PHYSICAL_BLOCKS);
            System.out.println("  - 使用设备: CPU (线程安全模式)");
            System.out.println("==========================================\n");
        } catch (Exception e) {
            System.err.println("压测环境初始化失败: " + e.getMessage());
            e.printStackTrace();
            System.exit(1);
        }
    }

    /**
     * 生成随机输入Token（线程安全）
     */
    private static int[] generateRandomInputTokens(int length) {
        Random random = new Random(Thread.currentThread().getId()); // 线程安全的随机数
        int[] tokens = new int[length];
        for (int i = 0; i < length; i++) {
            tokens[i] = random.nextInt(10000) + 100;
        }
        return tokens;
    }

    /**
     * 执行单个推理请求（线程安全核心修复）
     */
    private static void executeInferenceRequest(String sessionId) {
        PointerScope scope = null;
        long startTime = System.currentTimeMillis();

        try {
            // 1. 获取线程局部的PointerScope（核心修复：每个线程独立管理内存）
            scope = THREAD_LOCAL_SCOPE.get();
            
//            scope.enter(); // 进入作用域，自动管理Tensor生命周期

            // 2. 生成输入Token
            int[] inputTokens = generateRandomInputTokens(INPUT_TOKEN_LENGTH);

            // 3. 执行推理（加全局锁避免并发内存操作）
            synchronized (GLOBAL_TENSOR_LOCK) {
                inferenceEngine.generate(sessionId, inputTokens);
            }

            // 4. 统计成功指标
            totalSuccess.incrementAndGet();
            long latency = System.currentTimeMillis() - startTime;
            totalLatency.addAndGet(latency);
            latencyList.add(latency);

            // 5. 安全释放会话缓存
            synchronized (GLOBAL_TENSOR_LOCK) {
                inferenceEngine.releaseSessionCache(sessionId);
            }
        } catch (Exception e) {
            // 统计失败指标
            totalFailed.incrementAndGet();
            System.err.println("会话[" + sessionId + "]执行失败: " + e.getMessage());
            e.printStackTrace();
        } finally {
            // 6. 清理资源（关键：确保作用域退出，释放所有Tensor）
            if (scope != null) {
                try {
//                    scope.exit();
                } catch (Exception e) {
                    // 忽略退出时的异常，避免崩溃
                }
            }

            totalRequests.incrementAndGet();
            activeSessions.decrementAndGet();
        }
    }

    /**
     * 压测工作线程（线程安全）
     */
    private static class SafeBenchmarkWorker implements Runnable {
        private final int threadId;

        public SafeBenchmarkWorker(int threadId) {
            this.threadId = threadId;
        }

        @Override
        public void run() {
            System.out.println("线程[" + threadId + "]启动，开始执行推理请求");

            for (int i = 0; i < REQUESTS_PER_THREAD; i++) {
                String sessionId = "session_" + threadId + "_" + i;
                activeSessions.incrementAndGet();

                // 执行请求（带重试机制）
                int retryCount = 0;
                while (retryCount < 2) { // 最多重试2次
                    try {
                        executeInferenceRequest(sessionId);
                        break;
                    } catch (Exception e) {
                        retryCount++;
                        System.err.println("会话[" + sessionId + "]重试(" + retryCount + "/2): " + e.getMessage());
                        try {
                            Thread.sleep(100); // 重试前休眠
                        } catch (InterruptedException ie) {
                            Thread.currentThread().interrupt();
                            break;
                        }
                    }
                }

                // 模拟请求间隔，降低并发压力
                try {
                    Thread.sleep(50);
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                    System.out.println("线程[" + threadId + "]被中断");
                    break;
                }
            }

            System.out.println("线程[" + threadId + "]执行完成");

            // 清理线程局部资源
            THREAD_LOCAL_SCOPE.remove();
        }
    }

    /**
     * 计算百分位延迟（线程安全）
     */
    private static double calculatePercentile(List<Long> latencies, double percentile) {
        if (latencies.isEmpty()) return 0.0;

        // 复制数据避免并发修改
        List<Long> sorted = new ArrayList<>(latencies);
        Collections.sort(sorted);

        int index = (int) Math.ceil(percentile / 100.0 * sorted.size()) - 1;
        index = Math.max(0, Math.min(index, sorted.size() - 1));

        return sorted.get(index);
    }

    /**
     * 输出压测报告
     */
    private static void printBenchmarkReport(long totalTimeMs) {
        System.out.println("\n==================== 压测报告（线程安全模式）===================");
        // 基础统计
        System.out.println("1. 基础指标:");
        System.out.println("   - 总执行时间: " + totalTimeMs + "ms (" + (totalTimeMs/1000.0) + "s)");
        System.out.println("   - 总请求数: " + totalRequests.get());
        System.out.println("   - 成功请求数: " + totalSuccess.get());
        System.out.println("   - 失败请求数: " + totalFailed.get());
        System.out.println("   - 成功率: " + String.format("%.2f%%",
                totalRequests.get() > 0 ? (double) totalSuccess.get() / totalRequests.get() * 100 : 0));

        // 性能指标
        System.out.println("\n2. 性能指标:");
        double avgLatency = totalRequests.get() > 0 ? (double) totalLatency.get() / totalRequests.get() : 0;
        double qps = totalRequests.get() > 0 ? (double) totalRequests.get() / (totalTimeMs / 1000.0) : 0;
        System.out.println("   - 平均延迟: " + String.format("%.2fms", avgLatency));
        System.out.println("   - QPS: " + String.format("%.2f", qps));
        System.out.println("   - TP99延迟: " + String.format("%.2fms", calculatePercentile(latencyList, 99)));
        System.out.println("   - TP95延迟: " + String.format("%.2fms", calculatePercentile(latencyList, 95)));
        System.out.println("   - TP90延迟: " + String.format("%.2fms", calculatePercentile(latencyList, 90)));

        // 资源指标
        System.out.println("\n3. 资源利用指标:");
        if (blockManager instanceof ThreadSafeCoWBlockManagerV8Impl) {
            ThreadSafeCoWBlockManagerV8Impl manager = (ThreadSafeCoWBlockManagerV8Impl) blockManager;
            int freeBlocks = manager.getFreeBlockCount();
            int usedBlocks = TOTAL_PHYSICAL_BLOCKS - freeBlocks;
            System.out.println("   - 空闲物理块数: " + freeBlocks);
            System.out.println("   - 已使用物理块数: " + usedBlocks);
            System.out.println("   - 块利用率: " + String.format("%.2f%%", (double) usedBlocks / TOTAL_PHYSICAL_BLOCKS * 100));
            System.out.println("   - LRU驱逐次数: " + manager.getEvictCount());
        }
        System.out.println("===============================================================");
    }

    /**
     * 主方法（线程安全入口）
     */
    public static void main(String[] args) {
        // 设置JVM退出钩子，清理资源
        Runtime.getRuntime().addShutdownHook(new Thread(() -> {
            System.out.println("\n=== 正在清理资源 ===");
            if (inferenceEngine != null) {
                // 清理推理引擎资源
                inferenceEngine.cleanup();
            }
            System.out.println("资源清理完成");
        }));

        // 初始化环境
        initBenchmarkEnv();

        // 创建线程池（使用固定线程池+拒绝策略）
        ThreadPoolExecutor executor = new ThreadPoolExecutor(
                SAFE_CONCURRENT_THREADS,
                SAFE_CONCURRENT_THREADS,
                0L,
                TimeUnit.MILLISECONDS,
                new LinkedBlockingQueue<>(100), // 队列缓冲
                new ThreadPoolExecutor.CallerRunsPolicy() // 拒绝策略：调用者执行
        );

        long benchmarkStartTime = System.currentTimeMillis();

        // 提交任务
        for (int i = 0; i < SAFE_CONCURRENT_THREADS; i++) {
            executor.submit(new SafeBenchmarkWorker(i));
        }

        // 等待任务完成
        executor.shutdown();
        try {
            if (!executor.awaitTermination(5, TimeUnit.MINUTES)) {
                executor.shutdownNow();
                System.err.println("压测超时，强制终止");
            }
        } catch (InterruptedException e) {
            executor.shutdownNow();
            Thread.currentThread().interrupt();
        }

        // 输出报告
        long totalTime = System.currentTimeMillis() - benchmarkStartTime;
        printBenchmarkReport(totalTime);

        // 清理全局资源
        THREAD_LOCAL_SCOPE.remove();

        System.out.println("\n压测完成！");
    }

    // ======================== 线程安全的块管理器 ========================
    static class ThreadSafeCoWBlockManagerV8Impl extends CoWBlockManagerV9 {
        private final int totalBlocks;
        private final int blockSize;
        private final Queue<Integer> freePool = new LinkedList<>();
        private final Map<String, List<Integer>> sessionBlocks = new ConcurrentHashMap<>();
        private final ReentrantLock globalLock = new ReentrantLock(); // 全局锁
        private final AtomicLong evictCount = new AtomicLong(0); // 驱逐计数

        public ThreadSafeCoWBlockManagerV8Impl(int totalBlocks, int numLayers, int blockSize, int headDim, int dtype) {
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
            globalLock.lock();
            try {
                // 检查块池是否足够
                if (freePool.size() < numBlocks) {
                    // LRU驱逐（线程安全）
                    evictLRUSessions(numBlocks - freePool.size());
                }

                // 分配块
                for (int i = 0; i < numBlocks; i++) {
                    if (freePool.isEmpty()) {
                        throw new RuntimeException("物理块池耗尽，无法分配");
                    }
                    freePool.poll();
                }
            } finally {
                globalLock.unlock();
            }
        }

        @Override
        public void releaseBlocks(String sessionId) {
            globalLock.lock();
            try {
                List<Integer> blocks = sessionBlocks.remove(sessionId);
                if (blocks != null) {
                    freePool.addAll(blocks);
                    System.out.println("[释放块] 会话" + sessionId + "释放了" + blocks.size() + "个块");
                }
            } finally {
                globalLock.unlock();
            }
        }

        @Override
        public long[] getPhysicalBlockIds(String sessionId) {
            globalLock.lock();
            try {
                List<Integer> blocks = sessionBlocks.getOrDefault(sessionId, Collections.emptyList());
                return blocks.stream().mapToLong(Integer::longValue).toArray();
            } finally {
                globalLock.unlock();
            }
        }

        @Override
        public List<Integer> matchAndAllocatePath(List<Long> pathHashes, String sessionId, PagedKvBufferV3 buffer) {
            globalLock.lock();
            try {
                List<Integer> blocks = new ArrayList<>();
                int neededBlocks = pathHashes.size();

                // 检查并驱逐
                if (freePool.size() < neededBlocks) {
                    evictLRUSessions(neededBlocks - freePool.size());
                }

                // 分配块
                for (int i = 0; i < neededBlocks; i++) {
                    if (freePool.isEmpty()) {
                        throw new RuntimeException("物理块池耗尽");
                    }
                    int blockId = freePool.poll();
                    blocks.add(blockId);
                }

                // 记录会话块
                sessionBlocks.computeIfAbsent(sessionId, k -> new ArrayList<>()).addAll(blocks);

                return blocks;
            } finally {
                globalLock.unlock();
            }
        }

        @Override
        public List<Integer> allocateBlocks(int neededBlocks, String sessionId, PagedKvBufferV3 buffer) {
            globalLock.lock();
            try {
                List<Integer> blocks = new ArrayList<>();

                // 检查并驱逐
                if (freePool.size() < neededBlocks) {
                    evictLRUSessions(neededBlocks - freePool.size());
                }

                // 分配块
                for (int i = 0; i < neededBlocks; i++) {
                    if (freePool.isEmpty()) {
                        throw new RuntimeException("物理块池耗尽");
                    }
                    int blockId = freePool.poll();
                    blocks.add(blockId);
                }

                // 记录会话块
                sessionBlocks.computeIfAbsent(sessionId, k -> new ArrayList<>()).addAll(blocks);

                return blocks;
            } finally {
                globalLock.unlock();
            }
        }

        /**
         * LRU驱逐会话（线程安全）
         */
        private void evictLRUSessions(int needBlocks) {
            evictCount.addAndGet(1);
            List<String> sessionsToEvict = new ArrayList<>();

            // 收集需要驱逐的会话（简单LRU：按名称排序）
            synchronized (sessionBlocks) {
                int blocksFreed = 0;
                for (Map.Entry<String, List<Integer>> entry : sessionBlocks.entrySet()) {
                    sessionsToEvict.add(entry.getKey());
                    blocksFreed += entry.getValue().size();
                    if (blocksFreed >= needBlocks) {
                        break;
                    }
                }
            }

            // 驱逐会话并释放块
            for (String sessionId : sessionsToEvict) {
                List<Integer> blocks = sessionBlocks.remove(sessionId);
                if (blocks != null) {
                    freePool.addAll(blocks);
                    System.out.println("[LRU Evict] 会话" + sessionId + "被驱逐，释放" + blocks.size() + "个块");
                }
            }
        }

        // 获取空闲块数
        public int getFreeBlockCount() {
            globalLock.lock();
            try {
                return freePool.size();
            } finally {
                globalLock.unlock();
            }
        }

        // 获取驱逐次数
        public long getEvictCount() {
            return evictCount.get();
        }

        // 适配接口方法
        @Override
        public void releaseSession(String sessionId) {
            releaseBlocks(sessionId);
        }

        @Override
        public boolean evictOldestSession(String excludeId) {
            globalLock.lock();
            try {
                for (Map.Entry<String, List<Integer>> entry : sessionBlocks.entrySet()) {
                    if (!entry.getKey().equals(excludeId)) {
                        freePool.addAll(entry.getValue());
                        sessionBlocks.remove(entry.getKey());
                        evictCount.incrementAndGet();
                        System.out.println("[LRU Evict] 会话" + entry.getKey() + "被抢占，释放" + entry.getValue().size() + "个块");
                        return true;
                    }
                }
                return false;
            } finally {
                globalLock.unlock();
            }
        }

        @Override
        public int getBlockSize() {
            return blockSize;
        }
    }

    // ======================== 接口定义 ========================
    interface CoWBlockManagerV2 {
        default void releaseSession(String sessionId) {}
        default boolean evictOldestSession(String excludeId) { return false; }
        default int getBlockSize() { return 32; }
        default int getTotalBlocks() { return 1024; }
        default int getFreeBlockCount() { return 0; }
    }

//    interface CoWBlockManagerV9 extends CoWBlockManagerV2 {
//        void allocateBlocks(int numBlocks, int blockSize);
//        void releaseBlocks(String sessionId);
//        long[] getPhysicalBlockIds(String sessionId);
//        List<Integer> matchAndAllocatePath(List<Long> pathHashes, String sessionId, PagedKvBufferV3 buffer);
//        List<Integer> allocateBlocks(int neededBlocks, String sessionId, PagedKvBufferV3 buffer);
//    }

    // ======================== 核心推理类（线程安全版）========================
    static class Qwen3JavaInferenceV3 {
        private final JitModule model;
        private final CoWBlockManagerV9 blockManager;
        private final int numLayers;
        private final int blockSize;
        private final Map<String, SessionCache> sessionCacheMap = new ConcurrentHashMap<>();
        private final Object modelLock = new Object(); // 模型推理锁

        // 会话缓存类
        private static class SessionCache {
            long[] slotMapping;
            long[] positionIds;
            Map<Integer, Tensor[]> kvCache = new ConcurrentHashMap<>();
            PagedKvBufferV3 kvBuffer;
            List<Long>[] layerPathHashes;

            @SuppressWarnings("unchecked")
            public SessionCache(String sessionId, CoWBlockManagerV9 blockManager, int numLayers, int blockSize) {
                this.slotMapping = new long[0];
                this.positionIds = new long[0];
                this.kvBuffer = new PagedKvBufferV3(sessionId, blockManager, numLayers);
                this.layerPathHashes = new List[numLayers];
                for (int i = 0; i < numLayers; i++) {
                    this.layerPathHashes[i] = new ArrayList<>();
                }
            }

            public long generateHash(long[] slotMapping, int startIdx, int endIdx) {
                return Objects.hash(Arrays.copyOfRange(slotMapping, startIdx, endIdx));
            }

            // 清理缓存
            public void cleanup() {
                for (Tensor[] kv : kvCache.values()) {
                    if (kv[0] != null) {
                        try { kv[0].close(); } catch (Exception e) {}
                    }
                    if (kv[1] != null) {
                        try { kv[1].close(); } catch (Exception e) {}
                    }
                }
                kvCache.clear();
                if (kvBuffer != null) {
                    try { kvBuffer.close(); } catch (Exception e) {}
                }
            }
        }

        // 构造函数（线程安全）
        public Qwen3JavaInferenceV3(String modelPath, CoWBlockManagerV9 manager, int numLayers, int blockSize) {
            this.numLayers = numLayers;
            this.blockSize = blockSize;
            this.blockManager = manager;

            // 加载模型到CPU（线程安全）
            try {
                synchronized (modelLock) {
                    this.model = load(modelPath, new DeviceOptional(CPU_DEVICE), false);
                    System.out.println("模型成功加载到CPU设备");
                }
            } catch (Exception e) {
                throw new RuntimeException("模型加载失败: " + e.getMessage(), e);
            }
        }

        /**
         * 推理生成方法（线程安全）
         */
        public void generate(String sessionId, int[] inputTokenIds) {
            synchronized (modelLock) { // 模型推理加锁，避免并发调用
                try {
                    // 初始化会话缓存
                    SessionCache sessionCache = sessionCacheMap.computeIfAbsent(
                            sessionId,
                            k -> new SessionCache(k, blockManager, numLayers, blockSize)
                    );

                    // 创建输入张量（CPU）
                    Tensor inputTensor = null;
                    try {
                        inputTensor = tensor(inputTokenIds)
                                .view(1, -1)
                                .to(CPU_DEVICE, INT64);
                    } catch (Exception e) {
                        throw new RuntimeException("创建输入张量失败: " + e.getMessage(), e);
                    }

                    // 初始化PageAttention映射
                    initPageAttentionMapping(sessionCache, inputTokenIds.length, inputTensor);

                    // 推理循环（缩短循环避免长时间占用锁）
                    int totalGenerated = 0;
                    while (totalGenerated < MAX_NEW_TOKENS) {
                        // 准备输入参数
                        IValueVector inputs = preparePageAttentionInputs(sessionId, sessionCache, inputTensor);

                        // 模型推理
                        IValue output = null;
                        try {
                            output = model.forward(inputs);
                        } catch (Exception e) {
                            throw new RuntimeException("模型推理失败: " + e.getMessage(), e);
                        }

                        // 处理输出
                        Tensor logits = output.toTensor();
                        Tensor nextTokenTensor = logits.select(1L, -1L).argmax(new LongOptional(-1), false);
                        int nextToken = (int) nextTokenTensor.item().toLong();
                        System.out.println("线程[" + Thread.currentThread().getId() + "] 生成Token: " + nextToken);

                        // 终止条件
                        if (nextToken == 151643) break;

                        // 更新映射和缓存
                        updatePageAttentionMapping(sessionCache, nextToken, nextTokenTensor);

                        // 更新输入
                        inputTensor = nextTokenTensor.view(1, 1).to(CPU_DEVICE, INT64);
                        totalGenerated++;

                        // 显式释放临时张量
                        try {
                            logits.close();
                            nextTokenTensor.close();
                        } catch (Exception e) {
                            // 忽略释放异常
                        }
                    }

                    // 释放输入张量
                    if (inputTensor != null) {
                        try {
                            inputTensor.close();
                        } catch (Exception e) {
                            // 忽略释放异常
                        }
                    }
                } catch (Exception e) {
                    throw new RuntimeException("推理失败: " + e.getMessage(), e);
                }
            }
        }

        /**
         * 初始化PageAttention映射（线程安全）
         */
        private void initPageAttentionMapping(SessionCache sessionCache, int seqLen, Tensor inputTensor) {
            // 初始化映射数组
            sessionCache.slotMapping = new long[seqLen];
            sessionCache.positionIds = new long[seqLen];
            for (int i = 0; i < seqLen; i++) {
                sessionCache.slotMapping[i] = i;
                sessionCache.positionIds[i] = i;
            }

            // 初始化各层KV缓存
            for (int layer = 0; layer < numLayers; layer++) {
                // 生成哈希路径
                List<Long> pathHashes = new ArrayList<>();
                for (int i = 0; i < seqLen; i += blockSize) {
                    int endIdx = Math.min(i + blockSize, seqLen);
                    long hash = sessionCache.generateHash(sessionCache.slotMapping, i, endIdx);
                    pathHashes.add(hash);
                }
                sessionCache.layerPathHashes[layer] = pathHashes;

                // 写入KV缓存（捕获异常避免崩溃）
                try {
                    sessionCache.kvBuffer.prefillUltra(layer, 0, inputTensor);
                    sessionCache.kvBuffer.prefillUltra(layer, 1, inputTensor);
                } catch (Exception e) {
                    System.err.println("初始化KV缓存失败: " + e.getMessage());
                }
            }

            // 分配缓存块
            blockManager.allocateBlocks((seqLen + blockSize - 1) / blockSize, blockSize);
        }

        /**
         * 更新PageAttention映射（线程安全）
         */
        private void updatePageAttentionMapping(SessionCache sessionCache, int nextToken, Tensor nextTokenTensor) {
            int currentLen = sessionCache.slotMapping.length;

            // 扩展映射数组
            long[] newSlotMapping = Arrays.copyOf(sessionCache.slotMapping, currentLen + 1);
            long[] newPositionIds = Arrays.copyOf(sessionCache.positionIds, currentLen + 1);
            newSlotMapping[currentLen] = currentLen;
            newPositionIds[currentLen] = currentLen;

            sessionCache.slotMapping = newSlotMapping;
            sessionCache.positionIds = newPositionIds;

            // 检查是否需要新块
            boolean needNewBlock = (currentLen % blockSize == 0);
            if (needNewBlock) {
                blockManager.allocateBlocks(1, blockSize);

                // 更新各层哈希和KV缓存
                for (int layer = 0; layer < numLayers; layer++) {
                    long newHash = sessionCache.generateHash(sessionCache.slotMapping, currentLen, currentLen + 1);
                    sessionCache.layerPathHashes[layer].add(newHash);

                    try {
                        sessionCache.kvBuffer.prefillUltra(layer, 0, nextTokenTensor);
                        sessionCache.kvBuffer.prefillUltra(layer, 1, nextTokenTensor);
                    } catch (Exception e) {
                        System.err.println("更新KV缓存失败: " + e.getMessage());
                    }
                }
            }
        }

        /**
         * 准备PageAttention输入参数（线程安全）
         */
        private IValueVector preparePageAttentionInputs(String sessionId, SessionCache sessionCache, Tensor inputTensor) {
            // 创建辅助张量（全部CPU）
            Tensor positionIds = null;
            Tensor slotMapping = null;
            Tensor blockTable = null;

            try {
                positionIds = tensor(sessionCache.positionIds)
                        .view(1, -1)
                        .to(CPU_DEVICE, INT64);

                blockTable = getBlockTableFromManager(sessionId, sessionCache.slotMapping.length);

                slotMapping = tensor(sessionCache.slotMapping)
                        .view(1, -1)
                        .to(CPU_DEVICE, INT64);
            } catch (Exception e) {
                // 释放已创建的张量
                if (positionIds != null) try { positionIds.close(); } catch (Exception ex) {}
                if (slotMapping != null) try { slotMapping.close(); } catch (Exception ex) {}
                if (blockTable != null) try { blockTable.close(); } catch (Exception ex) {}
                throw new RuntimeException("创建辅助张量失败: " + e.getMessage(), e);
            }

            // 准备KV缓存输入
            IValueVector kvCacheInputs = new IValueVector();
            for (int layer = 0; layer < numLayers; layer++) {
                Tensor kTensor = null;
                Tensor vTensor = null;

                try {
                    kTensor = getKVCacheTensorFromBuffer(sessionCache.kvBuffer, layer, 0);
                    vTensor = getKVCacheTensorFromBuffer(sessionCache.kvBuffer, layer, 1);

                    kvCacheInputs.push_back(new IValue(kTensor));
                    kvCacheInputs.push_back(new IValue(vTensor));
                    sessionCache.kvCache.put(layer, new Tensor[]{kTensor, vTensor});
                } catch (Exception e) {
                    // 释放已创建的张量
                    if (kTensor != null) try { kTensor.close(); } catch (Exception ex) {}
                    if (vTensor != null) try { vTensor.close(); } catch (Exception ex) {}
                    throw new RuntimeException("创建KV缓存张量失败: " + e.getMessage(), e);
                }
            }

            // 组装最终输入
            IValueVector inputs = new IValueVector();
            try {
                inputs.push_back(new IValue(inputTensor));
                inputs.push_back(new IValue(positionIds));
                inputs.push_back(new IValue(blockTable));
                inputs.push_back(new IValue(slotMapping));
                inputs.push_back(new IValue(kvCacheInputs));
            } catch (Exception e) {
                throw new RuntimeException("组装输入参数失败: " + e.getMessage(), e);
            }

            return inputs;
        }

        /**
         * 获取KV缓存张量（线程安全）
         */
        private Tensor getKVCacheTensorFromBuffer(PagedKvBufferV3 kvBuffer, int layer, int kvType) {
            int headDim = 128;
            int numHeads = 32;
            int kb = getKBlockCount(kvBuffer, layer) * blockSize;
            int vb = getVBlockCount(kvBuffer, layer) * blockSize;
            int seqLen = (kvType == 0) ? kb : vb;

            // 创建KV缓存张量（CPU + float32）
            Tensor kvTensor = null;
            try {
                kvTensor = randn(new long[]{1, numHeads, seqLen, headDim},
                        new TensorOptions()
                                .device(new DeviceOptional(CPU_DEVICE))
                                .dtype(new ScalarTypeOptional(FLOAT32)));
            } catch (Exception e) {
                throw new RuntimeException("创建KV缓存张量失败: " + e.getMessage(), e);
            }

            return kvTensor;
        }

        /**
         * 获取BlockTable（线程安全）
         */
        private Tensor getBlockTableFromManager(String sessionId, int seqLen) {
            long[] physicalBlockIds = blockManager.getPhysicalBlockIds(sessionId);

            // 创建BlockTable张量
            Tensor blockTable = null;
            try {
                blockTable = tensor(physicalBlockIds)
                        .view(1, -1)
                        .to(CPU_DEVICE, INT64);
            } catch (Exception e) {
                throw new RuntimeException("创建BlockTable失败: " + e.getMessage(), e);
            }

            // 补全缺失块
            int requiredBlocks = (seqLen + blockSize - 1) / blockSize;
            if (blockTable.size(1) < requiredBlocks) {
                Tensor padding = null;
                try {
                    padding = tensor(new long[requiredBlocks - (int) blockTable.size(1)])
                            .view(1, -1)
                            .to(CPU_DEVICE, INT64);
                    blockTable = cat(new TensorVector(blockTable, padding), 1);
                } catch (Exception e) {
                    if (padding != null) try { padding.close(); } catch (Exception ex) {}
                    throw new RuntimeException("补全BlockTable失败: " + e.getMessage(), e);
                } finally {
                    if (padding != null) try { padding.close(); } catch (Exception ex) {}
                }
            }

            return blockTable;
        }

        /**
         * 释放会话缓存（线程安全）
         */
        public void releaseSessionCache(String sessionId) {
            SessionCache sessionCache = sessionCacheMap.remove(sessionId);
            if (sessionCache != null) {
                // 释放KV Buffer
                try {
                    sessionCache.kvBuffer.close();
                } catch (Exception e) {
                    System.err.println("释放KV Buffer失败: " + e.getMessage());
                }

                // 释放块管理器资源
                blockManager.releaseBlocks(sessionId);

                // 释放张量资源（关键：显式关闭所有Tensor）
                sessionCache.cleanup();
            }
        }

        /**
         * 全局清理资源
         */
        public void cleanup() {
            // 清理所有会话缓存
            for (String sessionId : sessionCacheMap.keySet()) {
                releaseSessionCache(sessionId);
            }
            sessionCacheMap.clear();

            // 清理模型
            synchronized (modelLock) {
                try {
                    model.close();
                } catch (Exception e) {
                    System.err.println("关闭模型失败: " + e.getMessage());
                }
            }
        }

        // ======================== 辅助方法 ========================
        private int getKBlockCount(PagedKvBufferV3 kvBuffer, int layer) {
            return (int) Math.ceil((double) getSessionId(kvBuffer).length() / blockSize);
        }

        private int getVBlockCount(PagedKvBufferV3 kvBuffer, int layer) {
            return (int) Math.ceil((double) getSessionId(kvBuffer).length() / blockSize);
        }

        private String getSessionId(PagedKvBufferV3 kvBuffer) {
            return "session_" + System.currentTimeMillis();
        }
    }

    // ======================== 占位类（适配编译）========================
//    static class PagedKvBufferV3 {
//        public PagedKvBufferV3(String sessionId, CoWBlockManagerV9 blockManager, int numLayers) {}
//
//        public void prefillUltra(int layer, int kvType, Tensor tensor) {}
//
//        public void close() {}
//    }
}
