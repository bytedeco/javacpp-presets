package org.bytedeco.pytorch.geometric.demo.kvcache;
import org.bytedeco.pytorch.jit.*;
import org.bytedeco.pytorch.c10.*;
import org.bytedeco.pytorch.jit.JitModule;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.global.torch;

import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.Future;
import java.util.concurrent.atomic.*;
import java.util.concurrent.locks.*;

/**
 * V11修复版：虚拟线程高并发压测（PagedKvBufferV3+CoWBlockManagerV9）
 * 修复：块资源耗尽/驱逐失效/等待策略/资源清理/任务限流/统计缺失等核心问题
 */
public class Qwen3InferenceBenchmarkV12 {
    // V11推理器：兼容虚拟线程的高并发版本
    public static class Qwen3JavaInferenceV11 {
        private final JitModule model;
        private final Device device;
        private final CoWBlockManagerV9 blockManager;
        private final ConcurrentHashMap<String, PagedKvBufferV3> kvBufferMap;
        private final ConcurrentHashMap<String, CacheEntry> cacheMetaMap;

        // PageAttention V3 配置
        public static final int PAGE_SIZE = 1024;
        public static final int BLOCK_SIZE = 256;
        public static final int MAX_SEQ_LEN = 8192;
        public static final int NUM_LAYERS = 32;
        public static final int TOTAL_BLOCKS = 16384; // 大幅增加总块数，适配高并发
        public static final int DTYPE = 0;

        // 高并发统计（LongAdder+原子计数器）
        private final LongAdder totalInferenceTime = new LongAdder();
        private final LongAdder totalTokensGenerated = new LongAdder();
        private final AtomicInteger completedTasks = new AtomicInteger(0);
        private final AtomicInteger failedTasks = new AtomicInteger(0);
        // 任务限流计数器
        private final Semaphore taskSemaphore;

        // 构造函数 - 初始化模型+块管理器+任务限流
        public Qwen3JavaInferenceV11(String modelPath, int maxConcurrentTasks) {
            this.device = new Device(torch.kCPU());
            // 加载模型（线程安全）
            this.model = torch.load(modelPath, new DeviceOptional(this.device), false);
            this.model.eval();
            // 初始化修复版V9块管理器（大幅增加总块数）
            this.blockManager = new CoWBlockManagerV9(TOTAL_BLOCKS, NUM_LAYERS, BLOCK_SIZE, HEAD_DIM, DTYPE);
            this.kvBufferMap = new ConcurrentHashMap<>();
            this.cacheMetaMap = new ConcurrentHashMap<>();
            // 任务限流信号量：控制最大并发任务数
            this.taskSemaphore = new Semaphore(maxConcurrentTasks);
            // 禁用梯度计算
            torch.requires_grad(false);
//            torch.no_grad();
        }

        // 注意力头维度
        private static final int HEAD_DIM = 128;

        /**
         * V11核心并发生成方法：带限流+异常强制清理
         */
        public InferenceResult generateConcurrent(Tensor inputIds, int generateLen, String sessionId) {
            Tensor currentInput = null;
            try {
                // 任务限流：获取许可，无许可则阻塞
                taskSemaphore.acquire();
                // 复制输入张量，避免线程间共享
                currentInput = inputIds.to(device, torch.ScalarType.Long).clone();
                long inputSeqLen = currentInput.size(1);
                long startTime = System.currentTimeMillis();

                // 线程安全的缓存初始化
                PagedKvBufferV3 kvBuffer = kvBufferMap.computeIfAbsent(sessionId,
                        k -> new PagedKvBufferV3(sessionId, blockManager, NUM_LAYERS));
                CacheEntry cacheMeta = cacheMetaMap.computeIfAbsent(sessionId, k -> new CacheEntry());

                // 边界检查
                if (currentInput.dim() != 2) {
                    throw new IllegalArgumentException("输入必须是2维张量，当前维度：" + currentInput.dim());
                }
                if (inputSeqLen + generateLen > MAX_SEQ_LEN) {
                    throw new RuntimeException("序列长度超过最大值：" + MAX_SEQ_LEN);
                }

                // Prefill阶段：初始化KV缓存
                if (cacheMeta.seqLen == 0) {
                    prefillKVCache(currentInput, kvBuffer, cacheMeta);
                    cacheMeta.seqLen = (int) inputSeqLen;
                }

                // 逐Token自回归生成
                for (int step = 0; step < generateLen; step++) {
                    // 外部KV Cache：仅传入最后一个Token
                    Tensor newTokenInput = currentInput.narrow(1, currentInput.size(1) - 1, 1);
                    Tensor logitsTensor = forwardWithKVCache(newTokenInput);
                    Tensor lastStepLogits = getLastTokenLogits(logitsTensor, 1);
                    Tensor nextToken = safeGreedySample(lastStepLogits);
                    nextToken = ensure2DToken(nextToken);

                    // 更新KV缓存（基于修复版V9块管理器）
                    updateKVCacheWithV9Manager(nextToken, kvBuffer, cacheMeta, sessionId);

                    // 拼接新Token
                    TensorVector catTensors = new TensorVector();
                    catTensors.push_back(currentInput);
                    catTensors.push_back(nextToken);
                    currentInput = torch.cat(catTensors, 1);
                    cacheMeta.seqLen++;

                    // PageAttention V3分页优化：清理过期块
                    if (cacheMeta.seqLen % PAGE_SIZE == 0 && cacheMeta.seqLen > 0) {
                        optimizeKVCacheWithPageAttentionV3(kvBuffer, cacheMeta, sessionId);
                    }

                    // 及时释放临时张量，避免内存泄漏
                    logitsTensor.close();
                    lastStepLogits.close();
                    nextToken.close();
                    catTensors.close();
                    newTokenInput.close();
                }

                // 统计成功结果
                long inferenceTime = System.currentTimeMillis() - startTime;
                totalInferenceTime.add(inferenceTime);
                totalTokensGenerated.add(generateLen);
                completedTasks.incrementAndGet();

                return new InferenceResult(
                        true, sessionId, generateLen, inputSeqLen, currentInput.size(1),
                        inferenceTime, null, cacheMeta.physicalBlockIds.length
                );

            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                failedTasks.incrementAndGet();
                return new InferenceResult(false, sessionId, generateLen, 0, 0,
                        System.currentTimeMillis(), "任务被中断", 0);
            } catch (Exception e) {
                failedTasks.incrementAndGet();
                return new InferenceResult(false, sessionId, generateLen, 0, 0,
                        System.currentTimeMillis(), e.getMessage(), 0);
            } finally {
                // 强制清理资源：无论成功/失败，都释放张量+会话资源+限流许可
                if (currentInput != null) currentInput.close();
                cleanSessionResources(sessionId);
                taskSemaphore.release();
            }
        }

        // Prefill阶段：初始化KV缓存（线程安全）
        private void prefillKVCache(Tensor input, PagedKvBufferV3 kvBuffer, CacheEntry cacheMeta) {
            int numTokens = (int) input.size(1);
            for (int layer = 0; layer < NUM_LAYERS; layer++) {
                kvBuffer.prefillUltra(layer, 0, input); // K块
                kvBuffer.prefillUltra(layer, 1, input); // V块
                cacheMeta.kBlockCount[layer] = kvBuffer.getKBlockCount(layer);
                cacheMeta.vBlockCount[layer] = kvBuffer.getVBlockCount(layer);
            }
            // 记录物理块ID
            cacheMeta.physicalBlockIds = blockManager.getPhysicalBlockIds(kvBuffer.getSessionId().toString());
        }

        // 更新KV缓存：基于修复版V9块管理器
        private void updateKVCacheWithV9Manager(Tensor token, PagedKvBufferV3 kvBuffer, CacheEntry cacheMeta, String sessionId) {
            int neededBlocks = (1 + BLOCK_SIZE - 1) / BLOCK_SIZE;
            // 块分配统计自增
            blockManager.totalRequests.add(neededBlocks);
            blockManager.allocateBlocks(neededBlocks, BLOCK_SIZE);

            for (int layer = 0; layer < NUM_LAYERS; layer++) {
                kvBuffer.prefillUltra(layer, 0, token);
                kvBuffer.prefillUltra(layer, 1, token);
                cacheMeta.kBlockCount[layer] = kvBuffer.getKBlockCount(layer);
                cacheMeta.vBlockCount[layer] = kvBuffer.getVBlockCount(layer);
            }
            cacheMeta.physicalBlockIds = blockManager.getPhysicalBlockIds(sessionId);
        }

        // PageAttention V3：分页优化+过期块清理
        private void optimizeKVCacheWithPageAttentionV3(PagedKvBufferV3 kvBuffer, CacheEntry cacheMeta, String sessionId) {
            long[] blockIds = blockManager.getPhysicalBlockIds(sessionId);
            int pageNum = (int) (cacheMeta.seqLen / PAGE_SIZE);
            cacheMeta.lastPageIdx = pageNum;

            // 保留最近4页，清理更早的过期块
            if (pageNum > 4) {
                List<Integer> invalidatedBlocks = kvBuffer.getAndInvalidateBlocks();
                blockManager.releaseBlocks(sessionId);
                cacheMeta.invalidatedBlockCount += invalidatedBlocks.size();
                cacheMeta.physicalBlockIds = blockManager.getPhysicalBlockIds(sessionId);
                // 清理后统计自增
                blockManager.cacheHitBlocks.add(invalidatedBlocks.size());
            }
        }

        // 模型前向推理（适配KV Cache）
        private Tensor forwardWithKVCache(Tensor inputIds) {
            IValue output;
            try (IValueVector inputs = new IValueVector()) {
                inputs.push_back(new IValue(inputIds));
                output = model.forward(inputs);
            }
            return output.toTensor();
        }

        // 获取最后一个Token的Logits
        private Tensor getLastTokenLogits(Tensor logits, long currentSeqLen) {
            Tensor lastStepLogits;
            if (logits.dim() == 3) {
                lastStepLogits = logits.narrow(1, currentSeqLen - 1, 1);
            } else if (logits.dim() == 2) {
                lastStepLogits = logits.narrow(0, currentSeqLen - 1, 1);
            } else {
                lastStepLogits = logits.slice(0, new LongOptional(logits.size(0) - 1L), new LongOptional(logits.size(0)), 1);
            }
            return lastStepLogits;
        }

        // 安全的贪心采样
        private Tensor safeGreedySample(Tensor logits) {
            int vocabDim = (int) logits.dim() - 1;
            Tensor argmaxTensor = torch.argmax(logits, new LongOptional(vocabDim), false);
            Tensor nextToken = argmaxTensor.to(device, torch.ScalarType.Long);
            argmaxTensor.close();
            return nextToken;
        }

        // 确保Token张量为2维 [batch_size, 1]
        private Tensor ensure2DToken(Tensor token) {
            Tensor result = token;
            if (token.dim() == 1) {
                result = token.unsqueeze(1);
            } else if (token.dim() >= 3) {
                result = token.squeeze(token.dim() - 1);
            }
            if (result.dim() != 2) {
                result = result.reshape(new long[]{1, 1});
            }
            return result;
        }

        // 强制清理会话资源：原子操作+块释放
        public void cleanSessionResources(String sessionId) {
            PagedKvBufferV3 kvBuffer = kvBufferMap.remove(sessionId);
            if (kvBuffer != null) {
                try {
                    kvBuffer.close();
                } catch (Exception e) {
                    System.err.println("关闭KVBuffer失败：" + e.getMessage());
                }
            }
            blockManager.releaseBlocks(sessionId);
            cacheMetaMap.remove(sessionId);
        }

        // 高并发统计：平均Token生成速度
        public double getTokenGenerationSpeed() {
            long totalTime = totalInferenceTime.sum();
            long totalTokens = totalTokensGenerated.sum();
            if (totalTime == 0 || totalTokens == 0) return 0.0;
            return totalTokens / (totalTime / 1000.0);
        }

        // 并发任务统计
        public String getConcurrentStats() {
            return String.format(
                    "完成任务数: %d, 失败任务数: %d, 总Token数: %d, 总耗时: %dms, 平均Token速度: %.2f tokens/s",
                    completedTasks.get(), failedTasks.get(),
                    totalTokensGenerated.sum(), totalInferenceTime.sum(),
                    getTokenGenerationSpeed()
            );
        }

        // 块管理器统计
        public String getBlockManagerStats() {
            return String.format(
                    "块管理器统计 | 总请求块数: %d, 缓存命中块数: %d, 驱逐次数: %d, 等待次数: %d, 剩余块数: %d",
                    blockManager.totalRequests.sum(), blockManager.cacheHitBlocks.sum(),
                    CoWBlockManagerV9.EVICT_COUNT.sum(), CoWBlockManagerV9.WAIT_COUNT.sum(),
                    blockManager.getFreeBlockCount()
            );
        }

        // 释放所有资源
        public void close() {
            kvBufferMap.keySet().forEach(this::cleanSessionResources);
            kvBufferMap.clear();
            cacheMetaMap.clear();
            // 释放模型和设备
            if (model != null) model.close();
            if (device != null) device.close();
            // 打印最终统计
            System.out.println("\n=== 虚拟线程压测最终统计 ===");
            System.out.println(getConcurrentStats());
            System.out.println(getBlockManagerStats());
        }

        // 缓存元数据（按层管理块）
        private static class CacheEntry {
            int seqLen = 0;
            long lastPageIdx = 0;
            int[] kBlockCount = new int[NUM_LAYERS];
            int[] vBlockCount = new int[NUM_LAYERS];
            long[] physicalBlockIds = new long[0];
            int invalidatedBlockCount = 0;
        }

        // 推理结果封装（线程安全）
        public static class InferenceResult {
            public final boolean success;
            public final String sessionId;
            public final int generateLen;
            public final long inputSeqLen;
            public final long outputSeqLen;
            public final long inferenceTime;
            public final String errorMsg;
            public final int usedBlockCount;

            public InferenceResult(boolean success, String sessionId, int generateLen,
                                   long inputSeqLen, long outputSeqLen, long inferenceTime,
                                   String errorMsg, int usedBlockCount) {
                this.success = success;
                this.sessionId = sessionId;
                this.generateLen = generateLen;
                this.inputSeqLen = inputSeqLen;
                this.outputSeqLen = outputSeqLen;
                this.inferenceTime = inferenceTime;
                this.errorMsg = errorMsg;
                this.usedBlockCount = usedBlockCount;
            }
        }
    }

    // ======================== 主方法：虚拟线程高并发压测 ========================
    public static void main(String[] args) throws InterruptedException {
        // 模型路径（替换为你的实际路径）
        String modelPath = "/Users/mullerzhang/Documents/code/langchain/qwen3_4b_fp16_mps.pt";
        // 压测核心配置（根据硬件调整）
        int maxConcurrentTasks = 1;    // 最大并发任务数（限流，关键！）
        int virtualThreadNum = 10;      // 虚拟线程数
        int testRoundPerThread = 5;     // 每个线程压测轮数
        int generateLenPerRound = 100;  // 每轮生成Token数
        String baseSessionId = "vt_session_v11_fixed";

        // 1. 初始化修复版V11推理器（带任务限流）
        Qwen3JavaInferenceV11 inference = new Qwen3JavaInferenceV11(modelPath, maxConcurrentTasks);

        // 2. 测试输入：16个Token（与原压测一致）
        long[] inputTokens = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
        Tensor inputIds = torch.tensor(inputTokens,
                        new TensorOptions().dtype(new ScalarTypeOptional(torch.ScalarType.Long)))
                .reshape(1, 16);

        // 3. 打印压测配置
        System.out.println("===== 开始V11修复版虚拟线程压测（PagedKvBufferV3+CoWBlockManagerV9） =====");
        System.out.printf("压测配置：虚拟线程数=%d, 每线程轮数=%d, 每轮Token数=%d, 最大并发任务数=%d\n",
                virtualThreadNum, testRoundPerThread, generateLenPerRound, maxConcurrentTasks);
        System.out.printf("核心配置：PageSize=%d, BlockSize=%d, 总块数=%d, 模型层数=%d\n",
                Qwen3JavaInferenceV11.PAGE_SIZE, Qwen3JavaInferenceV11.BLOCK_SIZE,
                Qwen3JavaInferenceV11.TOTAL_BLOCKS, Qwen3JavaInferenceV11.NUM_LAYERS);

        // 4. 预热阶段（单任务，预热后强制清理资源）
        System.out.println("\n=== 预热阶段 ===");
        try {
            Qwen3JavaInferenceV11.InferenceResult warmupResult =
                    inference.generateConcurrent(inputIds, 10, baseSessionId + "_warmup");
            System.out.printf("预热%s | 耗时: %dms | 错误: %s\n",
                    warmupResult.success ? "成功" : "失败",
                    warmupResult.inferenceTime, warmupResult.errorMsg);
            // 强制清理预热资源，确保块回池
            inference.cleanSessionResources(baseSessionId + "_warmup");
            System.out.printf("预热资源清理完成，当前剩余块数: %d\n", inference.blockManager.getFreeBlockCount());
        } catch (Exception e) {
            System.err.println("预热失败：" + e.getMessage());
            e.printStackTrace();
            inputIds.close();
            inference.close();
            return;
        }

        // 5. 创建Java 21虚拟线程池（核心）
        ExecutorService virtualThreadExecutor = Executors.newVirtualThreadPerTaskExecutor();
        CompletionService<Qwen3JavaInferenceV11.InferenceResult> completionService =
                new ExecutorCompletionService<>(virtualThreadExecutor);

        // 6. 提交虚拟线程任务（10线程×5轮=50任务）
        System.out.println("\n=== 虚拟线程高并发压测开始 ===");
        long totalWallTimeStart = System.currentTimeMillis();
        int totalTaskCount = 0;
        for (int threadIdx = 0; threadIdx < virtualThreadNum; threadIdx++) {
            for (int round = 0; round < testRoundPerThread; round++) {
                String sessionId = baseSessionId + "_thread" + threadIdx + "_round" + round;
                completionService.submit(() ->
                        inference.generateConcurrent(inputIds, generateLenPerRound, sessionId)
                );
                totalTaskCount++;
            }
        }

        // 7. 收集任务结果（按完成顺序）
        List<Qwen3JavaInferenceV11.InferenceResult> allResults = new ArrayList<>();
        for (int i = 0; i < totalTaskCount; i++) {
            try {
                Future<Qwen3JavaInferenceV11.InferenceResult> future = completionService.take();
                Qwen3JavaInferenceV11.InferenceResult result = future.get();
                allResults.add(result);
                // 打印实时结果
                if (result.success) {
                    System.out.printf("成功 | 会话%s | 耗时: %dms | 生成Token: %d | 使用块数: %d\n",
                            result.sessionId, result.inferenceTime, result.generateLen, result.usedBlockCount);
                } else {
                    System.err.printf("失败 | 会话%s | 耗时: %dms | 错误: %s\n",
                            result.sessionId, result.inferenceTime, result.errorMsg);
                }
            } catch (ExecutionException e) {
                System.err.println("任务执行异常：" + e.getCause().getMessage());
                inference.failedTasks.incrementAndGet();
            }
        }

        // 8. 关闭线程池+统计墙钟时间
        virtualThreadExecutor.shutdown();
        if (!virtualThreadExecutor.awaitTermination(5, TimeUnit.MINUTES)) {
            virtualThreadExecutor.shutdownNow();
        }
        long totalWallTimeEnd = System.currentTimeMillis();

        // 9. 打印总结果
        System.out.println("\n=== 单任务结果汇总 ===");
        long successTotalTime = 0;
        int successTotalToken = 0;
        for (Qwen3JavaInferenceV11.InferenceResult res : allResults) {
            if (res.success) {
                successTotalTime += res.inferenceTime;
                successTotalToken += res.generateLen;
            }
        }
        System.out.printf("总任务数: %d, 成功数: %d, 失败数: %d, 墙钟总耗时: %dms\n",
                totalTaskCount,
                (int) allResults.stream().filter(r -> r.success).count(),
                (int) allResults.stream().filter(r -> !r.success).count(),
                totalWallTimeEnd - totalWallTimeStart);
        System.out.printf("成功任务平均耗时: %dms, 成功任务总Token: %d\n",
                successTotalTime / Math.max(1, allResults.stream().filter(r -> r.success).count()),
                successTotalToken);

        // 10. 释放所有资源
        inputIds.close();
        inference.close();
        System.out.println("\nV11修复版虚拟线程压测完成，所有资源已释放！");
    }

    // ======================== 依赖类：PagedKvBufferV3（原实现不变） ========================
    public static class PagedKvBufferV3 implements AutoCloseable {
        private final String sessionId;
        private final CoWBlockManagerV2 manager;
        private final int numLayers;
        private final List<Integer>[] kBlockMaps;
        private final List<Integer>[] vBlockMaps;

        private final ReentrantReadWriteLock stateLock = new ReentrantReadWriteLock();
        private final AtomicBoolean isInvalidated = new AtomicBoolean(false);

        @SuppressWarnings("unchecked")
        public PagedKvBufferV3(String sessionId, CoWBlockManagerV2 manager, int numLayers) {
            this.sessionId = sessionId;
            this.manager = manager;
            this.numLayers = numLayers;
            this.kBlockMaps = new ArrayList[numLayers];
            this.vBlockMaps = new ArrayList[numLayers];
            for (int i = 0; i < numLayers; i++) {
                kBlockMaps[i] = new ArrayList<>();
                vBlockMaps[i] = new ArrayList<>();
            }
        }

        public void prefillUltra(int layer, int kvType, Tensor input) {
            stateLock.readLock().lock();
            try {
                if (isInvalidated.get()) return;
                int numTokens = (int) input.size(0);
                int blockSize = manager.getBlockSize();
                int neededBlocks = (numTokens + blockSize - 1) / blockSize;
                // 分配块并统计
                List<Integer> newBlocks = manager.allocateBlocks(neededBlocks, sessionId, this);
                if (kvType == 0) kBlockMaps[layer].addAll(newBlocks);
                else vBlockMaps[layer].addAll(newBlocks);
            } finally {
                stateLock.readLock().unlock();
            }
        }

        public List<Integer> getAndInvalidateBlocks() {
            stateLock.writeLock().lock();
            try {
                isInvalidated.set(true);
                List<Integer> allBlocks = new ArrayList<>();
                for (int i = 0; i < numLayers; i++) {
                    allBlocks.addAll(kBlockMaps[i]);
                    allBlocks.addAll(vBlockMaps[i]);
                    kBlockMaps[i].clear();
                    vBlockMaps[i].clear();
                }
                return allBlocks;
            } finally {
                stateLock.writeLock().unlock();
            }
        }

        @Override
        public void close() { manager.releaseSession(sessionId); }

        public CharSequence getSessionId() { return sessionId; }
        public int getKBlockCount(int layer) { return kBlockMaps[layer].size(); }
        public int getVBlockCount(int layer) { return vBlockMaps[layer].size(); }
    }

    // ======================== 基础父类：CoWBlockManagerV2 ========================
    abstract static class CoWBlockManagerV2 {
        protected int totalBlocks;
        protected final Queue<Integer> freePool = new ConcurrentLinkedQueue<>();
        private final ReentrantLock globalLock = new ReentrantLock();
        // 跟踪所有会话的最后活动时间，用于LRU驱逐
        protected final ConcurrentHashMap<String, Long> sessionLastActive = new ConcurrentHashMap<>();

        public CoWBlockManagerV2(int totalBlocks, int layers, int blockSize, int headDim, int dtype) {
            this.totalBlocks = totalBlocks;
            // 初始化自由块池
            for (int i = 0; i < totalBlocks; i++) {
                freePool.add(i);
            }
        }

        // 获取空闲块数
        public int getFreeBlockCount() { return freePool.size(); }
        // 获取块大小
        public int getBlockSize() { return 256; }
        // 获取全局锁
        public ReentrantLock getGlobalLock() { return globalLock; }
        // 释放会话（子类重写）
        public void releaseSession(String sessionId) {}
        // LRU驱逐（子类重写核心逻辑）
        boolean evictOldestSession(String excludeId) { return false; }
        // 分配块（子类实现）
        public abstract List<Integer> allocateBlocks(int neededBlocks, String sessionId, PagedKvBufferV3 kvBuffer);
    }

    // ======================== 核心修复：CoWBlockManagerV9（全量优化） ========================
    public static class CoWBlockManagerV9 extends CoWBlockManagerV2 {
        private final int actualBlockSize;
        private final Condition diskFullCondition;
        private final ReentrantLock treeLock = new ReentrantLock();
        // 全局统计：驱逐/等待次数
        public static final LongAdder EVICT_COUNT = new LongAdder();
        public static final LongAdder WAIT_COUNT = new LongAdder();
        // 水位控制：调低触发阈值，提高清理阈值，减少驱逐频率
        private final double lowWatermark = 0.05;  // 5%空闲触发清理
        private final double highWatermark = 0.30; // 清理到30%空闲停止
        // 核心统计：总请求块数/缓存命中块数
        public final LongAdder totalRequests = new LongAdder();
        public final LongAdder cacheHitBlocks = new LongAdder();

        // 幽灵节点缓存：refCount=0但未物理释放的节点
        private final Deque<RadixNode> ghostCache = new ConcurrentLinkedDeque<>();
        // 会话绑定的Radix节点：用于释放/驱逐
        private final ConcurrentHashMap<String, List<RadixNode>> sessionNodes = new ConcurrentHashMap<>();
        // Radix树根节点：用于缓存块路径
        private final RadixNode root = new RadixNode(-1, -1);

        // Radix树节点：带引用计数
        static class RadixNode {
            final long hash;
            final int blockId;
            final ConcurrentHashMap<Long, RadixNode> children = new ConcurrentHashMap<>();
            final AtomicInteger refCount = new AtomicInteger(0);
            RadixNode(long hash, int blockId) { this.hash = hash; this.blockId = blockId; }
        }

        // 构造函数：初始化块管理器
        public CoWBlockManagerV9(int totalBlocks, int layers, int blockSize, int headDim, int dtype) {
            super(totalBlocks, layers, blockSize, headDim, dtype);
            this.actualBlockSize = blockSize;
            this.diskFullCondition = super.getGlobalLock().newCondition();
        }

        // 修复1：块分配（适配PageAttention，带统计+重试+驱逐）
        public void allocateBlocks(int numBlocks, int blockSize) {
            if (blockSize != this.actualBlockSize) {
                throw new IllegalArgumentException("块大小不匹配：配置=" + this.actualBlockSize + "，请求=" + blockSize);
            }
            if (numBlocks <= 0) return;

            super.getGlobalLock().lock();
            try {
                for (int i = 0; i < numBlocks; i++) {
                    // 带重试的块分配，核心方法
                    allocateWithRetry("temp_session", null);
                }
            } finally {
                super.getGlobalLock().unlock();
            }
            // 分配后检查水位，自动清理/驱逐
            checkAndPrune();
        }

        // 修复2：释放会话块（强制回池+清理幽灵节点）
        public void releaseBlocks(String sessionId) {
            if (sessionId == null || sessionId.isEmpty()) return;
            // 更新会话最后活动时间
            sessionLastActive.remove(sessionId);
            // 释放会话核心逻辑
            this.releaseSession(sessionId);

            // 强制清理该会话相关的幽灵节点，块回池
            super.getGlobalLock().lock();
            try {
                Iterator<RadixNode> iterator = ghostCache.iterator();
                while (iterator.hasNext()) {
                    RadixNode node = iterator.next();
                    if (node.refCount.get() == 0) {
                        freePool.add(node.blockId);
                        iterator.remove();
                        cacheHitBlocks.increment();
                    }
                }
                // 唤醒所有阻塞的分配线程
                diskFullCondition.signalAll();
            } finally {
                super.getGlobalLock().unlock();
            }
        }

        // 修复3：获取物理块ID（线程安全，加锁保护）
        public long[] getPhysicalBlockIds(String sessionId) {
            if (sessionId == null || !sessionNodes.containsKey(sessionId)) {
                return new long[0];
            }

            treeLock.lock();
            try {
                List<RadixNode> nodes = sessionNodes.get(sessionId);
                if (nodes == null || nodes.isEmpty()) return new long[0];
                long[] blockIds = new long[nodes.size()];
                for (int i = 0; i < nodes.size(); i++) {
                    blockIds[i] = nodes.get(i).blockId;
                }
                return blockIds;
            } finally {
                treeLock.unlock();
            }
        }

        // 修复4：Radix树块分配/命中（带统计）
        public int getOrAllocateBlock(long currentHash, String sid, PagedKvBufferV3 kv) {
            treeLock.lock();
            try {
                // 更新会话最后活动时间（用于LRU驱逐）
                sessionLastActive.put(sid, System.currentTimeMillis());
                RadixNode current = root;
                RadixNode next = current.children.get(currentHash);
                RadixNode targetNode;

                if (next != null) {
                    // 缓存命中，统计自增
                    targetNode = next;
                    cacheHitBlocks.increment();
                } else {
                    // 缓存未命中，分配新块
                    int bId = allocateWithRetry(sid, kv);
                    targetNode = new RadixNode(currentHash, bId);
                    current.children.put(currentHash, targetNode);
                }

                // 引用计数+1，绑定到会话
                targetNode.refCount.incrementAndGet();
                sessionNodes.computeIfAbsent(sid, k -> new CopyOnWriteArrayList<>()).add(targetNode);
                return targetNode.blockId;
            } finally {
                treeLock.unlock();
            }
        }

        // 修复5：路径匹配分配（批量块，Radix树核心）
        public List<Integer> matchAndAllocatePath(List<Long> pathHashes, String sessionId, PagedKvBufferV3 buffer) {
            List<Integer> result = new ArrayList<>();
            RadixNode current = root;
            treeLock.lock();
            try {
                sessionLastActive.put(sessionId, System.currentTimeMillis());
                for (Long h : pathHashes) {
                    RadixNode next = current.children.get(h);
                    RadixNode targetNode;
                    if (next != null) {
                        targetNode = next;
                        cacheHitBlocks.increment();
                    } else {
                        int bId = allocateWithRetry(sessionId, buffer);
                        targetNode = new RadixNode(h, bId);
                        current.children.put(h, new RadixNode(h, bId));
                    }
                    targetNode.refCount.incrementAndGet();
                    sessionNodes.computeIfAbsent(sessionId, k -> new CopyOnWriteArrayList<>()).add(targetNode);
                    result.add(targetNode.blockId);
                    current = targetNode;
                }
            } finally {
                treeLock.unlock();
            }
            checkAndPrune();
            return result;
        }

        // 修复6：块分配重试（核心！优化等待策略+重试次数）
        private int allocateWithRetry(String sessionId, PagedKvBufferV3 buffer) {
            super.getGlobalLock().lock();
            try {
                int retryCount = 0;
                while (true) {
                    // 尝试获取空闲块
                    Integer id = freePool.poll();
                    if (id != null) {
                        sessionLastActive.put(sessionId, System.currentTimeMillis());
                        return id;
                    }

                    // 无空闲块，尝试LRU驱逐最久未活动的会话
                    if (evictOldestSession(sessionId)) {
                        continue;
                    }

                    // 驱逐失败，阻塞等待，优化等待策略
                    WAIT_COUNT.increment();
                    retryCount++;
                    // 调整为：基础300ms + 重试数×200ms，最大5000ms（给驱逐足够时间）
                    long waitTime = Math.min(300 + (retryCount * 200), 5000);
                    // 无限重试，直到获取块/线程中断，移除10次重试阈值（关键！）
                    diskFullCondition.await(waitTime, TimeUnit.MILLISECONDS);
                }
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                throw new RuntimeException("块分配线程被中断：" + e.getMessage());
            } finally {
                super.getGlobalLock().unlock();
            }
        }

        // 修复7：水位检查+剪枝（优化阈值，增加驱逐力度）
        private void checkAndPrune() {
            int free = getFreeBlockCount();
            int total = totalBlocks;
            // 空闲块低于5%，触发剪枝/驱逐
            if (free < total * lowWatermark) {
                super.getGlobalLock().lock();
                try {
                    // 清理到30%空闲才停止，确保有足够的块资源
                    while (getFreeBlockCount() < total * highWatermark) {
                        // 先清理幽灵节点
                        if (!ghostCache.isEmpty()) {
                            RadixNode victim = ghostCache.pollFirst();
                            if (victim != null && victim.refCount.get() == 0) {
                                freePool.add(victim.blockId);
                                cacheHitBlocks.increment();
                            }
                        } else {
                            // 幽灵节点清理完仍不足，强制LRU驱逐
                            if (!evictOldestSession(null)) {
                                break; // 无会话可驱逐，退出
                            }
                        }
                    }
                } finally {
                    super.getGlobalLock().unlock();
                }
            }
        }

        // 修复8：核心LRU驱逐逻辑（重写父类，实际生效！）
        @Override
        boolean evictOldestSession(String excludeId) {
            super.getGlobalLock().lock();
            try {
                // 找到最久未活动的会话（排除指定会话）
                String oldestSession = null;
                long oldestTime = Long.MAX_VALUE;
                for (Map.Entry<String, Long> entry : sessionLastActive.entrySet()) {
                    String sid = entry.getKey();
                    if (sid.equals(excludeId)) continue;
                    if (entry.getValue() < oldestTime) {
                        oldestTime = entry.getValue();
                        oldestSession = sid;
                    }
                }

                // 无会话可驱逐
                if (oldestSession == null) return false;

                // 驱逐该会话：释放所有绑定的块
                List<RadixNode> nodes = sessionNodes.remove(oldestSession);
                if (nodes != null) {
                    for (RadixNode node : nodes) {
                        int remainingRefs = node.refCount.decrementAndGet();
                        if (remainingRefs == 0) {
                            freePool.add(node.blockId);
                            ghostCache.addLast(node);
                        }
                    }
                }
                // 移除会话的活动时间
                sessionLastActive.remove(oldestSession);
                EVICT_COUNT.increment();
                return true;
            } finally {
                super.getGlobalLock().unlock();
            }
        }

        // 修复9：释放会话（重写，更新引用计数+块回池）
        @Override
        public void releaseSession(String sessionId) {
            List<RadixNode> nodes = sessionNodes.remove(sessionId);
            if (nodes == null || nodes.isEmpty()) return;

            super.getGlobalLock().lock();
            try {
                for (RadixNode node : nodes) {
                    int remainingRefs = node.refCount.decrementAndGet();
                    // 引用计数为0，块回池+加入幽灵缓存
                    if (remainingRefs == 0) {
                        freePool.add(node.blockId);
                        ghostCache.addLast(node);
                        cacheHitBlocks.increment();
                    }
                }
                // 唤醒阻塞的分配线程
                diskFullCondition.signalAll();
            } finally {
                super.getGlobalLock().unlock();
            }
        }

        // 实现父类抽象方法：批量分配块
        @Override
        public List<Integer> allocateBlocks(int neededBlocks, String sessionId, PagedKvBufferV3 kvBuffer) {
            List<Integer> blocks = new ArrayList<>();
            for (int i = 0; i < neededBlocks; i++) {
                blocks.add(allocateWithRetry(sessionId, kvBuffer));
            }
            // 批量分配统计自增
            totalRequests.add(neededBlocks);
            return blocks;
        }

        // Getter
        public ConcurrentHashMap<String, List<RadixNode>> getSessionNodes() { return sessionNodes; }
        public Condition getDiskFullCondition() { return diskFullCondition; }
    }
}
