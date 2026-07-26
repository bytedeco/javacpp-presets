package org.bytedeco.pytorch.geometric.demo.kvcache;
import org.bytedeco.pytorch.jit.*;
import org.bytedeco.pytorch.c10.*;
import org.bytedeco.pytorch.jit.JitModule;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.global.torch;

import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.Future;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.LongAdder;
import java.util.concurrent.locks.Condition;
import java.util.concurrent.locks.ReentrantLock;
import java.util.concurrent.locks.ReentrantReadWriteLock;

/**
 * V11版本：集成虚拟线程的高并发压测版
 * 基于V10核心优化 + Java 21虚拟线程 + 高并发统计
 */
public class Qwen3InferenceBenchmarkV11 {
    // V11推理器：兼容虚拟线程的高并发版本
    public static class Qwen3JavaInferenceV11 {
        private final JitModule model; // 加载好的Qwen模型
        private final Device device;   // CPU设备
        private final CoWBlockManagerV9 blockManager; // V9版本块管理器
        private final ConcurrentHashMap<String, PagedKvBufferV3> kvBufferMap; // PagedKV缓存
        private final ConcurrentHashMap<String, CacheEntry> cacheMetaMap; // 缓存元数据

        // PageAttention V3 配置
        private static final int PAGE_SIZE = 1024; // 每页Token数
        private static final int BLOCK_SIZE = 256; // 块大小（与CoWBlockManagerV9对齐）
        private static final int MAX_SEQ_LEN = 8192; // 最大序列长度
        private static final int NUM_LAYERS = 32; // Qwen3-4B的层数
        private static final int HEAD_DIM = 128; // 注意力头维度
        private static final int TOTAL_BLOCKS = 2048; // 增大总块数，适配高并发
        private static final int DTYPE = 0; // Float32

        // 高并发统计（替换AtomicLong为LongAdder）
        private final LongAdder totalInferenceTime = new LongAdder();
        private final LongAdder totalTokensGenerated = new LongAdder();
        private final AtomicInteger completedTasks = new AtomicInteger(0);
        private final AtomicInteger failedTasks = new AtomicInteger(0);

        // 构造函数 - 初始化模型和V9块管理器
        public Qwen3JavaInferenceV11(String modelPath) {
            // 初始化设备
            this.device = new Device(torch.kCPU());

            // 加载模型到CPU（线程安全模式）
            this.model = torch.load(modelPath, new DeviceOptional(this.device), false);
            this.model.eval(); // 设置为推理模式

            // 初始化V9块管理器（增大总块数，适配高并发）
            this.blockManager = new CoWBlockManagerV9(TOTAL_BLOCKS, NUM_LAYERS, BLOCK_SIZE, HEAD_DIM, DTYPE);

            // 初始化缓存映射（并发安全）
            this.kvBufferMap = new ConcurrentHashMap<>();
            this.cacheMetaMap = new ConcurrentHashMap<>();

            // 禁用梯度计算，提升推理性能
            torch.requires_grad(false);
//            torch.no_grad();
        }

        /**
         * V11核心generate方法：支持虚拟线程并发调用
         * @param inputIds 输入Token的张量 (shape: [1, seq_len])
         * @param generateLen 生成Token长度
         * @param sessionId 会话ID（虚拟线程隔离）
         * @return 推理结果封装
         */
        public InferenceResult generateConcurrent(Tensor inputIds, int generateLen, String sessionId) {
            // 复制输入张量，避免线程间共享
            Tensor currentInput = inputIds.to(device, torch.ScalarType.Long).clone();
            long inputSeqLen = currentInput.size(1);
            long startTime = System.currentTimeMillis();

            // 线程安全的缓存初始化
            PagedKvBufferV3 kvBuffer = kvBufferMap.computeIfAbsent(sessionId,
                    k -> new PagedKvBufferV3(sessionId, blockManager, NUM_LAYERS));
            CacheEntry cacheMeta = cacheMetaMap.computeIfAbsent(sessionId, k -> new CacheEntry());

            try {
                // 边界检查
                if (currentInput.dim() != 2) {
                    throw new IllegalArgumentException("输入必须是2维张量，当前维度：" + currentInput.dim());
                }

                // Prefill阶段
                if (cacheMeta.seqLen == 0) {
                    prefillKVCache(currentInput, kvBuffer, cacheMeta);
                    cacheMeta.seqLen = (int) inputSeqLen;
                }

                // 逐Token生成（自回归）
                for (int step = 0; step < generateLen; step++) {
                    // 边界检查：不超过最大序列长度
                    if (cacheMeta.seqLen + inputSeqLen >= MAX_SEQ_LEN) {
                        throw new RuntimeException("序列长度超过最大值：" + MAX_SEQ_LEN);
                    }

                    // 仅传入最后一个Token
                    Tensor newTokenInput = currentInput.narrow(1, currentInput.size(1) - 1, 1);
                    Tensor logitsTensor = forwardWithKVCache(newTokenInput);
                    Tensor lastStepLogits = getLastTokenLogits(logitsTensor, 1);
                    Tensor nextToken = safeGreedySample(lastStepLogits);
                    nextToken = ensure2DToken(nextToken);

                    // 更新KV缓存（线程安全）
                    updateKVCacheWithV9Manager(nextToken, kvBuffer, cacheMeta, sessionId);

                    // 拼接Token
                    TensorVector catTensors = new TensorVector();
                    catTensors.push_back(currentInput);
                    catTensors.push_back(nextToken);
                    currentInput = torch.cat(catTensors, 1);

                    // 更新序列长度
                    cacheMeta.seqLen++;

                    // PageAttention分页优化
                    if (cacheMeta.seqLen % PAGE_SIZE == 0 && cacheMeta.seqLen > 0) {
                        optimizeKVCacheWithPageAttentionV3(kvBuffer, cacheMeta, sessionId);
                    }

                    // 释放临时张量
                    logitsTensor.close();
                    lastStepLogits.close();
                    nextToken.close();
                    catTensors.close();
                    newTokenInput.close();
                }

                // 统计耗时和Token数
                long endTime = System.currentTimeMillis();
                long inferenceTime = endTime - startTime;
                totalInferenceTime.add(inferenceTime);
                totalTokensGenerated.add(generateLen);
                completedTasks.incrementAndGet();

                // 返回结果
                return new InferenceResult(
                        true,
                        sessionId,
                        generateLen,
                        inputSeqLen,
                        currentInput.size(1),
                        inferenceTime,
                        null
                );

            } catch (Exception e) {
                failedTasks.incrementAndGet();
                cleanSessionResources(sessionId); // 出错清理资源
                return new InferenceResult(
                        false,
                        sessionId,
                        generateLen,
                        inputSeqLen,
                        0,
                        System.currentTimeMillis() - startTime,
                        e.getMessage()
                );
            } finally {
                currentInput.close(); // 确保张量释放
            }
        }

        // ========== 核心方法复用V10（略作线程安全优化） ==========
        private void prefillKVCache(Tensor input, PagedKvBufferV3 kvBuffer, CacheEntry cacheMeta) {
            int numTokens = (int) input.size(1);
            for (int layer = 0; layer < NUM_LAYERS; layer++) {
                kvBuffer.prefillUltra(layer, 0, input);
                kvBuffer.prefillUltra(layer, 1, input);
                cacheMeta.kBlockCount[layer] = kvBuffer.getKBlockCount(layer);
                cacheMeta.vBlockCount[layer] = kvBuffer.getVBlockCount(layer);
            }
            cacheMeta.physicalBlockIds = blockManager.getPhysicalBlockIds(kvBuffer.getSessionId().toString());
        }

        private void updateKVCacheWithV9Manager(Tensor token, PagedKvBufferV3 kvBuffer, CacheEntry cacheMeta, String sessionId) {
            int neededBlocks = (1 + BLOCK_SIZE - 1) / BLOCK_SIZE;
            blockManager.allocateBlocks(neededBlocks, BLOCK_SIZE);

            for (int layer = 0; layer < NUM_LAYERS; layer++) {
                kvBuffer.prefillUltra(layer, 0, token);
                kvBuffer.prefillUltra(layer, 1, token);
                cacheMeta.kBlockCount[layer] = kvBuffer.getKBlockCount(layer);
                cacheMeta.vBlockCount[layer] = kvBuffer.getVBlockCount(layer);
            }
            cacheMeta.physicalBlockIds = blockManager.getPhysicalBlockIds(sessionId);
        }

        private void optimizeKVCacheWithPageAttentionV3(PagedKvBufferV3 kvBuffer, CacheEntry cacheMeta, String sessionId) {
            long[] blockIds = blockManager.getPhysicalBlockIds(sessionId);
            int pageNum = (int) (cacheMeta.seqLen / PAGE_SIZE);
            cacheMeta.lastPageIdx = pageNum;

            if (pageNum > 3) {
                List<Integer> invalidatedBlocks = kvBuffer.getAndInvalidateBlocks();
                blockManager.releaseBlocks(sessionId);
                cacheMeta.invalidatedBlockCount += invalidatedBlocks.size();
                cacheMeta.physicalBlockIds = blockManager.getPhysicalBlockIds(sessionId);
            }
        }

        private Tensor forwardWithKVCache(Tensor inputIds) {
            IValue output;
            try (IValueVector inputs = new IValueVector()) {
                inputs.push_back(new IValue(inputIds));
                output = model.forward(inputs);
            }
            return output.toTensor();
        }

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

        private Tensor safeGreedySample(Tensor logits) {
            int vocabDim = (int) logits.dim() - 1;
            Tensor argmaxTensor = torch.argmax(logits, new LongOptional(vocabDim), false);
            Tensor nextToken = argmaxTensor.to(device, torch.ScalarType.Long);
            argmaxTensor.close();
            return nextToken;
        }

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

        // ========== 线程安全的资源管理 ==========
        public void cleanSessionResources(String sessionId) {
            // 原子操作移除缓存
            PagedKvBufferV3 kvBuffer = kvBufferMap.remove(sessionId);
            if (kvBuffer != null) {
                kvBuffer.close();
            }
            blockManager.releaseBlocks(sessionId);
            cacheMetaMap.remove(sessionId);
        }

        // ========== 高并发统计方法 ==========
        public double getTokenGenerationSpeed() {
            long totalTime = totalInferenceTime.sum();
            long totalTokens = totalTokensGenerated.sum();
            if (totalTime == 0 || totalTokens == 0) {
                return 0.0;
            }
            return totalTokens / (totalTime / 1000.0);
        }

        public String getConcurrentStats() {
            return String.format(
                    "完成任务数: %d, 失败任务数: %d, 总Token数: %d, 总耗时: %dms, 平均Token速度: %.2f tokens/s",
                    completedTasks.get(),
                    failedTasks.get(),
                    totalTokensGenerated.sum(),
                    totalInferenceTime.sum(),
                    getTokenGenerationSpeed()
            );
        }

        public String getBlockManagerStats() {
            return String.format(
                    "块管理器统计 | 总请求数: %d, 缓存命中块数: %d, 驱逐次数: %d, 等待次数: %d",
                    blockManager.totalRequests.sum(),
                    blockManager.cacheHitBlocks.sum(),
                    CoWBlockManagerV9.EVICT_COUNT.sum(),
                    CoWBlockManagerV9.WAIT_COUNT.sum()
            );
        }

        // 关闭资源
        public void close() {
            // 批量清理所有会话
            kvBufferMap.keySet().forEach(this::cleanSessionResources);
            kvBufferMap.clear();
            cacheMetaMap.clear();

            // 释放模型和设备
            if (model != null) model.close();
            if (device != null) device.close();

            // 打印统计
            System.out.println("\n=== 虚拟线程压测统计 ===");
            System.out.println(getConcurrentStats());
            System.out.println(getBlockManagerStats());
        }

        // 缓存元数据
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

            public InferenceResult(boolean success, String sessionId, int generateLen,
                                   long inputSeqLen, long outputSeqLen, long inferenceTime, String errorMsg) {
                this.success = success;
                this.sessionId = sessionId;
                this.generateLen = generateLen;
                this.inputSeqLen = inputSeqLen;
                this.outputSeqLen = outputSeqLen;
                this.inferenceTime = inferenceTime;
                this.errorMsg = errorMsg;
            }
        }
    }

    // ======================== 虚拟线程压测主方法 ========================
    public static void main(String[] args) throws InterruptedException {
        // Java 21+ 虚拟线程需要的VM参数：--enable-preview
        String modelPath = "/Users/mullerzhang/Documents/code/langchain/qwen3_4b_fp16_mps.pt";

        // 1. 初始化V11推理器
        Qwen3JavaInferenceV11 inference = new Qwen3JavaInferenceV11(modelPath);

        // 2. 压测配置（高并发）
        int concurrentThreads = 10;    // 虚拟线程数（可设为100+）
        int testRoundPerThread = 5;    // 每个线程压测轮数
        int generateLenPerRound = 100; // 每轮生成Token数
        String baseSessionId = "vt_session_v11";

        // 3. 测试输入
        long[] inputTokens = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
        Tensor inputIds = torch.tensor(inputTokens,
                        new TensorOptions().dtype(new ScalarTypeOptional(torch.ScalarType.Long)))
                .reshape(1, 16);

        // 4. 预热（单线程）
        System.out.println("===== 开始V11虚拟线程压测（PagedKvBufferV3+CoWBlockManagerV9） =====");
        System.out.printf("压测配置：虚拟线程数=%d, 每线程轮数=%d, 每轮Token数=%d, PageSize=%d, BlockSize=%d\n",
                concurrentThreads, testRoundPerThread, generateLenPerRound,
                Qwen3JavaInferenceV11.PAGE_SIZE, Qwen3JavaInferenceV11.BLOCK_SIZE);

        System.out.println("\n=== 预热阶段 ===");
        try {
            Qwen3JavaInferenceV11.InferenceResult warmupResult =
                    inference.generateConcurrent(inputIds, 10, baseSessionId + "_warmup");
            System.out.println("预热" + (warmupResult.success ? "成功" : "失败") +
                    " | 耗时: " + warmupResult.inferenceTime + "ms");
            inference.cleanSessionResources(baseSessionId + "_warmup");
        } catch (Exception e) {
            System.err.println("预热失败：" + e.getMessage());
            e.printStackTrace();
            return;
        }

        // 5. 创建虚拟线程池（Java 21+）
        ExecutorService virtualThreadExecutor = Executors.newVirtualThreadPerTaskExecutor();
        CompletionService<Qwen3JavaInferenceV11.InferenceResult> completionService =
                new ExecutorCompletionService<>(virtualThreadExecutor);

        // 6. 提交虚拟线程任务
        System.out.println("\n=== 虚拟线程高并发压测 ===");
        long totalStartTime = System.currentTimeMillis();
        int taskCount = 0;

        // 为每个虚拟线程分配压测任务
        for (int threadIdx = 0; threadIdx < concurrentThreads; threadIdx++) {
            for (int round = 0; round < testRoundPerThread; round++) {
                String sessionId = baseSessionId + "_thread" + threadIdx + "_round" + round;
                completionService.submit(() ->
                        inference.generateConcurrent(inputIds, generateLenPerRound, sessionId)
                );
                taskCount++;
            }
        }

        // 7. 收集任务结果
        List<Qwen3JavaInferenceV11.InferenceResult> results = new ArrayList<>();
        for (int i = 0; i < taskCount; i++) {
            try {
                Future<Qwen3JavaInferenceV11.InferenceResult> future = completionService.take();
                results.add(future.get());
            } catch (ExecutionException e) {
                System.err.println("任务执行异常：" + e.getCause().getMessage());
                inference.failedTasks.incrementAndGet();
            }
        }

        // 8. 关闭线程池
        virtualThreadExecutor.shutdown();
        virtualThreadExecutor.awaitTermination(1, TimeUnit.MINUTES);
        long totalEndTime = System.currentTimeMillis();

        // 9. 输出结果
        System.out.println("\n=== 单任务结果详情 ===");
        for (Qwen3JavaInferenceV11.InferenceResult result : results) {
            if (result.success) {
                System.out.printf("会话%s | 成功 | 耗时: %dms | 生成Token: %d | 输入长度: %d | 输出长度: %d\n",
                        result.sessionId, result.inferenceTime, result.generateLen,
                        result.inputSeqLen, result.outputSeqLen);
            } else {
                System.err.printf("会话%s | 失败 | 耗时: %dms | 错误: %s\n",
                        result.sessionId, result.inferenceTime, result.errorMsg);
            }
        }

        // 10. 全局统计
        System.out.println("\n===== 高并发压测统计 =====");
        System.out.printf("总任务数: %d, 总耗时: %dms (墙钟时间)\n", taskCount, totalEndTime - totalStartTime);
        System.out.println(inference.getConcurrentStats());
        System.out.println(inference.getBlockManagerStats());

        // 11. 释放资源
        inputIds.close();
        inference.close();
        System.out.println("\nV11虚拟线程压测完成，所有资源已释放！");
    }

    // ======================== 依赖类（保持原有实现） ========================
    // CoWBlockManagerV2父类
//    abstract static class CoWBlockManagerV2 {
//        protected int totalBlocks;
//        protected final Queue<Integer> freePool = new ConcurrentLinkedQueue<>();
//        private final ReentrantLock globalLock = new ReentrantLock();
//
//        public CoWBlockManagerV2(int totalBlocks, int layers, int blockSize, int headDim, int dtype) {
//            this.totalBlocks = totalBlocks;
//            for (int i = 0; i < totalBlocks; i++) {
//                freePool.add(i);
//            }
//        }
//
//        public int getFreeBlockCount() {
//            return freePool.size();
//        }
//
//        public int getBlockSize() {
//            return 256;
//        }
//
//        public ReentrantLock getGlobalLock() {
//            return globalLock;
//        }
//
//        public void releaseSession(String sessionId) {}
//
//        boolean evictOldestSession(String excludeId) { return false; }
//    }

    // PagedKvBufferV3（保持原有实现）
//    public static class PagedKvBufferV3 implements AutoCloseable {
//        private final String sessionId;
//        private final CoWBlockManagerV2 manager;
//        private final int numLayers;
//        private final List<Integer>[] kBlockMaps;
//        private final List<Integer>[] vBlockMaps;
//
//        private final ReentrantReadWriteLock stateLock = new ReentrantReadWriteLock();
//        private final AtomicBoolean isInvalidated = new AtomicBoolean(false);
//
//        @SuppressWarnings("unchecked")
//        public PagedKvBufferV3(String sessionId, CoWBlockManagerV2 manager, int numLayers) {
//            this.sessionId = sessionId;
//            this.manager = manager;
//            this.numLayers = numLayers;
//            this.kBlockMaps = new ArrayList[numLayers];
//            this.vBlockMaps = new ArrayList[numLayers];
//            for (int i = 0; i < numLayers; i++) {
//                kBlockMaps[i] = new ArrayList<>();
//                vBlockMaps[i] = new ArrayList<>();
//            }
//        }
//
//        public void prefillUltra(int layer, int kvType, Tensor input) {
//            stateLock.readLock().lock();
//            try {
//                if (isInvalidated.get()) return;
//
//                int numTokens = (int) input.size(0);
//                int blockSize = manager.getBlockSize();
//                int neededBlocks = (numTokens + blockSize - 1) / blockSize;
//
//                List<Integer> newBlocks = manager.allocateBlocks(neededBlocks, sessionId, this);
//                if (kvType == 0) kBlockMaps[layer].addAll(newBlocks);
//                else vBlockMaps[layer].addAll(newBlocks);
//            } finally {
//                stateLock.readLock().unlock();
//            }
//        }
//
//        public List<Integer> getAndInvalidateBlocks() {
//            stateLock.writeLock().lock();
//            try {
//                isInvalidated.set(true);
//                List<Integer> allBlocks = new ArrayList<>();
//                for (int i = 0; i < numLayers; i++) {
//                    allBlocks.addAll(kBlockMaps[i]);
//                    allBlocks.addAll(vBlockMaps[i]);
//                    kBlockMaps[i].clear();
//                    vBlockMaps[i].clear();
//                }
//                return allBlocks;
//            } finally {
//                stateLock.writeLock().unlock();
//            }
//        }
//
//        @Override
//        public void close() { manager.releaseSession(sessionId); }
//
//        public CharSequence getSessionId() { return sessionId; }
//
//        public int getKBlockCount(int layer) { return kBlockMaps[layer].size(); }
//
//        public int getVBlockCount(int layer) { return vBlockMaps[layer].size(); }
//    }

    // CoWBlockManagerV9（保持原有实现）
    public static class CoWBlockManagerV9 extends CoWBlockManagerV2 {
        private final int actualBlockSize;
        private final Condition diskFullCondition;
        private final ReentrantLock treeLock = new ReentrantLock();
        public static final LongAdder EVICT_COUNT = new LongAdder();
        public static final LongAdder WAIT_COUNT = new LongAdder();
        private final double lowWatermark = 0.10;
        private final double highWatermark = 0.20;

        public final LongAdder totalRequests = new LongAdder();
        public final LongAdder cacheHitBlocks = new LongAdder();

        private final Deque<RadixNode> ghostCache = new ConcurrentLinkedDeque<>();
        private final ConcurrentHashMap<String, List<RadixNode>> sessionNodes = new ConcurrentHashMap<>();

        static class RadixNode {
            final long hash;
            final int blockId;
            final ConcurrentHashMap<Long, RadixNode> children = new ConcurrentHashMap<>();
            final AtomicInteger refCount = new AtomicInteger(0);
            RadixNode(long hash, int blockId) { this.hash = hash; this.blockId = blockId; }
        }

        private final RadixNode root = new RadixNode(-1, -1);

        public CoWBlockManagerV9(int totalBlocks, int layers, int blockSize, int headDim, int dtype) {
            super(totalBlocks, layers, blockSize, headDim, dtype);
            this.actualBlockSize = blockSize;
            this.diskFullCondition = super.getGlobalLock().newCondition();
        }

        public void allocateBlocks(int numBlocks, int blockSize) {
            if (blockSize != this.actualBlockSize) {
                throw new IllegalArgumentException("块大小不匹配：" + this.actualBlockSize + " vs " + blockSize);
            }
            if (numBlocks <= 0) return;

            super.getGlobalLock().lock();
            try {
                for (int i = 0; i < numBlocks; i++) {
                    allocateWithRetry("temp_session", null);
                }
            } finally {
                super.getGlobalLock().unlock();
            }
            checkAndPrune();
        }

        public void releaseBlocks(String sessionId) {
            if (sessionId == null || sessionId.isEmpty()) return;
            this.releaseSession(sessionId);

            super.getGlobalLock().lock();
            try {
                Iterator<RadixNode> iterator = ghostCache.iterator();
                while (iterator.hasNext()) {
                    RadixNode node = iterator.next();
                    if (node.refCount.get() == 0) {
                        freePool.add(node.blockId);
                        iterator.remove();
                    }
                }
                diskFullCondition.signalAll();
            } finally {
                super.getGlobalLock().unlock();
            }
        }

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

        public int getOrAllocateBlock(long currentHash, String sid, PagedKvBufferV3 kv) {
            treeLock.lock();
            try {
                RadixNode current = root;
                RadixNode next = current.children.get(currentHash);
                RadixNode targetNode;

                if (next != null) {
                    targetNode = next;
                } else {
                    int bId = allocateWithRetry(sid, kv);
                    RadixNode newNode = new RadixNode(currentHash, bId);
                    newNode.refCount.set(0);
                    current.children.put(currentHash, newNode);
                    targetNode = newNode;
                }

                targetNode.refCount.incrementAndGet();
                sessionNodes.computeIfAbsent(sid, k -> new CopyOnWriteArrayList<>()).add(targetNode);

                return targetNode.blockId;
            } finally {
                treeLock.unlock();
            }
        }

        public List<Integer> matchAndAllocatePath(List<Long> pathHashes, String sessionId, PagedKvBufferV3 buffer) {
            List<Integer> result = new ArrayList<>();
            RadixNode current = root;

            treeLock.lock();
            try {
                for (Long h : pathHashes) {
                    RadixNode next = current.children.get(h);
                    RadixNode targetNode;

                    if (next != null) {
                        targetNode = next;
                    } else {
                        int bId = allocateWithRetry(sessionId, buffer);
                        RadixNode newNode = new RadixNode(h, bId);
                        newNode.refCount.set(0);
                        current.children.put(h, newNode);
                        targetNode = newNode;
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

        private int allocateWithRetry(String sessionId, PagedKvBufferV3 buffer) {
            super.getGlobalLock().lock();
            try {
                int retryCount = 0;
                while (true) {
                    Integer id = freePool.poll();
                    if (id != null) return id;

                    if (evictOldestSession(sessionId)) {
                        continue;
                    }
                    WAIT_COUNT.increment();
                    retryCount++;
                    long waitTime = Math.min(100 + (retryCount * 100), 2000);

                    if (!diskFullCondition.await(waitTime, TimeUnit.MILLISECONDS)) {
                        if (retryCount > 10) {
                            throw new RuntimeException("GPU Memory Timeout after " + retryCount + " retries");
                        }
                    }
                }
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                throw new RuntimeException(e);
            } finally {
                super.getGlobalLock().unlock();
            }
        }

        private void checkAndPrune() {
            int free = getFreeBlockCount();
            int total = getTotalBlocks();

            if (free < total * lowWatermark) {
                super.getGlobalLock().lock();
                try {
                    while (getFreeBlockCount() < total * highWatermark && !ghostCache.isEmpty()) {
                        RadixNode victim = ghostCache.pollFirst();
                        if (victim != null && victim.refCount.get() == 0) {
                            freePool.add(victim.blockId);
                        }
                    }
                } finally {
                    super.getGlobalLock().unlock();
                }
            }
        }

        private int getTotalBlocks() {
            return super.totalBlocks;
        }

        private int allocateWithRetry2(String sessionId, PagedKvBufferV3 buffer) {
            super.getGlobalLock().lock();
            try {
                while (true) {
                    Integer id = freePool.poll();
                    if (id != null) return id;

                    if (evictOldestSession(sessionId)) {
                        continue;
                    }
                    WAIT_COUNT.increment();
                    if (!diskFullCondition.await(500, TimeUnit.MILLISECONDS)) {
                        throw new RuntimeException("GPU Memory Timeout: System saturated.");
                    }
                }
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                throw new RuntimeException(e);
            } finally {
                super.getGlobalLock().unlock();
            }
        }

        @Override
        boolean evictOldestSession(String excludeId) {
            boolean success = super.evictOldestSession(excludeId);
            if (success) {
                EVICT_COUNT.increment();
                super.getGlobalLock().lock();
                try {
                    diskFullCondition.signalAll();
                } finally {
                    super.getGlobalLock().unlock();
                }
            }
            return success;
        }

        @Override
        public void releaseSession(String sessionId) {
            List<RadixNode> nodes = getSessionNodes().remove(sessionId);

            if (nodes != null) {
                super.getGlobalLock().lock();
                try {
                    for (RadixNode node : nodes) {
                        int remainingRefs = node.refCount.decrementAndGet();
                        if (remainingRefs == 0) {
                            freePool.add(node.blockId);
                            ghostCache.addLast(node);
                        }
                    }
                    getDiskFullCondition().signalAll();
                } finally {
                    super.getGlobalLock().unlock();
                }
            }
        }

        public ConcurrentHashMap<String, List<RadixNode>> getSessionNodes() {
            return sessionNodes;
        }

        public Condition getDiskFullCondition() {
            return diskFullCondition;
        }

        // 补充父类未实现的方法
        public List<Integer> allocateBlocks(int neededBlocks, String sessionId, PagedKvBufferV3 kvBuffer) {
            List<Integer> blocks = new ArrayList<>();
            for (int i = 0; i < neededBlocks; i++) {
                blocks.add(allocateWithRetry(sessionId, kvBuffer));
            }
            return blocks;
        }
    }
}
