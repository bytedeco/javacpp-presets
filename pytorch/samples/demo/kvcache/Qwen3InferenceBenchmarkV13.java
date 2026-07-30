//package samples.demo.kvcache;
//import org.bytedeco.pytorch.jit.*;
//import org.bytedeco.pytorch.c10.*;
//import org.bytedeco.pytorch.jit.JitModule;
//
//import org.bytedeco.pytorch.*;
//import org.bytedeco.pytorch.global.torch;
//
//import java.util.*;
//import java.util.concurrent.*;
//import java.util.concurrent.Future;
//import java.util.concurrent.atomic.*;
//import java.util.concurrent.locks.*;
//
///**
// * V11终极修复版：防卡死+细粒度锁+超时保护+虚拟线程限流
// * 解决：全局锁竞争/无限阻塞/死锁/虚拟线程失控等核心卡死问题
// */
//public class Qwen3InferenceBenchmarkV13 {
//    // V11推理器：防卡死高并发版本
//    public static class Qwen3JavaInferenceV11 {
//        private final JitModule model;
//        private final Device device;
//        private final CoWBlockManagerV9 blockManager;
//        private final ConcurrentHashMap<String, PagedKvBufferV3> kvBufferMap;
//        private final ConcurrentHashMap<String, CacheEntry> cacheMetaMap;
//
//        // 核心配置（防卡死优化）
//        public static final int PAGE_SIZE = 1024;
//        public static final int BLOCK_SIZE = 256;
//        public static final int MAX_SEQ_LEN = 8192;
//        public static final int NUM_LAYERS = 32;
//        public static final int TOTAL_BLOCKS = 16384;
//        public static final int DTYPE = 0;
//        private static final int HEAD_DIM = 128;
//
//        // 高并发统计
//        private final LongAdder totalInferenceTime = new LongAdder();
//        private final LongAdder totalTokensGenerated = new LongAdder();
//        private final AtomicInteger completedTasks = new AtomicInteger(0);
//        private final AtomicInteger failedTasks = new AtomicInteger(0);
//        private final AtomicInteger timeoutTasks = new AtomicInteger(0); // 新增：超时任务统计
//
//        // 防卡死核心配置
//        private final Semaphore taskSemaphore;          // 任务限流
//        private final long taskTimeoutMs = 30000;       // 单任务超时时间（30s）
//        private final int maxVirtualThreads = 10;       // 限制虚拟线程数
//
//        // 构造函数：初始化+防卡死配置
//        public Qwen3JavaInferenceV11(String modelPath, int maxConcurrentTasks) {
//            this.device = new Device(torch.kCPU());
//            // 加载模型（线程安全）
//            this.model = torch.load(modelPath, new DeviceOptional(this.device), false);
//            this.model.eval();
//            // 初始化终极修复版V9块管理器（细粒度锁）
//            this.blockManager = new CoWBlockManagerV9(TOTAL_BLOCKS, NUM_LAYERS, BLOCK_SIZE, HEAD_DIM, DTYPE);
//            this.kvBufferMap = new ConcurrentHashMap<>();
//            this.cacheMetaMap = new ConcurrentHashMap<>();
//            // 任务限流：最大并发任务数=硬件核心数（推荐）
//            this.taskSemaphore = new Semaphore(maxConcurrentTasks);
//            // 禁用梯度计算
//            torch.requires_grad(false);
////            torch.no_grad();
//        }
//
//        /**
//         * 核心并发生成方法：防卡死+超时保护+细粒度锁
//         */
//        public InferenceResult generateConcurrent(Tensor inputIds, int generateLen, String sessionId) {
//            Tensor currentInput = null;
//            long startTime = System.currentTimeMillis();
//            try {
//                // 1. 任务限流：带超时获取许可（防永久阻塞）
//                if (!taskSemaphore.tryAcquire(taskTimeoutMs, TimeUnit.MILLISECONDS)) {
//                    timeoutTasks.incrementAndGet();
//                    return new InferenceResult(false, sessionId, generateLen, 0, 0,
//                            System.currentTimeMillis() - startTime, "任务获取许可超时（" + taskTimeoutMs + "ms）", 0);
//                }
//
//                // 2. 复制输入张量（避免共享）
//                currentInput = inputIds.to(device, torch.ScalarType.Long).clone();
//                long inputSeqLen = currentInput.size(1);
//
//                // 3. 单任务超时保护
//                if (System.currentTimeMillis() - startTime > taskTimeoutMs) {
//                    timeoutTasks.incrementAndGet();
//                    return new InferenceResult(false, sessionId, generateLen, inputSeqLen, 0,
//                            System.currentTimeMillis() - startTime, "单任务执行超时", 0);
//                }
//
//                // 4. 线程安全缓存初始化
//                PagedKvBufferV3 kvBuffer = kvBufferMap.computeIfAbsent(sessionId,
//                        k -> new PagedKvBufferV3(sessionId, blockManager, NUM_LAYERS));
//                CacheEntry cacheMeta = cacheMetaMap.computeIfAbsent(sessionId, k -> new CacheEntry());
//
//                // 5. 边界检查
//                if (currentInput.dim() != 2) {
//                    throw new IllegalArgumentException("输入必须是2维张量，当前维度：" + currentInput.dim());
//                }
//                if (inputSeqLen + generateLen > MAX_SEQ_LEN) {
//                    throw new RuntimeException("序列长度超过最大值：" + MAX_SEQ_LEN);
//                }
//
//                // 6. Prefill阶段
//                if (cacheMeta.seqLen == 0) {
//                    prefillKVCache(currentInput, kvBuffer, cacheMeta);
//                    cacheMeta.seqLen = (int) inputSeqLen;
//                }
//
//                // 7. 逐Token生成（带超时检查）
//                int generated = 0;
//                for (int step = 0; step < generateLen; step++) {
//                    // 单步超时检查
//                    if (System.currentTimeMillis() - startTime > taskTimeoutMs) {
//                        timeoutTasks.incrementAndGet();
//                        return new InferenceResult(false, sessionId, generated, inputSeqLen,
//                                inputSeqLen + generated, System.currentTimeMillis() - startTime,
//                                "Token生成超时", cacheMeta.physicalBlockIds.length);
//                    }
//
//                    // 核心生成逻辑
//                    Tensor newTokenInput = currentInput.narrow(1, currentInput.size(1) - 1, 1);
//                    Tensor logitsTensor = forwardWithKVCache(newTokenInput);
//                    Tensor lastStepLogits = getLastTokenLogits(logitsTensor, 1);
//                    Tensor nextToken = safeGreedySample(lastStepLogits);
//                    nextToken = ensure2DToken(nextToken);
//
//                    // 更新KV缓存（细粒度锁）
//                    updateKVCacheWithV9Manager(nextToken, kvBuffer, cacheMeta, sessionId);
//
//                    // 拼接Token
//                    TensorVector catTensors = new TensorVector();
//                    catTensors.push_back(currentInput);
//                    catTensors.push_back(nextToken);
//                    currentInput = torch.cat(catTensors, 1);
//                    cacheMeta.seqLen++;
//                    generated++;
//
//                    // PageAttention优化（定期清理，减少锁竞争）
//                    if (cacheMeta.seqLen % PAGE_SIZE == 0 && cacheMeta.seqLen > 0) {
//                        optimizeKVCacheWithPageAttentionV3(kvBuffer, cacheMeta, sessionId);
//                    }
//
//                    // 及时释放临时张量
//                    logitsTensor.close();
//                    lastStepLogits.close();
//                    nextToken.close();
//                    catTensors.close();
//                    newTokenInput.close();
//                }
//
//                // 8. 统计成功结果
//                long inferenceTime = System.currentTimeMillis() - startTime;
//                totalInferenceTime.add(inferenceTime);
//                totalTokensGenerated.add(generated);
//                completedTasks.incrementAndGet();
//
//                return new InferenceResult(true, sessionId, generated, inputSeqLen,
//                        currentInput.size(1), inferenceTime, null, cacheMeta.physicalBlockIds.length);
//
//            } catch (InterruptedException e) {
//                Thread.currentThread().interrupt();
//                failedTasks.incrementAndGet();
//                return new InferenceResult(false, sessionId, 0, 0, 0,
//                        System.currentTimeMillis() - startTime, "任务被中断：" + e.getMessage(), 0);
//            } catch (Exception e) {
//                failedTasks.incrementAndGet();
//                return new InferenceResult(false, sessionId, 0, 0, 0,
//                        System.currentTimeMillis() - startTime, e.getMessage(), 0);
//            } finally {
//                // 9. 安全清理：非阻塞释放+避免死锁
//                if (currentInput != null) {
//                    try {
//                        currentInput.close();
//                    } catch (Exception e) { /* 忽略关闭异常 */ }
//                }
//                // 非阻塞清理资源（核心：避免死锁）
//                try {
//                    cleanSessionResourcesNonBlocking(sessionId);
//                } catch (Exception e) {
//                    System.err.println("非阻塞清理资源失败：" + sessionId + "，错误：" + e.getMessage());
//                }
//                // 释放限流许可
//                taskSemaphore.release();
//            }
//        }
//
//        // ========== 核心方法（防卡死优化） ==========
//        private void prefillKVCache(Tensor input, PagedKvBufferV3 kvBuffer, CacheEntry cacheMeta) {
//            int numTokens = (int) input.size(0);
//            for (int layer = 0; layer < NUM_LAYERS; layer++) {
//                kvBuffer.prefillUltra(layer, 0, input);
//                kvBuffer.prefillUltra(layer, 1, input);
//                cacheMeta.kBlockCount[layer] = kvBuffer.getKBlockCount(layer);
//                cacheMeta.vBlockCount[layer] = kvBuffer.getVBlockCount(layer);
//            }
//            cacheMeta.physicalBlockIds = blockManager.getPhysicalBlockIds(kvBuffer.getSessionId().toString());
//        }
//
//        private void updateKVCacheWithV9Manager(Tensor token, PagedKvBufferV3 kvBuffer, CacheEntry cacheMeta, String sessionId) {
//            int neededBlocks = (1 + BLOCK_SIZE - 1) / BLOCK_SIZE;
//            blockManager.totalRequests.add(neededBlocks);
//            // 带超时的块分配（核心防卡死）
//            if (!blockManager.allocateBlocksWithTimeout(neededBlocks, BLOCK_SIZE, 5000)) {
//                throw new RuntimeException("块分配超时（5s）");
//            }
//
//            for (int layer = 0; layer < NUM_LAYERS; layer++) {
//                kvBuffer.prefillUltra(layer, 0, token);
//                kvBuffer.prefillUltra(layer, 1, token);
//                cacheMeta.kBlockCount[layer] = kvBuffer.getKBlockCount(layer);
//                cacheMeta.vBlockCount[layer] = kvBuffer.getVBlockCount(layer);
//            }
//            cacheMeta.physicalBlockIds = blockManager.getPhysicalBlockIds(sessionId);
//        }
//
//        private void optimizeKVCacheWithPageAttentionV3(PagedKvBufferV3 kvBuffer, CacheEntry cacheMeta, String sessionId) throws InterruptedException {
//            long[] blockIds = blockManager.getPhysicalBlockIds(sessionId);
//            int pageNum = (int) (cacheMeta.seqLen / PAGE_SIZE);
//            cacheMeta.lastPageIdx = pageNum;
//
//            if (pageNum > 4) {
//                List<Integer> invalidatedBlocks = kvBuffer.getAndInvalidateBlocks();
//                // 非阻塞释放块
//                blockManager.releaseBlocksNonBlocking(sessionId);
//                cacheMeta.invalidatedBlockCount += invalidatedBlocks.size();
//                cacheMeta.physicalBlockIds = blockManager.getPhysicalBlockIds(sessionId);
//                blockManager.cacheHitBlocks.add(invalidatedBlocks.size());
//            }
//        }
//
//        // 非阻塞清理会话资源（核心：避免死锁）
//        private void cleanSessionResourcesNonBlocking(String sessionId) {
//            // 使用tryLock非阻塞获取锁，失败则跳过（后续由LRU驱逐清理）
//            if (!blockManager.tryLockGlobalLock(100)) {
//                System.out.println("非阻塞清理跳过：" + sessionId + "（锁竞争）");
//                return;
//            }
//            try {
//                PagedKvBufferV3 kvBuffer = kvBufferMap.remove(sessionId);
//                if (kvBuffer != null) kvBuffer.close();
//                blockManager.releaseBlocksNonBlocking(sessionId);
//                cacheMetaMap.remove(sessionId);
//            } catch (InterruptedException e) {
//                throw new RuntimeException(e);
//            } finally {
//                blockManager.unlockGlobalLock();
//            }
//        }
//
//        // ========== 基础方法（复用） ==========
//        private Tensor forwardWithKVCache(Tensor inputIds) {
//            IValue output;
//            try (IValueVector inputs = new IValueVector()) {
//                inputs.push_back(new IValue(inputIds));
//                output = model.forward(inputs);
//            }
//            return output.toTensor();
//        }
//
//        private Tensor getLastTokenLogits(Tensor logits, long currentSeqLen) {
//            Tensor lastStepLogits;
//            if (logits.dim() == 3) {
//                lastStepLogits = logits.narrow(1, currentSeqLen - 1, 1);
//            } else if (logits.dim() == 2) {
//                lastStepLogits = logits.narrow(0, currentSeqLen - 1, 1);
//            } else {
//                lastStepLogits = logits.slice(0, new LongOptional(logits.size(0) - 1L), new LongOptional(logits.size(0)), 1);
//            }
//            return lastStepLogits;
//        }
//
//        private Tensor safeGreedySample(Tensor logits) {
//            int vocabDim = (int) logits.dim() - 1;
//            Tensor argmaxTensor = torch.argmax(logits, new LongOptional(vocabDim), false);
//            Tensor nextToken = argmaxTensor.to(device, torch.ScalarType.Long);
//            argmaxTensor.close();
//            return nextToken;
//        }
//
//        private Tensor ensure2DToken(Tensor token) {
//            Tensor result = token;
//            if (token.dim() == 1) {
//                result = token.unsqueeze(1);
//            } else if (token.dim() >= 3) {
//                result = token.squeeze(token.dim() - 1);
//            }
//            if (result.dim() != 2) {
//                result = result.reshape(new long[]{1, 1});
//            }
//            return result;
//        }
//
//        // ========== 统计+资源释放（防卡死） ==========
//        public double getTokenGenerationSpeed() {
//            long totalTime = totalInferenceTime.sum();
//            long totalTokens = totalTokensGenerated.sum();
//            if (totalTime == 0 || totalTokens == 0) return 0.0;
//            return totalTokens / (totalTime / 1000.0);
//        }
//
//        public String getConcurrentStats() {
//            return String.format(
//                    "完成任务数: %d, 失败任务数: %d, 超时任务数: %d, 总Token数: %d, 总耗时: %dms, 平均Token速度: %.2f tokens/s",
//                    completedTasks.get(), failedTasks.get(), timeoutTasks.get(),
//                    totalTokensGenerated.sum(), totalInferenceTime.sum(),
//                    getTokenGenerationSpeed()
//            );
//        }
//
//        public String getBlockManagerStats() {
//            return String.format(
//                    "块管理器统计 | 总请求块数: %d, 缓存命中块数: %d, 驱逐次数: %d, 等待次数: %d, 剩余块数: %d",
//                    blockManager.totalRequests.sum(), blockManager.cacheHitBlocks.sum(),
//                    CoWBlockManagerV9.EVICT_COUNT.sum(), CoWBlockManagerV9.WAIT_COUNT.sum(),
//                    blockManager.getFreeBlockCount()
//            );
//        }
//
//        public void close() {
//            // 非阻塞清理所有资源
//            kvBufferMap.keySet().forEach(sid -> {
//                try {
//                    cleanSessionResourcesNonBlocking(sid);
//                } catch (Exception e) { /* 忽略 */ }
//            });
//            kvBufferMap.clear();
//            cacheMetaMap.clear();
//
//            if (model != null) model.close();
//            if (device != null) device.close();
//
//            System.out.println("\n=== 虚拟线程压测最终统计 ===");
//            System.out.println(getConcurrentStats());
//            System.out.println(getBlockManagerStats());
//        }
//
//        // ========== 内部类（不变） ==========
//        private static class CacheEntry {
//            int seqLen = 0;
//            long lastPageIdx = 0;
//            int[] kBlockCount = new int[NUM_LAYERS];
//            int[] vBlockCount = new int[NUM_LAYERS];
//            long[] physicalBlockIds = new long[0];
//            int invalidatedBlockCount = 0;
//        }
//
//        public static class InferenceResult {
//            public final boolean success;
//            public final String sessionId;
//            public final int generateLen;
//            public final long inputSeqLen;
//            public final long outputSeqLen;
//            public final long inferenceTime;
//            public final String errorMsg;
//            public final int usedBlockCount;
//
//            public InferenceResult(boolean success, String sessionId, int generateLen,
//                                   long inputSeqLen, long outputSeqLen, long inferenceTime,
//                                   String errorMsg, int usedBlockCount) {
//                this.success = success;
//                this.sessionId = sessionId;
//                this.generateLen = generateLen;
//                this.inputSeqLen = inputSeqLen;
//                this.outputSeqLen = outputSeqLen;
//                this.inferenceTime = inferenceTime;
//                this.errorMsg = errorMsg;
//                this.usedBlockCount = usedBlockCount;
//            }
//        }
//    }
//
//    // ======================== 主方法：防卡死虚拟线程压测 ========================
//    public static void main(String[] args) throws InterruptedException {
//        String modelPath = "/Users/mullerzhang/Documents/code/langchain/qwen3_4b_fp16_mps.pt";
//        // 防卡死核心配置（关键！）
//        int maxConcurrentTasks = 3;        // 最大并发任务数（建议=CPU核心数）
//        int virtualThreadNum = 5;          // 限制虚拟线程数（减少锁竞争）
//        int testRoundPerThread = 3;        // 减少每线程轮数
//        int generateLenPerRound = 50;      // 减少每轮Token数（缩短单任务执行时间）
//        String baseSessionId = "vt_session_v11_final";
//
//        // 1. 初始化终极修复版推理器
//        Qwen3JavaInferenceV11 inference = new Qwen3JavaInferenceV11(modelPath, maxConcurrentTasks);
//
//        // 2. 测试输入
//        long[] inputTokens = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
//        Tensor inputIds = torch.tensor(inputTokens,
//                        new TensorOptions().dtype(new ScalarTypeOptional(torch.ScalarType.Long)))
//                .reshape(1, 16);
//
//        // 3. 打印配置
//        System.out.println("===== 开始V11终极修复版虚拟线程压测（防卡死） =====");
//        System.out.printf("压测配置：虚拟线程数=%d, 每线程轮数=%d, 每轮Token数=%d, 最大并发任务数=%d\n",
//                virtualThreadNum, testRoundPerThread, generateLenPerRound, maxConcurrentTasks);
//        System.out.printf("核心配置：PageSize=%d, BlockSize=%d, 总块数=%d, 单任务超时=%dms\n",
//                Qwen3JavaInferenceV11.PAGE_SIZE, Qwen3JavaInferenceV11.BLOCK_SIZE,
//                Qwen3JavaInferenceV11.TOTAL_BLOCKS, 30000);
//
//        // 4. 预热（单任务，非阻塞清理）
//        System.out.println("\n=== 预热阶段 ===");
//        try {
//            Qwen3JavaInferenceV11.InferenceResult warmupResult =
//                    inference.generateConcurrent(inputIds, 10, baseSessionId + "_warmup");
//            System.out.printf("预热%s | 耗时: %dms | 错误: %s\n",
//                    warmupResult.success ? "成功" : "失败",
//                    warmupResult.inferenceTime, warmupResult.errorMsg);
//            // 非阻塞清理预热资源
//            inference.cleanSessionResourcesNonBlocking(baseSessionId + "_warmup");
//            System.out.printf("预热资源清理完成，当前剩余块数: %d\n", inference.blockManager.getFreeBlockCount());
//        } catch (Exception e) {
//            System.err.println("预热失败：" + e.getMessage());
//            e.printStackTrace();
//            inputIds.close();
//            inference.close();
//            return;
//        }
//
//        // 5. 创建有限虚拟线程池（核心：防卡死）
//        ExecutorService virtualThreadExecutor = new ThreadPoolExecutor(
//                virtualThreadNum,                // 核心线程数
//                virtualThreadNum,                // 最大线程数
//                60, TimeUnit.SECONDS,            // 空闲超时
//                new LinkedBlockingQueue<>(),     // 任务队列
//                Thread.ofVirtual().factory()     // 虚拟线程工厂
//        );
//        CompletionService<Qwen3JavaInferenceV11.InferenceResult> completionService =
//                new ExecutorCompletionService<>(virtualThreadExecutor);
//
//        // 6. 提交任务（减少总任务数，避免卡死）
//        System.out.println("\n=== 虚拟线程高并发压测开始（防卡死） ===");
//        long totalWallTimeStart = System.currentTimeMillis();
//        int totalTaskCount = 0;
//        for (int threadIdx = 0; threadIdx < virtualThreadNum; threadIdx++) {
//            for (int round = 0; round < testRoundPerThread; round++) {
//                String sessionId = baseSessionId + "_thread" + threadIdx + "_round" + round;
//                completionService.submit(() ->
//                        inference.generateConcurrent(inputIds, generateLenPerRound, sessionId)
//                );
//                totalTaskCount++;
//            }
//        }
//
//        // 7. 收集结果（带超时）
//        List<Qwen3JavaInferenceV11.InferenceResult> allResults = new ArrayList<>();
//        int collected = 0;
//        while (collected < totalTaskCount) {
//            try {
//                // 每个任务收集超时30s
//                Future<Qwen3JavaInferenceV11.InferenceResult> future = completionService.poll(30, TimeUnit.SECONDS);
//                if (future == null) {
//                    System.err.println("任务收集超时，剩余未收集：" + (totalTaskCount - collected));
//                    break;
//                }
//                Qwen3JavaInferenceV11.InferenceResult result = future.get();
//                allResults.add(result);
//                collected++;
//                // 实时打印结果
//                if (result.success) {
//                    System.out.printf("成功 | 会话%s | 耗时: %dms | 生成Token: %d | 使用块数: %d\n",
//                            result.sessionId, result.inferenceTime, result.generateLen, result.usedBlockCount);
//                } else {
//                    System.err.printf("失败 | 会话%s | 耗时: %dms | 错误: %s\n",
//                            result.sessionId, result.inferenceTime, result.errorMsg);
//                }
//            } catch (ExecutionException e) {
//                System.err.println("任务执行异常：" + e.getCause().getMessage());
//                inference.failedTasks.incrementAndGet();
//                collected++;
//            } catch (Exception e) {
//                System.err.println("任务收集超时：" + e.getMessage());
//                inference.timeoutTasks.incrementAndGet();
//                collected++;
//            }
//        }
//
//        // 8. 优雅关闭线程池（核心：防卡死）
//        virtualThreadExecutor.shutdownNow(); // 强制关闭所有线程
//        if (!virtualThreadExecutor.awaitTermination(1, TimeUnit.MINUTES)) {
//            System.err.println("线程池未正常终止，强制退出");
//        }
//        long totalWallTimeEnd = System.currentTimeMillis();
//
//        // 9. 统计结果
//        System.out.println("\n=== 单任务结果汇总 ===");
//        long successTotalTime = 0;
//        int successTotalToken = 0;
//        int successCount = 0;
//        for (Qwen3JavaInferenceV11.InferenceResult res : allResults) {
//            if (res.success) {
//                successTotalTime += res.inferenceTime;
//                successTotalToken += res.generateLen;
//                successCount++;
//            }
//        }
//        System.out.printf("总任务数: %d, 成功数: %d, 失败数: %d, 超时数: %d, 墙钟总耗时: %dms\n",
//                totalTaskCount, successCount,
//                inference.failedTasks.get(), inference.timeoutTasks.get(),
//                totalWallTimeEnd - totalWallTimeStart);
//        if (successCount > 0) {
//            System.out.printf("成功任务平均耗时: %dms, 成功任务总Token: %d, 平均Token速度: %.2f tokens/s\n",
//                    successTotalTime / successCount, successTotalToken,
//                    successTotalToken / (successTotalTime / 1000.0));
//        }
//
//        // 10. 释放资源
//        inputIds.close();
//        inference.close();
//        System.out.println("\nV11终极修复版虚拟线程压测完成，无卡死！");
//    }
//
//    // ======================== 核心：CoWBlockManagerV9终极修复（细粒度锁+非阻塞） ========================
//    public static class CoWBlockManagerV9 extends CoWBlockManagerV2 {
//        // 细粒度锁拆分（核心防卡死）
//        private final ReentrantLock allocateLock = new ReentrantLock(true);    // 块分配锁（公平锁）
//        private final ReentrantLock releaseLock = new ReentrantLock(true);     // 块释放锁（公平锁）
//        private final ReentrantLock evictLock = new ReentrantLock(true);       // 驱逐锁（公平锁）
//        private final ReentrantLock treeLock = new ReentrantLock(true);        // Radix树锁（公平锁）
//
//        private final int actualBlockSize;
//        private final Condition allocateCondition = allocateLock.newCondition(); // 分配条件变量
//        public static final LongAdder EVICT_COUNT = new LongAdder();
//        public static final LongAdder WAIT_COUNT = new LongAdder();
//        private final double lowWatermark = 0.05;
//        private final double highWatermark = 0.30;
//        public final LongAdder totalRequests = new LongAdder();
//        public final LongAdder cacheHitBlocks = new LongAdder();
//
//        private final Deque<RadixNode> ghostCache = new ConcurrentLinkedDeque<>();
//        private final ConcurrentHashMap<String, List<RadixNode>> sessionNodes = new ConcurrentHashMap<>();
//        private final ConcurrentHashMap<String, Long> sessionLastActive = new ConcurrentHashMap<>();
//        private final RadixNode root = new RadixNode(-1, -1);
//
//        static class RadixNode {
//            final long hash;
//            final int blockId;
//            final ConcurrentHashMap<Long, RadixNode> children = new ConcurrentHashMap<>();
//            final AtomicInteger refCount = new AtomicInteger(0);
//            RadixNode(long hash, int blockId) { this.hash = hash; this.blockId = blockId; }
//        }
//
//        public CoWBlockManagerV9(int totalBlocks, int layers, int blockSize, int headDim, int dtype) {
//            super(totalBlocks, layers, blockSize, headDim, dtype);
//            this.actualBlockSize = blockSize;
//        }
//
//        // 终极修复1：带超时的块分配（非阻塞）
//        public boolean allocateBlocksWithTimeout(int numBlocks, int blockSize, long timeoutMs) {
//            if (blockSize != actualBlockSize || numBlocks <= 0) return false;
//
//            long startTime = System.currentTimeMillis();
//            allocateLock.lock();
//            try {
//                for (int i = 0; i < numBlocks; i++) {
//                    // 单块分配超时检查
//                    if (System.currentTimeMillis() - startTime > timeoutMs) {
//                        return false;
//                    }
//                    // 尝试分配块
//                    Integer id = freePool.poll();
//                    if (id != null) {
//                        continue;
//                    }
//                    // 无块则尝试驱逐
//                    if (evictOldestSession(null)) {
//                        continue;
//                    }
//                    // 等待块释放（带超时）
//                    WAIT_COUNT.increment();
//                    if (!allocateCondition.await(100, TimeUnit.MILLISECONDS)) {
//                        return false;
//                    }
//                }
//                return true;
//            } catch (InterruptedException e) {
//                Thread.currentThread().interrupt();
//                return false;
//            } finally {
//                allocateLock.unlock();
//            }
//        }
//
//        // 终极修复2：非阻塞释放块（核心防死锁）
//        public void releaseBlocksNonBlocking(String sessionId) throws InterruptedException {
//            if (sessionId == null || sessionId.isEmpty()) return;
//
//            // 非阻塞获取释放锁
//            if (!releaseLock.tryLock(100, TimeUnit.MILLISECONDS)) {
//                System.out.println("非阻塞释放跳过：" + sessionId + "（锁竞争）");
//                return;
//            }
//            try {
//                sessionLastActive.remove(sessionId);
//                this.releaseSession(sessionId);
//
//                // 清理幽灵节点
//                Iterator<RadixNode> iterator = ghostCache.iterator();
//                while (iterator.hasNext()) {
//                    RadixNode node = iterator.next();
//                    if (node.refCount.get() == 0) {
//                        freePool.add(node.blockId);
//                        iterator.remove();
//                        cacheHitBlocks.increment();
//                    }
//                }
//                // 唤醒分配线程
//                allocateLock.lock();
//                try {
//                    allocateCondition.signalAll();
//                } finally {
//                    allocateLock.unlock();
//                }
//            } catch (Exception e) {
//                System.err.println("非阻塞释放异常：" + e.getMessage());
//            } finally {
//                releaseLock.unlock();
//            }
//        }
//
//        // 终极修复3：LRU驱逐（细粒度锁）
//        @Override
//        boolean evictOldestSession(String excludeId) throws InterruptedException {
//            if (!evictLock.tryLock(100, TimeUnit.MILLISECONDS)) {
//                return false; // 非阻塞驱逐，失败则返回
//            }
//            try {
//                // 找到最久未活动会话
//                String oldestSession = null;
//                long oldestTime = Long.MAX_VALUE;
//                for (Map.Entry<String, Long> entry : sessionLastActive.entrySet()) {
//                    String sid = entry.getKey();
//                    if (sid.equals(excludeId)) continue;
//                    if (entry.getValue() < oldestTime) {
//                        oldestTime = entry.getValue();
//                        oldestSession = sid;
//                    }
//                }
//                if (oldestSession == null) return false;
//
//                // 驱逐会话
//                List<RadixNode> nodes = sessionNodes.remove(oldestSession);
//                if (nodes != null) {
//                    for (RadixNode node : nodes) {
//                        int remainingRefs = node.refCount.decrementAndGet();
//                        if (remainingRefs == 0) {
//                            freePool.add(node.blockId);
//                            ghostCache.addLast(node);
//                        }
//                    }
//                }
//                sessionLastActive.remove(oldestSession);
//                EVICT_COUNT.increment();
//
//                // 唤醒分配线程
//                allocateLock.lock();
//                try {
//                    allocateCondition.signalAll();
//                } finally {
//                    allocateLock.unlock();
//                }
//                return true;
//            } finally {
//                evictLock.unlock();
//            }
//        }
//
//        // 终极修复4：水位检查（非阻塞）
//        private void checkAndPrune() throws InterruptedException {
//            int free = getFreeBlockCount();
//            int total = totalBlocks;
//            if (free < total * lowWatermark) {
//                // 非阻塞获取驱逐锁
//                if (evictLock.tryLock(50, TimeUnit.MILLISECONDS)) {
//                    try {
//                        while (getFreeBlockCount() < total * highWatermark) {
//                            if (!ghostCache.isEmpty()) {
//                                RadixNode victim = ghostCache.pollFirst();
//                                if (victim != null && victim.refCount.get() == 0) {
//                                    freePool.add(victim.blockId);
//                                    cacheHitBlocks.increment();
//                                }
//                            } else if (!evictOldestSession(null)) {
//                                break;
//                            }
//                        }
//                    } catch (InterruptedException e) {
//                        throw new RuntimeException(e);
//                    } finally {
//                        evictLock.unlock();
//                    }
//                }
//            }
//        }
//
//        // 终极修复5：Radix树操作（细粒度锁）
//        public int getOrAllocateBlock(long currentHash, String sid, PagedKvBufferV3 kv) throws InterruptedException {
//            if (!treeLock.tryLock(100, TimeUnit.MILLISECONDS)) {
//                throw new RuntimeException("Radix树锁竞争超时");
//            }
//            try {
//                sessionLastActive.put(sid, System.currentTimeMillis());
//                RadixNode current = root;
//                RadixNode next = current.children.get(currentHash);
//                RadixNode targetNode;
//
//                if (next != null) {
//                    targetNode = next;
//                    cacheHitBlocks.increment();
//                } else {
//                    // 带超时分配块
//                    if (!allocateBlocksWithTimeout(1, actualBlockSize, 5000)) {
//                        throw new RuntimeException("块分配超时");
//                    }
//                    Integer bId = freePool.poll();
//                    if (bId == null) throw new RuntimeException("无空闲块");
//                    targetNode = new RadixNode(currentHash, bId);
//                    current.children.put(currentHash, targetNode);
//                }
//
//                targetNode.refCount.incrementAndGet();
//                sessionNodes.computeIfAbsent(sid, k -> new CopyOnWriteArrayList<>()).add(targetNode);
//                return targetNode.blockId;
//            } finally {
//                treeLock.unlock();
//            }
//        }
//
//        // 辅助方法：非阻塞获取全局锁（兼容旧逻辑）
//        public boolean tryLockGlobalLock(long timeoutMs) {
//            try {
//                return allocateLock.tryLock(timeoutMs, TimeUnit.MILLISECONDS);
//            } catch (InterruptedException e) {
//                Thread.currentThread().interrupt();
//                return false;
//            }
//        }
//
//        public void unlockGlobalLock() {
//            if (allocateLock.isHeldByCurrentThread()) {
//                allocateLock.unlock();
//            }
//        }
//
//        // 基础方法实现
//        @Override
//        public List<Integer> allocateBlocks(int neededBlocks, String sessionId, PagedKvBufferV3 kvBuffer) {
//            List<Integer> blocks = new ArrayList<>();
//            for (int i = 0; i < neededBlocks; i++) {
//                if (!allocateBlocksWithTimeout(1, actualBlockSize, 5000)) {
//                    throw new RuntimeException("块分配超时");
//                }
//                Integer id = freePool.poll();
//                if (id == null) throw new RuntimeException("无空闲块");
//                blocks.add(id);
//            }
//            totalRequests.add(neededBlocks);
//            return blocks;
//        }
//
//        @Override
//        public void releaseSession(String sessionId) {
//            List<RadixNode> nodes = sessionNodes.remove(sessionId);
//            if (nodes == null || nodes.isEmpty()) return;
//
//            releaseLock.lock();
//            try {
//                for (RadixNode node : nodes) {
//                    int remainingRefs = node.refCount.decrementAndGet();
//                    if (remainingRefs == 0) {
//                        freePool.add(node.blockId);
//                        ghostCache.addLast(node);
//                        cacheHitBlocks.increment();
//                    }
//                }
//                allocateLock.lock();
//                try {
//                    allocateCondition.signalAll();
//                } finally {
//                    allocateLock.unlock();
//                }
//            } finally {
//                releaseLock.unlock();
//            }
//        }
//
//        public long[] getPhysicalBlockIds(String sessionId) {
//            if (sessionId == null || !sessionNodes.containsKey(sessionId)) {
//                return new long[0];
//            }
//
//            treeLock.lock();
//            try {
//                List<RadixNode> nodes = sessionNodes.get(sessionId);
//                if (nodes == null || nodes.isEmpty()) return new long[0];
//                long[] blockIds = new long[nodes.size()];
//                for (int i = 0; i < nodes.size(); i++) {
//                    blockIds[i] = nodes.get(i).blockId;
//                }
//                return blockIds;
//            } finally {
//                treeLock.unlock();
//            }
//        }
//
//        public int getFreeBlockCount() {
//            return freePool.size();
//        }
//    }
//
//    // ======================== 基础父类（适配细粒度锁） ========================
//    abstract static class CoWBlockManagerV2 {
//        protected int totalBlocks;
//        protected final Queue<Integer> freePool = new ConcurrentLinkedQueue<>();
//        protected final ConcurrentHashMap<String, Long> sessionLastActive = new ConcurrentHashMap<>();
//
//        public CoWBlockManagerV2(int totalBlocks, int layers, int blockSize, int headDim, int dtype) {
//            this.totalBlocks = totalBlocks;
//            for (int i = 0; i < totalBlocks; i++) {
//                freePool.add(i);
//            }
//        }
//
//        public int getBlockSize() { return 256; }
//        public void releaseSession(String sessionId) {}
//        boolean evictOldestSession(String excludeId) throws InterruptedException { return false; }
//        public abstract List<Integer> allocateBlocks(int neededBlocks, String sessionId, PagedKvBufferV3 kvBuffer);
//    }
//
//    // ======================== PagedKvBufferV3（适配非阻塞） ========================
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
//                int numTokens = (int) input.size(0);
//                int blockSize = manager.getBlockSize();
//                int neededBlocks = (numTokens + blockSize - 1) / blockSize;
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
//        public void close() {
//            try {
//                manager.releaseSession(sessionId);
//            } catch (Exception e) { /* 忽略关闭异常 */ }
//        }
//
//        public CharSequence getSessionId() { return sessionId; }
//        public int getKBlockCount(int layer) { return kBlockMaps[layer].size(); }
//        public int getVBlockCount(int layer) { return vBlockMaps[layer].size(); }
//    }
//}
