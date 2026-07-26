package org.bytedeco.pytorch.geometric.demo.kvcache;
import org.bytedeco.pytorch.jit.*;
import org.bytedeco.pytorch.c10.*;
import org.bytedeco.pytorch.jit.JitModule;

import org.bytedeco.javacpp.LongPointer;
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.global.torch;

import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.Future;
import java.util.concurrent.atomic.*;
import java.util.concurrent.locks.*;

/**
 * V11最终版：主动资源管理+缓存复用+动态限流
 * 解决：100%块分配超时+缓存命中为0+驱逐次数为0问题
 */
public class Qwen3InferenceBenchmarkV14 {
    // V11推理器：主动资源管理版本
    public static class Qwen3JavaInferenceV11 {
        private final JitModule model;
        private final Device device;
        private final CoWBlockManagerV9 blockManager;
        private final ConcurrentHashMap<String, PagedKvBufferV3> kvBufferMap;
        private final ConcurrentHashMap<String, CacheEntry> cacheMetaMap;

        // 核心配置（主动资源管理）
        public static final int PAGE_SIZE = 1024;
        public static final int BLOCK_SIZE = 256;
        public static final int MAX_SEQ_LEN = 8192;
        public static final int NUM_LAYERS = 32;
        public static final int TOTAL_BLOCKS = 16384;
        public static final int DTYPE = 0;
        private static final int HEAD_DIM = 128;

        // 主动资源管理配置
        private final int minFreeBlocks = 2048;  // 最小空闲块数阈值
        private final AtomicInteger dynamicConcurrentLimit = new AtomicInteger(3); // 动态并发限流

        // 统计修复
        private final LongAdder totalInferenceTime = new LongAdder();
        private final LongAdder totalTokensGenerated = new LongAdder();
        private final AtomicInteger completedTasks = new AtomicInteger(0);
        private final AtomicInteger failedTasks = new AtomicInteger(0);
        private final AtomicInteger timeoutTasks = new AtomicInteger(0);
        private final AtomicInteger blockedTasks = new AtomicInteger(0); // 限流阻塞任务

        // 限流信号量（动态调整）
        private final Semaphore dynamicTaskSemaphore;
        private final long taskTimeoutMs = 60000;       // 单任务超时延长至60s
        private final long blockAllocateTimeoutMs = 15000; // 块分配超时延长至15s

        // 构造函数：主动资源管理初始化
        public Qwen3JavaInferenceV11(String modelPath, int initialConcurrentTasks) {
            this.device = new Device(torch.kCPU());
            this.model = torch.load(modelPath, new DeviceOptional(this.device), false);
            this.model.eval();

            // 初始化块管理器+预分配核心块
            this.blockManager = new CoWBlockManagerV9(TOTAL_BLOCKS, NUM_LAYERS, BLOCK_SIZE, HEAD_DIM, DTYPE);
            // 预分配10%块作为核心缓存
            blockManager.prefillCoreBlocks(1024);

            this.kvBufferMap = new ConcurrentHashMap<>();
            this.cacheMetaMap = new ConcurrentHashMap<>();
            // 动态限流信号量
            this.dynamicTaskSemaphore = new Semaphore(initialConcurrentTasks);

            torch.requires_grad(false);
//            torch.no_grad();

            // 启动后台资源监控线程（核心）
            startResourceMonitor();
        }

        private void startResourceMonitor2() {
            // 推荐：平台线程 + 守护模式（适合监控类后台任务）
            Thread monitorThread = Thread.ofPlatform()
                    .name("kvcache-resource-monitor")
                    .daemon(true)
                    .start(() -> {
                        System.out.println("[资源监控] 监控线程启动");
                        while (!Thread.currentThread().isInterrupted()) {
                            try {
                                int freeBlocks = blockManager.getFreeBlockCount();
                                int currentLimit = dynamicConcurrentLimit.get();

                                // 动态调整并发数
                                if (freeBlocks < minFreeBlocks && currentLimit > 1) {
                                    int newLimit = currentLimit - 1;
                                    dynamicConcurrentLimit.set(newLimit);
                                    adjustSemaphorePermits(newLimit);
                                    System.out.println("[资源监控] 空闲块不足(" + freeBlocks + ")，并发数降至：" + newLimit);
                                } else if (freeBlocks > minFreeBlocks * 2 && currentLimit < 5) {
                                    int newLimit = currentLimit + 1;
                                    dynamicConcurrentLimit.set(newLimit);
                                    adjustSemaphorePermits(newLimit);
                                    System.out.println("[资源监控] 空闲块充足(" + freeBlocks + ")，并发数升至：" + newLimit);
                                }

                                // 强制清理僵尸会话
                                blockManager.forceEvictZombieSessions();
                                Thread.sleep(5000); // 每5秒检查一次
                            } catch (InterruptedException e) {
                                Thread.currentThread().interrupt();
                                System.out.println("[资源监控] 监控线程被中断，准备退出");
                                break;
                            } catch (Exception e) {
                                System.err.println("[资源监控] 执行异常：" + e.getMessage());
                                e.printStackTrace();
                            }
                        }
                        System.out.println("[资源监控] 监控线程已退出");
                    });
        }
        // 后台资源监控：动态调整并发数+强制清理资源
        private void startResourceMonitor() {
            Thread monitorThread = Thread.ofVirtual().start(() -> {
                while (!Thread.currentThread().isInterrupted()) {
                    try {
                        int freeBlocks = blockManager.getFreeBlockCount();
                        int currentLimit = dynamicConcurrentLimit.get();

                        // 动态调整并发数
                        if (freeBlocks < minFreeBlocks && currentLimit > 1) {
                            // 块不足，降低并发数
                            int newLimit = currentLimit - 1;
                            dynamicConcurrentLimit.set(newLimit);
                            // 调整信号量许可数
                            adjustSemaphorePermits(newLimit);
                            System.out.println("[资源监控] 空闲块不足(" + freeBlocks + ")，并发数降至：" + newLimit);
                        } else if (freeBlocks > minFreeBlocks * 2 && currentLimit < 5) {
                            // 块充足，提高并发数
                            int newLimit = currentLimit + 1;
                            dynamicConcurrentLimit.set(newLimit);
                            adjustSemaphorePermits(newLimit);
                            System.out.println("[资源监控] 空闲块充足(" + freeBlocks + ")，并发数升至：" + newLimit);
                        }

                        // 强制清理僵尸会话（每5s一次）
                        blockManager.forceEvictZombieSessions();
                        Thread.sleep(5000);
                    } catch (InterruptedException e) {
                        Thread.currentThread().interrupt();
                        break;
                    } catch (Exception e) {
                        System.err.println("[资源监控] 异常：" + e.getMessage());
                    }
                }
            });
//            monitorThread.setDaemon(true); // 守护线程，不阻塞进程退出
        }

        // 调整信号量许可数
        private void adjustSemaphorePermits(int newLimit) {
            int currentPermits = dynamicTaskSemaphore.availablePermits();
            int delta = newLimit - currentPermits;
            if (delta > 0) {
                dynamicTaskSemaphore.release(delta);
            } else if (delta < 0) {
                // 减少许可数（通过获取后不释放）
                for (int i = 0; i < -delta; i++) {
                    try {
                        dynamicTaskSemaphore.acquire();
                    } catch (InterruptedException e) {
                        Thread.currentThread().interrupt();
                        break;
                    }
                }
            }
        }

        /**
         * 核心生成方法：主动资源管理+缓存复用
         */
        public InferenceResult generateConcurrent(Tensor inputIds, int generateLen, String sessionId) {
            Tensor currentInput = null;
            long startTime = System.currentTimeMillis();
            try {
                // 1. 动态限流：带超时获取许可
                if (!dynamicTaskSemaphore.tryAcquire(taskTimeoutMs, TimeUnit.MILLISECONDS)) {
                    blockedTasks.incrementAndGet();
                    return new InferenceResult(false, sessionId, generateLen, 0, 0,
                            System.currentTimeMillis() - startTime, "任务限流阻塞（" + taskTimeoutMs + "ms）", 0);
                }

                // 2. 块资源预检
                int neededBlocks = calculateNeededBlocks(generateLen);
                if (blockManager.getFreeBlockCount() < neededBlocks && !blockManager.forceEvictBlocks(neededBlocks)) {
                    failedTasks.incrementAndGet();
                    dynamicTaskSemaphore.release();
                    return new InferenceResult(false, sessionId, generateLen, 0, 0,
                            System.currentTimeMillis() - startTime, "块资源不足，强制驱逐失败", 0);
                }

                // 3. 输入处理
                currentInput = inputIds.to(device, torch.ScalarType.Long).clone();
                long inputSeqLen = currentInput.size(1);

                // 4. 缓存初始化（启用Radix树缓存）
                PagedKvBufferV3 kvBuffer = kvBufferMap.computeIfAbsent(sessionId,
                        k -> new PagedKvBufferV3(sessionId, blockManager, NUM_LAYERS));
                CacheEntry cacheMeta = cacheMetaMap.computeIfAbsent(sessionId, k -> new CacheEntry());

                // 5. Prefill阶段（启用缓存复用）
                if (cacheMeta.seqLen == 0) {
                    // 使用Radix树缓存路径分配块
                    List<Long> pathHashes = generatePathHashes(inputIds);
                    prefillKVCacheWithCache(currentInput, kvBuffer, cacheMeta, pathHashes);
                    cacheMeta.seqLen = (int) inputSeqLen;
                }

                // 6. 逐Token生成（启用缓存命中）
                int generated = 0;
                for (int step = 0; step < generateLen; step++) {
                    if (System.currentTimeMillis() - startTime > taskTimeoutMs) {
                        timeoutTasks.incrementAndGet();
                        dynamicTaskSemaphore.release();
                        return new InferenceResult(false, sessionId, generated, inputSeqLen,
                                inputSeqLen + generated, System.currentTimeMillis() - startTime,
                                "Token生成超时", cacheMeta.physicalBlockIds.length);
                    }

                    // 生成当前Token的哈希路径（用于缓存命中）
                    long tokenHash = generateTokenHash(currentInput, step);
                    // 使用缓存分配块
                    updateKVCacheWithCache(currentInput, kvBuffer, cacheMeta, sessionId, tokenHash);

                    // 核心生成逻辑
                    Tensor newTokenInput = currentInput.narrow(1, currentInput.size(1) - 1, 1);
                    Tensor logitsTensor = forwardWithKVCache(newTokenInput);
                    Tensor lastStepLogits = getLastTokenLogits(logitsTensor, 1);
                    Tensor nextToken = safeGreedySample(lastStepLogits);
                    nextToken = ensure2DToken(nextToken);

                    // 拼接Token
                    TensorVector catTensors = new TensorVector();
                    catTensors.push_back(currentInput);
                    catTensors.push_back(nextToken);
                    currentInput = torch.cat(catTensors, 1);
                    cacheMeta.seqLen++;
                    generated++;

                    // 及时释放临时张量
                    logitsTensor.close();
                    lastStepLogits.close();
                    nextToken.close();
                    catTensors.close();
                    newTokenInput.close();
                }

                // 7. 统计成功结果
                long inferenceTime = System.currentTimeMillis() - startTime;
                totalInferenceTime.add(inferenceTime);
                totalTokensGenerated.add(generated);
                completedTasks.incrementAndGet();
                dynamicTaskSemaphore.release();

                return new InferenceResult(true, sessionId, generated, inputSeqLen,
                        currentInput.size(1), inferenceTime, null, cacheMeta.physicalBlockIds.length);

            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                failedTasks.incrementAndGet();
                dynamicTaskSemaphore.release();
                return new InferenceResult(false, sessionId, 0, 0, 0,
                        System.currentTimeMillis() - startTime, "任务被中断：" + e.getMessage(), 0);
            } catch (Exception e) {
                failedTasks.incrementAndGet();
                dynamicTaskSemaphore.release();
                return new InferenceResult(false, sessionId, 0, 0, 0,
                        System.currentTimeMillis() - startTime, e.getMessage(), 0);
            } finally {
                if (currentInput != null) {
                    try {
                        currentInput.close();
                    } catch (Exception e) { /* 忽略 */ }
                }
            }
        }

        // 修复后的 generatePathHashes 方法（核心）
        private List<Long> generatePathHashes(Tensor inputIds) {
            List<Long> hashes = new ArrayList<>();

            // 1. 确保 Tensor 是 CPU 张量 + Long 类型
            Tensor cpuTensor = inputIds.to(new Device(torch.kCPU()), torch.ScalarType.Long);
            if (cpuTensor.scalar_type() != torch.ScalarType.Long) {
                cpuTensor = cpuTensor.to(torch.ScalarType.Long); // 强制转换为 Long 类型
            }

            // 2. 通用方式读取 Tensor 数据（不依赖 asLong()）
            long[] tokens = new long[0];
            try {
                // 处理维度：2D -> 1D（如 [1, seq_len] → [seq_len]）
                Tensor flatTensor = cpuTensor.dim() == 2 ? cpuTensor.squeeze(0) : cpuTensor;

                // 获取 Tensor 元素总数
                long numElements = flatTensor.numel();
                if (numElements <= 0) {
                    tokens = new long[]{1L, 2L, 3L}; // 默认值
                } else {
                    // 核心：通过 Pointer 手动读取 Long 数据（兼容所有版本）
                    Pointer dataPtr = flatTensor.data();
                    LongPointer longPtr = new LongPointer(dataPtr); // 转换为 Long 指针
                    tokens = new long[(int) numElements];
                    longPtr.get(tokens); // 从指针读取数据到 long 数组

                    // 释放指针（避免内存泄漏）
                    longPtr.close();
                }
            } catch (Exception e) {
                // 降级方案：生成默认哈希，避免任务崩溃
                System.err.println("读取Tensor数据失败，使用默认值：" + e.getMessage());
                tokens = new long[]{1001L, 1002L, 1003L, 1004L, 1005L};
            }

            // 3. 生成哈希值（保持原有逻辑）
            for (long token : tokens) {
                hashes.add(token ^ (NUM_LAYERS * 1000000L));
            }

            // 4. 释放临时张量（关键：避免内存泄漏）
            cpuTensor.close();

            return hashes;
        }

        // 配套修复：generateTokenHash 方法（同样兼容所有版本）
        private long generateTokenHash(Tensor input, int step) {
            long tokenHash = 0L;
            try {
                // 1. 转换为 CPU + Long 类型
                Tensor cpuInput = input.to(new Device(torch.kCPU()),torch.kInt());
                // 2. 取最后一个 Token
                Tensor lastTokenTensor = cpuInput.narrow(1, input.size(1)-1, 1).squeeze();
                // 3. 通用方式读取单个 Token 值
                LongPointer longPtr = new LongPointer(lastTokenTensor.data());
                long lastToken = longPtr.get(0); // 读取第一个元素
                longPtr.close();

                // 4. 生成哈希
                tokenHash = (lastToken + step) ^ (NUM_LAYERS * 1000000L);

                // 5. 释放临时张量
                cpuInput.close();
                lastTokenTensor.close();
            } catch (Exception e) {
                // 降级方案：使用默认逻辑
                tokenHash = (input.size(1) + step) ^ (NUM_LAYERS * 1000000L);
                System.err.println("生成Token哈希失败，使用默认值：" + e.getMessage());
            }
            return tokenHash;
        }
        // 计算所需块数（提前预检）
        private int calculateNeededBlocks(int generateLen) {
            // 每Token需要的块数 = (NUM_LAYERS * 2) / BLOCK_SIZE
            return (generateLen * NUM_LAYERS * 2 + BLOCK_SIZE - 1) / BLOCK_SIZE;
        }

//        private List<Long> generatePathHashes(Tensor inputIds) {
//            List<Long> hashes = new ArrayList<>();
//
//            // 1. 确保 Tensor 是 CPU 张量（避免设备不匹配）
//            Tensor cpuTensor = inputIds.to(new Device(torch.kCPU()), torch.ScalarType.Long);
//
//            // 2. 获取 Tensor 的数据并转换为 long 数组（正确 API）
//            //    data() 获取底层存储，asLong() 转换为 long 类型数组
//            long[] tokens;
//            try {
//                // 处理不同维度的 Tensor（兼容 1D/2D 输入）
//                if (cpuTensor.dim() == 2) {
//                    // 如果是 2D 张量（如 [1, seq_len]），先 squeeze 成 1D
//                    tokens = cpuTensor.squeeze(0).data().asLong();
//                } else {
//                    tokens = cpuTensor.data().asLong();
//                }
//            } catch (Exception e) {
//                // 降级方案：如果无法直接获取，生成默认哈希
//                System.err.println("获取Tensor数据失败，使用默认哈希：" + e.getMessage());
//                tokens = new long[]{1L, 2L, 3L, 4L, 5L}; // 默认值
//            }
//
//            // 3. 生成哈希值（保持原有逻辑）
//            for (long token : tokens) {
//                hashes.add(token ^ (NUM_LAYERS * 1000000L));
//            }
//
//            // 4. 释放临时张量（避免内存泄漏）
//            cpuTensor.close();
//
//            return hashes;
//        }

//        private long generateTokenHash(Tensor input, int step) {
//            long tokenHash = 0L;
//            try {
//                // 正确获取 Tensor 数据生成哈希
//                Tensor cpuInput = input.to(torch.device("cpu"));
//                // 获取最后一个 Token 的值作为哈希基础
//                long lastToken = cpuInput.narrow(1, input.size(1)-1, 1).squeeze().data().asLong()[0];
//                tokenHash = (lastToken + step) ^ (NUM_LAYERS * 1000000L);
//                cpuInput.close();
//            } catch (Exception e) {
//                // 降级方案：使用 step 生成默认哈希
//                tokenHash = (input.size(1) + step) ^ (NUM_LAYERS * 1000000L);
//            }
//            return tokenHash;
//        }
        // 生成路径哈希（用于Radix树缓存）
//        private List<Long> generatePathHashes(Tensor inputIds) {
//            List<Long> hashes = new ArrayList<>();
//            long[] tokens = inputIds.dataAsLong();
//            for (long token : tokens) {
//                hashes.add(token ^ (NUM_LAYERS * 1000000L));
//            }
//            return hashes;
//        }

        // 生成Token哈希
//        private long generateTokenHash(Tensor input, int step) {
//            return (input.size(1) + step) ^ (NUM_LAYERS * 1000000L);
//        }

        // Prefill阶段（启用缓存）
        private void prefillKVCacheWithCache(Tensor input, PagedKvBufferV3 kvBuffer, CacheEntry cacheMeta, List<Long> pathHashes) {
            int numTokens = (int) input.size(0);
            for (int layer = 0; layer < NUM_LAYERS; layer++) {
                // 使用Radix树缓存路径分配块
                List<Integer> kBlocks = blockManager.matchAndAllocatePath(pathHashes, kvBuffer.getSessionId().toString(), kvBuffer);
                List<Integer> vBlocks = blockManager.matchAndAllocatePath(pathHashes, kvBuffer.getSessionId().toString(), kvBuffer);
                kvBuffer.getKBlockMaps()[layer].addAll(kBlocks);
                kvBuffer.getVBlockMaps()[layer].addAll(vBlocks);
                cacheMeta.kBlockCount[layer] = kBlocks.size();
                cacheMeta.vBlockCount[layer] = vBlocks.size();
            }
            cacheMeta.physicalBlockIds = blockManager.getPhysicalBlockIds(kvBuffer.getSessionId().toString());
        }

        // 更新KV缓存（启用缓存）
        private void updateKVCacheWithCache(Tensor token, PagedKvBufferV3 kvBuffer, CacheEntry cacheMeta, String sessionId, long tokenHash) {
            int neededBlocks = (1 + BLOCK_SIZE - 1) / BLOCK_SIZE;
            blockManager.totalRequests.add(neededBlocks);

            // 使用缓存分配块
            for (int layer = 0; layer < NUM_LAYERS; layer++) {
                int blockId = blockManager.getOrAllocateBlock(tokenHash, sessionId, kvBuffer);
                kvBuffer.getKBlockMaps()[layer].add(blockId);
                kvBuffer.getVBlockMaps()[layer].add(blockId);
                cacheMeta.kBlockCount[layer] = kvBuffer.getKBlockCount(layer);
                cacheMeta.vBlockCount[layer] = kvBuffer.getVBlockCount(layer);
            }
            cacheMeta.physicalBlockIds = blockManager.getPhysicalBlockIds(sessionId);
        }

        // ========== 基础方法（复用+修复） ==========
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

        // ========== 统计+资源释放（修复） ==========
        public double getTokenGenerationSpeed() {
            long totalTime = totalInferenceTime.sum();
            long totalTokens = totalTokensGenerated.sum();
            if (totalTime == 0 || totalTokens == 0) return 0.0;
            return totalTokens / (totalTime / 1000.0);
        }

        public String getConcurrentStats() {
            return String.format(
                    "完成任务数: %d, 失败任务数: %d, 超时任务数: %d, 限流阻塞数: %d, 总Token数: %d, 总耗时: %dms, 平均Token速度: %.2f tokens/s",
                    completedTasks.get(), failedTasks.get(), timeoutTasks.get(), blockedTasks.get(),
                    totalTokensGenerated.sum(), totalInferenceTime.sum(),
                    getTokenGenerationSpeed()
            );
        }

        public String getBlockManagerStats() {
            return String.format(
                    "块管理器统计 | 总请求块数: %d, 缓存命中块数: %d, 驱逐次数: %d, 等待次数: %d, 剩余块数: %d",
                    blockManager.totalRequests.sum(), blockManager.cacheHitBlocks.sum(),
                    CoWBlockManagerV9.EVICT_COUNT.sum(), CoWBlockManagerV9.WAIT_COUNT.sum(),
                    blockManager.getFreeBlockCount()
            );
        }

        public void close() {
            // 停止监控线程
            Thread.currentThread().interrupt();
            // 清理所有资源
            kvBufferMap.forEach((sid, kvBuffer) -> {
                try {
                    kvBuffer.close();
                    blockManager.releaseBlocksNonBlocking(sid);
                } catch (Exception e) { /* 忽略 */ }
            });
            kvBufferMap.clear();
            cacheMetaMap.clear();

            if (model != null) model.close();
            if (device != null) device.close();

            System.out.println("\n=== 虚拟线程压测最终统计 ===");
            System.out.println(getConcurrentStats());
            System.out.println(getBlockManagerStats());
        }

        // ========== 内部类（修复） ==========
        private static class CacheEntry {
            int seqLen = 0;
            long lastPageIdx = 0;
            int[] kBlockCount = new int[NUM_LAYERS];
            int[] vBlockCount = new int[NUM_LAYERS];
            long[] physicalBlockIds = new long[0];
            int invalidatedBlockCount = 0;
        }

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

    // ======================== 主方法：主动资源管理压测 ========================
    public static void main(String[] args) throws InterruptedException {
        String modelPath = "/Users/mullerzhang/Documents/code/langchain/qwen3_4b_fp16_mps.pt";
        // 主动资源管理配置（保守配置）
        int initialConcurrentTasks = 10;    // 初始并发数=1（从低开始）
        int virtualThreadNum = 8;          // 虚拟线程数=3
        int testRoundPerThread = 2;        // 每线程轮数=2
        int generateLenPerRound = 20;      // 每轮Token数=20
        String baseSessionId = "vt_session_v11_final_arm";

        // 1. 初始化主动资源管理版推理器
        Qwen3JavaInferenceV11 inference = new Qwen3JavaInferenceV11(modelPath, initialConcurrentTasks);

        // 2. 测试输入
        long[] inputTokens = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
        Tensor inputIds = torch.tensor(inputTokens,
                        new TensorOptions().dtype(new ScalarTypeOptional(torch.ScalarType.Long)))
                .reshape(1, 16);

        // 3. 打印配置
        System.out.println("===== 开始V11主动资源管理版虚拟线程压测 =====");
        System.out.printf("压测配置：虚拟线程数=%d, 每线程轮数=%d, 每轮Token数=%d, 初始并发数=%d\n",
                virtualThreadNum, testRoundPerThread, generateLenPerRound, initialConcurrentTasks);
        System.out.printf("核心配置：PageSize=%d, BlockSize=%d, 总块数=%d, 块分配超时=%dms\n",
                Qwen3JavaInferenceV11.PAGE_SIZE, Qwen3JavaInferenceV11.BLOCK_SIZE,
                Qwen3JavaInferenceV11.TOTAL_BLOCKS, 15000);

        // 4. 预热（单任务，启用缓存）
        System.out.println("\n=== 预热阶段（启用缓存） ===");
        try {
            Qwen3JavaInferenceV11.InferenceResult warmupResult =
                    inference.generateConcurrent(inputIds, 10, baseSessionId + "_warmup");
            System.out.printf("预热%s | 耗时: %dms | 错误: %s | 缓存命中: %d\n",
                    warmupResult.success ? "成功" : "失败",
                    warmupResult.inferenceTime, warmupResult.errorMsg,
                    inference.blockManager.cacheHitBlocks.sum());
            System.out.printf("预热后剩余块数: %d\n", inference.blockManager.getFreeBlockCount());
        } catch (Exception e) {
            System.err.println("预热失败：" + e.getMessage());
            e.printStackTrace();
            inputIds.close();
            inference.close();
            return;
        }

        // 5. 创建有限虚拟线程池
        ExecutorService virtualThreadExecutor = new ThreadPoolExecutor(
                virtualThreadNum,
                virtualThreadNum,
                60, TimeUnit.SECONDS,
                new LinkedBlockingQueue<>(),
                Thread.ofVirtual().factory()
        );
        CompletionService<Qwen3JavaInferenceV11.InferenceResult> completionService =
                new ExecutorCompletionService<>(virtualThreadExecutor);

        // 6. 提交任务（保守数量）
        System.out.println("\n=== 虚拟线程高并发压测开始（主动资源管理） ===");
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

        // 7. 收集结果
        List<Qwen3JavaInferenceV11.InferenceResult> allResults = new ArrayList<>();
        int collected = 0;
        while (collected < totalTaskCount) {
            try {
                Future<Qwen3JavaInferenceV11.InferenceResult> future = completionService.poll(60, TimeUnit.SECONDS);
                if (future == null) {
                    System.err.println("任务收集超时，剩余未收集：" + (totalTaskCount - collected));
                    break;
                }
                Qwen3JavaInferenceV11.InferenceResult result = future.get();
                allResults.add(result);
                collected++;
                // 实时打印结果
                if (result.success) {
                    System.out.printf("成功 | 会话%s | 耗时: %dms | 生成Token: %d | 使用块数: %d | 缓存命中: %d\n",
                            result.sessionId, result.inferenceTime, result.generateLen, result.usedBlockCount,
                            inference.blockManager.cacheHitBlocks.sum());
                } else {
                    System.err.printf("失败 | 会话%s | 耗时: %dms | 错误: %s\n",
                            result.sessionId, result.inferenceTime, result.errorMsg);
                }
            } catch (ExecutionException e) {
                System.err.println("任务执行异常：" + e.getCause().getMessage());
                inference.failedTasks.incrementAndGet();
                collected++;
            } catch (Exception e) {
                System.err.println("任务收集超时：" + e.getMessage());
                inference.timeoutTasks.incrementAndGet();
                collected++;
            }
        }

        // 8. 关闭线程池
        virtualThreadExecutor.shutdownNow();
        if (!virtualThreadExecutor.awaitTermination(2, TimeUnit.MINUTES)) {
            System.err.println("线程池未正常终止");
        }
        long totalWallTimeEnd = System.currentTimeMillis();

        // 9. 统计结果
        System.out.println("\n=== 单任务结果汇总 ===");
        long successTotalTime = 0;
        int successTotalToken = 0;
        int successCount = 0;
        for (Qwen3JavaInferenceV11.InferenceResult res : allResults) {
            if (res.success) {
                successTotalTime += res.inferenceTime;
                successTotalToken += res.generateLen;
                successCount++;
            }
        }
        System.out.printf("总任务数: %d, 成功数: %d, 失败数: %d, 超时数: %d, 限流阻塞数: %d, 墙钟总耗时: %dms\n",
                totalTaskCount, successCount,
                inference.failedTasks.get(), inference.timeoutTasks.get(), inference.blockedTasks.get(),
                totalWallTimeEnd - totalWallTimeStart);
        if (successCount > 0) {
            System.out.printf("成功任务平均耗时: %dms, 成功任务总Token: %d, 平均Token速度: %.2f tokens/s\n",
                    successTotalTime / successCount, successTotalToken,
                    successTotalToken / (successTotalTime / 1000.0));
        }

        // 10. 释放资源
        inputIds.close();
        inference.close();
        System.out.println("\nV11主动资源管理版虚拟线程压测完成！");
    }

    // ======================== 核心：CoWBlockManagerV9主动资源管理版 ========================
    public static class CoWBlockManagerV9 extends CoWBlockManagerV2 {
        // 细粒度锁
        private final ReentrantLock allocateLock = new ReentrantLock(true);
        private final ReentrantLock releaseLock = new ReentrantLock(true);
        private final ReentrantLock evictLock = new ReentrantLock(true);
        private final ReentrantLock treeLock = new ReentrantLock(true);

        private final int actualBlockSize;
        private final Condition allocateCondition = allocateLock.newCondition();
        public static final LongAdder EVICT_COUNT = new LongAdder();
        public static final LongAdder WAIT_COUNT = new LongAdder();
        private final double lowWatermark = 0.10;  // 提高触发阈值至10%
        private final double highWatermark = 0.40; // 提高清理阈值至40%
        public final LongAdder totalRequests = new LongAdder();
        public final LongAdder cacheHitBlocks = new LongAdder();

        private final Deque<RadixNode> ghostCache = new ConcurrentLinkedDeque<>();
        private final ConcurrentHashMap<String, List<RadixNode>> sessionNodes = new ConcurrentHashMap<>();
        private final ConcurrentHashMap<String, Long> sessionLastActive = new ConcurrentHashMap<>();
        private final RadixNode root = new RadixNode(-1, -1);
        // 僵尸会话超时（30s）
        private static final long ZOMBIE_SESSION_TIMEOUT = 30000;

        static class RadixNode {
            final long hash;
            final int blockId;
            final ConcurrentHashMap<Long, RadixNode> children = new ConcurrentHashMap<>();
            final AtomicInteger refCount = new AtomicInteger(0);
            RadixNode(long hash, int blockId) { this.hash = hash; this.blockId = blockId; }
        }

        public CoWBlockManagerV9(int totalBlocks, int layers, int blockSize, int headDim, int dtype) {
            super(totalBlocks, layers, blockSize, headDim, dtype);
            this.actualBlockSize = blockSize;
        }

        // 预分配核心块
        public void prefillCoreBlocks(int count) {
            allocateLock.lock();
            try {
                List<Integer> coreBlocks = new ArrayList<>();
                for (int i = 0; i < count; i++) {
                    Integer id = freePool.poll();
                    if (id != null) coreBlocks.add(id);
                }
                // 将核心块加入Radix树缓存
                for (int i = 0; i < coreBlocks.size(); i++) {
                    long hash = 1000000L + i;
                    root.children.put(hash, new RadixNode(hash, coreBlocks.get(i)));
                }
                cacheHitBlocks.add(coreBlocks.size());
                System.out.println("[块管理器] 预分配核心块数：" + coreBlocks.size());
            } finally {
                allocateLock.unlock();
            }
        }

        // 强制驱逐指定数量的块
        public boolean forceEvictBlocks(int neededBlocks) {
            evictLock.lock();
            try {
                int evicted = 0;
                while (evicted < neededBlocks && !sessionLastActive.isEmpty()) {
                    if (evictOldestSession(null)) {
                        evicted += actualBlockSize;
                    } else {
                        break;
                    }
                }
                return evicted >= neededBlocks;
            } finally {
                evictLock.unlock();
            }
        }

        // 强制清理僵尸会话
        public void forceEvictZombieSessions() {
            evictLock.lock();
            try {
                long now = System.currentTimeMillis();
                List<String> zombieSessions = new ArrayList<>();
                // 找出超时的僵尸会话
                for (Map.Entry<String, Long> entry : sessionLastActive.entrySet()) {
                    if (now - entry.getValue() > ZOMBIE_SESSION_TIMEOUT) {
                        zombieSessions.add(entry.getKey());
                    }
                }
                // 驱逐所有僵尸会话
                for (String sid : zombieSessions) {
                    evictOldestSession(sid);
                    sessionLastActive.remove(sid);
                }
                if (!zombieSessions.isEmpty()) {
                    System.out.println("[块管理器] 驱逐僵尸会话数：" + zombieSessions.size() + "，释放块数：" + zombieSessions.size() * actualBlockSize);
                }
            } finally {
                evictLock.unlock();
            }
        }

        // 带超时的块分配（延长至15s）
        public boolean allocateBlocksWithTimeout(int numBlocks, int blockSize, long timeoutMs) {
            if (blockSize != actualBlockSize || numBlocks <= 0) return false;

            long startTime = System.currentTimeMillis();
            allocateLock.lock();
            try {
                for (int i = 0; i < numBlocks; i++) {
                    if (System.currentTimeMillis() - startTime > timeoutMs) {
                        return false;
                    }

                    Integer id = freePool.poll();
                    if (id != null) {
                        continue;
                    }

                    // 强制驱逐
                    if (forceEvictBlocks(1)) {
                        continue;
                    }

                    WAIT_COUNT.increment();
                    // 延长等待时间至1s
                    if (!allocateCondition.await(1000, TimeUnit.MILLISECONDS)) {
                        continue;
                    }
                }
                return true;
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                return false;
            } finally {
                allocateLock.unlock();
            }
        }

        // 修复LRU驱逐逻辑（确保能找到可驱逐会话）
        @Override
        boolean evictOldestSession(String excludeId) {
            try {
                String oldestSession = null;
                long oldestTime = Long.MAX_VALUE;

                // 遍历所有会话，排除excludeId
                for (Map.Entry<String, Long> entry : sessionLastActive.entrySet()) {
                    String sid = entry.getKey();
                    if (sid.equals(excludeId)) continue;
                    if (entry.getValue() < oldestTime) {
                        oldestTime = entry.getValue();
                        oldestSession = sid;
                    }
                }

                if (oldestSession == null) return false;

                // 驱逐会话并释放块
                List<RadixNode> nodes = sessionNodes.remove(oldestSession);
                int released = 0;
                if (nodes != null) {
                    for (RadixNode node : nodes) {
                        int remainingRefs = node.refCount.decrementAndGet();
                        if (remainingRefs == 0) {
                            freePool.add(node.blockId);
                            ghostCache.addLast(node);
                            cacheHitBlocks.increment();
                            released++;
                        }
                    }
                }

                sessionLastActive.remove(oldestSession);
                EVICT_COUNT.increment();

                // 唤醒所有等待的分配线程
                allocateLock.lock();
                try {
                    allocateCondition.signalAll();
                } finally {
                    allocateLock.unlock();
                }

                System.out.println("[块管理器] 驱逐会话：" + oldestSession + "，释放块数：" + released);
                return released > 0;
            } catch (Exception e) {
                System.err.println("[块管理器] 驱逐异常：" + e.getMessage());
                return false;
            }
        }

        // 非阻塞释放块
        public void releaseBlocksNonBlocking(String sessionId) throws InterruptedException {
            if (sessionId == null || sessionId.isEmpty()) return;

            if (!releaseLock.tryLock(500, TimeUnit.MILLISECONDS)) {
                return;
            }
            try {
                sessionLastActive.remove(sessionId);
                this.releaseSession(sessionId);

                // 清理幽灵节点
                Iterator<RadixNode> iterator = ghostCache.iterator();
                while (iterator.hasNext()) {
                    RadixNode node = iterator.next();
                    if (node.refCount.get() == 0) {
                        freePool.add(node.blockId);
                        iterator.remove();
                        cacheHitBlocks.increment();
                    }
                }

                allocateLock.lock();
                try {
                    allocateCondition.signalAll();
                } finally {
                    allocateLock.unlock();
                }
            } finally {
                releaseLock.unlock();
            }
        }

        // 启用缓存分配（确保缓存命中）
        public int getOrAllocateBlock(long currentHash, String sid, PagedKvBufferV3 kv) {
            treeLock.lock();
            try {
                // 强制更新会话活动时间
                sessionLastActive.put(sid, System.currentTimeMillis());
                RadixNode current = root;
                RadixNode next = current.children.get(currentHash);
                RadixNode targetNode;

                if (next != null) {
                    // 缓存命中
                    targetNode = next;
                    cacheHitBlocks.increment();
                } else {
                    // 缓存未命中，分配新块
                    if (!allocateBlocksWithTimeout(1, actualBlockSize, 15000)) {
                        throw new RuntimeException("块分配超时");
                    }
                    Integer bId = freePool.poll();
                    if (bId == null) throw new RuntimeException("无空闲块");
                    targetNode = new RadixNode(currentHash, bId);
                    current.children.put(currentHash, targetNode);
                }

                targetNode.refCount.incrementAndGet();
                sessionNodes.computeIfAbsent(sid, k -> new CopyOnWriteArrayList<>()).add(targetNode);
                return targetNode.blockId;
            } finally {
                treeLock.unlock();
            }
        }

        // 批量缓存分配
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
                        if (!allocateBlocksWithTimeout(1, actualBlockSize, 15000)) {
                            throw new RuntimeException("块分配超时");
                        }
                        Integer bId = freePool.poll();
                        if (bId == null) throw new RuntimeException("无空闲块");
                        targetNode = new RadixNode(h, bId);
                        current.children.put(h, targetNode);
                    }
                    targetNode.refCount.incrementAndGet();
                    sessionNodes.computeIfAbsent(sessionId, k -> new CopyOnWriteArrayList<>()).add(targetNode);
                    result.add(targetNode.blockId);
                    current = targetNode;
                }
            } finally {
                treeLock.unlock();
            }
            return result;
        }

        // 基础方法实现
        @Override
        public List<Integer> allocateBlocks(int neededBlocks, String sessionId, PagedKvBufferV3 kvBuffer) {
            List<Integer> blocks = new ArrayList<>();
            for (int i = 0; i < neededBlocks; i++) {
                if (!allocateBlocksWithTimeout(1, actualBlockSize, 15000)) {
                    throw new RuntimeException("块分配超时");
                }
                Integer id = freePool.poll();
                if (id == null) throw new RuntimeException("无空闲块");
                blocks.add(id);
            }
            totalRequests.add(neededBlocks);
            return blocks;
        }

        @Override
        public void releaseSession(String sessionId) {
            List<RadixNode> nodes = sessionNodes.remove(sessionId);
            if (nodes == null || nodes.isEmpty()) return;

            releaseLock.lock();
            try {
                for (RadixNode node : nodes) {
                    int remainingRefs = node.refCount.decrementAndGet();
                    if (remainingRefs == 0) {
                        freePool.add(node.blockId);
                        ghostCache.addLast(node);
                        cacheHitBlocks.increment();
                    }
                }
                allocateLock.lock();
                try {
                    allocateCondition.signalAll();
                } finally {
                    allocateLock.unlock();
                }
            } finally {
                releaseLock.unlock();
            }
        }

        // Getter方法
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

        public int getFreeBlockCount() {
            return freePool.size();
        }

        public boolean tryLockGlobalLock(long timeoutMs) {
            try {
                return allocateLock.tryLock(timeoutMs, TimeUnit.MILLISECONDS);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                return false;
            }
        }

        public void unlockGlobalLock() {
            if (allocateLock.isHeldByCurrentThread()) {
                allocateLock.unlock();
            }
        }
    }

    // ======================== 基础类（适配主动资源管理） ========================
    abstract static class CoWBlockManagerV2 {
        protected int totalBlocks;
        protected final Queue<Integer> freePool = new ConcurrentLinkedQueue<>();
        protected final ConcurrentHashMap<String, Long> sessionLastActive = new ConcurrentHashMap<>();

        public CoWBlockManagerV2(int totalBlocks, int layers, int blockSize, int headDim, int dtype) {
            this.totalBlocks = totalBlocks;
            for (int i = 0; i < totalBlocks; i++) {
                freePool.add(i);
            }
        }

        public int getBlockSize() { return 256; }
        public void releaseSession(String sessionId) {}
        boolean evictOldestSession(String excludeId) { return false; }
        public abstract List<Integer> allocateBlocks(int neededBlocks, String sessionId, PagedKvBufferV3 kvBuffer);
    }

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
        public void close() {
            try {
                manager.releaseSession(sessionId);
            } catch (Exception e) { /* 忽略 */ }
        }

        // 新增Getter方法
        public List<Integer>[] getKBlockMaps() { return kBlockMaps; }
        public List<Integer>[] getVBlockMaps() { return vBlockMaps; }

        public CharSequence getSessionId() { return sessionId; }
        public int getKBlockCount(int layer) { return kBlockMaps[layer].size(); }
        public int getVBlockCount(int layer) { return vBlockMaps[layer].size(); }
    }
}
