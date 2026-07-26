package org.bytedeco.pytorch.geometric.demo.kvcache;
import org.bytedeco.pytorch.jit.*;
import org.bytedeco.pytorch.jit.JitModule;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.global.torch;

import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.locks.ReentrantLock;

public class Qwen3InferenceBenchmarkV10 {
    // V10 版本推理器：深度集成 PagedKvBufferV3 + CoWBlockManagerV9
    public static class Qwen3JavaInferenceV10 {
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
        private static final int TOTAL_BLOCKS = 10240; // 总块数
        private static final int DTYPE = 0; // Float32

        // 压测统计
        private final AtomicLong totalInferenceTime = new AtomicLong(0);
        private final AtomicLong totalTokensGenerated = new AtomicLong(0);

        // 构造函数 - 初始化模型和V9块管理器
        public Qwen3JavaInferenceV10(String modelPath) {
            // 初始化设备
            this.device = new Device(torch.kCPU());

            // 加载模型到CPU
            this.model = torch.load(modelPath, new DeviceOptional(this.device), false);
            this.model.eval(); // 设置为推理模式

            // 初始化V9块管理器（核心组件）
            this.blockManager = new CoWBlockManagerV9(TOTAL_BLOCKS, NUM_LAYERS, BLOCK_SIZE, HEAD_DIM, DTYPE);

            // 初始化缓存映射
            this.kvBufferMap = new ConcurrentHashMap<>();
            this.cacheMetaMap = new ConcurrentHashMap<>();

            // 禁用梯度计算，提升推理性能
            torch.requires_grad(false);
//            torch.no_grad();
        }

        /**
         * V10核心generate方法：深度集成PagedKvBufferV3 + CoWBlockManagerV9
         * @param inputIds 输入Token的张量 (shape: [1, seq_len])
         * @param generateLen 生成Token长度
         * @param sessionId 会话ID（用于隔离KV Cache）
         * @return 完整的输出Token张量 [batch_size, input_len + generate_len]
         */
        public Tensor generate(Tensor inputIds, int generateLen, String sessionId) {
            // 确保输入是2维LongTensor且在CPU上
            Tensor currentInput = inputIds.to(device, torch.ScalarType.Long).clone();
            long batchSize = currentInput.size(0);
            long inputSeqLen = currentInput.size(1);

            // 强制保证输入是2维张量 [batch_size, seq_len]
            if (currentInput.dim() != 2) {
                throw new IllegalArgumentException("输入必须是2维张量 [batch_size, seq_len]，当前维度：" + currentInput.dim());
            }

            // 初始化PagedKvBufferV3和缓存元数据
            PagedKvBufferV3 kvBuffer = kvBufferMap.computeIfAbsent(sessionId,
                    k -> new PagedKvBufferV3(sessionId, blockManager, NUM_LAYERS));
            CacheEntry cacheMeta = cacheMetaMap.computeIfAbsent(sessionId, k -> new CacheEntry());

            try {
                // 记录压测开始时间
                long startTime = System.currentTimeMillis();

                // 第一步：Prefill阶段 - 初始化KV缓存（使用PagedKvBufferV3）
                if (cacheMeta.seqLen == 0) {
                    prefillKVCache(currentInput, kvBuffer, cacheMeta);
                    cacheMeta.seqLen = (int) inputSeqLen;
                }

                // 逐Token生成（自回归生成）- 使用V9块管理器优化缓存
                for (int step = 0; step < generateLen; step++) {
                    // 边界检查：不超过最大序列长度
                    if (cacheMeta.seqLen + inputSeqLen >= MAX_SEQ_LEN) {
                        throw new RuntimeException("序列长度超过最大值：" + MAX_SEQ_LEN);
                    }

                    // 外部KV Cache核心：只传入最后一个Token作为输入
                    Tensor newTokenInput = currentInput.narrow(1, currentInput.size(1) - 1, 1);

                    // 1. 调用模型forward（仅传入input_ids，适配模型接口）
                    Tensor logitsTensor = forwardWithKVCache(newTokenInput);

                    // 2. 动态获取最后一个Token的logits
                    Tensor lastStepLogits = getLastTokenLogits(logitsTensor, 1);

                    // 3. 安全的贪心采样
                    Tensor nextToken = safeGreedySample(lastStepLogits);

                    // 4. 确保nextToken是2维 [batch_size, 1]
                    nextToken = ensure2DToken(nextToken);

                    // 5. 使用V9块管理器分配新块并更新缓存
                    updateKVCacheWithV9Manager(nextToken, kvBuffer, cacheMeta, sessionId);

                    // 6. 拼接新生成的Token到输入中
                    TensorVector catTensors = new TensorVector();
                    catTensors.push_back(currentInput);
                    catTensors.push_back(nextToken);
                    currentInput = torch.cat(catTensors, 1);

                    // 更新缓存序列长度
                    cacheMeta.seqLen++;

                    // PageAttention V3：分页整理（基于块管理器的物理块）
                    if (cacheMeta.seqLen % PAGE_SIZE == 0 && cacheMeta.seqLen > 0) {
                        optimizeKVCacheWithPageAttentionV3(kvBuffer, cacheMeta, sessionId);
                    }

                    // 释放临时张量，避免内存泄漏
                    logitsTensor.close();
                    lastStepLogits.close();
                    nextToken.close();
                    catTensors.close();
                    newTokenInput.close();
                }

                // 压测统计
                long endTime = System.currentTimeMillis();
                totalInferenceTime.addAndGet(endTime - startTime);
                totalTokensGenerated.addAndGet(generateLen);

                // 返回完整的Token序列
                return currentInput;

            } catch (Exception e) {
                // 出错时清理该会话的所有缓存
                cleanSessionResources(sessionId);
                throw new RuntimeException("V10推理失败: " + e.getMessage(), e);
            }
        }

        /**
         * Prefill阶段：初始化KV缓存（使用PagedKvBufferV3）
         */
        private void prefillKVCache(Tensor input, PagedKvBufferV3 kvBuffer, CacheEntry cacheMeta) {
            int numTokens = (int) input.size(1);

            // 为每一层分配K/V块
            for (int layer = 0; layer < NUM_LAYERS; layer++) {
                // 分配K块
                kvBuffer.prefillUltra(layer, 0, input);
                // 分配V块
                kvBuffer.prefillUltra(layer, 1, input);

                // 更新缓存元数据
                cacheMeta.kBlockCount[layer] = kvBuffer.getKBlockCount(layer);
                cacheMeta.vBlockCount[layer] = kvBuffer.getVBlockCount(layer);
            }

            // 记录分配的物理块ID
            cacheMeta.physicalBlockIds = blockManager.getPhysicalBlockIds(kvBuffer.getSessionId().toString());
        }

        /**
         * 使用V9块管理器更新KV缓存
         */
        private void updateKVCacheWithV9Manager(Tensor token, PagedKvBufferV3 kvBuffer, CacheEntry cacheMeta, String sessionId) {
            // 为新Token分配块
            int neededBlocks = (1 + BLOCK_SIZE - 1) / BLOCK_SIZE; // 向上取整
            blockManager.allocateBlocks(neededBlocks, BLOCK_SIZE);

            // 更新每一层的缓存
            for (int layer = 0; layer < NUM_LAYERS; layer++) {
                kvBuffer.prefillUltra(layer, 0, token); // K块
                kvBuffer.prefillUltra(layer, 1, token); // V块

                // 更新块计数
                cacheMeta.kBlockCount[layer] = kvBuffer.getKBlockCount(layer);
                cacheMeta.vBlockCount[layer] = kvBuffer.getVBlockCount(layer);
            }

            // 更新物理块ID
            cacheMeta.physicalBlockIds = blockManager.getPhysicalBlockIds(sessionId);
        }

        /**
         * PageAttention V3实现：基于V9块管理器的分页优化
         */
        private void optimizeKVCacheWithPageAttentionV3(PagedKvBufferV3 kvBuffer, CacheEntry cacheMeta, String sessionId) {
            // 1. 获取当前会话的物理块
            long[] blockIds = blockManager.getPhysicalBlockIds(sessionId);

            // 2. 计算分页信息
            int pageNum = (int) (cacheMeta.seqLen / PAGE_SIZE);
            cacheMeta.lastPageIdx = pageNum;

            // 3. 清理过期块（保留最近3页）
            if (pageNum > 3) {
                List<Integer> invalidatedBlocks = kvBuffer.getAndInvalidateBlocks();
                // 释放过期块
                blockManager.releaseBlocks(sessionId);

                // 更新元数据
                cacheMeta.invalidatedBlockCount += invalidatedBlocks.size();
                cacheMeta.physicalBlockIds = blockManager.getPhysicalBlockIds(sessionId);
            }
        }

        /**
         * 适配模型接口的前向推理
         */
        private Tensor forwardWithKVCache(Tensor inputIds) {
            IValue output;
            try (IValueVector inputs = new IValueVector()) {
                // 仅传入input_ids一个参数（适配模型forward接口）
                inputs.push_back(new IValue(inputIds));

                // 执行前向推理
                output = model.forward(inputs);
            }

            // 返回logits
            return output.toTensor();
        }

        /**
         * 动态获取最后一个Token的logits
         */
        private Tensor getLastTokenLogits(Tensor logits, long currentSeqLen) {
            Tensor lastStepLogits;
            if (logits.dim() == 3) {
                // 标准LLM输出：[batch_size, seq_len, vocab_size]
                lastStepLogits = logits.narrow(1, currentSeqLen - 1, 1);
            } else if (logits.dim() == 2) {
                // 特殊情况：[batch_size * seq_len, vocab_size]
                lastStepLogits = logits.narrow(0, currentSeqLen - 1, 1);
            } else {
                // 只取最后一个元素
                lastStepLogits = logits.slice(0, new LongOptional(logits.size(0) - 1L), new LongOptional(logits.size(0)), 1);
            }
            return lastStepLogits;
        }

        /**
         * 安全的贪心采样
         */
        private Tensor safeGreedySample(Tensor logits) {
            Tensor argmaxTensor;

            // 动态判断vocab维度的索引
            int vocabDim = (int) logits.dim() - 1;

            // 对最后一维做argmax
            argmaxTensor = torch.argmax(logits, new LongOptional(vocabDim), false);

            // 转换为Long类型
            Tensor nextToken = argmaxTensor.to(device, torch.ScalarType.Long);

            // 释放临时张量
            argmaxTensor.close();

            return nextToken;
        }

        /**
         * 确保Token张量是2维 [batch_size, 1]
         */
        private Tensor ensure2DToken(Tensor token) {
            Tensor result = token;

            if (token.dim() == 1) {
                result = token.unsqueeze(1);
            } else if (token.dim() >= 3) {
                result = token.squeeze(token.dim() - 1);
            }

            // 最终兜底
            if (result.dim() != 2) {
                result = result.reshape(new long[]{1, 1});
            }

            return result;
        }

        /**
         * 清理会话所有资源
         */
        public void cleanSessionResources(String sessionId) {
            // 关闭PagedKvBuffer
            PagedKvBufferV3 kvBuffer = kvBufferMap.remove(sessionId);
            if (kvBuffer != null) {
                kvBuffer.close();
            }

            // 释放块管理器资源
            blockManager.releaseBlocks(sessionId);

            // 移除缓存元数据
            cacheMetaMap.remove(sessionId);
        }

        /**
         * 压测统计 - 获取平均Token生成速度
         */
        public double getTokenGenerationSpeed() {
            if (totalInferenceTime.get() == 0 || totalTokensGenerated.get() == 0) {
                return 0.0;
            }
            return totalTokensGenerated.get() / (totalInferenceTime.get() / 1000.0);
        }

        /**
         * 获取V9块管理器统计信息
         */
        public String getBlockManagerStats() {
            return String.format("总请求数: %d, 缓存命中块数: %d, 驱逐次数: %d, 等待次数: %d",
                    blockManager.totalRequests.sum(),
                    blockManager.cacheHitBlocks.sum(),
                    CoWBlockManagerV9.EVICT_COUNT.sum(),
                    CoWBlockManagerV9.WAIT_COUNT.sum());
        }

        /**
         * 清理所有资源
         */
        public void close() {
            // 清理所有会话
            kvBufferMap.keySet().forEach(this::cleanSessionResources);

            // 清理块管理器
            kvBufferMap.clear();
            cacheMetaMap.clear();

            // 释放模型和设备
            if (model != null) model.close();
            if (device != null) device.close();

            // 打印块管理器统计
            System.out.println("\n=== CoWBlockManagerV9 统计 ===");
            System.out.println(getBlockManagerStats());
        }

        // 缓存元数据（适配V9块管理器）
        private static class CacheEntry {
            int seqLen = 0;                      // 当前序列长度
            long lastPageIdx = 0;                // 最后一页索引
            int[] kBlockCount = new int[NUM_LAYERS]; // 每层K块数
            int[] vBlockCount = new int[NUM_LAYERS]; // 每层V块数
            long[] physicalBlockIds = new long[0];   // 物理块ID
            int invalidatedBlockCount = 0;        // 失效块数
        }
    }

    // 压测主方法
    public static void main(String[] args) {
        // 初始化V10推理器
        String modelPath = "/Users/mullerzhang/Documents/code/langchain/qwen3_4b_fp16_mps.pt";
        Qwen3JavaInferenceV10 inference = new Qwen3JavaInferenceV10(modelPath);

        // 压测配置
        int testRound = 5;          // 压测轮数
        int generateLen = 100;      // 每轮生成Token数
        String baseSessionId = "benchmark_session_v10";

        // 测试输入：16个Token的输入序列
        long[] inputTokens = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
        Tensor inputIds = torch.tensor(inputTokens,
                        new TensorOptions().dtype(new ScalarTypeOptional(torch.ScalarType.Long)))
                .reshape(1, 16);

        // 执行压测
        System.out.println("===== 开始压测（集成PagedKvBufferV3 + CoWBlockManagerV9） =====");
        System.out.printf("压测配置：轮数=%d, 每轮生成Token数=%d, PageSize=%d, BlockSize=%d\n",
                testRound, generateLen, Qwen3JavaInferenceV10.PAGE_SIZE, Qwen3JavaInferenceV10.BLOCK_SIZE);

        // 预热轮次
        System.out.println("\n=== 预热轮次 ===");
        try {
            Tensor warmupOutput = inference.generate(inputIds, 10, baseSessionId + "_warmup");
            warmupOutput.close();
            inference.cleanSessionResources(baseSessionId + "_warmup");
            System.out.println("预热完成");
        } catch (Exception e) {
            System.err.println("预热失败：" + e.getMessage());
            e.printStackTrace();
            return;
        }

        // 正式压测
        System.out.println("\n=== 正式压测 ===");
        for (int round = 0; round < testRound; round++) {
            String sessionId = baseSessionId + "_" + round;
            long roundStartTime = System.currentTimeMillis();

            try {
                Tensor output = inference.generate(inputIds, generateLen, sessionId);
                long roundEndTime = System.currentTimeMillis();

                System.out.printf("第%d轮压测完成 | 耗时: %dms | 生成Token数: %d | 输入长度: %d | 输出长度: %d\n",
                        round + 1,
                        roundEndTime - roundStartTime,
                        generateLen,
                        inputIds.size(1),
                        output.size(1));

                output.close();
                // 每轮结束清理资源
                inference.cleanSessionResources(sessionId);

            } catch (Exception e) {
                System.err.printf("第%d轮压测失败：%s\n", round + 1, e.getMessage());
                e.printStackTrace();
                break;
            }
        }

        // 输出压测统计
        System.out.println("\n===== 压测结果统计 =====");
        System.out.printf("平均Token生成速度: %.2f tokens/second\n", inference.getTokenGenerationSpeed());
        System.out.printf("总生成Token数: %d\n", inference.totalTokensGenerated.get());
        System.out.printf("总耗时: %dms\n", inference.totalInferenceTime.get());
        System.out.println("\n" + inference.getBlockManagerStats());

        // 释放资源
        inputIds.close();
        inference.close();
        System.out.println("\nV10压测完成，所有资源已释放！");
    }
}

// 父类CoWBlockManagerV2的基础实现（适配V9继承）
//abstract class CoWBlockManagerV2 {
//    protected int totalBlocks;
//    protected final java.util.Queue<Integer> freePool = new ConcurrentLinkedQueue<>();
//    private final ReentrantLock globalLock = new ReentrantLock();
//
//    public CoWBlockManagerV2(int totalBlocks, int layers, int blockSize, int headDim, int dtype) {
//        this.totalBlocks = totalBlocks;
//        // 初始化自由池
//        for (int i = 0; i < totalBlocks; i++) {
//            freePool.add(i);
//        }
//    }
//
//    public int getFreeBlockCount() {
//        return freePool.size();
//    }
//
//    public int getBlockSize() {
//        return 256; // 默认块大小
//    }
//
//    public ReentrantLock getGlobalLock() {
//        return globalLock;
//    }
//
//    public void releaseSession(String sessionId) {
//        // 基础实现，由子类V9重写
//    }
//
//    boolean evictOldestSession(String excludeId) {
//        // 基础驱逐逻辑，由子类V9重写
//        return false;
//    }
//}