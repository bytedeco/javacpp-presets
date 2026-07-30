package samples.demo.kvcache;
import org.bytedeco.pytorch.jit.*;
import org.bytedeco.pytorch.jit.JitModule;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.global.torch;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;

public class Qwen3InferenceBenchmarkv6 {
    // 简化的推理实现类
    public static class Qwen3JavaInferenceV3 {
        private final JitModule model; // 加载好的Qwen模型
        private final Device device;   // CPU设备
        private final ConcurrentHashMap<String, CacheEntry> kvCacheMap; // KV缓存

        // PageAttention配置（分页注意力）
        private static final int PAGE_SIZE = 1024; // 每页Token数
        private static final int MAX_SEQ_LEN = 8192; // 最大序列长度

        // 压测统计
        private final AtomicLong totalInferenceTime = new AtomicLong(0);
        private final AtomicLong totalTokensGenerated = new AtomicLong(0);

        // 构造函数 - 初始化模型和设备
        public Qwen3JavaInferenceV3(String modelPath) {
            // 正确的设备初始化（枚举常量，不带括号）
            this.device = new Device(torch.kCPU());
            // 加载模型到CPU（支持KV Cache的模型加载方式）
            this.model = torch.load(modelPath, new DeviceOptional(this.device),
                    false);
            this.model.eval(); // 设置为推理模式
            this.kvCacheMap = new ConcurrentHashMap<>();

            
            // 禁用梯度计算，提升推理性能
//            torch.no_grad();
        }

        /**
         * 集成KV Cache和PageAttention的generate方法 - 适配压测场景
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

            // 初始化/获取KV Cache
            CacheEntry cache = kvCacheMap.computeIfAbsent(sessionId, k -> new CacheEntry());

            try {
                // 记录压测开始时间
                long startTime = System.currentTimeMillis();

                // 第一步：处理初始输入，构建初始KV Cache
                if (cache.seqLen == 0) {
                    Tensor initialLogits = forwardWithKVCache(currentInput, cache, true);
                    initialLogits.close(); // 初始logits仅用于构建缓存，不参与采样
                    cache.seqLen = (int) inputSeqLen;
                }

                // 逐Token生成（自回归生成）- 仅传入新增Token，复用KV Cache
                for (int step = 0; step < generateLen; step++) {
                    // 只取最后一个Token作为输入（KV Cache优化核心）
                    Tensor newTokenInput = currentInput.narrow(1, currentInput.size(1) - 1, 1);

                    // 1. 调用模型forward，传入KV Cache
                    Tensor logitsTensor = forwardWithKVCache(newTokenInput, cache, false);

                    // 2. 动态获取最后一个Token的logits（适配不同维度）
                    Tensor lastStepLogits = getLastTokenLogits(logitsTensor, 1); // newTokenInput长度为1

                    // 3. 安全的贪心采样（动态适配维度）
                    Tensor nextToken = safeGreedySample(lastStepLogits);

                    // 4. 确保nextToken是2维 [batch_size, 1]，和currentInput维度匹配
                    nextToken = ensure2DToken(nextToken);

                    // 5. 拼接新生成的Token到输入中
                    TensorVector catTensors = new TensorVector();
                    catTensors.push_back(currentInput);
                    catTensors.push_back(nextToken);
                    currentInput = torch.cat(catTensors, 1);

                    // 更新缓存序列长度
                    cache.seqLen++;

                    // 分页注意力：当缓存超过页大小，进行分页整理（PageAttention核心）
                    if (cache.seqLen % PAGE_SIZE == 0 && cache.seqLen > 0) {
                        cache = optimizeKVCacheWithPageAttention(cache);
                        kvCacheMap.put(sessionId, cache);
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

                // 返回完整的Token序列（原始输入 + 生成的Token）
                return currentInput;

            } catch (Exception e) {
                // 出错时清理该会话的缓存
                kvCacheMap.remove(sessionId);
                throw new RuntimeException("推理失败: 模型推理失败: " + e.getMessage(), e);
            }
        }

        /**
         * 带KV Cache的前向推理 - 核心优化点
         * @param inputIds 输入Token（单Token或初始序列）
         * @param cache KV缓存
         * @param isInitial 是否是初始输入（首次推理）
         * @return logits张量
         */
        private Tensor forwardWithKVCache(Tensor inputIds, CacheEntry cache, boolean isInitial) {
            IValue output;
            try (IValueVector inputs = new IValueVector()) {
                // 基础输入：input_ids
                inputs.push_back(new IValue(inputIds));

                TensorOptions options = inputIds.options();
                // 添加KV Cache输入（适配Qwen模型的KV Cache接口）
                if (isInitial) {
                    // 初始推理：传入空缓存
                    inputs.push_back(new IValue(torch.empty(new long[]{0}, options,new MemoryFormatOptional()))); // k_cache
                    inputs.push_back(new IValue(torch.empty(new long[]{0}, options,new MemoryFormatOptional()))); // v_cache
                    inputs.push_back(new IValue(torch.tensor(0L))); // seq_len
                } else {
                    // 增量推理：传入已有缓存
                    inputs.push_back(new IValue(cache.kCache != null ? cache.kCache : torch.empty(new long[]{0}, options,new MemoryFormatOptional())));
                    inputs.push_back(new IValue(cache.vCache != null ? cache.vCache : torch.empty(new long[]{0}, options,new MemoryFormatOptional())));
                    inputs.push_back(new IValue(torch.tensor((long) cache.seqLen))); // 当前序列长度
                }

                // 执行前向推理
                output = model.forward(inputs);
            }

            // 解析输出：Qwen模型带KV Cache的输出格式 [logits, new_k_cache, new_v_cache]
            if (output.isTuple()) {
                IValueVector outputs = output.toTuple().elements().asIValueVector();
                Tensor logits = outputs.get(0).toTensor();

                // 更新KV Cache
                if (outputs.size() >= 3) {
                    if (cache.kCache != null) cache.kCache.close();
                    if (cache.vCache != null) cache.vCache.close();

                    cache.kCache = outputs.get(1).toTensor().to(device, torch.ScalarType.Float);
                    cache.vCache = outputs.get(2).toTensor().to(device, torch.ScalarType.Float);
                }

                return logits;
            } else {
                // 兼容无KV Cache输出的模型
                return output.toTensor();
            }
        }

        /**
         * PageAttention实现 - 分页管理KV Cache，优化长序列性能
         * @param cache 原始KV缓存
         * @return 分页优化后的缓存
         */
        private CacheEntry optimizeKVCacheWithPageAttention(CacheEntry cache) {
            if (cache.kCache == null || cache.vCache == null) {
                return cache;
            }

            // PageAttention核心：将连续的KV Cache拆分为固定大小的页
            // K/V Cache形状通常为 [num_layers, batch_size, num_heads, seq_len, head_dim]
            long[] kShape = cache.kCache.shape();
            long[] vShape = cache.vCache.shape();

            // 只对seq_len维度进行分页整理
            if (kShape.length >= 4 && vShape.length >= 4) {
                long seqLenDim = 3; // seq_len维度索引
                long currentSeqLen = kShape[(int) seqLenDim];

                // 计算页数
                long pageNum = currentSeqLen / PAGE_SIZE;

                // 对K Cache分页
                Tensor newKCache = torch.empty(
                        new long[]{kShape[0], kShape[1], kShape[2], pageNum, PAGE_SIZE, kShape[4]},
                        cache.kCache.options(),new MemoryFormatOptional()
                );

                // 对V Cache分页
                Tensor newVCache = torch.empty(
                        new long[]{vShape[0], vShape[1], vShape[2], pageNum, PAGE_SIZE, vShape[4]},
                        cache.vCache.options(),new MemoryFormatOptional()
                );

                // 填充分页数据
                for (long i = 0; i < pageNum; i++) {
                    long start = i * PAGE_SIZE;
                    Tensor kPage = cache.kCache.narrow((int) seqLenDim, start, PAGE_SIZE);
                    Tensor vPage = cache.vCache.narrow((int) seqLenDim, start, PAGE_SIZE);

                    newKCache.narrow(3, i, 1).copy_(kPage.unsqueeze(3));
                    newVCache.narrow(3, i, 1).copy_(vPage.unsqueeze(3));

                    kPage.close();
                    vPage.close();
                }

                // 释放旧缓存，替换为分页缓存
                cache.kCache.close();
                cache.vCache.close();
                cache.kCache = newKCache;
                cache.vCache = newVCache;
            }

            return cache;
        }

        /**
         * 动态获取最后一个Token的logits - 适配不同维度的模型输出
         * @param logits 模型输出的logits张量
         * @param currentSeqLen 当前输入序列长度
         * @return 最后一个Token的logits
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
         * 安全的贪心采样 - 动态适配张量维度，避免越界
         * @param logits 最后一个Token的logits
         * @return 采样后的Token张量
         */
        private Tensor safeGreedySample(Tensor logits) {
            Tensor argmaxTensor;

            // 动态判断vocab维度的索引
            int vocabDim = (int) logits.dim() - 1; // vocab维度永远是最后一维

            // 对最后一维（vocab维度）做argmax，不保留维度
            argmaxTensor = torch.argmax(logits, new LongOptional(vocabDim), false);

            // 转换为Long类型
            Tensor nextToken = argmaxTensor.to(device, torch.ScalarType.Long);

            // 释放临时张量
            argmaxTensor.close();

            return nextToken;
        }

        /**
         * 确保Token张量是2维 [batch_size, 1]
         * @param token 采样后的Token张量
         * @return 2维Token张量
         */
        private Tensor ensure2DToken(Tensor token) {
            Tensor result = token;

            if (token.dim() == 1) {
                // 1维 -> 2维: [batch_size] -> [batch_size, 1]
                result = token.unsqueeze(1);
            } else if (token.dim() >= 3) {
                // 3维及以上 -> 压缩为2维: [batch_size, 1, 1] -> [batch_size, 1]
                // 动态压缩最后一维，避免越界
                result = token.squeeze(token.dim() - 1);
            }

            // 最终兜底：确保是2维
            if (result.dim() != 2) {
                result = result.reshape(new long[]{1, 1}); // batch_size=1
            }

            return result;
        }

        /**
         * 压测统计 - 获取平均Token生成速度 (tokens/second)
         */
        public double getTokenGenerationSpeed() {
            if (totalInferenceTime.get() == 0 || totalTokensGenerated.get() == 0) {
                return 0.0;
            }
            return totalTokensGenerated.get() / (totalInferenceTime.get() / 1000.0);
        }

        /**
         * 清理指定会话的KV Cache
         */
        public void clearKVCache(String sessionId) {
            CacheEntry cache = kvCacheMap.remove(sessionId);
            if (cache != null) {
                if (cache.kCache != null) cache.kCache.close();
                if (cache.vCache != null) cache.vCache.close();
            }
        }

        /**
         * 清理所有KV Cache
         */
        public void clearAllKVCache() {
            for (Map.Entry<String, CacheEntry> entry : kvCacheMap.entrySet()) {
                CacheEntry cache = entry.getValue();
                if (cache.kCache != null) cache.kCache.close();
                if (cache.vCache != null) cache.vCache.close();
            }
            kvCacheMap.clear();
        }

        // 释放资源
        public void close() {
            clearAllKVCache();
            if (model != null) model.close();
            if (device != null) device.close();
        }

        // KV缓存条目（增强版，支持PageAttention）
        private static class CacheEntry {
            Tensor kCache;       // Key Cache
            Tensor vCache;       // Value Cache
            int seqLen;          // 当前序列长度
            long lastPageIdx = 0;// PageAttention：最后一页索引
        }
    }

    // 压测主方法
    public static void main(String[] args) {
        // 初始化推理器
        String modelPath = "/Users/mullerzhang/Documents/code/langchain/qwen3_4b_fp16_mps.pt";
        Qwen3JavaInferenceV3 inference = new Qwen3JavaInferenceV3(modelPath);

        // 压测配置
        int testRound = 5;          // 压测轮数
        int generateLen = 100;      // 每轮生成Token数
        String sessionId = "benchmark_session_001";

        // 测试输入：16个Token的输入序列（匹配压测配置）
        long[] inputTokens = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
        Tensor inputIds = torch.tensor(inputTokens, new TensorOptions().dtype(new ScalarTypeOptional(torch.ScalarType.Long)))
                .reshape(1, 16);

        // 执行压测
        System.out.println("===== 开始压测（集成KV Cache + PageAttention） =====");
        System.out.printf("压测配置：轮数=%d, 每轮生成Token数=%d\n", testRound, generateLen);

        for (int round = 0; round < testRound; round++) {
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
                // 每轮结束清理缓存（模拟不同会话）
                inference.clearKVCache(sessionId);

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

        // 释放资源
        inputIds.close();
        inference.close();
        System.out.println("\n压测完成，资源已释放！");
    }

    /**
     * Top-K采样（可选，替代贪心采样）
     */
    private Tensor topKSample(Tensor logits, int k) {
        int vocabDim = (int) logits.dim() - 1;
        // 获取Top-K的value和indices
        T_TensorTensor_T topKResult = torch.topk(logits, k, vocabDim, true, true);
        Tensor topKLogits = topKResult.get0();
        Tensor topKIndices = topKResult.get1();

        // 计算softmax概率
        Tensor probs = torch.softmax(topKLogits, vocabDim);

        // 随机采样
        Tensor randomIdx = torch.multinomial(probs, 1, true, new GeneratorOptional());
        Tensor nextToken = torch.gather(topKIndices, vocabDim, randomIdx);

        // 释放资源
        topKLogits.close();
        topKIndices.close();
        probs.close();
        randomIdx.close();

        return nextToken.to(torch.ScalarType.Long);
    }
}
