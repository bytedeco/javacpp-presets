package samples.demo.kvcache;
import org.bytedeco.pytorch.jit.*;
import org.bytedeco.pytorch.jit.JitModule;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.global.torch;

import java.util.concurrent.ConcurrentHashMap;

public class Qwen3InferenceBenchmarkv5 {
    // 简化的推理实现类
    public static class Qwen3JavaInferenceV3 {
        private final JitModule model; // 加载好的Qwen模型
        private final Device device;   // CPU设备
        private final ConcurrentHashMap<String, CacheEntry> kvCacheMap; // KV缓存

        // 构造函数 - 初始化模型和设备
        public Qwen3JavaInferenceV3(String modelPath) {
            // 正确的设备初始化（枚举常量，不带括号）
            this.device = new Device(torch.kCPU());
            // 加载模型到CPU
            this.model = torch.load(modelPath, new DeviceOptional(this.device),false);
            this.model.eval(); // 设置为推理模式
            this.kvCacheMap = new ConcurrentHashMap<>();
        }

        /**
         * 最终修复版generate方法 - 动态适配维度，解决越界问题
         * @param inputIds 输入Token的张量 (shape: [1, seq_len])
         * @param generateLen 生成Token长度
         * @param sessionId 会话ID
         * @return 完整的输出Token张量 [batch_size, input_len + generate_len]
         */
        public Tensor generate(Tensor inputIds, int generateLen, String sessionId) {
            // 确保输入是2维LongTensor且在CPU上
            Tensor currentInput = inputIds.to(device,torch.ScalarType.Long).clone();

            // 强制保证输入是2维张量 [batch_size, seq_len]
            if (currentInput.dim() != 2) {
                throw new IllegalArgumentException("输入必须是2维张量 [batch_size, seq_len]，当前维度：" + currentInput.dim());
            }

            try {
                // 逐Token生成（自回归生成）
                for (int step = 0; step < generateLen; step++) {
                    // 1. 调用模型forward，仅传入input_ids参数
                    IValue output;
                    try (IValueVector inputs = new IValueVector()) {
                        inputs.push_back(new IValue(currentInput));
                        output = model.forward(inputs);
                    }

                    // 2. 解析模型输出（logits）并打印维度信息（调试用）
                    Tensor logitsTensor = output.toTensor();
                    System.out.printf("第%d步 - logits维度: %d, shape: [%d, %d]\n",
                            step+1, logitsTensor.dim(),
                            logitsTensor.size(0), logitsTensor.size(1));

                    // 3. 动态获取最后一个Token的logits（适配不同维度）
                    Tensor lastStepLogits = getLastTokenLogits(logitsTensor, currentInput.size(1));

                    // 4. 安全的贪心采样（动态适配维度）
                    Tensor nextToken = safeGreedySample(lastStepLogits);

                    // 5. 确保nextToken是2维 [batch_size, 1]，和currentInput维度匹配
                    nextToken = ensure2DToken(nextToken);

                    // 6. 安全拼接：仅当维度匹配时拼接
                    TensorVector catTensors = new TensorVector();
                    catTensors.push_back(currentInput);
                    catTensors.push_back(nextToken);
                    currentInput = torch.cat(catTensors, 1); // dim=1 按序列长度拼接

                    // 释放临时张量，避免内存泄漏
                    logitsTensor.close();
                    lastStepLogits.close();
                    nextToken.close();
                    catTensors.close();
                }

                // 返回完整的Token序列（原始输入 + 生成的Token）
                return currentInput;

            } catch (Exception e) {
                throw new RuntimeException("推理失败: 模型推理失败: " + e.getMessage(), e);
            }
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
                lastStepLogits = logits.slice(0, new LongOptional(logits.size(0) - 1l), new LongOptional(logits.size(0)),1);
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
            int vocabDim = (int)logits.dim() - 1; // vocab维度永远是最后一维
            System.out.println("采样维度 - logits维度: " + logits.dim() + ", vocab维度索引: " + vocabDim);

            // 对最后一维（vocab维度）做argmax，不保留维度
            argmaxTensor = torch.argmax(logits, new LongOptional(vocabDim), false);

            // 转换为Long类型
            Tensor nextToken = argmaxTensor.to(device,torch.ScalarType.Long);

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
                result = token.squeeze((token.dim() - 1));
            }

            // 最终兜底：确保是2维
            if (result.dim() != 2) {
                result = result.reshape(new long[]{1, 1}); // batch_size=1
            }

            System.out.println("Token维度转换 - 原始维度: " + token.dim() + ", 转换后维度: " + result.dim());
            return result;
        }

        // 释放资源
        public void close() {
            if (model != null) model.close();
            if (device != null) device.close();
            kvCacheMap.clear();
        }

        // KV缓存条目（简化定义）
        private static class CacheEntry {
            Tensor kCache;
            Tensor vCache;
            int seqLen;
        }
    }

    // 测试方法
    public static void main(String[] args) {
        // 初始化推理器
        String modelPath = "/Users/mullerzhang/Documents/code/langchain/qwen3_4b_fp16_mps.pt";

        Qwen3JavaInferenceV3 inference = new Qwen3JavaInferenceV3(modelPath);

        // 测试输入：16个Token的输入序列（匹配压测配置）
        long[] inputTokens = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
        // 显式指定类型为Long，避免类型不匹配
        Tensor inputIds = torch.tensor(inputTokens, new TensorOptions().dtype(new ScalarTypeOptional(torch.ScalarType.Long)))
                .reshape(1, 16);

        // 执行推理（生成10个Token，匹配压测配置）
        try {
            Tensor output = inference.generate(inputIds, 10, "test_session_0");
            System.out.println("\n===== 推理结果 =====");
            System.out.println("输入Token长度: " + inputIds.size(1));
            System.out.println("输出Token总长度: " + output.size(1));
            System.out.println("成功生成Token数: " + (output.size(1) - inputIds.size(1)));
            System.out.println("推理成功！");
        } catch (Exception e) {
            System.err.println("\n推理失败：" + e.getMessage());
            e.printStackTrace();
        } finally {
            // 释放所有张量资源
            inputIds.close();
            inference.close();
        }
    }


    private Tensor topKSample(Tensor logits, int k) {
        int vocabDim = (int)logits.dim() - 1;
        // 获取Top-K的value和indices
        T_TensorTensor_T topKResult = torch.topk(logits, k, vocabDim, true, true);
        Tensor topKLogits = topKResult.get0();
        Tensor topKIndices = topKResult.get1();

        // 计算softmax概率
        Tensor probs = torch.softmax(topKLogits,vocabDim);// new LongOptional(vocabDim));

        // 随机采样（需要引入随机数生成）
        Tensor randomIdx = torch.multinomial(probs, 1, true,new GeneratorOptional());
        Tensor nextToken = torch.gather(topKIndices, vocabDim, randomIdx);

        // 释放资源
        topKLogits.close();
        topKIndices.close();
        probs.close();
        randomIdx.close();

        return nextToken.to(torch.ScalarType.Long);
    }

}
