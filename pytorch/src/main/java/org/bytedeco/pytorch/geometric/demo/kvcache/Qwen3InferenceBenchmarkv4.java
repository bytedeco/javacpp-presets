package org.bytedeco.pytorch.geometric.demo.kvcache;
import org.bytedeco.pytorch.jit.*;
import org.bytedeco.pytorch.jit.JitModule;


import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.global.torch;

import java.util.concurrent.ConcurrentHashMap;

public class Qwen3InferenceBenchmarkv4 {
    // 简化的推理实现类
    public static class Qwen3JavaInferenceV3 {
        private final JitModule model; // 加载好的Qwen模型
        private final Device device;   // CPU设备
        private final ConcurrentHashMap<String, CacheEntry> kvCacheMap; // KV缓存

        // 构造函数 - 初始化模型和设备
        public Qwen3JavaInferenceV3(String modelPath) {
            // 修复1：torch.kCPU是枚举常量，不带括号
            this.device = new Device(torch.kCPU());
            // 加载模型到CPU（简化写法）
            this.model = torch.load(modelPath, new DeviceOptional(this.device),false);
            this.model.eval(); // 设置为推理模式
            this.kvCacheMap = new ConcurrentHashMap<>();
        }

        /**
         * 修复后的generate方法 - 解决维度不匹配问题
         * @param inputIds 输入Token的张量 (shape: [1, seq_len])
         * @param generateLen 生成Token长度
         * @param sessionId 会话ID
         * @return 完整的输出Token张量 [batch_size, input_len + generate_len]
         */
        public Tensor generate(Tensor inputIds, int generateLen, String sessionId) {
            // 确保输入是2维LongTensor且在CPU上
            Tensor currentInput = inputIds.to(device,torch.ScalarType.Long).clone();
            // 强制保证是2维张量 [batch_size, seq_len]
            if (currentInput.dim() != 2) {
                throw new IllegalArgumentException("输入必须是2维张量 [batch_size, seq_len]");
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

                    // 2. 解析模型输出（logits），处理维度
                    Tensor logitsTensor = output.toTensor();

                    // 检查logits维度，确保是 [batch_size, seq_len, vocab_size]
                    if (logitsTensor.dim() != 3) {
                        throw new RuntimeException("模型输出logits必须是3维张量，当前维度：" + logitsTensor.dim());
                    }

                    // 3. 取最后一个Token的logits: [batch_size, 1, vocab_size]
                    Tensor lastStepLogits = logitsTensor.narrow(1, currentInput.size(1) - 1, 1);

                    // 4. 贪心采样（argmax）- 返回2维张量 [batch_size, 1]
                    Tensor nextToken = greedySample(lastStepLogits);

                    // 5. 关键修复：确保nextToken是2维，和currentInput维度一致
                    if (nextToken.dim() != 2) {
                        throw new RuntimeException("采样后的Token必须是2维张量，当前维度：" + nextToken.dim());
                    }

                    // 6. 拼接新生成的Token到输入中（维度匹配：2维 + 2维）
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
         * 修复后的贪心采样实现 - 确保返回2维张量 [batch_size, 1]
         * @param logits [batch_size, 1, vocab_size]
         * @return nextToken [batch_size, 1] (2维张量)
         */
        private Tensor greedySample(Tensor logits) {
            // 1. 对vocab维度（dim=2）做argmax，得到最大概率的Token索引
            // keepdim=False 避免生成多余维度
            Tensor argmaxTensor = torch.argmax(logits, new LongOptional(2), false);

            // 2. 关键修复：压缩维度，确保是2维 [batch_size, 1]
            // squeeze(2) 移除第2维（vocab维度），从3维变为2维
            Tensor nextToken = argmaxTensor.squeeze(2)
                    .to(device,torch.ScalarType.Long);

            // 3. 再次确保维度是2维（防止极端情况）
            if (nextToken.dim() == 1) {
                nextToken = nextToken.unsqueeze(1); // 变为 [batch_size, 1]
            }

            // 4. 释放临时张量
            argmaxTensor.close();

            return nextToken;
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
            System.out.println("推理成功！");
            System.out.println("输入Token长度: " + inputIds.size(1));
            System.out.println("输出Token总长度: " + output.size(1));
            System.out.println("生成的Token数: " + (output.size(1) - inputIds.size(1)));
        } catch (Exception e) {
            System.err.println("推理失败：" + e.getMessage());
            e.printStackTrace();
        } finally {
            // 释放所有张量资源
            inputIds.close();
            inference.close();
        }
    }
}
