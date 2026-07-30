package samples.demo.kvcache;
import org.bytedeco.pytorch.jit.*;
import org.bytedeco.pytorch.jit.JitModule;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.global.torch;

import java.util.concurrent.ConcurrentHashMap;

public class Qwen3InferenceBenchmarkv3 {
    // 简化的推理实现类
    public static class Qwen3JavaInferenceV3 {
        private final JitModule model; // 加载好的Qwen模型
        private final Device device;   // CPU设备
        private final ConcurrentHashMap<String, CacheEntry> kvCacheMap; // KV缓存

        // 构造函数 - 初始化模型和设备
        public Qwen3JavaInferenceV3(String modelPath) {
            // 加载模型到CPU
            this.device = new Device(torch.kCPU());
            this.model = torch.load(modelPath, new DeviceOptional(this.device), false);
            this.model.eval(); // 设置为推理模式
            this.kvCacheMap = new ConcurrentHashMap<>();
        }

        /**
         * 修复后的generate方法 - 只传入input_ids参数给forward
         * @param inputIds 输入Token的张量 (shape: [1, seq_len])
         * @param generateLen 生成Token长度
         * @param sessionId 会话ID
         * @return 生成的Token序列
         */
//        public Tensor generate(Tensor inputIds, int generateLen, String sessionId) {
//            try (IValueVector inputs = new IValueVector()) {
//                // 关键修复：只传入input_ids一个参数（self由模型自身隐式传入）
//                // forward方法实际只接收 input_ids 这一个显式参数
//                inputs.push_back(new IValue(inputIds.to(device, torch.ScalarType.Long)));
//
//                // 调用模型forward方法 - 仅传入input_ids参数
//                IValue output = model.forward(inputs);//.toTensor();
//
//                // 简单的采样逻辑（实际场景需替换为真实的采样/解码逻辑）
//                Tensor logits = output.toTensor();
//                Tensor generatedTokens = extractGeneratedTokens(logits, generateLen);
//
//                return generatedTokens;
//            } catch (Exception e) {
//                throw new RuntimeException("推理失败: " + e.getMessage(), e);
//            }
//        }

        // 辅助方法：从logits中提取生成的Token（简化实现）
//        private Tensor extractGeneratedTokens(Tensor logits, int generateLen) {
//            // 这里是简化的采样逻辑，实际需根据模型输出维度处理
//            long[] tokens = new long[generateLen];
//            for (int i = 0; i < generateLen; i++) {
//                // 示例：取最后一个Token的预测值（实际需用argmax/采样）
//                tokens[i] = logits.get(0, logits.size(1) - 1, 0).get();
//            }
//            return torch.tensor(tokens).reshape(1, generateLen);
//        }

        /**
         * 修复后的generate方法 - 包含正确的Token采样逻辑
         * @param inputIds 输入Token张量 [batch_size, seq_len]
         * @param generateLen 要生成的Token长度
         * @param sessionId 会话ID
         * @return 完整的输出Token张量 [batch_size, input_len + generate_len]
         */
        public Tensor generate(Tensor inputIds, int generateLen, String sessionId) {
            // 保存原始输入，用于拼接生成的Token
            Tensor currentInput = inputIds.to(device, torch.ScalarType.Long).clone();
            long batchSize = currentInput.size(0);

            try {
                // 逐Token生成（自回归生成）
                for (int step = 0; step < generateLen; step++) {
                    // 1. 调用模型forward，仅传入input_ids（修复参数不匹配问题）
                    IValue output;
                    try (IValueVector inputs = new IValueVector()) {
                        inputs.push_back(new IValue(currentInput));
                        output = model.forward(inputs);
                    }

                    // 2. 解析模型输出（logits），处理维度
                    Tensor logitsTensor = output.toTensor();

                    // 关键修复：logits维度是 [batch_size, seq_len, vocab_size]，取最后一个Token的logits
                    // 切片获取最后一个位置的logits: [batch_size, 1, vocab_size]
                    Tensor lastStepLogits = logitsTensor.narrow(1, currentInput.size(1) - 1, 1);

                    // 3. 贪心采样（argmax）- 实际场景可替换为top_k/top_p采样
                    Tensor nextToken = greedySample(lastStepLogits);

                    // 4. 拼接新生成的Token到输入中，用于下一轮推理
                    currentInput = torch.cat(new TensorVector(currentInput, nextToken), 1);

                    // 释放临时张量，避免内存泄漏
                    logitsTensor.close();
                    lastStepLogits.close();
                }

                // 返回完整的Token序列（原始输入 + 生成的Token）
                return currentInput.to(torch.ScalarType.Long);

            } catch (Exception e) {
                throw new RuntimeException("推理失败: 模型推理失败: " + e.getMessage(), e);
            }
        }

        /**
         * 贪心采样实现 - 从logits中取概率最大的Token
         * @param logits [batch_size, 1, vocab_size]
         * @return nextToken [batch_size, 1]
         */
        private Tensor greedySample(Tensor logits) {
            // 1. 对vocab维度做argmax，得到最大概率的Token索引
            // dim=2: vocab维度；keepdim=True保持维度，方便后续拼接
            Tensor argmaxTensor = torch.argmax(logits, new LongOptional(2), true);

            // 2. 转换为LongTensor并移动到CPU（避免设备不匹配）
            Tensor nextToken = argmaxTensor.to(device, torch.ScalarType.Long);

            // 3. 释放临时张量
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
        Tensor inputIds = torch.tensor(inputTokens).reshape(1, 16);

        // 执行推理（生成10个Token，匹配压测配置）
        try {
            Tensor output = inference.generate(inputIds, 10, "test_session_0");
            System.out.println("推理成功，生成Token数：" + output.size(1));
        } catch (Exception e) {
            System.err.println("推理失败：" + e.getMessage());
            e.printStackTrace();
        } finally {
            inference.close();
        }
    }
}