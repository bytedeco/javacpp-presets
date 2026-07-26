package org.bytedeco.pytorch.geometric.demo.profiler;

import org.bytedeco.pytorch.*;
import static org.bytedeco.pytorch.global.torch.*;
import java.util.concurrent.atomic.AtomicLong;

/**
 * 这是一个手动显存/内存追踪器
 * 适用于 Mac/Linux/Windows，不受 Profiler 限制
 */
public class MiniMindMemoryTracker {
    private static final AtomicLong currentAllocated = new AtomicLong(0);
    private static final AtomicLong peakAllocated = new AtomicLong(0);

    // 记录一个 Tensor 产生的内存增加
    public static void track(Tensor t, String label) {
        if (t == null) return;
        long bytes = t.numel() * t.element_size();
        long current = currentAllocated.addAndGet(bytes);
        if (current > peakAllocated.get()) {
            peakAllocated.set(current);
        }
        System.out.printf("[Memory] %-20s | 增量: %8.2f MB | 当前占用: %8.2f MB%n",
                label, bytes / 1024.0 / 1024.0, current / 1024.0 / 1024.0);
    }

    // 释放记录（当 Tensor 被销毁时手动调用，或在推理结束时重置）
    public static void reset() {
        currentAllocated.set(0);
        peakAllocated.set(0);
    }

    public static void main(String[] args) {
        System.out.println("🚀 开始 MiniMind 手动显存分析...");
        reset();

        try {
            TensorOptions options = new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float));

            // 1. 模拟加载权重
            Tensor weight = randn(new long[]{4096, 4096}, options);
            track(weight, "Model Weights");

            // 2. 模拟输入数据
            Tensor input = randn(new long[]{1024, 4096}, options);
            track(input, "Input Batch");

            // 3. 模拟中间层计算 (例如 MatMul)
            Tensor output = mm(input, weight);
            track(output, "Hidden States (MM)");

            // 4. 模拟激活函数 (通常是 In-place 操作，不增加内存)
            relu_(output);
            track(output, "Post ReLU (In-place)");

            System.out.println("-------------------------------------------");
            System.out.printf("🏁 推理完成。峰值显存占用: %.2f MB%n", peakAllocated.get() / 1024.0 / 1024.0);

        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}



//
//import org.bytedeco.pytorch.*;
//        import java.util.HashMap;
//import java.util.Map;
//
//public class MiniMindAnalyzer {
//
//    /**
//     * 自动统计 Module 的所有参数 (Weights)
//     */
//    public static void analyzeParameters(Layer layer) {
//        long totalParams = 0;
//        long totalBytes = 0;
//
//        // 获取所有参数（包括 weight 和 bias）
//        StringTensorDict params = layer.parameters();
//        StringVector keys = params.keys();
//
//        System.out.println("📂 --- 模型权重分布 ---");
//        for (int i = 0; i < keys.size(); i++) {
//            String key = keys.get(i).getString();
//            Tensor p = params.get(keys.get(i));
//            long numel = p.numel();
//            long bytes = numel * p.element_size();
//
//            totalParams += numel;
//            totalBytes += bytes;
//
//            System.out.printf("Layer: %-30s | Params: %10d | Size: %8.2f KB%n",
//                    key, numel, bytes / 1024.0);
//        }
//
//        System.out.println("-------------------------------------------");
//        System.out.printf("📊 总参数量: %.2f M%n", totalParams / 1000000.0);
//        System.out.printf("💾 权重占用: %.2f MB%n", totalBytes / 1024.0 / 1024.0);
//    }
//
//    /**
//     * 预估 MiniMind 推理峰值 (Activation + KV Cache)
//     */
//    public static void estimateInferencePeak(MiniMindConfig config, int batchSize, int seqLen) {
//        // 1. 激活层峰值 (简单粗暴估算法：最宽的一层 Tensor)
//        // [batch, seq, hidden_size] * float32
//        long activationSize = (long) batchSize * seqLen * config.hidden_size * 4;
//
//        // 2. KV Cache 占用 (KV Cache 会随 seqLen 增长)
//        // 每层: 2 (K和V) * batch * heads * head_dim * seqLen * float32
//        int headDim = config.hidden_size / config.num_attention_heads;
//        long kvCachePerLayer = 2L * batchSize * config.num_key_value_heads * headDim * seqLen * 4;
//        long totalKVCache = kvCachePerLayer * config.num_hidden_layers;
//
//        System.out.println("\n💡 --- 推理显存预估 (Batch=" + batchSize + ", Seq=" + seqLen + ") ---");
//        System.out.printf("单次激活占用: %8.2f MB%n", activationSize / 1024.0 / 1024.0);
//        System.out.printf("KV Cache 总量: %8.2f MB%n", totalKVCache / 1024.0 / 1024.0);
//        System.out.printf("🏁 预计运行时总显存: > %.2f MB%n", (activationSize + totalKVCache) / 1024.0 / 1024.0);
//    }
//}