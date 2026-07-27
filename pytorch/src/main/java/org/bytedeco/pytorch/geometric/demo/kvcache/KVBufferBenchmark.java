import org.bytedeco.pytorch.data.*;
import org.bytedeco.pytorch.jit.*;

//package org.bytedeco.pytorch.geometric.demo.kvcache;
//
//import org.bytedeco.javacpp.LongPointer;
//import org.bytedeco.pytorch.*;
//
//import static org.bytedeco.cuda.global.cudart.cudaDeviceSynchronize;
//import static org.bytedeco.pytorch.global.torch.*;
//
//public class KVBufferBenchmark {
//    // 模拟参数
//    static int HEAD_DIM = 128;
//    static int NUM_HEADS = 16;
//    static int BLOCK_SIZE = 16; // Paged 特有
//
//    public static void main(String[] args) {
//        int[] batchSizes = {4, 8, 16, 32, 64, 128};
//
//        System.out.println("BatchSize | Naive TPS | Paged TPS | Memory Saved");
//        System.out.println("------------------------------------------------");
//
//        for (int batchSize : batchSizes) {
//            // 1. 测试标准连续 KV Cache (Naive)
//            double naiveTps = runNaiveBenchmark(batchSize);
//
//            // 2. 测试 PagedKVBuffer (你实现的版本)
//            double pagedTps = runPagedBenchmark(batchSize);
//
//            System.out.printf("%10d | %9.2f | %9.2f | %s\n",
//                    batchSize, naiveTps, pagedTps, calculateMemoryDiff(batchSize));
//        }
//    }
//
//    // 模拟标准 PyTorch 行为：预分配连续大张量
//    static double runNaiveBenchmark(int batchSize) {
//        try {
//            // 模拟一次性分配最大长度的连续显存 (例如 SeqLen=2048)
//            Tensor naiveK = empty(new long[]{batchSize, NUM_HEADS, 2048, HEAD_DIM}, float16().device(kCUDA));
//            // 开始计时推理...
//            return simulateInference(batchSize);
//        } catch (Exception e) {
//            return 0.0; // 代表 OOM
//        }
//    }
//
//    // 模拟 Paged 行为：按需分配物理块
//    static double runPagedBenchmark(int batchSize) {
//        // 这里调用你实现的 PagedKvBuffer
//        // PagedKvBuffer buffer = new PagedKvBuffer(maxBlocks, BLOCK_SIZE, ...);
//        // 它允许更高的 BatchSize 而不崩溃
//        return simulateInference(batchSize * 1.5); // 模拟 Paged 能承载更多并发
//    }
//
//    /**
//     * 模拟 Decode 阶段的推理循环
//     * @param batchSize 当前并发用户数
//     * @param iterations 迭代次数（模拟生成多少个 Token）
//     * @return 吞吐量 (tokens/sec)
//     */
//    static double simulateInference(int batchSize, int iterations) {
//        if (batchSize <= 0) return 0;
//        TensorOptions options =new TensorOptions().dtype(new ScalarTypeOptional(kFloat())).device(new DeviceOptional(new Device(kMPS())));
//
//        // 构造输入张量 (Batch, 1, HiddenDim)
//        Device device = new Device(kCUDA());
//        Tensor input = randn(new long[]{batchSize, 1, 1024}, options);
//
//        // 预热 (Warmup)
//        for (int i = 0; i < 5; i++) {
//            // 这里可以调用你的注意力算子或者简单的 matmul 模拟
//            input.matmul(input.transpose(-1, -2));
//        }
//        cudaDeviceSynchronize();
//
//        long start = System.nanoTime();
//
//        // 核心压测循环
//        for (int i = 0; i < iterations; i++) {
//            // 模拟注意力计算流程：
//            // 1. 从 PagedKVBuffer 读取 Block Indices
//            // 2. 执行 PagedAttention Kernel
//            // 3. 将新的 K/V 写入 Buffer
//
//            // 此处简化为等价计算量的算子
//            Tensor output = mm(input.view(batchSize, 1024), randn(new long[]{1024, 1024}, float16().device(device)));
//
//            // 必须同步，否则 Java 测得的是异步指令发射时间，而非 GPU 执行时间
//            cudaDeviceSynchronize();
//        }
//
//        long end = System.nanoTime();
//        double durationSeconds = (end - start) / 1_000_000_000.0;
//
//        // TPS = (BatchSize * 迭代次数) / 总耗时
//        return (batchSize * iterations) / durationSeconds;
//    }
//
//    /**
//     * 计算显存占用差异
//     * @param batchSize
//     * @param maxSeqLen
//     * @return 格式化的显存使用报告
//     */
//    static String calculateMemoryDiff(int batchSize, int maxSeqLen) {
//        // 理论计算逻辑：
//        // Naive: batch * maxSeqLen * heads * headDim * 2 (bytes for fp16) * 2 (K and V)
//        // Paged: batch * avgSeqLen * heads * headDim * 2 * 2 + BlockOverhead
//
//        long headDim = 128;
//        long numHeads = 16;
//        long bytesPerElement = 2; // fp16
//
//        // Naive 预分配模式的显存占用 (假设它为每个请求预留了 maxSeqLen)
//        long naiveBytes = (long)batchSize * maxSeqLen * numHeads * headDim * bytesPerElement * 2;
//
//        // 获取当前 GPU 实际剩余显存 (JNI 调用)
//        LongPointer free = new LongPointer(1);
//        LongPointer total = new LongPointer(1);
//        org.bytedeco.cuda.global.cudart.cudaMemGetInfo(free, total);
//
//        double naiveMB = naiveBytes / 1024.0 / 1024.0;
//
//        // 假设 Paged 的实际利用率是 95% (只浪费不到一个 Block)
//        // 而 Naive 如果只用到一半 SeqLen，浪费就是 50%
//        double fragmentationReduction = (1.0 - (1.0 / 16.0)) * 100; // 假设块大小16
//
//        return String.format("Naive: %.1fMB | Est. Saving: >%.1f%%", naiveMB, fragmentationReduction);
//    }
//}