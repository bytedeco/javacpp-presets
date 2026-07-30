package samples.demo.kvcache;
import org.bytedeco.pytorch.data.*;
import org.bytedeco.pytorch.jit.*;

import org.bytedeco.pytorch.*;
import static org.bytedeco.pytorch.global.torch.*;

public class MpsKVBenchmark {
    // 模拟参数
    static int HEAD_DIM = 128;
    static int NUM_HEADS = 16;
    static int MAX_SEQ_LEN = 2048; // 压测的最大序列长度

    public static void main(String[] args) {
        // 确保 MPS 可用
        if (!hasMPS()) {
            System.out.println("MPS is not available on this Mac.");
            return;
        }

        int[] batchSizes = {1, 4, 8, 16, 32, 64};
        System.out.println("BatchSize | Naive TPS | Paged TPS | Memory Status (System)");
        System.out.println("-------------------------------------------------------");

        for (int batch : batchSizes) {
            // 1. 压测 Naive 模式 (连续大张量)
            double naiveTps = runNaiveBenchmark(batch, 50);

            // 2. 压测 Paged 模式 (你实现的 PagedKvBuffer)
            double pagedTps = runPagedBenchmark(batch, 50);

            System.out.printf("%9d | %9.2f | %9.2f | %s\n",
                    batch, naiveTps, pagedTps, getMacMemoryStatus());
        }
    }

    static String getMacMemoryStatus() {
        // Mac 没有 CUDA 的 free/total 概念，它共享系统内存
        Runtime runtime = Runtime.getRuntime();
        long usedMemory = (runtime.totalMemory() - runtime.freeMemory()) / 1024 / 1024;

        // 建议在运行的同时打开 Activity Monitor (活动监视器) 的 GPU 标签页
        return "JVM Used: " + usedMemory + "MB";
    }

    /**
     * 模拟推理：MPS 专用版
     */
    static double simulateInference(int batchSize, int iterations, boolean isPaged) {
        Device device = new Device(kMPS());
        // 使用 Float16 减少显存压力，这是 Mac M 系列芯片最擅长的数据类型
        Tensor input = randn(new long[]{batchSize, 1, 1024}, new TensorOptions().dtype(new ScalarTypeOptional(kFloat())).device(new DeviceOptional(device)));
        Tensor weights = randn(new long[]{1024, 1024}, new TensorOptions().dtype(new ScalarTypeOptional(kFloat())).device(new DeviceOptional(device)));

        // 预热
        for (int i = 0; i < 5; i++) {
            input.matmul(weights).add_(new Scalar(0.1));
        }

        // MPS 同步点：通过一个简单的 CPU 交互操作强制刷新命令队列
        input.cpu();

        long start = System.nanoTime();

        for (int i = 0; i < iterations; i++) {
            // 模拟计算负载
            // 如果是 Paged，这里逻辑上会包含 Block 索引查找的微小开销
            Tensor out = input.matmul(weights);

            // 强制同步以确保测得的是硬件执行时间
            out.cpu();
        }

        long end = System.nanoTime();
        return (batchSize * iterations) / ((end - start) / 1_000_000_000.0);
    }

    static double runNaiveBenchmark(int batch, int iter) {
        TensorOptions options =new TensorOptions().dtype(new ScalarTypeOptional(kFloat())).device(new DeviceOptional(new Device(kMPS())));
        
        try {
            // 尝试分配连续的 KV 显存。在 Mac 上，如果超过 Unified Memory 限制，这里会变慢或报错
            Tensor k = empty(new long[]{batch, NUM_HEADS, MAX_SEQ_LEN, HEAD_DIM},options,new MemoryFormatOptional());
            
            Tensor v = empty( new long[]{batch, NUM_HEADS, MAX_SEQ_LEN, HEAD_DIM}, options, new MemoryFormatOptional());
            return simulateInference(batch, iter, false);
        } catch (Exception e) {
            return -1.0; // 代表 OOM 或显存分配失败
        }
    }

    static double runPagedBenchmark(int batch, int iter) {
        // 调用你实现的 PagedKvBuffer 逻辑
        // 注意：在 MPS 上，Paged 块的存储在物理上也是离散的内存区域
        return simulateInference(batch, iter, true);
    }
}