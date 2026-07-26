package org.bytedeco.pytorch.geometric.demo.kvcache;
import org.bytedeco.pytorch.jit.*;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.global.torch;

import java.time.Duration;
import java.time.Instant;
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.LongAdder;

import static org.bytedeco.pytorch.global.torch.*;

public class VirtualThreadMegaBenchmarkV3 {
    private static Tensor dummyA;
    private static Tensor dummyB;
    private static TensorOptions opts =new TensorOptions().dtype(new ScalarTypeOptional(kFloat())).device(new DeviceOptional(new Device(kMPS())));

    // 初始化（只需一次）
    static {
        // 选一个能跑满 10ms 的维度，比如 4096 或 8192，取决于你的 Mac 芯片 (M1/M2/M3 Pro/Max)
        long dim = 4096;

        dummyA = randn(new long[]{dim, dim}, opts);
        dummyB = randn(new long[]{dim, dim}, opts);
    }
    private static final Tensor DUMMY_A = randn(new long[]{2048, 2048},opts).retainReference();
    private static final Tensor DUMMY_B = randn(new long[]{2048, 2048},
            opts).retainReference();
    private static void burnGpuTime() {
        // 1. 规模减小到 1024 或 2048，减少单次提交的指令长度
        long dim = 2048;
        TensorOptions opts =new TensorOptions().dtype(new ScalarTypeOptional(kFloat())).device(new DeviceOptional(new Device(kMPS())));

        // 2. 局部创建（虽然开销大一点，但对并发更友好），并立即释放
        try (Tensor a = randn(new long[]{dim, dim}, opts);
             Tensor b = randn(new long[]{dim, dim}, opts);
             Tensor res = matmul(a, b)) {

            // 3. 关键：不要用全局同步 synchronize()，因为它会同步整个设备
            // 改为读取一个标量，这只会阻塞当前线程等待这个特定的 res 完成
            try (Tensor scalar = res.select(0, 0).select(0, 0)) {
                scalar.item().toFloat();
//                org.bytedeco.pytorch.global.torch.commit();
//                org.bytedeco.pytorch.global.torch.synchronize();
            }
        }
    }
    private static void burnGpuTime3() {
        // 使用 try-with-resources 强制释放 matmul 产生的临时结果 Tensor
        try (Tensor result = matmul(DUMMY_A, DUMMY_B)) {
            // 同步，确保这一层计算真的占用了 GPU 时间
            org.bytedeco.pytorch.global.torch.synchronize();
        }
        // result 在这里会被立即执行释放逻辑，归还给 MPS 缓冲池
    }
    private static void burnGpuTime2() {
        // 执行 C = A * B
        // matmul 是同步发射，但 MPS 是异步执行的，
        // 为了模拟真实的阻塞等待，我们需要显式同步或读取一个标量
        try (Tensor result = matmul(dummyA, dummyB)) {
            // 重要：PyTorch 的 MPS/CUDA 调用是异步的。
            // 如果不执行同步，虚拟线程会瞬间跑完循环，起不到模拟耗时的作用。
            // 读取一个元素会强制 CPU 等待 GPU 计算完成。
            try (Tensor firstElement = result.select(0, 0).select(0, 0)) {
                float val = firstElement.item().toFloat();
            }
//            float val = result.item().toFloat();
        }
    }
    public static void main(String[] args) throws InterruptedException {
        // --- 1. 严格限制物理池，逼出驱逐 ---
        int totalBlocks = 3000;  // 只有 2000 块
        int numLayers = 32;      // 32 层模型
        int blockSize = 16;
        CoWBlockManagerV8 manager = new CoWBlockManagerV8(totalBlocks, numLayers, blockSize, 128, 0);
        // 限制并发执行 GPU 计算的虚拟线程数，模拟 MPS 核心限制
        Semaphore mpsComputeSlots = new Semaphore(36);
        System.out.println("totalBlocks: " + totalBlocks + ", blockSize: " + blockSize+", numLayers: " + numLayers+", mpsComputeSlots maxConcurrentMPS: " + 6);

        int totalUsers = 80;  // 一万个虚拟线程
        LongAdder successCount = new LongAdder();
        LongAdder totalTokens = new LongAdder();

        // 模拟公共前缀（Radix Tree 共享点）
        List<Long> sharedPrefix = Arrays.asList(111L, 222L, 333L);

        System.out.println("🔥 启动万级虚拟线程压力测试...");
        System.out.println("内存限制: " + totalBlocks + " Blocks | 目标用户: " + totalUsers);

        Instant startTime = Instant.now();


        TensorOptions options =new TensorOptions().dtype(new ScalarTypeOptional(kFloat())).device(new DeviceOptional(new Device(kMPS())));
// ... 前面初始化代码不变 ...
        try (var executor = Executors.newVirtualThreadPerTaskExecutor()) {
            for (int i = 0; i < totalUsers; i++) {
                final int userId = i;
                executor.submit(() -> {
                    String sid = "session-" + userId;
                    try {
                        // 1. 构造一个真实的模拟 Tensor (Shape: [1, blockSize, headDim])
                        try (Tensor input = randn(new long[]{1, blockSize, 128}, options)) {

                            try (PagedKvBufferV3 kv = new PagedKvBufferV3(sid, manager, numLayers)) {
                                for (int l = 0; l < numLayers; l++) {
                                    mpsComputeSlots.acquire();
                                    try {
                                        // 模拟 Radix 路径：前缀 + 当前层 Hash
                                        List<Long> path = new ArrayList<>(sharedPrefix);
                                        path.add((long) l);

                                        // A. 分配/匹配块
                                        manager.matchAndAllocatePath(path, sid, kv);

                                        // B. 【核心修正】调用真实 Prefill，产生显存 IO
                                        kv.prefillUltra(l, 0, input);
                                        burnGpuTime();
                                        // C. 模拟 GPU 计算耗时
//                                        Thread.sleep(Duration.ofMillis(10));
                                    } finally {
                                        mpsComputeSlots.release();
                                    }
                                }
                                successCount.increment();
                                // 每一层处理 blockSize 个 token
                                totalTokens.add(numLayers * blockSize);
                            }
                        }
                    } catch (Exception e) {
                        System.err.println("Fatal: " + e.getMessage());
                    } finally {
                        manager.releaseSession(sid);
                    }
                });
            }
        }
//        try (var executor = Executors.newVirtualThreadPerTaskExecutor()) {
//            for (int i = 0; i < totalUsers; i++) {
//                final int userId = i;
//                executor.submit(() -> {
//                    String sid = "session-" + userId;
//                    try {
//                        // 构造请求路径：公共前缀 + 个人私有数据
//                        List<Long> myPath = new ArrayList<>(sharedPrefix);
//                        myPath.add((long) (userId + 10000)); // 每个用户一个独特的 Block
//
//                        // 每个 Session 必须逐层完成 Prefill
//                        // 在虚拟线程内部
//                        try (PagedKvBufferV3 kv = new PagedKvBufferV3(sid, manager, numLayers)) {
//                            // 构造模拟输入 Tensor (例如 1x16 的 token)
//                            try (Tensor input = randn(new long[]{1, blockSize, 128}, options)) {
//
//                                for (int layer = 0; layer < numLayers; layer++) {
//                                    mpsComputeSlots.acquire();
//                                    try {
//                                        // 1. 先通过 Radix Tree 获取/分配该层所需的 Block 映射
//                                        // 这里的 Hash 模拟：layerHash = sessionId + layerId (或者 content hash)
//                                        List<Long> path = new ArrayList<>(sharedPrefix);
//                                        path.add((long) (userId * 1000 + layer));
//
//                                        // 执行分配逻辑 (触发可能的 Evict/Wait)
//                                        manager.matchAndAllocatePath(path, sid, kv);
//
//                                        // 2. 【关键修正】调用真实的 Prefill 逻辑
//                                        // 这会把 input 写入到刚才分配的物理块中
//                                        kv.prefillUltra(layer, 0, input);
//
//                                        // 3. 模拟计算延迟
//                                        Thread.sleep(Duration.ofMillis(6));
//                                    } finally {
//                                        mpsComputeSlots.release();
//                                    }
//                                }
//                            }
//                            successCount.increment();
//                        }
////                        try (PagedKvBufferV3 kv = new PagedKvBufferV3(sid, manager, numLayers)) {
////
////                            for (int layer = 0; layer < numLayers; layer++) {
////                                // 模拟竞争 GPU 计算资源
////                                mpsComputeSlots.acquire();
////                                try {
////                                    // 这一步会触发：1. Radix 命中 2. 物理分配 3. 阻塞等待 4. 驱逐
////                                    manager.matchAndAllocatePath(myPath, sid, kv);
////
////                                    // 模拟真实的 GPU 算子耗时（1ms）
////                                    // 虚拟线程在 sleep 时会切出 CPU，从而让其它线程去竞争内存
////                                    Thread.sleep(Duration.ofMillis(6));
////                                } finally {
////                                    mpsComputeSlots.release();
////                                }
////                            }
////
////                            successCount.increment();
////                            totalTokens.add(myPath.size() * blockSize);
////                        }
//                    } catch (Exception e) {
//                        // 如果系统实在是饱和到超时，会在这里报错
//                        System.err.println("Fatal: " + e.getMessage());
//                    } finally {
//                        // 任务结束，释放显存
//                        manager.releaseSession(sid);
//                    }
//                });
//            }
//        }

        Instant endTime = Instant.now();
        long durationMs = Duration.between(startTime, endTime).toMillis();

        System.out.println("\n" + "=".repeat(44));
        System.out.println("万级并发测试报告:");
        System.out.println("运行时间: " + (durationMs / 1000.0) + "s");
        System.out.println("成功处理请求: " + successCount.sum());
        System.out.println("总 Token 吞吐量: " + totalTokens.sum());
        System.out.println("平均 TPS: " + String.format("%.2f", totalTokens.sum() / (durationMs / 1000.0)));
        System.out.println("最终空闲块: " + manager.getFreeBlockCount());
        System.out.println("=".repeat(44));
    }
}