package org.bytedeco.pytorch.geometric.demo.layer;

import org.bytedeco.javacpp.PointerScope;
import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.geometric.aggr.Set2Set;

import java.util.Arrays;

public class DemoSet2Set {

    public static void main(String[] args) {
        System.out.println("=== Testing org.bytedeco.pytorch.geometric.aggr.Set2Set org.bytedeco.pytorch.geometric.aggr.Aggregation ===");

        // 1. 设置环境
//        Device device = torch_cuda.is_available() ? new Device("cuda") : new Device("cpu");
//        System.out.println("Device: " + (device.is_cuda() ? "CUDA" : "CPU"));

        if (!torch.hasCUDA()) {
            System.out.println("===(CUDA) not support ===");
//            throw new RuntimeException("Need CUDA for this enterprise demo!");
        }
        Device device = new Device("cpu");// new Device("cuda"); //mps 不支持 index_reduce
        long dimSize = 3;   // Batch Size = 3 graphs
        long inChannels = 16;
        long steps = 3;     // LSTM 迭代次数

        // 2. 构造数据
        // 假设总共有 10 个节点
        TensorOptions opts = new TensorOptions().device(new DeviceOptional(device)).dtype(new ScalarTypeOptional(torch.ScalarType.Float));
        TensorOptions idxOpts = new TensorOptions().device(new DeviceOptional(device)).dtype(new ScalarTypeOptional(torch.ScalarType.Long));

        Tensor x = torch.randn(new long[]{10, inChannels}, opts);

        // Index: 
        // Graph 0: 3 nodes
        // Graph 1: 2 nodes
        // Graph 2: 5 nodes
        Tensor index = torch.tensor(new long[]{
                0, 0, 0,
                1, 1,
                2, 2, 2, 2, 2
        }, idxOpts);

        // 3. 初始化 org.bytedeco.pytorch.geometric.aggr.Set2Set
        Set2Set set2set = new Set2Set(inChannels, steps, 1);
        set2set.to(device, true);

        // 4. 前向传播
        try (PointerScope scope = new PointerScope()) {
            Tensor out = set2set.forward(x, index, dimSize);

            System.out.println("Input Shape:  " + Arrays.toString(x.shape()));
            System.out.println("Output Shape: " + Arrays.toString(out.shape()));

            // 验证
            // Output dim 应该是 [Batch, 2 * In]
            long expectedFeatures = 2 * inChannels;

            boolean shapeCorrect = (out.size(0) == dimSize) && (out.size(1) == expectedFeatures);

            if (shapeCorrect) {
                System.out.println("PASS: Output dimensions are correct [Batch, 2*In].");
            } else {
                System.err.println("FAIL: Expected [3, " + expectedFeatures + "]");
            }

            // 打印部分数据确保没崩溃且有数值
            System.out.println("First Graph Embedding:\n" + out.slice(0, new LongOptional(0), new LongOptional(1), 1));
        }
    }
}
