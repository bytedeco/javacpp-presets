package org.bytedeco.pytorch.geometric.demo.layer;

import org.bytedeco.pytorch.ScalarTypeOptional;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorOptions;
import org.bytedeco.pytorch.geometric.nn.conv.PNAConv;
import org.bytedeco.pytorch.geometric.nn.model.PNA;

import java.util.Arrays;

import static org.bytedeco.pytorch.global.torch.*;

public class DemoPNA {


    public static void main(String[] args) {
        // 初始化运行
        testPNA();
    }


    public static void testPNA() {
        System.out.println("=== Starting PNA & PNAConv Test ===");

        // 1. 准备基础参数
        long inChannels = 16;
        long hiddenChannels = 32;
        long outChannels = 64;
        int numLayers = 2;
        double avgDegree = 1.0; // 简单起见，设为 1.0

        String[] aggregators = {"mean", "max", "sum"};
        String[] scalers = {"identity", "amplification", "attenuation"};

        // 2. 构造模拟数据
        // 5 个节点，每个节点 16 维特征
        Tensor x = randn(new long[]{5, inChannels},
                new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)));

        // 构造边索引 [2, 6]：一个简单的有向图
        // 0 -> 1, 1 -> 2, 2 -> 3, 3 -> 4, 4 -> 0, 0 -> 2
        long[] edgeData = {0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 2};
        Tensor edge_index = tensor(edgeData,
                new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Long))).view(2, 6);

        try {
            // --- 测试单层 PNAConv ---
            System.out.println("Testing Single PNAConv Layer...");
            PNAConv conv = new PNAConv(inChannels, hiddenChannels, aggregators, scalers, avgDegree);

            Tensor convOut = conv.forward(x, edge_index);
            System.out.println("PNAConv Output Shape: " + Arrays.toString(convOut.shape()));
            // 验证维度是否匹配 hiddenChannels
            if (convOut.size(1) != hiddenChannels) throw new RuntimeException("PNAConv dimension mismatch!");

            // --- 测试多层 PNA 模型 ---
            System.out.println("\nTesting Multi-layer PNA Model...");
            PNA pnaModel = new PNA(inChannels, hiddenChannels, outChannels, numLayers,
                    aggregators, scalers, avgDegree);

            // 确保 PNA 继承自 GenericModule，可以处理变长参数
            Tensor pnaOut = pnaModel.forward(x, edge_index);
            System.out.println("PNA Model Output Shape: " + Arrays.toString(pnaOut.shape()));

            // 验证最终输出维度
            if (pnaOut.size(1) != outChannels) throw new RuntimeException("PNA Model dimension mismatch!");

            // 检查梯度流 (模拟 backward)
            System.out.println("\nTesting Backward Pass...");
            Tensor loss = pnaOut.sum();
            loss.backward();
            System.out.println("Backward Pass Successful (No Crashes)!");

            System.out.println("\nAll PNA Tests Passed!");

        } catch (Exception e) {
            System.err.println("Test Failed!");
            e.printStackTrace();
            // 打印堆栈信息以便排查是否又是某个 Scalar 包装问题
        }
    }


}