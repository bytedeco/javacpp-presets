package org.bytedeco.pytorch.geometric.demo.layer;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.geometric.nn.model.GAT;
//import org.gnn.demo.model.org.bytedeco.pytorch.geometric.nn.model.GAT;

import java.util.Arrays;

public class DemoGAT {

    public static void main(String[] args) {
        System.out.println("=== Testing org.bytedeco.pytorch.geometric.nn.model.GAT Full Model ===");

        // 1. 模拟数据
        long numNodes = 10;
        long inChannels = 16;   // 输入特征
        long hiddenDim = 8;     // 隐层特征
        long outChannels = 2;   // 二分类
        long heads = 8;         // 隐层 8 头
        long outHeads = 4;      // 输出层 4 头 (最后会取平均)

        Tensor x = torch.randn(new long[]{numNodes, inChannels});
        Tensor edge_index = torch.tensor(new long[]{
                0, 1, 1, 2, 2, 3, 3, 4, 4, 0, // source
                1, 0, 2, 1, 3, 2, 4, 3, 0, 4  // target
        }).reshape(2, 10);

        System.out.println("Input Shape: " + Arrays.toString(x.shape()));

        // 2. 实例化 org.bytedeco.pytorch.geometric.nn.model.GAT
        GAT gat = new GAT(inChannels, hiddenDim, outChannels, heads, outHeads, 0.6);

        // 3. 运行前向传播
        gat.train(true); // 启用 Dropout
        Tensor out = gat.forward(x, edge_index);

        // 4. 验证输出
        System.out.println("Output Shape: " + Arrays.toString(out.shape()));

        // 预期验证:
        // Layer 1 输出应为: [10, 8 * 8] = [10, 64]
        // Layer 2 原始输出: [10, 4 * 2] = [10, 8]
        // Layer 2 平均后输出: [10, 2]

        long[] expected = new long[]{numNodes, outChannels};
        if (Arrays.equals(out.shape(), expected)) {
            System.out.println("PASS: Output shape matches [NumNodes, OutChannels]");
        } else {
            System.err.println("FAIL: Expected " + Arrays.toString(expected) +
                    " but got " + Arrays.toString(out.shape()));
        }

        // 5. 打印部分结果以检查数值 (不应全为0或NaN)
//        System.out.println("First node logits: " + out.slice(0, 0, 1, 1));
    }
}
