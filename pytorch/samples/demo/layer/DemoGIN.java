package samples.demo.layer;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.geometric.nn.model.GIN;
//import org.gnn.demo.model.org.bytedeco.pytorch.geometric.nn.model.GIN;

import java.util.Arrays;

public class DemoGIN {
    public static void main(String[] args) {
        System.out.println("=== Testing org.bytedeco.pytorch.geometric.nn.model.GIN Model Structure ===");

        long numNodes = 10;
        long inChannels = 16;
        long hiddenChannels = 32;
        long outChannels = 2; // 二分类
        int numLayers = 3;

        // 1. 准备数据
        Tensor x = torch.randn(new long[]{numNodes, inChannels});
        Tensor edge_index = torch.tensor(new long[]{
                0, 1, 1, 2, 2, 3, 3, 4, // source
                1, 0, 2, 1, 3, 2, 4, 3  // target
        }).reshape(2, 8);

        // 2. 初始化 org.bytedeco.pytorch.geometric.nn.model.GIN
        // org.bytedeco.pytorch.geometric.nn.conv.GINConv 内部包含 MLP，初始化会比较慢，属于正常
        GIN model = new GIN(inChannels, hiddenChannels, outChannels, numLayers, 0.5);
        System.out.println("org.bytedeco.pytorch.geometric.nn.model.GIN Model Initialized.");

        // 3. 前向传播
        model.train(true); // 开启 BN 和 Dropout
        Tensor out = model.forward(x, edge_index);

        // 4. 验证
        System.out.println("Input Shape:  " + Arrays.toString(x.shape()));
        System.out.println("Output Shape: " + Arrays.toString(out.shape()));

        if (out.size(0) == numNodes && out.size(1) == outChannels) {
            System.out.println("PASS: Output dimensions are correct.");
        } else {
            System.err.println("FAIL: Dimension mismatch!");
        }
    }
}