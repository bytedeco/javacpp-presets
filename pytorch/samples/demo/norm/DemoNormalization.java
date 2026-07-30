package samples.demo.norm;
import org.bytedeco.pytorch.autograd.*;
import org.bytedeco.pytorch.nn.modules.*;

import org.bytedeco.javacpp.PointerScope;
import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.geometric.nn.norm.GraphNorm;
import org.bytedeco.pytorch.geometric.nn.norm.HeteroLayerNorm;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

import static org.bytedeco.pytorch.global.torch.randn;

public class DemoNormalization {

    public static void main(String[] args) {
        System.out.println("=== Testing Normalization Layers ===");

        try (PointerScope scope = new PointerScope()) {
            testGraphNorm();
            testHeteroNorm();
        }
    }

    public static void testHeteroNorm() {
        System.out.println("=== Starting HeteroLayerNorm Test ===");

        // 1. 定义超参数
        long channels = 128; // 确保输入维度与初始化维度严格一致
        String[] nodeTypes = {"paper", "author"};

        // 2. 初始化设备选项 (建议显式声明)
        TensorOptions options = new TensorOptions()
                .dtype(new ScalarTypeOptional(torch.ScalarType.Float))
                .device(new DeviceOptional(new Device("cpu")));

        // 3. 创建 HeteroLayerNorm 实例
        // 假设我们的 LayerNorm 构造函数签名是: (long inChannels, double eps, boolean affine, String mode)
        HeteroLayerNorm heteroLN = new HeteroLayerNorm(channels, nodeTypes, 1e-5, true, "node");

        // 4. 构造异构输入数据 (xDict)
        Map<String, Tensor> xDict = new HashMap<>();

        // paper 节点: 5个节点, 128维特征
        Tensor paperX = randn(new long[]{5, channels}, options);
        // author 节点: 3个节点, 128维特征
        Tensor authorX = randn(new long[]{3, channels}, options);

        xDict.put("paper", paperX);
        xDict.put("author", authorX);

        // 5. 构造 Batch 数据 (用于测试 mode="graph", 如果是 node 模式可传空)
        // 注意：Batch 必须是 Long 类型，防止底层 index_select 崩溃
        Map<String, Tensor> batchDict = new HashMap<>();
        TensorOptions longOptions = new TensorOptions()
                .dtype(new ScalarTypeOptional(torch.ScalarType.Long));

        batchDict.put("paper", torch.zeros(new long[]{5}, longOptions));
        batchDict.put("author", torch.zeros(new long[]{3}, longOptions));

        try {
            // 6. 执行前向传播
            System.out.println("Running forward pass...");
            Map<String, Tensor> outDict = heteroLN.forward(xDict, batchDict);

            // 7. 验证结果形状
            for (String type : nodeTypes) {
                Tensor out = outDict.get(type);
                long[] shape = out.shape();
                System.out.println("Node Type: [" + type + "] Output Shape: " +
                        Arrays.toString(shape));

                // 验证均值是否接近 0 (标准化特性)
                // 对特征维度求均值，期望均值接近 0
                double meanVal = out.mean().item().toDouble();
                System.out.println("Node Type: [" + type + "] Global Mean: " + meanVal);
            }

            System.out.println("Test Passed Successfully!");

        } catch (Exception e) {
            System.err.println("Test Failed during forward pass!");
            e.printStackTrace();
        }
    }
    private static void testGraphNorm() {
        System.out.println("\n[Test 1] GraphNorm");

        long C = 4;
        // 构造数据: 2个图
        // Graph 0: 3 nodes
        // Graph 1: 2 nodes
        Tensor x = torch.tensor(new float[]{
                1, 2, 3, 4, // G0_N0
                2, 3, 4, 5, // G0_N1
                0, 1, 2, 3, // G0_N2

                10, 10, 10, 10, // G1_N0
                20, 20, 20, 20  // G1_N1
        }).reshape(5, 4);

        Tensor batch = torch.tensor(new long[]{0, 0, 0, 1, 1});

        GraphNorm gn = new GraphNorm(C);

        // Forward
        Tensor out = gn.forward(x, batch);

        System.out.println("Input X:\n" + x);
        System.out.println("Output (GraphNorm):\n" + out);

        // 验证 Graph 1 (最后两行)
        // Mean = 15, Std = 5 (approx)
        // (10 - 15) / 5 = -1
        // (20 - 15) / 5 = 1
        Tensor g1Out = out.slice(0, new LongOptional(3), new LongOptional(5), 1);
        System.out.println("Graph 1 Normalized (Expect close to -1 and 1):\n" + g1Out);
    }

    private static void testHeteroNorm1() {
        System.out.println("\n[Test 2] org.bytedeco.pytorch.geometric.nn.norm.HeteroLayerNorm");

        // 配置异构图结构
        Map<String, Long> channels = new HashMap<>();
        channels.put("user", 4L);
        channels.put("item", 2L);

        // 构造数据
        Map<String, Tensor> xDict = new HashMap<>();
        xDict.put("user", randn(3, 4)); // 3 users, 4 feat
        xDict.put("item", randn(5, 2)); // 5 items, 2 feat
        String[] types = {"user", "item"};
//        String[] types = {"paper", "author"};
        HeteroLayerNorm heteroLN = new HeteroLayerNorm(2, types);

// 3. 模拟异构数据
//        Map<String, Tensor> xDict = new HashMap<>();
//        xDict.put("paper", randn(new long[]{5, 128}));
//        xDict.put("author", randn(new long[]{2, 128}));

// 4. 前向传播
        Map<String, Tensor> outDict = heteroLN.forward(xDict, null);
        // 初始化层
//        HeteroLayerNorm heteroLN = new HeteroLayerNorm(channels, types, 1e-5, true, "batch");
        // Forward
//        Map<String, Tensor> outDict = heteroLN.forward(xDict);

        for (String key : outDict.keySet()) {
            Tensor t = outDict.get(key);
            System.out.println("Type: " + key + ", Out Shape: " + Arrays.toString(t.shape()));

            // 简单验证 LayerNorm 效果: 均值应接近0，方差接近1 (dim=1)
            Tensor mean = t.mean(new long[]{1}, false,new ScalarTypeOptional());
            Tensor std = t.std(new long[]{1}, false, true);

            // 打印第一个节点的统计信息
            System.out.printf("  Node 0 Stats -> Mean: %.4f, Std: %.4f%n",
                    mean.select(0, 0).item().toFloat(),
                    std.select(0, 0).item().toFloat());
        }
    }
}