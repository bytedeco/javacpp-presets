package org.bytedeco.pytorch.geometric.demo.norm;
import org.bytedeco.pytorch.nn.modules.*;

import org.bytedeco.pytorch.ScalarTypeOptional;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorOptions;
import org.bytedeco.pytorch.geometric.nn.norm.*;

import java.util.Arrays;

import static org.bytedeco.pytorch.global.torch.*;

public class DemoNormTest {

    public static void main(String[] args) {
        long numNodes = 5;
        long channels = 16;
        Tensor x = randn(new long[]{numNodes, channels}, new TensorOptions().dtype(new ScalarTypeOptional(kFloat())));
        // 模拟边索引
        Tensor edge_index = tensor(new long[]{0, 1, 1, 2, 2, 3, 3, 4, 0, 4}, new TensorOptions().dtype(new ScalarTypeOptional(kLong()))).view(2, 5);

        // 1. LayerNorm: 独立节点特征归一化
        testLayerNorm(x, channels);

        // 2. PairNorm: 解决过平滑，使节点间保持距离
        testPairNorm(x);

        // 3. MessageNorm: 归一化消息传递后的特征
        testMessageNorm(x, x); // 假设消息 msg = x

        testMeanSubtractionNorm(x);
        testDiffGroupNorm(x);
        testNewNorms();
        testBatchNorm();
    }

    public static void testLayerNorm(Tensor x, long channels) {
        System.out.println("\n[LayerNorm Test]");
        LayerNorm norm = new LayerNorm(channels, 1e-5, true);
        Tensor out = norm.forward(x);
        System.out.println("Output Shape: " + Arrays.toString(out.shape()));
        // 验证均值接近0 (在特征维度)
//        System.out.println("Mean sample: " + out.mean(new long[]{1}, false, new ScalarTypeOptional()).slice(0, 0, 1).item().toFloat());
    }

    public static void testPairNorm(Tensor x) {
        System.out.println("\n[PairNorm Test]");
        // scale=1.0, sub_graph_batching=false
        PairNorm norm = new PairNorm();
        Tensor out = ((PairNorm)norm).forward(x, (Tensor)null);
        System.out.println("Output Shape: " + Arrays.toString(out.shape()));
    }

    public static void testMessageNorm(Tensor x, Tensor msg) {
        System.out.println("\n[MessageNorm Test]");
        MessageNorm norm = new MessageNorm(x.size(1));//, true);
        Tensor out = norm.forward(x, msg);
        System.out.println("Output Shape: " + Arrays.toString(out.shape()));
    }

    public static void testMeanSubtractionNorm(Tensor x) {
        System.out.println("\n[MeanSubtractionNorm Test]");
        // 减去图中节点的均值
        MeanSubtractionNorm norm = new MeanSubtractionNorm();
        Tensor out = ((MeanSubtractionNorm)norm).forward(x, (Tensor)null); // 假设单图
        System.out.println("Sum of output (should be near 0): " + out.sum().item().toFloat());
    }

    public static void testDiffGroupNorm(Tensor x) {
        System.out.println("\n[DiffGroupNorm Test]");
        // 将节点动态分组归一化，解决过度同质化
        DiffGroupNorm norm = new DiffGroupNorm(x.size(1), 4, 0.1); // 4个组
        Tensor out = norm.forward(x);
        System.out.println("Output Shape: " + Arrays.toString(out.shape()));
    }

    public static void testNewNorms() {
        System.out.println("=== Testing New Norms ===");

        long C = 8;
        Tensor x = randn(new long[]{10, C}, new TensorOptions().dtype(new ScalarTypeOptional(kFloat())));
        Tensor batch = tensor(new long[]{0, 0, 0, 0, 0, 1, 1, 1, 1, 1}, new TensorOptions().dtype(new ScalarTypeOptional(kLong())));

        // 1. 测试 InstanceNorm
        InstanceNorm iNorm = new InstanceNorm(C, 1e-5, 0.1, true, false);
        Tensor outI = iNorm.forward(x, batch);
        System.out.println("InstanceNorm Output: " + outI.size(0) + "x" + outI.size(1));

        // 2. 测试 HeteroBatchNorm
        // 假设 2 种类型
        Tensor typeIdx = tensor(new long[]{0, 0, 1, 1, 0, 0, 1, 1, 0, 1}, new TensorOptions().dtype(new ScalarTypeOptional(kLong())));
        HeteroBatchNorm hNorm = new HeteroBatchNorm(C, 2, 1e-5, 0.1, true, true); //new DoubleOptional(0.1)
        Tensor outH = hNorm.forward(x, typeIdx);
        System.out.println("HeteroBatchNorm Output: " + outH.size(0) + "x" + outH.size(1));
    }

    public static void testBatchNorm() {
        System.out.println("=== Starting Custom BatchNorm Test ===");

        long inChannels = 4;

        // 1. 初始化模型
        // 参数：channels=4, eps=1e-5, momentum=0.1, affine=true, track=true, allowSingle=true
        BatchNorm bn = new BatchNorm(inChannels, 1e-5, 0.1, true, true, true);

        try {
            // --- 场景 A: 正常 Batch (N > 1) ---
            System.out.println("\n[Test A: Normal Batch]");
            Tensor xNormal = randn(new long[]{5, inChannels}, new TensorOptions().dtype(new ScalarTypeOptional(kFloat())));

            bn.train(true);
            Tensor outNormal = bn.forward(xNormal);
            System.out.println("Normal Output Shape: " + Arrays.toString(outNormal.shape()));
            System.out.println("Running Mean (after 1 step): " + bn.innerBN.running_mean());

            // --- 场景 B: 单元素 Batch (N = 1) ---
            // 这是最关键的测试点。原生 BN 在 train() 模式下处理 N=1 会抛出异常
            System.out.println("\n[Test B: Single Element Batch]");
            Tensor xSingle = randn(new long[]{1, inChannels}, new TensorOptions().dtype(new ScalarTypeOptional(kFloat())));

            // 依然保持 train() 状态
            bn.train(true);

            // 这里的 forward 内部会触发：eval() -> forward -> train()
            Tensor outSingle = bn.forward(xSingle);

            System.out.println("Single Element Output Shape: " + Arrays.toString(outSingle.shape()));
            if (bn.is_training()) {
                System.out.println("PASS: Model state recovered to TRAINING mode.");
            }

            // --- 场景 C: 梯度验证 ---
            System.out.println("\n[Test C: Gradient Flow]");
            Tensor loss = outNormal.sum().add(outSingle.sum());
            loss.backward();
            System.out.println("Backward Pass: SUCCESS (Gradients computed)");

        } catch (Exception e) {
            System.err.println("Test FAILED!");
            e.printStackTrace();
        }
    }
//    public static void testHeteroNorm() {
//        System.out.println("\n=== Hetero Normalization Test ===");
//
//        // 构造异构数据: 'user' 节点和 'item' 节点
//        Tensor userX = randn(new long[]{10, 32},new TensorOptions().dtype(new ScalarTypeOptional( kFloat())));
//        Tensor itemX = randn(new long[]{20, 64}, new TensorOptions().dtype(new ScalarTypeOptional( kFloat())));
//
//        // 1. HeteroBatchNorm
//        System.out.println("[HeteroBatchNorm]");
//        // 为不同类型的节点指定不同的 BatchNorm
//        HeteroBatchNorm hBatch = new HeteroBatchNorm();
//        hBatch.register_type("user", 32);
//        hBatch.register_type("item", 64);
//
//        // 2. HeteroLayerNorm
//        System.out.println("[HeteroLayerNorm]");
//        HeteroLayerNorm hLayer = new HeteroLayerNorm();
//        hLayer.register_type("user", 32);
//        hLayer.register_type("item", 64);
//
//        // 模拟 Forward
//        // 注意: 实际代码中通常使用自定义的 MultiTensorData 结构
//        Tensor outUser = hLayer.forward("user", userX);
//        Tensor outItem = hLayer.forward("item", itemX);
//
//        System.out.println("User Output Shape: " + Arrays.toString(outUser.shape()));
//        System.out.println("Item Output Shape: " + Arrays.toString(outItem.shape()));
//    }
}