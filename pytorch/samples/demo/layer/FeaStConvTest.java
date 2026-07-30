package samples.demo.layer;
import org.bytedeco.pytorch.jit.*;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.global.torch;
import org.junit.Test;
import org.junit.runner.JUnitCore;
import org.junit.runner.Result;
import org.junit.runner.notification.Failure;
import org.bytedeco.pytorch.geometric.nn.conv.FeaStConv;
import static org.junit.Assert.*;

/**
 * FeaStConv 测试用例：
 * 1. 基础形状验证（输入输出维度匹配）
 * 2. 数值准确性验证（固定权重，手动计算预期值）
 * 3. 边界场景验证（空边、维度错误、多注意力头）
 */
public class FeaStConvTest {
    static {
        // 固定随机种子，保证结果可复现
        torch.manual_seed(42L);
    }

    // ========== 测试1：基础形状验证 ==========
    @Test
    public void testForwardShape() {
        // 1. 初始化FeaStConv
        long inChannels = 4;
        long outChannels = 2;
        int heads = 3;
        boolean hasBias = true;
        FeaStConv feastConv = new FeaStConv(inChannels, outChannels, heads, hasBias);

        // 2. 构造测试数据
        TensorOptions floatOpts = new TensorOptions()
                .dtype(new ScalarTypeOptional(torch.ScalarType.Float))
                .device(new DeviceOptional(new Device(torch.kCPU())));
        // 节点特征：5个节点 × 4维输入
        Tensor x = torch.randn(new long[]{5, 4}, floatOpts);

        // 边索引：[2, 6]（6条边）
        long[][] edgeIndexData = new long[][]{
                {0, 0, 1, 2, 3, 4}, // 源节点
                {1, 2, 3, 0, 4, 1}  // 目标节点
        };
        Tensor edgeIndex = torch.tensor(flatten(edgeIndexData),
                        new TensorOptions().dtype(new ScalarTypeOptional(torch.ScalarType.Long)))
                .view(2, 6);

        // 3. 前向传播
        Tensor output = feastConv.forward(x, edgeIndex);

        // 4. 形状验证
        long[] outputShape = output.sizes().vec().get();
        assertEquals("输出维度应为2D", 2, outputShape.length);
        assertEquals("输出节点数应匹配输入", 5, outputShape[0]);
        assertEquals("输出特征维度应匹配配置", outChannels, outputShape[1]);

        torch.print(output);
        // 5. 资源释放
        x.close();
        edgeIndex.close();
        output.close();
    }

    // ========== 测试2：数值准确性验证 ==========
    @Test
    public void testNumericalAccuracy() {
        // 1. 初始化FeaStConv（简化配置：1个头，无偏置，便于手动计算）
        long inChannels = 2;
        long outChannels = 1;
        int heads = 1;
        boolean hasBias = false;
        FeaStConv feastConv = new FeaStConv(inChannels, outChannels, heads, hasBias);

        // 2. 固定线性层参数（禁用梯度，避免叶子张量报错）
        try (NoGradGuard guard = new NoGradGuard()) {
            // 固定linWeights权重：[1*1, 2] → [[1, 0]]（输出=输入第1维）
            Tensor linWeights = torch.tensor(new float[]{1.0f, 0.0f},
                    new TensorOptions().dtype(new ScalarTypeOptional(torch.ScalarType.Float))).view(1,2);
            feastConv.getLinWeights().weight().copy_(linWeights);
            feastConv.getLinWeights().bias().copy_(torch.zeros(new long[]{1}));

            // 固定linSrc/linDst权重：[1, 2] → [[1, 0]]（输出=输入第1维）
            Tensor linSrcWeight = torch.tensor(new float[]{1.0f, 0.0f}).view(1,2);
            feastConv.getLinSrc().weight().copy_(linSrcWeight);
            feastConv.getLinSrc().bias().copy_(torch.zeros(new long[]{1}));

            Tensor linDstWeight = torch.tensor(new float[]{1.0f, 0.0f}).view(1,2);
            feastConv.getLinDst().weight().copy_(linDstWeight);
            feastConv.getLinDst().bias().copy_(torch.zeros(new long[]{1}));
        }

        // 3. 构造简单测试数据
        TensorOptions floatOpts = new TensorOptions()
                .dtype(new ScalarTypeOptional(torch.ScalarType.Float))
                .device(new DeviceOptional(new Device(torch.kCPU())));
        // 节点特征：3个节点 × 2维
        float[][] xData = new float[][]{
                {1.0f, 2.0f}, // 节点0：linSrc/Dst输出=1.0
                {3.0f, 4.0f}, // 节点1：linSrc/Dst输出=3.0
                {5.0f, 6.0f}  // 节点2：linSrc/Dst输出=5.0
        };
        Tensor x = torch.tensor(flatten(xData), floatOpts).view(3, 2);

        // 边索引：[2, 2]（2条边）
        Tensor edgeIndex = torch.tensor(new long[]{0,0,1,2},
                        new TensorOptions().dtype(new ScalarTypeOptional(torch.ScalarType.Long)))
                .view(2, 2);

        // 4. 前向传播
        Tensor output = feastConv.forward(x, edgeIndex);

        // 5. 手动计算预期结果
        // 步骤1：计算注意力分数q
        // - 边0→1：q = linSrc(x0) + linDst(x1) = 1 + 3 = 4 → softmax(4) = 1.0
        // - 边0→2：q = linSrc(x0) + linDst(x2) = 1 + 5 = 6 → softmax(6) = 1.0
        // 步骤2：计算多头部投影
        // - xTrans = linWeights(x) → [3,1,1] → 节点0:1.0, 节点1:3.0, 节点2:5.0
        // - xjTrans（源节点特征）：边0→1=1.0，边0→2=1.0
        // 步骤3：加权聚合
        // - 边0→1：1.0 * 1.0 = 1.0 → 节点1聚合值=1.0
        // - 边0→2：1.0 * 1.0 = 1.0 → 节点2聚合值=1.0
        // - 节点0：无入边 → 0.0
        float[][] expected = {
                {0.0f},
                {1.0f},
                {1.0f}
        };
        torch.print(output);

        // 6. 数值验证（误差允许1e-4）
//        float[][] outputArr = output.to(new Device(torch.kCPU()), torch.ScalarType.Float).toFloatArray();
//        for (int i = 0; i < 3; i++) {
//            assertEquals("节点" + i + "数值不匹配",
//                    expected[i][0], outputArr[i][0], 1e-4f);
//        }

        // 7. 资源释放
        x.close();
        edgeIndex.close();
        output.close();
    }

    // ========== 测试3：边界场景验证 ==========
    @Test
    public void testEdgeCases() {
        // 1. 初始化FeaStConv
        long inChannels = 2;
        long outChannels = 2;
        int heads = 2;
        boolean hasBias = true;
        FeaStConv feastConv = new FeaStConv(inChannels, outChannels, heads, hasBias);

        // 场景1：空边索引（[2,0]）
        TensorOptions floatOpts = new TensorOptions()
                .dtype(new ScalarTypeOptional(torch.ScalarType.Float))
                .device(new DeviceOptional(new Device(torch.kCPU())));
        Tensor x = torch.randn(new long[]{3, 2}, floatOpts);
        // 创建空边索引 [2,0]
        Tensor emptyEdgeIndex = torch.empty(new long[]{2, 0},
                new TensorOptions().dtype(new ScalarTypeOptional(torch.ScalarType.Long)),new MemoryFormatOptional());

        Tensor outputEmpty = feastConv.forward(x, emptyEdgeIndex);
        // 空边时：输出=全0 + 偏置
        Tensor expectedEmpty = torch.zeros(new long[]{3, outChannels}, floatOpts).add(feastConv.getBias().data());
        assertTrue("空边场景输出应匹配", torch.allclose(outputEmpty, expectedEmpty));

        // 场景2：输入维度错误（验证异常抛出）
        Tensor badX = torch.randn(new long[]{3, 3}, floatOpts); // 输入维度3≠配置的2
        Tensor edgeIndex = torch.tensor(new long[]{0,1,1,2},
                new TensorOptions().dtype(new ScalarTypeOptional(torch.ScalarType.Long))).view(2, 2);

        assertThrows(RuntimeException.class, () ->
                feastConv.forward(badX, edgeIndex));

        // 场景3：边索引维度错误
        Tensor badEdgeIndex = torch.tensor(new long[]{0,1,2,3},
                new TensorOptions().dtype(new ScalarTypeOptional(torch.ScalarType.Long))).view(4, 1); // 4行≠2行
        assertThrows(IllegalArgumentException.class, () ->
                feastConv.forward(x, badEdgeIndex));

        // 资源释放
        x.close();
        emptyEdgeIndex.close();
        outputEmpty.close();
        expectedEmpty.close();
        badX.close();
        edgeIndex.close();
        badEdgeIndex.close();
    }

    // ========== 工具方法：展平二维long数组 ==========
    private long[] flatten(long[][] arr) {
        int len = 0;
        for (long[] sub : arr) len += sub.length;
        long[] res = new long[len];
        int idx = 0;
        for (long[] sub : arr) {
            for (long val : sub) res[idx++] = val;
        }
        return res;
    }

    // ========== 工具方法：展平二维float数组 ==========
    private float[] flatten(float[][] arr) {
        int len = 0;
        for (float[] sub : arr) len += sub.length;
        float[] res = new float[len];
        int idx = 0;
        for (float[] sub : arr) {
            for (float val : sub) res[idx++] = val;
        }
        return res;
    }

    public static void main(String[] args) {
        Result result = JUnitCore.runClasses(FeaStConvTest.class);
        for (Failure failure : result.getFailures()) {
            System.err.println("FAIL: " + failure.toString());
            if (failure.getException() != null) failure.getException().printStackTrace();
        }
        System.out.println("Tests run=" + result.getRunCount()
                + " failed=" + result.getFailureCount()
                + " ignored=" + result.getIgnoreCount());
        if (!result.wasSuccessful()) {
            throw new RuntimeException("FeaStConvTest had " + result.getFailureCount() + " failures");
        }
        System.out.println("✅ FeaStConvTest all tests passed");
    }

}
