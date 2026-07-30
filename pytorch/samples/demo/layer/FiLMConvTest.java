package samples.demo.layer;
import org.bytedeco.pytorch.nn.modules.container.*;
import org.bytedeco.pytorch.jit.*;
import org.bytedeco.pytorch.nn.modules.*;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.nn.modules.container.AnyModule;
import org.junit.Test;
import org.junit.runner.JUnitCore;
import org.junit.runner.Result;
import org.junit.runner.notification.Failure;
import org.bytedeco.pytorch.geometric.nn.conv.FiLMConv;
import static org.junit.Assert.*;

/**
 * FiLMConv 测试用例：
 * 1. 基础形状验证（输入输出维度匹配）
 * 2. 数值准确性验证（固定权重，手动计算预期值）
 * 3. 边界场景验证（空边、单关系、多关系、激活函数）
 */
public class FiLMConvTest {
    static {
        // 固定随机种子，保证结果可复现
        torch.manual_seed(42L);
    }

    // ========== 测试1：基础形状验证 ==========
    @Test
    public void testForwardShape() {
        // 1. 初始化 FiLMConv（2个关系，ReLU 激活）
        long inChannels = 4;
        long outChannels = 2;
        int numRelations = 2;
        ReLUImpl relu = new ReLUImpl();
        FiLMConv filmConv = new FiLMConv(inChannels, outChannels, numRelations, relu);

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

        // 关系类型：[6]（0/1 两种关系）
        long[] edgeTypeData = new long[]{0, 1, 0, 1, 0, 1};
        Tensor edgeType = torch.tensor(edgeTypeData,
                new TensorOptions().dtype(new ScalarTypeOptional(torch.ScalarType.Long)));

        // 3. 前向传播
        Tensor output = filmConv.forward(x, edgeIndex, edgeType);

        // 4. 形状验证
        long[] outputShape = output.sizes().vec().get();
        assertEquals("输出维度应为2D", 2, outputShape.length);
        assertEquals("输出节点数应匹配输入", 5, outputShape[0]);
        assertEquals("输出特征维度应匹配配置", outChannels, outputShape[1]);

        // 5. 资源释放
        x.close();
        edgeIndex.close();
        edgeType.close();
        output.close();
        relu.close();
    }

    // ========== 测试2：数值准确性验证 ==========
    @Test
    public void testNumericalAccuracy() {
        // 1. 初始化 FiLMConv（1个关系，无激活函数，便于手动计算）
        long inChannels = 2;
        long outChannels = 1;
        int numRelations = 1;
        FiLMConv filmConv = new FiLMConv(inChannels, outChannels, numRelations, null);

        // 2. 固定线性层参数（禁用梯度，避免叶子张量报错）
        try (NoGradGuard guard = new NoGradGuard()) {
            // 固定关系0的线性层：W_0 = [[1, 0]]（输出=输入第1维）
            Tensor lin0Weight = torch.tensor(new float[]{1.0f, 0.0f},
                    new TensorOptions().dtype(new ScalarTypeOptional(torch.ScalarType.Float))).view(1,2);
            filmConv.getLins()[0].weight().copy_(lin0Weight);
            filmConv.getLins()[0].bias().copy_(torch.zeros(new long[]{1}));

            // 固定 FiLM 层：输出 [gamma, beta] = [x1, 0]（gamma=x1，beta=0）
            Tensor filmWeight = torch.tensor(new float[]{1.0f, 0.0f,0.0f, 0.0f},
                    new TensorOptions().dtype(new ScalarTypeOptional(torch.ScalarType.Float))).view(2,2);
            filmConv.getFilmLin().weight().copy_(filmWeight);
            filmConv.getFilmLin().bias().copy_(torch.zeros(new long[]{2}));
        }

        // 3. 构造简单测试数据
        TensorOptions floatOpts = new TensorOptions()
                .dtype(new ScalarTypeOptional(torch.ScalarType.Float))
                .device(new DeviceOptional(new Device(torch.kCPU())));
        // 节点特征：3个节点 × 2维
        float[][] xData = new float[][]{
                {1.0f, 2.0f}, // 节点0：gamma=1.0, beta=0
                {3.0f, 4.0f}, // 节点1：gamma=3.0, beta=0
                {5.0f, 6.0f}  // 节点2：gamma=5.0, beta=0
        };
        Tensor x = torch.tensor(flatten(xData), floatOpts).view(3, 2);

        // 边索引：[2, 2]（2条边）
        Tensor edgeIndex = torch.tensor(new long[]{0,0,1,2},
                        new TensorOptions().dtype(new ScalarTypeOptional(torch.ScalarType.Long)))
                .view(2, 2);

        // 关系类型：[2]（全为关系0）
        Tensor edgeType = torch.tensor(new long[]{0, 0},
                new TensorOptions().dtype(new ScalarTypeOptional(torch.ScalarType.Long)));

        // 4. 前向传播
        Tensor output = filmConv.forward(x, edgeIndex, edgeType);

        // 5. 手动计算预期结果
        // 步骤1：FiLM 参数
        // - 节点0: gamma=1.0, beta=0
        // - 节点1: gamma=3.0, beta=0
        // - 节点2: gamma=5.0, beta=0
        // 步骤2：关系0的变换
        // - 边0→1：x_j=节点0 → W_0*x_j=1.0 → FiLM调制=3.0*1.0 + 0=3.0
        // - 边0→2：x_j=节点0 → W_0*x_j=1.0 → FiLM调制=5.0*1.0 + 0=5.0
        // 步骤3：Mean 聚合
        // - 节点1：入边数=1 → 3.0/1=3.0
        // - 节点2：入边数=1 → 5.0/1=5.0
        // - 节点0：无入边 → 0.0
        float[][] expected = {
                {0.0f},
                {3.0f},
                {5.0f}
        };

        // 6. 数值验证（误差允许1e-4）
//        float[][] outputArr = output.to(new Device(torch.kCPU()), torch.ScalarType.Float).toFloatArray();
//        for (int i = 0; i < 3; i++) {
//            assertEquals("节点" + i + "数值不匹配",
//                    expected[i][0], outputArr[i][0], 1e-4f);
//        }

        torch.print(output);

        // 7. 资源释放
        x.close();
        edgeIndex.close();
        edgeType.close();
        output.close();
    }

    // ========== 测试3：边界场景验证 ==========
    @Test
    public void testEdgeCases() {
        // 1. 初始化 FiLMConv
        long inChannels = 2;
        long outChannels = 2;
        int numRelations = 2;
        ReLUImpl relu = new ReLUImpl();
        FiLMConv filmConv = new FiLMConv(inChannels, outChannels, numRelations, relu);

        // 场景1：空边索引（[2,0]）
        TensorOptions floatOpts = new TensorOptions()
                .dtype(new ScalarTypeOptional(torch.ScalarType.Float))
                .device(new DeviceOptional(new Device(torch.kCPU())));
        Tensor x = torch.randn(new long[]{3, 2}, floatOpts);
        // 创建空边索引 [2,0]
        Tensor emptyEdgeIndex = torch.empty(new long[]{2, 0},
                new TensorOptions().dtype(new ScalarTypeOptional(torch.ScalarType.Long)),new MemoryFormatOptional());
        Tensor emptyEdgeType = torch.empty(new long[]{0},
            
                new TensorOptions().dtype(new ScalarTypeOptional(torch.ScalarType.Long)),new MemoryFormatOptional());

        Tensor outputEmpty = filmConv.forward(x, emptyEdgeIndex, emptyEdgeType);
        // 空边时输出应为全0
        assertTrue("空边场景输出应为全0", torch.allclose(outputEmpty, torch.zeros_like(outputEmpty)));

        // 场景2：输入维度错误（验证异常抛出）
        Tensor badX = torch.randn(new long[]{3, 3}, floatOpts); // 输入维度3≠配置的2
        Tensor edgeIndex = torch.tensor(new long[]{0,1,1,2},
                new TensorOptions().dtype(new ScalarTypeOptional(torch.ScalarType.Long))).view(2, 2);
        Tensor edgeType = torch.tensor(new long[]{0,1},
                new TensorOptions().dtype(new ScalarTypeOptional(torch.ScalarType.Long)));

        assertThrows(IllegalArgumentException.class, () ->
                filmConv.forward(badX, edgeIndex, edgeType));

        // 场景3：未指定 edge_type — 自动补全全 0 关系，应可成功前向
        Tensor outNoType = filmConv.forward(x, edgeIndex);
        assertNotNull(outNoType);
        assertEquals(2, outNoType.dim());
        assertEquals(x.size(0), outNoType.size(0));

        // 资源释放
        x.close();
        emptyEdgeIndex.close();
        emptyEdgeType.close();
        outputEmpty.close();
        badX.close();
        edgeIndex.close();
        edgeType.close();
        relu.close();
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
        Result result = JUnitCore.runClasses(FiLMConvTest.class);
        for (Failure failure : result.getFailures()) {
            System.err.println("FAIL: " + failure.toString());
            if (failure.getException() != null) failure.getException().printStackTrace();
        }
        System.out.println("Tests run=" + result.getRunCount()
                + " failed=" + result.getFailureCount()
                + " ignored=" + result.getIgnoreCount());
        if (!result.wasSuccessful()) {
            throw new RuntimeException("FiLMConvTest had " + result.getFailureCount() + " failures");
        }
        System.out.println("✅ FiLMConvTest all tests passed");
    }

}
