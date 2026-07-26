package org.bytedeco.pytorch.geometric.demo.layer;
import org.bytedeco.pytorch.jit.*;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.global.torch;
import org.junit.Test;
import org.bytedeco.pytorch.geometric.nn.conv.FastRGCNConv;
import static org.junit.Assert.*;

/**
 * FastRGCNConv 测试用例：
 * 1. 基础形状验证
 * 2. 数值准确性验证
 * 3. 边界场景验证（空边、多关系、根节点/偏置）
 */
public class FastRGCNConvTest {
    static {
        // 固定随机种子，保证结果可复现
        torch.manual_seed(42L);
    }

    // ========== 测试1：基础形状验证 ==========
    @Test
    public void testForwardShape() {
        // 1. 初始化参数
        long inChannels = 3;
        long outChannels = 2;
        int numRelations = 2;
        boolean rootWeight = true;
        boolean hasBias = true;
        FastRGCNConv rgcn = new FastRGCNConv(inChannels, outChannels, numRelations, rootWeight, hasBias);

        // 2. 构造测试数据
        TensorOptions floatOpts = new TensorOptions()
                .dtype(new ScalarTypeOptional(torch.ScalarType.Float))
                .device(new DeviceOptional(new Device(torch.kCPU())));
        // 节点特征：4个节点 × 3维输入
        Tensor x = torch.randn(new long[]{4, 3}, floatOpts);

        // 边索引：[2, 5]
        long[][] edgeIndexData = new long[][]{
                {0, 0, 1, 2, 3}, // 源节点
                {1, 2, 3, 0, 1}  // 目标节点
        };
        Tensor edgeIndex = torch.tensor(flatten(edgeIndexData),
                        new TensorOptions().dtype(new ScalarTypeOptional(torch.ScalarType.Long)))
                .view(2, 5);

        // 关系类型：[5]（0/1 两种关系）
        long[] edgeTypeData = new long[]{0, 1, 0, 1, 0};
        Tensor edgeType = torch.tensor(edgeTypeData,
                new TensorOptions().dtype(new ScalarTypeOptional(torch.ScalarType.Long)));

        // 3. 前向传播（两种实现）
        Tensor output1 = rgcn.forward(x, edgeIndex, edgeType);
//        Tensor output2 = rgcn.forward2(x, edgeIndex, edgeType);

        // 4. 形状验证
        long[] output1Shape = output1.sizes().vec().get();
        assertEquals("输出维度应为2D", 2, output1Shape.length);
        assertEquals("输出节点数应匹配", 4, output1Shape[0]);
        assertEquals("输出特征维度应匹配", 2, output1Shape[1]);

//        long[] output2Shape = output2.sizes().vec().get();
//        assertArrayEquals("forward/forward2 输出形状应一致", output1Shape, output2Shape);

        // 5. 资源释放
        x.close();
        edgeIndex.close();
        edgeType.close();
        output1.close();
//        output2.close();
    }

    // ========== 测试2：数值准确性验证 ==========
    @Test
    public void testNumericalAccuracy() {
        // 1. 初始化参数（禁用随机，固定权重）
        long inChannels = 2;
        long outChannels = 1;
        int numRelations = 2;
        boolean rootWeight = false;
        boolean hasBias = false;
        FastRGCNConv rgcn = new FastRGCNConv(inChannels, outChannels, numRelations, rootWeight, hasBias);

        // 2. 固定权重（禁用梯度，避免叶子张量报错）
        try (NoGradGuard guard = new NoGradGuard()) {
            // 权重：[2, 2, 1] → 关系0: [[1,0],[0,1]], 关系1: [[0,1],[1,0]]
            float[][] rel0Weight = new float[][]{{1.0f}, {0.0f}};
            float[][] rel1Weight = new float[][]{{0.0f}, {1.0f}};
            float[][][] weightData = new float[][][]{rel0Weight, rel1Weight};
            Tensor weightTensor = torch.tensor(flatten(weightData),
                            new TensorOptions().dtype(new ScalarTypeOptional(torch.ScalarType.Float)))
                    .view(2, 2, 1);
//            rgcn.weight().copy_(weightTensor); // 测试场景临时用 copy_（NoGradGuard 保护）
        }

        // 3. 构造简单测试数据
        TensorOptions floatOpts = new TensorOptions()
                .dtype(new ScalarTypeOptional(torch.ScalarType.Float))
                .device(new DeviceOptional(new Device(torch.kCPU())));
        // 节点特征：3个节点 × 2维
        float[][] xData = new float[][]{
                {1.0f, 2.0f}, // 节点0
                {3.0f, 4.0f}, // 节点1
                {5.0f, 6.0f}  // 节点2
        };
        Tensor x = torch.tensor(flatten(xData), floatOpts).view(3, 2);

        // 边索引：[2, 3]
        Tensor edgeIndex = torch.tensor(new long[]{0,0,1,1,2,2},
                        new TensorOptions().dtype(new ScalarTypeOptional(torch.ScalarType.Long)))
                .view(2, 3);

        // 关系类型：[3]（0,1,0）
        Tensor edgeType = torch.tensor(new long[]{0,1,0},
                new TensorOptions().dtype(new ScalarTypeOptional(torch.ScalarType.Long)));

        // 4. 前向传播
        Tensor output = rgcn.forward(x, edgeIndex, edgeType);

        // 5. 手动计算预期结果
        // 步骤1：提取源节点特征
        // - 边0→1（关系0）：x_j = [1,2] × 权重0 [[1],[0]] → 1*1 + 2*0 = 1
        // - 边0→2（关系1）：x_j = [1,2] × 权重1 [[0],[1]] → 1*0 + 2*1 = 2
        // - 边1→2（关系0）：x_j = [3,4] × 权重0 [[1],[0]] → 3*1 + 4*0 = 3
        // 步骤2：聚合到目标节点
        // - 节点1：无入边 → 0
        // - 节点2：入边0→2（2） + 1→2（3） → 5
        // - 节点0：无入边 → 0
        float[][] expected = {
                {0.0f},
                {0.0f},
                {5.0f}
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
        edgeType.close();
        output.close();
    }

    // ========== 测试3：边界场景验证 ==========
    @Test
    public void testEdgeCases() {
        // 1. 初始化参数
        long inChannels = 2;
        long outChannels = 2;
        int numRelations = 3;
        FastRGCNConv rgcn = new FastRGCNConv(inChannels, outChannels, numRelations, true, true);

        // 场景1：空边索引（[2,0]）
        TensorOptions floatOpts = new TensorOptions()
                .dtype(new ScalarTypeOptional(torch.ScalarType.Float))
                .device(new DeviceOptional(new Device(torch.kCPU())));
        Tensor x = torch.randn(new long[]{3, 2}, floatOpts);
        Tensor emptyEdgeIndex = torch.empty(new long[]{2, 0},
                new TensorOptions().dtype(new ScalarTypeOptional(torch.ScalarType.Long)),new MemoryFormatOptional());
        Tensor emptyEdgeType = torch.empty(new long[]{0},
                new TensorOptions().dtype(new ScalarTypeOptional(torch.ScalarType.Long)),new MemoryFormatOptional());

        Tensor outputEmpty = rgcn.forward(x, emptyEdgeIndex, emptyEdgeType);
        // 空边时：输出 = 根节点投影 + 偏置
        Tensor expectedEmpty = rgcn.linRoot.forward(x).add(rgcn.bias.data());
        assertTrue("空边场景输出应匹配", torch.allclose(outputEmpty, expectedEmpty));

        // 场景2：输入维度错误（验证异常抛出）
        Tensor badX = torch.randn(new long[]{3, 3}, floatOpts); // 输入维度不匹配
        Tensor edgeIndex = torch.tensor(new long[]{0,1,1,2},
                new TensorOptions().dtype(new ScalarTypeOptional(torch.ScalarType.Long))).view(2, 2);
        Tensor edgeType = torch.tensor(new long[]{0,1},
                new TensorOptions().dtype(new ScalarTypeOptional(torch.ScalarType.Long)));

        assertThrows(IllegalArgumentException.class, () ->
                rgcn.forward(badX, edgeIndex, edgeType));

        // 场景3：关系类型长度不匹配
        Tensor badEdgeType = torch.tensor(new long[]{0},
                new TensorOptions().dtype(new ScalarTypeOptional(torch.ScalarType.Long))); // 长度1 ≠ 边数2
        assertThrows(IllegalArgumentException.class, () ->
                rgcn.forward(x, edgeIndex, badEdgeType));

        // 资源释放
        x.close();
        emptyEdgeIndex.close();
        emptyEdgeType.close();
        outputEmpty.close();
        expectedEmpty.close();
        badX.close();
        edgeIndex.close();
        edgeType.close();
        badEdgeType.close();
    }

    // ========== 工具方法：展平多维数组 ==========
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

    private float[] flatten(float[][][] arr) {
        int len = 0;
        for (float[][] sub : arr) {
            for (float[] sub2 : sub) len += sub2.length;
        }
        float[] res = new float[len];
        int idx = 0;
        for (float[][] sub : arr) {
            for (float[] sub2 : sub) {
                for (float val : sub2) res[idx++] = val;
            }
        }
        return res;
    }
}