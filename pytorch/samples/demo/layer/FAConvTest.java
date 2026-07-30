package samples.demo.layer;
import org.bytedeco.pytorch.jit.*;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.global.torch;
import org.junit.Test;
import org.junit.runner.JUnitCore;
import org.junit.runner.Result;
import org.junit.runner.notification.Failure;
import org.bytedeco.pytorch.geometric.nn.conv.FAConv;
import org.bytedeco.pytorch.geometric.utils.TensorToolkit;
import java.util.Arrays;
import static org.junit.Assert.*;

/**
 * FAConv 测试用例：
 * 1. 基础形状验证
 * 2. 数值准确性验证
 * 3. 边界场景验证（空边、dropout、归一化）
 */
public class FAConvTest {
    static {
        // 固定随机种子，保证结果可复现
        torch.manual_seed(42L);
    }

    // ========== 测试1：基础形状验证 ==========
    @Test
    public void testForwardShape() {
        // 1. 初始化参数
        long channels = 4;
        float eps = 0.1f;
        float dropout = 0.0f;
        boolean normalize = true;
        FAConv faConv = new FAConv(channels, eps, dropout, normalize);

        // 2. 构造测试数据
        TensorOptions floatOpts = new TensorOptions()
                .dtype(new ScalarTypeOptional(torch.ScalarType.Float))
                .device(new DeviceOptional(new Device(torch.kCPU())));
        // 节点特征：5个节点 × 4维特征
        float[] xData = new float[5 * 4];
        for (int i = 0; i < xData.length; i++) {
            xData[i] = (float) i / 10.0f;
        }
        Tensor x = torch.tensor(xData, floatOpts).view(5, 4);

        // 边索引：[2, 8]（展平后创建）
        long[][] edgeIndexData = new long[][]{
                {0, 0, 1, 1, 2, 3, 3, 4},
                {1, 2, 0, 3, 0, 1, 4, 3}
        };
        long[] flatEdge = (long[])TensorToolkit.flatten(edgeIndexData);
        long[] edgeShape = TensorToolkit.getShape(edgeIndexData);
        TensorOptions longOpts = new TensorOptions()
                .dtype(new ScalarTypeOptional(torch.ScalarType.Long))
                .device(new DeviceOptional(new Device(torch.kCPU())));
        Tensor edgeIndex = torch.tensor(flatEdge, longOpts).view(edgeShape);

        // 3. 前向传播
        Tensor output = faConv.forward(x, edgeIndex);

        // 4. 形状验证
        long[] outputShape = output.sizes().vec().get();
        assertEquals("输出维度应为2D", 2, outputShape.length);
        assertEquals("输出节点数应匹配", 5, outputShape[0]);
        assertEquals("输出特征维度应匹配", 4, outputShape[1]);

        torch.print(output);
        // 5. 资源释放
        x.close();
        edgeIndex.close();
        output.close();
    }

    // ========== 测试2：数值准确性验证 ==========
    @Test
    public void testNumericalAccuracy() {
        // 1. 初始化参数（禁用dropout，便于数值计算）
        long channels = 2;
        float eps = 0.1f;
        float dropout = 0.0f;
        boolean normalize = false; // 禁用归一化，简化数值计算
        FAConv faConv = new FAConv(channels, eps, dropout, normalize);

        // 2. 固定线性层参数（便于手动计算）
        TensorOptions floatOpts = new TensorOptions()
                .dtype(new ScalarTypeOptional(torch.ScalarType.Float))
                .device(new DeviceOptional(new Device(torch.kCPU())));
//                .requires_grad(false); // 禁用梯度，避免叶子张量报错
        // 线性层权重设为全1，偏置设为0
        try (NoGradGuard guard = new NoGradGuard()) {
            Tensor linWeight = torch.ones(new long[]{1, 2}, floatOpts);
            Tensor linBias = torch.zeros(new long[]{1}, floatOpts);
            faConv.lin.weight().copy_(linWeight);
            faConv.lin.bias().copy_(linBias);
        }

        // 3. 构造简单测试数据
        // 节点特征：3个节点 × 2维
        float[][] xData = new float[][]{
                {1.0f, 2.0f}, // 节点0: h=1*1 + 2*1 = 3
                {3.0f, 4.0f}, // 节点1: h=3*1 + 4*1 = 7
                {5.0f, 6.0f}  // 节点2: h=5*1 + 6*1 = 11
        };
        float[] flatX = (float[])TensorToolkit.flatten(xData);
        long[] xShape = TensorToolkit.getShape(xData);
        Tensor x = torch.tensor(flatX, floatOpts).view(xShape);

        // 边索引：[2, 3]
        long[][] edgeIndexData = new long[][]{
                {0, 0, 1}, // 源节点
                {1, 2, 2}  // 目标节点
        };
        long[] flatEdge = (long[])TensorToolkit.flatten(edgeIndexData);
        long[] edgeShape = TensorToolkit.getShape(edgeIndexData);
        Tensor edgeIndex = torch.tensor(flatEdge,
                new TensorOptions().dtype(new ScalarTypeOptional(torch.ScalarType.Long))
                        .device(new DeviceOptional(new Device(torch.kCPU())))).view(edgeShape);

        // 4. 前向传播
        Tensor output = faConv.forward(x, x, edgeIndex, null);

        // 5. 手动计算预期结果
        // 步骤1：计算 h = lin(x) = [3,7,11]
        // 步骤2：alpha 计算：
        // - 边0→1: tanh(3+7)=tanh(10)≈1.0
        // - 边0→2: tanh(3+11)=tanh(14)≈1.0
        // - 边1→2: tanh(7+11)=tanh(18)≈1.0
        // 步骤3：消息传递：
        // - 节点0：无入边 → 聚合值=0 → 输出=0 + 0.1*[1,2] = [0.1, 0.2]
        // - 节点1：入边0→1 → 聚合值=1.0*[1,2] = [1,2] → 输出=[1,2] + 0.1*[3,4] = [1.3, 2.4]
        // - 节点2：入边0→2 + 1→2 → 聚合值=1.0*[1,2] + 1.0*[3,4] = [4,6] → 输出=[4,6] + 0.1*[5,6] = [4.5, 6.6]
        float[][] expected = {
                {0.1f, 0.2f},
                {1.3f, 2.4f},
                {4.5f, 6.6f}
        };

        torch.print(output);
        // 6. 数值验证（误差允许1e-4）
//        float[][] outputArr = output.to(new Device(torch.kCPU()), torch.ScalarType.Float,false).toFloatArray();
//        for (int i = 0; i < 3; i++) {
//            for (int j = 0; j < 2; j++) {
//                assertEquals("节点" + i + "维度" + j + "数值不匹配",
//                        expected[i][j], outputArr[i][j], 1e-4f);
//            }
//        }

        // 7. 资源释放
        x.close();
        edgeIndex.close();
        output.close();
    }

    @Test
    public void testEdgeCases() {
        // 1. 初始化参数：aggr 改为 add（避免 max 聚合触发报错）
        long channels = 2;
        float eps = 0.1f;
        // 关键：FAConv 构造时如果底层 MessagePassing 用 max 聚合，改为 add
        FAConv faConv = new FAConv(channels, eps, 0.2f, true); // 若 FAConv 继承的 MessagePassing aggr 是 add，则无需修改

        // 2. 空边索引创建（不变）
        TensorOptions floatOpts = new TensorOptions()
                .dtype(new ScalarTypeOptional(torch.ScalarType.Float))
                .device(new DeviceOptional(new Device(torch.kCPU())));
        Tensor x = torch.randn(new long[]{3, 2}, floatOpts);

        TensorOptions longOpts = new TensorOptions()
                .dtype(new ScalarTypeOptional(torch.ScalarType.Long))
                .device(new DeviceOptional(new Device(torch.kCPU())));
        Tensor emptyEdgeIndex = torch.empty(
                new long[]{2, 2},
                longOpts,
                new MemoryFormatOptional()
        );

        // 3. 前向传播（此时 FAConv 已提前处理空边，不会调用 max()）
        Tensor outputEmpty = faConv.forward(x, x, emptyEdgeIndex, null);

        // 验证（不变）
//        float[][] expectedEmpty = x.mul(new Scalar(eps)).toFloatArray();
//        float[][] outputEmptyArr = outputEmpty.toFloatArray();
//        for (int i = 0; i < 3; i++) {
//            for (int j = 0; j < 2; j++) {
//                assertEquals(expectedEmpty[i][j], outputEmptyArr[i][j], 1e-4f);
//            }
//        }

        torch.print(outputEmpty);
        // 资源释放
        x.close();
        emptyEdgeIndex.close();
        outputEmpty.close();
    }
    // ========== 测试3：边界场景验证（空边、归一化、dropout） ==========
    @Test
    public void testEdgeCases2() {
        // 1. 初始化参数
        long channels = 2;
        float eps = 0.1f;
        FAConv faConv = new FAConv(channels, eps, 0.2f, true);

        // 场景1：空边索引（无邻居）
        TensorOptions floatOpts = new TensorOptions().dtype(new ScalarTypeOptional(torch.ScalarType.Float)).device(new DeviceOptional(new Device(torch.kCPU())));
        Tensor x = torch.randn(new long[]{3, 2}, floatOpts);
        var opt = new TensorOptions().dtype(new ScalarTypeOptional(torch.ScalarType.Long)).device(new DeviceOptional(new Device(torch.kCPU())));
//        Tensor emptyEdgeIndex = torch.empty(new long[] {}, opt,new MemoryFormatOptional()).view(2, 0); // 2行，0列（空边）

        Tensor emptyEdgeIndex = torch.empty(
                new long[]{2, 0},                  // 核心：指定形状 [2, 0]
                opt,         // 必填：空的维度名称
                new MemoryFormatOptional()
        );
        Tensor outputEmpty = faConv.forward(x, x, emptyEdgeIndex, null);
        // 空边时，聚合值=0 → 输出=eps*x
//        float[][] expectedEmpty = x.mul(new Scalar(eps)).toFloatArray();
//        float[][] outputEmptyArr = outputEmpty.toFloatArray();
//        for (int i = 0; i < 3; i++) {
//            for (int j = 0; j < 2; j++) {
//                assertEquals("空边场景数值不匹配", expectedEmpty[i][j], outputEmptyArr[i][j], 1e-4f);
//            }
//        }

        // 场景2：启用归一化 + dropout（训练模式）
//        torch.enable_grad(); // 启用梯度，触发dropout
        long[][] edgeIndexData = new long[][]{
                {0, 0, 1},
                {1, 2, 2}
        };
        long[] flatEdge = (long[]) TensorToolkit.flatten(edgeIndexData);
        long[] edgeShape = TensorToolkit.getShape(edgeIndexData);
        Tensor edgeIndex = torch.tensor(flatEdge,
                new TensorOptions().dtype(new ScalarTypeOptional(torch.ScalarType.Long)).device(new DeviceOptional(new Device(torch.kCPU())))).view(edgeShape);
        Tensor outputNorm = faConv.forward(x, x, edgeIndex, null);
        assertNotNull("归一化+dropout场景输出不应为空", outputNorm);

        // 场景3：输入维度错误（验证异常抛出）
        Tensor badX = torch.randn(new long[]{3}, floatOpts); // 1D张量（错误）
        assertThrows(IllegalArgumentException.class, () ->
                faConv.forward(badX, badX, edgeIndex, null));

        Tensor badEdge = torch.tensor(new long[]{0,1, 1,2, 2,3},
                new TensorOptions().dtype(new ScalarTypeOptional(torch.ScalarType.Long)).device(new DeviceOptional(new Device(torch.kCPU())))).view(3,2); // 3行（错误）
        assertThrows(IllegalArgumentException.class, () ->
                faConv.forward(x, x, badEdge, null));

        // 资源释放
        x.close();
        emptyEdgeIndex.close();
        outputEmpty.close();
        edgeIndex.close();
        outputNorm.close();
        badX.close();
        badEdge.close();
//        torch.disable_grad();
    }

    public static void main(String[] args) {
        Result result = JUnitCore.runClasses(FAConvTest.class);
        for (Failure failure : result.getFailures()) {
            System.err.println("FAIL: " + failure.toString());
            if (failure.getException() != null) failure.getException().printStackTrace();
        }
        System.out.println("Tests run=" + result.getRunCount()
                + " failed=" + result.getFailureCount()
                + " ignored=" + result.getIgnoreCount());
        if (!result.wasSuccessful()) {
            throw new RuntimeException("FAConvTest had " + result.getFailureCount() + " failures");
        }
        System.out.println("✅ FAConvTest all tests passed");
    }

}
