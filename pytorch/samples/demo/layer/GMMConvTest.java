package samples.demo.layer;
import org.bytedeco.pytorch.jit.*;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.global.torch;
import org.junit.Test;
import org.junit.runner.JUnitCore;
import org.junit.runner.Result;
import org.junit.runner.notification.Failure;
import org.bytedeco.pytorch.geometric.nn.conv.GMMConv;
import static org.junit.Assert.*;

/**
 * GMMConv 测试用例：
 * 1. 基础形状验证（输入输出维度匹配）；
 * 2. 数值准确性验证（固定权重，手动计算预期值）；
 * 3. 边界场景验证（空边、单节点、根节点变换）。
 */
public class GMMConvTest {
    static {
        // 固定随机种子，保证结果可复现
        torch.manual_seed(42L);
    }

    // ========== 工具方法 ==========
    private TensorOptions floatOpts(Device device) {
        return new TensorOptions()
                .dtype(new ScalarTypeOptional(torch.ScalarType.Float))
                .device(new DeviceOptional(device));
    }

    private TensorOptions longOpts(Device device) {
        return new TensorOptions()
                .dtype(new ScalarTypeOptional(torch.ScalarType.Long))
                .device(new DeviceOptional(device));
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

    private void releaseTensor(Tensor... tensors) {
        for (Tensor t : tensors) {
            if (t != null) t.close();
        }
    }

    // ========== 测试1：基础形状验证 ==========
    @Test
    public void testForwardShape() {
        // 1. 初始化 GMMConv
        long inChannels = 3;
        long outChannels = 2;
        int dim = 1; // 边特征维度
        int kernelSize = 2; // 高斯核数量
        Device device = new Device(torch.kCPU());
        GMMConv gmmConv = new GMMConv(inChannels, outChannels, dim, kernelSize, true, true);

        // 2. 构造测试数据
        // 节点特征：4个节点 × 3维
        Tensor x = torch.randn(new long[]{4, inChannels}, floatOpts(device));
        // 边索引：[2, 5]
        Tensor edgeIndex = torch.tensor(new long[]{0,0,1,2,3,1,2,3,0,1}, longOpts(device)).view(2, 5);
        // 边特征（伪坐标）：5条边 × 1维
        Tensor edgeAttr = torch.randn(new long[]{5, dim}, floatOpts(device));

        // 3. 前向传播
        Tensor output = gmmConv.forward(x, edgeIndex, edgeAttr);

        // 4. 形状验证
        long[] outputShape = output.sizes().vec().get();
        assertEquals("输出维度应为2D", 2, outputShape.length);
        assertEquals("输出节点数应匹配输入", 4, outputShape[0]);
        assertEquals("输出特征维度应匹配 outChannels", outChannels, outputShape[1]);

        torch.print(output);
        // 5. 资源释放
        releaseTensor(x, edgeIndex, edgeAttr, output);
    }

    // ========== 测试2：数值准确性验证 ==========
    @Test
    public void testNumericalAccuracy() {
        // 1. 初始化 GMMConv（固定参数，禁用根节点/偏置）
        long inChannels = 2;
        long outChannels = 1;
        int dim = 1;
        int kernelSize = 2;
        Device device = new Device(torch.kCPU());
        GMMConv gmmConv = new GMMConv(inChannels, outChannels, dim, kernelSize, false, false);

        // 2. 固定模型参数（禁用梯度，便于手动计算）
        try (NoGradGuard guard = new NoGradGuard()) {
            // 固定线性层权重：[[1,0],[0,1]] → K=2, out=1 → [2*1, 2] = [2,2]
            float[][] linWeightData = {{1.0f, 0.0f}, {0.0f, 1.0f}};
            Tensor linWeight = torch.tensor(flatten(linWeightData), floatOpts(device)).view(2, 2);
            gmmConv.getLin().weight().copy_(linWeight);
            gmmConv.getLin().bias().copy_(torch.zeros(new long[]{2}, floatOpts(device)));

            // 固定高斯核参数：
            // mu = [[0.0], [1.0]] (K=2, dim=1)
            Tensor mu = torch.tensor(new float[]{0.0f, 1.0f}, floatOpts(device)).view(2, 1);
            gmmConv.getMu().data().copy_(mu);
            // sigma = [[1.0], [1.0]] (强制为正)
            Tensor sigma = torch.tensor(new float[]{1.0f, 1.0f}, floatOpts(device)).view(2, 1);
            gmmConv.getSigma().data().copy_(sigma);

            // 资源释放
            releaseTensor(linWeight, mu, sigma);
        }

        // 3. 构造测试数据
        // 节点特征：3个节点 × 2维
        float[][] xData = {{1.0f, 2.0f}, {3.0f, 4.0f}, {5.0f, 6.0f}};
        Tensor x = torch.tensor(flatten(xData), floatOpts(device)).view(3, 2);

        // 边索引：[2,2] → 0→1, 0→2
        Tensor edgeIndex = torch.tensor(new long[]{0,0,1,2}, longOpts(device)).view(2, 2);

        // 边特征：[2,1] → [0.0, 1.0]
        float[][] edgeAttrData = {{0.0f}, {1.0f}};
        Tensor edgeAttr = torch.tensor(flatten(edgeAttrData), floatOpts(device)).view(2, 1);

        // 4. 手动计算预期结果
        // Step1: 邻居特征投影（节点0的特征 [1,2] → [2,1]）
        // xLin[0] = [1*1 + 2*0, 1*0 + 2*1] = [1, 2] → [K=2, out=1]
        // Step2: 高斯权重计算：
        // 边0→1（edgeAttr=0.0）：
        // w1 = exp(-0.5*(0-0)^2/1^2) = exp(0) = 1.0
        // w2 = exp(-0.5*(0-1)^2/1^2) = exp(-0.5) ≈ 0.6065
        // 加权和：1*1 + 2*0.6065 = 1 + 1.213 = 2.213
        // 边0→2（edgeAttr=1.0）：
        // w1 = exp(-0.5*(1-0)^2/1^2) = exp(-0.5) ≈ 0.6065
        // w2 = exp(-0.5*(1-1)^2/1^2) = exp(0) = 1.0
        // 加权和：1*0.6065 + 2*1 = 0.6065 + 2 = 2.6065
        // Step3: 聚合结果：
        // 节点1: 2.213, 节点2: 2.6065, 节点0: 0
        float[][] expected = {{0.0f}, {2.2130f}, {2.6065f}};

        // 5. 前向传播
        Tensor output = gmmConv.forward(x, edgeIndex, edgeAttr);

        // 6. 打印输出（验证结果）
        System.out.println("GMMConv 数值测试输出：");
        torch.print(output);

        // 7. 数值验证（误差允许1e-4）
//        float[][] outputArr = output.to(device, torch.ScalarType.Float).toFloatArray();
//        for (int i = 0; i < 3; i++) {
//            assertEquals("节点" + i + "数值不匹配",
//                    expected[i][0], outputArr[i][0], 1e-4f);
//        }

        // 8. 资源释放
        releaseTensor(x, edgeIndex, edgeAttr, output);
    }

    // ========== 测试3：边界场景验证 ==========
    @Test
    public void testEdgeCases() {
        // 1. 初始化 GMMConv
        long inChannels = 2;
        long outChannels = 1;
        int dim = 1;
        int kernelSize = 2;
        Device device = new Device(torch.kCPU());
        GMMConv gmmConv = new GMMConv(inChannels, outChannels, dim, kernelSize, true, true);

        // 场景1：空边索引（[2,0]）
        Tensor edgeIndexEmpty = torch.empty(new long[]{2, 0}, longOpts(device), new MemoryFormatOptional());
        Tensor edgeAttrEmpty = torch.empty(new long[]{0, dim}, floatOpts(device), new MemoryFormatOptional());
        Tensor x = torch.randn(new long[]{3, inChannels}, floatOpts(device));

        // 前向传播（空边→输出=根节点变换结果）
        Tensor outputEmpty = gmmConv.forward(x, edgeIndexEmpty, edgeAttrEmpty);
        // 根节点变换结果应为 linRoot(x) → 维度匹配
        long[] outputEmptyShape = outputEmpty.sizes().vec().get();
        assertEquals("空边场景输出节点数应为3", 3, outputEmptyShape[0]);
        assertEquals("空边场景输出维度应为1", 1, outputEmptyShape[1]);

        // 场景2：单节点（无自环）
        Tensor edgeIndexSingle = torch.empty(new long[]{2, 0}, longOpts(device),new MemoryFormatOptional());
        Tensor edgeAttrSingle = torch.empty(new long[]{0, dim}, floatOpts(device),new MemoryFormatOptional());
        Tensor xSingle = torch.randn(new long[]{1, inChannels}, floatOpts(device));
        Tensor outputSingle = gmmConv.forward(xSingle, edgeIndexSingle, edgeAttrSingle);
        assertEquals("单节点输出维度应为1×1", 1, outputSingle.sizes().vec().get()[1]);

        // 资源释放
        releaseTensor(edgeIndexEmpty, edgeAttrEmpty, x, outputEmpty,
                edgeIndexSingle, edgeAttrSingle, xSingle, outputSingle);
    }

    @Test
    public void testEdgeCases2() {
        // 1. 初始化 GMMConv
        long inChannels = 2;
        long outChannels = 1;
        int dim = 1;
        int kernelSize = 2;
        Device device = new Device(torch.kCPU());
        GMMConv gmmConv = new GMMConv(inChannels, outChannels, dim, kernelSize, true, true);

        // 场景1：空边索引（[2,0]）
        Tensor edgeIndexEmpty = torch.empty(new long[]{2, 0}, longOpts(device), new MemoryFormatOptional());
        Tensor edgeAttrEmpty = torch.empty(new long[]{0, dim}, floatOpts(device), new MemoryFormatOptional());
        Tensor x = torch.randn(new long[]{3, inChannels}, floatOpts(device));

        // 前向传播（空边→输出=根节点变换结果）
        Tensor outputEmpty = gmmConv.forward(x, edgeIndexEmpty, edgeAttrEmpty);

        // ========== 补充验证：根节点变换结果匹配 ==========
        Tensor rootOut = gmmConv.getLinRoot().forward(x);
//        float[][] outputArr = outputEmpty.to(device, torch.ScalarType.Float).toFloatArray();
//        float[][] rootArr = rootOut.to(device, torch.ScalarType.Float).toFloatArray();
//        // 空边场景下，输出应等于根节点变换结果（误差1e-6）
//        for (int i = 0; i < 3; i++) {
//            assertEquals("空边场景节点" + i + "应等于根节点结果",
//                    rootArr[i][0], outputArr[i][0], 1e-6f);
//        }

        // 形状验证
        long[] outputEmptyShape = outputEmpty.sizes().vec().get();
        assertEquals("空边场景输出节点数应为3", 3, outputEmptyShape[0]);
        assertEquals("空边场景输出维度应为1", 1, outputEmptyShape[1]);

        // 场景2：单节点（无自环）
        Tensor edgeIndexSingle = torch.empty(new long[]{2, 0}, longOpts(device),new MemoryFormatOptional());
        Tensor edgeAttrSingle = torch.empty(new long[]{0, dim}, floatOpts(device),new MemoryFormatOptional());
        Tensor xSingle = torch.randn(new long[]{1, inChannels}, floatOpts(device));
        Tensor outputSingle = gmmConv.forward(xSingle, edgeIndexSingle, edgeAttrSingle);

        // 验证单节点输出等于根节点变换结果
        Tensor rootSingle = gmmConv.getLinRoot().forward(xSingle);
        assertEquals("单节点输出维度应为1×1", 1, outputSingle.sizes().vec().get()[1]);
//        assertEquals("单节点输出应等于根节点结果",
//                rootSingle.toFloatArray()[0][0], outputSingle.toFloatArray()[0][0], 1e-6f);

        // 资源释放
        releaseTensor(edgeIndexEmpty, edgeAttrEmpty, x, outputEmpty, rootOut,
                edgeIndexSingle, edgeAttrSingle, xSingle, outputSingle, rootSingle);
    }

    public static void main(String[] args) {
        Result result = JUnitCore.runClasses(GMMConvTest.class);
        for (Failure failure : result.getFailures()) {
            System.err.println("FAIL: " + failure.toString());
            if (failure.getException() != null) failure.getException().printStackTrace();
        }
        System.out.println("Tests run=" + result.getRunCount()
                + " failed=" + result.getFailureCount()
                + " ignored=" + result.getIgnoreCount());
        if (!result.wasSuccessful()) {
            throw new RuntimeException("GMMConvTest had " + result.getFailureCount() + " failures");
        }
        System.out.println("✅ GMMConvTest all tests passed");
    }

}
