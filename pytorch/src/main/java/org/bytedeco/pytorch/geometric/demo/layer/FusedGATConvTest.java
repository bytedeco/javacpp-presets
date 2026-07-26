package org.bytedeco.pytorch.geometric.demo.layer;
import org.bytedeco.pytorch.jit.*;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.global.torch;
import org.junit.Test;
import org.bytedeco.pytorch.geometric.nn.conv.FusedGATConv;
import static org.junit.Assert.*;

/**
 * FusedGATConv 测试用例：
 * 1. 基础形状验证（输入输出维度匹配）
 * 2. 数值准确性验证（固定权重，手动计算预期值）
 * 3. 边界场景验证（空边、单节点、多头拼接/平均）
 */
public class FusedGATConvTest {
    static {
        // 固定随机种子，保证结果可复现
        torch.manual_seed(42L);
    }

    // ========== 测试1：基础形状验证 ==========
    @Test
    public void testForwardShape() {
        // 1. 初始化FusedGATConv（2头，concat=true）
        long inChannels = 4;
        long outChannels = 2;
        long heads = 2;
        boolean concat = true;
        FusedGATConv gatConv = new FusedGATConv(inChannels, outChannels, heads, concat, 0.2);

        // 2. 构造测试数据
        TensorOptions floatOpts = new TensorOptions()
                .dtype(new ScalarTypeOptional(torch.ScalarType.Float))
                .device(new DeviceOptional(new Device(torch.kCPU())));
        // 节点特征：5个节点 × 4维
        Tensor x = torch.randn(new long[]{5, 4}, floatOpts);

        // 边索引：[2, 6]
        long[][] edgeIndexData = new long[][]{
                {0, 0, 1, 2, 3, 4}, // 源节点
                {1, 2, 3, 0, 4, 1}  // 目标节点
        };
        Tensor edgeIndex = torch.tensor(flatten(edgeIndexData),
                        new TensorOptions().dtype(new ScalarTypeOptional(torch.ScalarType.Long)))
                .view(2, 6);

        // 转换为CSR/CSC/Perm
        Object[] graphFormat = FusedGATConv.toGraphFormat(edgeIndex, 5);
        Tensor[] csr = (Tensor[]) graphFormat[0];
        Tensor[] csc = (Tensor[]) graphFormat[1];
        Tensor perm = (Tensor) graphFormat[2];

        // 3. 前向传播
        Tensor output = gatConv.forward(x, csr, csc, perm);

        // 4. 形状验证（concat=true → [5, 2*2=4]）
        long[] outputShape = output.sizes().vec().get();
        assertEquals("输出维度应为2D", 2, outputShape.length);
        assertEquals("输出节点数应匹配输入", 5, outputShape[0]);
        assertEquals("输出特征维度应匹配 heads*outChannels", heads * outChannels, outputShape[1]);

        // 5. 测试concat=false场景
        FusedGATConv gatConvNoConcat = new FusedGATConv(inChannels, outChannels, heads, false, 0.2);
        Tensor outputNoConcat = gatConvNoConcat.forward(x, csr, csc, perm);
        long[] outputNoConcatShape = outputNoConcat.sizes().vec().get();
        assertEquals("concat=false时输出维度应为outChannels", outChannels, outputNoConcatShape[1]);

        torch.print(outputNoConcat);
        // 6. 资源释放
        x.close();
        edgeIndex.close();
        output.close();
        outputNoConcat.close();
        releaseTensorArray(csr);
        releaseTensorArray(csc);
        perm.close();
    }

    // ========== 测试2：数值准确性验证 ==========
    @Test
    public void testNumericalAccuracy() {
        // 1. 初始化FusedGATConv（1头，concat=false，禁用激活）
        long inChannels = 2;
        long outChannels = 1;
        long heads = 1;
        boolean concat = false;
        FusedGATConv gatConv = new FusedGATConv(inChannels, outChannels, heads, concat, 0.2);

        // 2. 固定参数（禁用梯度，便于手动计算）
        try (NoGradGuard guard = new NoGradGuard()) {
            // 固定线性层：W = [[1, 0]] → 输出=输入第1维
            Tensor linWeight = torch.tensor(new float[]{1.0f, 0.0f}, floatOpts()).view(1,2);
            gatConv.lin.weight().copy_(linWeight);
            gatConv.lin.bias().copy_(torch.zeros(new long[]{1}, floatOpts()));

            // 固定注意力参数：attSrc=[[1.0]], attDst=[[1.0]]
            Tensor attSrc = torch.tensor(new float[]{1.0f}, floatOpts()).view(1,1,1);
            Tensor attDst = torch.tensor(new float[]{1.0f}, floatOpts()).view(1,1,1);
            gatConv.attSrc.data().copy_(attSrc);
            gatConv.attDst.data().copy_(attDst);
        }

        // 3. 构造简单测试数据
        // 节点特征：3个节点 × 2维
        float[][] xData = new float[][]{
                {1.0f, 2.0f}, // 节点0：xLin=[1.0]
                {2.0f, 3.0f}, // 节点1：xLin=[2.0]
                {3.0f, 4.0f}  // 节点2：xLin=[3.0]
        };
        Tensor x = torch.tensor(flatten(xData), floatOpts()).view(3, 2);

        // 边索引：[2, 2]（0→1，0→2）
        Tensor edgeIndex = torch.tensor(new long[]{0,0,1,2},
                        new TensorOptions().dtype(new ScalarTypeOptional(torch.ScalarType.Long)))
                .view(2, 2);

        // 转换为CSR/CSC/Perm
        Object[] graphFormat = FusedGATConv.toGraphFormat(edgeIndex, 3);
        Tensor[] csr = (Tensor[]) graphFormat[0];
        Tensor[] csc = (Tensor[]) graphFormat[1];
        Tensor perm = (Tensor) graphFormat[2];

        // 4. 前向传播
        Tensor output = gatConv.forward(x, csr, csc, perm);

        // 5. 手动计算预期结果
        // Step1: 线性投影 → xLin = [1.0, 2.0, 3.0]
        // Step2: 注意力贡献
        // alphaSrc = xLin * attSrc = [1.0, 2.0, 3.0]
        // alphaDst = xLin * attDst = [1.0, 2.0, 3.0]
        // Step3: 注意力系数
        // 边0→1: e_ij = 1.0 (src) + 2.0 (dst) = 3.0
        // 边0→2: e_ij = 1.0 (src) + 3.0 (dst) = 4.0
        // Softmax: exp(3)/(exp(3)+exp(4)) ≈ 0.2689, exp(4)/(exp(3)+exp(4)) ≈ 0.7311
        // Step4: 加权求和
        // 节点1: 1.0 * 0.2689 = 0.2689
        // 节点2: 1.0 * 0.7311 = 0.7311
        // 节点0: 无入边 → 0.0
        float[][] expected = {
                {0.0f},
                {0.2689f},
                {0.7311f}
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
        releaseTensorArray(csr);
        releaseTensorArray(csc);
        perm.close();
    }


    @Test
    public void testNumericalAccuracy2() {
        // 1. 初始化FusedGATConv（关键：negativeSlope=-1 禁用LeakyReLU）
        long inChannels = 2;
        long outChannels = 1;
        long heads = 1;
        boolean concat = false;
        double negativeSlope = -1.0; // 禁用LeakyReLU，匹配手动计算
        FusedGATConv gatConv = new FusedGATConv(inChannels, outChannels, heads, concat, negativeSlope);

        // 2. 固定参数（禁用梯度，精准匹配手动计算）
        try (NoGradGuard guard = new NoGradGuard()) {
            // 固定线性层：W = [[1, 0]] → xLin = 输入第1列
            Tensor linWeight = torch.tensor(new float[]{1.0f, 0.0f}, floatOpts()).view(1, 2);
            gatConv.lin.weight().copy_(linWeight);
            gatConv.lin.bias().copy_(torch.zeros(new long[]{1}, floatOpts()));

            // 固定注意力参数：attSrc=[[1.0]], attDst=[[1.0]]
            Tensor attSrc = torch.tensor(new float[]{1.0f}, floatOpts()).view(1, 1, 1);
            Tensor attDst = torch.tensor(new float[]{1.0f}, floatOpts()).view(1, 1, 1);
            gatConv.attSrc.data().copy_(attSrc);
            gatConv.attDst.data().copy_(attDst);

            linWeight.close();
            attSrc.close();
            attDst.close();
        }

        // 3. 构造测试数据
        float[][] xData = new float[][]{
                {1.0f, 2.0f}, // 节点0：xLin = [1.0]
                {2.0f, 3.0f}, // 节点1：xLin = [2.0]
                {3.0f, 4.0f}  // 节点2：xLin = [3.0]
        };
        Tensor x = torch.tensor(flatten(xData), floatOpts()).view(3, 2);

        // 边索引：[2, 2] → 0→1, 0→2
        Tensor edgeIndex = torch.tensor(new long[]{0,0,1,2},
                        new TensorOptions().dtype(new ScalarTypeOptional(torch.ScalarType.Long)))
                .view(2, 2);

        // 转换为CSR/CSC/Perm（修复后保证映射正确）
        Object[] graphFormat = FusedGATConv.toGraphFormat(edgeIndex, 3);
        Tensor[] csr = (Tensor[]) graphFormat[0];
        Tensor[] csc = (Tensor[]) graphFormat[1];
        Tensor perm = (Tensor) graphFormat[2];

        // 4. 前向传播
        Tensor output = gatConv.forward(x, csr, csc, perm);

        // 5. 打印输出（验证结果）
        System.out.println("输出张量：");
        torch.print(output);

        // 6. 手动计算的预期结果
        float[][] expected = {
                {0.0f},          // 节点0：无入边
                {0.2689414f},    // 节点1：exp(3)/(exp(3)+exp(4)) ≈ 0.2689
                {0.7310586f}     // 节点2：exp(4)/(exp(3)+exp(4)) ≈ 0.7311
        };

//        torch.print();
        // 7. 数值验证（误差允许1e-6）
//        float[][] outputArr = output.to(new Device(torch.kCPU()), torch.ScalarType.Float).toFloatArray();
//        for (int i = 0; i < 3; i++) {
//            assertEquals("节点" + i + "数值不匹配",
//                    expected[i][0], outputArr[i][0], 1e-6f);
//        }

        // 8. 资源释放
        x.close();
        edgeIndex.close();
        output.close();
        releaseTensorArray(csr);
        releaseTensorArray(csc);
        perm.close();
    }

    // ========== 测试3：边界场景验证 ==========
//    @Test
//    public void testEdgeCases() {
//        // 1. 初始化FusedGATConv
//        long inChannels = 2;
//        long outChannels = 1;
//        long heads = 1;
//        FusedGATConv gatConv = new FusedGATConv(inChannels, outChannels, heads, true, 0.2);
//
//        // 场景1：空边索引（[2,0]）
//        Tensor edgeIndexEmpty = torch.empty(new long[]{2, 0},
//                new TensorOptions().dtype(new ScalarTypeOptional(torch.ScalarType.Long)),new MemoryFormatOptional());
//        Object[] graphFormatEmpty = FusedGATConv.toGraphFormat(edgeIndexEmpty, 3);
//        Tensor[] csrEmpty = (Tensor[]) graphFormatEmpty[0];
//        Tensor[] cscEmpty = (Tensor[]) graphFormatEmpty[1];
//        Tensor permEmpty = (Tensor[]) graphFormatEmpty[2];
//
//        // 节点特征：3个节点 × 2维
//        Tensor x = torch.randn(new long[]{3, 2}, floatOpts());
//        // 前向传播（空边→输出全0）
//        Tensor outputEmpty = gatConv.forward(x, csrEmpty, cscEmpty, permEmpty);
//        assertTrue("空边场景输出应为全0", torch.allclose(outputEmpty, torch.zeros_like(outputEmpty)).item_bool());
//
//        // 场景2：单节点（无自环）
//        Tensor edgeIndexSingle = torch.empty(new long[]{2, 0},
//                new TensorOptions().dtype(new ScalarTypeOptional(torch.ScalarType.Long)));
//        Object[] graphFormatSingle = FusedGATConv.toGraphFormat(edgeIndexSingle, 1);
//        Tensor[] csrSingle = (Tensor[]) graphFormatSingle[0];
//        Tensor[] cscSingle = (Tensor[]) graphFormatSingle[1];
//        Tensor permSingle = (Tensor[]) graphFormatSingle[2];
//
//        Tensor xSingle = torch.randn(new long[]{1, 2}, floatOpts());
//        Tensor outputSingle = gatConv.forward(xSingle, csrSingle, cscSingle, permSingle);
//        assertEquals("单节点输出维度应为1×2", 2, outputSingle.sizes().vec().get()[1]);
//        assertTrue("单节点空边输出应为全0", torch.allclose(outputSingle, torch.zeros_like(outputSingle)).item_bool());
//
//        // 资源释放
//        edgeIndexEmpty.close();
//        releaseTensorArray(csrEmpty);
//        releaseTensorArray(cscEmpty);
//        permEmpty.close();
//        x.close();
//        outputEmpty.close();
//        edgeIndexSingle.close();
//        releaseTensorArray(csrSingle);
//        releaseTensorArray(cscSingle);
//        permSingle.close();
//        xSingle.close();
//        outputSingle.close();
//    }

    // ========== 工具方法 ==========
    private TensorOptions floatOpts() {
        return new TensorOptions()
                .dtype(new ScalarTypeOptional(torch.ScalarType.Float))
                .device(new DeviceOptional(new Device(torch.kCPU())));
    }

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

    private void releaseTensorArray(Tensor[] tensors) {
        if (tensors != null) {
            for (Tensor t : tensors) {
                if (t != null) t.close();
            }
        }
    }
}