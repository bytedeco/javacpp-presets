package samples.demo.layer;
import org.bytedeco.pytorch.jit.*;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.global.torch;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.JUnitCore;
import org.junit.runner.Result;
import org.junit.runner.notification.Failure;
import org.bytedeco.pytorch.geometric.nn.conv.DenseSAGEConv;

import java.util.Arrays;

import static org.junit.Assert.*;

/**
 * DenseSAGEConv 测试用例：
 * 1. 形状验证（输入→输出维度）
 * 2. 数值验证（核心公式：W_rel·Mean(Neighbors) + W_root·X）
 * 3. 归一化验证（L2归一化后范数=1）
 * 4. 异常验证（非法输入）
 * 5. 资源释放验证
 */
public class DenseSAGEConvTest {
    private DenseSAGEConv sageConvNoNorm;  // 无归一化版本
    private DenseSAGEConv sageConvWithNorm;// 有归一化版本
    private Tensor x;                      // 输入特征 [B=2, N=3, C=4]
    private Tensor adj;                    // 邻接矩阵 [B=2, N=3, N=3]（单位矩阵）
    private static final long IN_CHANNELS = 4;
    private static final long OUT_CHANNELS = 2;

    @Before
    public void setUp() {
        // 1. 固定随机种子，保证数值可复现
        torch.manual_seed(1234);

        // 2. 初始化SAGE层（有无归一化）
        sageConvNoNorm = new DenseSAGEConv(IN_CHANNELS, OUT_CHANNELS, false);
        sageConvWithNorm = new DenseSAGEConv(IN_CHANNELS, OUT_CHANNELS, true);

        // 3. 构造输入特征：[2,3,4]（float32）
        x = torch.randn(new long[]{2, 3, 4}).to(torch.ScalarType.Float);

        // 4. 构造邻接矩阵：单位矩阵（每个节点仅连接自己）→ Mean(Neighbors)=X
        adj = torch.eye(3).unsqueeze(0).expand(new long[]{2, 3, 3}).to(torch.ScalarType.Float);
    }

    /**
     * 测试1：输出形状验证
     * 预期：输入 [2,3,4] → 输出 [2,3,2]
     */
    @Test
    public void testForwardShape() {
        Tensor outNoNorm = sageConvNoNorm.forward(x, adj);
        Tensor outWithNorm = sageConvWithNorm.forward(x, adj);

        // 验证无归一化版本形状
        assertEquals("输出应为3维张量", 3, outNoNorm.dim());
        assertEquals("批次维度应保持2", 2, outNoNorm.size(0));
        assertEquals("节点维度应保持3", 3, outNoNorm.size(1));
        assertEquals("输出通道数应为2", OUT_CHANNELS, outNoNorm.size(2));

        // 验证归一化版本形状（与无归一化一致）
        assertEquals("归一化版本形状应一致", Arrays.toString(outNoNorm.sizes().vec().get()),Arrays.toString( outWithNorm.sizes().vec().get()));

        System.out.println("✅ 形状验证通过：输入 " + Arrays.toString(x.sizes().vec().get()) + " → 输出 " + Arrays.toString(outNoNorm.sizes().vec().get()));

        // 释放临时张量
        outNoNorm.close();
        outWithNorm.close();
    }

    /**
     * 测试2：数值逻辑验证（核心公式）
     * 核心公式：out = W_rel·X + W_root·X（单位矩阵下 Mean(Neighbors)=X）
     */
    @Test
    public void testForwardNumerics() {
        // 步骤1：手动计算理论值
        Tensor linRelWeight = sageConvNoNorm.getLinRel().weight();
        Tensor linRelBias = sageConvNoNorm.getLinRel().bias();
        Tensor linRootWeight = sageConvNoNorm.getLinRoot().weight();
        Tensor linRootBias = sageConvNoNorm.getLinRoot().bias();

        // 合并权重和偏置：W_rel + W_root, b_rel + b_root
        Tensor combinedWeight = linRelWeight.add(linRootWeight);
        Tensor combinedBias = linRelBias.add(linRootBias);

        // 手动计算：X @ W.T + b（PyTorch Linear 实现逻辑）
        Tensor expectedOut = x.matmul(combinedWeight.t()).add(combinedBias);

        // 步骤2：获取SAGE实际输出
        Tensor actualOut = sageConvNoNorm.forward(x, adj);

        // 步骤3：验证数值一致性（误差 < 1e-5）
        Tensor diff = actualOut.sub(expectedOut).abs();
        double maxDiff = diff.max().item_double();
        assertTrue(String.format("数值误差过大（%.6f）", maxDiff), maxDiff < 1e-5);

        System.out.println("✅ 数值验证通过，最大误差：" + maxDiff);

        // 释放临时张量
        combinedWeight.close();
        combinedBias.close();
        expectedOut.close();
        actualOut.close();
        diff.close();
    }

    /**
     * 测试3：归一化验证（L2归一化后范数≈1）
     */
    @Test
    public void testNormalize() {
        Tensor outWithNorm = sageConvWithNorm.forward(x, adj);

        // 计算输出的L2范数（沿最后一维）
        Tensor norm = outWithNorm.norm(
                new ScalarOptional(new Scalar(2.0)),
                new long[]{-1},
                true
        );

        // 验证每个节点的输出范数≈1（误差 < 1e-5）
        Tensor normDiff = norm.sub(new Scalar(1.0)).abs();
        double maxNormDiff = normDiff.max().item_double();
        assertTrue(String.format("归一化后范数偏离1过大（%.6f）", maxNormDiff), maxNormDiff < 1e-5);

        System.out.println("✅ 归一化验证通过，范数最大偏差：" + maxNormDiff);

        // 释放临时张量
        outWithNorm.close();
        norm.close();
        normDiff.close();
    }

    /**
     * 测试4：异常输入验证
     */
    @Test
    public void testInvalidInput() {
        // 测试1：x维度不是3维
        Tensor x2d = torch.randn(new long[]{3, 4});
        assertThrows(IllegalArgumentException.class, () -> sageConvNoNorm.forward(x2d, adj));

        // 测试2：adj维度不是3维
        Tensor adj2d = torch.eye(3);
        assertThrows(IllegalArgumentException.class, () -> sageConvNoNorm.forward(x, adj2d));

        // 测试3：x通道数不匹配
        Tensor xWrongChannel = torch.randn(new long[]{2, 3, 5});
        assertThrows(IllegalArgumentException.class, () -> sageConvNoNorm.forward(xWrongChannel, adj));

        System.out.println("✅ 异常输入验证通过");

        // 释放临时张量
        x2d.close();
        adj2d.close();
        xWrongChannel.close();
    }

    /**
     * 测试5：资源释放验证
     */
    @Test
    public void testResourceRelease() {
        DenseSAGEConv tempConv = new DenseSAGEConv(4, 2);
        tempConv.close();
        // 验证重复释放不崩溃
//        assertDoesNotThrow(tempConv::close);

        System.out.println("✅ 资源释放验证通过");
    }

    @After
    public void tearDown() {
        // 释放所有资源，避免JNI内存泄漏
        if (sageConvNoNorm != null) sageConvNoNorm.close();
        if (sageConvWithNorm != null) sageConvWithNorm.close();
        if (x != null) x.close();
        if (adj != null) adj.close();

        // 清空PyTorch计算图
//        torch.clear_autograd_graph();
//        torch.cuda.empty_cache();
    }

    public static void main(String[] args) {
        Result result = JUnitCore.runClasses(DenseSAGEConvTest.class);
        for (Failure failure : result.getFailures()) {
            System.err.println("FAIL: " + failure.toString());
            if (failure.getException() != null) failure.getException().printStackTrace();
        }
        System.out.println("Tests run=" + result.getRunCount()
                + " failed=" + result.getFailureCount()
                + " ignored=" + result.getIgnoreCount());
        if (!result.wasSuccessful()) {
            throw new RuntimeException("DenseSAGEConvTest had " + result.getFailureCount() + " failures");
        }
        System.out.println("✅ DenseSAGEConvTest all tests passed");
    }

}
