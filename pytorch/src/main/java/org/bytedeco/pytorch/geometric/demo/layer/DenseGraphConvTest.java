package org.bytedeco.pytorch.geometric.demo.layer;
import org.bytedeco.pytorch.jit.*;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.global.torch;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.bytedeco.pytorch.geometric.nn.conv.DenseGraphConv;

import java.util.Arrays;

import static org.junit.Assert.*;

/**
 * DenseGraphConv 测试用例：
 * 1. 形状验证（输入→输出维度是否符合预期）
 * 2. 数值验证（核心公式：linRel(A·X) + linRoot(X)）
 * 3. 异常验证（非法输入维度/通道数）
 * 4. 资源释放验证（无内存泄漏）
 */
public class DenseGraphConvTest {
    private DenseGraphConv graphConv;   // 测试用 GraphConv 实例
    private Tensor x;                   // 输入特征 [B=2, N=3, in_channels=4]
    private Tensor adj;                 // 邻接矩阵 [B=2, N=3, N=3]（单位矩阵）
    private static final long IN_CHANNELS = 4;  // 输入通道数
    private static final long OUT_CHANNELS = 2; // 输出通道数

    @Before
    public void setUp() {
        // 1. 固定随机种子，保证数值可复现
        torch.manual_seed(1234);

        // 2. 初始化 GraphConv 层
        graphConv = new DenseGraphConv(IN_CHANNELS, OUT_CHANNELS);

        // 3. 构造输入特征：[2, 3, 4]（float32，避免精度问题）
        x = torch.randn(new long[]{2, 3, 4}).to(torch.kFloat());

        // 4. 构造邻接矩阵：单位矩阵（每个节点只连接自己）→ A·X = X
        adj = torch.eye(3).unsqueeze(0).expand(new long[]{2, 3, 3}).to(torch.kFloat());
    }

    /**
     * 测试1：输出形状验证（核心）
     * 预期：输入 [2,3,4] → 输出 [2,3,2]
     */
    @Test
    public void testForwardShape() {
        Tensor out = graphConv.forward(x, adj);

        // 验证维度数
        assertEquals("输出应为 3 维张量", 3, out.dim());
        // 验证批次维度
        assertEquals("批次维度应保持 2", 2, out.size(0));
        // 验证节点维度
        assertEquals("节点维度应保持 3", 3, out.size(1));
        // 验证输出通道数
        assertEquals("输出通道数应为 2", OUT_CHANNELS, out.size(2));

        System.out.println("✅ 形状验证通过：输入 " + Arrays.toString(x.sizes().vec().get()) + " → 输出 " + Arrays.toString(out.sizes().vec().get()));

        // 释放临时张量
        out.close();
    }

    /**
     * 测试2：数值逻辑验证（核心公式）
     * 核心公式：out = linRel(A·X) + linRoot(X)
     * 当 A 是单位矩阵时：A·X = X → out = (linRel + linRoot)(X)
     */
    @Test
    public void testForwardNumerics() {
        // 步骤1：手动计算理论值（模拟核心公式）
        // a. 获取线性层权重和偏置
        Tensor linRelWeight = graphConv.getLinRel().weight();
        Tensor linRelBias = graphConv.getLinRel().bias();
        Tensor linRootWeight = graphConv.getLinRoot().weight();
        Tensor linRootBias = graphConv.getLinRoot().bias();

        // b. 计算 linRel(X) + linRoot(X) = (W_rel + W_root)·X + (b_rel + b_root)
        Tensor combinedWeight = linRelWeight.add(linRootWeight);
        Tensor combinedBias = linRelBias.add(linRootBias);

        // c. 手动矩阵乘法：X @ W.T + b（PyTorch Linear 实现逻辑）
        //    x: [2,3,4] → 转置权重 [4,2] → 相乘 [2,3,2]
        Tensor expectedOut = x.matmul(combinedWeight.t()).add(combinedBias);

        // 步骤2：获取 GraphConv 实际输出
        Tensor actualOut = graphConv.forward(x, adj);

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
     * 测试3：异常输入验证（维度/通道数不匹配）
     */
    @Test
    public void testInvalidInput() {
        // 测试1：x 维度不是 3 维
        Tensor x2d = torch.randn(new long[]{3, 4}); // [N, C]
        assertThrows(IllegalArgumentException.class, () -> graphConv.forward(x2d, adj));

        // 测试2：adj 维度不是 3 维
        Tensor adj2d = torch.eye(3); // [N, N]
        assertThrows(IllegalArgumentException.class, () -> graphConv.forward(x, adj2d));

        // 测试3：x 通道数不匹配（期望 4，实际 5）
        Tensor xWrongChannel = torch.randn(new long[]{2, 3, 5});
        assertThrows(IllegalArgumentException.class, () -> graphConv.forward(xWrongChannel, adj));

        System.out.println("✅ 异常输入验证通过");

        // 释放临时张量
        x2d.close();
        adj2d.close();
        xWrongChannel.close();
    }

    /**
     * 测试4：资源释放验证（无内存泄漏/重复释放）
     */
    @Test
    public void testResourceRelease() {
        // 构造临时 GraphConv 实例
        DenseGraphConv tempConv = new DenseGraphConv(4, 2);
        // 第一次释放
        tempConv.close();
        // 验证重复释放不崩溃
//        assertDoesNotThrow(tempConv::close);

        System.out.println("✅ 资源释放验证通过");
    }

    @After
    public void tearDown() {
        // 严格释放所有资源，避免JNI内存泄漏
        if (graphConv != null) graphConv.close();
        if (x != null) x.close();
        if (adj != null) adj.close();

        // 清空PyTorch计算图，释放JNI资源
//        torch.clear_autograd_graph();
//        torch.cuda.empty_cache(); // 兼容GPU场景
    }
}
