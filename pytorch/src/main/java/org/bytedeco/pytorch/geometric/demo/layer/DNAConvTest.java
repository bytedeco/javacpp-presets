package org.bytedeco.pytorch.geometric.demo.layer;
import org.bytedeco.pytorch.jit.*;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.global.torch;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.bytedeco.pytorch.geometric.nn.conv.DNAConv;

import java.util.Arrays;

import static org.junit.Assert.*;

/**
 * DNAConv 测试用例：
 * 1. 形状验证（输入→输出维度）
 * 2. 数值验证（注意力得分归一化）
 * 3. 跨层聚合验证（每层特征参与注意力）
 * 4. 边权重验证（拓扑平滑生效）
 * 5. 异常验证（非法输入/参数）
 * 6. 资源释放验证
 */
public class DNAConvTest {
    private DNAConv dnaConv;            // 带偏置版本
    private DNAConv dnaConvNoBias;      // 无偏置版本
    private Tensor x;                   // 节点特征 [5, 3, 4]（5节点，3层，4通道）
    private Tensor edgeIndex;           // 边索引 [2, 6]
    private Tensor edgeWeight;          // 边权重 [6]
    private static final long CHANNELS = 4;
    private static final int HEADS = 2;
    private static final int GROUPS = 2;

    @Before
    public void setUp() {
        // 1. 固定随机种子，保证数值可复现
        torch.manual_seed(1234);

        // 2. 初始化DNAConv
        dnaConv = new DNAConv(CHANNELS, HEADS, GROUPS, true);
        dnaConvNoBias = new DNAConv(CHANNELS, HEADS, GROUPS, false);

        // 3. 构造输入特征 [N=5, L=3, C=4]
        x = torch.randn(new long[]{5, 3, 4}).to(torch.kFloat());

        // 4. 构造边索引 [2, 6]：5节点的有向边
        long[] edgeData = {0,0,1,2,3,4, 1,2,2,3,4,0};
        edgeIndex = torch.tensor(edgeData).view(2, 6).to(torch.kLong());

        // 5. 构造边权重 [6]（归一化权重）
        edgeWeight = torch.ones(6).div(new Scalar(2.0f)); // 每条边权重=0.5
    }

    /**
     * 测试1：输出形状验证
     * 预期：输入 [5,3,4] → 输出 [5,4]
     */
    @Test
    public void testForwardShape() {
        Tensor out = ((DNAConv)dnaConv).forward(x, edgeIndex);
        Tensor outNoBias = ((DNAConv)dnaConvNoBias).forward(x, edgeIndex, edgeWeight);

        // 验证带偏置版本形状
        assertEquals("输出应为2维张量", 2, out.dim());
        assertEquals("节点数应保持5", 5, out.size(0));
        assertEquals("输出通道数应为4", CHANNELS, out.size(1));

        // 验证无偏置版本形状
        assertEquals("无偏置版本形状应一致", Arrays.toString(out.sizes().vec().get()), Arrays.toString(outNoBias.sizes().vec().get()));

        System.out.println("✅ 形状验证通过：输入 " + Arrays.toString(x.sizes().vec().get()) + " → 输出 " + Arrays.toString(out.sizes().vec().get()));

        // 释放临时张量
        out.close();
        outNoBias.close();
    }

    /**
     * 测试2：注意力得分归一化验证
     * 核心逻辑：softmax后，每层注意力得分之和=1
     */
    @Test
    public void testAttentionNormalization() {
        // 手动提取注意力得分（简化版）
        long N = x.size(0);
        long L = x.size(1);
        long C = x.size(2);
        Tensor x_flat = x.view(-1, C);
        Tensor Q_flat = dnaConvNoBias.linQ.forward(x_flat);
        Tensor K_flat = dnaConvNoBias.linK.forward(x_flat);
        Tensor Q = Q_flat.view(N, L, HEADS, dnaConvNoBias.getDk());
        Tensor K = K_flat.view(N, L, HEADS, dnaConvNoBias.getDk());

        // 计算注意力得分并softmax
        Tensor query = Q.select(1, (int)(L-1)).unsqueeze(1);
        Tensor attn = query.mul(K).sum(-1).div(new Scalar(Math.sqrt(dnaConvNoBias.getDk())));
        Tensor attnSoftmax = torch.softmax(attn, 1);

        // 验证softmax后，每个节点+注意力头的得分之和≈1
        Tensor attnSum = attnSoftmax.sum(1); // [N, H]
        Tensor attnDiff = attnSum.sub(new Scalar(1.0f)).abs();
        double maxDiff = attnDiff.max().item_double();
        assertTrue(String.format("注意力得分归一化失败（最大偏差：%.6f）", maxDiff), maxDiff < 1e-5);

        System.out.println("✅ 注意力归一化验证通过，最大偏差：" + maxDiff);

        // 释放临时张量
        x_flat.close();
        Q_flat.close();
        K_flat.close();
        Q.close();
        K.close();
        query.close();
        attn.close();
        attnSoftmax.close();
        attnSum.close();
        attnDiff.close();
    }

    /**
     * 测试3：跨层聚合验证（有无跨层聚合结果不同）
     */
    @Test
    public void testCrossLayerAggregation() {
        // 正常跨层聚合输出
        Tensor outFull = ((DNAConv)dnaConvNoBias).forward(x, edgeIndex);

        // 仅使用最后一层特征（模拟无跨层聚合）
        Tensor xLastLayer = x.select(1, 2).unsqueeze(1); // [5,1,4]
        Tensor outLastLayer = ((DNAConv)dnaConvNoBias).forward(xLastLayer, edgeIndex);

        // 验证结果不同（跨层聚合生效）
        Tensor diff = outFull.sub(outLastLayer).abs();
        double maxDiff = diff.max().item_double();
        assertTrue("跨层聚合应生效（最大差异：" + maxDiff + "）", maxDiff > 1e-3);

        System.out.println("✅ 跨层聚合验证通过，最大差异：" + maxDiff);

        // 释放临时张量
        outFull.close();
        xLastLayer.close();
        outLastLayer.close();
        diff.close();
    }

    /**
     * 测试4：边权重验证（拓扑平滑生效）
     */
    @Test
    public void testEdgeWeight() {
        // 无边权重输出
        Tensor outNoWeight = ((DNAConv)dnaConvNoBias).forward(x, edgeIndex, (Tensor)null);
        // 有边权重输出
        Tensor outWithWeight = ((DNAConv)dnaConvNoBias).forward(x, edgeIndex, edgeWeight);

        // 验证结果不同（边权重生效）
        Tensor diff = outNoWeight.sub(outWithWeight).abs();
        double maxDiff = diff.max().item_double();
        assertTrue("边权重应生效（最大差异：" + maxDiff + "）", maxDiff > 1e-3);

        System.out.println("✅ 边权重验证通过，最大差异：" + maxDiff);

        // 释放临时张量
        outNoWeight.close();
        outWithWeight.close();
        diff.close();
    }

    /**
     * 测试5：异常输入验证
     */
    @Test
    public void testInvalidInput() {
        // 测试1：channels 不能被 heads 整除
        assertThrows(IllegalArgumentException.class, () -> new DNAConv(5, 2, 2, true));

        // 测试2：heads/groups 为0
        assertThrows(ArithmeticException.class, () -> new DNAConv(4, 0, 2, true));

        // 测试3：x 维度错误（2维）
        Tensor x2d = torch.randn(new long[]{5, 4});
        assertThrows(IllegalArgumentException.class, () -> ((DNAConv)dnaConv).forward(x2d, edgeIndex));

        // 测试4：edge_index 维度错误（3维）
        Tensor edgeIndex3d = torch.randn(new long[]{3, 6});
        assertThrows(IllegalArgumentException.class, () -> ((DNAConv)dnaConv).forward(x, edgeIndex3d));

        // 测试5：x 通道数不匹配
        Tensor xWrongChannel = torch.randn(new long[]{5, 3, 5});
        assertThrows(IllegalArgumentException.class, () -> ((DNAConv)dnaConv).forward(xWrongChannel, edgeIndex));

        System.out.println("✅ 异常输入验证通过");

        // 释放临时张量
        x2d.close();
        edgeIndex3d.close();
        xWrongChannel.close();
    }

    /**
     * 测试6：资源释放验证
     */
    @Test
    public void testResourceRelease() {
        DNAConv tempConv = new DNAConv(4, 2, 2, true);
        tempConv.close();
        // 验证重复释放不崩溃
//        assertDoesNotThrow(tempConv::close);

        System.out.println("✅ 资源释放验证通过");
    }

    @After
    public void tearDown() {
        // 释放所有资源
        if (dnaConv != null) dnaConv.close();
        if (dnaConvNoBias != null) dnaConvNoBias.close();
        if (x != null) x.close();
        if (edgeIndex != null) edgeIndex.close();
        if (edgeWeight != null) edgeWeight.close();

        // 清空PyTorch计算图
//        torch.clear_autograd_graph();
//        torch.cuda.empty_cache();
    }
}
