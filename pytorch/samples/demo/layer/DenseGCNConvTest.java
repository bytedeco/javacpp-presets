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
import org.bytedeco.pytorch.geometric.nn.conv.DenseGCNConv;
import static org.junit.Assert.*;
import org.bytedeco.pytorch.*;


import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.global.torch;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.JUnitCore;
import org.junit.runner.Result;
import org.junit.runner.notification.Failure;
import org.bytedeco.pytorch.geometric.nn.conv.DenseGCNConv;

import java.util.Arrays;

import static org.junit.Assert.*;

public class DenseGCNConvTest {
    private DenseGCNConv gcnConv;
    private DenseGCNConv improvedGcnConv;
    private Tensor x; // [B=2, N=3, in_channels=4]
    private Tensor adj; // [B=2, N=3, N=3] 单位矩阵（仅自环）
    private Tensor mask; // [B=2, N=3] 掩码（屏蔽最后一个节点）

    @Before
    public void setUp() {
        // 1. 初始化 GCN 层：in=4, out=2（普通版/改进版）
        gcnConv = new DenseGCNConv(4, 2, false);
        improvedGcnConv = new DenseGCNConv(4, 2, true);

        // 2. 固定随机种子，保证数值可复现
        torch.manual_seed(1234);

        // 3. 构造输入特征：[2, 3, 4]
        x = torch.randn(new long[]{2, 3, 4});

        // 4. 构造邻接矩阵：单位矩阵（每个节点只连接自己）
        adj = torch.eye(3).unsqueeze(0).expand(new long[]{2, 3, 3});

        // 5. 构造掩码：[2, 3]，屏蔽最后一个节点（值为 0）
        // ========== 最终修复：构造 0 维标量张量 ==========
        mask = torch.ones(new long[]{2, 3});
        // 方案1：直接构造 0 维标量（推荐）
        Scalar zeroScalar = new Scalar(0f);
        mask.select(0, 0).select(0, 2).fill_(zeroScalar); // 0维标量填充
        mask.select(0, 1).select(0, 2).fill_(zeroScalar);

        // 方案2（备选）：将 1 维张量 squeeze 为 0 维
        // Tensor zeroTensor = torch.tensor(0f).squeeze(); // 从 [1] → []
        // mask.select(0, 0).select(0, 2).copy_(zeroTensor);
        // mask.select(0, 1).select(0, 2).copy_(zeroTensor);
    }

    @Test
    public void testForwardShape() {
        // 普通版 GCN 前向传播
        Tensor out = gcnConv.forward(x, adj, new TensorOptional());
        // 验证输出形状：[2, 3, 2]
        assertEquals(3, out.dim());
        assertEquals(2, out.size(0)); // B=2
        assertEquals(3, out.size(1)); // N=3
        assertEquals(2, out.size(2)); // out_channels=2

        // 改进版 GCN 形状验证
        Tensor improvedOut = improvedGcnConv.forward(x, adj, new TensorOptional());
        assertArrayEquals(out.sizes().vec().get(), improvedOut.sizes().vec().get());

        System.out.println("输出形状验证通过：" + Arrays.toString(out.sizes().vec().get()));
    }

    @Test
    public void testForwardNumerics() {
        // 1. 普通版 GCN 前向传播（自环邻接矩阵）
        Tensor out = gcnConv.forward(x, adj, new TensorOptional());
        // 2. 线性投影值（X @ W）
        Tensor xW = gcnConv.getLin().forward(x);

        // 3. 验证数值一致性：自环场景下，归一化邻接矩阵为单位矩阵，out = xW
        Tensor diff = out.sub(xW).abs();
        double maxDiff = diff.max().item_double();
        assertTrue("数值误差应小于 1e-5，实际：" + maxDiff, maxDiff < 1e-5);

        System.out.println("普通版 GCN 数值验证通过，最大误差：" + maxDiff);

        // 4. 改进版 GCN 数值验证（A+2I）
        Tensor improvedOut = improvedGcnConv.forward(x, adj, new TensorOptional());
        // 改进版自环数翻倍，度矩阵变化，数值应与普通版不同
        Tensor improvedDiff = improvedOut.sub(xW).abs();
        assertTrue("改进版 GCN 数值应与普通版不同", improvedDiff.max().item_double() > 1e-3);
        System.out.println("改进版 GCN 数值差异验证通过");
    }

    @Test
    public void testMaskFunction() {
        // 带掩码的前向传播
        TensorOptional maskOptional = new TensorOptional(mask);
        Tensor out = gcnConv.forward(x, adj, maskOptional);

        // 验证最后一个节点的输出为 0（被掩码屏蔽）
        Tensor lastNodeOut = out.select(1, 2); // 取第 3 个节点（索引 2）
        double maxLastNode = lastNodeOut.abs().max().item_double();
        assertTrue("掩码节点输出应接近 0，实际：" + maxLastNode, maxLastNode < 1e-5);

        System.out.println("掩码功能验证通过，屏蔽节点输出最大值：" + maxLastNode);
    }

    @After
    public void tearDown() {
        // 释放所有资源
        if (gcnConv != null) gcnConv.close();
        if (improvedGcnConv != null) improvedGcnConv.close();
        if (x != null) x.close();
        if (adj != null) adj.close();
        if (mask != null) mask.close();
//        torch.clear_autograd_graph();
    }

    public static void main(String[] args) {
        Result result = JUnitCore.runClasses(DenseGCNConvTest.class);
        for (Failure failure : result.getFailures()) {
            System.err.println("FAIL: " + failure.toString());
            if (failure.getException() != null) failure.getException().printStackTrace();
        }
        System.out.println("Tests run=" + result.getRunCount()
                + " failed=" + result.getFailureCount()
                + " ignored=" + result.getIgnoreCount());
        if (!result.wasSuccessful()) {
            throw new RuntimeException("DenseGCNConvTest had " + result.getFailureCount() + " failures");
        }
        System.out.println("✅ DenseGCNConvTest all tests passed");
    }
}
