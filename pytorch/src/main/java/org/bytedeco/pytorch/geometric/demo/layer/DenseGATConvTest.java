package org.bytedeco.pytorch.geometric.demo.layer;
import org.bytedeco.pytorch.jit.*;


import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.global.torch;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.bytedeco.pytorch.geometric.nn.conv.DenseGATConv;

import java.util.Arrays;

import static org.junit.Assert.*;

public class DenseGATConvTest {
    private DenseGATConv gatConv;
    private Tensor x; // [B=2, N=3, in_channels=4]
    private Tensor adj; // [B=2, N=3, N=3] 单位矩阵（自环）

    @Before
    public void setUp() {
        // 初始化 GAT 层：in=4, out=2, heads=3
        gatConv = new DenseGATConv(4, 2, 3);

        // 固定随机种子，保证数值可复现
        torch.manual_seed(1234);
        x = torch.randn(new long[]{2, 3, 4});

        // 邻接矩阵：单位矩阵（每个节点只连接自己）
        adj = torch.eye(3).unsqueeze(0).expand(new long[]{2, 3, 3});
    }

    @Test
    public void testForwardShape() {
        Tensor out = gatConv.forward(x, adj);
        // 验证输出形状：[2, 3, 3*2=6]
        assertEquals(3, out.dim());
        assertEquals(2, out.size(0));
        assertEquals(3, out.size(1));
        assertEquals(6, out.size(2));
        System.out.println("输出形状验证通过：" + Arrays.toString(out.shape()));
    }

    @Test
    public void testForwardNumerics() {
        Tensor out = gatConv.forward(x, adj);
        // 自环场景下：输出 = 线性投影后的值（alpha 对角线为 1）
        Tensor xProj = gatConv.lin.forward(x); // [2, 3, 6]
        // 验证数值一致性（浮点误差 < 1e-5）
        double maxDiff = out.sub(xProj).abs().max().item_double();
        assertTrue("数值误差应小于 1e-5，实际：" + maxDiff, maxDiff < 1e-5);
        System.out.println("数值逻辑验证通过，最大误差：" + maxDiff);
    }

//    @Test
//    public void testParameterRegistration() {
//        StringTensorDict params = gatConv.named_parameters();
//        // 验证参数数量：lin.weight + lin.bias + attSrc + attDst = 4
//        assertEquals(4, params.size());
//        // 验证注意力参数形状
//        boolean hasAttSrc = false, hasAttDst = false;
//        for (int i = 0; i < params.size(); i++) {
//            String name = params.get(i).first.getString();
//            Tensor param = params.get(i).second;
//            if (name.equals("attSrc")) {
//                hasAttSrc = true;
//                assertArrayEquals(new long[]{1, 3, 2}, param.sizes());
//            } else if (name.equals("attDst")) {
//                hasAttDst = true;
//                assertArrayEquals(new long[]{1, 3, 2}, param.sizes());
//            }
//        }
//        assertTrue("attSrc 参数未注册", hasAttSrc);
//        assertTrue("attDst 参数未注册", hasAttDst);
//        System.out.println("参数注册验证通过");
//    }

    @Test
    public void testMessagePassingLogic() {
        // 验证 MessagePassing 核心方法调用逻辑
        gatConv.forward(x, adj);
        // 验证注意力权重 alpha 非空（message 方法已使用）
        assertNotNull("alpha 权重应为非空", gatConv.alpha);
        // 验证 alpha 形状：[2, 3, 3, 3]
        assertArrayEquals(new long[]{2, 3, 3, 3}, gatConv.alpha.sizes().vec().get());
        System.out.println("MessagePassing 核心逻辑验证通过");
    }

    @After
    public void tearDown() {
        // 释放所有资源
        if (gatConv != null) gatConv.close();
        if (x != null) x.close();
        if (adj != null) adj.close();
//        torch.clear_autograd_graph();
    }
}
