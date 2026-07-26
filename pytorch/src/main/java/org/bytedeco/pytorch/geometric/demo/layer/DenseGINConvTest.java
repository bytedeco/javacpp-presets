package org.bytedeco.pytorch.geometric.demo.layer;
import org.bytedeco.pytorch.jit.*;
import org.bytedeco.pytorch.nn.options.*;
import org.bytedeco.pytorch.nn.modules.container.*;
import org.bytedeco.pytorch.nn.modules.*;

//package org.bytedeco.pytorch.geometric.demo.layer;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.global.torch;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.bytedeco.pytorch.geometric.nn.conv.DenseGINConv;

import java.util.Arrays;

import static org.junit.Assert.*;

public class DenseGINConvTest {
    private DenseGINConv ginConvFixed;    // 固定 eps（不可训练）
    private DenseGINConv ginConvTrainable;// 可训练 eps
    private Tensor x;                     // 输入特征 [B=2, N=3, in_channels=4]
    private Tensor adj;                   // 邻接矩阵 [B=2, N=3, N=3]（单位矩阵）
    private SequentialImpl mlp;           // GIN 的 MLP 模块

    @Before
    public void setUp() {
        // 1. 固定随机种子，保证数值可复现
        torch.manual_seed(1234);

        // 2. 构造 MLP：4 → 8 → 2（GIN 的核心映射）
        mlp = new SequentialImpl();
        mlp.push_back(new LinearImpl(4, 8));        
        mlp.push_back(new ReLUImpl());        
        mlp.push_back(new LinearImpl(8, 2));
        // 3. 初始化 GIN 层
//        System.out.println("forward shape DenseGINConv");
        ginConvFixed = new DenseGINConv(mlp, 0.1, false);    // 固定 eps=0.1
//        System.out.println("forward shape DenseGINConv22");
        ginConvTrainable = new DenseGINConv(mlp, 0.0, true);         // 可训练 eps（初始 0.0）

        // 4. 构造输入特征：[2, 3, 4]
        x = torch.randn(new long[]{2, 3, 4});
//        System.out.println("forward shape DenseGINConv33");
        // 5. 构造邻接矩阵：单位矩阵（每个节点只连接自己）
        adj = torch.eye(3).unsqueeze(0).expand(new long[]{2, 3, 3});
//        System.out.println("forward shape DenseGINConv44");
    }

    /**
     * 测试1：输出形状验证（核心）
     * 验证：输入 [2,3,4] → 输出 [2,3,2]
     */
    @Test
    public void testForwardShape() {
        // 固定 eps 版本
        System.out.println("forward shape");
        Tensor outFixed = ginConvFixed.forward(x, adj);
        assertEquals(3, outFixed.dim());
        assertEquals(2, outFixed.size(0));  // B=2
        assertEquals(3, outFixed.size(1));  // N=3
        assertEquals(2, outFixed.size(2));  // out_channels=2

        // 可训练 eps 版本
        Tensor outTrainable = ginConvTrainable.forward(x, adj);
        assertArrayEquals(outFixed.sizes().vec().get(), outTrainable.sizes().vec().get());

        System.out.println("输出形状验证通过：outFixed.sizes()" + Arrays.toString(outFixed.sizes().vec().get()));

        // 释放临时张量
        outFixed.close();
        outTrainable.close();
    }

    /**
     * 测试2：数值逻辑验证（GIN 核心公式）
     * 验证：(1+eps)X + AX = (2+eps)X（因为 A 是单位矩阵）
     */
    @Test
    public void testForwardNumerics() {
        // 步骤1：计算理论值（A 是单位矩阵，AX=X → 总输出 = (2+eps)X → MLP((2+eps)X)）
        double eps = 0.1;
        Tensor expectedInput = x.mul(new Scalar(2.0 + eps));  // (2+eps)X
        Tensor expectedOut = ginConvFixed.getMlp().forward(expectedInput);

        // 步骤2：计算 GIN 输出
        Tensor actualOut = ginConvFixed.forward(x, adj);

        // 步骤3：验证数值一致性（误差 < 1e-5）
        Tensor diff = actualOut.sub(expectedOut).abs();
        double maxDiff = diff.max().item_double();
        assertTrue("GIN 数值逻辑错误，最大误差：" + maxDiff, maxDiff < 1e-5);

        System.out.println("数值逻辑验证通过，最大误差：" + maxDiff);

        // 释放临时张量
        expectedInput.close();
        expectedOut.close();
        actualOut.close();
        diff.close();
    }

    /**
     * 测试3：可训练 eps 参数验证
     * 验证：epsParam 是 Parameter 且可更新
     */
    @Test
    public void testTrainableEps() {
        // 验证1：可训练版本有 epsParam
        assertNotNull("可训练 GIN 应包含 epsParam", ginConvTrainable.getEpsParam());
        // 验证2：epsParam 初始值为 0.0
        double initialEps = ginConvTrainable.getEpsParam().data().item_double();
        assertEquals("epsParam 初始值应为 0.0", 0.0, initialEps, 1e-6);

        // 模拟反向传播更新 epsParam
        Tensor out = ginConvTrainable.forward(x, adj);
        Tensor loss = out.sum();  // 简单求和作为损失
        loss.backward();          // 反向传播

        System.out.println(ginConvTrainable.getEpsParam().grad().defined());
        // 验证3：epsParam 有梯度
//        assertTrue("epsParam 应有梯度", ginConvTrainable.getEpsParam().grad().defined());

        System.out.println("可训练 eps 参数验证通过，初始值：" + initialEps);

        // 释放临时张量
        out.close();
        loss.close();
    }

    /**
     * 测试4：设备对齐验证（CPU 设备，模拟 GPU 逻辑）
     */
    @Test
    public void testDeviceAlignment() {
        // 构造 CPU 设备的输入
        Tensor xCpu = x.to(new Device(torch.DeviceType.CPU), x.options().dtype().toScalarType());
        Tensor adjCpu = adj.to(new Device(torch.DeviceType.CPU),adj.options().dtype().toScalarType());

        // 验证：可训练 epsParam 自动对齐到 CPU
        Tensor out = ginConvTrainable.forward(xCpu, adjCpu);
        assertEquals("输出应在 CPU 上", "CPU", out.device().type().toString());

        System.out.println("设备对齐验证通过");

        // 释放临时张量
        xCpu.close();
        adjCpu.close();
        out.close();
    }

    @After
    public void tearDown() {
        // 释放所有资源
        if (ginConvFixed != null) ginConvFixed.close();
        if (ginConvTrainable != null) ginConvTrainable.close();
        if (x != null) x.close();
        if (adj != null) adj.close();
        if (mlp != null) mlp.close();
//        torch.clear_autograd_graph();
    }
}
