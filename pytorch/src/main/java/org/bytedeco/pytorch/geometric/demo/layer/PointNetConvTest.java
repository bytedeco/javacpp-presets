package org.bytedeco.pytorch.geometric.demo.layer;
import org.bytedeco.pytorch.jit.*;
import org.bytedeco.pytorch.nn.modules.container.*;
import org.bytedeco.pytorch.nn.modules.*;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.geometric.nn.conv.PointNetConv;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import static org.junit.Assert.*;
import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.geometric.nn.conv.PointNetConv;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import static org.junit.Assert.*;

/**
 * 修复版测试用例：
 * 1. 确保 localNN 输出维度 = globalNN 输入维度
 * 2. 全链路维度校验
 */
public class PointNetConvTest {
    private PointNetConv conv;
    private Tensor x;          // 节点特征 [10, 4]
    private Tensor pos;        // 3D 坐标 [10, 3]
    private Tensor edgeIndex;  // 边索引 [2, 20]
    private SequentialImpl localNN;  // 局部 MLP: 7 → 64 → 32
    private SequentialImpl globalNN; // 全局 MLP: 32 → 64 → 32（输入维度=localNN输出）

    @Before
    public void setUp() {
        // 固定随机种子
        torch.manual_seed(42);

        // 1. 初始化测试数据（维度严格对齐）
        x = torch.randn(new long[]{10, 4});          // 10节点，4维特征
        pos = torch.randn(new long[]{10, 3});        // 10节点，3维坐标
        edgeIndex = torch.randint(0, 10, new long[]{2, 20}); // 20条边

        // 2. 初始化 localNN：输入=7（4+3），输出=32（关键：与globalNN输入对齐）
        localNN = new SequentialImpl();
        localNN.push_back(new LinearImpl(7, 64));
        localNN.push_back(new ReLUImpl());
        localNN.push_back(new LinearImpl(64, 32));

        // 3. 初始化 globalNN：输入=32（=localNN输出），输出=32
        globalNN = new SequentialImpl();

        globalNN.push_back(new LinearImpl(32, 64));
        globalNN.push_back(new ReLUImpl());
        globalNN.push_back(new LinearImpl(64, 32));


        // 4. 初始化 PointNetConv（构造函数会校验维度）
        conv = new PointNetConv(localNN, globalNN, true);
    }

    @After
    public void tearDown() {
        // 释放所有资源
        if (x != null) x.close();
        if (pos != null) pos.close();
        if (edgeIndex != null) edgeIndex.close();
        if (conv != null) conv.close();
        if (localNN != null) localNN.close();
        if (globalNN != null) globalNN.close();
    }

    // ========== 核心测试：基础前向+维度校验 ==========
    @Test
    public void testBasicForward() {
        // 前向传播
        Tensor output = conv.forward(x, pos, edgeIndex);

        // 1. 验证输出维度（globalNN 输出 [10,32]）
        assertEquals("输出维度应为 2", 2, output.dim());
        assertEquals("输出节点数应为 10", 10, output.size(0));
        assertEquals("输出通道数应为 32", 32, output.size(1));
        System.out.println("✅ 维度校验通过：输出形状 = " + output.size(0) + "x" + output.size(1));

        // 2. 验证数值有效性（Max Pooling 生效）
        Tensor maxOutput = torch.max(output);
        Tensor minOutput = torch.min(output);
        assertFalse("输出不应全为极小值", minOutput.item_float() == -1e9);
        assertTrue("输出应有有效数值", maxOutput.item_float() > -1e9);
        System.out.println("✅ 数值有效性测试通过：max=" + maxOutput.item_float() + ", min=" + minOutput.item_float());

        // 释放临时张量
        output.close();
        maxOutput.close();
        minOutput.close();
    }

    // ========== 无节点特征测试（x=null） ==========
    @Test
    public void testForwardWithoutX() {
        // 初始化 localNN：输入=3（仅坐标），输出=32
        SequentialImpl localNNWithoutX = new SequentialImpl();
        localNNWithoutX.push_back(new LinearImpl(3, 64));
        localNNWithoutX.push_back(new ReLUImpl());
        localNNWithoutX.push_back(new LinearImpl(64, 32));
//
        // 初始化 globalNN：输入=32（匹配 localNN 输出）
        SequentialImpl globalNNWithoutX = new SequentialImpl();
        globalNNWithoutX.push_back(new LinearImpl(32, 64));
        globalNNWithoutX.push_back(new ReLUImpl());
        globalNNWithoutX.push_back(new LinearImpl(64, 32));
//
        PointNetConv convWithoutX = new PointNetConv(localNNWithoutX, globalNNWithoutX, true);

        // 前向传播（x=null）
        Tensor output = convWithoutX.forward(null, pos, edgeIndex);

        // 验证维度
        assertEquals("输出节点数应为 10", 10, output.size(0));
        assertEquals("输出通道数应为 32", 32, output.size(1));
        System.out.println("✅ 无特征 x 测试通过：输出形状 = " + output.size(0) + "x" + output.size(1));

        // 释放资源
        output.close();
        convWithoutX.close();
        localNNWithoutX.close();
        globalNNWithoutX.close();
    }

    // ========== 异常测试：维度不匹配的 localNN/globalNN ==========
    @Test
    public void testMLPDimMismatch() {
        // 构造维度不匹配的 MLP：localNN 输出 64，globalNN 输入 32
        SequentialImpl badLocalNN = new SequentialImpl();
        badLocalNN.push_back(new LinearImpl(7, 64));
        badLocalNN.push_back(new ReLUImpl());
        badLocalNN.push_back(new LinearImpl(64, 64));
//
        SequentialImpl badGlobalNN = new SequentialImpl();
        badGlobalNN.push_back( new LinearImpl(32, 64));

        try {
            new PointNetConv(badLocalNN, badGlobalNN, true);
            fail("应抛出维度不匹配异常");
        } catch (IllegalArgumentException e) {
            assertTrue("异常信息不匹配", e.getMessage().contains("localNN 输出维度(64) 与 globalNN 输入维度(32) 不匹配"));
            System.out.println("✅ 维度不匹配异常测试通过：" + e.getMessage());
        } finally {
            badLocalNN.close();
            badGlobalNN.close();
        }
    }

    // ========== 其他异常测试（无修改） ==========
    @Test
    public void testNullPos() {
        try {
            conv.forward(x, null, edgeIndex);
            fail("应抛出 pos 为空异常");
        } catch (IllegalArgumentException e) {
            assertTrue("异常信息不匹配", e.getMessage().contains("必须传入 pos 坐标参数"));
            System.out.println("✅ 空 pos 异常测试通过：" + e.getMessage());
        }
    }

    @Test
    public void testInvalidEdgeIndex() {
        Tensor wrongEdgeIndex = torch.randint(0, 10, new long[]{3, 20});
        try {
            conv.forward(x, pos, wrongEdgeIndex);
            fail("应抛出 edge_index 非法异常");
        } catch (IllegalArgumentException e) {
            assertTrue("异常信息不匹配", e.getMessage().contains("edge_index 必须是 [2, E] 形状"));
            System.out.println("✅ 非法 edge_index 异常测试通过：" + e.getMessage());
        } finally {
            wrongEdgeIndex.close();
        }
    }

    @Test
    public void testEdgeIndexOutOfBound() {
        Tensor outBoundEdgeIndex = torch.cat(
                new TensorVector(edgeIndex, torch.tensor(new long[]{10, 5}).view(2,1)),
                1
        );
        try {
            conv.forward(x, pos, outBoundEdgeIndex);
            fail("应抛出边索引越界异常");
        } catch (IllegalArgumentException e) {
            assertTrue("异常信息不匹配", e.getMessage().contains("包含非法节点索引"));
            System.out.println("✅ 边索引越界异常测试通过：" + e.getMessage());
        } finally {
            outBoundEdgeIndex.close();
        }
    }

    @Test
    public void testResourceRelease() {
        conv.close();
        try {
            conv.forward(x, pos, edgeIndex);
            fail("释放后应抛出异常");
        } catch (IllegalStateException e) {
            assertTrue("异常信息不匹配", e.getMessage().contains("已释放资源"));
            System.out.println("✅ 释放后调用 forward 测试通过：" + e.getMessage());
        }
        conv.close();
        System.out.println("✅ 重复释放测试通过");
    }

    // ========== 主函数：运行所有测试 ==========
    public static void main(String[] args) {
        PointNetConvTest test = new PointNetConvTest();
        try {
            test.setUp();
            test.testBasicForward();
            test.testForwardWithoutX();
            test.testMLPDimMismatch();
            test.testNullPos();
            test.testInvalidEdgeIndex();
            test.testEdgeIndexOutOfBound();
            test.testResourceRelease();
            System.out.println("\n🎉 所有 PointNetConv 测试通过！");
        } catch (Exception e) {
            System.err.println("❌ 测试失败：" + e.getMessage());
            e.printStackTrace();
        } finally {
            test.tearDown();
        }
    }
}

//public class PointNetConvTest {
//    private PointNetConv conv;
//    private Tensor x;          // 节点特征 [10, 4]
//    private Tensor pos;        // 3D 坐标 [10, 3]
//    private Tensor edgeIndex;  // 边索引 [2, 20]
//    private SequentialImpl localNN;  // 局部 MLP: 4+3=7 → 32
//    private SequentialImpl globalNN; // 全局 MLP: 32 → 32
//
//    @Before
//    public void setUp() {
//        // 固定随机种子，保证结果可复现
//        torch.manual_seed(42);
//
//        // 1. 初始化测试数据
//        x = torch.randn(new long[]{10, 4});          // 10个节点，4维特征
//        pos = torch.randn(new long[]{10, 3});        // 10个节点，3维坐标（3D点云）
//        edgeIndex = torch.randint(0, 10, new long[]{2, 20}); // 20条边
//
//        // 2. 初始化局部 MLP：输入=7（4特征+3坐标），输出=32
//        localNN = new SequentialImpl();
//        localNN.push_back(new LinearImpl(7, 64));
//        localNN.push_back(new ReLUImpl());
//        localNN.push_back(new LinearImpl(64, 32));
//
//        // 3. 初始化全局 MLP：输入=32，输出=32
//        globalNN = new SequentialImpl();
//        globalNN.push_back(new LinearImpl(32, 64));
//        globalNN.push_back(new ReLUImpl());
//        globalNN.push_back(new LinearImpl(64, 32));
//
//        // 4. 初始化 PointNetConv（添加自环）
//        conv = new PointNetConv(localNN, globalNN, true);
//    }
//
//    @After
//    public void tearDown() {
//        // 释放所有资源
//        if (x != null) x.close();
//        if (pos != null) pos.close();
//        if (edgeIndex != null) edgeIndex.close();
//        if (conv != null) conv.close();
//        if (localNN != null) localNN.close();
//        if (globalNN != null) globalNN.close();
//    }
//
//    // ========== 1. 基础功能测试 ==========
//    @Test
//    public void testBasicForward() {
//        // 前向传播
//        Tensor output = conv.forward(x, pos, edgeIndex);
//
//        // 验证输出形状：[10, 32]
//        assertEquals("输出维度应为 2", 2, output.dim());
//        assertEquals("输出节点数应为 10", 10, output.size(0));
//        assertEquals("输出通道数应为 32", 32, output.size(1));
//        System.out.println("✅ 基础前向测试通过：输出形状 = " + output.size(0) + "x" + output.size(1));
//
//        // 验证数值非全零（确保 Max Pooling 和 MLP 生效）
//        Tensor maxOutput = torch.max(output);
//        Tensor minOutput = torch.min(output);
//        assertFalse("输出不应全为极小值", minOutput.item_float() == -1e9);
//        assertTrue("输出应有有效数值", maxOutput.item_float() > -1e9);
//        System.out.println("✅ 数值有效性测试通过：max=" + maxOutput.item_float() + ", min=" + minOutput.item_float());
//
//        // 释放临时张量
//        output.close();
//        maxOutput.close();
//        minOutput.close();
//    }
//
//    // ========== 2. 无节点特征（x=null）测试 ==========
//    @Test
//    public void testForwardWithoutX() {
//        // 重新初始化局部 MLP：输入=3（仅坐标），输出=32
//        SequentialImpl localNNWithoutX = new SequentialImpl();
//        localNNWithoutX.push_back(new LinearImpl(3, 64));
//        localNNWithoutX.push_back(new ReLUImpl());
//        localNNWithoutX.push_back(new LinearImpl(64, 32));
//
//        PointNetConv convWithoutX = new PointNetConv(localNNWithoutX, globalNN, true);
//
//        // 前向传播（x=null）
//        Tensor output = convWithoutX.forward(null, pos, edgeIndex);
//
//        // 验证形状
//        assertEquals("输出节点数应为 10", 10, output.size(0));
//        assertEquals("输出通道数应为 32", 32, output.size(1));
//        System.out.println("✅ 无特征 x 测试通过：输出形状 = " + output.size(0) + "x" + output.size(1));
//
//        // 释放资源
//        output.close();
//        convWithoutX.close();
//        localNNWithoutX.close();
//    }
//
//    // ========== 3. 异常场景测试 ==========
//    // 3.1 空 pos 测试
//    @Test
//    public void testNullPos() {
//        try {
//            conv.forward(x, null, edgeIndex);
//            fail("应抛出 pos 为空异常");
//        } catch (IllegalArgumentException e) {
//            assertTrue("异常信息不匹配", e.getMessage().contains("必须传入 pos 坐标参数"));
//            System.out.println("✅ 空 pos 异常测试通过：" + e.getMessage());
//        }
//    }
//
//    // 3.2 非法 edge_index 测试（形状 3x20）
//    @Test
//    public void testInvalidEdgeIndex() {
//        Tensor wrongEdgeIndex = torch.randint(0, 10, new long[]{3, 20});
//        try {
//            conv.forward(x, pos, wrongEdgeIndex);
//            fail("应抛出 edge_index 非法异常");
//        } catch (IllegalArgumentException e) {
//            assertTrue("异常信息不匹配", e.getMessage().contains("edge_index 必须是 [2, E] 形状"));
//            System.out.println("✅ 非法 edge_index 异常测试通过：" + e.getMessage());
//        } finally {
//            wrongEdgeIndex.close();
//        }
//    }
//
//    // 3.3 边索引越界测试
//    @Test
//    public void testEdgeIndexOutOfBound() {
//        // 边索引包含 10（节点数为10，合法索引是 0-9）
//        Tensor outBoundEdgeIndex = torch.cat(
//                new TensorVector(edgeIndex, torch.tensor(new long[]{10, 5}).view(2,1)),
//                1
//        );
//        try {
//            conv.forward(x, pos, outBoundEdgeIndex);
//            fail("应抛出边索引越界异常");
//        } catch (IllegalArgumentException e) {
//            assertTrue("异常信息不匹配", e.getMessage().contains("包含非法节点索引"));
//            System.out.println("✅ 边索引越界异常测试通过：" + e.getMessage());
//        } finally {
//            outBoundEdgeIndex.close();
//        }
//    }
//
//    // ========== 4. 资源释放测试 ==========
//    @Test
//    public void testResourceRelease() {
//        // 释放 conv
//        conv.close();
//
//        // 测试释放后调用 forward
//        try {
//            conv.forward(x, pos, edgeIndex);
//            fail("释放后应抛出异常");
//        } catch (IllegalStateException e) {
//            assertTrue("异常信息不匹配", e.getMessage().contains("已释放资源"));
//            System.out.println("✅ 释放后调用 forward 测试通过：" + e.getMessage());
//        }
//
//        // 重复释放测试
//        conv.close();
//        System.out.println("✅ 重复释放测试通过");
//    }
//
//    // ========== 主函数：运行所有测试 ==========
//    public static void main(String[] args) {
//        PointNetConvTest test = new PointNetConvTest();
//        try {
//            test.setUp();
//            test.testBasicForward();
//            test.testForwardWithoutX();
//            test.testNullPos();
//            test.testInvalidEdgeIndex();
//            test.testEdgeIndexOutOfBound();
//            test.testResourceRelease();
//            System.out.println("\n✅ 所有 PointNetConv 测试通过！");
//        } catch (Exception e) {
//            System.err.println("❌ 测试失败：" + e.getMessage());
//            e.printStackTrace();
//        } finally {
//            test.tearDown();
//        }
//    }
//}
