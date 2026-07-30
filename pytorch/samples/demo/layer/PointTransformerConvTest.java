package samples.demo.layer;
import org.bytedeco.pytorch.jit.*;
import org.bytedeco.pytorch.nn.options.*;
import org.bytedeco.pytorch.nn.modules.container.*;
import org.bytedeco.pytorch.nn.modules.*;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.geometric.nn.conv.PointTransformerConv;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import static org.junit.Assert.*;

/**
 * PointTransformerConv 测试用例：
 * 1. 基础功能测试（形状/数值）
 * 2. 异常场景测试
 * 3. 资源释放测试
 */
public class PointTransformerConvTest {
    private PointTransformerConv conv;
    private Tensor x;          // 节点特征 [8, 4]
    private Tensor pos;        // 位置编码 [8, 3]
    private Tensor edgeIndex;  // 边索引 [2, 16]
    private SequentialImpl posNN;      // 位置编码网络
    private SequentialImpl attnNN;     // 注意力网络

    @Before
    public void setUp() {
        // 固定随机种子
        torch.manual_seed(42);

        // 1. 初始化测试数据
        // 节点特征：8个点，4维特征
        x = torch.randn(new long[]{8, 4});
        // 位置编码：8个点，3维坐标（点云）
        pos = torch.randn(new long[]{8, 3});
        // 边索引：[2, 16]（随机连接）
        edgeIndex = torch.randint(0, 8, new long[]{2, 16});

        // 2. 初始化位置编码网络（posNN：3维→4维）
        posNN = new SequentialImpl();
        posNN.push_back(new LinearImpl(3, 4));
        posNN.push_back(new ReLUImpl());
        posNN.push_back(new LinearImpl(4, 4));
        // 初始化注意力网络（attnNN：4维→4维）
        attnNN = new SequentialImpl();
        attnNN.push_back(new LinearImpl(4, 4));
        attnNN.push_back(new ReLUImpl());
        attnNN.push_back(new LinearImpl(4, 4));

        // 3. 初始化 PointTransformerConv
        conv = new PointTransformerConv(4, 4,3, posNN, attnNN);
    }

    @After
    public void tearDown() {
        // 释放所有资源
        if (x != null) x.close();
        if (pos != null) pos.close();
        if (edgeIndex != null) edgeIndex.close();
        if (conv != null) conv.close();
        if (posNN != null) posNN.close();
        if (attnNN != null) attnNN.close();
    }

    // ========== 1. 基础功能测试 ==========
    @Test
    public void testBasicForward() {
        // 前向传播
        Tensor output = conv.forward(x, pos, edgeIndex);

        // 验证输出形状：[8, 4]
        assertEquals("输出维度应为 2", 2, output.dim());
        assertEquals("输出节点数应为 8", 8, output.size(0));
        assertEquals("输出通道数应为 4", 4, output.size(1));
        System.out.println("✅ 基础前向测试通过：输出形状 = " + output.size(0) + "x" + output.size(1));

        // 验证数值非全零（确保逻辑生效）
        Tensor sumOutput = torch.sum(output);
        assertFalse("输出不应全零", sumOutput.item_float() == 0.0f);
        System.out.println("✅ 数值有效性测试通过：输出总和 = " + sumOutput.item_float());

        // 释放临时张量
        output.close();
        sumOutput.close();
    }

    // ========== 2. pos 为 null 时的测试 ==========
    @Test
    public void testForwardWithNullPos() {
        Tensor output = conv.forward(x, null, edgeIndex);

        assertEquals("输出节点数应为 8", 8, output.size(0));
        assertEquals("输出通道数应为 4", 4, output.size(1));
        System.out.println("✅ pos 为 null 测试通过：输出形状 = " + output.size(0) + "x" + output.size(1));

        output.close();
    }

    // ========== 3. 异常场景测试 ==========
    // 3.1 空 x 测试
    @Test
    public void testNullX() {
        try {
            conv.forward(null, pos, edgeIndex);
            fail("应抛出 x 为空异常");
        } catch (IllegalArgumentException e) {
            assertTrue("异常信息不匹配", e.getMessage().contains("节点特征 x 不能为空"));
            System.out.println("✅ 空 x 异常测试通过：" + e.getMessage());
        }
    }

    // 3.2 非法 edge_index 测试
    @Test
    public void testInvalidEdgeIndex() {
        Tensor wrongEdgeIndex = torch.randn(new long[]{3, 16}); // 3x16 非法形状
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

    // 3.3 pos 节点数不匹配测试
    @Test
    public void testPosNodeMismatch() {
        Tensor wrongPos = torch.randn(new long[]{10, 3}); // 10个节点（x 是8个）
        try {
            conv.forward(x, wrongPos, edgeIndex);
            fail("应抛出 pos 节点数不匹配异常");
        } catch (IllegalArgumentException e) {
            assertTrue("异常信息不匹配", e.getMessage().contains("pos 节点数必须与 x 一致"));
            System.out.println("✅ pos 节点数不匹配异常测试通过：" + e.getMessage());
        } finally {
            wrongPos.close();
        }
    }

    // ========== 4. 资源释放测试 ==========
    @Test
    public void testResourceRelease() {
        // 释放 conv
        conv.close();

        // 测试释放后调用 forward
        try {
            conv.forward(x, pos, edgeIndex);
            fail("释放后应抛出异常");
        } catch (IllegalStateException e) {
            assertTrue("异常信息不匹配", e.getMessage().contains("已释放资源"));
            System.out.println("✅ 释放后调用 forward 测试通过：" + e.getMessage());
        }

        // 重复释放测试
        conv.close();
        System.out.println("✅ 重复释放测试通过");
    }

    // ========== 主函数：运行所有测试 ==========
    public static void main(String[] args) {
        PointTransformerConvTest test = new PointTransformerConvTest();
        try {
            test.setUp();
            test.testBasicForward();
            test.testForwardWithNullPos();
            test.testNullX();
            test.testInvalidEdgeIndex();
            test.testPosNodeMismatch();
            test.testResourceRelease();
            System.out.println("\n✅ 所有 PointTransformerConv 测试通过！");
        } catch (Exception e) {
            System.err.println("❌ 测试失败：" + e.getMessage());
            e.printStackTrace();
        } finally {
            test.tearDown();
        }
    }
}