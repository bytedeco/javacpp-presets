package samples.demo.layer;
import org.bytedeco.pytorch.jit.*;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.geometric.nn.conv.HypergraphConv;
import static org.junit.Assert.*;

public class HypergraphConvTest {
    private HypergraphConv conv;
    private HypergraphConv convWithAttention;
    private Tensor x;          // 节点特征 [5, 4]
    private Tensor hyperedgeIndex; // 超图索引 [2, 8]
    private Tensor hyperedgeWeight; // 超边权重 [3]

    public static void main(String[] args) {
        // 固定随机种子
        torch.manual_seed(42);

        HypergraphConvTest test = new HypergraphConvTest();
        try {
            test.setUp();
            test.testBasicForward();
            test.testForwardWithAttention();
            test.testForwardWithHyperedgeWeight();
            test.testInputValidation();
            test.testResourceRelease();
            System.out.println("\n🎉 所有 HypergraphConv 测试通过！");
        } catch (Exception e) {
            System.err.println("❌ 测试失败：" + e.getMessage());
            e.printStackTrace();
        } finally {
            test.tearDown();
        }
    }

    private void setUp() {
        // 1. 初始化测试数据
        // 节点特征：5个节点，每个节点4维特征
        x = torch.randn(new long[]{5, 4});

        // 超图索引：[2,8] 
        // 超边0：包含节点0,1,2
        // 超边1：包含节点2,3
        // 超边2：包含节点3,4,0
        hyperedgeIndex = torch.tensor(new long[]
                {0, 1, 2, 2, 3, 3, 4, 0, // 节点索引
                0, 0, 0, 1, 1, 2, 2, 2}  // 超边索引
        ).view(2,8);

        // 超边权重：3个超边的权重
        hyperedgeWeight = torch.tensor(new float[]{1.0f, 2.0f, 0.5f});

        // 2. 初始化卷积层
        // 无注意力：输入4维 → 单头输出2维
        conv = new HypergraphConv(4, 2, false, 1, true);
        // 有注意力：输入4维 → 2头，每头输出2维，拼接后4维
        convWithAttention = new HypergraphConv(4, 2, true, 2, true);
    }

    private void tearDown() {
        // 释放所有资源
        safeClose(x, hyperedgeIndex, hyperedgeWeight);
        safeClose(conv, convWithAttention);
    }

    /**
     * 安全释放工具方法
     */
    private void safeClose(AutoCloseable... closeables) {
        for (AutoCloseable c : closeables) {
            if (c != null) {
                try {
                    c.close();
                } catch (Exception e) {
                    System.err.println("测试释放资源警告：" + e.getMessage());
                }
            }
        }
    }

    /**
     * 基础前向传播测试（无注意力）
     */
    private void testBasicForward() {
        Tensor output = ((HypergraphConv)conv).forward(x, hyperedgeIndex, (Tensor)null);

        // 验证输出维度：5个节点，1头×2维=2维
        assertEquals("输出维度应为2", 2, output.dim());
        assertEquals("输出节点数应为5", 5, output.size(0));
        assertEquals("输出特征维度应为2", 2, output.size(1));

        // 验证数值有效性（非全零）
        Tensor sumOutput = torch.sum(output);
        assertFalse("输出不应全为零", sumOutput.item_float() == 0.0f);

        System.out.println("✅ 基础前向传播测试通过：输出形状 = " + output.size(0) + "x" + output.size(1));

        // 释放临时张量
        safeClose(output, sumOutput);
    }

    /**
     * 注意力机制测试
     */
    private void testForwardWithAttention() {
        Tensor output = ((HypergraphConv)convWithAttention).forward(x, hyperedgeIndex, (Tensor)null);

        // 验证输出维度：5个节点，2头×2维=4维
        assertEquals("输出节点数应为5", 5, output.size(0));
        assertEquals("输出特征维度应为4", 4, output.size(1));

        System.out.println("✅ 注意力机制测试通过：输出形状 = " + output.size(0) + "x" + output.size(1));

        // 释放临时张量
        safeClose(output);
    }

    /**
     * 超边权重测试
     */
    private void testForwardWithHyperedgeWeight() {
        Tensor output1 = ((HypergraphConv)conv).forward(x, hyperedgeIndex, (Tensor)null);
        Tensor output2 = ((HypergraphConv)conv).forward(x, hyperedgeIndex, hyperedgeWeight);

        // 验证权重生效（输出不应相同）
        Tensor diff = torch.abs(output1.sub(output2)).sum();
        assertTrue("超边权重应改变输出结果", diff.item_float() > 1e-6);

        System.out.println("✅ 超边权重测试通过：权重生效，差值=" + diff.item_float());

        // 释放临时张量
        safeClose(output1, output2, diff);
    }

    /**
     * 输入参数校验测试（异常场景）
     */
    private void testInputValidation() {
        // 测试1：空节点特征
        try {
            ((HypergraphConv)conv).forward(null, hyperedgeIndex, (Tensor)null);
            fail("应抛出空节点特征异常");
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage().contains("节点特征 x 不能为空"));
            System.out.println("✅ 空节点特征异常测试通过");
        }

        // 测试2：空超图索引
        try {
            ((HypergraphConv)conv).forward(x, null, (Tensor)null);
            fail("应抛出空超图索引异常");
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage().contains("超图索引 hyperedge_index 不能为空"));
            System.out.println("✅ 空超图索引异常测试通过");
        }

        // 测试3：维度不匹配的节点特征
        Tensor badX = torch.randn(new long[]{5, 5}); // 输入维度应为4，实际5
        try {
            ((HypergraphConv)conv).forward(badX, hyperedgeIndex, (Tensor)null);
            fail("应抛出维度不匹配异常");
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage().contains("节点特征维度不匹配"));
            System.out.println("✅ 节点特征维度异常测试通过");
        }
        safeClose(badX);

        // 测试4：非法超图索引形状
        Tensor badEdgeIndex = torch.randn(new long[]{3, 8}); // 应为[2,8]，实际[3,8]
        try {
            ((HypergraphConv)conv).forward(x, badEdgeIndex, (Tensor)null);
            fail("应抛出超图索引形状异常");
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage().contains("超图索引 hyperedge_index 必须是 [2, num_incidences] 形状"));
            System.out.println("✅ 超图索引形状异常测试通过");
        }
        safeClose(badEdgeIndex);

        // 测试5：超边权重数量不匹配
        Tensor badWeight = torch.tensor(new float[]{1.0f, 2.0f}); // 应为3个，实际2个
        try {
            ((HypergraphConv)conv).forward(x, hyperedgeIndex, badWeight);
            fail("应抛出超边权重数量异常");
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage().contains("超边权重数量不匹配"));
            System.out.println("✅ 超边权重数量异常测试通过");
        }
        safeClose(badWeight);
    }

    /**
     * 资源释放测试
     */
    private void testResourceRelease() {
        // 释放卷积层
        conv.close();

        // 尝试再次前向传播，应抛出异常
        try {
            ((HypergraphConv)conv).forward(x, hyperedgeIndex, (Tensor)null);
            fail("释放后应抛出异常");
        } catch (IllegalStateException e) {
            assertTrue(e.getMessage().contains("已释放资源，无法继续使用"));
            System.out.println("✅ 资源释放测试通过");
        }

        // 重复释放测试（不应报错）
        conv.close();
        System.out.println("✅ 重复释放测试通过");
    }
}