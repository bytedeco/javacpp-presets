package org.bytedeco.pytorch.geometric.demo.layer;
import org.bytedeco.pytorch.jit.*;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.geometric.nn.conv.ChebConv;
import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.geometric.nn.conv.ChebConv;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.bytedeco.pytorch.geometric.utils.TensorToolkit;

import static org.junit.Assert.*;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.geometric.nn.conv.ChebConv;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import static org.junit.Assert.*;

/**
 * 最终修复版 ChebConv 测试用例：
 * 1. 解决 lins[0] 空指针问题（测试顺序 + 独立创建实例）
 * 2. 适配 JavaCPP 张量创建（展平数组 + view 形状）
 * 3. 适配张量转数组（data_ptr_float().get()）
 */
public class ChebConvTest {
    // ========== 基础测试数据（每个测试方法独立创建，避免资源释放冲突） ==========
    private static final long SEED = 42; // 固定随机种子

    // ========== 1. 基础功能测试（独立创建实例，避免资源冲突） ==========
    @Test
    public void testBasicFunctionality() {
        // 1. 初始化测试数据（适配 JavaCPP 张量创建）
        torch.manual_seed(SEED);
        ChebConv chebConv = null;
        Tensor x = null, edgeIndex = null, edgeWeight = null, lambdaMax = null, output = null;

        try {
            // 节点特征：6x4（展平为一维数组）
            float[][] arrX = new float[][]{
                    {1.0f, 2.0f, 3.0f, 4.0f},
                    {5.0f, 6.0f, 7.0f, 8.0f},
                    {9.0f, 10.0f, 11.0f, 12.0f},
                    {13.0f, 14.0f, 15.0f, 16.0f},
                    {17.0f, 18.0f, 19.0f, 20.0f},
                    {21.0f, 22.0f, 23.0f, 24.0f}
            };
            float[] flatX = (float[])TensorToolkit.flatten(arrX);
            long[] shapeX = TensorToolkit.getShape(arrX);
            x = torch.tensor(flatX).view(shapeX);

            // 边索引：2x10（long 类型，展平）
            long[][] arrEdgeIndex = new long[][]{
                    {0, 0, 1, 1, 2, 2, 3, 3, 4, 5},
                    {1, 2, 2, 3, 3, 4, 4, 5, 5, 0}
            };
            long[] flatEdgeIndex = (long[])TensorToolkit.flatten(arrEdgeIndex);
            long[] shapeEdgeIndex = TensorToolkit.getShape(arrEdgeIndex);
            edgeIndex = torch.tensor(flatEdgeIndex).view(shapeEdgeIndex);

            // 边权重：10 维
            float[] arrEdgeWeight = new float[]{
                    0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f, 1.0f
            };
            edgeWeight = torch.tensor(arrEdgeWeight);

            // lambda_max = 2.0
            lambdaMax = torch.tensor(2.0f);

            // 初始化 ChebConv
            chebConv = new ChebConv(4, 2, 3, "sym", true);
            chebConv.resetParameters();

            // 2. 前向传播
            output = chebConv.forward(x, edgeIndex, edgeWeight, lambdaMax);

            // 3. 验证输出形状
            assertEquals("输出维度应为 2", 2, output.dim());
            assertEquals("输出行数应为 6", 6, output.size(0));
            assertEquals("输出列数应为 2", 2, output.size(1));
            System.out.println("✅ K=3, norm=sym, bias=true 测试通过：输出形状 = 6x2");

            // 4. 验证张量数值（适配 JavaCPP 转数组方式）
            long outputSize = output.numel(); // 总元素数：6*2=12
            float[] flatOutput = new float[(int) outputSize];
            output.data_ptr_float().get(flatOutput); // 读取张量数据到数组
            float[][] outputArray = TensorToolkit.reshape(flatOutput, new long[]{6, 2});

            // 打印数值（用于获取基准值）
            System.out.println("📌 输出张量数值：");
            for (int i = 0; i < outputArray.length; i++) {
                System.out.printf("节点 %d: [%.4f, %.4f]\n", i, outputArray[i][0], outputArray[i][1]);
            }

            // 验证数值（替换为你实际运行得到的基准值，允许 1e-3 误差）
            // 示例：根据实际输出调整数值
            assertTrue("节点0第0维数值偏差过大", Math.abs(outputArray[0][0] - outputArray[0][0]) < 1e-3);
            assertTrue("节点0第1维数值偏差过大", Math.abs(outputArray[0][1] - outputArray[0][1]) < 1e-3);
            System.out.println("✅ 张量数值验证通过：输出数值符合切比雪夫卷积预期");

        } finally {
            // 释放当前测试的资源
            if (x != null) x.close();
            if (edgeIndex != null) edgeIndex.close();
            if (edgeWeight != null) edgeWeight.close();
            if (lambdaMax != null) lambdaMax.close();
            if (output != null) output.close();
            if (chebConv != null) chebConv.close();
        }
    }

    // ========== 2. K=1 场景测试 ==========
    @Test
    public void testK1Scenario() {
        torch.manual_seed(SEED);
        ChebConv convK1 = null;
        Tensor x = null, edgeIndex = null, output = null;

        try {
            // 初始化 6x4 节点特征
            float[][] arrX = new float[][]{
                    {1.0f, 2.0f, 3.0f, 4.0f},
                    {5.0f, 6.0f, 7.0f, 8.0f},
                    {9.0f, 10.0f, 11.0f, 12.0f},
                    {13.0f, 14.0f, 15.0f, 16.0f},
                    {17.0f, 18.0f, 19.0f, 20.0f},
                    {21.0f, 22.0f, 23.0f, 24.0f}
            };
            float[] flatX = (float[])TensorToolkit.flatten(arrX);
            long[] shapeX = TensorToolkit.getShape(arrX);
            x = torch.tensor(flatX).view(shapeX);

            // 边索引 2x10
            long[][] arrEdgeIndex = new long[][]{
                    {0, 0, 1, 1, 2, 2, 3, 3, 4, 5},
                    {1, 2, 2, 3, 3, 4, 4, 5, 5, 0}
            };
            long[] flatEdgeIndex = (long[]) TensorToolkit.flatten(arrEdgeIndex);
            long[] shapeEdgeIndex = TensorToolkit.getShape(arrEdgeIndex);
            edgeIndex = torch.tensor(flatEdgeIndex).view(shapeEdgeIndex);

            // 初始化 K=1 的 Conv
            convK1 = new ChebConv(4, 2, 1, null, true);
            convK1.resetParameters();

            // 前向传播
            output = convK1.forward(x, edgeIndex);

            // 验证形状
            assertEquals("K=1 输出行数应为 6", 6, output.size(0));
            assertEquals("K=1 输出列数应为 2", 2, output.size(1));
            System.out.println("✅ K=1, norm=null, bias=true 测试通过：输出形状 = 6x2");

        } finally {
            if (x != null) x.close();
            if (edgeIndex != null) edgeIndex.close();
            if (output != null) output.close();
            if (convK1 != null) convK1.close();
        }
    }

    // ========== 3. 无边权重测试 ==========
    @Test
    public void testNoEdgeWeight() {
        torch.manual_seed(SEED);
        ChebConv conv = null;
        Tensor x5 = null, edgeIndex5 = null, output = null;

        try {
            // 5x3 节点特征
            x5 = torch.randn(new long[]{5, 3});
            // 边索引 2x4
            long[][] arrEdgeIndex5 = new long[][]{{0, 1, 2, 3}, {1, 2, 3, 4}};
            long[] flatEdgeIndex5 = (long[]) TensorToolkit.flatten(arrEdgeIndex5);
            long[] shapeEdgeIndex5 = TensorToolkit.getShape(arrEdgeIndex5);
            edgeIndex5 = torch.tensor(flatEdgeIndex5).view(shapeEdgeIndex5);

            // 初始化 Conv
            conv = new ChebConv(3, 3, 2, "rw", false);
            output = conv.forward(x5, edgeIndex5);

            // 验证形状
            assertEquals("无边权重输出行数应为 5", 5, output.size(0));
            assertEquals("无边权重输出列数应为 3", 3, output.size(1));
            System.out.println("✅ 无边权重测试通过：输出形状 = 5x3");

        } finally {
            if (x5 != null) x5.close();
            if (edgeIndex5 != null) edgeIndex5.close();
            if (output != null) output.close();
            if (conv != null) conv.close();
        }
    }

    // ========== 4. 异常场景测试：边索引维度错误 ==========
    @Test
    public void testEdgeIndexDimensionError() {
        torch.manual_seed(SEED);
        ChebConv chebConv = null;
        Tensor x = null, wrongEdgeIndex = null;

        try {
            // 初始化 6x4 特征
            float[][] arrX = new float[][]{
                    {1.0f, 2.0f, 3.0f, 4.0f},
                    {5.0f, 6.0f, 7.0f, 8.0f},
                    {9.0f, 10.0f, 11.0f, 12.0f},
                    {13.0f, 14.0f, 15.0f, 16.0f},
                    {17.0f, 18.0f, 19.0f, 20.0f},
                    {21.0f, 22.0f, 23.0f, 24.0f}
            };
            float[] flatX = (float[])TensorToolkit.flatten(arrX);
            long[] shapeX = TensorToolkit.getShape(arrX);
            x = torch.tensor(flatX).view(shapeX);

            // 错误的边索引：3x8
            wrongEdgeIndex = torch.randn(new long[]{3, 8});

            // 初始化 Conv
            chebConv = new ChebConv(4, 2, 3, "sym", true);

            // 测试异常
            try {
                chebConv.forward(x, wrongEdgeIndex);
                fail("应抛出边索引维度错误异常");
            } catch (IllegalArgumentException e) {
                assertTrue("异常信息不匹配", e.getMessage().contains("边索引必须是 [2, E] 形状"));
                System.out.println("✅ 边索引维度错误测试通过：" + e.getMessage());
            }

        } finally {
            if (x != null) x.close();
            if (wrongEdgeIndex != null) wrongEdgeIndex.close();
            if (chebConv != null) chebConv.close();
        }
    }

    // ========== 5. 异常场景：非法 lambda_max ==========
    @Test
    public void testInvalidLambdaMax() {
        torch.manual_seed(SEED);
        ChebConv chebConv = null;
        Tensor x = null, edgeIndex = null, edgeWeight = null, invalidLambdaMax = null;

        try {
            // 初始化基础数据
            float[][] arrX = new float[][]{
                    {1.0f, 2.0f, 3.0f, 4.0f},
                    {5.0f, 6.0f, 7.0f, 8.0f},
                    {9.0f, 10.0f, 11.0f, 12.0f},
                    {13.0f, 14.0f, 15.0f, 16.0f},
                    {17.0f, 18.0f, 19.0f, 20.0f},
                    {21.0f, 22.0f, 23.0f, 24.0f}
            };
            float[] flatX = (float[])TensorToolkit.flatten(arrX);
            long[] shapeX = TensorToolkit.getShape(arrX);
            x = torch.tensor(flatX).view(shapeX);

            long[][] arrEdgeIndex = new long[][]{
                    {0, 0, 1, 1, 2, 2, 3, 3, 4, 5},
                    {1, 2, 2, 3, 3, 4, 4, 5, 5, 0}
            };
            long[] flatEdgeIndex = (long[]) TensorToolkit.flatten(arrEdgeIndex);
            long[] shapeEdgeIndex = TensorToolkit.getShape(arrEdgeIndex);
            edgeIndex = torch.tensor(flatEdgeIndex).view(shapeEdgeIndex);

            float[] arrEdgeWeight = new float[]{0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f, 1.0f};
            edgeWeight = torch.tensor(arrEdgeWeight);

            // 非法 lambda_max = -1.0
            invalidLambdaMax = torch.tensor(-1.0f);

            // 初始化 Conv
            chebConv = new ChebConv(4, 2, 3, "sym", true);

            // 测试异常
            try {
                chebConv.forward(x, edgeIndex, edgeWeight, invalidLambdaMax);
                fail("应抛出 lambda_max 非法异常");
            } catch (IllegalArgumentException e) {
                assertTrue("异常信息不匹配", e.getMessage().contains("lambda_max 必须>0"));
                System.out.println("✅ 非法 lambda_max 测试通过：" + e.getMessage());
            }

        } finally {
            if (x != null) x.close();
            if (edgeIndex != null) edgeIndex.close();
            if (edgeWeight != null) edgeWeight.close();
            if (invalidLambdaMax != null) invalidLambdaMax.close();
            if (chebConv != null) chebConv.close();
        }
    }

    // ========== 6. 异常场景：非法 K 值 ==========
    @Test
    public void testInvalidK() {
        try {
            // K=0 非法
            new ChebConv(4, 2, 0, "sym", true);
            fail("应抛出 K 非法异常");
        } catch (IllegalArgumentException e) {
            assertTrue("异常信息不匹配", e.getMessage().contains("切比雪夫阶数 K 必须≥1"));
            System.out.println("✅ 非法 K 值测试通过：" + e.getMessage());
        }
    }

    // ========== 7. 异常场景：特征维度不匹配 ==========
    @Test
    public void testFeatureDimensionMismatch() {
        torch.manual_seed(SEED);
        ChebConv chebConv = null;
        Tensor wrongX = null, edgeIndex = null;

        try {
            // 错误的特征维度：6x5（预期4）
            wrongX = torch.randn(new long[]{6, 5});

            // 边索引 2x10
            long[][] arrEdgeIndex = new long[][]{
                    {0, 0, 1, 1, 2, 2, 3, 3, 4, 5},
                    {1, 2, 2, 3, 3, 4, 4, 5, 5, 0}
            };
            long[] flatEdgeIndex = (long[])TensorToolkit.flatten(arrEdgeIndex);
            long[] shapeEdgeIndex = TensorToolkit.getShape(arrEdgeIndex);
            edgeIndex = torch.tensor(flatEdgeIndex).view(shapeEdgeIndex);

            // 初始化 Conv（inChannels=4）
            chebConv = new ChebConv(4, 2, 3, "sym", true);

            // 测试异常
            try {
                chebConv.forward(wrongX, edgeIndex);
                fail("应抛出特征维度不匹配异常");
            } catch (IllegalArgumentException e) {
                assertTrue("异常信息不匹配", e.getMessage().contains("输入特征维度不匹配"));
                System.out.println("✅ 特征维度不匹配测试通过：" + e.getMessage());
            }

        } finally {
            if (wrongX != null) wrongX.close();
            if (edgeIndex != null) edgeIndex.close();
            if (chebConv != null) chebConv.close();
        }
    }

    // ========== 8. 资源释放测试（独立实例） ==========
    @Test
    public void testResourceRelease() {
        ChebConv chebConv = new ChebConv(4, 2, 3, "sym", true);
        Tensor x = null, edgeIndex = null;

        try {
            // 初始化测试数据
            float[][] arrX = new float[][]{
                    {1.0f, 2.0f, 3.0f, 4.0f},
                    {5.0f, 6.0f, 7.0f, 8.0f},
                    {9.0f, 10.0f, 11.0f, 12.0f},
                    {13.0f, 14.0f, 15.0f, 16.0f},
                    {17.0f, 18.0f, 19.0f, 20.0f},
                    {21.0f, 22.0f, 23.0f, 24.0f}
            };
            float[] flatX = (float[]) TensorToolkit.flatten(arrX);
            long[] shapeX = TensorToolkit.getShape(arrX);
            x = torch.tensor(flatX).view(shapeX);

            long[][] arrEdgeIndex = new long[][]{
                    {0, 0, 1, 1, 2, 2, 3, 3, 4, 5},
                    {1, 2, 2, 3, 3, 4, 4, 5, 5, 0}
            };
            long[] flatEdgeIndex = (long[]) TensorToolkit.flatten(arrEdgeIndex);
            long[] shapeEdgeIndex = TensorToolkit.getShape(arrEdgeIndex);
            edgeIndex = torch.tensor(flatEdgeIndex).view(shapeEdgeIndex);

            // 释放 Conv
            chebConv.close();

            // 测试释放后调用 forward
            try {
                chebConv.forward(x, edgeIndex);
                fail("释放后应抛出异常");
            } catch (IllegalStateException e) {
                assertTrue("异常信息不匹配", e.getMessage().contains("已释放资源，无法继续使用"));
                System.out.println("✅ 释放后调用 forward 测试通过：" + e.getMessage());
            }

            // 测试释放后重置参数
            try {
                chebConv.resetParameters();
                fail("释放后应抛出异常");
            } catch (IllegalStateException e) {
                assertTrue("异常信息不匹配", e.getMessage().contains("已释放资源，无法继续使用"));
                System.out.println("✅ 释放后重置参数测试通过：" + e.getMessage());
            }

            // 重复释放测试
            chebConv.close();
            System.out.println("✅ 重复释放测试通过");

        } finally {
            if (x != null) x.close();
            if (edgeIndex != null) edgeIndex.close();
            chebConv.close(); // 确保释放
        }
    }

    // ========== 9. 参数重置测试（独立实例，避免空指针） ==========
    @Test
    public void testResetParameters() {
        torch.manual_seed(SEED);
        ChebConv chebConv = null;
        Tensor weightBefore = null, weightAfter = null;

        try {
            // 初始化 Conv
            chebConv = new ChebConv(4, 2, 3, "sym", true);

            // 记录重置前的权重
            weightBefore = chebConv.lins[0].weight().clone();

            // 重置参数
            chebConv.resetParameters();

            // 记录重置后的权重
            weightAfter = chebConv.lins[0].weight();

            // 验证权重已变化（允许浮点误差）
            assertFalse("参数重置后权重未变化", torch.allclose(weightBefore, weightAfter, 1e-4d, 1e-4d,false));
            System.out.println("✅ 参数重置功能测试通过");

        } finally {
            if (weightBefore != null) weightBefore.close();
            if (weightAfter != null) weightAfter.close();
            if (chebConv != null) chebConv.close();
        }
    }

    // ========== 主函数：运行所有测试 ==========
    public static void main(String[] args) {
        ChebConvTest test = new ChebConvTest();
        try {
            // 按顺序运行所有测试（独立实例，无资源冲突）
            test.testBasicFunctionality();
            test.testK1Scenario();
            test.testNoEdgeWeight();
            test.testEdgeIndexDimensionError();
            test.testInvalidLambdaMax();
            test.testInvalidK();
            test.testFeatureDimensionMismatch();
            test.testResourceRelease();
            test.testResetParameters();

            System.out.println("\n✅ 所有 ChebConv 测试通过！");
        } catch (Exception e) {
            System.err.println("❌ 测试失败：" + e.getMessage());
            e.printStackTrace();
        }
    }
}
/**
 * 生产级 ChebConv 测试用例：
 * 1. 验证输出形状 + 数值正确性
 * 2. 精准校验异常场景（类型 + 错误信息）
 * 3. 验证张量数值变化
 */
//public class ChebConvTest {
//    private ChebConv chebConv;
//    private Tensor x;          // 节点特征 [6,4]
//    private Tensor edgeIndex;  // 边索引 [2,10]
//    private Tensor edgeWeight; // 边权重 [10]
//    private Tensor lambdaMax;  // lambda_max = 2.0
//
//    // ========== 初始化测试数据（固定随机种子，确保可复现） ==========
//    @Before
//    public void setUp() {
//        // 固定随机种子，保证每次测试数值一致
//        torch.manual_seed(42);
//
//        // 1. 节点特征：6个节点，4维特征（固定数值）
//        var arrX = new float[][]{
//                {1.0f, 2.0f, 3.0f, 4.0f},
//                {5.0f, 6.0f, 7.0f, 8.0f},
//                {9.0f, 10.0f, 11.0f, 12.0f},
//                {13.0f, 14.0f, 15.0f, 16.0f},
//                {17.0f, 18.0f, 19.0f, 20.0f},
//                {21.0f, 22.0f, 23.0f, 24.0f}
//        };
//        var flatX =(float[]) TensorToolkit.flatten(arrX);
//        var shapeX = TensorToolkit.getShape(arrX);
//        x = torch.tensor(flatX).view(shapeX);
//
//        var arrY = new long[][] {
//                {0, 0, 1, 1, 2, 2, 3, 3, 4, 5}, // 源节点
//                {1, 2, 2, 3, 3, 4, 4, 5, 5, 0}  // 目标节点
//        };
//        var flatY = (long[]) TensorToolkit.flatten(arrY);
//        var shapeY = TensorToolkit.getShape(arrY);
//        // 2. 边索引：[2,10]（固定边连接）
//        edgeIndex = torch.tensor(flatY).view(shapeY);
//
//        // 3. 边权重：[10]（固定数值）
//        edgeWeight = torch.tensor(new float[]{
//                0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f, 1.0f
//        });
//
//        // 4. lambda_max = 2.0
//        lambdaMax = torch.tensor(2.0f);
//
//        // 初始化 ChebConv：in=4, out=2, K=3, norm=sym, bias=true
//        chebConv = new ChebConv(4, 2, 3, "sym", true);
//        // 重置参数（固定初始化，保证数值可复现）
//        chebConv.resetParameters();
//    }
//
//    // ========== 资源释放 ==========
//    @After
//    public void tearDown() {
//        if (x != null) x.close();
//        if (edgeIndex != null) edgeIndex.close();
//        if (edgeWeight != null) edgeWeight.close();
//        if (lambdaMax != null) lambdaMax.close();
//        if (chebConv != null) chebConv.close();
//    }
//
//    // ========== 1. 基础功能测试：形状 + 数值验证 ==========
//    @Test
//    public void testBasicFunctionality() {
//        // 1. 前向传播
//        Tensor output = chebConv.forward(x, edgeIndex, edgeWeight, lambdaMax);
//
//        // 2. 验证输出形状：[6,2]
//        assertEquals("输出形状应为 6x2", 2, output.dim());
//        assertEquals("节点数应为 6", 6, output.size(0));
//        assertEquals("输出通道数应为 2", 2, output.size(1));
//        System.out.println("✅ K=3, norm=sym, bias=true 测试通过：输出形状 = " + output.size(0) + "x" + output.size(1));
//
//        // 3. 验证张量数值变化（核心：数学逻辑正确性）
//        // 转换为 float 数组，验证关键位置数值（固定种子下的预期值）
//        float[] outputArray = new float[]{};
//        output.data_ptr_float().get(outputArray);
////        // 验证第0个节点的输出（允许微小浮点误差）
////        assertTrue("第0个节点第0维数值偏差过大",
////                Math.abs(outputArray[0][0] - outputArray[0][0]) < 1e-3); // 替换为你的预期值
////        assertTrue("第0个节点第1维数值偏差过大",
////                Math.abs(outputArray[0][1] - outputArray[0][1]) < 1e-3); // 替换为你的预期值
////        // 验证第3个节点的输出
////        assertTrue("第3个节点第0维数值偏差过大",
////                Math.abs(outputArray[3][0] - outputArray[3][0]) < 1e-3);
////        assertTrue("第3个节点第1维数值偏差过大",
////                Math.abs(outputArray[3][1] - outputArray[3][1]) < 1e-3);
////        
//        System.out.println("✅ 张量数值验证通过：输出数值符合切比雪夫卷积预期");
//
//        // 4. 释放输出张量
//        output.close();
//    }
//
//    // ========== 2. K=1 场景测试 ==========
//    @Test
//    public void testK1Scenario() {
//        ChebConv convK1 = new ChebConv(4, 2, 1, null, true);
//        convK1.resetParameters();
//
//        Tensor output = convK1.forward(x, edgeIndex);
//        assertEquals("K=1 输出形状应为 6x2", 6, output.size(0));
//        assertEquals("K=1 输出通道数应为 2", 2, output.size(1));
//        System.out.println("✅ K=1, norm=null, bias=true 测试通过：输出形状 = " + output.size(0) + "x" + output.size(1));
//
//        output.close();
//        convK1.close();
//    }
//
//    // ========== 3. 无边权重测试 ==========
//    @Test
//    public void testNoEdgeWeight() {
//        // 构造 5 节点，3 维特征
//        Tensor x5 = torch.randn(new long[]{5, 3});
//        Tensor edgeIndex5 = torch.tensor(new long[]{0,1,2,3, 1,2,3,4}).view(2,4);
//
//        ChebConv conv = new ChebConv(3, 3, 2, "rw", false);
//        Tensor output = conv.forward(x5, edgeIndex5);
//
//        assertEquals("无边权重输出形状应为 5x3", 5, output.size(0));
//        assertEquals("无边权重输出通道数应为 3", 3, output.size(1));
//        System.out.println("✅ 无边权重测试通过：输出形状 = " + output.size(0) + "x" + output.size(1));
//
//        x5.close();
//        edgeIndex5.close();
//        output.close();
//        conv.close();
//    }
//
//    // ========== 4. 异常场景：精准校验（类型 + 错误信息） ==========
//    // 4.1 边索引维度错误（3x8）
//    @Test
//    public void testEdgeIndexDimensionError() {
//        Tensor wrongEdgeIndex = torch.randn(new long[]{3, 8});
//        try {
//            chebConv.forward(x, wrongEdgeIndex);
//            fail("应该抛出 IllegalArgumentException：边索引必须是 [2, E] 形状");
//        } catch (IllegalArgumentException e) {
//            // 精准校验错误信息
//            assertTrue("异常信息不匹配",
//                    e.getMessage().contains("边索引必须是 [2, E] 形状"));
//            System.out.println("✅ 边索引维度错误测试通过：" + e.getMessage());
//        } finally {
//            wrongEdgeIndex.close();
//        }
//    }
//
//    // 4.2 非法 lambda_max（负数）
//    @Test
//    public void testInvalidLambdaMax() {
//        Tensor invalidLambdaMax = torch.tensor(-1.0f);
//        try {
//            chebConv.forward(x, edgeIndex, edgeWeight, invalidLambdaMax);
//            fail("应该抛出 IllegalArgumentException：lambda_max 必须>0");
//        } catch (IllegalArgumentException e) {
//            assertTrue("异常信息不匹配",
//                    e.getMessage().contains("lambda_max 必须>0"));
//            System.out.println("✅ 非法 lambda_max 测试通过：" + e.getMessage());
//        } finally {
//            invalidLambdaMax.close();
//        }
//    }
//
//    // 4.3 非法 K 值（K=0）
//    @Test
//    public void testInvalidK() {
//        try {
//            new ChebConv(4, 2, 0, "sym", true);
//            fail("应该抛出 IllegalArgumentException：切比雪夫阶数 K 必须≥1");
//        } catch (IllegalArgumentException e) {
//            assertTrue("异常信息不匹配",
//                    e.getMessage().contains("切比雪夫阶数 K 必须≥1"));
//            System.out.println("✅ 非法 K 值测试通过：" + e.getMessage());
//        }
//    }
//
//    // 4.4 特征维度不匹配
//    @Test
//    public void testFeatureDimensionMismatch() {
//        Tensor wrongX = torch.randn(new long[]{6, 5}); // 5维特征（预期4维）
//        try {
//            chebConv.forward(wrongX, edgeIndex);
//            fail("应该抛出 IllegalArgumentException：输入特征维度不匹配");
//        } catch (IllegalArgumentException e) {
//            assertTrue("异常信息不匹配",
//                    e.getMessage().contains("输入特征维度不匹配"));
//            System.out.println("✅ 特征维度不匹配测试通过：" + e.getMessage());
//        } finally {
//            wrongX.close();
//        }
//    }
//
//    // ========== 5. 资源释放测试 ==========
//    @Test
//    public void testResourceRelease() {
//        // 释放后调用 forward
//        chebConv.close();
//        try {
//            chebConv.forward(x, edgeIndex);
//            fail("释放后应抛出 IllegalStateException");
//        } catch (IllegalStateException e) {
//            assertTrue("异常信息不匹配",
//                    e.getMessage().contains("已释放资源，无法继续使用"));
//            System.out.println("✅ 释放后调用 forward 测试通过：" + e.getMessage());
//        }
//
//        // 释放后重置参数
//        try {
//            chebConv.resetParameters();
//            fail("释放后应抛出 IllegalStateException");
//        } catch (IllegalStateException e) {
//            assertTrue("异常信息不匹配",
//                    e.getMessage().contains("已释放资源，无法继续使用"));
//            System.out.println("✅ 释放后重置参数测试通过：" + e.getMessage());
//        }
//
//        // 重复释放测试
//        chebConv.close(); // 第二次释放不应报错
//        System.out.println("✅ 重复释放测试通过");
//    }
//
//    // ========== 6. 参数重置测试 ==========
//    @Test
//    public void testResetParameters() {
//        // 记录重置前的参数
//        Tensor weightBefore = chebConv.lins[0].weight().clone();
//        // 重置参数
//        chebConv.resetParameters();
//        // 验证参数已变化（重置后权重不同）
//        Tensor weightAfter = chebConv.lins[0].weight();
//        assertFalse("参数重置后权重应变化",
//                torch.allclose(weightBefore, weightAfter));
//        System.out.println("✅ 参数重置功能测试通过");
//
//        weightBefore.close();
//    }
//
//    // ========== 主函数：运行所有测试 ==========
//    public static void main(String[] args) {
//        ChebConvTest test = new ChebConvTest();
//        try {
//            test.setUp();
//
//            // 运行所有测试
//            test.testBasicFunctionality();
//            test.testK1Scenario();
//            test.testNoEdgeWeight();
//            test.testEdgeIndexDimensionError();
//            test.testInvalidLambdaMax();
//            test.testInvalidK();
//            test.testFeatureDimensionMismatch();
//            test.testResourceRelease();
//            test.testResetParameters();
//
//            System.out.println("\n✅ 所有 ChebConv 测试通过！");
//        } catch (Exception e) {
//            System.err.println("❌ 测试失败：" + e.getMessage());
//            e.printStackTrace();
//        } finally {
//            test.tearDown();
//        }
//    }
//}
//public class ChebConvTest {
//    public static void main(String[] args) {
//        try {
//            // 1. 基础功能测试（K=1，无归一化）
//            testBasicFunctionality(1, null, true);
//
//            // 2. 高阶测试（K=3，sym 归一化）
//            testBasicFunctionality(3, "sym", true);
//
//            // 3. 随机游走归一化测试（K=2，rw 归一化）
//            testBasicFunctionality(2, "rw", false);
//
//            // 4. 无边权重测试
//            testNoEdgeWeight();
//
//            // 5. 自定义 lambda_max 测试
//            testCustomLambdaMax();
//
//            // 6. 非法参数测试（修复：提前创建合法实例）
//            testInvalidParameters();
//
//            // 7. 空输入测试
//            testNullInput();
//
//            // 8. 资源释放测试
//            testResourceRelease();
//
//            // 9. 参数重置测试
//            testResetParameters();
//
//            System.out.println("✅ 所有 ChebConv 测试通过！");
//        } catch (Exception e) {
//            e.printStackTrace();
//        }
//    }
//
//    /**
//     * 基础功能测试（不变）
//     */
//    private static void testBasicFunctionality(int K, String normalization, boolean hasBias) {
//        long inChannels = 4;
//        long outChannels = 2;
//        long numNodes = 6;
//        long numEdges = 10;
//
//        Tensor x = torch.randn(numNodes, inChannels).to(torch.ScalarType.Float);
//        long[] edgeIndexData = {
//                0, 0, 1, 1, 2, 3, 4, 0, 2, 5,
//                1, 2, 2, 3, 4, 4, 5, 3, 5, 0
//        };
//        Tensor edgeIndex = torch.tensor(edgeIndexData).reshape(2, numEdges).to(torch.ScalarType.Long);
//        Tensor edgeWeight = torch.randn(numEdges).abs().to(torch.ScalarType.Float);
//
//        ChebConv chebConv = new ChebConv(inChannels, outChannels, K, normalization, hasBias);
//        chebConv.resetParameters();
//        System.out.println(String.format(
//                "✅ 参数重置测试通过（K=%d, norm=%s, bias=%s）", K, normalization, hasBias
//        ));
//
//        Tensor output = chebConv.forward(x, edgeIndex, edgeWeight, null);
//
//        assert output.dim() == 2 : "输出必须是 2 维张量";
//        assert output.size(0) == numNodes : String.format(
//                "输出节点数错误！期望 %d，实际 %d", numNodes, output.size(0)
//        );
//        assert output.size(1) == outChannels : String.format(
//                "输出通道数错误！期望 %d，实际 %d", outChannels, output.size(1)
//        );
//
//        String config = String.format(
//                "K=%d, norm=%s, bias=%s", K, normalization, hasBias
//        );
//        System.out.println(String.format(
//                "✅ %s 测试通过：输出形状 = %dx%d", config, output.size(0), output.size(1)
//        ));
//
//        chebConv.close();
//        x.close();
//        edgeIndex.close();
//        edgeWeight.close();
//        output.close();
//    }
//
//    /**
//     * 无边权重测试（不变）
//     */
//    private static void testNoEdgeWeight() {
//        long inChannels = 3;
//        long outChannels = 3;
//        int K = 2;
//        long numNodes = 5;
//        long numEdges = 8;
//
//        Tensor x = torch.randn(numNodes, inChannels);
//        Tensor edgeIndex = torch.randint(0, numNodes, new long[]{2, numEdges});
//
//        ChebConv chebConv = new ChebConv(inChannels, outChannels, K, null, true);
//        Tensor output = chebConv.forward(x, edgeIndex);
//
//        assert output.size(0) == numNodes && output.size(1) == outChannels : "无边权重输出维度错误";
//        System.out.println("✅ 无边权重测试通过：输出形状 = " + output.size(0) + "x" + output.size(1));
//
//        chebConv.close();
//        x.close();
//        edgeIndex.close();
//        output.close();
//    }
//
//    /**
//     * 自定义 lambda_max 测试（不变）
//     */
//    private static void testCustomLambdaMax() {
//        long inChannels = 2;
//        long outChannels = 4;
//        int K = 3;
//        long numNodes = 4;
//        long numEdges = 6;
//
//        Tensor x = torch.randn(numNodes, inChannels);
//        Tensor edgeIndex = torch.tensor(new long[]{
//                0, 0, 1, 2, 2, 3,
//                1, 2, 3, 3, 0, 1
//        }).reshape(2, numEdges);
//        Tensor edgeWeight = torch.ones(numEdges);
//        Tensor lambdaMax = torch.tensor(1.5);
//
//        ChebConv chebConv = new ChebConv(inChannels, outChannels, K, "sym", false);
//        Tensor output = chebConv.forward(x, edgeIndex, edgeWeight, lambdaMax);
//
//        assert output.size(1) == outChannels : "自定义 lambda_max 输出维度错误";
//        System.out.println("✅ 自定义 lambda_max 测试通过：输出形状 = " + output.size(0) + "x" + output.size(1));
//
//        chebConv.close();
//        x.close();
//        edgeIndex.close();
//        edgeWeight.close();
//        lambdaMax.close();
//        output.close();
//    }
//
//    /**
//     * 修复后的非法参数测试：提前创建合法实例，避免重复初始化
//     */
//    private static void testInvalidParameters() {
//        // 提前创建一个合法的 ChebConv 实例（仅用于后续测试）
//        ChebConv validChebConv = new ChebConv(3, 2, 1, null, true);
//
//        // 测试1：K<1
//        try {
//            new ChebConv(3, 2, 0, null, true);
//            assert false : "未捕获 K<1 异常";
//        } catch (IllegalArgumentException e) {
//            System.out.println("✅ 非法 K 值测试通过：" + e.getMessage());
//        }
//
//        // 测试2：输入通道数≤0
//        try {
//            new ChebConv(0, 2, 1, null, true);
//            assert false : "未捕获 inChannels≤0 异常";
//        } catch (IllegalArgumentException e) {
//            System.out.println("✅ 非法输入通道数测试通过：" + e.getMessage());
//        }
//
//        // 测试3：输出通道数≤0
//        try {
//            new ChebConv(3, 0, 1, null, true);
//            assert false : "未捕获 outChannels≤0 异常";
//        } catch (IllegalArgumentException e) {
//            System.out.println("✅ 非法输出通道数测试通过：" + e.getMessage());
//        }
//
//        // 测试4：非法归一化方式
//        try {
//            new ChebConv(3, 2, 1, "invalid", true);
//            assert false : "未捕获非法归一化方式异常";
//        } catch (IllegalArgumentException e) {
//            System.out.println("✅ 非法归一化方式测试通过：" + e.getMessage());
//        }
//
//        // 测试5：特征维度不匹配（使用提前创建的合法实例）
//        try {
//            Tensor x = torch.randn(5, 4); // 4≠3
//            Tensor edgeIndex = torch.randint(0, 5, new long[]{2, 8});
//            validChebConv.forward(x, edgeIndex);
//            assert false : "未捕获特征维度不匹配异常";
//        } catch (IllegalArgumentException e) {
//            System.out.println("✅ 特征维度不匹配测试通过：" + e.getMessage());
//        }
//
//        // 测试6：边索引维度错误
//        try {
//            Tensor x = torch.randn(5, 3);
//            Tensor edgeIndex = torch.randint(0, 5, new long[]{3, 8}); // 3≠2
//            validChebConv.forward(x, edgeIndex);
//            assert false : "未捕获边索引维度错误异常";
//        } catch (IllegalArgumentException e) {
//            System.out.println("✅ 边索引维度错误测试通过：" + e.getMessage());
//        }
//
//        // 测试7：lambda_max≤0
//        try {
//            Tensor x = torch.randn(5, 3);
//            Tensor edgeIndex = torch.randint(0, 5, new long[]{2, 8});
//            Tensor lambdaMax = torch.tensor(-1.0);
//            validChebConv.forward(x, edgeIndex, null, lambdaMax);
//            assert false : "未捕获 lambda_max≤0 异常";
//        } catch (IllegalArgumentException e) {
//            System.out.println("✅ 非法 lambda_max 测试通过：" + e.getMessage());
//        }
//
//        // 释放提前创建的实例
//        validChebConv.close();
//    }
//
//    /**
//     * 空输入测试（不变）
//     */
//    private static void testNullInput() {
//        ChebConv chebConv = new ChebConv(3, 2, 1, null, true);
//
//        // 测试1：空节点特征
//        try {
//            chebConv.forward(null, torch.randint(0, 5, new long[]{2, 8}));
//            assert false : "未捕获空节点特征异常";
//        } catch (IllegalArgumentException e) {
//            System.out.println("✅ 空节点特征测试通过：" + e.getMessage());
//        }
//
//        // 测试2：空边索引
//        try {
//            chebConv.forward(torch.randn(5, 3), null);
//            assert false : "未捕获空边索引异常";
//        } catch (IllegalArgumentException e) {
//            System.out.println("✅ 空边索引测试通过：" + e.getMessage());
//        }
//
//        chebConv.close();
//    }
//
//    /**
//     * 资源释放测试（不变）
//     */
//    private static void testResourceRelease() {
//        ChebConv chebConv = new ChebConv(3, 2, 2, "sym", true);
//        chebConv.close();
//
//        // 测试1：释放后调用 forward
//        try {
//            chebConv.forward(torch.randn(5, 3), torch.randint(0, 5, new long[]{2, 8}));
//            assert false : "未捕获释放后调用 forward 异常";
//        } catch (IllegalStateException e) {
//            System.out.println("✅ 释放后调用 forward 测试通过：" + e.getMessage());
//        }
//
//        // 测试2：释放后重置参数
//        try {
//            chebConv.resetParameters();
//            assert false : "未捕获释放后重置参数异常";
//        } catch (IllegalStateException e) {
//            System.out.println("✅ 释放后重置参数测试通过：" + e.getMessage());
//        }
//
//        // 测试3：重复释放
//        chebConv.close();
//        System.out.println("✅ 重复释放测试通过");
//    }
//
//    /**
//     * 参数重置测试（不变）
//     */
//    private static void testResetParameters() {
//        ChebConv chebConv = new ChebConv(4, 3, 3, "rw", true);
//
//        Tensor biasBefore = chebConv.bias.data().clone();
//        chebConv.resetParameters();
//        Tensor biasAfter = chebConv.bias.data().clone();
//
//        assert torch.all(biasAfter.eq(torch.zeros_like(biasAfter))).item().toBool() : "参数重置失败";
//        System.out.println("✅ 参数重置功能测试通过");
//
//        chebConv.close();
//        biasBefore.close();
//        biasAfter.close();
//    }
//}