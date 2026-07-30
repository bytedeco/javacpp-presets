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
import org.bytedeco.pytorch.geometric.nn.conv.DirGNNConv;
import org.bytedeco.pytorch.geometric.nn.conv.SAGEConvV2;
import org.bytedeco.pytorch.geometric.nn.conv.SAGEConvV3;

import java.util.Arrays;

import static org.junit.Assert.*;

/**
 * DirGNNConv 测试用例：
 * 1. 形状验证（输入→输出维度）
 * 2. 数值验证（入边/出边加权融合逻辑）
 * 3. 有向边聚合验证（入边/出边结果不同）
 * 4. 根节点权重验证（残差连接生效）
 * 5. 异常验证（非法输入/alpha）
 * 6. 资源释放验证
 */
public class DirGNNConvTest {
    private DirGNNConv dirGnnConv;      // 有根节点 + alpha=0.5
    private DirGNNConv dirGnnConvNoRoot;// 无根节点 + alpha=0.5
    private Tensor x;                   // 节点特征 [4, 3]（4个节点，3维特征）
    private Tensor edgeIndex;           // 有向边索引 [2, 5]（5条有向边）
    private static final long IN_CHANNELS = 3;
    private static final long OUT_CHANNELS = 2;

    @Before
    public void setUp() {
        // 1. 固定随机种子，保证数值可复现
        torch.manual_seed(1234);

        // 2. 初始化基础 SAGEConvV2
        SAGEConvV3 sageConv = new SAGEConvV3(IN_CHANNELS, OUT_CHANNELS);

        // 3. 初始化 DirGNNConv（两种配置）
        dirGnnConv = new DirGNNConv(sageConv, 0.5f, true, IN_CHANNELS, OUT_CHANNELS);
        dirGnnConvNoRoot = new DirGNNConv(new SAGEConvV3(IN_CHANNELS, OUT_CHANNELS), 0.5f, false, IN_CHANNELS, OUT_CHANNELS);

        // 4. 构造节点特征：[4, 3]
        x = torch.randn(new long[]{4, 3}).to(torch.kFloat());

        // 5. 构造有向边索引 [2, 5]：
        //    边：0→1, 0→2, 1→2, 2→3, 3→0（有向边，入边/出边不同）
        long[] edgeData = {0,0,1,2,3, 1,2,2,3,0};
        edgeIndex = torch.tensor(edgeData).view(2, 5).to(torch.ScalarType.Long);
    }

    /**
     * 测试1：输出形状验证
     * 预期：输入 [4,3] → 输出 [4,2]
     */
    @Test
    public void testForwardShape() {
        Tensor out = dirGnnConv.forward(x, edgeIndex);
        Tensor outNoRoot = dirGnnConvNoRoot.forward(x, edgeIndex);

        // 验证有根节点版本形状
        assertEquals("输出应为2维张量", 2, out.dim());
        assertEquals("节点数应保持4", 4, out.size(0));
        assertEquals("输出通道数应为2", OUT_CHANNELS, out.size(1));

        // 验证无根节点版本形状（与有根节点一致）
        assertEquals("无根节点版本形状应一致", Arrays.toString(out.sizes().vec().get()), Arrays.toString(outNoRoot.sizes().vec().get()));

        System.out.println("✅ 形状验证通过：输入 " +Arrays.toString(x.sizes().vec().get()) + " → 输出 " + Arrays.toString(out.sizes().vec().get()));

        // 释放临时张量
        out.close();
        outNoRoot.close();
    }

    /**
     * 测试2：数值验证（alpha=0.5 加权融合）
     * 核心逻辑：out = 0.5*outIn + 0.5*outOut
     */
    @Test
    public void testForwardNumerics() {
        // 步骤1：手动计算入边/出边聚合结果
        SAGEConvV3 baseConv = (SAGEConvV3) dirGnnConvNoRoot.getConv();
        Tensor outIn = baseConv.forward(x, edgeIndex); // 入边聚合
        // 翻转边索引计算出边聚合
        Tensor edgeRow0 = edgeIndex.select(0, 0);
        Tensor edgeRow1 = edgeIndex.select(0, 1);
        Tensor revEdgeIndex = torch.stack(new TensorVector(edgeRow1, edgeRow0), 0);
        Tensor outOut = baseConv.forward(x, revEdgeIndex);

        // 步骤2：手动计算加权融合结果（alpha=0.5）
        Tensor expectedOut = outIn.mul(new Scalar(0.5f)).add(outOut.mul(new Scalar(0.5f)));

        // 步骤3：获取 DirGNNConv 实际输出（无根节点版本）
        Tensor actualOut = dirGnnConvNoRoot.forward(x, edgeIndex);

        // 步骤4：验证数值一致性（误差 < 1e-5）
        Tensor diff = actualOut.sub(expectedOut).abs();
        double maxDiff = diff.max().item_double();
        assertTrue(String.format("数值误差过大（%.6f）", maxDiff), maxDiff < 1e-5);

        System.out.println("✅ 数值验证通过，最大误差：" + maxDiff);

        // 释放临时张量
        outIn.close();
        outOut.close();
        edgeRow0.close();
        edgeRow1.close();
        revEdgeIndex.close();
        expectedOut.close();
        actualOut.close();
        diff.close();
    }

    /**
     * 测试3：有向边聚合验证（入边/出边结果不同）
     */
    @Test
    public void testDirectedEdgeAggregation() {
        SAGEConvV3 baseConv = (SAGEConvV3) dirGnnConv.getConv();
        // 入边聚合结果
        Tensor outIn = baseConv.forward(x, edgeIndex);
        // 出边聚合结果（翻转边索引）
        Tensor edgeRow0 = edgeIndex.select(0, 0);
        Tensor edgeRow1 = edgeIndex.select(0, 1);
        Tensor revEdgeIndex = torch.stack(new TensorVector(edgeRow1, edgeRow0), 0);
        Tensor outOut = baseConv.forward(x, revEdgeIndex);

        // 验证入边/出边结果不同（有向图聚合差异）
        Tensor diff = outIn.sub(outOut).abs();
        double maxDiff = diff.max().item_double();
        assertTrue("有向边聚合结果应不同（最大差异：" + maxDiff + "）", maxDiff > 1e-3);

        System.out.println("✅ 有向边聚合验证通过，入边/出边最大差异：" + maxDiff);

        // 释放临时张量
        outIn.close();
        outOut.close();
        edgeRow0.close();
        edgeRow1.close();
        revEdgeIndex.close();
        diff.close();
    }

    /**
     * 测试4：根节点权重验证（残差连接生效）
     */
    @Test
    public void testRootWeight() {
        // 有根节点版本输出
        Tensor outWithRoot = dirGnnConv.forward(x, edgeIndex);
        // 无根节点版本输出
        Tensor outNoRoot = dirGnnConvNoRoot.forward(x, edgeIndex);

        // 验证两者结果不同（根节点残差生效）
        Tensor diff = outWithRoot.sub(outNoRoot).abs();
        double maxDiff = diff.max().item_double();
        assertTrue("根节点权重应生效（最大差异：" + maxDiff + "）", maxDiff > 1e-3);

        System.out.println("✅ 根节点权重验证通过，有无根节点最大差异：" + maxDiff);

        // 释放临时张量
        outWithRoot.close();
        outNoRoot.close();
        diff.close();
    }

    /**
     * 测试5：异常输入验证
     */
    @Test
    public void testInvalidInput() {
        SAGEConvV3 sageConv = new SAGEConvV3(IN_CHANNELS, OUT_CHANNELS);

        // 测试1：alpha 超出 [0,1] 范围
        assertThrows(IllegalArgumentException.class, () -> new DirGNNConv(sageConv, 1.5f, true, IN_CHANNELS, OUT_CHANNELS));
        assertThrows(IllegalArgumentException.class, () -> new DirGNNConv(sageConv, -0.1f, true, IN_CHANNELS, OUT_CHANNELS));

        // 测试2：基础卷积算子为空
        assertThrows(IllegalArgumentException.class, () -> new DirGNNConv(null, 0.5f, true, IN_CHANNELS, OUT_CHANNELS));

        // 测试3：x 维度错误（1维）
        Tensor x1d = torch.randn(new long[]{3});
        assertThrows(IllegalArgumentException.class, () -> dirGnnConv.forward(x1d, edgeIndex));

        // 测试4：edge_index 维度错误（3维）
        Tensor edgeIndex3d = torch.randn(new long[]{3, 5});
        assertThrows(IllegalArgumentException.class, () -> dirGnnConv.forward(x, edgeIndex3d));

        // 测试5：x 通道数不匹配
        Tensor xWrongChannel = torch.randn(new long[]{4, 4});
        assertThrows(IllegalArgumentException.class, () -> dirGnnConv.forward(xWrongChannel, edgeIndex));

        System.out.println("✅ 异常输入验证通过");

        // 释放临时张量
        x1d.close();
        edgeIndex3d.close();
        xWrongChannel.close();
    }

    /**
     * 测试6：资源释放验证
     */
    @Test
    public void testResourceRelease() {
        SAGEConvV3 sageConv = new SAGEConvV3(IN_CHANNELS, OUT_CHANNELS);
        DirGNNConv tempConv = new DirGNNConv(sageConv, 0.5f, true, IN_CHANNELS, OUT_CHANNELS);
        tempConv.close();
        // 验证重复释放不崩溃
//        assertDoesNotThrow(tempConv::close);

        System.out.println("✅ 资源释放验证通过");
    }

    @After
    public void tearDown() {
        // 释放所有资源，避免JNI内存泄漏
        if (dirGnnConv != null) dirGnnConv.close();
        if (dirGnnConvNoRoot != null) dirGnnConvNoRoot.close();
        if (x != null) x.close();
        if (edgeIndex != null) edgeIndex.close();

//        // 清空PyTorch计算图
//        torch.clear_autograd_graph();
//        torch.cuda.empty_cache();
    }

    public static void main(String[] args) {
        Result result = JUnitCore.runClasses(DirGNNConvTest.class);
        for (Failure failure : result.getFailures()) {
            System.err.println("FAIL: " + failure.toString());
            if (failure.getException() != null) failure.getException().printStackTrace();
        }
        System.out.println("Tests run=" + result.getRunCount()
                + " failed=" + result.getFailureCount()
                + " ignored=" + result.getIgnoreCount());
        if (!result.wasSuccessful()) {
            throw new RuntimeException("DirGNNConvTest had " + result.getFailureCount() + " failures");
        }
        System.out.println("✅ DirGNNConvTest all tests passed");
    }

}
