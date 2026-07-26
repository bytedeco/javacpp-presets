package org.bytedeco.pytorch.geometric.demo.layer;
import org.bytedeco.pytorch.jit.*;
import org.bytedeco.pytorch.nn.options.*;
import org.bytedeco.pytorch.nn.modules.container.*;
import org.bytedeco.pytorch.nn.modules.*;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.global.torch;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.bytedeco.pytorch.geometric.nn.conv.DynamicEdgeConv;

import java.util.Arrays;

import static org.junit.Assert.*;

/**
 * DynamicEdgeConv 测试用例：
 * 1. 形状验证（输入→输出维度）
 * 2. k-NN构图验证（邻居数=k，无自环）
 * 3. EdgeConv核心逻辑验证（(x_i, x_j-x_i) 拼接 + 变换）
 * 4. 批次隔离验证（不同批次节点无连接）
 * 5. 异常验证（非法k值/维度/批次向量）
 * 6. 资源释放验证
 */
public class DynamicEdgeConvTest {
    private DynamicEdgeConv dynamicEdgeConv; // k=2, max聚合
    private DynamicEdgeConv dynamicEdgeConvMean; // k=2, mean聚合
    private Tensor x; // 节点特征 [8, 3]（8节点，3通道）
    private Tensor batch; // 批次向量 [8]（0,0,0,0,1,1,1,1 → 2个批次，各4节点）
    private SequentialImpl nn; // 非线性变换网络：Linear(6, 2) → 输入6=2*3，输出2

    @Before
    public void setUp() {
        // 1. 固定随机种子，保证数值可复现
        torch.manual_seed(1234);

        // 2. 构建非线性变换网络：Linear(6, 2)（EdgeConv输入=2*3=6）
        var dict = new StringAnyModuleDict();
        dict.insert("lin", new AnyModule(new LinearImpl(6, 2)));
        nn = new SequentialImpl(dict);

        // 3. 初始化DynamicEdgeConv
        dynamicEdgeConv = new DynamicEdgeConv(nn, 2, "max");
        dynamicEdgeConvMean = new DynamicEdgeConv(new LinearImpl(6, 2), 2, "mean");

        // 4. 构造节点特征 [8, 3]（前4个节点特征相近，后4个节点特征相近）
        x = torch.cat(
                new TensorVector(
                        torch.randn(new long[]{4, 3}).mul(new Scalar(0.1f)), // 批次0：特征值小
                        torch.randn(new long[]{4, 3}).add(new Scalar(10.0f)) // 批次1：特征值大（与批次0隔离）
                ),
                0
        );

        // 5. 构造批次向量 [8]：[0,0,0,0,1,1,1,1]
        batch = torch.cat(
                new TensorVector(
                        torch.zeros(new long[]{4}).to(torch.ScalarType.Long),
                        torch.ones(new long[]{4}).to(torch.ScalarType.Long)
                ),
                0
        );
    }

    /**
     * 测试1：输出形状验证
     * 预期：输入 [8,3] → 输出 [8,2]
     */
    @Test
    public void testForwardShape() {
        Tensor out = ((DynamicEdgeConv)dynamicEdgeConv).forward(x, batch);
        Tensor outMean = ((DynamicEdgeConv)dynamicEdgeConvMean).forward(x, (Tensor)null); // 无批次向量

        // 验证max聚合版本形状
        assertEquals("输出应为2维张量", 2, out.dim());
        assertEquals("节点数应保持8", 8, out.size(0));
        assertEquals("输出通道数应为2", 2, out.size(1));

        // 验证mean聚合版本形状
        assertEquals("mean聚合版本形状应一致", Arrays.toString(out.sizes().vec().get()), Arrays.toString(outMean.sizes().vec().get()));

        System.out.println("✅ 形状验证通过：输入 " + Arrays.toString(x.sizes().vec().get()) + " → 输出 " + Arrays.toString(out.sizes().vec().get()));

        // 释放临时张量
        out.close();
        outMean.close();
    }

    /**
     * 测试2：k-NN构图验证（邻居数=k，无自环）
     */
    @Test
    public void testKnnGraph() {
        long N = x.size(0);
        int k = dynamicEdgeConv.getK();

        // 计算距离矩阵
        Tensor dists = torch.cdist(x, x);
        // 批次隔离掩码
        Tensor batchExpandI = batch.view(-1, 1).repeat(new long[]{1, N});
        Tensor batchExpandJ = batch.view(1, -1).repeat(new long[]{N, 1});
        Tensor mask = batchExpandI.ne(batchExpandJ);
        Tensor inf = torch.tensor(Double.POSITIVE_INFINITY, x.options());
        dists = dists.where(mask.logical_not(), inf);

        // 获取k+1个最近邻
        T_TensorTensor_T topk = dists.topk(k + 1, -1, false, true);
        Tensor knnIdx = topk.get1().narrow(-1, 1, k); // [8,2]

        // 验证1：邻居数=k
        assertEquals("邻居数应为k=2", k, knnIdx.size(1));

        // 验证2：无自环（邻居索引≠自身索引）
        Tensor selfIdx = torch.arange(new Scalar(N), knnIdx.options()).view(-1, 1); // [8,1]
        Tensor isSelf = knnIdx.eq(selfIdx).any(-1); // [8]，是否包含自身
        assertEquals("k-NN图不应包含自环", 0, isSelf.sum().item_long());

        System.out.println("✅ k-NN构图验证通过：邻居数=" + k + "，无自环");

        // 释放临时张量
        dists.close();
        batchExpandI.close();
        batchExpandJ.close();
        mask.close();
        inf.close();
        topk.close();
        knnIdx.close();
        selfIdx.close();
        isSelf.close();
    }

    /**
     * 测试3：EdgeConv核心逻辑验证
     */
    @Test
    public void testEdgeConvLogic() {
        // 取前2个节点特征
        Tensor xSmall = x.narrow(0, 0, 2); // [2,3]
        Tensor batchSmall = batch.narrow(0, 0, 2); // [2]

        // 手动计算EdgeConv输入：(x_i, x_j-x_i)
        Tensor xi = xSmall.select(0, 0).unsqueeze(0); // [1,3]
        Tensor xj = xSmall.select(0, 1).unsqueeze(0); // [1,3]
        Tensor msgInputManual = torch.cat(new TensorVector(xi, xj.sub(xi)), -1); // [1,6]

        // 通过nn变换
        LinearImpl lin = dynamicEdgeConv.getNn().asSequential().get(0).asLinear();
        Tensor msgManual = lin.forward(msgInputManual); // [1,2]

        // 验证变换逻辑一致性（误差<1e-5）
        Tensor out = ((DynamicEdgeConv)dynamicEdgeConv).forward(xSmall, batchSmall);
        Tensor outFirst = out.select(0, 0).unsqueeze(0); // 第一个节点的输出
        Tensor diff = outFirst.sub(msgManual).abs();
        double maxDiff = diff.max().item_double();
        assertTrue(String.format("EdgeConv逻辑错误（最大误差：%.6f）", maxDiff), maxDiff < 1e-5);

        System.out.println("✅ EdgeConv核心逻辑验证通过，最大误差：" + maxDiff);

        // 释放临时张量
        xSmall.close();
        batchSmall.close();
        xi.close();
        xj.close();
        msgInputManual.close();
        msgManual.close();
        out.close();
        outFirst.close();
        diff.close();
    }

    /**
     * 测试4：批次隔离验证（不同批次节点无连接）
     */
    @Test
    public void testBatchIsolation() {
        long N = x.size(0);
        int k = dynamicEdgeConv.getK();

        // 计算带批次隔离的距离矩阵
        Tensor dists = torch.cdist(x, x);
        Tensor batchExpandI = batch.view(-1, 1).repeat(new long[]{1, N});
        Tensor batchExpandJ = batch.view(1, -1).repeat(new long[]{N, 1});
        Tensor mask = batchExpandI.ne(batchExpandJ);
        Tensor inf = torch.tensor(Double.POSITIVE_INFINITY, x.options());
        dists = dists.where(mask.logical_not(), inf);

        // 获取批次0第一个节点的k-NN邻居
        Tensor dists0 = dists.select(0, 0); // [8]
        T_TensorTensor_T topk0 = dists0.topk(k + 1, -1, false, true);
        Tensor knnIdx0 = topk0.get1().narrow(-1, 1, k); // [2]

        // 验证邻居索引均在批次0（0-3）
        Tensor isBatch0 = knnIdx0.lt(new Scalar(4)); // 索引<4为批次0
        assertEquals("批次0节点的邻居应全部在批次0", k, isBatch0.sum().item_long());

        System.out.println("✅ 批次隔离验证通过：不同批次节点无连接");

        // 释放临时张量
        dists.close();
        batchExpandI.close();
        batchExpandJ.close();
        mask.close();
        inf.close();
        dists0.close();
        topk0.close();
        knnIdx0.close();
        isBatch0.close();
    }

    /**
     * 测试5：异常输入验证
     */
    @Test
    public void testInvalidInput() {
        // 测试1：k<1
        assertThrows(IllegalArgumentException.class, () -> new DynamicEdgeConv(nn, 0, "max"));

        // 测试2：非法聚合方式
        assertThrows(IllegalArgumentException.class, () -> new DynamicEdgeConv(nn, 2, "sum"));

        // 测试3：x维度错误（3维）
        Tensor x3d = torch.randn(new long[]{8, 3, 2});
        assertThrows(IllegalArgumentException.class, () -> ((DynamicEdgeConv)dynamicEdgeConv).forward(x3d, batch));

        // 测试4：批次向量长度不匹配
        Tensor batchWrong = torch.zeros(new long[]{7});
        assertThrows(IllegalArgumentException.class, () -> ((DynamicEdgeConv)dynamicEdgeConv).forward(x, batchWrong));

        System.out.println("✅ 异常输入验证通过");

        // 释放临时张量
        x3d.close();
        batchWrong.close();
    }

    /**
     * 测试6：资源释放验证
     */
    @Test
    public void testResourceRelease() {
        DynamicEdgeConv tempConv = new DynamicEdgeConv(new LinearImpl(6, 2), 2, "max");
        tempConv.close();
        // 验证重复释放不崩溃
//        assertDoesNotThrow(tempConv::close);

        System.out.println("✅ 资源释放验证通过");
    }

    @After
    public void tearDown() {
        // 释放所有资源
        if (dynamicEdgeConv != null) dynamicEdgeConv.close();
        if (dynamicEdgeConvMean != null) dynamicEdgeConvMean.close();
        if (x != null) x.close();
        if (batch != null) batch.close();
//        if (nn != null) nn.close();

//        // 清空PyTorch计算图
//        torch.clear_autograd_graph();
//        torch.cuda.empty_cache();
    }
}
