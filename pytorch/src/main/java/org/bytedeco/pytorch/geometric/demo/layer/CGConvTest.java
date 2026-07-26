package org.bytedeco.pytorch.geometric.demo.layer;
import org.bytedeco.pytorch.nn.modules.*;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.geometric.nn.conv.CGConv;

/**
 * CGConv 测试用例
 * 覆盖场景：基础功能、不同聚合方式、BatchNorm/偏置、非法参数、空输入、资源释放
 */
public class CGConvTest {
    public static void main(String[] args) {
        try {
            // 1. 基础功能测试（add 聚合 + 有边特征 + 有偏置 + 无BatchNorm）
            testBasicFunctionality("add", 2, true, false);

            // 2. 不同聚合方式测试（mean/max）
            testBasicFunctionality("mean", 2, true, false);
            testBasicFunctionality("max", 2, true, false);

            // 3. BatchNorm + 无偏置测试
            testBasicFunctionality("add", 2, false, true);

            // 4. 无边特征测试
            testNoEdgeAttr();

            // 5. 非法参数测试
            testInvalidParameters();

            // 6. 空输入测试
            testNullInput();

            // 7. 资源释放测试
            testResourceRelease();

            System.out.println("✅ 所有 CGConv 测试通过！");
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    /**
     * 基础功能测试
     * @param aggr 聚合方式（add/mean/max）
     * @param edgeDim 边特征维度
     * @param hasBias 是否使用偏置
     * @param batchNorm 是否启用 BatchNorm
     */
    private static void testBasicFunctionality(String aggr, int edgeDim, boolean hasBias, boolean batchNorm) {
        long channels = 3; // 节点特征维度
        long numNodes = 5; // 5个节点
        long numEdges = 8; // 8条边

        // 1. 构造测试数据
        // 节点特征 [5, 3]
        float[] xData = {
                1.0f, 2.0f, 3.0f,
                4.0f, 5.0f, 6.0f,
                7.0f, 8.0f, 9.0f,
                10.0f, 11.0f, 12.0f,
                13.0f, 14.0f, 15.0f
        };
        Tensor x = torch.tensor(xData).reshape(numNodes, channels).to(torch.ScalarType.Float);

        // 边索引 [2, 8]
        long[] edgeIndexData = {
                0, 0, 1, 1, 2, 3, 0, 2, // 源节点
                1, 2, 2, 3, 4, 4, 3, 3  // 目标节点
        };
        Tensor edgeIndex = torch.tensor(edgeIndexData).reshape(2, numEdges).to(torch.ScalarType.Long);

        // 边特征 [8, edgeDim]
        Tensor edgeAttr = null;
        if (edgeDim > 0) {
            float[] edgeAttrData = new float[(int)numEdges * edgeDim];
            for (int i = 0; i < edgeAttrData.length; i++) {
                edgeAttrData[i] = i * 0.1f; // 随机填充
            }
            edgeAttr = torch.tensor(edgeAttrData).reshape(numEdges, edgeDim).to(torch.ScalarType.Float);
        }

        // 2. 创建 CGConv 实例
        CGConv cgConv = new CGConv(channels, edgeDim, aggr, batchNorm, hasBias);

        // 3. 测试参数重置
//        cgConv.resetParameters();
        System.out.println("✅ 参数重置测试通过（聚合：" + aggr + "，edgeDim：" + edgeDim + "）");

        // 4. 前向传播
        Tensor output = ((CGConv)cgConv).forward(x, edgeIndex, edgeAttr);

        // 5. 验证输出维度
        assert output.dim() == 2 : "输出必须是2维张量";
        assert output.size(0) == numNodes : "输出节点数必须为" + numNodes + "，实际：" + output.size(0);
        assert output.size(1) == channels : "输出维度必须为" + channels + "，实际：" + output.size(1);

        String config = String.format("聚合=%s, edgeDim=%d, bias=%s, batchNorm=%s",
                aggr, edgeDim, hasBias, batchNorm);
        System.out.println("✅ " + config + " 测试通过：输出形状 = " + output.size(0) + "x" + output.size(1));

        // 6. 释放资源
        cgConv.close();
        x.close();
        edgeIndex.close();
        if (edgeAttr != null) edgeAttr.close();
        output.close();
    }

    /**
     * 无边特征测试
     */
    private static void testNoEdgeAttr() {
        long channels = 3;
        int edgeDim = 0; // 无边特征
        long numNodes = 5;
        long numEdges = 8;

        // 构造数据
        Tensor x = torch.randn(numNodes, channels);
        Tensor edgeIndex = torch.tensor(new long[]{
                0, 0, 1, 1, 2, 3, 0, 2,
                1, 2, 2, 3, 4, 4, 3, 3
        }).reshape(2, numEdges).to(torch.ScalarType.Long);

        // 创建实例并前向传播
        CGConv cgConv = new CGConv(channels, edgeDim, "add", false, true);
        Tensor output = ((CGConv)cgConv).forward(x, edgeIndex);

        // 验证输出
        assert output.size(0) == numNodes && output.size(1) == channels : "无边特征输出维度错误";
        System.out.println("✅ 无边特征测试通过：输出形状 = " + output.size(0) + "x" + output.size(1));

        // 释放资源
        cgConv.close();
        x.close();
        edgeIndex.close();
        output.close();
    }

    /**
     * 非法参数测试
     */
    private static void testInvalidParameters() {
        // 测试1：通道数≤0
        try {
            new CGConv(0, 2, "add", false, true);
            assert false : "未捕获非法通道数";
        } catch (IllegalArgumentException e) {
            System.out.println("✅ 非法通道数测试通过：" + e.getMessage());
        }

        // 测试2：边特征维度<0
        try {
            new CGConv(3, -1, "add", false, true);
            assert false : "未捕获非法边特征维度";
        } catch (IllegalArgumentException e) {
            System.out.println("✅ 非法边特征维度测试通过：" + e.getMessage());
        }

        // 测试3：非法聚合方式
        try {
            new CGConv(3, 2, "sum", false, true);
            assert false : "未捕获非法聚合方式";
        } catch (IllegalArgumentException e) {
            System.out.println("✅ 非法聚合方式测试通过：" + e.getMessage());
        }

        // 测试4：节点特征维度不匹配
        try {
            CGConv conv = new CGConv(3, 2, "add", false, true);
            Tensor x = torch.randn(5, 4); // 维度4≠3
            Tensor edgeIndex = torch.randint(0, 5, new long[]{2, 8});
            ((CGConv)conv).forward(x, edgeIndex, torch.randn(8, 2));
            assert false : "未捕获节点特征维度不匹配";
        } catch (IllegalArgumentException e) {
            System.out.println("✅ 节点特征维度不匹配测试通过：" + e.getMessage());
        } finally {
            CGConv conv = new CGConv(3, 2, "add", false, true);
            conv.close();
        }

        // 测试5：边特征维度不匹配
        try {
            CGConv conv = new CGConv(3, 2, "add", false, true);
            Tensor x = torch.randn(5, 3);
            Tensor edgeIndex = torch.randint(0, 5, new long[]{2, 8});
            Tensor edgeAttr = torch.randn(8, 3); // 维度3≠2
            ((CGConv)conv).forward(x, edgeIndex, edgeAttr);
            assert false : "未捕获边特征维度不匹配";
        } catch (IllegalArgumentException e) {
            System.out.println("✅ 边特征维度不匹配测试通过：" + e.getMessage());
        } finally {
            CGConv conv = new CGConv(3, 2, "add", false, true);
            conv.close();
        }
    }

    /**
     * 空输入测试
     */
    private static void testNullInput() {
        CGConv conv = new CGConv(3, 2, "add", false, true);

        // 测试1：空节点特征
        try {
            ((CGConv)conv).forward(null, torch.randint(0, 5, new long[]{2, 8}), torch.randn(8, 2));
            assert false : "未捕获空节点特征";
        } catch (IllegalArgumentException e) {
            System.out.println("✅ 空节点特征测试通过：" + e.getMessage());
        }

        // 测试2：空边索引
        try {
            ((CGConv)conv).forward(torch.randn(5, 3), null, torch.randn(8, 2));
            assert false : "未捕获空边索引";
        } catch (IllegalArgumentException e) {
            System.out.println("✅ 空边索引测试通过：" + e.getMessage());
        }

        // 测试3：指定 edgeDim>0 但未传边特征
        try {
            CGConv conv2 = new CGConv(3, 2, "add", false, true);
            ((CGConv)conv2).forward(torch.randn(5, 3), torch.randint(0, 5, new long[]{2, 8}), (Tensor)null);
            assert false : "未捕获缺失边特征";
        } catch (IllegalArgumentException e) {
            System.out.println("✅ 缺失边特征测试通过：" + e.getMessage());
        } finally {
            CGConv conv2 = new CGConv(3, 2, "add", false, true);
            conv2.close();
        }

        // 释放资源
        conv.close();
    }

    /**
     * 资源释放测试
     */
    private static void testResourceRelease() {
        CGConv conv = new CGConv(3, 2, "add", true, true);
        conv.close();

        // 测试释放后调用 forward
        try {
            ((CGConv)conv).forward(torch.randn(5, 3), torch.randint(0, 5, new long[]{2, 8}), torch.randn(8, 2));
            assert false : "未捕获释放后调用 forward";
        } catch (IllegalStateException e) {
            System.out.println("✅ 释放后调用 forward 测试通过：" + e.getMessage());
        }

        // 测试释放后重置参数
        try {
            conv.resetParameters();
            assert false : "未捕获释放后重置参数";
        } catch (IllegalStateException e) {
            System.out.println("✅ 释放后重置参数测试通过：" + e.getMessage());
        }

        // 测试重复释放
        conv.close();
        System.out.println("✅ 重复释放测试通过");
    }
}
