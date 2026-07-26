package org.bytedeco.pytorch.geometric.demo.layer;
//package org.bytedeco.pytorch.geometric.nn.conv;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.geometric.nn.conv.TAGConv;

/**
 * TAGConv 测试用例（修复形状不匹配问题）
 */
public class TAGConvTest {
    public static void main(String[] args) {
        // 1. 基础测试：小图验证
        testBasicFunctionality();

        // 2. 非法参数测试
        testInvalidParameters();

        // 3. 内存释放测试
        testResourceRelease();

        System.out.println("✅ 所有测试通过！");
    }

    /**
     * 基础功能测试：
     * - 节点特征：5个节点，输入维度3
     * - 边索引：简单无向图（8条边，2×8=16个元素）
     * - 跳数K=2，输出维度4
     */
    private static void testBasicFunctionality() {
        // 1. 构造测试数据
        long inChannels = 3;
        long outChannels = 4;
        int K = 2;

        // 节点特征 [5, 3]（5×3=15个元素）
        float[] xData = {
                1.0f, 2.0f, 3.0f, // 节点0
                4.0f, 5.0f, 6.0f, // 节点1
                7.0f, 8.0f, 9.0f, // 节点2
                10.0f, 11.0f, 12.0f, // 节点3
                13.0f, 14.0f, 15.0f  // 节点4
        };
        Tensor x = torch.tensor(xData).reshape(5, 3).to(torch.ScalarType.Float);

        // 边索引 [2, 8]（2×8=16个元素，8条无向边）
        // 格式说明：第一行是目标节点，第二行是源节点（邻居）
        long[] edgeIndexData = {
                0, 0, 1, 1, 2, 3, 0, 2, // 目标节点（第一行）
                1, 2, 2, 3, 4, 4, 3, 3  // 源节点（第二行）
        };
        // 修正：确保元素数=2×8=16，reshape形状匹配
        Tensor edgeIndex = torch.tensor(edgeIndexData).reshape(2, 8).to(torch.ScalarType.Long);

        // 2. 创建 TAGConv 实例
        TAGConv tagConv = new TAGConv(inChannels, outChannels, K);

        // 3. 前向传播
        Tensor output = tagConv.forward(x, edgeIndex);

        // 4. 验证输出维度
        assert output.dim() == 2 : "输出必须是2维张量";
        assert output.size(0) == 5 : "输出节点数必须为5，实际：" + output.size(0);
        assert output.size(1) == 4 : "输出维度必须为4，实际：" + output.size(1);
        System.out.println("✅ 基础功能测试通过：输出形状 = " + output.size(0) + "x" + output.size(1));

        // 5. 测试参数重置
        tagConv.reset_parameters();
        System.out.println("✅ 参数重置测试通过");

        // 6. 释放资源
        tagConv.close();
        x.close();
        edgeIndex.close();
        output.close();
    }

    /**
     * 非法参数测试：验证参数校验逻辑
     */
    private static void testInvalidParameters() {
        // 测试1：输入通道数≤0
        try {
            new TAGConv(0, 4, 2);
            assert false : "未捕获非法输入通道数";
        } catch (IllegalArgumentException e) {
            System.out.println("✅ 非法输入通道数测试通过：" + e.getMessage());
        }

        // 测试2：输出通道数≤0
        try {
            new TAGConv(3, -1, 2);
            assert false : "未捕获非法输出通道数";
        } catch (IllegalArgumentException e) {
            System.out.println("✅ 非法输出通道数测试通过：" + e.getMessage());
        }

        // 测试3：跳数K<0
        try {
            new TAGConv(3, 4, -1);
            assert false : "未捕获非法跳数K";
        } catch (IllegalArgumentException e) {
            System.out.println("✅ 非法跳数K测试通过：" + e.getMessage());
        }

        // 测试4：节点特征维度不匹配
        try {
            TAGConv tagConv = new TAGConv(3, 4, 2);
            Tensor x = torch.tensor(new float[]{1,2}).reshape(1,2); // 维度2≠3
            Tensor edgeIndex = torch.tensor(new long[]{0,1,1,0}).reshape(2,2); // 合法边索引
            tagConv.forward(x, edgeIndex);
            assert false : "未捕获维度不匹配";
        } catch (IllegalArgumentException e) {
            System.out.println("✅ 维度不匹配测试通过：" + e.getMessage());
        } finally {
            // 确保资源释放
            try {
                TAGConv tagConv = new TAGConv(3, 4, 2);
                tagConv.close();
            } catch (Exception e) {
                // 忽略释放异常
            }
        }
    }

    /**
     * 内存释放测试：验证资源释放逻辑
     */
    private static void testResourceRelease() {
        TAGConv tagConv = new TAGConv(3, 4, 2);
        tagConv.close();

        // 测试释放后调用方法
        try {
            tagConv.reset_parameters();
            assert false : "未捕获释放后调用";
        } catch (IllegalStateException e) {
            System.out.println("✅ 资源释放后调用测试通过：" + e.getMessage());
        }

        // 测试重复释放（无异常）
        tagConv.close();
        System.out.println("✅ 重复释放测试通过");
    }
}