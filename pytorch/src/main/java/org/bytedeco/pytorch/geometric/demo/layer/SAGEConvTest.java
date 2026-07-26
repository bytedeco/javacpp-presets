package org.bytedeco.pytorch.geometric.demo.layer;

//package org.bytedeco.pytorch.geometric.nn.conv;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.geometric.nn.conv.SAGEConv;

import static org.bytedeco.pytorch.global.torch.*;

/**
 * SAGEConv 测试用例
 * 覆盖：普通图、二部图、维度校验、聚合逻辑、归一化
 */
public class SAGEConvTest {

    public static void main(String[] args) {
        // 1. 初始化环境
        torch.manual_seed(42); // 固定种子，保证结果可复现
        Device cpu = new Device(torch.kCPU());
        TensorOptions floatOpts = new TensorOptions()
                .dtype(new ScalarTypeOptional(torch.kFloat()))
                .device(new DeviceOptional(cpu));
        TensorOptions longOpts = new TensorOptions()
                .dtype(new ScalarTypeOptional(torch.kLong()))
                .device(new DeviceOptional(cpu));

        // 2. 测试普通图场景（核心）
        System.out.println("===== 测试1：普通图 SAGEConv =====");
        testNormalGraph(floatOpts, longOpts);

        // 3. 测试二部图场景
        System.out.println("\n===== 测试2：二部图 SAGEConv =====");
        testBipartiteGraph(floatOpts, longOpts);

        // 4. 测试归一化功能
        System.out.println("\n===== 测试3：归一化功能验证 =====");
        testNormalization(floatOpts, longOpts);
    }

    /**
     * 测试普通图（单节点特征）
     */
    private static void testNormalGraph(TensorOptions floatOpts, TensorOptions longOpts) {
        // 超参数
        long N = 4;          // 节点数
        long inDim = 2;      // 输入维度
        long outDim = 3;     // 输出维度

        // 构造测试数据
        // 节点特征 [4, 2]
        float[] xData = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f};
        Tensor x = tensor(xData, floatOpts).view(N, inDim);
        // 边索引 [2, 4]：0→1, 1→2, 2→3, 3→0（环形图）
        long[] edgeData = {0, 1, 2, 3, 1, 2, 3, 0};
        Tensor edge_index = tensor(edgeData, longOpts).view(2, 4);

        // 创建SAGEConv
        SAGEConv sageConv = new SAGEConv(inDim, outDim);

        // 前向传播
        Tensor out = sageConv.forward(x, edge_index);

        // 维度校验
        verifyShape("普通图输出", out, new long[]{N, outDim});
        System.out.println("✅ 普通图前向传播成功！");
        System.out.println("输出特征维度：[" + out.size(0) + ", " + out.size(1) + "]");
        System.out.println("输出特征：");
        torch.print(out);
    }

    /**
     * 测试二部图（源/目标节点特征分离）
     */
    private static void testBipartiteGraph(TensorOptions floatOpts, TensorOptions longOpts) {
        // 超参数
        long N_src = 3;      // 源节点数
        long N_dst = 2;      // 目标节点数
        long inDim = 2;      // 输入维度
        long outDim = 3;     // 输出维度

        // 构造测试数据
        Tensor xSrc = torch.randn(new long[]{N_src, inDim}, floatOpts); // 源节点特征 [3,2]
        Tensor xDst = torch.randn(new long[]{N_dst, inDim}, floatOpts); // 目标节点特征 [2,2]
        // 边索引 [2, 3]：0→0, 1→1, 2→0（源→目标）
        long[] edgeData = {0, 1, 2, 0, 1, 0};
        Tensor edge_index = tensor(edgeData, longOpts).view(2, 3);

        // 创建SAGEConv
        SAGEConv sageConv = new SAGEConv(inDim, outDim);

        // 前向传播（二部图）
        Tensor out = sageConv.forward(xSrc, xDst, edge_index);

        // 维度校验（输出维度为目标节点数×outDim）
        verifyShape("二部图输出", out, new long[]{N_dst, outDim});
        System.out.println("✅ 二部图前向传播成功！");
        System.out.println("输出特征维度：[" + out.size(0) + ", " + out.size(1) + "]");
    }

    /**
     * 测试归一化功能
     */
    private static void testNormalization(TensorOptions floatOpts, TensorOptions longOpts) {
        // 超参数
        long N = 3;
        long inDim = 2;
        long outDim = 2;

        // 构造测试数据
        Tensor x = torch.ones(new long[]{N, inDim}, floatOpts); // 全1特征 [3,2]
        long[] edgeData = {0, 1, 1, 2, 2, 0};
        Tensor edge_index = tensor(edgeData, longOpts).view(2, 3);

        // 创建带归一化的SAGEConv
        SAGEConv sageConv = new SAGEConv(inDim, outDim, true, true);

        // 前向传播
        Tensor out = sageConv.forward(x, edge_index);

        // 验证归一化：每行L2范数≈1
        Tensor norm = out.norm(new ScalarOptional(new Scalar(2)), new long[]{1}, true);
        System.out.println("归一化后每行L2范数：");
        torch.print(norm);
        // 校验范数是否接近1（误差<1e-6）
        boolean isNormalized = norm.sub(new Scalar(1.0f)).abs().lt(new Scalar(1e-6)).all().item_bool();
        if (isNormalized) {
            System.out.println("✅ 归一化功能验证通过！");
        } else {
            System.out.println("❌ 归一化功能验证失败！");
        }
    }

    /**
     * 通用维度校验工具
     */
    private static void verifyShape(String name, Tensor tensor, long[] expectedShape) {
        long[] actualShape = tensor.shape();// new long[tensor.dim()];
        for (int i = 0; i < tensor.dim(); i++) {
            actualShape[i] = tensor.size(i);
        }

        boolean match = true;
        if (actualShape.length != expectedShape.length) {
            match = false;
        } else {
            for (int i = 0; i < actualShape.length; i++) {
                if (actualShape[i] != expectedShape[i]) {
                    match = false;
                    break;
                }
            }
        }

        if (match) {
            System.out.println("✅ " + name + " 维度校验通过！");
        } else {
            throw new IllegalArgumentException(
                    "❌ " + name + " 维度校验失败！预期：" + arrayToString(expectedShape) +
                            "，实际：" + arrayToString(actualShape)
            );
        }
    }

    /**
     * 数组转字符串
     */
    private static String arrayToString(long[] arr) {
        StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < arr.length; i++) {
            sb.append(arr[i]);
            if (i < arr.length - 1) sb.append(", ");
        }
        sb.append("]");
        return sb.toString();
    }
}
