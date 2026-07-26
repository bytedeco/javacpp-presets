package org.bytedeco.pytorch.geometric.demo.layer;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.geometric.nn.conv.SAGEConvV3;

import static org.bytedeco.pytorch.global.torch.*;

/**
 * 对齐官方API的SAGEConv测试用例
 * 覆盖：普通图/二部图、root_weight、project、不同聚合方式、归一化
 */
public class SAGEConvOfficialTest {

    public static void main(String[] args) {
        // 初始化环境
        torch.manual_seed(42);
        Device cpu = new Device(torch.kCPU());
        TensorOptions floatOpts = new TensorOptions()
                .dtype(new ScalarTypeOptional(torch.kFloat()))
                .device(new DeviceOptional(cpu));
        TensorOptions longOpts = new TensorOptions()
                .dtype(new ScalarTypeOptional(torch.kLong()))
                .device(new DeviceOptional(cpu));

        // 测试1：普通图（默认参数）
        System.out.println("===== 测试1：普通图（默认参数） =====");
        testNormalGraphDefault(floatOpts, longOpts);

        // 测试2：二部图（源/目标维度不同）
        System.out.println("\n===== 测试2：二部图（源3维/目标2维） =====");
        testBipartiteGraph(floatOpts, longOpts);

        // 测试3：关闭root_weight（无自身特征）
        System.out.println("\n===== 测试3：关闭root_weight =====");
        testNoRootWeight(floatOpts, longOpts);

        // 测试4：启用project（聚合前投影）
        System.out.println("\n===== 测试4：启用project投影 =====");
        testProject(floatOpts, longOpts);

        // 测试5：max聚合 + 归一化
        System.out.println("\n===== 测试5：max聚合 + L2归一化 =====");
        testMaxAggrAndNormalize(floatOpts, longOpts);
    }

    /**
     * 测试1：普通图（默认参数）
     */
    private static void testNormalGraphDefault(TensorOptions floatOpts, TensorOptions longOpts) {
        long inDim = 2;
        long outDim = 3;
        long N = 4;

        // 构造数据
        Tensor x = torch.randn(new long[]{N, inDim}, floatOpts);
        long[] edgeData = {0, 1, 2, 3, 1, 2, 3, 0};
        Tensor edgeIndex = tensor(edgeData, longOpts).view(2, 4);

        // 创建SAGEConv（默认参数）
        SAGEConvV3 sageConv = new SAGEConvV3(inDim, outDim);

        // 前向传播
        Tensor out = sageConv.forward(x, edgeIndex);

        // 校验
        verifyShape("普通图默认参数输出", out, new long[]{N, outDim});
        System.out.println("✅ 普通图默认参数测试通过！");
        printTensor(out);
    }

    /**
     * 测试2：二部图（源3维/目标2维）
     */
    private static void testBipartiteGraph(TensorOptions floatOpts, TensorOptions longOpts) {
        long[] inChannels = {3, 2}; // 源3维，目标2维
        long outDim = 4;
        long N_src = 5;
        long N_dst = 3;

        // 构造二部图数据
        Tensor xSrc = torch.randn(new long[]{N_src, inChannels[0]}, floatOpts);
        Tensor xDst = torch.randn(new long[]{N_dst, inChannels[1]}, floatOpts);
        long[] edgeData = {0, 1, 2, 3, 4, 0, 1, 1, 2, 2};
        Tensor edgeIndex = tensor(edgeData, longOpts).view(2, 5);

        // 创建二部图SAGEConv
        SAGEConvV3 sageConv = new SAGEConvV3(inChannels, outDim);

        // 前向传播
        Tensor out = sageConv.forward(xSrc, xDst, edgeIndex);

        // 校验
        verifyShape("二部图输出", out, new long[]{N_dst, outDim});
        System.out.println("✅ 二部图测试通过！");
    }

    /**
     * 测试3：关闭root_weight（无自身特征）
     */
    private static void testNoRootWeight(TensorOptions floatOpts, TensorOptions longOpts) {
        long inDim = 2;
        long outDim = 3;
        long N = 4;

        Tensor x = torch.randn(new long[]{N, inDim}, floatOpts);
        long[] edgeData = {0, 1, 2, 3, 1, 2, 3, 0};
        Tensor edgeIndex = tensor(edgeData, longOpts).view(2, 4);

        // 创建SAGEConv（关闭root_weight）
        SAGEConvV3 sageConv = new SAGEConvV3(inDim, outDim, "mean", false, false, false, true);

        // 前向传播
        Tensor out = sageConv.forward(x, edgeIndex);

        // 校验（维度仍正确，但无自身特征）
        verifyShape("关闭root_weight输出", out, new long[]{N, outDim});
        System.out.println("✅ 关闭root_weight测试通过！");
    }

    /**
     * 测试4：启用project（聚合前投影）
     */
    private static void testProject(TensorOptions floatOpts, TensorOptions longOpts) {
        long inDim = 2;
        long outDim = 3;
        long N = 4;

        Tensor x = torch.randn(new long[]{N, inDim}, floatOpts);
        long[] edgeData = {0, 1, 2, 3, 1, 2, 3, 0};
        Tensor edgeIndex = tensor(edgeData, longOpts).view(2, 4);

        // 创建SAGEConv（启用project）
        SAGEConvV3 sageConv = new SAGEConvV3(inDim, outDim, "mean", false, true, true, true);

        // 验证project层存在
        if (sageConv.getLinProj() == null) {
            throw new RuntimeException("❌ project层未初始化！");
        }

        // 前向传播
        Tensor out = sageConv.forward(x, edgeIndex);

        verifyShape("启用project输出", out, new long[]{N, outDim});
        System.out.println("✅ 启用project测试通过！");
    }

    /**
     * 测试5：max聚合 + L2归一化
     */
    private static void testMaxAggrAndNormalize(TensorOptions floatOpts, TensorOptions longOpts) {
        long inDim = 2;
        long outDim = 2;
        long N = 3;

        Tensor x = torch.ones(new long[]{N, inDim}, floatOpts); // 全1特征
        long[] edgeData = {0, 1, 1, 2, 2, 0};
        Tensor edgeIndex = tensor(edgeData, longOpts).view(2, 3);

        // 创建SAGEConv（max聚合 + 归一化）
        SAGEConvV3 sageConv = new SAGEConvV3(inDim, outDim, "max", true, true, false, true);

        // 前向传播
        Tensor out = sageConv.forward(x, edgeIndex);

        // 校验归一化（每行范数≈1）
        Tensor norm = out.norm(new ScalarOptional(new Scalar(2)), new long[]{1}, true);
        boolean isNormalized = norm.sub(new Scalar(1.0f)).abs().lt(new Scalar(1e-6)).all().item_bool();

        verifyShape("max聚合+归一化输出", out, new long[]{N, outDim});
        if (isNormalized) {
            System.out.println("✅ max聚合 + L2归一化测试通过！");
            printTensor(out);
            printTensor(norm);
        } else {
            throw new RuntimeException("❌ 归一化验证失败！");
        }
    }

    // ========== 通用工具方法 ==========
    private static void verifyShape(String name, Tensor tensor, long[] expectedShape) {
        long[] actualShape = tensor.shape();// new long[tensor.dim()];
        for (int i = 0; i < tensor.dim(); i++) {
            actualShape[i] = tensor.size(i);
        }

        boolean match = true;
        if (actualShape.length != expectedShape.length) match = false;
        else {
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

    private static void printTensor(Tensor tensor) {
        System.out.flush();
        System.out.println("Tensor values:");
        torch.print(tensor);
        System.out.flush();
    }

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
