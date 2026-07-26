package org.bytedeco.pytorch.geometric.demo.layer;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.geometric.nn.conv.GATConvFinal;

import static org.bytedeco.pytorch.global.torch.*;

/**
 * 复杂图结构 + 差异化注意力分数（非全1）
 */
public class GATConvComplexTest {

    public static void main(String[] args) {
        // 1. 固定随机种子（保证结果可复现）
        torch.manual_seed(42);

        // 2. 测试参数（保持不变）
        long N = 3;              // 节点数（仍为3，但边更复杂）
        long inChannels = 2;     // 输入特征维度
        long outChannels = 2;    // 单头输出维度
        long heads = 2;          // 注意力头数（2个头，分数会更丰富）
        boolean concat = true;   // 拼接多头
        double negativeSlope = 0.2;

        // 3. 创建TensorOptions
        Device cpu = new Device(torch.kCPU());
        TensorOptions floatOpts = new TensorOptions()
                .dtype(new ScalarTypeOptional(torch.kFloat()))
                .device(new DeviceOptional(cpu));
        TensorOptions longOpts = new TensorOptions()
                .dtype(new ScalarTypeOptional(torch.kLong()))
                .device(new DeviceOptional(cpu));

        // 4. 构造【复杂图结构】（核心改造：让节点2收到2条入边）
        // 边索引 [2,4]：4条边，目标节点分布：1(1条)、2(2条)、0(1条)
        // 边0：0→1，边1：0→2，边2：1→2，边3：2→0
        long[] edgeData = {0, 0, 1, 2, 1, 2, 2, 0}; // [源节点, 目标节点]
        Tensor edge_index = tensor(edgeData, longOpts).view(2, 4);
        verifyTensorShape("edge_index", edge_index, new long[]{2, 4});

        // 5. 构造【非对称节点特征】（让注意力分数有差异）
        // 不再用0/1，改用差异化的浮点值
        float[][] xData = {{0.5f, 0.1f}, {0.2f, 0.8f}, {0.9f, 0.3f}};
        float[] flatX = flattenFloatArray(xData);
        Tensor x = tensor(flatX, floatOpts).view(3, 2);
        verifyTensorShape("x", x, new long[]{3, 2});

        // 6. 创建GATConv并初始化【随机权重】（避免恒等变换）
        GATConvFinal gat = new GATConvFinal(inChannels, outChannels, heads, concat, negativeSlope);
        long linearOutDim = heads * outChannels;

        // 随机权重（非单位矩阵，引入数值差异）
        Tensor linWeight = torch.randn(new long[]{linearOutDim, inChannels}, floatOpts).mul(new Scalar(0.1));
        gat.getLin().weight().set_data(linWeight);
        verifyTensorShape("linWeight", linWeight, new long[]{linearOutDim, inChannels});
        gat.getLin().bias().set_data(zeros(new long[]{linearOutDim}, floatOpts));

        // 7. 前向传播
        Tensor out = gat.forward(x, edge_index);

        // 8. 打印核心信息（清晰看到数值）
        System.out.println("===== 图结构说明 =====");
        System.out.println("节点数：3，边数：4");
        System.out.println("边列表：0→1、0→2、1→2、2→0");
        System.out.println("目标节点分布：节点0(1条边)、节点1(1条边)、节点2(2条边)");

        System.out.println("\n===== 节点特征（差异化） =====");
        torch.print(x);

        System.out.println("\n===== 边索引 =====");
        torch.print(edge_index);

        System.out.println("\n===== 线性层权重 =====");
        torch.print(gat.getLin().weight());

        // 9. 计算并打印【差异化注意力分数】（核心）
        System.out.println("\n===== 注意力分数（非全1） =====");
        Tensor alpha = calculateAttentionScore(gat, x, edge_index, floatOpts);
        torch.print(alpha);

        // 10. 验证注意力分数特性（总和为1）
        verifyAttentionSum(alpha, edge_index, N);

        System.out.println("\n===== GATConv 输出（多头拼接） =====");
        torch.print(out);

        // 11. 验证维度
        verifyDimension(out, N, concat ? heads * outChannels : outChannels);
    }

    /**
     * 计算注意力分数（提取核心逻辑）
     */
    private static Tensor calculateAttentionScore(GATConvFinal gat, Tensor x, Tensor edge_index, TensorOptions floatOpts) {
        long N = x.size(0);
        // 线性变换 + 重塑为多头
        Tensor xLin = gat.getLin().forward(x).view(N, gat.getHeads(), gat.getOutChannels());

        // 提取源/目标节点索引
        Tensor srcIdx = edge_index.select(0, 0);
        Tensor dstIdx = edge_index.select(0, 1);

        // 提取边维度特征
        Tensor x_j = xLin.index_select(0, srcIdx);
        Tensor x_i = xLin.index_select(0, dstIdx);

        // 拼接特征 [E, heads, 2*outChannels]
        Tensor catFeat = torch.cat(new TensorVector(x_i, x_j), 2);

        // 扩展注意力向量并相乘
        Tensor attExpanded = gat.getAttParam().expand(catFeat.size(0), catFeat.size(1), catFeat.size(2));
        Tensor alpha = catFeat.mul(attExpanded);

        // 手动对最后一维求和
        alpha = sumLastDim(alpha, floatOpts);

        // LeakyReLU激活
        alpha = torch.leaky_relu(alpha, new Scalar(gat.getNegativeSlope()));

        // Scatter Softmax（核心：对同一目标节点的边做归一化）
        alpha = gat.scatter_softmax(alpha, dstIdx, N);

        return alpha;
    }

    /**
     * 验证注意力分数：同一目标节点的分数总和≈1
     */
    private static void verifyAttentionSum(Tensor alpha, Tensor edge_index, long numNodes) {
        Tensor dstIdx = edge_index.select(0, 1);
        System.out.println("\n===== 注意力分数总和验证 =====");

        for (long node = 0; node < numNodes; node++) {
            // 找到目标节点为node的所有边索引
            Tensor mask = dstIdx.eq(new Scalar(node));
            Tensor nodeAlpha = alpha.index_select(0, torch.nonzero(mask).select(1, 0));

            if (nodeAlpha.numel() == 0) {
                System.out.println("节点" + node + "：无入边");
                continue;
            }

            // 对每个头的分数求和
            System.out.println("节点" + node + " 入边注意力分数：");
            for (long head = 0; head < alpha.size(1); head++) {
                Tensor headAlpha = nodeAlpha.select(1, head);
                Tensor sum = headAlpha.sum();
                System.out.printf("  头%d：总和=%.4f（应≈1）\n", head, sum.item_float());
            }
        }
    }

    /**
     * 手动对最后一维求和（核心逻辑不变）
     */
    private static Tensor sumLastDim(Tensor tensor, TensorOptions floatOpts) {
        int dims = (int)tensor.dim();
        if (dims == 0) return tensor;

        long[] shape = new long[(int) dims];
        for (int i = 0; i < dims; i++) shape[i] = tensor.size(i);

        long lastDimSize = shape[dims - 1];
        long prefixSize = 1;
        for (int i = 0; i < dims - 1; i++) prefixSize *= shape[i];

        Tensor flat = tensor.view(prefixSize, lastDimSize);
        Tensor sumResult = torch.zeros(new long[]{prefixSize}, floatOpts);

        for (long i = 0; i < prefixSize; i++) {
            Tensor row = flat.index_select(0, torch.tensor(new long[]{i}, floatOpts).to(torch.kLong()));
            sumResult.put(torch.tensor(new long[]{i}), row.sum());
        }

        long[] newShape = new long[(int) dims - 1];
        System.arraycopy(shape, 0, newShape, 0, (int) dims - 1);
        return sumResult.view(newShape);
    }

    /**
     * 工具：展平浮点数组
     */
    private static float[] flattenFloatArray(float[][] arr) {
        int rows = arr.length;
        int cols = arr[0].length;
        float[] flat = new float[rows * cols];
        int idx = 0;
        for (float[] row : arr) {
            for (float val : row) {
                flat[idx++] = val;
            }
        }
        return flat;
    }

    /**
     * 验证维度
     */
    private static void verifyDimension(Tensor out, long expectedN, long expectedF) {
        long actualN = out.size(0);
        long actualF = out.size(1);
        if (actualN == expectedN && actualF == expectedF) {
            System.out.println("\n✅ 维度验证通过！输出维度: [" + actualN + ", " + actualF + "]");
        } else {
            System.out.println("\n❌ 维度验证失败！预期: [" + expectedN + ", " + expectedF + "]，实际: [" + actualN + ", " + actualF + "]");
        }
    }

    /**
     * 校验张量维度
     */
    private static void verifyTensorShape(String name, Tensor tensor, long[] expectedShape) {
        long expectedTotal = 1;
        for (long s : expectedShape) expectedTotal *= s;
        if (tensor.numel() != expectedTotal) {
            throw new IllegalArgumentException(name + " 元素数不匹配：预期 " + expectedTotal + "，实际 " + tensor.numel());
        }
        if (tensor.dim() != expectedShape.length) {
            throw new IllegalArgumentException(name + " 维度数不匹配：预期 " + expectedShape.length + "，实际 " + tensor.dim());
        }
        for (int i = 0; i < expectedShape.length; i++) {
            if (tensor.size(i) != expectedShape[i]) {
                throw new IllegalArgumentException(name + " 维度" + i + "不匹配：预期 " + expectedShape[i] + "，实际 " + tensor.size(i));
            }
        }
        System.out.println("✅ 张量 " + name + " 维度校验通过：[" + String.join(", ", java.util.Arrays.stream(expectedShape).mapToObj(String::valueOf).toArray(String[]::new)) + "]");
    }
}
