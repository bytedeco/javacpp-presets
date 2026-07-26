package org.bytedeco.pytorch.geometric.demo.layer;



import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.geometric.nn.conv.GATConvFinal;
import org.bytedeco.pytorch.geometric.utils.TensorToolkit;

import static org.bytedeco.pytorch.global.torch.*;

/**
 * 修改后测试用例：生成有意义的数值输出
 */
public class GATConvUltimateTestV2 {

    public static void main(String[] args) {
        // 1. 固定随机种子（保证结果可复现）
        torch.manual_seed(42);

        // 2. 测试参数
        long N = 3;              // 节点数
        long inChannels = 2;     // 输入特征维度
        long outChannels = 2;    // 单头输出维度
        long heads = 2;          // 注意力头数
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

        // 4. 构造测试数据（核心修改：随机输入特征，非0/1）
        // 边索引 [2,2]（保持不变）
        long[] edgeData = {0, 1, 1, 2};
        Tensor edge_index = tensor(edgeData, longOpts).view(2, 2);
        verifyTensorShape("edge_index", edge_index, new long[]{2, 2});

        // ✅ 核心修改1：随机节点特征（非0/1）
        Tensor x = torch.randn(new long[]{3, 2}, floatOpts); // 正态分布随机数
        verifyTensorShape("x", x, new long[]{3, 2});

        // 5. 创建GATConv并初始化权重（核心修改：随机权重）
        GATConvFinal gat = new GATConvFinal(inChannels, outChannels, heads, concat, negativeSlope);
        long linearOutDim = heads * outChannels;

        // ✅ 核心修改2：随机初始化线性层权重（而非单位矩阵）
        Tensor linWeight = torch.randn(new long[]{linearOutDim, inChannels}, floatOpts);
        // 权重缩放（保证数值范围合理）
        linWeight = linWeight.mul(new Scalar(0.1));
        gat.getLin().weight().set_data(linWeight);
        verifyTensorShape("linWeight", linWeight, new long[]{linearOutDim, inChannels});

        // 偏置清零（保持不变）
        gat.getLin().bias().set_data(zeros(new long[]{linearOutDim}, floatOpts));

        // 6. 前向传播
        Tensor out = gat.forward(x, edge_index);

        torch.print(out);
        // 7. 打印结果（核心修改：优化打印格式，更易读）
        System.out.println("===== 输入节点特征（随机值） =====");
        printTensor(x);
        System.out.println("\n===== 边索引 =====");
        printTensor(edge_index);
        System.out.println("\n===== 线性层权重 =====");
        printTensor(gat.getLin().weight());
        System.out.println("\n===== 注意力向量 =====");
        printTensor(gat.getAttParam());
        System.out.println("\n===== GATConv 输出（多头拼接） =====");
        printTensor(out);

        // 8. 验证维度
        verifyDimension(out, N, concat ? heads * outChannels : outChannels);

        // 9. 验证注意力分数
        verifyAttentionScore(gat, x, edge_index, floatOpts);
    }

    /**
     * 优化张量打印格式（解决输出错乱问题）
     */
    private static void printTensor(Tensor tensor) {
        // 创建临时张量避免修改原数据
        Tensor temp = tensor.clone();
        // 打印张量（适配bytedeco-pytorch的打印逻辑）
        torch.print(temp);
//        System.out.print(temp.toString());
        System.out.println("\r\n");
    }

    /**
     * 验证输出维度
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
     * 验证注意力分数
     */
    private static void verifyAttentionScore(GATConvFinal gat, Tensor x, Tensor edge_index, TensorOptions floatOpts) {
        try {
            Device cpu = new Device(torch.kCPU());
            long N = x.size(0);
            Tensor xLin = gat.getLin().forward(x).view(N, gat.getHeads(), gat.getOutChannels());
            verifyTensorShape("xLin", xLin, new long[]{N, gat.getHeads(), gat.getOutChannels()});

            Tensor edge_cpu = edge_index.to(cpu, torch.kLong());

            // 提取源/目标节点索引
            Tensor srcIdx = edge_cpu.select(0, 0);
            Tensor dstIdx = edge_cpu.select(0, 1);

            // 提取边维度特征
            Tensor x_j = xLin.index_select(0, srcIdx);
            Tensor x_i = xLin.index_select(0, dstIdx);

            // 计算注意力分数
            Tensor catFeat = torch.cat(new TensorVector(x_i, x_j), 2);
            long[] catShape = catFeat.shape();
            long E = catShape[0];
            long heads = catShape[1];
            long lastDim = catShape[2];

            // 扩展att并相乘
            Tensor attExpanded = gat.getAttParam().expand(new long[]{E, heads, lastDim});
            Tensor alpha = catFeat.mul(attExpanded);

            // 手动求和
            alpha = sumLastDim(alpha, floatOpts);
            verifyTensorShape("alpha after sum", alpha, new long[]{E, heads});

            alpha = torch.leaky_relu(alpha, new Scalar(gat.getNegativeSlope()));
            alpha = gat.scatter_softmax(alpha, dstIdx, N);

            System.out.println("\n===== 注意力分数 =====");
            printTensor(alpha);
            System.out.println("\r\n===== 注意力分数（简化版） =====");
            torch.print(alpha);

            // 验证注意力分数非负
            boolean nonNegative = alpha.ge(new Scalar(0.0f)).all().item_bool();

            if (nonNegative) {
                System.out.println("✅ 注意力分数非负验证通过！");
            } else {
                System.out.println("❌ 注意力分数存在负值！");
            }
        } catch (Exception e) {
            System.out.println("⚠️ 注意力分数验证跳过：" + e.getMessage());
            e.printStackTrace();
        }
    }

    /**
     * 手动对最后一维求和
     */
    private static Tensor sumLastDim(Tensor tensor, TensorOptions floatOpts) {
        int dims = (int)tensor.dim();
        if (dims == 0) return tensor;

        long[] shape = new long[dims];
        for (int i = 0; i < dims; i++) {
            shape[i] = tensor.size(i);
        }

        long lastDimSize = shape[dims - 1];
        long prefixSize = 1;
        for (int i = 0; i < dims - 1; i++) {
            prefixSize *= shape[i];
        }

        Tensor flat = tensor.view(prefixSize, lastDimSize);
        Tensor sumResult = torch.zeros(new long[]{prefixSize}, floatOpts);

        for (long i = 0; i < prefixSize; i++) {
            Tensor row = flat.index_select(0, torch.tensor(new long[]{i}, floatOpts).to(torch.kLong()));
            Tensor rowSum = row.sum();
            sumResult.put(torch.tensor(new long[]{i}), rowSum);
        }

        long[] newShape = new long[dims - 1];
        System.arraycopy(shape, 0, newShape, 0, dims - 1);
        return sumResult.view(newShape);
    }

    /**
     * 工具方法：校验张量维度
     */
    private static void verifyTensorShape(String name, Tensor tensor, long[] expectedShape) {
        long expectedTotal = 1;
        for (long s : expectedShape) {
            expectedTotal *= s;
        }
        long actualTotal = tensor.numel();

        if (actualTotal != expectedTotal) {
            throw new IllegalArgumentException(
                    "张量 " + name + " 元素数不匹配：预期 " + expectedTotal + "，实际 " + actualTotal
            );
        }

        if (tensor.dim() != expectedShape.length) {
            throw new IllegalArgumentException(
                    "张量 " + name + " 维度数不匹配：预期 " + expectedShape.length + "，实际 " + tensor.dim()
            );
        }

        long[] actualShape = new long[expectedShape.length];
        for (int i = 0; i < expectedShape.length; i++) {
            actualShape[i] = tensor.size(i);
            if (actualShape[i] != expectedShape[i]) {
                throw new IllegalArgumentException(
                        "张量 " + name + " 维度" + i + "不匹配：预期 " + expectedShape[i] + "，实际 " + actualShape[i]
                );
            }
        }
        System.out.println("✅ 张量 " + name + " 维度校验通过：" + arrayToString(expectedShape));
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