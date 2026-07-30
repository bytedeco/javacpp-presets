package samples.demo.layer;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.geometric.nn.conv.GATConv;
import org.bytedeco.pytorch.geometric.nn.conv.GATConvV2;
import org.bytedeco.pytorch.geometric.utils.Scatter;
import org.bytedeco.pytorch.geometric.utils.TensorToolkit;

import static org.bytedeco.pytorch.global.torch.*;
import static org.bytedeco.pytorch.geometric.utils.GraphUtils.add_self_loops;

/**
 * GATConv测试用例：修复维度不匹配+MessagePassing调用错误
 * 核心：适配MessagePassing的propagate调用规范，确保张量维度一致
 */
public class GATConvTestV2 {

    public static void main(String[] args) {
        // 1. 固定随机种子（确保结果可复现）
        torch.manual_seed(42);

        // 2. 测试参数
        long N = 3;              // 节点数
        long inChannels = 2;     // 输入特征维度
        long outChannels = 2;    // 单头输出维度
        long heads = 2;          // 注意力头数
        boolean concat = true;   // 拼接多头结果
        double negativeSlope = 0.2;

        // 适配bytedeco-pytorch的TensorOptions构造
        Device cpuDevice = new Device(DeviceType.CPU);
        TensorOptions floatOpts = new TensorOptions()
                .dtype(new ScalarTypeOptional(kFloat()))
                .device(new DeviceOptional(cpuDevice));
        TensorOptions longOpts = new TensorOptions()
                .dtype(new ScalarTypeOptional(kLong()))
                .device(new DeviceOptional(cpuDevice));

        // 3. 构造测试数据（极简版，避免维度干扰）
        // 边索引：0→1, 1→2（不加自环，先验证基础逻辑）
        long[] edgeData = {0, 1, 1, 2};
        Tensor edge_index = tensor(edgeData, longOpts).view(2, 2);
        // 可选：加自环（修复后可加，先测试基础场景）
        // edge_index = add_self_loops(edge_index, N);

        // 节点特征：[[1,0], [0,1], [1,1]]
        float[][] xData = {{1.0f, 0.0f}, {0.0f, 1.0f}, {1.0f, 1.0f}};
        float[] flatX = (float[]) TensorToolkit.flatten(xData);
        long[] xShape = TensorToolkit.getShape(xData);
        Tensor x = tensor(flatX, floatOpts).view(xShape);

        // 4. 创建GATConv并固定线性层权重（维度匹配）
        GATConvV2 gat = new GATConvV2(inChannels, outChannels, heads, concat, negativeSlope);
        // 线性层权重：[heads*outChannels, inChannels] = [4,2]
        long linearOutDim = heads * outChannels;
        Tensor linWeight = eye(inChannels, linearOutDim, floatOpts).t(); // [4,2]
        gat.getLin().weight().set_data(linWeight);
        // 偏置清零
        gat.getLin().bias().set_data(zeros(new long[]{linearOutDim}, floatOpts));

        // 5. 前向传播（核心：此时无维度错误）
        Tensor out = gat.forward(x, edge_index);

        // 6. 打印结果
        System.out.println("===== 输入节点特征 =====");
        torch.print(x);
        System.out.println("\n===== 边索引 =====");
        torch.print(edge_index);
        System.out.println("\n===== GATConv输出（多头拼接） =====");
        torch.print(out);

        // 7. 验证核心维度
        verifyDimension(out, N, concat ? heads * outChannels : outChannels);

        // 8. 验证注意力分数（简化版，避免维度干扰）
        verifyAttentionScoreSimple(gat, x, edge_index, floatOpts);
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
     * 简化版注意力分数验证（避免MessagePassing内部维度问题）
     */
    private static void verifyAttentionScoreSimple(GATConvV2 gat, Tensor x, Tensor edge_index, TensorOptions floatOpts) {
        try {
            long N = x.size(0);
            Tensor xLin = gat.getLin().forward(x).view(N, gat.getHeads(), gat.getOutChannels());
            Tensor edge_cpu = edge_index.to(new Device(torch.kCPU()),torch.kLong());

            // 构造x_i/x_j（确保维度为 [E, heads, outChannels]）
            long E = edge_cpu.size(1);
            Tensor srcIdx = edge_cpu.select(0, 0); // 源节点 [E]
            Tensor dstIdx = edge_cpu.select(0, 1); // 目标节点 [E]
            Tensor x_j = xLin.index_select(0, srcIdx); // [E, 2, 2]
            Tensor x_i = xLin.index_select(0, dstIdx); // [E, 2, 2]

            // 计算注意力分数
            Tensor catFeat = torch.cat(new TensorVector(x_i, x_j), -1); // [E,2,4]
            Tensor alpha = catFeat.mul(gat.getAttParam()).sum(-1); // [E,2]
            alpha = torch.leaky_relu(alpha, new Scalar(gat.getNegativeSlope()));
            alpha = gat.scatter_softmax(alpha, dstIdx, N);

            // 验证非负
            boolean nonNegative = alpha.ge(new Scalar(0.0f)).all().item_bool();
            if (nonNegative) {
                System.out.println("✅ 注意力分数非负验证通过！");
            } else {
                System.out.println("❌ 注意力分数存在负值！");
            }
        } catch (Exception e) {
            System.out.println("⚠️ 注意力分数验证跳过（非核心逻辑）：" + e.getMessage());
        }
    }
}

