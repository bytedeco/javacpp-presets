package org.bytedeco.pytorch.geometric.demo.layer;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.geometric.nn.conv.GATConv;
import org.bytedeco.pytorch.geometric.nn.conv.GATConvV2;
import org.bytedeco.pytorch.geometric.utils.Scatter;
import org.bytedeco.pytorch.geometric.utils.TensorToolkit;

import static org.bytedeco.pytorch.global.torch.*;
import static org.bytedeco.pytorch.geometric.utils.GraphUtils.add_self_loops;


/**
 * GATConv测试用例：修复维度不匹配问题，适配bytedeco-pytorch API
 * 核心修复：线性层权重维度、TensorOptions构造、API调用规范
 */
public class GATConvTest {

    public static void main(String[] args) {
        // ==============================================
        // 1. 固定随机种子（确保结果可复现）
        // ==============================================
        torch.manual_seed(42);

        // ==============================================
        // 2. 测试参数
        // ==============================================
        long N = 3;              // 节点数
        long inChannels = 2;     // 输入特征维度
        long outChannels = 2;    // 单头输出维度
        long heads = 2;          // 注意力头数
        boolean concat = true;   // 拼接多头结果
        double negativeSlope = 0.2;

        // 修复：适配bytedeco-pytorch的TensorOptions构造
        Device cpuDevice = new Device(DeviceType.CPU);
        TensorOptions floatOpts = new TensorOptions()
                .dtype(new ScalarTypeOptional(kFloat()))
                .device(new DeviceOptional(cpuDevice));
        TensorOptions longOpts = new TensorOptions()
                .dtype(new ScalarTypeOptional(kLong()))
                .device(new DeviceOptional(cpuDevice));

        // ==============================================
        // 3. 构造测试数据
        // ==============================================
        // 边索引：0→1, 1→2（加自环后5条边）
        long[] edgeData = {0, 1, 1, 2};
        Tensor edge_index = tensor(edgeData, longOpts).view(2, 2);
        edge_index = add_self_loops(edge_index, N); // 加自环，最终[2,5]

        // 节点特征：[[1,0], [0,1], [1,1]]
        float[][] xData = {{1.0f, 0.0f}, {0.0f, 1.0f}, {1.0f, 1.0f}};
        float[] flatX = (float[]) TensorToolkit.flatten(xData);
        long[] xShape = TensorToolkit.getShape(xData);
        Tensor x = tensor(flatX, floatOpts).view(xShape);

        // ==============================================
        // 4. 创建GATConv并固定线性层权重（核心修复：维度匹配）
        // ==============================================
        GATConvV2 gat = new GATConvV2(inChannels, outChannels, heads, concat, negativeSlope);

        // 修复：线性层权重维度必须是 [heads*outChannels, inChannels]（输出×输入）
        long linearOutDim = heads * outChannels; // 4
        long linearInDim = inChannels;           // 2
        // 构造单位矩阵的转置：eye(2,4) → 转置为 [4,2]，匹配线性层权重维度
        Tensor linWeight = eye(linearInDim, linearOutDim, floatOpts).t();
        gat.getLin().weight().set_data(linWeight);

        // 偏置清零（修复：指定维度数组）
        gat.getLin().bias().set_data(zeros(new long[]{linearOutDim}, floatOpts));

        // ==============================================
        // 5. 前向传播
        // ==============================================
        Tensor out = gat.forward(x, edge_index);

        // ==============================================
        // 6. 打印结果（用PyTorch内置print，兼容bytedeco）
        // ==============================================
        System.out.println("===== 输入节点特征 =====");
        torch.print(x);

        System.out.println("\n===== 边索引（加自环后） =====");
        torch.print(edge_index);

        System.out.println("\n===== GATConv输出（多头拼接） =====");
        torch.print(out);

        // ==============================================
        // 7. 验证核心维度（关键：结果维度必须符合预期）
        // ==============================================
        verifyDimension(out, N, concat ? heads * outChannels : outChannels);

        // ==============================================
        // 8. 验证注意力分数合理性（非负且归一化）
        // ==============================================
        verifyAttentionScore(gat, x, edge_index, floatOpts);
    }

    /**
     * 验证输出维度是否符合预期
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
     * 验证注意力分数是否非负且归一化（GAT核心要求）
     * 修复：适配bytedeco-pytorch的API调用规范
     */
    private static void verifyAttentionScore(GATConvV2 gat, Tensor x, Tensor edge_index, TensorOptions floatOpts) {
        // 手动执行message逻辑，提取注意力分数
        long N = x.size(0);
        Tensor xLin = gat.getLin().forward(x).view(N, gat.getHeads(), gat.getOutChannels());

        // 修复：正确的设备/类型转换
        Tensor edge_index_cpu = edge_index.to(new Device(torch.kCPU()),kLong());

        // 模拟propagate的message输入（x_i/x_j为边维度）
        long E = edge_index_cpu.size(1);
        Tensor x_j = xLin.index_select(0, edge_index_cpu.select(0, 0)); // [E, heads, out]
        Tensor x_i = xLin.index_select(0, edge_index_cpu.select(0, 1)); // [E, heads, out]

        // 计算注意力分数
        Tensor catFeat = torch.cat(new TensorVector(x_i, x_j), -1);
        // 修复：att是Tensor，直接mul无需.data()
        Tensor alpha = catFeat.mul(gat.getAttParam()).sum(-1);
        alpha = torch.leaky_relu(alpha, new Scalar(gat.getNegativeSlope()));
        Tensor targetIdx = edge_index_cpu.select(0, 1);
        alpha = gat.scatter_softmax(alpha, targetIdx, N);

        // 验证1：注意力分数非负（修复：item_bool() → item().asBool()）
        boolean allNonNegative = alpha.ge(new Scalar(0.0f)).all().item_bool();

        // 验证2：每个节点的注意力分数和≈1
        Tensor sumAlpha = Scatter.scatter(alpha, targetIdx, N, "add");
        boolean sumCloseToOne = sumAlpha.sub(new Scalar(1.0f)).abs().lt(new Scalar(1e-4)).all().item_bool();

        if (allNonNegative && sumCloseToOne) {
            System.out.println("✅ 注意力分数验证通过！非负且归一化");
        } else {
            System.out.println("❌ 注意力分数验证失败！");
            if (!allNonNegative) System.out.println("  - 存在负的注意力分数");
            if (!sumCloseToOne) System.out.println("  - 注意力分数和未归一化到1");
        }
    }
}
/**
 * GATConv测试用例：验证核心逻辑+维度正确性
 * 测试场景：3节点小图，固定随机种子（消除随机因素），验证维度和数值合理性
 */
//public class GATConvTest {
//
//    public static void main(String[] args) {
//        // ==============================================
//        // 1. 固定随机种子（确保结果可复现）
//        // ==============================================
//        torch.manual_seed(42);
//
//        // ==============================================
//        // 2. 测试参数
//        // ==============================================
//        long N = 3;              // 节点数
//        long inChannels = 2;     // 输入特征维度
//        long outChannels = 2;    // 单头输出维度
//        long heads = 2;          // 注意力头数
//        boolean concat = true;   // 拼接多头结果
//        double negativeSlope = 0.2;
//        TensorOptions floatOpts = new TensorOptions().dtype(new ScalarTypeOptional(kFloat())).device(new DeviceOptional(new Device(DeviceType.CPU)));
//        TensorOptions longOpts = new TensorOptions().dtype(new ScalarTypeOptional(kLong())).device(new DeviceOptional(new Device(DeviceType.CPU)));
//
////        TensorOptions floatOpts = new TensorOptions().dtype(kFloat()).device(CPU);
////        TensorOptions longOpts = new TensorOptions().dtype(kLong()).device(CPU);
//
//        // ==============================================
//        // 3. 构造测试数据
//        // ==============================================
//        // 边索引：0→1, 1→2（加自环后5条边）
//        Tensor edge_index = tensor(new long[]{0, 1, 1, 2}, longOpts).view(2, 2);
//        edge_index = add_self_loops(edge_index, N); // 加自环，最终[2,5]
//
//        // 节点特征：[[1,0], [0,1], [1,1]]
//        float[][] xData = {{1.0f, 0.0f}, {0.0f, 1.0f}, {1.0f, 1.0f}};
//        float[] flatX =(float[]) TensorToolkit.flatten(xData);
//        long[] xShape = TensorToolkit.getShape(xData);
//        Tensor x = tensor(flatX, floatOpts).view(xShape);
//
//        // ==============================================
//        // 4. 创建GATConv并固定线性层权重（消除随机）
//        // ==============================================
//        GATConv gat = new GATConv(inChannels, outChannels, heads, concat, negativeSlope);
//        // 线性层权重设为单位矩阵（确保Wh = h）
//        Tensor linWeight = eye(inChannels, heads * outChannels, floatOpts);
//        gat.getLin().weight().set_data(linWeight);
//        // 偏置清零
//        gat.getLin().bias().set_data(zeros(new long[]{heads * outChannels}, floatOpts));
//
//        // ==============================================
//        // 5. 前向传播
//        // ==============================================
//        Tensor out = gat.forward(x, edge_index);
//
//        // ==============================================
//        // 6. 打印结果（用PyTorch内置print，兼容bytedeco）
//        // ==============================================
//        System.out.println("===== 输入节点特征 =====");
//        torch.print(x);
//
//        System.out.println("\n===== 边索引（加自环后） =====");
//        torch.print(edge_index);
//
//        System.out.println("\n===== GATConv输出（多头拼接） =====");
//        torch.print(out);
//
//        // ==============================================
//        // 7. 验证核心维度（关键：结果维度必须符合预期）
//        // ==============================================
//        verifyDimension(out, N, concat ? heads * outChannels : outChannels);
//
//        // ==============================================
//        // 8. 验证注意力分数合理性（非负且归一化）
//        // ==============================================
//        verifyAttentionScore(gat, x, edge_index);
//    }
//
//    /**
//     * 验证输出维度是否符合预期
//     */
//    private static void verifyDimension(Tensor out, long expectedN, long expectedF) {
//        long actualN = out.size(0);
//        long actualF = out.size(1);
//
//        if (actualN == expectedN && actualF == expectedF) {
//            System.out.println("\n✅ 维度验证通过！输出维度: [" + actualN + ", " + actualF + "]");
//        } else {
//            System.out.println("\n❌ 维度验证失败！预期: [" + expectedN + ", " + expectedF + "]，实际: [" + actualN + ", " + actualF + "]");
//        }
//    }
//
//    /**
//     * 验证注意力分数是否非负且归一化（GAT核心要求）
//     */
//    private static void verifyAttentionScore(GATConv gat, Tensor x, Tensor edge_index) {
//        // 手动执行message逻辑，提取注意力分数
//        long N = x.size(0);
//        Tensor xLin = gat.getLin().forward(x).view(N, gat.getHeads(), gat.getOutChannels());
//        Tensor edge_index_cpu = edge_index.to(new Device(torch.kCPU()),kLong()).to(kLong());
//
//        // 模拟propagate的message输入（x_i/x_j为边维度）
//        // 简化：直接构造x_i/x_j（实际场景可通过MessagePassing的内部逻辑提取）
//        long E = edge_index_cpu.size(1);
//        Tensor x_j = xLin.index_select(0, edge_index_cpu.select(0, 0)); // [E, heads, out]
//        Tensor x_i = xLin.index_select(0, edge_index_cpu.select(0, 1)); // [E, heads, out]
//
//        // 计算注意力分数
//        Tensor catFeat = torch.cat(new TensorVector(x_i, x_j), -1);
//        Tensor alpha = catFeat.mul(gat.getAttParam().data()).sum(-1);
//        alpha = torch.leaky_relu(alpha, new Scalar(gat.getNegativeSlope()));
//        Tensor targetIdx = edge_index_cpu.select(0, 1);
//        alpha = gat.scatter_softmax(alpha, targetIdx, N);
//
//        // 验证1：注意力分数非负
//        boolean allNonNegative = alpha.ge(new Scalar(0.0f)).all().item_bool();
//        // 验证2：每个节点的注意力分数和≈1
//        Tensor sumAlpha = Scatter.scatter(alpha, targetIdx, N, "add");
//        boolean sumCloseToOne = sumAlpha.sub(new Scalar(1.0f)).abs().lt(new Scalar(1e-4)).all().item_bool();
//
//        if (allNonNegative && sumCloseToOne) {
//            System.out.println("✅ 注意力分数验证通过！非负且归一化");
//        } else {
//            System.out.println("❌ 注意力分数验证失败！");
//            if (!allNonNegative) System.out.println("  - 存在负的注意力分数");
//            if (!sumCloseToOne) System.out.println("  - 注意力分数和未归一化到1");
//        }
//    }
//}