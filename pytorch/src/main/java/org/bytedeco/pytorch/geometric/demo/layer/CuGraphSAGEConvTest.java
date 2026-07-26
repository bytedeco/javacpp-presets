package org.bytedeco.pytorch.geometric.demo.layer;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.geometric.nn.conv.CuGraphSAGEConv;

/**
 * CuGraphSAGEConv 最终完整测试用例：
 * 1. 适配 bytedeco-pytorch 的 ParameterDict 遍历规范（StringTensorDict）
 * 2. 覆盖 sum/mean/max 聚合方式、归一化、孤立节点、内存释放等所有场景
 * 3. 所有 API 调用 100% 适配 bytedeco-pytorch 实际规范
 */
public class CuGraphSAGEConvTest {
    public static void main(String[] args) {
        // ========== 0. 定义数据类型（适配 bytedeco-pytorch 规范） ==========
        TensorOptions float32 = torch.dtype(torch.ScalarType.Float);
        TensorOptions int64 = torch.dtype(torch.ScalarType.Long);

        // ========== 1. 初始化卷积层核心参数 ==========
        System.out.println("===== 测试1：卷积层参数初始化 =====");
        long inChannels = 8;     // 输入特征维度
        long outChannels = 4;    // 输出特征维度
        boolean normalize = true;// 启用 L2 归一化（验证修正后的 norm API）
        boolean rootWeight = true;// 使用根节点权重
        boolean hasBias = true;  // 使用偏置
        String[] aggrTypes = {"sum", "mean", "max"}; // 测试所有聚合方式

        // ========== 2. 构造CSC格式测试数据（严格适配 dtype API） ==========
        System.out.println("\n===== 测试2：CSC格式数据构造 =====");
        long N = 4; // 4个节点（含1个孤立节点）
        // CSC列指针：[0,2,3,5,5] → 节点0:2条边，节点1:1条边，节点2:2条边，节点3:0条边（孤立）
        long[] colptrData = {0, 2, 3, 5, 5};
        Tensor colptr = torch.tensor(colptrData, int64);

        // CSC行索引（源节点）：[1,2,0,1,3] → 边：0←1,0←2; 1←0; 2←1,2←3
        long[] rowData = {1, 2, 0, 1, 3};
        Tensor row = torch.tensor(rowData, int64);

        // 节点特征：[4,8]（随机初始化，适配 float32）
        Tensor x = torch.randn(new long[]{N, inChannels}, float32);

        // 打印基础数据信息
        System.out.println("节点数 N：" + N);
        System.out.println("边数 E：" + row.size(0)); // 预期5
        System.out.println("colptr 张量：" + colptr);
        System.out.println("row 张量：" + row);
        System.out.println("节点特征 x 维度：" + x.size(0) + " x " + x.size(1));

        // ========== 3. 遍历所有聚合方式测试核心功能 ==========
        for (String aggr : aggrTypes) {
            System.out.println("\n===== 测试3：" + aggr + " 聚合方式校验 =====");
            // 创建卷积层实例
            CuGraphSAGEConv conv = new CuGraphSAGEConv(
                    inChannels, outChannels, aggr,
                    normalize, rootWeight, hasBias
            );

            try {
                // 前向传播（核心逻辑验证）
                Tensor output = conv.forward(x, row, colptr);

                // 维度校验
                System.out.println("输入特征维度：" + x.size(0) + " x " + x.size(1)); // 4x8
                System.out.println("输出特征维度：" + output.size(0) + " x " + output.size(1)); // 4x4
                if (output.size(0) == N && output.size(1) == outChannels) {
                    System.out.println("✅ " + aggr + " 聚合 - 维度校验通过");
                } else {
                    System.out.println("❌ " + aggr + " 聚合 - 维度校验失败（预期：" + N + "x" + outChannels + "）");
                }

                // 数值稳定性校验（无nan/inf）
                boolean hasNan = output.isnan().any().item().toBool();
                boolean hasInf = output.isinf().any().item().toBool();
                if (!hasNan && !hasInf) {
                    System.out.println("✅ " + aggr + " 聚合 - 数值稳定性校验通过（无nan/inf）");
                } else {
                    System.out.println("❌ " + aggr + " 聚合 - 数值稳定性校验失败（nan：" + hasNan + "，inf：" + hasInf + "）");
                }

                // 归一化校验（启用normalize时，L2范数≈1）
                if (normalize) {
//                    LongPointer normDim = new LongPointer(new long[]{-1});
                    Tensor norm = output.norm(new ScalarOptional(new Scalar(2)), new long[]{-1}, true); // 验证修正后的norm API
                    // 校验范数是否接近1（允许微小浮点误差）
                    boolean normValid = norm.sub(new Scalar(1.0)).abs().max().item_float() < 1e-5;
                    if (normValid) {
                        System.out.println("✅ " + aggr + " 聚合 - 归一化校验通过（L2范数≈1）");
                    } else {
                        System.out.println("❌ " + aggr + " 聚合 - 归一化校验失败（L2范数：" + norm + "）");
                    }
                    norm.close(); // 释放临时张量
                }

                // 释放当前聚合方式的资源
                output.close();
                conv.close();
            } catch (Exception e) {
                System.out.println("❌ " + aggr + " 聚合 - 运行异常：" + e.getMessage());
                e.printStackTrace();
            }
        }

        // ========== 4. 边界场景测试：孤立节点 ==========
        System.out.println("\n===== 测试4：孤立节点校验 =====");
        CuGraphSAGEConv convIsolated = new CuGraphSAGEConv(
                inChannels, outChannels, "mean", normalize, rootWeight, hasBias
        );
        try {
            Tensor outputIsolated = convIsolated.forward(x, row, colptr);
            // 提取孤立节点（节点3）的输出
            Tensor isolatedIdx = torch.tensor(new long[]{3}, int64);
            Tensor isolatedNodeOut = outputIsolated.index_select(0, isolatedIdx);
            System.out.println("孤立节点（节点3）输出值：" + isolatedNodeOut);
            // 验证孤立节点输出无异常
            if (!isolatedNodeOut.isnan().any().item().toBool() && !isolatedNodeOut.isinf().any().item().toBool()) {
                System.out.println("✅ 孤立节点校验通过（输出无异常）");
            } else {
                System.out.println("❌ 孤立节点校验失败（输出含nan/inf）");
            }
            // 释放临时张量
            isolatedIdx.close();
            isolatedNodeOut.close();
            outputIsolated.close();
            convIsolated.close();
        } catch (Exception e) {
            System.out.println("❌ 孤立节点校验异常：" + e.getMessage());
        }

        // ========== 5. 可训练参数校验（核心：适配 StringTensorDict 遍历） ==========
        System.out.println("\n===== 测试5：可训练参数遍历校验 =====");
        CuGraphSAGEConv convParams = new CuGraphSAGEConv(
                inChannels, outChannels, "sum", normalize, rootWeight, hasBias
        );
        // 适配 bytedeco-pytorch 的 ParameterDict 遍历规范
        StringTensorDict params = convParams.named_parameters();
        System.out.println("可训练参数列表：size " + params.size());
        var dictBegin = params.begin();
        var dictEnd = params.end();
        // 遍历所有参数（严格适配迭代器规范）
        while (!dictBegin.equals(dictEnd)) {
            var entry = dictBegin.get();
            String paramName = entry.key().getString();
            Tensor param = entry.access();
            // 打印参数名称和维度（适配多维度张量）
            StringBuilder dimStr = new StringBuilder();
            for (int i = 0; i < param.data().dim(); i++) {
                if (i > 0) dimStr.append("x");
                dimStr.append(param.data().size(i));
            }
            System.out.println("- " + paramName + "：维度 " + dimStr);
            dictBegin.increment(); // 迭代器自增
        }
        // 校验核心参数是否存在
        boolean hasLinLWeight = params.contains("lin_l.weight");
        boolean hasLinRWeight = params.contains("lin_r.weight");
        boolean hasBiass = params.contains("bias");
        if (hasLinLWeight && hasLinRWeight && hasBiass) {
            System.out.println("✅ 可训练参数校验通过（核心参数均存在）");
        } else {
            System.out.println("❌ 可训练参数校验失败（lin_l.weight：" + hasLinLWeight +
                    "，lin_r.weight：" + hasLinRWeight + "，bias：" + hasBiass + "）");
        }
        convParams.close();

        // ========== 6. 内存释放最终校验 ==========
        System.out.println("\n===== 测试6：内存释放校验 =====");
        try {
            // 释放所有基础数据张量
            x.close();
            row.close();
            colptr.close();
            // 验证已释放的张量访问会报错（内存释放成功）
            x.size(0); // 已释放，触发异常
            System.out.println("❌ 内存释放校验失败（张量未真正释放）");
        } catch (Exception e) {
            System.out.println("✅ 内存释放校验通过（访问已释放张量触发异常）");
        }

        System.out.println("\n===== 所有测试完成 =====");
    }
}