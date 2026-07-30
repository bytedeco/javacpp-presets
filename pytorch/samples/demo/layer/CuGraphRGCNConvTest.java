package samples.demo.layer;

import org.bytedeco.pytorch.StringTensorDict;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorOptions;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.geometric.nn.conv.CuGraphRGCNConv;

/**
 * CuGraphRGCNConv 完整测试用例：
 * 1. 适配 bytedeco-pytorch API（替换 torch.int64()/float32() 为 dtype(ScalarType)）
 * 2. 覆盖 sum/mean/max 三种聚合方式
 * 3. 验证 CSC 索引、孤立节点、空关系等边界场景
 * 4. 校验内存释放和数值稳定性
 */
public class CuGraphRGCNConvTest {
    public static void main(String[] args) {
        // ========== 0. 定义数据类型（适配 bytedeco-pytorch API） ==========
        TensorOptions float32 = torch.dtype(torch.ScalarType.Float);
        TensorOptions int64 = torch.dtype(torch.ScalarType.Long);

        // ========== 1. 初始化卷积层参数 ==========
        System.out.println("===== 测试1：卷积层参数初始化 =====");
        long inChannels = 8;    // 输入特征维度
        long outChannels = 4;   // 输出特征维度
        int numRelations = 2;   // 关系类型数量
        boolean rootWeight = true; // 是否使用根节点权重
        boolean hasBias = true;    // 是否使用偏置
        String[] aggrTypes = {"sum", "mean", "max"}; // 测试所有聚合方式

        // ========== 2. 构造CSC格式测试数据（核心：适配 dtype API） ==========
        System.out.println("\n===== 测试2：CSC格式数据构造 =====");
        long N = 4; // 4个节点
        // CSC列指针：[0,2,3,5,5] → 节点0:2条边，节点1:1条边，节点2:2条边，节点3:0条边（孤立）
        long[] colptrData = {0, 2, 3, 5, 5};
        Tensor colptr = torch.tensor(colptrData, int64); // 替换 torch.int64() 为 int64

        // CSC行索引（源节点）：[1,2,0,1,3] → 边：0←1,0←2; 1←0; 2←1,2←3
        long[] rowData = {1, 2, 0, 1, 3};
        Tensor row = torch.tensor(rowData, int64);

        // 边关系类型：[0,1,0,1,0] → 边0:关系0，边1:关系1，边2:关系0，边3:关系1，边4:关系0
        long[] edgeTypeData = {0, 1, 0, 1, 0};
        Tensor edgeType = torch.tensor(edgeTypeData, int64);

        // 节点特征：[4,8]（替换 torch.float32() 为 float32）
        Tensor x = torch.randn(new long[]{N, inChannels}, float32);

        // 打印数据信息
        System.out.println("节点数 N：" + N);
        System.out.println("边数 E：" + row.size(0)); // 5
        System.out.println("关系类型数 R：" + numRelations); // 2
        System.out.println("colptr 张量：" + colptr);
        System.out.println("row 张量：" + row);
        System.out.println("edge_type 张量：" + edgeType);
        System.out.println("节点特征 x 维度：" + x.size(0) + " x " + x.size(1));

        // ========== 3. 遍历所有聚合方式测试 ==========
        for (String aggr : aggrTypes) {
            System.out.println("\n===== 测试3：" + aggr + " 聚合方式校验 =====");
            // 创建卷积层实例
            CuGraphRGCNConv conv = new CuGraphRGCNConv(
                    inChannels, outChannels, numRelations,
                    rootWeight, hasBias, aggr
            );

            try {
                // 前向传播
                Tensor output = conv.forward(x, row, colptr, edgeType);

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
        CuGraphRGCNConv convIsolated = new CuGraphRGCNConv(
                inChannels, outChannels, numRelations, rootWeight, hasBias, "sum"
        );
        try {
            Tensor outputIsolated = convIsolated.forward(x, row, colptr, edgeType);
            // 提取孤立节点（节点3）的输出
            Tensor isolatedNodeOut = outputIsolated.index_select(0, torch.tensor(new long[]{3}, int64));
            System.out.println("孤立节点（节点3）输出值：" + isolatedNodeOut);
            // 验证孤立节点输出无异常
            if (!isolatedNodeOut.isnan().any().item().toBool() && !isolatedNodeOut.isinf().any().item().toBool()) {
                System.out.println("✅ 孤立节点校验通过（输出无异常）");
            } else {
                System.out.println("❌ 孤立节点校验失败（输出含nan/inf）");
            }
            isolatedNodeOut.close();
            outputIsolated.close();
            convIsolated.close();
        } catch (Exception e) {
            System.out.println("❌ 孤立节点校验异常：" + e.getMessage());
        }

        // ========== 5. 边界场景测试：空关系类型 ==========
        System.out.println("\n===== 测试5：空关系类型校验 =====");
        CuGraphRGCNConv convEmptyRel = new CuGraphRGCNConv(
                inChannels, outChannels, numRelations, rootWeight, hasBias, "sum"
        );
        try {
            // 构造空关系数据（仅关系0有边，关系1无边）
            long[] edgeTypeEmptyData = {0, 0, 0, 0, 0};
            Tensor edgeTypeEmpty = torch.tensor(edgeTypeEmptyData, int64);
            // 前向传播
            Tensor outputEmpty = convEmptyRel.forward(x, row, colptr, edgeTypeEmpty);
            System.out.println("✅ 空关系类型校验通过（无报错）");
            outputEmpty.close();
            edgeTypeEmpty.close();
            convEmptyRel.close();
        } catch (Exception e) {
            System.out.println("❌ 空关系类型校验异常：" + e.getMessage());
        }

        // ========== 6. 可训练参数校验 ==========
        System.out.println("\n===== 测试6：可训练参数校验 =====");
        CuGraphRGCNConv convParams = new CuGraphRGCNConv(
                inChannels, outChannels, numRelations, rootWeight, hasBias, "sum"
        );
        StringTensorDict params = convParams.named_parameters();
        System.out.println("可训练参数列表：" + params.size());
        var dictBegin = params.begin();
        var dictEnd = params.end();
        System.out.println("可训练参数列表： size " + params.size());
        while (!dictBegin.equals(dictEnd)) {
            var entry = dictBegin.get();
            String paramName = entry.key().getString();
            Tensor param = entry.access();
            System.out.println("- " + paramName + "：维度 " + param.data().size(0) + " x " + (param.data().dim() > 1 ? param.data().size(1) : 1));
            dictBegin.increment();
        }

        // 校验核心参数是否存在
        boolean hasWeight = params.contains("weight");
        boolean hasLinRootWeight = params.contains("lin_root.weight");
        boolean hasBiass = params.contains("bias");
        if (hasWeight && hasLinRootWeight && hasBiass) {
            System.out.println("✅ 可训练参数校验通过（核心参数均存在）");
        } else {
            System.out.println("❌ 可训练参数校验失败（weight：" + hasWeight + "，lin_root.weight：" + hasLinRootWeight + "，bias：" + hasBias + "）");
        }
        convParams.close();

        // ========== 7. 内存释放最终校验 ==========
        System.out.println("\n===== 测试7：内存释放校验 =====");
        try {
            // 释放所有基础数据张量
            x.close();
            row.close();
            colptr.close();
            edgeType.close();
            // 验证已释放的张量访问会报错
            x.size(0); // 已释放，触发异常
            System.out.println("❌ 内存释放校验失败（张量未真正释放）");
        } catch (Exception e) {
            System.out.println("✅ 内存释放校验通过（访问已释放张量触发异常）");
        }

        System.out.println("\n===== 所有测试完成 =====");
    }

    /**
     * 辅助方法：获取张量维度的字符串描述（适配多维度）
     */
    private static String getTensorDimStr(Tensor tensor) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < tensor.dim(); i++) {
            if (i > 0) sb.append("x");
            sb.append(tensor.size(i));
        }
        return sb.toString();
    }
}
