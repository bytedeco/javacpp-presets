package org.bytedeco.pytorch.geometric.nn.conv;
import org.bytedeco.pytorch.nn.modules.*;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.geometric.nn.Parameter;

/**
 * 修正版 FastRGCNConv（高性能 RGCN 卷积）
 * 核心优化：
 * 1. 替换原地操作，避免叶子张量梯度报错
 * 2. 完善张量初始化配置（dtype/device 对齐）
 * 3. 修复维度匹配错误，补充边界校验
 * 4. 增加资源释放逻辑，避免内存泄漏
 */
public class FastRGCNConv extends Module {
    public Parameter weight; // [numRelations, inChannels, outChannels]（改为 Parameter 管理梯度）
    public LinearImpl linRoot;
    public Parameter bias;
    private long numRelations;
    private long inChannels;
    private long outChannels;

    public FastRGCNConv(long inChannels, long outChannels, long numRelations,
                        boolean rootWeight, boolean hasBias) {
        super();
        this.inChannels = inChannels;
        this.outChannels = outChannels;
        this.numRelations = numRelations;

        // ========== 1. 权重初始化（补充 dtype/device，改为 Parameter） ==========
        TensorOptions paramOpts = new TensorOptions()
                .dtype(new ScalarTypeOptional(torch.ScalarType.Float))
                .device(new DeviceOptional(new Device(torch.kCPU())));
//                .requires_grad(new BoolOptional(true));

        // 初始化权重张量并应用 Xavier 初始化（非原地操作）
        Tensor weightTensor = torch.empty( new long[]{numRelations, inChannels, outChannels}, paramOpts,new MemoryFormatOptional());
        weightTensor = torch.xavier_uniform_(weightTensor); // 替换原地操作 xavier_uniform_
        this.weight = new Parameter(weightTensor);
        register_parameter("weight", this.weight);

        // ========== 2. 根节点线性层 ==========
        if (rootWeight) {
            this.linRoot = new LinearImpl(inChannels, outChannels);
            // 初始化线性层参数（确保 dtype/device 一致）
            Tensor linWeight = torch.randn(new long[]{outChannels, inChannels}, paramOpts);
            Parameter linWeightParam = new Parameter(linWeight);
            this.linRoot.weight(linWeightParam);

            if (this.linRoot.bias() != null) {
                Tensor linBias = torch.zeros(new long[]{outChannels}, paramOpts);
                Parameter linBiasParam = new Parameter(linBias);
                this.linRoot.bias(linBiasParam);
            }
            register_module("lin_root", linRoot);
        }

        // ========== 3. 偏置初始化 ==========
        if (hasBias) {
            Tensor biasTensor = torch.zeros(new long[]{outChannels}, paramOpts);
            this.bias = new Parameter(biasTensor);
            register_parameter("bias", this.bias);
        }
    }

    /**
     * 基础前向传播（批量矩阵乘法版）
     * @param x          节点特征 [N, inChannels]
     * @param edge_index 边索引 [2, E]
     * @param edge_type  关系类型 [E]
     * @return 输出特征 [N, outChannels]
     */
    public Tensor forward(Tensor x, Tensor edge_index, Tensor edge_type) {
        // ========== 输入校验 ==========
        long[] xShape = x.sizes().vec().get();
        if (xShape.length != 2 || xShape[1] != inChannels) {
            throw new IllegalArgumentException(
                    "节点特征 x 形状必须为 [N, " + inChannels + "]，当前：" + shapeToString(xShape));
        }
        long N = xShape[0];

        long[] edgeIndexShape = edge_index.sizes().vec().get();
        if (edgeIndexShape.length != 2 || edgeIndexShape[0] != 2) {
            throw new IllegalArgumentException(
                    "边索引 edge_index 形状必须为 [2, E]，当前：" + shapeToString(edgeIndexShape));
        }
        long E = edgeIndexShape[1];

        long[] edgeTypeShape = edge_type.sizes().vec().get();
        if (edgeTypeShape.length != 1 || edgeTypeShape[0] != E) {
            throw new IllegalArgumentException(
                    "关系类型 edge_type 形状必须为 [E]，当前：" + shapeToString(edgeTypeShape));
        }

        // ========== 1. 提取源/目标节点索引 ==========
        Tensor srcIdx = edge_index.select(0, 0); // [E]
        Tensor dstIdx = edge_index.select(0, 1); // [E]

        // ========== 2. 处理空边场景 ==========
        if (E == 0) {
            Tensor res = torch.zeros(new long[]{N, outChannels},
                    new TensorOptions().dtype(new ScalarTypeOptional(x.options().dtype())).device(new DeviceOptional(x.options().device())));
            if (linRoot != null) {
                res = res.add(linRoot.forward(x));
            }
            if (bias != null) {
                res = res.add(bias);
            }
            srcIdx.close();
            dstIdx.close();
            return res;
        }

        // ========== 3. 提取源节点特征和对应关系权重 ==========
        Tensor x_j = x.index_select(0, srcIdx); // [E, inChannels]
        Tensor w = weight.index_select(0, edge_type); // [E, inChannels, outChannels]

        // ========== 4. 批量矩阵乘法（核心） ==========
        // x_j.unsqueeze(1) → [E, 1, inChannels]
        // w → [E, inChannels, outChannels]
        // matmul → [E, 1, outChannels] → squeeze → [E, outChannels]
        Tensor out = torch.matmul(x_j.unsqueeze(1), w).squeeze(1);

        // ========== 5. 聚合到目标节点（替换原地操作 scatter_add_） ==========
        Tensor res = torch.zeros(new long[]{N, outChannels}, x.options());
        res = res.scatter_add(0, dstIdx.unsqueeze(-1).expand_as(out), out); // 非原地操作

        // ========== 6. 根节点处理 ==========
        if (linRoot != null) {
            res = res.add(linRoot.forward(x));
        }

        // ========== 7. 加偏置 ==========
        if (bias != null) {
            res = res.add(bias);
        }

        // ========== 资源释放 ==========
        srcIdx.close();
        dstIdx.close();
        x_j.close();
        w.close();
        out.close();

        return res;
    }

    /**
     * 优化版前向传播（预计算所有关系投影）
     * 适合关系类型少、节点数多的场景
     * @param x          节点特征 [N, inChannels]
     * @param edge_index 边索引 [2, E]
     * @param edge_type  关系类型 [E]
     * @return 输出特征 [N, outChannels]
     */
    public Tensor forward2(Tensor x, Tensor edge_index, Tensor edge_type) {
        // ========== 输入校验 ==========
        long[] xShape = x.sizes().vec().get();
        if (xShape.length != 2 || xShape[1] != inChannels) {
            throw new IllegalArgumentException(
                    "节点特征 x 形状必须为 [N, " + inChannels + "]，当前：" + shapeToString(xShape));
        }
        long N = xShape[0];

        long[] edgeIndexShape = edge_index.sizes().vec().get();
        if (edgeIndexShape.length != 2 || edgeIndexShape[0] != 2) {
            throw new IllegalArgumentException(
                    "边索引 edge_index 形状必须为 [2, E]，当前：" + shapeToString(edgeIndexShape));
        }
        long E = edgeIndexShape[1];

        long[] edgeTypeShape = edge_type.sizes().vec().get();
        if (edgeTypeShape.length != 1 || edgeTypeShape[0] != E) {
            throw new IllegalArgumentException(
                    "关系类型 edge_type 形状必须为 [E]，当前：" + shapeToString(edgeTypeShape));
        }

        // ========== 1. 处理空边场景 ==========
        if (E == 0) {
            Tensor out = torch.zeros(new long[]{N, outChannels}, x.options());
            if (linRoot != null) {
                out = out.add(linRoot.forward(x));
            }
            if (bias != null) {
                out = out.add(bias);
            }
            return out;
        }

        // ========== 2. 预计算所有关系的投影结果 ==========
        Tensor x_expanded = x.unsqueeze(1); // [N, 1, inChannels]
        Tensor out_all_rel = torch.matmul(x_expanded, weight); // [N, numRelations, outChannels]

        // ========== 3. 提取源/目标节点索引 ==========
        Tensor sourceIdx = edge_index.select(0, 0); // [E]
        Tensor targetIdx = edge_index.select(0, 1); // [E]

        // ========== 4. 提取对应关系的消息（修复 gather 维度） ==========
        Tensor msg = out_all_rel.index_select(0, sourceIdx); // [E, numRelations, outChannels]

        // 构造 gather 索引：[E, 1, outChannels]（修复维度不匹配）
        Tensor rel_idx = edge_type.view(-1, 1, 1)
                .expand(new long[]{E, 1, outChannels}); // [E, 1, outChannels]
        msg = torch.gather(msg, 1, rel_idx).squeeze(1); // [E, outChannels]

        // ========== 5. 聚合消息到目标节点（非原地操作） ==========
        Tensor out = torch.zeros(new long[]{N, outChannels}, x.options());
        out = out.scatter_add(0, targetIdx.view(-1, 1).expand(new long[]{E, outChannels}), msg);

        // ========== 6. 根节点 + 偏置 ==========
        if (linRoot != null) {
            out = out.add(linRoot.forward(x));
        }
        if (bias != null) {
            out = out.add(bias);
        }

        // ========== 资源释放 ==========
        x_expanded.close();
        out_all_rel.close();
        sourceIdx.close();
        targetIdx.close();
        rel_idx.close();
        msg.close();

        return out;
    }

    // ========== 工具方法：形状转字符串 ==========
    private String shapeToString(long[] shape) {
        StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < shape.length; i++) {
            if (i > 0) sb.append(", ");
            sb.append(shape[i]);
        }
        sb.append("]");
        return sb.toString();
    }

    // ========== 资源释放 ==========
//    @Override
//    protected void finalize() throws Throwable {
//        try {
//            if (weight != null) weight.close();
//            if (linRoot != null) linRoot.close();
//            if (bias != null) bias.close();
//        } finally {
//            super.finalize();
//        }
//    }
}

//package org.bytedeco.pytorch.geometric.nn.conv;
//
//import org.bytedeco.pytorch.*;
//import org.bytedeco.pytorch.nn.Module;
//import org.bytedeco.pytorch.global.torch;
//
///**
// * 实现 torch_geometric.nn.conv.FastRGCNConv
// * 高性能版 RGCNConv，通过并行化处理所有关系类型来提升 GPU 利用率。
// */
//public class FastRGCNConv extends Module {
//    private Tensor weight; // [numRelations, inChannels, outChannels]
//    private LinearImpl linRoot;
//    private Tensor bias;
//    private int numRelations;
//    private long inChannels;
//    private long outChannels;
//
//    public FastRGCNConv(long inChannels, long outChannels, int numRelations, boolean rootWeight, boolean hasBias) {
//        super();
//        this.inChannels = inChannels;
//        this.outChannels = outChannels;
//        this.numRelations = numRelations;
//
//        // 1. 将所有关系的权重打包成一个 Tensor，方便批量矩阵乘法
//        this.weight = torch.empty(new long[]{numRelations, inChannels, outChannels});
//        torch.xavier_uniform_(this.weight);
//        register_parameter("weight", weight);
//
//        if (rootWeight) {
//            this.linRoot = new LinearImpl(inChannels, outChannels);
//            register_module("lin_root", linRoot);
//        }
//
//        if (hasBias) {
//            this.bias = torch.zeros(new long[]{outChannels});
//            register_parameter("bias", bias);
//        }
//    }
//
//    /**
//     * @param x          节点特征 [N, inChannels]
//     * @param edge_index 边索引 [2, E]
//     * @param edge_type  关系类型 [E]
//     */
//    public Tensor forward(Tensor x, Tensor edge_index, Tensor edge_type) {
//        long E = edge_index.size(1);
//        long N = x.size(0);
//
//        Tensor srcIdx = edge_index.select(0, 0);
//        Tensor dstIdx = edge_index.select(0, 1);
//
//        // 1. 获取源节点特征 [E, inChannels]
//        Tensor x_j = x.index_select(0, srcIdx);
//
//        // 2. 核心修复：根据关系类型提取对应的权重矩阵 [E, inC, outC]
//        // weight 形状应为 [numRelations, inChannels, outChannels]
//        // edge_type 形状为 [E]
//        Tensor w = weight.index_select(0, edge_type);
//
//        // 3. 执行批量矩阵乘法 (BMM)
//        // x_j.unsqueeze(1) -> [E, 1, inC]
//        // w -> [E, inC, outC]
//        // 结果 [E, 1, outC] -> squeeze -> [E, outC]
//        Tensor out = torch.matmul(x_j.unsqueeze(1), w).squeeze(1);
//
//        // 4. 聚合到目标节点
//        Tensor res = torch.zeros(new long[]{N, outChannels}, x.options());
//        res.scatter_add_(0, dstIdx.unsqueeze(-1).expand_as(out), out);
//
//        // 5. 根节点处理
//        if (linRoot != null) {
//            res = res.add(linRoot.forward(x));
//        }
//
//        return res;
//    }
//    public Tensor forward2(Tensor x, Tensor edge_index, Tensor edge_type) {
//        long N = x.size(0);
//        long E = edge_index.size(1);
//
//        // 1. 预计算所有可能的投影结果
//        // x: [N, In], weight: [R, In, Out]
//        // 结果: [N, R, Out] (注意：这里在 N 大、R 多时会非常耗显存)
//        Tensor x_expanded = x.unsqueeze(1); // [N, 1, In]
//        Tensor out_all_rel = x_expanded.matmul(weight); // [N, R, Out]
//
//        // 2. 消息传递：利用 edge_type 和 edge_index[0] (source) 提取对应关系的消息
//        Tensor sourceIdx = edge_index.select(0, 0);
//        Tensor targetIdx = edge_index.select(0, 1);
//
//        // 构造索引：我们需要从 out_all_rel 中提取 (sourceIdx, edge_type) 对应的特征
//        // 这一步模拟了消息提取逻辑
//        Tensor msg = out_all_rel.index_select(0, sourceIdx); // [E, R, Out]
//
//        // 利用 edge_type 进一步筛选特定的关系维度 [E, Out]
//        // 这里在 JavaCPP 中通过 gather 操作实现
//        Tensor rel_idx = edge_type.view(-1, 1, 1).expand(new long[]{E, 1, outChannels});
//        msg = msg.gather(1, rel_idx).squeeze(1); // [E, Out]
//
//        // 3. 聚合消息到目标节点
//        Tensor out = torch.zeros(new long[]{N, outChannels}, x.options());
//        out.scatter_add_(0, targetIdx.view(-1, 1).expand(new long[]{E, outChannels}), msg);
//
//        // 加上根节点权重
//        if (linRoot != null) {
//            out = out.add(linRoot.forward(x));
//        }
//
//        if (bias != null) {
//            out = out.add(bias);
//        }
//
//        return out;
//    }
//}