package org.bytedeco.pytorch.geometric.nn.conv;
import org.bytedeco.pytorch.jit.*;
import org.bytedeco.pytorch.autograd.*;
import org.bytedeco.pytorch.nn.modules.*;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.nn.Parameter;

import java.util.Arrays;

import static org.bytedeco.pytorch.geometric.utils.Scatter.scatter_max;

/**
 * 修正版 CuGraphRGCNConv：
 * 1. 基于 CSC 格式优化的关系图卷积，保证低显存+高性能
 * 2. 修复 CSC 索引、scatter 聚合等核心逻辑
 * 3. 完善内存管理和边界条件处理
 */
public class CuGraphRGCNConv extends Module implements AutoCloseable {
    private final TensorVector tensors = new TensorVector(); // 管理临时Tensor，防止内存泄漏
    private Parameter weight; // [numRelations, inChannels, outChannels]（包装为可训练参数）
    private LinearImpl linRoot; // 根节点线性层
    private Parameter bias; // 偏置参数
    private int numRelations;
    private long inChannels;
    private long outChannels;
    private String aggr; // 聚合方式：sum/mean/max

    /**
     * 构造函数
     *
     * @param inChannels   输入特征维度
     * @param outChannels  输出特征维度
     * @param numRelations 关系类型数量
     * @param rootWeight   是否使用根节点权重
     * @param hasBias      是否使用偏置
     * @param aggr         聚合方式（sum/mean/max）
     */
    public CuGraphRGCNConv(long inChannels, long outChannels, int numRelations,
                           boolean rootWeight, boolean hasBias, String aggr) {
        super();
        this.inChannels = inChannels;
        this.outChannels = outChannels;
        this.numRelations = numRelations;

        // 校验聚合方式合法性
        if (!Arrays.asList("sum", "mean", "max").contains(aggr)) {
            throw new IllegalArgumentException("仅支持 sum/mean/max 聚合方式，当前：" + aggr);
        }
        this.aggr = aggr;

        // 1. 初始化关系权重 [R, In, Out]，包装为Parameter支持反向传播
        Tensor weightTensor = torch.empty(new long[]{numRelations, inChannels, outChannels}, torch.dtype(torch.ScalarType.Float), new MemoryFormatOptional());
        torch.xavier_uniform_(weightTensor); // Xavier初始化
        this.weight = new Parameter(weightTensor);
        register_parameter("weight", this.weight); // 注册可训练参数

        // 2. 根节点线性层（可选）
        if (rootWeight) {
            this.linRoot = new LinearImpl(inChannels, outChannels);
            register_module("lin_root", this.linRoot); // 注册子模块
        } else {
            this.linRoot = null;
        }

        // 3. 偏置参数（可选）
        if (hasBias) {
            Tensor biasTensor = torch.zeros(new long[]{outChannels}, torch.dtype(torch.ScalarType.Float));
            this.bias = new Parameter(biasTensor);
            register_parameter("bias", this.bias);
        } else {
            this.bias = null;
        }
    }

    /**
     * 前向传播核心逻辑（CSC格式优化）
     *
     * @param x         节点特征 [N, In]
     * @param row       CSC行索引（源节点ID）[E]
     * @param colptr    CSC列指针（目标节点偏移）[N+1]
     * @param edge_type 每条边的关系类型 [E]
     * @return 输出特征 [N, Out]
     */
    public Tensor forward(Tensor x, Tensor row, Tensor colptr, Tensor edge_type) {
        // ========== 1. 输入合法性校验 ==========
        if (x.dim() != 2) throw new IllegalArgumentException("x必须是2维张量 [N, In]，当前维度：" + x.dim());
        if (row.dim() != 1) throw new IllegalArgumentException("row必须是1维张量 [E]，当前维度：" + row.dim());
        if (colptr.dim() != 1) throw new IllegalArgumentException("colptr必须是1维张量 [N+1]，当前维度：" + colptr.dim());
        if (edge_type.dim() != 1)
            throw new IllegalArgumentException("edge_type必须是1维张量 [E]，当前维度：" + edge_type.dim());

        long N = x.size(0); // 节点数
        long E = row.size(0); // 边数

        if (colptr.size(0) != N + 1) {
            throw new IllegalArgumentException("colptr长度必须为N+1（N=" + N + "），当前：" + colptr.size(0));
        }
        if (edge_type.size(0) != E) {
            throw new IllegalArgumentException("edge_type长度必须等于边数E（E=" + E + "），当前：" + edge_type.size(0));
        }
        // 校验关系类型范围
        long maxRel = edge_type.max().item_long();
        long minRel = edge_type.min().item_long();
        if (maxRel >= numRelations || minRel < 0) {
            throw new IllegalArgumentException("edge_type必须在 [0, " + (numRelations - 1) + "] 范围内，当前：[" + minRel + ", " + maxRel + "]");
        }

        // ========== 2. 生成CSC目标节点索引（核心修正） ==========
        // 替代错误的 repeat_interleave，生成真正的目标节点ID [E]
        Tensor targetIdx = torch.zeros(new long[]{E}, torch.dtype(torch.ScalarType.Long));
        tensors.push_back(targetIdx);
        for (long i = 0; i < N; i++) {
            long start = colptr.get(i).item_long();//.longValue();
            long end = colptr.get(i + 1).item_long();//.longValue();
            if (start >= end) continue; // 孤立节点，无边
            targetIdx.narrow(0, start, end - start).fill_(new Scalar(i)); // 填充目标节点i
        }

        // ========== 3. 低显存版消息计算（避免生成 [E, In, Out] 大张量） ==========
        // 思路：按关系类型分组计算，而非逐边计算，减少显存占用
        Tensor out = torch.zeros(new long[]{N, outChannels}, x.options());
        tensors.push_back(out);

        for (int r = 0; r < numRelations; r++) {
            // 3.1 筛选当前关系类型的边索引
            Tensor relMask = edge_type.eq(new Scalar(r)); // [E]，标记当前关系的边
            tensors.push_back(relMask);
            if (relMask.sum().item_long() == 0) continue; // 无该关系的边，跳过

            // 3.2 提取当前关系的源节点、目标节点、权重
            Tensor relRow = row.masked_select(relMask); // [E_r]，当前关系的源节点
            Tensor relTargetIdx = targetIdx.masked_select(relMask); // [E_r]，当前关系的目标节点
            Tensor w_r = weight.data().index_select(0, torch.tensor(new long[]{r}, torch.dtype(torch.ScalarType.Long))).squeeze(0); // [In, Out]
            tensors.push_back(relRow);
            tensors.push_back(relTargetIdx);
            tensors.push_back(w_r);

            // 3.3 源节点特征变换：x[relRow] @ w_r → [E_r, Out]
            Tensor x_rel = x.index_select(0, relRow); // [E_r, In]
            Tensor msg_rel = torch.matmul(x_rel, w_r); // [E_r, Out]
            tensors.push_back(x_rel);
            tensors.push_back(msg_rel);

            // 3.4 按关系聚合到目标节点（分步聚合，降低显存）
            aggregateScatter(out, msg_rel, relTargetIdx);
        }

        // ========== 4. 根节点特征融合（可选） ==========
        if (linRoot != null) {
            Tensor rootOut = linRoot.forward(x); // [N, Out]
            tensors.push_back(rootOut);
            out = out.add(rootOut);
        }

        // ========== 5. 添加偏置（可选） ==========
        if (bias != null) {
            out = out.add(bias.data());
            tensors.push_back(out);
        }

        return out;
    }

    /**
     * 自定义聚合函数（适配 bytedeco-pytorch，替代 scatter_reduce_）
     *
     * @param out       输出张量 [N, Out]
     * @param msg       消息张量 [E_r, Out]
     * @param targetIdx 目标节点索引 [E_r]
     */
    private void aggregateScatter(Tensor out, Tensor msg, Tensor targetIdx) {
        long N = out.size(0);
        long Out = out.size(1);
        Tensor expandIdx = targetIdx.view(new long[]{-1, 1}).expand_as(msg); // [E_r, Out]
        tensors.push_back(expandIdx);

        switch (aggr) {
            case "sum":
                out.scatter_add_(0, expandIdx, msg);
                break;
            case "mean":
                // 先求和，再计算均值
                Tensor sumTemp = torch.zeros(new long[]{N, Out}, out.options());
                sumTemp.scatter_add_(0, expandIdx, msg);
                // 计算每个节点的边数
                Tensor count = torch.zeros(new long[]{N}, out.options());
                count.scatter_add_(0, targetIdx, torch.ones(targetIdx.sizes(), out.options()));
                count.clamp_(new ScalarOptional(new Scalar(1.0))); // 避免除以0
                // 均值聚合
                out.add_(sumTemp.div(count.view(new long[]{N, 1}).expand_as(sumTemp)));
                tensors.push_back(sumTemp);
                tensors.push_back(count);
                break;
            case "max":
                Tensor maxOut = scatter_max(msg, targetIdx, 0, N);
                out.copy_(maxOut);
                tensors.push_back(maxOut);
                break;
        }
    }

    /**
     * 重写参数获取方法：返回所有可训练参数
     */
//    @Override
//    public ParameterDict named_parameters() {
//        ParameterDict params = new ParameterDict();
//        if (weight != null) params.put("weight", weight);
//        if (linRoot != null) {
//            params.put("lin_root.weight", linRoot.weight());
//            if (linRoot.bias() != null) params.put("lin_root.bias", linRoot.bias());
//        }
//        if (bias != null) params.put("bias", bias);
//        return params;
//    }

    /**
     * 资源释放：批量释放所有临时Tensor和模块
     */
    @Override
    public void close() {
        // 1. 释放临时Tensor
        for (int i = 0; i < tensors.size(); i++) {
            Tensor t = tensors.get(i);
            if (t != null && !t.isNull()) t.close();
        }
        tensors.close();
        // 2. 释放子模块
        if (linRoot != null) linRoot.close();
        // 3. 释放参数
        if (weight != null) weight.close();
        if (bias != null) bias.close();
    }
}
/**
 * 模拟 CuGraphRGCNConv
 * 基于 CSC 格式优化的关系图卷积，旨在提供最低的显存足迹和最高的执行效率。
 */
//public class CuGraphRGCNConv extends Module {
//    private Tensor weight; // [numRelations, inChannels, outChannels]
//    private LinearImpl linRoot;
//    private Tensor bias;
//    private int numRelations;
//    private long inChannels;
//    private long outChannels;
//    private String aggr;
//
//    public CuGraphRGCNConv(long inChannels, long outChannels, int numRelations, boolean rootWeight, boolean hasBias, String aggr) {
//        super();
//        this.inChannels = inChannels;
//        this.outChannels = outChannels;
//        this.numRelations = numRelations;
//        this.aggr = aggr;
//
//        // 1. 初始化权重张量 [R, In, Out]
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
//     * @param x         节点特征 [N, In]
//     * @param row       CSC 的行索引 (Source Nodes)
//     * @param colptr    CSC 的列指针 (Target Node offsets)
//     * @param edge_type 每条边的关系类型 [E]
//     */
//    public Tensor forward(Tensor x, Tensor row, Tensor colptr, Tensor edge_type) {
//        long N = x.size(0);
//        long E = row.size(0);
//
//        // --- 核心优化步骤：模拟 Fused Aggregation ---
//        // 1. 展开得到 targetIdx，用于 scatter 聚合
//        Tensor targetIdx = torch.repeat_interleave(colptr.diff());
//
//        // 2. 根据 edge_type 获取每条边对应的权重 W_r
//        // W_r 形状: [E, In, Out]
//        Tensor w_per_edge = weight.index_select(0, edge_type);
//
//        // 3. 提取源节点特征并进行变换
//        // x_j 形状: [E, 1, In]
//        Tensor x_j = x.index_select(0, row).unsqueeze(1);
//
//        // 消息计算: x_j @ W_r -> [E, 1, Out] -> [E, Out]
//        Tensor msg = torch.matmul(x_j, w_per_edge).squeeze(1);
//
//        // 4. 聚合到目标节点
//        Tensor out = torch.zeros(new long[]{N, outChannels}, x.options());
//        out.scatter_reduce_(0, targetIdx.view(-1, 1).expand_as(msg), msg, aggr, false);
//
//        // 5. 加上根节点
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