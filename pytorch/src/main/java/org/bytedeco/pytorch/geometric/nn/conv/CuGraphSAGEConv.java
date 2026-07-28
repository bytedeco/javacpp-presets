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
 * 修正版 CuGraphSAGEConv：
 * 1. 针对 CSC 格式优化的 GraphSAGE 算子，修复核心逻辑错误
 * 2. 适配 bytedeco-pytorch API 规范，保证可运行性
 * 3. 完善内存管理和边界条件处理
 */
public class CuGraphSAGEConv extends Module implements AutoCloseable {
    private final TensorVector tensors = new TensorVector(); // 管理临时Tensor，防止内存泄漏
    private LinearImpl linL; // 邻居聚合权重
    private LinearImpl linR; // 根节点权重
    private String aggr; // 支持 "mean", "sum", "max"
    private boolean normalize;
    private Parameter bias; // 包装为可训练参数

    public CuGraphSAGEConv(long inChannels, long outChannels, String aggr,
                           boolean normalize, boolean rootWeight, boolean hasBias) {
        super();

        // 校验聚合方式合法性
        if (!Arrays.asList("mean", "sum", "max").contains(aggr)) {
            throw new IllegalArgumentException("仅支持 mean/sum/max 聚合方式，当前：" + aggr);
        }
        this.aggr = aggr;
        this.normalize = normalize;

        // 1. 邻居变换层
        this.linL = new LinearImpl(inChannels, outChannels);
        register_module("lin_l", linL); // 注册子模块

        // 2. 根节点（自身）变换层（可选）
        if (rootWeight) {
            this.linR = new LinearImpl(inChannels, outChannels);
            register_module("lin_r", linR);
        } else {
            this.linR = null;
        }

        // 3. 偏置参数（可选，包装为Parameter支持训练）
        if (hasBias) {
            Tensor biasTensor = torch.zeros(new long[]{outChannels}, torch.dtype(torch.ScalarType.Float));
            this.bias = new Parameter(biasTensor);
            register_parameter("bias", this.bias); // 注册可训练参数
        } else {
            this.bias = null;
        }
    }

    /**
     * 前向传播核心逻辑
     *
     * @param x      节点特征 [N, In]
     * @param row    CSC行索引（源节点ID）[E]
     * @param colptr CSC列指针（目标节点偏移）[N+1]
     * @return 输出特征 [N, Out]
     */
    public Tensor forward(Tensor x, Tensor row, Tensor colptr) {
        // ========== 1. 输入合法性校验 ==========
        if (x.dim() != 2) throw new IllegalArgumentException("x必须是2维张量 [N, In]，当前维度：" + x.dim());
        if (row.dim() != 1) throw new IllegalArgumentException("row必须是1维张量 [E]，当前维度：" + row.dim());
        if (colptr.dim() != 1) throw new IllegalArgumentException("colptr必须是1维张量 [N+1]，当前维度：" + colptr.dim());

        long numNodes = x.size(0);
        long numEdges = row.size(0);

        if (colptr.size(0) != numNodes + 1) {
            throw new IllegalArgumentException("colptr长度必须为N+1（N=" + numNodes + "），当前：" + colptr.size(0));
        }
        // 校验 colptr.diff() 求和等于边数
        Tensor colptrDiff = colptr.diff();
        tensors.push_back(colptrDiff);
        if (colptrDiff.sum().item_long() != numEdges) {
            throw new IllegalArgumentException("colptr.diff()求和应等于边数E（E=" + numEdges + "），当前：" + colptrDiff.sum().item_long());
        }

        // ========== 2. 邻居聚合（核心修正） ==========
        Tensor neighborAggr = aggregateCSC(x, row, colptr, numNodes);
        tensors.push_back(neighborAggr);

        // ========== 3. 线性变换 ==========
        // 邻居特征变换
        Tensor out = linL.forward(neighborAggr);
        tensors.push_back(out);

        // 根节点特征变换（可选）
        if (linR != null) {
            Tensor rootOut = linR.forward(x);
            tensors.push_back(rootOut);
            out = out.add(rootOut);
        }

        // 添加偏置（可选）
        if (bias != null) {
            out = out.add(bias.data());
            tensors.push_back(out);
        }

        // ========== 4. L2 归一化（修正 API 调用） ==========
        if (normalize) {
            // bytedeco-pytorch 中 norm 的正确调用方式
//            LongPointer normDim = new LongPointer(new long[]{-1});
//            Tensor norm = out.norm(torch.TensorNorm.Two, normDim, true); // L2 范数
            Tensor norm = out.norm(new ScalarOptional(new Scalar(2)), new long[]{-1}, true);
            tensors.push_back(norm);
            // 避免除以0， clamp_min 后广播除法
            out = out.div(norm.clamp_min(new Scalar(1e-12)));
            tensors.push_back(out);
        }

        return out;
    }

    /**
     * 修正版 CSC 聚合逻辑（模拟 cugraph-ops）
     *
     * @param x        节点特征 [N, In]
     * @param row      CSC行索引 [E]
     * @param colptr   CSC列指针 [N+1]
     * @param numNodes 节点数 N
     * @return 聚合后的邻居特征 [N, In]
     */
    private Tensor aggregateCSC(Tensor x, Tensor row, Tensor colptr, long numNodes) {
        long numEdges = row.size(0);
        // ========== 1. 生成正确的 CSC 目标节点索引（核心修正） ==========
        // 替换错误的 repeat_interleave，生成真正的目标节点 ID [E]
        Tensor targetIdx = torch.zeros(new long[]{numEdges}, torch.dtype(torch.ScalarType.Long));
        tensors.push_back(targetIdx);
        for (long i = 0; i < numNodes; i++) {
            long start = colptr.get(i).item_long();
            long end = colptr.get(i + 1).item_long();
            if (start >= end) continue; // 孤立节点，无边
            // 填充目标节点i到 [start, end) 区间
            targetIdx.narrow(0, start, end - start).fill_(new Scalar(i));
        }

        // ========== 2. 提取源节点特征 [E, In] ==========
        Tensor x_j = x.index_select(0, row);
        tensors.push_back(x_j);

        // ========== 3. 初始化输出张量 ==========
        Tensor out = torch.zeros(new long[]{numNodes, x.size(1)}, x.options());
        tensors.push_back(out);

        // ========== 4. 自定义聚合（替代 scatter_reduce_） ==========
        Tensor expandIdx = targetIdx.unsqueeze(-1).expand_as(x_j);
        tensors.push_back(expandIdx);

        switch (aggr) {
            case "sum":
                out.scatter_add_(0, expandIdx, x_j);
                break;
            case "mean":
                // 先求和，再计算均值
                Tensor sumTemp = torch.zeros(new long[]{numNodes, x.size(1)}, x.options());
                sumTemp.scatter_add_(0, expandIdx, x_j);
                // 计算每个节点的边数（度）
                Tensor degree = torch.zeros(new long[]{numNodes}, x.options());
                degree.scatter_add_(0, targetIdx, torch.ones(targetIdx.sizes(), x.options()));
                // 避免除以0（孤立节点度为0，设为1）
                degree.clamp_(new ScalarOptional(new Scalar(1.0)));
                // 广播度到特征维度，计算均值
                out = sumTemp.div(degree.view(new long[]{numNodes, 1}).expand_as(sumTemp));
                tensors.push_back(sumTemp);
                tensors.push_back(degree);
                break;
            case "max":
                Tensor maxOut = scatter_max(x_j, targetIdx, 0, numNodes);
                out.copy_(maxOut);
                tensors.push_back(maxOut);
                break;
            //Tensor out, Tensor msg, Tensor targetIdx)
//                Tensor maxOut = scatter_max(row, targetIdx, 0, numEdges);
//                out.copy_(maxOut);
//                tensors.push_back(maxOut);
//                break;
            // Scatter Max 聚合（适配 bytedeco-pytorch）
//                PairScalarTensor maxResult = torch.scatter_max(x_j, 0, expandIdx, out);
//                out.copy_(maxResult.second());
//                tensors.push_back(maxResult.second());
//                break;
        }

        return out;
    }

    /**
     * 重写参数获取方法：返回所有可训练参数
     */
//    @Override
//    public ParameterDict named_parameters() {
//        ParameterDict params = new ParameterDict();
//        // 邻居变换层参数
//        if (linL != null) {
//            params.put("lin_l.weight", linL.weight());
//            if (linL.bias() != null) params.put("lin_l.bias", linL.bias());
//        }
//        // 根节点变换层参数
//        if (linR != null) {
//            params.put("lin_r.weight", linR.weight());
//            if (linR.bias() != null) params.put("lin_r.bias", linR.bias());
//        }
//        // 偏置参数
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
        if (linL != null) linL.close();
        if (linR != null) linR.close();
        // 3. 释放参数
        if (bias != null) bias.close();
    }
}
/**
 * 模拟 CuGraphSAGEConv
 * 针对 CSC 格式优化的 GraphSAGE 算子
 */
//public class CuGraphSAGEConv extends Module {
//    private LinearImpl linL; // 邻居聚合权重
//    private LinearImpl linR; // 根节点权重
//    private String aggr; // "mean", "sum", "max"
//    private boolean normalize;
//    private Tensor bias;
//
//    public CuGraphSAGEConv(long inChannels, long outChannels, String aggr, boolean normalize, boolean rootWeight, boolean hasBias) {
//        super();
//        this.aggr = aggr;
//        this.normalize = normalize;
//
//        // 邻居变换
//        this.linL = new LinearImpl(inChannels, outChannels);
//        register_module("lin_l", linL);
//
//        // 根节点（自身）变换
//        if (rootWeight) {
//            this.linR = new LinearImpl(inChannels, outChannels);
//            register_module("lin_r", linR);
//        }
//
//        if (hasBias) {
//            this.bias = torch.zeros(new long[]{outChannels});
//            register_parameter("bias", bias);
//        }
//    }
//
//    /**
//     * @param x      节点特征 [N, In]
//     * @param row    CSC 的行索引 (Source Nodes)
//     * @param colptr CSC 的列指针 (Target Node offsets)
//     */
//    public Tensor forward(Tensor x, Tensor row, Tensor colptr) {
//        long numNodes = x.size(0);
//
//        // 1. 邻居聚合 (基于 CSC 格式的手动聚合逻辑)
//        // 在 C++/CUDA 层面，这一步是融合的。在 Java 层面，我们模拟这个过程：
//        Tensor neighborAggr = aggregateCSC(x, row, colptr, numNodes);
//
//        // 2. 线性变换
//        Tensor out = linL.forward(neighborAggr);
//
//        if (linR != null) {
//            out = out.add(linR.forward(x));
//        }
//
//        if (bias != null) {
//            out = out.add(bias);
//        }
//
//        // 3. L2 归一化
//        if (normalize) {
//            out = out.div(out.norm(new ScalarOptional(new Scalar(2)),new long[]{-1}, true).clamp_min(new Scalar(1e-12)));
//        }
//
//        return out;
//    }
//
//    /**
//     * 模拟 cugraph-ops 的 CSC 聚合逻辑
//     */
//    private Tensor aggregateCSC(Tensor x, Tensor row, Tensor colptr, long numNodes) {
//        // 由于 Java 无法直接执行 CUDA 内核，我们通过将 CSC 转回 COO 或使用 index_select 模拟
//        // 在生产高性能场景下，这一步通常由自定义 C++ Op 完成
//
//        // 转换逻辑：从 colptr 展开得到 targetIdx
//        // 虽然性能不如原生 CUDA，但保证了算法的等价性
//        Tensor targetIdx = expandColptr(colptr, row.size(0));
//
//        // 执行聚合
//        Tensor x_j = x.index_select(0, row);
//        Tensor out = torch.zeros(new long[]{numNodes, x.size(1)}, x.options());
//
//        // 使用 scatter_reduce 模拟聚合
//        out.scatter_reduce_(0, targetIdx.unsqueeze(-1).expand_as(x_j), x_j, aggr, false);
//
//        return out;
//    }
//
//    private Tensor expandColptr(Tensor colptr, long numEdges) {
//        // 将 [N+1] 的 colptr 转换为 [E] 的 targetIdx
//        // 例如 colptr=[0, 2, 5] -> targetIdx=[0, 0, 1, 1, 1]
//        Tensor diff = colptr.diff(); // 每个节点的度 [N]
//        return torch.repeat_interleave(diff);
//    }
//}