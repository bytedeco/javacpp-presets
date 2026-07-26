package org.bytedeco.pytorch.geometric.nn.conv;
import org.bytedeco.pytorch.jit.*;
import org.bytedeco.pytorch.autograd.*;
import org.bytedeco.pytorch.nn.modules.*;

import org.bytedeco.javacpp.LongPointer;
import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.geometric.nn.Parameter;
import org.bytedeco.pytorch.geometric.utils.Scatter;

/**
 * 修正版 CuGraphGATConv：
 * 1. 不继承 MessagePassing（CSC 融合算子无需通用消息传递框架）
 * 2. 修复 CSC 索引映射、注意力方向等核心逻辑
 * 3. 正确注册可训练参数，完善内存管理
 * 4. 实现 Scatter 工具方法，保证 scatter_softmax 可运行
 */
public class CuGraphGATConv extends Module implements AutoCloseable {
    private LinearImpl lin;
    private final TensorVector tensors = new TensorVector(); // 管理临时Tensor，防止内存泄漏
    private long heads;
    private long outChannels;
    private boolean concat;
    private double negativeSlope;
    private Parameter attSrc, attDst; // 注意力参数（可训练）
    private Parameter bias;

    // 构造函数：初始化参数+注册模块
    public CuGraphGATConv(long inChannels, long outChannels, long heads, boolean concat, double negativeSlope, boolean hasBias) {
        super(); // 调用父类Module构造器
        this.heads = heads;
        this.outChannels = outChannels;
        this.concat = concat;
        this.negativeSlope = negativeSlope;

        // 1. 线性变换层：[inChannels] -> [heads * outChannels]
        this.lin = new LinearImpl(inChannels, heads * outChannels);
        register_module("lin", lin); // 注册子模块到父类Module

        // 2. 注意力参数初始化：[1, heads, outChannels]
        Tensor attSrcTensor = torch.randn(new long[]{1, heads, outChannels}, torch.dtype(torch.ScalarType.Float));
        Tensor attDstTensor = torch.randn(new long[]{1, heads, outChannels}, torch.dtype(torch.ScalarType.Float));
        // Xavier均匀初始化（符合GAT最佳实践）
        torch.xavier_uniform_(attSrcTensor);
        torch.xavier_uniform_(attDstTensor);
        // 包装为Parameter，支持反向传播
        this.attSrc = new Parameter(attSrcTensor);
        this.attDst = new Parameter(attDstTensor);
        // 注册参数到父类Module
        register_parameter("att_src", this.attSrc);
        register_parameter("att_dst", this.attDst);

        // 3. 偏置参数初始化
        if (hasBias) {
            long biasDim = concat ? heads * outChannels : outChannels;
            Tensor biasTensor = torch.zeros(new long[]{biasDim}, torch.dtype(torch.ScalarType.Float));
            this.bias = new Parameter(biasTensor);
            register_parameter("bias", this.bias);
        } else {
            this.bias = null;
        }
    }

    /**
     * 前向传播核心逻辑
     * @param x 节点特征 [N, inChannels]
     * @param row CSC行索引（源节点ID）[E]
     * @param colptr CSC列指针（目标节点偏移）[N+1]
     * @return 输出特征 [N, heads*outChannels] (concat=true) / [N, outChannels] (concat=false)
     */
    public Tensor forward(Tensor x, Tensor row, Tensor colptr) {
        // ========== 1. 输入合法性校验 ==========
        if (x.dim() != 2) {
            throw new IllegalArgumentException("节点特征x必须是2维张量，当前维度：" + x.dim());
        }
        if (row.dim() != 1) {
            throw new IllegalArgumentException("CSC行索引row必须是1维张量，当前维度：" + row.dim());
        }
        if (colptr.dim() != 1) {
            throw new IllegalArgumentException("CSC列指针colptr必须是1维张量，当前维度：" + colptr.dim());
        }
        long N = x.size(0); // 节点数
        if (colptr.size(0) != N + 1) {
            throw new IllegalArgumentException("colptr长度必须为N+1（N=" + N + "），当前长度：" + colptr.size(0));
        }
        long E = row.size(0); // 边数
        Tensor colptrDiff = colptr.diff();
        tensors.push_back(colptrDiff);
        if (colptrDiff.sum().item_long() != E) {
            throw new IllegalArgumentException("colptr.diff()求和应等于边数E（E=" + E + "），当前求和：" + colptrDiff.sum().item_long());
        }

        // ========== 2. 线性变换 + 维度重塑 ==========
        // xLin: [N, inChannels] -> [N, heads*outChannels] -> [N, heads, outChannels]
        Tensor xLin = lin.forward(x).view(new long[]{N, heads, outChannels});
        tensors.push_back(xLin);

        // ========== 3. 计算注意力分量 ==========
        // alphaSrc: (xLin * attSrc).sum(-1) -> [N, heads]
        // alphaDst: (xLin * attDst).sum(-1) -> [N, heads]
        LongPointer sumDim = new LongPointer(new long[]{-1}); // bytedeco指定最后一维的正确方式
        Tensor alphaSrc = xLin.mul(attSrc.data()).sum(-1).squeeze(-1);
        Tensor alphaDst = xLin.mul(attDst.data()).sum(-1).squeeze(-1);
        tensors.push_back(alphaSrc);
        tensors.push_back(alphaDst);

        // ========== 4. CSC格式核心聚合 ==========
        Tensor out = aggregateGATCSC(xLin, alphaSrc, alphaDst, row, colptr, N);
        tensors.push_back(out);

        // ========== 5. 多头注意力合并 ==========
        if (concat) {
            // concat模式：[N, heads, outChannels] -> [N, heads*outChannels]
            out = out.view(new long[]{N, heads * outChannels});
        } else {
            // 平均模式：[N, heads, outChannels] -> [N, outChannels]
            out = out.mean(1).squeeze(1);
        }
        tensors.push_back(out);

        // ========== 6. 添加偏置 ==========
        if (bias != null) {
            out = out.add(bias.data());
            tensors.push_back(out);
        }

        return out;
    }

    /**
     * CSC格式GAT核心聚合：模拟CuGraph融合逻辑
     * @param xLin 线性变换后的节点特征 [N, heads, outChannels]
     * @param alphaSrc 源节点注意力分量 [N, heads]
     * @param alphaDst 目标节点注意力分量 [N, heads]
     * @param row CSC行索引（源节点）[E]
     * @param colptr CSC列指针（目标节点偏移）[N+1]
     * @param numNodes 节点数 N
     * @return 聚合后的特征 [N, heads, outChannels]
     */
    private Tensor aggregateGATCSC(Tensor xLin, Tensor alphaSrc, Tensor alphaDst, Tensor row, Tensor colptr, long numNodes) {
        long E = row.size(0);

        // ========== 1. 生成CSC目标节点索引（核心修正） ==========
        // colptr: [0, e1, e1+e2, ..., E] → 目标节点i对应边的范围是 [colptr[i], colptr[i+1])
        // targetIdx: [0,0,...,1,1,...,2,...] → 长度E，每个元素是边对应的目标节点ID
        Tensor targetIdx = torch.zeros(new long[]{E}, torch.dtype(torch.ScalarType.Long));
        tensors.push_back(targetIdx);
        for (long i = 0; i < numNodes; i++) {
            long start = colptr.get(i).item_long();//.longValue();
            long end = colptr.get(i + 1).item_long();//.longValue();
            if (start >= end) continue; // 孤立节点，无边
            // 填充目标节点i到 [start, end) 区间
            targetIdx.narrow(0, start, end - start).fill_(new Scalar(i));
        }

        // ========== 2. 计算未归一化注意力分数（修正方向） ==========
        // GAT正确逻辑：e_ij = LeakyReLU(alphaDst[i] + alphaSrc[j])
        // i=目标节点（targetIdx），j=源节点（row）
        Tensor alphaDstTarget = alphaDst.index_select(0, targetIdx); // [E, heads]
        Tensor alphaSrcRow = alphaSrc.index_select(0, row); // [E, heads]
        Tensor e_ij = alphaDstTarget.add(alphaSrcRow);
        tensors.push_back(alphaDstTarget);
        tensors.push_back(alphaSrcRow);
        tensors.push_back(e_ij);

        // LeakyReLU激活
        e_ij = torch.leaky_relu(e_ij, new Scalar(negativeSlope));
        tensors.push_back(e_ij);

        // ========== 3. CSC局部Scatter Softmax（数值稳定） ==========
        Tensor alpha = scatter_softmax(e_ij, targetIdx, numNodes);
        tensors.push_back(alpha);

        // ========== 4. 加权聚合源节点特征 ==========
        // xLin[row]: [E, heads, outChannels] → 源节点特征
        // alpha.unsqueeze(-1): [E, heads, 1] → 注意力权重广播
        Tensor msg = xLin.index_select(0, row).mul(alpha.unsqueeze(-1));
        tensors.push_back(msg);

        // Scatter Add聚合到目标节点：[N, heads, outChannels]
        Tensor out = torch.zeros(new long[]{numNodes, heads, outChannels}, xLin.options());
        tensors.push_back(out);
        // 扩展targetIdx到[E,1,1]，匹配msg维度
        Tensor targetIdxExpand = targetIdx.view(new long[]{E, 1, 1}).expand_as(msg);
        tensors.push_back(targetIdxExpand);
        out.scatter_add_(0, targetIdxExpand, msg);

        return out;
    }

    /**
     * 数值稳定的Scatter Softmax实现
     *
     * @param src     输入张量 [E, heads]
     * @param index   目标节点索引 [E]
     * @param dimSize 节点数 N
     * @return 归一化后的注意力权重 [E, heads]
     */
    private Tensor scatter_softmax(Tensor src, Tensor index, long dimSize) {
        // Step1: 按目标节点取最大值（数值稳定）
        Tensor maxVal = Scatter.scatter(src, index, dimSize, "max");
//        Tensor maxVal = scatter(src, index, dimSize, "max");
        tensors.push_back(maxVal);
        // Step2: 减去最大值后exp
        Tensor srcShifted = src.sub(maxVal.index_select(0, index));
        Tensor out = torch.exp(srcShifted);
        tensors.push_back(srcShifted);
        tensors.push_back(out);
        // Step3: 按目标节点求和
        Tensor sumVal = Scatter.scatter(out, index, dimSize, "add");
        tensors.push_back(sumVal);
        // Step4: 归一化（加小常数避免除以0）
        sumVal = sumVal.add(new Scalar(1e-16));
        out = out.div(sumVal.index_select(0, index));
        // 处理nan/inf（边界保护）
        out.masked_fill_(torch.logical_or(out.isnan(), out.isinf()), new Scalar(0.0));
        return out;
    }

    /**
     * 通用Scatter工具方法（替代缺失的org.bytedeco.pytorch.geometric.utils.Scatter）
     * @param src 输入张量 [E, *]
     * @param index 目标索引 [E]
     * @param dimSize 输出维度大小 N
     * @param reduce 聚合方式：add/max/mean
     * @return 聚合结果 [N, *]
     */
//    private Tensor scatter(Tensor src, Tensor index, long dimSize, String reduce) {
//        long[] srcShape = new long[src.dim()];
//        for (int i = 0; i < src.dim(); i++) srcShape[i] = src.size(i);
//        // 构造输出形状：[dimSize, srcShape[1], ...]
//        long[] outShape = new long[src.dim()];
//        outShape[0] = dimSize;
//        System.arraycopy(srcShape, 1, outShape, 1, src.dim() - 1);
//
//        Tensor out = null;
//        switch (reduce) {
//            case "add":
//                out = torch.zeros(outShape, src.options());
//                out.scatter_add_(0, index.view(new long[]{src.size(0), 1}).expand_as(src), src);
//                break;
//            case "max":
//                out = torch.full(outShape, new Scalar(Float.NEGATIVE_INFINITY), src.options());
//                PairScalarTensor maxResult = scatter_max(src, 0, index.view(new long[]{src.size(0), 1}).expand_as(src), out);
//                out = maxResult.second();
//                out.masked_fill_(out.eq(new Scalar(Double.NEGATIVE_INFINITY)), new Scalar(0.0));
//                break;
//            case "mean":
//                out = torch.zeros(outShape, src.options());
//                out.scatter_add_(0, index.view(new long[]{src.size(0), 1}).expand_as(src), src);
//                // 计算每个索引的计数
//                Tensor count = torch.zeros(new long[]{dimSize}, src.options());
//                count.scatter_add_(0, index, torch.ones(new long[]{index.size(0)}, src.options()));
//                count.clamp_(new ScalarOptional(new Scalar(1.0))); // 避免除以0
//                out = out.div(count.view(new long[]{dimSize, 1}).expand_as(out));
//                tensors.push_back(count);
//                break;
//            default:
//                throw new IllegalArgumentException("不支持的聚合方式：" + reduce);
//        }
//        tensors.push_back(out);
//        return out;
//    }

    /**
     * 重写参数获取方法：返回所有可训练参数
     */
//    @Override
//    public ParameterDict named_parameters() {
//        ParameterDict params = new ParameterDict();
//        if (lin != null) {
//            params.put("lin.weight", lin.weight());
//            if (lin.bias() != null) params.put("lin.bias", lin.bias());
//        }
//        if (attSrc != null) params.put("att_src", attSrc);
//        if (attDst != null) params.put("att_dst", attDst);
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
        if (lin != null) lin.close();
        // 3. 释放参数
        if (attSrc != null) attSrc.close();
        if (attDst != null) attDst.close();
        if (bias != null) bias.close();
    }
}


/**
 * 模拟 CuGraphGATConv
 * 针对 CSC 格式优化的 GAT 算子，追求显存节约与访存局部性。
 */
//public class CuGraphGATConv extends Module {
//    private LinearImpl lin;
//    private Tensor attSrc, attDst; // 对应 a_src 和 a_dst
//    private long heads;
//    private long outChannels;
//    private boolean concat;
//    private double negativeSlope;
//    private Tensor bias;
//
//    public CuGraphGATConv(long inChannels, long outChannels, long heads, boolean concat, double negativeSlope, boolean hasBias) {
//        super();
//        this.heads = heads;
//        this.outChannels = outChannels;
//        this.concat = concat;
//        this.negativeSlope = negativeSlope;
//
//        // 1. 线性变换: [In] -> [Heads * Out]
//        this.lin = new LinearImpl(inChannels, heads * outChannels);
//
//        // 2. 注意力参数向量: 分为源节点部分和目标节点部分 [1, Heads, Out]
//        this.attSrc = torch.randn(new long[]{1, heads, outChannels});
//        this.attDst = torch.randn(new long[]{1, heads, outChannels});
//        torch.xavier_uniform_(this.attSrc);
//        torch.xavier_uniform_(this.attDst);
//
//        register_module("lin", lin);
//        register_parameter("att_src", attSrc);
//        register_parameter("att_dst", attDst);
//
//        if (hasBias) {
//            long biasDim = concat ? heads * outChannels : outChannels;
//            this.bias = torch.zeros(new long[]{biasDim});
//            register_parameter("bias", bias);
//        }
//    }
//
//    /**
//     * @param x      节点特征 [N, In]
//     * @param row    CSC 行索引 (Source Nodes)
//     * @param colptr CSC 列指针 (Target Node offsets)
//     */
//    public Tensor forward(Tensor x, Tensor row, Tensor colptr) {
//        long N = x.size(0);
//
//        // 1. 线性变换与维度重塑: [N, H, C]
//        Tensor xLin = lin.forward(x).view(N, heads, outChannels);
//
//        // 2. 预计算注意力分量 (以减少重复计算)
//        // alpha_src = (x * att_src).sum(-1) -> [N, H]
//        // alpha_dst = (x * att_dst).sum(-1) -> [N, H]
//        Tensor alphaSrc = (xLin.mul(attSrc)).sum(-1);
//        Tensor alphaDst = (xLin.mul(attDst)).sum(-1);
//
//        // 3. 模拟 CuGraph 的 Fused 聚合逻辑
//        // 在原生 CUDA 中，以下步骤不产生 [E, H] 的大张量
//        Tensor out = aggregateGATCSC(xLin, alphaSrc, alphaDst, row, colptr, N);
//
//        // 4. 多头合并
//        if (concat) {
//            out = out.view(N, heads * outChannels);
//        } else {
//            out = out.mean(1);
//        }
//
//        if (bias != null) {
//            out = out.add(bias);
//        }
//
//        return out;
//    }
//
//    /**
//     * 核心聚合方法：模拟 CSC 局部 Softmax
//     */
//    private Tensor aggregateGATCSC(Tensor xLin, Tensor alphaSrc, Tensor alphaDst, Tensor row, Tensor colptr, long numNodes) {
//        // 为了在 Java 中保持高性能并模拟 CSC，我们分步操作：
//
//        // 1. 根据 colptr 展开得到目标节点索引 targetIdx (E)
//        // 这在 CuGraph 内部由 CUDA 线程格直接映射
//        Tensor targetIdx = torch.repeat_interleave(colptr.diff());
//
//        // 2. 计算边上的未归一化注意力 score: e_ij = LeakyReLU(a_src*h_j + a_dst*h_i)
//        // [E, H] = [E, H] + [E, H]
//        Tensor e_ij = alphaSrc.index_select(0, row).add(alphaDst.index_select(0, targetIdx));
//        e_ij = torch.leaky_relu(e_ij, new Scalar(negativeSlope));
//
//        // 3. CSC 局部 Softmax
//        // 注意：由于我们使用了 CSC 的顺序，相同 target 的边在内存中是连续的
//        // 这里的 scatter_softmax 效果等同于 Fused 局部归一化
//        Tensor alpha = scatter_softmax(e_ij, targetIdx, numNodes);
//
//        // 4. 加权聚合: sum(alpha_ij * x_j)
//        // [E, H, C] * [E, H, 1]
//        Tensor msg = xLin.index_select(0, row).mul(alpha.unsqueeze(-1));
//
//        // 聚合到目标节点
//        Tensor out = torch.zeros(new long[]{numNodes, heads, outChannels}, xLin.options());
//        out.scatter_add_(0, targetIdx.view(-1, 1, 1).expand_as(msg), msg);
//
//        return out;
//    }
//
//    private Tensor scatter_softmax(Tensor src, Tensor index, long dimSize) {
//        // 数值稳定的 Softmax 实现 (使用之前定义的 Scatter 工具)
//        Tensor maxVal = Scatter.scatter(src, index, dimSize, "max");
//        Tensor out = src.sub(maxVal.index_select(0, index)).exp();
//        Tensor sum = Scatter.scatter(out, index, dimSize, "add");
//        return out.div(sum.index_select(0, index).add(new Scalar(1e-16)));
//    }
//}