package org.bytedeco.pytorch.geometric.nn.conv;
import org.bytedeco.pytorch.nn.modules.container.*;
import org.bytedeco.pytorch.nn.options.*;
import org.bytedeco.pytorch.nn.modules.*;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.geometric.nn.Parameter;
import org.bytedeco.pytorch.geometric.utils.Scatter;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.global.torch;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.global.torch;

public class FusedGATConv extends Module {
    public LinearImpl lin;
    public Parameter attSrc;
    public Parameter attDst;
    private long heads;
    private long outChannels;
    private boolean concat;
    private double negativeSlope;

    public FusedGATConv(long inChannels, long outChannels, long heads, boolean concat, double negativeSlope) {
        super();
        this.heads = heads;
        this.outChannels = outChannels;
        this.concat = concat;
        this.negativeSlope = negativeSlope;

        TensorOptions paramOpts = new TensorOptions()
                .dtype(new ScalarTypeOptional(torch.ScalarType.Float))
                .device(new DeviceOptional(new Device(torch.kCPU())));
//                .requires_grad(new BoolOptional(true));

        this.lin = new LinearImpl(inChannels, heads * outChannels);
        initLinearParams(lin, inChannels, heads * outChannels, paramOpts);
        register_module("lin", lin);

        this.attSrc = new Parameter(torch.randn(new long[]{1, heads, outChannels}, paramOpts));
        this.attDst = new Parameter(torch.randn(new long[]{1, heads, outChannels}, paramOpts));
        torch.xavier_uniform_(attSrc.data());
        torch.xavier_uniform_(attDst.data());
        register_parameter("att_src", attSrc);
        register_parameter("att_dst", attDst);
    }

    public Tensor forward(Tensor x, Tensor[] csr, Tensor[] csc, Tensor perm) {
        // ========== 输入校验 ==========
        long[] xShape = x.sizes().vec().get();
        if (xShape.length != 2) {
            throw new IllegalArgumentException("节点特征x必须是2D张量，当前维度：" + xShape.length);
        }
        long N = xShape[0];

        if (csr == null || csr.length != 2 || csc == null || csc.length != 2) {
            throw new IllegalArgumentException("CSR/CSC 必须是 [rowptr/col] 和 [row/colptr] 格式");
        }
        Tensor rowptr = csr[0];
        Tensor col = csr[1];
        Tensor row = csc[0];       // CSC的row：按目标节点排序后的源节点 [E]
        Tensor colptr = csc[1];    // CSC的colptr：目标节点的边起始位置 [N+1]

        long E = col.size(0);
        if (E == 0) {
            long outDim = concat ? heads * outChannels : outChannels;
            return torch.zeros(new long[]{N, outDim}, x.options());
        }

        // ========== 1. 线性投影 [N, H*C] → [N, H, C] ==========
        Tensor xLin = lin.forward(x).view(N, heads, outChannels);

        // ========== 2. 预计算节点注意力贡献 [N, H] ==========
        Tensor alphaSrc = (xLin.mul(attSrc)).sum(-1); // [N, H]
        Tensor alphaDst = (xLin.mul(attDst)).sum(-1); // [N, H]

        // ========== 3. Fused聚合（修复：基于CSC的正确映射） ==========
        Tensor out = aggregateFused(xLin, alphaSrc, alphaDst, row, colptr, N, perm);

        // ========== 4. 多头合并 ==========
        if (concat) {
            out = out.view(N, heads * outChannels);
        } else {
            out = out.mean(1);
        }

        xLin.close();
        alphaSrc.close();
        alphaDst.close();

        return out;
    }

    /**
     * 修复版聚合逻辑：
     * 1. 基于CSC的colptr生成正确的targetIdx（与row对齐）
     * 2. 支持禁用LeakyReLU（negativeSlope=-1时）
     */
    private Tensor aggregateFused(Tensor xLin, Tensor alphaSrc, Tensor alphaDst,
                                  Tensor row, Tensor colptr, long numNodes, Tensor perm) {
        long E = row.size(0);
        long H = heads;
        long C = outChannels;

        // ========== 修复1：生成与CSC row对齐的targetIdx ==========
        // colptr.diff() → 每个目标节点的边数 [N]
        Tensor colDiff = colptr.diff(); // [numNodes]
        // 生成目标节点索引：节点i重复colDiff[i]次 → [E]（与CSC row完全对齐）
        Tensor targetIdx = torch.arange(new Scalar(numNodes), colDiff.options()).repeat_interleave(colDiff);

        // ========== 修复2：注意力系数计算（支持禁用LeakyReLU） ==========
        // 源节点注意力（row是CSC排序后的源节点索引）
        Tensor alphaSrcRow = alphaSrc.index_select(0, row); // [E, H]
        // 目标节点注意力
        Tensor alphaDstTarget = alphaDst.index_select(0, targetIdx); // [E, H]
        // 原始注意力系数 e_ij = src + dst
        Tensor e_ij = alphaSrcRow.add(alphaDstTarget); // [E, H]

        // 可选：禁用LeakyReLU（测试时negativeSlope=-1）
        if (this.negativeSlope >= 0) {
            e_ij = torch.leaky_relu(e_ij, new Scalar(this.negativeSlope));
        }

        // ========== 3. Scatter Softmax（按目标节点） ==========
        Tensor alpha = scatter_softmax(e_ij, targetIdx, numNodes); // [E, H]

        // ========== 4. 加权消息计算 ==========
        Tensor xLinRow = xLin.index_select(0, row); // [E, H, C]
        Tensor msg = xLinRow.mul(alpha.unsqueeze(-1)); // [E, H, C]

        // ========== 5. 聚合到目标节点（非原地操作） ==========
        Tensor out = torch.zeros(new long[]{numNodes, H, C}, xLin.options());
        Tensor expandIdx = targetIdx.view(-1, 1, 1).expand_as(msg);
        out = out.scatter_add(0, expandIdx, msg);

        // ========== 资源释放 ==========
        colDiff.close();
        targetIdx.close();
        alphaSrcRow.close();
        alphaDstTarget.close();
        e_ij.close();
        alpha.close();
        xLinRow.close();
        msg.close();
        expandIdx.close();

        return out;
    }

    /**
     * 修复版scatter_softmax（public + 兼容JavaCPP API）
     */
    public Tensor scatter_softmax(Tensor src, Tensor index, long dimSize) {
        if (src == null || index == null) {
            throw new IllegalArgumentException("src和index不能为空");
        }
        long[] srcShape = src.sizes().vec().get();
        if (srcShape.length != 2) {
            throw new IllegalArgumentException("src必须是2D张量 [E, H]，当前维度：" + srcShape.length);
        }
        long[] indexShape = index.sizes().vec().get();
        if (indexShape.length != 1 || indexShape[0] != srcShape[0]) {
            throw new IllegalArgumentException("index必须是1D张量且长度等于src第一维，当前：" + indexShape);
        }

        // Step1: 按索引取最大值
        Tensor maxValInit = torch.full(new long[]{dimSize, srcShape[1]}, new Scalar(Float.NEGATIVE_INFINITY), src.options());
        Tensor maxVal = torch.scatter_reduce(
                maxValInit,
                0,
                index.unsqueeze(-1).expand_as(src),
                src,
                "max",
                false
        );

        // Step2: 减去最大值后exp
        Tensor out = src.sub(maxVal.index_select(0, index)).exp();

        // Step3: 按索引求和
        Tensor sumInit = torch.zeros(new long[]{dimSize, srcShape[1]}, src.options());
        Tensor sum = torch.scatter_reduce(
                sumInit,
                0,
                index.unsqueeze(-1).expand_as(out),
                out,
                "sum",
                false
        );

        // Step4: 除以和（加epsilon避免除0）
        out = out.div(sum.index_select(0, index).add(new Scalar(1e-16)));

        maxValInit.close();
        maxVal.close();
        sumInit.close();
        sum.close();

        return out;
    }

    private void initLinearParams(LinearImpl linear, long inDim, long outDim, TensorOptions paramOpts) {
        Tensor weight = torch.empty(new long[]{outDim, inDim}, paramOpts, new MemoryFormatOptional());
        weight = torch.xavier_uniform_(weight);
        linear.weight(new Parameter(weight));

        if (linear.bias() != null) {
            Tensor bias = torch.zeros(new long[]{outDim}, paramOpts);
            linear.bias(new Parameter(bias));
        }
    }

    /**
     * 修复版toGraphFormat：保证CSC的row/targetIdx严格对齐
     */
    public static Object[] toGraphFormat(Tensor edge_index, long numNodes) {
        long[] edgeShape = edge_index.sizes().vec().get();
        if (edgeShape.length != 2 || edgeShape[0] != 2) {
            throw new IllegalArgumentException("edge_index必须是[2, E]形状，当前：" + edgeShape);
        }
        long E = edgeShape[1];
        if (E == 0) {
            // 空边处理
            Tensor rowptr = torch.zeros(new long[]{numNodes + 1}, edge_index.options().dtype(new ScalarTypeOptional(torch.ScalarType.Long)));
            Tensor col = torch.empty(new long[]{0}, edge_index.options().dtype(new ScalarTypeOptional(torch.ScalarType.Long)),new MemoryFormatOptional());
            Tensor row = torch.empty(new long[]{0}, edge_index.options().dtype(new ScalarTypeOptional(torch.ScalarType.Long)),new MemoryFormatOptional());
            Tensor colptr = torch.zeros(new long[]{numNodes + 1}, edge_index.options().dtype(new ScalarTypeOptional(torch.ScalarType.Long)));
            Tensor perm = torch.empty(new long[]{0}, edge_index.options().dtype(new ScalarTypeOptional(torch.ScalarType.Long)),new MemoryFormatOptional());
            return new Object[]{new Tensor[]{rowptr, col}, new Tensor[]{row, colptr}, perm};
        }
//        if (E == 0) {
//            Tensor rowptr = torch.zeros(new long[]{numNodes + 1}, edge_index.options().dtype(torch.ScalarType.Long));
//            Tensor col = torch.empty(new long[]{0}, edge_index.options().dtype(torch.ScalarType.Long));
//            Tensor row = torch.empty(new long[]{0}, edge_index.options().dtype(torch.ScalarType.Long));
//            Tensor colptr = torch.zeros(new long[]{numNodes + 1}, edge_index.options().dtype(torch.ScalarType.Long));
//            Tensor perm = torch.empty(new long[]{0}, edge_index.options().dtype(torch.ScalarType.Long));
//            return new Object[]{new Tensor[]{rowptr, col}, new Tensor[]{row, colptr}, perm};
//        }

        Tensor srcNodes = edge_index.select(0, 0); // [E]
        Tensor dstNodes = edge_index.select(0, 1); // [E]

        // 1. 生成CSR（按源节点排序）
        Tensor sortedCSR = srcNodes.argsort();
        Tensor col = dstNodes.index_select(0, sortedCSR);
        Tensor srcCount = torch.bincount(srcNodes, null, numNodes);
        Tensor rowptr = torch.cat(new TensorVector(
                torch.zeros(new long[]{1}, srcCount.options()),
                srcCount.cumsum(0)
        ), 0);

        // 2. 生成CSC（按目标节点排序）
        Tensor sortedCSC = dstNodes.argsort();
        Tensor row = srcNodes.index_select(0, sortedCSC); // 按目标节点排序后的源节点
        Tensor dstCount = torch.bincount(dstNodes, null, numNodes);
        Tensor colptr = torch.cat(new TensorVector(
                torch.zeros(new long[]{1}, dstCount.options()),
                dstCount.cumsum(0)
        ), 0);

        // 3. 生成Permutation
        Tensor perm = sortedCSR.argsort().index_select(0, sortedCSC);

        // 资源释放
        srcNodes.close();
        dstNodes.close();
        sortedCSR.close();
        srcCount.close();
        sortedCSC.close();
        dstCount.close();

        return new Object[]{new Tensor[]{rowptr, col}, new Tensor[]{row, colptr}, perm};
    }

//    @Override
//    protected void finalize() throws Throwable {
//        try {
//            if (lin != null) lin.close();
//            if (attSrc != null) attSrc.close();
//            if (attDst != null) attDst.close();
//        } finally {
//            super.finalize();
//        }
//    }

    // Getter
    public long getHeads() { return heads; }
    public long getOutChannels() { return outChannels; }
    public boolean isConcat() { return concat; }
    public double getNegativeSlope() { return negativeSlope; }
}


//public class FusedGATConv extends Module {
//    public LinearImpl lin;
//    public Parameter attSrc;  // 修复：改用Parameter支持梯度
//    public Parameter attDst;
//    private long heads;
//    private long outChannels;
//    private boolean concat;
//    private double negativeSlope;
//
//    public FusedGATConv(long inChannels, long outChannels, long heads, boolean concat, double negativeSlope) {
//        super();
//        this.heads = heads;
//        this.outChannels = outChannels;
//        this.concat = concat;
//        this.negativeSlope = negativeSlope;
//
//        // 统一参数配置：Float + CPU，确保设备/类型一致
//        TensorOptions paramOpts = new TensorOptions()
//                .dtype(new ScalarTypeOptional(torch.ScalarType.Float))
//                .device(new DeviceOptional(new Device(torch.kCPU())));
////                .requires_grad(new BoolOptional(true));
//
//        // 1. 权重层：in → heads*out
//        this.lin = new LinearImpl(inChannels, heads * outChannels);
//        initLinearParams(lin, inChannels, heads * outChannels, paramOpts);
//        register_module("lin", lin);
//
//        // 2. 注意力参数（修复：包装为Parameter，支持梯度）
//        this.attSrc = new Parameter(torch.randn(new long[]{1, heads, outChannels}, paramOpts));
//        this.attDst = new Parameter(torch.randn(new long[]{1, heads, outChannels}, paramOpts));
//        // Xavier初始化
//        torch.xavier_uniform_(attSrc.data());
//        torch.xavier_uniform_(attDst.data());
//        // 注册参数
//        register_parameter("att_src", attSrc);
//        register_parameter("att_dst", attDst);
//    }
//
//    /**
//     * 核心前向传播
//     * @param x    节点特征 [N, inChannels]
//     * @param csr  CSR格式: [rowptr (N+1), col (E)]
//     * @param csc  CSC格式: [row (E), colptr (N+1)]
//     * @param perm 重排索引（CSR→CSC映射）[E]
//     * @return 输出特征 [N, heads*out] (concat=true) 或 [N, out] (concat=false)
//     */
//    public Tensor forward(Tensor x, Tensor[] csr, Tensor[] csc, Tensor perm) {
//        // ========== 输入校验 ==========
//        long[] xShape = x.sizes().vec().get();
//        if (xShape.length != 2) {
//            throw new IllegalArgumentException("节点特征x必须是2D张量，当前维度：" + xShape.length);
//        }
//        long N = xShape[0]; // 节点数
//
//        // CSR/CSC 校验
//        if (csr == null || csr.length != 2 || csc == null || csc.length != 2) {
//            throw new IllegalArgumentException("CSR/CSC 必须是 [rowptr/col] 和 [row/colptr] 格式");
//        }
//        Tensor rowptr = csr[0];
//        Tensor col = csr[1];
//        Tensor row = csc[0];
//        Tensor colptr = csc[1];
//
//        // 空边处理
//        long E = col.size(0);
//        if (E == 0) {
//            long outDim = concat ? heads * outChannels : outChannels;
//            return torch.zeros(new long[]{N, outDim}, x.options());
//        }
//
//        // ========== 1. 线性投影 [N, H*C] → [N, H, C] ==========
//        Tensor xLin = lin.forward(x).view(N, heads, outChannels);
//
//        // ========== 2. 预计算节点注意力贡献 [N, H] ==========
//        Tensor alphaSrc = (xLin.mul(attSrc)).sum(-1); // [N, H]
//        Tensor alphaDst = (xLin.mul(attDst)).sum(-1); // [N, H]
//
//        // ========== 3. Fused聚合（CSC局部性优化） ==========
//        Tensor out = aggregateFused(xLin, alphaSrc, alphaDst, row, colptr, N);
//
//        // ========== 4. 多头合并 ==========
//        if (concat) {
//            out = out.view(N, heads * outChannels); // [N, H*C]
//        } else {
//            out = out.mean(1); // [N, C] (沿头维度平均)
//        }
//
//        // ========== 资源释放 ==========
//        xLin.close();
//        alphaSrc.close();
//        alphaDst.close();
//
//        return out;
//    }
//
//    /**
//     * 融合聚合逻辑（CSC列局部性优化）
//     */
//    private Tensor aggregateFused(Tensor xLin, Tensor alphaSrc, Tensor alphaDst, Tensor row, Tensor colptr, long numNodes) {
//        long E = row.size(0); // 边数
//        long H = heads;
//        long C = outChannels;
//
//        // ========== 修复：正确生成目标节点索引（CSC的colptr→targetIdx） ==========
//        // colptr: [0, e1, e1+e2, ..., E] → diff得到每个节点的边数 [e1, e2, ..., eN]
//        Tensor colDiff = colptr.diff(); // [numNodes]
//        // 生成每个节点对应的边索引：节点i对应 E_i 条边 → targetIdx [E]
//        Tensor targetIdx = torch.arange(new Scalar(numNodes), colDiff.options()).repeat_interleave(colDiff);
//
//        // ========== 注意力系数计算（模拟Fused Kernel） ==========
//        // alphaSrc[row]: 源节点注意力 [E, H]
//        // alphaDst[targetIdx]: 目标节点注意力 [E, H]
//        Tensor e_ij = alphaSrc.index_select(0, row).add(alphaDst.index_select(0, targetIdx));
//        // LeakyReLU激活
//        e_ij = torch.leaky_relu(e_ij, new Scalar(negativeSlope));
//
//        // ========== 按目标节点Softmax（修复：scatter_softmax兼容PyTorch） ==========
//        Tensor alpha = scatter_softmax(e_ij, targetIdx, numNodes); // [E, H]
//
//        // ========== 加权消息计算 ==========
//        // xLin[row]: 源节点特征 [E, H, C]
//        // alpha.unsqueeze(-1): [E, H, 1] → 广播到 [E, H, C]
//        Tensor msg = xLin.index_select(0, row).mul(alpha.unsqueeze(-1)); // [E, H, C]
//
//        // ========== 聚合到目标节点（修复：非原地scatter_add） ==========
//        Tensor out = torch.zeros(new long[]{numNodes, H, C}, xLin.options());
//        // 扩展targetIdx到 [E, 1, 1] → 广播到 [E, H, C]
//        Tensor expandIdx = targetIdx.view(-1, 1, 1).expand_as(msg);
//        out = out.scatter_add(0, expandIdx, msg); // 非原地操作，支持梯度
//
//        // ========== 资源释放 ==========
//        colDiff.close();
//        targetIdx.close();
//        e_ij.close();
//        alpha.close();
//        msg.close();
//        expandIdx.close();
//
//        return out;
//    }
//
//    /**
//     * 修复版：按索引的Scatter Softmax（兼容JavaCPP scatter_reduce API，支持梯度）
//     * 关键修复：
//     * 1. 改用字符串类型的reduce操作（"max"/"sum"），匹配JavaCPP封装的scatter_reduce
//     * 2. 方法改为public，满足访问权限要求
//     * 3. 完善参数校验和资源管理
//     *
//     * @param src 输入张量 [E, H] (E=边数, H=头数)
//     * @param index 目标节点索引 [E]
//     * @param dimSize 目标维度大小（节点数N）
//     * @return 按索引softmax后的张量 [E, H]
//     */
//    public Tensor scatter_softmax(Tensor src, Tensor index, long dimSize) {
//        // ========== 输入校验 ==========
//        if (src == null || index == null) {
//            throw new IllegalArgumentException("src和index不能为空");
//        }
//        long[] srcShape = src.sizes().vec().get();
//        if (srcShape.length != 2) {
//            throw new IllegalArgumentException("src必须是2D张量 [E, H]，当前维度：" + srcShape.length);
//        }
//        long[] indexShape = index.sizes().vec().get();
//        if (indexShape.length != 1 || indexShape[0] != srcShape[0]) {
//            throw new IllegalArgumentException("index必须是1D张量且长度等于src第一维，当前：" + indexShape);
//        }
//
//        // ========== Step1: 按索引取最大值（数值稳定） ==========
//        // 初始化maxVal为负无穷：[dimSize, H]
//        Tensor maxValInit = torch.full(
//                new long[]{dimSize, srcShape[1]},
//                new Scalar(Float.NEGATIVE_INFINITY),
//                src.options()
//        );
//        // 调用scatter_reduce（max操作）：按index将src的最大值写入maxValInit
//        Tensor maxVal = torch.scatter_reduce(
//                maxValInit,                // 目标张量
//                0,                         // 操作维度
//                index.unsqueeze(-1).expand_as(src),  // 索引张量 [E, H]
//                src,                       // 源张量
//                "max",                     // reduce操作（字符串类型，匹配JavaCPP API）
//                false                      // 不允许原地操作
//        );
//
//        // ========== Step2: 减去最大值后exp（避免数值溢出） ==========
//        Tensor out = src.sub(maxVal.index_select(0, index)).exp();
//
//        // ========== Step3: 按索引求和 ==========
//        // 初始化sum为0：[dimSize, H]
//        Tensor sumInit = torch.zeros(
//                new long[]{dimSize, srcShape[1]},
//                src.options()
//        );
//        // 调用scatter_reduce（sum操作）：按index将out求和写入sumInit
//        Tensor sum = torch.scatter_reduce(
//                sumInit,                   // 目标张量
//                0,                         // 操作维度
//                index.unsqueeze(-1).expand_as(out), // 索引张量 [E, H]
//                out,                       // 源张量
//                "sum",                     // reduce操作（字符串类型）
//                false                      // 不允许原地操作
//        );
//
//        // ========== Step4: 除以和（加epsilon避免除0） ==========
//        out = out.div(sum.index_select(0, index).add(new Scalar(1e-16)));
//
//        // ========== 资源释放（避免内存泄漏） ==========
//        maxValInit.close();
//        maxVal.close();
//        sumInit.close();
//        sum.close();
//
//        return out;
//    }
//
//    /**
//     * 初始化线性层参数（Xavier）
//     */
//    private void initLinearParams(LinearImpl linear, long inDim, long outDim, TensorOptions paramOpts) {
//        Tensor weight = torch.empty(new long[]{outDim, inDim}, paramOpts,new MemoryFormatOptional());
//        weight = torch.xavier_uniform_(weight);
//        linear.weight(new Parameter(weight));
//
//        if (linear.bias() != null) {
//            Tensor bias = torch.zeros(new long[]{outDim}, paramOpts);
//            linear.bias(new Parameter(bias));
//        }
//    }
//
//    /**
//     * 修复版：EdgeIndex → CSR/CSC/Perm 转换
//     * @param edge_index 边索引 [2, E]
//     * @param numNodes 节点数 N
//     * @return Object[]{CSR, CSC, perm}
//     */
//    public static Object[] toGraphFormat(Tensor edge_index, long numNodes) {
//        // ========== 输入校验 ==========
//        long[] edgeShape = edge_index.sizes().vec().get();
//        if (edgeShape.length != 2 || edgeShape[0] != 2) {
//            throw new IllegalArgumentException("edge_index必须是[2, E]形状，当前：" + edgeShape);
//        }
//        long E = edgeShape[1];
//        if (E == 0) {
//            // 空边处理
//            Tensor rowptr = torch.zeros(new long[]{numNodes + 1}, edge_index.options().dtype(new ScalarTypeOptional(torch.ScalarType.Long)));
//            Tensor col = torch.empty(new long[]{0}, edge_index.options().dtype(new ScalarTypeOptional(torch.ScalarType.Long)),new MemoryFormatOptional());
//            Tensor row = torch.empty(new long[]{0}, edge_index.options().dtype(new ScalarTypeOptional(torch.ScalarType.Long)),new MemoryFormatOptional());
//            Tensor colptr = torch.zeros(new long[]{numNodes + 1}, edge_index.options().dtype(new ScalarTypeOptional(torch.ScalarType.Long)));
//            Tensor perm = torch.empty(new long[]{0}, edge_index.options().dtype(new ScalarTypeOptional(torch.ScalarType.Long)),new MemoryFormatOptional());
//            return new Object[]{new Tensor[]{rowptr, col}, new Tensor[]{row, colptr}, perm};
//        }
//
//        // ========== 1. 生成CSR（按源节点排序） ==========
//        Tensor srcNodes = edge_index.select(0, 0); // 源节点 [E]
//        Tensor dstNodes = edge_index.select(0, 1); // 目标节点 [E]
//        // 按源节点排序 → sortedCSR: 排序后的索引 [E]
//        Tensor sortedCSR = srcNodes.argsort();
//        // CSR: rowptr (N+1), col (E)
//        Tensor col = dstNodes.index_select(0, sortedCSR); // 排序后的目标节点
//        Tensor srcCount = torch.bincount(srcNodes, null, numNodes); // 每个源节点的边数 [N]
//        Tensor rowptr = torch.cat(new TensorVector(
//                torch.zeros(new long[]{1}, srcCount.options()),
//                srcCount.cumsum(0)
//        ), 0); // [N+1]
//
//        // ========== 2. 生成CSC（按目标节点排序） ==========
//        // 按目标节点排序 → sortedCSC: 排序后的索引 [E]
//        Tensor sortedCSC = dstNodes.argsort();
//        // CSC: row (E), colptr (N+1)
//        Tensor row = srcNodes.index_select(0, sortedCSC); // 排序后的源节点
//        Tensor dstCount = torch.bincount(dstNodes, null, numNodes); // 每个目标节点的边数 [N]
//        Tensor colptr = torch.cat(new TensorVector(
//                torch.zeros(new long[]{1}, dstCount.options()),
//                dstCount.cumsum(0)
//        ), 0); // [N+1]
//
//        // ========== 3. 生成Permutation（CSR→CSC映射） ==========
//        // sortedCSR.argsort(): CSR排序索引的逆映射 → 原始索引→CSR索引
//        // index_select(sortedCSC): 映射到CSC索引
//        Tensor perm = sortedCSR.argsort().index_select(0, sortedCSC);
//
//        // ========== 资源释放 ==========
//        srcNodes.close();
//        dstNodes.close();
//        sortedCSR.close();
//        srcCount.close();
//        sortedCSC.close();
//        dstCount.close();
//
//        return new Object[]{new Tensor[]{rowptr, col}, new Tensor[]{row, colptr}, perm};
//    }
//
//    /**
//     * 资源释放（避免内存泄漏）
//     */
////    @Override
////    protected void finalize() throws Throwable {
////        try {
////            if (lin != null) lin.close();
////            if (attSrc != null) attSrc.close();
////            if (attDst != null) attDst.close();
////        } finally {
////            super.finalize();
////        }
////    }
//
//    // ========== Getter（测试用） ==========
//    public long getHeads() { return heads; }
//    public long getOutChannels() { return outChannels; }
//    public boolean isConcat() { return concat; }
//}


//public class FusedGATConv extends Module {
//    public LinearImpl lin;
//    public Tensor attSrc;
//    public Tensor attDst;
//    private long heads;
//    private long outChannels;
//    private boolean concat;
//    private double negativeSlope;
//
//    public FusedGATConv(long inChannels, long outChannels, long heads, boolean concat, double negativeSlope) {
//        super();
//        this.heads = heads;
//        this.outChannels = outChannels;
//        this.concat = concat;
//        this.negativeSlope = negativeSlope;
//
//        // 权重层
//        this.lin = new LinearImpl(inChannels, heads * outChannels);
//
//        // 注意力参数
//        this.attSrc = torch.randn(new long[]{1, heads, outChannels});
//        this.attDst = torch.randn(new long[]{1, heads, outChannels});
//        torch.xavier_uniform_(this.attSrc);
//        torch.xavier_uniform_(this.attDst);
//
//        register_module("lin", lin);
//        register_parameter("att_src", attSrc);
//        register_parameter("att_dst", attDst);
//    }
//
//    /**
//     * @param x    节点特征 [N, In]
//     * @param csr  (rowptr, col) - 用于高效扫描源节点
//     * @param csc  (row, colptr) - 用于高效聚合到目标节点
//     * @param perm 重排索引，用于在 CSR 和 CSC 数据间快速同步
//     */
//    public Tensor forward(Tensor x, Tensor[] csr, Tensor[] csc, Tensor perm) {
//        long N = x.size(0);
//        Tensor rowptr = csr[0];
//        Tensor col = csr[1];
//        Tensor row = csc[0];
//        Tensor colptr = csc[1];
//
//        // 1. 线性投影 [N, H, C]
//        Tensor xLin = lin.forward(x).view(N, heads, outChannels);
//
//        // 2. 预计算节点端的注意力贡献 (基于 CSR/CSC 扫描)
//        Tensor alphaSrc = (xLin.mul(attSrc)).sum(-1); // [N, H]
//        Tensor alphaDst = (xLin.mul(attDst)).sum(-1); // [N, H]
//
//        // 3. Fused 传播逻辑
//        // 在 dgNN 论文中，这里会启动一个自定义 Kernel
//        // Java 层模拟：利用 csc 带来的列局部性进行聚合
//        Tensor out = aggregateFused(xLin, alphaSrc, alphaDst, row, colptr, N);
//
//        // 4. 合并多头
//        if (concat) {
//            return out.view(N, heads * outChannels);
//        } else {
//            return out.mean(1);
//        }
//    }
//
//    private Tensor aggregateFused(Tensor xLin, Tensor alphaSrc, Tensor alphaDst, Tensor row, Tensor colptr, long numNodes) {
//        // 利用 CSC 的 colptr 模拟无锁聚合
//        // 这避免了 COO 格式下 index_select 产生的大型中间边张量
//
//        // 展开 targetIdx [E]
//        Tensor targetIdx = torch.repeat_interleave(colptr.diff());
//
//        // 计算注意力系数（这里模拟了 Fusion，实际底层会合并这些步骤）
//        Tensor e_ij = alphaSrc.index_select(0, row).add(alphaDst.index_select(0, targetIdx));
//        e_ij = torch.leaky_relu(e_ij, new Scalar(negativeSlope));
//
//        // 基于目标节点局部性的 Softmax
//        Tensor alpha = scatter_softmax(e_ij, targetIdx, numNodes);
//
//        // 加权求和
//        Tensor msg = xLin.index_select(0, row).mul(alpha.unsqueeze(-1));
//        Tensor out = torch.zeros(new long[]{numNodes, heads, outChannels}, xLin.options());
//        out.scatter_add_(0, targetIdx.view(-1, 1, 1).expand_as(msg), msg);
//
//        return out;
//    }
//
//    /**
//     * 将 edge_index 转换为 FusedGAT 格式的静态工具方法
//     */
//    public static Object[] toGraphFormat(Tensor edge_index, long numNodes) {
//        // 1. 生成 CSR
//        Tensor sortedCSR = edge_index.select(0, 0).argsort();
//        Tensor col = edge_index.select(0, 1).index_select(0, sortedCSR);
//        Tensor rowptr = torch.cat(new TensorVector(
//                torch.zeros(new long[]{1}, col.options()),
//                torch.bincount(edge_index.select(0, 0), null, numNodes).cumsum(0)
//        ), 0);
//
//        // 2. 生成 CSC
//        Tensor sortedCSC = edge_index.select(0, 1).argsort();
//        Tensor row = edge_index.select(0, 0).index_select(0, sortedCSC);
//        Tensor colptr = torch.cat(new TensorVector(
//                torch.zeros(new long[]{1}, row.options()),
//                torch.bincount(edge_index.select(0, 1), null, numNodes).cumsum(0)
//        ), 0);
//
//        // 3. 生成 Permutation (CSR 边到 CSC 边的映射)
//        // 这一步在 dgNN 中用于在计算过程中快速索引
//        Tensor perm = sortedCSR.argsort().index_select(0, sortedCSC);
//
//        return new Object[]{ new Tensor[]{rowptr, col}, new Tensor[]{row, colptr}, perm };
//    }
//
//    private Tensor scatter_softmax(Tensor src, Tensor index, long dimSize) {
//        Tensor maxVal = Scatter.scatter(src, index, dimSize, "max");
//        Tensor out = src.sub(maxVal.index_select(0, index)).exp();
//        Tensor sum = Scatter.scatter(out, index, dimSize, "add");
//        return out.div(sum.index_select(0, index).add(new Scalar(1e-16)));
//    }
//}
