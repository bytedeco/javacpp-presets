package org.bytedeco.pytorch.geometric.nn.conv;
import org.bytedeco.pytorch.nn.modules.*;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.global.torch;

import static org.bytedeco.pytorch.global.torch.kLong;

/**
 * 实现 torch_geometric.nn.conv.SplineConv
 * 基于 B-样条基函数的几何卷积算子。
 * 注意：输入 edge_attr (伪坐标) 必须在 [0, 1] 范围内。
 */
public class SplineConv extends MessagePassing {
    private long inChannels, outChannels;
    private int dim;
    private int kernelSize;
    private int degree;

    private Tensor weight;        // 样条基权重张量 [prod(kernel_size), inChannels, outChannels]
    private LinearImpl linRoot;       // 根节点变换
    private Tensor bias;

    public SplineConv(long inChannels, long outChannels, int dim, int kernelSize, int degree, boolean rootWeight, boolean hasBias) {
        super("mean");
        this.inChannels = inChannels;
        this.outChannels = outChannels;
        this.dim = dim;
        this.kernelSize = kernelSize;
        this.degree = degree;

        // 计算总权重网格大小 (对于多维情况是各维度的乘积)
        long totalKernelSize = (long) Math.pow(kernelSize, dim);

        // 权重张量: 对应样条插值网格上的每个控制点
        this.weight = torch.empty(new long[]{totalKernelSize, inChannels, outChannels});
        torch.xavier_uniform_(this.weight);
        register_parameter("weight", weight);

        if (rootWeight) {
            this.linRoot = new LinearImpl(inChannels, outChannels);
            register_module("lin_root", linRoot);
        }

        if (hasBias) {
            this.bias = torch.zeros(new long[]{outChannels});
            register_parameter("bias", bias);
        }
    }

    @Override
    public Tensor forward(Tensor x, Tensor edge_index) {
        return forward(x, edge_index, (Tensor)null);
    }
    
    public Tensor forward(Tensor x, Tensor edge_index, Tensor edge_attr) {
        long N = x.size(0);

        // 1. 计算样条基系数和控制点索引
        // 这里简化为 degree=1 的线性插值实现
        // 获取每个边对应的 B-样条权重值 (basis) 和权重索引 (weight_index)
        Object[] splineData = computeSpline(edge_attr);
        Tensor basis = (Tensor) splineData[0];        // [E, 2^dim]
        Tensor weightIdx = (Tensor) splineData[1];    // [E, 2^dim]

        // 2. 消息传递
        return propagate(edge_index, x, basis, weightIdx);
    }


    public Tensor propagate(Tensor edge_index, Tensor x, Tensor basis, Tensor weightIdx) {
        long N = x.size(0);
        long E = edge_index.size(1);
        Tensor sourceIdx = edge_index.select(0, 0);
        Tensor targetIdx = edge_index.select(0, 1);

        Tensor xj = x.index_select(0, sourceIdx); // [E, InC]
        Tensor msg = torch.zeros(new long[]{E, outChannels}, xj.options());

        // 样条插值核心循环
        for (int k = 0; k < basis.size(-1); k++) {
            // 1. 获取基函数系数并确保是 [E, 1]
            Tensor b = basis.select(-1, k).reshape(new long[]{E, 1});

            // 2. 核心修复：确保 wi 是一个 1D Vector
            // select 之后调用 reshape(-1) 强制转为一维向量
            Tensor wi = weightIdx.select(-1, k).reshape(new long[]{-1}).to(torch.kLong());

            // 3. 提取权重矩阵 [E, InC, OutC]
            // weight: [TotalKernelSize, InC, OutC]
            // wi: [E] -> w: [E, InC, OutC]
            Tensor w = weight.index_select(0, wi);

            // 4. 执行批量矩阵乘法 [E, 1, InC] x [E, InC, OutC] -> [E, 1, OutC]
            Tensor res = torch.matmul(xj.unsqueeze(1), w).reshape(new long[]{E, (int)outChannels});

            // 5. 累加消息
            msg = msg.add(res.mul(b));
        }

        Tensor out = aggregate(msg, targetIdx, N);

        if (linRoot != null) {
            out = out.add(linRoot.forward(x));
        }
        if (bias != null) {
            out = out.add(bias);
        }

        return out;
    }

    /**
     * 修正后的 computeSpline
     */
    private Object[] computeSpline(Tensor edge_attr) {
        // 假设 edge_attr: [E, dim]
        long E = edge_attr.size(0);

        // 映射到网格空间 [0, kernelSize - 1]
        Tensor scaled_attr = edge_attr.mul(new Scalar(kernelSize - 1));
        Tensor idx_left = scaled_attr.floor().to(torch.kLong());
        Tensor frac = scaled_attr.sub(idx_left.to(torch.kFloat()));

        // 这里以 1D 为例简化，实际多维需要 Cartesian Product (笛卡尔积)
        // 确保返回的 Tensor 形状为 [E, 2]
        Tensor basis = torch.stack(new TensorVector(frac.neg().add(new Scalar(1.0)), frac), -1);
        Tensor weightIdx = torch.stack(new TensorVector(idx_left, idx_left.add(new Scalar(1.0))), -1);

        // 如果维度是多维的，需要对索引进行偏移处理：idx = idx_d1 + idx_d2 * K + idx_d3 * K^2 ...
        // 这里暂时返回 1D 映射逻辑
        return new Object[]{ basis.reshape(new long[]{E, -1}), weightIdx.reshape(new long[]{E, -1}) };
    }

    /**
     * 简化的线性 B-样条计算 (1D 示例，多维需张量积)
     */
    private Object[] computeSpline2(Tensor edge_attr) {
        // 映射伪坐标到网格: 0.5 -> index=kernelSize*0.5
        Tensor scaled_attr = edge_attr.mul(new Scalar(kernelSize - 1));
        Tensor idx_left = scaled_attr.floor().to(kLong());
        Tensor frac = scaled_attr.sub(idx_left.to(torch.kFloat())); // 插值比例

        // 线性样条只需左右两个控制点
        // basis_left = 1 - frac, basis_right = frac
        return new Object[]{
                torch.stack(new TensorVector(frac.neg().add(new Scalar(1.0)), frac), -1),
                torch.stack(new TensorVector(idx_left, idx_left.add(new Scalar(1.0))), -1)
        };
    }

    public Tensor propagate3(Tensor edge_index, Tensor x, Tensor basis, Tensor weightIdx) {
        long N = x.size(0);
        long E = edge_index.size(1);
        Tensor sourceIdx = edge_index.select(0, 0);
        Tensor targetIdx = edge_index.select(0, 1);

        Tensor xj = x.index_select(0, sourceIdx); // [E, InC]
        Tensor msg = torch.zeros(new long[]{E, outChannels}, xj.options());

        // 样条插值核心循环
        for (int k = 0; k < basis.size(-1); k++) {
            // 1. 获取第 k 个基函数系数 [E, 1]
            Tensor b = basis.select(-1, k).unsqueeze(-1);

            // 2. 获取第 k 个权重网格索引 [E]
            // 修复点：不要使用 view(-1)，直接 select 得到的已经是 [E] 形状
            Tensor wi = weightIdx.select(-1, k).contiguous().to(kLong());

            // 3. 提取权重矩阵 [E, InC, OutC]
            // 这里的 weight 是 [TotalKernelSize, InC, OutC]
            Tensor w = weight.index_select(0, wi);

            // 4. 批量矩阵乘法 (BMM)
            // xj: [E, InC] -> unsqueeze(1) -> [E, 1, InC]
            // w:  [E, InC, OutC]
            // res: [E, 1, OutC]
            Tensor res = torch.matmul(xj.unsqueeze(1), w).squeeze(1);

            // 5. 累加到消息 [E, OutC]
            msg = msg.add(res.mul(b));
        }

        Tensor out = aggregate(msg, targetIdx, N);

        if (linRoot != null) out = out.add(linRoot.forward(x));
        if (bias != null) out = out.add(bias);

        return out;
    }
    public Tensor propagate2(Tensor edge_index, Tensor x, Tensor basis, Tensor weightIdx) {
        long N = x.size(0);
        Tensor sourceIdx = edge_index.select(0, 0);
        Tensor targetIdx = edge_index.select(0, 1);

        Tensor xj = x.index_select(0, sourceIdx); // [E, InC]

        // 初始化输出消息张量 [E, OutC]
        Tensor msg = torch.zeros(new long[]{xj.size(0), outChannels}, xj.options());

        // 样条插值核心循环
        // basis 形状: [E, K], weightIdx 形状: [E, K]
        // 对于线性样条，K = 2^dim
        for (int k = 0; k < basis.size(-1); k++) {
            // 1. 获取第 k 个基函数系数 [E, 1]
            Tensor b = basis.select(-1, k).unsqueeze(-1);

            // 2. 获取第 k 个权重网格索引 [E]
            // 必须连续化并转为 1D 向量，确保 index_select 只选出 E 个矩阵
            Tensor wi = weightIdx.select(-1, k).contiguous().view(new long[]{-1}).to(kLong());

            // 3. 提取权重矩阵 [E, InC, OutC]
            // weight 形状为 [TotalKernelSize, InC, OutC]
            Tensor w = weight.index_select(0, wi);

            // 4. 批量矩阵乘法 (BMM)
            // xj.unsqueeze(1) 是 [E, 1, InC]
            // w 是 [E, InC, OutC]
            // 结果 res 是 [E, 1, OutC] -> squeeze 后变为 [E, OutC]
            Tensor res = torch.matmul(xj.unsqueeze(1), w).squeeze(1);

            // 5. 按照基函数系数加权累加到 msg
            msg = msg.add(res.mul(b));
        }

        // 6. 执行图聚合 (基于 mean, sum 或 max)
        // 注意：SplineConv 的标准实现通常在聚合前应用消息，
        // 这里我们将插值后的 msg 进行聚合
        Tensor out = aggregate(msg, targetIdx, N);

        if (linRoot != null) out = out.add(linRoot.forward(x));
        if (bias != null) out = out.add(bias);

        return out;
    }

    @Override
    public Tensor message(Tensor x_j, Tensor x_i, Tensor edge_index, Tensor edge_attr, long numNodes) {
        return x_j;
    }
}



//    public Tensor propagate(Tensor edge_index, Tensor x, Tensor basis, Tensor weightIdx) {
//        long N = x.size(0);
//        Tensor sourceIdx = edge_index.select(0, 0);
//        Tensor targetIdx = edge_index.select(0, 1);
//
//        Tensor xj = x.index_select(0, sourceIdx); // [E, In]
//
//        // --- 样条内核插值计算 ---
//        // 对每一条边，根据其伪坐标在权重网格中进行插值
//        // 这里的逻辑模拟了：sum_k basis_k * (xj @ Weight[weightIdx_k])
//        Tensor msg = torch.zeros(new long[]{xj.size(0), outChannels}, xj.options());
//
////        for (int k = 0; k < basis.size(-1); k++) {
////            Tensor b = basis.select(-1, k).view(-1, 1); // 权重系数
////            Tensor wi = weightIdx.select(-1, k);        // 控制点索引
////
////            // 提取对应的权重矩阵并计算变换
////            // 注意：大规模下应使用批量 gather 进行优化
////            for (int i = 0; i < wi.size(0); i++) {
////                long currentWeightIdx = wi.select(0, i).item_long();
////                Tensor w = weight.select(0, currentWeightIdx);
////                // 仅为逻辑演示，实际应使用 batch matmul
////            }
////        }
////        for (int k = 0; k < basis.size(-1); k++) {
////            // 1. 获取第 k 个基函数的系数 [E, 1]
////            Tensor b = basis.select(-1, k).unsqueeze(-1);
////
////            // 2. 获取第 k 个权重索引 [E]
////            Tensor wi = weightIdx.select(-1, k);
////
////            // 3. 核心修复：批量提取权重矩阵
////            // 不要用循环调用 item_long()，使用 index_select
////            // weight 形状为 [TotalWeights, InC, OutC]
////            Tensor w = weight.index_select(0, wi); // 结果形状 [E, InC, OutC]
////
////            // 4. 计算：(x_j @ w) * b
////            // x_j 形状 [E, 1, InC]
////            Tensor res = torch.matmul(xj.unsqueeze(1), w).squeeze(1); // [E, OutC]
////            msg = msg.add(res.mul(b));
////        }
//
//        for (int k = 0; k < basis.size(-1); k++) {
//            // 1. 获取第 k 个基函数系数 [E, 1]
//            Tensor b = basis.select(-1, k).unsqueeze(-1);
//
//            // 2. 修复点：确保 wi 是一个 1D 向量 (Vector)
//            // 使用 .view(-1) 强制展平，并确保是 kLong 类型
//            Tensor wi = weightIdx.select(-1, k).contiguous().view(new long[]{-1}).to(kLong());
//
//            // 3. 执行 index_select (此时 wi 必然是 vector)
//            // weight 形状: [TotalWeights, InC, OutC]
//            Tensor w = weight.index_select(0, wi); // 结果: [E, InC, OutC]
//
//            // 4. 计算变换: (x_j @ w) * b
//            // x_j: [E, 1, InC], w: [E, InC, OutC]
//            Tensor res = torch.matmul(xj.unsqueeze(1), w).squeeze(1);
//            msg = msg.add(res.mul(b));
//        }
//        // 假设 basis 形状 [E, K], weightIdx 形状 [E, K]
////        for (int k = 0; k < basis.size(-1); k++) {
////            // 1. 获取第 k 个基函数的系数 [E, 1]
////            Tensor b = basis.select(-1, k).unsqueeze(-1);
////
////            // 2. 获取第 k 个权重索引 [E]
////            // 必须确保 wi 是 Long 且 形状为 [E]
////            Tensor wi = weightIdx.select(-1, k).contiguous().to(kLong());
////
////            // 3. 批量提取权重矩阵
////            // weight 形状: [TotalWeights, InC, OutC]
////            // w 形状: [E, InC, OutC]
////            Tensor w = weight.index_select(0, wi);
////
////            // 4. 检查维度对齐
////            // x_j 形状: [E, InC] -> [E, 1, InC]
////            Tensor x_j_expanded = xj.unsqueeze(1);
////
////            // 5. 执行批量矩阵乘法
////            // [E, 1, InC] @ [E, InC, OutC] -> [E, 1, OutC]
////            Tensor res = torch.matmul(x_j_expanded, w).squeeze(1);
////
////            // 6. 累加结果
////            msg = msg.add(res.mul(b));
////        }
////        for (int k = 0; k < basis.size(-1); k++) {
////            // 1. 获取第 k 个基函数的系数并确保维度为 [E, 1]
////            Tensor b = basis.select(-1, k).unsqueeze(-1);
////
////            // 2. 获取第 k 个权重索引，并强制展平为 1D 向量
////            // .contiguous() 确保内存连续，.view(-1) 确保是 vector
////            Tensor wi = weightIdx.select(-1, k).contiguous().view(new long[]{-1});
////
////            // 3. 批量提取权重矩阵
////            // weight 形状为 [TotalWeights, InC, OutC]
////            // wi 形状必须是 [E]
////            Tensor w = weight.index_select(0, wi.to(torch.kLong())); // 结果形状 [E, InC, OutC]
////
////            // 4. 执行批量矩阵乘法 (BMM)
////            // x_j: [E, InC] -> unsqueeze(1) -> [E, 1, InC]
////            // w: [E, InC, OutC]
////            // 结果: [E, 1, OutC] -> squeeze(1) -> [E, OutC]
////            Tensor res = torch.matmul(xj.unsqueeze(1), w).squeeze(1);
////
////            // 5. 加上带系数的结果
////            msg = msg.add(res.mul(b));
////        }
//
//        // 此处通常调用底层 C++ 算子实现 Fused Spline Kernel
//        // 在 Java 模拟中，我们直接实现最终聚合结果
//        Tensor out = aggregate(xj, targetIdx, N);
//
//        if (linRoot != null) out = out.add(linRoot.forward(x));
//        if (bias != null) out = out.add(bias);
//
//        return out;
//    }
