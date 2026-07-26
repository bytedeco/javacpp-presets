package org.bytedeco.pytorch.geometric.nn.pooling;

import org.bytedeco.pytorch.*;
import static org.bytedeco.pytorch.global.torch.*;

public class DensePooling {

    /**
     * Dense DiffPool (Differentiable Pooling)
     * 核心公式: X' = S^T * X,  A' = S^T * A * S
     */
    public static Tensor[] dense_diff_pool(Tensor x, Tensor adj, Tensor s) {
        // x: [B, N, D], adj: [B, N, N], s: [B, N, M]
        s = softmax(s, -1); // 确保分配矩阵每一行和为1

        // 1. 计算粗化后的特征: [B, M, D]
        Tensor out_x = s.transpose(1, 2).matmul(x);

        // 2. 计算粗化后的邻接矩阵: [B, M, M]
        Tensor out_adj = s.transpose(1, 2).matmul(adj).matmul(s);

        return new Tensor[]{out_x, out_adj};
    }

    /**
     * Dense MinCutPool
     * 基于谱聚类思想，不仅输出粗化图，还计算辅助损失以优化切分
     */
    public static Tensor[] dense_mincut_pool(Tensor x, Tensor adj, Tensor s) {
        s = softmax(s, -1); // [B, N, M]
        long B = s.size(0);
        long N = s.size(1);
        long M = s.size(2);

        // 1. 池化逻辑
        Tensor out_x = s.transpose(1, 2).matmul(x);
        Tensor out_adj = s.transpose(1, 2).matmul(adj).matmul(s);

        // 2. MinCut Loss 计算
        // num: Trace(S^T A S) -> 提取对角线求和
        Tensor num = out_adj.diagonal(0, 1, 2).sum();

        // den: Trace(S^T D S) 
        // d 原本计算: adj.sum(2, false) -> [B, N]
        // 显式 view 确保它是 [B, N, 1] 而不是 [B, N, 1, 1]
        Tensor d = adj.sum(new long[]{2}, true, new ScalarTypeOptional()).view(new long[]{B, N, 1});

        // 关键修正：不再使用容易出错的 expand 数组传递，改用 mul 的自动广播
        // 在 JavaCPP 中，如果维度是 [B, N, M] * [B, N, 1]，mul 是非常稳定的
        Tensor den = s.pow(new Scalar(2)).mul(d).sum();

        // 3. 计算 Cut Loss
        Tensor cut_loss = num.div(den.add(new Scalar(1e-6))).neg();

        // 4. Orthogonality Loss (修复 Identity 的维度)
        Tensor st_s = s.transpose(1, 2).matmul(s);
        // 直接生成 [M, M] 单位阵，mul 会自动处理 Batch 广播，不需要手动 expand
        Tensor identity = eye(M, s.options());

        // Frobenius Norm: ||S^T S - I||_F
        Tensor ortho_loss = norm(st_s.sub(identity));

        return new Tensor[]{out_x, out_adj, cut_loss.add(ortho_loss)};
    }
    
    public static Tensor[] dense_mincut_pool_bak4(Tensor x, Tensor adj, Tensor s) {
        // s: [B, N, M]
        s = softmax(s, -1);
        long B = s.size(0);
        long N = s.size(1);
        long M = s.size(2);

        // 1. 池化特征与邻接矩阵
        Tensor out_x = s.transpose(1, 2).matmul(x); // [B, M, D]
        Tensor out_adj = s.transpose(1, 2).matmul(adj).matmul(s); // [B, M, M]

        // 2. MinCut Loss 计算
        // 分子 num = Trace(Sum_over_B(out_adj))
        // 因为 trace 不支持 Batch，我们手动计算对角线和
        Tensor num = out_adj.diagonal(0, 1, 2).sum();

        // 分母 den = Trace(S^T D S)
        // d: [B, N] -> [B, N, 1]
        Tensor d = adj.sum(new long[]{2}, true, new ScalarTypeOptional()).unsqueeze(-1);

        // 关键修正：显式展开 d 以匹配 s.pow(2) 的形状 [B, N, M]
        Tensor s_sq = s.pow(new Scalar(2));
        Tensor d_exp = d.expand(new long[]{B, N, M});

        // 现在进行点乘，维度完全一致
        Tensor den = s_sq.mul(d_exp).sum();

        // 3. 计算 Cut Loss
        Tensor cut_loss = num.div(den.add(new Scalar(1e-6))).neg();

        // 4. Orthogonality Loss: 保持簇的独立性
        // 使得 S^T S 接近单位阵
        Tensor st_s = s.transpose(1, 2).matmul(s); // [B, M, M]
        Tensor identity = eye(M, s.options()).unsqueeze(0).expand(new long[]{B, M, M});

        // 计算 Frobenius 范数: ||S^T S - I||_F
        Tensor ortho_loss = norm(st_s.sub(identity)).div(new Scalar(B * M));
        // 矩阵 Frobenius 范数简化计算
//        Tensor ortho_loss = norm(st_s.div(norm(st_s)).sub(identity.div(norm(identity))));

        return new Tensor[]{out_x, out_adj, cut_loss.add(ortho_loss)};
    }
    public static Tensor[] dense_mincut_pool_bak3(Tensor x, Tensor adj, Tensor s) {
        // 确保 s 是经过 Softmax 归一化的分配矩阵 [B, N, M]
        s = softmax(s, -1);

        // 1. 特征池化: [B, M, D]
        // s.transpose(1, 2) 是 [B, M, N], x 是 [B, N, D]
        Tensor out_x = s.transpose(1, 2).matmul(x);

        // 2. 邻接矩阵池化: [B, M, M]
        // out_adj = S^T * A * S
        Tensor out_adj = s.transpose(1, 2).matmul(adj).matmul(s);

        // 3. MinCut Loss 计算 (修复报错的核心部分)

        // 分子: Tr(S^T A S) -> 
        // 直接取池化后邻接矩阵的迹。由于是 Batch，先对 Batch 求和再取迹，或者取均值
        Tensor num = trace(out_adj.sum(0));

        // 分母: Tr(S^T D S) -> 
        // d 是度向量 [B, N] (按行求和)
        Tensor d = adj.sum(new long[]{2}, true, new ScalarTypeOptional());

        // 重点：使用点乘代替矩阵乘法以绕过广播错误
        // Tr(S^T D S) 等价于 sum( (S .* S) * d_expanded )
        // s.pow(2) 是 [B, N, M]
        // d.unsqueeze(-1) 是 [B, N, 1]
        // 这里的 mul 会自动广播 [B, N, M] * [B, N, 1] -> [B, N, M]
        Tensor den = s.pow(new Scalar(2)).mul(d.unsqueeze(-1)).sum();

        // 计算 Cut Loss
        Tensor cut_loss = num.div(den.add(new Scalar(1e-6))).neg();

        // 4. 计算 Orthogonality Loss (MinCut 通常需要的辅助项)
        // 使得 S 尽量接近正交分配，防止所有节点聚类到同一个簇
        Tensor st_s = s.transpose(1, 2).matmul(s); // [B, M, M]
        long M = s.size(2);
        Tensor identity = eye(M, s.options());

        // 矩阵 Frobenius 范数简化计算
        Tensor ortho_loss = norm(st_s.div(norm(st_s)).sub(identity.div(norm(identity))));

        // 返回：粗化特征, 粗化邻接阵, 综合 Loss
        return new Tensor[]{out_x, out_adj, cut_loss.add(ortho_loss)};
    }
    public static Tensor[] dense_mincut_pool_bak2(Tensor x, Tensor adj, Tensor s) {
        s = softmax(s, -1); // [B, N, M]

        // 1. 特征与邻接矩阵粗化
        Tensor out_x = s.transpose(1, 2).matmul(x); // [B, M, D]
        Tensor out_adj = s.transpose(1, 2).matmul(adj).matmul(s); // [B, M, M]

        // 2. MinCut Loss 计算 (修复维度匹配)
        // d: 按行求和得到度向量 [B, N]
//        Tensor d = adj.sum(2, false);
        Tensor d = adj.sum(new long[]{2}, true, new ScalarTypeOptional());
        // D_matrix: [B, N, N] 对角矩阵
        Tensor d_diag = d.diag_embed(0, 1, 2);

        // 计算分母 trace(S^T D S)
        // s.t() @ d_diag @ s -> [B, M, M]
        Tensor st_d_s = s.transpose(1, 2).matmul(d_diag).matmul(s);

        // 对 Batch 维度求和后取 Trace
        // 注意：trace 只能作用于 2D 矩阵
        Tensor num = trace(out_adj.sum(0));
        Tensor den = trace(st_d_s.sum(0));

        // 防止除以0
        Tensor cut_loss = num.div(den.add( new Scalar(1e-6))).neg();

        return new Tensor[]{out_x, out_adj, cut_loss};
    }
    public static Tensor[] dense_mincut_pool_bak(Tensor x, Tensor adj, Tensor s) {
        s = softmax(s, -1);

        // 核心池化逻辑同 DiffPool
        Tensor out_x = s.transpose(1, 2).matmul(x);
        Tensor out_adj = s.transpose(1, 2).matmul(adj).matmul(s);

        // 计算 MinCut Loss (用于最小化切割权重)
        // Loss = -trace(S^T A S) / trace(S^T D S)
        Tensor d = adj.sum(new long[]{2}, true, new ScalarTypeOptional());
        Tensor num = trace(out_adj.sum(new long[]{0}, false, new ScalarTypeOptional()));
        Tensor den = trace(s.transpose(1, 2).matmul(d.diag_embed()).matmul(s).sum(0));
        Tensor cut_loss = num.div(den).neg();

        return new Tensor[]{out_x, out_adj, cut_loss};
    }

    /**
     * DMoNPooling (Deep Modularity Networks)
     * 基于模块度（Modularity）优化图聚类
     */
    public static Tensor[] dmon_pooling(Tensor x, Tensor adj, Tensor s) {
        s = softmax(s, -1);

        // 粗化逻辑
        Tensor out_x = s.transpose(1, 2).matmul(x);
        Tensor out_adj = s.transpose(1, 2).matmul(adj).matmul(s);

        // 计算模块度损失 (Spectral Modularity)
        // 衡量簇内连接是否比随机图更紧密
        long degrees = (long) adj.sum().item_float();
        Tensor d = adj.sum(new long[]{2}, true, new ScalarTypeOptional());
        Tensor expected_adj = d.matmul(d.transpose(1, 2)).div(new Scalar(degrees));
        Tensor modularity = trace(s.transpose(1, 2).matmul(adj.sub(expected_adj)).matmul(s).sum(0));
        Tensor dmon_loss = modularity.div( new Scalar(2.0 * degrees)).neg();

        return new Tensor[]{out_x, out_adj, dmon_loss};
    }
}