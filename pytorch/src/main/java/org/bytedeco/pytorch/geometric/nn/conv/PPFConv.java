package org.bytedeco.pytorch.geometric.nn.conv;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.global.torch;

/**
 * 实现 torch_geometric.nn.conv.PPFConv
 * 基于点对特征（位置+法向量夹角）的卷积算子，具有极强的旋转不变性。
 */
public class PPFConv extends MessagePassing {
    private Module localNN;  // 局部 MLP: h(x_j, PPF(i, j))
    private Module globalNN; // 全局 MLP: γ(...)
    private boolean addSelfLoops;

    public PPFConv(Module localNN, Module globalNN, boolean addSelfLoops) {
        super("max"); // 论文建议使用最大池化以获取鲁棒特征
        this.localNN = localNN;
        this.globalNN = globalNN;
        this.addSelfLoops = addSelfLoops;

        if (localNN != null) register_module("local_nn", localNN);
        if (globalNN != null) register_module("global_nn", globalNN);
    }

    @Override
    public Tensor forward(Tensor x, Tensor edge_index) {
        return forward(x, edge_index, (Tensor)null);
    }
    /**
     * @param x       节点特征 [N, C]
     * @param pos     节点位置 [N, 3]
     * @param normal  节点法向量 [N, 3]
     * @param edge_index 边索引 [2, E]
     */
    public Tensor forward(Tensor x, Tensor pos, Tensor normal, Tensor edge_index) {
        long N = pos.size(0);

        // 消息传递：输入包含特征、位置和法向量
        Tensor out = propagate(edge_index, x, pos, normal);

        if (globalNN != null) {
            out = globalNN.asSequential().forward(out);
        }

        return out;
    }

    /**
     * 重写 propagate 以计算 4 维 PPF 特征
     */
    public Tensor propagate(Tensor edge_index, Tensor x, Tensor pos, Tensor normal) {
        long N = pos.size(0);
        Tensor sourceIdx = edge_index.select(0, 0);
        Tensor targetIdx = edge_index.select(0, 1);

        // 获取 i 和 j 的位置与法向量
        Tensor pos_i = pos.index_select(0, targetIdx);
        Tensor pos_j = pos.index_select(0, sourceIdx);
        Tensor norm_i = normal.index_select(0, targetIdx);
        Tensor norm_j = normal.index_select(0, sourceIdx);

        // 1. 计算相对位移 d_ij = pos_j - pos_i
        Tensor rel_pos = pos_j.sub(pos_i);
        Tensor dist = rel_pos.norm(new ScalarOptional(new Scalar(2)), new long[]{-1}, true); // 距离 ||d_ij||
        Tensor d_norm = rel_pos.div(dist.clamp_min(new Scalar(1e-12))); // 单位化方向向量

        // 2. 计算 PPF 的四个分量:
        // f1: ||d_ij||
        // f2: angle(norm_i, d_ij)
        // f3: angle(norm_j, d_ij)
        // f4: angle(norm_i, norm_j)
        Tensor f2 = torch.atan2(torch.cross(norm_i, d_norm).norm(new ScalarOptional(new Scalar(2)), new long[]{-1}, true), (norm_i.mul(d_norm)).sum(new long[]{-1}, true, new ScalarTypeOptional(torch.kFloat())));
        Tensor f3 = torch.atan2(torch.cross(norm_j, d_norm).norm(new ScalarOptional(new Scalar(2)), new long[]{-1}, true), (norm_j.mul(d_norm)).sum(new long[]{-1}, true,new ScalarTypeOptional(torch.kFloat())));
        Tensor f4 = torch.atan2(torch.cross(norm_i, norm_j).norm(new ScalarOptional(new Scalar(2)), new long[]{-1}, true), (norm_i.mul(norm_j)).sum(new long[]{-1}, true,new ScalarTypeOptional(torch.kFloat())));

        Tensor ppf = torch.cat(new TensorVector(dist, f2, f3, f4), -1);

        // 3. 拼接节点特征 x_j (如果存在)
        Tensor msgInput = ppf;
        if (x != null) {
            msgInput = torch.cat(new TensorVector(x.index_select(0, sourceIdx), ppf), -1);
        }

        // 4. 局部非线性变换与聚合
        Tensor msg = localNN.asSequential().forward(msgInput);
        return aggregate(msg, targetIdx, N);
    }

    @Override
    public Tensor message(Tensor x_j, Tensor x_i, Tensor edge_index, Tensor edge_attr, long numNodes) {
        // 由于上面使用了自定义的 propagate_gated，这里的基类实现作为备用签名
        return x_j;
    }
}