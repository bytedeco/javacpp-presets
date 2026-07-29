package org.bytedeco.pytorch.geometric.nn.conv;
import org.bytedeco.pytorch.nn.modules.LinearImpl;
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.global.torch;

import static org.bytedeco.pytorch.global.torch.*;
import static org.bytedeco.pytorch.geometric.utils.GraphUtils.add_self_loops;

public class GCNConvV2 extends MessagePassing {

    private LinearImpl lin;
    private long inChannels;
    private long outChannels;

    public GCNConvV2(Pointer p) {
        super(p);
    }

    public GCNConvV2(long inChannels, long outChannels) {
        super("add");
        this.inChannels = inChannels;
        this.outChannels = outChannels;
        this.lin = new LinearImpl(inChannels, outChannels);
        register_module("lin", lin);
    }

    @Override
    public Tensor forward(Tensor x, Tensor edge_index) {
        // 1. 防御性检查
        if (x.dim() != 2 || edge_index.dim() != 2 || edge_index.size(0) != 2) {
            throw new IllegalArgumentException("输入维度错误！x=[N,F], edge_index=[2,E]");
        }
        TensorOptions floatOpts = new TensorOptions().dtype(new ScalarTypeOptional(kFloat())).device(new DeviceOptional(new Device(DeviceType.CPU)));

        // 2. 线性变换（强制float类型）
        lin.to(x.device(),ScalarType.Float,false);
        Tensor weight = eye(inChannels, outChannels, floatOpts);
        lin.weight().set_data(weight);
        lin.bias().set_data(zeros(new long[]{outChannels}, floatOpts));
        Tensor xTransformed = lin.forward(x).to(kFloat());

        // 3. 先加自环，再计算归一化（关键：自环必须参与度统计）
        edge_index = add_self_loops(edge_index, x.size(0)).to(kLong());

        // 4. 计算对称归一化系数（核心修复）
        Tensor norm = compute_gcn_norm(edge_index, x.size(0), x.device());

        // 5. 消息传递（确保norm是float类型）
        return propagate(edge_index, xTransformed, norm.to(kFloat()));
    }

    /**
     * 核心修复：正确计算GCN对称归一化系数
     * 公式：norm_j,i = 1 / sqrt(deg[j]) * 1 / sqrt(deg[i])
     */
    private Tensor compute_gcn_norm(Tensor edge_index, long numNodes, Device device) {
//        TensorOptions longOpts = new TensorOptions().dtype(kLong()).device(device);
        TensorOptions floatOpts = new TensorOptions().dtype(new ScalarTypeOptional(kFloat())).device(new DeviceOptional(device));

        // 1. 提取源/目标节点索引（add_self_loops后的完整边）
        Tensor row = edge_index.select(0, 0).to(kLong()); // 源节点 j
        Tensor col = edge_index.select(0, 1).to(kLong()); // 目标节点 i

        // 2. 计算节点度（包含自环，对称统计）
        Tensor deg = zeros(new long[]{numNodes}, floatOpts);
        Tensor ones = ones(new long[]{row.size(0)}, floatOpts);

        // 关键：GCN的度是"入度+出度"（无向图），所以要累加row和col
        deg.scatter_add_(0, row, ones); // 源节点度
        deg.scatter_add_(0, col, ones); // 目标节点度 → 合并为总度

        // 3. 计算D^-0.5，处理度为0的情况
        Tensor deg_inv_sqrt = deg.pow(new Scalar(-0.5f));
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt.isinf(), new Scalar(0.0f));

        // 4. 计算每条边的归一化系数（对称）
        Tensor norm_j = deg_inv_sqrt.index_select(0, row);
        Tensor norm_i = deg_inv_sqrt.index_select(0, col);
        return norm_j.mul(norm_i);
    }

    @Override
    public Tensor message(Tensor x_j, Tensor x_i, Tensor edge_index, Tensor edge_attr, long numNodes) {
        if (edge_attr != null) {
            // 确保维度广播：[E] → [E,1]
            return x_j.mul(edge_attr.view(-1, 1).to(kFloat()));
        }
        return x_j;
    }

    // Getter
    public LinearImpl getLin() {
        return lin;
    }

    public long getInChannels() {
        return inChannels;
    }

    public long getOutChannels() {
        return outChannels;
    }
}