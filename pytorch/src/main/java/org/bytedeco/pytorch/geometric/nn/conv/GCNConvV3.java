package org.bytedeco.pytorch.geometric.nn.conv;
import org.bytedeco.pytorch.nn.modules.LinearImpl;
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.global.torch;

import static org.bytedeco.pytorch.geometric.utils.GraphUtils.add_self_loops;
import static org.bytedeco.pytorch.global.torch.*;

/**
 * 适配float类型的GCNConvV2实现
 * 核心调整：全量使用float类型，替换TensorOptions创建方式
 */
public class GCNConvV3 extends MessagePassing {

    private LinearImpl lin; // 线性变换层
    private long inChannels; // 输入特征维度
    private long outChannels; // 输出特征维度

    // ======================== 构造方法 ========================
    public GCNConvV3(Pointer p) {
        super(p); // 调用基类Pointer构造
    }

    public GCNConvV3(long inChannels, long outChannels) {
        super("add"); // GCN固定使用add聚合
        this.inChannels = inChannels;
        this.outChannels = outChannels;

        // 初始化线性层（float类型）
        this.lin = new LinearImpl(inChannels, outChannels);
        register_module("lin", lin); // 注册为子模块，管理参数
    }

    // ======================== 核心前向传播 ========================
    @Override
    public Tensor forward(Tensor x, Tensor edge_index) {
        // --------------- 防御性检查 ---------------
        if (x.dim() != 2) {
            throw new IllegalArgumentException("节点特征x必须是2维张量[N, F]，当前维度: " + x.dim());
        }
        if (edge_index.dim() != 2 || edge_index.size(0) != 2) {
            throw new IllegalArgumentException("边索引edge_index必须是[2, E]维度，当前维度: " + edge_index.dim());
        }
        if (x.size(1) != inChannels) {
            throw new IllegalArgumentException("输入特征维度不匹配：期望" + inChannels + "，实际" + x.size(1));
        }

        // 1. 线性变换：X * Theta → [N, outChannels]（确保float类型）
        lin.to(x.device(),ScalarType.Float,false);
        Tensor xTransformed = lin.forward(x);

        // 2. 添加自环：A → A + I（GCN核心步骤）
        edge_index = add_self_loops(edge_index, x.size(0));
        // 确保edge_index是Long类型（index_select要求）
        edge_index = edge_index.to(kLong());

        // 3. 计算GCN对称归一化系数（float类型）
        Tensor norm = gcn_norm(edge_index, x.size(0), kFloat(), x.device());

        // 4. 关键：调用带edge_attr的propagate重载（norm作为edge_attr传入）
        return propagate(edge_index, xTransformed, norm);
    }

    // ======================== 必须重写的抽象message方法 ========================
    @Override
    public Tensor message(Tensor x_j, Tensor x_i, Tensor edge_index, Tensor edge_attr, long numNodes) {
        // edge_attr就是归一化系数norm，确保float类型+维度广播
        if (edge_attr != null) {
            // 转为float类型，重塑为[E,1]支持广播乘法
            Tensor norm = edge_attr.to(kFloat()).view(-1, 1);
            return x_j.mul(norm);
        }
        return x_j;
    }

    // ======================== 辅助方法：计算GCN归一化系数（全float类型） ========================
    private Tensor gcn_norm(Tensor edge_index, long numNodes, torch.ScalarType dtype, Device device) {
        // 1. 提取源/目标节点索引（已确保是Long类型）
        Tensor row = edge_index.select(0, 0); // 源节点（j）
        Tensor col = edge_index.select(0, 1); // 目标节点（i）

        // 2. 初始化度张量（float类型，与输入设备一致）
        TensorOptions floatOptions = new TensorOptions().dtype(new ScalarTypeOptional(dtype)).device(new DeviceOptional(device));
        Tensor deg_src = zeros(new long[]{numNodes}, floatOptions);
        Tensor deg_dst = zeros(new long[]{numNodes}, floatOptions);
        Tensor ones = ones(new long[]{row.size(0)}, floatOptions);

        // 3. 计算源节点度和目标节点度（对称归一化核心）
        deg_src.scatter_add_(0, row, ones);
        deg_dst.scatter_add_(0, col, ones);

        // 4. 计算D^-0.5，处理度为0的情况（float类型）
        Tensor deg_src_inv_sqrt = deg_src.pow(new Scalar(-0.5f));
        Tensor deg_dst_inv_sqrt = deg_dst.pow(new Scalar(-0.5f));
        // 将无穷大替换为0（float类型的0）
        deg_src_inv_sqrt.masked_fill_(deg_src_inv_sqrt.isinf(), new Scalar(0.0f));
        deg_dst_inv_sqrt.masked_fill_(deg_dst_inv_sqrt.isinf(), new Scalar(0.0f));

        // 5. 计算每条边的归一化系数（float类型）
        Tensor norm = deg_src_inv_sqrt.index_select(0, row).mul(deg_dst_inv_sqrt.index_select(0, col));
        return norm;
    }

    // ======================== Getter方法 ========================
    public long getInChannels() {
        return inChannels;
    }

    public long getOutChannels() {
        return outChannels;
    }

    public LinearImpl getLin() {
        return lin;
    }
}
