package org.bytedeco.pytorch.geometric.nn.conv;
import org.bytedeco.pytorch.nn.modules.LinearImpl;
import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.geometric.utils.Scatter;

public class GATConvV2 extends MessagePassing {
    private LinearImpl lin; // 线性变换层: [inChannels] → [heads*outChannels]
    private Tensor att;
    private long heads; // 注意力头数
    private long outChannels; // 单头输出维度
    private double negativeSlope;  // LeakyReLU负斜率
    private boolean concat; // 是否拼接多头结果（true:拼接，false:平均）

    public GATConvV2(long inChannels, long outChannels, long heads, double negativeSlope) {
        super("add");
        this.heads = heads;
        this.outChannels = outChannels;
        this.negativeSlope = negativeSlope;

        // 初始化线性层：输入 inChannels, 输出 heads * outChannels
        this.lin = new LinearImpl(inChannels, heads * outChannels);

        // 注意力向量 a: [1, heads, 2 * outChannels]
        this.att = torch.randn(new long[]{1, heads, 2 * outChannels});
        torch.xavier_uniform_(this.att);
        this.concat = true;

        register_module("lin", lin);
        register_parameter("att", att);
    }

    public GATConvV2(long inChannels, long outChannels, long heads, boolean concat, double negativeSlope) {
        super("add");
        this.heads = heads;
        this.outChannels = outChannels;
        this.negativeSlope = negativeSlope;

        // 初始化线性层：输入 inChannels, 输出 heads * outChannels
        this.lin = new LinearImpl(inChannels, heads * outChannels);

        // 注意力向量 a: [1, heads, 2 * outChannels]
        this.att = torch.randn(new long[]{1, heads, 2 * outChannels});
        torch.xavier_uniform_(this.att);
        this.concat = concat;

        register_module("lin", lin);
        register_parameter("att", att);
    }

    @Override
    public Tensor forward(Tensor x, Tensor edge_index) {
        // 设备/类型对齐（关键：避免维度不匹配的基础）
        x = x.to(new Device(torch.kCPU()),torch.kFloat());
        edge_index = edge_index.to(new Device(torch.kCPU()),torch.kLong());
        this.att = this.att.to(x.device(),x.dtype());

        long N = x.size(0);

        // 1. 线性变换: [N, inChannels] → [N, heads*outChannels]
        Tensor xLin = lin.forward(x);

        // 2. 重塑为多头形状: [N, heads, outChannels]
        xLin = xLin.view(N, heads, outChannels);

        // 🔴 核心修复：删除第三个参数，调用正确的 propagate 重载
        // 错误：propagate(edge_index, xLin, new long[]{x.size(0), x.size(0)});
        // 正确：仅传 edge_index 和 xLin，让 MessagePassing 自动推导目标维度
        Tensor out = propagate(edge_index, xLin);

        // 4. 多头结果处理（拼接/平均）
        if (concat) {
            // 拼接：[N, heads, outChannels] → [N, heads*outChannels]
            out = out.view(N, heads * outChannels);
        } else {
            // 平均：[N, heads, outChannels] → [N, outChannels]
            out = out.mean(1);
        }

        return out;
    }

    @Override
    public Tensor message(Tensor x_j, Tensor x_i, Tensor edge_index, Tensor edge_attr, long numNodes) {
        // x_i, x_j shape: [E, heads, outChannels]
        Tensor targetIdx = edge_index.select(0, 1);

        // 计算 e_ij = a^T [Wh_i || Wh_j]
        Tensor catFeat = torch.cat(new TensorVector(x_i, x_j), -1); // [E, heads, 2 * outChannels]

        // 与 att 做计算 [1, heads, 2 * outChannels]
        Tensor alpha = (catFeat.mul(this.att)).sum(-1); // [E, heads]
        alpha = torch.leaky_relu(alpha, new Scalar(negativeSlope));

        // 数值稳定的 Softmax
        alpha = scatter_softmax(alpha, targetIdx, numNodes);

        return x_j.mul(alpha.unsqueeze(-1));
    }

    public Tensor scatter_softmax(Tensor src, Tensor index, long numNodes) {
        // 修复：确保 src/index 设备/类型对齐
        src = src.to(new Device(torch.kCPU()),torch.kFloat());
        index = index.to(new Device(torch.kCPU()),torch.kLong());

        Tensor maxVal = Scatter.scatter(src, index, numNodes, "max");
        Tensor out = src.sub(maxVal.index_select(0, index)).exp();
        Tensor sum = Scatter.scatter(out, index, numNodes, "add");
        return out.div(sum.index_select(0, index).add(new Scalar(1e-16)));
    }

    // 🔴 修复：重写 update 方法，避免重复 view 导致维度错误
    @Override
    public Tensor update(Tensor inputs, Tensor x) {
        // inputs 已经是 [N, heads, outChannels]，无需再次 view
        return inputs;
    }

    // Getter
    public LinearImpl getLin() {
        return lin;
    }
    public long getHeads() {
        return heads;
    }
    public long getOutChannels() {
        return outChannels;
    }
    public double getNegativeSlope() {
        return negativeSlope;
    }
    public boolean isConcat() {
        return concat;
    }
    public Tensor getAttParam() {
        return att;
    }
}

