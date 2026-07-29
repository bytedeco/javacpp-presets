package org.bytedeco.pytorch.geometric.nn.conv;
import org.bytedeco.pytorch.nn.modules.LinearImpl;
import org.bytedeco.pytorch.*;

import static org.bytedeco.pytorch.global.torch.relu;

public class SAGEConvV2 extends MessagePassing {

    private LinearImpl linL; // 邻居权重 W_neighbor
    private LinearImpl linR; // 自身权重 W_root
    private boolean normalize;
    //, boolean bias    
    public SAGEConvV2(long inDimSrc, long inDimDst, long outDim, boolean normalize, boolean bias) {
        super("mean");
        this.normalize = normalize;
        // 注意构造函数参数可能是 (in, out)
        this.linL = register_module("linL", new LinearImpl(inDimSrc, outDim));
        this.linR = register_module("linR", new LinearImpl(inDimDst, outDim));
    }

    public Tensor forward(Tensor xSrc, Tensor xDst, Tensor edge_index) {

        Tensor out = this.linL.forward(xSrc);
        // 2. 关键：传播消息。在二部图中，size 必须是 {srcNodes, dstNodes}
        // 这样 propagate 内部才知道将消息聚合到 100 个目标节点上
        long[] size = new long[]{xSrc.size(0), xDst.size(0)};
        out = propagate(edge_index, out, size);
        // 3. 结合目标节点自身的特征 (Update)
        Tensor xSelf = this.linR.forward(xDst);
        return out.add(xSelf);
    }
    /**
     * 必须匹配基类签名：(x_j, x_i, edge_index, edge_attr)
     * 哪怕 SAGE 只需要 x_j，参数也必须写全！
     */
    @Override
    public Tensor message(Tensor x_j, Tensor x_i, Tensor edge_index, Tensor edge_attr, long numNodes) {
        // GraphSAGE 的 message 就是邻居特征本身
        // 如果以后要支持带权重的 SAGE，可以在这里处理 edge_attr
        return x_j;
    }

    // 显式告诉 propagate 目标节点的数量是 xDst.size(0)
//        return propagate(edge_index, xSrc, xDst, xDst.size(0));

    @Override
    public Tensor forward(Tensor x, Tensor edge_index) {
        // 1. 聚合: [N, in] -> [N, in] (mean)
//        Tensor aggrOut = propagate(edge_index, x);
//
//        // 2. 变换: Wl * aggr + Wr * x
//        Tensor neighborFeat = linL.forward(aggrOut);
//        Tensor selfFeat = linR.forward(x);
//
//        return neighborFeat.add(selfFeat);
        // 1. 邻居聚合: [N, in] -> [N, in]
        // 这里会根据你的 MessagePassing 基类调用 scatter_mean
        Tensor out = propagate(edge_index, x, new long[]{x.size(0), x.size(0)});

        // 2. 变换邻居特征: W_l * aggr
        out = linL.forward(out);

        // 3. 融合自身特征: out = W_l * aggr + W_r * x
        // 注意：GraphSAGE 允许不带 root weight，但标准实现通常包含
        out = out.add(linR.forward(x));

        // 4. L2 归一化 (GraphSAGE 的标志性步骤)
        if (this.normalize) {
            // 对每一行计算 L2 范数并除以它
            out = relu(out); // 习惯上 GraphSAGE 之后接激活，但算子内只做归一化
            out = out.div(out.norm(new ScalarOptional(new Scalar(2)), new long[]{ -1}, true).clamp_min(new Scalar(1e-12)));
        }

        return out;
    }

    // 2. 带有默认值的简化构造函数
    public SAGEConvV2(long inDimSrc, long inDimDst, long outDim) {
        // 调用上面的构造函数，设置默认 normalize=false, bias=true
        this(inDimSrc, inDimDst, outDim, false, true);
    }
    public SAGEConvV2(long inDimSrc, long outDim) {
        // 调用上面的构造函数，设置默认 normalize=false, bias=true
        this(inDimSrc, outDim, outDim, false, true);
    }
}

