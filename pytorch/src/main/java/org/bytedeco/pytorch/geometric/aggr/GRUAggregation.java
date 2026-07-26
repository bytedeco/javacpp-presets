package org.bytedeco.pytorch.geometric.aggr;
import org.bytedeco.pytorch.nn.options.*;
import org.bytedeco.pytorch.c10.*;
import org.bytedeco.pytorch.nn.modules.*;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.nn.options.GRUOptions;
import org.bytedeco.pytorch.geometric.utils.AggrUtils;
//import org.gnn.framework.utils.org.bytedeco.pytorch.geometric.utils.AggrUtils;

/**
 * GRU org.bytedeco.pytorch.geometric.aggr.Aggregation
 * 使用 GRU 处理邻居序列。
 * 输入: [N, C]
 * 输出: [N, OutChannels]
 */
public class GRUAggregation extends Aggregation {
    private GRUImpl gru;
    private long outChannels;

    public GRUAggregation(long inChannels, long outChannels) {
        this.outChannels = outChannels;

        // GRU 配置: batch_first=true -> [Batch, Seq, Feat]
        GRUOptions opts = new GRUOptions(inChannels, outChannels);
        opts.batch_first().put(true);
        this.gru = new GRUImpl(opts);

        register_module("gru", gru);
    }

    @Override
    public Tensor forward(Tensor x, Tensor index, long dimSize) {
        // 1. 稀疏转稠密: [N, MaxDeg, In]
        // 填充 0.0
        Tensor[] denseData = AggrUtils.to_dense_batch(x, index, dimSize, 0.0f);
        Tensor denseX = denseData[0];
        Tensor lengths = denseData[2]; // [N]

        long maxDeg = denseX.size(1);
        if (maxDeg == 0) {
            return torch.zeros(new long[]{dimSize, outChannels}, x.options());
        }

        // 2. GRU Forward
        // GRU 返回 Tuple (output, h_n)
        // output: [N, MaxDeg, Out]
        T_TensorTensor_T ret = gru.forwardT_TensorTensor_T(denseX);
        Tensor output = ret.get0();

        // 3. 提取有效输出 (Gather Last Valid Timestep)
        // 我们需要取 lengths - 1 位置的输出

        // idx = clamp(lengths - 1, min=0).view(N, 1, 1)
        Tensor idx = lengths.sub(new Scalar(1)).clamp_min(new Scalar(0)).view(dimSize, 1, 1);

        // 扩展到特征维度: [N, 1, Out]
        Tensor gatherIdx = idx.expand(new long[]{dimSize, 1, outChannels});

        // gather: [N, 1, Out] -> squeeze -> [N, Out]
        Tensor finalOut = output.gather(1, gatherIdx).squeeze(1);

        // 4. 处理度为0的节点 (mask out)
        Tensor mask = lengths.gt(new Scalar(0)).unsqueeze(1).expand_as(finalOut);

        return finalOut.mul(mask.to(torch.ScalarType.Float));
    }
}