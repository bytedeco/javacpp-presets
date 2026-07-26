package org.bytedeco.pytorch.geometric.aggr;
import org.bytedeco.pytorch.nn.options.*;
import org.bytedeco.pytorch.c10.*;
import org.bytedeco.pytorch.nn.modules.*;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.nn.options.LSTMOptions;
import org.bytedeco.pytorch.geometric.utils.AggrUtils;
//import org.gnn.framework.utils.org.bytedeco.pytorch.geometric.utils.AggrUtils;

/**
 * LSTM org.bytedeco.pytorch.geometric.aggr.Aggregation
 * 将邻居视为序列，通过 LSTM 聚合。
 * 适用于邻居特征有隐含顺序或特征复杂的场景。
 */
public class LSTMAggregation extends Aggregation {
    private LSTMImpl lstm;
    private long inChannels;
    private long outChannels;

    public LSTMAggregation(long inChannels, long outChannels) {
        this.inChannels = inChannels;
        this.outChannels = outChannels;

        // batch_first=true: 输入格式 [Batch, Seq, Feat]
        LSTMOptions opts = new LSTMOptions(inChannels, outChannels);
        opts.batch_first().put(true);
        this.lstm = new LSTMImpl(opts);

        register_module("lstm", lstm);
    }

    @Override
    public Tensor forward(Tensor x, Tensor index, long dimSize) {
        // 1. 转为 Dense: [N, MaxDeg, F], 填充 0
        Tensor[] denseData = AggrUtils.to_dense_batch(x, index, dimSize, 0.0f);
        Tensor denseX = denseData[0]; // [N, MaxDeg, In]
        Tensor lengths = denseData[2]; // [N]

        long maxDeg = denseX.size(1);

        // 如果 maxDeg == 0 (没有任何边)，直接返回 0
        if (maxDeg == 0) {
            return torch.zeros(new long[]{dimSize, outChannels}, x.options());
        }

        // 2. LSTM Forward
        // 输入: [N, MaxDeg, In]
        // 输出 Tuple: (output, (h_n, c_n))
        // output: [N, MaxDeg, Out]
        // h_n: [1, N, Out]
        T_TensorT_TensorTensor_T_T ret = lstm.forwardT_TensorT_TensorTensor_T_T(denseX);
        Tensor output = ret.get0();

        // 3. 提取有效输出
        // 我们不能简单取最后一个时间步 (output[:, -1, :])，因为有 Padding
        // 我们需要取 lengths - 1 位置的输出

        // 构造 Gather Index
        // lengths: [N] -> indices: [N, 1, 1]
        // 我们要取的时间步索引是 lengths - 1 (注意要 clamp_min(0) 防止 -1)
        Tensor idx = lengths.sub(new Scalar(1)).clamp_min(new Scalar(0)).view(dimSize, 1, 1);

        // 扩展 Index 到特征维度 [N, 1, Out]
        Tensor gatherIdx = idx.expand(new long[]{dimSize, 1, outChannels});

        // gather(dim=1, index) -> [N, 1, Out]
        Tensor finalOut = output.gather(1, gatherIdx).squeeze(1);

        // 4. 处理度为 0 的节点 (lengths=0)
        // 它们应该保持为 0，但上面 gather 可能取到了 idx=0 的脏数据
        // Mask: lengths > 0
        Tensor mask = lengths.gt(new Scalar(0)).unsqueeze(1).expand_as(finalOut);

        return finalOut.mul(mask.to(torch.ScalarType.Float));
    }
}