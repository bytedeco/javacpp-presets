package org.bytedeco.pytorch.geometric.aggr;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.geometric.utils.AggrUtils;
//import org.gnn.framework.utils.org.bytedeco.pytorch.geometric.utils.AggrUtils;


/**
 * Latent Core Matching org.bytedeco.pytorch.geometric.aggr.Aggregation
 * 将节点特征映射到 Latent Space，与 Cores 交互后再聚合。
 */
public class LCMAggregation extends Aggregation {
    private Tensor cores; // [NumCores, InChannels]  Parameter
    private long numCores;

    public LCMAggregation(long inChannels, long numCores) {
        this.numCores = numCores;
        this.cores = new Tensor(torch.randn(new long[]{numCores, inChannels})); //Parameter
        register_parameter("cores", cores);
    }

    @Override
    public Tensor forward(Tensor x, Tensor index, long dimSize) {
        // x: [N, C], cores: [K, C]

        // 1. Calculate similarity with cores: x @ cores^T -> [N, K]
        Tensor sim = x.matmul(cores.t());
        Tensor attention = torch.softmax(sim, 1); // Over cores

        // 2. Encode: x_encoded = attention @ cores -> [N, C]
        // 实际上这是一种 VQ (Vector Quantization) 的软版本
        Tensor xEncoded = attention.matmul(cores);

        // 3. Aggregate encoded features
        return AggrUtils.scatter(xEncoded, index, dimSize, "sum");
    }
}