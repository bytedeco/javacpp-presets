package org.bytedeco.pytorch.geometric.nn.norm;
import org.bytedeco.pytorch.data.*;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.nn.Parameter;
import org.bytedeco.pytorch.geometric.utils.AggrUtils;

/**
 * MeanSubtractionNorm
 * 减去图内节点的均值 (Centering)
 */
public class MeanSubtractionNorm extends Module {

    public MeanSubtractionNorm() {
        super();
    }

    public Tensor forward(Tensor x, Tensor batch) {
        if (batch == null) {
            // Global Mean
            Tensor mean = x.mean(new long[]{0}, true, new ScalarTypeOptional(torch.ScalarType.Float)); // [1, C]
            return x.sub(mean);
        }

        long batchSize = batch.max().item().toLong() + 1;

        // 1. Calculate Mean per graph: [BatchSize, C]
        Tensor mean = AggrUtils.scatter(x, batch, batchSize, "mean");

        // 2. Broadcast back to nodes: [N, C]
        Tensor meanExpanded = mean.index_select(0, batch);

        // 3. Subtract
        return x.sub(meanExpanded);
    }
}