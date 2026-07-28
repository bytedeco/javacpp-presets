package org.bytedeco.pytorch.geometric.nn.norm;

import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.geometric.utils.AggrUtils;
import org.bytedeco.pytorch.nn.Module;

/**
 * Graph-size normalization: scale features by {@code 1/√|V_g|} per graph.
 *
 * <pre>
 *   x'_i = x_i / √ n_{batch(i)}
 * </pre>
 * Common in graph Transformers. When {@code batch == null}, uses global N.
 */
public class GraphSizeNorm extends Module {

    public GraphSizeNorm() {
        super();
    }

    /** Single-graph convenience. */
    public Tensor forward(Tensor x) {
        return forward(x, (Tensor) null);
    }

    /**
     * @param x     [N, C]
     * @param batch [N] long graph ids, or null
     */
    public Tensor forward(Tensor x, Tensor batch) {
        if (x == null || x.dim() != 2) {
            throw new IllegalArgumentException("x must be [N, C]");
        }
        if (batch == null) {
            double scale = 1.0 / Math.sqrt(Math.max(1, x.size(0)));
            return x.mul(new Scalar(scale));
        }
        batch = AggrUtils.asLongIndex(batch);
        long numGraphs = batch.size(0) == 0 ? 1 : batch.max().item_long() + 1;
        Tensor graphSizes = AggrUtils.compute_degree(batch, numGraphs); // [G]
        Tensor scale = graphSizes.index_select(0, batch).rsqrt();       // [N]
        return x.mul(scale.unsqueeze(1));
    }
}
