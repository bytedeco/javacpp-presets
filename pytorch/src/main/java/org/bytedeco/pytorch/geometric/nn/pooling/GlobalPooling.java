package org.bytedeco.pytorch.geometric.nn.pooling;

import org.bytedeco.pytorch.ScalarTypeOptional;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.geometric.utils.AggrUtils;

import static org.bytedeco.pytorch.global.torch.*;

/**
 * Global (graph-level) readout pooling.
 *
 * <pre>
 *   [N, C] + batch[N]  →  [num_graphs, C]
 * </pre>
 * Supports sum / mean / max. When {@code batch == null}, treats input as one graph
 * and returns shape {@code [1, C]}.
 */
public final class GlobalPooling {

    private GlobalPooling() {}

    /**
     * @param x     node features [N, C]
     * @param batch node→graph index [N], or null for a single graph
     * @param type  {@code "sum"} | {@code "mean"} | {@code "max"} (also add/avg aliases)
     */
    public static Tensor pool(Tensor x, Tensor batch, String type) {
        if (x == null || x.dim() != 2) {
            throw new IllegalArgumentException("x must be [N, C]");
        }
        if (type == null) {
            throw new IllegalArgumentException("type must not be null");
        }
        switch (type.toLowerCase()) {
            case "sum":
            case "add":
                return global_add_pool(x, batch);
            case "mean":
            case "avg":
            case "average":
                return global_mean_pool(x, batch);
            case "max":
                return global_max_pool(x, batch);
            default:
                throw new IllegalArgumentException(
                        "Unsupported pooling type='" + type + "' (use sum|mean|max)");
        }
    }

    public static Tensor global_add_pool(Tensor x, Tensor batch) {
        if (batch == null) {
            return x.sum(new long[]{0}, true, new ScalarTypeOptional());
        }
        batch = AggrUtils.asLongIndex(batch);
        long batchSize = batch.size(0) == 0 ? 1 : batch.max().item_long() + 1;
        return AggrUtils.scatter(x, batch, batchSize, "sum");
    }

    public static Tensor global_mean_pool(Tensor x, Tensor batch) {
        if (batch == null) {
            return x.mean(new long[]{0}, true, new ScalarTypeOptional());
        }
        batch = AggrUtils.asLongIndex(batch);
        long batchSize = batch.size(0) == 0 ? 1 : batch.max().item_long() + 1;
        return AggrUtils.scatter(x, batch, batchSize, "mean");
    }

    public static Tensor global_max_pool(Tensor x, Tensor batch) {
        if (batch == null) {
            return x.max(0, true).get0();
        }
        batch = AggrUtils.asLongIndex(batch);
        long batchSize = batch.size(0) == 0 ? 1 : batch.max().item_long() + 1;
        return AggrUtils.scatter(x, batch, batchSize, "max");
    }
}
