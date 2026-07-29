package org.bytedeco.pytorch.geometric.transforms;

import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorVector;
import org.bytedeco.pytorch.geometric.data.GraphData;
import org.bytedeco.pytorch.geometric.utils.AggrUtils;

import static org.bytedeco.pytorch.global.torch.cat;
import static org.bytedeco.pytorch.global.torch.stack;

/**
 * LocalDegreeProfile (LDP): append per-node degree statistics.
 *
 * <p>Features concatenated onto {@code x}:
 * {@code [deg, min(deg_nbr), max(deg_nbr), mean(deg_nbr), std(deg_nbr)]}
 * (5 columns). Isolated nodes get neighbor stats = 0.
 *
 * <p>Aligned with PyG {@code LocalDegreeProfile}.
 */
public class LocalDegreeProfile implements BaseTransform {
    @Override
    public GraphData apply(GraphData data) {
        Tensor x = TransformUtils.requireX(data);
        Tensor ei = TransformUtils.requireEdgeIndex(data);
        long N = x.size(0);

        // Degree from target (col) indices — same convention as GCNNorm / TransformUtils.degree
        Tensor col = TransformUtils.col(ei);
        Tensor deg = TransformUtils.degree(col, N, x); // [N], float-like

        // Neighbor degree messages: for each edge j→i, message = deg[j]
        Tensor row = TransformUtils.row(ei);
        Tensor degSrc = deg.index_select(0, row); // [E]

        // min / max / mean / std of neighbor degrees via scatter
        Tensor degMin = AggrUtils.scatter(degSrc, col, N, "min");
        Tensor degMax = AggrUtils.scatter(degSrc, col, N, "max");
        Tensor degMean = AggrUtils.scatter(degSrc, col, N, "mean");
        // std = sqrt(mean(x^2) - mean(x)^2), clamp for numerical stability
        Tensor degSqMean = AggrUtils.scatter(degSrc.mul(degSrc), col, N, "mean");
        Tensor degVar = degSqMean.sub(degMean.mul(degMean)).clamp_min(new Scalar(0.0));
        Tensor degStd = degVar.sqrt();

        // Isolated nodes: AggrUtils leaves 0 for sum/mean; min/max also 0 after our policy.
        Tensor profile = stack(new TensorVector(
                deg, degMin, degMax, degMean, degStd
        ), 1); // [N, 5]

        data.x = cat(new TensorVector(x, profile.to(x.dtype())), 1);
        return data;
    }
}
