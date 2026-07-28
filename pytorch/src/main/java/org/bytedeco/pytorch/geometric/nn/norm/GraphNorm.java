package org.bytedeco.pytorch.geometric.nn.norm;

import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.ScalarTypeOptional;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorOptions;
import org.bytedeco.pytorch.nn.Parameter;
import org.bytedeco.pytorch.geometric.utils.AggrUtils;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.nn.Module;

/**
 * GraphNorm (Cai et al.): per-graph feature normalization with learnable mean shift.
 *
 * <pre>
 *   μ_g, σ²_g  computed over nodes of graph g (batch vector)
 *   y = ((x − α ⊙ μ) / √(σ² + ε)) ⊙ γ + β
 * </pre>
 * When {@code batch == null}, treats the whole tensor as one graph.
 */
public class GraphNorm extends Module {

    private final long inChannels;
    private final double eps;
    private final Parameter weight;     // γ [C]
    private final Parameter bias;       // β [C]
    private final Parameter meanScale;  // α [C]

    public GraphNorm(long inChannels) {
        this(inChannels, 1e-5);
    }

    public GraphNorm(long inChannels, double eps) {
        super();
        if (inChannels <= 0) {
            throw new IllegalArgumentException("inChannels must be > 0");
        }
        this.inChannels = inChannels;
        this.eps = eps;

        TensorOptions fOpt = new TensorOptions()
                .dtype(new ScalarTypeOptional(torch.ScalarType.Float));
        Tensor w = torch.ones(new long[]{inChannels}, fOpt).clone().requires_grad_(true);
        Tensor b = torch.zeros(new long[]{inChannels}, fOpt).clone().requires_grad_(true);
        Tensor a = torch.ones(new long[]{inChannels}, fOpt).clone().requires_grad_(true);
        this.weight = new Parameter(w, true);
        this.bias = new Parameter(b, true);
        this.meanScale = new Parameter(a, true);
        register_parameter("weight", this.weight);
        register_parameter("bias", this.bias);
        register_parameter("mean_scale", this.meanScale);
    }

    /**
     * @param x     [N, C]
     * @param batch [N] long graph ids, or null (single graph)
     */
    public Tensor forward(Tensor x, Tensor batch) {
        if (x == null || x.dim() != 2 || x.size(1) != inChannels) {
            throw new IllegalArgumentException("x must be [N," + inChannels + "]");
        }
        long N = x.size(0);
        if (batch == null) {
            batch = torch.zeros(new long[]{N},
                    new TensorOptions().dtype(new ScalarTypeOptional(torch.ScalarType.Long)));
        }
        batch = AggrUtils.asLongIndex(batch);
        long numGraphs = batch.size(0) == 0 ? 1 : batch.max().item_long() + 1;

        // Per-graph mean / mean-of-squares via scatter
        Tensor mean = AggrUtils.scatter(x, batch, numGraphs, "mean");           // [G,C]
        Tensor meanSq = AggrUtils.scatter(x.mul(x), batch, numGraphs, "mean");
        Tensor var = torch.relu(meanSq.sub(mean.pow(new Scalar(2))));           // ≥ 0

        Tensor nodeMean = mean.index_select(0, batch);                          // [N,C]
        Tensor nodeVar = var.index_select(0, batch);

        Tensor alpha = meanScale.view(1, inChannels);
        Tensor out = x.sub(nodeMean.mul(alpha));
        out = out.div(nodeVar.add(new Scalar(eps)).sqrt());
        return out.mul(weight.view(1, inChannels)).add(bias.view(1, inChannels));
    }

    /** Single-graph convenience. */
    public Tensor forward(Tensor x) {
        return forward(x, (Tensor) null);
    }

    public long getInChannels() {
        return inChannels;
    }
}
