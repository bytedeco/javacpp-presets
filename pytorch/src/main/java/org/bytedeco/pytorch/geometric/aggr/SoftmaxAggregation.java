package org.bytedeco.pytorch.geometric.aggr;

import org.bytedeco.pytorch.ScalarTypeOptional;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorOptions;
import org.bytedeco.pytorch.nn.Parameter;
import org.bytedeco.pytorch.geometric.utils.AggrUtils;
import org.bytedeco.pytorch.global.torch;

/**
 * Softmax aggregation (Li et al. / PyG).
 *
 * <pre>
 *   α_j = softmax_j( t ⊙ x_j )
 *   y_i = Σ_{j ∈ N(i)} α_j ⊙ x_j
 * </pre>
 * Learnable per-channel temperature {@code t} (or fixed ones).
 */
public class SoftmaxAggregation extends Aggregation {

    private final Parameter t;       // [1, C] learnable
    private final Tensor tFixed;     // [1, C] buffer when not learning
    private final boolean learnT;
    private final long channels;

    public SoftmaxAggregation(long channels) {
        this(channels, true);
    }

    public SoftmaxAggregation(long channels, boolean learnT) {
        super();
        if (channels <= 0) {
            throw new IllegalArgumentException("channels must be > 0");
        }
        this.channels = channels;
        this.learnT = learnT;
        TensorOptions fOpt = new TensorOptions()
                .dtype(new ScalarTypeOptional(torch.ScalarType.Float));
        Tensor init = torch.ones(new long[]{1, channels}, fOpt).clone();
        if (learnT) {
            init.requires_grad_(true);
            this.t = new Parameter(init, true);
            register_parameter("t", this.t);
            this.tFixed = null;
        } else {
            this.t = null;
            this.tFixed = init;
            register_buffer("t", this.tFixed);
        }
    }

    private Tensor temperature() {
        return learnT ? t : tFixed;
    }

    @Override
    public Tensor forward(Tensor x, Tensor index, long dimSize) {
        if (x == null || index == null) {
            throw new NullPointerException("x and index must not be null");
        }
        // t: [1,C] broadcasts over [E,C]
        Tensor score = x.mul(temperature());
        Tensor alpha = AggrUtils.scatter_softmax(score, index, dimSize);
        return AggrUtils.scatter(x.mul(alpha), index, dimSize, "sum");
    }

    public boolean isLearnT() {
        return learnT;
    }

    public long getChannels() {
        return channels;
    }
}
