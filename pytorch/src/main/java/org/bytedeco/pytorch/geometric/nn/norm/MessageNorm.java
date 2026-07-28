package org.bytedeco.pytorch.geometric.nn.norm;

import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.ScalarOptional;
import org.bytedeco.pytorch.ScalarTypeOptional;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorOptions;
import org.bytedeco.pytorch.nn.Parameter;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.nn.Module;

/**
 * MessageNorm (Li et al., DeepGCNs): scale aggregated messages to match node feature norms.
 *
 * <pre>
 *   m' = s · m · (‖x‖₂ / (‖m‖₂ + ε))
 * </pre>
 * {@code s} is a learnable scalar.
 */
public class MessageNorm extends Module {

    private final Parameter scale;
    private final double eps;

    public MessageNorm() {
        this(1.0);
    }

    public MessageNorm(double initScale) {
        this(initScale, 1e-6);
    }

    public MessageNorm(double initScale, double eps) {
        super();
        this.eps = eps;
        TensorOptions fOpt = new TensorOptions()
                .dtype(new ScalarTypeOptional(torch.ScalarType.Float));
        Tensor s = torch.tensor(new float[]{(float) initScale}, fOpt).clone().requires_grad_(true);
        this.scale = new Parameter(s, true);
        register_parameter("scale", this.scale);
    }

    /**
     * @param x   node features [N, C] (provides target norm)
     * @param msg aggregated messages [N, C]
     * @return scaled messages [N, C]
     */
    public Tensor forward(Tensor x, Tensor msg) {
        if (x == null || msg == null) {
            throw new NullPointerException("x and msg must not be null");
        }
        if (x.dim() != msg.dim() || x.size(0) != msg.size(0)) {
            throw new IllegalArgumentException("x and msg must share rank and size(0)");
        }
        Tensor normX = x.norm(new ScalarOptional(new Scalar(2)), new long[]{1}, true);
        Tensor normMsg = msg.norm(new ScalarOptional(new Scalar(2)), new long[]{1}, true);
        Tensor ratio = normX.div(normMsg.add(new Scalar(eps)));
        return msg.mul(ratio).mul(scale);
    }
}
