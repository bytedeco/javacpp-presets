package org.bytedeco.pytorch.geometric.nn.conv;

import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.ScalarTypeOptional;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorOptions;
import org.bytedeco.pytorch.nn.Parameter;
import org.bytedeco.pytorch.geometric.utils.GraphUtils;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.nn.modules.LinearImpl;
import org.bytedeco.pytorch.nn.options.LinearOptions;

/**
 * Cluster-GCN convolution (Chiang et al., 2019).
 *
 * <pre>
 *   X' = ((Ã_norm + λ I) X) W + b
 * </pre>
 * where {@code Ã_norm} is the symmetrically normalized adjacency (optional self-loops).
 * Uses the industrial {@link MessagePassing} pipeline — no temporary TensorVector bookkeeping.
 */
public class ClusterGCNConv extends MessagePassing {

    private final LinearImpl lin;
    private final float diagLambda;
    private final boolean addSelfLoops;
    private final Parameter bias;
    private final long inChannels;
    private final long outChannels;

    public ClusterGCNConv(long inChannels, long outChannels) {
        this(inChannels, outChannels, 0.0f, true, true);
    }

    public ClusterGCNConv(long inChannels, long outChannels, float diagLambda,
                          boolean addSelfLoops, boolean hasBias) {
        super("sum");
        if (inChannels <= 0 || outChannels <= 0) {
            throw new IllegalArgumentException("in/out channels must be > 0");
        }
        this.inChannels = inChannels;
        this.outChannels = outChannels;
        this.diagLambda = diagLambda;
        this.addSelfLoops = addSelfLoops;

        LinearOptions opt = new LinearOptions(inChannels, outChannels);
        opt.bias().put(false); // explicit bias for PyG parity
        this.lin = register_module("lin", new LinearImpl(opt));

        if (hasBias) {
            Tensor b = torch.zeros(new long[]{outChannels},
                    new TensorOptions().dtype(new ScalarTypeOptional(torch.ScalarType.Float)));
            this.bias = new Parameter(b, true);
            register_parameter("bias", this.bias);
        } else {
            this.bias = null;
        }
    }

    @Override
    public Tensor forward(Tensor x, Tensor edge_index) {
        return forward(x, edge_index, (Tensor) null);
    }

    @Override
    public Tensor forward(Tensor x, Tensor edge_index, Tensor edge_weight) {
        if (x == null || edge_index == null) {
            throw new NullPointerException("x and edge_index must not be null");
        }
        if (x.dim() != 2) {
            throw new IllegalArgumentException("x must be [N, F], dim=" + x.dim());
        }
        if (edge_index.dim() != 2 || edge_index.size(0) != 2) {
            throw new IllegalArgumentException("edge_index must be [2, E]");
        }
        if (x.size(1) != inChannels) {
            throw new IllegalArgumentException(
                    "x.size(1)=" + x.size(1) + " != inChannels=" + inChannels);
        }

        long N = x.size(0);
        torch.ScalarType dtype = x.scalar_type().intern();

        Tensor[] normed = GraphUtils.gcn_norm(edge_index, edge_weight, N, addSelfLoops, dtype);
        Tensor ei = normed[0];
        Tensor norm = normed[1];

        Tensor out = propagate(ei, x, norm);

        if (diagLambda != 0.0f) {
            out = out.add(x.mul(new Scalar(diagLambda)));
        }

        out = lin.forward(out);
        if (bias != null) {
            out = out.add(bias);
        }
        return out;
    }

    @Override
    public Tensor message(Tensor x_j, Tensor x_i, Tensor edge_index, Tensor edge_attr, long numNodes) {
        if (edge_attr != null) {
            if (edge_attr.dim() == 1) {
                return x_j.mul(edge_attr.view(new long[]{-1, 1}));
            }
            return x_j.mul(edge_attr);
        }
        return x_j;
    }

    public LinearImpl getLin() {
        return lin;
    }

    public float getDiagLambda() {
        return diagLambda;
    }

    public long getOutChannels() {
        return outChannels;
    }
}
