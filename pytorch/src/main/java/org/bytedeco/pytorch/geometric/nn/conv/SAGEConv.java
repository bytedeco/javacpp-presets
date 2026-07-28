package org.bytedeco.pytorch.geometric.nn.conv;

import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.ScalarOptional;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.nn.modules.LinearImpl;
import org.bytedeco.pytorch.nn.options.LinearOptions;

import static org.bytedeco.pytorch.global.torch.relu;

/**
 * GraphSAGE convolution (Hamilton et al.).
 *
 * <pre>
 *   x'_i = W_1 x_i + W_2 · mean_{j ∈ N(i)} x_j
 * </pre>
 * Optional L2 row-normalization after fusion (+ optional ReLU for project compatibility).
 */
public class SAGEConv extends MessagePassing {

    private final LinearImpl linNeighbor;
    private final LinearImpl linSelf;
    private final boolean normalize;
    private final boolean bias;
    private final boolean rootWeight;
    private final boolean applyRelu;

    public SAGEConv(long inDim, long outDim, boolean normalize, boolean bias) {
        this(inDim, outDim, normalize, bias, true, true);
    }

    /**
     * @param applyRelu project-compat flag (historical SAGEConv applied ReLU); set false for pure PyG parity
     */
    public SAGEConv(long inDim, long outDim, boolean normalize, boolean bias,
                    boolean rootWeight, boolean applyRelu) {
        super("mean");
        this.normalize = normalize;
        this.bias = bias;
        this.rootWeight = rootWeight;
        this.applyRelu = applyRelu;

        LinearOptions neighOpt = new LinearOptions(inDim, outDim);
        neighOpt.bias().put(bias);
        this.linNeighbor = register_module("linNeighbor", new LinearImpl(neighOpt));

        if (rootWeight) {
            LinearOptions selfOpt = new LinearOptions(inDim, outDim);
            selfOpt.bias().put(bias);
            this.linSelf = register_module("linSelf", new LinearImpl(selfOpt));
        } else {
            this.linSelf = null;
        }
    }

    public SAGEConv(long inDim, long outDim) {
        this(inDim, outDim, false, true);
    }

    /** Homogeneous forward. */
    @Override
    public Tensor forward(Tensor x, Tensor edge_index) {
        return forward(x, x, edge_index);
    }

    /**
     * Bipartite forward: aggregate from {@code xSrc} onto {@code xDst} nodes.
     *
     * @param xSrc       source node features [N_src, inDim]
     * @param xDst       destination node features [N_dst, inDim]
     * @param edge_index [2, E] with src→dst
     * @return [N_dst, outDim]
     */
    public Tensor forward(Tensor xSrc, Tensor xDst, Tensor edge_index) {
        long[] size = new long[]{xSrc.size(0), xDst.size(0)};
        // Mean-aggregate neighbors; size ensures correct N_dst even when bipartite
        Tensor aggrNeighbor = propagate(edge_index, xSrc, size);
        Tensor neighborFeat = linNeighbor.forward(aggrNeighbor);

        Tensor out;
        if (rootWeight && linSelf != null) {
            Tensor selfFeat = linSelf.forward(xDst);
            out = neighborFeat.add(selfFeat);
        } else {
            out = neighborFeat;
        }

        if (applyRelu) {
            out = relu(out);
        }
        if (normalize) {
            Tensor norm = out.norm(new ScalarOptional(new Scalar(2)), new long[]{1}, true);
            norm = norm.clamp_min(new Scalar(1e-12));
            out = out.div(norm);
        }
        return out;
    }

    @Override
    public Tensor message(Tensor x_j, Tensor x_i, Tensor edge_index, Tensor edge_attr, long numNodes) {
        return x_j;
    }

    public LinearImpl getLinNeighbor() {
        return linNeighbor;
    }

    public LinearImpl getLinSelf() {
        return linSelf;
    }
}
