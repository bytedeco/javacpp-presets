package org.bytedeco.pytorch.geometric.nn.conv;

import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.geometric.utils.GraphUtils;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.nn.modules.LinearImpl;

import java.util.ArrayList;
import java.util.List;

import static org.bytedeco.pytorch.global.torch.relu;

/**
 * ARMA graph convolutional operator (Bianchi et al.).
 *
 * <pre>
 *   X̄^{(0)} = σ( W_init X )
 *   X̄^{(t+1)} = σ( Ã X̄^{(t)} + W_root X̄^{(t)} )
 *   Y = mean_s X̄_s^{(T)}
 * </pre>
 * Multiple independent stacks are averaged. Uses industrial MessagePassing;
 * no intermediate Tensor.close().
 */
public class ARMAConv extends MessagePassing {

    private final List<LinearImpl> initLins;
    private final List<LinearImpl> rootLins;
    private final int numStacks;
    private final int numLayers;
    private final long inChannels;
    private final long outChannels;
    private final boolean normalize;
    private final boolean addSelfLoops;
    private final double dropout;

    public ARMAConv(long inChannels, long outChannels, int numStacks, int numLayers) {
        this(inChannels, outChannels, numStacks, numLayers, true, true, 0.0);
    }

    public ARMAConv(long inChannels, long outChannels, int numStacks, int numLayers,
                    boolean normalize, boolean addSelfLoops, double dropout) {
        super("sum");
        if (inChannels <= 0 || outChannels <= 0) {
            throw new IllegalArgumentException("in/out channels must be > 0");
        }
        if (numStacks <= 0) {
            throw new IllegalArgumentException("numStacks must be > 0");
        }
        if (numLayers < 0) {
            throw new IllegalArgumentException("numLayers must be >= 0");
        }
        this.inChannels = inChannels;
        this.outChannels = outChannels;
        this.numStacks = numStacks;
        this.numLayers = numLayers;
        this.normalize = normalize;
        this.addSelfLoops = addSelfLoops;
        this.dropout = dropout;

        this.initLins = new ArrayList<>(numStacks);
        this.rootLins = new ArrayList<>(numStacks);
        for (int i = 0; i < numStacks; i++) {
            LinearImpl initLin = new LinearImpl(inChannels, outChannels);
            LinearImpl rootLin = new LinearImpl(outChannels, outChannels);
            initLins.add(initLin);
            rootLins.add(rootLin);
            register_module("init_" + i, initLin);
            register_module("root_" + i, rootLin);
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
        if (x.dim() != 2 || x.size(1) != inChannels) {
            throw new IllegalArgumentException(
                    "x must be [N," + inChannels + "], got F=" + (x.dim() > 1 ? x.size(1) : -1));
        }
        if (edge_index.dim() != 2 || edge_index.size(0) != 2) {
            throw new IllegalArgumentException("edge_index must be [2,E]");
        }

        long N = x.size(0);
        Tensor ei = edge_index;
        Tensor ew = edge_weight;
        if (normalize) {
            torch.ScalarType dtype = x.scalar_type().intern();
            Tensor[] normed = GraphUtils.gcn_norm(ei, ew, N, addSelfLoops, dtype);
            ei = normed[0];
            ew = normed[1];
        } else if (addSelfLoops) {
            if (ew == null) {
                ei = GraphUtils.add_self_loops(ei, N);
            } else {
                Tensor[] pair = GraphUtils.add_self_loops(ei, ew, N, 1.0);
                ei = pair[0];
                ew = pair[1];
            }
        }

        if (dropout > 0.0 && this.is_training() && ew != null) {
            ew = torch.dropout(ew, dropout, true);
        }

        Tensor globalOut = null;
        for (int s = 0; s < numStacks; s++) {
            Tensor out = relu(initLins.get(s).forward(x));
            for (int t = 0; t < numLayers; t++) {
                Tensor aggr = propagate(ei, out, ew);
                Tensor root = rootLins.get(s).forward(out);
                out = relu(aggr.add(root));
            }
            globalOut = globalOut == null ? out : globalOut.add(out);
        }
        return globalOut.div(new Scalar(numStacks));
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

    public void resetParameters() {
        for (LinearImpl lin : initLins) {
            lin.reset_parameters();
        }
        for (LinearImpl lin : rootLins) {
            lin.reset_parameters();
        }
    }

    public int getNumStacks() {
        return numStacks;
    }

    public int getNumLayers() {
        return numLayers;
    }

    public long getInChannels() {
        return inChannels;
    }

    public long getOutChannels() {
        return outChannels;
    }
}
