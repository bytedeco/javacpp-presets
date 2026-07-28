package org.bytedeco.pytorch.geometric.nn.conv;

import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorVector;
import org.bytedeco.pytorch.geometric.utils.AggrUtils;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.modules.LinearImpl;
import org.bytedeco.pytorch.nn.modules.ReLUImpl;
import org.bytedeco.pytorch.nn.modules.container.SequentialImpl;

/**
 * Feature-wise Linear Modulation graph conv (Brockschmidt, FiLMConv).
 *
 * <pre>
 *   γ_i, β_i = split(W_film x_i)
 *   m_{ij}^{(r)} = act( γ_i ⊙ (W_r x_j) + β_i )   for relation r
 *   x'_i = mean_j m_{ij}
 * </pre>
 * Per-relation neighbor transforms with destination-conditioned FiLM, then
 * degree-weighted mean across relations.
 */
public class FiLMConv extends MessagePassing {

    private final LinearImpl[] lins;
    private final LinearImpl filmLin;
    private final Module act;
    private final long inChannels;
    private final long outChannels;
    private final int numRelations;

    private Tensor _gamma;
    private Tensor _beta;

    public FiLMConv(long inChannels, long outChannels, int numRelations) {
        this(inChannels, outChannels, numRelations, true);
    }

    public FiLMConv(long inChannels, long outChannels, int numRelations, boolean useRelu) {
        this(inChannels, outChannels, numRelations, useRelu ? new ReLUImpl() : null);
    }

    public FiLMConv(long inChannels, long outChannels, int numRelations, Module act) {
        super("sum"); // we do degree-weighted mean ourselves across relations
        if (inChannels <= 0 || outChannels <= 0 || numRelations <= 0) {
            throw new IllegalArgumentException("in/out/numRelations must be > 0");
        }
        this.inChannels = inChannels;
        this.outChannels = outChannels;
        this.numRelations = numRelations;
        this.act = act;

        this.lins = new LinearImpl[numRelations];
        for (int r = 0; r < numRelations; r++) {
            lins[r] = register_module("lin_" + r, new LinearImpl(inChannels, outChannels));
        }
        this.filmLin = register_module("film_lin", new LinearImpl(inChannels, 2 * outChannels));
    }

    @Override
    public Tensor forward(Tensor x, Tensor edge_index) {
        long E = edge_index.size(1);
        Tensor edgeType = torch.zeros(new long[]{E},
                edge_index.options().dtype(new org.bytedeco.pytorch.ScalarTypeOptional(torch.ScalarType.Long)));
        return forward(x, edge_index, edgeType);
    }

    /**
     * @param x          [N, inChannels]
     * @param edge_index [2, E]
     * @param edge_type  [E] relation ids in {@code [0, numRelations)}
     */
    @Override
    public Tensor forward(Tensor x, Tensor edge_index, Tensor edge_type) {
        if (x == null || edge_index == null) {
            throw new NullPointerException("x and edge_index must not be null");
        }
        if (x.size(1) != inChannels) {
            throw new IllegalArgumentException(
                    "x.size(1)=" + x.size(1) + " != inChannels=" + inChannels);
        }
        if (edge_type == null) {
            return forward(x, edge_index);
        }
        edge_type = AggrUtils.asLongIndex(edge_type);

        long N = x.size(0);
        Tensor filmParams = filmLin.forward(x);
        this._gamma = filmParams.narrow(-1, 0, outChannels);
        this._beta = filmParams.narrow(-1, outChannels, outChannels);

        Tensor out = torch.zeros(new long[]{N, outChannels}, x.options());
        Tensor count = torch.zeros(new long[]{N, 1}, x.options());

        try {
            Tensor row = AggrUtils.asLongIndex(edge_index.select(0, 0));
            Tensor col = AggrUtils.asLongIndex(edge_index.select(0, 1));

            for (int r = 0; r < numRelations; r++) {
                Tensor mask = edge_type.eq(new Scalar(r));
                if (!mask.any().item().toBool()) {
                    continue;
                }
                Tensor subSrc = row.masked_select(mask);
                Tensor subDst = col.masked_select(mask);
                Tensor subEi = torch.stack(new TensorVector(subSrc, subDst), 0);

                // Neighbor transform for this relation, then FiLM in message
                Tensor xRel = lins[r].forward(x);
                Tensor part = super.propagate(subEi, xRel); // sum-aggr
                out = out.add(part);

                Tensor ones = torch.ones(new long[]{subDst.size(0), 1}, x.options());
                count = count.add(AggrUtils.scatter(ones, subDst, N, "sum"));
            }
            out = out.div(count.clamp_min(new Scalar(1.0)));
        } finally {
            this._gamma = null;
            this._beta = null;
        }
        return out;
    }

    @Override
    public Tensor message(Tensor x_j, Tensor x_i, Tensor edge_index, Tensor edge_attr, long numNodes) {
        Tensor index_i = AggrUtils.asLongIndex(
                _index_i != null ? _index_i : edge_index.select(0, 1));
        Tensor g = _gamma.index_select(0, index_i);
        Tensor b = _beta.index_select(0, index_i);
        Tensor msg = x_j.mul(g).add(b);
        if (act == null) {
            return msg;
        }
        if (act instanceof ReLUImpl) {
            return torch.relu(msg);
        }
        if (act instanceof SequentialImpl) {
            return ((SequentialImpl) act).forward(msg);
        }
        try {
            return act.asSequential().forward(msg);
        } catch (Exception e) {
            return torch.relu(msg);
        }
    }

    public long getOutChannels() {
        return outChannels;
    }

    public int getNumRelations() {
        return numRelations;
    }
}
