package org.bytedeco.pytorch.geometric.nn.conv;

import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorVector;
import org.bytedeco.pytorch.geometric.utils.AggrUtils;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.modules.container.SequentialImpl;

/**
 * PointGNN convolution (Shi & Rajkumar et al.) with alignment offsets.
 *
 * <pre>
 *   Δp_i = MLP_h(x_i)
 *   r_{ij} = p_j - (p_i + Δp_i)
 *   m_{ij} = MLP_f([ x_j || r_{ij} ])
 *   x'_i   = x_i + MLP_g( max_j m_{ij} )   (residual if dims match)
 * </pre>
 * Uses industrial MessagePassing (max aggr by default) with transient pos/delta.
 */
public class PointGNNConv extends MessagePassing {

    private final Module mlpH;
    private final Module mlpF;
    private final Module mlpG;

    // Transient geometry for one forward
    private Tensor _pos;
    private Tensor _deltaPos;

    public PointGNNConv(Module mlpH, Module mlpF, Module mlpG) {
        this(mlpH, mlpF, mlpG, "max");
    }

    public PointGNNConv(Module mlpH, Module mlpF, Module mlpG, String aggr) {
        super(aggr != null ? aggr : "max");
        if (mlpH == null || mlpF == null || mlpG == null) {
            throw new IllegalArgumentException("mlpH/mlpF/mlpG must not be null");
        }
        this.mlpH = register_module("mlp_h", mlpH);
        this.mlpF = register_module("mlp_f", mlpF);
        this.mlpG = register_module("mlp_g", mlpG);
    }

    /** Convenience accepting SequentialImpl (common in demos). */
    public PointGNNConv(SequentialImpl mlpH, SequentialImpl mlpF, SequentialImpl mlpG) {
        this((Module) mlpH, (Module) mlpF, (Module) mlpG);
    }

    @Override
    public Tensor forward(Tensor x, Tensor edge_index) {
        throw new UnsupportedOperationException(
                "PointGNNConv requires positions — use forward(x, pos, edge_index)");
    }

    /**
     * @param x          node features [N, F]
     * @param pos        coordinates [N, 3]
     * @param edge_index [2, E]
     */
    public Tensor forward(Tensor x, Tensor pos, Tensor edge_index) {
        if (x == null || pos == null || edge_index == null) {
            throw new NullPointerException("x, pos, edge_index must not be null");
        }
        if (x.size(0) != pos.size(0)) {
            throw new IllegalArgumentException("x and pos must have same N");
        }
        if (edge_index.dim() != 2 || edge_index.size(0) != 2) {
            throw new IllegalArgumentException("edge_index must be [2,E]");
        }

        this._pos = pos;
        this._deltaPos = forwardMlp(mlpH, x);
        try {
            // propagateImpl already invokes update() — do not call it again
            // (would feed mlpG a post-projection tensor and break Linear shapes).
            return propagate(edge_index, x);
        } finally {
            this._pos = null;
            this._deltaPos = null;
        }
    }

    @Override
    public Tensor message(Tensor x_j, Tensor x_i, Tensor edge_index, Tensor edge_attr, long numNodes) {
        if (_pos == null || _deltaPos == null) {
            throw new IllegalStateException("PointGNNConv message requires active forward(pos)");
        }
        // source_to_target: index_j = row (source), index_i = col (target)
        Tensor index_j = _index_j != null ? _index_j : edge_index.select(0, 0);
        Tensor index_i = _index_i != null ? _index_i : edge_index.select(0, 1);
        index_j = AggrUtils.asLongIndex(index_j);
        index_i = AggrUtils.asLongIndex(index_i);

        Tensor pos_j = _pos.index_select(0, index_j);
        Tensor pos_i = _pos.index_select(0, index_i);
        Tensor delta_i = _deltaPos.index_select(0, index_i);

        // Translation-invariant relative position with alignment
        Tensor relPos = pos_j.sub(pos_i.add(delta_i));
        Tensor fInput = torch.cat(new TensorVector(x_j, relPos), -1);
        return forwardMlp(mlpF, fInput);
    }

    @Override
    public Tensor update(Tensor aggrOut, Tensor x) {
        Tensor updated = forwardMlp(mlpG, aggrOut);
        // Residual only when feature dims match
        if (x != null && x.size(1) == updated.size(1)) {
            return x.add(updated);
        }
        return updated;
    }

    private static Tensor forwardMlp(Module m, Tensor in) {
        if (m instanceof SequentialImpl) {
            return ((SequentialImpl) m).forward(in);
        }
        return m.asSequential().forward(in);
    }

    public Module getMlpH() {
        return mlpH;
    }

    public Module getMlpF() {
        return mlpF;
    }

    public Module getMlpG() {
        return mlpG;
    }
}
