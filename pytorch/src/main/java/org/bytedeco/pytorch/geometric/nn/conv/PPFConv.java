package org.bytedeco.pytorch.geometric.nn.conv;

import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.ScalarOptional;
import org.bytedeco.pytorch.ScalarTypeOptional;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorVector;
import org.bytedeco.pytorch.geometric.utils.AggrUtils;
import org.bytedeco.pytorch.geometric.utils.GraphUtils;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.modules.container.SequentialImpl;

/**
 * Point Pair Feature convolution (PPFConv) — rotation-robust geometric messages.
 *
 * <pre>
 *   d = p_j - p_i
 *   PPF = [ ‖d‖, ∠(n_i,d), ∠(n_j,d), ∠(n_i,n_j) ]
 *   m_{ij} = LocalNN([ x_j || PPF ])
 *   x'_i   = GlobalNN( max_j m_{ij} )
 * </pre>
 */
public class PPFConv extends MessagePassing {

    private final Module localNN;
    private final Module globalNN;
    private final boolean addSelfLoops;

    private Tensor _pos;
    private Tensor _normal;

    public PPFConv(Module localNN, Module globalNN, boolean addSelfLoops) {
        super("max");
        if (localNN == null) {
            throw new IllegalArgumentException("localNN must not be null");
        }
        this.localNN = register_module("local_nn", localNN);
        this.globalNN = globalNN != null ? register_module("global_nn", globalNN) : null;
        this.addSelfLoops = addSelfLoops;
    }

    @Override
    public Tensor forward(Tensor x, Tensor edge_index) {
        throw new UnsupportedOperationException(
                "PPFConv requires pos and normal — use forward(x, pos, normal, edge_index)");
    }

    /**
     * @param x          node features [N, C] (may be null → PPF-only messages)
     * @param pos        positions [N, 3]
     * @param normal     normals [N, 3]
     * @param edge_index [2, E]
     */
    public Tensor forward(Tensor x, Tensor pos, Tensor normal, Tensor edge_index) {
        if (pos == null || normal == null || edge_index == null) {
            throw new NullPointerException("pos, normal, edge_index must not be null");
        }
        if (pos.size(0) != normal.size(0)) {
            throw new IllegalArgumentException("pos and normal must have same N");
        }
        long N = pos.size(0);
        Tensor ei = edge_index;
        if (addSelfLoops) {
            ei = GraphUtils.add_self_loops(ei, N);
        }

        // Dummy features if x is null — lift still needs a tensor; use ones
        Tensor feat = x != null ? x : torch.ones(new long[]{N, 1}, pos.options());

        this._pos = pos;
        this._normal = normal;
        try {
            Tensor out = propagate(ei, feat);
            if (globalNN != null) {
                out = forwardMlp(globalNN, out);
            }
            return out;
        } finally {
            this._pos = null;
            this._normal = null;
        }
    }

    @Override
    public Tensor message(Tensor x_j, Tensor x_i, Tensor edge_index, Tensor edge_attr, long numNodes) {
        if (_pos == null || _normal == null) {
            throw new IllegalStateException("PPFConv message requires active forward(pos,normal)");
        }
        Tensor index_j = AggrUtils.asLongIndex(
                _index_j != null ? _index_j : edge_index.select(0, 0));
        Tensor index_i = AggrUtils.asLongIndex(
                _index_i != null ? _index_i : edge_index.select(0, 1));

        Tensor pos_i = _pos.index_select(0, index_i);
        Tensor pos_j = _pos.index_select(0, index_j);
        Tensor n_i = _normal.index_select(0, index_i);
        Tensor n_j = _normal.index_select(0, index_j);

        Tensor rel = pos_j.sub(pos_i);
        Tensor dist = rel.norm(new ScalarOptional(new Scalar(2)), new long[]{-1}, true);
        Tensor dHat = rel.div(dist.clamp_min(new Scalar(1e-12)));

        // angle(a,b) = atan2(‖a×b‖, a·b)
        Tensor f2 = angle(n_i, dHat);
        Tensor f3 = angle(n_j, dHat);
        Tensor f4 = angle(n_i, n_j);
        Tensor ppf = torch.cat(new TensorVector(dist, f2, f3, f4), -1); // [E,4]

        Tensor msgInput = ppf;
        // If original x was provided, x_j carries real features (not dummy ones)
        if (x_j.size(1) > 1 || (x_j.size(1) == 1 && _pos.size(1) != 1)) {
            // Prefer concatenating neighbor features when they look like real features.
            // Always concat x_j — callers that pass null x get dummy 1-d ones.
            msgInput = torch.cat(new TensorVector(x_j, ppf), -1);
        } else {
            msgInput = torch.cat(new TensorVector(x_j, ppf), -1);
        }
        return forwardMlp(localNN, msgInput);
    }

    private static Tensor angle(Tensor a, Tensor b) {
        Tensor crossN = torch.cross(a, b)
                .norm(new ScalarOptional(new Scalar(2)), new long[]{-1}, true);
        Tensor dot = a.mul(b).sum(new long[]{-1}, true,
                new ScalarTypeOptional(torch.ScalarType.Float));
        return torch.atan2(crossN, dot);
    }

    private static Tensor forwardMlp(Module m, Tensor in) {
        if (m instanceof SequentialImpl) {
            return ((SequentialImpl) m).forward(in);
        }
        return m.asSequential().forward(in);
    }
}
