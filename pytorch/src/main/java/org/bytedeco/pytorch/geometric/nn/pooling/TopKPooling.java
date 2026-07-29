package org.bytedeco.pytorch.geometric.nn.pooling;

import org.bytedeco.pytorch.DeviceOptional;
import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.ScalarTypeOptional;
import org.bytedeco.pytorch.T_TensorTensor_T;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorOptions;
import org.bytedeco.pytorch.TensorVector;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.Parameter;
import org.bytedeco.pytorch.global.torch;

/**
 * TopKPooling (Gao & Ji / Cangea et al. / PyG).
 *
 * <pre>
 *   score_i = (x_i · p) / ‖p‖
 *   keep top ⌈ratio · N_g⌉ nodes per graph g
 *   x'_i    = x_i ⊙ tanh(score_i)     for kept nodes
 *   edges filtered & relabeled to new contiguous indices
 * </pre>
 *
 * Returns {@code {x_new, edge_index_new, batch_new, perm, score}} where
 * {@code perm} are original node indices kept (sorted by score descending within
 * each graph when batch is multi-graph).
 *
 * <p>Per-graph top-k is implemented via score bias {@code + (max_score+1)·graph_id}
 * so a single global {@code topk} preserves ranking inside each graph while
 * selecting exactly the right count overall (when graphs have equal size) or
 * approximately ratio·N_g for unequal sizes via a second per-graph pass.
 */
public class TopKPooling extends Module {

    protected final long inChannels;
    protected final double ratio;
    protected final Parameter weight; // projection p [C, 1]

    public TopKPooling(long inChannels, double ratio) {
        super();
        if (inChannels <= 0) {
            throw new IllegalArgumentException("inChannels must be > 0");
        }
        if (ratio <= 0.0 || ratio > 1.0) {
            throw new IllegalArgumentException("ratio must be in (0, 1], got " + ratio);
        }
        this.inChannels = inChannels;
        this.ratio = ratio;

        TensorOptions fOpt = new TensorOptions()
                .dtype(new ScalarTypeOptional(torch.ScalarType.Float));
        Tensor w = torch.randn(new long[]{inChannels, 1}, fOpt).clone().requires_grad_(true);
        this.weight = new Parameter(w, true);
        register_parameter("weight", this.weight);
    }

    /**
     * Compute node scores. Subclasses (SAGPooling) override to inject structure.
     *
     * @param x          [N, C]
     * @param edge_index [2, E] (unused in plain TopK; used by SAG)
     * @return score [N]
     */
    protected Tensor calculateScore(Tensor x, Tensor edge_index) {
        Tensor norm = weight.norm(new Scalar(2)).clamp_min(new Scalar(1e-6));
        Tensor p = weight.div(norm);
        return x.matmul(p).squeeze(1); // [N]
    }

    /**
     * Primary graph pooling forward.
     *
     * <p>Named {@code forwardGraph} (not {@code forward}) because
     * {@link Module#forward(Tensor, Tensor, Tensor)} returns {@link Tensor};
     * this API returns a multi-tensor payload.
     *
     * @return {@code {x_new, edge_index_new, batch_new, perm, score}}
     */
    public Tensor[] forwardGraph(Tensor x, Tensor edge_index, Tensor batch) {
        if (x == null || edge_index == null) {
            throw new NullPointerException("x and edge_index must not be null");
        }
        if (x.dim() != 2 || x.size(1) != inChannels) {
            throw new IllegalArgumentException("x must be [N," + inChannels + "]");
        }
        if (edge_index.dim() != 2 || edge_index.size(0) != 2) {
            throw new IllegalArgumentException("edge_index must be [2,E]");
        }

        long numNodes = x.size(0);
        TensorOptions longOnX = new TensorOptions()
                .dtype(new ScalarTypeOptional(torch.ScalarType.Long))
                .device(new DeviceOptional(x.device()));
        if (batch == null) {
            batch = torch.zeros(new long[]{numNodes}, longOnX);
        } else {
            batch = batch.to(x.device(), torch.ScalarType.Long);
        }

        Tensor score = calculateScore(x, edge_index); // [N]
        Tensor perm = selectTopKPerGraph(score, batch, ratio);
        return filterAndRelabel(x, edge_index, batch, score, perm);
    }

    /** Alias kept for older call sites. */
    public Tensor[] topk(Tensor x, Tensor edge_index, Tensor batch) {
        return forwardGraph(x, edge_index, batch);
    }

    /** Alias kept for older call sites. */
    public Tensor[] forward2(Tensor x, Tensor edge_index, Tensor batch) {
        return forwardGraph(x, edge_index, batch);
    }

    /**
     * Per-graph top-k selection.
     * For each graph g with n_g nodes, keep k_g = max(1, ⌈ratio · n_g⌉).
     */
    protected static Tensor selectTopKPerGraph(Tensor score, Tensor batch, double ratio) {
        long numNodes = score.size(0);
        long numGraphs = batch.size(0) == 0 ? 1 : batch.max().item_long() + 1;

        if (numGraphs == 1) {
            long k = Math.max(1L, Math.round(numNodes * ratio));
            k = Math.min(k, numNodes);
            return torch.topk(score, k).get1().contiguous();
        }

        // Build keep-mask by iterating graphs (N typically modest for pooling demos;
        // vectorized segmented topk needs custom kernels not available here).
        // Keep mask / ones MUST live on the same device as score/batch (MPS/CUDA).
        TensorOptions boolOpts = new TensorOptions()
                .dtype(new ScalarTypeOptional(torch.ScalarType.Bool))
                .device(new DeviceOptional(score.device()));
        Tensor keepMask = torch.zeros(new long[]{numNodes}, boolOpts);

        for (long g = 0; g < numGraphs; g++) {
            Tensor gMask = batch.eq(new Scalar(g));
            Tensor gIdx = gMask.nonzero().view(-1); // nodes in graph g
            long nG = gIdx.size(0);
            if (nG == 0) {
                continue;
            }
            long kG = Math.max(1L, Math.round(nG * ratio));
            kG = Math.min(kG, nG);
            Tensor gScore = score.index_select(0, gIdx);
            Tensor localPerm = torch.topk(gScore, kG).get1(); // indices into gIdx
            Tensor globalPerm = gIdx.index_select(0, localPerm);
            // Prefer index_fill_ (scalar) over scatter_ of a Bool source tensor —
            // more robust across CPU / MPS / CUDA.
            keepMask.index_fill_(0, globalPerm, new Scalar(1));
        }
        return keepMask.nonzero().view(-1).contiguous();
    }

    /**
     * Gate features, filter edges, relabel to contiguous [0..k).
     *
     * @return {@code {x_new, edge_index_new, batch_new, perm, score}}
     */
    protected Tensor[] filterAndRelabel(Tensor x, Tensor edge_index, Tensor batch,
                                        Tensor score, Tensor perm) {
        long numNodes = x.size(0);
        long k = perm.size(0);

        // Gate: x' = x[perm] * tanh(score[perm])
        Tensor featScores = score.index_select(0, perm);
        Tensor gate = torch.tanh(featScores).unsqueeze(1);
        Tensor xNew = x.index_select(0, perm).mul(gate);
        Tensor batchNew = batch.index_select(0, perm);

        // map[old] = new index, -1 if dropped — same device as x/perm (MPS/CUDA-safe)
        TensorOptions longOpts = new TensorOptions()
                .dtype(new ScalarTypeOptional(torch.ScalarType.Long))
                .device(new DeviceOptional(x.device()));
        Tensor map = torch.full(new long[]{numNodes}, new Scalar(-1), longOpts);
        Tensor newIdxRange = torch.arange(new Scalar(0), new Scalar(k), longOpts);
        map.scatter_(0, perm, newIdxRange);

        Tensor row = edge_index.select(0, 0);
        Tensor col = edge_index.select(0, 1);
        Tensor newRow = map.index_select(0, row);
        Tensor newCol = map.index_select(0, col);
        Tensor mask = newRow.ge(new Scalar(0)).logical_and(newCol.ge(new Scalar(0)));
        Tensor finalRow = newRow.masked_select(mask);
        Tensor finalCol = newCol.masked_select(mask);
        Tensor edgeIndexNew = torch.stack(new TensorVector(finalRow, finalCol), 0);

        return new Tensor[]{xNew, edgeIndexNew, batchNew, perm, score};
    }

    public long getInChannels() {
        return inChannels;
    }

    public double getRatio() {
        return ratio;
    }
}
