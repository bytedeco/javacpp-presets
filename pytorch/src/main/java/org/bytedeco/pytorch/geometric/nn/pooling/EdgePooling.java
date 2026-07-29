/*
 * Copyright (C) 2026 bytedeco.org and pytorch JavaCPP presets contributors
 *
 * Hand-written peer for PyG torch_geometric.nn.pool.EdgePooling
 * (Diehl et al., "Towards Graph Pooling by Edge Contraction", 2019).
 */
package org.bytedeco.pytorch.geometric.nn.pooling;

import org.bytedeco.pytorch.AbstractTensor;
import org.bytedeco.pytorch.Device;
import org.bytedeco.pytorch.DeviceOptional;
import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.ScalarTypeOptional;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorOptions;
import org.bytedeco.pytorch.TensorVector;
import org.bytedeco.pytorch.geometric.utils.AggrUtils;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.modules.LinearImpl;

/**
 * EdgePooling — greedy edge contraction by learned scores (Diehl et al. / PyG).
 *
 * <h2>Algorithm</h2>
 * <pre>
 *   raw_e  = Linear([x_i ‖ x_j])                  // [E, 1] → [E]
 *   s_e    = score_fn(raw_e)                      // softmax / tanh / sigmoid
 *   if add_to_edge_score:
 *       s_e ← s_e + s_e.detach()                  // residual (PyG default)
 *   s_e    = dropout(s_e, p=dropout)
 *   greedily contract highest-scoring non-incident edges (matching)
 *   uncontracted nodes become singleton clusters
 *   x'_c   = mean_{i ∈ c} x_i
 *   A'     = coarsen(A)  (map endpoints → clusters, drop self-loops, optional coalesce)
 * </pre>
 *
 * <h2>Design notes (enterprise / JavaCPP)</h2>
 * <ul>
 *   <li>Greedy matching is inherently sequential — runs on a CPU long copy of
 *       {@code edge_index}/{@code score} via {@code data_ptr_long()}. Feature
 *       pooling and edge coarsening stay on the input device so gradients flow
 *       through {@code lin} and the mean-pool path.</li>
 *   <li>Batch-aware: matching is independent per graph (nodes of graph g only
 *       match within g). Coarse {@code batch} is derived by taking any fine
 *       node's batch id per cluster (all members share the same graph).</li>
 *   <li>Ownership: {@link LinearImpl} is registered via {@code register_module};
 *       do not store {@code register_parameter} ByRef returns elsewhere.</li>
 *   <li>Aligned with {@link TopKPooling} / {@link SAGPooling} validation style
 *       and with dense poolers ({@link DenseDiffPool}, {@link DenseMinCutPool})
 *       on the "return structured payload + auxiliary signal" pattern
 *       (here: {@code edgeScore} instead of link/cut loss).</li>
 * </ul>
 *
 * <h2>Score functions</h2>
 * Matching PyG static helpers:
 * <ul>
 *   <li>{@link ScoreFunction#SOFTMAX} — {@code softmax} over all edges (default)</li>
 *   <li>{@link ScoreFunction#TANH} — {@code tanh}</li>
 *   <li>{@link ScoreFunction#SIGMOID} — {@code sigmoid}</li>
 * </ul>
 *
 * @see EdgePoolingOutput
 * @see <a href="https://arxiv.org/abs/1905.10990">Diehl et al., 2019</a>
 */
public class EdgePooling extends Module {

    /** Edge-score normalizer, matching PyG {@code compute_edge_score_*}. */
    public enum ScoreFunction {
        /** Softmax over the full edge set (PyG default). */
        SOFTMAX,
        /** Element-wise tanh. */
        TANH,
        /** Element-wise sigmoid. */
        SIGMOID
    }

    private final long inChannels;
    private final boolean addToEdgeScore;
    private final double dropout;
    private final ScoreFunction scoreFunction;
    private final boolean coalesceCoarseEdges;
    private final LinearImpl lin;

    // -------------------------------------------------------------------------
    // Constructors
    // -------------------------------------------------------------------------

    /** {@code EdgePooling(inChannels)} — softmax, add_to_edge_score=true, dropout=0. */
    public EdgePooling(long inChannels) {
        this(inChannels, true, 0.0, ScoreFunction.SOFTMAX, true);
    }

    /** PyG-style two-arg ctor. */
    public EdgePooling(long inChannels, boolean addToEdgeScore) {
        this(inChannels, addToEdgeScore, 0.0, ScoreFunction.SOFTMAX, true);
    }

    /**
     * Full constructor.
     *
     * @param inChannels           node feature dim C
     * @param addToEdgeScore       PyG residual {@code s = s + s.detach()}
     * @param dropout              edge-score dropout in train mode (0 disables)
     * @param scoreFunction        raw→score map
     * @param coalesceCoarseEdges  dedup coarse multi-edges after contraction
     */
    public EdgePooling(long inChannels, boolean addToEdgeScore, double dropout,
                       ScoreFunction scoreFunction, boolean coalesceCoarseEdges) {
        super();
        if (inChannels <= 0) {
            throw new IllegalArgumentException("inChannels must be > 0, got " + inChannels);
        }
        if (dropout < 0.0 || dropout >= 1.0) {
            throw new IllegalArgumentException(
                    "dropout must be in [0, 1), got " + dropout);
        }
        if (scoreFunction == null) {
            throw new NullPointerException("scoreFunction");
        }
        this.inChannels = inChannels;
        this.addToEdgeScore = addToEdgeScore;
        this.dropout = dropout;
        this.scoreFunction = scoreFunction;
        this.coalesceCoarseEdges = coalesceCoarseEdges;
        // Score head: [x_i ‖ x_j] → R   (bias on, matching nn.Linear default)
        this.lin = register_module("lin", new LinearImpl(2L * inChannels, 1L));
    }

    // -------------------------------------------------------------------------
    // Forward
    // -------------------------------------------------------------------------

    /**
     * Single-graph / no-batch forward.
     *
     * <p>Named {@code forwardGraph} (not {@code forward}) because
     * {@link Module#forward(Tensor, Tensor)} returns {@link Tensor}; this API
     * returns the full {@link EdgePoolingOutput} payload.
     *
     * @param x          [N, C]
     * @param edge_index [2, E]
     */
    public EdgePoolingOutput forwardGraph(Tensor x, Tensor edge_index) {
        return forwardGraph(x, edge_index, /*batch=*/null);
    }

    /**
     * Primary forward (PyG signature minus edge_attr — scores are purely
     * feature-based; edge_attr can be folded in by a subclass override of
     * {@link #computeRawScores}).
     *
     * @param x          [N, C]
     * @param edge_index [2, E]  (directed or undirected; both orientations OK)
     * @param batch      [N] graph ids, or {@code null} → single graph
     */
    public EdgePoolingOutput forwardGraph(Tensor x, Tensor edge_index, Tensor batch) {
        validate(x, edge_index);
        x = x.contiguous();
        edge_index = edge_index.contiguous();

        final long numNodes = x.size(0);
        final long numEdges = edge_index.size(1);

        if (batch == null) {
            batch = torch.zeros(new long[]{numNodes}, longOpts(x.device()));
        } else {
            if (batch.dim() != 1 || batch.size(0) != numNodes) {
                throw new IllegalArgumentException(
                        "batch must be [N=" + numNodes + "], got shape mismatch");
            }
            batch = batch.to(torch.ScalarType.Long).contiguous();
        }

        // ---------- empty edge set: identity clusters ----------
        if (numEdges == 0) {
            Tensor cluster = torch.arange(
                    new Scalar(0), new Scalar(numNodes), longOpts(x.device()));
            Tensor emptyEi = torch.zeros(new long[]{2, 0}, longOpts(x.device()));
            return new EdgePoolingOutput(x.clone(), emptyEi, batch.clone(), cluster,
                    /*edgeScore=*/null);
        }

        // ---------- 1. edge scores (device-resident, grad-connected) ----------
        Tensor raw = computeRawScores(x, edge_index);          // [E]
        Tensor score = applyScoreFunction(raw);                // [E]
        if (addToEdgeScore) {
            // PyG: edge_score = edge_score + edge_score.detach()
            // Doubles the forward value while keeping full grad through `score`.
            score = score.add(score.detach());
        }
        if (dropout > 0.0 && is_training()) {
            score = torch.dropout(score, dropout, /*train=*/true);
        }

        // ---------- 2. greedy matching (CPU, no grad) ----------
        // cluster_cpu[i] ∈ [0, N') ; built on host then lifted to x.device()
        MatchResult match = greedyMatch(edge_index, score, batch, numNodes);
        Tensor cluster = match.cluster.to(x.device(), torch.ScalarType.Long);
        long numClusters = match.numClusters;

        // ---------- 3. mean-pool features (device, grad through x) ----------
        // cluster is a pure index tensor (no grad) — scatter mean is fine.
        Tensor newX = AggrUtils.scatter(x, cluster, numClusters, "mean");

        // ---------- 4. coarse batch: any member's batch id per cluster ----------
        // max works because all members of a cluster share the same graph id
        // (matching is batch-restricted). Cast via float for scatter max path.
        Tensor batchF = batch.to(torch.ScalarType.Float);
        Tensor newBatch = AggrUtils.scatter(batchF, cluster, numClusters, "max")
                .to(torch.ScalarType.Long);

        // ---------- 5. coarsen topology ----------
        Tensor newEdgeIndex = coarsenEdges(edge_index, cluster, numClusters,
                coalesceCoarseEdges);

        return new EdgePoolingOutput(newX, newEdgeIndex, newBatch, cluster, score);
    }

    /** Alias kept for older call sites / demos. */
    public EdgePoolingOutput edgePool(Tensor x, Tensor edge_index) {
        return forwardGraph(x, edge_index, null);
    }

    /** Alias kept for older call sites. */
    public EdgePoolingOutput edgePool(Tensor x, Tensor edge_index, Tensor batch) {
        return forwardGraph(x, edge_index, batch);
    }

    /** Alias kept for older call sites. */
    public EdgePoolingOutput forward2(Tensor x, Tensor edge_index) {
        return forwardGraph(x, edge_index, null);
    }

    /**
     * Unpool helper (stateless) — expands coarse features with a cluster map.
     * Prefer {@link EdgePoolingOutput#unpool(Tensor)} when you hold the output.
     */
    public static Tensor unpool(Tensor xCoarse, Tensor cluster) {
        if (xCoarse == null || cluster == null) {
            throw new NullPointerException("xCoarse and cluster must not be null");
        }
        return xCoarse.index_select(0, cluster);
    }

    // -------------------------------------------------------------------------
    // Score path (overridable)
    // -------------------------------------------------------------------------

    /**
     * Raw edge logits before the score function.
     * Default: {@code Linear([x_i ‖ x_j]).squeeze(-1)}.
     * Subclass to inject edge_attr / attention / etc.
     *
     * @return [E] float tensor on {@code x}'s device, connected to {@code lin}
     */
    protected Tensor computeRawScores(Tensor x, Tensor edge_index) {
        Tensor row = edge_index.select(0, 0);
        Tensor col = edge_index.select(0, 1);
        Tensor xI = x.index_select(0, row);                         // [E, C]
        Tensor xJ = x.index_select(0, col);                         // [E, C]
        Tensor cat = torch.cat(new TensorVector(xI, xJ), 1);        // [E, 2C]
        Tensor raw = lin.forward(cat);                              // [E, 1]
        if (raw.dim() == 2 && raw.size(1) == 1) {
            raw = raw.view(-1);
        }
        return raw;
    }

    /** Map raw logits → ranking scores (PyG {@code compute_edge_score_*}). */
    protected Tensor applyScoreFunction(Tensor raw) {
        switch (scoreFunction) {
            case SOFTMAX:
                // Global softmax over edges — relative ranking, sum_e s_e = 1.
                return torch.softmax(raw, 0);
            case TANH:
                return torch.tanh(raw);
            case SIGMOID:
                return torch.sigmoid(raw);
            default:
                throw new IllegalStateException("Unknown scoreFunction: " + scoreFunction);
        }
    }

    // -------------------------------------------------------------------------
    // Greedy matching (CPU)
    // -------------------------------------------------------------------------

    private static final class MatchResult {
        final Tensor cluster;     // Long CPU [N]
        final long numClusters;

        MatchResult(Tensor cluster, long numClusters) {
            this.cluster = cluster;
            this.numClusters = numClusters;
        }
    }

    /**
     * Greedy non-incident edge contraction, restricted per batch graph.
     *
     * <p>Complexity O(E log E) for the sort + O(E) scan. Operates on CPU long
     * copies so device tensors (with grad) are never indexed from Java loops.
     */
    private static MatchResult greedyMatch(Tensor edgeIndex, Tensor score,
                                           Tensor batch, long numNodes) {
        final long numEdges = edgeIndex.size(1);
        // Host-side boolean[] / long[] require N fit in int — true for any
        // graph EdgePooling is practical on (CPU greedy match is O(E log E)).
        if (numNodes > Integer.MAX_VALUE || numEdges > Integer.MAX_VALUE) {
            throw new IllegalArgumentException(
                    "EdgePooling greedy match supports at most Integer.MAX_VALUE nodes/edges, got N="
                            + numNodes + " E=" + numEdges);
        }
        final int n = (int) numNodes;
        final int E = (int) numEdges;

        // Sort edges by score descending. argsort returns int64 indices.
        // Flatten to 1-D contiguous CPU long buffers and read via data_ptr_long —
        // LongIndexer on multi-dim tensors is not reliable for flat get(i).
        Tensor orderT = score.detach().argsort(/*dim=*/0, /*descending=*/true)
                .cpu().contiguous().to(torch.ScalarType.Long).view(-1);
        Tensor rowT = edgeIndex.select(0, 0).detach().cpu().contiguous()
                .to(torch.ScalarType.Long).view(-1);
        Tensor colT = edgeIndex.select(0, 1).detach().cpu().contiguous()
                .to(torch.ScalarType.Long).view(-1);
        Tensor batchT = batch.detach().cpu().contiguous()
                .to(torch.ScalarType.Long).view(-1);

        long[] order = new long[E];
        long[] row = new long[E];
        long[] col = new long[E];
        long[] batchArr = new long[n];
        orderT.data_ptr_long().get(order);
        rowT.data_ptr_long().get(row);
        colT.data_ptr_long().get(col);
        batchT.data_ptr_long().get(batchArr);

        boolean[] matched = new boolean[n];
        long[] clusterArr = new long[n];
        java.util.Arrays.fill(clusterArr, -1L);
        long newNumNodes = 0;

        for (int i = 0; i < E; i++) {
            long e = order[i];
            if (e < 0 || e >= E) {
                continue;
            }
            int ei = (int) e;
            long u = row[ei];
            long v = col[ei];
            if (u < 0 || v < 0 || u >= numNodes || v >= numNodes) {
                continue;
            }
            if (u == v) {
                continue; // self-loop — never contract
            }
            int ui = (int) u;
            int vi = (int) v;
            // Batch restriction: only contract within the same graph
            if (batchArr[ui] != batchArr[vi]) {
                continue;
            }
            if (!matched[ui] && !matched[vi]) {
                matched[ui] = true;
                matched[vi] = true;
                clusterArr[ui] = newNumNodes;
                clusterArr[vi] = newNumNodes;
                newNumNodes++;
            }
        }
        // Singletons keep their own cluster id
        for (int i = 0; i < n; i++) {
            if (clusterArr[i] < 0) {
                clusterArr[i] = newNumNodes++;
            }
        }

        // Build Long CPU tensor via AbstractTensor.create (safe, no torch.tensor overload games)
        Tensor cluster = AbstractTensor.create(clusterArr, numNodes);
        return new MatchResult(cluster, newNumNodes);
    }

    // -------------------------------------------------------------------------
    // Topology coarsening
    // -------------------------------------------------------------------------

    /**
     * Map fine edges → coarse cluster edges; drop self-loops; optionally
     * coalesce duplicate (src,dst) pairs (keeps first occurrence order-stable
     * via sort+unique_consecutive on pair keys).
     *
     * @return [2, E'] Long tensor on the same device as {@code edgeIndex}
     */
    static Tensor coarsenEdges(Tensor edgeIndex, Tensor cluster, long numClusters,
                               boolean coalesce) {
        Tensor row = cluster.index_select(0, edgeIndex.select(0, 0));
        Tensor col = cluster.index_select(0, edgeIndex.select(0, 1));
        Tensor mask = row.ne(col);
        Tensor r = row.masked_select(mask);
        Tensor c = col.masked_select(mask);

        if (r.numel() == 0) {
            return torch.zeros(new long[]{2, 0}, longOpts(edgeIndex.device()));
        }

        if (!coalesce) {
            return torch.stack(new TensorVector(r, c), 0);
        }

        // Dedup directed pairs: key = r * numClusters + c
        // (numClusters fits in int64 for any realistic graph; overflow would
        //  require N' > sqrt(2^63) which is outside EdgePooling's CPU-match regime.)
        Tensor keys = r.to(torch.ScalarType.Long)
                .mul(new Scalar(numClusters))
                .add(c.to(torch.ScalarType.Long));
        // sort + unique_consecutive ≈ unique; recover (r,c) from keys
        Tensor sortedKeys = keys.sort(/*dim=*/0, /*descending=*/false).get0();
        Tensor uniq = torch.unique_consecutive(sortedKeys).get0(); // [E_uniq]
        if (uniq.numel() == 0) {
            return torch.zeros(new long[]{2, 0}, longOpts(edgeIndex.device()));
        }
        // floor_divide keeps integer semantics on Long tensors (plain div may promote).
        Tensor newR = uniq.floor_divide(new Scalar(numClusters));
        Tensor newC = uniq.remainder(new Scalar(numClusters));
        return torch.stack(new TensorVector(newR, newC), 0);
    }

    // -------------------------------------------------------------------------
    // Validation / accessors
    // -------------------------------------------------------------------------

    /** Long options pinned to {@code device} (cluster / edge_index construction). */
    private static TensorOptions longOpts(Device device) {
        return new TensorOptions()
                .dtype(new ScalarTypeOptional(torch.ScalarType.Long))
                .device(new DeviceOptional(device));
    }

    private void validate(Tensor x, Tensor edge_index) {
        if (x == null || edge_index == null) {
            throw new NullPointerException("x and edge_index must not be null");
        }
        if (!x.defined() || !edge_index.defined()) {
            throw new IllegalArgumentException("x and edge_index must be defined Tensors");
        }
        if (x.dim() != 2 || x.size(1) != inChannels) {
            throw new IllegalArgumentException(
                    "x must be [N," + inChannels + "], got dim=" + x.dim()
                            + " size(1)=" + (x.dim() >= 2 ? x.size(1) : -1));
        }
        if (edge_index.dim() != 2 || edge_index.size(0) != 2) {
            throw new IllegalArgumentException(
                    "edge_index must be [2,E], got dim=" + edge_index.dim()
                            + " size(0)=" + (edge_index.dim() >= 1 ? edge_index.size(0) : -1));
        }
    }

    public long getInChannels() {
        return inChannels;
    }

    public boolean isAddToEdgeScore() {
        return addToEdgeScore;
    }

    public double getDropout() {
        return dropout;
    }

    public ScoreFunction getScoreFunction() {
        return scoreFunction;
    }

    public LinearImpl getLin() {
        return lin;
    }

    @Override
    public String toString() {
        return "EdgePooling(inChannels=" + inChannels
                + ", addToEdgeScore=" + addToEdgeScore
                + ", dropout=" + dropout
                + ", score=" + scoreFunction
                + ", coalesce=" + coalesceCoarseEdges + ")";
    }
}
