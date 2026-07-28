/*
 * Copyright (C) 2026 bytedeco.org and pytorch JavaCPP presets contributors
 * Hand-written peer for PyG EdgePooling return payload.
 */
package org.bytedeco.pytorch.geometric.nn.pooling;

import org.bytedeco.pytorch.Tensor;

/**
 * Result of {@link EdgePooling#forwardGraph}.
 *
 * <pre>
 *   x         [N', C]   pooled node features (mean over cluster)
 *   edgeIndex [2, E']  coarse edges among clusters (self-loops removed)
 *   batch     [N']      graph id of each cluster (null if input batch was null
 *                       and single-graph default was used — still non-null in practice)
 *   cluster   [N]       cluster[i] = new node id of original node i  (unpool map)
 *   edgeScore [E]       learned edge scores used for greedy matching (may be null)
 * </pre>
 *
 * <p>{@link #unpool(Tensor)} expands a coarse feature tensor back to the fine
 * node set via {@code x_fine = x_coarse[cluster]} (PyG {@code EdgePooling.unpool}).
 *
 * <p>Fields are public for demo / migration compatibility; treat as read-only
 * after construction.
 */
public final class EdgePoolingOutput {

    /** Pooled features [N', C]. */
    public final Tensor x;

    /** Coarse edge_index [2, E'] (may be empty [2,0]). Never null. */
    public final Tensor edgeIndex;

    /** Coarse batch [N']. Never null after a successful forward. */
    public final Tensor batch;

    /** Fine→coarse assignment [N], values in {@code [0, N')}. */
    public final Tensor cluster;

    /** Edge scores [E] used for ranking (may be null if no edges). */
    public final Tensor edgeScore;

    public EdgePoolingOutput(Tensor x, Tensor cluster) {
        this(x, /*edgeIndex=*/null, /*batch=*/null, cluster, /*edgeScore=*/null);
    }

    public EdgePoolingOutput(Tensor x, Tensor edgeIndex, Tensor batch, Tensor cluster) {
        this(x, edgeIndex, batch, cluster, /*edgeScore=*/null);
    }

    public EdgePoolingOutput(Tensor x, Tensor edgeIndex, Tensor batch,
                             Tensor cluster, Tensor edgeScore) {
        if (x == null) {
            throw new NullPointerException("x");
        }
        if (cluster == null) {
            throw new NullPointerException("cluster");
        }
        this.x = x;
        this.edgeIndex = edgeIndex;
        this.batch = batch;
        this.cluster = cluster;
        this.edgeScore = edgeScore;
    }

    // -------------------------------------------------------------------------
    // Accessors (enterprise surface; fields remain for demos)
    // -------------------------------------------------------------------------

    public Tensor getX() {
        return x;
    }

    public Tensor getEdgeIndex() {
        return edgeIndex;
    }

    public Tensor getBatch() {
        return batch;
    }

    public Tensor getCluster() {
        return cluster;
    }

    public Tensor getEdgeScore() {
        return edgeScore;
    }

    /** Number of coarse nodes N'. */
    public long numClusters() {
        return x.size(0);
    }

    /** Number of fine nodes N. */
    public long numNodes() {
        return cluster.size(0);
    }

    /**
     * Unpool: expand coarse features to the fine node set.
     * {@code x_fine[i] = x_coarse[cluster[i]]} — PyG {@code EdgePooling.unpool}.
     *
     * @param xCoarse [N', C] (typically this.x, or a later layer output on coarse graph)
     * @return [N, C]
     */
    public Tensor unpool(Tensor xCoarse) {
        if (xCoarse == null) {
            throw new NullPointerException("xCoarse");
        }
        if (xCoarse.dim() != 2) {
            throw new IllegalArgumentException(
                    "xCoarse must be [N',C], got dim=" + xCoarse.dim());
        }
        if (xCoarse.size(0) != x.size(0)) {
            throw new IllegalArgumentException(
                    "xCoarse.size(0)=" + xCoarse.size(0)
                            + " != numClusters=" + x.size(0));
        }
        return xCoarse.index_select(0, cluster);
    }

    /**
     * Unpack as {@code {x, edgeIndex, batch, cluster}} — matches common
     * multi-return call sites (TopK / SAG style).
     */
    public Tensor[] asArray() {
        return new Tensor[]{x, edgeIndex, batch, cluster};
    }

    @Override
    public String toString() {
        long nPrime = x != null && x.defined() ? x.size(0) : -1;
        long n = cluster != null && cluster.defined() ? cluster.size(0) : -1;
        long e = edgeIndex != null && edgeIndex.defined() ? edgeIndex.size(1) : -1;
        return "EdgePoolingOutput{N=" + n + " → N'=" + nPrime + ", E'=" + e + "}";
    }
}
