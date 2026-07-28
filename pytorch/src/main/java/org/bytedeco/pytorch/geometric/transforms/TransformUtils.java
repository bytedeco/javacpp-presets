/*
 * Copyright (C) 2026 bytedeco.org and pytorch JavaCPP presets contributors
 * Shared helpers for geometric transforms (enterprise surface).
 */
package org.bytedeco.pytorch.geometric.transforms;

import org.bytedeco.pytorch.Device;
import org.bytedeco.pytorch.DeviceOptional;
import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.ScalarTypeOptional;
import org.bytedeco.pytorch.T_TensorTensorTensor_T;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorOptions;
import org.bytedeco.pytorch.TensorVector;
import org.bytedeco.pytorch.geometric.data.GraphData;
import org.bytedeco.pytorch.global.torch;

import static org.bytedeco.pytorch.global.torch.*;

/**
 * Validation + topology helpers shared by {@link BaseTransform} implementations.
 *
 * <p>Rules:
 * <ul>
 *   <li>Fail fast with clear messages (never NPE deep inside torch ops).</li>
 *   <li>{@code edge_index} is always Long {@code [2, E]}.</li>
 *   <li>Prefer {@code select(0, r)} over awkward {@code index(TensorIndex...)}.</li>
 *   <li>Coalesce / undirected helpers are pure functions — callers assign results.</li>
 * </ul>
 */
public final class TransformUtils {

    private TransformUtils() {}

    // -------------------------------------------------------------------------
    // Require / validate
    // -------------------------------------------------------------------------

    public static GraphData requireData(GraphData data) {
        if (data == null) {
            throw new NullPointerException("GraphData must not be null");
        }
        return data;
    }

    public static Tensor requireX(GraphData data) {
        requireData(data);
        if (data.x == null || !data.x.defined()) {
            throw new IllegalArgumentException("GraphData.x is required but missing/undefined");
        }
        return data.x;
    }

    public static Tensor requirePos(GraphData data) {
        requireData(data);
        if (data.pos == null || !data.pos.defined()) {
            throw new IllegalArgumentException("GraphData.pos is required but missing/undefined");
        }
        return data.pos;
    }

    public static Tensor requireEdgeIndex(GraphData data) {
        requireData(data);
        if (data.edge_index == null || !data.edge_index.defined()) {
            throw new IllegalArgumentException(
                    "GraphData.edge_index is required but missing/undefined");
        }
        if (data.edge_index.dim() != 2 || data.edge_index.size(0) != 2) {
            throw new IllegalArgumentException(
                    "edge_index must be [2,E], got dim=" + data.edge_index.dim()
                            + " size(0)=" + (data.edge_index.dim() > 0 ? data.edge_index.size(0) : -1));
        }
        return data.edge_index;
    }

    public static long numNodes(GraphData data) {
        requireData(data);
        long n = data.numNodes();
        if (n <= 0) {
            throw new IllegalArgumentException("GraphData has zero nodes");
        }
        return n;
    }

    // -------------------------------------------------------------------------
    // Options builders
    // -------------------------------------------------------------------------

    public static TensorOptions longOpts(Device device) {
        return new TensorOptions()
                .dtype(new ScalarTypeOptional(torch.ScalarType.Long))
                .device(new DeviceOptional(device));
    }

    public static TensorOptions longOptsLike(Tensor ref) {
        return longOpts(ref.device());
    }

    public static TensorOptions floatOptsLike(Tensor ref) {
        return new TensorOptions()
                .dtype(new ScalarTypeOptional(torch.ScalarType.Float))
                .device(new DeviceOptional(ref.device()));
    }

    public static TensorOptions boolOptsLike(Tensor ref) {
        return new TensorOptions()
                .dtype(new ScalarTypeOptional(torch.ScalarType.Bool))
                .device(new DeviceOptional(ref.device()));
    }

    // -------------------------------------------------------------------------
    // Edge helpers
    // -------------------------------------------------------------------------

    /** {@code edge_index} row / col as 1-D Long views. */
    public static Tensor row(Tensor edgeIndex) {
        return edgeIndex.select(0, 0);
    }

    public static Tensor col(Tensor edgeIndex) {
        return edgeIndex.select(0, 1);
    }

    /** Stack {@code [row; col]} → {@code [2, E]}. */
    public static Tensor stackEdges(Tensor row, Tensor col) {
        return stack(new TensorVector(row, col), 0);
    }

    /**
     * Drop self-loops from {@code edge_index}. Optionally filter a 1-D edge attr
     * in lockstep (pass null to skip).
     *
     * @return {@code {edge_index, edge_attr_or_null}}
     */
    public static Tensor[] removeSelfLoops(Tensor edgeIndex, Tensor edgeAttr) {
        Tensor r = row(edgeIndex);
        Tensor c = col(edgeIndex);
        Tensor mask = r.ne(c);
        Tensor keep = mask.nonzero().view(-1);
        Tensor ei = edgeIndex.index_select(1, keep);
        Tensor ea = edgeAttr == null ? null : edgeAttr.index_select(0, keep);
        return new Tensor[]{ei, ea};
    }

    public static Tensor removeSelfLoops(Tensor edgeIndex) {
        return removeSelfLoops(edgeIndex, null)[0];
    }

    /**
     * Make undirected: append reverse edges, then optionally coalesce duplicates.
     */
    public static Tensor toUndirected(Tensor edgeIndex, boolean coalesce) {
        Tensor r = row(edgeIndex);
        Tensor c = col(edgeIndex);
        Tensor both = cat(new TensorVector(
                stackEdges(r, c),
                stackEdges(c, r)), 1);
        return coalesce ? coalesce(both) : both;
    }

    /**
     * Deduplicate directed edges via sort of pair-keys + unique_consecutive.
     * Preserves Long dtype / device of input.
     */
    public static Tensor coalesce(Tensor edgeIndex) {
        if (edgeIndex.size(1) == 0) {
            return edgeIndex;
        }
        Tensor r = row(edgeIndex).to(torch.ScalarType.Long);
        Tensor c = col(edgeIndex).to(torch.ScalarType.Long);
        // key = r * (max+1) + c  — safe for realistic N
        long span = Math.max(r.max().item_long(), c.max().item_long()) + 1;
        Tensor keys = r.mul(new Scalar(span)).add(c);
        Tensor sorted = keys.sort(/*dim=*/0, /*descending=*/false).get0();
        Tensor uniq = unique_consecutive(sorted).get0();
        Tensor newR = uniq.floor_divide(new Scalar(span));
        Tensor newC = uniq.remainder(new Scalar(span));
        return stackEdges(newR, newC).to(edgeIndex.device(), torch.ScalarType.Long);
    }

    /**
     * Add a self-loop for every node in {@code [0, numNodes)} (may create duplicates
     * if loops already exist — pair with {@link #removeSelfLoops} for "remaining").
     */
    public static Tensor addSelfLoops(Tensor edgeIndex, long numNodes) {
        Tensor loop = arange(new Scalar(0), new Scalar(numNodes), longOptsLike(edgeIndex));
        Tensor edgeLoop = stackEdges(loop, loop);
        if (edgeIndex == null || !edgeIndex.defined() || edgeIndex.size(1) == 0) {
            return edgeLoop;
        }
        return cat(new TensorVector(edgeIndex, edgeLoop), 1);
    }

    /**
     * Degree of nodes referenced by {@code index} [E] → [numNodes] (same dtype as {@code like}).
     */
    public static Tensor degree(Tensor index, long numNodes, Tensor like) {
        Tensor idx = index.to(torch.ScalarType.Long);
        Tensor deg = zeros(new long[]{numNodes}, like.options());
        Tensor ones = ones(new long[]{idx.size(0)}, like.options());
        return deg.scatter_add_(0, idx, ones);
    }

    /**
     * First-occurrence indices for each unique value of a 1-D Long label tensor
     * of length N with values in {@code [0, K)}. Result length = K, ordered by
     * ascending label id (matches sorted {@code unique}).
     */
    public static Tensor firstOccurrenceIndex(Tensor labels, long numLabels) {
        long n = labels.size(0);
        // sort labels ascending; stable-ish via paired positions
        org.bytedeco.pytorch.T_TensorTensor_T sorted = labels.sort(/*dim=*/0, /*descending=*/false);
        Tensor sortedLabs = sorted.get0();
        Tensor sortedIdx = sorted.get1(); // original positions
        // unique_consecutive with counts → group sizes in sorted order
        T_TensorTensorTensor_T uc = unique_consecutive(sortedLabs,
                /*return_inverse=*/false, /*return_counts=*/true,
                new org.bytedeco.pytorch.LongOptional());
        Tensor counts = uc.get2(); // [K]
        long k = counts.size(0);
        if (k == 0) {
            return zeros(new long[]{0}, longOptsLike(labels));
        }
        // starts[0]=0; starts[i] = sum(counts[0..i))
        Tensor prefix = counts.cumsum(0);                 // [K] ending positions
        Tensor starts = cat(new TensorVector(
                zeros(new long[]{1}, longOptsLike(labels)),
                prefix.narrow(0, 0, k - 1)
        ), 0);                                            // [K]
        return sortedIdx.index_select(0, starts.to(torch.ScalarType.Long));
    }

    /**
     * Cat edge_attr with a new [E, F] feature block (or set if null).
     */
    public static Tensor catEdgeAttr(Tensor existing, Tensor neu) {
        if (existing == null || !existing.defined()) {
            return neu;
        }
        return cat(new TensorVector(existing, neu), 1);
    }

    /**
     * Move every defined Tensor field of {@code data} to {@code device}
     * (dtype preserved except edge_index forced Long).
     */
    public static GraphData toDevice(GraphData data, Device device) {
        requireData(data);
        if (data.x != null && data.x.defined()) {
            data.x = data.x.to(device, data.x.scalar_type());
        }
        if (data.edge_index != null && data.edge_index.defined()) {
            data.edge_index = data.edge_index.to(device, torch.ScalarType.Long);
        }
        if (data.edge_attr != null && data.edge_attr.defined()) {
            data.edge_attr = data.edge_attr.to(device, data.edge_attr.scalar_type());
        }
        if (data.edge_weight != null && data.edge_weight.defined()) {
            data.edge_weight = data.edge_weight.to(device, data.edge_weight.scalar_type());
        }
        if (data.y != null && data.y.defined()) {
            data.y = data.y.to(device, data.y.scalar_type());
        }
        if (data.pos != null && data.pos.defined()) {
            data.pos = data.pos.to(device, data.pos.scalar_type());
        }
        if (data.adj != null && data.adj.defined()) {
            data.adj = data.adj.to(device, data.adj.scalar_type());
        }
        // dynamic attrs
        for (String k : new java.util.ArrayList<>(data.keys())) {
            // skip core fields already handled
            if ("x".equals(k) || "edge_index".equals(k) || "edge_attr".equals(k)
                    || "edge_weight".equals(k) || "y".equals(k) || "pos".equals(k)) {
                continue;
            }
            Tensor t = data.get(k);
            if (t != null && t.defined()) {
                data.put(k, t.to(device, t.scalar_type()));
            }
        }
        return data;
    }
}
