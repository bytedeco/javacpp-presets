/*
 * Copyright (C) 2026 bytedeco.org and pytorch JavaCPP presets contributors
 */
package org.bytedeco.pytorch.geometric.transforms;

import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorVector;
import org.bytedeco.pytorch.geometric.data.GraphData;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.bytedeco.pytorch.global.torch.*;

/** Higher-order / global-structure transforms. */
public final class HighOrderTransforms {
    private HighOrderTransforms() {}

    /**
     * Meta-path logger for hetero pipelines.
     *
     * <p>Full hetero metapath materialization needs {@code HeteroData}; for
     * homogeneous {@link GraphData} this records the requested path length as
     * {@code data['_metapath_len']} and is a no-op on topology.
     */
    public static class AddMetaPaths implements BaseTransform {
        private final String[] metapath;

        public AddMetaPaths(String[] metapath) {
            this.metapath = metapath == null ? new String[0] : metapath.clone();
        }

        public String[] getMetapath() {
            return metapath.clone();
        }

        @Override
        public GraphData apply(GraphData data) {
            TransformUtils.requireData(data);
            if (metapath.length > 0) {
                Tensor ref = data.x != null ? data.x
                        : (data.edge_index != null ? data.edge_index : zeros(new long[]{1}));
                data.put("_metapath_len",
                        tensor(new long[]{metapath.length}, TransformUtils.longOptsLike(ref)));
            }
            System.out.println("AddMetaPaths: " + String.join(" -> ", metapath));
            return data;
        }
    }

    /**
     * Add one virtual node connected bidirectionally to every existing node.
     * Nodes: N→N+1 ; Edges: E → E + 2N.
     */
    public static class VirtualNode implements BaseTransform {
        @Override
        public GraphData apply(GraphData data) {
            Tensor x = TransformUtils.requireX(data);
            Tensor ei = TransformUtils.requireEdgeIndex(data);
            long numNodes = x.size(0);
            long dim = x.size(1);
            Tensor vFeat = zeros(new long[]{1, dim}, x.options());
            data.x = cat(new TensorVector(x, vFeat), 0);

            Tensor indices = arange(new Scalar(0), new Scalar(numNodes),
                    TransformUtils.longOptsLike(ei));
            Tensor vIndex = full(new long[]{numNodes}, new Scalar(numNodes),
                    TransformUtils.longOptsLike(ei));
            Tensor v2n = stack(new TensorVector(vIndex, indices), 0);
            Tensor n2v = stack(new TensorVector(indices, vIndex), 0);
            data.edge_index = cat(new TensorVector(ei, v2n, n2v), 1);
            return data;
        }
    }

    /**
     * Keep only the largest weakly-connected component(s) (undirected view).
     *
     * <p>Host BFS over the edge list; relabels survivors to contiguous
     * {@code [0..n')} and subsets {@code x}/{@code pos}/{@code y} when present.
     * Isolated nodes are size-1 components.
     */
    public static class LargestConnectedComponents implements BaseTransform {
        private final int numComponents;

        /** Keep the single largest component (PyG default {@code num_components=1}). */
        public LargestConnectedComponents() {
            this(1);
        }

        /**
         * @param numComponents number of largest components to keep (union). Must be &gt;= 1.
         */
        public LargestConnectedComponents(int numComponents) {
            if (numComponents < 1) {
                throw new IllegalArgumentException("numComponents must be >= 1");
            }
            this.numComponents = numComponents;
        }

        public int getNumComponents() {
            return numComponents;
        }

        @Override
        public GraphData apply(GraphData data) {
            TransformUtils.requireData(data);
            long n = TransformUtils.numNodes(data);
            if (n == 0) {
                return data;
            }
            if (n > Integer.MAX_VALUE) {
                throw new IllegalArgumentException(
                        "LargestConnectedComponents supports at most Integer.MAX_VALUE nodes");
            }
            final int N = (int) n;

            // Undirected adjacency
            List<List<Integer>> adj = new ArrayList<>(N);
            for (int i = 0; i < N; i++) {
                adj.add(new ArrayList<>());
            }
            long origE = 0;
            long[] edgeRow = null;
            long[] edgeCol = null;
            if (data.edge_index != null && data.edge_index.defined()
                    && data.edge_index.numel() > 0) {
                Tensor ei = data.edge_index.cpu().contiguous()
                        .to(org.bytedeco.pytorch.global.torch.ScalarType.Long);
                origE = ei.size(1);
                edgeRow = new long[(int) origE];
                edgeCol = new long[(int) origE];
                ei.select(0, 0).contiguous().view(-1).data_ptr_long().get(edgeRow);
                ei.select(0, 1).contiguous().view(-1).data_ptr_long().get(edgeCol);
                for (int e = 0; e < origE; e++) {
                    int u = (int) edgeRow[e];
                    int v = (int) edgeCol[e];
                    if (u < 0 || v < 0 || u >= N || v >= N || u == v) {
                        continue;
                    }
                    adj.get(u).add(v);
                    adj.get(v).add(u);
                }
            }

            // BFS components
            int[] compId = new int[N];
            Arrays.fill(compId, -1);
            List<Integer> compSizes = new ArrayList<>();
            int cid = 0;
            int[] queue = new int[N];
            for (int s = 0; s < N; s++) {
                if (compId[s] >= 0) {
                    continue;
                }
                int qh = 0, qt = 0;
                queue[qt++] = s;
                compId[s] = cid;
                int size = 0;
                while (qh < qt) {
                    int u = queue[qh++];
                    size++;
                    for (int v : adj.get(u)) {
                        if (compId[v] < 0) {
                            compId[v] = cid;
                            queue[qt++] = v;
                        }
                    }
                }
                compSizes.add(size);
                cid++;
            }

            // Rank by size desc; keep top-K
            Integer[] order = new Integer[compSizes.size()];
            for (int i = 0; i < order.length; i++) {
                order[i] = i;
            }
            Arrays.sort(order, (a, b) -> Integer.compare(compSizes.get(b), compSizes.get(a)));
            boolean[] keepComp = new boolean[compSizes.size()];
            int kKeep = Math.min(numComponents, order.length);
            for (int i = 0; i < kKeep; i++) {
                keepComp[order[i]] = true;
            }

            boolean[] keepNode = new boolean[N];
            int kept = 0;
            for (int i = 0; i < N; i++) {
                if (keepComp[compId[i]]) {
                    keepNode[i] = true;
                    kept++;
                }
            }
            if (kept == N) {
                return data;
            }
            if (kept == 0) {
                keepNode[0] = true;
                kept = 1;
            }

            long[] remap = new long[N];
            Arrays.fill(remap, -1L);
            long[] keptIdx = new long[kept];
            long next = 0;
            int p = 0;
            for (int i = 0; i < N; i++) {
                if (keepNode[i]) {
                    remap[i] = next;
                    keptIdx[p++] = i;
                    next++;
                }
            }
            Tensor ref = data.edge_index != null ? data.edge_index
                    : (data.x != null ? data.x : zeros(new long[]{1}));
            Tensor keptTensor = tensor(keptIdx, TransformUtils.longOptsLike(ref));

            if (data.x != null && data.x.defined()) {
                data.x = data.x.index_select(0, keptTensor);
            }
            if (data.pos != null && data.pos.defined()) {
                data.pos = data.pos.index_select(0, keptTensor);
            }
            if (data.y != null && data.y.defined() && data.y.dim() >= 1 && data.y.size(0) == N) {
                data.y = data.y.index_select(0, keptTensor);
            }

            if (edgeRow != null) {
                List<Long> newRow = new ArrayList<>();
                List<Long> newCol = new ArrayList<>();
                List<Integer> keptEdgePos = new ArrayList<>();
                for (int e = 0; e < origE; e++) {
                    int u = (int) edgeRow[e];
                    int v = (int) edgeCol[e];
                    if (u < 0 || v < 0 || u >= N || v >= N) {
                        continue;
                    }
                    if (remap[u] < 0 || remap[v] < 0) {
                        continue;
                    }
                    newRow.add(remap[u]);
                    newCol.add(remap[v]);
                    keptEdgePos.add(e);
                }
                int nE = newRow.size();
                long[] flat = new long[2 * nE];
                for (int e = 0; e < nE; e++) {
                    flat[e] = newRow.get(e);
                    flat[nE + e] = newCol.get(e);
                }
                data.edge_index = tensor(flat, TransformUtils.longOptsLike(ref)).reshape(2, nE);

                if (data.edge_attr != null && data.edge_attr.defined()
                        && data.edge_attr.size(0) == origE && nE > 0) {
                    long[] eIdx = new long[nE];
                    for (int i = 0; i < nE; i++) {
                        eIdx[i] = keptEdgePos.get(i);
                    }
                    data.edge_attr = data.edge_attr.index_select(0,
                            tensor(eIdx, TransformUtils.longOptsLike(ref)));
                } else if (nE == 0) {
                    data.edge_attr = null;
                }
                if (data.edge_weight != null && data.edge_weight.defined()
                        && data.edge_weight.size(0) == origE && nE > 0) {
                    long[] eIdx = new long[nE];
                    for (int i = 0; i < nE; i++) {
                        eIdx[i] = keptEdgePos.get(i);
                    }
                    data.edge_weight = data.edge_weight.index_select(0,
                            tensor(eIdx, TransformUtils.longOptsLike(ref)));
                } else if (nE == 0) {
                    data.edge_weight = null;
                }
            }

            data.put("lcc_num_kept",
                    tensor(new long[]{kept}, TransformUtils.longOptsLike(keptTensor)));
            return data;
        }
    }
}
