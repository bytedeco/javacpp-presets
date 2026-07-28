package org.bytedeco.pytorch.geometric.nn.conv;

import org.bytedeco.javacpp.Pointer;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.geometric.aggr.Aggregation;
import org.bytedeco.pytorch.geometric.utils.AggrUtils;
import org.bytedeco.pytorch.geometric.utils.GraphUtils;
import org.bytedeco.pytorch.nn.Module;

/**
 * Industrial MessagePassing base (PyG-aligned) for sparse GNN layers.
 *
 * <p>Pipeline for every {@code propagate} overload:
 * <pre>
 *   edge_index [2,E] → resolve flow → (index_j, index_i)
 *   → resolve size [N_src, N_dst]
 *   → optional fused message_and_aggregate
 *   else:
 *     x_j = lift(x_src, index_j)
 *     x_i = lift(x_dst, index_i)   // when bipartite or needsX_i()
 *     msg = collectMessage(...)
 *     out = aggregate(msg, index_i, N_dst)
 *     out = update(out, x_dst)
 * </pre>
 *
 * <p>Java has no {@code **kwargs}, so call sites use typed overloads that all funnel
 * into {@link #propagateImpl}. Subclasses override {@code message} / {@code update} /
 * {@code needsX_i} / {@code messageAndAggregate}; the 5-arg message is non-abstract so
 * simple layers need not implement it (existing {@code @Override}s still work).
 *
 * <p><b>Do not</b> close intermediate tensors in this base (JavaCPP ByRef / leaf retain).
 */
public abstract class MessagePassing extends Module {

    // ---- configuration ----
    protected String aggr;                 // string reduce when aggrModule == null
    protected Aggregation aggrModule;      // wins over string when non-null
    protected String flow;                 // source_to_target | target_to_source
    protected int nodeDim = -2;            // feature layout docs (scatter always dim 0)

    // ---- transient propagate state (valid only during a propagate call) ----
    protected Tensor _edgeIndex;
    protected Tensor _index_i;             // target indices [E]
    protected Tensor _index_j;             // source indices [E]
    protected long[] _size;                // {N_src, N_dst}

    // ========================================================================
    // Constructors
    // ========================================================================

    public MessagePassing() {
        this("add", "source_to_target");
    }

    public MessagePassing(String aggr) {
        this(aggr, "source_to_target");
    }

    public MessagePassing(String aggr, String flow) {
        super();
        this.aggr = normalizeAggr(aggr);
        this.flow = normalizeFlow(flow);
        this.aggrModule = null;
    }

    public MessagePassing(Aggregation aggr) {
        this(aggr, "source_to_target");
    }

    public MessagePassing(Aggregation aggr, String flow) {
        super();
        if (aggr == null) {
            throw new IllegalArgumentException("Aggregation module must not be null");
        }
        this.aggr = null;
        this.flow = normalizeFlow(flow);
        this.aggrModule = aggr;
        register_module("aggr_module", aggr);
    }

    /** JavaCPP pointer ctor (interop). Defaults to sum / source_to_target. */
    public MessagePassing(Pointer p) {
        super(p);
        this.aggr = "sum";
        this.flow = "source_to_target";
        this.aggrModule = null;
    }

    // ========================================================================
    // Configuration mutators
    // ========================================================================

    public void setAggr(String reduce) {
        this.aggr = normalizeAggr(reduce);
        this.aggrModule = null;
    }

    public void setAggr(Aggregation module) {
        if (module == null) {
            throw new IllegalArgumentException("Aggregation module must not be null");
        }
        this.aggrModule = module;
        this.aggr = null;
        register_module("aggr_module", module);
    }

    public String getAggr() {
        return aggrModule != null ? aggrModule.getClass().getSimpleName() : aggr;
    }

    public Aggregation getAggrModule() {
        return aggrModule;
    }

    public String getFlow() {
        return flow;
    }

    public void setFlow(String flow) {
        this.flow = normalizeFlow(flow);
    }

    public int getNodeDim() {
        return nodeDim;
    }

    // ========================================================================
    // Forward API
    // ========================================================================

    /** Standard sparse-layer forward. Dense / special layers may throw or override freely. */
    public abstract Tensor forward(Tensor x, Tensor edge_index);

    /**
     * Edge-attr forward. Default throws — subclasses that need edge features override.
     * Layers that can ignore edge_attr may call {@link #forward(Tensor, Tensor)}.
     */
    public Tensor forward(Tensor x, Tensor edge_index, Tensor edge_attr) {
        throw new UnsupportedOperationException(
                getClass().getName() + " does not implement forward(x, edge_index, edge_attr)");
    }

    /** Varargs dispatch for generic callers. */
    public final Tensor forward(Tensor[] args) {
        if (args == null || args.length < 2) {
            throw new IllegalArgumentException("forward requires at least (x, edge_index)");
        }
        if (args.length == 2) {
            return forward(args[0], args[1]);
        }
        if (args.length == 3) {
            return forward(args[0], args[1], args[2]);
        }
        throw new IllegalArgumentException(
                "Unsupported number of arguments for GNN forward: " + args.length);
    }

    // ========================================================================
    // Public propagate overloads → single propagateImpl
    //
    // NOTE: Java cannot overload (Tensor,Tensor,Tensor) twice. The 3-tensor
    // form is therefore ALWAYS interpreted as homogeneous (x, edgeAttr).
    // Bipartite without edge features uses:
    //   propagate(edge, xSrc, xDst, (Tensor) null)
    //   propagate(edge, xSrc, long[] size)
    //   propagate(edge, xSrc, xDst, long numDst)
    // Subclasses may override these for specialized paths (GEN/Spline/…).
    // ========================================================================

    /** Homogeneous: x is both source and destination features. */
    public Tensor propagate(Tensor edgeIndex, Tensor x) {
        return propagateImpl(edgeIndex, x, x, null, null);
    }

    /**
     * Homogeneous with edge attributes / weights (or bipartite if a subclass
     * overrides). Base implementation treats the third tensor as edgeAttr.
     */
    public Tensor propagate(Tensor edgeIndex, Tensor x, Tensor edgeAttr) {
        return propagateImpl(edgeIndex, x, x, edgeAttr, null);
    }

    /**
     * Size override: {@code size = {N_src, N_dst}}.
     * Both ends use {@code x} as features; aggregation dim is {@code size[1]}.
     * For true bipartite features prefer
     * {@link #propagate(Tensor, Tensor, Tensor, Tensor, long[])}.
     */
    public Tensor propagate(Tensor edgeIndex, Tensor x, long[] size) {
        // Bipartite-size with a single feature tensor: messages from x, write to size[1].
        // If size[0] != size[1], still lift from x (caller must ensure indices fit).
        return propagateImpl(edgeIndex, x, x, null, size);
    }

    /** Bipartite + optional edge attributes ({@code edgeAttr} may be null). */
    public Tensor propagate(Tensor edgeIndex, Tensor xSrc, Tensor xDst, Tensor edgeAttr) {
        return propagateImpl(edgeIndex, xSrc, xDst, edgeAttr, null);
    }

    /** Bipartite + edge attributes + explicit size. */
    public Tensor propagate(Tensor edgeIndex, Tensor xSrc, Tensor xDst,
                            Tensor edgeAttr, long[] size) {
        return propagateImpl(edgeIndex, xSrc, xDst, edgeAttr, size);
    }

    /**
     * Legacy DNAConv / CGConv shape: {@code numNodes} is N_dst
     * (N_src from {@code xSrc} when available). {@code edgeAttr} may be edge weight.
     */
    public Tensor propagate(Tensor edgeIndex, Tensor xSrc, Tensor xDst,
                            Tensor edgeAttr, long numNodes) {
        long nSrc = xSrc != null ? xSrc.size(0) : numNodes;
        return propagateImpl(edgeIndex, xSrc, xDst, edgeAttr, new long[]{nSrc, numNodes});
    }

    /**
     * Bipartite with explicit destination count only.
     * {@code size} is N_dst; N_src is taken from {@code xSrc}.
     */
    public Tensor propagate(Tensor edgeIndex, Tensor xSrc, Tensor xDst, long size) {
        long nSrc = xSrc != null ? xSrc.size(0) : size;
        return propagateImpl(edgeIndex, xSrc, xDst, null, new long[]{nSrc, size});
    }

    // ========================================================================
    // Core implementation
    // ========================================================================

    /**
     * Unified message-passing implementation.
     *
     * @param edgeIndex [2, E]
     * @param xSrc      source node features [N_src, F...] (may equal xDst)
     * @param xDst      destination node features [N_dst, F...]
     * @param edgeAttr  optional edge features / weights [E] or [E, F_e]
     * @param size      optional {N_src, N_dst}; inferred when null
     */
    protected Tensor propagateImpl(Tensor edgeIndex, Tensor xSrc, Tensor xDst,
                                   Tensor edgeAttr, long[] size) {
        if (edgeIndex == null) {
            throw new NullPointerException("edgeIndex must not be null");
        }
        if (edgeIndex.dim() != 2 || edgeIndex.size(0) != 2) {
            throw new IllegalArgumentException(
                    "edgeIndex must have shape [2, E], got dim=" + edgeIndex.dim()
                            + " size0=" + (edgeIndex.dim() > 0 ? edgeIndex.size(0) : -1));
        }

        // 1. Flow → (index_j source, index_i target)
        Tensor[] endpoints = resolveFlow(edgeIndex);
        Tensor index_j = endpoints[0];
        Tensor index_i = endpoints[1];

        // 2. Size
        long[] resolved = resolveSize(edgeIndex, xSrc, xDst, index_j, index_i, size);
        long nSrc = resolved[0];
        long nDst = resolved[1];

        // 3. Stash transient state for hooks (attention etc.)
        this._edgeIndex = edgeIndex;
        this._index_j = index_j;
        this._index_i = index_i;
        this._size = resolved;

        try {
            // 4. Fused path
            if (xSrc != null) {
                Tensor fused = messageAndAggregate(edgeIndex, xSrc);
                if (fused != null) {
                    return update(fused, xDst != null ? xDst : xSrc);
                }
            }

            if (xSrc == null) {
                throw new IllegalArgumentException("xSrc must not be null when fused path is unused");
            }

            // 5. Bounds check on source indices
            if (index_j.size(0) > 0) {
                long maxJ = index_j.max().item_long();
                if (maxJ >= xSrc.size(0)) {
                    throw new IndexOutOfBoundsException(String.format(
                            "source index %d out of bounds for xSrc with %d nodes",
                            maxJ, xSrc.size(0)));
                }
            }

            // 6. Lift
            Tensor x_j = lift(xSrc, index_j);
            Tensor x_i = null;
            boolean needXi = needsX_i() || (xDst != null && xDst != xSrc);
            if (needXi) {
                Tensor dstFeat = xDst != null ? xDst : xSrc;
                if (index_i.size(0) > 0) {
                    long maxI = index_i.max().item_long();
                    if (maxI >= dstFeat.size(0)) {
                        throw new IndexOutOfBoundsException(String.format(
                                "target index %d out of bounds for xDst with %d nodes",
                                maxI, dstFeat.size(0)));
                    }
                }
                x_i = lift(dstFeat, index_i);
            }

            // 7. Message
            Tensor msg = collectMessage(x_j, x_i, edgeIndex, edgeAttr, nDst);

            // 8. Aggregate
            Tensor out = aggregate(msg, index_i, nDst);

            // 9. Update
            return update(out, xDst != null ? xDst : xSrc);
        } finally {
            // Clear transient refs (do not close tensors — caller / autograd owns them)
            this._edgeIndex = null;
            this._index_j = null;
            this._index_i = null;
            this._size = null;
        }
    }

    /**
     * Collect message: always invoke the 5-arg hook so existing subclass {@code @Override}s
     * (GAT/GCN/…) run. Default 5-arg delegates down the arity chain.
     */
    protected Tensor collectMessage(Tensor x_j, Tensor x_i, Tensor edgeIndex,
                                    Tensor edgeAttr, long numNodes) {
        // Provide a non-null x_i placeholder when not lifted so 5-arg overrides that only
        // use x_j still work (they typically ignore x_i).
        Tensor xi = x_i != null ? x_i : x_j;
        return message(x_j, xi, edgeIndex, edgeAttr, numNodes);
    }

    // ========================================================================
    // Hooks (non-abstract defaults)
    // ========================================================================

    /** Identity message (GraphSAGE / GIN style). */
    protected Tensor message(Tensor x_j) {
        return x_j;
    }

    /**
     * Edge-weighted message. Rank-1 {@code edgeAttr} [E] is broadcast as [E,1];
     * higher-rank attrs are left to subclasses (default: ignore and return x_j).
     */
    protected Tensor message(Tensor x_j, Tensor edgeAttr) {
        if (edgeAttr == null) {
            return message(x_j);
        }
        if (edgeAttr.dim() == 1) {
            return x_j.mul(edgeAttr.view(new long[]{-1, 1}));
        }
        // [E, 1] already
        if (edgeAttr.dim() == 2 && edgeAttr.size(1) == 1) {
            return x_j.mul(edgeAttr);
        }
        return message(x_j);
    }

    /** Default: ignore x_i, use edge-weighted path. Attention layers override. */
    public Tensor message(Tensor x_j, Tensor x_i, Tensor edgeAttr) {
        return message(x_j, edgeAttr);
    }

    /**
     * Full message hook used by the pipeline. <b>Non-abstract</b> — default delegates
     * to the 3-arg form. Existing layers may {@code @Override} this signature.
     */
    public Tensor message(Tensor x_j, Tensor x_i, Tensor edge_index,
                          Tensor edge_attr, long numNodes) {
        return message(x_j, x_i, edge_attr);
    }

    /**
     * 4-arg bridge kept for older call sites / overrides.
     */
    protected Tensor message(Tensor x_j, Tensor x_i, Tensor edgeIndex, Tensor edgeAttr) {
        long n = _size != null ? _size[1] : (x_j != null ? x_j.size(0) : 0);
        return message(x_j, x_i, edgeIndex, edgeAttr, n);
    }

    /**
     * Aggregate edge messages into node embeddings.
     * Uses {@link #aggrModule} when set, else string {@link #aggr} via {@link AggrUtils}.
     */
    public Tensor aggregate(Tensor inputs, Tensor index, long dimSize) {
        if (aggrModule != null) {
            return aggrModule.forward(inputs, index, dimSize);
        }
        String reduce = aggr != null ? aggr : "sum";
        return AggrUtils.scatter(inputs, index, dimSize, reduce);
    }

    /** CSR-style aggregate; default ignores ptr. */
    public Tensor aggregate(Tensor inputs, Tensor index, Tensor ptr, long dimSize) {
        if (aggrModule != null) {
            return aggrModule.forward(inputs, index, ptr, dimSize);
        }
        return aggregate(inputs, index, dimSize);
    }

    /** Update without residual features. */
    public Tensor update(Tensor inputs) {
        return inputs;
    }

    /** Update with original (destination) node features available. Default: identity. */
    public Tensor update(Tensor inputs, Tensor x) {
        return update(inputs);
    }

    /**
     * Optional fused message+aggregate (e.g. SpMM). Return non-null to skip the
     * lift → message → aggregate path. Default: {@code null}.
     */
    protected Tensor messageAndAggregate(Tensor edgeIndex, Tensor x) {
        return null;
    }

    /**
     * When true, destination features are always lifted to {@code x_i} even in the
     * homogeneous case (required by attention). Default false for efficiency.
     */
    protected boolean needsX_i() {
        return false;
    }

    // ========================================================================
    // Helpers
    // ========================================================================

    /** Gather node features onto edges: {@code src.index_select(0, index)}. */
    protected Tensor lift(Tensor src, Tensor index) {
        if (src == null) {
            throw new NullPointerException("lift src must not be null");
        }
        index = AggrUtils.asLongIndex(index);
        if (index.size(0) == 0) {
            // Empty edge set: return empty [0, F...]
            long[] shape = src.shape();
            long[] outShape = new long[shape.length];
            outShape[0] = 0;
            System.arraycopy(shape, 1, outShape, 1, shape.length - 1);
            return src.new_empty(outShape);
        }
        return src.index_select(0, index);
    }

    /**
     * Apply flow: returns {index_j (source), index_i (target)}.
     */
    protected Tensor[] resolveFlow(Tensor edgeIndex) {
        Tensor row = AggrUtils.asLongIndex(edgeIndex.select(0, 0));
        Tensor col = AggrUtils.asLongIndex(edgeIndex.select(0, 1));
        if ("target_to_source".equals(flow)) {
            return new Tensor[]{col, row};
        }
        // default source_to_target: message flows j→i, index_j=row, index_i=col
        return new Tensor[]{row, col};
    }

    /**
     * Resolve {N_src, N_dst}.
     */
    protected long[] resolveSize(Tensor edgeIndex, Tensor xSrc, Tensor xDst,
                                 Tensor index_j, Tensor index_i, long[] size) {
        if (size != null) {
            if (size.length != 2) {
                throw new IllegalArgumentException("size must be long[2] {N_src, N_dst}");
            }
            return new long[]{size[0], size[1]};
        }
        return GraphUtils.bipartite_size(edgeIndex, xSrc, xDst);
    }

    // ========================================================================
    // Validation
    // ========================================================================

    /**
     * Normalize common aliases. Unknown strings (e.g. GENConv "softmax"/"powermean")
     * are kept as-is — subclasses that override {@link #aggregate} may use custom names.
     * Default {@link #aggregate} still routes through {@link AggrUtils} which validates.
     */
    private static String normalizeAggr(String reduce) {
        if (reduce == null) {
            return "sum";
        }
        switch (reduce) {
            case "add":
            case "sum":
                return "sum";
            case "mean":
                return "mean";
            case "max":
                return "max";
            case "min":
                return "min";
            case "mul":
            case "prod":
                return "prod";
            default:
                // Custom / layer-specific reduce keys (softmax, powermean, …)
                return reduce;
        }
    }

    private static String normalizeFlow(String flow) {
        if (flow == null) {
            return "source_to_target";
        }
        if (!"source_to_target".equals(flow) && !"target_to_source".equals(flow)) {
            throw new IllegalArgumentException(
                    "Unsupported flow='" + flow + "' (use source_to_target or target_to_source)");
        }
        return flow;
    }
}
