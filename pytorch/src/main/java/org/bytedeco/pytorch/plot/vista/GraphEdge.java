package org.bytedeco.pytorch.plot.vista;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;

/**
 * Directed edge in the forward-pass adjacency list.
 *
 * <p>Mirrors torchvista's edge dict:
 * {@code {target, dims, edge_data_id?, is_implied_edge?}}.
 */
public final class GraphEdge {
    private final String target;
    private final String dims;
    private final Long edgeDataId;
    private final boolean implied;

    public GraphEdge(String target, String dims) {
        this(target, dims, null, false);
    }

    public GraphEdge(String target, String dims, Long edgeDataId, boolean implied) {
        this.target = Objects.requireNonNull(target, "target");
        this.dims = dims == null ? "" : dims;
        this.edgeDataId = edgeDataId;
        this.implied = implied;
    }

    public String target() {
        return target;
    }

    public String dims() {
        return dims;
    }

    public Long edgeDataId() {
        return edgeDataId;
    }

    public boolean implied() {
        return implied;
    }

    /** JSON-friendly map matching torchvista edge schema. */
    public Map<String, Object> toMap() {
        Map<String, Object> m = new LinkedHashMap<>();
        m.put("target", target);
        m.put("dims", dims);
        if (edgeDataId != null) {
            m.put("edge_data_id", edgeDataId);
        }
        if (implied) {
            m.put("is_implied_edge", true);
        }
        return m;
    }

    @Override
    public String toString() {
        return "GraphEdge{" + target + " dims=" + dims + (implied ? " implied" : "") + "}";
    }
}
