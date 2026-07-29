package org.bytedeco.pytorch.utils.vista;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * One node in the torchvista-style adjacency list.
 *
 * <p>Schema: {@code {edges, failed, node_type}} plus optional original dims
 * used by nested-graph transforms / compressed view.
 */
public final class GraphNode {
    private final List<GraphEdge> edges = new ArrayList<>();
    private boolean failed;
    private NodeType nodeType;
    private List<String> originalIncomingDims;
    private List<String> originalOutgoingDims;

    public GraphNode(NodeType nodeType, boolean failed) {
        this.nodeType = nodeType == null ? NodeType.MODULE : nodeType;
        this.failed = failed;
    }

    public static GraphNode of(NodeType type) {
        return new GraphNode(type, false);
    }

    public static GraphNode failed(NodeType type) {
        return new GraphNode(type, true);
    }

    public List<GraphEdge> edges() {
        return edges;
    }

    public void addEdge(GraphEdge edge) {
        if (edge != null) edges.add(edge);
    }

    public boolean failed() {
        return failed;
    }

    public void setFailed(boolean failed) {
        this.failed = failed;
    }

    public NodeType nodeType() {
        return nodeType;
    }

    public void setNodeType(NodeType nodeType) {
        this.nodeType = nodeType == null ? NodeType.MODULE : nodeType;
    }

    public List<String> originalIncomingDims() {
        return originalIncomingDims;
    }

    public void setOriginalIncomingDims(List<String> dims) {
        this.originalIncomingDims = dims;
    }

    public List<String> originalOutgoingDims() {
        return originalOutgoingDims;
    }

    public void setOriginalOutgoingDims(List<String> dims) {
        this.originalOutgoingDims = dims;
    }

    /** JSON-friendly map matching torchvista adj_list entry. */
    public Map<String, Object> toMap() {
        Map<String, Object> m = new LinkedHashMap<>();
        List<Map<String, Object>> edgeMaps = new ArrayList<>(edges.size());
        for (GraphEdge e : edges) {
            edgeMaps.add(e.toMap());
        }
        m.put("edges", edgeMaps);
        m.put("failed", failed);
        m.put("node_type", nodeType.value());
        if (originalIncomingDims != null) {
            m.put("original_incoming_dims", new ArrayList<>(originalIncomingDims));
        }
        if (originalOutgoingDims != null) {
            m.put("original_outgoing_dims", new ArrayList<>(originalOutgoingDims));
        }
        return m;
    }

    public GraphNode copyShallow() {
        GraphNode n = new GraphNode(nodeType, failed);
        n.edges.addAll(edges);
        if (originalIncomingDims != null) {
            n.originalIncomingDims = new ArrayList<>(originalIncomingDims);
        }
        if (originalOutgoingDims != null) {
            n.originalOutgoingDims = new ArrayList<>(originalOutgoingDims);
        }
        return n;
    }

    @Override
    public String toString() {
        return "GraphNode{type=" + nodeType + ", failed=" + failed + ", edges=" + edges.size() + "}";
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof GraphNode)) return false;
        GraphNode that = (GraphNode) o;
        return failed == that.failed
                && nodeType == that.nodeType
                && Objects.equals(edges, that.edges);
    }

    @Override
    public int hashCode() {
        return Objects.hash(failed, nodeType, edges);
    }
}
