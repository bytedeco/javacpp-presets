package org.bytedeco.pytorch.utils.vista;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Full result of a torchvista-style forward-pass trace.
 *
 * <p>Holds the adjacency list plus all side tables the HTML renderer needs
 * (module_info, func_info, parent maps, display names, ancestor map, …).
 * Serialises to the same JSON payload keys torchvista injects into
 * {@code graph.html}.
 */
public final class TraceGraph {
    private final Map<String, GraphNode> adjList = new LinkedHashMap<>();
    private final Map<String, ModuleInfo> moduleInfo = new LinkedHashMap<>();
    private final Map<String, Map<String, Object>> funcInfo = new LinkedHashMap<>();
    private final Map<String, String> nodeToModulePath = new LinkedHashMap<>();
    private final Map<String, List<String>> parentModuleToNodes = new LinkedHashMap<>();
    private final Map<String, Integer> parentModuleToDepth = new LinkedHashMap<>();
    private final Map<String, String> graphNodeNameToWithoutSuffix = new LinkedHashMap<>();
    private final Map<String, String> graphNodeDisplayNames = new LinkedHashMap<>();
    private final Map<String, String> nodeToAttrName = new LinkedHashMap<>();
    private final Map<String, List<String>> nodeToAncestors = new LinkedHashMap<>();
    private final Set<String> repeatContainers = new LinkedHashSet<>();
    /**
     * Per-node feature / label metadata for Input & Output cards.
     * Keys are node ids ({@code input_user_id}, {@code output}, …).
     * Values are free-form maps: {@code kind, name, feature_type, shape, dtype,
     * vocab_size, embed_dim, pooling, max_len, task, label, …}.
     */
    private final Map<String, Map<String, Object>> nodeMeta = new LinkedHashMap<>();
    private Throwable exception;

    public Map<String, GraphNode> adjList() {
        return adjList;
    }

    public Map<String, ModuleInfo> moduleInfo() {
        return moduleInfo;
    }

    public Map<String, Map<String, Object>> funcInfo() {
        return funcInfo;
    }

    public Map<String, String> nodeToModulePath() {
        return nodeToModulePath;
    }

    public Map<String, List<String>> parentModuleToNodes() {
        return parentModuleToNodes;
    }

    public Map<String, Integer> parentModuleToDepth() {
        return parentModuleToDepth;
    }

    public Map<String, String> graphNodeNameToWithoutSuffix() {
        return graphNodeNameToWithoutSuffix;
    }

    public Map<String, String> graphNodeDisplayNames() {
        return graphNodeDisplayNames;
    }

    public Map<String, String> nodeToAttrName() {
        return nodeToAttrName;
    }

    public Map<String, List<String>> nodeToAncestors() {
        return nodeToAncestors;
    }

    public Set<String> repeatContainers() {
        return repeatContainers;
    }

    public Map<String, Map<String, Object>> nodeMeta() {
        return nodeMeta;
    }

    public Throwable exception() {
        return exception;
    }

    public void setException(Throwable exception) {
        this.exception = exception;
    }

    public boolean hasNodes() {
        return !adjList.isEmpty();
    }

    public int nodeCount() {
        return adjList.size();
    }

    public int edgeCount() {
        int n = 0;
        for (GraphNode node : adjList.values()) {
            n += node.edges().size();
        }
        return n;
    }

    /**
     * Immediate-ancestor map used by the collapsible frontend
     * (torchvista {@code build_immediate_ancestor_map}): {@code child → parent}.
     */
    public Map<String, String> buildImmediateAncestorMap() {
        return GraphTransforms.buildImmediateAncestorMapFlat(nodeToAncestors, adjList);
    }

    /** Full JSON payload for the interactive HTML template. */
    public Map<String, Object> toJsonPayload() {
        // Derive per-node in/out dims from edges so the renderer can paint
        // shape chips without re-walking the graph in JS.
        Map<String, List<String>> incoming = new LinkedHashMap<>();
        Map<String, List<String>> outgoing = new LinkedHashMap<>();
        for (Map.Entry<String, GraphNode> e : adjList.entrySet()) {
            for (GraphEdge edge : e.getValue().edges()) {
                String d = edge.dims() == null ? "" : edge.dims();
                outgoing.computeIfAbsent(e.getKey(), k -> new ArrayList<>()).add(d);
                incoming.computeIfAbsent(edge.target(), k -> new ArrayList<>()).add(d);
            }
        }
        for (Map.Entry<String, GraphNode> e : adjList.entrySet()) {
            GraphNode n = e.getValue();
            if (n.originalIncomingDims() == null) {
                n.setOriginalIncomingDims(incoming.getOrDefault(e.getKey(), List.of()));
            }
            if (n.originalOutgoingDims() == null) {
                n.setOriginalOutgoingDims(outgoing.getOrDefault(e.getKey(), List.of()));
            }
        }

        Map<String, Object> payload = new LinkedHashMap<>();

        Map<String, Object> adj = new LinkedHashMap<>();
        for (Map.Entry<String, GraphNode> e : adjList.entrySet()) {
            adj.put(e.getKey(), e.getValue().toMap());
        }
        payload.put("adj_list", adj);

        Map<String, Object> modInfo = new LinkedHashMap<>();
        for (Map.Entry<String, ModuleInfo> e : moduleInfo.entrySet()) {
            modInfo.put(e.getKey(), e.getValue().toMap());
        }
        payload.put("module_info", modInfo);
        payload.put("func_info", deepCopyMaps(funcInfo));
        payload.put("node_to_module_path", new LinkedHashMap<>(nodeToModulePath));
        payload.put("parent_module_to_nodes", copyListMap(parentModuleToNodes));
        payload.put("parent_module_to_depth", new LinkedHashMap<>(parentModuleToDepth));
        payload.put("graph_node_name_to_without_suffix", new LinkedHashMap<>(graphNodeNameToWithoutSuffix));
        payload.put("graph_node_display_names", new LinkedHashMap<>(graphNodeDisplayNames));
        payload.put("node_to_attr_name", new LinkedHashMap<>(nodeToAttrName));
        // torchvista schema: ancestor_map[node] = parent (string)
        payload.put("ancestor_map", buildImmediateAncestorMap());
        payload.put("repeat_containers", new ArrayList<>(repeatContainers));
        payload.put("node_meta", deepCopyMaps(nodeMeta));
        return payload;
    }

    private static Map<String, Object> deepCopyMaps(Map<String, Map<String, Object>> src) {
        Map<String, Object> out = new LinkedHashMap<>();
        for (Map.Entry<String, Map<String, Object>> e : src.entrySet()) {
            out.put(e.getKey(), new LinkedHashMap<>(e.getValue()));
        }
        return out;
    }

    private static Map<String, Object> copyListMap(Map<String, List<String>> src) {
        Map<String, Object> out = new LinkedHashMap<>();
        for (Map.Entry<String, List<String>> e : src.entrySet()) {
            out.put(e.getKey(), new ArrayList<>(e.getValue()));
        }
        return out;
    }

    /** Human-readable summary for console / tests. */
    public String summary() {
        StringBuilder sb = new StringBuilder();
        sb.append("TraceGraph{nodes=").append(nodeCount())
                .append(", edges=").append(edgeCount());
        if (exception != null) {
            sb.append(", error=").append(exception.getClass().getSimpleName())
                    .append(": ").append(exception.getMessage());
        }
        sb.append('}');
        return sb.toString();
    }

    @Override
    public String toString() {
        return summary();
    }
}
