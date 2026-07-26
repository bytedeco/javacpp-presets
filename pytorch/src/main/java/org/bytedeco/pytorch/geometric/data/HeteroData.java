package org.bytedeco.pytorch.geometric.data;
import org.bytedeco.pytorch.autograd.*;

import java.util.HashMap;
import java.util.Map;

public class HeteroData extends BaseData {
    // Disjoint storage for nodes and edges
    private final Map<String, NodeStorage> nodeStores = new HashMap<>();
    private final Map<EdgeRel, EdgeStorage> edgeStores = new HashMap<>();

    // Accessor for Node Storage: data.get("paper")
    public NodeStorage get(String nodeType) {
        return nodeStores.computeIfAbsent(nodeType, k -> new NodeStorage());
    }

    // Accessor for Edge Storage: data.get("author", "writes", "paper")
    public EdgeStorage get(String src, String rel, String dst) {
        EdgeRel key = new EdgeRel(src, rel, dst);
        return edgeStores.computeIfAbsent(key, k -> new EdgeStorage());
    }

    @Override
    public BaseData to(String device) {
        return null;
    }

    @Override
    public BaseData pinMemory() {
        return null;
    }

    @Override
    public boolean validate() {
        return false;
    }

    // Record for Edge Relation Key
    public record EdgeRel(String src, String rel, String dst) {}
}