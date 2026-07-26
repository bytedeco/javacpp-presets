package org.bytedeco.pytorch.data.dataframe.dtype;
import java.io.Serializable;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;
import java.util.Objects; /**
 * 图边（对齐Neo4j Relationship模型）
 */
public class GraphEdge implements Serializable {
    private static final long serialVersionUID = 1L;
    
    // 边唯一ID
    private final String edgeId;
    // 起始节点ID
    private final String startNodeId;
    // 结束节点ID
    private final String endNodeId;
    // 边类型（如"KNOWS"、"WORKS_AT"）
    private final String relationshipType;
    // 边属性
    private final Map<String, Object> properties;
    // 是否有向边
    private boolean directed;

    public GraphEdge(String edgeId, String startNodeId, String endNodeId, String relationshipType, boolean directed) {
        this(edgeId, startNodeId, endNodeId, relationshipType, directed, new HashMap<>());
    }

    public GraphEdge(String edgeId, String startNodeId, String endNodeId, String relationshipType, 
                     boolean directed, Map<String, Object> properties) {
        this.edgeId = Objects.requireNonNull(edgeId, "边ID不能为空");
        this.startNodeId = Objects.requireNonNull(startNodeId, "起始节点ID不能为空");
        this.endNodeId = Objects.requireNonNull(endNodeId, "结束节点ID不能为空");
        this.relationshipType = Objects.requireNonNull(relationshipType, "边类型不能为空");
        this.directed = directed;
        this.properties = new HashMap<>(Objects.requireNonNull(properties, "边属性不能为空"));
    }

    // 设置/获取属性
    public void setProperty(String key, Object value) {
        properties.put(Objects.requireNonNull(key), value);
    }

    public Object getProperty(String key) {
        return properties.get(key);
    }

    // Getter
    public String getEdgeId() {
        return edgeId;
    }

    public String getStartNodeId() {
        return startNodeId;
    }

    public String getEndNodeId() {
        return endNodeId;
    }

    public String getRelationshipType() {
        return relationshipType;
    }

    public boolean isDirected() {
        return directed;
    }

    public Map<String, Object> getProperties() {
        return Collections.unmodifiableMap(properties);
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        GraphEdge graphEdge = (GraphEdge) o;
        return Objects.equals(edgeId, graphEdge.edgeId);
    }

    @Override
    public int hashCode() {
        return Objects.hash(edgeId);
    }

    @Override
    public String toString() {
        return String.format("Edge[id=%s, %s-%s->%s (directed=%s), props=%s]",
                edgeId, startNodeId, relationshipType, endNodeId, directed, properties);
    }

    public void setDirected(boolean b) {
        this.directed = b;
    }
}
