package org.bytedeco.pytorch.dataframe.dtype;
import java.io.Serializable;
import java.util.*; /**
 * 图节点（对齐Neo4j Node模型）
 */
public class GraphNode implements Serializable {
    private static final long serialVersionUID = 1L;
    
    // 节点唯一ID（类比Neo4j Node ID）
    private final String nodeId;
    // 节点标签（支持多标签，如["Person", "Employee"]）
    private final Set<String> labels;
    // 节点属性（键值对，支持任意类型）
    private final Map<String, Object> properties;

    public GraphNode(String nodeId, String label) {
        this(nodeId, Collections.singleton(label), new HashMap<>());
    }

    public GraphNode(String nodeId, Set<String> labels, Map<String, Object> properties) {
        this.nodeId = Objects.requireNonNull(nodeId, "节点ID不能为空");
        this.labels = new LinkedHashSet<>(Objects.requireNonNull(labels, "节点标签不能为空"));
        this.properties = new HashMap<>(Objects.requireNonNull(properties, "节点属性不能为空"));
    }

    /**
     * 单标签+属性构造函数（兼容目标调用方式）
     * @param nodeId 节点ID
     * @param label 单个节点标签
     * @param properties 节点属性
     */
    public GraphNode(String nodeId, String label, Map<String, Object> properties) {
        // 复用已有完整构造函数，保证逻辑一致性
        this(
                nodeId,
                Collections.singleton(Objects.requireNonNull(label, "节点标签不能为空")),
                properties
        );
    }

    public GraphNode(String nodeId) {
        this(nodeId, new LinkedHashSet<>(), new HashMap<>());
    }

    // 添加/移除标签
    public void addLabel(String label) {
        labels.add(Objects.requireNonNull(label));
    }

    public void removeLabel(String label) {
        labels.remove(label);
    }

    // 设置/获取属性
    public void setProperty(String key, Object value) {
        properties.put(Objects.requireNonNull(key), value);
    }

    public Object getProperty(String key) {
        return properties.get(key);
    }

    public Object getProperty(String key, Object defaultValue) {
        return properties.getOrDefault(key, defaultValue);
    }

    // Getter（返回不可变视图）
    public String getNodeId() {
        return nodeId;
    }

    public Set<String> getLabels() {
        return Collections.unmodifiableSet(labels);
    }

    public Map<String, Object> getProperties() {
        return Collections.unmodifiableMap(properties);
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        GraphNode graphNode = (GraphNode) o;
        return Objects.equals(nodeId, graphNode.nodeId);
    }

    @Override
    public int hashCode() {
        return Objects.hash(nodeId);
    }

    @Override
    public String toString() {
        return String.format("Node[id=%s, labels=%s, props=%s]", nodeId, labels, properties);
    }
}
