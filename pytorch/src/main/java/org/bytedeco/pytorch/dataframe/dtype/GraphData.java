package org.bytedeco.pytorch.dataframe.dtype;

import java.util.*;
import java.util.stream.Collectors;

/**
 * 图数据容器（GRAPH_VIEW）
 * 支持同构图/异构图、有向图/无向图，对齐Neo4j图模型设计
 */
public class GraphData extends AbstractDataValue implements StructuredData {
    private static final long serialVersionUID = 1L;

    // 图名称
    private String graphName;
    // 是否异构图（节点/边类型多样）
    private boolean heterogeneous;
    // 默认是否有向图
    private boolean directed;
    // 节点集合（ID -> 节点）
    private final Map<String, GraphNode> nodes = new HashMap<>();
    // 边集合（ID -> 边）
    private final Map<String, GraphEdge> edges = new HashMap<>();
    // 图级属性
    private final Map<String, Object> graphProperties = new HashMap<>();

    // 空构造
    public GraphData() {
        this("default_graph", false, true);
    }

    /**
     * 核心构造函数
     * @param graphName 图名称
     * @param heterogeneous 是否异构图
     * @param directed 是否有向图
     */
    public GraphData(String graphName, boolean heterogeneous, boolean directed) {
        this.graphName = Objects.requireNonNull(graphName, "图名称不能为空");
        this.heterogeneous = heterogeneous;
        this.directed = directed;
    }

    // ========== 节点操作 ==========
    public void addNode(GraphNode node) {
        Objects.requireNonNull(node, "节点不能为空");
        if (nodes.containsKey(node.getNodeId())) {
            throw new IllegalArgumentException("节点ID已存在：" + node.getNodeId());
        }
        nodes.put(node.getNodeId(), node);
    }

    public GraphNode getNode(String nodeId) {
        return nodes.get(Objects.requireNonNull(nodeId));
    }

    public void removeNode(String nodeId) {
        String id = Objects.requireNonNull(nodeId);
        // 删除节点时同时删除关联的边
        edges.values().removeIf(edge -> edge.getStartNodeId().equals(id) || edge.getEndNodeId().equals(id));
        nodes.remove(id);
    }

    // ========== 边操作 ==========
    public void addEdge(GraphEdge edge) {
        Objects.requireNonNull(edge, "边不能为空");
        // 校验起始/结束节点是否存在
        if (!nodes.containsKey(edge.getStartNodeId())) {
            throw new IllegalArgumentException("起始节点不存在：" + edge.getStartNodeId());
        }
        if (!nodes.containsKey(edge.getEndNodeId())) {
            throw new IllegalArgumentException("结束节点不存在：" + edge.getEndNodeId());
        }
        if (edges.containsKey(edge.getEdgeId())) {
            throw new IllegalArgumentException("边ID已存在：" + edge.getEdgeId());
        }
        edges.put(edge.getEdgeId(), edge);
    }

    public GraphEdge getEdge(String edgeId) {
        return edges.get(Objects.requireNonNull(edgeId));
    }

    public void removeEdge(String edgeId) {
        edges.remove(Objects.requireNonNull(edgeId));
    }

    // ========== 图属性操作 ==========
    public void setGraphProperty(String key, Object value) {
        graphProperties.put(Objects.requireNonNull(key), value);
    }

    public Object getGraphProperty(String key) {
        return graphProperties.get(Objects.requireNonNull(key));
    }

    // ========== 图查询辅助方法 ==========
    /**
     * 获取节点的所有出边（有向图）/关联边（无向图）
     */
    public List<GraphEdge> getEdgesFromNode(String nodeId) {
        String id = Objects.requireNonNull(nodeId);
        return edges.values().stream()
                .filter(edge -> edge.getStartNodeId().equals(id) || (!edge.isDirected() && edge.getEndNodeId().equals(id)))
                .collect(Collectors.toList());
    }

    /**
     * 获取节点的所有入边（仅有向图）
     */
    public List<GraphEdge> getEdgesToNode(String nodeId) {
        String id = Objects.requireNonNull(nodeId);
        if (!directed) {
            throw new UnsupportedOperationException("无向图不支持入边查询");
        }
        return edges.values().stream()
                .filter(edge -> edge.getEndNodeId().equals(id))
                .collect(Collectors.toList());
    }

    /**
     * 获取指定标签的所有节点
     */
    public List<GraphNode> getNodesByLabel(String label) {
        Objects.requireNonNull(label);
        return nodes.values().stream()
                .filter(node -> node.getLabels().contains(label))
                .collect(Collectors.toList());
    }

    /**
     * 获取指定类型的所有边
     */
    public List<GraphEdge> getEdgesByType(String relationshipType) {
        Objects.requireNonNull(relationshipType);
        return edges.values().stream()
                .filter(edge -> edge.getRelationshipType().equals(relationshipType))
                .collect(Collectors.toList());
    }

    // ========== 实现AbstractLanceData抽象方法 ==========
    @Override
    public String getDataType() {
        return "GRAPH_VIEW";
    }

    @Override
    public Object toArrowCompatible() {
        // Arrow适配：返回图的核心元数据（节点/边数量、类型、属性等）
        Map<String, Object> arrowData = new LinkedHashMap<>();
        arrowData.put("graphName", graphName);
        arrowData.put("heterogeneous", heterogeneous);
        arrowData.put("directed", directed);
        arrowData.put("nodeCount", nodes.size());
        arrowData.put("edgeCount", edges.size());
        arrowData.put("graphProperties", graphProperties);

        // 节点/边的元数据（避免传输完整数据，仅传统计信息）
        Map<String, Object> nodeMeta = new HashMap<>();
        nodeMeta.put("labels", nodes.values().stream()
                .flatMap(node -> node.getLabels().stream())
                .distinct()
                .collect(Collectors.toList()));
        arrowData.put("nodeMetadata", nodeMeta);

        Map<String, Object> edgeMeta = new HashMap<>();
        edgeMeta.put("relationshipTypes", edges.values().stream()
                .map(GraphEdge::getRelationshipType)
                .distinct()
                .collect(Collectors.toList()));
        arrowData.put("edgeMetadata", edgeMeta);

        return arrowData;
    }

    @Override
    public String getShortDesc() {
        return String.format("name=%s, type=%s, directed=%s, nodes=%d, edges=%d",
                graphName,
                heterogeneous ? "heterogeneous" : "homogeneous",
                directed,
                nodes.size(),
                edges.size());
    }

    @Override
    public boolean isValid() {
        // 基础校验 + 图专属校验
        return super.isValid()
                && graphName != null && !graphName.isEmpty()
                // 节点ID唯一、边ID唯一
                && new HashSet<>(nodes.keySet()).size() == nodes.size()
                && new HashSet<>(edges.keySet()).size() == edges.size()
                // 所有边的起始/结束节点都存在
                && edges.values().stream().allMatch(edge ->
                        nodes.containsKey(edge.getStartNodeId()) && nodes.containsKey(edge.getEndNodeId()));
    }

    // ========== 实现StructuredData接口 ==========
    @Override
    public int getSize() {
        // 图大小：节点数 + 边数
        return nodes.size() + edges.size();
    }

    @Override
    public Map<String, Object> toMap() {
        Map<String, Object> map = new LinkedHashMap<>();
        map.put("graphName", graphName);
        map.put("heterogeneous", heterogeneous);
        map.put("directed", directed);
        map.put("nodes", nodes);
        map.put("edges", edges);
        map.put("graphProperties", graphProperties);
        map.put("nodeCount", nodes.size());
        map.put("edgeCount", edges.size());
        return map;
    }

    // ========== Getter & Setter ==========
    public String getGraphName() {
        return graphName;
    }

    public void setGraphName(String graphName) {
        this.graphName = Objects.requireNonNull(graphName);
    }

    public boolean isHeterogeneous() {
        return heterogeneous;
    }

    public void setHeterogeneous(boolean heterogeneous) {
        this.heterogeneous = heterogeneous;
    }

    public boolean isDirected() {
        return directed;
    }

    public void setDirected(boolean directed) {
        this.directed = directed;
    }

    public Map<String, GraphNode> getNodes() {
        return Collections.unmodifiableMap(nodes);
    }

    public Map<String, GraphEdge> getEdges() {
        return Collections.unmodifiableMap(edges);
    }

    public int getNodeCount() {
        return nodes.size();
    }

    public int getEdgeCount() {
        return edges.size();
    }

    @Override
    public Number getNumericValue() {
        return null;
    }

    @Override
    public String toString() {
        return String.format("GraphData[name=%s, heterogeneous=%s, directed=%s, nodes=%d, edges=%d]",
                graphName, heterogeneous, directed, nodes.size(), edges.size());
    }

    // ========== 图算法功能 ==========

    /**
     * 社区发现算法 - Louvain算法（简化版）
     * 将图中的节点划分为社区，使得社区内部连接密集，社区间连接稀疏
     */
    public Map<String, Set<String>> communityDetection() {
        Map<String, Set<String>> communities = new HashMap<>();
        Map<String, String> nodeToCommunityyy = new HashMap<>();

        // 初始化：每个节点为一个社区
        int communityId = 0;
        for (String nodeId : nodes.keySet()) {
            String community = "community_" + (communityId++);
            nodeToCommunityyy.put(nodeId, community);
            communities.put(community, new HashSet<>(Collections.singleton(nodeId)));
        }

        boolean improved = true;
        int maxIterations = 10;
        int iteration = 0;

        while (improved && iteration < maxIterations) {
            improved = false;
            iteration++;

            for (String nodeId : nodes.keySet()) {
                String currentCommunity = nodeToCommunityyy.get(nodeId);
                String bestCommunity = currentCommunity;
                double bestModularityGain = 0.0;

                // 获取邻居社区
                Set<String> neighborCommunities = new HashSet<>();
                for (GraphEdge edge : getEdgesFromNode(nodeId)) {
                    String neighborId = edge.getStartNodeId().equals(nodeId) ?
                                      edge.getEndNodeId() : edge.getStartNodeId();
                    neighborCommunities.add(nodeToCommunityyy.get(neighborId));
                }

                // 尝试移动到邻居社区
                for (String neighborCommunity : neighborCommunities) {
                    if (!neighborCommunity.equals(currentCommunity)) {
                        double gain = calculateModularityGain(nodeId, currentCommunity, neighborCommunity, nodeToCommunityyy);
                        if (gain > bestModularityGain) {
                            bestModularityGain = gain;
                            bestCommunity = neighborCommunity;
                        }
                    }
                }

                // 移动节点到最优社区
                if (!bestCommunity.equals(currentCommunity)) {
                    communities.get(currentCommunity).remove(nodeId);
                    if (communities.get(currentCommunity).isEmpty()) {
                        communities.remove(currentCommunity);
                    }
                    communities.get(bestCommunity).add(nodeId);
                    nodeToCommunityyy.put(nodeId, bestCommunity);
                    improved = true;
                }
            }
        }

        return communities;
    }

    /**
     * 计算模块度增益（简化版）
     */
    private double calculateModularityGain(String nodeId, String fromCommunity, String toCommunity,
                                         Map<String, String> nodeToCommunityyy) {
        // 简化的模块度计算：基于邻居连接数
        long connectionsToFrom = getEdgesFromNode(nodeId).stream()
            .filter(edge -> {
                String neighborId = edge.getStartNodeId().equals(nodeId) ?
                                  edge.getEndNodeId() : edge.getStartNodeId();
                return nodeToCommunityyy.get(neighborId).equals(fromCommunity);
            })
            .count();

        long connectionsToTo = getEdgesFromNode(nodeId).stream()
            .filter(edge -> {
                String neighborId = edge.getStartNodeId().equals(nodeId) ?
                                  edge.getEndNodeId() : edge.getStartNodeId();
                return nodeToCommunityyy.get(neighborId).equals(toCommunity);
            })
            .count();

        return (connectionsToTo - connectionsToFrom) / (double) edges.size();
    }

    /**
     * 邻居域异常检测
     * 检测图中的异常节点，基于节点的邻居连接模式
     */
    public List<AnomalyResult> neighborhoodAnomalyDetection(double threshold) {
        List<AnomalyResult> anomalies = new ArrayList<>();

        // 计算每个节点的特征
        Map<String, NodeFeatures> nodeFeatures = new HashMap<>();
        for (String nodeId : nodes.keySet()) {
            NodeFeatures features = calculateNodeFeatures(nodeId);
            nodeFeatures.put(nodeId, features);
        }

        // 计算异常分数
        for (String nodeId : nodes.keySet()) {
            double anomalyScore = calculateAnomalyScore(nodeId, nodeFeatures);
            if (anomalyScore > threshold) {
                anomalies.add(new AnomalyResult(nodeId, anomalyScore, "Neighborhood pattern anomaly"));
            }
        }

        // 按异常分数排序
        anomalies.sort((a, b) -> Double.compare(b.getAnomalyScore(), a.getAnomalyScore()));
        return anomalies;
    }

    /**
     * 计算节点特征
     */
    private NodeFeatures calculateNodeFeatures(String nodeId) {
        List<GraphEdge> edges = getEdgesFromNode(nodeId);
        int degree = edges.size();

        // 计算聚类系数
        Set<String> neighbors = new HashSet<>();
        for (GraphEdge edge : edges) {
            String neighborId = edge.getStartNodeId().equals(nodeId) ?
                              edge.getEndNodeId() : edge.getStartNodeId();
            neighbors.add(neighborId);
        }

        int triangles = 0;
        for (String neighbor1 : neighbors) {
            for (String neighbor2 : neighbors) {
                if (!neighbor1.equals(neighbor2)) {
                    if (hasEdge(neighbor1, neighbor2)) {
                        triangles++;
                    }
                }
            }
        }

        double clusteringCoefficient = neighbors.size() > 1 ?
            triangles / (double) (neighbors.size() * (neighbors.size() - 1)) : 0.0;

        return new NodeFeatures(degree, clusteringCoefficient);
    }

    /**
     * 计算异常分数
     */
    private double calculateAnomalyScore(String nodeId, Map<String, NodeFeatures> allFeatures) {
        NodeFeatures nodeFeatures = allFeatures.get(nodeId);

        // 计算度数和聚类系数的平均值和标准差
        double[] degrees = allFeatures.values().stream().mapToDouble(f -> f.degree).toArray();
        double[] clusteringCoeffs = allFeatures.values().stream().mapToDouble(f -> f.clusteringCoefficient).toArray();

        double avgDegree = Arrays.stream(degrees).average().orElse(0.0);
        double stdDegree = calculateStandardDeviation(degrees, avgDegree);

        double avgClustering = Arrays.stream(clusteringCoeffs).average().orElse(0.0);
        double stdClustering = calculateStandardDeviation(clusteringCoeffs, avgClustering);

        // 标准化分数
        double degreeScore = stdDegree > 0 ? Math.abs(nodeFeatures.degree - avgDegree) / stdDegree : 0;
        double clusteringScore = stdClustering > 0 ? Math.abs(nodeFeatures.clusteringCoefficient - avgClustering) / stdClustering : 0;

        return (degreeScore + clusteringScore) / 2.0;
    }

    private double calculateStandardDeviation(double[] values, double mean) {
        double sum = 0.0;
        for (double value : values) {
            sum += Math.pow(value - mean, 2);
        }
        return Math.sqrt(sum / values.length);
    }

    /**
     * 最短路径发现 - Dijkstra算法
     * 找到两个节点之间的最短路径
     */
    public ShortestPathResult findShortestPath(String startNodeId, String endNodeId) {
        if (!nodes.containsKey(startNodeId) || !nodes.containsKey(endNodeId)) {
            throw new IllegalArgumentException("起始或结束节点不存在");
        }

        Map<String, Double> distances = new HashMap<>();
        Map<String, String> previous = new HashMap<>();
        PriorityQueue<NodeDistance> pq = new PriorityQueue<>(Comparator.comparing(NodeDistance::getDistance));
        Set<String> visited = new HashSet<>();

        // 初始化
        for (String nodeId : nodes.keySet()) {
            distances.put(nodeId, Double.MAX_VALUE);
        }
        distances.put(startNodeId, 0.0);
        pq.offer(new NodeDistance(startNodeId, 0.0));

        while (!pq.isEmpty()) {
            NodeDistance current = pq.poll();
            String currentNodeId = current.getNodeId();

            if (visited.contains(currentNodeId)) {
                continue;
            }
            visited.add(currentNodeId);

            if (currentNodeId.equals(endNodeId)) {
                break;
            }

            // 检查邻居
            for (GraphEdge edge : getEdgesFromNode(currentNodeId)) {
                String neighborId = edge.getStartNodeId().equals(currentNodeId) ?
                                  edge.getEndNodeId() : edge.getStartNodeId();

                if (!visited.contains(neighborId)) {
                    double weight = getEdgeWeight(edge);
                    double newDistance = distances.get(currentNodeId) + weight;

                    if (newDistance < distances.get(neighborId)) {
                        distances.put(neighborId, newDistance);
                        previous.put(neighborId, currentNodeId);
                        pq.offer(new NodeDistance(neighborId, newDistance));
                    }
                }
            }
        }

        // 重构路径
        List<String> path = new ArrayList<>();
        double finalDistance = distances.get(endNodeId);
        boolean pathExists = finalDistance != Double.MAX_VALUE;

        if (pathExists) {
            path = reconstructPath(previous, startNodeId, endNodeId);
        }

        return new ShortestPathResult(startNodeId, endNodeId, path, finalDistance, pathExists);
    }

    private double getEdgeWeight(GraphEdge edge) {
        Object weight = edge.getProperty("weight");
        if (weight instanceof Number) {
            return ((Number) weight).doubleValue();
        }
        return 1.0; // 默认权重
    }

    private List<String> reconstructPath(Map<String, String> previous, String start, String end) {
        List<String> path = new ArrayList<>();
        String current = end;

        while (current != null) {
            path.add(0, current);
            current = previous.get(current);
        }

        if (!path.get(0).equals(start)) {
            return new ArrayList<>(); // 无路径
        }

        return path;
    }

    /**
     * 环路发现算法
     * 使用深度优先搜索检测图中的环路
     */
    public List<CycleResult> findCycles() {
        List<CycleResult> cycles = new ArrayList<>();
        Set<String> visited = new HashSet<>();
        Set<String> recursionStack = new HashSet<>();

        for (String nodeId : nodes.keySet()) {
            if (!visited.contains(nodeId)) {
                List<String> currentPath = new ArrayList<>();
                findCycleDFS(nodeId, visited, recursionStack, currentPath, cycles);
            }
        }

        return cycles;
    }

    private void findCycleDFS(String nodeId, Set<String> visited, Set<String> recursionStack,
                             List<String> currentPath, List<CycleResult> cycles) {
        visited.add(nodeId);
        recursionStack.add(nodeId);
        currentPath.add(nodeId);

        for (GraphEdge edge : getEdgesFromNode(nodeId)) {
            String neighborId = edge.getStartNodeId().equals(nodeId) ?
                              edge.getEndNodeId() : edge.getStartNodeId();

            if (!visited.contains(neighborId)) {
                findCycleDFS(neighborId, visited, recursionStack, currentPath, cycles);
            } else if (recursionStack.contains(neighborId)) {
                // 找到环路
                int cycleStart = currentPath.indexOf(neighborId);
                List<String> cycle = new ArrayList<>(currentPath.subList(cycleStart, currentPath.size()));
                cycle.add(neighborId); // 闭合环路
                cycles.add(new CycleResult(cycle, cycle.size() - 1));
            }
        }

        recursionStack.remove(nodeId);
        currentPath.remove(currentPath.size() - 1);
    }

    /**
     * 团（Clique）发现算法 - Bron-Kerbosch算法
     * 找到图中的最大团
     */
    public List<CliqueResult> findCliques(int minSize) {
        List<CliqueResult> cliques = new ArrayList<>();
        Set<String> R = new HashSet<>(); // 当前团
        Set<String> P = new HashSet<>(nodes.keySet()); // 候选节点
        Set<String> X = new HashSet<>(); // 已处理节点

        bronKerbosch(R, P, X, cliques);

        // 过滤小于最小尺寸的团
        return cliques.stream()
                .filter(clique -> clique.getNodes().size() >= minSize)
                .sorted((a, b) -> Integer.compare(b.getSize(), a.getSize()))
                .collect(Collectors.toList());
    }

    private void bronKerbosch(Set<String> R, Set<String> P, Set<String> X, List<CliqueResult> cliques) {
        if (P.isEmpty() && X.isEmpty()) {
            // 找到最大团
            if (!R.isEmpty()) {
                cliques.add(new CliqueResult(new HashSet<>(R)));
            }
            return;
        }

        // 选择pivot
        String pivot = null;
        int maxConnections = -1;
        Set<String> union = new HashSet<>(P);
        union.addAll(X);

        for (String node : union) {
            int connections = getNeighbors(node, union).size();
            if (connections > maxConnections) {
                maxConnections = connections;
                pivot = node;
            }
        }

        Set<String> pivotNeighbors = pivot != null ? getNeighbors(pivot, P) : new HashSet<>();
        Set<String> candidates = new HashSet<>(P);
        candidates.removeAll(pivotNeighbors);

        for (String v : candidates) {
            Set<String> newR = new HashSet<>(R);
            newR.add(v);

            Set<String> neighbors = getNeighbors(v, nodes.keySet());
            Set<String> newP = new HashSet<>(P);
            newP.retainAll(neighbors);

            Set<String> newX = new HashSet<>(X);
            newX.retainAll(neighbors);

            bronKerbosch(newR, newP, newX, cliques);

            P.remove(v);
            X.add(v);
        }
    }

    private Set<String> getNeighbors(String nodeId, Set<String> candidateNodes) {
        Set<String> neighbors = new HashSet<>();
        for (GraphEdge edge : getEdgesFromNode(nodeId)) {
            String neighborId = edge.getStartNodeId().equals(nodeId) ?
                              edge.getEndNodeId() : edge.getStartNodeId();
            if (candidateNodes.contains(neighborId)) {
                neighbors.add(neighborId);
            }
        }
        return neighbors;
    }

    private boolean hasEdge(String nodeId1, String nodeId2) {
        return edges.values().stream()
                .anyMatch(edge ->
                    (edge.getStartNodeId().equals(nodeId1) && edge.getEndNodeId().equals(nodeId2)) ||
                    (!edge.isDirected() && edge.getStartNodeId().equals(nodeId2) && edge.getEndNodeId().equals(nodeId1))
                );
    }

    // ========== 结果类定义 ==========

    /**
     * 异常检测结果
     */
    public static class AnomalyResult {
        private final String nodeId;
        private final double anomalyScore;
        private final String reason;

        public AnomalyResult(String nodeId, double anomalyScore, String reason) {
            this.nodeId = nodeId;
            this.anomalyScore = anomalyScore;
            this.reason = reason;
        }

        public String getNodeId() { return nodeId; }
        public double getAnomalyScore() { return anomalyScore; }
        public String getReason() { return reason; }

        @Override
        public String toString() {
            return String.format("AnomalyResult[nodeId=%s, score=%.3f, reason=%s]",
                               nodeId, anomalyScore, reason);
        }
    }

    /**
     * 最短路径结果
     */
    public static class ShortestPathResult {
        private final String startNodeId;
        private final String endNodeId;
        private final List<String> path;
        private final double totalDistance;
        private final boolean pathExists;

        public ShortestPathResult(String startNodeId, String endNodeId, List<String> path,
                                double totalDistance, boolean pathExists) {
            this.startNodeId = startNodeId;
            this.endNodeId = endNodeId;
            this.path = path;
            this.totalDistance = totalDistance;
            this.pathExists = pathExists;
        }

        public String getStartNodeId() { return startNodeId; }
        public String getEndNodeId() { return endNodeId; }
        public List<String> getPath() { return path; }
        public double getTotalDistance() { return totalDistance; }
        public boolean isPathExists() { return pathExists; }
        public int getHopCount() { return pathExists ? path.size() - 1 : -1; }

        @Override
        public String toString() {
            if (!pathExists) {
                return String.format("ShortestPath[from=%s, to=%s, status=NO_PATH]",
                                   startNodeId, endNodeId);
            }
            return String.format("ShortestPath[from=%s, to=%s, distance=%.2f, hops=%d, path=%s]",
                               startNodeId, endNodeId, totalDistance, getHopCount(), path);
        }
    }

    /**
     * 环路结果
     */
    public static class CycleResult {
        private final List<String> cycle;
        private final int length;

        public CycleResult(List<String> cycle, int length) {
            this.cycle = cycle;
            this.length = length;
        }

        public List<String> getCycle() { return cycle; }
        public int getLength() { return length; }

        @Override
        public String toString() {
            return String.format("Cycle[length=%d, nodes=%s]", length, cycle);
        }
    }

    /**
     * 团发现结果
     */
    public static class CliqueResult {
        private final Set<String> nodes;

        public CliqueResult(Set<String> nodes) {
            this.nodes = nodes;
        }

        public Set<String> getNodes() { return nodes; }
        public int getSize() { return nodes.size(); }

        @Override
        public String toString() {
            return String.format("Clique[size=%d, nodes=%s]", getSize(), nodes);
        }
    }

    /**
     * 节点特征
     */
    private static class NodeFeatures {
        final int degree;
        final double clusteringCoefficient;

        NodeFeatures(int degree, double clusteringCoefficient) {
            this.degree = degree;
            this.clusteringCoefficient = clusteringCoefficient;
        }
    }

    /**
     * 节点距离（用于Dijkstra算法）
     */
    private static class NodeDistance {
        private final String nodeId;
        private final double distance;

        NodeDistance(String nodeId, double distance) {
            this.nodeId = nodeId;
            this.distance = distance;
        }

        String getNodeId() { return nodeId; }
        double getDistance() { return distance; }
    }

}