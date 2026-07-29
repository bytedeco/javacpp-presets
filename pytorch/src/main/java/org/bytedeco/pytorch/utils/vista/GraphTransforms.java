package org.bytedeco.pytorch.utils.vista;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;

/**
 * Graph post-processing transforms ported from torchvista {@code graph_transforms.py}.
 *
 * <p>Provides:
 * <ul>
 *   <li>{@link #buildImmediateAncestorMap} — collapsible UI parent pointers</li>
 *   <li>{@link #transformToNestedGraph} / {@link #transformToUnnestedGraph} —
 *       hierarchy reshape used by the compressed view</li>
 *   <li>{@link #applyCompressedView} — experimental repeat-block compression
 *       for Sequential chains with identical type + in/out dims</li>
 * </ul>
 *
 * <p>The interactive HTML renderer primarily needs the flat adj_list plus
 * {@code ancestor_map} / {@code parent_module_to_nodes}. Nested/compress paths
 * are optional ({@link VistaOptions#showCompressedView()}).
 */
public final class GraphTransforms {
    private GraphTransforms() {}

    /**
     * torchvista {@code build_immediate_ancestor_map}:
     * each node → its immediate parent; also wires parent → grandparent so the
     * frontend can walk the full container chain.
     *
     * <p>Returns a map of node → single parent name (string), matching the
     * JSON shape the original template expects ({@code ancestor_map[node] = parent}).
     * For multi-level chains we emit every link {@code child → parent}.
     */
    public static Map<String, List<String>> buildImmediateAncestorMap(
            Map<String, List<String>> ancestorDict,
            Map<String, GraphNode> adjList) {
        // Frontend actually uses a map node → parent (string) in torchvista;
        // we keep List<String> of size 0..1 for uniformity with TraceGraph
        // serialisation, and also emit the full chain links.
        // Looking at render.py: 'ancestor_map': json.dumps(ancestor_map)
        // and build_immediate_ancestor_map returns {node: parent_str, ...}.
        // TraceGraph.toJsonPayload puts this under "ancestor_map". Our HTML
        // renderer accepts either string or list-of-one.
        Map<String, List<String>> out = new LinkedHashMap<>();
        if (ancestorDict == null || adjList == null) return out;

        for (Map.Entry<String, List<String>> e : ancestorDict.entrySet()) {
            String node = e.getKey();
            List<String> ancestors = e.getValue();
            if (ancestors == null || ancestors.isEmpty()) continue;
            if (!adjList.containsKey(node)) continue;

            // immediate parent
            putParent(out, node, ancestors.get(0));
            for (int i = 0; i < ancestors.size() - 1; i++) {
                putParent(out, ancestors.get(i), ancestors.get(i + 1));
            }
        }
        return out;
    }

    /** Variant that returns the torchvista-exact {@code Map<String,String>}. */
    public static Map<String, String> buildImmediateAncestorMapFlat(
            Map<String, List<String>> ancestorDict,
            Map<String, GraphNode> adjList) {
        Map<String, String> flat = new LinkedHashMap<>();
        Map<String, List<String>> multi = buildImmediateAncestorMap(ancestorDict, adjList);
        for (Map.Entry<String, List<String>> e : multi.entrySet()) {
            if (e.getValue() != null && !e.getValue().isEmpty()) {
                flat.put(e.getKey(), e.getValue().get(0));
            }
        }
        return flat;
    }

    private static void putParent(Map<String, List<String>> out, String child, String parent) {
        if (child == null || parent == null) return;
        if (out.containsKey(child)) return; // first assignment wins (immediate)
        out.put(child, Collections.singletonList(parent));
    }

    // =========================================================================
    // Nested graph
    // =========================================================================

    /**
     * Nested node used only during compress transforms. Not part of the public
     * flat adj_list schema.
     */
    public static final class NestedNode {
        public final List<GraphEdge> edges = new ArrayList<>();
        public final Map<String, NestedNode> subgraphs = new LinkedHashMap<>();
        public boolean failed;
        public NodeType nodeType = NodeType.MODULE;
        public List<String> originalIncomingDims = new ArrayList<>();
        public List<String> originalOutgoingDims = new ArrayList<>();

        public NestedNode copyMeta() {
            NestedNode n = new NestedNode();
            n.failed = failed;
            n.nodeType = nodeType;
            n.originalIncomingDims = new ArrayList<>(originalIncomingDims);
            n.originalOutgoingDims = new ArrayList<>(originalOutgoingDims);
            n.edges.addAll(edges);
            return n;
        }
    }

    /**
     * torchvista {@code transform_to_nested_graph}.
     * Builds a hierarchy from {@code node_to_ancestors} and redirects edges to
     * representative nodes at each LCA boundary.
     */
    public static Map<String, NestedNode> transformToNestedGraph(
            Map<String, GraphNode> adjList,
            Map<String, List<String>> nodeToAncestors) {

        Map<String, NestedNode> nodes = new LinkedHashMap<>();
        Set<String> allNodes = new LinkedHashSet<>(adjList.keySet());
        for (List<String> anc : nodeToAncestors.values()) {
            if (anc != null) allNodes.addAll(anc);
        }

        // Pre-compute original dims from flat edges
        Map<String, List<String>> originalIncoming = new HashMap<>();
        Map<String, List<String>> originalOutgoing = new HashMap<>();
        for (Map.Entry<String, GraphNode> e : adjList.entrySet()) {
            for (GraphEdge edge : e.getValue().edges()) {
                originalOutgoing.computeIfAbsent(e.getKey(), k -> new ArrayList<>()).add(edge.dims());
                originalIncoming.computeIfAbsent(edge.target(), k -> new ArrayList<>()).add(edge.dims());
            }
        }

        // Container boundary dims
        Set<String> ancestorOnly = new HashSet<>(allNodes);
        ancestorOnly.removeAll(adjList.keySet());
        for (String container : ancestorOnly) {
            Set<String> descendants = new HashSet<>();
            for (Map.Entry<String, List<String>> e : nodeToAncestors.entrySet()) {
                if (e.getValue() != null && e.getValue().contains(container)) {
                    descendants.add(e.getKey());
                }
            }
            Set<Long> seenIn = new HashSet<>();
            for (String target : descendants) {
                for (Map.Entry<String, GraphNode> se : adjList.entrySet()) {
                    if (descendants.contains(se.getKey())) continue;
                    for (GraphEdge edge : se.getValue().edges()) {
                        if (!target.equals(edge.target())) continue;
                        if (edge.edgeDataId() != null) {
                            if (!seenIn.add(edge.edgeDataId())) continue;
                        }
                        originalIncoming.computeIfAbsent(container, k -> new ArrayList<>())
                                .add(edge.dims());
                    }
                }
            }
            Set<Long> seenOut = new HashSet<>();
            for (String source : descendants) {
                GraphNode sn = adjList.get(source);
                if (sn == null) continue;
                for (GraphEdge edge : sn.edges()) {
                    if (descendants.contains(edge.target())) continue;
                    if (edge.edgeDataId() != null) {
                        if (!seenOut.add(edge.edgeDataId())) continue;
                    }
                    originalOutgoing.computeIfAbsent(container, k -> new ArrayList<>())
                            .add(edge.dims());
                }
            }
        }

        for (String node : allNodes) {
            NestedNode nn = new NestedNode();
            if (adjList.containsKey(node)) {
                GraphNode gn = adjList.get(node);
                nn.failed = gn.failed();
                nn.nodeType = gn.nodeType();
            }
            nn.originalIncomingDims = sortedCopy(originalIncoming.get(node));
            nn.originalOutgoingDims = sortedCopy(originalOutgoing.get(node));
            nodes.put(node, nn);
        }

        // Redirect edges to representative nodes
        Set<String> seenEdgeKeys = new HashSet<>();
        for (Map.Entry<String, GraphNode> e : adjList.entrySet()) {
            String source = e.getKey();
            for (GraphEdge edge : e.getValue().edges()) {
                String target = edge.target();
                String[] reps = representativeNodes(source, target, nodeToAncestors);
                String repSrc = reps[0];
                String repTgt = reps[1];
                if (edge.edgeDataId() != null) {
                    String key = repSrc + "->" + repTgt + "#" + edge.edgeDataId();
                    if (!seenEdgeKeys.add(key)) continue;
                }
                NestedNode srcNode = nodes.get(repSrc);
                if (srcNode == null) continue;
                srcNode.edges.add(new GraphEdge(
                        repTgt, edge.dims(), edge.edgeDataId(), edge.implied()));
            }
        }

        // Nest under immediate parents
        Map<String, NestedNode> root = new LinkedHashMap<>(nodes);
        for (Map.Entry<String, List<String>> e : nodeToAncestors.entrySet()) {
            String node = e.getKey();
            List<String> ancestors = e.getValue();
            if (ancestors == null || ancestors.isEmpty()) continue;
            // zip([node] + ancestors[:-1], ancestors)
            List<String> children = new ArrayList<>();
            children.add(node);
            for (int i = 0; i < ancestors.size() - 1; i++) {
                children.add(ancestors.get(i));
            }
            for (int i = 0; i < children.size() && i < ancestors.size(); i++) {
                String child = children.get(i);
                String parent = ancestors.get(i);
                NestedNode childN = nodes.get(child);
                NestedNode parentN = nodes.get(parent);
                if (childN == null || parentN == null) continue;
                parentN.subgraphs.put(child, childN);
                root.remove(child);
            }
        }
        return root;
    }

    private static String[] representativeNodes(
            String node1, String node2, Map<String, List<String>> nodeToAncestors) {
        List<String> a1 = nodeToAncestors.getOrDefault(node1, Collections.emptyList());
        List<String> a2 = nodeToAncestors.getOrDefault(node2, Collections.emptyList());
        if (a1.isEmpty() && a2.isEmpty()) return new String[]{node1, node2};
        if (a1.isEmpty()) return new String[]{node1, a2.get(a2.size() - 1)};
        if (a2.isEmpty()) return new String[]{a1.get(a1.size() - 1), node2};

        String lca = findLca(a1, a2);
        if (lca == null) {
            return new String[]{a1.get(a1.size() - 1), a2.get(a2.size() - 1)};
        }
        String r1 = elementBefore(a1, lca);
        String r2 = elementBefore(a2, lca);
        return new String[]{
                r1 == null ? node1 : r1,
                r2 == null ? node2 : r2
        };
    }

    private static String findLca(List<String> path1, List<String> path2) {
        // paths ordered immediate-parent-first; compare from root (end)
        String lca = null;
        int i = path1.size() - 1;
        int j = path2.size() - 1;
        while (i >= 0 && j >= 0) {
            if (Objects.equals(path1.get(i), path2.get(j))) {
                lca = path1.get(i);
                i--;
                j--;
            } else {
                break;
            }
        }
        return lca;
    }

    private static String elementBefore(List<String> lst, String target) {
        int idx = lst.indexOf(target);
        if (idx <= 0) return null;
        return lst.get(idx - 1);
    }

    private static List<String> sortedCopy(List<String> src) {
        if (src == null || src.isEmpty()) return new ArrayList<>();
        List<String> copy = new ArrayList<>(src);
        Collections.sort(copy);
        return copy;
    }

    // =========================================================================
    // Unnest
    // =========================================================================

    /**
     * Flatten a nested graph back to leaf adj_list + rebuilt ancestors.
     * Returns {@code [unnestedAdjList, rebuiltAncestors]}.
     */
    @SuppressWarnings("unchecked")
    public static Object[] transformToUnnestedGraph(
            Map<String, NestedNode> nestedGraph,
            Map<String, List<String>> nodeToAncestors) {

        Map<String, NestedNode> allNodes = new LinkedHashMap<>();
        Map<String, List<String>> rebuilt = new LinkedHashMap<>();
        collectNodes(nestedGraph, new ArrayList<>(), allNodes, rebuilt);

        Map<String, GraphNode> unnested = new LinkedHashMap<>();
        for (Map.Entry<String, NestedNode> e : allNodes.entrySet()) {
            if (e.getValue().subgraphs.isEmpty()) {
                GraphNode gn = new GraphNode(e.getValue().nodeType, e.getValue().failed);
                unnested.put(e.getKey(), gn);
            }
        }

        Map<String, List<String>> ingressCache = new HashMap<>();
        Map<String, List<String>> egressCache = new HashMap<>();

        for (Map.Entry<String, NestedNode> e : allNodes.entrySet()) {
            String sourceName = e.getKey();
            List<String> sourceLeaves = getEgressLeaves(sourceName, allNodes, egressCache);
            if (sourceLeaves.isEmpty()) continue;
            for (GraphEdge edge : e.getValue().edges) {
                String targetName = edge.target();
                List<String> targetLeaves = allNodes.containsKey(targetName)
                        ? getIngressLeaves(targetName, allNodes, ingressCache)
                        : Collections.emptyList();
                if (targetLeaves.isEmpty()) continue;
                for (String srcLeaf : sourceLeaves) {
                    for (String tgtLeaf : targetLeaves) {
                        GraphNode srcNode = unnested.get(srcLeaf);
                        if (srcNode == null) continue;
                        srcNode.addEdge(new GraphEdge(
                                tgtLeaf, edge.dims(), edge.edgeDataId(), edge.implied()));
                    }
                }
            }
        }
        return new Object[]{unnested, rebuilt};
    }

    private static void collectNodes(
            Map<String, NestedNode> subgraph,
            List<String> ancestors,
            Map<String, NestedNode> allNodes,
            Map<String, List<String>> rebuilt) {
        for (Map.Entry<String, NestedNode> e : subgraph.entrySet()) {
            allNodes.put(e.getKey(), e.getValue());
            // rebuilt: immediate parent first = reversed ancestors
            List<String> rev = new ArrayList<>(ancestors.size());
            for (int i = ancestors.size() - 1; i >= 0; i--) rev.add(ancestors.get(i));
            rebuilt.put(e.getKey(), rev);
            List<String> next = new ArrayList<>(ancestors);
            next.add(e.getKey());
            collectNodes(e.getValue().subgraphs, next, allNodes, rebuilt);
        }
    }

    private static List<String> getIngressLeaves(
            String nodeName,
            Map<String, NestedNode> allNodes,
            Map<String, List<String>> cache) {
        if (cache.containsKey(nodeName)) return cache.get(nodeName);
        NestedNode node = allNodes.get(nodeName);
        if (node == null || node.subgraphs.isEmpty()) {
            List<String> self = Collections.singletonList(nodeName);
            cache.put(nodeName, self);
            return self;
        }
        Map<String, Integer> incoming = new HashMap<>();
        for (String c : node.subgraphs.keySet()) incoming.put(c, 0);
        for (Map.Entry<String, NestedNode> c : node.subgraphs.entrySet()) {
            for (GraphEdge edge : c.getValue().edges) {
                if (node.subgraphs.containsKey(edge.target())) {
                    incoming.merge(edge.target(), 1, Integer::sum);
                }
            }
        }
        List<String> candidates = new ArrayList<>();
        for (Map.Entry<String, Integer> e : incoming.entrySet()) {
            if (e.getValue() == 0) candidates.add(e.getKey());
        }
        if (candidates.isEmpty()) candidates.addAll(node.subgraphs.keySet());
        List<String> leaves = new ArrayList<>();
        for (String c : candidates) {
            leaves.addAll(getIngressLeaves(c, allNodes, cache));
        }
        cache.put(nodeName, leaves);
        return leaves;
    }

    private static List<String> getEgressLeaves(
            String nodeName,
            Map<String, NestedNode> allNodes,
            Map<String, List<String>> cache) {
        if (cache.containsKey(nodeName)) return cache.get(nodeName);
        NestedNode node = allNodes.get(nodeName);
        if (node == null || node.subgraphs.isEmpty()) {
            List<String> self = Collections.singletonList(nodeName);
            cache.put(nodeName, self);
            return self;
        }
        Map<String, Integer> outgoing = new HashMap<>();
        for (String c : node.subgraphs.keySet()) outgoing.put(c, 0);
        for (Map.Entry<String, NestedNode> c : node.subgraphs.entrySet()) {
            for (GraphEdge edge : c.getValue().edges) {
                if (node.subgraphs.containsKey(edge.target())) {
                    outgoing.merge(c.getKey(), 1, Integer::sum);
                }
            }
        }
        List<String> candidates = new ArrayList<>();
        for (Map.Entry<String, Integer> e : outgoing.entrySet()) {
            if (e.getValue() == 0) candidates.add(e.getKey());
        }
        if (candidates.isEmpty()) candidates.addAll(node.subgraphs.keySet());
        List<String> leaves = new ArrayList<>();
        for (String c : candidates) {
            leaves.addAll(getEgressLeaves(c, allNodes, cache));
        }
        cache.put(nodeName, leaves);
        return leaves;
    }

    // =========================================================================
    // Compressed view (experimental, Sequential repeat blocks)
    // =========================================================================

    /**
     * Apply torchvista-style compressed view in-place on {@link TraceGraph}:
     * nest → simple Sequential-chain repeat compression → unnest, then replace
     * the working adj_list / ancestors / repeat_containers.
     *
     * <p>This is a simplified port of {@code compress_nested_graph}: it detects
     * runs of sibling nodes under a Sequential-named ancestor that share the
     * same type-without-suffix and identical sorted original in/out dims, and
     * collapses each run of length ≥ 2 into a {@code repeat_N_K} container
     * holding the first instance. Full ModuleList injection is omitted (requires
     * live module instance identity that we already discarded after the forward
     * pass); Sequential-level repeats cover the common case.
     */
    @SuppressWarnings("unchecked")
    public static void applyCompressedView(TraceGraph graph) {
        if (graph == null || graph.adjList().isEmpty()) return;

        Map<String, NestedNode> nested = transformToNestedGraph(
                graph.adjList(), graph.nodeToAncestors());

        Set<String> repeatNodes = new LinkedHashSet<>();
        int[] counter = {0};
        Map<String, NestedNode> compressed = compressRoot(
                nested, graph, repeatNodes, counter);

        Object[] pair = transformToUnnestedGraph(compressed, graph.nodeToAncestors());
        Map<String, GraphNode> unnested = (Map<String, GraphNode>) pair[0];
        Map<String, List<String>> rebuilt = (Map<String, List<String>>) pair[1];

        // Strip ModuleList-labelled ancestors from display chains
        for (Map.Entry<String, List<String>> e : rebuilt.entrySet()) {
            List<String> filtered = new ArrayList<>();
            for (String a : e.getValue()) {
                String bare = graph.graphNodeNameToWithoutSuffix().getOrDefault(a, a);
                if (!"ModuleList".equals(bare)) filtered.add(a);
            }
            e.setValue(filtered);
        }

        graph.adjList().clear();
        graph.adjList().putAll(unnested);
        graph.nodeToAncestors().clear();
        graph.nodeToAncestors().putAll(rebuilt);
        graph.repeatContainers().clear();
        graph.repeatContainers().addAll(repeatNodes);
    }

    private static Map<String, NestedNode> compressRoot(
            Map<String, NestedNode> nested,
            TraceGraph graph,
            Set<String> repeatNodes,
            int[] counter) {
        Map<String, NestedNode> out = new LinkedHashMap<>();
        for (Map.Entry<String, NestedNode> e : nested.entrySet()) {
            out.put(e.getKey(), compressNode(e.getKey(), e.getValue(), graph, repeatNodes, counter));
        }
        return out;
    }

    private static NestedNode compressNode(
            String nodeName,
            NestedNode nodeData,
            TraceGraph graph,
            Set<String> repeatNodes,
            int[] counter) {
        NestedNode neu = nodeData.copyMeta();
        String display = graph.graphNodeNameToWithoutSuffix()
                .getOrDefault(nodeName, nodeName);
        boolean isSeq = "Sequential".equals(display) || "ModuleList".equals(display)
                || display.startsWith("Sequential") || display.startsWith("ModuleList");

        if (isSeq && !nodeData.subgraphs.isEmpty()) {
            List<String> chain = getChainFromSubgraph(nodeData.subgraphs);
            neu.subgraphs.putAll(compressChain(chain, nodeData.subgraphs, graph, repeatNodes, counter));
        } else {
            for (Map.Entry<String, NestedNode> e : nodeData.subgraphs.entrySet()) {
                neu.subgraphs.put(e.getKey(),
                        compressNode(e.getKey(), e.getValue(), graph, repeatNodes, counter));
            }
        }
        return neu;
    }

    /** Topological-ish chain: nodes with a single successor inside the subgraph. */
    private static List<String> getChainFromSubgraph(Map<String, NestedNode> subgraph) {
        if (subgraph.isEmpty()) return Collections.emptyList();
        // Pick start: zero in-degree inside subgraph
        Map<String, Integer> indeg = new HashMap<>();
        for (String n : subgraph.keySet()) indeg.put(n, 0);
        for (Map.Entry<String, NestedNode> e : subgraph.entrySet()) {
            for (GraphEdge edge : e.getValue().edges) {
                if (subgraph.containsKey(edge.target())) {
                    indeg.merge(edge.target(), 1, Integer::sum);
                }
            }
        }
        String start = null;
        for (Map.Entry<String, Integer> e : indeg.entrySet()) {
            if (e.getValue() == 0) {
                start = e.getKey();
                break;
            }
        }
        if (start == null) start = subgraph.keySet().iterator().next();

        List<String> chain = new ArrayList<>();
        Set<String> seen = new HashSet<>();
        String cur = start;
        while (cur != null && subgraph.containsKey(cur) && seen.add(cur)) {
            chain.add(cur);
            String next = null;
            for (GraphEdge edge : subgraph.get(cur).edges) {
                if (subgraph.containsKey(edge.target()) && !seen.contains(edge.target())) {
                    next = edge.target();
                    break;
                }
            }
            cur = next;
        }
        // Append any leftovers (branches) in key order
        for (String n : subgraph.keySet()) {
            if (!seen.contains(n)) chain.add(n);
        }
        return chain;
    }

    private static Map<String, NestedNode> compressChain(
            List<String> chain,
            Map<String, NestedNode> parentSubgraph,
            TraceGraph graph,
            Set<String> repeatNodes,
            int[] counter) {
        Map<String, NestedNode> newSub = new LinkedHashMap<>();
        int i = 0;
        while (i < chain.size()) {
            String current = chain.get(i);
            NestedNode curData = parentSubgraph.get(current);
            if (curData == null) {
                i++;
                continue;
            }
            // Try to extend a repeat run
            int j = i + 1;
            while (j < chain.size()) {
                String cand = chain.get(j);
                NestedNode candData = parentSubgraph.get(cand);
                if (candData == null) break;
                if (!sameSignature(current, curData, cand, candData, graph)) break;
                // Also require linear link current-run → cand
                j++;
            }
            int runLen = j - i;
            if (runLen >= 2) {
                String repeatName = "repeat_" + runLen + "_" + (counter[0]++);
                NestedNode rep = new NestedNode();
                rep.nodeType = NodeType.MODULE;
                rep.failed = false;
                // Keep first instance as the sole subgraph exemplar
                NestedNode first = compressNode(current, curData, graph, repeatNodes, counter);
                rep.subgraphs.put(current, first);
                // Edge out of the run toward next node after the run
                if (j < chain.size()) {
                    String nextNode = chain.get(j);
                    String dims = "( )";
                    Long edgeDataId = null;
                    boolean implied = false;
                    NestedNode last = parentSubgraph.get(chain.get(j - 1));
                    if (last != null) {
                        for (GraphEdge edge : last.edges) {
                            dims = edge.dims();
                            edgeDataId = edge.edgeDataId();
                            implied = edge.implied();
                            break;
                        }
                    }
                    rep.edges.add(new GraphEdge(nextNode, dims, edgeDataId, implied));
                }
                // Redirect previous node's edge that pointed at `current`
                if (i > 0) {
                    // previous may already be in newSub
                    String prev = chain.get(i - 1);
                    NestedNode prevNode = newSub.get(prev);
                    if (prevNode == null) {
                        for (NestedNode n : newSub.values()) {
                            if (n.subgraphs.containsKey(prev)) {
                                prevNode = n;
                                break;
                            }
                        }
                    }
                    if (prevNode != null) {
                        for (int ei = 0; ei < prevNode.edges.size(); ei++) {
                            GraphEdge edge = prevNode.edges.get(ei);
                            if (current.equals(edge.target())) {
                                prevNode.edges.set(ei, new GraphEdge(
                                        repeatName, edge.dims(), edge.edgeDataId(), edge.implied()));
                            }
                        }
                    }
                }
                graph.graphNodeNameToWithoutSuffix().put(repeatName, "REPEAT X " + runLen);
                graph.graphNodeDisplayNames().put(repeatName, "REPEAT " + runLen + "x");
                graph.nodeToModulePath().put(repeatName, "");
                newSub.put(repeatName, rep);
                repeatNodes.add(repeatName);
                i = j;
            } else {
                newSub.put(current, compressNode(current, curData, graph, repeatNodes, counter));
                i++;
            }
        }
        return newSub;
    }

    private static boolean sameSignature(
            String aName, NestedNode a,
            String bName, NestedNode b,
            TraceGraph graph) {
        String ta = graph.graphNodeNameToWithoutSuffix().getOrDefault(aName, aName);
        String tb = graph.graphNodeNameToWithoutSuffix().getOrDefault(bName, bName);
        if (!Objects.equals(ta, tb)) return false;
        return Objects.equals(a.originalIncomingDims, b.originalIncomingDims)
                && Objects.equals(a.originalOutgoingDims, b.originalOutgoingDims);
    }
}
