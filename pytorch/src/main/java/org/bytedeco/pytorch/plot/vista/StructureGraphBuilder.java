package org.bytedeco.pytorch.plot.vista;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import org.bytedeco.pytorch.data.serialize.StructureSpec;
import org.bytedeco.pytorch.inductor.AOTIModelPackageLoader;

/**
 * Build a {@link TraceGraph} from static structure (no live forward pass).
 *
 * <p>Used when only {@code *.structure.json}, a weight bag without sample
 * inputs, or an AOTI package metadata is available. Edges are topological
 * (parent→child / Sequential chain order) with empty dims; the interactive
 * viewer still supports collapse / inspect of hyperparameters and parameter
 * names from the structure spec.
 */
public final class StructureGraphBuilder {
    private StructureGraphBuilder() {}

    public static TraceGraph fromStructure(StructureSpec spec) {
        TraceGraph g = new TraceGraph();
        if (spec == null || spec.nodes == null || spec.nodes.isEmpty()) {
            return g;
        }

        String rootPath = "";
        StructureSpec.Node root = spec.nodes.get(rootPath);
        if (root == null) {
            // pick first
            rootPath = spec.nodes.keySet().iterator().next();
            root = spec.nodes.get(rootPath);
        }

        // Input / output placeholders so the viewer has anchors
        g.adjList().put("input_0", GraphNode.of(NodeType.INPUT));
        g.graphNodeDisplayNames().put("input_0", "input");
        g.graphNodeNameToWithoutSuffix().put("input_0", "input");
        g.nodeToAncestors().put("input_0", new ArrayList<>());

        int[] counter = {1};
        Map<String, String> pathToNode = new LinkedHashMap<>();
        String rootNode = emitNode(g, spec, rootPath, root, null, counter, pathToNode,
                /*isRoot=*/true);

        // Wire input → first sequential-ish child or root
        String first = firstLeaf(spec, rootPath, pathToNode);
        if (first != null) {
            g.adjList().get("input_0").addEdge(new GraphEdge(first, "(structure)"));
        } else if (rootNode != null) {
            g.adjList().get("input_0").addEdge(new GraphEdge(rootNode, "(structure)"));
        }

        // Sequential chain edges among siblings
        wireSequentialChains(g, spec, rootPath, pathToNode);

        String last = lastLeaf(spec, rootPath, pathToNode);
        g.adjList().put("output", GraphNode.of(NodeType.OUTPUT));
        g.graphNodeDisplayNames().put("output", "output");
        g.graphNodeNameToWithoutSuffix().put("output", "output");
        if (last != null) {
            GraphNode n = g.adjList().get(last);
            if (n != null) n.addEdge(new GraphEdge("output", "(structure)"));
        }

        return g;
    }

    /**
     * AOTI package: constants as Parameter nodes + a single Operation "aoti_run"
     * between input and output, annotated with call_spec.
     */
    public static TraceGraph fromAoti(AOTIModelPackageLoader loader, String sourcePath) {
        TraceGraph g = new TraceGraph();
        g.adjList().put("input_0", GraphNode.of(NodeType.INPUT));
        g.graphNodeDisplayNames().put("input_0", "input");
        g.graphNodeNameToWithoutSuffix().put("input_0", "input");
        g.nodeToAncestors().put("input_0", new ArrayList<>());

        List<String> consts = VistaModelFiles.aotiConstantFqns(loader);
        List<String> callSpec = VistaModelFiles.aotiCallSpec(loader);
        int i = 0;
        for (String c : consts) {
            i++;
            String name = "param_" + i;
            g.adjList().put(name, GraphNode.of(NodeType.PARAMETER));
            g.graphNodeDisplayNames().put(name, shortName(c));
            g.graphNodeNameToWithoutSuffix().put(name, "Parameter");
            g.nodeToModulePath().put(name, c);
            Map<String, Object> info = new LinkedHashMap<>();
            info.put("positional_args", List.of(c));
            info.put("keyword_args", Map.of());
            g.funcInfo().put(name, info);
            g.adjList().get(name).addEdge(new GraphEdge("aoti_run_1", "(const)"));
        }

        GraphNode run = GraphNode.of(NodeType.OPERATION);
        g.adjList().put("aoti_run_1", run);
        g.graphNodeDisplayNames().put("aoti_run_1", "aoti_run");
        g.graphNodeNameToWithoutSuffix().put("aoti_run_1", "aoti_run");
        g.nodeToModulePath().put("aoti_run_1", "torch.inductor.aoti");
        Map<String, Object> fi = new LinkedHashMap<>();
        fi.put("positional_args", callSpec);
        fi.put("keyword_args", Map.of(
                "source", sourcePath == null ? "" : sourcePath,
                "n_constants", consts.size()));
        g.funcInfo().put("aoti_run_1", fi);

        g.adjList().get("input_0").addEdge(new GraphEdge("aoti_run_1", "(?)"));

        g.adjList().put("output", GraphNode.of(NodeType.OUTPUT));
        g.graphNodeDisplayNames().put("output", "output");
        g.graphNodeNameToWithoutSuffix().put("output", "output");
        run.addEdge(new GraphEdge("output", "(?)"));

        // Container for the package
        String pkg = "AOTIPackage_0";
        g.graphNodeDisplayNames().put(pkg, "AOTIPackage");
        g.graphNodeNameToWithoutSuffix().put(pkg, "AOTIPackage");
        g.parentModuleToNodes().put(pkg, new ArrayList<>(List.of("aoti_run_1")));
        g.parentModuleToDepth().put(pkg, 1);
        g.nodeToAncestors().put("aoti_run_1", new ArrayList<>(List.of(pkg)));
        for (int p = 1; p <= i; p++) {
            g.nodeToAncestors().put("param_" + p, new ArrayList<>(List.of(pkg)));
            g.parentModuleToNodes().get(pkg).add("param_" + p);
        }

        ModuleInfo mi = new ModuleInfo("AOTIPackage",
                Map.of(),
                Map.of("constants", consts.size(), "call_spec", callSpec.toString()),
                sourcePath);
        g.moduleInfo().put(pkg, mi);
        return g;
    }

    // -------------------------------------------------------------------------

    private static String emitNode(TraceGraph g, StructureSpec spec, String path,
                                   StructureSpec.Node node, String parentNodeName,
                                   int[] counter, Map<String, String> pathToNode,
                                   boolean isRoot) {
        if (node == null) return null;
        counter[0]++;
        String simple = simpleClass(node);
        String nodeName = simple + "_" + counter[0];
        pathToNode.put(path, nodeName);

        boolean container = node.isContainer() || (node.children != null && !node.children.isEmpty());
        g.graphNodeNameToWithoutSuffix().put(nodeName, simple);
        String display = path.isEmpty() ? (spec.root != null ? spec.root : simple)
                : shortName(path);
        g.graphNodeDisplayNames().put(nodeName, display);
        g.nodeToModulePath().put(nodeName, node.className != null ? node.className : node.kind);
        if (!path.isEmpty()) {
            g.nodeToAttrName().put(nodeName, shortName(path));
        }

        // module_info from hyper + own parameters
        Map<String, ModuleInfo.ParamInfo> params = new LinkedHashMap<>();
        if (node.ownParameters != null) {
            for (String p : node.ownParameters) {
                params.put(shortName(p), new ModuleInfo.ParamInfo(new long[0], true));
            }
        }
        Map<String, Object> attrs = new LinkedHashMap<>();
        if (node.hyper != null) attrs.putAll(node.hyper);
        attrs.put("kind", node.kind);
        if (node.className != null) attrs.put("class", node.className);
        g.moduleInfo().put(nodeName, new ModuleInfo(
                node.className != null ? node.className : simple, params, attrs, null));

        List<String> ancestors = new ArrayList<>();
        if (parentNodeName != null) {
            ancestors.add(parentNodeName);
            List<String> pa = g.nodeToAncestors().get(parentNodeName);
            if (pa != null) ancestors.addAll(pa);
            g.parentModuleToNodes()
                    .computeIfAbsent(parentNodeName, k -> new ArrayList<>())
                    .add(nodeName);
            g.parentModuleToDepth().merge(parentNodeName, 1, Math::max);
        }
        g.nodeToAncestors().put(nodeName, ancestors);

        if (!container) {
            // Leaf module in adj_list
            g.adjList().put(nodeName, GraphNode.of(NodeType.MODULE));
        } else {
            // Container: not in adj_list as leaf; children will be. Still keep
            // display metadata. For root with only containers, add a proxy node
            // so something is clickable.
            if (isRoot && (node.children == null || node.children.isEmpty())) {
                g.adjList().put(nodeName, GraphNode.of(NodeType.MODULE));
            }
        }

        if (node.children != null) {
            for (String childKey : node.children) {
                String childPath = path.isEmpty() ? childKey : path + "." + childKey;
                StructureSpec.Node child = spec.nodes.get(childPath);
                if (child == null) {
                    // try relative only
                    child = spec.nodes.get(childKey);
                    if (child != null) childPath = childKey;
                }
                if (child != null) {
                    emitNode(g, spec, childPath, child, nodeName, counter, pathToNode, false);
                }
            }
        }
        return nodeName;
    }

    private static void wireSequentialChains(TraceGraph g, StructureSpec spec,
                                             String path, Map<String, String> pathToNode) {
        StructureSpec.Node node = spec.nodes.get(path);
        if (node == null) return;
        if (node.isSequential() && node.children != null && node.children.size() >= 2) {
            List<String> leafChain = new ArrayList<>();
            for (String childKey : node.children) {
                String childPath = path.isEmpty() ? childKey : path + "." + childKey;
                String leaf = firstLeaf(spec, childPath, pathToNode);
                if (leaf == null) leaf = pathToNode.get(childPath);
                if (leaf != null && g.adjList().containsKey(leaf)) leafChain.add(leaf);
                // recurse into child containers first
                wireSequentialChains(g, spec, childPath, pathToNode);
            }
            for (int i = 0; i < leafChain.size() - 1; i++) {
                String a = leafChain.get(i);
                String b = leafChain.get(i + 1);
                // also connect last leaf of child i to first leaf of child i+1 more carefully
                String lastA = lastLeaf(spec,
                        path.isEmpty() ? node.children.get(i) : path + "." + node.children.get(i),
                        pathToNode);
                String firstB = firstLeaf(spec,
                        path.isEmpty() ? node.children.get(i + 1) : path + "." + node.children.get(i + 1),
                        pathToNode);
                if (lastA == null) lastA = a;
                if (firstB == null) firstB = b;
                GraphNode gn = g.adjList().get(lastA);
                if (gn != null && g.adjList().containsKey(firstB)) {
                    boolean exists = false;
                    for (GraphEdge e : gn.edges()) {
                        if (firstB.equals(e.target())) {
                            exists = true;
                            break;
                        }
                    }
                    if (!exists) gn.addEdge(new GraphEdge(firstB, "(structure)"));
                }
            }
            return;
        }
        if (node.children != null) {
            for (String childKey : node.children) {
                String childPath = path.isEmpty() ? childKey : path + "." + childKey;
                wireSequentialChains(g, spec, childPath, pathToNode);
            }
        }
    }

    private static String firstLeaf(StructureSpec spec, String path, Map<String, String> pathToNode) {
        StructureSpec.Node n = spec.nodes.get(path);
        if (n == null) return pathToNode.get(path);
        if (n.children == null || n.children.isEmpty()) return pathToNode.get(path);
        String childKey = n.children.get(0);
        String childPath = path.isEmpty() ? childKey : path + "." + childKey;
        return firstLeaf(spec, childPath, pathToNode);
    }

    private static String lastLeaf(StructureSpec spec, String path, Map<String, String> pathToNode) {
        StructureSpec.Node n = spec.nodes.get(path);
        if (n == null) return pathToNode.get(path);
        if (n.children == null || n.children.isEmpty()) return pathToNode.get(path);
        String childKey = n.children.get(n.children.size() - 1);
        String childPath = path.isEmpty() ? childKey : path + "." + childKey;
        return lastLeaf(spec, childPath, pathToNode);
    }

    private static String simpleClass(StructureSpec.Node node) {
        if (node.className != null && !node.className.isEmpty()) {
            String c = node.className;
            int i = Math.max(c.lastIndexOf('.'), c.lastIndexOf(':'));
            String s = i >= 0 ? c.substring(i + 1) : c;
            if (!s.isEmpty()) return s.replace(":", "");
        }
        if (node.kind != null && !node.kind.isEmpty()) {
            String k = node.kind;
            // LINEAR → Linear, RELU → ReLU
            if (k.equals(k.toUpperCase(Locale.ROOT)) && k.length() > 1) {
                return k.charAt(0) + k.substring(1).toLowerCase(Locale.ROOT);
            }
            return k;
        }
        return "Module";
    }

    private static String shortName(String path) {
        if (path == null || path.isEmpty()) return "root";
        int i = path.lastIndexOf('.');
        return i >= 0 ? path.substring(i + 1) : path;
    }
}
