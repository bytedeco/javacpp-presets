/*
 * Feature lineage graph — view ← source ← transform edges.
 * Used for impact analysis (Databricks / Feast / internal Meta feature graph).
 */
package org.bytedeco.pytorch.feature.registry;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;

/** Thread-safe directed lineage graph. */
public final class LineageGraph {

    public static final class Edge {
        public final String from;
        public final String to;
        public final String relation;

        public Edge(String from, String to, String relation) {
            this.from = Objects.requireNonNull(from, "from");
            this.to = Objects.requireNonNull(to, "to");
            this.relation = relation != null ? relation : "depends_on";
        }

        @Override
        public String toString() {
            return from + " -[" + relation + "]-> " + to;
        }

        @Override
        public boolean equals(Object o) {
            if (this == o) return true;
            if (!(o instanceof Edge)) return false;
            Edge edge = (Edge) o;
            return from.equals(edge.from) && to.equals(edge.to) && relation.equals(edge.relation);
        }

        @Override
        public int hashCode() {
            return Objects.hash(from, to, relation);
        }
    }

    private final ConcurrentHashMap<String, Set<Edge>> outgoing = new ConcurrentHashMap<>();
    private final ConcurrentHashMap<String, Set<Edge>> incoming = new ConcurrentHashMap<>();

    public void addEdge(String from, String to, String relation) {
        Edge e = new Edge(from, to, relation);
        outgoing.computeIfAbsent(from, k -> ConcurrentHashMap.newKeySet()).add(e);
        incoming.computeIfAbsent(to, k -> ConcurrentHashMap.newKeySet()).add(e);
    }

    public List<Edge> outgoing(String node) {
        Set<Edge> s = outgoing.get(node);
        return s == null ? List.of() : new ArrayList<>(s);
    }

    public List<Edge> incoming(String node) {
        Set<Edge> s = incoming.get(node);
        return s == null ? List.of() : new ArrayList<>(s);
    }

    /** Upstream dependencies (BFS). */
    public Set<String> upstream(String node, int maxDepth) {
        return walk(node, true, maxDepth);
    }

    /** Downstream dependents (BFS). */
    public Set<String> downstream(String node, int maxDepth) {
        return walk(node, false, maxDepth);
    }

    private Set<String> walk(String start, boolean up, int maxDepth) {
        Set<String> visited = new LinkedHashSet<>();
        List<String> frontier = new ArrayList<>();
        frontier.add(start);
        int depth = 0;
        while (!frontier.isEmpty() && depth < maxDepth) {
            List<String> next = new ArrayList<>();
            for (String n : frontier) {
                List<Edge> edges = up ? outgoing(n) : incoming(n);
                for (Edge e : edges) {
                    String other = up ? e.to : e.from;
                    if (visited.add(other)) {
                        next.add(other);
                    }
                }
            }
            frontier = next;
            depth++;
        }
        visited.remove(start);
        return visited;
    }

    public Map<String, List<Edge>> snapshot() {
        Map<String, List<Edge>> out = new LinkedHashMap<>();
        for (Map.Entry<String, Set<Edge>> e : outgoing.entrySet()) {
            out.put(e.getKey(), new ArrayList<>(e.getValue()));
        }
        return Collections.unmodifiableMap(out);
    }

    public int edgeCount() {
        int n = 0;
        for (Set<Edge> s : outgoing.values()) n += s.size();
        return n;
    }

    public void clear() {
        outgoing.clear();
        incoming.clear();
    }
}
