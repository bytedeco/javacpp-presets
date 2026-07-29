/*
 * Incremental materialization cursor (watermark) per feature view.
 */
package org.bytedeco.pytorch.utils.feature.materialize;

import java.util.Objects;
import java.util.concurrent.ConcurrentHashMap;

/** Tracks last successfully materialized event timestamp per view. */
public final class IncrementalCursor {

    private final ConcurrentHashMap<String, Long> watermarks = new ConcurrentHashMap<>();

    private static String key(String project, String viewName) {
        return (project == null || project.isEmpty() ? "default" : project) + "/" + viewName;
    }

    public long get(String project, String viewName) {
        return watermarks.getOrDefault(key(project, viewName), 0L);
    }

    public void advance(String project, String viewName, long eventTsMs) {
        watermarks.merge(key(project, viewName), eventTsMs, Math::max);
    }

    public void set(String project, String viewName, long eventTsMs) {
        watermarks.put(key(project, viewName), eventTsMs);
    }

    public void clear(String project, String viewName) {
        watermarks.remove(key(project, viewName));
    }

    public void clearAll() {
        watermarks.clear();
    }

    public int size() {
        return watermarks.size();
    }

    @Override
    public String toString() {
        return "IncrementalCursor" + watermarks;
    }

    public ConcurrentHashMap<String, Long> snapshot() {
        return new ConcurrentHashMap<>(watermarks);
    }

    public void load(ConcurrentHashMap<String, Long> other) {
        Objects.requireNonNull(other, "other");
        watermarks.clear();
        watermarks.putAll(other);
    }
}
