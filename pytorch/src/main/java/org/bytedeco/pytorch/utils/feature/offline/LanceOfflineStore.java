/*
 * Lance-oriented offline store for multimodal / embedding feature rows.
 * Delegates tabular storage to FileOfflineStore; records lance URI hints per view.
 *
 * Production would use utils.lance / dataframe.lance for vector columns;
 * this adapter keeps embedding float[] in row maps and optional path metadata.
 */
package org.bytedeco.pytorch.utils.feature.offline;

import java.nio.file.Path;
import java.time.Instant;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.ConcurrentHashMap;

/** Offline store optimized for embedding/multimodal feature payloads. */
public final class LanceOfflineStore implements OfflineStore {

    private final FileOfflineStore mirror;
    private final ConcurrentHashMap<String, String> lanceUris = new ConcurrentHashMap<>();

    public LanceOfflineStore() {
        this.mirror = FileOfflineStore.inMemory();
    }

    public LanceOfflineStore(Path fileRoot) {
        this.mirror = new FileOfflineStore(fileRoot);
    }

    public FileOfflineStore mirror() {
        return mirror;
    }

    public void bindLanceUri(String project, String viewName, String uri) {
        lanceUris.put(key(project, viewName), uri);
    }

    public Optional<String> lanceUri(String project, String viewName) {
        return Optional.ofNullable(lanceUris.get(key(project, viewName)));
    }

    private static String key(String project, String viewName) {
        return (project == null || project.isEmpty() ? "default" : project) + "/" + viewName;
    }

    @Override
    public void put(String project, String viewName, List<Map<String, Object>> rows) {
        mirror.put(project, viewName, rows);
    }

    @Override
    public void replace(String project, String viewName, List<Map<String, Object>> rows) {
        mirror.replace(project, viewName, rows);
    }

    @Override
    public List<Map<String, Object>> readAll(String project, String viewName) {
        return mirror.readAll(project, viewName);
    }

    @Override
    public List<Map<String, Object>> readRange(String project, String viewName,
                                               Instant start, Instant end,
                                               String timestampColumn) {
        return mirror.readRange(project, viewName, start, end, timestampColumn);
    }

    @Override
    public Optional<Long> latestTimestamp(String project, String viewName, String timestampColumn) {
        return mirror.latestTimestamp(project, viewName, timestampColumn);
    }

    @Override
    public long rowCount(String project, String viewName) {
        return mirror.rowCount(project, viewName);
    }

    @Override
    public void close() {
        mirror.close();
    }
}
