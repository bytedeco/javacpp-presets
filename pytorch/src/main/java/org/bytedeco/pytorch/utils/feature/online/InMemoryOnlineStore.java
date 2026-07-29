/*
 * Concurrent in-memory online store with optional TTL eviction on read.
 */
package org.bytedeco.pytorch.utils.feature.online;

import java.util.Collection;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.ConcurrentHashMap;

/** Default online store for single-process serving. */
public final class InMemoryOnlineStore implements OnlineStore {

    private final ConcurrentHashMap<String, OnlineFeatureRow> data = new ConcurrentHashMap<>();
    private final boolean evictExpiredOnRead;

    public InMemoryOnlineStore() {
        this(true);
    }

    public InMemoryOnlineStore(boolean evictExpiredOnRead) {
        this.evictExpiredOnRead = evictExpiredOnRead;
    }

    private static String sk(String project, String viewName, String entityKey) {
        return (project == null || project.isEmpty() ? "default" : project)
                + "#" + viewName + "#" + entityKey;
    }

    @Override
    public void onlineWrite(OnlineWriteBatch batch) {
        if (batch == null) return;
        for (OnlineFeatureRow row : batch.rows()) {
            data.put(row.storageKey(), row);
        }
    }

    @Override
    public Optional<OnlineFeatureRow> onlineRead(String project, String viewName, String entityKey) {
        String k = sk(project, viewName, entityKey);
        OnlineFeatureRow row = data.get(k);
        if (row == null) return Optional.empty();
        if (evictExpiredOnRead && row.isExpired(System.currentTimeMillis())) {
            data.remove(k, row);
            return Optional.empty();
        }
        return Optional.of(row);
    }

    @Override
    public Map<String, OnlineFeatureRow> onlineReadBatch(String project, String viewName,
                                                         Collection<String> entityKeys) {
        Map<String, OnlineFeatureRow> out = new LinkedHashMap<>();
        if (entityKeys == null) return out;
        long now = System.currentTimeMillis();
        for (String ek : entityKeys) {
            String k = sk(project, viewName, ek);
            OnlineFeatureRow row = data.get(k);
            if (row == null) continue;
            if (evictExpiredOnRead && row.isExpired(now)) {
                data.remove(k, row);
                continue;
            }
            out.put(ek, row);
        }
        return out;
    }

    @Override
    public long size(String project, String viewName) {
        String prefix = (project == null || project.isEmpty() ? "default" : project) + "#" + viewName + "#";
        long n = 0;
        for (String k : data.keySet()) {
            if (k.startsWith(prefix)) n++;
        }
        return n;
    }

    @Override
    public void delete(String project, String viewName, String entityKey) {
        data.remove(sk(project, viewName, entityKey));
    }

    @Override
    public void clearView(String project, String viewName) {
        String prefix = (project == null || project.isEmpty() ? "default" : project) + "#" + viewName + "#";
        data.keySet().removeIf(k -> k.startsWith(prefix));
    }

    public void clear() {
        data.clear();
    }

    public long totalSize() {
        return data.size();
    }
}
