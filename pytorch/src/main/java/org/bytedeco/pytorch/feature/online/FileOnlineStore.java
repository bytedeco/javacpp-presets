/*
 * File-backed online store — crash-recoverable snapshot of the in-memory KV.
 * Format: JSONL under {root}/online/{project}/{view}.jsonl
 */
package org.bytedeco.pytorch.feature.online;

import org.bytedeco.pytorch.feature.offline.FileOfflineStore;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.Collection;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;

/** Durable online store with InMemory mirror. */
public final class FileOnlineStore implements OnlineStore {

    private final Path root;
    private final InMemoryOnlineStore memory = new InMemoryOnlineStore();

    public FileOnlineStore(Path root) {
        this.root = root;
        try {
            Files.createDirectories(root);
            loadAll();
        } catch (IOException e) {
            throw new IllegalStateException("FileOnlineStore init failed: " + root, e);
        }
    }

    public Path root() {
        return root;
    }

    private Path viewFile(String project, String viewName) {
        String p = project == null || project.isEmpty() ? "default" : project;
        return root.resolve(p).resolve(viewName + ".jsonl");
    }

    private void loadAll() throws IOException {
        if (!Files.isDirectory(root)) return;
        try (var projects = Files.list(root)) {
            for (Path proj : (Iterable<Path>) projects.filter(Files::isDirectory)::iterator) {
                String project = proj.getFileName().toString();
                try (var files = Files.list(proj)) {
                    for (Path f : (Iterable<Path>) files.filter(p -> p.toString().endsWith(".jsonl"))::iterator) {
                        String viewName = f.getFileName().toString().replace(".jsonl", "");
                        List<OnlineFeatureRow> rows = new ArrayList<>();
                        try (BufferedReader br = Files.newBufferedReader(f, StandardCharsets.UTF_8)) {
                            String line;
                            while ((line = br.readLine()) != null) {
                                if (line.isBlank()) continue;
                                OnlineFeatureRow row = parseRow(project, viewName, line);
                                if (row != null) rows.add(row);
                            }
                        }
                        memory.onlineWrite(OnlineWriteBatch.of(rows));
                    }
                }
            }
        }
    }

    private OnlineFeatureRow parseRow(String project, String viewName, String line) {
        Map<String, Object> m = FileOfflineStore.parseJsonLine(line);
        Object ek = m.get("_entity_key");
        if (ek == null) return null;
        long eventTs = FileOfflineStore.toEpochMillis(m.get("_event_ts"));
        long written = FileOfflineStore.toEpochMillis(m.get("_written_at"));
        long ttl = m.get("_ttl") instanceof Number ? ((Number) m.get("_ttl")).longValue() : 0L;
        Map<String, Object> values = new LinkedHashMap<>();
        for (Map.Entry<String, Object> e : m.entrySet()) {
            if (e.getKey().startsWith("_")) continue;
            values.put(e.getKey(), e.getValue());
        }
        return OnlineFeatureRow.builder(viewName, String.valueOf(ek))
                .project(project)
                .values(values)
                .eventTimestampMs(eventTs)
                .writtenAtMs(written)
                .ttlMs(ttl)
                .build();
    }

    private void persistView(String project, String viewName) {
        // rewrite entire view file from memory scan
        Path f = viewFile(project, viewName);
        try {
            Files.createDirectories(f.getParent());
            Path tmp = f.resolveSibling(f.getFileName().toString() + ".tmp");
            // Collect keys for this view by probing — we don't expose iterator; rewrite via batch read is hard.
            // Strategy: keep side index of entity keys per view.
            List<String> keys = entityKeysIndex.computeIfAbsent(sk(project, viewName), x -> new ArrayList<>());
            try (BufferedWriter w = Files.newBufferedWriter(tmp, StandardCharsets.UTF_8)) {
                Map<String, OnlineFeatureRow> batch = memory.onlineReadBatch(project, viewName, new ArrayList<>(keys));
                for (OnlineFeatureRow row : batch.values()) {
                    w.write(toLine(row));
                    w.newLine();
                }
            }
            Files.move(tmp, f, StandardCopyOption.REPLACE_EXISTING, StandardCopyOption.ATOMIC_MOVE);
        } catch (IOException e) {
            throw new IllegalStateException("online persist failed", e);
        }
    }

    private final Map<String, List<String>> entityKeysIndex = new ConcurrentHashMapCompat();

    /** Tiny ConcurrentHashMap subclass avoided — use ConcurrentHashMap directly via field init below. */
    private static final class ConcurrentHashMapCompat extends java.util.concurrent.ConcurrentHashMap<String, List<String>> {}

    private static String sk(String project, String viewName) {
        return (project == null || project.isEmpty() ? "default" : project) + "/" + viewName;
    }

    private static String toLine(OnlineFeatureRow row) {
        Map<String, Object> m = new LinkedHashMap<>();
        m.put("_entity_key", row.entityKey());
        m.put("_event_ts", row.eventTimestampMs());
        m.put("_written_at", row.writtenAtMs());
        m.put("_ttl", row.ttlMs());
        m.putAll(row.values());
        return FileOfflineStore.toJsonLine(m);
    }

    @Override
    public void onlineWrite(OnlineWriteBatch batch) {
        if (batch == null) return;
        memory.onlineWrite(batch);
        // group by view for persistence
        Map<String, List<OnlineFeatureRow>> byView = new LinkedHashMap<>();
        for (OnlineFeatureRow row : batch.rows()) {
            String vk = sk(row.project(), row.viewName());
            byView.computeIfAbsent(vk, x -> new ArrayList<>()).add(row);
            entityKeysIndex.computeIfAbsent(vk, x -> new ArrayList<>());
            List<String> keys = entityKeysIndex.get(vk);
            synchronized (keys) {
                if (!keys.contains(row.entityKey())) keys.add(row.entityKey());
            }
        }
        for (List<OnlineFeatureRow> group : byView.values()) {
            if (group.isEmpty()) continue;
            OnlineFeatureRow sample = group.get(0);
            persistView(sample.project(), sample.viewName());
        }
    }

    @Override
    public Optional<OnlineFeatureRow> onlineRead(String project, String viewName, String entityKey) {
        return memory.onlineRead(project, viewName, entityKey);
    }

    @Override
    public Map<String, OnlineFeatureRow> onlineReadBatch(String project, String viewName,
                                                         Collection<String> entityKeys) {
        return memory.onlineReadBatch(project, viewName, entityKeys);
    }

    @Override
    public long size(String project, String viewName) {
        return memory.size(project, viewName);
    }

    @Override
    public void delete(String project, String viewName, String entityKey) {
        memory.delete(project, viewName, entityKey);
        List<String> keys = entityKeysIndex.get(sk(project, viewName));
        if (keys != null) {
            synchronized (keys) {
                keys.remove(entityKey);
            }
        }
        persistView(project, viewName);
    }

    @Override
    public void clearView(String project, String viewName) {
        memory.clearView(project, viewName);
        entityKeysIndex.remove(sk(project, viewName));
        try {
            Path f = viewFile(project, viewName);
            Files.deleteIfExists(f);
        } catch (IOException ignored) {
        }
    }

    @Override
    public void close() {
        memory.close();
    }
}
