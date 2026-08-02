package org.bytedeco.pytorch.serving.tritonserver;

import java.util.*;

/**
 * Dictionary of models returned by {@link TServer#models}.
 *
 * <p>Corresponds to Python {@code ModelDictionary} (mapping name → {@link TritonModel}).
 * When multiple versions exist, the entry for a bare name typically points at the
 * version selected by server policy; versioned keys use {@code "name/version"}.
 */
public final class TritonModelDictionary implements Iterable<Map.Entry<String, TritonModel>> {
    private final Map<String, TritonModel> models;

    public TritonModelDictionary(Map<String, TritonModel> models) {
        Objects.requireNonNull(models, "models");
        this.models = Collections.unmodifiableMap(new LinkedHashMap<>(models));
    }

    public static TritonModelDictionary empty() {
        return new TritonModelDictionary(Map.of());
    }

    public TritonModel get(String name) {
        return models.get(name);
    }

    public TritonModel get(String name, long version) {
        TritonModel m = models.get(name + "/" + version);
        if (m != null) {
            return m;
        }
        m = models.get(name);
        if (m != null && m.version() == version) {
            return m;
        }
        return null;
    }

    public boolean contains(String name) {
        return models.containsKey(name);
    }

    public Set<String> names() {
        return models.keySet();
    }

    public Collection<TritonModel> values() {
        return models.values();
    }

    public Map<String, TritonModel> asMap() {
        return models;
    }

    public int size() {
        return models.size();
    }

    public boolean isEmpty() {
        return models.isEmpty();
    }

    @Override
    public Iterator<Map.Entry<String, TritonModel>> iterator() {
        return models.entrySet().iterator();
    }

    @Override
    public String toString() {
        return "ModelDictionary" + models.keySet();
    }
}
