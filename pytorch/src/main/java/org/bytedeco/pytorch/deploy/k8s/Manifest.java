package org.bytedeco.pytorch.deploy.k8s;

import org.bytedeco.pytorch.utils.yaml.Yaml;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Multi-document Kubernetes manifest helper.
 */
public final class Manifest {

    private final List<Object> documents;

    public Manifest() {
        this.documents = new ArrayList<>();
    }

    public Manifest(List<Object> documents) {
        this.documents = new ArrayList<>();
        if (documents != null) this.documents.addAll(documents);
    }

    public static Manifest load(Path path) throws IOException {
        return new Manifest(Yaml.loadAll(path));
    }

    public static Manifest load(String yamlText) throws IOException {
        return new Manifest(Yaml.loadAll(yamlText));
    }

    public static Manifest of(Object... docs) {
        Manifest m = new Manifest();
        if (docs != null) {
            for (Object d : docs) if (d != null) m.documents.add(d);
        }
        return m;
    }

    public Manifest add(Object doc) {
        if (doc != null) documents.add(doc);
        return this;
    }

    public Manifest addAll(List<?> docs) {
        if (docs != null) {
            for (Object d : docs) if (d != null) documents.add(d);
        }
        return this;
    }

    public List<Object> documents() {
        return Collections.unmodifiableList(documents);
    }

    public int size() {
        return documents.size();
    }

    @SuppressWarnings("unchecked")
    public Map<String, Object> get(int index) {
        Object d = documents.get(index);
        if (d instanceof Map<?, ?> m) return (Map<String, Object>) m;
        throw new IllegalStateException("document " + index + " is not a mapping");
    }

    /** Index by {@code kind/namespace/name} (namespace may be empty). */
    public Map<String, Map<String, Object>> index() {
        Map<String, Map<String, Object>> out = new LinkedHashMap<>();
        for (Object doc : documents) {
            if (!(doc instanceof Map<?, ?>)) continue;
            @SuppressWarnings("unchecked")
            Map<String, Object> m = (Map<String, Object>) doc;
            String key = keyOf(m);
            if (key != null) out.put(key, m);
        }
        return out;
    }

    public Map<String, Object> find(String kind, String name) {
        return find(kind, null, name);
    }

    public Map<String, Object> find(String kind, String namespace, String name) {
        for (Object doc : documents) {
            if (!(doc instanceof Map<?, ?>)) continue;
            @SuppressWarnings("unchecked")
            Map<String, Object> m = (Map<String, Object>) doc;
            if (kind != null && !kind.equalsIgnoreCase(str(m.get("kind")))) continue;
            Map<String, Object> meta = meta(m);
            if (name != null && !name.equals(str(meta.get("name")))) continue;
            if (namespace != null) {
                String ns = str(meta.get("namespace"));
                if (ns == null) ns = "";
                if (!namespace.equals(ns)) continue;
            }
            return m;
        }
        return null;
    }

    public String toYaml() {
        return Yaml.dumpAll(documents);
    }

    public void save(Path path) throws IOException {
        Yaml.dumpAll(path, documents);
    }

    public static String keyOf(Map<String, Object> doc) {
        if (doc == null) return null;
        String kind = str(doc.get("kind"));
        Map<String, Object> meta = meta(doc);
        String name = str(meta.get("name"));
        String ns = str(meta.get("namespace"));
        if (kind == null || name == null) return null;
        return kind + "/" + (ns == null ? "" : ns) + "/" + name;
    }

    @SuppressWarnings("unchecked")
    public static Map<String, Object> meta(Map<String, Object> doc) {
        Object m = doc.get("metadata");
        if (m instanceof Map<?, ?> map) return (Map<String, Object>) map;
        return Map.of();
    }

    private static String str(Object o) {
        return o == null ? null : String.valueOf(o);
    }

    @Override
    public String toString() {
        return "Manifest{docs=" + documents.size() + ", keys=" + index().keySet() + "}";
    }
}
