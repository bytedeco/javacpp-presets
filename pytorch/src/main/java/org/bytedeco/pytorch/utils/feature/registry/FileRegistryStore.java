/*
 * File-backed RegistryStore — JSON lines under a root directory.
 * Zero extra deps: hand-rolled minimal JSON for metadata round-trip.
 *
 * Layout:
 *   {root}/projects/{name}.json
 *   {root}/{project}/entities/{name}.json
 *   {root}/{project}/views/{name}.json
 *   {root}/{project}/services/{name}.json
 *   {root}/{project}/versions/{type}_{name}_{versionId}.json
 *   {root}/{project}/prod/{type}_{name}.ptr
 *
 * Full payload objects (FeatureView etc.) stay in the companion in-memory cache;
 * file store persists identity + stage + schema summary for restart demos.
 */
package org.bytedeco.pytorch.utils.feature.registry;

import org.bytedeco.pytorch.utils.feature.core.Entity;
import org.bytedeco.pytorch.utils.feature.core.FeatureService;
import org.bytedeco.pytorch.utils.feature.core.FeatureView;
import org.bytedeco.pytorch.utils.feature.core.Field;
import org.bytedeco.pytorch.utils.feature.core.OnDemandFeatureView;
import org.bytedeco.pytorch.utils.feature.core.Project;
import org.bytedeco.pytorch.utils.feature.core.StreamFeatureView;
import org.bytedeco.pytorch.utils.feature.core.ValueType;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.time.Duration;
import java.time.Instant;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.stream.Collectors;
import java.util.stream.Stream;

/** Durable file registry with in-memory mirror for full object graphs. */
public final class FileRegistryStore implements RegistryStore {

    private final Path root;
    private final InMemoryRegistryStore memory = new InMemoryRegistryStore();

    public FileRegistryStore(Path root) {
        this.root = root;
        try {
            Files.createDirectories(root);
            loadAll();
        } catch (IOException e) {
            throw new IllegalStateException("cannot init FileRegistryStore at " + root, e);
        }
    }

    public Path root() {
        return root;
    }

    private void loadAll() throws IOException {
        Path projectsDir = root.resolve("projects");
        if (Files.isDirectory(projectsDir)) {
            try (Stream<Path> s = Files.list(projectsDir)) {
                s.filter(p -> p.toString().endsWith(".json")).forEach(p -> {
                    try {
                        Map<String, String> m = readSimpleJson(p);
                        memory.saveProject(Project.builder(m.getOrDefault("name", p.getFileName().toString().replace(".json", "")))
                                .description(m.getOrDefault("description", ""))
                                .owner(m.getOrDefault("owner", ""))
                                .build());
                    } catch (IOException ignored) {
                    }
                });
            }
        }
        // Full object reload is best-effort via companion sidecars written on save.
        // Entities / views are re-hydrated from compact schema JSON.
        if (!Files.isDirectory(root)) return;
        try (Stream<Path> projects = Files.list(root)) {
            for (Path projDir : projects.filter(Files::isDirectory).filter(p -> !p.getFileName().toString().equals("projects")).collect(Collectors.toList())) {
                String project = projDir.getFileName().toString();
                loadEntities(project, projDir.resolve("entities"));
                loadViews(project, projDir.resolve("views"));
                loadServices(project, projDir.resolve("services"));
                loadOnDemand(project, projDir.resolve("ondemand"));
                loadStreams(project, projDir.resolve("streams"));
                loadVersions(project, projDir.resolve("versions"));
                loadProd(project, projDir.resolve("prod"));
            }
        }
    }

    private void loadEntities(String project, Path dir) throws IOException {
        if (!Files.isDirectory(dir)) return;
        try (Stream<Path> s = Files.list(dir)) {
            s.filter(p -> p.toString().endsWith(".json")).forEach(p -> {
                try {
                    Map<String, String> m = readSimpleJson(p);
                    Entity e = Entity.builder(m.getOrDefault("name", "unknown"))
                            .project(project)
                            .valueType(ValueType.parse(m.getOrDefault("valueType", "INT64")))
                            .joinKey(m.getOrDefault("joinKey", m.getOrDefault("name", "unknown")))
                            .description(m.getOrDefault("description", ""))
                            .build();
                    memory.saveEntity(e);
                } catch (IOException ignored) {
                }
            });
        }
    }

    private void loadViews(String project, Path dir) throws IOException {
        if (!Files.isDirectory(dir)) return;
        try (Stream<Path> s = Files.list(dir)) {
            s.filter(p -> p.toString().endsWith(".json")).forEach(p -> {
                try {
                    String raw = Files.readString(p, StandardCharsets.UTF_8);
                    FeatureView view = parseFeatureView(project, raw);
                    if (view != null) memory.saveFeatureView(view);
                } catch (IOException ignored) {
                }
            });
        }
    }

    private FeatureView parseFeatureView(String project, String raw) {
        Map<String, String> m = parseFlatJson(raw);
        String name = m.getOrDefault("name", "");
        if (name.isEmpty()) return null;
        FeatureView.Builder b = FeatureView.builder(name).project(project)
                .description(m.getOrDefault("description", ""))
                .owner(m.getOrDefault("owner", ""))
                .online(!"false".equalsIgnoreCase(m.getOrDefault("online", "true")));
        String ttlMs = m.get("ttlMs");
        if (ttlMs != null && !ttlMs.isEmpty()) {
            try {
                b.ttl(Duration.ofMillis(Long.parseLong(ttlMs)));
            } catch (NumberFormatException ignored) {
            }
        }
        String fields = m.getOrDefault("fields", "");
        if (!fields.isEmpty()) {
            for (String part : fields.split(";")) {
                String[] kv = part.split(":", 2);
                if (kv.length == 2) {
                    b.field(Field.of(kv[0].trim(), ValueType.parse(kv[1].trim())));
                }
            }
        }
        String entities = m.getOrDefault("entities", "");
        if (!entities.isEmpty()) {
            for (String en : entities.split(",")) {
                String ename = en.trim();
                if (!ename.isEmpty()) {
                    b.entity(memory.getEntity(project, ename).orElse(Entity.of(ename)));
                }
            }
        }
        return b.build();
    }

    private void loadServices(String project, Path dir) throws IOException {
        if (!Files.isDirectory(dir)) return;
        try (Stream<Path> s = Files.list(dir)) {
            s.filter(p -> p.toString().endsWith(".json")).forEach(p -> {
                try {
                    Map<String, String> m = readSimpleJson(p);
                    FeatureService.Builder b = FeatureService.builder(m.getOrDefault("name", "svc"))
                            .project(project)
                            .description(m.getOrDefault("description", ""));
                    String views = m.getOrDefault("views", "");
                    if (!views.isEmpty()) {
                        for (String v : views.split(",")) {
                            if (!v.trim().isEmpty()) b.view(v.trim());
                        }
                    }
                    String od = m.getOrDefault("onDemand", "");
                    if (!od.isEmpty()) {
                        for (String v : od.split(",")) {
                            if (!v.trim().isEmpty()) b.onDemandView(v.trim());
                        }
                    }
                    memory.saveFeatureService(b.build());
                } catch (IOException ignored) {
                }
            });
        }
    }

    private void loadOnDemand(String project, Path dir) throws IOException {
        if (!Files.isDirectory(dir)) return;
        try (Stream<Path> s = Files.list(dir)) {
            s.filter(p -> p.toString().endsWith(".json")).forEach(p -> {
                try {
                    Map<String, String> m = readSimpleJson(p);
                    OnDemandFeatureView.Builder b = OnDemandFeatureView.builder(m.getOrDefault("name", "od"))
                            .project(project)
                            .description(m.getOrDefault("description", ""));
                    String fields = m.getOrDefault("fields", "");
                    if (!fields.isEmpty()) {
                        for (String part : fields.split(";")) {
                            String[] kv = part.split(":", 2);
                            if (kv.length == 2) b.schema(Field.of(kv[0].trim(), ValueType.parse(kv[1].trim())));
                        }
                    }
                    memory.saveOnDemandFeatureView(b.build());
                } catch (IOException ignored) {
                }
            });
        }
    }

    private void loadStreams(String project, Path dir) throws IOException {
        if (!Files.isDirectory(dir)) return;
        try (Stream<Path> s = Files.list(dir)) {
            s.filter(p -> p.toString().endsWith(".json")).forEach(p -> {
                try {
                    Map<String, String> m = readSimpleJson(p);
                    StreamFeatureView.Builder b = StreamFeatureView.builder(m.getOrDefault("name", "sfv"))
                            .project(project)
                            .description(m.getOrDefault("description", ""));
                    String fields = m.getOrDefault("fields", "");
                    if (!fields.isEmpty()) {
                        for (String part : fields.split(";")) {
                            String[] kv = part.split(":", 2);
                            if (kv.length == 2) b.schema(Field.of(kv[0].trim(), ValueType.parse(kv[1].trim())));
                        }
                    }
                    memory.saveStreamFeatureView(b.build());
                } catch (IOException ignored) {
                }
            });
        }
    }

    private void loadVersions(String project, Path dir) throws IOException {
        if (!Files.isDirectory(dir)) return;
        try (Stream<Path> s = Files.list(dir)) {
            s.filter(p -> p.toString().endsWith(".json")).forEach(p -> {
                try {
                    Map<String, String> m = readSimpleJson(p);
                    FeatureVersion.ResourceType type = FeatureVersion.ResourceType.valueOf(
                            m.getOrDefault("resourceType", "FEATURE_VIEW"));
                    FeatureVersion v = FeatureVersion.builder(m.getOrDefault("resourceName", "unknown"), type)
                            .versionId(m.getOrDefault("versionId", "v1"))
                            .project(project)
                            .stage(LifecycleStage.parse(m.getOrDefault("stage", "DRAFT")))
                            .schemaHash(m.getOrDefault("schemaHash", ""))
                            .description(m.getOrDefault("description", ""))
                            .createdAt(parseInstant(m.get("createdAt")))
                            .createdBy(m.getOrDefault("createdBy", ""))
                            .build();
                    memory.saveVersion(v);
                } catch (Exception ignored) {
                }
            });
        }
    }

    private void loadProd(String project, Path dir) throws IOException {
        if (!Files.isDirectory(dir)) return;
        try (Stream<Path> s = Files.list(dir)) {
            s.filter(p -> p.toString().endsWith(".ptr")).forEach(p -> {
                try {
                    String name = p.getFileName().toString().replace(".ptr", "");
                    int us = name.indexOf('_');
                    if (us <= 0) return;
                    FeatureVersion.ResourceType type = FeatureVersion.ResourceType.valueOf(name.substring(0, us));
                    String resource = name.substring(us + 1);
                    String versionId = Files.readString(p, StandardCharsets.UTF_8).trim();
                    memory.setProductionPointer(project, type, resource, versionId);
                } catch (Exception ignored) {
                }
            });
        }
    }

    private static Instant parseInstant(String s) {
        if (s == null || s.isEmpty()) return Instant.now();
        try {
            return Instant.parse(s);
        } catch (Exception e) {
            return Instant.now();
        }
    }

    private void atomicWrite(Path path, String content) throws IOException {
        Files.createDirectories(path.getParent());
        Path tmp = path.resolveSibling(path.getFileName().toString() + ".tmp");
        Files.writeString(tmp, content, StandardCharsets.UTF_8);
        Files.move(tmp, path, StandardCopyOption.REPLACE_EXISTING, StandardCopyOption.ATOMIC_MOVE);
    }

    private static String esc(String s) {
        if (s == null) return "";
        return s.replace("\\", "\\\\").replace("\"", "\\\"").replace("\n", "\\n");
    }

    private static String quote(String s) {
        return "\"" + esc(s) + "\"";
    }

    private static Map<String, String> readSimpleJson(Path path) throws IOException {
        return parseFlatJson(Files.readString(path, StandardCharsets.UTF_8));
    }

    /** Minimal flat string-valued JSON object parser (no nested objects). */
    static Map<String, String> parseFlatJson(String raw) {
        Map<String, String> out = new LinkedHashMap<>();
        if (raw == null) return out;
        String s = raw.trim();
        if (s.startsWith("{")) s = s.substring(1);
        if (s.endsWith("}")) s = s.substring(0, s.length() - 1);
        // split on "," not inside quotes
        StringBuilder cur = new StringBuilder();
        boolean inQ = false;
        List<String> parts = new ArrayList<>();
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            if (c == '"' && (i == 0 || s.charAt(i - 1) != '\\')) inQ = !inQ;
            if (c == ',' && !inQ) {
                parts.add(cur.toString());
                cur.setLength(0);
            } else {
                cur.append(c);
            }
        }
        if (cur.length() > 0) parts.add(cur.toString());
        for (String part : parts) {
            int colon = indexOfColonOutsideQuotes(part);
            if (colon < 0) continue;
            String k = unquote(part.substring(0, colon).trim());
            String v = unquote(part.substring(colon + 1).trim());
            out.put(k, v);
        }
        return out;
    }

    private static int indexOfColonOutsideQuotes(String s) {
        boolean inQ = false;
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            if (c == '"' && (i == 0 || s.charAt(i - 1) != '\\')) inQ = !inQ;
            if (c == ':' && !inQ) return i;
        }
        return -1;
    }

    private static String unquote(String s) {
        s = s.trim();
        if (s.startsWith("\"") && s.endsWith("\"") && s.length() >= 2) {
            s = s.substring(1, s.length() - 1);
        }
        return s.replace("\\\"", "\"").replace("\\n", "\n").replace("\\\\", "\\");
    }

    private static String fieldsSummary(List<Field> fields) {
        return fields.stream().map(f -> f.name() + ":" + f.valueType().name()).collect(Collectors.joining(";"));
    }

    @Override
    public void saveProject(Project project) {
        memory.saveProject(project);
        try {
            String json = "{"
                    + quote("name") + ":" + quote(project.name()) + ","
                    + quote("description") + ":" + quote(project.description()) + ","
                    + quote("owner") + ":" + quote(project.owner())
                    + "}";
            atomicWrite(root.resolve("projects").resolve(project.name() + ".json"), json);
        } catch (IOException e) {
            throw new IllegalStateException("saveProject failed", e);
        }
    }

    @Override
    public Optional<Project> getProject(String name) {
        return memory.getProject(name);
    }

    @Override
    public List<Project> listProjects() {
        return memory.listProjects();
    }

    @Override
    public void saveEntity(Entity entity) {
        memory.saveEntity(entity);
        try {
            String json = "{"
                    + quote("name") + ":" + quote(entity.name()) + ","
                    + quote("valueType") + ":" + quote(entity.valueType().name()) + ","
                    + quote("joinKey") + ":" + quote(entity.joinKey()) + ","
                    + quote("description") + ":" + quote(entity.description())
                    + "}";
            atomicWrite(root.resolve(entity.project()).resolve("entities").resolve(entity.name() + ".json"), json);
        } catch (IOException e) {
            throw new IllegalStateException("saveEntity failed", e);
        }
    }

    @Override
    public Optional<Entity> getEntity(String project, String name) {
        return memory.getEntity(project, name);
    }

    @Override
    public List<Entity> listEntities(String project) {
        return memory.listEntities(project);
    }

    @Override
    public void saveFeatureView(FeatureView view) {
        memory.saveFeatureView(view);
        try {
            String json = "{"
                    + quote("name") + ":" + quote(view.name()) + ","
                    + quote("description") + ":" + quote(view.description()) + ","
                    + quote("owner") + ":" + quote(view.owner()) + ","
                    + quote("online") + ":" + quote(String.valueOf(view.online())) + ","
                    + quote("ttlMs") + ":" + quote(String.valueOf(view.ttlMillis())) + ","
                    + quote("entities") + ":" + quote(String.join(",", view.entityNames())) + ","
                    + quote("fields") + ":" + quote(fieldsSummary(view.schema()))
                    + "}";
            atomicWrite(root.resolve(view.project()).resolve("views").resolve(view.name() + ".json"), json);
        } catch (IOException e) {
            throw new IllegalStateException("saveFeatureView failed", e);
        }
    }

    @Override
    public Optional<FeatureView> getFeatureView(String project, String name) {
        return memory.getFeatureView(project, name);
    }

    @Override
    public List<FeatureView> listFeatureViews(String project) {
        return memory.listFeatureViews(project);
    }

    @Override
    public void saveOnDemandFeatureView(OnDemandFeatureView view) {
        memory.saveOnDemandFeatureView(view);
        try {
            String json = "{"
                    + quote("name") + ":" + quote(view.name()) + ","
                    + quote("description") + ":" + quote(view.description()) + ","
                    + quote("fields") + ":" + quote(fieldsSummary(view.schema()))
                    + "}";
            atomicWrite(root.resolve(view.project()).resolve("ondemand").resolve(view.name() + ".json"), json);
        } catch (IOException e) {
            throw new IllegalStateException("saveOnDemandFeatureView failed", e);
        }
    }

    @Override
    public Optional<OnDemandFeatureView> getOnDemandFeatureView(String project, String name) {
        return memory.getOnDemandFeatureView(project, name);
    }

    @Override
    public List<OnDemandFeatureView> listOnDemandFeatureViews(String project) {
        return memory.listOnDemandFeatureViews(project);
    }

    @Override
    public void saveStreamFeatureView(StreamFeatureView view) {
        memory.saveStreamFeatureView(view);
        try {
            String json = "{"
                    + quote("name") + ":" + quote(view.name()) + ","
                    + quote("description") + ":" + quote(view.description()) + ","
                    + quote("fields") + ":" + quote(fieldsSummary(view.schema()))
                    + "}";
            atomicWrite(root.resolve(view.project()).resolve("streams").resolve(view.name() + ".json"), json);
        } catch (IOException e) {
            throw new IllegalStateException("saveStreamFeatureView failed", e);
        }
    }

    @Override
    public Optional<StreamFeatureView> getStreamFeatureView(String project, String name) {
        return memory.getStreamFeatureView(project, name);
    }

    @Override
    public List<StreamFeatureView> listStreamFeatureViews(String project) {
        return memory.listStreamFeatureViews(project);
    }

    @Override
    public void saveFeatureService(FeatureService service) {
        memory.saveFeatureService(service);
        try {
            String json = "{"
                    + quote("name") + ":" + quote(service.name()) + ","
                    + quote("description") + ":" + quote(service.description()) + ","
                    + quote("views") + ":" + quote(String.join(",", service.viewNames())) + ","
                    + quote("onDemand") + ":" + quote(String.join(",", service.onDemandViewNames()))
                    + "}";
            atomicWrite(root.resolve(service.project()).resolve("services").resolve(service.name() + ".json"), json);
        } catch (IOException e) {
            throw new IllegalStateException("saveFeatureService failed", e);
        }
    }

    @Override
    public Optional<FeatureService> getFeatureService(String project, String name) {
        return memory.getFeatureService(project, name);
    }

    @Override
    public List<FeatureService> listFeatureServices(String project) {
        return memory.listFeatureServices(project);
    }

    @Override
    public void saveVersion(FeatureVersion version) {
        memory.saveVersion(version);
        try {
            String json = "{"
                    + quote("versionId") + ":" + quote(version.versionId()) + ","
                    + quote("resourceName") + ":" + quote(version.resourceName()) + ","
                    + quote("resourceType") + ":" + quote(version.resourceType().name()) + ","
                    + quote("stage") + ":" + quote(version.stage().name()) + ","
                    + quote("schemaHash") + ":" + quote(version.schemaHash()) + ","
                    + quote("description") + ":" + quote(version.description()) + ","
                    + quote("createdAt") + ":" + quote(version.createdAt().toString()) + ","
                    + quote("createdBy") + ":" + quote(version.createdBy())
                    + "}";
            String file = version.resourceType().name() + "_" + version.resourceName() + "_" + version.versionId() + ".json";
            atomicWrite(root.resolve(version.project()).resolve("versions").resolve(file), json);
        } catch (IOException e) {
            throw new IllegalStateException("saveVersion failed", e);
        }
    }

    @Override
    public Optional<FeatureVersion> getVersion(String project, FeatureVersion.ResourceType type,
                                               String resourceName, String versionId) {
        return memory.getVersion(project, type, resourceName, versionId);
    }

    @Override
    public List<FeatureVersion> listVersions(String project, FeatureVersion.ResourceType type,
                                             String resourceName) {
        return memory.listVersions(project, type, resourceName);
    }

    @Override
    public void setProductionPointer(String project, FeatureVersion.ResourceType type,
                                     String resourceName, String versionId) {
        memory.setProductionPointer(project, type, resourceName, versionId);
        try {
            String file = type.name() + "_" + resourceName + ".ptr";
            atomicWrite(root.resolve(project).resolve("prod").resolve(file), versionId);
        } catch (IOException e) {
            throw new IllegalStateException("setProductionPointer failed", e);
        }
    }

    @Override
    public Optional<String> getProductionPointer(String project, FeatureVersion.ResourceType type,
                                                 String resourceName) {
        return memory.getProductionPointer(project, type, resourceName);
    }

    @Override
    public void flush() {
        // writes are already durable per-op
    }
}
