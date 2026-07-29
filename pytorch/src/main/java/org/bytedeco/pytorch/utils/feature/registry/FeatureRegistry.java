/*
 * Feature Registry — source of truth for feature metadata and lifecycle.
 *
 * Mirrors Feast registry / Databricks Feature Registry / Featureform,
 * API shape aligned with recommend.modelops.ModelRegistry.
 */
package org.bytedeco.pytorch.utils.feature.registry;

import org.bytedeco.pytorch.utils.feature.core.Entity;
import org.bytedeco.pytorch.utils.feature.core.FeatureService;
import org.bytedeco.pytorch.utils.feature.core.FeatureView;
import org.bytedeco.pytorch.utils.feature.core.OnDemandFeatureView;
import org.bytedeco.pytorch.utils.feature.core.Project;
import org.bytedeco.pytorch.utils.feature.core.StreamFeatureView;

import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.util.ArrayList;
import java.util.HexFormat;
import java.util.List;
import java.util.Objects;
import java.util.Optional;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.function.Consumer;
import java.util.stream.Collectors;

/** Thread-safe feature metadata registry with stage transitions. */
public final class FeatureRegistry {

    public static final class RegistryEvent {
        public enum Kind { REGISTERED, STAGE_CHANGED, PROD_SET }

        public final Kind kind;
        public final FeatureVersion version;
        public final LifecycleStage from;
        public final LifecycleStage to;
        public final long atMs;

        private RegistryEvent(Kind kind, FeatureVersion version, LifecycleStage from, LifecycleStage to) {
            this.kind = kind;
            this.version = version;
            this.from = from;
            this.to = to;
            this.atMs = System.currentTimeMillis();
        }

        public static RegistryEvent registered(FeatureVersion v) {
            return new RegistryEvent(Kind.REGISTERED, v, null, v.stage());
        }

        public static RegistryEvent stageChanged(FeatureVersion v, LifecycleStage from, LifecycleStage to) {
            return new RegistryEvent(Kind.STAGE_CHANGED, v, from, to);
        }

        public static RegistryEvent prodSet(FeatureVersion v) {
            return new RegistryEvent(Kind.PROD_SET, v, null, LifecycleStage.PROD);
        }

        @Override
        public String toString() {
            return "RegistryEvent{" + kind + "," + version.fullyQualifiedId() + "}";
        }
    }

    private final RegistryStore store;
    private final LineageGraph lineage = new LineageGraph();
    private final CopyOnWriteArrayList<Consumer<RegistryEvent>> listeners = new CopyOnWriteArrayList<>();

    public FeatureRegistry() {
        this(new InMemoryRegistryStore());
    }

    public FeatureRegistry(RegistryStore store) {
        this.store = Objects.requireNonNull(store, "store");
        // ensure default project
        if (store.getProject(Project.DEFAULT).isEmpty()) {
            store.saveProject(Project.of(Project.DEFAULT));
        }
    }

    public RegistryStore store() {
        return store;
    }

    public LineageGraph lineage() {
        return lineage;
    }

    public void addListener(Consumer<RegistryEvent> listener) {
        listeners.add(Objects.requireNonNull(listener));
    }

    private void emit(RegistryEvent event) {
        for (Consumer<RegistryEvent> l : listeners) {
            try {
                l.accept(event);
            } catch (RuntimeException ignored) {
            }
        }
    }

    // ── projects ────────────────────────────────────────────────────────────

    public Project registerProject(Project project) {
        Objects.requireNonNull(project, "project");
        store.saveProject(project);
        return project;
    }

    public Optional<Project> getProject(String name) {
        return store.getProject(name);
    }

    public List<Project> listProjects() {
        return store.listProjects();
    }

    // ── entities ────────────────────────────────────────────────────────────

    public Entity registerEntity(Entity entity) {
        Objects.requireNonNull(entity, "entity");
        ensureProject(entity.project());
        store.saveEntity(entity);
        FeatureVersion v = FeatureVersion.builder(entity.name(), FeatureVersion.ResourceType.ENTITY)
                .project(entity.project())
                .stage(LifecycleStage.DRAFT)
                .schemaHash(hash(entity.name() + ":" + entity.valueType()))
                .payload(entity)
                .build();
        store.saveVersion(v);
        emit(RegistryEvent.registered(v));
        return entity;
    }

    public Optional<Entity> getEntity(String project, String name) {
        return store.getEntity(project, name);
    }

    public Entity requireEntity(String project, String name) {
        return getEntity(project, name)
                .orElseThrow(() -> new IllegalArgumentException("unknown entity: " + project + "/" + name));
    }

    public List<Entity> listEntities(String project) {
        return store.listEntities(project);
    }

    // ── feature views ───────────────────────────────────────────────────────

    public FeatureView registerFeatureView(FeatureView view) {
        Objects.requireNonNull(view, "view");
        ensureProject(view.project());
        store.saveFeatureView(view);
        for (Entity e : view.entities()) {
            lineage.addEdge(view.qualifiedName(), "entity:" + e.project() + "/" + e.name(), "joins");
        }
        if (view.source() != null) {
            lineage.addEdge(view.qualifiedName(), "source:" + view.source().name(), "reads");
        }
        FeatureVersion v = FeatureVersion.builder(view.name(), FeatureVersion.ResourceType.FEATURE_VIEW)
                .project(view.project())
                .stage(LifecycleStage.DRAFT)
                .schemaHash(hashSchema(view))
                .description(view.description())
                .payload(view)
                .build();
        store.saveVersion(v);
        emit(RegistryEvent.registered(v));
        return view;
    }

    public Optional<FeatureView> getFeatureView(String project, String name) {
        return store.getFeatureView(project, name);
    }

    public FeatureView requireFeatureView(String project, String name) {
        return getFeatureView(project, name)
                .orElseThrow(() -> new IllegalArgumentException("unknown feature view: " + project + "/" + name));
    }

    public List<FeatureView> listFeatureViews(String project) {
        return store.listFeatureViews(project);
    }

    // ── on-demand ───────────────────────────────────────────────────────────

    public OnDemandFeatureView registerOnDemandFeatureView(OnDemandFeatureView view) {
        Objects.requireNonNull(view, "view");
        ensureProject(view.project());
        store.saveOnDemandFeatureView(view);
        for (String src : view.sourceViewNames()) {
            lineage.addEdge(view.qualifiedName(), view.project() + "/" + src, "depends_on");
        }
        FeatureVersion v = FeatureVersion.builder(view.name(), FeatureVersion.ResourceType.ON_DEMAND_FEATURE_VIEW)
                .project(view.project())
                .stage(LifecycleStage.DRAFT)
                .schemaHash(hash(view.name() + view.schema().toString()))
                .payload(view)
                .build();
        store.saveVersion(v);
        emit(RegistryEvent.registered(v));
        return view;
    }

    public Optional<OnDemandFeatureView> getOnDemandFeatureView(String project, String name) {
        return store.getOnDemandFeatureView(project, name);
    }

    public List<OnDemandFeatureView> listOnDemandFeatureViews(String project) {
        return store.listOnDemandFeatureViews(project);
    }

    // ── stream ──────────────────────────────────────────────────────────────

    public StreamFeatureView registerStreamFeatureView(StreamFeatureView view) {
        Objects.requireNonNull(view, "view");
        ensureProject(view.project());
        store.saveStreamFeatureView(view);
        FeatureVersion v = FeatureVersion.builder(view.name(), FeatureVersion.ResourceType.STREAM_FEATURE_VIEW)
                .project(view.project())
                .stage(LifecycleStage.DRAFT)
                .schemaHash(hash(view.name() + view.schema().toString()))
                .payload(view)
                .build();
        store.saveVersion(v);
        emit(RegistryEvent.registered(v));
        return view;
    }

    public Optional<StreamFeatureView> getStreamFeatureView(String project, String name) {
        return store.getStreamFeatureView(project, name);
    }

    public List<StreamFeatureView> listStreamFeatureViews(String project) {
        return store.listStreamFeatureViews(project);
    }

    // ── feature services ────────────────────────────────────────────────────

    public FeatureService registerFeatureService(FeatureService service) {
        Objects.requireNonNull(service, "service");
        ensureProject(service.project());
        store.saveFeatureService(service);
        for (String vn : service.allViewNames()) {
            lineage.addEdge(service.qualifiedName(), service.project() + "/" + vn, "serves");
        }
        FeatureVersion v = FeatureVersion.builder(service.name(), FeatureVersion.ResourceType.FEATURE_SERVICE)
                .project(service.project())
                .stage(LifecycleStage.DRAFT)
                .schemaHash(hash(service.name() + service.viewNames().toString()))
                .payload(service)
                .build();
        store.saveVersion(v);
        emit(RegistryEvent.registered(v));
        return service;
    }

    public Optional<FeatureService> getFeatureService(String project, String name) {
        return store.getFeatureService(project, name);
    }

    public FeatureService requireFeatureService(String project, String name) {
        return getFeatureService(project, name)
                .orElseThrow(() -> new IllegalArgumentException(
                        "unknown feature service: " + project + "/" + name));
    }

    public List<FeatureService> listFeatureServices(String project) {
        return store.listFeatureServices(project);
    }

    // ── lifecycle ───────────────────────────────────────────────────────────

    public List<FeatureVersion> listVersions(String project, FeatureVersion.ResourceType type,
                                             String resourceName) {
        return store.listVersions(project, type, resourceName);
    }

    public Optional<FeatureVersion> getVersion(String project, FeatureVersion.ResourceType type,
                                               String resourceName, String versionId) {
        return store.getVersion(project, type, resourceName, versionId);
    }

    public synchronized FeatureVersion transition(String project, FeatureVersion.ResourceType type,
                                                  String resourceName, String versionId,
                                                  LifecycleStage to) {
        FeatureVersion current = store.getVersion(project, type, resourceName, versionId)
                .orElseThrow(() -> new IllegalArgumentException(
                        "unknown version: " + project + "/" + resourceName + ":" + versionId));
        LifecycleStage from = current.stage();
        if (!from.canTransitionTo(to)) {
            throw new IllegalStateException("illegal transition " + from + " → " + to
                    + " for " + current.fullyQualifiedId());
        }
        FeatureVersion updated = current.withStage(to);
        store.saveVersion(updated);
        if (to == LifecycleStage.PROD) {
            // archive previous prod
            store.getProductionPointer(project, type, resourceName).ifPresent(prevId -> {
                if (!prevId.equals(versionId)) {
                    store.getVersion(project, type, resourceName, prevId).ifPresent(prev -> {
                        if (prev.stage() == LifecycleStage.PROD) {
                            FeatureVersion archived = prev.withStage(LifecycleStage.ARCHIVED);
                            store.saveVersion(archived);
                            emit(RegistryEvent.stageChanged(archived, LifecycleStage.PROD, LifecycleStage.ARCHIVED));
                        }
                    });
                }
            });
            store.setProductionPointer(project, type, resourceName, versionId);
            emit(RegistryEvent.prodSet(updated));
        }
        if (from == LifecycleStage.PROD && to != LifecycleStage.PROD) {
            store.getProductionPointer(project, type, resourceName).ifPresent(ptr -> {
                if (ptr.equals(versionId)) {
                    // clear by setting empty — store impl may keep last; we overwrite only on PROD
                }
            });
        }
        emit(RegistryEvent.stageChanged(updated, from, to));
        return updated;
    }

    /**
     * Promote one step on happy path: DRAFT→VALIDATED→PROD→DEPRECATED→ARCHIVED.
     */
    public synchronized FeatureVersion promote(String project, FeatureVersion.ResourceType type,
                                               String resourceName, String versionId) {
        FeatureVersion current = store.getVersion(project, type, resourceName, versionId)
                .orElseThrow(() -> new IllegalArgumentException(
                        "unknown version: " + project + "/" + resourceName + ":" + versionId));
        LifecycleStage next = current.stage().nextPromote();
        if (next == null) {
            throw new IllegalStateException("cannot promote further from " + current.stage()
                    + " for " + current.fullyQualifiedId());
        }
        return transition(project, type, resourceName, versionId, next);
    }

    public Optional<FeatureVersion> productionOf(String project, FeatureVersion.ResourceType type,
                                                 String resourceName) {
        return store.getProductionPointer(project, type, resourceName)
                .flatMap(vid -> store.getVersion(project, type, resourceName, vid));
    }

    /** Latest version id for a resource (by createdAt). */
    public Optional<FeatureVersion> latestVersion(String project, FeatureVersion.ResourceType type,
                                                  String resourceName) {
        List<FeatureVersion> list = store.listVersions(project, type, resourceName);
        return list.isEmpty() ? Optional.empty() : Optional.of(list.get(0));
    }

    public void flush() {
        store.flush();
    }

    public void close() {
        store.close();
    }

    private void ensureProject(String project) {
        String p = project == null || project.isEmpty() ? Project.DEFAULT : project;
        if (store.getProject(p).isEmpty()) {
            store.saveProject(Project.of(p));
        }
    }

    private static String hashSchema(FeatureView view) {
        String raw = view.featureNames().stream().sorted().collect(Collectors.joining(","))
                + "|" + view.entityNames().stream().sorted().collect(Collectors.joining(","));
        return hash(raw);
    }

    private static String hash(String raw) {
        try {
            MessageDigest md = MessageDigest.getInstance("SHA-256");
            byte[] dig = md.digest(raw.getBytes(StandardCharsets.UTF_8));
            return HexFormat.of().formatHex(dig).substring(0, 16);
        } catch (Exception e) {
            return Integer.toHexString(raw.hashCode());
        }
    }

    /** Resolve all batch views for a feature service (project-scoped). */
    public List<FeatureView> resolveViews(FeatureService service) {
        Objects.requireNonNull(service, "service");
        List<FeatureView> out = new ArrayList<>();
        for (String vn : service.viewNames()) {
            getFeatureView(service.project(), vn).ifPresent(out::add);
            // also try stream views projected as batch
            getStreamFeatureView(service.project(), vn).ifPresent(sfv -> out.add(sfv.asBatchView()));
        }
        return out;
    }

    public List<OnDemandFeatureView> resolveOnDemand(FeatureService service) {
        Objects.requireNonNull(service, "service");
        List<OnDemandFeatureView> out = new ArrayList<>();
        for (String vn : service.onDemandViewNames()) {
            getOnDemandFeatureView(service.project(), vn).ifPresent(out::add);
        }
        return out;
    }
}
