/*
 * Thread-safe in-memory RegistryStore — default for tests and single-process demos.
 */
package org.bytedeco.pytorch.utils.feature.registry;

import org.bytedeco.pytorch.utils.feature.core.Entity;
import org.bytedeco.pytorch.utils.feature.core.FeatureService;
import org.bytedeco.pytorch.utils.feature.core.FeatureView;
import org.bytedeco.pytorch.utils.feature.core.OnDemandFeatureView;
import org.bytedeco.pytorch.utils.feature.core.Project;
import org.bytedeco.pytorch.utils.feature.core.StreamFeatureView;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Optional;
import java.util.concurrent.ConcurrentHashMap;

/** Concurrent in-memory implementation of {@link RegistryStore}. */
public final class InMemoryRegistryStore implements RegistryStore {

    private final ConcurrentHashMap<String, Project> projects = new ConcurrentHashMap<>();
    private final ConcurrentHashMap<String, Entity> entities = new ConcurrentHashMap<>();
    private final ConcurrentHashMap<String, FeatureView> views = new ConcurrentHashMap<>();
    private final ConcurrentHashMap<String, OnDemandFeatureView> onDemand = new ConcurrentHashMap<>();
    private final ConcurrentHashMap<String, StreamFeatureView> streams = new ConcurrentHashMap<>();
    private final ConcurrentHashMap<String, FeatureService> services = new ConcurrentHashMap<>();
    private final ConcurrentHashMap<String, FeatureVersion> versions = new ConcurrentHashMap<>();
    private final ConcurrentHashMap<String, String> productionPointers = new ConcurrentHashMap<>();

    private static String key(String project, String name) {
        return (project == null || project.isEmpty() ? "default" : project) + "/" + name;
    }

    private static String versionKey(String project, FeatureVersion.ResourceType type,
                                     String resourceName, String versionId) {
        return key(project, type.name() + ":" + resourceName + ":" + versionId);
    }

    private static String prodKey(String project, FeatureVersion.ResourceType type, String resourceName) {
        return key(project, type.name() + ":" + resourceName);
    }

    @Override
    public void saveProject(Project project) {
        projects.put(project.name(), project);
    }

    @Override
    public Optional<Project> getProject(String name) {
        return Optional.ofNullable(projects.get(name));
    }

    @Override
    public List<Project> listProjects() {
        return new ArrayList<>(projects.values());
    }

    @Override
    public void saveEntity(Entity entity) {
        entities.put(key(entity.project(), entity.name()), entity);
    }

    @Override
    public Optional<Entity> getEntity(String project, String name) {
        return Optional.ofNullable(entities.get(key(project, name)));
    }

    @Override
    public List<Entity> listEntities(String project) {
        String prefix = (project == null || project.isEmpty() ? "default" : project) + "/";
        List<Entity> out = new ArrayList<>();
        for (var e : entities.entrySet()) {
            if (e.getKey().startsWith(prefix)) out.add(e.getValue());
        }
        return out;
    }

    @Override
    public void saveFeatureView(FeatureView view) {
        views.put(key(view.project(), view.name()), view);
    }

    @Override
    public Optional<FeatureView> getFeatureView(String project, String name) {
        return Optional.ofNullable(views.get(key(project, name)));
    }

    @Override
    public List<FeatureView> listFeatureViews(String project) {
        String prefix = (project == null || project.isEmpty() ? "default" : project) + "/";
        List<FeatureView> out = new ArrayList<>();
        for (var e : views.entrySet()) {
            if (e.getKey().startsWith(prefix)) out.add(e.getValue());
        }
        return out;
    }

    @Override
    public void saveOnDemandFeatureView(OnDemandFeatureView view) {
        onDemand.put(key(view.project(), view.name()), view);
    }

    @Override
    public Optional<OnDemandFeatureView> getOnDemandFeatureView(String project, String name) {
        return Optional.ofNullable(onDemand.get(key(project, name)));
    }

    @Override
    public List<OnDemandFeatureView> listOnDemandFeatureViews(String project) {
        String prefix = (project == null || project.isEmpty() ? "default" : project) + "/";
        List<OnDemandFeatureView> out = new ArrayList<>();
        for (var e : onDemand.entrySet()) {
            if (e.getKey().startsWith(prefix)) out.add(e.getValue());
        }
        return out;
    }

    @Override
    public void saveStreamFeatureView(StreamFeatureView view) {
        streams.put(key(view.project(), view.name()), view);
    }

    @Override
    public Optional<StreamFeatureView> getStreamFeatureView(String project, String name) {
        return Optional.ofNullable(streams.get(key(project, name)));
    }

    @Override
    public List<StreamFeatureView> listStreamFeatureViews(String project) {
        String prefix = (project == null || project.isEmpty() ? "default" : project) + "/";
        List<StreamFeatureView> out = new ArrayList<>();
        for (var e : streams.entrySet()) {
            if (e.getKey().startsWith(prefix)) out.add(e.getValue());
        }
        return out;
    }

    @Override
    public void saveFeatureService(FeatureService service) {
        services.put(key(service.project(), service.name()), service);
    }

    @Override
    public Optional<FeatureService> getFeatureService(String project, String name) {
        return Optional.ofNullable(services.get(key(project, name)));
    }

    @Override
    public List<FeatureService> listFeatureServices(String project) {
        String prefix = (project == null || project.isEmpty() ? "default" : project) + "/";
        List<FeatureService> out = new ArrayList<>();
        for (var e : services.entrySet()) {
            if (e.getKey().startsWith(prefix)) out.add(e.getValue());
        }
        return out;
    }

    @Override
    public void saveVersion(FeatureVersion version) {
        versions.put(versionKey(version.project(), version.resourceType(),
                version.resourceName(), version.versionId()), version);
    }

    @Override
    public Optional<FeatureVersion> getVersion(String project, FeatureVersion.ResourceType type,
                                               String resourceName, String versionId) {
        return Optional.ofNullable(versions.get(versionKey(project, type, resourceName, versionId)));
    }

    @Override
    public List<FeatureVersion> listVersions(String project, FeatureVersion.ResourceType type,
                                             String resourceName) {
        String p = project == null || project.isEmpty() ? "default" : project;
        List<FeatureVersion> out = new ArrayList<>();
        for (FeatureVersion v : versions.values()) {
            if (v.project().equals(p) && v.resourceType() == type && v.resourceName().equals(resourceName)) {
                out.add(v);
            }
        }
        out.sort(Comparator.comparing(FeatureVersion::createdAt).reversed());
        return out;
    }

    @Override
    public void setProductionPointer(String project, FeatureVersion.ResourceType type,
                                     String resourceName, String versionId) {
        productionPointers.put(prodKey(project, type, resourceName), versionId);
    }

    @Override
    public Optional<String> getProductionPointer(String project, FeatureVersion.ResourceType type,
                                                 String resourceName) {
        return Optional.ofNullable(productionPointers.get(prodKey(project, type, resourceName)));
    }

    public void clear() {
        projects.clear();
        entities.clear();
        views.clear();
        onDemand.clear();
        streams.clear();
        services.clear();
        versions.clear();
        productionPointers.clear();
    }
}
