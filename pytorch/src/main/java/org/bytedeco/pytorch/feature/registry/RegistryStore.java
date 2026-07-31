/*
 * Pluggable persistence for Feature Registry metadata.
 */
package org.bytedeco.pytorch.feature.registry;

import org.bytedeco.pytorch.feature.core.Entity;
import org.bytedeco.pytorch.feature.core.FeatureService;
import org.bytedeco.pytorch.feature.core.FeatureView;
import org.bytedeco.pytorch.feature.core.OnDemandFeatureView;
import org.bytedeco.pytorch.feature.core.Project;
import org.bytedeco.pytorch.feature.core.StreamFeatureView;

import java.util.List;
import java.util.Optional;

/** Registry storage SPI. */
public interface RegistryStore {

    void saveProject(Project project);

    Optional<Project> getProject(String name);

    List<Project> listProjects();

    void saveEntity(Entity entity);

    Optional<Entity> getEntity(String project, String name);

    List<Entity> listEntities(String project);

    void saveFeatureView(FeatureView view);

    Optional<FeatureView> getFeatureView(String project, String name);

    List<FeatureView> listFeatureViews(String project);

    void saveOnDemandFeatureView(OnDemandFeatureView view);

    Optional<OnDemandFeatureView> getOnDemandFeatureView(String project, String name);

    List<OnDemandFeatureView> listOnDemandFeatureViews(String project);

    void saveStreamFeatureView(StreamFeatureView view);

    Optional<StreamFeatureView> getStreamFeatureView(String project, String name);

    List<StreamFeatureView> listStreamFeatureViews(String project);

    void saveFeatureService(FeatureService service);

    Optional<FeatureService> getFeatureService(String project, String name);

    List<FeatureService> listFeatureServices(String project);

    void saveVersion(FeatureVersion version);

    Optional<FeatureVersion> getVersion(String project, FeatureVersion.ResourceType type,
                                        String resourceName, String versionId);

    List<FeatureVersion> listVersions(String project, FeatureVersion.ResourceType type,
                                      String resourceName);

    void setProductionPointer(String project, FeatureVersion.ResourceType type,
                              String resourceName, String versionId);

    Optional<String> getProductionPointer(String project, FeatureVersion.ResourceType type,
                                          String resourceName);

    /** Flush durable state if any. */
    default void flush() {}

    default void close() {}
}
