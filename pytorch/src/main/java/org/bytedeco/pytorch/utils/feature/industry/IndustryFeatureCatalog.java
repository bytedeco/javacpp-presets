/*
 * Industry feature catalog SPI — declarative templates that register
 * entities / views / services into a FeaturePlatform / FeatureRegistry.
 */
package org.bytedeco.pytorch.utils.feature.industry;

import org.bytedeco.pytorch.utils.feature.FeaturePlatform;
import org.bytedeco.pytorch.utils.feature.core.FeatureService;
import org.bytedeco.pytorch.utils.feature.core.FeatureView;
import org.bytedeco.pytorch.utils.feature.registry.FeatureRegistry;

import java.util.List;
import java.util.Map;

/** Loads a vertical's feature warehouse skeleton into the registry. */
public interface IndustryFeatureCatalog {

    IndustryDomain domain();

    /** Registry project name used by this catalog. */
    String project();

    /**
     * Register entities, feature views, on-demand views, and feature services.
     *
     * @return registered feature service names
     */
    List<String> registerAll(FeatureRegistry registry);

    default List<String> registerAll(FeaturePlatform platform) {
        return registerAll(platform.registry());
    }

    /** Primary ranking / scoring FeatureService name. */
    String primaryService();

    /** Batch feature views created by this catalog. */
    List<FeatureView> featureViews();

    /** Optional toy offline rows for smoke / benchmark (viewName → rows). */
    Map<String, List<Map<String, Object>>> sampleOfflineData(long nowMs, int nUsers, int nItems);

    default FeatureService requirePrimary(FeatureRegistry registry) {
        return registry.requireFeatureService(project(), primaryService());
    }
}
