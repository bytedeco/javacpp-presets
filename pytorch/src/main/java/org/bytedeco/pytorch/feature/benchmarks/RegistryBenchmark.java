/*
 * RegistryBenchmark — register / promote / list / file round-trip.
 */
package org.bytedeco.pytorch.feature.benchmarks;

import org.bytedeco.pytorch.feature.core.Entity;
import org.bytedeco.pytorch.feature.core.FeatureView;
import org.bytedeco.pytorch.feature.core.Field;
import org.bytedeco.pytorch.feature.core.ValueType;
import org.bytedeco.pytorch.feature.registry.FeatureRegistry;
import org.bytedeco.pytorch.feature.registry.FeatureVersion;
import org.bytedeco.pytorch.feature.registry.FileRegistryStore;
import org.bytedeco.pytorch.feature.registry.LifecycleStage;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

/** Registry correctness + throughput bench. */
public final class RegistryBenchmark {

    private RegistryBenchmark() {}

    public static void run(BenchCase.Suite suite, int nViews) {
        long t0 = System.nanoTime();
        try {
            FeatureRegistry reg = new FeatureRegistry();
            Entity user = Entity.of("user_id");
            reg.registerEntity(user);
            for (int i = 0; i < nViews; i++) {
                FeatureView v = FeatureView.builder("view_" + i)
                        .entities(user)
                        .schema(Field.of("f_" + i, ValueType.FLOAT64),
                                Field.of("id_" + i, ValueType.INT64))
                        .online(true)
                        .build();
                reg.registerFeatureView(v);
            }
            List<FeatureView> listed = reg.listFeatureViews("default");
            if (listed.size() != nViews) {
                suite.add(BenchCase.fail("registry_register",
                        "expected " + nViews + " views, got " + listed.size(), System.nanoTime() - t0));
                return;
            }
            // promote latest of view_0: DRAFT→VALIDATED→PROD
            FeatureVersion latest = reg.latestVersion("default",
                    FeatureVersion.ResourceType.FEATURE_VIEW, "view_0").orElseThrow();
            reg.promote("default", FeatureVersion.ResourceType.FEATURE_VIEW, "view_0", latest.versionId());
            reg.promote("default", FeatureVersion.ResourceType.FEATURE_VIEW, "view_0", latest.versionId());
            FeatureVersion prod = reg.productionOf("default",
                    FeatureVersion.ResourceType.FEATURE_VIEW, "view_0").orElseThrow();
            if (prod.stage() != LifecycleStage.PROD) {
                suite.add(BenchCase.fail("registry_promote", "stage=" + prod.stage(), System.nanoTime() - t0));
                return;
            }
            // illegal transition
            boolean rejected = false;
            try {
                reg.transition("default", FeatureVersion.ResourceType.FEATURE_VIEW,
                        "view_0", latest.versionId(), LifecycleStage.DRAFT);
            } catch (IllegalStateException ex) {
                rejected = true;
            }
            if (!rejected) {
                suite.add(BenchCase.fail("registry_lifecycle", "illegal transition accepted", System.nanoTime() - t0));
                return;
            }
            suite.add(BenchCase.pass("registry_register_promote",
                    "views=" + nViews + ", prod=" + prod.versionId(), System.nanoTime() - t0));
        } catch (Exception e) {
            suite.add(BenchCase.fail("registry_register_promote", e.toString(), System.nanoTime() - t0));
        }

        // file round-trip
        long t1 = System.nanoTime();
        try {
            Path tmp = Files.createTempDirectory("feature-registry-bench");
            FeatureRegistry reg = new FeatureRegistry(new FileRegistryStore(tmp));
            reg.registerEntity(Entity.of("item_id"));
            FeatureView v = FeatureView.builder("file_view")
                    .entities(Entity.of("item_id"))
                    .schema(Field.of("score", ValueType.FLOAT64))
                    .build();
            reg.registerFeatureView(v);
            reg.flush();

            FeatureRegistry reloaded = new FeatureRegistry(new FileRegistryStore(tmp));
            if (reloaded.getFeatureView("default", "file_view").isEmpty()) {
                suite.add(BenchCase.fail("registry_file_roundtrip", "view missing after reload", System.nanoTime() - t1));
            } else {
                suite.add(BenchCase.pass("registry_file_roundtrip", "root=" + tmp, System.nanoTime() - t1));
            }
        } catch (Exception e) {
            suite.add(BenchCase.fail("registry_file_roundtrip", e.toString(), System.nanoTime() - t1));
        }
    }
}
