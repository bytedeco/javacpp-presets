package org.bytedeco.pytorch.deploy.docker;

import org.bytedeco.pytorch.deploy.k8s.K8s;
import org.bytedeco.pytorch.deploy.k8s.K8sClusterOps;
import org.bytedeco.pytorch.deploy.k8s.K8sOptions;
import org.bytedeco.pytorch.deploy.k8s.Manifest;
import org.bytedeco.pytorch.deploy.k8s.ModelServingManifest;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.time.Instant;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Unified model-service deploy entry spanning Docker Compose and Kubernetes.
 *
 * <p>Full lifecycle: {@link #plan} → write YAML → {@link #apply} → {@link #waitHealthy}
 * → {@link #status} → {@link #rollback} / {@link #teardown}. Supports {@code dryRun}
 * (generate YAML only).
 *
 * <pre>{@code
 * ModelServiceDeployer deployer = ModelServiceDeployer.builder()
 *     .target(ModelServiceDeployer.DeployTarget.DOCKER_COMPOSE)
 *     .workDir(Path.of("deploy/ranker"))
 *     .build();
 * ModelContainerSpec spec = ModelContainerSpec.builder("ranker", "my/ranker:v1")
 *     .ports(8080, 8000).gpus(1).healthHttp(8000, "/health").build();
 * DeployPlan plan = deployer.planCompose(spec);
 * DeployResult result = deployer.apply(plan);
 * }</pre>
 */
public final class ModelServiceDeployer {

    public enum DeployTarget {
        DOCKER_COMPOSE,
        K8S
    }

    public enum DeployPhase {
        PLANNED,
        APPLIED,
        HEALTHY,
        FAILED,
        ROLLED_BACK,
        TORN_DOWN
    }

    private final DeployTarget target;
    private final Path workDir;
    private final DockerOptions dockerOptions;
    private final K8sOptions k8sOptions;
    private final boolean dryRun;
    private final Duration healthTimeout;
    private final Duration healthPollInterval;

    private ModelServiceDeployer(Builder b) {
        this.target = b.target == null ? DeployTarget.DOCKER_COMPOSE : b.target;
        this.workDir = b.workDir == null ? Path.of(".") : b.workDir;
        this.dockerOptions = b.dockerOptions == null ? DockerOptions.defaults() : b.dockerOptions;
        this.k8sOptions = b.k8sOptions == null ? K8sOptions.defaults() : b.k8sOptions;
        this.dryRun = b.dryRun;
        this.healthTimeout = b.healthTimeout == null ? Duration.ofMinutes(3) : b.healthTimeout;
        this.healthPollInterval = b.healthPollInterval == null ? Duration.ofSeconds(3) : b.healthPollInterval;
    }

    public static Builder builder() {
        return new Builder();
    }

    public DeployTarget target() { return target; }
    public Path workDir() { return workDir; }
    public boolean dryRun() { return dryRun; }

    // ---- plan ----

    public DeployPlan planCompose(ModelContainerSpec spec) throws IOException {
        Objects.requireNonNull(spec, "spec");
        Files.createDirectories(workDir);
        ComposeFile cf = ComposeFile.create()
                .name(spec.serviceName())
                .addModelService(spec);
        Path composePath = workDir.resolve("docker-compose.yml");
        String yaml = cf.toYaml();
        if (!dryRun) {
            Files.writeString(composePath, yaml);
        }
        DeployPlan plan = new DeployPlan(
                DeployTarget.DOCKER_COMPOSE,
                spec.serviceName(),
                composePath,
                yaml,
                List.of(),
                Instant.now(),
                Map.of("service", spec.serviceName(), "image", spec.image()));
        return plan;
    }

    public DeployPlan planCompose(ComposeFile composeFile, String projectName) throws IOException {
        Objects.requireNonNull(composeFile, "composeFile");
        Files.createDirectories(workDir);
        if (projectName != null) composeFile.name(projectName);
        Path composePath = workDir.resolve("docker-compose.yml");
        String yaml = composeFile.toYaml();
        if (!dryRun) Files.writeString(composePath, yaml);
        return new DeployPlan(
                DeployTarget.DOCKER_COMPOSE,
                projectName == null ? "compose" : projectName,
                composePath,
                yaml,
                List.of(),
                Instant.now(),
                Map.of("services", String.valueOf(composeFile.services().keySet())));
    }

    public DeployPlan planK8s(ModelServingManifest.ModelServiceSpec spec) throws IOException {
        Objects.requireNonNull(spec, "spec");
        Files.createDirectories(workDir);
        ModelServingManifest msm = ModelServingManifest.from(spec);
        Manifest manifest = msm.toManifest();
        String yaml = manifest.toYaml();
        Path manifestPath = workDir.resolve(spec.name() + ".yaml");
        if (!dryRun) Files.writeString(manifestPath, yaml);
        return new DeployPlan(
                DeployTarget.K8S,
                spec.name(),
                manifestPath,
                yaml,
                manifest.documents(),
                Instant.now(),
                Map.of(
                        "name", spec.name(),
                        "namespace", spec.namespace() == null ? "default" : spec.namespace(),
                        "image", spec.image(),
                        "replicas", String.valueOf(spec.replicas())));
    }

    public DeployPlan planK8s(Manifest manifest, String name) throws IOException {
        Objects.requireNonNull(manifest, "manifest");
        Files.createDirectories(workDir);
        String yaml = manifest.toYaml();
        Path manifestPath = workDir.resolve((name == null ? "manifest" : name) + ".yaml");
        if (!dryRun) Files.writeString(manifestPath, yaml);
        return new DeployPlan(
                DeployTarget.K8S,
                name == null ? "app" : name,
                manifestPath,
                yaml,
                manifest.documents(),
                Instant.now(),
                Map.of());
    }

    // ---- apply / teardown ----

    public DeployResult apply(DeployPlan plan) {
        Objects.requireNonNull(plan, "plan");
        if (dryRun) {
            return DeployResult.ok(plan, DeployPhase.PLANNED, "dry-run: yaml generated at " + plan.artifactPath());
        }
        try {
            if (plan.target() == DeployTarget.DOCKER_COMPOSE) {
                DockerCompose compose = DockerCompose.connect(dockerOptions);
                if (!compose.available()) {
                    return DeployResult.fail(plan, "docker compose not available");
                }
                Path file = plan.artifactPath();
                Path dir = file.getParent();
                compose.up(file, dir, true, false, false);
                return DeployResult.ok(plan, DeployPhase.APPLIED, "compose up -d");
            }
            // K8s
            try (K8s k8s = K8s.connect(k8sOptions)) {
                if (plan.artifactPath() != null && Files.isRegularFile(plan.artifactPath())) {
                    k8s.kubectl().apply(plan.artifactPath());
                } else if (plan.documents() != null && !plan.documents().isEmpty()) {
                    k8s.applyManifest(new Manifest(plan.documents()));
                } else {
                    Path tmp = Files.createTempFile("jnitorch-mdeploy-", ".yaml");
                    Files.writeString(tmp, plan.yaml());
                    try {
                        k8s.kubectl().apply(tmp);
                    } finally {
                        Files.deleteIfExists(tmp);
                    }
                }
                return DeployResult.ok(plan, DeployPhase.APPLIED, "kubectl apply");
            }
        } catch (Exception e) {
            return DeployResult.fail(plan, e.getMessage() == null ? e.toString() : e.getMessage());
        }
    }

    public DeployResult waitHealthy(DeployPlan plan, DeployResult previous) {
        if (dryRun) {
            return DeployResult.ok(plan, DeployPhase.HEALTHY, "dry-run skip health");
        }
        if (previous != null && previous.phase() == DeployPhase.FAILED) return previous;
        long deadline = System.currentTimeMillis() + healthTimeout.toMillis();
        try {
            if (plan.target() == DeployTarget.DOCKER_COMPOSE) {
                DockerCompose compose = DockerCompose.connect(dockerOptions);
                Path file = plan.artifactPath();
                Path dir = file == null ? workDir : file.getParent();
                while (System.currentTimeMillis() < deadline) {
                    List<Map<String, Object>> rows = compose.psJson(file, dir);
                    boolean any = false;
                    boolean allOk = true;
                    for (Map<String, Object> row : rows) {
                        any = true;
                        String state = String.valueOf(row.getOrDefault("State",
                                row.getOrDefault("State", "")));
                        String health = String.valueOf(row.getOrDefault("Health", ""));
                        if (!state.toLowerCase().contains("running")) allOk = false;
                        if (health != null && !health.isBlank() && !"null".equals(health)
                                && !health.equalsIgnoreCase("healthy")
                                && !health.equalsIgnoreCase("")) {
                            // if health present and not healthy
                            if (!"starting".equalsIgnoreCase(health) && !"healthy".equalsIgnoreCase(health)) {
                                allOk = false;
                            }
                            if ("starting".equalsIgnoreCase(health)) allOk = false;
                        }
                    }
                    if (any && allOk) {
                        return DeployResult.ok(plan, DeployPhase.HEALTHY, "compose services running");
                    }
                    Thread.sleep(healthPollInterval.toMillis());
                }
                return DeployResult.fail(plan, "compose health timeout after " + healthTimeout);
            }
            // K8s: wait deployment available
            try (K8s k8s = K8s.connect(k8sOptions)) {
                String name = plan.name();
                String ns = plan.meta().getOrDefault("namespace", k8sOptions.namespace);
                k8s.kubectl().rolloutStatus("deployment/" + name, ns, healthTimeout);
                return DeployResult.ok(plan, DeployPhase.HEALTHY, "deployment available");
            }
        } catch (Exception e) {
            return DeployResult.fail(plan, "health: " + e.getMessage());
        }
    }

    public Map<String, Object> status(DeployPlan plan) {
        Map<String, Object> out = new LinkedHashMap<>();
        out.put("target", plan.target().name());
        out.put("name", plan.name());
        out.put("artifact", plan.artifactPath() == null ? null : plan.artifactPath().toString());
        try {
            if (plan.target() == DeployTarget.DOCKER_COMPOSE) {
                DockerCompose compose = DockerCompose.connect(dockerOptions);
                Path file = plan.artifactPath();
                Path dir = file == null ? workDir : file.getParent();
                out.put("ps", compose.ps(file, dir));
                out.put("services", compose.psJson(file, dir));
            } else {
                try (K8s k8s = K8s.connect(k8sOptions)) {
                    String ns = plan.meta().getOrDefault("namespace", k8sOptions.namespace);
                    out.put("deployments", k8s.kubectl().get("deploy", plan.name(), ns, true));
                    out.put("pods", k8s.kubectl().get("pods", null, ns, true));
                }
            }
        } catch (Exception e) {
            out.put("error", e.getMessage());
        }
        return out;
    }

    public DeployResult teardown(DeployPlan plan) {
        if (dryRun) {
            return DeployResult.ok(plan, DeployPhase.TORN_DOWN, "dry-run teardown");
        }
        try {
            if (plan.target() == DeployTarget.DOCKER_COMPOSE) {
                DockerCompose compose = DockerCompose.connect(dockerOptions);
                Path file = plan.artifactPath();
                Path dir = file == null ? workDir : file.getParent();
                compose.down(file, dir, false, false);
                return DeployResult.ok(plan, DeployPhase.TORN_DOWN, "compose down");
            }
            try (K8s k8s = K8s.connect(k8sOptions)) {
                if (plan.artifactPath() != null && Files.isRegularFile(plan.artifactPath())) {
                    k8s.kubectl().delete(plan.artifactPath());
                } else {
                    Path tmp = Files.createTempFile("jnitorch-mdeploy-del-", ".yaml");
                    Files.writeString(tmp, plan.yaml());
                    try {
                        k8s.kubectl().delete(tmp);
                    } finally {
                        Files.deleteIfExists(tmp);
                    }
                }
                return DeployResult.ok(plan, DeployPhase.TORN_DOWN, "kubectl delete -f");
            }
        } catch (Exception e) {
            return DeployResult.fail(plan, "teardown: " + e.getMessage());
        }
    }

    /**
     * K8s-only: rollout undo for deployment {@code plan.name()}.
     */
    public DeployResult rollback(DeployPlan plan) {
        if (plan.target() != DeployTarget.K8S) {
            return DeployResult.fail(plan, "rollback only supported for K8S target (compose: redeploy previous yaml)");
        }
        if (dryRun) {
            return DeployResult.ok(plan, DeployPhase.ROLLED_BACK, "dry-run rollback");
        }
        try (K8s k8s = K8s.connect(k8sOptions)) {
            String ns = plan.meta().getOrDefault("namespace", k8sOptions.namespace);
            k8s.kubectl().rolloutUndo("deployment/" + plan.name(), ns);
            return DeployResult.ok(plan, DeployPhase.ROLLED_BACK, "rollout undo");
        } catch (Exception e) {
            return DeployResult.fail(plan, "rollback: " + e.getMessage());
        }
    }

    /**
     * Full happy-path: plan + apply + waitHealthy for a model container on Compose.
     */
    public DeployResult deployCompose(ModelContainerSpec spec) throws IOException {
        DeployPlan plan = planCompose(spec);
        DeployResult applied = apply(plan);
        if (applied.phase() == DeployPhase.FAILED || dryRun) return applied;
        return waitHealthy(plan, applied);
    }

    /**
     * Full happy-path on Kubernetes.
     */
    public DeployResult deployK8s(ModelServingManifest.ModelServiceSpec spec) throws IOException {
        DeployPlan plan = planK8s(spec);
        DeployResult applied = apply(plan);
        if (applied.phase() == DeployPhase.FAILED || dryRun) return applied;
        return waitHealthy(plan, applied);
    }

    /** Create {@link K8sClusterOps} bound to these options (for DeploymentController). */
    public K8sClusterOps clusterOps() {
        return new K8sClusterOps(k8sOptions);
    }

    // ---- types ----

    public static final class DeployPlan {
        private final DeployTarget target;
        private final String name;
        private final Path artifactPath;
        private final String yaml;
        private final List<Object> documents;
        private final Instant createdAt;
        private final Map<String, String> meta;

        public DeployPlan(
                DeployTarget target,
                String name,
                Path artifactPath,
                String yaml,
                List<Object> documents,
                Instant createdAt,
                Map<String, String> meta) {
            this.target = target;
            this.name = name;
            this.artifactPath = artifactPath;
            this.yaml = yaml == null ? "" : yaml;
            this.documents = documents == null ? List.of() : List.copyOf(documents);
            this.createdAt = createdAt == null ? Instant.now() : createdAt;
            this.meta = meta == null ? Map.of() : Map.copyOf(new LinkedHashMap<>(meta));
        }

        public DeployTarget target() { return target; }
        public String name() { return name; }
        public Path artifactPath() { return artifactPath; }
        public String yaml() { return yaml; }
        public List<Object> documents() { return documents; }
        public Instant createdAt() { return createdAt; }
        public Map<String, String> meta() { return meta; }

        @Override
        public String toString() {
            return "DeployPlan{target=" + target + ", name=" + name + ", path=" + artifactPath + "}";
        }
    }

    public static final class DeployResult {
        private final DeployPlan plan;
        private final DeployPhase phase;
        private final boolean success;
        private final String message;
        private final Instant at;

        private DeployResult(DeployPlan plan, DeployPhase phase, boolean success, String message) {
            this.plan = plan;
            this.phase = phase;
            this.success = success;
            this.message = message == null ? "" : message;
            this.at = Instant.now();
        }

        public static DeployResult ok(DeployPlan plan, DeployPhase phase, String message) {
            return new DeployResult(plan, phase, true, message);
        }

        public static DeployResult fail(DeployPlan plan, String message) {
            return new DeployResult(plan, DeployPhase.FAILED, false, message);
        }

        public DeployPlan plan() { return plan; }
        public DeployPhase phase() { return phase; }
        public boolean success() { return success; }
        public String message() { return message; }
        public Instant at() { return at; }

        @Override
        public String toString() {
            return "DeployResult{phase=" + phase + ", ok=" + success + ", msg=" + message + "}";
        }
    }

    public static final class Builder {
        private DeployTarget target = DeployTarget.DOCKER_COMPOSE;
        private Path workDir = Path.of(".");
        private DockerOptions dockerOptions;
        private K8sOptions k8sOptions;
        private boolean dryRun;
        private Duration healthTimeout = Duration.ofMinutes(3);
        private Duration healthPollInterval = Duration.ofSeconds(3);

        public Builder target(DeployTarget t) { this.target = t; return this; }
        public Builder workDir(Path p) { this.workDir = p; return this; }
        public Builder dockerOptions(DockerOptions o) { this.dockerOptions = o; return this; }
        public Builder k8sOptions(K8sOptions o) { this.k8sOptions = o; return this; }
        public Builder dryRun(boolean v) { this.dryRun = v; return this; }
        public Builder healthTimeout(Duration d) { this.healthTimeout = d; return this; }
        public Builder healthPollInterval(Duration d) { this.healthPollInterval = d; return this; }

        public ModelServiceDeployer build() {
            return new ModelServiceDeployer(this);
        }
    }
}
