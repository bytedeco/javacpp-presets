package org.bytedeco.pytorch.deploy.k8s;

import org.bytedeco.pytorch.deploy.serving.gateway.TrafficRouter;
import org.bytedeco.pytorch.deploy.serving.deploy.DeploymentController;

import java.time.Duration;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.ConcurrentHashMap;
import java.util.function.Function;

/**
 * {@link DeploymentController.ClusterOps} backed by real Kubernetes (kubectl by default).
 *
 * <p>Mapping:
 * <ul>
 *   <li>{@code versionId} → Deployment name (optionally via {@link #deploymentNameMapper})</li>
 *   <li>{@link #scale} → {@code kubectl scale} + wait readyReplicas</li>
 *   <li>{@link #inplaceRestart} → {@code kubectl rollout restart}</li>
 *   <li>{@link #healthPercent} → {@code 100 * ready / desired}</li>
 *   <li>{@link #setTrafficWeight} → annotation
 *       {@code serving.jnitorch.io/traffic-weight} on Deployment (+ optional Service)</li>
 * </ul>
 *
 * <p>Gateway / mesh integration reads the traffic-weight annotation (or use
 * {@link TrafficRouter} in-process).
 */
public final class K8sClusterOps implements DeploymentController.ClusterOps, AutoCloseable {

    public static final String TRAFFIC_WEIGHT_ANNOTATION = "serving.jnitorch.io/traffic-weight";

    private final K8sOptions options;
    private final Kubectl kubectl;
    private final Duration scaleWait;
    private final Function<String, String> deploymentNameMapper;
    private final boolean annotateService;
    private final ConcurrentHashMap<String, Double> trafficCache = new ConcurrentHashMap<>();
    private K8sClient client; // optional lazy REST

    public K8sClusterOps(K8sOptions options) {
        this(options, null, Duration.ofMinutes(2), Function.identity(), false);
    }

    public K8sClusterOps(
            K8sOptions options,
            Kubectl kubectl,
            Duration scaleWait,
            Function<String, String> deploymentNameMapper,
            boolean annotateService) {
        this.options = options == null ? K8sOptions.defaults() : options;
        this.kubectl = kubectl == null ? new Kubectl(this.options) : kubectl;
        this.scaleWait = scaleWait == null ? Duration.ofMinutes(2) : scaleWait;
        this.deploymentNameMapper = deploymentNameMapper == null
                ? Function.identity()
                : deploymentNameMapper;
        this.annotateService = annotateService;
    }

    public static Builder builder() {
        return new Builder();
    }

    public Kubectl kubectl() {
        return kubectl;
    }

    public K8sOptions options() {
        return options;
    }

    private String dep(String versionId) {
        return deploymentNameMapper.apply(Objects.requireNonNull(versionId, "versionId"));
    }

    private String ns() {
        return options.namespace;
    }

    @Override
    public int scale(String versionId, int desiredReplicas) throws Exception {
        String name = dep(versionId);
        if (!kubectl.available()) {
            // try REST
            ensureClient();
            client.scaleDeployment(ns(), name, desiredReplicas);
            long deadline = System.currentTimeMillis() + scaleWait.toMillis();
            int ready = 0;
            while (System.currentTimeMillis() < deadline) {
                ready = client.readyReplicas(ns(), name);
                if (desiredReplicas <= 0 || ready >= desiredReplicas) return ready;
                Thread.sleep(1500);
            }
            return ready;
        }
        return kubectl.scaleAndWait(name, desiredReplicas, ns(), scaleWait);
    }

    @Override
    public int inplaceRestart(String versionId, int podCount) throws Exception {
        String name = dep(versionId);
        if (kubectl.available()) {
            kubectl.rolloutRestart("deployment/" + name, ns());
            // best-effort wait
            try {
                kubectl.rolloutStatus("deployment/" + name, ns(), scaleWait);
            } catch (Exception ignored) {
            }
            return Math.max(podCount, kubectl.readyReplicas(name, ns()));
        }
        ensureClient();
        // delete is heavier; patch template annotation to force rollout
        Map<String, Object> patch = K8sClient.JsonMap.mapOf(
                "spec", K8sClient.JsonMap.mapOf(
                        "template", K8sClient.JsonMap.mapOf(
                                "metadata", K8sClient.JsonMap.mapOf(
                                        "annotations", K8sClient.JsonMap.mapOf(
                                                "kubectl.kubernetes.io/restartedAt",
                                                java.time.Instant.now().toString())))));
        client.patch("apps", "v1", "deployments", ns(), name, patch);
        return podCount;
    }

    @Override
    public double healthPercent(String versionId) {
        String name = dep(versionId);
        try {
            int desired;
            int ready;
            if (kubectl.available()) {
                desired = Math.max(1, kubectl.desiredReplicas(name, ns()));
                ready = kubectl.readyReplicas(name, ns());
            } else {
                ensureClient();
                Map<String, Object> dep = client.getDeployment(ns(), name);
                desired = 1;
                ready = 0;
                Object spec = dep.get("spec");
                if (spec instanceof Map<?, ?> sm && sm.get("replicas") instanceof Number n) {
                    desired = Math.max(1, n.intValue());
                }
                Object status = dep.get("status");
                if (status instanceof Map<?, ?> st && st.get("readyReplicas") instanceof Number n) {
                    ready = n.intValue();
                }
            }
            return 100.0 * ready / (double) desired;
        } catch (Exception e) {
            return 0.0;
        }
    }

    @Override
    public void setTrafficWeight(String versionId, double weightPercent) throws Exception {
        String name = dep(versionId);
        double w = Math.max(0.0, Math.min(100.0, weightPercent));
        trafficCache.put(versionId, w);
        String value = String.valueOf(w);
        Map<String, String> ann = new LinkedHashMap<>();
        ann.put(TRAFFIC_WEIGHT_ANNOTATION, value);
        if (kubectl.available()) {
            kubectl.annotate("deployment", name, ns(), ann, true);
            if (annotateService) {
                try {
                    kubectl.annotate("service", name, ns(), ann, true);
                } catch (Exception ignored) {
                }
            }
            return;
        }
        ensureClient();
        Map<String, Object> patch = K8sClient.JsonMap.mapOf(
                "metadata", K8sClient.JsonMap.mapOf(
                        "annotations", K8sClient.JsonMap.mapOf(TRAFFIC_WEIGHT_ANNOTATION, value)));
        client.patch("apps", "v1", "deployments", ns(), name, patch);
    }

    /** Last weight set via this ops instance (local cache). */
    public double trafficWeight(String versionId) {
        return trafficCache.getOrDefault(versionId, 0.0);
    }

    private synchronized void ensureClient() {
        if (client == null) {
            client = K8sClient.connect(options);
        }
    }

    @Override
    public synchronized void close() {
        if (client != null) {
            client.close();
            client = null;
        }
    }

    public static final class Builder {
        private K8sOptions options;
        private Kubectl kubectl;
        private Duration scaleWait = Duration.ofMinutes(2);
        private Function<String, String> mapper = Function.identity();
        private boolean annotateService;

        public Builder options(K8sOptions o) { this.options = o; return this; }
        public Builder kubectl(Kubectl k) { this.kubectl = k; return this; }
        public Builder scaleWait(Duration d) { this.scaleWait = d; return this; }
        public Builder deploymentNameMapper(Function<String, String> m) {
            this.mapper = m;
            return this;
        }
        public Builder annotateService(boolean v) { this.annotateService = v; return this; }

        public K8sClusterOps build() {
            return new K8sClusterOps(options, kubectl, scaleWait, mapper, annotateService);
        }
    }
}
