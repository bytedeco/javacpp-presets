package org.bytedeco.pytorch.deploy.k8s;

import java.nio.file.Path;
import java.time.Duration;
import java.util.Map;
import java.util.Objects;

/**
 * Facade for zero-dep Kubernetes access: kubectl (default) + optional apiserver REST.
 *
 * <pre>{@code
 * try (K8s k8s = K8s.connect()) {
 *     if (!k8s.kubectlAvailable()) throw new IllegalStateException("kubectl missing");
 *     ModelServingManifest.ModelServiceSpec spec =
 *         ModelServingManifest.ModelServiceSpec.builder("ranker", "my/ranker:v1")
 *             .replicas(2).gpuCount(1).build();
 *     k8s.deployModelService(spec);
 *     k8s.rolloutWait("ranker", Duration.ofMinutes(2));
 * }
 * }</pre>
 */
public final class K8s implements AutoCloseable {

    private final K8sOptions options;
    private final Kubectl kubectl;
    private K8sClient client; // lazy
    private KubeConfig kubeConfig; // lazy

    private K8s(K8sOptions options) {
        this.options = Objects.requireNonNull(options, "options");
        this.kubectl = new Kubectl(this.options);
    }

    public static K8s connect() {
        return connect(K8sOptions.defaults());
    }

    public static K8s connect(K8sOptions options) {
        return new K8s(options == null ? K8sOptions.defaults() : options);
    }

    public static K8s connect(String namespace) {
        return connect(K8sOptions.builder().fromEnv().namespace(namespace).build());
    }

    public K8sOptions options() {
        return options;
    }

    public Kubectl kubectl() {
        return kubectl;
    }

    public boolean kubectlAvailable() {
        return kubectl.available();
    }

    public synchronized KubeConfig kubeConfig() {
        if (kubeConfig == null) {
            try {
                if (options.kubeconfig != null) {
                    kubeConfig = KubeConfig.load(options.kubeconfig);
                } else {
                    kubeConfig = KubeConfig.load();
                }
            } catch (Exception e) {
                throw new K8sException("kubeconfig: " + e.getMessage(), e);
            }
        }
        return kubeConfig;
    }

    public synchronized K8sClient client() {
        if (client == null) {
            client = K8sClient.connect(options);
        }
        return client;
    }

    public void applyManifest(Manifest manifest) {
        Objects.requireNonNull(manifest, "manifest");
        if (kubectl.available()) {
            kubectl.applyStdin(manifest.toYaml());
            return;
        }
        // REST create/replace best-effort per document
        K8sClient c = client();
        for (Object doc : manifest.documents()) {
            if (!(doc instanceof Map<?, ?>)) continue;
            @SuppressWarnings("unchecked")
            Map<String, Object> m = (Map<String, Object>) doc;
            applyOne(c, m);
        }
    }

    public void applyFile(Path path) {
        if (kubectl.available()) {
            kubectl.apply(path);
            return;
        }
        try {
            applyManifest(Manifest.load(path));
        } catch (Exception e) {
            throw new K8sException("applyFile: " + e.getMessage(), e);
        }
    }

    public void deployModelService(ModelServingManifest.ModelServiceSpec spec) {
        ModelServingManifest msm = ModelServingManifest.from(spec);
        applyManifest(msm.toManifest());
    }

    public String deployModelServiceYaml(ModelServingManifest.ModelServiceSpec spec) {
        return ModelServingManifest.from(spec).toYaml();
    }

    public void rolloutWait(String deploymentName, Duration timeout) {
        kubectl.rolloutStatus("deployment/" + deploymentName, options.namespace, timeout);
    }

    public void scale(String deploymentName, int replicas) {
        kubectl.scaleDeployment(deploymentName, replicas, options.namespace);
    }

    public K8sClusterOps clusterOps() {
        return new K8sClusterOps(options, kubectl, Duration.ofMinutes(2), s -> s, false);
    }

    @SuppressWarnings("unchecked")
    private void applyOne(K8sClient c, Map<String, Object> doc) {
        String apiVersion = String.valueOf(doc.getOrDefault("apiVersion", "v1"));
        String kind = String.valueOf(doc.get("kind"));
        Map<String, Object> meta = doc.get("metadata") instanceof Map<?, ?> m
                ? (Map<String, Object>) m : Map.of();
        String name = String.valueOf(meta.get("name"));
        String ns = meta.get("namespace") == null ? options.namespace : String.valueOf(meta.get("namespace"));
        String group;
        String version;
        if (apiVersion.contains("/")) {
            String[] sp = apiVersion.split("/", 2);
            group = sp[0];
            version = sp[1];
        } else {
            group = "";
            version = apiVersion;
        }
        String plural = kindToPlural(kind);
        try {
            c.get(group, version, plural, namespaced(kind) ? ns : null, name);
            // exists → replace
            c.replace(group, version, plural, namespaced(kind) ? ns : null, name, doc);
        } catch (K8sException e) {
            if (e.httpStatus() == 404) {
                c.create(group, version, plural, namespaced(kind) ? ns : null, doc);
            } else {
                throw e;
            }
        }
    }

    private static boolean namespaced(String kind) {
        if (kind == null) return true;
        return switch (kind) {
            case "Namespace", "PersistentVolume", "ClusterRole", "ClusterRoleBinding",
                 "StorageClass", "CustomResourceDefinition", "Node" -> false;
            default -> true;
        };
    }

    private static String kindToPlural(String kind) {
        if (kind == null || kind.isBlank()) return "unknowns";
        // very small heuristic
        if (kind.endsWith("s")) return kind.toLowerCase() + "es";
        if (kind.endsWith("y")) return kind.substring(0, kind.length() - 1).toLowerCase() + "ies";
        return switch (kind) {
            case "NetworkPolicy" -> "networkpolicies";
            case "Ingress" -> "ingresses";
            case "Endpoints" -> "endpoints";
            default -> kind.toLowerCase() + "s";
        };
    }

    @Override
    public synchronized void close() {
        if (client != null) {
            client.close();
            client = null;
        }
    }
}
