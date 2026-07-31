package org.bytedeco.pytorch.deploy.k8s;

import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Base64;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Builders that emit Kubernetes resource maps (JSON/YAML-serializable).
 *
 * <p>No codegen POJOs — plain {@link LinkedHashMap} trees matching apps/v1 &amp; core/v1.
 */
public final class Resources {

    private Resources() {}

    public static Map<String, Object> metadata(String name, String namespace, Map<String, String> labels) {
        LinkedHashMap<String, Object> meta = new LinkedHashMap<>();
        meta.put("name", Objects.requireNonNull(name, "name"));
        if (namespace != null && !namespace.isBlank()) meta.put("namespace", namespace);
        if (labels != null && !labels.isEmpty()) meta.put("labels", new LinkedHashMap<>(labels));
        return meta;
    }

    public static Map<String, Object> metadata(
            String name, String namespace, Map<String, String> labels, Map<String, String> annotations) {
        Map<String, Object> meta = metadata(name, namespace, labels);
        if (annotations != null && !annotations.isEmpty()) {
            meta.put("annotations", new LinkedHashMap<>(annotations));
        }
        return meta;
    }

    // ---- Deployment ----

    public static final class DeploymentBuilder {
        private String name;
        private String namespace = "default";
        private int replicas = 1;
        private final Map<String, String> labels = new LinkedHashMap<>();
        private final Map<String, String> selector = new LinkedHashMap<>();
        private final Map<String, String> annotations = new LinkedHashMap<>();
        private final List<Map<String, Object>> containers = new ArrayList<>();
        private final List<Map<String, Object>> volumes = new ArrayList<>();
        private String serviceAccountName;
        private int terminationGracePeriodSeconds = 30;
        private final Map<String, String> nodeSelector = new LinkedHashMap<>();
        private String restartPolicy = "Always";
        private String strategyType = "RollingUpdate";
        private Integer maxUnavailable;
        private Integer maxSurge;

        public DeploymentBuilder name(String n) { this.name = n; return this; }
        public DeploymentBuilder namespace(String ns) { this.namespace = ns; return this; }
        public DeploymentBuilder replicas(int r) { this.replicas = r; return this; }
        public DeploymentBuilder label(String k, String v) {
            labels.put(k, v);
            selector.putIfAbsent(k, v);
            return this;
        }
        public DeploymentBuilder labels(Map<String, String> m) {
            if (m != null) {
                labels.putAll(m);
                m.forEach(selector::putIfAbsent);
            }
            return this;
        }
        public DeploymentBuilder selector(String k, String v) {
            selector.put(k, v);
            return this;
        }
        public DeploymentBuilder annotation(String k, String v) {
            annotations.put(k, v);
            return this;
        }
        public DeploymentBuilder container(Map<String, Object> c) {
            containers.add(c);
            return this;
        }
        public DeploymentBuilder container(ContainerBuilder cb) {
            containers.add(cb.build());
            return this;
        }
        public DeploymentBuilder volume(Map<String, Object> v) {
            volumes.add(v);
            return this;
        }
        public DeploymentBuilder emptyDirVolume(String name) {
            LinkedHashMap<String, Object> v = new LinkedHashMap<>();
            v.put("name", name);
            v.put("emptyDir", Map.of());
            volumes.add(v);
            return this;
        }
        public DeploymentBuilder hostPathVolume(String name, String path) {
            LinkedHashMap<String, Object> v = new LinkedHashMap<>();
            v.put("name", name);
            v.put("hostPath", Map.of("path", path));
            volumes.add(v);
            return this;
        }
        public DeploymentBuilder pvcVolume(String name, String claimName) {
            LinkedHashMap<String, Object> v = new LinkedHashMap<>();
            v.put("name", name);
            v.put("persistentVolumeClaim", Map.of("claimName", claimName));
            volumes.add(v);
            return this;
        }
        public DeploymentBuilder configMapVolume(String name, String configMapName) {
            LinkedHashMap<String, Object> v = new LinkedHashMap<>();
            v.put("name", name);
            v.put("configMap", Map.of("name", configMapName));
            volumes.add(v);
            return this;
        }
        public DeploymentBuilder serviceAccountName(String s) { this.serviceAccountName = s; return this; }
        public DeploymentBuilder nodeSelector(String k, String v) {
            nodeSelector.put(k, v);
            return this;
        }
        public DeploymentBuilder strategyRolling(Integer maxUnavailable, Integer maxSurge) {
            this.strategyType = "RollingUpdate";
            this.maxUnavailable = maxUnavailable;
            this.maxSurge = maxSurge;
            return this;
        }
        public DeploymentBuilder strategyRecreate() {
            this.strategyType = "Recreate";
            return this;
        }

        public Map<String, Object> build() {
            Objects.requireNonNull(name, "name");
            if (labels.isEmpty()) {
                labels.put("app", name);
                selector.putIfAbsent("app", name);
            }
            LinkedHashMap<String, Object> root = new LinkedHashMap<>();
            root.put("apiVersion", "apps/v1");
            root.put("kind", "Deployment");
            root.put("metadata", metadata(name, namespace, labels, annotations));

            LinkedHashMap<String, Object> spec = new LinkedHashMap<>();
            spec.put("replicas", replicas);
            spec.put("selector", Map.of("matchLabels", new LinkedHashMap<>(selector)));

            LinkedHashMap<String, Object> strategy = new LinkedHashMap<>();
            strategy.put("type", strategyType);
            if ("RollingUpdate".equals(strategyType)
                    && (maxUnavailable != null || maxSurge != null)) {
                LinkedHashMap<String, Object> ru = new LinkedHashMap<>();
                if (maxUnavailable != null) ru.put("maxUnavailable", maxUnavailable);
                if (maxSurge != null) ru.put("maxSurge", maxSurge);
                strategy.put("rollingUpdate", ru);
            }
            spec.put("strategy", strategy);

            LinkedHashMap<String, Object> podSpec = new LinkedHashMap<>();
            podSpec.put("containers", containers);
            if (!volumes.isEmpty()) podSpec.put("volumes", volumes);
            if (serviceAccountName != null) podSpec.put("serviceAccountName", serviceAccountName);
            podSpec.put("restartPolicy", restartPolicy);
            podSpec.put("terminationGracePeriodSeconds", terminationGracePeriodSeconds);
            if (!nodeSelector.isEmpty()) podSpec.put("nodeSelector", new LinkedHashMap<>(nodeSelector));

            LinkedHashMap<String, Object> podTemplate = new LinkedHashMap<>();
            LinkedHashMap<String, Object> podMeta = new LinkedHashMap<>();
            podMeta.put("labels", new LinkedHashMap<>(labels));
            if (!annotations.isEmpty()) podMeta.put("annotations", new LinkedHashMap<>(annotations));
            podTemplate.put("metadata", podMeta);
            podTemplate.put("spec", podSpec);

            spec.put("template", podTemplate);
            root.put("spec", spec);
            return root;
        }
    }

    public static DeploymentBuilder deployment(String name) {
        return new DeploymentBuilder().name(name).label("app", name);
    }

    // ---- Container ----

    public static final class ContainerBuilder {
        private String name = "app";
        private String image;
        private final List<Object> command = new ArrayList<>();
        private final List<Object> args = new ArrayList<>();
        private final List<Map<String, Object>> ports = new ArrayList<>();
        private final List<Map<String, Object>> env = new ArrayList<>();
        private final List<Map<String, Object>> volumeMounts = new ArrayList<>();
        private final Map<String, Object> resources = new LinkedHashMap<>();
        private Map<String, Object> readinessProbe;
        private Map<String, Object> livenessProbe;
        private String imagePullPolicy = "IfNotPresent";
        private String workingDir;

        public ContainerBuilder name(String n) { this.name = n; return this; }
        public ContainerBuilder image(String img) { this.image = img; return this; }
        public ContainerBuilder command(String... cmd) {
            command.clear();
            if (cmd != null) for (String c : cmd) command.add(c);
            return this;
        }
        public ContainerBuilder args(String... a) {
            args.clear();
            if (a != null) for (String c : a) args.add(c);
            return this;
        }
        public ContainerBuilder port(int containerPort) {
            return port(containerPort, "http", "TCP");
        }
        public ContainerBuilder port(int containerPort, String name, String protocol) {
            LinkedHashMap<String, Object> p = new LinkedHashMap<>();
            p.put("containerPort", containerPort);
            if (name != null) p.put("name", name);
            if (protocol != null) p.put("protocol", protocol);
            ports.add(p);
            return this;
        }
        public ContainerBuilder env(String key, String value) {
            LinkedHashMap<String, Object> e = new LinkedHashMap<>();
            e.put("name", key);
            e.put("value", value);
            env.add(e);
            return this;
        }
        public ContainerBuilder env(Map<String, String> map) {
            if (map != null) map.forEach(this::env);
            return this;
        }
        public ContainerBuilder envFromSecret(String envName, String secretName, String secretKey) {
            LinkedHashMap<String, Object> e = new LinkedHashMap<>();
            e.put("name", envName);
            e.put("valueFrom", Map.of("secretKeyRef", Map.of("name", secretName, "key", secretKey)));
            env.add(e);
            return this;
        }
        public ContainerBuilder volumeMount(String name, String mountPath, boolean readOnly) {
            LinkedHashMap<String, Object> m = new LinkedHashMap<>();
            m.put("name", name);
            m.put("mountPath", mountPath);
            if (readOnly) m.put("readOnly", true);
            volumeMounts.add(m);
            return this;
        }
        public ContainerBuilder resources(String cpuReq, String memReq, String cpuLim, String memLim) {
            // Merge into existing maps so a prior gpu() is not wiped.
            @SuppressWarnings("unchecked")
            Map<String, Object> req = (Map<String, Object>) resources.computeIfAbsent(
                    "requests", k -> new LinkedHashMap<>());
            @SuppressWarnings("unchecked")
            Map<String, Object> lim = (Map<String, Object>) resources.computeIfAbsent(
                    "limits", k -> new LinkedHashMap<>());
            if (cpuReq != null) req.put("cpu", cpuReq);
            if (memReq != null) req.put("memory", memReq);
            if (cpuLim != null) lim.put("cpu", cpuLim);
            if (memLim != null) lim.put("memory", memLim);
            if (req.isEmpty()) resources.remove("requests");
            if (lim.isEmpty()) resources.remove("limits");
            return this;
        }
        public ContainerBuilder gpu(int count) {
            @SuppressWarnings("unchecked")
            Map<String, Object> lim = (Map<String, Object>) resources.computeIfAbsent(
                    "limits", k -> new LinkedHashMap<>());
            lim.put("nvidia.com/gpu", count);
            return this;
        }
        public ContainerBuilder httpReadiness(int port, String path, int initialDelay, int period) {
            readinessProbe = httpProbe(port, path, initialDelay, period);
            return this;
        }
        public ContainerBuilder httpLiveness(int port, String path, int initialDelay, int period) {
            livenessProbe = httpProbe(port, path, initialDelay, period);
            return this;
        }
        public ContainerBuilder imagePullPolicy(String p) { this.imagePullPolicy = p; return this; }
        public ContainerBuilder workingDir(String d) { this.workingDir = d; return this; }

        private static Map<String, Object> httpProbe(int port, String path, int initialDelay, int period) {
            LinkedHashMap<String, Object> probe = new LinkedHashMap<>();
            probe.put("httpGet", Map.of("path", path == null ? "/health" : path, "port", port));
            probe.put("initialDelaySeconds", initialDelay);
            probe.put("periodSeconds", period);
            probe.put("timeoutSeconds", 5);
            probe.put("failureThreshold", 3);
            return probe;
        }

        public Map<String, Object> build() {
            Objects.requireNonNull(image, "image");
            LinkedHashMap<String, Object> c = new LinkedHashMap<>();
            c.put("name", name);
            c.put("image", image);
            if (imagePullPolicy != null) c.put("imagePullPolicy", imagePullPolicy);
            if (!command.isEmpty()) c.put("command", new ArrayList<>(command));
            if (!args.isEmpty()) c.put("args", new ArrayList<>(args));
            if (!ports.isEmpty()) c.put("ports", ports);
            if (!env.isEmpty()) c.put("env", env);
            if (!volumeMounts.isEmpty()) c.put("volumeMounts", volumeMounts);
            if (!resources.isEmpty()) c.put("resources", resources);
            if (readinessProbe != null) c.put("readinessProbe", readinessProbe);
            if (livenessProbe != null) c.put("livenessProbe", livenessProbe);
            if (workingDir != null) c.put("workingDir", workingDir);
            return c;
        }
    }

    public static ContainerBuilder container(String name, String image) {
        return new ContainerBuilder().name(name).image(image);
    }

    // ---- Service ----

    public static Map<String, Object> service(
            String name, String namespace, Map<String, String> selector,
            int port, int targetPort, String type) {
        LinkedHashMap<String, Object> root = new LinkedHashMap<>();
        root.put("apiVersion", "v1");
        root.put("kind", "Service");
        Map<String, String> labels = new LinkedHashMap<>();
        if (selector != null) labels.putAll(selector);
        labels.putIfAbsent("app", name);
        root.put("metadata", metadata(name, namespace, labels));
        LinkedHashMap<String, Object> spec = new LinkedHashMap<>();
        spec.put("selector", selector == null ? Map.of("app", name) : new LinkedHashMap<>(selector));
        LinkedHashMap<String, Object> p = new LinkedHashMap<>();
        p.put("name", "http");
        p.put("port", port);
        p.put("targetPort", targetPort);
        p.put("protocol", "TCP");
        spec.put("ports", List.of(p));
        spec.put("type", type == null ? "ClusterIP" : type);
        root.put("spec", spec);
        return root;
    }

    // ---- ConfigMap / Secret ----

    public static Map<String, Object> configMap(
            String name, String namespace, Map<String, String> data) {
        LinkedHashMap<String, Object> root = new LinkedHashMap<>();
        root.put("apiVersion", "v1");
        root.put("kind", "ConfigMap");
        root.put("metadata", metadata(name, namespace, Map.of("app", name)));
        root.put("data", data == null ? Map.of() : new LinkedHashMap<>(data));
        return root;
    }

    public static Map<String, Object> secretOpaque(
            String name, String namespace, Map<String, String> stringData) {
        LinkedHashMap<String, Object> root = new LinkedHashMap<>();
        root.put("apiVersion", "v1");
        root.put("kind", "Secret");
        root.put("metadata", metadata(name, namespace, Map.of("app", name)));
        root.put("type", "Opaque");
        LinkedHashMap<String, Object> data = new LinkedHashMap<>();
        if (stringData != null) {
            for (Map.Entry<String, String> e : stringData.entrySet()) {
                String v = e.getValue() == null ? "" : e.getValue();
                data.put(e.getKey(), Base64.getEncoder().encodeToString(v.getBytes(StandardCharsets.UTF_8)));
            }
        }
        root.put("data", data);
        return root;
    }

    // ---- HPA ----

    public static Map<String, Object> horizontalPodAutoscaler(
            String name, String namespace, String deploymentName,
            int minReplicas, int maxReplicas, int targetCpuUtilization) {
        LinkedHashMap<String, Object> root = new LinkedHashMap<>();
        root.put("apiVersion", "autoscaling/v2");
        root.put("kind", "HorizontalPodAutoscaler");
        root.put("metadata", metadata(name, namespace, Map.of("app", deploymentName)));
        LinkedHashMap<String, Object> spec = new LinkedHashMap<>();
        spec.put("scaleTargetRef", Map.of(
                "apiVersion", "apps/v1",
                "kind", "Deployment",
                "name", deploymentName));
        spec.put("minReplicas", minReplicas);
        spec.put("maxReplicas", maxReplicas);
        LinkedHashMap<String, Object> metric = new LinkedHashMap<>();
        metric.put("type", "Resource");
        metric.put("resource", Map.of(
                "name", "cpu",
                "target", Map.of(
                        "type", "Utilization",
                        "averageUtilization", targetCpuUtilization)));
        spec.put("metrics", List.of(metric));
        root.put("spec", spec);
        return root;
    }

    // ---- SA / PVC / Ingress (minimal) ----

    public static Map<String, Object> serviceAccount(String name, String namespace) {
        LinkedHashMap<String, Object> root = new LinkedHashMap<>();
        root.put("apiVersion", "v1");
        root.put("kind", "ServiceAccount");
        root.put("metadata", metadata(name, namespace, Map.of("app", name)));
        return root;
    }

    public static Map<String, Object> persistentVolumeClaim(
            String name, String namespace, String storage, String accessMode) {
        LinkedHashMap<String, Object> root = new LinkedHashMap<>();
        root.put("apiVersion", "v1");
        root.put("kind", "PersistentVolumeClaim");
        root.put("metadata", metadata(name, namespace, Map.of("app", name)));
        LinkedHashMap<String, Object> spec = new LinkedHashMap<>();
        spec.put("accessModes", List.of(accessMode == null ? "ReadWriteOnce" : accessMode));
        spec.put("resources", Map.of("requests", Map.of("storage", storage == null ? "10Gi" : storage)));
        root.put("spec", spec);
        return root;
    }

    public static Map<String, Object> ingress(
            String name, String namespace, String host, String serviceName, int servicePort, String path) {
        LinkedHashMap<String, Object> root = new LinkedHashMap<>();
        root.put("apiVersion", "networking.k8s.io/v1");
        root.put("kind", "Ingress");
        root.put("metadata", metadata(name, namespace, Map.of("app", serviceName)));
        LinkedHashMap<String, Object> rule = new LinkedHashMap<>();
        if (host != null) rule.put("host", host);
        LinkedHashMap<String, Object> httpPath = new LinkedHashMap<>();
        httpPath.put("path", path == null ? "/" : path);
        httpPath.put("pathType", "Prefix");
        httpPath.put("backend", Map.of(
                "service", Map.of(
                        "name", serviceName,
                        "port", Map.of("number", servicePort))));
        rule.put("http", Map.of("paths", List.of(httpPath)));
        root.put("spec", Map.of("rules", List.of(rule)));
        return root;
    }
}
