package org.bytedeco.pytorch.utils.k8s;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Generate a full model-serving Kubernetes manifest set from a high-level spec.
 *
 * <p>Produces Deployment + Service + optional ConfigMap / HPA / PVC, with GPU,
 * readiness/liveness, and model volume support.
 */
public final class ModelServingManifest {

    private final ModelServiceSpec spec;
    private final Manifest manifest;

    private ModelServingManifest(ModelServiceSpec spec, Manifest manifest) {
        this.spec = spec;
        this.manifest = manifest;
    }

    public static ModelServingManifest from(ModelServiceSpec spec) {
        Objects.requireNonNull(spec, "spec");
        Manifest m = new Manifest();
        String ns = spec.namespace() == null ? "default" : spec.namespace();
        String name = spec.name();

        if (spec.configData() != null && !spec.configData().isEmpty()) {
            m.add(Resources.configMap(name + "-config", ns, spec.configData()));
        }
        if (spec.modelPvcSize() != null && !spec.modelPvcSize().isBlank()) {
            m.add(Resources.persistentVolumeClaim(
                    name + "-models", ns, spec.modelPvcSize(), "ReadWriteMany"));
        }

        Resources.ContainerBuilder cb = Resources.container(name, spec.image())
                .port(spec.containerPort(), "http", "TCP")
                .env(spec.env())
                .imagePullPolicy(spec.imagePullPolicy());
        if (spec.command() != null && !spec.command().isEmpty()) {
            cb.command(spec.command().toArray(new String[0]));
        }
        if (spec.args() != null && !spec.args().isEmpty()) {
            cb.args(spec.args().toArray(new String[0]));
        }
        if (spec.cpuRequest() != null || spec.memoryRequest() != null
                || spec.cpuLimit() != null || spec.memoryLimit() != null) {
            cb.resources(spec.cpuRequest(), spec.memoryRequest(), spec.cpuLimit(), spec.memoryLimit());
        }
        if (spec.gpuCount() > 0) cb.gpu(spec.gpuCount());
        if (spec.healthPath() != null) {
            cb.httpReadiness(spec.containerPort(), spec.healthPath(), 10, 10);
            cb.httpLiveness(spec.containerPort(), spec.healthPath(), 30, 20);
        }
        if (spec.modelVolumeName() != null) {
            cb.volumeMount(spec.modelVolumeName(), spec.modelMountPath(), true);
        } else if (spec.modelPvcSize() != null) {
            cb.volumeMount("models", spec.modelMountPath(), true);
        }
        if (spec.configData() != null && !spec.configData().isEmpty()) {
            cb.volumeMount("config", "/config", true);
        }

        Resources.DeploymentBuilder db = Resources.deployment(name)
                .namespace(ns)
                .replicas(spec.replicas())
                .labels(spec.labels())
                .label("app", name)
                .label("jnitorch.model", spec.modelName() == null ? name : spec.modelName())
                .container(cb);
        if (spec.version() != null) {
            db.label("jnitorch.version", spec.version());
            db.annotation("serving.jnitorch.io/version", spec.version());
        }
        if (spec.modelVolumeName() != null && spec.modelHostPath() != null) {
            db.hostPathVolume(spec.modelVolumeName(), spec.modelHostPath());
        } else if (spec.modelPvcSize() != null) {
            db.pvcVolume("models", name + "-models");
        }
        if (spec.configData() != null && !spec.configData().isEmpty()) {
            db.configMapVolume("config", name + "-config");
        }
        if (spec.nodeSelector() != null) {
            spec.nodeSelector().forEach(db::nodeSelector);
        }

        m.add(db.build());
        m.add(Resources.service(
                name, ns, Map.of("app", name),
                spec.servicePort(), spec.containerPort(), spec.serviceType()));

        if (spec.hpaMin() > 0 && spec.hpaMax() >= spec.hpaMin()) {
            m.add(Resources.horizontalPodAutoscaler(
                    name + "-hpa", ns, name, spec.hpaMin(), spec.hpaMax(), spec.hpaCpuTarget()));
        }
        if (spec.ingressHost() != null && !spec.ingressHost().isBlank()) {
            m.add(Resources.ingress(
                    name + "-ing", ns, spec.ingressHost(), name, spec.servicePort(), "/"));
        }
        return new ModelServingManifest(spec, m);
    }

    public ModelServiceSpec spec() { return spec; }
    public Manifest toManifest() { return manifest; }
    public String toYaml() { return manifest.toYaml(); }

    /**
     * High-level model service description.
     */
    public static final class ModelServiceSpec {
        private final String name;
        private final String namespace;
        private final String image;
        private final String modelName;
        private final String version;
        private final int replicas;
        private final int containerPort;
        private final int servicePort;
        private final String serviceType;
        private final String healthPath;
        private final int gpuCount;
        private final String cpuRequest;
        private final String memoryRequest;
        private final String cpuLimit;
        private final String memoryLimit;
        private final String modelHostPath;
        private final String modelVolumeName;
        private final String modelMountPath;
        private final String modelPvcSize;
        private final Map<String, String> env;
        private final Map<String, String> labels;
        private final Map<String, String> configData;
        private final Map<String, String> nodeSelector;
        private final List<String> command;
        private final List<String> args;
        private final String imagePullPolicy;
        private final int hpaMin;
        private final int hpaMax;
        private final int hpaCpuTarget;
        private final String ingressHost;

        private ModelServiceSpec(Builder b) {
            this.name = Objects.requireNonNull(b.name, "name");
            this.namespace = b.namespace;
            this.image = Objects.requireNonNull(b.image, "image");
            this.modelName = b.modelName;
            this.version = b.version;
            this.replicas = b.replicas;
            this.containerPort = b.containerPort;
            this.servicePort = b.servicePort;
            this.serviceType = b.serviceType;
            this.healthPath = b.healthPath;
            this.gpuCount = b.gpuCount;
            this.cpuRequest = b.cpuRequest;
            this.memoryRequest = b.memoryRequest;
            this.cpuLimit = b.cpuLimit;
            this.memoryLimit = b.memoryLimit;
            this.modelHostPath = b.modelHostPath;
            this.modelVolumeName = b.modelVolumeName;
            this.modelMountPath = b.modelMountPath == null ? "/models" : b.modelMountPath;
            this.modelPvcSize = b.modelPvcSize;
            this.env = freeze(b.env);
            this.labels = freeze(b.labels);
            this.configData = freeze(b.configData);
            this.nodeSelector = freeze(b.nodeSelector);
            this.command = b.command == null ? List.of() : List.copyOf(b.command);
            this.args = b.args == null ? List.of() : List.copyOf(b.args);
            this.imagePullPolicy = b.imagePullPolicy == null ? "IfNotPresent" : b.imagePullPolicy;
            this.hpaMin = b.hpaMin;
            this.hpaMax = b.hpaMax;
            this.hpaCpuTarget = b.hpaCpuTarget;
            this.ingressHost = b.ingressHost;
        }

        public static Builder builder(String name, String image) {
            return new Builder(name, image);
        }

        public String name() { return name; }
        public String namespace() { return namespace; }
        public String image() { return image; }
        public String modelName() { return modelName; }
        public String version() { return version; }
        public int replicas() { return replicas; }
        public int containerPort() { return containerPort; }
        public int servicePort() { return servicePort; }
        public String serviceType() { return serviceType; }
        public String healthPath() { return healthPath; }
        public int gpuCount() { return gpuCount; }
        public String cpuRequest() { return cpuRequest; }
        public String memoryRequest() { return memoryRequest; }
        public String cpuLimit() { return cpuLimit; }
        public String memoryLimit() { return memoryLimit; }
        public String modelHostPath() { return modelHostPath; }
        public String modelVolumeName() { return modelVolumeName; }
        public String modelMountPath() { return modelMountPath; }
        public String modelPvcSize() { return modelPvcSize; }
        public Map<String, String> env() { return env; }
        public Map<String, String> labels() { return labels; }
        public Map<String, String> configData() { return configData; }
        public Map<String, String> nodeSelector() { return nodeSelector; }
        public List<String> command() { return command; }
        public List<String> args() { return args; }
        public String imagePullPolicy() { return imagePullPolicy; }
        public int hpaMin() { return hpaMin; }
        public int hpaMax() { return hpaMax; }
        public int hpaCpuTarget() { return hpaCpuTarget; }
        public String ingressHost() { return ingressHost; }

        private static Map<String, String> freeze(Map<String, String> m) {
            if (m == null || m.isEmpty()) return Map.of();
            return Collections.unmodifiableMap(new LinkedHashMap<>(m));
        }

        public static final class Builder {
            private final String name;
            private final String image;
            private String namespace = "default";
            private String modelName;
            private String version;
            private int replicas = 1;
            private int containerPort = 8000;
            private int servicePort = 80;
            private String serviceType = "ClusterIP";
            private String healthPath = "/health";
            private int gpuCount;
            private String cpuRequest = "500m";
            private String memoryRequest = "1Gi";
            private String cpuLimit = "4";
            private String memoryLimit = "8Gi";
            private String modelHostPath;
            private String modelVolumeName;
            private String modelMountPath = "/models";
            private String modelPvcSize;
            private final Map<String, String> env = new LinkedHashMap<>();
            private final Map<String, String> labels = new LinkedHashMap<>();
            private final Map<String, String> configData = new LinkedHashMap<>();
            private final Map<String, String> nodeSelector = new LinkedHashMap<>();
            private List<String> command;
            private List<String> args;
            private String imagePullPolicy = "IfNotPresent";
            private int hpaMin;
            private int hpaMax;
            private int hpaCpuTarget = 70;
            private String ingressHost;

            public Builder(String name, String image) {
                this.name = name;
                this.image = image;
            }

            public Builder namespace(String v) { this.namespace = v; return this; }
            public Builder modelName(String v) { this.modelName = v; return this; }
            public Builder version(String v) { this.version = v; return this; }
            public Builder replicas(int v) { this.replicas = v; return this; }
            public Builder containerPort(int v) { this.containerPort = v; return this; }
            public Builder servicePort(int v) { this.servicePort = v; return this; }
            public Builder serviceType(String v) { this.serviceType = v; return this; }
            public Builder healthPath(String v) { this.healthPath = v; return this; }
            public Builder gpuCount(int v) { this.gpuCount = v; return this; }
            public Builder resources(String cpuReq, String memReq, String cpuLim, String memLim) {
                this.cpuRequest = cpuReq;
                this.memoryRequest = memReq;
                this.cpuLimit = cpuLim;
                this.memoryLimit = memLim;
                return this;
            }
            public Builder modelHostPath(String hostPath, String volumeName) {
                this.modelHostPath = hostPath;
                this.modelVolumeName = volumeName == null ? "models" : volumeName;
                return this;
            }
            public Builder modelPvc(String size) { this.modelPvcSize = size; return this; }
            public Builder modelMountPath(String p) { this.modelMountPath = p; return this; }
            public Builder env(String k, String v) {
                if (k != null && v != null) env.put(k, v);
                return this;
            }
            public Builder env(Map<String, String> m) {
                if (m != null) env.putAll(m);
                return this;
            }
            public Builder label(String k, String v) {
                if (k != null && v != null) labels.put(k, v);
                return this;
            }
            public Builder config(String k, String v) {
                if (k != null && v != null) configData.put(k, v);
                return this;
            }
            public Builder nodeSelector(String k, String v) {
                if (k != null && v != null) nodeSelector.put(k, v);
                return this;
            }
            public Builder command(List<String> c) { this.command = c; return this; }
            public Builder args(List<String> a) { this.args = a; return this; }
            public Builder imagePullPolicy(String p) { this.imagePullPolicy = p; return this; }
            public Builder hpa(int min, int max, int cpuTarget) {
                this.hpaMin = min;
                this.hpaMax = max;
                this.hpaCpuTarget = cpuTarget;
                return this;
            }
            public Builder ingressHost(String host) { this.ingressHost = host; return this; }

            public ModelServiceSpec build() {
                return new ModelServiceSpec(this);
            }
        }
    }
}
