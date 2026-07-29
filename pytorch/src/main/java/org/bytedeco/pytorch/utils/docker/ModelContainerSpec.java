package org.bytedeco.pytorch.utils.docker;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Model-service oriented container / compose service specification.
 *
 * <p>Bridges training/registry artifacts to runnable Docker units: image, model mount,
 * inference port, GPU, health check, resource limits.
 */
public final class ModelContainerSpec {

    private final String serviceName;
    private final String image;
    private final String containerName;
    private final int hostPort;
    private final int containerPort;
    private final String modelHostPath;
    private final String modelContainerPath;
    private final String modelVolumeName;
    private final Map<String, String> env;
    private final List<String> command;
    private final List<String> entrypoint;
    private final int gpuCount; // 0 = none, -1 = all
    private final String cpus;
    private final String memory;
    private final String shmSize;
    private final String healthCmd;
    private final String healthInterval;
    private final String restart;
    private final Map<String, String> labels;
    private final List<String> extraVolumes;
    private final String workdir;

    private ModelContainerSpec(Builder b) {
        this.serviceName = Objects.requireNonNull(b.serviceName, "serviceName");
        this.image = Objects.requireNonNull(b.image, "image");
        this.containerName = b.containerName;
        this.hostPort = b.hostPort;
        this.containerPort = b.containerPort;
        this.modelHostPath = b.modelHostPath;
        this.modelContainerPath = b.modelContainerPath == null ? "/models" : b.modelContainerPath;
        this.modelVolumeName = b.modelVolumeName;
        this.env = freezeMap(b.env);
        this.command = freezeList(b.command);
        this.entrypoint = freezeList(b.entrypoint);
        this.gpuCount = b.gpuCount;
        this.cpus = b.cpus;
        this.memory = b.memory;
        this.shmSize = b.shmSize;
        this.healthCmd = b.healthCmd;
        this.healthInterval = b.healthInterval == null ? "10s" : b.healthInterval;
        this.restart = b.restart == null ? "unless-stopped" : b.restart;
        this.labels = freezeMap(b.labels);
        this.extraVolumes = freezeList(b.extraVolumes);
        this.workdir = b.workdir;
    }

    public static Builder builder(String serviceName, String image) {
        return new Builder(serviceName, image);
    }

    public String serviceName() { return serviceName; }
    public String image() { return image; }
    public String containerName() { return containerName; }
    public int hostPort() { return hostPort; }
    public int containerPort() { return containerPort; }
    public String modelHostPath() { return modelHostPath; }
    public String modelContainerPath() { return modelContainerPath; }
    public String modelVolumeName() { return modelVolumeName; }
    public Map<String, String> env() { return env; }
    public List<String> command() { return command; }
    public int gpuCount() { return gpuCount; }

    /** Convert to {@link DockerModels.RunSpec} for plain {@code docker run}. */
    public DockerModels.RunSpec toRunSpec() {
        DockerModels.RunSpec.Builder b = DockerModels.RunSpec.builder(image)
                .detach(true)
                .restart(restart)
                .env(env);
        if (containerName != null) b.name(containerName);
        else b.name(serviceName);
        if (hostPort > 0 && containerPort > 0) b.publish(hostPort, containerPort);
        if (modelHostPath != null && !modelHostPath.isBlank()) {
            b.volume(modelHostPath + ":" + modelContainerPath);
        } else if (modelVolumeName != null && !modelVolumeName.isBlank()) {
            b.volume(modelVolumeName + ":" + modelContainerPath);
        }
        for (String v : extraVolumes) b.volume(v);
        if (gpuCount != 0) {
            b.gpus(gpuCount < 0 ? "all" : ("device=" + joinGpuDevices(gpuCount)));
            b.env("NVIDIA_VISIBLE_DEVICES", gpuCount < 0 ? "all" : joinGpuDevices(gpuCount));
        }
        if (cpus != null) b.cpus(cpus);
        if (memory != null) b.memory(memory);
        if (shmSize != null) b.shmSize(shmSize);
        if (workdir != null) b.workdir(workdir);
        if (healthCmd != null) {
            b.healthCmd(healthCmd).healthInterval(healthInterval).healthRetries(3);
        }
        for (Map.Entry<String, String> e : labels.entrySet()) b.label(e.getKey(), e.getValue());
        b.label("jnitorch.service", serviceName);
        if (!command.isEmpty()) b.command(command);
        if (!entrypoint.isEmpty()) b.entrypoint(entrypoint.toArray(new String[0]));
        return b.build();
    }

    /** Convert to a compose service map. */
    public Map<String, Object> toComposeService() {
        ComposeFile.ServiceBuilder sb = new ComposeFile().serviceBuilder(serviceName).image(image);
        if (containerName != null) sb.containerName(containerName);
        sb.restart(restart);
        if (hostPort > 0 && containerPort > 0) sb.port(hostPort, containerPort);
        sb.env(env);
        if (modelHostPath != null && !modelHostPath.isBlank()) {
            sb.volume(modelHostPath + ":" + modelContainerPath);
        } else if (modelVolumeName != null && !modelVolumeName.isBlank()) {
            sb.volume(modelVolumeName + ":" + modelContainerPath);
        }
        for (String v : extraVolumes) sb.volume(v);
        if (gpuCount != 0) sb.gpus(gpuCount < 0 ? 0 : gpuCount);
        if (cpus != null) {
            try { sb.cpus(Double.parseDouble(cpus)); } catch (NumberFormatException ignored) {}
        }
        if (memory != null) sb.memory(memory);
        if (shmSize != null) sb.shmSize(shmSize);
        if (healthCmd != null) sb.healthcheck(healthCmd, healthInterval, "5s", 3);
        for (Map.Entry<String, String> e : labels.entrySet()) sb.label(e.getKey(), e.getValue());
        sb.label("jnitorch.service", serviceName);
        if (!command.isEmpty()) sb.command(command.toArray(new String[0]));
        if (!entrypoint.isEmpty()) sb.entrypoint(entrypoint.toArray(new String[0]));
        if (workdir != null) sb.raw("working_dir", workdir);
        return sb.buildSpec();
    }

    private static String joinGpuDevices(int count) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < count; i++) {
            if (i > 0) sb.append(',');
            sb.append(i);
        }
        return sb.toString();
    }

    private static List<String> freezeList(List<String> in) {
        if (in == null || in.isEmpty()) return List.of();
        return Collections.unmodifiableList(new ArrayList<>(in));
    }

    private static Map<String, String> freezeMap(Map<String, String> in) {
        if (in == null || in.isEmpty()) return Map.of();
        return Collections.unmodifiableMap(new LinkedHashMap<>(in));
    }

    public static final class Builder {
        private final String serviceName;
        private final String image;
        private String containerName;
        private int hostPort = 8000;
        private int containerPort = 8000;
        private String modelHostPath;
        private String modelContainerPath = "/models";
        private String modelVolumeName;
        private final Map<String, String> env = new LinkedHashMap<>();
        private final List<String> command = new ArrayList<>();
        private final List<String> entrypoint = new ArrayList<>();
        private int gpuCount;
        private String cpus;
        private String memory;
        private String shmSize = "1g";
        private String healthCmd;
        private String healthInterval = "10s";
        private String restart = "unless-stopped";
        private final Map<String, String> labels = new LinkedHashMap<>();
        private final List<String> extraVolumes = new ArrayList<>();
        private String workdir;

        public Builder(String serviceName, String image) {
            this.serviceName = serviceName;
            this.image = image;
        }

        public Builder containerName(String v) { this.containerName = v; return this; }
        public Builder ports(int host, int container) {
            this.hostPort = host;
            this.containerPort = container;
            return this;
        }
        public Builder modelHostPath(String hostPath) { this.modelHostPath = hostPath; return this; }
        public Builder modelContainerPath(String path) { this.modelContainerPath = path; return this; }
        public Builder modelVolumeName(String name) { this.modelVolumeName = name; return this; }
        public Builder env(String k, String v) {
            if (k != null && v != null) env.put(k, v);
            return this;
        }
        public Builder env(Map<String, String> m) {
            if (m != null) env.putAll(m);
            return this;
        }
        public Builder command(String... cmd) {
            command.clear();
            if (cmd != null) for (String c : cmd) if (c != null) command.add(c);
            return this;
        }
        public Builder entrypoint(String... ep) {
            entrypoint.clear();
            if (ep != null) for (String c : ep) if (c != null) entrypoint.add(c);
            return this;
        }
        /** 0 = none, negative = all GPUs, positive = count. */
        public Builder gpus(int count) { this.gpuCount = count; return this; }
        public Builder gpusAll() { this.gpuCount = -1; return this; }
        public Builder cpus(String v) { this.cpus = v; return this; }
        public Builder memory(String v) { this.memory = v; return this; }
        public Builder shmSize(String v) { this.shmSize = v; return this; }
        public Builder healthHttp(int port, String path) {
            String p = path == null ? "/health" : path;
            this.healthCmd = "curl -f http://127.0.0.1:" + port + p + " || exit 1";
            return this;
        }
        public Builder healthCmd(String cmd) { this.healthCmd = cmd; return this; }
        public Builder healthInterval(String v) { this.healthInterval = v; return this; }
        public Builder restart(String v) { this.restart = v; return this; }
        public Builder label(String k, String v) {
            if (k != null && v != null) labels.put(k, v);
            return this;
        }
        public Builder volume(String spec) {
            if (spec != null) extraVolumes.add(spec);
            return this;
        }
        public Builder workdir(String v) { this.workdir = v; return this; }
        /** Convenience: MODEL_NAME + MODEL_PATH env for common serving stacks. */
        public Builder modelMeta(String modelName, String modelVersion) {
            if (modelName != null) env.put("MODEL_NAME", modelName);
            if (modelVersion != null) env.put("MODEL_VERSION", modelVersion);
            env.put("MODEL_PATH", modelContainerPath);
            return this;
        }

        public ModelContainerSpec build() {
            return new ModelContainerSpec(this);
        }
    }
}
