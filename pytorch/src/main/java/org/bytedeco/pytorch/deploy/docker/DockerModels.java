package org.bytedeco.pytorch.deploy.docker;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Immutable DTOs for Docker containers / images / run specs.
 */
public final class DockerModels {

    private DockerModels() {}

    public static final class PortBinding {
        public final String hostIp;
        public final int hostPort;
        public final int containerPort;
        public final String protocol; // tcp / udp

        public PortBinding(int hostPort, int containerPort) {
            this("", hostPort, containerPort, "tcp");
        }

        public PortBinding(String hostIp, int hostPort, int containerPort, String protocol) {
            this.hostIp = hostIp == null ? "" : hostIp;
            this.hostPort = hostPort;
            this.containerPort = containerPort;
            this.protocol = protocol == null || protocol.isBlank() ? "tcp" : protocol;
        }

        /** Parse {@code 8080:80}, {@code 127.0.0.1:8080:80/udp}, {@code 80}. */
        public static PortBinding parse(String spec) {
            if (spec == null || spec.isBlank()) {
                throw new IllegalArgumentException("empty port spec");
            }
            String s = spec.trim();
            String protocol = "tcp";
            int slash = s.lastIndexOf('/');
            if (slash > 0) {
                protocol = s.substring(slash + 1);
                s = s.substring(0, slash);
            }
            String[] parts = s.split(":");
            if (parts.length == 1) {
                int p = Integer.parseInt(parts[0]);
                return new PortBinding("", p, p, protocol);
            }
            if (parts.length == 2) {
                return new PortBinding("", Integer.parseInt(parts[0]), Integer.parseInt(parts[1]), protocol);
            }
            if (parts.length == 3) {
                return new PortBinding(parts[0], Integer.parseInt(parts[1]), Integer.parseInt(parts[2]), protocol);
            }
            throw new IllegalArgumentException("bad port spec: " + spec);
        }

        public String toPublishFlag() {
            StringBuilder sb = new StringBuilder();
            if (!hostIp.isEmpty()) sb.append(hostIp).append(':');
            sb.append(hostPort).append(':').append(containerPort);
            if (!"tcp".equalsIgnoreCase(protocol)) sb.append('/').append(protocol);
            return sb.toString();
        }

        @Override
        public String toString() { return toPublishFlag(); }
    }

    public static final class ContainerInfo {
        public final String id;
        public final String name;
        public final String image;
        public final String status;
        public final String state;
        public final List<String> ports;
        public final Map<String, String> labels;

        public ContainerInfo(
                String id, String name, String image, String status, String state,
                List<String> ports, Map<String, String> labels) {
            this.id = id == null ? "" : id;
            this.name = name == null ? "" : name;
            this.image = image == null ? "" : image;
            this.status = status == null ? "" : status;
            this.state = state == null ? "" : state;
            this.ports = ports == null ? List.of() : Collections.unmodifiableList(new ArrayList<>(ports));
            this.labels = labels == null
                    ? Map.of()
                    : Collections.unmodifiableMap(new LinkedHashMap<>(labels));
        }

        public boolean running() {
            return "running".equalsIgnoreCase(state)
                    || status.toLowerCase().startsWith("up");
        }

        @Override
        public String toString() {
            return "ContainerInfo{id=" + shortId() + ", name=" + name + ", image=" + image
                    + ", state=" + state + "}";
        }

        public String shortId() {
            return id.length() > 12 ? id.substring(0, 12) : id;
        }
    }

    public static final class ImageInfo {
        public final String id;
        public final String repository;
        public final String tag;
        public final long sizeBytes;

        public ImageInfo(String id, String repository, String tag, long sizeBytes) {
            this.id = id == null ? "" : id;
            this.repository = repository == null ? "" : repository;
            this.tag = tag == null ? "" : tag;
            this.sizeBytes = sizeBytes;
        }

        public String ref() {
            if (repository.isEmpty()) return id;
            if (tag.isEmpty() || "<none>".equals(tag)) return repository;
            return repository + ":" + tag;
        }

        @Override
        public String toString() {
            return "ImageInfo{" + ref() + ", size=" + sizeBytes + "}";
        }
    }

    public static final class HealthStatus {
        public final String status; // starting / healthy / unhealthy / none
        public final int failingStreak;
        public final String log;

        public HealthStatus(String status, int failingStreak, String log) {
            this.status = status == null ? "none" : status;
            this.failingStreak = failingStreak;
            this.log = log == null ? "" : log;
        }

        public boolean healthy() {
            return "healthy".equalsIgnoreCase(status) || "none".equalsIgnoreCase(status);
        }
    }

    /**
     * Spec for {@code docker run}.
     */
    public static final class RunSpec {
        public final String image;
        public final String name;
        public final boolean detach;
        public final boolean remove;
        public final boolean tty;
        public final boolean interactive;
        public final List<String> command;
        public final List<String> entrypoint;
        public final Map<String, String> env;
        public final List<String> envFiles;
        public final List<PortBinding> ports;
        public final List<String> volumes;
        public final List<String> networks;
        public final Map<String, String> labels;
        public final String restart;
        public final String networkMode;
        public final String ipc;
        public final String pid;
        public final boolean privileged;
        public final List<String> devices;
        public final List<String> gpus; // e.g. "all", "device=0"
        public final String cpus;
        public final String memory;
        public final String shmSize;
        public final String workdir;
        public final String user;
        public final String hostname;
        public final Map<String, String> sysctls;
        public final List<String> extraHosts;
        public final String healthCmd;
        public final String healthInterval;
        public final String healthTimeout;
        public final int healthRetries;
        public final String pull; // missing / always / never
        public final List<String> rawArgs;

        private RunSpec(Builder b) {
            this.image = Objects.requireNonNull(b.image, "image");
            this.name = b.name;
            this.detach = b.detach;
            this.remove = b.remove;
            this.tty = b.tty;
            this.interactive = b.interactive;
            this.command = freezeList(b.command);
            this.entrypoint = freezeList(b.entrypoint);
            this.env = freezeMap(b.env);
            this.envFiles = freezeList(b.envFiles);
            this.ports = b.ports == null ? List.of() : Collections.unmodifiableList(new ArrayList<>(b.ports));
            this.volumes = freezeList(b.volumes);
            this.networks = freezeList(b.networks);
            this.labels = freezeMap(b.labels);
            this.restart = b.restart;
            this.networkMode = b.networkMode;
            this.ipc = b.ipc;
            this.pid = b.pid;
            this.privileged = b.privileged;
            this.devices = freezeList(b.devices);
            this.gpus = freezeList(b.gpus);
            this.cpus = b.cpus;
            this.memory = b.memory;
            this.shmSize = b.shmSize;
            this.workdir = b.workdir;
            this.user = b.user;
            this.hostname = b.hostname;
            this.sysctls = freezeMap(b.sysctls);
            this.extraHosts = freezeList(b.extraHosts);
            this.healthCmd = b.healthCmd;
            this.healthInterval = b.healthInterval;
            this.healthTimeout = b.healthTimeout;
            this.healthRetries = b.healthRetries;
            this.pull = b.pull;
            this.rawArgs = freezeList(b.rawArgs);
        }

        public static Builder builder(String image) {
            return new Builder(image);
        }

        /** Build {@code docker run ...} argv after binary (excluding {@code docker} itself). */
        public List<String> toCliArgs() {
            List<String> args = new ArrayList<>();
            args.add("run");
            if (detach) args.add("-d");
            if (remove) args.add("--rm");
            if (tty) args.add("-t");
            if (interactive) args.add("-i");
            if (name != null && !name.isBlank()) {
                args.add("--name");
                args.add(name);
            }
            if (restart != null && !restart.isBlank()) {
                args.add("--restart");
                args.add(restart);
            }
            for (Map.Entry<String, String> e : env.entrySet()) {
                args.add("-e");
                args.add(e.getKey() + "=" + e.getValue());
            }
            for (String f : envFiles) {
                args.add("--env-file");
                args.add(f);
            }
            for (PortBinding p : ports) {
                args.add("-p");
                args.add(p.toPublishFlag());
            }
            for (String v : volumes) {
                args.add("-v");
                args.add(v);
            }
            for (String n : networks) {
                args.add("--network");
                args.add(n);
            }
            if (networkMode != null && !networkMode.isBlank()) {
                args.add("--network");
                args.add(networkMode);
            }
            for (Map.Entry<String, String> e : labels.entrySet()) {
                args.add("--label");
                args.add(e.getKey() + "=" + e.getValue());
            }
            if (ipc != null) { args.add("--ipc"); args.add(ipc); }
            if (pid != null) { args.add("--pid"); args.add(pid); }
            if (privileged) args.add("--privileged");
            for (String d : devices) {
                args.add("--device");
                args.add(d);
            }
            for (String g : gpus) {
                args.add("--gpus");
                args.add(g);
            }
            if (cpus != null) { args.add("--cpus"); args.add(cpus); }
            if (memory != null) { args.add("--memory"); args.add(memory); }
            if (shmSize != null) { args.add("--shm-size"); args.add(shmSize); }
            if (workdir != null) { args.add("-w"); args.add(workdir); }
            if (user != null) { args.add("-u"); args.add(user); }
            if (hostname != null) { args.add("--hostname"); args.add(hostname); }
            for (Map.Entry<String, String> e : sysctls.entrySet()) {
                args.add("--sysctl");
                args.add(e.getKey() + "=" + e.getValue());
            }
            for (String h : extraHosts) {
                args.add("--add-host");
                args.add(h);
            }
            if (healthCmd != null) {
                args.add("--health-cmd");
                args.add(healthCmd);
            }
            if (healthInterval != null) {
                args.add("--health-interval");
                args.add(healthInterval);
            }
            if (healthTimeout != null) {
                args.add("--health-timeout");
                args.add(healthTimeout);
            }
            if (healthRetries > 0) {
                args.add("--health-retries");
                args.add(String.valueOf(healthRetries));
            }
            if (pull != null) {
                args.add("--pull");
                args.add(pull);
            }
            if (!entrypoint.isEmpty()) {
                args.add("--entrypoint");
                // docker CLI accepts single string entrypoint; join if multiple
                args.add(String.join(" ", entrypoint));
            }
            args.addAll(rawArgs);
            args.add(image);
            args.addAll(command);
            return args;
        }

        public static final class Builder {
            private final String image;
            private String name;
            private boolean detach = true;
            private boolean remove;
            private boolean tty;
            private boolean interactive;
            private final List<String> command = new ArrayList<>();
            private final List<String> entrypoint = new ArrayList<>();
            private final Map<String, String> env = new LinkedHashMap<>();
            private final List<String> envFiles = new ArrayList<>();
            private final List<PortBinding> ports = new ArrayList<>();
            private final List<String> volumes = new ArrayList<>();
            private final List<String> networks = new ArrayList<>();
            private final Map<String, String> labels = new LinkedHashMap<>();
            private String restart;
            private String networkMode;
            private String ipc;
            private String pid;
            private boolean privileged;
            private final List<String> devices = new ArrayList<>();
            private final List<String> gpus = new ArrayList<>();
            private String cpus;
            private String memory;
            private String shmSize;
            private String workdir;
            private String user;
            private String hostname;
            private final Map<String, String> sysctls = new LinkedHashMap<>();
            private final List<String> extraHosts = new ArrayList<>();
            private String healthCmd;
            private String healthInterval;
            private String healthTimeout;
            private int healthRetries;
            private String pull;
            private final List<String> rawArgs = new ArrayList<>();

            public Builder(String image) {
                this.image = Objects.requireNonNull(image, "image");
            }

            public Builder name(String v) { this.name = v; return this; }
            public Builder detach(boolean v) { this.detach = v; return this; }
            public Builder remove(boolean v) { this.remove = v; return this; }
            public Builder tty(boolean v) { this.tty = v; return this; }
            public Builder interactive(boolean v) { this.interactive = v; return this; }
            public Builder command(String... cmd) {
                command.clear();
                if (cmd != null) for (String c : cmd) if (c != null) command.add(c);
                return this;
            }
            public Builder command(List<String> cmd) {
                command.clear();
                if (cmd != null) command.addAll(cmd);
                return this;
            }
            public Builder entrypoint(String... ep) {
                entrypoint.clear();
                if (ep != null) for (String c : ep) if (c != null) entrypoint.add(c);
                return this;
            }
            public Builder env(String k, String v) {
                if (k != null && v != null) env.put(k, v);
                return this;
            }
            public Builder env(Map<String, String> m) {
                if (m != null) env.putAll(m);
                return this;
            }
            public Builder envFile(String path) {
                if (path != null) envFiles.add(path);
                return this;
            }
            public Builder publish(String portSpec) {
                ports.add(PortBinding.parse(portSpec));
                return this;
            }
            public Builder publish(int host, int container) {
                ports.add(new PortBinding(host, container));
                return this;
            }
            public Builder volume(String spec) {
                if (spec != null) volumes.add(spec);
                return this;
            }
            public Builder network(String n) {
                if (n != null) networks.add(n);
                return this;
            }
            public Builder label(String k, String v) {
                if (k != null && v != null) labels.put(k, v);
                return this;
            }
            public Builder restart(String v) { this.restart = v; return this; }
            public Builder networkMode(String v) { this.networkMode = v; return this; }
            public Builder ipc(String v) { this.ipc = v; return this; }
            public Builder pid(String v) { this.pid = v; return this; }
            public Builder privileged(boolean v) { this.privileged = v; return this; }
            public Builder device(String d) { if (d != null) devices.add(d); return this; }
            public Builder gpus(String g) { if (g != null) gpus.add(g); return this; }
            public Builder gpusAll() { return gpus("all"); }
            public Builder cpus(String v) { this.cpus = v; return this; }
            public Builder memory(String v) { this.memory = v; return this; }
            public Builder shmSize(String v) { this.shmSize = v; return this; }
            public Builder workdir(String v) { this.workdir = v; return this; }
            public Builder user(String v) { this.user = v; return this; }
            public Builder hostname(String v) { this.hostname = v; return this; }
            public Builder sysctl(String k, String v) {
                if (k != null && v != null) sysctls.put(k, v);
                return this;
            }
            public Builder extraHost(String h) { if (h != null) extraHosts.add(h); return this; }
            public Builder healthCmd(String v) { this.healthCmd = v; return this; }
            public Builder healthInterval(String v) { this.healthInterval = v; return this; }
            public Builder healthTimeout(String v) { this.healthTimeout = v; return this; }
            public Builder healthRetries(int v) { this.healthRetries = v; return this; }
            public Builder pull(String v) { this.pull = v; return this; }
            public Builder rawArg(String a) { if (a != null) rawArgs.add(a); return this; }

            public RunSpec build() { return new RunSpec(this); }
        }
    }

    private static List<String> freezeList(List<String> in) {
        if (in == null || in.isEmpty()) return List.of();
        return Collections.unmodifiableList(new ArrayList<>(in));
    }

    private static Map<String, String> freezeMap(Map<String, String> in) {
        if (in == null || in.isEmpty()) return Map.of();
        return Collections.unmodifiableMap(new LinkedHashMap<>(in));
    }
}
