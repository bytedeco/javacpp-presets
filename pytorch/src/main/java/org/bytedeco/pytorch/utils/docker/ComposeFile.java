package org.bytedeco.pytorch.utils.docker;

import org.bytedeco.pytorch.utils.yaml.Yaml;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Programmatic docker-compose.yml model: load / dump / mutate services.
 *
 * <p>Covers the subset needed for model-service deploy: image, build, ports, environment,
 * volumes, deploy.resources, gpus, healthcheck, depends_on, networks, command.
 */
public final class ComposeFile {

    private String version; // optional in compose spec v2+
    private String name;
    private final LinkedHashMap<String, Object> services = new LinkedHashMap<>();
    private final LinkedHashMap<String, Object> networks = new LinkedHashMap<>();
    private final LinkedHashMap<String, Object> volumes = new LinkedHashMap<>();
    private final LinkedHashMap<String, Object> secrets = new LinkedHashMap<>();
    private final LinkedHashMap<String, Object> configs = new LinkedHashMap<>();
    private final LinkedHashMap<String, Object> extra = new LinkedHashMap<>();

    public ComposeFile() {}

    public static ComposeFile create() {
        return new ComposeFile();
    }

    public static ComposeFile load(Path path) throws IOException {
        return fromMap(Yaml.loadMap(path));
    }

    public static ComposeFile load(String yamlText) throws IOException {
        return fromMap(Yaml.loadMap(yamlText));
    }

    @SuppressWarnings("unchecked")
    public static ComposeFile fromMap(Map<String, Object> root) {
        ComposeFile cf = new ComposeFile();
        if (root == null) return cf;
        if (root.get("version") != null) cf.version = String.valueOf(root.get("version"));
        if (root.get("name") != null) cf.name = String.valueOf(root.get("name"));
        Object svc = root.get("services");
        if (svc instanceof Map<?, ?> m) {
            for (Map.Entry<?, ?> e : m.entrySet()) {
                cf.services.put(String.valueOf(e.getKey()), e.getValue());
            }
        }
        Object nets = root.get("networks");
        if (nets instanceof Map<?, ?> m) {
            for (Map.Entry<?, ?> e : m.entrySet()) {
                cf.networks.put(String.valueOf(e.getKey()), e.getValue());
            }
        }
        Object vols = root.get("volumes");
        if (vols instanceof Map<?, ?> m) {
            for (Map.Entry<?, ?> e : m.entrySet()) {
                cf.volumes.put(String.valueOf(e.getKey()), e.getValue());
            }
        }
        Object secs = root.get("secrets");
        if (secs instanceof Map<?, ?> m) {
            for (Map.Entry<?, ?> e : m.entrySet()) {
                cf.secrets.put(String.valueOf(e.getKey()), e.getValue());
            }
        }
        Object cfgs = root.get("configs");
        if (cfgs instanceof Map<?, ?> m) {
            for (Map.Entry<?, ?> e : m.entrySet()) {
                cf.configs.put(String.valueOf(e.getKey()), e.getValue());
            }
        }
        for (Map.Entry<String, Object> e : root.entrySet()) {
            String k = e.getKey();
            if ("version".equals(k) || "name".equals(k) || "services".equals(k)
                    || "networks".equals(k) || "volumes".equals(k)
                    || "secrets".equals(k) || "configs".equals(k)) {
                continue;
            }
            cf.extra.put(k, e.getValue());
        }
        return cf;
    }

    public ComposeFile version(String v) { this.version = v; return this; }
    public ComposeFile name(String n) { this.name = n; return this; }

    public String version() { return version; }
    public String name() { return name; }

    public Map<String, Object> services() {
        return Collections.unmodifiableMap(services);
    }

    @SuppressWarnings("unchecked")
    public Map<String, Object> service(String name) {
        Object s = services.get(name);
        if (s instanceof Map<?, ?> m) return (Map<String, Object>) m;
        return null;
    }

    public ComposeFile putService(String name, Map<String, Object> spec) {
        Objects.requireNonNull(name, "name");
        services.put(name, spec == null ? new LinkedHashMap<>() : new LinkedHashMap<>(spec));
        return this;
    }

    public ComposeFile removeService(String name) {
        services.remove(name);
        return this;
    }

    public ComposeFile putNetwork(String name, Map<String, Object> spec) {
        networks.put(name, spec == null ? new LinkedHashMap<>() : new LinkedHashMap<>(spec));
        return this;
    }

    public ComposeFile putVolume(String name, Map<String, Object> spec) {
        volumes.put(name, spec == null ? null : new LinkedHashMap<>(spec));
        return this;
    }

    /**
     * Fluent service builder that writes into this compose file.
     */
    public ServiceBuilder serviceBuilder(String serviceName) {
        return new ServiceBuilder(this, serviceName);
    }

    /** Build a model-serving oriented service (image + port + model volume + optional GPU). */
    public ComposeFile addModelService(ModelContainerSpec spec) {
        Objects.requireNonNull(spec, "spec");
        putService(spec.serviceName(), spec.toComposeService());
        if (spec.modelVolumeName() != null && !spec.modelVolumeName().isBlank()) {
            volumes.putIfAbsent(spec.modelVolumeName(), null);
        }
        return this;
    }

    public Map<String, Object> toMap() {
        LinkedHashMap<String, Object> root = new LinkedHashMap<>();
        if (version != null && !version.isBlank()) root.put("version", version);
        if (name != null && !name.isBlank()) root.put("name", name);
        root.put("services", new LinkedHashMap<>(services));
        if (!networks.isEmpty()) root.put("networks", new LinkedHashMap<>(networks));
        if (!volumes.isEmpty()) root.put("volumes", new LinkedHashMap<>(volumes));
        if (!secrets.isEmpty()) root.put("secrets", new LinkedHashMap<>(secrets));
        if (!configs.isEmpty()) root.put("configs", new LinkedHashMap<>(configs));
        root.putAll(extra);
        return root;
    }

    public String toYaml() {
        return Yaml.dump(toMap());
    }

    public void save(Path path) throws IOException {
        Yaml.dump(path, toMap());
    }

    @Override
    public String toString() {
        return "ComposeFile{name=" + name + ", services=" + services.keySet() + "}";
    }

    // ---- service builder ----

    public static final class ServiceBuilder {
        private final ComposeFile parent;
        private final String serviceName;
        private final LinkedHashMap<String, Object> spec = new LinkedHashMap<>();

        ServiceBuilder(ComposeFile parent, String serviceName) {
            this.parent = parent;
            this.serviceName = Objects.requireNonNull(serviceName, "serviceName");
        }

        public ServiceBuilder image(String image) {
            spec.put("image", image);
            return this;
        }

        public ServiceBuilder build(String context) {
            spec.put("build", context);
            return this;
        }

        public ServiceBuilder build(String context, String dockerfile) {
            LinkedHashMap<String, Object> b = new LinkedHashMap<>();
            b.put("context", context);
            if (dockerfile != null) b.put("dockerfile", dockerfile);
            spec.put("build", b);
            return this;
        }

        public ServiceBuilder containerName(String name) {
            spec.put("container_name", name);
            return this;
        }

        public ServiceBuilder restart(String policy) {
            spec.put("restart", policy);
            return this;
        }

        public ServiceBuilder command(String... cmd) {
            if (cmd == null || cmd.length == 0) return this;
            if (cmd.length == 1) spec.put("command", cmd[0]);
            else {
                List<String> list = new ArrayList<>();
                Collections.addAll(list, cmd);
                spec.put("command", list);
            }
            return this;
        }

        public ServiceBuilder entrypoint(String... ep) {
            if (ep == null) return this;
            List<String> list = new ArrayList<>();
            Collections.addAll(list, ep);
            spec.put("entrypoint", list);
            return this;
        }

        public ServiceBuilder port(String mapping) {
            @SuppressWarnings("unchecked")
            List<Object> ports = (List<Object>) spec.computeIfAbsent("ports", k -> new ArrayList<>());
            ports.add(mapping);
            return this;
        }

        public ServiceBuilder port(int host, int container) {
            return port(host + ":" + container);
        }

        public ServiceBuilder env(String key, String value) {
            Object existing = spec.get("environment");
            if (existing instanceof List<?>) {
                @SuppressWarnings("unchecked")
                List<Object> list = (List<Object>) existing;
                list.add(key + "=" + value);
            } else {
                @SuppressWarnings("unchecked")
                Map<String, Object> env = (Map<String, Object>) spec.computeIfAbsent(
                        "environment", k -> new LinkedHashMap<>());
                env.put(key, value);
            }
            return this;
        }

        public ServiceBuilder env(Map<String, String> map) {
            if (map != null) map.forEach(this::env);
            return this;
        }

        public ServiceBuilder volume(String mapping) {
            @SuppressWarnings("unchecked")
            List<Object> vols = (List<Object>) spec.computeIfAbsent("volumes", k -> new ArrayList<>());
            vols.add(mapping);
            return this;
        }

        public ServiceBuilder network(String name) {
            @SuppressWarnings("unchecked")
            List<Object> nets = (List<Object>) spec.computeIfAbsent("networks", k -> new ArrayList<>());
            nets.add(name);
            return this;
        }

        public ServiceBuilder dependsOn(String... services) {
            if (services == null) return this;
            List<Object> deps = new ArrayList<>();
            Collections.addAll(deps, services);
            spec.put("depends_on", deps);
            return this;
        }

        public ServiceBuilder healthcheck(String test, String interval, String timeout, int retries) {
            LinkedHashMap<String, Object> h = new LinkedHashMap<>();
            h.put("test", List.of("CMD-SHELL", test));
            if (interval != null) h.put("interval", interval);
            if (timeout != null) h.put("timeout", timeout);
            if (retries > 0) h.put("retries", retries);
            spec.put("healthcheck", h);
            return this;
        }

        public ServiceBuilder gpus(int count) {
            // compose deploy / device_requests style
            LinkedHashMap<String, Object> reserve = new LinkedHashMap<>();
            List<Object> devices = new ArrayList<>();
            LinkedHashMap<String, Object> dev = new LinkedHashMap<>();
            dev.put("driver", "nvidia");
            dev.put("count", count <= 0 ? "all" : count);
            dev.put("capabilities", List.of(List.of("gpu")));
            devices.add(dev);
            reserve.put("devices", devices);
            @SuppressWarnings("unchecked")
            Map<String, Object> deploy = (Map<String, Object>) spec.computeIfAbsent(
                    "deploy", k -> new LinkedHashMap<>());
            @SuppressWarnings("unchecked")
            Map<String, Object> resources = (Map<String, Object>) deploy.computeIfAbsent(
                    "resources", k -> new LinkedHashMap<>());
            resources.put("reservations", reserve);
            // also set classic runtime-friendly env
            env("NVIDIA_VISIBLE_DEVICES", count <= 0 ? "all" : "0");
            return this;
        }

        public ServiceBuilder cpus(double cpus) {
            @SuppressWarnings("unchecked")
            Map<String, Object> deploy = (Map<String, Object>) spec.computeIfAbsent(
                    "deploy", k -> new LinkedHashMap<>());
            @SuppressWarnings("unchecked")
            Map<String, Object> resources = (Map<String, Object>) deploy.computeIfAbsent(
                    "resources", k -> new LinkedHashMap<>());
            @SuppressWarnings("unchecked")
            Map<String, Object> limits = (Map<String, Object>) resources.computeIfAbsent(
                    "limits", k -> new LinkedHashMap<>());
            limits.put("cpus", String.valueOf(cpus));
            return this;
        }

        public ServiceBuilder memory(String mem) {
            @SuppressWarnings("unchecked")
            Map<String, Object> deploy = (Map<String, Object>) spec.computeIfAbsent(
                    "deploy", k -> new LinkedHashMap<>());
            @SuppressWarnings("unchecked")
            Map<String, Object> resources = (Map<String, Object>) deploy.computeIfAbsent(
                    "resources", k -> new LinkedHashMap<>());
            @SuppressWarnings("unchecked")
            Map<String, Object> limits = (Map<String, Object>) resources.computeIfAbsent(
                    "limits", k -> new LinkedHashMap<>());
            limits.put("memory", mem);
            return this;
        }

        public ServiceBuilder shmSize(String size) {
            spec.put("shm_size", size);
            return this;
        }

        public ServiceBuilder label(String k, String v) {
            @SuppressWarnings("unchecked")
            Map<String, Object> labels = (Map<String, Object>) spec.computeIfAbsent(
                    "labels", x -> new LinkedHashMap<>());
            labels.put(k, v);
            return this;
        }

        public ServiceBuilder raw(String key, Object value) {
            spec.put(key, value);
            return this;
        }

        public ComposeFile apply() {
            parent.putService(serviceName, spec);
            return parent;
        }

        public Map<String, Object> buildSpec() {
            return new LinkedHashMap<>(spec);
        }
    }
}
