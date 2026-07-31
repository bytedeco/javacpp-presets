package org.bytedeco.pytorch.deploy.docker;

import java.time.Duration;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;

/**
 * Connection / binary / global-flag options for Docker CLI and Engine API.
 *
 * <p>Resolves from builder fields, then env ({@code DOCKER_HOST}, {@code DOCKER_API_VERSION},
 * {@code DOCKER_CONTEXT}, {@code DOCKER_CONFIG}, {@code DOCKER_TLS_VERIFY}, {@code DOCKER_CERT_PATH}),
 * then platform defaults ({@code unix:///var/run/docker.sock} on Linux/macOS).
 *
 * <p>Global CLI flags injected by {@link DockerCli}:
 * {@code --config}, {@code --context}, {@code --host/-H}, {@code --debug}, {@code --log-level},
 * {@code --tls}, {@code --tlscacert/cert/key}, {@code --tlsverify}.
 */
public final class DockerOptions {

    public final String dockerBin;
    public final String composeBin;
    /** Engine host: {@code unix:///var/run/docker.sock}, {@code tcp://127.0.0.1:2375}, {@code http://...}. */
    public final String host;
    public final String apiVersion;
    public final Duration timeout;
    public final boolean tls;
    public final boolean tlsVerify;
    public final String certPath;
    public final String tlsCaCert;
    public final String tlsCert;
    public final String tlsKey;
    /** {@code --context} name (overrides DOCKER_HOST when set by docker CLI). */
    public final String context;
    /** {@code --config} client config dir (default {@code ~/.docker}). */
    public final String configDir;
    public final boolean debug;
    public final String logLevel;
    public final Map<String, String> extraEnv;
    /** Prefer CLI even when engine host is set. */
    public final boolean preferCli;

    private DockerOptions(Builder b) {
        this.dockerBin = b.dockerBin == null || b.dockerBin.isBlank() ? "docker" : b.dockerBin;
        this.composeBin = b.composeBin;
        this.host = b.host;
        this.apiVersion = b.apiVersion == null || b.apiVersion.isBlank() ? "1.43" : b.apiVersion;
        this.timeout = b.timeout == null ? Duration.ofSeconds(120) : b.timeout;
        this.tls = b.tls;
        this.tlsVerify = b.tlsVerify;
        this.certPath = b.certPath;
        this.tlsCaCert = b.tlsCaCert;
        this.tlsCert = b.tlsCert;
        this.tlsKey = b.tlsKey;
        this.context = b.context;
        this.configDir = b.configDir;
        this.debug = b.debug;
        this.logLevel = b.logLevel;
        this.extraEnv = b.extraEnv == null
                ? Map.of()
                : Collections.unmodifiableMap(new LinkedHashMap<>(b.extraEnv));
        this.preferCli = b.preferCli;
    }

    public static DockerOptions defaults() {
        return new Builder().fromEnv().build();
    }

    public static Builder builder() {
        return new Builder();
    }

    public Builder toBuilder() {
        Builder b = new Builder();
        b.dockerBin = dockerBin;
        b.composeBin = composeBin;
        b.host = host;
        b.apiVersion = apiVersion;
        b.timeout = timeout;
        b.tls = tls;
        b.tlsVerify = tlsVerify;
        b.certPath = certPath;
        b.tlsCaCert = tlsCaCert;
        b.tlsCert = tlsCert;
        b.tlsKey = tlsKey;
        b.context = context;
        b.configDir = configDir;
        b.debug = debug;
        b.logLevel = logLevel;
        b.extraEnv = new LinkedHashMap<>(extraEnv);
        b.preferCli = preferCli;
        return b;
    }

    /** Effective DOCKER_HOST-style endpoint (may be null → CLI default). */
    public String effectiveHost() {
        if (host != null && !host.isBlank()) return host;
        String env = System.getenv("DOCKER_HOST");
        if (env != null && !env.isBlank()) return env;
        return defaultUnixHost();
    }

    public static String defaultUnixHost() {
        String os = System.getProperty("os.name", "").toLowerCase();
        if (os.contains("win")) {
            return "npipe:////./pipe/docker_engine";
        }
        return "unix:///var/run/docker.sock";
    }

    public static final class Builder {
        private String dockerBin = "docker";
        private String composeBin;
        private String host;
        private String apiVersion = "1.43";
        private Duration timeout = Duration.ofSeconds(120);
        private boolean tls;
        private boolean tlsVerify;
        private String certPath;
        private String tlsCaCert;
        private String tlsCert;
        private String tlsKey;
        private String context;
        private String configDir;
        private boolean debug;
        private String logLevel;
        private Map<String, String> extraEnv = new LinkedHashMap<>();
        private boolean preferCli = true;

        public Builder dockerBin(String v) { this.dockerBin = v; return this; }
        public Builder composeBin(String v) { this.composeBin = v; return this; }
        public Builder host(String v) { this.host = v; return this; }
        public Builder apiVersion(String v) { this.apiVersion = v; return this; }
        public Builder timeout(Duration d) { this.timeout = Objects.requireNonNull(d); return this; }
        public Builder tls(boolean v) { this.tls = v; return this; }
        public Builder tlsVerify(boolean v) { this.tlsVerify = v; return this; }
        public Builder certPath(String v) { this.certPath = v; return this; }
        public Builder tlsCaCert(String v) { this.tlsCaCert = v; return this; }
        public Builder tlsCert(String v) { this.tlsCert = v; return this; }
        public Builder tlsKey(String v) { this.tlsKey = v; return this; }
        public Builder context(String v) { this.context = v; return this; }
        public Builder configDir(String v) { this.configDir = v; return this; }
        public Builder debug(boolean v) { this.debug = v; return this; }
        public Builder logLevel(String v) { this.logLevel = v; return this; }
        public Builder env(String k, String v) {
            if (k != null && v != null) extraEnv.put(k, v);
            return this;
        }
        public Builder env(Map<String, String> m) {
            if (m != null) extraEnv.putAll(m);
            return this;
        }
        public Builder preferCli(boolean v) { this.preferCli = v; return this; }

        public Builder fromEnv() {
            String h = System.getenv("DOCKER_HOST");
            if (h != null && !h.isBlank()) this.host = h;
            String api = System.getenv("DOCKER_API_VERSION");
            if (api != null && !api.isBlank()) this.apiVersion = api;
            String tlsV = System.getenv("DOCKER_TLS_VERIFY");
            if (tlsV != null && ("1".equals(tlsV) || "true".equalsIgnoreCase(tlsV))) {
                this.tlsVerify = true;
                this.tls = true;
            }
            String cert = System.getenv("DOCKER_CERT_PATH");
            if (cert != null && !cert.isBlank()) this.certPath = cert;
            String ctx = System.getenv("DOCKER_CONTEXT");
            if (ctx != null && !ctx.isBlank()) this.context = ctx;
            String cfg = System.getenv("DOCKER_CONFIG");
            if (cfg != null && !cfg.isBlank()) this.configDir = cfg;
            return this;
        }

        public DockerOptions build() {
            return new DockerOptions(this);
        }
    }
}
