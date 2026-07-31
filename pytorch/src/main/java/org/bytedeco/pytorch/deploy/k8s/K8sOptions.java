package org.bytedeco.pytorch.deploy.k8s;

import java.nio.file.Path;
import java.time.Duration;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;

/**
 * Options for kubectl CLI and apiserver REST client.
 */
public final class K8sOptions {

    public final String kubectlBin;
    public final Path kubeconfig;
    public final String context;
    public final String namespace;
    public final Duration timeout;
    public final boolean insecureSkipTls;
    /** Direct apiserver URL override (otherwise from kubeconfig / in-cluster). */
    public final String apiServer;
    public final String bearerToken;
    public final Path clientCert;
    public final Path clientKey;
    public final Path caCert;
    public final Map<String, String> extraEnv;
    public final boolean preferKubectl;

    private K8sOptions(Builder b) {
        this.kubectlBin = b.kubectlBin == null || b.kubectlBin.isBlank() ? "kubectl" : b.kubectlBin;
        this.kubeconfig = b.kubeconfig;
        this.context = b.context;
        this.namespace = b.namespace == null || b.namespace.isBlank() ? "default" : b.namespace;
        this.timeout = b.timeout == null ? Duration.ofSeconds(120) : b.timeout;
        this.insecureSkipTls = b.insecureSkipTls;
        this.apiServer = b.apiServer;
        this.bearerToken = b.bearerToken;
        this.clientCert = b.clientCert;
        this.clientKey = b.clientKey;
        this.caCert = b.caCert;
        this.extraEnv = b.extraEnv == null
                ? Map.of()
                : Collections.unmodifiableMap(new LinkedHashMap<>(b.extraEnv));
        this.preferKubectl = b.preferKubectl;
    }

    public static K8sOptions defaults() {
        return new Builder().fromEnv().build();
    }

    public static Builder builder() {
        return new Builder();
    }

    public Builder toBuilder() {
        Builder b = new Builder();
        b.kubectlBin = kubectlBin;
        b.kubeconfig = kubeconfig;
        b.context = context;
        b.namespace = namespace;
        b.timeout = timeout;
        b.insecureSkipTls = insecureSkipTls;
        b.apiServer = apiServer;
        b.bearerToken = bearerToken;
        b.clientCert = clientCert;
        b.clientKey = clientKey;
        b.caCert = caCert;
        b.extraEnv = new LinkedHashMap<>(extraEnv);
        b.preferKubectl = preferKubectl;
        return b;
    }

    public static final class Builder {
        private String kubectlBin = "kubectl";
        private Path kubeconfig;
        private String context;
        private String namespace = "default";
        private Duration timeout = Duration.ofSeconds(120);
        private boolean insecureSkipTls;
        private String apiServer;
        private String bearerToken;
        private Path clientCert;
        private Path clientKey;
        private Path caCert;
        private Map<String, String> extraEnv = new LinkedHashMap<>();
        private boolean preferKubectl = true;

        public Builder kubectlBin(String v) { this.kubectlBin = v; return this; }
        public Builder kubeconfig(Path p) { this.kubeconfig = p; return this; }
        public Builder kubeconfig(String p) {
            this.kubeconfig = p == null ? null : Path.of(p);
            return this;
        }
        public Builder context(String v) { this.context = v; return this; }
        public Builder namespace(String v) { this.namespace = v; return this; }
        public Builder timeout(Duration d) { this.timeout = Objects.requireNonNull(d); return this; }
        public Builder insecureSkipTls(boolean v) { this.insecureSkipTls = v; return this; }
        public Builder apiServer(String v) { this.apiServer = v; return this; }
        public Builder bearerToken(String v) { this.bearerToken = v; return this; }
        public Builder clientCert(Path p) { this.clientCert = p; return this; }
        public Builder clientKey(Path p) { this.clientKey = p; return this; }
        public Builder caCert(Path p) { this.caCert = p; return this; }
        public Builder env(String k, String v) {
            if (k != null && v != null) extraEnv.put(k, v);
            return this;
        }
        public Builder env(Map<String, String> m) {
            if (m != null) extraEnv.putAll(m);
            return this;
        }
        public Builder preferKubectl(boolean v) { this.preferKubectl = v; return this; }

        public Builder fromEnv() {
            String kc = System.getenv("KUBECONFIG");
            if (kc != null && !kc.isBlank()) {
                // take first path if multi
                String first = kc.split(java.io.File.pathSeparator, 2)[0];
                this.kubeconfig = Path.of(first);
            }
            String ns = System.getenv("KUBECTL_NAMESPACE");
            if (ns == null || ns.isBlank()) ns = System.getenv("NAMESPACE");
            if (ns != null && !ns.isBlank()) this.namespace = ns;
            String ctx = System.getenv("KUBECTL_CONTEXT");
            if (ctx != null && !ctx.isBlank()) this.context = ctx;
            String server = System.getenv("KUBERNETES_SERVICE_HOST");
            String port = System.getenv("KUBERNETES_SERVICE_PORT");
            if (server != null && !server.isBlank()) {
                String p = (port == null || port.isBlank()) ? "443" : port;
                this.apiServer = "https://" + server + ":" + p;
            }
            return this;
        }

        public K8sOptions build() {
            return new K8sOptions(this);
        }
    }
}
