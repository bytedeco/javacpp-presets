package org.bytedeco.pytorch.utils.k8s;

import org.bytedeco.pytorch.utils.yaml.Yaml;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Minimal kubeconfig parser (clusters / users / contexts / current-context).
 *
 * <p>Also supports in-cluster config from
 * {@code /var/run/secrets/kubernetes.io/serviceaccount/}.
 */
public final class KubeConfig {

    public static final Path IN_CLUSTER_TOKEN =
            Path.of("/var/run/secrets/kubernetes.io/serviceaccount/token");
    public static final Path IN_CLUSTER_CA =
            Path.of("/var/run/secrets/kubernetes.io/serviceaccount/ca.crt");
    public static final Path IN_CLUSTER_NS =
            Path.of("/var/run/secrets/kubernetes.io/serviceaccount/namespace");

    private final String currentContext;
    private final String server;
    private final String namespace;
    private final String token;
    private final Path clientCertificate;
    private final Path clientKey;
    private final Path certificateAuthority;
    private final boolean insecureSkipTls;
    private final String username;
    private final String password;
    private final Map<String, Object> raw;

    private KubeConfig(
            String currentContext,
            String server,
            String namespace,
            String token,
            Path clientCertificate,
            Path clientKey,
            Path certificateAuthority,
            boolean insecureSkipTls,
            String username,
            String password,
            Map<String, Object> raw) {
        this.currentContext = currentContext;
        this.server = server;
        this.namespace = namespace;
        this.token = token;
        this.clientCertificate = clientCertificate;
        this.clientKey = clientKey;
        this.certificateAuthority = certificateAuthority;
        this.insecureSkipTls = insecureSkipTls;
        this.username = username;
        this.password = password;
        this.raw = raw == null ? Map.of() : raw;
    }

    public static Path defaultPath() {
        String home = System.getProperty("user.home", ".");
        return Path.of(home, ".kube", "config");
    }

    public static boolean inCluster() {
        return Files.isRegularFile(IN_CLUSTER_TOKEN);
    }

    public static KubeConfig loadInCluster() throws IOException {
        String token = Files.readString(IN_CLUSTER_TOKEN, StandardCharsets.UTF_8).trim();
        String ns = Files.isRegularFile(IN_CLUSTER_NS)
                ? Files.readString(IN_CLUSTER_NS, StandardCharsets.UTF_8).trim()
                : "default";
        String host = System.getenv("KUBERNETES_SERVICE_HOST");
        String port = System.getenv("KUBERNETES_SERVICE_PORT");
        if (host == null || host.isBlank()) {
            throw new K8sException("in-cluster: KUBERNETES_SERVICE_HOST not set");
        }
        if (port == null || port.isBlank()) port = "443";
        String server = "https://" + host + ":" + port;
        Path ca = Files.isRegularFile(IN_CLUSTER_CA) ? IN_CLUSTER_CA : null;
        return new KubeConfig(
                "in-cluster", server, ns, token, null, null, ca, false, null, null, Map.of());
    }

    public static KubeConfig load() throws IOException {
        if (inCluster()) {
            try {
                return loadInCluster();
            } catch (Exception ignored) {
                // fall through to file
            }
        }
        String env = System.getenv("KUBECONFIG");
        if (env != null && !env.isBlank()) {
            String first = env.split(java.io.File.pathSeparator, 2)[0];
            return load(Path.of(first));
        }
        return load(defaultPath());
    }

    public static KubeConfig load(Path path) throws IOException {
        Objects.requireNonNull(path, "path");
        Map<String, Object> root = Yaml.loadMap(path);
        return fromMap(root);
    }

    public static KubeConfig load(String yamlText) throws IOException {
        return fromMap(Yaml.loadMap(yamlText));
    }

    @SuppressWarnings("unchecked")
    public static KubeConfig fromMap(Map<String, Object> root) {
        if (root == null) root = Map.of();
        String current = str(root.get("current-context"), "");
        Map<String, Map<String, Object>> contexts = indexByName(root.get("contexts"));
        Map<String, Map<String, Object>> clusters = indexByName(root.get("clusters"));
        Map<String, Map<String, Object>> users = indexByName(root.get("users"));

        String ctxName = current;
        Map<String, Object> ctx = contexts.get(ctxName);
        if (ctx == null && !contexts.isEmpty()) {
            ctxName = contexts.keySet().iterator().next();
            ctx = contexts.get(ctxName);
        }
        Map<String, Object> ctxInner = ctx == null ? Map.of()
                : (ctx.get("context") instanceof Map<?, ?> m ? (Map<String, Object>) m : Map.of());
        String clusterName = str(ctxInner.get("cluster"), "");
        String userName = str(ctxInner.get("user"), "");
        String namespace = str(ctxInner.get("namespace"), "default");

        Map<String, Object> clusterWrap = clusters.getOrDefault(clusterName, Map.of());
        Map<String, Object> cluster = clusterWrap.get("cluster") instanceof Map<?, ?> m
                ? (Map<String, Object>) m : Map.of();
        String server = str(cluster.get("server"), "");
        boolean insecure = Yaml.asBool(cluster.get("insecure-skip-tls-verify"), false);
        Path ca = fileOrData(cluster, "certificate-authority", "certificate-authority-data", "ca.crt");

        Map<String, Object> userWrap = users.getOrDefault(userName, Map.of());
        Map<String, Object> user = userWrap.get("user") instanceof Map<?, ?> m
                ? (Map<String, Object>) m : Map.of();
        String token = str(user.get("token"), null);
        if ((token == null || token.isBlank()) && user.get("tokenFile") != null) {
            try {
                token = Files.readString(Path.of(String.valueOf(user.get("tokenFile")))).trim();
            } catch (Exception ignored) {
            }
        }
        // exec auth not supported — leave token null
        Path clientCert = fileOrData(user, "client-certificate", "client-certificate-data", "client.crt");
        Path clientKey = fileOrData(user, "client-key", "client-key-data", "client.key");
        String username = str(user.get("username"), null);
        String password = str(user.get("password"), null);

        return new KubeConfig(
                ctxName, server, namespace, token, clientCert, clientKey, ca,
                insecure, username, password, root);
    }

    public String currentContext() { return currentContext; }
    public String server() { return server; }
    public String namespace() { return namespace; }
    public String token() { return token; }
    public Path clientCertificate() { return clientCertificate; }
    public Path clientKey() { return clientKey; }
    public Path certificateAuthority() { return certificateAuthority; }
    public boolean insecureSkipTls() { return insecureSkipTls; }
    public String username() { return username; }
    public String password() { return password; }
    public Map<String, Object> raw() { return raw; }

    public boolean hasToken() {
        return token != null && !token.isBlank();
    }

    /** Merge into K8sOptions builder. */
    public K8sOptions.Builder applyTo(K8sOptions.Builder b) {
        if (server != null && !server.isBlank()) b.apiServer(server);
        if (namespace != null && !namespace.isBlank()) b.namespace(namespace);
        if (hasToken()) b.bearerToken(token);
        if (clientCertificate != null) b.clientCert(clientCertificate);
        if (clientKey != null) b.clientKey(clientKey);
        if (certificateAuthority != null) b.caCert(certificateAuthority);
        if (insecureSkipTls) b.insecureSkipTls(true);
        if (currentContext != null) b.context(currentContext);
        return b;
    }

    @SuppressWarnings("unchecked")
    private static Map<String, Map<String, Object>> indexByName(Object listObj) {
        Map<String, Map<String, Object>> out = new LinkedHashMap<>();
        if (!(listObj instanceof List<?> list)) return out;
        for (Object o : list) {
            if (!(o instanceof Map<?, ?> m)) continue;
            Map<String, Object> map = (Map<String, Object>) m;
            String name = str(map.get("name"), null);
            if (name != null) out.put(name, map);
        }
        return out;
    }

    private static Path fileOrData(Map<String, Object> m, String fileKey, String dataKey, String tmpName) {
        Object file = m.get(fileKey);
        if (file != null) {
            Path p = Path.of(String.valueOf(file));
            if (Files.isRegularFile(p)) return p;
        }
        Object data = m.get(dataKey);
        if (data != null) {
            try {
                byte[] raw = java.util.Base64.getDecoder().decode(String.valueOf(data).replaceAll("\\s", ""));
                Path tmp = Files.createTempFile("jnitorch-kube-", "-" + tmpName);
                Files.write(tmp, raw);
                tmp.toFile().deleteOnExit();
                return tmp;
            } catch (Exception ignored) {
            }
        }
        return null;
    }

    private static String str(Object o, String def) {
        if (o == null) return def;
        String s = String.valueOf(o);
        return s.isBlank() ? def : s;
    }

    @Override
    public String toString() {
        return "KubeConfig{context=" + currentContext + ", server=" + server
                + ", ns=" + namespace + ", token=" + (hasToken() ? "***" : "none") + "}";
    }
}
