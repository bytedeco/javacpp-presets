package org.bytedeco.pytorch.deploy.k8s;

import org.bytedeco.pytorch.utils.json.Json;

import javax.net.ssl.SSLContext;
import javax.net.ssl.TrustManager;
import javax.net.ssl.X509TrustManager;
import java.io.IOException;
import java.net.URI;
import java.net.URLEncoder;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;
import java.security.cert.X509Certificate;
import java.time.Duration;
import java.util.LinkedHashMap;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;

/**
 * Minimal Kubernetes apiserver REST client (JSON only, no client-java).
 *
 * <p>Auth: Bearer token (kubeconfig / SA) preferred. Optional insecure TLS skip
 * for lab clusters. Client-cert mTLS is not fully wired in v1 (use kubectl).
 *
 * <pre>{@code
 * try (K8sClient c = K8sClient.connect(opts)) {
 *     Map&lt;String, Object&gt; dep = c.get("apps", "v1", "deployments", "default", "ranker");
 * }
 * }</pre>
 */
public final class K8sClient implements AutoCloseable {

    private final K8sOptions options;
    private final HttpClient http;
    private final URI base;
    private final String token;

    public K8sClient(K8sOptions options, String apiServer, String bearerToken) {
        this.options = options == null ? K8sOptions.defaults() : options;
        String server = apiServer;
        if (server == null || server.isBlank()) {
            server = this.options.apiServer;
        }
        if (server == null || server.isBlank()) {
            throw new K8sException("apiServer required for K8sClient (set K8sOptions.apiServer or kubeconfig)");
        }
        if (server.endsWith("/")) server = server.substring(0, server.length() - 1);
        this.base = URI.create(server);
        this.token = bearerToken != null ? bearerToken : this.options.bearerToken;
        Duration timeout = this.options.timeout;
        HttpClient.Builder hb = HttpClient.newBuilder()
                .connectTimeout(timeout)
                .followRedirects(HttpClient.Redirect.NORMAL);
        if (this.options.insecureSkipTls) {
            try {
                SSLContext ctx = SSLContext.getInstance("TLS");
                ctx.init(null, new TrustManager[]{new TrustAll()}, new java.security.SecureRandom());
                hb.sslContext(ctx);
            } catch (Exception e) {
                throw new K8sException("failed to init insecure TLS: " + e.getMessage(), e);
            }
        }
        this.http = hb.build();
    }

    public static K8sClient connect(K8sOptions options) {
        K8sOptions opts = options == null ? K8sOptions.defaults() : options;
        String server = opts.apiServer;
        String token = opts.bearerToken;
        if ((server == null || server.isBlank()) || (token == null || token.isBlank())) {
            try {
                KubeConfig kc;
                if (opts.kubeconfig != null) {
                    kc = KubeConfig.load(opts.kubeconfig);
                } else {
                    kc = KubeConfig.load();
                }
                if (server == null || server.isBlank()) server = kc.server();
                if (token == null || token.isBlank()) token = kc.token();
                if (!opts.insecureSkipTls && kc.insecureSkipTls()) {
                    opts = opts.toBuilder().insecureSkipTls(true).build();
                }
            } catch (IOException e) {
                throw new K8sException("load kubeconfig: " + e.getMessage(), e);
            }
        }
        return new K8sClient(opts, server, token);
    }

    public K8sOptions options() {
        return options;
    }

    public URI base() {
        return base;
    }

    // ---- generic CRUD ----

    public Object getRaw(String path) {
        return exchange("GET", path, null);
    }

    public Map<String, Object> get(
            String group, String version, String resourcePlural, String namespace, String name) {
        return asMap(exchange("GET", resourcePath(group, version, resourcePlural, namespace, name), null));
    }

    public Map<String, Object> list(
            String group, String version, String resourcePlural, String namespace) {
        return asMap(exchange("GET", resourcePath(group, version, resourcePlural, namespace, null), null));
    }

    public Map<String, Object> create(
            String group, String version, String resourcePlural, String namespace, Object body) {
        return asMap(exchange("POST", resourcePath(group, version, resourcePlural, namespace, null), body));
    }

    public Map<String, Object> replace(
            String group, String version, String resourcePlural, String namespace, String name, Object body) {
        return asMap(exchange("PUT", resourcePath(group, version, resourcePlural, namespace, name), body));
    }

    public Map<String, Object> patch(
            String group, String version, String resourcePlural, String namespace, String name, Object patchBody) {
        // strategic merge patch as JSON
        String path = resourcePath(group, version, resourcePlural, namespace, name);
        return asMap(exchange("PATCH", path, patchBody, "application/strategic-merge-patch+json"));
    }

    public Map<String, Object> patchJson(
            String group, String version, String resourcePlural, String namespace, String name, Object patchBody) {
        String path = resourcePath(group, version, resourcePlural, namespace, name);
        return asMap(exchange("PATCH", path, patchBody, "application/json-patch+json"));
    }

    public void delete(
            String group, String version, String resourcePlural, String namespace, String name) {
        exchange("DELETE", resourcePath(group, version, resourcePlural, namespace, name), null);
    }

    // ---- convenience ----

    public Map<String, Object> getDeployment(String namespace, String name) {
        return get("apps", "v1", "deployments", namespace, name);
    }

    public Map<String, Object> scaleDeployment(String namespace, String name, int replicas) {
        Map<String, Object> body = JsonMap.mapOf(
                "apiVersion", "autoscaling/v1",
                "kind", "Scale",
                "metadata", JsonMap.mapOf("name", name, "namespace", namespace),
                "spec", JsonMap.mapOf("replicas", replicas));
        // scale subresource
        String path = resourcePath("apps", "v1", "deployments", namespace, name) + "/scale";
        return asMap(exchange("PUT", path, body));
    }

    public int readyReplicas(String namespace, String name) {
        try {
            Map<String, Object> dep = getDeployment(namespace, name);
            Object status = dep.get("status");
            if (status instanceof Map<?, ?> m) {
                Object r = m.get("readyReplicas");
                if (r instanceof Number n) return n.intValue();
            }
        } catch (Exception ignored) {
        }
        return 0;
    }

    public Object exchange(String method, String path, Object body) {
        return exchange(method, path, body, "application/json");
    }

    public Object exchange(String method, String path, Object body, String contentType) {
        try {
            String p = path == null ? "/" : path;
            if (!p.startsWith("/")) p = "/" + p;
            URI uri = base.resolve(p);
            HttpRequest.Builder rb = HttpRequest.newBuilder(uri)
                    .timeout(options.timeout)
                    .header("Accept", "application/json");
            if (token != null && !token.isBlank()) {
                rb.header("Authorization", "Bearer " + token);
            }
            String m = method == null ? "GET" : method.toUpperCase(Locale.ROOT);
            if ("GET".equals(m) || "DELETE".equals(m) || "HEAD".equals(m)) {
                if (body == null) {
                    rb.method(m, HttpRequest.BodyPublishers.noBody());
                } else {
                    String payload = body instanceof String s ? s : Json.encode(body);
                    rb.header("Content-Type", contentType == null ? "application/json" : contentType);
                    rb.method(m, HttpRequest.BodyPublishers.ofString(payload, StandardCharsets.UTF_8));
                }
            } else {
                String payload;
                if (body == null) payload = "{}";
                else if (body instanceof String s) payload = s;
                else payload = Json.encode(body);
                rb.header("Content-Type", contentType == null ? "application/json" : contentType);
                rb.method(m, HttpRequest.BodyPublishers.ofString(payload, StandardCharsets.UTF_8));
            }
            HttpResponse<String> resp = http.send(rb.build(),
                    HttpResponse.BodyHandlers.ofString(StandardCharsets.UTF_8));
            int code = resp.statusCode();
            String text = resp.body() == null ? "" : resp.body();
            if (code >= 200 && code < 300) {
                if (text.isBlank()) return null;
                try {
                    return Json.decode(text);
                } catch (Exception e) {
                    return text;
                }
            }
            throw K8sException.ofHttp(m + " " + p, code, text);
        } catch (K8sException e) {
            throw e;
        } catch (IOException e) {
            throw new K8sException("k8s http I/O: " + e.getMessage(), e, -1, -1, method, path);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new K8sException("k8s http interrupted", e, -1, -1, method, path);
        }
    }

    public static String resourcePath(
            String group, String version, String resourcePlural, String namespace, String name) {
        Objects.requireNonNull(version, "version");
        Objects.requireNonNull(resourcePlural, "resourcePlural");
        StringBuilder sb = new StringBuilder();
        if (group == null || group.isBlank() || "core".equals(group)) {
            sb.append("/api/").append(version);
        } else {
            sb.append("/apis/").append(group).append('/').append(version);
        }
        if (namespace != null && !namespace.isBlank()) {
            sb.append("/namespaces/").append(enc(namespace));
        }
        sb.append('/').append(resourcePlural);
        if (name != null && !name.isBlank()) {
            sb.append('/').append(enc(name));
        }
        return sb.toString();
    }

    private static String enc(String s) {
        return URLEncoder.encode(s, StandardCharsets.UTF_8).replace("+", "%20");
    }

    @SuppressWarnings("unchecked")
    private static Map<String, Object> asMap(Object o) {
        if (o == null) return new LinkedHashMap<>();
        if (o instanceof Map<?, ?> m) return (Map<String, Object>) m;
        throw new K8sException("expected JSON object, got " + o.getClass().getSimpleName());
    }

    @Override
    public void close() {
        // HttpClient has no close on Java 17
    }

    private static final class TrustAll implements X509TrustManager {
        @Override public void checkClientTrusted(X509Certificate[] chain, String authType) {}
        @Override public void checkServerTrusted(X509Certificate[] chain, String authType) {}
        @Override public X509Certificate[] getAcceptedIssuers() { return new X509Certificate[0]; }
    }

    /** Tiny map helper without pulling dataframe HttpJson. */
    public static final class JsonMap {
        private JsonMap() {}
        public static Map<String, Object> mapOf(Object... kv) {
            if (kv == null || kv.length == 0) return new LinkedHashMap<>();
            if ((kv.length & 1) != 0) throw new IllegalArgumentException("odd kv");
            Map<String, Object> m = new LinkedHashMap<>();
            for (int i = 0; i < kv.length; i += 2) {
                m.put(String.valueOf(kv[i]), kv[i + 1]);
            }
            return m;
        }
    }
}
