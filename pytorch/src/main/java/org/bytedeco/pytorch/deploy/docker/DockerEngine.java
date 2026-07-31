package org.bytedeco.pytorch.deploy.docker;

import org.bytedeco.pytorch.utils.json.Json;

import java.io.IOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

/**
 * Docker Engine REST API client (no docker-java).
 *
 * <p>Transports:
 * <ul>
 *   <li>{@code unix:///path/to.sock} — {@link UnixSocketHttp}</li>
 *   <li>{@code tcp://host:port} / {@code http(s)://} — JDK {@link HttpClient}</li>
 * </ul>
 *
 * <p>API version prefix: {@code /v1.43/...} from {@link DockerOptions#apiVersion}.
 *
 * @see <a href="https://docs.docker.com/engine/api/">Docker Engine API</a>
 */
public final class DockerEngine implements AutoCloseable {

    private final DockerOptions options;
    private final String apiPrefix;
    private final Transport transport;

    public DockerEngine(DockerOptions options) {
        this.options = options == null ? DockerOptions.defaults() : options;
        String ver = this.options.apiVersion;
        if (ver.startsWith("v")) ver = ver.substring(1);
        this.apiPrefix = "/v" + ver;
        this.transport = Transport.open(this.options);
    }

    public static DockerEngine connect(DockerOptions options) {
        return new DockerEngine(options);
    }

    public DockerOptions options() {
        return options;
    }

    public void ping() {
        EngineResponse r = transport.exchange("GET", "/_ping", null, null);
        if (!r.ok() && r.status != 200) {
            // some engines return OK body "OK"
            throw DockerException.ofHttp("ping", r.status, r.body);
        }
    }

    public Map<String, Object> version() {
        return getJson("/version");
    }

    public Map<String, Object> info() {
        return getJson("/info");
    }

    @SuppressWarnings("unchecked")
    public List<Map<String, Object>> listContainers(boolean all) {
        Object v = exchangeJson("GET", "/containers/json?all=" + (all ? "1" : "0"), null);
        if (v instanceof List<?> list) {
            List<Map<String, Object>> out = new ArrayList<>();
            for (Object o : list) {
                if (o instanceof Map<?, ?> m) out.add((Map<String, Object>) m);
            }
            return out;
        }
        return List.of();
    }

    public Map<String, Object> inspectContainer(String id) {
        return getJson("/containers/" + enc(id) + "/json");
    }

    public String createContainer(String name, Map<String, Object> body) {
        String path = "/containers/create";
        if (name != null && !name.isBlank()) {
            path += "?name=" + enc(name);
        }
        Map<String, Object> resp = postJson(path, body);
        Object id = resp.get("Id");
        if (id == null) throw new DockerException("createContainer: missing Id in response");
        return String.valueOf(id);
    }

    public void startContainer(String id) {
        EngineResponse r = transport.exchange("POST", apiPrefix + "/containers/" + enc(id) + "/start",
                Map.of("Content-Type", "application/json"), new byte[0]);
        if (!r.ok() && r.status != 204 && r.status != 304) {
            throw DockerException.ofHttp("start", r.status, r.body);
        }
    }

    public void stopContainer(String id, Integer timeoutSeconds) {
        String q = timeoutSeconds == null ? "" : "?t=" + timeoutSeconds;
        EngineResponse r = transport.exchange("POST",
                apiPrefix + "/containers/" + enc(id) + "/stop" + q,
                null, null);
        if (!r.ok() && r.status != 204 && r.status != 304) {
            throw DockerException.ofHttp("stop", r.status, r.body);
        }
    }

    public void removeContainer(String id, boolean force, boolean removeVolumes) {
        String q = "?force=" + (force ? "1" : "0") + "&v=" + (removeVolumes ? "1" : "0");
        EngineResponse r = transport.exchange("DELETE",
                apiPrefix + "/containers/" + enc(id) + q, null, null);
        if (!r.ok() && r.status != 204) {
            throw DockerException.ofHttp("rm", r.status, r.body);
        }
    }

    public String containerLogs(String id, boolean stdout, boolean stderr, Integer tail) {
        StringBuilder q = new StringBuilder("?stdout=")
                .append(stdout ? "1" : "0")
                .append("&stderr=").append(stderr ? "1" : "0");
        if (tail != null) q.append("&tail=").append(tail);
        EngineResponse r = transport.exchange("GET",
                apiPrefix + "/containers/" + enc(id) + "/logs" + q, null, null);
        if (!r.ok()) throw DockerException.ofHttp("logs", r.status, r.body);
        // multiplexed stream header may be present; return raw string best-effort
        return r.body;
    }

    @SuppressWarnings("unchecked")
    public List<Map<String, Object>> listImages() {
        Object v = exchangeJson("GET", "/images/json", null);
        if (v instanceof List<?> list) {
            List<Map<String, Object>> out = new ArrayList<>();
            for (Object o : list) {
                if (o instanceof Map<?, ?> m) out.add((Map<String, Object>) m);
            }
            return out;
        }
        return List.of();
    }

    public void pingOrThrow() {
        ping();
    }

    // ---- internal ----

    private Map<String, Object> getJson(String path) {
        Object v = exchangeJson("GET", path, null);
        if (v instanceof Map<?, ?> m) {
            @SuppressWarnings("unchecked")
            Map<String, Object> map = (Map<String, Object>) m;
            return map;
        }
        throw new DockerException("expected JSON object from " + path);
    }

    private Map<String, Object> postJson(String path, Object body) {
        Object v = exchangeJson("POST", path, body);
        if (v == null) return new LinkedHashMap<>();
        if (v instanceof Map<?, ?> m) {
            @SuppressWarnings("unchecked")
            Map<String, Object> map = (Map<String, Object>) m;
            return map;
        }
        throw new DockerException("expected JSON object from POST " + path);
    }

    private Object exchangeJson(String method, String path, Object body) {
        String full = path.startsWith("/_") ? path : apiPrefix + path;
        byte[] raw = null;
        Map<String, String> headers = new LinkedHashMap<>();
        if (body != null) {
            String json = body instanceof String s ? s : Json.encode(body);
            raw = json.getBytes(StandardCharsets.UTF_8);
            headers.put("Content-Type", "application/json");
        }
        EngineResponse r = transport.exchange(method, full, headers, raw);
        if (!r.ok()) {
            throw DockerException.ofHttp(method + " " + path, r.status, r.body);
        }
        if (r.body == null || r.body.isBlank()) return null;
        try {
            return Json.decode(r.body);
        } catch (IOException e) {
            return r.body;
        }
    }

    private static String enc(String s) {
        // path segment encode minimal
        return s.replace("/", "%2F").replace(" ", "%20");
    }

    @Override
    public void close() {
        transport.close();
    }

    // ---- transport ----

    private interface Transport extends AutoCloseable {
        EngineResponse exchange(String method, String path, Map<String, String> headers, byte[] body);

        @Override
        void close();

        static Transport open(DockerOptions opts) {
            String host = opts.effectiveHost();
            if (host == null || host.isBlank()) {
                host = DockerOptions.defaultUnixHost();
            }
            String h = host.toLowerCase(Locale.ROOT);
            if (h.startsWith("unix://") || h.startsWith("npipe://")) {
                if (h.startsWith("npipe://")) {
                    throw new DockerException(
                            "Windows named pipe Engine transport not implemented; use Docker CLI",
                            -1, "engine");
                }
                UnixSocketHttp http = UnixSocketHttp.open(host, opts.timeout);
                return new UnixTransport(http);
            }
            // tcp://127.0.0.1:2375 → http://127.0.0.1:2375
            String base = host;
            if (h.startsWith("tcp://")) {
                base = (opts.tlsVerify ? "https://" : "http://") + host.substring("tcp://".length());
            } else if (!h.startsWith("http://") && !h.startsWith("https://")) {
                base = "http://" + host;
            }
            return new HttpTransport(base, opts.timeout);
        }
    }

    private static final class UnixTransport implements Transport {
        private final UnixSocketHttp http;

        UnixTransport(UnixSocketHttp http) { this.http = http; }

        @Override
        public EngineResponse exchange(String method, String path, Map<String, String> headers, byte[] body) {
            try {
                UnixSocketHttp.Response r = http.exchange(method, path, headers, body);
                return new EngineResponse(r.status, r.bodyString());
            } catch (IOException e) {
                throw new DockerException("unix engine I/O: " + e.getMessage(), e, -1, -1, "engine");
            }
        }

        @Override
        public void close() { http.close(); }
    }

    private static final class HttpTransport implements Transport {
        private final HttpClient client;
        private final URI base;
        private final Duration timeout;

        HttpTransport(String baseUrl, Duration timeout) {
            String b = baseUrl.endsWith("/") ? baseUrl.substring(0, baseUrl.length() - 1) : baseUrl;
            this.base = URI.create(b);
            this.timeout = timeout == null ? Duration.ofSeconds(30) : timeout;
            this.client = HttpClient.newBuilder()
                    .connectTimeout(this.timeout)
                    .followRedirects(HttpClient.Redirect.NORMAL)
                    .build();
        }

        @Override
        public EngineResponse exchange(String method, String path, Map<String, String> headers, byte[] body) {
            try {
                String p = path == null ? "/" : path;
                URI uri = base.resolve(p);
                HttpRequest.Builder rb = HttpRequest.newBuilder(uri).timeout(timeout);
                if (headers != null) {
                    for (Map.Entry<String, String> h : headers.entrySet()) {
                        if (h.getKey() != null && h.getValue() != null) {
                            rb.header(h.getKey(), h.getValue());
                        }
                    }
                }
                String m = method == null ? "GET" : method.toUpperCase(Locale.ROOT);
                if ("GET".equals(m) || "DELETE".equals(m) || "HEAD".equals(m)) {
                    if (body == null || body.length == 0) {
                        rb.method(m, HttpRequest.BodyPublishers.noBody());
                    } else {
                        rb.method(m, HttpRequest.BodyPublishers.ofByteArray(body));
                    }
                } else {
                    rb.method(m, body == null
                            ? HttpRequest.BodyPublishers.noBody()
                            : HttpRequest.BodyPublishers.ofByteArray(body));
                }
                HttpResponse<String> resp = client.send(rb.build(),
                        HttpResponse.BodyHandlers.ofString(StandardCharsets.UTF_8));
                return new EngineResponse(resp.statusCode(), resp.body() == null ? "" : resp.body());
            } catch (IOException e) {
                throw new DockerException("http engine I/O: " + e.getMessage(), e, -1, -1, "engine");
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                throw new DockerException("http engine interrupted", e, -1, -1, "engine");
            }
        }

        @Override
        public void close() { /* HttpClient no-op */ }
    }

    static final class EngineResponse {
        final int status;
        final String body;

        EngineResponse(int status, String body) {
            this.status = status;
            this.body = body == null ? "" : body;
        }

        boolean ok() { return status >= 200 && status < 300; }
    }
}
