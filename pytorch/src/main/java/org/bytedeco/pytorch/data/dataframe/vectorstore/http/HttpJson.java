package org.bytedeco.pytorch.data.dataframe.vectorstore.http;

import org.bytedeco.pytorch.data.dataframe.vectorstore.VectorStoreException;
import org.bytedeco.pytorch.utils.json.Json;

import java.io.IOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Tiny JSON-over-HTTP helper used by Qdrant / Milvus / OpenSearch / Mongo adapters.
 * Reuses the same zero-dep {@link Json} codec as WandB / Visdom clients.
 */
public final class HttpJson implements AutoCloseable {

    private final HttpClient http;
    private final URI base;
    private final Map<String, String> defaultHeaders;
    private final Duration timeout;
    private final String backend;

    public HttpJson(String baseUrl, String backend, Duration timeout, Map<String, String> headers) {
        Objects.requireNonNull(baseUrl, "baseUrl");
        String b = baseUrl.endsWith("/") ? baseUrl.substring(0, baseUrl.length() - 1) : baseUrl;
        this.base = URI.create(b);
        this.backend = backend == null ? "http" : backend;
        this.timeout = timeout == null ? Duration.ofSeconds(30) : timeout;
        this.defaultHeaders = headers == null ? Map.of() : Map.copyOf(headers);
        this.http = HttpClient.newBuilder()
            .connectTimeout(this.timeout)
            .followRedirects(HttpClient.Redirect.NORMAL)
            .build();
    }

    public static Builder builder(String baseUrl) {
        return new Builder(baseUrl);
    }

    public URI base() { return base; }
    public String backend() { return backend; }

    public Object get(String path) {
        return exchange("GET", path, null);
    }

    public Object post(String path, Object body) {
        return exchange("POST", path, body);
    }

    public Object put(String path, Object body) {
        return exchange("PUT", path, body);
    }

    public Object patch(String path, Object body) {
        return exchange("PATCH", path, body);
    }

    public Object delete(String path) {
        return exchange("DELETE", path, null);
    }

    public Object delete(String path, Object body) {
        return exchange("DELETE", path, body);
    }

    public Object exchange(String method, String path, Object body) {
        return exchange(method, path, body, "application/json");
    }

    /**
     * POST raw body with an explicit Content-Type (e.g. OpenSearch
     * {@code application/x-ndjson} bulk API).
     */
    public Object postRaw(String path, String rawBody, String contentType) {
        return exchange("POST", path, rawBody, contentType == null ? "application/json" : contentType);
    }

    /** OpenSearch / ES {@code _bulk} helper — body must already be NDJSON (+ trailing newline). */
    public Object postNdjson(String path, String ndjson) {
        return postRaw(path, ndjson, "application/x-ndjson");
    }

    public Object exchange(String method, String path, Object body, String contentType) {
        try {
            String p = path == null ? "" : path;
            if (!p.startsWith("/")) p = "/" + p;
            URI uri = base.resolve(p);

            HttpRequest.Builder rb = HttpRequest.newBuilder(uri)
                .timeout(timeout)
                .header("Accept", "application/json");

            for (Map.Entry<String, String> h : defaultHeaders.entrySet()) {
                if (h.getKey() != null && h.getValue() != null) {
                    rb.header(h.getKey(), h.getValue());
                }
            }

            String m = method == null ? "GET" : method.toUpperCase();
            if ("GET".equals(m) || ("DELETE".equals(m) && body == null)) {
                rb.method(m, HttpRequest.BodyPublishers.noBody());
            } else {
                String payload;
                String ct = contentType == null ? "application/json" : contentType;
                if (body == null) {
                    payload = "application/x-ndjson".equals(ct) ? "" : "{}";
                } else if (body instanceof String s) {
                    // raw string body (NDJSON or pre-encoded JSON)
                    payload = s;
                } else if (body instanceof byte[] bytes) {
                    rb.header("Content-Type", ct);
                    rb.method(m, HttpRequest.BodyPublishers.ofByteArray(bytes));
                    return send(rb, m, uri);
                } else {
                    payload = Json.encode(body);
                    ct = "application/json";
                }
                rb.header("Content-Type", ct);
                rb.method(m, HttpRequest.BodyPublishers.ofString(payload, StandardCharsets.UTF_8));
            }

            return send(rb, m, uri);
        } catch (VectorStoreException e) {
            throw e;
        } catch (IOException e) {
            throw new VectorStoreException(backend + " I/O: " + e.getMessage(), e, -1, backend);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new VectorStoreException(backend + " interrupted", e, -1, backend);
        }
    }

    private Object send(HttpRequest.Builder rb, String m, URI uri) throws IOException, InterruptedException {
        HttpResponse<String> resp = http.send(rb.build(), HttpResponse.BodyHandlers.ofString(StandardCharsets.UTF_8));
        int code = resp.statusCode();
        String text = resp.body() == null ? "" : resp.body();
        if (code >= 200 && code < 300) {
            if (text.isBlank()) return null;
            try {
                return Json.decode(text);
            } catch (Exception parseEx) {
                return text;
            }
        }
        throw new VectorStoreException(
            backend + " HTTP " + code + " " + m + " " + uri + ": " + truncate(text, 500),
            code, backend);
    }

    // ---- JSON navigation helpers ----

    @SuppressWarnings("unchecked")
    public static Map<String, Object> asMap(Object o) {
        if (o == null) return Map.of();
        if (o instanceof Map<?, ?> m) return (Map<String, Object>) m;
        throw new VectorStoreException("expected JSON object, got " + o.getClass().getSimpleName());
    }

    @SuppressWarnings("unchecked")
    public static List<Object> asList(Object o) {
        if (o == null) return List.of();
        if (o instanceof List<?> l) return (List<Object>) l;
        throw new VectorStoreException("expected JSON array, got " + o.getClass().getSimpleName());
    }

    public static Object dig(Object root, String... path) {
        Object cur = root;
        for (String p : path) {
            if (cur == null) return null;
            if (cur instanceof Map<?, ?> m) {
                cur = m.get(p);
            } else {
                return null;
            }
        }
        return cur;
    }

    public static String asString(Object o) {
        return o == null ? null : String.valueOf(o);
    }

    public static int asInt(Object o, int def) {
        if (o instanceof Number n) return n.intValue();
        if (o instanceof String s) {
            try { return Integer.parseInt(s.trim()); } catch (NumberFormatException ignored) {}
        }
        return def;
    }

    public static long asLong(Object o, long def) {
        if (o instanceof Number n) return n.longValue();
        if (o instanceof String s) {
            try { return Long.parseLong(s.trim()); } catch (NumberFormatException ignored) {}
        }
        return def;
    }

    public static double asDouble(Object o, double def) {
        if (o instanceof Number n) return n.doubleValue();
        if (o instanceof String s) {
            try { return Double.parseDouble(s.trim()); } catch (NumberFormatException ignored) {}
        }
        return def;
    }

    public static float asFloat(Object o, float def) {
        return (float) asDouble(o, def);
    }

    public static float[] asFloatArray(Object o) {
        if (o == null) return null;
        if (o instanceof float[] f) return f;
        if (o instanceof double[] d) {
            float[] f = new float[d.length];
            for (int i = 0; i < d.length; i++) f[i] = (float) d[i];
            return f;
        }
        if (o instanceof List<?> list) {
            float[] f = new float[list.size()];
            for (int i = 0; i < list.size(); i++) {
                f[i] = asFloat(list.get(i), 0f);
            }
            return f;
        }
        return null;
    }

    public static List<Double> toDoubleList(float[] v) {
        java.util.ArrayList<Double> list = new java.util.ArrayList<>(v.length);
        for (float x : v) list.add((double) x);
        return list;
    }

    public static Map<String, Object> mapOf(Object... kv) {
        if (kv == null || kv.length == 0) return new LinkedHashMap<>();
        if ((kv.length & 1) != 0) throw new IllegalArgumentException("odd kv length");
        Map<String, Object> m = new LinkedHashMap<>();
        for (int i = 0; i < kv.length; i += 2) {
            m.put(String.valueOf(kv[i]), kv[i + 1]);
        }
        return m;
    }

    private static String truncate(String s, int max) {
        if (s == null) return "";
        return s.length() <= max ? s : s.substring(0, max) + "…";
    }

    @Override
    public void close() {
        // HttpClient has no close in Java 17; retained for AutoCloseable symmetry.
    }

    public static final class Builder {
        private final String baseUrl;
        private String backend = "http";
        private Duration timeout = Duration.ofSeconds(30);
        private final Map<String, String> headers = new LinkedHashMap<>();

        Builder(String baseUrl) { this.baseUrl = baseUrl; }

        public Builder backend(String b) { this.backend = b; return this; }
        public Builder timeout(Duration d) { this.timeout = d; return this; }
        public Builder header(String k, String v) {
            if (k != null && v != null) headers.put(k, v);
            return this;
        }
        public Builder bearer(String token) {
            if (token != null && !token.isEmpty()) {
                headers.put("Authorization", "Bearer " + token);
            }
            return this;
        }
        public Builder basic(String user, String pass) {
            String u = user == null ? "" : user;
            String p = pass == null ? "" : pass;
            String enc = java.util.Base64.getEncoder()
                .encodeToString((u + ":" + p).getBytes(StandardCharsets.UTF_8));
            headers.put("Authorization", "Basic " + enc);
            return this;
        }
        public Builder apiKey(String headerName, String key) {
            if (headerName != null && key != null) headers.put(headerName, key);
            return this;
        }

        public HttpJson build() {
            return new HttpJson(baseUrl, backend, timeout, headers);
        }
    }
}
