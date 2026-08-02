/*
 * Copyright (C) 2020-2026 Eduardo Gonzalez, Hervé Guillemet, Samuel Audet, mullerhai
 *
 * Licensed either under the Apache License, Version 2.0, or (at your option)
 * under the terms of the GNU General Public License as published by
 * the Free Software Foundation (subject to the "Classpath" exception),
 * either version 2, or any later version (collectively, the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *     http://www.gnu.org/licenses/
 *     http://www.gnu.org/software/classpath/license.html
 *
 * or as provided in the LICENSE.txt file that accompanied this code.
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.bytedeco.pytorch.utils.gravitino;

import com.google.gson.Gson;
import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import org.bytedeco.pytorch.utils.lake.LakeException;
import org.bytedeco.pytorch.utils.lake.LakeFormat;

import java.io.IOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;
import java.nio.file.DirectoryStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.ArrayList;
import java.util.Base64;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Gravitino metalake client: REST (JDK HttpClient + Gson) or offline mock registry.
 *
 * <p>Public REST paths align with Apache Gravitino docs:
 * {@code /api/metalakes/{metalake}/catalogs[/{catalog}/schemas[/{schema}/tables[/{table}]]]}.</p>
 *
 * @see <a href="https://gravitino.apache.org/docs/latest/api/rest/">Gravitino REST</a>
 */
public final class GravitinoMetalake implements AutoCloseable {

    private static final Gson GSON = new Gson();

    private final GravitinoOptions options;
    private final HttpClient http;
    private final Map<String, JsonObject> mockTables;
    private final boolean mockMode;

    public GravitinoMetalake(GravitinoOptions options) {
        this.options = Objects.requireNonNull(options, "options");
        this.mockMode = options.mockRegistryPath() != null && !options.mockRegistryPath().isBlank();
        this.mockTables = new ConcurrentHashMap<>();
        if (mockMode) {
            loadMockRegistry(Path.of(options.mockRegistryPath()));
            this.http = null;
        } else {
            if (options.uri() == null || options.uri().isBlank()) {
                throw new LakeException(LakeFormat.GRAVITINO, "metalake",
                        "uri required (or set mock_registry for offline mode)");
            }
            Duration connect = Duration.ofMillis(Math.max(1, options.connectTimeoutMs()));
            this.http = HttpClient.newBuilder()
                    .connectTimeout(connect)
                    .followRedirects(HttpClient.Redirect.NORMAL)
                    .build();
        }
    }

    public GravitinoOptions options() {
        return options;
    }

    public boolean isMockMode() {
        return mockMode;
    }

    /** Register a mock table entry (offline tests / local federation). */
    public void registerMockTable(String fullName, String provider, String location,
                                  Map<String, String> properties) {
        Map<String, Object> entry = GravitinoResolver.mockTableEntry(
                fullName, provider, location, null);
        if (properties != null) {
            @SuppressWarnings("unchecked")
            Map<String, String> props = (Map<String, String>) entry.get("properties");
            props.putAll(properties);
        }
        mockTables.put(fullName, GSON.toJsonTree(entry).getAsJsonObject());
    }

    public List<String> listCatalogs() {
        if (mockMode) {
            return distinctSegment(0);
        }
        String metalake = requireMetalake();
        JsonObject body = getJson(apiBase() + "/" + enc(metalake) + "/catalogs");
        return readNameList(body, "catalogs", "identifiers");
    }

    public List<String> listSchemas(String catalog) {
        if (mockMode) {
            return distinctSegment(1, catalog);
        }
        String metalake = requireMetalake();
        JsonObject body = getJson(apiBase() + "/" + enc(metalake) + "/catalogs/"
                + enc(catalog) + "/schemas");
        return readNameList(body, "schemas", "identifiers");
    }

    public List<String> listTables(String catalog, String schema) {
        if (mockMode) {
            return distinctSegment(2, catalog, schema);
        }
        String metalake = requireMetalake();
        JsonObject body = getJson(apiBase() + "/" + enc(metalake) + "/catalogs/"
                + enc(catalog) + "/schemas/" + enc(schema) + "/tables");
        return readNameList(body, "tables", "identifiers");
    }

    /**
     * Load table metadata JSON (provider, location, properties, columns).
     */
    public JsonObject loadTableMeta(String catalog, String schema, String table) {
        String full = buildFullName(catalog, schema, table);
        if (mockMode) {
            JsonObject mock = mockTables.get(full);
            if (mock == null) {
                // try short name
                mock = mockTables.get(table);
            }
            if (mock == null) {
                throw new LakeException(LakeFormat.GRAVITINO, "loadTable",
                        "mock table not found: " + full);
            }
            return mock;
        }
        String metalake = requireMetalake();
        return getJson(apiBase() + "/" + enc(metalake) + "/catalogs/"
                + enc(catalog) + "/schemas/" + enc(schema) + "/tables/" + enc(table));
    }

    public GravitinoResolver.Resolved resolveTable(String catalog, String schema, String table) {
        JsonObject meta = loadTableMeta(catalog, schema, table);
        String full = buildFullName(catalog, schema, table);
        String provider = text(meta, "provider");
        if (provider == null && meta.has("table")) {
            provider = text(meta.getAsJsonObject("table"), "provider");
        }
        String location = text(meta, "storageLocation");
        if (location == null) location = text(meta, "location");
        if (location == null && meta.has("table")) {
            JsonObject t = meta.getAsJsonObject("table");
            location = text(t, "storageLocation");
            if (location == null) location = text(t, "location");
        }
        Map<String, String> props = readProperties(meta);
        if (meta.has("table")) {
            props.putAll(readProperties(meta.getAsJsonObject("table")));
        }
        if (provider == null) provider = props.get("provider");
        if (location == null) location = props.get("location");
        return GravitinoResolver.resolve(full, provider, location, props, options);
    }

    private void loadMockRegistry(Path path) {
        try {
            if (Files.isRegularFile(path)) {
                String raw = Files.readString(path, StandardCharsets.UTF_8);
                ingestMockJson(raw);
            } else if (Files.isDirectory(path)) {
                try (DirectoryStream<Path> stream = Files.newDirectoryStream(path, "*.json")) {
                    for (Path p : stream) {
                        ingestMockJson(Files.readString(p, StandardCharsets.UTF_8));
                    }
                }
            } else {
                // create empty dir so registerMockTable works
                Files.createDirectories(path);
            }
        } catch (IOException e) {
            throw new LakeException(LakeFormat.GRAVITINO, "mockRegistry",
                    "failed to load " + path, e);
        }
    }

    private void ingestMockJson(String raw) {
        JsonElement el = JsonParser.parseString(raw);
        if (el.isJsonArray()) {
            for (JsonElement e : el.getAsJsonArray()) {
                if (e.isJsonObject()) putMock(e.getAsJsonObject());
            }
        } else if (el.isJsonObject()) {
            JsonObject o = el.getAsJsonObject();
            if (o.has("tables") && o.get("tables").isJsonArray()) {
                for (JsonElement e : o.getAsJsonArray("tables")) {
                    if (e.isJsonObject()) putMock(e.getAsJsonObject());
                }
            } else {
                putMock(o);
            }
        }
    }

    private void putMock(JsonObject o) {
        String name = text(o, "name");
        if (name == null) name = text(o, "fullName");
        if (name == null) return;
        mockTables.put(name, o);
    }

    private List<String> distinctSegment(int index, String... prefix) {
        List<String> out = new ArrayList<>();
        for (String full : mockTables.keySet()) {
            String[] parts = full.split("\\.");
            // full may be metalake.catalog.schema.table → shift if metalake present
            int off = 0;
            if (options.metalake() != null && parts.length > 0
                    && parts[0].equals(options.metalake())) {
                off = 1;
            }
            int idx = off + index;
            if (parts.length <= idx) continue;
            boolean ok = true;
            for (int i = 0; i < prefix.length; i++) {
                int pi = off + i;
                if (pi >= parts.length || !parts[pi].equals(prefix[i])) {
                    ok = false;
                    break;
                }
            }
            if (!ok) continue;
            String seg = parts[idx];
            if (!out.contains(seg)) out.add(seg);
        }
        return List.copyOf(out);
    }

    private String buildFullName(String catalog, String schema, String table) {
        StringBuilder sb = new StringBuilder();
        if (options.metalake() != null) sb.append(options.metalake()).append('.');
        if (catalog != null) sb.append(catalog).append('.');
        if (schema != null) sb.append(schema).append('.');
        sb.append(table == null ? "" : table);
        return sb.toString();
    }

    private String requireMetalake() {
        if (options.metalake() == null || options.metalake().isBlank()) {
            throw new LakeException(LakeFormat.GRAVITINO, "metalake", "metalake name required");
        }
        return options.metalake();
    }

    private String apiBase() {
        String base = options.uri();
        if (base.endsWith("/")) base = base.substring(0, base.length() - 1);
        String prefix = options.apiPrefix();
        if (!prefix.startsWith("/")) prefix = "/" + prefix;
        return base + prefix;
    }

    private JsonObject getJson(String url) {
        try {
            HttpRequest.Builder rb = HttpRequest.newBuilder()
                    .uri(URI.create(url))
                    .timeout(Duration.ofMillis(Math.max(1, options.socketTimeoutMs())))
                    .GET()
                    .header("Accept", "application/vnd.gravitino.v1+json, application/json");
            applyAuth(rb);
            HttpResponse<String> resp = http.send(rb.build(), HttpResponse.BodyHandlers.ofString());
            if (resp.statusCode() >= 400) {
                throw new LakeException(LakeFormat.GRAVITINO, "http.get",
                        "HTTP " + resp.statusCode() + " for " + url + ": " + truncate(resp.body()));
            }
            JsonElement el = JsonParser.parseString(resp.body() == null ? "{}" : resp.body());
            return el.isJsonObject() ? el.getAsJsonObject() : new JsonObject();
        } catch (LakeException e) {
            throw e;
        } catch (Exception e) {
            throw new LakeException(LakeFormat.GRAVITINO, "http.get",
                    "failed GET " + url, e);
        }
    }

    private void applyAuth(HttpRequest.Builder rb) {
        if (options.authToken() != null && !options.authToken().isBlank()) {
            rb.header("Authorization", "Bearer " + options.authToken());
        } else if (options.username() != null) {
            String raw = options.username() + ":" + (options.password() == null ? "" : options.password());
            String b64 = Base64.getEncoder().encodeToString(raw.getBytes(StandardCharsets.UTF_8));
            rb.header("Authorization", "Basic " + b64);
        }
    }

    private static List<String> readNameList(JsonObject body, String... keys) {
        if (body == null) return List.of();
        for (String key : keys) {
            if (!body.has(key)) continue;
            JsonElement el = body.get(key);
            if (el.isJsonArray()) {
                List<String> out = new ArrayList<>();
                JsonArray arr = el.getAsJsonArray();
                for (JsonElement e : arr) {
                    if (e.isJsonPrimitive()) out.add(e.getAsString());
                    else if (e.isJsonObject()) {
                        String n = text(e.getAsJsonObject(), "name");
                        if (n == null) n = text(e.getAsJsonObject(), "fullName");
                        if (n != null) {
                            // take last segment if dotted
                            int dot = n.lastIndexOf('.');
                            out.add(dot >= 0 ? n.substring(dot + 1) : n);
                        }
                    }
                }
                return List.copyOf(out);
            }
        }
        // nested under "data"
        if (body.has("data") && body.get("data").isJsonObject()) {
            return readNameList(body.getAsJsonObject("data"), keys);
        }
        return List.of();
    }

    static Map<String, String> readProperties(JsonObject obj) {
        if (obj == null || !obj.has("properties")) return new LinkedHashMap<>();
        JsonElement el = obj.get("properties");
        Map<String, String> m = new LinkedHashMap<>();
        if (el.isJsonObject()) {
            for (var e : el.getAsJsonObject().entrySet()) {
                if (e.getValue().isJsonPrimitive()) {
                    m.put(e.getKey(), e.getValue().getAsString());
                } else {
                    m.put(e.getKey(), e.getValue().toString());
                }
            }
        }
        return m;
    }

    static String text(JsonObject o, String key) {
        if (o == null || !o.has(key) || o.get(key).isJsonNull()) return null;
        JsonElement el = o.get(key);
        if (el.isJsonPrimitive()) return el.getAsString();
        return null;
    }

    private static String enc(String s) {
        return java.net.URLEncoder.encode(s, StandardCharsets.UTF_8).replace("+", "%20");
    }

    private static String truncate(String s) {
        if (s == null) return "";
        return s.length() > 200 ? s.substring(0, 200) + "..." : s;
    }

    public Map<String, JsonObject> mockTablesView() {
        return Collections.unmodifiableMap(mockTables);
    }

    @Override
    public void close() {
        // HttpClient needs no explicit close on older JDKs; mock has nothing
    }
}
