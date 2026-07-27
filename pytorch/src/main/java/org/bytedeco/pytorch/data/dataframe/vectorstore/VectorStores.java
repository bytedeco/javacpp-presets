package org.bytedeco.pytorch.data.dataframe.vectorstore;

import org.bytedeco.pytorch.data.dataframe.vectorstore.memory.InMemoryVectorStore;
import org.bytedeco.pytorch.data.dataframe.vectorstore.milvus.MilvusVectorStore;
import org.bytedeco.pytorch.data.dataframe.vectorstore.mongo.MongoAtlasVectorStore;
import org.bytedeco.pytorch.data.dataframe.vectorstore.opensearch.OpenSearchVectorStore;
import org.bytedeco.pytorch.data.dataframe.vectorstore.pgvector.PgVectorStore;
import org.bytedeco.pytorch.data.dataframe.vectorstore.qdrant.QdrantVectorStore;
import org.bytedeco.pytorch.data.dataframe.vectorstore.redis.RedisVectorStore;

import java.time.Duration;
import java.util.LinkedHashMap;
import java.util.Locale;
import java.util.Map;
import java.util.ServiceLoader;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Factory for built-in zero-SDK vector stores + {@link ServiceLoader} plugins.
 *
 * <h2>Why no vendor JARs?</h2>
 * Bundling Jedis / milvus-sdk-java / qdrant-client / opensearch-java / mongo-driver
 * would pull tens of MB of transitive deps into every javacpp-pytorch user.
 * Instead we speak the wire protocols directly:
 * <ul>
 *   <li>HTTP + JSON — Qdrant, Milvus REST v2, OpenSearch, Mongo Atlas Data API</li>
 *   <li>RESP2 — Redis / RediSearch (tiny hand-rolled client)</li>
 *   <li>JDBC — pgvector (driver stays on the <em>app</em> classpath)</li>
 *   <li>In-process HNSW — pure Java, always available</li>
 * </ul>
 *
 * <p>If you already depend on a full SDK, implement {@link VectorStoreProvider}
 * and register it via {@code META-INF/services}; then {@link #open(String, Map)}
 * will find it.
 *
 * <pre>{@code
 * // in-process
 * VectorStore mem = VectorStores.memory("local", 384, VectorMetric.COSINE);
 *
 * // qdrant
 * VectorStore q = VectorStores.qdrant("http://localhost:6333", "clips", 768, VectorMetric.COSINE);
 *
 * // redis stack
 * VectorStore r = VectorStores.redis("127.0.0.1", 6379, "idx:clips", 768, VectorMetric.COSINE);
 *
 * // generic
 * VectorStore vs = VectorStores.open("milvus", Map.of(
 *     "url", "http://localhost:9091",
 *     "collection", "clips",
 *     "dim", 768,
 *     "token", "root:Milvus"
 * ));
 * }</pre>
 */
public final class VectorStores {
    private VectorStores() {}

    private static final Map<String, VectorStoreProvider> PLUGINS = new ConcurrentHashMap<>();
    static {
        reloadProviders();
    }

    /** Re-scan {@link ServiceLoader} for {@link VectorStoreProvider} plugins. */
    public static void reloadProviders() {
        PLUGINS.clear();
        try {
            for (VectorStoreProvider p : ServiceLoader.load(VectorStoreProvider.class)) {
                if (p != null && p.name() != null) {
                    PLUGINS.put(p.name().toLowerCase(Locale.ROOT), p);
                }
            }
        } catch (Throwable ignored) {
            // broken provider on classpath — ignore
        }
    }

    public static void registerProvider(VectorStoreProvider provider) {
        if (provider != null && provider.name() != null) {
            PLUGINS.put(provider.name().toLowerCase(Locale.ROOT), provider);
        }
    }

    // ── convenience factories ────────────────────────────────────────────

    public static VectorStore memory(String name, int dim, VectorMetric metric) {
        return InMemoryVectorStore.builder(dim).name(name).metric(metric).build();
    }

    public static VectorStore memory(int dim) {
        return memory("memory", dim, VectorMetric.L2);
    }

    public static VectorStore qdrant(String url, String collection, int dim, VectorMetric metric) {
        return QdrantVectorStore.builder(url).collection(collection).dim(dim).metric(metric).build();
    }

    public static VectorStore qdrant(String url, String collection, int dim, VectorMetric metric, String apiKey) {
        return QdrantVectorStore.builder(url).collection(collection).dim(dim).metric(metric).apiKey(apiKey).build();
    }

    public static VectorStore redis(String host, int port, String index, int dim, VectorMetric metric) {
        return RedisVectorStore.builder().host(host).port(port).index(index).dim(dim).metric(metric).build();
    }

    public static VectorStore redis(String host, int port, String index, int dim, VectorMetric metric,
                                    String password) {
        return RedisVectorStore.builder().host(host).port(port).index(index).dim(dim).metric(metric)
            .password(password).build();
    }

    public static VectorStore milvus(String url, String collection, int dim, VectorMetric metric) {
        return MilvusVectorStore.builder(url).collection(collection).dim(dim).metric(metric).build();
    }

    public static VectorStore milvus(String url, String collection, int dim, VectorMetric metric, String token) {
        return MilvusVectorStore.builder(url).collection(collection).dim(dim).metric(metric).token(token).build();
    }

    public static VectorStore openSearch(String url, String index, int dim, VectorMetric metric) {
        return OpenSearchVectorStore.builder(url).index(index).dim(dim).metric(metric).build();
    }

    public static VectorStore openSearch(String url, String index, int dim, VectorMetric metric,
                                         String user, String password) {
        return OpenSearchVectorStore.builder(url).index(index).dim(dim).metric(metric)
            .basicAuth(user, password).build();
    }

    public static VectorStore pgvector(String jdbcUrl, String user, String password,
                                       String table, int dim, VectorMetric metric) {
        return PgVectorStore.builder().url(jdbcUrl).user(user).password(password)
            .table(table).dim(dim).metric(metric).build();
    }

    public static VectorStore mongoAtlas(String dataApiUrl, String apiKey,
                                         String dataSource, String database, String collection,
                                         int dim, VectorMetric metric) {
        return MongoAtlasVectorStore.builder(dataApiUrl)
            .apiKey(apiKey).dataSource(dataSource).database(database).collection(collection)
            .dim(dim).metric(metric).build();
    }

    /**
     * Open by scheme name + free-form config.
     *
     * <p>Built-in schemes: {@code memory}, {@code qdrant}, {@code redis},
     * {@code milvus}, {@code opensearch}, {@code pgvector}/{@code postgres},
     * {@code mongo}/{@code mongodb}/{@code atlas}.
     * Anything else is resolved via {@link VectorStoreProvider} plugins.
     */
    public static VectorStore open(String scheme, Map<String, Object> config) {
        if (scheme == null || scheme.isBlank()) {
            throw new VectorStoreException("scheme required");
        }
        String s = scheme.toLowerCase(Locale.ROOT).trim();
        Map<String, Object> cfg = config == null ? Map.of() : config;

        return switch (s) {
            case "memory", "hnsw", "local" -> openMemory(cfg);
            case "qdrant" -> openQdrant(cfg);
            case "redis", "redisearch" -> openRedis(cfg);
            case "milvus", "zilliz" -> openMilvus(cfg);
            case "opensearch", "elasticsearch", "es" -> openOpenSearch(cfg);
            case "pgvector", "postgres", "postgresql", "pg" -> openPg(cfg);
            case "mongo", "mongodb", "atlas", "mongoatlas" -> openMongo(cfg);
            default -> {
                VectorStoreProvider p = PLUGINS.get(s);
                if (p == null) {
                    throw new VectorStoreException(
                        "Unknown vector store scheme '" + scheme
                            + "'. Built-ins: memory, qdrant, redis, milvus, opensearch, pgvector, mongo. "
                            + "Or register a VectorStoreProvider SPI plugin.");
                }
                yield p.open(cfg);
            }
        };
    }

    /** Parse a simple URI-like spec: {@code qdrant://localhost:6333/clips?dim=768&metric=cosine}. */
    public static VectorStore open(String uri) {
        if (uri == null || uri.isBlank()) throw new VectorStoreException("uri required");
        // scheme://host:port/path?k=v
        int schemeEnd = uri.indexOf("://");
        if (schemeEnd <= 0) {
            // bare scheme name with no authority — treat as open(scheme, empty)
            return open(uri, Map.of());
        }
        String scheme = uri.substring(0, schemeEnd);
        String rest = uri.substring(schemeEnd + 3);
        String path = rest;
        String query = null;
        int q = rest.indexOf('?');
        if (q >= 0) {
            path = rest.substring(0, q);
            query = rest.substring(q + 1);
        }
        Map<String, Object> cfg = new LinkedHashMap<>();
        // authority + path
        String hostPort = path;
        String collection = null;
        int slash = path.indexOf('/');
        if (slash >= 0) {
            hostPort = path.substring(0, slash);
            collection = path.substring(slash + 1);
            if (collection.isEmpty()) collection = null;
        }
        if (!hostPort.isEmpty()) {
            // rebuild url for HTTP backends
            String urlScheme = switch (scheme.toLowerCase(Locale.ROOT)) {
                case "redis", "redisearch" -> "redis";
                case "pgvector", "postgres", "postgresql", "pg" -> "jdbc:postgresql";
                case "mongo", "mongodb", "atlas" -> "https";
                default -> hostPort.startsWith("http") ? "" : "http";
            };
            if ("redis".equals(urlScheme)) {
                String host = hostPort;
                int port = 6379;
                int colon = hostPort.lastIndexOf(':');
                if (colon > 0) {
                    host = hostPort.substring(0, colon);
                    try { port = Integer.parseInt(hostPort.substring(colon + 1)); } catch (Exception ignored) {}
                }
                cfg.put("host", host);
                cfg.put("port", port);
            } else if (urlScheme.startsWith("jdbc")) {
                cfg.put("url", urlScheme + "://" + hostPort + (collection != null ? "/" + collection : ""));
                // collection segment for jdbc is db name; table comes from query
            } else if (!urlScheme.isEmpty()) {
                cfg.put("url", urlScheme + "://" + hostPort);
            } else {
                cfg.put("url", hostPort);
            }
        }
        if (collection != null) {
            cfg.put("collection", collection);
            cfg.put("index", collection);
            cfg.put("table", collection);
            cfg.put("name", collection);
        }
        if (query != null) {
            for (String pair : query.split("&")) {
                if (pair.isEmpty()) continue;
                int eq = pair.indexOf('=');
                String k = eq < 0 ? pair : pair.substring(0, eq);
                String v = eq < 0 ? "" : pair.substring(eq + 1);
                k = urlDecode(k);
                v = urlDecode(v);
                cfg.put(k, coerce(v));
            }
        }
        return open(scheme, cfg);
    }

    // ── internal openers ─────────────────────────────────────────────────

    private static VectorStore openMemory(Map<String, Object> cfg) {
        int dim = intVal(cfg, "dim", 0);
        if (dim <= 0) throw new VectorStoreException("memory store requires dim");
        String name = strVal(cfg, "name", strVal(cfg, "collection", "memory"));
        VectorMetric metric = metricVal(cfg);
        int M = intVal(cfg, "M", 16);
        int ef = intVal(cfg, "efConstruction", 200);
        boolean norm = boolVal(cfg, "normalize", false);
        return InMemoryVectorStore.builder(dim).name(name).metric(metric).M(M)
            .efConstruction(ef).normalize(norm).build();
    }

    private static VectorStore openQdrant(Map<String, Object> cfg) {
        String url = strVal(cfg, "url", "http://localhost:6333");
        String collection = strVal(cfg, "collection", strVal(cfg, "name", "vectors"));
        int dim = intVal(cfg, "dim", 0);
        return QdrantVectorStore.builder(url)
            .collection(collection)
            .dim(dim)
            .metric(metricVal(cfg))
            .apiKey(strVal(cfg, "apiKey", strVal(cfg, "api_key", null)))
            .timeout(durationVal(cfg))
            .build();
    }

    private static VectorStore openRedis(Map<String, Object> cfg) {
        return RedisVectorStore.builder()
            .host(strVal(cfg, "host", "127.0.0.1"))
            .port(intVal(cfg, "port", 6379))
            .username(strVal(cfg, "username", null))
            .password(strVal(cfg, "password", null))
            .index(strVal(cfg, "index", strVal(cfg, "collection", "idx:vectors")))
            .prefix(strVal(cfg, "prefix", "doc:"))
            .vectorField(strVal(cfg, "vectorField", "vector"))
            .dim(intVal(cfg, "dim", 0))
            .metric(metricVal(cfg))
            .algorithm(strVal(cfg, "algorithm", "HNSW"))
            .M(intVal(cfg, "M", 16))
            .efConstruction(intVal(cfg, "efConstruction", 200))
            .timeout(durationVal(cfg))
            .build();
    }

    private static VectorStore openMilvus(Map<String, Object> cfg) {
        String url = strVal(cfg, "url", "http://localhost:9091");
        return MilvusVectorStore.builder(url)
            .collection(strVal(cfg, "collection", "vectors"))
            .dbName(strVal(cfg, "dbName", strVal(cfg, "database", "default")))
            .dim(intVal(cfg, "dim", 0))
            .metric(metricVal(cfg))
            .token(strVal(cfg, "token", null))
            .apiKey(strVal(cfg, "apiKey", strVal(cfg, "api_key", null)))
            .timeout(durationVal(cfg))
            .build();
    }

    private static VectorStore openOpenSearch(Map<String, Object> cfg) {
        String url = strVal(cfg, "url", "http://localhost:9200");
        OpenSearchVectorStore.Builder b = OpenSearchVectorStore.builder(url)
            .index(strVal(cfg, "index", strVal(cfg, "collection", "vectors")))
            .dim(intVal(cfg, "dim", 0))
            .metric(metricVal(cfg))
            .vectorField(strVal(cfg, "vectorField", "vector"))
            .engine(strVal(cfg, "engine", "faiss"))
            .timeout(durationVal(cfg));
        String user = strVal(cfg, "username", strVal(cfg, "user", null));
        if (user != null) b.basicAuth(user, strVal(cfg, "password", ""));
        String apiKey = strVal(cfg, "apiKey", strVal(cfg, "api_key", null));
        if (apiKey != null) b.apiKey(apiKey);
        return b.build();
    }

    private static VectorStore openPg(Map<String, Object> cfg) {
        return PgVectorStore.builder()
            .url(strVal(cfg, "url", strVal(cfg, "jdbcUrl", null)))
            .user(strVal(cfg, "user", strVal(cfg, "username", null)))
            .password(strVal(cfg, "password", null))
            .table(strVal(cfg, "table", strVal(cfg, "collection", "vectors")))
            .dim(intVal(cfg, "dim", 0))
            .metric(metricVal(cfg))
            .build();
    }

    private static VectorStore openMongo(Map<String, Object> cfg) {
        String url = strVal(cfg, "url", null);
        if (url == null) throw new VectorStoreException("mongo store requires url (Data API base)");
        return MongoAtlasVectorStore.builder(url)
            .apiKey(strVal(cfg, "apiKey", strVal(cfg, "api_key", null)))
            .dataSource(strVal(cfg, "dataSource", strVal(cfg, "cluster", "Cluster0")))
            .database(strVal(cfg, "database", strVal(cfg, "db", "test")))
            .collection(strVal(cfg, "collection", "vectors"))
            .dim(intVal(cfg, "dim", 0))
            .metric(metricVal(cfg))
            .vectorPath(strVal(cfg, "vectorPath", strVal(cfg, "vectorField", "embedding")))
            .indexName(strVal(cfg, "indexName", strVal(cfg, "index", "vector_index")))
            .timeout(durationVal(cfg))
            .build();
    }

    // ── config helpers ───────────────────────────────────────────────────

    private static String strVal(Map<String, Object> cfg, String key, String def) {
        Object v = cfg.get(key);
        if (v == null) return def;
        String s = String.valueOf(v);
        return s.isEmpty() ? def : s;
    }

    private static int intVal(Map<String, Object> cfg, String key, int def) {
        Object v = cfg.get(key);
        if (v instanceof Number n) return n.intValue();
        if (v instanceof String s) {
            try { return Integer.parseInt(s.trim()); } catch (NumberFormatException ignored) {}
        }
        return def;
    }

    private static boolean boolVal(Map<String, Object> cfg, String key, boolean def) {
        Object v = cfg.get(key);
        if (v instanceof Boolean b) return b;
        if (v instanceof String s) return Boolean.parseBoolean(s) || "1".equals(s) || "yes".equalsIgnoreCase(s);
        return def;
    }

    private static VectorMetric metricVal(Map<String, Object> cfg) {
        Object v = cfg.get("metric");
        if (v == null) v = cfg.get("distance");
        if (v == null) v = cfg.get("space");
        if (v instanceof VectorMetric m) return m;
        if (v == null) return VectorMetric.COSINE;
        String s = String.valueOf(v).trim().toUpperCase(Locale.ROOT);
        return switch (s) {
            case "L2", "EUCLID", "EUCLIDEAN", "SQUARED_L2" -> VectorMetric.L2;
            case "IP", "DOT", "INNER_PRODUCT", "INNERPRODUCT", "DOTPRODUCT" -> VectorMetric.IP;
            case "COSINE", "COS", "COSINESIMIL" -> VectorMetric.COSINE;
            default -> VectorMetric.COSINE;
        };
    }

    private static Duration durationVal(Map<String, Object> cfg) {
        Object v = cfg.get("timeout");
        if (v instanceof Duration d) return d;
        if (v instanceof Number n) return Duration.ofMillis(n.longValue());
        if (v instanceof String s) {
            try { return Duration.ofMillis(Long.parseLong(s.trim())); } catch (Exception ignored) {}
        }
        return Duration.ofSeconds(30);
    }

    private static Object coerce(String v) {
        if (v == null) return null;
        if ("true".equalsIgnoreCase(v) || "false".equalsIgnoreCase(v)) return Boolean.parseBoolean(v);
        try { return Integer.parseInt(v); } catch (NumberFormatException ignored) {}
        try { return Long.parseLong(v); } catch (NumberFormatException ignored) {}
        try { return Double.parseDouble(v); } catch (NumberFormatException ignored) {}
        return v;
    }

    private static String urlDecode(String s) {
        try {
            return java.net.URLDecoder.decode(s, java.nio.charset.StandardCharsets.UTF_8);
        } catch (Exception e) {
            return s;
        }
    }
}
