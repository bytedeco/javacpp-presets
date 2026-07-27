package org.bytedeco.pytorch.data.dataframe.vectorstore.redis;

import org.bytedeco.pytorch.data.dataframe.vectorstore.PayloadField;
import org.bytedeco.pytorch.data.dataframe.vectorstore.VectorHit;
import org.bytedeco.pytorch.data.dataframe.vectorstore.VectorMetric;
import org.bytedeco.pytorch.data.dataframe.vectorstore.VectorQuery;
import org.bytedeco.pytorch.data.dataframe.vectorstore.VectorRecord;
import org.bytedeco.pytorch.data.dataframe.vectorstore.VectorSearchResult;
import org.bytedeco.pytorch.data.dataframe.vectorstore.VectorStore;
import org.bytedeco.pytorch.data.dataframe.vectorstore.VectorStoreException;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.util.ArrayList;
import java.util.Collection;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Redis + RediSearch (Redis Stack) vector adapter — pure RESP, no Jedis.
 *
 * <p>Features:
 * <ul>
 *   <li>{@code FT.CREATE} with {@code VECTOR} + multi-field SCHEMA
 *       ({@link PayloadField} TEXT / TAG / NUMERIC)</li>
 *   <li>Pipelined {@code HSET} bulk upsert</li>
 *   <li>{@code FT.SEARCH} KNN (dialect 2), {@code HMGET}/{@code HGETALL} fetch,
 *       cursor-style scroll via {@code FT.SEARCH ... LIMIT}</li>
 * </ul>
 *
 * <pre>{@code
 * try (VectorStore vs = RedisVectorStore.builder()
 *         .host("127.0.0.1").port(6379)
 *         .index("idx:clips").prefix("doc:")
 *         .dim(768).metric(VectorMetric.COSINE)
 *         .payloadField(PayloadField.text("title"))
 *         .payloadField(PayloadField.tag("category"))
 *         .payloadField(PayloadField.numeric("year").sortable())
 *         .build()) {
 *     vs.ensureCollection();
 *     vs.upsert(batch);                 // pipelined
 *     vs.search(emb, 5);
 *     List<VectorRecord> page = vs.scroll(100, 0).records();
 * }
 * }</pre>
 */
public final class RedisVectorStore implements VectorStore {

    private final RespClient client;
    private final String index;
    private final String prefix;
    private final String vectorField;
    private final int dim;
    private final VectorMetric metric;
    private final String algorithm;
    private final int M;
    private final int efConstruction;
    private final boolean ownClient;
    private final List<PayloadField> payloadFields;
    private final int pipelineBatch;
    /** Default TTL applied to each doc key on upsert; null / non-positive = no expiry. */
    private final Duration defaultTtl;

    private RedisVectorStore(Builder b) {
        this.index = Objects.requireNonNull(b.index, "index");
        this.prefix = b.prefix == null ? "doc:" : b.prefix;
        this.vectorField = b.vectorField == null ? "vector" : b.vectorField;
        this.dim = b.dim;
        this.metric = b.metric == null ? VectorMetric.COSINE : b.metric;
        this.algorithm = b.algorithm == null ? "HNSW" : b.algorithm;
        this.M = b.M;
        this.efConstruction = b.efConstruction;
        this.payloadFields = List.copyOf(b.payloadFields);
        this.pipelineBatch = Math.max(1, b.pipelineBatch);
        this.defaultTtl = b.ttl;
        if (b.client != null) {
            this.client = b.client;
            this.ownClient = false;
        } else {
            this.client = new RespClient(b.host, b.port, b.username, b.password, b.timeout);
            this.ownClient = true;
        }
    }

    public static Builder builder() { return new Builder(); }

    @Override public String backend() { return "redis"; }
    @Override public String name() { return index; }
    @Override public int dim() { return dim; }
    @Override public VectorMetric metric() { return metric; }

    @Override
    public void ensureCollection() {
        try {
            client.call("FT.INFO", index);
            return;
        } catch (VectorStoreException e) {
            String msg = e.getMessage() == null ? "" : e.getMessage().toLowerCase();
            if (!msg.contains("unknown") && !msg.contains("no such") && !msg.contains("not found")
                && !msg.contains("unknown index")) {
                throw e;
            }
        }
        if (dim <= 0) {
            throw new VectorStoreException("dim required to create Redis vector index", -1, backend());
        }
        List<Object> args = new ArrayList<>();
        args.add("FT.CREATE");
        args.add(index);
        args.add("ON");
        args.add("HASH");
        args.add("PREFIX");
        args.add("1");
        args.add(prefix);
        args.add("SCHEMA");

        // id as TAG for filter/return
        args.add("id");
        args.add("TAG");
        args.add("SORTABLE");

        // vector field
        args.add(vectorField);
        args.add("VECTOR");
        args.add(algorithm);
        if ("HNSW".equalsIgnoreCase(algorithm)) {
            int extra = 0;
            if (M > 0) extra += 2;
            if (efConstruction > 0) extra += 2;
            args.add(String.valueOf(6 + extra));
            args.add("TYPE");
            args.add("FLOAT32");
            args.add("DIM");
            args.add(String.valueOf(dim));
            args.add("DISTANCE_METRIC");
            args.add(metric.redis());
            if (M > 0) {
                args.add("M");
                args.add(String.valueOf(M));
            }
            if (efConstruction > 0) {
                args.add("EF_CONSTRUCTION");
                args.add(String.valueOf(efConstruction));
            }
        } else {
            args.add("6");
            args.add("TYPE");
            args.add("FLOAT32");
            args.add("DIM");
            args.add(String.valueOf(dim));
            args.add("DISTANCE_METRIC");
            args.add(metric.redis());
        }

        // multi-field payload SCHEMA
        for (PayloadField pf : payloadFields) {
            if ("id".equals(pf.name()) || vectorField.equals(pf.name())) continue;
            pf.appendRedisSchema(args);
        }

        client.call(args.toArray());
    }

    @Override
    public void dropCollection() {
        try {
            client.call("FT.DROPINDEX", index, "DD");
        } catch (VectorStoreException e) {
            String msg = e.getMessage() == null ? "" : e.getMessage().toLowerCase();
            if (!msg.contains("unknown") && !msg.contains("no such")) throw e;
        }
    }

    @Override
    public long count() {
        try {
            List<Object> info = client.callArray("FT.INFO", index);
            for (int i = 0; i + 1 < info.size(); i += 2) {
                String k = RespClient.str(info.get(i));
                if ("num_docs".equalsIgnoreCase(k) || "num_records".equalsIgnoreCase(k)) {
                    return parseLong(info.get(i + 1));
                }
            }
        } catch (VectorStoreException e) {
            return -1L;
        }
        return -1L;
    }

    @Override
    public void upsert(Collection<VectorRecord> records) {
        upsert(records, defaultTtl);
    }

    /**
     * Upsert with a per-call TTL override applied to every {@code doc:{id}} key
     * via pipelined {@code EXPIRE}/{@code PEXPIRE}. {@code null} TTL means
     * "use builder default"; non-positive / zero means "no expiry for this call".
     */
    public void upsert(Collection<VectorRecord> records, Duration ttl) {
        if (records == null || records.isEmpty()) return;
        Duration effective = ttl != null ? ttl : defaultTtl;
        List<Object[]> pipeline = new ArrayList<>(Math.min(records.size(), pipelineBatch));
        List<String> keysForTtl = (effective != null && !effective.isZero() && !effective.isNegative())
                ? new ArrayList<>(Math.min(records.size(), pipelineBatch))
                : null;
        for (VectorRecord r : records) {
            if (dim > 0 && r.vector().length != dim) {
                throw new VectorStoreException(
                    "dim mismatch: got " + r.vector().length + ", expected " + dim, -1, backend());
            }
            String key = prefix + r.resolvedId();
            pipeline.add(hsetArgs(r));
            if (keysForTtl != null) keysForTtl.add(key);
            if (pipeline.size() >= pipelineBatch) {
                client.pipeline(pipeline);
                pipeline.clear();
                if (keysForTtl != null && !keysForTtl.isEmpty()) {
                    expireKeys(keysForTtl, effective);
                    keysForTtl.clear();
                }
            }
        }
        if (!pipeline.isEmpty()) client.pipeline(pipeline);
        if (keysForTtl != null && !keysForTtl.isEmpty()) {
            expireKeys(keysForTtl, effective);
        }
    }

    /** Upsert a single record with TTL (null → builder default). */
    public void upsert(VectorRecord record, Duration ttl) {
        if (record != null) upsert(List.of(record), ttl);
    }

    /**
     * DataFrame upsert with TTL on every {@code doc:{id}} key.
     * Builds {@link VectorRecord}s then {@link #upsert(Collection, Duration)} so
     * HSET + EXPIRE share the same pipeline batches.
     */
    public void upsertDataFrame(org.bytedeco.pytorch.data.dataframe.DataFrame df,
                                String idCol, String vectorCol, Duration ttl,
                                String... payloadCols) {
        Objects.requireNonNull(df, "df");
        Objects.requireNonNull(vectorCol, "vectorCol");
        org.bytedeco.pytorch.data.dataframe.Column vcol = df.column(vectorCol);
        org.bytedeco.pytorch.data.dataframe.Column icol =
                idCol == null ? null : df.column(idCol);
        if (vcol.dtype() == org.bytedeco.pytorch.data.dataframe.Column.DType.TENSOR) {
            throw new VectorStoreException(
                    "vectorCol '" + vectorCol + "' is TENSOR (multi-dim).", -1, backend());
        }
        List<String> payloadNames = new ArrayList<>();
        if (payloadCols == null || payloadCols.length == 0) {
            for (int c = 0; c < df.columnCount(); c++) {
                String n = df.column(c).name();
                if (!n.equals(vectorCol) && (idCol == null || !n.equals(idCol))) {
                    payloadNames.add(n);
                }
            }
        } else {
            for (String p : payloadCols) {
                if (p != null && !p.isEmpty()) payloadNames.add(p);
            }
        }
        List<VectorRecord> batch = new ArrayList<>(df.rowCount());
        for (int r = 0; r < df.rowCount(); r++) {
            float[] vec = VectorStore.toFloatArray(vcol.get(r));
            if (vec == null) continue;
            VectorRecord.Builder b = VectorRecord.builder().vector(vec);
            if (icol != null) {
                Object idv = icol.get(r);
                if (idv instanceof Number n) b.id(n.longValue());
                else if (idv != null) b.id(String.valueOf(idv));
                else b.id((long) r);
            } else {
                b.id((long) r);
            }
            if (!payloadNames.isEmpty()) {
                Map<String, Object> payload = new LinkedHashMap<>();
                for (String pn : payloadNames) payload.put(pn, df.get(r, pn));
                b.payload(payload);
            }
            batch.add(b.build());
        }
        if (!batch.isEmpty()) upsert(batch, ttl);
    }

    /**
     * Apply / refresh TTL on existing document keys (prefix + id).
     *
     * @return number of keys that received a TTL (EXPIRE returned 1)
     */
    public long expire(Collection<String> ids, Duration ttl) {
        if (ids == null || ids.isEmpty() || ttl == null || ttl.isZero() || ttl.isNegative()) {
            return 0L;
        }
        List<String> keys = new ArrayList<>(ids.size());
        for (String id : ids) {
            if (id != null) keys.add(prefix + id);
        }
        return expireKeys(keys, ttl);
    }

    public long expire(Duration ttl, String... ids) {
        if (ids == null || ids.length == 0) return 0L;
        return expire(List.of(ids), ttl);
    }

    /** Remove TTL from document keys ({@code PERSIST}). */
    public long persist(Collection<String> ids) {
        if (ids == null || ids.isEmpty()) return 0L;
        List<Object[]> cmds = new ArrayList<>(ids.size());
        for (String id : ids) {
            if (id != null) cmds.add(new Object[]{"PERSIST", prefix + id});
        }
        if (cmds.isEmpty()) return 0L;
        List<Object> replies = client.pipeline(cmds);
        long n = 0;
        for (Object r : replies) {
            if (r instanceof Number num && num.longValue() == 1L) n++;
        }
        return n;
    }

    /** TTL seconds for a document id, or Redis semantics (-1 no expire, -2 missing). */
    public long ttl(String id) {
        return client.callLong("TTL", prefix + id);
    }

    public Duration defaultTtl() {
        return defaultTtl;
    }

    private long expireKeys(List<String> keys, Duration ttl) {
        if (keys == null || keys.isEmpty() || ttl == null || ttl.isZero() || ttl.isNegative()) {
            return 0L;
        }
        long ms = ttl.toMillis();
        boolean useMillis = ms > 0 && (ms < 1000 || ms % 1000 != 0);
        List<Object[]> cmds = new ArrayList<>(keys.size());
        for (String key : keys) {
            if (useMillis) {
                cmds.add(new Object[]{"PEXPIRE", key, String.valueOf(ms)});
            } else {
                cmds.add(new Object[]{"EXPIRE", key, String.valueOf(Math.max(1L, ttl.getSeconds()))});
            }
        }
        List<Object> replies = client.pipeline(cmds);
        long n = 0;
        for (Object r : replies) {
            if (r instanceof Number num && num.longValue() == 1L) n++;
        }
        return n;
    }

    private Object[] hsetArgs(VectorRecord r) {
        List<Object> args = new ArrayList<>();
        args.add("HSET");
        args.add(prefix + r.resolvedId());
        args.add(vectorField);
        args.add(toLittleEndianBytes(r.vector()));
        args.add("id");
        args.add(r.resolvedId());
        if (r.payload() != null) {
            for (Map.Entry<String, Object> e : r.payload().entrySet()) {
                if (e.getKey() == null || e.getKey().equals(vectorField) || "id".equals(e.getKey())) continue;
                args.add(e.getKey());
                args.add(payloadToString(e.getKey(), e.getValue()));
            }
        }
        return args.toArray();
    }

    @Override
    public void delete(Collection<String> ids) {
        if (ids == null || ids.isEmpty()) return;
        // pipeline UNLINK/DEL in chunks
        List<String> list = ids instanceof List ? (List<String>) ids : new ArrayList<>(ids);
        for (int i = 0; i < list.size(); i += pipelineBatch) {
            List<String> slice = list.subList(i, Math.min(i + pipelineBatch, list.size()));
            List<Object> args = new ArrayList<>(slice.size() + 1);
            args.add("UNLINK");
            for (String id : slice) args.add(prefix + id);
            try {
                client.call(args.toArray());
            } catch (VectorStoreException e) {
                // UNLINK may be missing on very old Redis — fall back to DEL
                args.set(0, "DEL");
                client.call(args.toArray());
            }
        }
    }

    @Override
    public List<VectorRecord> fetch(Collection<String> ids) {
        if (ids == null || ids.isEmpty()) return List.of();
        List<VectorRecord> out = new ArrayList<>();
        // pipeline HGETALL
        List<Object[]> cmds = new ArrayList<>();
        List<String> idOrder = new ArrayList<>();
        for (String id : ids) {
            if (id == null) continue;
            idOrder.add(id);
            cmds.add(new Object[]{"HGETALL", prefix + id});
            if (cmds.size() >= pipelineBatch) {
                drainFetch(cmds, idOrder, out);
                cmds.clear();
                idOrder.clear();
            }
        }
        if (!cmds.isEmpty()) drainFetch(cmds, idOrder, out);
        return out;
    }

    private void drainFetch(List<Object[]> cmds, List<String> idOrder, List<VectorRecord> out) {
        List<Object> replies = client.pipeline(cmds);
        for (int i = 0; i < replies.size(); i++) {
            Object rep = replies.get(i);
            if (!(rep instanceof List<?> flat) || flat.isEmpty()) continue;
            Map<String, Object> fields = flatToMap(flat);
            String id = idOrder.get(i);
            if (fields.containsKey("id")) id = String.valueOf(fields.get("id"));
            float[] vec = null;
            Object vb = fields.get(vectorField);
            if (vb instanceof byte[] b) vec = fromLittleEndianBytes(b);
            else if (vb instanceof String s) {
                // unlikely for binary, but handle
                vec = fromLittleEndianBytes(s.getBytes(StandardCharsets.ISO_8859_1));
            }
            if (vec == null) continue;
            Map<String, Object> payload = new LinkedHashMap<>();
            for (Map.Entry<String, Object> e : fields.entrySet()) {
                if (vectorField.equals(e.getKey()) || "id".equals(e.getKey())) continue;
                payload.put(e.getKey(), e.getValue());
            }
            out.add(VectorRecord.of(id, vec, payload));
        }
    }

    @Override
    public ScrollPage scroll(int limit, Object cursor) {
        int lim = Math.max(1, limit);
        int offset = 0;
        if (cursor instanceof Number n) offset = Math.max(0, n.intValue());
        else if (cursor instanceof String s) {
            try { offset = Integer.parseInt(s); } catch (NumberFormatException ignored) {}
        }
        // FT.SEARCH idx * LIMIT offset lim RETURN ...
        List<Object> args = new ArrayList<>();
        args.add("FT.SEARCH");
        args.add(index);
        args.add("*");
        args.add("LIMIT");
        args.add(String.valueOf(offset));
        args.add(String.valueOf(lim));
        args.add("DIALECT");
        args.add("2");

        List<Object> raw;
        try {
            raw = client.callArray(args.toArray());
        } catch (VectorStoreException e) {
            return ScrollPage.empty();
        }
        long total = raw.isEmpty() ? 0 : parseLong(raw.get(0));
        List<VectorRecord> page = new ArrayList<>();
        for (int i = 1; i + 1 < raw.size(); i += 2) {
            String docKey = RespClient.str(raw.get(i));
            Object fieldsObj = raw.get(i + 1);
            Map<String, Object> fields = new LinkedHashMap<>();
            if (fieldsObj instanceof List<?> fl) fields = flatToMap(fl);
            String id = fields.containsKey("id") ? String.valueOf(fields.get("id")) : stripPrefix(docKey);
            float[] vec = null;
            Object vb = fields.get(vectorField);
            if (vb instanceof byte[] b) vec = fromLittleEndianBytes(b);
            Map<String, Object> payload = new LinkedHashMap<>();
            for (Map.Entry<String, Object> e : fields.entrySet()) {
                if (vectorField.equals(e.getKey()) || "id".equals(e.getKey())) continue;
                payload.put(e.getKey(), e.getValue());
            }
            // if vector missing from RETURN default, HMGET
            if (vec == null) {
                try {
                    Object bulk = client.call("HGET", prefix + id, vectorField);
                    if (bulk instanceof byte[] bb) vec = fromLittleEndianBytes(bb);
                } catch (VectorStoreException ignored) {}
            }
            if (vec == null) vec = new float[Math.max(dim, 0)];
            page.add(VectorRecord.of(id, vec, payload));
        }
        int next = offset + page.size();
        Object nextCur = (page.isEmpty() || next >= total) ? null : Integer.valueOf(next);
        return new ScrollPage(page, nextCur);
    }

    @Override
    public VectorSearchResult search(VectorQuery query) {
        Objects.requireNonNull(query, "query");
        long t0 = System.nanoTime();
        int k = query.topK();
        byte[] blob = toLittleEndianBytes(query.vector());

        String filterPrefix = "*";
        if (query.filter() instanceof String s && !s.isBlank()) {
            filterPrefix = s.trim();
        }
        String knn = filterPrefix + "=>[KNN " + k + " @" + vectorField + " $blob AS __score]";

        List<Object> args = new ArrayList<>();
        args.add("FT.SEARCH");
        args.add(index);
        args.add(knn);
        args.add("PARAMS");
        args.add("2");
        args.add("blob");
        args.add(blob);
        args.add("SORTBY");
        args.add("__score");
        args.add("DIALECT");
        args.add("2");
        if (!query.includeVector()) {
            // return id + score + declared payload fields
            List<String> ret = new ArrayList<>();
            ret.add("id");
            ret.add("__score");
            for (PayloadField pf : payloadFields) ret.add(pf.name());
            args.add("RETURN");
            args.add(String.valueOf(ret.size()));
            args.addAll(ret);
        }

        List<Object> raw = client.callArray(args.toArray());
        List<VectorHit> hits = new ArrayList<>();
        if (raw.isEmpty()) return VectorSearchResult.empty();
        for (int i = 1; i + 1 < raw.size(); i += 2) {
            String docKey = RespClient.str(raw.get(i));
            Object fieldsObj = raw.get(i + 1);
            Map<String, Object> fields = new LinkedHashMap<>();
            if (fieldsObj instanceof List<?> fl) fields = flatToMap(fl);
            String id = fields.containsKey("id")
                ? String.valueOf(fields.get("id"))
                : stripPrefix(docKey);
            float score = 0f;
            Object sc = fields.get("__score");
            if (sc != null) {
                try { score = Float.parseFloat(String.valueOf(sc)); }
                catch (NumberFormatException ignored) {}
            }
            float[] vec = null;
            if (query.includeVector()) {
                Object vb = fields.get(vectorField);
                if (vb instanceof byte[] b) vec = fromLittleEndianBytes(b);
            }
            Map<String, Object> payload = new LinkedHashMap<>();
            if (query.includePayload()) {
                for (Map.Entry<String, Object> e : fields.entrySet()) {
                    String k2 = e.getKey();
                    if (vectorField.equals(k2) || "__score".equals(k2) || "id".equals(k2)) continue;
                    payload.put(k2, e.getValue());
                }
            }
            hits.add(new VectorHit(id, -1L, false, score, score, vec, payload));
            if (hits.size() >= k) break;
        }
        long took = (System.nanoTime() - t0) / 1_000_000L;
        return new VectorSearchResult(hits, took);
    }

    @Override
    public void close() {
        if (ownClient) client.close();
    }

    private String stripPrefix(String key) {
        if (key != null && key.startsWith(prefix)) return key.substring(prefix.length());
        return key;
    }

    private String payloadToString(String key, Object v) {
        if (v == null) return "";
        if (v instanceof byte[] b) return new String(b, StandardCharsets.UTF_8);
        // NUMERIC fields must be plain numbers
        for (PayloadField pf : payloadFields) {
            if (pf.name().equals(key) && pf.type() == PayloadField.Type.NUMERIC) {
                if (v instanceof Number n) return n.toString();
            }
            if (pf.name().equals(key) && pf.type() == PayloadField.Type.BOOLEAN) {
                return Boolean.parseBoolean(String.valueOf(v)) ? "true" : "false";
            }
        }
        return String.valueOf(v);
    }

    private static Map<String, Object> flatToMap(List<?> flat) {
        Map<String, Object> fields = new LinkedHashMap<>();
        for (int j = 0; j + 1 < flat.size(); j += 2) {
            String fk = RespClient.str(flat.get(j));
            Object fv = flat.get(j + 1);
            if (fv instanceof byte[] b) {
                // keep binary for vector; decode text otherwise later
                fields.put(fk, b);
                // also provide string view for non-vector
                fields.putIfAbsent(fk + "#str", new String(b, StandardCharsets.UTF_8));
            } else {
                fields.put(fk, fv);
            }
        }
        // normalize: for non-vector keys, prefer string
        Map<String, Object> norm = new LinkedHashMap<>();
        for (Map.Entry<String, Object> e : fields.entrySet()) {
            if (e.getKey().endsWith("#str")) continue;
            Object v = e.getValue();
            if (v instanceof byte[] b) {
                // leave as bytes — caller decides
                norm.put(e.getKey(), b);
            } else {
                norm.put(e.getKey(), v);
            }
        }
        // convert non-vector byte[] to string for convenience
        for (Map.Entry<String, Object> e : new ArrayList<>(norm.entrySet())) {
            if (e.getValue() instanceof byte[] b && !e.getKey().equals("vector")
                && !e.getKey().contains("embedding") && !e.getKey().contains("vec")) {
                // heuristic: if looks like text, decode
                norm.put(e.getKey(), new String(b, StandardCharsets.UTF_8));
            }
        }
        return norm;
    }

    private static long parseLong(Object o) {
        if (o instanceof Number n) return n.longValue();
        try { return Long.parseLong(RespClient.str(o)); }
        catch (Exception e) { return -1L; }
    }

    public static byte[] toLittleEndianBytes(float[] v) {
        ByteBuffer buf = ByteBuffer.allocate(v.length * 4).order(ByteOrder.LITTLE_ENDIAN);
        for (float x : v) buf.putFloat(x);
        return buf.array();
    }

    public static float[] fromLittleEndianBytes(byte[] b) {
        if (b == null || b.length < 4) return new float[0];
        ByteBuffer buf = ByteBuffer.wrap(b).order(ByteOrder.LITTLE_ENDIAN);
        float[] v = new float[b.length / 4];
        for (int i = 0; i < v.length; i++) v[i] = buf.getFloat();
        return v;
    }

    public static final class Builder {
        private String host = "127.0.0.1";
        private int port = 6379;
        private String username;
        private String password;
        private Duration timeout = Duration.ofSeconds(10);
        private RespClient client;
        private String index = "idx:vectors";
        private String prefix = "doc:";
        private String vectorField = "vector";
        private int dim;
        private VectorMetric metric = VectorMetric.COSINE;
        private String algorithm = "HNSW";
        private int M = 16;
        private int efConstruction = 200;
        private final List<PayloadField> payloadFields = new ArrayList<>();
        private int pipelineBatch = 256;
        private Duration ttl;

        public Builder host(String h) { this.host = h; return this; }
        public Builder port(int p) { this.port = p; return this; }
        public Builder username(String u) { this.username = u; return this; }
        public Builder password(String p) { this.password = p; return this; }
        public Builder timeout(Duration d) { this.timeout = d; return this; }
        public Builder client(RespClient c) { this.client = c; return this; }
        public Builder index(String i) { this.index = i; return this; }
        public Builder prefix(String p) { this.prefix = p; return this; }
        public Builder vectorField(String f) { this.vectorField = f; return this; }
        public Builder dim(int d) { this.dim = d; return this; }
        public Builder metric(VectorMetric m) { this.metric = m; return this; }
        public Builder algorithm(String a) { this.algorithm = a; return this; }
        public Builder M(int m) { this.M = m; return this; }
        public Builder efConstruction(int ef) { this.efConstruction = ef; return this; }
        public Builder pipelineBatch(int n) { this.pipelineBatch = n; return this; }

        /**
         * Default TTL applied to every {@code doc:{id}} key on upsert
         * ({@code EXPIRE}/{@code PEXPIRE}). {@code null} = no expiry.
         */
        public Builder ttl(Duration d) { this.ttl = d; return this; }
        public Builder ttlSeconds(long seconds) {
            this.ttl = seconds <= 0 ? null : Duration.ofSeconds(seconds);
            return this;
        }

        public Builder payloadField(PayloadField f) {
            if (f != null) payloadFields.add(f);
            return this;
        }
        public Builder payloadFields(Collection<PayloadField> fields) {
            if (fields != null) payloadFields.addAll(fields);
            return this;
        }
        /** Convenience: TAG fields. */
        public Builder tagFields(String... names) {
            if (names != null) for (String n : names) if (n != null) payloadFields.add(PayloadField.tag(n));
            return this;
        }
        public Builder textFields(String... names) {
            if (names != null) for (String n : names) if (n != null) payloadFields.add(PayloadField.text(n));
            return this;
        }
        public Builder numericFields(String... names) {
            if (names != null) for (String n : names) if (n != null) payloadFields.add(PayloadField.numeric(n));
            return this;
        }

        public RedisVectorStore build() {
            return new RedisVectorStore(this);
        }
    }
}
