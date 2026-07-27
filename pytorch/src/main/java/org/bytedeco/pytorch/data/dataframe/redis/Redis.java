package org.bytedeco.pytorch.data.dataframe.redis;

import org.bytedeco.pytorch.data.dataframe.Column;
import org.bytedeco.pytorch.data.dataframe.DataFrame;
import org.bytedeco.pytorch.data.dataframe.vectorstore.redis.RespClient;
import org.bytedeco.pytorch.data.dataframe.vectorstore.VectorStoreException;
import org.bytedeco.pytorch.utils.json.Json;

import java.io.Closeable;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.util.ArrayList;
import java.util.Base64;
import java.util.Collection;
import java.util.Collections;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.function.Consumer;

/**
 * Full-featured Redis client for DataFrame I/O — pure RESP2 via {@link RespClient},
 * no Jedis / Lettuce dependency.
 *
 * <h2>Coverage (Jedis-parity subset used by DataFrame workflows)</h2>
 * <ul>
 *   <li><b>Connection</b> — connect / auth / ping / select / quit / close</li>
 *   <li><b>Keys</b> — exists, del, unlink, expire, pexpire, ttl, pttl, persist,
 *       type, rename, keys (prefer scan), scan, scanAll</li>
 *   <li><b>Strings</b> — get/set/setex/psetex/setnx, mget/mset, incr/decr, append, getrange/setrange</li>
 *   <li><b>Hashes</b> — hget/hset/hmget/hmset/hgetall/hdel/hexists/hkeys/hvals/hlen/hincrby</li>
 *   <li><b>Lists</b> — lpush/rpush/lpop/rpop/lrange/llen</li>
 *   <li><b>Sets</b> — sadd/srem/smembers/sismember/scard</li>
 *   <li><b>Sorted sets</b> — zadd/zrem/zrange/zrangebyscore/zcard/zscore</li>
 *   <li><b>Pipeline</b> — multi-command write + ordered replies</li>
 *   <li><b>DataFrame</b> — {@link #writeDataFrame}, {@link #readDataFrame}, hash/json/frame layouts + TTL</li>
 * </ul>
 *
 * <pre>{@code
 * try (Redis r = Redis.connect("127.0.0.1", 6379)) {
 *     r.ping();
 *     df.toRedis(r, RedisOptions.hash("df:people:", Duration.ofHours(1)));
 *     DataFrame back = DataFrame.readRedis(r, RedisOptions.hash("df:people:"));
 *
 *     r.setex("cache:x", 60, "v");
 *     r.hset("user:1", Map.of("name", "alice", "score", "9.5"));
 *     r.expire("user:1", Duration.ofMinutes(30));
 * }
 * }</pre>
 */
public final class Redis implements Closeable {

    public static final int DEFAULT_PORT = 6379;
    public static final Duration DEFAULT_TIMEOUT = Duration.ofSeconds(10);

    private final RespClient client;
    private final boolean ownClient;
    private final String host;
    private final int port;
    private int pipelineBatch = 256;

    private Redis(RespClient client, boolean ownClient, String host, int port) {
        this.client = Objects.requireNonNull(client, "client");
        this.ownClient = ownClient;
        this.host = host == null ? "127.0.0.1" : host;
        this.port = port <= 0 ? DEFAULT_PORT : port;
    }

    // ── factories ──────────────────────────────────────────────────────────

    public static Redis connect() {
        return connect("127.0.0.1", DEFAULT_PORT);
    }

    public static Redis connect(String host) {
        return connect(host, DEFAULT_PORT);
    }

    public static Redis connect(String host, int port) {
        return connect(host, port, null, null, DEFAULT_TIMEOUT);
    }

    public static Redis connect(String host, int port, String password) {
        return connect(host, port, null, password, DEFAULT_TIMEOUT);
    }

    public static Redis connect(String host, int port, String username, String password) {
        return connect(host, port, username, password, DEFAULT_TIMEOUT);
    }

    public static Redis connect(String host, int port, String username, String password, Duration timeout) {
        RespClient c = RespClient.connect(host, port, username, password,
                timeout == null ? DEFAULT_TIMEOUT : timeout);
        return new Redis(c, true, host, port);
    }

    /** Parse {@code redis://[:password@]host:port[/db]} or {@code host:port}. */
    public static Redis connectUri(String uri) {
        Objects.requireNonNull(uri, "uri");
        String s = uri.trim();
        String host = "127.0.0.1";
        int port = DEFAULT_PORT;
        String user = null;
        String pass = null;
        Integer db = null;
        if (s.startsWith("redis://") || s.startsWith("rediss://")) {
            s = s.substring(s.indexOf("://") + 3);
        }
        // user:pass@host:port/db
        int at = s.lastIndexOf('@');
        String auth = null;
        String rest = s;
        if (at >= 0) {
            auth = s.substring(0, at);
            rest = s.substring(at + 1);
            int colon = auth.indexOf(':');
            if (colon >= 0) {
                user = auth.substring(0, colon);
                pass = auth.substring(colon + 1);
            } else {
                pass = auth;
            }
        }
        int slash = rest.indexOf('/');
        String hostPort = slash >= 0 ? rest.substring(0, slash) : rest;
        if (slash >= 0 && slash + 1 < rest.length()) {
            try { db = Integer.parseInt(rest.substring(slash + 1).split("[?]", 2)[0]); }
            catch (NumberFormatException ignored) {}
        }
        int colon = hostPort.lastIndexOf(':');
        if (colon > 0) {
            host = hostPort.substring(0, colon);
            try { port = Integer.parseInt(hostPort.substring(colon + 1)); }
            catch (NumberFormatException ignored) {}
        } else if (!hostPort.isBlank()) {
            host = hostPort;
        }
        Redis r = connect(host, port, user, pass, DEFAULT_TIMEOUT);
        if (db != null && db >= 0) r.select(db);
        return r;
    }

    /** Wrap an existing {@link RespClient} without taking ownership. */
    public static Redis wrap(RespClient client) {
        return new Redis(client, false, null, DEFAULT_PORT);
    }

    public RespClient client() { return client; }
    public String host() { return host; }
    public int port() { return port; }

    public Redis pipelineBatch(int n) {
        this.pipelineBatch = Math.max(1, n);
        return this;
    }

    public int pipelineBatch() { return pipelineBatch; }

    // ── connection ─────────────────────────────────────────────────────────

    public String ping() {
        return callString("PING");
    }

    public String echo(String msg) {
        return callString("ECHO", msg);
    }

    public String select(int db) {
        return callString("SELECT", String.valueOf(db));
    }

    public String flushDb() {
        return callString("FLUSHDB");
    }

    public String flushAll() {
        return callString("FLUSHALL");
    }

    public String info() {
        return callString("INFO");
    }

    public String info(String section) {
        return callString("INFO", section);
    }

    public long dbSize() {
        return callLong("DBSIZE");
    }

    public String clientSetName(String name) {
        return callString("CLIENT", "SETNAME", name);
    }

    // ── raw / pipeline ─────────────────────────────────────────────────────

    public Object call(Object... args) {
        try {
            return client.call(args);
        } catch (VectorStoreException e) {
            throw wrap(e, args.length > 0 ? String.valueOf(args[0]) : null);
        }
    }

    public String callString(Object... args) {
        try {
            return client.callString(args);
        } catch (VectorStoreException e) {
            throw wrap(e, args.length > 0 ? String.valueOf(args[0]) : null);
        }
    }

    public long callLong(Object... args) {
        try {
            return client.callLong(args);
        } catch (VectorStoreException e) {
            throw wrap(e, args.length > 0 ? String.valueOf(args[0]) : null);
        }
    }

    @SuppressWarnings("unchecked")
    public List<Object> callArray(Object... args) {
        try {
            return client.callArray(args);
        } catch (VectorStoreException e) {
            throw wrap(e, args.length > 0 ? String.valueOf(args[0]) : null);
        }
    }

    public List<Object> pipeline(List<Object[]> commands) {
        try {
            return client.pipeline(commands);
        } catch (VectorStoreException e) {
            throw wrap(e, "PIPELINE");
        }
    }

    public void pipelineVoid(List<Object[]> commands) {
        pipeline(commands);
    }

    // ── keys ───────────────────────────────────────────────────────────────

    public boolean exists(String key) {
        return callLong("EXISTS", key) > 0;
    }

    public long exists(String... keys) {
        if (keys == null || keys.length == 0) return 0;
        Object[] args = new Object[keys.length + 1];
        args[0] = "EXISTS";
        System.arraycopy(keys, 0, args, 1, keys.length);
        return callLong(args);
    }

    public long del(String... keys) {
        if (keys == null || keys.length == 0) return 0;
        Object[] args = new Object[keys.length + 1];
        args[0] = "DEL";
        System.arraycopy(keys, 0, args, 1, keys.length);
        return callLong(args);
    }

    public long del(Collection<String> keys) {
        if (keys == null || keys.isEmpty()) return 0;
        return del(keys.toArray(new String[0]));
    }

    public long unlink(String... keys) {
        if (keys == null || keys.length == 0) return 0;
        Object[] args = new Object[keys.length + 1];
        args[0] = "UNLINK";
        System.arraycopy(keys, 0, args, 1, keys.length);
        try {
            return callLong(args);
        } catch (RedisException e) {
            return del(keys);
        }
    }

    public boolean expire(String key, long seconds) {
        if (seconds < 0) return persist(key);
        return callLong("EXPIRE", key, String.valueOf(seconds)) == 1L;
    }

    public boolean expire(String key, Duration ttl) {
        if (ttl == null || ttl.isZero() || ttl.isNegative()) return persist(key);
        long ms = ttl.toMillis();
        if (ms > 0 && ms < 1000) return pexpire(key, ms);
        return expire(key, Math.max(1L, ttl.getSeconds()));
    }

    public boolean pexpire(String key, long millis) {
        if (millis < 0) return persist(key);
        return callLong("PEXPIRE", key, String.valueOf(millis)) == 1L;
    }

    public boolean expireAt(String key, long unixSeconds) {
        return callLong("EXPIREAT", key, String.valueOf(unixSeconds)) == 1L;
    }

    public boolean pexpireAt(String key, long unixMillis) {
        return callLong("PEXPIREAT", key, String.valueOf(unixMillis)) == 1L;
    }

    public long ttl(String key) {
        return callLong("TTL", key);
    }

    public long pttl(String key) {
        return callLong("PTTL", key);
    }

    public boolean persist(String key) {
        return callLong("PERSIST", key) == 1L;
    }

    public String type(String key) {
        return callString("TYPE", key);
    }

    public String rename(String oldKey, String newKey) {
        return callString("RENAME", oldKey, newKey);
    }

    public boolean renamenx(String oldKey, String newKey) {
        return callLong("RENAMENX", oldKey, newKey) == 1L;
    }

    /** Prefer {@link #scan(String, String, int)} — KEYS blocks the server. */
    public List<String> keys(String pattern) {
        List<Object> raw = callArray("KEYS", pattern == null ? "*" : pattern);
        List<String> out = new ArrayList<>(raw.size());
        for (Object o : raw) out.add(RespClient.str(o));
        return out;
    }

    /**
     * One SCAN page.
     *
     * @return next cursor (0 when done) + keys
     */
    public ScanPage scan(String cursor, String match, int count) {
        List<Object> args = new ArrayList<>();
        args.add("SCAN");
        args.add(cursor == null ? "0" : cursor);
        if (match != null && !match.isEmpty()) {
            args.add("MATCH");
            args.add(match);
        }
        if (count > 0) {
            args.add("COUNT");
            args.add(String.valueOf(count));
        }
        List<Object> raw = callArray(args.toArray());
        if (raw.size() < 2) return new ScanPage("0", List.of());
        String next = RespClient.str(raw.get(0));
        List<String> keys = new ArrayList<>();
        if (raw.get(1) instanceof List<?> list) {
            for (Object o : list) keys.add(RespClient.str(o));
        }
        return new ScanPage(next, keys);
    }

    public List<String> scanAll(String match) {
        return scanAll(match, 200);
    }

    public List<String> scanAll(String match, int count) {
        List<String> all = new ArrayList<>();
        String cursor = "0";
        do {
            ScanPage page = scan(cursor, match, count);
            all.addAll(page.keys());
            cursor = page.cursor();
        } while (cursor != null && !"0".equals(cursor));
        return all;
    }

    public void scanEach(String match, int count, Consumer<String> consumer) {
        String cursor = "0";
        do {
            ScanPage page = scan(cursor, match, count);
            for (String k : page.keys()) consumer.accept(k);
            cursor = page.cursor();
        } while (cursor != null && !"0".equals(cursor));
    }

    public record ScanPage(String cursor, List<String> keys) {
        public boolean done() {
            return cursor == null || "0".equals(cursor);
        }
    }

    // ── strings ────────────────────────────────────────────────────────────

    public String get(String key) {
        Object r = call("GET", key);
        return RespClient.str(r);
    }

    public byte[] getBytes(String key) {
        Object r = call("GET", key);
        if (r == null) return null;
        if (r instanceof byte[] b) return b;
        return String.valueOf(r).getBytes(StandardCharsets.UTF_8);
    }

    public String set(String key, String value) {
        return callString("SET", key, value);
    }

    public String set(String key, byte[] value) {
        return callString("SET", key, value);
    }

    /** SET key value EX seconds. */
    public String setex(String key, long seconds, String value) {
        return callString("SETEX", key, String.valueOf(seconds), value);
    }

    public String setex(String key, Duration ttl, String value) {
        long s = ttl == null ? 0 : Math.max(1L, ttl.getSeconds());
        if (ttl != null && ttl.toMillis() > 0 && ttl.toMillis() < 1000) {
            return psetex(key, ttl.toMillis(), value);
        }
        return setex(key, s, value);
    }

    public String psetex(String key, long millis, String value) {
        return callString("PSETEX", key, String.valueOf(millis), value);
    }

    public boolean setnx(String key, String value) {
        return callLong("SETNX", key, value) == 1L;
    }

    /**
     * Full SET with optional EX/PX/NX/XX.
     *
     * @param ttl     optional expiry
     * @param nx      only set if not exists
     * @param xx      only set if exists
     * @return {@code true} if set (or "OK"), {@code false} if NX/XX prevented set
     */
    public boolean set(String key, String value, Duration ttl, boolean nx, boolean xx) {
        List<Object> args = new ArrayList<>();
        args.add("SET");
        args.add(key);
        args.add(value);
        if (ttl != null && !ttl.isZero() && !ttl.isNegative()) {
            long ms = ttl.toMillis();
            if (ms > 0 && (ms % 1000 != 0 || ms < 1000)) {
                args.add("PX");
                args.add(String.valueOf(ms));
            } else {
                args.add("EX");
                args.add(String.valueOf(Math.max(1L, ttl.getSeconds())));
            }
        }
        if (nx) args.add("NX");
        if (xx) args.add("XX");
        Object r = call(args.toArray());
        if (r == null) return false;
        String s = RespClient.str(r);
        return s != null && "OK".equalsIgnoreCase(s);
    }

    public List<String> mget(String... keys) {
        if (keys == null || keys.length == 0) return List.of();
        Object[] args = new Object[keys.length + 1];
        args[0] = "MGET";
        System.arraycopy(keys, 0, args, 1, keys.length);
        List<Object> raw = callArray(args);
        List<String> out = new ArrayList<>(raw.size());
        for (Object o : raw) out.add(RespClient.str(o));
        return out;
    }

    public String mset(Map<String, String> kvs) {
        if (kvs == null || kvs.isEmpty()) return "OK";
        Object[] args = new Object[1 + kvs.size() * 2];
        args[0] = "MSET";
        int i = 1;
        for (Map.Entry<String, String> e : kvs.entrySet()) {
            args[i++] = e.getKey();
            args[i++] = e.getValue() == null ? "" : e.getValue();
        }
        return callString(args);
    }

    public long incr(String key) { return callLong("INCR", key); }
    public long incrBy(String key, long n) { return callLong("INCRBY", key, String.valueOf(n)); }
    public long decr(String key) { return callLong("DECR", key); }
    public long decrBy(String key, long n) { return callLong("DECRBY", key, String.valueOf(n)); }

    public long append(String key, String value) {
        return callLong("APPEND", key, value);
    }

    public String getrange(String key, long start, long end) {
        return callString("GETRANGE", key, String.valueOf(start), String.valueOf(end));
    }

    public long setrange(String key, long offset, String value) {
        return callLong("SETRANGE", key, String.valueOf(offset), value);
    }

    public long strlen(String key) {
        return callLong("STRLEN", key);
    }

    // ── hashes ─────────────────────────────────────────────────────────────

    public String hget(String key, String field) {
        return RespClient.str(call("HGET", key, field));
    }

    public long hset(String key, String field, String value) {
        return callLong("HSET", key, field, value == null ? "" : value);
    }

    public long hset(String key, Map<String, ?> fields) {
        if (fields == null || fields.isEmpty()) return 0;
        Object[] args = new Object[2 + fields.size() * 2];
        args[0] = "HSET";
        args[1] = key;
        int i = 2;
        for (Map.Entry<String, ?> e : fields.entrySet()) {
            args[i++] = e.getKey();
            args[i++] = valueToString(e.getValue());
        }
        return callLong(args);
    }

    public boolean hsetnx(String key, String field, String value) {
        return callLong("HSETNX", key, field, value) == 1L;
    }

    /** HMSET (deprecated on server but widely supported) — uses HSET multi-field. */
    public String hmset(String key, Map<String, ?> fields) {
        hset(key, fields);
        return "OK";
    }

    public List<String> hmget(String key, String... fields) {
        if (fields == null || fields.length == 0) return List.of();
        Object[] args = new Object[2 + fields.length];
        args[0] = "HMGET";
        args[1] = key;
        System.arraycopy(fields, 0, args, 2, fields.length);
        List<Object> raw = callArray(args);
        List<String> out = new ArrayList<>(raw.size());
        for (Object o : raw) out.add(RespClient.str(o));
        return out;
    }

    public Map<String, String> hgetall(String key) {
        List<Object> raw = callArray("HGETALL", key);
        Map<String, String> out = new LinkedHashMap<>();
        for (int i = 0; i + 1 < raw.size(); i += 2) {
            out.put(RespClient.str(raw.get(i)), RespClient.str(raw.get(i + 1)));
        }
        return out;
    }

    public long hdel(String key, String... fields) {
        if (fields == null || fields.length == 0) return 0;
        Object[] args = new Object[2 + fields.length];
        args[0] = "HDEL";
        args[1] = key;
        System.arraycopy(fields, 0, args, 2, fields.length);
        return callLong(args);
    }

    public boolean hexists(String key, String field) {
        return callLong("HEXISTS", key, field) == 1L;
    }

    public Set<String> hkeys(String key) {
        List<Object> raw = callArray("HKEYS", key);
        Set<String> out = new LinkedHashSet<>();
        for (Object o : raw) out.add(RespClient.str(o));
        return out;
    }

    public List<String> hvals(String key) {
        List<Object> raw = callArray("HVALS", key);
        List<String> out = new ArrayList<>(raw.size());
        for (Object o : raw) out.add(RespClient.str(o));
        return out;
    }

    public long hlen(String key) {
        return callLong("HLEN", key);
    }

    public long hincrBy(String key, String field, long n) {
        return callLong("HINCRBY", key, field, String.valueOf(n));
    }

    // ── lists ──────────────────────────────────────────────────────────────

    public long lpush(String key, String... values) {
        return listPush("LPUSH", key, values);
    }

    public long rpush(String key, String... values) {
        return listPush("RPUSH", key, values);
    }

    private long listPush(String cmd, String key, String... values) {
        if (values == null || values.length == 0) return llen(key);
        Object[] args = new Object[2 + values.length];
        args[0] = cmd;
        args[1] = key;
        System.arraycopy(values, 0, args, 2, values.length);
        return callLong(args);
    }

    public String lpop(String key) {
        return RespClient.str(call("LPOP", key));
    }

    public String rpop(String key) {
        return RespClient.str(call("RPOP", key));
    }

    public List<String> lrange(String key, long start, long stop) {
        List<Object> raw = callArray("LRANGE", key, String.valueOf(start), String.valueOf(stop));
        List<String> out = new ArrayList<>(raw.size());
        for (Object o : raw) out.add(RespClient.str(o));
        return out;
    }

    public long llen(String key) {
        return callLong("LLEN", key);
    }

    // ── sets ───────────────────────────────────────────────────────────────

    public long sadd(String key, String... members) {
        if (members == null || members.length == 0) return 0;
        Object[] args = new Object[2 + members.length];
        args[0] = "SADD";
        args[1] = key;
        System.arraycopy(members, 0, args, 2, members.length);
        return callLong(args);
    }

    public long srem(String key, String... members) {
        if (members == null || members.length == 0) return 0;
        Object[] args = new Object[2 + members.length];
        args[0] = "SREM";
        args[1] = key;
        System.arraycopy(members, 0, args, 2, members.length);
        return callLong(args);
    }

    public Set<String> smembers(String key) {
        List<Object> raw = callArray("SMEMBERS", key);
        Set<String> out = new LinkedHashSet<>();
        for (Object o : raw) out.add(RespClient.str(o));
        return out;
    }

    public boolean sismember(String key, String member) {
        return callLong("SISMEMBER", key, member) == 1L;
    }

    public long scard(String key) {
        return callLong("SCARD", key);
    }

    // ── sorted sets ────────────────────────────────────────────────────────

    public long zadd(String key, double score, String member) {
        return callLong("ZADD", key, String.valueOf(score), member);
    }

    public long zadd(String key, Map<String, Double> scoreMembers) {
        if (scoreMembers == null || scoreMembers.isEmpty()) return 0;
        Object[] args = new Object[2 + scoreMembers.size() * 2];
        args[0] = "ZADD";
        args[1] = key;
        int i = 2;
        for (Map.Entry<String, Double> e : scoreMembers.entrySet()) {
            args[i++] = String.valueOf(e.getValue());
            args[i++] = e.getKey();
        }
        return callLong(args);
    }

    public long zrem(String key, String... members) {
        if (members == null || members.length == 0) return 0;
        Object[] args = new Object[2 + members.length];
        args[0] = "ZREM";
        args[1] = key;
        System.arraycopy(members, 0, args, 2, members.length);
        return callLong(args);
    }

    public List<String> zrange(String key, long start, long stop) {
        List<Object> raw = callArray("ZRANGE", key, String.valueOf(start), String.valueOf(stop));
        List<String> out = new ArrayList<>(raw.size());
        for (Object o : raw) out.add(RespClient.str(o));
        return out;
    }

    public List<String> zrangeByScore(String key, double min, double max) {
        List<Object> raw = callArray("ZRANGEBYSCORE", key, String.valueOf(min), String.valueOf(max));
        List<String> out = new ArrayList<>(raw.size());
        for (Object o : raw) out.add(RespClient.str(o));
        return out;
    }

    public long zcard(String key) {
        return callLong("ZCARD", key);
    }

    public Double zscore(String key, String member) {
        String s = RespClient.str(call("ZSCORE", key, member));
        if (s == null) return null;
        try { return Double.parseDouble(s); }
        catch (NumberFormatException e) { return null; }
    }

    // ── batch TTL helpers ──────────────────────────────────────────────────

    /** Pipeline EXPIRE for many keys (seconds). Returns number of keys that got a TTL. */
    public long expireMany(Collection<String> keys, long seconds) {
        if (keys == null || keys.isEmpty() || seconds < 0) return 0;
        List<Object[]> cmds = new ArrayList<>(keys.size());
        for (String k : keys) {
            if (k != null) cmds.add(new Object[]{"EXPIRE", k, String.valueOf(seconds)});
        }
        List<Object> replies = pipeline(cmds);
        long n = 0;
        for (Object r : replies) {
            if (r instanceof Long l && l == 1L) n++;
            else if (r instanceof Number num && num.longValue() == 1L) n++;
        }
        return n;
    }

    public long expireMany(Collection<String> keys, Duration ttl) {
        if (ttl == null || ttl.isZero() || ttl.isNegative()) return 0;
        long ms = ttl.toMillis();
        if (ms > 0 && ms < 1000) return pexpireMany(keys, ms);
        return expireMany(keys, Math.max(1L, ttl.getSeconds()));
    }

    public long pexpireMany(Collection<String> keys, long millis) {
        if (keys == null || keys.isEmpty() || millis < 0) return 0;
        List<Object[]> cmds = new ArrayList<>(keys.size());
        for (String k : keys) {
            if (k != null) cmds.add(new Object[]{"PEXPIRE", k, String.valueOf(millis)});
        }
        List<Object> replies = pipeline(cmds);
        long n = 0;
        for (Object r : replies) {
            if (r instanceof Number num && num.longValue() == 1L) n++;
        }
        return n;
    }

    // ── DataFrame I/O ──────────────────────────────────────────────────────

    /**
     * Write a DataFrame using {@link RedisOptions} (HASH / JSON / FRAME_* + TTL).
     *
     * @return number of Redis keys written (frame layouts return 1)
     */
    public int writeDataFrame(DataFrame df, RedisOptions options) {
        Objects.requireNonNull(df, "df");
        RedisOptions opt = options == null ? RedisOptions.defaults() : options;
        return switch (opt.layout()) {
            case HASH -> writeHashLayout(df, opt);
            case JSON -> writeJsonLayout(df, opt);
            case FRAME_JSON -> writeFrameJson(df, opt, false);
            case FRAME_JSONL -> writeFrameJson(df, opt, true);
        };
    }

    /** Convenience: HASH layout with prefix + optional TTL. */
    public int writeHash(DataFrame df, String prefix, String idColumn, Duration ttl) {
        return writeDataFrame(df, RedisOptions.builder()
                .layout(RedisOptions.Layout.HASH)
                .prefix(prefix)
                .idColumn(idColumn)
                .ttl(ttl)
                .build());
    }

    public int writeHash(DataFrame df, String prefix) {
        return writeHash(df, prefix, null, null);
    }

    /** Convenience: per-row JSON strings. */
    public int writeJson(DataFrame df, String prefix, String idColumn, Duration ttl) {
        return writeDataFrame(df, RedisOptions.builder()
                .layout(RedisOptions.Layout.JSON)
                .prefix(prefix)
                .idColumn(idColumn)
                .ttl(ttl)
                .build());
    }

    /** Single-key JSON array of records. */
    public int writeFrame(DataFrame df, String key, Duration ttl) {
        return writeDataFrame(df, RedisOptions.builder()
                .layout(RedisOptions.Layout.FRAME_JSON)
                .key(key)
                .ttl(ttl)
                .build());
    }

    /**
     * Read a DataFrame previously written with {@link #writeDataFrame}.
     */
    public DataFrame readDataFrame(RedisOptions options) {
        RedisOptions opt = options == null ? RedisOptions.defaults() : options;
        return switch (opt.layout()) {
            case HASH -> readHashLayout(opt);
            case JSON -> readJsonLayout(opt);
            case FRAME_JSON, FRAME_JSONL -> readFrameJson(opt);
        };
    }

    public DataFrame readHash(String prefix) {
        return readDataFrame(RedisOptions.hash(prefix));
    }

    public DataFrame readHash(String prefix, Map<String, Column.DType> dtype) {
        return readDataFrame(RedisOptions.builder()
                .layout(RedisOptions.Layout.HASH)
                .prefix(prefix)
                .dtype(dtype)
                .build());
    }

    public DataFrame readJsonRows(String prefix) {
        return readDataFrame(RedisOptions.json(prefix));
    }

    public DataFrame readFrame(String key) {
        return readDataFrame(RedisOptions.frame(key));
    }

    /**
     * Read specific keys as HASH rows into a DataFrame (keys order preserved).
     */
    public DataFrame readHashes(Collection<String> keys) {
        return readHashes(keys, null);
    }

    public DataFrame readHashes(Collection<String> keys, Map<String, Column.DType> dtype) {
        if (keys == null || keys.isEmpty()) return DataFrame.create();
        List<Map<String, String>> rows = new ArrayList<>();
        List<String> keyList = new ArrayList<>();
        List<Object[]> cmds = new ArrayList<>();
        for (String k : keys) {
            if (k == null) continue;
            keyList.add(k);
            cmds.add(new Object[]{"HGETALL", k});
            if (cmds.size() >= pipelineBatch) {
                drainHgetall(cmds, keyList, rows);
                cmds.clear();
                keyList.clear();
            }
        }
        if (!cmds.isEmpty()) drainHgetall(cmds, keyList, rows);
        return mapsToDataFrame(rows, dtype, true);
    }

    // ── layout writers ─────────────────────────────────────────────────────

    private int writeHashLayout(DataFrame df, RedisOptions opt) {
        String idCol = resolveIdColumn(df, opt);
        List<Object[]> cmds = new ArrayList<>();
        List<String> writtenKeys = new ArrayList<>();
        int count = 0;
        for (int r = 0; r < df.rowCount(); r++) {
            Object id = idCol != null ? df.get(r, idCol) : r;
            String key = opt.keyFor(id == null ? r : id);
            if (opt.ifExists() != RedisOptions.IfExists.REPLACE && exists(key)) {
                if (opt.ifExists() == RedisOptions.IfExists.FAIL) {
                    throw new RedisException("key exists: " + key, null, "HSET");
                }
                continue; // SKIP
            }
            List<Object> args = new ArrayList<>();
            args.add("HSET");
            args.add(key);
            if (idCol != null) {
                args.add(idCol);
                args.add(valueToString(id));
            }
            for (int c = 0; c < df.columnCount(); c++) {
                String name = df.column(c).name();
                if (idCol != null && name.equals(idCol)) continue;
                Object v = df.get(r, name);
                if (v == null && !opt.includeNulls()) continue;
                args.add(name);
                args.add(cellToRedis(v, opt));
            }
            if (args.size() <= 2) {
                // empty hash — still touch key with id
                args.add("_row");
                args.add(String.valueOf(r));
            }
            cmds.add(args.toArray());
            writtenKeys.add(key);
            count++;
            if (cmds.size() >= opt.pipelineBatch()) {
                pipeline(cmds);
                cmds.clear();
            }
        }
        if (!cmds.isEmpty()) pipeline(cmds);
        applyTtl(writtenKeys, opt);
        return count;
    }

    private int writeJsonLayout(DataFrame df, RedisOptions opt) {
        String idCol = resolveIdColumn(df, opt);
        List<Object[]> cmds = new ArrayList<>();
        List<String> writtenKeys = new ArrayList<>();
        int count = 0;
        for (int r = 0; r < df.rowCount(); r++) {
            Object id = idCol != null ? df.get(r, idCol) : r;
            String key = opt.keyFor(id == null ? r : id);
            if (opt.ifExists() != RedisOptions.IfExists.REPLACE && exists(key)) {
                if (opt.ifExists() == RedisOptions.IfExists.FAIL) {
                    throw new RedisException("key exists: " + key, null, "SET");
                }
                continue;
            }
            Map<String, Object> row = rowToJsonMap(df, r, opt);
            String json = Json.encode(row);
            if (opt.hasTtl()) {
                long ms = opt.ttlMillis();
                if (ms % 1000 == 0) {
                    cmds.add(new Object[]{"SETEX", key, String.valueOf(ms / 1000), json});
                } else {
                    cmds.add(new Object[]{"PSETEX", key, String.valueOf(ms), json});
                }
            } else {
                cmds.add(new Object[]{"SET", key, json});
            }
            writtenKeys.add(key);
            count++;
            if (cmds.size() >= opt.pipelineBatch()) {
                pipeline(cmds);
                cmds.clear();
            }
        }
        if (!cmds.isEmpty()) pipeline(cmds);
        // TTL already applied via SETEX/PSETEX when present
        if (!opt.hasTtl()) {
            // nothing
        }
        return count;
    }

    private int writeFrameJson(DataFrame df, RedisOptions opt, boolean jsonl) {
        String key = opt.frameKey();
        if (opt.ifExists() != RedisOptions.IfExists.REPLACE && exists(key)) {
            if (opt.ifExists() == RedisOptions.IfExists.FAIL) {
                throw new RedisException("key exists: " + key, null, "SET");
            }
            return 0;
        }
        String body;
        if (jsonl) {
            StringBuilder sb = new StringBuilder(df.rowCount() * 64);
            for (int r = 0; r < df.rowCount(); r++) {
                if (r > 0) sb.append('\n');
                sb.append(Json.encode(rowToJsonMap(df, r, opt)));
            }
            body = sb.toString();
        } else {
            List<Map<String, Object>> records = new ArrayList<>(df.rowCount());
            for (int r = 0; r < df.rowCount(); r++) {
                records.add(rowToJsonMap(df, r, opt));
            }
            body = Json.encode(records);
        }
        if (opt.hasTtl()) {
            long ms = opt.ttlMillis();
            if (ms % 1000 == 0) setex(key, ms / 1000, body);
            else psetex(key, ms, body);
        } else {
            set(key, body);
        }
        return 1;
    }

    private void applyTtl(List<String> keys, RedisOptions opt) {
        if (!opt.hasTtl() || keys == null || keys.isEmpty()) return;
        long ms = opt.ttlMillis();
        // chunk pipelines
        for (int i = 0; i < keys.size(); i += opt.pipelineBatch()) {
            List<String> slice = keys.subList(i, Math.min(i + opt.pipelineBatch(), keys.size()));
            if (ms % 1000 == 0) expireMany(slice, ms / 1000);
            else pexpireMany(slice, ms);
        }
    }

    // ── layout readers ─────────────────────────────────────────────────────

    private DataFrame readHashLayout(RedisOptions opt) {
        String match = opt.scanMatchPrefix()
                ? (opt.prefix().endsWith("*") ? opt.prefix() : opt.prefix() + "*")
                : (opt.prefix() + "*");
        List<String> keys = scanAll(match, opt.scanCount());
        // stable order
        Collections.sort(keys);
        return readHashes(keys, opt.dtype());
    }

    private DataFrame readJsonLayout(RedisOptions opt) {
        String match = opt.prefix().endsWith("*") ? opt.prefix() : opt.prefix() + "*";
        List<String> keys = scanAll(match, opt.scanCount());
        Collections.sort(keys);
        List<Map<String, String>> rows = new ArrayList<>();
        for (int i = 0; i < keys.size(); i += opt.pipelineBatch()) {
            List<String> slice = keys.subList(i, Math.min(i + opt.pipelineBatch(), keys.size()));
            List<String> values = mget(slice.toArray(new String[0]));
            for (int j = 0; j < values.size(); j++) {
                String json = values.get(j);
                if (json == null || json.isBlank()) continue;
                Map<String, String> row = jsonObjectToStringMap(json);
                if (!row.containsKey("_key")) row.put("_key", slice.get(j));
                rows.add(row);
            }
        }
        return mapsToDataFrame(rows, opt.dtype(), false);
    }

    private DataFrame readFrameJson(RedisOptions opt) {
        String key = opt.frameKey();
        String body = get(key);
        if (body == null || body.isBlank()) return DataFrame.create();
        String trimmed = body.stripLeading();
        List<Map<String, String>> rows = new ArrayList<>();
        if (trimmed.startsWith("[")) {
            try {
                Object parsed = Json.decode(body);
                if (parsed instanceof List<?> list) {
                    for (Object o : list) {
                        rows.add(objectToStringMap(o));
                    }
                }
            } catch (Exception e) {
                throw new RedisException("frame JSON decode failed for key " + key + ": " + e.getMessage(), e, "GET");
            }
        } else if (opt.layout() == RedisOptions.Layout.FRAME_JSONL || body.contains("\n")) {
            for (String line : body.split("\n")) {
                if (line.isBlank()) continue;
                rows.add(jsonObjectToStringMap(line));
            }
        } else {
            // single object
            rows.add(jsonObjectToStringMap(body));
        }
        return mapsToDataFrame(rows, opt.dtype(), false);
    }

    private void drainHgetall(List<Object[]> cmds, List<String> keyList, List<Map<String, String>> rows) {
        List<Object> replies = pipeline(cmds);
        for (int i = 0; i < replies.size(); i++) {
            Object rep = replies.get(i);
            Map<String, String> map = new LinkedHashMap<>();
            if (rep instanceof List<?> flat) {
                for (int j = 0; j + 1 < flat.size(); j += 2) {
                    map.put(RespClient.str(flat.get(j)), RespClient.str(flat.get(j + 1)));
                }
            }
            if (map.isEmpty()) continue;
            if (!map.containsKey("_key") && i < keyList.size()) {
                map.put("_key", keyList.get(i));
            }
            rows.add(map);
        }
    }

    // ── DataFrame materialization ──────────────────────────────────────────

    private static DataFrame mapsToDataFrame(List<Map<String, String>> rows,
                                              Map<String, Column.DType> dtype,
                                              boolean preferKeyAsId) {
        DataFrame df = DataFrame.create();
        if (rows == null || rows.isEmpty()) return df;

        // union of keys preserving first-seen order
        List<String> cols = new ArrayList<>();
        Set<String> seen = new LinkedHashSet<>();
        for (Map<String, String> row : rows) {
            for (String k : row.keySet()) {
                if ("_key".equals(k) && !preferKeyAsId) continue;
                if (seen.add(k)) cols.add(k);
            }
        }
        // drop internal _key if real id-like column exists
        if (cols.contains("_key") && (cols.contains("id") || cols.contains("ID"))) {
            cols.remove("_key");
        }

        for (String c : cols) {
            Column.DType dt = dtype != null && dtype.containsKey(c)
                    ? dtype.get(c)
                    : inferDType(rows, c);
            df.addColumn(c, dt);
        }
        for (Map<String, String> row : rows) {
            int ri = df.addEmptyRow();
            for (String c : cols) {
                String raw = row.get(c);
                if (raw == null) continue;
                Column.DType dt = df.column(c).dtype();
                df.set(ri, c, coerce(raw, dt));
            }
        }
        return df;
    }

    private static Column.DType inferDType(List<Map<String, String>> rows, String col) {
        boolean sawBool = false, sawInt = false, sawFloat = false, sawOther = false;
        int samples = 0;
        for (Map<String, String> row : rows) {
            String v = row.get(col);
            if (v == null || v.isEmpty()) continue;
            samples++;
            if ("true".equalsIgnoreCase(v) || "false".equalsIgnoreCase(v)) {
                sawBool = true;
            } else if (isLong(v)) {
                sawInt = true;
            } else if (isDouble(v)) {
                sawFloat = true;
            } else {
                sawOther = true;
                break;
            }
            if (samples >= 32) break;
        }
        if (sawOther || samples == 0) return Column.DType.STRING;
        if (sawBool && !sawInt && !sawFloat) return Column.DType.BOOLEAN;
        if (sawFloat) return Column.DType.FLOAT64;
        if (sawInt) return Column.DType.INT64;
        return Column.DType.STRING;
    }

    private static Object coerce(String raw, Column.DType dt) {
        if (raw == null) return null;
        try {
            return switch (dt) {
                case INT32 -> Integer.parseInt(raw.trim());
                case INT64 -> Long.parseLong(raw.trim());
                case FLOAT32 -> Float.parseFloat(raw.trim());
                case FLOAT64 -> Double.parseDouble(raw.trim());
                case BOOLEAN -> Boolean.parseBoolean(raw.trim())
                        || "1".equals(raw.trim())
                        || "yes".equalsIgnoreCase(raw.trim());
                default -> raw;
            };
        } catch (Exception e) {
            return raw;
        }
    }

    private static boolean isLong(String s) {
        try { Long.parseLong(s.trim()); return true; }
        catch (Exception e) { return false; }
    }

    private static boolean isDouble(String s) {
        try {
            Double.parseDouble(s.trim());
            return s.indexOf('.') >= 0 || s.toLowerCase(Locale.ROOT).contains("e");
        } catch (Exception e) { return false; }
    }

    private static String resolveIdColumn(DataFrame df, RedisOptions opt) {
        if (opt.idColumn() != null && !opt.idColumn().isBlank()) {
            if (!df.hasColumn(opt.idColumn())) {
                throw new RedisException("id column not found: " + opt.idColumn());
            }
            return opt.idColumn();
        }
        if (df.hasColumn("id")) return "id";
        if (df.hasColumn("ID")) return "ID";
        if (df.hasColumn("_id")) return "_id";
        return null;
    }

    private static Map<String, Object> rowToJsonMap(DataFrame df, int row, RedisOptions opt) {
        Map<String, Object> m = new LinkedHashMap<>();
        for (int c = 0; c < df.columnCount(); c++) {
            String name = df.column(c).name();
            Object v = df.get(row, name);
            if (v == null && !opt.includeNulls()) continue;
            m.put(name, cellToJson(v, opt));
        }
        return m;
    }

    private static Object cellToJson(Object v, RedisOptions opt) {
        if (v == null) return null;
        if (v instanceof float[] f) {
            if (opt.binaryVectorsAsBase64()) {
                return Map.of("__vec_b64", Base64.getEncoder().encodeToString(floatsToLe(f)),
                        "__dim", f.length);
            }
            List<Double> list = new ArrayList<>(f.length);
            for (float x : f) list.add((double) x);
            return list;
        }
        if (v instanceof double[] d) {
            List<Double> list = new ArrayList<>(d.length);
            for (double x : d) list.add(x);
            return list;
        }
        if (v instanceof byte[] b) {
            return Base64.getEncoder().encodeToString(b);
        }
        // DataValue / complex types → string
        if (!(v instanceof Number) && !(v instanceof Boolean) && !(v instanceof String)
                && !(v instanceof Map) && !(v instanceof Collection)) {
            return String.valueOf(v);
        }
        return v;
    }

    private static String cellToRedis(Object v, RedisOptions opt) {
        if (v == null) return "";
        if (v instanceof float[] f) {
            if (opt.binaryVectorsAsBase64()) {
                return Base64.getEncoder().encodeToString(floatsToLe(f));
            }
            return Json.encode(f);
        }
        if (v instanceof double[] d) {
            return Json.encode(d);
        }
        if (v instanceof byte[] b) {
            return Base64.getEncoder().encodeToString(b);
        }
        if (v instanceof Boolean b) return b ? "true" : "false";
        if (v instanceof Number || v instanceof String) return String.valueOf(v);
        // nested → json
        if (v instanceof Map || v instanceof Collection) return Json.encode(v);
        return String.valueOf(v);
    }

    private static String valueToString(Object v) {
        if (v == null) return "";
        if (v instanceof byte[] b) return new String(b, StandardCharsets.UTF_8);
        return String.valueOf(v);
    }

    private static byte[] floatsToLe(float[] f) {
        byte[] out = new byte[f.length * 4];
        for (int i = 0; i < f.length; i++) {
            int bits = Float.floatToIntBits(f[i]);
            int o = i * 4;
            out[o] = (byte) (bits);
            out[o + 1] = (byte) (bits >>> 8);
            out[o + 2] = (byte) (bits >>> 16);
            out[o + 3] = (byte) (bits >>> 24);
        }
        return out;
    }

    @SuppressWarnings("unchecked")
    private static Map<String, String> jsonObjectToStringMap(String json) {
        try {
            Object parsed = Json.decode(json);
            return objectToStringMap(parsed);
        } catch (Exception e) {
            Map<String, String> m = new LinkedHashMap<>();
            m.put("value", json);
            return m;
        }
    }

    @SuppressWarnings("unchecked")
    private static Map<String, String> objectToStringMap(Object parsed) {
        Map<String, String> m = new LinkedHashMap<>();
        if (parsed instanceof Map<?, ?> map) {
            for (Map.Entry<?, ?> e : map.entrySet()) {
                Object v = e.getValue();
                if (v == null) m.put(String.valueOf(e.getKey()), null);
                else if (v instanceof String || v instanceof Number || v instanceof Boolean) {
                    m.put(String.valueOf(e.getKey()), String.valueOf(v));
                } else {
                    m.put(String.valueOf(e.getKey()), Json.encode(v));
                }
            }
        } else if (parsed != null) {
            m.put("value", String.valueOf(parsed));
        }
        return m;
    }

    private static RedisException wrap(VectorStoreException e, String cmd) {
        return new RedisException(e.getMessage(), e, cmd);
    }

    @Override
    public void close() {
        if (ownClient) {
            try { client.close(); } catch (Exception ignored) {}
        }
    }

    @Override
    public String toString() {
        return "Redis{" + host + ":" + port + "}";
    }
}
