package org.bytedeco.pytorch.dataframe.vectorstore.redis;

import org.bytedeco.pytorch.dataframe.vectorstore.VectorStoreException;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.ByteArrayOutputStream;
import java.io.Closeable;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.InetSocketAddress;
import java.net.Socket;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;

/**
 * Minimal Redis RESP2 client — enough for RediSearch vector commands.
 * No Jedis / Lettuce dependency.
 *
 * <p>Supports simple strings, errors, integers, bulk strings, and arrays
 * (including nested arrays returned by {@code FT.SEARCH}).
 */
public final class RespClient implements Closeable {

    private final String host;
    private final int port;
    private final String username;
    private final String password;
    private final Duration timeout;
    private Socket socket;
    private InputStream in;
    private OutputStream out;
    private final Object lock = new Object();

    public RespClient(String host, int port, String username, String password, Duration timeout) {
        this.host = host == null ? "127.0.0.1" : host;
        this.port = port <= 0 ? 6379 : port;
        this.username = username;
        this.password = password;
        this.timeout = timeout == null ? Duration.ofSeconds(10) : timeout;
    }

    public static RespClient connect(String host, int port) {
        return connect(host, port, null, null, Duration.ofSeconds(10));
    }

    public static RespClient connect(String host, int port, String username, String password, Duration timeout) {
        RespClient c = new RespClient(host, port, username, password, timeout);
        c.ensureConnected();
        return c;
    }

    public void ensureConnected() {
        synchronized (lock) {
            if (socket != null && socket.isConnected() && !socket.isClosed()) return;
            try {
                socket = new Socket();
                socket.connect(new InetSocketAddress(host, port), (int) timeout.toMillis());
                socket.setSoTimeout((int) timeout.toMillis());
                socket.setTcpNoDelay(true);
                in = new BufferedInputStream(socket.getInputStream());
                out = new BufferedOutputStream(socket.getOutputStream());
                if (password != null && !password.isEmpty()) {
                    if (username != null && !username.isEmpty()) {
                        Object r = rawCommand("AUTH", username, password);
                        expectOk(r, "AUTH");
                    } else {
                        Object r = rawCommand("AUTH", password);
                        expectOk(r, "AUTH");
                    }
                }
            } catch (IOException e) {
                closeQuietly();
                throw new VectorStoreException("redis connect " + host + ":" + port + ": " + e.getMessage(), e, -1, "redis");
            }
        }
    }

    /** Send a command; args are String or byte[]. Returns parsed RESP value. */
    public Object call(Object... args) {
        synchronized (lock) {
            ensureConnected();
            try {
                return rawCommand(args);
            } catch (IOException e) {
                closeQuietly();
                throw new VectorStoreException("redis I/O: " + e.getMessage(), e, -1, "redis");
            }
        }
    }

    public String callString(Object... args) {
        Object r = call(args);
        if (r == null) return null;
        if (r instanceof byte[] b) return new String(b, StandardCharsets.UTF_8);
        if (r instanceof String s) return s;
        return String.valueOf(r);
    }

    public long callLong(Object... args) {
        Object r = call(args);
        if (r instanceof Long l) return l;
        if (r instanceof Integer i) return i.longValue();
        if (r instanceof byte[] b) {
            try { return Long.parseLong(new String(b, StandardCharsets.UTF_8)); }
            catch (NumberFormatException e) { return 0L; }
        }
        if (r instanceof String s) {
            try { return Long.parseLong(s); } catch (NumberFormatException e) { return 0L; }
        }
        return 0L;
    }

    @SuppressWarnings("unchecked")
    public List<Object> callArray(Object... args) {
        Object r = call(args);
        if (r == null) return List.of();
        if (r instanceof List<?> l) return (List<Object>) l;
        throw new VectorStoreException("redis expected array, got " + r.getClass().getSimpleName(), -1, "redis");
    }

    /**
     * Pipeline multiple commands in one write/flush, then read all replies in order.
     * Each element of {@code commands} is one command's args (String or byte[]).
     * Errors on individual commands become {@link VectorStoreException} entries only if
     * the server returns a RESP error — the exception is thrown for that slot.
     */
    public List<Object> pipeline(List<Object[]> commands) {
        if (commands == null || commands.isEmpty()) return List.of();
        synchronized (lock) {
            ensureConnected();
            try {
                for (Object[] args : commands) {
                    writeCommand(args);
                }
                out.flush();
                List<Object> replies = new ArrayList<>(commands.size());
                for (int i = 0; i < commands.size(); i++) {
                    replies.add(readResp());
                }
                return replies;
            } catch (IOException e) {
                closeQuietly();
                throw new VectorStoreException("redis pipeline I/O: " + e.getMessage(), e, -1, "redis");
            }
        }
    }

    private void writeCommand(Object... args) throws IOException {
        ByteArrayOutputStream buf = new ByteArrayOutputStream(128);
        writeCrLf(buf, "*" + args.length);
        for (Object a : args) {
            byte[] bytes;
            if (a == null) {
                bytes = new byte[0];
            } else if (a instanceof byte[] b) {
                bytes = b;
            } else {
                bytes = String.valueOf(a).getBytes(StandardCharsets.UTF_8);
            }
            writeCrLf(buf, "$" + bytes.length);
            buf.write(bytes);
            buf.write('\r');
            buf.write('\n');
        }
        out.write(buf.toByteArray());
    }

    private Object rawCommand(Object... args) throws IOException {
        writeCommand(args);
        out.flush();
        return readResp();
    }

    private Object readResp() throws IOException {
        int type = in.read();
        if (type < 0) throw new IOException("redis connection closed");
        switch (type) {
            case '+': // simple string
                return readLine();
            case '-': { // error
                String err = readLine();
                throw new VectorStoreException("redis error: " + err, -1, "redis");
            }
            case ':': { // integer
                String n = readLine();
                try { return Long.parseLong(n); }
                catch (NumberFormatException e) { return 0L; }
            }
            case '$': { // bulk string
                String lenStr = readLine();
                int len = Integer.parseInt(lenStr);
                if (len < 0) return null;
                byte[] data = in.readNBytes(len);
                if (data.length != len) throw new IOException("short bulk read");
                // consume trailing CRLF
                if (in.read() != '\r' || in.read() != '\n') {
                    throw new IOException("bad bulk terminator");
                }
                return data;
            }
            case '*': { // array
                String nStr = readLine();
                int n = Integer.parseInt(nStr);
                if (n < 0) return null;
                List<Object> list = new ArrayList<>(n);
                for (int i = 0; i < n; i++) list.add(readResp());
                return list;
            }
            default:
                throw new IOException("unknown RESP type: " + (char) type);
        }
    }

    private String readLine() throws IOException {
        ByteArrayOutputStream bos = new ByteArrayOutputStream(64);
        while (true) {
            int c = in.read();
            if (c < 0) throw new IOException("EOF in RESP line");
            if (c == '\r') {
                int n = in.read();
                if (n != '\n') throw new IOException("expected LF after CR");
                break;
            }
            bos.write(c);
        }
        return bos.toString(StandardCharsets.UTF_8);
    }

    private static void writeCrLf(ByteArrayOutputStream buf, String s) throws IOException {
        buf.write(s.getBytes(StandardCharsets.UTF_8));
        buf.write('\r');
        buf.write('\n');
    }

    private static void expectOk(Object r, String cmd) {
        String s;
        if (r instanceof byte[] b) s = new String(b, StandardCharsets.UTF_8);
        else s = String.valueOf(r);
        if (s == null || !("OK".equalsIgnoreCase(s) || "PONG".equalsIgnoreCase(s))) {
            // AUTH may return OK as simple string
            if (!"OK".equalsIgnoreCase(String.valueOf(r))) {
                // still accept truthy
                if (r instanceof Long l && l == 1L) return;
            }
        }
    }

    private void closeQuietly() {
        try { close(); } catch (Exception ignored) {}
    }

    @Override
    public void close() {
        synchronized (lock) {
            try { if (in != null) in.close(); } catch (Exception ignored) {}
            try { if (out != null) out.close(); } catch (Exception ignored) {}
            try { if (socket != null) socket.close(); } catch (Exception ignored) {}
            in = null;
            out = null;
            socket = null;
        }
    }

    /** UTF-8 decode helper for bulk replies. */
    public static String str(Object o) {
        if (o == null) return null;
        if (o instanceof byte[] b) return new String(b, StandardCharsets.UTF_8);
        return String.valueOf(o);
    }
}
