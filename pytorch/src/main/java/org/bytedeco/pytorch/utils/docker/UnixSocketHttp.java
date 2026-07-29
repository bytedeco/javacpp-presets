package org.bytedeco.pytorch.utils.docker;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.StandardProtocolFamily;
import java.net.UnixDomainSocketAddress;
import java.nio.ByteBuffer;
import java.nio.channels.Channels;
import java.nio.channels.SocketChannel;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.time.Duration;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;

/**
 * Minimal HTTP/1.1 client over a Unix domain socket (Docker Engine default transport).
 *
 * <p>JDK {@code HttpClient} does not speak Unix sockets portably on Java 17, so we
 * implement a tiny request/response exchange suitable for Docker's REST API.
 */
public final class UnixSocketHttp implements AutoCloseable {

    private final Path socketPath;
    private final Duration timeout;
    private final String defaultHostHeader;

    public UnixSocketHttp(Path socketPath, Duration timeout) {
        this.socketPath = Objects.requireNonNull(socketPath, "socketPath");
        this.timeout = timeout == null ? Duration.ofSeconds(30) : timeout;
        this.defaultHostHeader = "localhost";
    }

    public static UnixSocketHttp open(String unixUrl, Duration timeout) {
        Objects.requireNonNull(unixUrl, "unixUrl");
        String path = unixUrl;
        if (path.startsWith("unix://")) {
            path = path.substring("unix://".length());
        }
        // docker sometimes uses unix:///var/run/... (three slashes → absolute)
        if (path.startsWith("//")) {
            // unix:////rare — normalize
            while (path.startsWith("//")) path = path.substring(1);
        }
        if (!path.startsWith("/")) path = "/" + path;
        return new UnixSocketHttp(Path.of(path), timeout);
    }

    public Response exchange(String method, String pathAndQuery, Map<String, String> headers, byte[] body)
            throws IOException {
        Objects.requireNonNull(method, "method");
        String path = pathAndQuery == null || pathAndQuery.isEmpty() ? "/" : pathAndQuery;
        if (!path.startsWith("/")) path = "/" + path;

        byte[] payload = body == null ? new byte[0] : body;
        String m = method.toUpperCase(Locale.ROOT);

        StringBuilder req = new StringBuilder(256);
        req.append(m).append(' ').append(path).append(" HTTP/1.1\r\n");
        req.append("Host: ").append(defaultHostHeader).append("\r\n");
        req.append("User-Agent: jnitorch-docker/1.0\r\n");
        req.append("Accept: application/json, */*\r\n");
        req.append("Connection: close\r\n");
        boolean hasContentType = false;
        boolean hasContentLength = false;
        if (headers != null) {
            for (Map.Entry<String, String> h : headers.entrySet()) {
                if (h.getKey() == null || h.getValue() == null) continue;
                String hk = h.getKey();
                if (hk.equalsIgnoreCase("Host") || hk.equalsIgnoreCase("Content-Length")) {
                    if (hk.equalsIgnoreCase("Content-Length")) hasContentLength = true;
                    continue;
                }
                if (hk.equalsIgnoreCase("Content-Type")) hasContentType = true;
                req.append(hk).append(": ").append(h.getValue()).append("\r\n");
            }
        }
        if (payload.length > 0 && !hasContentType) {
            req.append("Content-Type: application/json\r\n");
        }
        if (!hasContentLength) {
            req.append("Content-Length: ").append(payload.length).append("\r\n");
        }
        req.append("\r\n");

        UnixDomainSocketAddress addr = UnixDomainSocketAddress.of(socketPath);
        try (SocketChannel ch = SocketChannel.open(StandardProtocolFamily.UNIX)) {
            ch.configureBlocking(true);
            // connect timeout approximated via SO timeout on streams after connect
            if (!ch.connect(addr)) {
                // non-blocking would need finishConnect; we are blocking
                ch.finishConnect();
            }
            OutputStream out = Channels.newOutputStream(ch);
            out.write(req.toString().getBytes(StandardCharsets.US_ASCII));
            if (payload.length > 0) out.write(payload);
            out.flush();

            InputStream in = Channels.newInputStream(ch);
            return readResponse(in);
        }
    }

    public Response get(String path) throws IOException {
        return exchange("GET", path, null, null);
    }

    public Response delete(String path) throws IOException {
        return exchange("DELETE", path, null, null);
    }

    public Response post(String path, byte[] body, String contentType) throws IOException {
        Map<String, String> h = new LinkedHashMap<>();
        if (contentType != null) h.put("Content-Type", contentType);
        return exchange("POST", path, h, body);
    }

    public Response postJson(String path, String json) throws IOException {
        byte[] body = json == null ? new byte[0] : json.getBytes(StandardCharsets.UTF_8);
        return post(path, body, "application/json");
    }

    private static Response readResponse(InputStream in) throws IOException {
        ByteArrayOutputStream headerBuf = new ByteArrayOutputStream(512);
        // read until \r\n\r\n
        int state = 0; // 0 none, 1 \r, 2 \r\n, 3 \r\n\r, 4 done
        while (state != 4) {
            int b = in.read();
            if (b < 0) throw new IOException("unexpected EOF reading HTTP headers from unix socket");
            headerBuf.write(b);
            if (b == '\r') {
                if (state == 0 || state == 2) state++;
                else state = 1;
            } else if (b == '\n') {
                if (state == 1) state = 2;
                else if (state == 3) state = 4;
                else state = 0;
            } else {
                state = 0;
            }
            if (headerBuf.size() > 1024 * 1024) {
                throw new IOException("HTTP headers too large over unix socket");
            }
        }
        String headerText = headerBuf.toString(StandardCharsets.US_ASCII);
        String[] lines = headerText.split("\r\n");
        if (lines.length == 0) throw new IOException("empty HTTP response");
        String statusLine = lines[0];
        String[] sp = statusLine.split(" ", 3);
        if (sp.length < 2) throw new IOException("bad status line: " + statusLine);
        int code = Integer.parseInt(sp[1]);
        String reason = sp.length >= 3 ? sp[2] : "";
        Map<String, String> headers = new LinkedHashMap<>();
        for (int i = 1; i < lines.length; i++) {
            String line = lines[i];
            if (line.isEmpty()) continue;
            int c = line.indexOf(':');
            if (c > 0) {
                headers.put(line.substring(0, c).trim().toLowerCase(Locale.ROOT),
                        line.substring(c + 1).trim());
            }
        }

        byte[] body;
        String te = headers.getOrDefault("transfer-encoding", "");
        if (te.toLowerCase(Locale.ROOT).contains("chunked")) {
            body = readChunked(in);
        } else if (headers.containsKey("content-length")) {
            int len = Integer.parseInt(headers.get("content-length").trim());
            body = in.readNBytes(len);
        } else {
            // connection close — read all
            body = in.readAllBytes();
        }
        return new Response(code, reason, Collections.unmodifiableMap(headers), body);
    }

    private static byte[] readChunked(InputStream in) throws IOException {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        while (true) {
            String sizeLine = readLine(in);
            if (sizeLine == null) break;
            sizeLine = sizeLine.trim();
            int semi = sizeLine.indexOf(';');
            if (semi >= 0) sizeLine = sizeLine.substring(0, semi).trim();
            int size = Integer.parseInt(sizeLine, 16);
            if (size == 0) {
                // trailers
                while (true) {
                    String t = readLine(in);
                    if (t == null || t.isEmpty()) break;
                }
                break;
            }
            byte[] chunk = in.readNBytes(size);
            out.write(chunk);
            // trailing CRLF
            readLine(in);
        }
        return out.toByteArray();
    }

    private static String readLine(InputStream in) throws IOException {
        ByteArrayOutputStream buf = new ByteArrayOutputStream(64);
        int prev = -1;
        while (true) {
            int b = in.read();
            if (b < 0) {
                if (buf.size() == 0) return null;
                break;
            }
            if (b == '\n') {
                // drop optional \r
                break;
            }
            if (prev == '\r') buf.write('\r');
            if (b != '\r') buf.write(b);
            prev = b;
        }
        return buf.toString(StandardCharsets.US_ASCII);
    }

    @Override
    public void close() {
        // stateless per-exchange channels
    }

    public static final class Response {
        public final int status;
        public final String reason;
        public final Map<String, String> headers;
        public final byte[] body;

        public Response(int status, String reason, Map<String, String> headers, byte[] body) {
            this.status = status;
            this.reason = reason == null ? "" : reason;
            this.headers = headers == null ? Map.of() : headers;
            this.body = body == null ? new byte[0] : body;
        }

        public String bodyString() {
            return new String(body, StandardCharsets.UTF_8);
        }

        public boolean ok() {
            return status >= 200 && status < 300;
        }

        @Override
        public String toString() {
            return "UnixHttpResponse{status=" + status + ", bytes=" + body.length + "}";
        }
    }
}
