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
package org.bytedeco.pytorch.utils.doris;

import com.google.gson.Gson;
import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import org.bytedeco.pytorch.dataframe.Column;
import org.bytedeco.pytorch.dataframe.DataFrame;
import org.bytedeco.pytorch.utils.lake.LakeException;
import org.bytedeco.pytorch.utils.lake.LakeFormat;
import org.bytedeco.pytorch.utils.lake.LakeMetrics;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.util.Base64;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.UUID;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Apache Doris HTTP Stream Load client (high-throughput bulk ingest).
 *
 * <p>Protocol: {@code PUT http://fe:http_port/api/{db}/{table}/_stream_load}
 * with Basic Auth, Expect: 100-continue, and format-specific headers.
 * Label provides idempotency for retries.</p>
 *
 * @see <a href="https://doris.apache.org/docs/data-operate/import/import-way/stream-load-manual">Stream Load</a>
 */
public final class DorisStreamLoad implements AutoCloseable {

    private static final Gson GSON = new Gson();
    private static final AtomicLong LABEL_SEQ = new AtomicLong();

    private final DorisOptions options;
    private final HttpClient http;
    private final LakeMetrics metrics;
    private final boolean ownHttp;

    public DorisStreamLoad(DorisOptions options) {
        this(options, null, LakeMetrics.of("doris-stream-load"));
    }

    public DorisStreamLoad(DorisOptions options, HttpClient http, LakeMetrics metrics) {
        this.options = Objects.requireNonNull(options, "options");
        if (http == null) {
            this.http = HttpClient.newBuilder()
                    .connectTimeout(Duration.ofMillis(Math.max(1, options.connectTimeoutMs())))
                    .followRedirects(HttpClient.Redirect.NORMAL)
                    .build();
            this.ownHttp = true;
        } else {
            this.http = http;
            this.ownHttp = false;
        }
        this.metrics = metrics == null ? LakeMetrics.of("doris-stream-load") : metrics;
    }

    public DorisOptions options() {
        return options;
    }

    public LakeMetrics metrics() {
        return metrics;
    }

    /**
     * Stream-load a DataFrame using configured format (default JSON rows array / NDJSON-style body).
     */
    public Result load(DataFrame df) {
        return load(df, null);
    }

    public Result load(DataFrame df, String label) {
        Objects.requireNonNull(df, "dataframe");
        if (options.database() == null || options.table() == null) {
            throw new LakeException(LakeFormat.DORIS, "stream_load", "database and table required");
        }
        byte[] body;
        try {
            body = encode(df);
        } catch (IOException e) {
            metrics.recordFailure();
            throw new LakeException(LakeFormat.DORIS, "stream_load.encode", e.getMessage(), e);
        }
        return loadBytes(body, label, df.rowCount());
    }

    public Result loadBytes(byte[] body, String label, long rowHint) {
        Objects.requireNonNull(body, "body");
        String effectiveLabel = (label == null || label.isBlank())
                ? options.labelPrefix() + "-" + UUID.randomUUID() + "-" + LABEL_SEQ.incrementAndGet()
                : label;
        String url = options.httpBaseUrl() + options.streamLoadPath();
        long t0 = System.nanoTime();
        try {
            HttpRequest.Builder rb = HttpRequest.newBuilder()
                    .uri(URI.create(url))
                    .timeout(Duration.ofMillis(Math.max(1, options.socketTimeoutMs())))
                    .header("Authorization", basicAuth(options.username(), options.password()))
                    .header("Expect", "100-continue")
                    .header("label", effectiveLabel)
                    .header("format", options.loadFormat().name().toLowerCase())
                    .header("max_filter_ratio",
                            Double.toString(options.maxFilterRatioPercent() / 100.0));
            if (options.twoPhaseCommit()) {
                rb.header("two_phase_commit", "true");
            }
            if (options.partialColumns()) {
                rb.header("partial_columns", "true");
            }
            switch (options.loadFormat()) {
                case JSON -> {
                    rb.header("Content-Type", "application/json; charset=utf-8");
                    rb.header("read_json_by_line", "true");
                    rb.header("strip_outer_array", "false");
                }
                case CSV -> {
                    rb.header("Content-Type", "text/plain; charset=utf-8");
                    rb.header("column_separator", options.columnSeparator());
                    rb.header("line_delimiter", options.lineDelimiter());
                }
                case PARQUET -> rb.header("Content-Type", "application/octet-stream");
            }
            for (Map.Entry<String, String> e : options.streamLoadHeaders().entrySet()) {
                rb.header(e.getKey(), e.getValue());
            }
            HttpRequest request = rb.PUT(HttpRequest.BodyPublishers.ofByteArray(body)).build();
            HttpResponse<String> response = http.send(request, HttpResponse.BodyHandlers.ofString(StandardCharsets.UTF_8));
            long latency = System.nanoTime() - t0;
            Result result = parseResult(response.statusCode(), response.body(), effectiveLabel, body.length, rowHint);
            if (result.success()) {
                metrics.recordWrite(result.numberTotalRows() > 0 ? result.numberTotalRows() : rowHint,
                        body.length, latency);
            } else {
                metrics.recordFailure();
                throw new LakeException(LakeFormat.DORIS, "stream_load",
                        "label=" + effectiveLabel + " status=" + result.status()
                                + " msg=" + result.message() + " body=" + truncate(response.body(), 512));
            }
            return result;
        } catch (LakeException e) {
            throw e;
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            metrics.recordFailure();
            throw new LakeException(LakeFormat.DORIS, "stream_load", "interrupted", e);
        } catch (Exception e) {
            metrics.recordFailure();
            throw new LakeException(LakeFormat.DORIS, "stream_load", e.getMessage(), e);
        }
    }

    /** Build request headers map (for tests / dry-run). */
    public Map<String, String> buildHeaders(String label) {
        Map<String, String> h = new LinkedHashMap<>();
        h.put("Authorization", basicAuth(options.username(), options.password()));
        h.put("Expect", "100-continue");
        h.put("label", label == null ? options.labelPrefix() + "-dry" : label);
        h.put("format", options.loadFormat().name().toLowerCase());
        h.put("max_filter_ratio", Double.toString(options.maxFilterRatioPercent() / 100.0));
        if (options.twoPhaseCommit()) h.put("two_phase_commit", "true");
        if (options.partialColumns()) h.put("partial_columns", "true");
        if (options.loadFormat() == DorisOptions.LoadFormat.JSON) {
            h.put("read_json_by_line", "true");
        }
        if (options.loadFormat() == DorisOptions.LoadFormat.CSV) {
            h.put("column_separator", options.columnSeparator());
            h.put("line_delimiter", options.lineDelimiter());
        }
        h.putAll(options.streamLoadHeaders());
        return h;
    }

    public byte[] encode(DataFrame df) throws IOException {
        return switch (options.loadFormat()) {
            case JSON -> encodeJsonLines(df);
            case CSV -> encodeCsv(df);
            case PARQUET -> encodeParquet(df);
        };
    }

    private static byte[] encodeJsonLines(DataFrame df) {
        List<Map<String, Object>> rows = df.toRecords();
        StringBuilder sb = new StringBuilder(Math.max(64, rows.size() * 64));
        for (int i = 0; i < rows.size(); i++) {
            if (i > 0) sb.append('\n');
            sb.append(GSON.toJson(rows.get(i)));
        }
        if (!rows.isEmpty()) sb.append('\n');
        return sb.toString().getBytes(StandardCharsets.UTF_8);
    }

    private byte[] encodeCsv(DataFrame df) {
        String sep = options.columnSeparator();
        String nl = options.lineDelimiter();
        int cols = df.columnCount();
        StringBuilder sb = new StringBuilder();
        for (int r = 0; r < df.rowCount(); r++) {
            for (int c = 0; c < cols; c++) {
                if (c > 0) sb.append(sep);
                Object v = df.column(c).get(r);
                sb.append(v == null ? "" : String.valueOf(v));
            }
            sb.append(nl);
        }
        return sb.toString().getBytes(StandardCharsets.UTF_8);
    }

    private static byte[] encodeParquet(DataFrame df) throws IOException {
        // Write to temp path then read bytes — DataFrame has writeParquet(path)
        java.nio.file.Path tmp = java.nio.file.Files.createTempFile("doris-sl-", ".parquet");
        try {
            df.writeParquet(tmp.toString());
            return java.nio.file.Files.readAllBytes(tmp);
        } catch (Exception e) {
            throw new IOException("parquet encode failed: " + e.getMessage(), e);
        } finally {
            try { java.nio.file.Files.deleteIfExists(tmp); } catch (IOException ignored) {}
        }
    }

    private static Result parseResult(int httpStatus, String body, String label, long bytes, long rowHint) {
        String status = "Fail";
        String message = body;
        String txnId = null;
        long loaded = 0;
        long filtered = 0;
        long total = rowHint;
        long unselected = 0;
        double loadBytes = bytes;
        try {
            if (body != null && !body.isBlank() && body.trim().startsWith("{")) {
                JsonObject o = JsonParser.parseString(body).getAsJsonObject();
                status = text(o, "Status", text(o, "status", status));
                message = text(o, "Message", text(o, "msg", message));
                txnId = text(o, "TxnId", text(o, "txnId", null));
                loaded = longVal(o, "NumberLoadedRows", longVal(o, "numberLoadedRows", 0));
                filtered = longVal(o, "NumberFilteredRows", longVal(o, "numberFilteredRows", 0));
                total = longVal(o, "NumberTotalRows", longVal(o, "numberTotalRows", rowHint));
                unselected = longVal(o, "NumberUnselectedRows", 0);
                loadBytes = doubleVal(o, "LoadBytes", bytes);
            } else if (httpStatus >= 200 && httpStatus < 300) {
                status = "Success";
            }
        } catch (Exception e) {
            message = "parse response failed: " + e.getMessage() + "; raw=" + truncate(body, 256);
        }
        boolean ok = "Success".equalsIgnoreCase(status)
                || "Publish Timeout".equalsIgnoreCase(status)
                || "Label Already Exists".equalsIgnoreCase(status);
        if (httpStatus >= 400) ok = false;
        return new Result(ok, status, message, label, txnId, httpStatus,
                total, loaded, filtered, unselected, (long) loadBytes);
    }

    private static String text(JsonObject o, String k, String dft) {
        JsonElement e = o.get(k);
        if (e == null || e.isJsonNull()) return dft;
        return e.getAsString();
    }

    private static long longVal(JsonObject o, String k, long dft) {
        JsonElement e = o.get(k);
        if (e == null || e.isJsonNull()) return dft;
        try { return e.getAsLong(); } catch (Exception ex) {
            try { return Long.parseLong(e.getAsString()); } catch (Exception ex2) { return dft; }
        }
    }

    private static double doubleVal(JsonObject o, String k, double dft) {
        JsonElement e = o.get(k);
        if (e == null || e.isJsonNull()) return dft;
        try { return e.getAsDouble(); } catch (Exception ex) { return dft; }
    }

    static String basicAuth(String user, String pass) {
        String raw = (user == null ? "" : user) + ":" + (pass == null ? "" : pass);
        return "Basic " + Base64.getEncoder().encodeToString(raw.getBytes(StandardCharsets.UTF_8));
    }

    private static String truncate(String s, int max) {
        if (s == null) return "";
        return s.length() <= max ? s : s.substring(0, max) + "...";
    }

    @Override
    public void close() {
        // HttpClient has no close in Java 17; nothing to release for shared clients
    }

    /**
     * Stream Load response summary.
     */
    public record Result(
            boolean success,
            String status,
            String message,
            String label,
            String txnId,
            int httpStatus,
            long numberTotalRows,
            long numberLoadedRows,
            long numberFilteredRows,
            long numberUnselectedRows,
            long loadBytes
    ) {}
}
