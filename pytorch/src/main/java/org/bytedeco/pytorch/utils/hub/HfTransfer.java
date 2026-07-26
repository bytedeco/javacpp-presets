/*
 * Copyright (C) 2020-2026 Eduardo Gonzalez, Hervé Guillemet, Samuel Audet
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
package org.bytedeco.pytorch.utils.hub;
import org.bytedeco.pytorch.c10.*;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.URI;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.atomic.LongAdder;
import java.util.function.BiConsumer;

/**
 * Accelerated multi-connection HTTP downloader inspired by {@code hf_transfer}.
 *
 * <p>Splits large responses into ranged GETs across a fixed thread pool, then
 * stitches the parts. Small responses use a single connection. Pure Java — no
 * native Rust crate required.
 *
 * <pre>{@code
 * HfTransfer xfer = HfTransfer.create().maxWorkers(8).chunkSize(8 << 20).build();
 * byte[] body = xfer.download("https://huggingface.co/.../model.safetensors", token);
 * xfer.downloadToFile(url, token, Path.of("model.safetensors"), (done, total) -> ...);
 * }</pre>
 */
public final class HfTransfer {

    public static final int DEFAULT_WORKERS = 4;
    public static final int DEFAULT_CHUNK = 8 * 1024 * 1024; // 8 MiB
    public static final int DEFAULT_CONNECT_TIMEOUT_MS = 15_000;
    public static final int DEFAULT_READ_TIMEOUT_MS = 60_000;

    private final int maxWorkers;
    private final int chunkSize;
    private final int connectTimeoutMs;
    private final int readTimeoutMs;
    private final boolean forceSingle;
    private final LongAdder bytesDownloaded = new LongAdder();

    private HfTransfer(Builder b) {
        this.maxWorkers = Math.max(1, b.maxWorkers);
        this.chunkSize = Math.max(64 * 1024, b.chunkSize);
        this.connectTimeoutMs = b.connectTimeoutMs;
        this.readTimeoutMs = b.readTimeoutMs;
        this.forceSingle = b.forceSingle;
    }

    public static Builder create() {
        return new Builder();
    }

    public static Builder builder() {
        return new Builder();
    }

    public long bytesDownloaded() {
        return bytesDownloaded.sum();
    }

    public void resetStats() {
        bytesDownloaded.reset();
    }

    /** Download entire URL body into a byte array (multi-connection when possible). */
    public byte[] download(String url, String token) throws IOException {
        Objects.requireNonNull(url, "url");
        long contentLength = -1L;
        boolean acceptRanges = false;
        if (!forceSingle) {
            try {
                HeadInfo head = head(url, token);
                contentLength = head.contentLength;
                acceptRanges = head.acceptRanges;
            } catch (IOException ignored) {
                // fall through to single GET
            }
        }
        if (forceSingle || !acceptRanges || contentLength <= chunkSize || contentLength <= 0) {
            return singleGet(url, token);
        }
        return multiGet(url, token, contentLength);
    }

    /**
     * Stream download to a file with optional progress callback
     * {@code (bytesDone, totalBytesOr-1)}.
     *
     * <p>Uses a streaming single-connection GET so multi-GB weight files never
     * materialize as a full {@code byte[]} in heap. (Parallel multi-GET still
     * available via {@link #download} for smaller payloads.)
     */
    public Path downloadToFile(String url, String token, Path dest,
                               BiConsumer<Long, Long> progress) throws IOException {
        Objects.requireNonNull(dest, "dest");
        Files.createDirectories(dest.getParent() == null ? Path.of(".") : dest.getParent());
        Path tmp = dest.resolveSibling(dest.getFileName().toString() + ".part");
        long total = -1L;
        try {
            HeadInfo head = null;
            try { head = head(url, token); total = head.contentLength; } catch (IOException ignored) {}
            streamGetToFile(url, token, tmp, total, progress);
            Files.move(tmp, dest, StandardCopyOption.REPLACE_EXISTING, StandardCopyOption.ATOMIC_MOVE);
        } catch (IOException e) {
            try { Files.deleteIfExists(tmp); } catch (IOException ignored) {}
            throw e;
        }
        return dest;
    }

    /** Streaming GET that never buffers the whole body in memory. */
    private void streamGetToFile(String url, String token, Path dest, long totalHint,
                                 BiConsumer<Long, Long> progress) throws IOException {
        HttpURLConnection conn = open(url, token, "GET");
        try {
            int code = conn.getResponseCode();
            if (code == 301 || code == 302 || code == 303 || code == 307 || code == 308) {
                String loc = conn.getHeaderField("Location");
                conn.disconnect();
                if (loc == null) throw new IOException("Redirect without Location: " + url);
                // Resolve relative Location against original URL
                String resolved = loc;
                if (!loc.startsWith("http://") && !loc.startsWith("https://")) {
                    URI base = URI.create(url);
                    if (loc.startsWith("/")) {
                        resolved = base.getScheme() + "://" + base.getHost() + loc;
                    } else {
                        resolved = base.resolve(loc).toString();
                    }
                }
                streamGetToFile(resolved, token, dest, totalHint, progress);
                return;
            }
            if (code >= 400) {
                throw new IOException("GET " + url + " -> HTTP " + code);
            }
            long total = totalHint > 0 ? totalHint : conn.getContentLengthLong();
            long done = 0L;
            try (InputStream in = conn.getInputStream();
                 var out = Files.newOutputStream(dest,
                         StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING)) {
                byte[] buf = new byte[1 << 20]; // 1 MiB
                int n;
                while ((n = in.read(buf)) >= 0) {
                    if (n == 0) continue;
                    out.write(buf, 0, n);
                    done += n;
                    bytesDownloaded.add(n);
                    if (progress != null) progress.accept(done, total);
                }
            }
        } finally {
            conn.disconnect();
        }
    }

    public Path downloadToFile(String url, String token, Path dest) throws IOException {
        return downloadToFile(url, token, dest, null);
    }

    // ---- internals -------------------------------------------------------

    private record HeadInfo(long contentLength, boolean acceptRanges) {}

    private HeadInfo head(String url, String token) throws IOException {
        HttpURLConnection conn = open(url, token, "HEAD");
        try {
            int code = conn.getResponseCode();
            if (code >= 400) {
                throw new IOException("HEAD " + url + " -> HTTP " + code);
            }
            long len = conn.getContentLengthLong();
            String ar = conn.getHeaderField("Accept-Ranges");
            boolean ranges = ar != null && ar.toLowerCase().contains("bytes");
            // some CDNs omit Accept-Ranges on HEAD but still honour Range on GET
            if (!ranges && len > chunkSize) {
                ranges = true;
            }
            return new HeadInfo(len, ranges);
        } finally {
            conn.disconnect();
        }
    }

    private byte[] singleGet(String url, String token) throws IOException {
        HttpURLConnection conn = open(url, token, "GET");
        try {
            int code = conn.getResponseCode();
            // follow one redirect manually if needed (HF often 302)
            if (code == 301 || code == 302 || code == 303 || code == 307 || code == 308) {
                String loc = conn.getHeaderField("Location");
                conn.disconnect();
                if (loc == null) {
                    throw new IOException("Redirect without Location: " + url);
                }
                // Resolve relative Location against original URL
                String resolved = loc;
                if (!loc.startsWith("http://") && !loc.startsWith("https://")) {
                    URI base = URI.create(url);
                    if (loc.startsWith("/")) {
                        resolved = base.getScheme() + "://" + base.getHost() + loc;
                    } else {
                        resolved = base.resolve(loc).toString();
                    }
                }
                return singleGet(resolved, token);
            }
            if (code >= 400) {
                throw new IOException("GET " + url + " -> HTTP " + code);
            }
            try (InputStream in = conn.getInputStream();
                 ByteArrayOutputStream bos = new ByteArrayOutputStream()) {
                in.transferTo(bos);
                byte[] body = bos.toByteArray();
                bytesDownloaded.add(body.length);
                return body;
            }
        } finally {
            conn.disconnect();
        }
    }

    private byte[] multiGet(String url, String token, long contentLength) throws IOException {
        int parts = (int) ((contentLength + chunkSize - 1) / chunkSize);
        parts = Math.min(parts, maxWorkers * 4);
        List<long[]> ranges = new ArrayList<>(parts);
        for (int i = 0; i < parts; i++) {
            long start = (long) i * chunkSize;
            long end = Math.min(contentLength - 1, start + chunkSize - 1);
            ranges.add(new long[]{start, end});
        }
        byte[] out = new byte[(int) contentLength];
        ExecutorService pool = Executors.newFixedThreadPool(Math.min(maxWorkers, parts));
        try {
            List<Future<Void>> futures = new ArrayList<>(parts);
            for (long[] r : ranges) {
                final long start = r[0];
                final long end = r[1];
                futures.add(pool.submit((Callable<Void>) () -> {
                    byte[] chunk = rangedGet(url, token, start, end);
                    System.arraycopy(chunk, 0, out, (int) start, chunk.length);
                    bytesDownloaded.add(chunk.length);
                    return null;
                }));
            }
            for (Future<Void> f : futures) {
                try {
                    f.get();
                } catch (Exception e) {
                    throw new IOException("Parallel download failed for " + url, e);
                }
            }
        } finally {
            pool.shutdownNow();
        }
        return out;
    }

    private byte[] rangedGet(String url, String token, long start, long end) throws IOException {
        HttpURLConnection conn = open(url, token, "GET");
        conn.setRequestProperty("Range", "bytes=" + start + "-" + end);
        try {
            int code = conn.getResponseCode();
            if (code == 301 || code == 302 || code == 307 || code == 308) {
                String loc = conn.getHeaderField("Location");
                conn.disconnect();
                String resolved = loc;
                if (!loc.startsWith("http://") && !loc.startsWith("https://")) {
                    URI base = URI.create(url);
                    if (loc.startsWith("/")) {
                        resolved = base.getScheme() + "://" + base.getHost() + loc;
                    } else {
                        resolved = base.resolve(loc).toString();
                    }
                }
                return rangedGet(resolved, token, start, end);
            }
            if (code != 206 && code != 200) {
                throw new IOException("Range GET " + url + " [" + start + "-" + end + "] -> HTTP " + code);
            }
            try (InputStream in = conn.getInputStream();
                 ByteArrayOutputStream bos = new ByteArrayOutputStream((int) (end - start + 1))) {
                in.transferTo(bos);
                return bos.toByteArray();
            }
        } finally {
            conn.disconnect();
        }
    }

    private HttpURLConnection open(String url, String token, String method) throws IOException {
        HttpURLConnection conn = (HttpURLConnection) URI.create(url).toURL().openConnection();
        conn.setRequestMethod(method);
        conn.setConnectTimeout(connectTimeoutMs);
        conn.setReadTimeout(readTimeoutMs);
        conn.setInstanceFollowRedirects(false);
        conn.setRequestProperty("User-Agent", "javacpp-pytorch-hf-transfer/1.0");
        if (token != null && !token.isBlank()) {
            conn.setRequestProperty("Authorization", "Bearer " + token);
        }
        return conn;
    }

    /** Copy stream utility (public for callers stitching custom pipelines). */
    public static long copy(InputStream in, OutputStream out, byte[] buf) throws IOException {
        long n = 0;
        int r;
        while ((r = in.read(buf)) >= 0) {
            out.write(buf, 0, r);
            n += r;
        }
        return n;
    }

    public static final class Builder {
        private int maxWorkers = DEFAULT_WORKERS;
        private int chunkSize = DEFAULT_CHUNK;
        private int connectTimeoutMs = DEFAULT_CONNECT_TIMEOUT_MS;
        private int readTimeoutMs = DEFAULT_READ_TIMEOUT_MS;
        private boolean forceSingle;

        public Builder maxWorkers(int maxWorkers) {
            this.maxWorkers = maxWorkers;
            return this;
        }

        public Builder chunkSize(int chunkSize) {
            this.chunkSize = chunkSize;
            return this;
        }

        public Builder connectTimeoutMs(int connectTimeoutMs) {
            this.connectTimeoutMs = connectTimeoutMs;
            return this;
        }

        public Builder readTimeoutMs(int readTimeoutMs) {
            this.readTimeoutMs = readTimeoutMs;
            return this;
        }

        /** Disable multi-connection even when server advertises ranges. */
        public Builder forceSingle(boolean forceSingle) {
            this.forceSingle = forceSingle;
            return this;
        }

        public HfTransfer build() {
            return new HfTransfer(this);
        }
    }
}
