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

package org.bytedeco.pytorch.llm.llamacpp;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.net.ServerSocket;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicReference;

/**
 * Spawns and supervises an official {@code llama-server} process.
 */
public final class LlamaProcessManager implements AutoCloseable {

    private final LlamaRuntimeConfig config;
    private final AtomicReference<Process> process = new AtomicReference<>();
    private volatile int boundPort;
    private volatile Path resolvedBin;

    public LlamaProcessManager(LlamaRuntimeConfig config) {
        this.config = Objects.requireNonNull(config);
    }

    public int boundPort() { return boundPort; }
    public Path resolvedBin() { return resolvedBin; }
    public boolean isAlive() {
        Process p = process.get();
        return p != null && p.isAlive();
    }

    public static Path findLlamaServer(LlamaRuntimeConfig config) {
        if (config.llamaServerBin().isPresent()) {
            Path p = config.llamaServerBin().get();
            if (Files.isExecutable(p)) return p;
        }
        String env = System.getenv("LLAMA_SERVER_BIN");
        if (env != null && !env.isBlank()) {
            Path p = Path.of(env);
            if (Files.isExecutable(p)) return p;
        }
        for (String c : List.of(
                "llama-server",
                "/usr/local/bin/llama-server",
                "/usr/bin/llama-server",
                System.getProperty("user.home", "") + "/.local/bin/llama-server")) {
            if (c == null || c.isBlank()) continue;
            Path p = Path.of(c);
            if (Files.isExecutable(p)) return p;
            // PATH lookup
            if (!c.contains("/") && !c.contains("\\")) {
                Path which = which(c);
                if (which != null) return which;
            }
        }
        return null;
    }

    public static Path which(String name) {
        String path = System.getenv("PATH");
        if (path == null) return null;
        for (String dir : path.split(java.io.File.pathSeparator)) {
            Path p = Path.of(dir, name);
            if (Files.isExecutable(p)) return p;
        }
        return null;
    }

    public int start() throws Exception {
        Path bin = findLlamaServer(config);
        if (bin == null) {
            throw new IllegalStateException(
                    "llama-server binary not found. Set LlamaRuntimeConfig.llamaServerBin or LLAMA_SERVER_BIN.");
        }
        this.resolvedBin = bin;
        int port = config.serverPort();
        if (port <= 0) {
            try (ServerSocket ss = new ServerSocket(0)) {
                port = ss.getLocalPort();
            }
        }
        this.boundPort = port;

        List<String> cmd = new ArrayList<>();
        cmd.add(bin.toAbsolutePath().toString());
        cmd.addAll(config.toServerArgList(port));

        ProcessBuilder pb = new ProcessBuilder(cmd);
        pb.redirectErrorStream(true);
        if (config.verbose()) {
            System.err.println("[llama-server] " + String.join(" ", cmd));
        }
        Process proc = pb.start();
        process.set(proc);

        // drain stdout in background to avoid buffer deadlock
        Thread t = new Thread(() -> {
            try (BufferedReader br = new BufferedReader(new InputStreamReader(proc.getInputStream()))) {
                String line;
                while ((line = br.readLine()) != null) {
                    if (config.verbose()) System.err.println("[llama-server] " + line);
                }
            } catch (IOException ignored) {}
        }, "llama-server-stdout");
        t.setDaemon(true);
        t.start();

        LlamaServerClient client = new LlamaServerClient(config.serverHost(), port);
        long deadline = System.currentTimeMillis() + config.serverStartTimeoutMs();
        while (System.currentTimeMillis() < deadline) {
            if (!proc.isAlive()) {
                throw new IllegalStateException("llama-server exited early, code=" + proc.exitValue());
            }
            if (client.healthy()) return port;
            Thread.sleep(200);
        }
        stop();
        throw new IllegalStateException("llama-server health check timed out on port " + port);
    }

    public void stop() {
        Process p = process.getAndSet(null);
        if (p == null) return;
        p.destroy();
        try {
            if (!p.waitFor(5, TimeUnit.SECONDS)) {
                p.destroyForcibly();
            }
        } catch (InterruptedException e) {
            p.destroyForcibly();
            Thread.currentThread().interrupt();
        }
    }

    @Override
    public void close() { stop(); }
}
