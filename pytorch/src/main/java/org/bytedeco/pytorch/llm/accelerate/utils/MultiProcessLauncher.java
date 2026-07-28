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
package org.bytedeco.pytorch.llm.accelerate.utils;
import org.bytedeco.pytorch.distributed.*;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.TimeUnit;
import java.util.stream.Collectors;

/**
 * HuggingFace {@code accelerate.launch}-style multi-process JVM launcher.
 *
 * <p>Spawns {@code worldSize} child JVMs sharing classpath and a FileStore path.
 * Each child receives {@code RANK}, {@code LOCAL_RANK}, {@code WORLD_SIZE},
 * {@code MASTER_ADDR}, {@code MASTER_PORT}, and {@code ACCELERATE_FILE_STORE}.
 *
 * <pre>{@code
 * LaunchResult r = MultiProcessLauncher.builder()
 *     .worldSize(2)
 *     .mainClass(MyWorker.class)
 *     .args("--smoke")
 *     .launch();
 * if (!r.ok()) throw new IllegalStateException(r.summary());
 * }</pre>
 */
public final class MultiProcessLauncher {

    public static final String ENV_RANK = "RANK";
    public static final String ENV_LOCAL_RANK = "LOCAL_RANK";
    public static final String ENV_WORLD_SIZE = "WORLD_SIZE";
    public static final String ENV_MASTER_ADDR = "MASTER_ADDR";
    public static final String ENV_MASTER_PORT = "MASTER_PORT";
    public static final String ENV_FILE_STORE = "ACCELERATE_FILE_STORE";

    private MultiProcessLauncher() {}

    public static Builder builder() {
        return new Builder();
    }

    public static LaunchResult launch(int worldSize, Class<?> mainClass, String... args)
            throws IOException, InterruptedException {
        return builder().worldSize(worldSize).mainClass(mainClass).args(args).launch();
    }

    public static final class LaunchResult {
        public final int worldSize;
        public final List<Integer> exitCodes;
        public final Map<Integer, String> stdout;
        public final Map<Integer, String> stderr;
        public final long elapsedMs;
        public final Path fileStorePath;

        LaunchResult(int worldSize, List<Integer> exitCodes,
                     Map<Integer, String> stdout, Map<Integer, String> stderr,
                     long elapsedMs, Path fileStorePath) {
            this.worldSize = worldSize;
            this.exitCodes = List.copyOf(exitCodes);
            this.stdout = Map.copyOf(stdout);
            this.stderr = Map.copyOf(stderr);
            this.elapsedMs = elapsedMs;
            this.fileStorePath = fileStorePath;
        }

        public boolean ok() {
            for (Integer c : exitCodes) {
                if (c == null || c != 0) return false;
            }
            return exitCodes.size() == worldSize;
        }

        public String summary() {
            StringBuilder sb = new StringBuilder();
            sb.append("LaunchResult{worldSize=").append(worldSize)
                    .append(", ok=").append(ok())
                    .append(", elapsedMs=").append(elapsedMs)
                    .append(", exits=").append(exitCodes).append('}');
            if (!ok()) {
                for (int r = 0; r < worldSize; r++) {
                    if (exitCodes.size() > r && exitCodes.get(r) != null && exitCodes.get(r) != 0) {
                        sb.append("\n--- rank ").append(r).append(" stderr ---\n")
                                .append(stderr.getOrDefault(r, ""));
                    }
                }
            }
            return sb.toString();
        }
    }

    public static final class Builder {
        private int worldSize = 2;
        private Class<?> mainClass;
        private String mainClassName;
        private final List<String> args = new ArrayList<>();
        private final Map<String, String> extraEnv = new LinkedHashMap<>();
        private final List<String> jvmArgs = new ArrayList<>();
        private String masterAddr = "127.0.0.1";
        private int masterPort = 29_511;
        private Path fileStorePath;
        private long timeoutMs = 180_000;
        private boolean inheritIo;
        private File workingDirectory;

        public Builder worldSize(int worldSize) {
            if (worldSize < 1) throw new IllegalArgumentException("worldSize must be >= 1");
            this.worldSize = worldSize;
            return this;
        }

        public Builder mainClass(Class<?> mainClass) {
            this.mainClass = mainClass;
            this.mainClassName = mainClass == null ? null : mainClass.getName();
            return this;
        }

        public Builder mainClassName(String name) {
            this.mainClassName = name;
            return this;
        }

        public Builder args(String... a) {
            if (a != null) {
                for (String s : a) if (s != null) args.add(s);
            }
            return this;
        }

        public Builder arg(String a) {
            if (a != null) args.add(a);
            return this;
        }

        public Builder env(String k, String v) {
            extraEnv.put(k, v);
            return this;
        }

        public Builder jvmArg(String a) {
            if (a != null) jvmArgs.add(a);
            return this;
        }

        public Builder masterAddr(String addr) {
            this.masterAddr = addr;
            return this;
        }

        public Builder masterPort(int port) {
            this.masterPort = port;
            return this;
        }

        public Builder fileStorePath(Path path) {
            this.fileStorePath = path;
            return this;
        }

        public Builder timeoutMs(long timeoutMs) {
            this.timeoutMs = timeoutMs;
            return this;
        }

        public Builder inheritIo(boolean inheritIo) {
            this.inheritIo = inheritIo;
            return this;
        }

        public Builder workingDirectory(File dir) {
            this.workingDirectory = dir;
            return this;
        }

        public LaunchResult launch() throws IOException, InterruptedException {
            Objects.requireNonNull(mainClassName, "mainClass");
            String javaHome = System.getProperty("java.home");
            String javaBin = javaHome + File.separator + "bin" + File.separator + "java";
            String cp = System.getProperty("java.class.path", "");
            Path store = fileStorePath;
            if (store == null) {
                store = Files.createTempDirectory("accelerate_filestore_");
            } else {
                Files.createDirectories(store);
            }
            // FileStore path used by DistributedStore defaults; point children at a unique dir
            // via env so benchmarks can also pass it explicitly.
            long t0 = System.currentTimeMillis();
            List<Process> procs = new ArrayList<>(worldSize);
            List<StringBuilder> outs = new ArrayList<>(worldSize);
            List<StringBuilder> errs = new ArrayList<>(worldSize);
            for (int rank = 0; rank < worldSize; rank++) {
                List<String> cmd = new ArrayList<>();
                cmd.add(javaBin);
                cmd.add("--add-opens=java.base/java.nio=ALL-UNNAMED");
                cmd.add("--enable-native-access=ALL-UNNAMED");
                // Inherit natives path so child JVMs can load libjnitorch / openblas
                String libPath = System.getProperty("java.library.path");
                if (libPath != null && !libPath.isEmpty()) {
                    cmd.add("-Djava.library.path=" + libPath);
                }
                cmd.addAll(jvmArgs);
                if (!cp.isEmpty()) {
                    cmd.add("-cp");
                    cmd.add(cp);
                }
                cmd.add(mainClassName);
                cmd.addAll(args);

                ProcessBuilder pb = new ProcessBuilder(cmd);
                if (workingDirectory != null) pb.directory(workingDirectory);
                Map<String, String> env = pb.environment();
                env.put(ENV_RANK, String.valueOf(rank));
                env.put(ENV_LOCAL_RANK, String.valueOf(rank));
                env.put(ENV_WORLD_SIZE, String.valueOf(worldSize));
                env.put(ENV_MASTER_ADDR, masterAddr);
                env.put(ENV_MASTER_PORT, String.valueOf(masterPort));
                env.put(ENV_FILE_STORE, store.toAbsolutePath().toString());
                env.putAll(extraEnv);
                if (inheritIo) {
                    pb.inheritIO();
                } else {
                    pb.redirectErrorStream(false);
                }
                Process p = pb.start();
                procs.add(p);
                outs.add(new StringBuilder());
                errs.add(new StringBuilder());
                if (!inheritIo) {
                    startPump(p, true, outs.get(rank));
                    startPump(p, false, errs.get(rank));
                }
            }

            List<Integer> codes = new ArrayList<>(worldSize);
            boolean timedOut = false;
            for (int i = 0; i < procs.size(); i++) {
                Process p = procs.get(i);
                boolean finished = p.waitFor(timeoutMs, TimeUnit.MILLISECONDS);
                if (!finished) {
                    timedOut = true;
                    p.destroyForcibly();
                    codes.add(-9);
                } else {
                    codes.add(p.exitValue());
                }
            }
            long elapsed = System.currentTimeMillis() - t0;
            Map<Integer, String> outMap = new HashMap<>();
            Map<Integer, String> errMap = new HashMap<>();
            for (int i = 0; i < worldSize; i++) {
                outMap.put(i, outs.get(i).toString());
                errMap.put(i, errs.get(i).toString());
            }
            if (timedOut) {
                errMap.put(-1, "launcher timeout after " + timeoutMs + "ms");
            }
            return new LaunchResult(worldSize, codes, outMap, errMap, elapsed, store);
        }

        private static void startPump(Process p, boolean stdout, StringBuilder sink) {
            Thread t = new Thread(() -> {
                try (BufferedReader br = new BufferedReader(new InputStreamReader(
                        stdout ? p.getInputStream() : p.getErrorStream(), StandardCharsets.UTF_8))) {
                    String line;
                    while ((line = br.readLine()) != null) {
                        synchronized (sink) {
                            sink.append(line).append('\n');
                        }
                    }
                } catch (IOException ignored) {
                }
            }, "mpl-pump-" + (stdout ? "out" : "err"));
            t.setDaemon(true);
            t.start();
        }
    }

    /**
     * Parse rank from env.
     * <p>Default is {@code 0} for compatibility with launched workers. To detect
     * whether this JVM was launched by {@link #launch}, use {@link #isLaunched()}
     * (checks that {@code RANK} env is actually present) — do <b>not</b> use
     * {@code envRank() < 0} as a single-process signal; that never triggers because
     * the default is 0, which previously caused single-process benchmarks to enter
     * multi-process Gloo rendezvous and hang for minutes.
     */
    public static int envRank() {
        return parseIntEnv(ENV_RANK, 0);
    }

    /**
     * True iff {@code RANK} environment variable is set (child of
     * {@link #launch} / torchrun-style launcher). Unset → single-process smoke.
     */
    public static boolean isLaunched() {
        String v = System.getenv(ENV_RANK);
        return v != null && !v.isEmpty();
    }

    public static int envWorldSize() {
        return parseIntEnv(ENV_WORLD_SIZE, 1);
    }

    public static int envLocalRank() {
        return parseIntEnv(ENV_LOCAL_RANK, envRank());
    }

    public static String envMasterAddr() {
        String v = System.getenv(ENV_MASTER_ADDR);
        return v == null || v.isEmpty() ? "127.0.0.1" : v;
    }

    public static int envMasterPort() {
        return parseIntEnv(ENV_MASTER_PORT, 29_500);
    }

    public static String envFileStore() {
        return System.getenv(ENV_FILE_STORE);
    }

    private static int parseIntEnv(String key, int def) {
        String v = System.getenv(key);
        if (v == null || v.isEmpty()) return def;
        try {
            return Integer.parseInt(v.trim());
        } catch (NumberFormatException e) {
            return def;
        }
    }

    /** Join child outputs for debugging. */
    public static String joinOutputs(LaunchResult r) {
        return r.stdout.entrySet().stream()
                .sorted(Map.Entry.comparingByKey())
                .map(e -> "=== rank " + e.getKey() + " ===\n" + e.getValue())
                .collect(Collectors.joining("\n"));
    }
}
