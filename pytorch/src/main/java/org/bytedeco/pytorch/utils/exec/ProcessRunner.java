/*
 * Zero-dep process / CLI runner used by Docker and Kubernetes adapters.
 *
 * Mirrors the timeout + stdout/stderr capture pattern from FFmpeg / MultiProcessLauncher
 * without pulling in any external process libraries.
 */
package org.bytedeco.pytorch.utils.exec;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.InputStream;
import java.nio.charset.Charset;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.TimeUnit;
import java.util.function.Function;

/**
 * Thin {@link ProcessBuilder} wrapper: timeout, env merge, cwd, combined or split streams.
 *
 * <pre>{@code
 * ProcessRunner.CommandResult r = ProcessRunner.run(
 *     List.of("docker", "version", "--format", "{{.Server.Version}}"),
 *     ProcessRunner.Options.defaults().timeout(Duration.ofSeconds(15)));
 * if (r.ok()) System.out.println(r.stdout().trim());
 * }</pre>
 */
public final class ProcessRunner {

    private ProcessRunner() {}

    /** Immutable run result. */
    public static final class CommandResult {
        private final int exitCode;
        private final String stdout;
        private final String stderr;
        private final long durationMs;
        private final List<String> command;

        public CommandResult(int exitCode, String stdout, String stderr, long durationMs, List<String> command) {
            this.exitCode = exitCode;
            this.stdout = stdout == null ? "" : stdout;
            this.stderr = stderr == null ? "" : stderr;
            this.durationMs = durationMs;
            this.command = command == null
                    ? List.of()
                    : Collections.unmodifiableList(new ArrayList<>(command));
        }

        public int exitCode() { return exitCode; }
        public String stdout() { return stdout; }
        public String stderr() { return stderr; }
        public long durationMs() { return durationMs; }
        public List<String> command() { return command; }

        public boolean ok() { return exitCode == 0; }

        /** Combined stdout+stderr for error messages. */
        public String output() {
            if (stderr.isEmpty()) return stdout;
            if (stdout.isEmpty()) return stderr;
            return stdout + (stdout.endsWith("\n") ? "" : "\n") + stderr;
        }

        public String requireOk() {
            return requireOk(msg -> new IllegalStateException(msg));
        }

        public String requireOk(Function<String, RuntimeException> exceptionFactory) {
            if (ok()) return stdout;
            String cmd = String.join(" ", command);
            String body = truncate(output(), 1200);
            throw exceptionFactory.apply(
                    "command failed exit=" + exitCode + " durationMs=" + durationMs
                            + " cmd=[" + cmd + "] output:\n" + body);
        }

        @Override
        public String toString() {
            return "CommandResult{exit=" + exitCode + ", ms=" + durationMs
                    + ", cmd=" + command + "}";
        }
    }

    /** Run options. */
    public static final class Options {
        public final Duration timeout;
        public final Path workingDirectory;
        public final Map<String, String> extraEnv;
        public final boolean clearEnv;
        public final boolean redirectErrorStream;
        public final boolean inheritIo;
        public final Charset charset;
        /** If non-null, stdin is fed this string (UTF-8 by default charset). */
        public final String stdin;

        private Options(Builder b) {
            this.timeout = b.timeout == null ? Duration.ofSeconds(120) : b.timeout;
            this.workingDirectory = b.workingDirectory;
            this.extraEnv = b.extraEnv == null
                    ? Map.of()
                    : Collections.unmodifiableMap(new LinkedHashMap<>(b.extraEnv));
            this.clearEnv = b.clearEnv;
            this.redirectErrorStream = b.redirectErrorStream;
            this.inheritIo = b.inheritIo;
            this.charset = b.charset == null ? StandardCharsets.UTF_8 : b.charset;
            this.stdin = b.stdin;
        }

        public static Options defaults() {
            return new Builder().build();
        }

        public static Builder builder() {
            return new Builder();
        }

        public Builder toBuilder() {
            Builder b = new Builder();
            b.timeout = timeout;
            b.workingDirectory = workingDirectory;
            b.extraEnv = new LinkedHashMap<>(extraEnv);
            b.clearEnv = clearEnv;
            b.redirectErrorStream = redirectErrorStream;
            b.inheritIo = inheritIo;
            b.charset = charset;
            b.stdin = stdin;
            return b;
        }

        public static final class Builder {
            private Duration timeout = Duration.ofSeconds(120);
            private Path workingDirectory;
            private Map<String, String> extraEnv = new LinkedHashMap<>();
            private boolean clearEnv;
            private boolean redirectErrorStream = true;
            private boolean inheritIo;
            private Charset charset = StandardCharsets.UTF_8;
            private String stdin;

            public Builder timeout(Duration d) { this.timeout = d; return this; }
            public Builder timeoutSeconds(long s) { this.timeout = Duration.ofSeconds(s); return this; }
            public Builder workingDirectory(Path p) { this.workingDirectory = p; return this; }
            public Builder workingDirectory(File f) {
                this.workingDirectory = f == null ? null : f.toPath();
                return this;
            }
            public Builder env(String k, String v) {
                if (k != null && v != null) extraEnv.put(k, v);
                return this;
            }
            public Builder env(Map<String, String> m) {
                if (m != null) extraEnv.putAll(m);
                return this;
            }
            public Builder clearEnv(boolean v) { this.clearEnv = v; return this; }
            public Builder redirectErrorStream(boolean v) { this.redirectErrorStream = v; return this; }
            public Builder inheritIo(boolean v) { this.inheritIo = v; return this; }
            public Builder charset(Charset c) { this.charset = c; return this; }
            public Builder stdin(String s) { this.stdin = s; return this; }
            public Options build() { return new Options(this); }
        }
    }

    /**
     * Run {@code command} and wait up to {@code opts.timeout}.
     * On timeout the process is destroyed forcibly and exit code is set to {@code -9}.
     */
    public static CommandResult run(List<String> command, Options opts) {
        Objects.requireNonNull(command, "command");
        if (command.isEmpty()) {
            throw new IllegalArgumentException("command is empty");
        }
        Options o = opts == null ? Options.defaults() : opts;
        List<String> cmd = new ArrayList<>(command);
        long t0 = System.currentTimeMillis();
        try {
            ProcessBuilder pb = new ProcessBuilder(cmd);
            if (o.workingDirectory != null) {
                File dir = o.workingDirectory.toFile();
                if (!dir.isDirectory()) {
                    throw new IllegalArgumentException("workingDirectory not a directory: " + dir);
                }
                pb.directory(dir);
            }
            Map<String, String> env = pb.environment();
            if (o.clearEnv) env.clear();
            if (!o.extraEnv.isEmpty()) env.putAll(o.extraEnv);

            if (o.inheritIo) {
                pb.inheritIO();
            } else if (o.redirectErrorStream) {
                pb.redirectErrorStream(true);
            }

            Process p = pb.start();

            if (o.stdin != null && !o.inheritIo) {
                try (var out = p.getOutputStream()) {
                    out.write(o.stdin.getBytes(o.charset));
                    out.flush();
                }
            } else if (!o.inheritIo) {
                try { p.getOutputStream().close(); } catch (Exception ignored) {}
            }

            ByteArrayOutputStream bout = new ByteArrayOutputStream();
            ByteArrayOutputStream berr = new ByteArrayOutputStream();
            Thread outPump = null;
            Thread errPump = null;
            if (!o.inheritIo) {
                outPump = pump(p.getInputStream(), bout, "proc-stdout");
                if (!o.redirectErrorStream) {
                    errPump = pump(p.getErrorStream(), berr, "proc-stderr");
                }
            }

            long timeoutMs = o.timeout.toMillis();
            if (timeoutMs <= 0) timeoutMs = TimeUnit.DAYS.toMillis(1);
            boolean finished = p.waitFor(timeoutMs, TimeUnit.MILLISECONDS);
            int code;
            if (!finished) {
                p.destroyForcibly();
                try { p.waitFor(5, TimeUnit.SECONDS); } catch (InterruptedException ie) {
                    Thread.currentThread().interrupt();
                }
                code = -9;
            } else {
                code = p.exitValue();
            }

            if (outPump != null) joinQuiet(outPump, 2000);
            if (errPump != null) joinQuiet(errPump, 2000);

            long dt = System.currentTimeMillis() - t0;
            String stdout = bout.toString(o.charset);
            String stderr = berr.toString(o.charset);
            if (!finished) {
                stderr = (stderr.isEmpty() ? "" : stderr + "\n") + "[ProcessRunner] timed out after "
                        + timeoutMs + "ms, process destroyed";
            }
            return new CommandResult(code, stdout, stderr, dt, cmd);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            long dt = System.currentTimeMillis() - t0;
            return new CommandResult(-1, "", "interrupted: " + e.getMessage(), dt, cmd);
        } catch (Exception e) {
            long dt = System.currentTimeMillis() - t0;
            return new CommandResult(-1, "", e.getClass().getSimpleName() + ": " + e.getMessage(), dt, cmd);
        }
    }

    public static CommandResult run(String... command) {
        List<String> cmd = new ArrayList<>();
        if (command != null) {
            for (String c : command) {
                if (c != null) cmd.add(c);
            }
        }
        return run(cmd, Options.defaults());
    }

    public static CommandResult run(List<String> command) {
        return run(command, Options.defaults());
    }

    /** True if {@code binary} resolves on PATH (or is an absolute existing file). */
    public static boolean onPath(String binary) {
        if (binary == null || binary.isBlank()) return false;
        Path p = Path.of(binary);
        if (p.isAbsolute() || binary.contains("/") || binary.contains("\\")) {
            return Files.isExecutable(p) || Files.isRegularFile(p);
        }
        String path = System.getenv("PATH");
        if (path == null || path.isEmpty()) path = System.getenv("Path");
        if (path == null) return false;
        String[] parts = path.split(File.pathSeparator);
        boolean windows = isWindows();
        for (String part : parts) {
            if (part == null || part.isEmpty()) continue;
            Path base = Path.of(part);
            Path cand = base.resolve(binary);
            if (Files.isExecutable(cand) || Files.isRegularFile(cand)) return true;
            if (windows) {
                for (String ext : new String[]{".exe", ".bat", ".cmd"}) {
                    Path c2 = base.resolve(binary + ext);
                    if (Files.isRegularFile(c2)) return true;
                }
            }
        }
        return false;
    }

    /**
     * Resolve binary: absolute path as-is; otherwise first hit on PATH; else {@code null}.
     */
    public static String which(String binary) {
        if (binary == null || binary.isBlank()) return null;
        Path p = Path.of(binary);
        if (p.isAbsolute() || binary.contains("/") || binary.contains("\\")) {
            if (Files.isRegularFile(p)) return p.toAbsolutePath().toString();
            return null;
        }
        String path = System.getenv("PATH");
        if (path == null || path.isEmpty()) path = System.getenv("Path");
        if (path == null) return null;
        boolean windows = isWindows();
        for (String part : path.split(File.pathSeparator)) {
            if (part == null || part.isEmpty()) continue;
            Path base = Path.of(part);
            Path cand = base.resolve(binary);
            if (Files.isRegularFile(cand)) return cand.toAbsolutePath().toString();
            if (windows) {
                for (String ext : new String[]{".exe", ".bat", ".cmd"}) {
                    Path c2 = base.resolve(binary + ext);
                    if (Files.isRegularFile(c2)) return c2.toAbsolutePath().toString();
                }
            }
        }
        return null;
    }

    private static Thread pump(InputStream in, ByteArrayOutputStream out, String name) {
        Thread t = new Thread(() -> {
            try {
                in.transferTo(out);
            } catch (Exception ignored) {
            }
        }, name);
        t.setDaemon(true);
        t.start();
        return t;
    }

    private static void joinQuiet(Thread t, long ms) {
        try {
            t.join(ms);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
    }

    private static boolean isWindows() {
        String os = System.getProperty("os.name", "");
        return os.toLowerCase(Locale.ROOT).contains("win");
    }

    private static String truncate(String s, int max) {
        if (s == null) return "";
        s = s.trim();
        if (s.length() <= max) return s;
        return s.substring(0, max) + "…";
    }
}
