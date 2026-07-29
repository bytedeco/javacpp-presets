package org.bytedeco.pytorch.utils.docker;

import org.bytedeco.pytorch.utils.exec.ProcessRunner;
import org.bytedeco.pytorch.utils.json.Json;

import java.io.IOException;
import java.nio.file.Path;
import java.time.Duration;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Docker Compose adapter ({@code docker compose} plugin, fallback {@code docker-compose}).
 *
 * <pre>{@code
 * DockerCompose compose = DockerCompose.connect(opts);
 * compose.up(Path.of("docker-compose.yml"), true, true);
 * compose.ps(file);
 * compose.down(file, true, false);
 * }</pre>
 */
public final class DockerCompose {

    public enum Flavor { DOCKER_COMPOSE_PLUGIN, DOCKER_COMPOSE_STANDALONE, NONE }

    private final DockerOptions options;
    private final Flavor flavor;
    private final String composeBinary; // for standalone

    public DockerCompose(DockerOptions options) {
        this.options = options == null ? DockerOptions.defaults() : options;
        Detected d = detect(this.options);
        this.flavor = d.flavor;
        this.composeBinary = d.standaloneBin;
    }

    public static DockerCompose connect() {
        return new DockerCompose(DockerOptions.defaults());
    }

    public static DockerCompose connect(DockerOptions options) {
        return new DockerCompose(options);
    }

    public Flavor flavor() { return flavor; }
    public boolean available() { return flavor != Flavor.NONE; }
    public DockerOptions options() { return options; }

    public ProcessRunner.CommandResult raw(List<String> composeArgs, Path projectDir, Path composeFile) {
        return raw(composeArgs, projectDir, composeFile, options.timeout);
    }

    public ProcessRunner.CommandResult raw(
            List<String> composeArgs, Path projectDir, Path composeFile, Duration timeout) {
        if (flavor == Flavor.NONE) {
            throw new DockerException("docker compose not available on PATH", -1, "compose");
        }
        List<String> cmd = new ArrayList<>();
        if (flavor == Flavor.DOCKER_COMPOSE_PLUGIN) {
            cmd.add(options.dockerBin);
            cmd.add("compose");
        } else {
            cmd.add(composeBinary != null ? composeBinary : "docker-compose");
        }
        if (composeFile != null) {
            cmd.add("-f");
            cmd.add(composeFile.toAbsolutePath().toString());
        }
        if (composeArgs != null) cmd.addAll(composeArgs);

        ProcessRunner.Options.Builder ob = ProcessRunner.Options.builder()
                .timeout(timeout == null ? options.timeout : timeout)
                .redirectErrorStream(false);
        if (projectDir != null) ob.workingDirectory(projectDir);
        if (options.host != null && !options.host.isBlank()) ob.env("DOCKER_HOST", options.host);
        if (options.tlsVerify) ob.env("DOCKER_TLS_VERIFY", "1");
        if (options.certPath != null) ob.env("DOCKER_CERT_PATH", options.certPath);
        if (!options.extraEnv.isEmpty()) ob.env(options.extraEnv);
        return ProcessRunner.run(cmd, ob.build());
    }

    public String requireOk(String op, List<String> args, Path projectDir, Path composeFile) {
        ProcessRunner.CommandResult r = raw(args, projectDir, composeFile);
        if (!r.ok()) throw DockerException.ofExit("compose." + op, r.exitCode(), r.output());
        return r.stdout();
    }

    /** Validate and render the compose file (resolved config). */
    public String config(Path composeFile, Path projectDir) {
        return requireOk("config", List.of("config"), projectDir, composeFile);
    }

    public Map<String, Object> configJson(Path composeFile, Path projectDir) {
        String out = requireOk("config", List.of("config", "--format", "json"), projectDir, composeFile);
        try {
            Object v = Json.decode(out.trim());
            if (v instanceof Map<?, ?> m) {
                @SuppressWarnings("unchecked")
                Map<String, Object> map = (Map<String, Object>) m;
                return map;
            }
            throw new DockerException("compose config --format json did not return object");
        } catch (IOException e) {
            throw new DockerException("compose config json: " + e.getMessage(), e);
        }
    }

    /**
     * {@code docker compose up}.
     *
     * @param detach     {@code -d}
     * @param build      {@code --build}
     * @param forceRecreate {@code --force-recreate}
     */
    public void up(Path composeFile, Path projectDir, boolean detach, boolean build, boolean forceRecreate) {
        List<String> args = new ArrayList<>();
        args.add("up");
        if (detach) args.add("-d");
        if (build) args.add("--build");
        if (forceRecreate) args.add("--force-recreate");
        requireOk("up", args, projectDir, composeFile);
    }

    public void up(Path composeFile, boolean detach, boolean build) {
        Path dir = composeFile == null ? null : composeFile.toAbsolutePath().getParent();
        up(composeFile, dir, detach, build, false);
    }

    public void down(Path composeFile, Path projectDir, boolean removeVolumes, boolean removeImages) {
        List<String> args = new ArrayList<>();
        args.add("down");
        if (removeVolumes) args.add("-v");
        if (removeImages) {
            args.add("--rmi");
            args.add("local");
        }
        requireOk("down", args, projectDir, composeFile);
    }

    public void down(Path composeFile, boolean removeVolumes) {
        Path dir = composeFile == null ? null : composeFile.toAbsolutePath().getParent();
        down(composeFile, dir, removeVolumes, false);
    }

    public String ps(Path composeFile, Path projectDir) {
        return requireOk("ps", List.of("ps"), projectDir, composeFile);
    }

    public List<Map<String, Object>> psJson(Path composeFile, Path projectDir) {
        String out = requireOk("ps", List.of("ps", "--format", "json"), projectDir, composeFile);
        return parseJsonLinesOrArray(out);
    }

    public String logs(Path composeFile, Path projectDir, String service, Integer tail) {
        List<String> args = new ArrayList<>();
        args.add("logs");
        if (tail != null) {
            args.add("--tail");
            args.add(String.valueOf(tail));
        }
        if (service != null) args.add(service);
        ProcessRunner.CommandResult r = raw(args, projectDir, composeFile);
        if (!r.ok()) throw DockerException.ofExit("compose.logs", r.exitCode(), r.output());
        return r.output();
    }

    public void restart(Path composeFile, Path projectDir, String service) {
        List<String> args = new ArrayList<>();
        args.add("restart");
        if (service != null) args.add(service);
        requireOk("restart", args, projectDir, composeFile);
    }

    public void pull(Path composeFile, Path projectDir) {
        requireOk("pull", List.of("pull"), projectDir, composeFile);
    }

    public void scale(Path composeFile, Path projectDir, String service, int replicas) {
        Objects.requireNonNull(service, "service");
        // modern: up -d --scale svc=n
        List<String> args = new ArrayList<>();
        args.add("up");
        args.add("-d");
        args.add("--no-recreate");
        args.add("--scale");
        args.add(service + "=" + Math.max(0, replicas));
        args.add(service);
        requireOk("scale", args, projectDir, composeFile);
    }

    // ---- detect ----

    private static final class Detected {
        final Flavor flavor;
        final String standaloneBin;
        Detected(Flavor f, String b) { this.flavor = f; this.standaloneBin = b; }
    }

    private static Detected detect(DockerOptions opts) {
        if (opts.composeBin != null && !opts.composeBin.isBlank()) {
            if (ProcessRunner.onPath(opts.composeBin) || opts.composeBin.contains("/")
                    || opts.composeBin.contains("\\")) {
                return new Detected(Flavor.DOCKER_COMPOSE_STANDALONE, opts.composeBin);
            }
        }
        // try docker compose version
        if (ProcessRunner.onPath(opts.dockerBin)) {
            ProcessRunner.CommandResult r = ProcessRunner.run(
                    List.of(opts.dockerBin, "compose", "version"),
                    ProcessRunner.Options.builder().timeout(Duration.ofSeconds(10)).build());
            if (r.ok()) return new Detected(Flavor.DOCKER_COMPOSE_PLUGIN, null);
        }
        if (ProcessRunner.onPath("docker-compose")) {
            return new Detected(Flavor.DOCKER_COMPOSE_STANDALONE, "docker-compose");
        }
        return new Detected(Flavor.NONE, null);
    }

    @SuppressWarnings("unchecked")
    private static List<Map<String, Object>> parseJsonLinesOrArray(String out) {
        List<Map<String, Object>> list = new ArrayList<>();
        if (out == null || out.isBlank()) return list;
        String t = out.trim();
        try {
            if (t.startsWith("[")) {
                Object v = Json.decode(t);
                if (v instanceof List<?> arr) {
                    for (Object o : arr) {
                        if (o instanceof Map<?, ?> m) list.add((Map<String, Object>) m);
                    }
                }
                return list;
            }
            // ndjson
            for (String line : t.split("\n")) {
                line = line.trim();
                if (line.isEmpty()) continue;
                Object v = Json.decode(line);
                if (v instanceof Map<?, ?> m) list.add((Map<String, Object>) m);
            }
        } catch (IOException e) {
            throw new DockerException("compose ps json: " + e.getMessage(), e);
        }
        return list;
    }
}
