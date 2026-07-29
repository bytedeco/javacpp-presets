package org.bytedeco.pytorch.utils.docker;

import org.bytedeco.pytorch.utils.exec.ProcessRunner;
import org.bytedeco.pytorch.utils.json.Json;

import java.io.IOException;
import java.nio.file.Path;
import java.time.Duration;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Full-surface {@code docker} CLI adapter — zero SDK dependency.
 *
 * <p>Covers every top-level command group from {@code docker --help}:
 * <ul>
 *   <li><b>Common</b>: run, exec, ps, build, bake, pull, push, images, login, logout,
 *       search, version, info</li>
 *   <li><b>Management</b>: builder, container, context, image, manifest, network,
 *       plugin, system, volume (+ compose via {@link DockerCompose})</li>
 *   <li><b>Swarm</b>: swarm init/join/leave/…</li>
 *   <li><b>Container ops</b>: attach, commit, cp, create, diff, events, export,
 *       history, import, inspect, kill, load, logs, pause, port, rename, restart,
 *       rm, rmi, save, start, stats, stop, tag, top, unpause, update, wait</li>
 * </ul>
 *
 * <p>Global options from {@link DockerOptions} are injected:
 * {@code --config}, {@code --context}, {@code -H/--host}, {@code -D/--debug},
 * {@code --log-level}, {@code --tls*}, {@code --tlsverify}.
 *
 * <pre>{@code
 * DockerCli d = new DockerCli(DockerOptions.defaults());
 * d.ping();
 * String id = d.run(DockerModels.RunSpec.builder("nginx:alpine").detach(true).publish(8080, 80).build());
 * d.cmd("container", "ls").flag("-a").formatJson().runOk();
 * d.image().ls();
 * d.network().create("ml-net");
 * d.volume().create("models");
 * d.system().df();
 * }</pre>
 *
 * @see <a href="https://docs.docker.com/reference/cli/docker/">docker CLI reference</a>
 */
public final class DockerCli {

    private final DockerOptions options;

    public DockerCli(DockerOptions options) {
        this.options = options == null ? DockerOptions.defaults() : options;
    }

    public DockerOptions options() {
        return options;
    }

    public boolean available() {
        return ProcessRunner.onPath(options.dockerBin);
    }

    // =========================================================================
    // Low-level execution
    // =========================================================================

    public ProcessRunner.CommandResult raw(List<String> args) {
        return raw(args, options.timeout, null);
    }

    public ProcessRunner.CommandResult raw(List<String> args, Duration timeout) {
        return raw(args, timeout, null);
    }

    public ProcessRunner.CommandResult raw(List<String> args, Duration timeout, String stdin) {
        List<String> cmd = new ArrayList<>();
        cmd.add(options.dockerBin);
        injectGlobalFlags(cmd);
        if (args != null) cmd.addAll(args);

        ProcessRunner.Options.Builder ob = ProcessRunner.Options.builder()
                .timeout(timeout == null ? options.timeout : timeout)
                .redirectErrorStream(false);
        if (stdin != null) ob.stdin(stdin);
        // Also export classic env so nested tools (buildx/compose) see them
        if (options.host != null && !options.host.isBlank()) {
            ob.env("DOCKER_HOST", options.host);
        }
        if (options.tlsVerify) ob.env("DOCKER_TLS_VERIFY", "1");
        if (options.certPath != null) ob.env("DOCKER_CERT_PATH", options.certPath);
        if (options.context != null) ob.env("DOCKER_CONTEXT", options.context);
        if (options.configDir != null) ob.env("DOCKER_CONFIG", options.configDir);
        if (!options.extraEnv.isEmpty()) ob.env(options.extraEnv);
        return ProcessRunner.run(cmd, ob.build());
    }

    public String requireOk(String operation, List<String> args) {
        return requireOk(operation, args, options.timeout, null);
    }

    public String requireOk(String operation, List<String> args, Duration timeout, String stdin) {
        ProcessRunner.CommandResult r = raw(args, timeout, stdin);
        if (!r.ok()) {
            throw DockerException.ofExit(operation, r.exitCode(), r.output());
        }
        return r.stdout();
    }

    /** Escape hatch: build any docker invocation. */
    public Cmd cmd(String... parts) {
        return new Cmd(this).add(parts);
    }

    public Cmd cmd(List<String> parts) {
        return new Cmd(this).add(parts);
    }

    private void injectGlobalFlags(List<String> cmd) {
        if (options.configDir != null && !options.configDir.isBlank()) {
            cmd.add("--config");
            cmd.add(options.configDir);
        }
        if (options.context != null && !options.context.isBlank()) {
            cmd.add("--context");
            cmd.add(options.context);
        }
        if (options.host != null && !options.host.isBlank()) {
            cmd.add("--host");
            cmd.add(options.host);
        }
        if (options.debug) cmd.add("--debug");
        if (options.logLevel != null && !options.logLevel.isBlank()) {
            cmd.add("--log-level");
            cmd.add(options.logLevel);
        }
        if (options.tls) cmd.add("--tls");
        if (options.tlsVerify) cmd.add("--tlsverify");
        // Explicit cert files take precedence; else DOCKER_CERT_PATH is used by docker CLI via env
        if (options.tlsCaCert != null) {
            cmd.add("--tlscacert");
            cmd.add(options.tlsCaCert);
        }
        if (options.tlsCert != null) {
            cmd.add("--tlscert");
            cmd.add(options.tlsCert);
        }
        if (options.tlsKey != null) {
            cmd.add("--tlskey");
            cmd.add(options.tlsKey);
        }
    }

    // =========================================================================
    // Fluent Cmd builder
    // =========================================================================

    /**
     * Fluent argv builder. Global flags from {@link DockerOptions} are injected at run time.
     *
     * <pre>{@code
     * d.cmd("ps").flag("-a").format("{{json .}}").runOk();
     * d.cmd("build").flag("-t", "app:1").add(".").runOk();
     * }</pre>
     */
    public static final class Cmd {
        private final DockerCli cli;
        private final List<String> args = new ArrayList<>();
        private Duration timeout;
        private String stdin;

        Cmd(DockerCli cli) {
            this.cli = cli;
            this.timeout = cli.options.timeout;
        }

        public Cmd add(String... parts) {
            if (parts != null) for (String p : parts) if (p != null) args.add(p);
            return this;
        }

        public Cmd add(Collection<String> parts) {
            if (parts != null) for (String p : parts) if (p != null) args.add(p);
            return this;
        }

        public Cmd flag(String name) {
            if (name != null) args.add(name.startsWith("-") ? name : "--" + name);
            return this;
        }

        public Cmd flag(String name, Object value) {
            if (name == null || value == null) return this;
            String n = name.startsWith("-") ? name : "--" + name;
            args.add(n);
            args.add(String.valueOf(value));
            return this;
        }

        public Cmd flagEq(String name, Object value) {
            if (name == null || value == null) return this;
            String n = name.startsWith("-") ? name : "--" + name;
            args.add(n + "=" + value);
            return this;
        }

        public Cmd format(String fmt) {
            return flag("--format", fmt);
        }

        public Cmd formatJson() {
            return format("{{json .}}");
        }

        public Cmd quiet() {
            return flag("-q");
        }

        public Cmd all() {
            return flag("-a");
        }

        public Cmd force() {
            return flag("-f");
        }

        public Cmd timeout(Duration d) {
            this.timeout = d;
            return this;
        }

        public Cmd stdin(String body) {
            this.stdin = body;
            return this;
        }

        public List<String> argv() {
            return new ArrayList<>(args);
        }

        public ProcessRunner.CommandResult run() {
            return cli.raw(argv(), timeout, stdin);
        }

        public String runOk() {
            String op = args.isEmpty() ? "docker" : args.get(0);
            return cli.requireOk(op, argv(), timeout, stdin);
        }

        public Map<String, Object> runJsonObject() {
            String out = runOk();
            try {
                return Json.decodeObject(out.trim());
            } catch (IOException e) {
                throw new DockerException("json parse: " + e.getMessage(), e);
            }
        }

        public List<Map<String, Object>> runJsonLines() {
            String out = runOk();
            return parseJsonLines(out);
        }

        @Override
        public String toString() {
            return "docker " + String.join(" ", argv());
        }
    }

    // =========================================================================
    // Common: version / info / ping / login / logout / search
    // =========================================================================

    public void ping() {
        requireOk("ping", List.of("info", "-f", "{{.ServerVersion}}"));
    }

    public String version() {
        // Prefer server version; fall back to client if daemon down
        ProcessRunner.CommandResult r = raw(List.of("version", "--format", "{{.Server.Version}}"));
        if (r.ok() && r.stdout() != null && !r.stdout().isBlank()) {
            return r.stdout().trim();
        }
        return requireOk("version", List.of("version", "--format", "{{.Client.Version}}")).trim();
    }

    public String versionRaw() {
        return requireOk("version", List.of("version"));
    }

    public Map<String, Object> versionJson() {
        String out = requireOk("version", List.of("version", "--format", "{{json .}}"));
        return decodeObject(out, "version");
    }

    public Map<String, Object> info() {
        String out = requireOk("info", List.of("info", "--format", "{{json .}}"));
        return decodeObject(out, "info");
    }

    public String infoRaw() {
        return requireOk("info", List.of("info"));
    }

    public String login(String server, String username, String password) {
        Cmd c = cmd("login");
        if (username != null) c.flag("-u", username);
        if (password != null) c.flag("--password-stdin").stdin(password);
        if (server != null) c.add(server);
        return c.runOk();
    }

    public String loginInteractive(String server) {
        Cmd c = cmd("login");
        if (server != null) c.add(server);
        return c.runOk();
    }

    public String logout(String server) {
        Cmd c = cmd("logout");
        if (server != null) c.add(server);
        return c.runOk();
    }

    public String search(String term, Integer limit, boolean noTrunc) {
        Cmd c = cmd("search", term);
        if (limit != null) c.flag("--limit", limit);
        if (noTrunc) c.flag("--no-trunc");
        return c.runOk();
    }

    public List<Map<String, Object>> searchJson(String term, Integer limit) {
        Cmd c = cmd("search", term).formatJson();
        if (limit != null) c.flag("--limit", limit);
        return c.runJsonLines();
    }

    // =========================================================================
    // Common: run / exec / ps / build / bake / pull / push / images
    // =========================================================================

    /** Run container; returns container id (trimmed) when detached. */
    public String run(DockerModels.RunSpec spec) {
        Objects.requireNonNull(spec, "spec");
        String out = requireOk("run", spec.toCliArgs());
        return firstLine(out);
    }

    public String run(String image, String... command) {
        return run(DockerModels.RunSpec.builder(image).command(command).detach(true).build());
    }

    /**
     * Low-level {@code docker run} with raw extra flags after global options.
     * Prefer {@link #run(DockerModels.RunSpec)}.
     */
    public String runRaw(List<String> runArgs) {
        List<String> args = new ArrayList<>();
        args.add("run");
        if (runArgs != null) args.addAll(runArgs);
        return firstLine(requireOk("run", args));
    }

    public String exec(String idOrName, List<String> command, boolean tty) {
        return exec(idOrName, command, false, tty, null, null);
    }

    public String exec(
            String idOrName, List<String> command, boolean interactive, boolean tty,
            String user, String workdir) {
        Objects.requireNonNull(command, "command");
        Cmd c = cmd("exec");
        if (interactive) c.flag("-i");
        if (tty) c.flag("-t");
        if (user != null) c.flag("-u", user);
        if (workdir != null) c.flag("-w", workdir);
        c.add(idOrName).add(command);
        return c.runOk();
    }

    public String execDetach(String idOrName, List<String> command) {
        Objects.requireNonNull(command, "command");
        return cmd("exec", "-d", idOrName).add(command).runOk();
    }

    public List<DockerModels.ContainerInfo> ps(boolean all) {
        Cmd c = cmd("ps");
        if (all) c.all();
        c.formatJson();
        String out = c.runOk();
        return parseContainerLines(out);
    }

    public List<DockerModels.ContainerInfo> ps(PsOptions opts) {
        PsOptions o = opts == null ? PsOptions.defaults() : opts;
        Cmd c = cmd("ps");
        if (o.all) c.all();
        if (o.quiet) c.quiet();
        if (o.size) c.flag("-s");
        if (o.latest) c.flag("-l");
        if (o.lastN != null) c.flag("-n", o.lastN);
        if (o.noTrunc) c.flag("--no-trunc");
        if (o.filter != null) {
            for (String f : o.filter) c.flag("--filter", f);
        }
        if (o.format != null) c.format(o.format);
        else if (!o.quiet) c.formatJson();
        String out = c.runOk();
        if (o.quiet) {
            List<DockerModels.ContainerInfo> list = new ArrayList<>();
            for (String line : out.split("\n")) {
                line = line.trim();
                if (!line.isEmpty()) {
                    list.add(new DockerModels.ContainerInfo(line, "", "", "", "", List.of(), Map.of()));
                }
            }
            return list;
        }
        return parseContainerLines(out);
    }

    public static final class PsOptions {
        public boolean all;
        public boolean quiet;
        public boolean size;
        public boolean latest;
        public Integer lastN;
        public boolean noTrunc;
        public List<String> filter;
        public String format;

        public static PsOptions defaults() { return new PsOptions(); }
        public PsOptions all(boolean v) { this.all = v; return this; }
        public PsOptions quiet(boolean v) { this.quiet = v; return this; }
        public PsOptions size(boolean v) { this.size = v; return this; }
        public PsOptions latest(boolean v) { this.latest = v; return this; }
        public PsOptions lastN(int v) { this.lastN = v; return this; }
        public PsOptions noTrunc(boolean v) { this.noTrunc = v; return this; }
        public PsOptions filter(String... f) {
            this.filter = f == null ? null : List.of(f);
            return this;
        }
        public PsOptions format(String v) { this.format = v; return this; }
    }

    public String build(Path context, String tag, Map<String, String> buildArgs, String dockerfile) {
        return build(BuildOptions.defaults()
                .context(context == null ? Path.of(".") : context)
                .tag(tag)
                .buildArgs(buildArgs)
                .dockerfile(dockerfile));
    }

    public String build(BuildOptions opts) {
        BuildOptions o = Objects.requireNonNull(opts, "opts");
        Cmd c = cmd("build");
        if (o.tags != null) for (String t : o.tags) c.flag("-t", t);
        if (o.dockerfile != null) c.flag("-f", o.dockerfile);
        if (o.buildArgs != null) {
            for (Map.Entry<String, String> e : o.buildArgs.entrySet()) {
                c.flag("--build-arg", e.getKey() + "=" + e.getValue());
            }
        }
        if (o.target != null) c.flag("--target", o.target);
        if (o.network != null) c.flag("--network", o.network);
        if (o.noCache) c.flag("--no-cache");
        if (o.pull) c.flag("--pull");
        if (o.quiet) c.quiet();
        if (o.rm) c.flag("--rm");
        if (o.forceRm) c.flag("--force-rm");
        if (o.platform != null) c.flag("--platform", o.platform);
        if (o.progress != null) c.flag("--progress", o.progress);
        if (o.secret != null) for (String s : o.secret) c.flag("--secret", s);
        if (o.ssh != null) c.flag("--ssh", o.ssh);
        if (o.labels != null) {
            for (Map.Entry<String, String> e : o.labels.entrySet()) {
                c.flag("--label", e.getKey() + "=" + e.getValue());
            }
        }
        if (o.extra != null) c.add(o.extra);
        c.add(o.context == null ? "." : o.context.toString());
        ProcessRunner.CommandResult r = raw(c.argv(), o.timeout == null ? options.timeout : o.timeout);
        if (!r.ok()) throw DockerException.ofExit("build", r.exitCode(), r.output());
        return o.tags == null || o.tags.isEmpty() ? null : o.tags.get(0);
    }

    public static final class BuildOptions {
        public Path context = Path.of(".");
        public List<String> tags = new ArrayList<>();
        public String dockerfile;
        public Map<String, String> buildArgs;
        public String target;
        public String network;
        public boolean noCache;
        public boolean pull;
        public boolean quiet;
        public boolean rm = true;
        public boolean forceRm;
        public String platform;
        public String progress;
        public List<String> secret;
        public String ssh;
        public Map<String, String> labels;
        public List<String> extra;
        public Duration timeout;

        public static BuildOptions defaults() { return new BuildOptions(); }
        public BuildOptions context(Path p) { this.context = p; return this; }
        public BuildOptions tag(String t) {
            if (t != null) this.tags.add(t);
            return this;
        }
        public BuildOptions tags(String... t) {
            if (t != null) this.tags.addAll(List.of(t));
            return this;
        }
        public BuildOptions dockerfile(String v) { this.dockerfile = v; return this; }
        public BuildOptions buildArgs(Map<String, String> m) { this.buildArgs = m; return this; }
        public BuildOptions target(String v) { this.target = v; return this; }
        public BuildOptions network(String v) { this.network = v; return this; }
        public BuildOptions noCache(boolean v) { this.noCache = v; return this; }
        public BuildOptions pull(boolean v) { this.pull = v; return this; }
        public BuildOptions quiet(boolean v) { this.quiet = v; return this; }
        public BuildOptions platform(String v) { this.platform = v; return this; }
        public BuildOptions progress(String v) { this.progress = v; return this; }
        public BuildOptions secret(String... s) {
            this.secret = s == null ? null : List.of(s);
            return this;
        }
        public BuildOptions ssh(String v) { this.ssh = v; return this; }
        public BuildOptions labels(Map<String, String> m) { this.labels = m; return this; }
        public BuildOptions extra(String... e) {
            this.extra = e == null ? null : List.of(e);
            return this;
        }
        public BuildOptions timeout(Duration d) { this.timeout = d; return this; }
    }

    /** {@code docker buildx bake} / {@code docker bake} (plugin). */
    public String bake(Path file, List<String> targets, boolean print, boolean push, boolean load) {
        Cmd c = cmd("buildx", "bake");
        if (file != null) c.flag("-f", file.toAbsolutePath().toString());
        if (print) c.flag("--print");
        if (push) c.flag("--push");
        if (load) c.flag("--load");
        if (targets != null) c.add(targets);
        ProcessRunner.CommandResult r = raw(c.argv());
        if (!r.ok()) {
            // fallback to bare `docker bake` on newer CLIs
            Cmd c2 = cmd("bake");
            if (file != null) c2.flag("-f", file.toAbsolutePath().toString());
            if (print) c2.flag("--print");
            if (push) c2.flag("--push");
            if (load) c2.flag("--load");
            if (targets != null) c2.add(targets);
            return c2.runOk();
        }
        return r.stdout();
    }

    public void pull(String image) {
        requireOk("pull", List.of("pull", image));
    }

    public void pull(String image, String platform, boolean quiet, boolean allTags) {
        Cmd c = cmd("pull", image);
        if (platform != null) c.flag("--platform", platform);
        if (quiet) c.quiet();
        if (allTags) c.flag("-a");
        c.runOk();
    }

    public void push(String image) {
        requireOk("push", List.of("push", image));
    }

    public void push(String image, boolean allTags, boolean quiet) {
        Cmd c = cmd("push", image);
        if (allTags) c.flag("-a");
        if (quiet) c.quiet();
        c.runOk();
    }

    public List<DockerModels.ImageInfo> images() {
        String out = requireOk("images", List.of("images", "--format", "{{json .}}"));
        return parseImageLines(out);
    }

    public List<DockerModels.ImageInfo> images(ImagesOptions opts) {
        ImagesOptions o = opts == null ? ImagesOptions.defaults() : opts;
        Cmd c = cmd("images");
        if (o.all) c.all();
        if (o.quiet) c.quiet();
        if (o.digests) c.flag("--digests");
        if (o.noTrunc) c.flag("--no-trunc");
        if (o.filter != null) for (String f : o.filter) c.flag("--filter", f);
        if (o.repository != null) c.add(o.repository);
        if (!o.quiet) c.formatJson();
        String out = c.runOk();
        if (o.quiet) {
            List<DockerModels.ImageInfo> list = new ArrayList<>();
            for (String line : out.split("\n")) {
                line = line.trim();
                if (!line.isEmpty()) list.add(new DockerModels.ImageInfo(line, "", "", 0));
            }
            return list;
        }
        return parseImageLines(out);
    }

    public static final class ImagesOptions {
        public boolean all;
        public boolean quiet;
        public boolean digests;
        public boolean noTrunc;
        public List<String> filter;
        public String repository;

        public static ImagesOptions defaults() { return new ImagesOptions(); }
        public ImagesOptions all(boolean v) { this.all = v; return this; }
        public ImagesOptions quiet(boolean v) { this.quiet = v; return this; }
        public ImagesOptions digests(boolean v) { this.digests = v; return this; }
        public ImagesOptions noTrunc(boolean v) { this.noTrunc = v; return this; }
        public ImagesOptions filter(String... f) {
            this.filter = f == null ? null : List.of(f);
            return this;
        }
        public ImagesOptions repository(String v) { this.repository = v; return this; }
    }

    // =========================================================================
    // Container lifecycle: create/start/stop/kill/restart/rm/pause/unpause/
    //                      rename/update/wait/attach/logs/top/port/stats/diff/cp
    // =========================================================================

    public String create(DockerModels.RunSpec spec) {
        // reuse run flags but with `create` verb
        Objects.requireNonNull(spec, "spec");
        List<String> runArgs = spec.toCliArgs();
        // toCliArgs starts with "run" — replace
        List<String> args = new ArrayList<>();
        args.add("create");
        if (runArgs.size() > 1) args.addAll(runArgs.subList(1, runArgs.size()));
        return firstLine(requireOk("create", args));
    }

    public String createRaw(List<String> createArgs) {
        List<String> args = new ArrayList<>();
        args.add("create");
        if (createArgs != null) args.addAll(createArgs);
        return firstLine(requireOk("create", args));
    }

    public void start(String... idOrNames) {
        Cmd c = cmd("start");
        c.add(idOrNames);
        c.runOk();
    }

    public void startAttach(String idOrName, boolean interactive) {
        Cmd c = cmd("start", "-a");
        if (interactive) c.flag("-i");
        c.add(idOrName).runOk();
    }

    public void stop(String idOrName) {
        stop(idOrName, null);
    }

    public void stop(String idOrName, Integer timeoutSeconds) {
        Cmd c = cmd("stop");
        if (timeoutSeconds != null) c.flag("-t", timeoutSeconds);
        c.add(idOrName).runOk();
    }

    public void stopAll(List<String> idOrNames, Integer timeoutSeconds) {
        Cmd c = cmd("stop");
        if (timeoutSeconds != null) c.flag("-t", timeoutSeconds);
        if (idOrNames != null) c.add(idOrNames);
        c.runOk();
    }

    public void kill(String idOrName, String signal) {
        Cmd c = cmd("kill");
        if (signal != null) c.flag("-s", signal);
        c.add(idOrName).runOk();
    }

    public void restart(String idOrName, Integer timeoutSeconds) {
        Cmd c = cmd("restart");
        if (timeoutSeconds != null) c.flag("-t", timeoutSeconds);
        c.add(idOrName).runOk();
    }

    public void rm(String idOrName, boolean force) {
        rm(idOrName, force, false);
    }

    public void rm(String idOrName, boolean force, boolean removeVolumes) {
        Cmd c = cmd("rm");
        if (force) c.force();
        if (removeVolumes) c.flag("-v");
        c.add(idOrName).runOk();
    }

    public void rmAll(List<String> ids, boolean force, boolean removeVolumes) {
        Cmd c = cmd("rm");
        if (force) c.force();
        if (removeVolumes) c.flag("-v");
        if (ids != null) c.add(ids);
        c.runOk();
    }

    public void pause(String... idOrNames) {
        cmd("pause").add(idOrNames).runOk();
    }

    public void unpause(String... idOrNames) {
        cmd("unpause").add(idOrNames).runOk();
    }

    public void rename(String oldName, String newName) {
        requireOk("rename", List.of("rename", oldName, newName));
    }

    public void update(String idOrName, UpdateOptions opts) {
        UpdateOptions o = opts == null ? UpdateOptions.defaults() : opts;
        Cmd c = cmd("update");
        if (o.cpus != null) c.flag("--cpus", o.cpus);
        if (o.memory != null) c.flag("--memory", o.memory);
        if (o.memorySwap != null) c.flag("--memory-swap", o.memorySwap);
        if (o.restart != null) c.flag("--restart", o.restart);
        if (o.pidsLimit != null) c.flag("--pids-limit", o.pidsLimit);
        if (o.blkioWeight != null) c.flag("--blkio-weight", o.blkioWeight);
        if (o.cpuShares != null) c.flag("--cpu-shares", o.cpuShares);
        if (o.extra != null) c.add(o.extra);
        c.add(idOrName).runOk();
    }

    public static final class UpdateOptions {
        public String cpus;
        public String memory;
        public String memorySwap;
        public String restart;
        public Integer pidsLimit;
        public Integer blkioWeight;
        public Integer cpuShares;
        public List<String> extra;

        public static UpdateOptions defaults() { return new UpdateOptions(); }
        public UpdateOptions cpus(String v) { this.cpus = v; return this; }
        public UpdateOptions memory(String v) { this.memory = v; return this; }
        public UpdateOptions memorySwap(String v) { this.memorySwap = v; return this; }
        public UpdateOptions restart(String v) { this.restart = v; return this; }
        public UpdateOptions pidsLimit(int v) { this.pidsLimit = v; return this; }
        public UpdateOptions blkioWeight(int v) { this.blkioWeight = v; return this; }
        public UpdateOptions cpuShares(int v) { this.cpuShares = v; return this; }
        public UpdateOptions extra(String... e) {
            this.extra = e == null ? null : List.of(e);
            return this;
        }
    }

    public String wait(String... idOrNames) {
        return cmd("wait").add(idOrNames).runOk().trim();
    }

    /** Attach — long-running; returns Process. */
    public Process attach(String idOrName, boolean stdin, boolean sigProxy) throws IOException {
        List<String> cmdLine = new ArrayList<>();
        cmdLine.add(options.dockerBin);
        injectGlobalFlags(cmdLine);
        cmdLine.add("attach");
        if (!sigProxy) cmdLine.add("--sig-proxy=false");
        if (!stdin) cmdLine.add("--no-stdin");
        cmdLine.add(idOrName);
        ProcessBuilder pb = new ProcessBuilder(cmdLine);
        if (!options.extraEnv.isEmpty()) pb.environment().putAll(options.extraEnv);
        return pb.start();
    }

    public String logs(String idOrName, boolean timestamps, Integer tail) {
        return logs(idOrName, LogsOptions.defaults().timestamps(timestamps).tail(tail));
    }

    public String logs(String idOrName, LogsOptions opts) {
        LogsOptions o = opts == null ? LogsOptions.defaults() : opts;
        Cmd c = cmd("logs", idOrName);
        if (o.timestamps) c.flag("-t");
        if (o.tail != null) c.flag("--tail", o.tail);
        if (o.since != null) c.flag("--since", o.since);
        if (o.until != null) c.flag("--until", o.until);
        if (o.details) c.flag("--details");
        ProcessRunner.CommandResult r = raw(c.argv(), o.timeout == null ? options.timeout : o.timeout);
        if (!r.ok()) throw DockerException.ofExit("logs", r.exitCode(), r.output());
        return r.output();
    }

    /** Follow logs in background. */
    public Process logsFollow(String idOrName, Integer tail, boolean timestamps) throws IOException {
        List<String> cmdLine = new ArrayList<>();
        cmdLine.add(options.dockerBin);
        injectGlobalFlags(cmdLine);
        cmdLine.add("logs");
        cmdLine.add("-f");
        if (timestamps) cmdLine.add("-t");
        if (tail != null) {
            cmdLine.add("--tail");
            cmdLine.add(String.valueOf(tail));
        }
        cmdLine.add(idOrName);
        ProcessBuilder pb = new ProcessBuilder(cmdLine);
        pb.redirectErrorStream(true);
        if (!options.extraEnv.isEmpty()) pb.environment().putAll(options.extraEnv);
        return pb.start();
    }

    public static final class LogsOptions {
        public boolean timestamps;
        public Integer tail;
        public String since;
        public String until;
        public boolean details;
        public Duration timeout;

        public static LogsOptions defaults() { return new LogsOptions(); }
        public LogsOptions timestamps(boolean v) { this.timestamps = v; return this; }
        public LogsOptions tail(Integer v) { this.tail = v; return this; }
        public LogsOptions since(String v) { this.since = v; return this; }
        public LogsOptions until(String v) { this.until = v; return this; }
        public LogsOptions details(boolean v) { this.details = v; return this; }
        public LogsOptions timeout(Duration v) { this.timeout = v; return this; }
    }

    public String top(String idOrName, String psArgs) {
        Cmd c = cmd("top", idOrName);
        if (psArgs != null) c.add(psArgs);
        return c.runOk();
    }

    public String port(String idOrName, String privatePort) {
        Cmd c = cmd("port", idOrName);
        if (privatePort != null) c.add(privatePort);
        return c.runOk().trim();
    }

    public String stats(List<String> idOrNames, boolean noStream, boolean all) {
        Cmd c = cmd("stats");
        if (noStream) c.flag("--no-stream");
        if (all) c.all();
        if (idOrNames != null) c.add(idOrNames);
        return c.runOk();
    }

    public String diff(String idOrName) {
        return requireOk("diff", List.of("diff", idOrName));
    }

    public void cp(String src, String dst) {
        requireOk("cp", List.of("cp", src, dst));
    }

    public void cpToContainer(Path local, String container, String remotePath) {
        cp(local.toAbsolutePath().toString(), container + ":" + remotePath);
    }

    public void cpFromContainer(String container, String remotePath, Path local) {
        cp(container + ":" + remotePath, local.toAbsolutePath().toString());
    }

    public String commit(String idOrName, String repository, String message, String author, Map<String, String> change) {
        Cmd c = cmd("commit");
        if (message != null) c.flag("-m", message);
        if (author != null) c.flag("-a", author);
        if (change != null) {
            for (Map.Entry<String, String> e : change.entrySet()) {
                c.flag("-c", e.getKey() + " " + e.getValue());
            }
        }
        c.add(idOrName);
        if (repository != null) c.add(repository);
        return firstLine(c.runOk());
    }

    public void export(String idOrName, Path outputTar) {
        Cmd c = cmd("export", idOrName);
        if (outputTar != null) c.flag("-o", outputTar.toAbsolutePath().toString());
        c.runOk();
    }

    // =========================================================================
    // Image ops: rmi / tag / save / load / history / import / inspect / prune
    // =========================================================================

    public void rmi(String image, boolean force, boolean noPrune) {
        Cmd c = cmd("rmi");
        if (force) c.force();
        if (noPrune) c.flag("--no-prune");
        c.add(image).runOk();
    }

    public void rmiAll(List<String> images, boolean force) {
        Cmd c = cmd("rmi");
        if (force) c.force();
        if (images != null) c.add(images);
        c.runOk();
    }

    public void tag(String source, String target) {
        requireOk("tag", List.of("tag", source, target));
    }

    public void save(List<String> images, Path outputTar) {
        Cmd c = cmd("save");
        if (outputTar != null) c.flag("-o", outputTar.toAbsolutePath().toString());
        if (images != null) c.add(images);
        c.runOk();
    }

    public void load(Path inputTar, boolean quiet) {
        Cmd c = cmd("load");
        if (inputTar != null) c.flag("-i", inputTar.toAbsolutePath().toString());
        if (quiet) c.quiet();
        c.runOk();
    }

    public void loadStdin(byte[] tarBytes, boolean quiet) {
        // ProcessRunner stdin is String — for binary use temp file path via load(Path)
        throw new DockerException("loadStdin: use load(Path) for binary tar input", -1, "load");
    }

    public String history(String image, boolean noTrunc, boolean quiet) {
        Cmd c = cmd("history", image);
        if (noTrunc) c.flag("--no-trunc");
        if (quiet) c.quiet();
        return c.runOk();
    }

    public List<Map<String, Object>> historyJson(String image) {
        return cmd("history", image).formatJson().runJsonLines();
    }

    public String importTar(Path tar, String repository, String message) {
        Cmd c = cmd("import");
        if (message != null) c.flag("-m", message);
        c.add(tar == null ? "-" : tar.toAbsolutePath().toString());
        if (repository != null) c.add(repository);
        return firstLine(c.runOk());
    }

    public DockerModels.ContainerInfo inspectContainer(String idOrName) {
        Objects.requireNonNull(idOrName, "idOrName");
        String out = requireOk("inspect", List.of("inspect", "--format", "{{json .}}", idOrName));
        Map<String, Object> m = decodeObject(out, "inspect");
        return fromInspect(m);
    }

    @SuppressWarnings("unchecked")
    public List<Map<String, Object>> inspectRaw(String... idOrNames) {
        List<String> args = new ArrayList<>();
        args.add("inspect");
        for (String id : idOrNames) if (id != null) args.add(id);
        String out = requireOk("inspect", args);
        try {
            Object v = Json.decode(out.trim());
            if (v instanceof List<?> list) {
                List<Map<String, Object>> result = new ArrayList<>();
                for (Object o : list) {
                    if (o instanceof Map<?, ?> map) result.add((Map<String, Object>) map);
                }
                return result;
            }
            if (v instanceof Map<?, ?> map) {
                return List.of((Map<String, Object>) map);
            }
            throw new DockerException("unexpected inspect payload");
        } catch (IOException e) {
            throw new DockerException("inspect json: " + e.getMessage(), e);
        }
    }

    public Map<String, Object> inspectObject(String idOrName) {
        String out = requireOk("inspect", List.of("inspect", "--format", "{{json .}}", idOrName));
        return decodeObject(out, "inspect");
    }

    public String inspectFormat(String idOrName, String goTemplate) {
        return requireOk("inspect", List.of("inspect", "--format", goTemplate, idOrName)).trim();
    }

    public DockerModels.HealthStatus health(String idOrName) {
        String out = requireOk("health", List.of(
                "inspect", "--format",
                "{{if .State.Health}}{{.State.Health.Status}}|{{.State.Health.FailingStreak}}{{else}}none|0{{end}}",
                idOrName));
        String line = firstLine(out);
        String[] sp = line.split("\\|", 2);
        String status = sp.length > 0 ? sp[0].trim() : "none";
        int streak = 0;
        if (sp.length > 1) {
            try { streak = Integer.parseInt(sp[1].trim()); } catch (NumberFormatException ignored) {}
        }
        return new DockerModels.HealthStatus(status, streak, "");
    }

    public String state(String idOrName) {
        return requireOk("state", List.of("inspect", "--format", "{{.State.Status}}", idOrName)).trim();
    }

    // =========================================================================
    // events
    // =========================================================================

    public String events(EventsOptions opts) {
        EventsOptions o = opts == null ? EventsOptions.defaults() : opts;
        Cmd c = cmd("events");
        if (o.since != null) c.flag("--since", o.since);
        if (o.until != null) c.flag("--until", o.until);
        if (o.filter != null) for (String f : o.filter) c.flag("--filter", f);
        if (o.format != null) c.format(o.format);
        return c.timeout(o.timeout == null ? Duration.ofSeconds(30) : o.timeout).runOk();
    }

    /** Stream events — caller reads process stdout. */
    public Process eventsFollow(List<String> filters) throws IOException {
        List<String> cmdLine = new ArrayList<>();
        cmdLine.add(options.dockerBin);
        injectGlobalFlags(cmdLine);
        cmdLine.add("events");
        if (filters != null) {
            for (String f : filters) {
                cmdLine.add("--filter");
                cmdLine.add(f);
            }
        }
        ProcessBuilder pb = new ProcessBuilder(cmdLine);
        pb.redirectErrorStream(true);
        if (!options.extraEnv.isEmpty()) pb.environment().putAll(options.extraEnv);
        return pb.start();
    }

    public static final class EventsOptions {
        public String since;
        public String until;
        public List<String> filter;
        public String format;
        public Duration timeout;

        public static EventsOptions defaults() { return new EventsOptions(); }
        public EventsOptions since(String v) { this.since = v; return this; }
        public EventsOptions until(String v) { this.until = v; return this; }
        public EventsOptions filter(String... f) {
            this.filter = f == null ? null : List.of(f);
            return this;
        }
        public EventsOptions format(String v) { this.format = v; return this; }
        public EventsOptions timeout(Duration v) { this.timeout = v; return this; }
    }

    // =========================================================================
    // Management namespaces: container / image / network / volume / system /
    //                        context / builder / plugin / manifest / swarm
    // =========================================================================

    public ContainerCmd container() { return new ContainerCmd(this); }
    public ImageCmd image() { return new ImageCmd(this); }
    public NetworkCmd network() { return new NetworkCmd(this); }
    public VolumeCmd volume() { return new VolumeCmd(this); }
    public SystemCmd system() { return new SystemCmd(this); }
    public ContextCmd context() { return new ContextCmd(this); }
    public BuilderCmd builder() { return new BuilderCmd(this); }
    public PluginCmd plugin() { return new PluginCmd(this); }
    public ManifestCmd manifest() { return new ManifestCmd(this); }
    public SwarmCmd swarm() { return new SwarmCmd(this); }

    // ---- container ----
    public static final class ContainerCmd {
        private final DockerCli d;
        ContainerCmd(DockerCli d) { this.d = d; }

        public String ls(boolean all) {
            Cmd c = d.cmd("container", "ls");
            if (all) c.all();
            return c.runOk();
        }

        public List<DockerModels.ContainerInfo> lsJson(boolean all) {
            return d.ps(all);
        }

        public String inspect(String id) { return d.inspectFormat(id, "{{json .}}"); }
        public void start(String... ids) { d.start(ids); }
        public void stop(String id, Integer t) { d.stop(id, t); }
        public void kill(String id, String sig) { d.kill(id, sig); }
        public void restart(String id, Integer t) { d.restart(id, t); }
        public void rm(String id, boolean force) { d.rm(id, force); }
        public void pause(String... ids) { d.pause(ids); }
        public void unpause(String... ids) { d.unpause(ids); }
        public void prune(boolean force, String filter) {
            Cmd c = d.cmd("container", "prune");
            if (force) c.flag("-f");
            if (filter != null) c.flag("--filter", filter);
            c.runOk();
        }
        public String logs(String id, Integer tail) { return d.logs(id, false, tail); }
        public String top(String id) { return d.top(id, null); }
        public String stats(String id) { return d.stats(List.of(id), true, false); }
        public String diff(String id) { return d.diff(id); }
        public void rename(String oldN, String newN) { d.rename(oldN, newN); }
        public void update(String id, UpdateOptions o) { d.update(id, o); }
        public String wait(String... ids) { return d.wait(ids); }
        public String run(DockerModels.RunSpec spec) { return d.run(spec); }
        public String create(DockerModels.RunSpec spec) { return d.create(spec); }
        public String exec(String id, List<String> cmd, boolean tty) { return d.exec(id, cmd, tty); }
        public void cp(String src, String dst) { d.cp(src, dst); }
        public String commit(String id, String repo) { return d.commit(id, repo, null, null, null); }
        public void export(String id, Path out) { d.export(id, out); }
        public String port(String id) { return d.port(id, null); }
    }

    // ---- image ----
    public static final class ImageCmd {
        private final DockerCli d;
        ImageCmd(DockerCli d) { this.d = d; }

        public List<DockerModels.ImageInfo> ls() { return d.images(); }
        public List<DockerModels.ImageInfo> ls(ImagesOptions o) { return d.images(o); }
        public void build(BuildOptions o) { d.build(o); }
        public void pull(String ref) { d.pull(ref); }
        public void push(String ref) { d.push(ref); }
        public void tag(String src, String dst) { d.tag(src, dst); }
        public void rm(String ref, boolean force) { d.rmi(ref, force, false); }
        public void save(List<String> refs, Path out) { d.save(refs, out); }
        public void load(Path in) { d.load(in, false); }
        public String history(String ref) { return d.history(ref, false, false); }
        public String inspect(String ref) { return d.inspectFormat(ref, "{{json .}}"); }
        public String importTar(Path tar, String repo) { return d.importTar(tar, repo, null); }
        public void prune(boolean all, boolean force, String filter) {
            Cmd c = d.cmd("image", "prune");
            if (all) c.flag("-a");
            if (force) c.flag("-f");
            if (filter != null) c.flag("--filter", filter);
            c.runOk();
        }
    }

    // ---- network ----
    public static final class NetworkCmd {
        private final DockerCli d;
        NetworkCmd(DockerCli d) { this.d = d; }

        public void create(String name) {
            d.requireOk("network_create", List.of("network", "create", name));
        }

        public void create(String name, NetworkCreateOptions opts) {
            NetworkCreateOptions o = opts == null ? NetworkCreateOptions.defaults() : opts;
            Cmd c = d.cmd("network", "create");
            if (o.driver != null) c.flag("-d", o.driver);
            if (o.subnet != null) c.flag("--subnet", o.subnet);
            if (o.gateway != null) c.flag("--gateway", o.gateway);
            if (o.ipRange != null) c.flag("--ip-range", o.ipRange);
            if (o.internal) c.flag("--internal");
            if (o.ipv6) c.flag("--ipv6");
            if (o.attachable) c.flag("--attachable");
            if (o.opt != null) {
                for (Map.Entry<String, String> e : o.opt.entrySet()) {
                    c.flag("-o", e.getKey() + "=" + e.getValue());
                }
            }
            if (o.labels != null) {
                for (Map.Entry<String, String> e : o.labels.entrySet()) {
                    c.flag("--label", e.getKey() + "=" + e.getValue());
                }
            }
            c.add(name).runOk();
        }

        public void rm(String name) {
            d.requireOk("network_rm", List.of("network", "rm", name));
        }

        public List<String> ls() {
            String out = d.requireOk("network_ls", List.of("network", "ls", "--format", "{{.Name}}"));
            List<String> names = new ArrayList<>();
            for (String line : out.split("\n")) {
                line = line.trim();
                if (!line.isEmpty()) names.add(line);
            }
            return names;
        }

        public List<Map<String, Object>> lsJson() {
            return d.cmd("network", "ls").formatJson().runJsonLines();
        }

        public Map<String, Object> inspect(String name) {
            return d.inspectObject(name);
        }

        public void connect(String network, String container, String ip, String alias) {
            Cmd c = d.cmd("network", "connect");
            if (ip != null) c.flag("--ip", ip);
            if (alias != null) c.flag("--alias", alias);
            c.add(network, container).runOk();
        }

        public void disconnect(String network, String container, boolean force) {
            Cmd c = d.cmd("network", "disconnect");
            if (force) c.force();
            c.add(network, container).runOk();
        }

        public void prune(boolean force, String filter) {
            Cmd c = d.cmd("network", "prune");
            if (force) c.flag("-f");
            if (filter != null) c.flag("--filter", filter);
            c.runOk();
        }
    }

    public static final class NetworkCreateOptions {
        public String driver;
        public String subnet;
        public String gateway;
        public String ipRange;
        public boolean internal;
        public boolean ipv6;
        public boolean attachable;
        public Map<String, String> opt;
        public Map<String, String> labels;

        public static NetworkCreateOptions defaults() { return new NetworkCreateOptions(); }
        public NetworkCreateOptions driver(String v) { this.driver = v; return this; }
        public NetworkCreateOptions subnet(String v) { this.subnet = v; return this; }
        public NetworkCreateOptions gateway(String v) { this.gateway = v; return this; }
        public NetworkCreateOptions ipRange(String v) { this.ipRange = v; return this; }
        public NetworkCreateOptions internal(boolean v) { this.internal = v; return this; }
        public NetworkCreateOptions ipv6(boolean v) { this.ipv6 = v; return this; }
        public NetworkCreateOptions attachable(boolean v) { this.attachable = v; return this; }
        public NetworkCreateOptions opt(Map<String, String> m) { this.opt = m; return this; }
        public NetworkCreateOptions labels(Map<String, String> m) { this.labels = m; return this; }
    }

    // ---- volume ----
    public static final class VolumeCmd {
        private final DockerCli d;
        VolumeCmd(DockerCli d) { this.d = d; }

        public void create(String name) {
            d.requireOk("volume_create", List.of("volume", "create", name));
        }

        public void create(String name, String driver, Map<String, String> opt, Map<String, String> labels) {
            Cmd c = d.cmd("volume", "create");
            if (driver != null) c.flag("-d", driver);
            if (opt != null) {
                for (Map.Entry<String, String> e : opt.entrySet()) {
                    c.flag("--opt", e.getKey() + "=" + e.getValue());
                }
            }
            if (labels != null) {
                for (Map.Entry<String, String> e : labels.entrySet()) {
                    c.flag("--label", e.getKey() + "=" + e.getValue());
                }
            }
            c.add(name).runOk();
        }

        public void rm(String name) {
            d.requireOk("volume_rm", List.of("volume", "rm", name));
        }

        public void rm(String name, boolean force) {
            Cmd c = d.cmd("volume", "rm");
            if (force) c.force();
            c.add(name).runOk();
        }

        public List<String> ls() {
            String out = d.requireOk("volume_ls", List.of("volume", "ls", "--format", "{{.Name}}"));
            List<String> names = new ArrayList<>();
            for (String line : out.split("\n")) {
                line = line.trim();
                if (!line.isEmpty()) names.add(line);
            }
            return names;
        }

        public List<Map<String, Object>> lsJson(String filter) {
            Cmd c = d.cmd("volume", "ls").formatJson();
            if (filter != null) c.flag("--filter", filter);
            return c.runJsonLines();
        }

        public Map<String, Object> inspect(String name) {
            return d.inspectObject(name);
        }

        public void prune(boolean force, String filter) {
            Cmd c = d.cmd("volume", "prune");
            if (force) c.flag("-f");
            if (filter != null) c.flag("--filter", filter);
            c.runOk();
        }
    }

    // ---- system ----
    public static final class SystemCmd {
        private final DockerCli d;
        SystemCmd(DockerCli d) { this.d = d; }

        public String df() {
            return d.cmd("system", "df").runOk();
        }

        public String dfVerbose() {
            return d.cmd("system", "df", "-v").runOk();
        }

        public Map<String, Object> info() {
            return d.info();
        }

        public String events(EventsOptions o) {
            return d.events(o);
        }

        public String prune(boolean all, boolean force, String filter, boolean volumes) {
            Cmd c = d.cmd("system", "prune");
            if (all) c.flag("-a");
            if (force) c.flag("-f");
            if (volumes) c.flag("--volumes");
            if (filter != null) c.flag("--filter", filter);
            return c.runOk();
        }
    }

    // ---- context ----
    public static final class ContextCmd {
        private final DockerCli d;
        ContextCmd(DockerCli d) { this.d = d; }

        public String ls() {
            return d.cmd("context", "ls").runOk();
        }

        public List<Map<String, Object>> lsJson() {
            return d.cmd("context", "ls").formatJson().runJsonLines();
        }

        public String create(String name, String dockerHost, String description) {
            Cmd c = d.cmd("context", "create", name);
            if (dockerHost != null) c.flag("--docker", "host=" + dockerHost);
            if (description != null) c.flag("--description", description);
            return c.runOk();
        }

        public void use(String name) {
            d.cmd("context", "use", name).runOk();
        }

        public void rm(String name, boolean force) {
            Cmd c = d.cmd("context", "rm");
            if (force) c.force();
            c.add(name).runOk();
        }

        public Map<String, Object> inspect(String name) {
            String out = d.cmd("context", "inspect", name).format("{{json .}}").runOk();
            // may be array
            try {
                Object v = Json.decode(out.trim());
                if (v instanceof List<?> list && !list.isEmpty() && list.get(0) instanceof Map<?, ?> m) {
                    @SuppressWarnings("unchecked")
                    Map<String, Object> map = (Map<String, Object>) m;
                    return map;
                }
                if (v instanceof Map<?, ?> m) {
                    @SuppressWarnings("unchecked")
                    Map<String, Object> map = (Map<String, Object>) m;
                    return map;
                }
            } catch (IOException e) {
                throw new DockerException("context inspect: " + e.getMessage(), e);
            }
            return Map.of();
        }

        public void update(String name, String description) {
            Cmd c = d.cmd("context", "update", name);
            if (description != null) c.flag("--description", description);
            c.runOk();
        }

        public void exportCtx(String name, Path file) {
            d.cmd("context", "export", name, file.toAbsolutePath().toString()).runOk();
        }

        public void importCtx(String name, Path file) {
            d.cmd("context", "import", name, file.toAbsolutePath().toString()).runOk();
        }

        public String show() {
            return d.cmd("context", "show").runOk().trim();
        }
    }

    // ---- builder / buildx ----
    public static final class BuilderCmd {
        private final DockerCli d;
        BuilderCmd(DockerCli d) { this.d = d; }

        public String prune(boolean all, boolean force, String filter) {
            Cmd c = d.cmd("builder", "prune");
            if (all) c.flag("-a");
            if (force) c.flag("-f");
            if (filter != null) c.flag("--filter", filter);
            return c.runOk();
        }

        public String buildxVersion() {
            return d.cmd("buildx", "version").runOk().trim();
        }

        public String buildxLs() {
            return d.cmd("buildx", "ls").runOk();
        }

        public void buildxCreate(String name, String driver, boolean use) {
            Cmd c = d.cmd("buildx", "create");
            if (name != null) c.flag("--name", name);
            if (driver != null) c.flag("--driver", driver);
            if (use) c.flag("--use");
            c.runOk();
        }

        public void buildxUse(String name) {
            d.cmd("buildx", "use", name).runOk();
        }

        public void buildxRm(String name) {
            d.cmd("buildx", "rm", name).runOk();
        }

        public String buildxInspect(String name) {
            Cmd c = d.cmd("buildx", "inspect");
            if (name != null) c.add(name);
            return c.runOk();
        }

        public String bake(Path file, List<String> targets, boolean push, boolean load) {
            return d.bake(file, targets, false, push, load);
        }
    }

    // ---- plugin ----
    public static final class PluginCmd {
        private final DockerCli d;
        PluginCmd(DockerCli d) { this.d = d; }

        public String ls() {
            return d.cmd("plugin", "ls").runOk();
        }

        public List<Map<String, Object>> lsJson() {
            return d.cmd("plugin", "ls").formatJson().runJsonLines();
        }

        public void install(String plugin, boolean grantAll) {
            Cmd c = d.cmd("plugin", "install");
            if (grantAll) c.flag("--grant-all-permissions");
            c.add(plugin).runOk();
        }

        public void enable(String plugin) {
            d.cmd("plugin", "enable", plugin).runOk();
        }

        public void disable(String plugin, boolean force) {
            Cmd c = d.cmd("plugin", "disable");
            if (force) c.force();
            c.add(plugin).runOk();
        }

        public void rm(String plugin, boolean force) {
            Cmd c = d.cmd("plugin", "rm");
            if (force) c.force();
            c.add(plugin).runOk();
        }

        public String inspect(String plugin) {
            return d.cmd("plugin", "inspect", plugin).runOk();
        }

        public void upgrade(String plugin) {
            d.cmd("plugin", "upgrade", plugin).runOk();
        }

        public void create(String name, Path path) {
            d.cmd("plugin", "create", name, path.toAbsolutePath().toString()).runOk();
        }

        public void push(String plugin) {
            d.cmd("plugin", "push", plugin).runOk();
        }

        public void set(String plugin, Map<String, String> conf) {
            Cmd c = d.cmd("plugin", "set", plugin);
            if (conf != null) {
                for (Map.Entry<String, String> e : conf.entrySet()) {
                    c.add(e.getKey() + "=" + e.getValue());
                }
            }
            c.runOk();
        }
    }

    // ---- manifest ----
    public static final class ManifestCmd {
        private final DockerCli d;
        ManifestCmd(DockerCli d) { this.d = d; }

        public void create(String list, List<String> manifests, boolean amend) {
            Cmd c = d.cmd("manifest", "create");
            if (amend) c.flag("-a");
            c.add(list);
            if (manifests != null) c.add(manifests);
            c.runOk();
        }

        public String inspect(String name) {
            return d.cmd("manifest", "inspect", name).runOk();
        }

        public void annotate(String list, String manifest, String arch, String os, String osVersion, String variant) {
            Cmd c = d.cmd("manifest", "annotate", list, manifest);
            if (arch != null) c.flag("--arch", arch);
            if (os != null) c.flag("--os", os);
            if (osVersion != null) c.flag("--os-version", osVersion);
            if (variant != null) c.flag("--variant", variant);
            c.runOk();
        }

        public void push(String list, boolean purge) {
            Cmd c = d.cmd("manifest", "push");
            if (purge) c.flag("--purge");
            c.add(list).runOk();
        }

        public void rm(String... lists) {
            d.cmd("manifest", "rm").add(lists).runOk();
        }
    }

    // ---- swarm ----
    public static final class SwarmCmd {
        private final DockerCli d;
        SwarmCmd(DockerCli d) { this.d = d; }

        public String init(String advertiseAddr, String listenAddr, boolean forceNewCluster) {
            Cmd c = d.cmd("swarm", "init");
            if (advertiseAddr != null) c.flag("--advertise-addr", advertiseAddr);
            if (listenAddr != null) c.flag("--listen-addr", listenAddr);
            if (forceNewCluster) c.flag("--force-new-cluster");
            return c.runOk();
        }

        public void join(String token, String managerAddr, String advertiseAddr, String listenAddr) {
            Cmd c = d.cmd("swarm", "join");
            if (token != null) c.flag("--token", token);
            if (advertiseAddr != null) c.flag("--advertise-addr", advertiseAddr);
            if (listenAddr != null) c.flag("--listen-addr", listenAddr);
            c.add(managerAddr).runOk();
        }

        public String joinToken(String workerOrManager, boolean rotate) {
            Cmd c = d.cmd("swarm", "join-token", workerOrManager == null ? "worker" : workerOrManager);
            if (rotate) c.flag("--rotate");
            return c.flag("-q").runOk().trim();
        }

        public void leave(boolean force) {
            Cmd c = d.cmd("swarm", "leave");
            if (force) c.force();
            c.runOk();
        }

        public void update(Integer taskHistoryLimit, String dispatcherHeartbeat, Boolean autolock) {
            Cmd c = d.cmd("swarm", "update");
            if (taskHistoryLimit != null) c.flag("--task-history-limit", taskHistoryLimit);
            if (dispatcherHeartbeat != null) c.flag("--dispatcher-heartbeat", dispatcherHeartbeat);
            if (autolock != null) c.flagEq("--autolock", autolock);
            c.runOk();
        }

        public String unlockKey(boolean rotate) {
            Cmd c = d.cmd("swarm", "unlock-key");
            if (rotate) c.flag("--rotate");
            return c.flag("-q").runOk().trim();
        }

        public void unlock(String key) {
            d.cmd("swarm", "unlock").stdin(key == null ? "" : key).runOk();
        }

        public String ca() {
            return d.cmd("swarm", "ca").runOk();
        }
    }

    // ---- legacy network/volume shims kept for callers ----

    public void networkCreate(String name) {
        network().create(name);
    }

    public void networkRm(String name) {
        network().rm(name);
    }

    public List<String> networkLs() {
        return network().ls();
    }

    public void volumeCreate(String name) {
        volume().create(name);
    }

    public void volumeRm(String name) {
        volume().rm(name);
    }

    // =========================================================================
    // Helpers
    // =========================================================================

    private List<DockerModels.ContainerInfo> parseContainerLines(String out) {
        List<DockerModels.ContainerInfo> list = new ArrayList<>();
        for (String line : out.split("\n")) {
            line = line.trim();
            if (line.isEmpty()) continue;
            try {
                Map<String, Object> m = Json.decodeObject(line);
                String id = str(m, "ID", str(m, "Id", ""));
                String name = str(m, "Names", str(m, "Name", ""));
                if (name.startsWith("/")) name = name.substring(1);
                if (name.contains(",")) name = name.split(",")[0].trim();
                String image = str(m, "Image", "");
                String status = str(m, "Status", "");
                String state = str(m, "State", "");
                List<String> ports = new ArrayList<>();
                Object p = m.get("Ports");
                if (p != null) ports.add(String.valueOf(p));
                Map<String, String> labels = new LinkedHashMap<>();
                Object lab = m.get("Labels");
                if (lab instanceof Map<?, ?> lm) {
                    for (Map.Entry<?, ?> e : lm.entrySet()) {
                        labels.put(String.valueOf(e.getKey()), String.valueOf(e.getValue()));
                    }
                } else if (lab instanceof String s && !s.isBlank()) {
                    for (String part : s.split(",")) {
                        int eq = part.indexOf('=');
                        if (eq > 0) labels.put(part.substring(0, eq), part.substring(eq + 1));
                    }
                }
                list.add(new DockerModels.ContainerInfo(id, name, image, status, state, ports, labels));
            } catch (Exception ignored) {
            }
        }
        return list;
    }

    private List<DockerModels.ImageInfo> parseImageLines(String out) {
        List<DockerModels.ImageInfo> list = new ArrayList<>();
        for (String line : out.split("\n")) {
            line = line.trim();
            if (line.isEmpty()) continue;
            try {
                Map<String, Object> m = Json.decodeObject(line);
                String id = str(m, "ID", str(m, "Id", ""));
                String repo = str(m, "Repository", "");
                String tag = str(m, "Tag", "");
                long size = parseSize(str(m, "Size", "0"));
                list.add(new DockerModels.ImageInfo(id, repo, tag, size));
            } catch (Exception ignored) {
            }
        }
        return list;
    }

    @SuppressWarnings("unchecked")
    private DockerModels.ContainerInfo fromInspect(Map<String, Object> m) {
        String id = str(m, "Id", "");
        String name = str(m, "Name", "");
        if (name.startsWith("/")) name = name.substring(1);
        Map<String, Object> config = m.get("Config") instanceof Map<?, ?> c
                ? (Map<String, Object>) c : Map.of();
        Map<String, Object> state = m.get("State") instanceof Map<?, ?> s
                ? (Map<String, Object>) s : Map.of();
        String image = str(config, "Image", "");
        String st = str(state, "Status", "");
        Map<String, String> labels = new LinkedHashMap<>();
        Object lab = config.get("Labels");
        if (lab instanceof Map<?, ?> lm) {
            for (Map.Entry<?, ?> e : lm.entrySet()) {
                labels.put(String.valueOf(e.getKey()), String.valueOf(e.getValue()));
            }
        }
        return new DockerModels.ContainerInfo(id, name, image, st, st, List.of(), labels);
    }

    static List<Map<String, Object>> parseJsonLines(String out) {
        List<Map<String, Object>> list = new ArrayList<>();
        if (out == null || out.isBlank()) return list;
        String t = out.trim();
        try {
            if (t.startsWith("[")) {
                Object v = Json.decode(t);
                if (v instanceof List<?> arr) {
                    for (Object o : arr) {
                        if (o instanceof Map<?, ?> m) {
                            @SuppressWarnings("unchecked")
                            Map<String, Object> row = (Map<String, Object>) m;
                            list.add(row);
                        }
                    }
                }
                return list;
            }
            for (String line : t.split("\n")) {
                line = line.trim();
                if (line.isEmpty()) continue;
                Object v = Json.decode(line);
                if (v instanceof Map<?, ?> m) {
                    @SuppressWarnings("unchecked")
                    Map<String, Object> row = (Map<String, Object>) m;
                    list.add(row);
                }
            }
        } catch (IOException e) {
            throw new DockerException("json lines: " + e.getMessage(), e);
        }
        return list;
    }

    private Map<String, Object> decodeObject(String out, String op) {
        try {
            return Json.decodeObject(out.trim());
        } catch (IOException e) {
            throw new DockerException("docker " + op + " json parse: " + e.getMessage(), e, -1, -1, op);
        }
    }

    private static String firstLine(String out) {
        if (out == null) return "";
        for (String line : out.split("\n")) {
            line = line.trim();
            if (!line.isEmpty()) return line;
        }
        return out.trim();
    }

    private static String str(Map<String, Object> m, String k, String def) {
        Object v = m.get(k);
        return v == null ? def : String.valueOf(v);
    }

    private static long parseSize(String s) {
        if (s == null || s.isBlank()) return 0;
        s = s.trim().toUpperCase();
        try {
            if (s.endsWith("GB") || s.endsWith("G")) {
                return (long) (Double.parseDouble(s.replace("GB", "").replace("G", "").trim()) * 1024 * 1024 * 1024);
            }
            if (s.endsWith("MB") || s.endsWith("M")) {
                return (long) (Double.parseDouble(s.replace("MB", "").replace("M", "").trim()) * 1024 * 1024);
            }
            if (s.endsWith("KB") || s.endsWith("K")) {
                return (long) (Double.parseDouble(s.replace("KB", "").replace("K", "").trim()) * 1024);
            }
            if (s.endsWith("B")) {
                return Long.parseLong(s.substring(0, s.length() - 1).trim());
            }
            return Long.parseLong(s.replaceAll("[^0-9]", ""));
        } catch (Exception e) {
            return 0;
        }
    }
}
