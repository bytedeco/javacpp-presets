package org.bytedeco.pytorch.deploy.k8s;

import org.bytedeco.pytorch.utils.exec.ProcessRunner;
import org.bytedeco.pytorch.utils.json.Json;

import java.io.IOException;
import java.nio.file.Path;
import java.time.Duration;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.TimeUnit;

/**
 * Full-surface {@code kubectl} CLI adapter — zero SDK dependency.
 *
 * <p>Covers every top-level command group from {@code kubectl --help}:
 * <ul>
 *   <li>Beginner: create, expose, run, set</li>
 *   <li>Intermediate: explain, get, edit, delete</li>
 *   <li>Deploy: rollout, scale, autoscale</li>
 *   <li>Cluster: certificate, cluster-info, top, cordon, uncordon, drain, taint</li>
 *   <li>Debug: describe, logs, attach, exec, port-forward, proxy, cp, auth, debug, events</li>
 *   <li>Advanced: diff, apply, patch, replace, wait, kustomize</li>
 *   <li>Settings: label, annotate, completion</li>
 *   <li>Other: api-resources, api-versions, config, plugin, version</li>
 * </ul>
 *
 * <p>Plus a fluent {@link Cmd} builder for arbitrary subcommands/flags, and
 * global option injection from {@link K8sOptions} ({@code --kubeconfig},
 * {@code --context}, {@code --namespace}, {@code --insecure-skip-tls-verify}).
 *
 * <pre>{@code
 * Kubectl k = new Kubectl(K8sOptions.defaults());
 * k.apply(Path.of("deploy.yaml"));
 * k.rollout().status("deployment/ranker").timeout(Duration.ofMinutes(2)).run();
 * k.cmd("get", "pods").ns("ml").output("json").label("app=ranker").runOk();
 * String ver = k.version(true, true);
 * }</pre>
 *
 * @see <a href="https://kubernetes.io/docs/reference/kubectl/">kubectl reference</a>
 */
public final class Kubectl {

    private final K8sOptions options;

    public Kubectl(K8sOptions options) {
        this.options = options == null ? K8sOptions.defaults() : options;
    }

    public K8sOptions options() {
        return options;
    }

    public boolean available() {
        return ProcessRunner.onPath(options.kubectlBin);
    }

    // =========================================================================
    // Low-level execution
    // =========================================================================

    public ProcessRunner.CommandResult raw(List<String> args) {
        return raw(args, options.timeout, null);
    }

    public ProcessRunner.CommandResult raw(List<String> args, Duration timeout, String stdin) {
        List<String> cmd = new ArrayList<>();
        cmd.add(options.kubectlBin);
        injectGlobalFlags(cmd);
        if (args != null) cmd.addAll(args);

        ProcessRunner.Options.Builder ob = ProcessRunner.Options.builder()
                .timeout(timeout == null ? options.timeout : timeout)
                .redirectErrorStream(false);
        if (stdin != null) ob.stdin(stdin);
        if (!options.extraEnv.isEmpty()) ob.env(options.extraEnv);
        return ProcessRunner.run(cmd, ob.build());
    }

    public String requireOk(String operation, List<String> args) {
        return requireOk(operation, args, options.timeout, null);
    }

    public String requireOk(String operation, List<String> args, Duration timeout, String stdin) {
        ProcessRunner.CommandResult r = raw(args, timeout, stdin);
        if (!r.ok()) throw K8sException.ofExit(operation, r.exitCode(), r.output());
        return r.stdout();
    }

    /** Escape hatch: build any kubectl invocation. */
    public Cmd cmd(String... parts) {
        return new Cmd(this).add(parts);
    }

    public Cmd cmd(List<String> parts) {
        return new Cmd(this).add(parts);
    }

    private void injectGlobalFlags(List<String> cmd) {
        if (options.kubeconfig != null) {
            cmd.add("--kubeconfig");
            cmd.add(options.kubeconfig.toAbsolutePath().toString());
        }
        if (options.context != null && !options.context.isBlank()) {
            cmd.add("--context");
            cmd.add(options.context);
        }
        if (options.insecureSkipTls) {
            cmd.add("--insecure-skip-tls-verify=true");
        }
    }

    private void addNamespace(List<String> args, String namespace) {
        String ns = namespace == null || namespace.isBlank() ? options.namespace : namespace;
        if (ns != null && !ns.isBlank()) {
            args.add("-n");
            args.add(ns);
        }
    }

    private static void addIf(List<String> args, boolean cond, String... flags) {
        if (cond && flags != null) {
            Collections.addAll(args, flags);
        }
    }

    private static void addOpt(List<String> args, String flag, Object value) {
        if (value == null) return;
        String s = String.valueOf(value);
        if (s.isBlank()) return;
        args.add(flag);
        args.add(s);
    }

    private static void addOptEq(List<String> args, String flag, Object value) {
        if (value == null) return;
        String s = String.valueOf(value);
        if (s.isBlank()) return;
        args.add(flag + "=" + s);
    }

    // =========================================================================
    // Fluent Cmd builder — arbitrary kubectl invocation
    // =========================================================================

    /**
     * Fluent argv builder. Global flags from {@link K8sOptions} are injected at run time.
     *
     * <pre>{@code
     * k.cmd("get", "pods").allNamespaces().output("wide").label("app=web").runOk();
     * k.cmd("apply").file(path).serverDryRun().runOk();
     * }</pre>
     */
    public static final class Cmd {
        private final Kubectl kubectl;
        private final List<String> args = new ArrayList<>();
        private Duration timeout;
        private String stdin;
        private boolean skipDefaultNamespace;
        private String explicitNamespace;

        Cmd(Kubectl kubectl) {
            this.kubectl = kubectl;
            this.timeout = kubectl.options.timeout;
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

        public Cmd ns(String namespace) {
            this.explicitNamespace = namespace;
            return this;
        }

        public Cmd allNamespaces() {
            this.skipDefaultNamespace = true;
            args.add("--all-namespaces");
            return this;
        }

        public Cmd noNamespace() {
            this.skipDefaultNamespace = true;
            return this;
        }

        public Cmd output(String format) {
            return flag("-o", format);
        }

        public Cmd json() { return output("json"); }
        public Cmd yaml() { return output("yaml"); }
        public Cmd wide() { return output("wide"); }
        public Cmd nameOnly() { return output("name"); }

        public Cmd label(String selector) {
            return flag("-l", selector);
        }

        public Cmd fieldSelector(String selector) {
            return flag("--field-selector", selector);
        }

        public Cmd file(Path path) {
            return flag("-f", path == null ? null : path.toAbsolutePath().toString());
        }

        public Cmd file(String path) {
            return flag("-f", path);
        }

        public Cmd recursive() { return flag("-R"); }

        public Cmd dryRun(String strategy) {
            return flagEq("--dry-run", strategy == null ? "client" : strategy);
        }

        public Cmd clientDryRun() { return dryRun("client"); }
        public Cmd serverDryRun() { return dryRun("server"); }

        public Cmd force() { return flag("--force"); }
        public Cmd gracePeriod(int seconds) { return flag("--grace-period", seconds); }
        public Cmd wait(boolean v) { return flagEq("--wait", v); }
        public Cmd timeout(Duration d) {
            this.timeout = d;
            if (d != null) flagEq("--timeout", Math.max(1, d.toSeconds()) + "s");
            return this;
        }

        public Cmd stdin(String body) {
            this.stdin = body;
            return this;
        }

        public Cmd kustomize(Path dir) {
            return flag("-k", dir == null ? null : dir.toAbsolutePath().toString());
        }

        public List<String> argv() {
            List<String> out = new ArrayList<>(args);
            // inject -n unless all-namespaces / cluster-scoped / already present
            if (!skipDefaultNamespace && explicitNamespace == null) {
                boolean hasNs = false;
                for (int i = 0; i < out.size(); i++) {
                    String a = out.get(i);
                    if ("-n".equals(a) || "--namespace".equals(a) || a.startsWith("--namespace=")) {
                        hasNs = true;
                        break;
                    }
                }
                if (!hasNs) {
                    String ns = kubectl.options.namespace;
                    if (ns != null && !ns.isBlank()) {
                        out.add("-n");
                        out.add(ns);
                    }
                }
            } else if (explicitNamespace != null && !explicitNamespace.isBlank()) {
                out.add("-n");
                out.add(explicitNamespace);
            }
            return out;
        }

        public ProcessRunner.CommandResult run() {
            return kubectl.raw(argv(), timeout, stdin);
        }

        public String runOk() {
            String op = args.isEmpty() ? "kubectl" : args.get(0);
            return kubectl.requireOk(op, argv(), timeout, stdin);
        }

        public Map<String, Object> runJsonObject() {
            if (!args.contains("-o") && !args.contains("--output")
                    && args.stream().noneMatch(a -> a.startsWith("-o=") || a.startsWith("--output="))) {
                output("json");
            }
            String out = runOk();
            try {
                return Json.decodeObject(out.trim());
            } catch (IOException e) {
                throw new K8sException("json parse: " + e.getMessage(), e);
            }
        }

        @Override
        public String toString() {
            return "kubectl " + String.join(" ", argv());
        }
    }

    // =========================================================================
    // Basic Commands (Beginner): create / expose / run / set
    // =========================================================================

    /** {@code kubectl create -f <file>}. */
    public String create(Path file) {
        return cmd("create").file(file).noNamespace().runOk();
    }

    public String createStdin(String yaml) {
        return cmd("create").file("-").noNamespace().stdin(yaml).runOk();
    }

    public String createDryRun(Path file, String strategy) {
        return cmd("create").file(file).dryRun(strategy).noNamespace().runOk();
    }

    /**
     * {@code kubectl create <resource>} with extra args, e.g.
     * {@code create("namespace", "ml")} or
     * {@code create("configmap", "cfg", "--from-literal=a=1")}.
     */
    public String createResource(String resourceType, String name, String... extra) {
        Cmd c = cmd("create", resourceType, name).add(extra);
        // namespace create is cluster-scoped
        if ("namespace".equals(resourceType) || "ns".equals(resourceType)
                || "clusterrole".equalsIgnoreCase(resourceType)
                || "clusterrolebinding".equalsIgnoreCase(resourceType)) {
            c.noNamespace();
        }
        return c.runOk();
    }

    public String createNamespace(String name) {
        return createResource("namespace", name);
    }

    public String createConfigMapFromLiteral(String name, String namespace, Map<String, String> literals) {
        Cmd c = cmd("create", "configmap", name).ns(namespace);
        if (literals != null) {
            for (Map.Entry<String, String> e : literals.entrySet()) {
                c.flag("--from-literal", e.getKey() + "=" + e.getValue());
            }
        }
        return c.runOk();
    }

    public String createConfigMapFromFile(String name, String namespace, Path file) {
        return cmd("create", "configmap", name)
                .ns(namespace)
                .flag("--from-file", file.toAbsolutePath().toString())
                .runOk();
    }

    public String createSecretGeneric(String name, String namespace, Map<String, String> literals) {
        Cmd c = cmd("create", "secret", "generic", name).ns(namespace);
        if (literals != null) {
            for (Map.Entry<String, String> e : literals.entrySet()) {
                c.flag("--from-literal", e.getKey() + "=" + e.getValue());
            }
        }
        return c.runOk();
    }

    public String createSecretDockerRegistry(
            String name, String namespace, String dockerServer, String username, String password, String email) {
        return cmd("create", "secret", "docker-registry", name)
                .ns(namespace)
                .flag("--docker-server", dockerServer)
                .flag("--docker-username", username)
                .flag("--docker-password", password)
                .flag("--docker-email", email)
                .runOk();
    }

    public String createServiceAccount(String name, String namespace) {
        return cmd("create", "serviceaccount", name).ns(namespace).runOk();
    }

    public String createJob(String name, String namespace, String image, String... command) {
        Cmd c = cmd("create", "job", name).ns(namespace).flag("--image", image);
        if (command != null && command.length > 0) {
            c.add("--").add(command);
        }
        return c.runOk();
    }

    public String createCronJob(
            String name, String namespace, String schedule, String image, String... command) {
        Cmd c = cmd("create", "cronjob", name)
                .ns(namespace)
                .flag("--image", image)
                .flag("--schedule", schedule);
        if (command != null && command.length > 0) {
            c.add("--").add(command);
        }
        return c.runOk();
    }

    public String createDeployment(String name, String namespace, String image, int replicas) {
        return cmd("create", "deployment", name)
                .ns(namespace)
                .flag("--image", image)
                .flag("--replicas", replicas)
                .runOk();
    }

    /**
     * {@code kubectl expose deployment/NAME --port=P --target-port=T --type=T}.
     */
    public String expose(
            String typeName, String namespace, int port, Integer targetPort, String serviceType, String name) {
        Cmd c = cmd("expose", typeName).ns(namespace).flag("--port", port);
        if (targetPort != null) c.flag("--target-port", targetPort);
        if (serviceType != null) c.flag("--type", serviceType);
        if (name != null) c.flag("--name", name);
        return c.runOk();
    }

    public String exposeDeployment(String deployment, String namespace, int port, int targetPort, String type) {
        return expose("deployment/" + deployment, namespace, port, targetPort, type, deployment);
    }

    /**
     * {@code kubectl run NAME --image=IMG ...}.
     */
    public String run(
            String name, String image, String namespace, boolean restartNever,
            Map<String, String> env, Integer port, String... command) {
        Cmd c = cmd("run", name).ns(namespace).flag("--image", image);
        if (restartNever) c.flag("--restart", "Never");
        if (port != null) c.flag("--port", port);
        if (env != null) {
            for (Map.Entry<String, String> e : env.entrySet()) {
                c.flag("--env", e.getKey() + "=" + e.getValue());
            }
        }
        if (command != null && command.length > 0) {
            c.add("--").add(command);
        }
        return c.runOk();
    }

    // ---- set ----

    public String setImage(String typeName, String namespace, Map<String, String> containerImages) {
        Cmd c = cmd("set", "image", typeName).ns(namespace);
        if (containerImages != null) {
            for (Map.Entry<String, String> e : containerImages.entrySet()) {
                c.add(e.getKey() + "=" + e.getValue());
            }
        }
        return c.runOk();
    }

    public String setResources(
            String typeName, String namespace, String limits, String requests, String containers) {
        Cmd c = cmd("set", "resources", typeName).ns(namespace);
        if (limits != null) c.flag("--limits", limits);
        if (requests != null) c.flag("--requests", requests);
        if (containers != null) c.flag("-c", containers);
        return c.runOk();
    }

    public String setEnv(String typeName, String namespace, Map<String, String> env) {
        Cmd c = cmd("set", "env", typeName).ns(namespace);
        if (env != null) {
            for (Map.Entry<String, String> e : env.entrySet()) {
                if (e.getValue() == null) c.add(e.getKey() + "-");
                else c.add(e.getKey() + "=" + e.getValue());
            }
        }
        return c.runOk();
    }

    public String setServiceAccount(String typeName, String serviceAccount, String namespace) {
        return cmd("set", "serviceaccount", typeName, serviceAccount).ns(namespace).runOk();
    }

    public String setSubject(String typeName, String namespace, String user, String group, String sa) {
        Cmd c = cmd("set", "subject", typeName).ns(namespace);
        if (user != null) c.flag("--user", user);
        if (group != null) c.flag("--group", group);
        if (sa != null) c.flag("--serviceaccount", sa);
        return c.runOk();
    }

    public String setSelector(String typeName, String namespace, String resourceSelector, boolean all) {
        Cmd c = cmd("set", "selector", typeName, resourceSelector).ns(namespace);
        if (all) c.flag("--all");
        return c.runOk();
    }

    // =========================================================================
    // Basic Commands (Intermediate): explain / get / edit / delete
    // =========================================================================

    public String explain(String resource) {
        return cmd("explain", resource).noNamespace().runOk();
    }

    public String explain(String resource, boolean recursive) {
        Cmd c = cmd("explain", resource).noNamespace();
        if (recursive) c.flag("--recursive");
        return c.runOk();
    }

    public String get(String type, String name, String namespace, boolean json) {
        Cmd c = cmd("get", type);
        if (name != null && !name.isBlank()) c.add(name);
        c.ns(namespace);
        if (json) c.json();
        return c.runOk();
    }

    public String get(String type, String name, String namespace, String output) {
        Cmd c = cmd("get", type);
        if (name != null && !name.isBlank()) c.add(name);
        c.ns(namespace);
        if (output != null) c.output(output);
        return c.runOk();
    }

    public String getAll(String type, String namespace, String labelSelector, String output) {
        Cmd c = cmd("get", type).ns(namespace);
        if (labelSelector != null) c.label(labelSelector);
        if (output != null) c.output(output);
        return c.runOk();
    }

    public String getAllNamespaces(String type, String labelSelector, String output) {
        Cmd c = cmd("get", type).allNamespaces();
        if (labelSelector != null) c.label(labelSelector);
        if (output != null) c.output(output);
        return c.runOk();
    }

    public Map<String, Object> getJson(String type, String name, String namespace) {
        String out = get(type, name, namespace, true);
        try {
            return Json.decodeObject(out.trim());
        } catch (IOException e) {
            throw new K8sException("get json: " + e.getMessage(), e);
        }
    }

    public List<Map<String, Object>> getJsonList(String type, String namespace, String labelSelector) {
        Cmd c = cmd("get", type).ns(namespace).json();
        if (labelSelector != null) c.label(labelSelector);
        String out = c.runOk();
        try {
            Object v = Json.decode(out.trim());
            if (v instanceof Map<?, ?> m) {
                Object items = m.get("items");
                if (items instanceof List<?> list) {
                    List<Map<String, Object>> result = new ArrayList<>();
                    for (Object o : list) {
                        if (o instanceof Map<?, ?> im) {
                            @SuppressWarnings("unchecked")
                            Map<String, Object> row = (Map<String, Object>) im;
                            result.add(row);
                        }
                    }
                    return result;
                }
            }
            return List.of();
        } catch (IOException e) {
            throw new K8sException("get list json: " + e.getMessage(), e);
        }
    }

    /**
     * {@code kubectl edit TYPE NAME}. Interactive — typically needs a TTY; prefer
     * {@link #patch} / {@link #apply} in automation.
     */
    public String edit(String type, String name, String namespace) {
        return cmd("edit", type, name).ns(namespace).runOk();
    }

    public String delete(Path file) {
        return cmd("delete").file(file).flag("--ignore-not-found=true").noNamespace().runOk();
    }

    public String deleteStdin(String yaml) {
        return cmd("delete").file("-").flag("--ignore-not-found=true").noNamespace().stdin(yaml).runOk();
    }

    public String deleteResource(String type, String name, String namespace) {
        return cmd("delete", type, name).ns(namespace).flag("--ignore-not-found=true").runOk();
    }

    public String deleteByLabel(String type, String namespace, String labelSelector) {
        return cmd("delete", type).ns(namespace).label(labelSelector)
                .flag("--ignore-not-found=true").runOk();
    }

    public String deleteAll(String type, String namespace, boolean force, Integer gracePeriod) {
        Cmd c = cmd("delete", type, "--all").ns(namespace).flag("--ignore-not-found=true");
        if (force) c.force();
        if (gracePeriod != null) c.gracePeriod(gracePeriod);
        return c.runOk();
    }

    public String deleteWithOptions(
            String type, String name, String namespace, boolean force, Integer gracePeriod, boolean wait) {
        Cmd c = cmd("delete", type, name).ns(namespace).flag("--ignore-not-found=true");
        if (force) c.force();
        if (gracePeriod != null) c.gracePeriod(gracePeriod);
        c.wait(wait);
        return c.runOk();
    }

    // =========================================================================
    // Deploy Commands: rollout / scale / autoscale
    // =========================================================================

    public RolloutCmd rollout() {
        return new RolloutCmd(this);
    }

    public static final class RolloutCmd {
        private final Kubectl k;
        RolloutCmd(Kubectl k) { this.k = k; }

        public String status(String typeName, String namespace, Duration timeout) {
            return k.rolloutStatus(typeName, namespace, timeout);
        }

        public String history(String typeName, String namespace) {
            return k.cmd("rollout", "history", typeName).ns(namespace).runOk();
        }

        public String history(String typeName, String namespace, Integer revision) {
            Cmd c = k.cmd("rollout", "history", typeName).ns(namespace);
            if (revision != null) c.flag("--revision", revision);
            return c.runOk();
        }

        public String undo(String typeName, String namespace) {
            return k.rolloutUndo(typeName, namespace);
        }

        public String undo(String typeName, String namespace, Integer toRevision) {
            Cmd c = k.cmd("rollout", "undo", typeName).ns(namespace);
            if (toRevision != null) c.flag("--to-revision", toRevision);
            return c.runOk();
        }

        public String restart(String typeName, String namespace) {
            return k.rolloutRestart(typeName, namespace);
        }

        public String pause(String typeName, String namespace) {
            return k.cmd("rollout", "pause", typeName).ns(namespace).runOk();
        }

        public String resume(String typeName, String namespace) {
            return k.cmd("rollout", "resume", typeName).ns(namespace).runOk();
        }
    }

    public String rolloutStatus(String typeName, String namespace, Duration timeout) {
        Cmd c = cmd("rollout", "status", typeName).ns(namespace);
        Duration t = timeout == null ? options.timeout : timeout;
        c.timeout(t);
        return requireOk("rollout_status", c.argv(), t.plusSeconds(30), null);
    }

    public String rolloutUndo(String typeName, String namespace) {
        return cmd("rollout", "undo", typeName).ns(namespace).runOk();
    }

    public String rolloutRestart(String typeName, String namespace) {
        return cmd("rollout", "restart", typeName).ns(namespace).runOk();
    }

    public String rolloutHistory(String typeName, String namespace) {
        return cmd("rollout", "history", typeName).ns(namespace).runOk();
    }

    public String rolloutPause(String typeName, String namespace) {
        return cmd("rollout", "pause", typeName).ns(namespace).runOk();
    }

    public String rolloutResume(String typeName, String namespace) {
        return cmd("rollout", "resume", typeName).ns(namespace).runOk();
    }

    public void scale(String typeName, int replicas, String namespace) {
        cmd("scale", typeName).ns(namespace).flagEq("--replicas", Math.max(0, replicas)).runOk();
    }

    public void scaleDeployment(String name, int replicas, String namespace) {
        scale("deployment/" + name, replicas, namespace);
    }

    public void scaleWithCurrent(
            String typeName, int replicas, String namespace, Integer currentReplicas) {
        Cmd c = cmd("scale", typeName).ns(namespace).flagEq("--replicas", Math.max(0, replicas));
        if (currentReplicas != null) c.flagEq("--current-replicas", currentReplicas);
        c.runOk();
    }

    /**
     * {@code kubectl autoscale deployment/NAME --min= --max= [--cpu-percent=]}.
     */
    public String autoscale(
            String typeName, String namespace, int min, int max, Integer cpuPercent, String hpaName) {
        Cmd c = cmd("autoscale", typeName).ns(namespace)
                .flag("--min", min)
                .flag("--max", max);
        if (cpuPercent != null) c.flag("--cpu-percent", cpuPercent);
        if (hpaName != null) c.flag("--name", hpaName);
        return c.runOk();
    }

    public String autoscaleDeployment(String name, String namespace, int min, int max, int cpuPercent) {
        return autoscale("deployment/" + name, namespace, min, max, cpuPercent, name);
    }

    // =========================================================================
    // Cluster Management: certificate / cluster-info / top / cordon / drain / taint
    // =========================================================================

    public String certificateApprove(String name) {
        return cmd("certificate", "approve", name).noNamespace().runOk();
    }

    public String certificateDeny(String name) {
        return cmd("certificate", "deny", name).noNamespace().runOk();
    }

    public String clusterInfo() {
        return cmd("cluster-info").noNamespace().runOk();
    }

    public String clusterInfoDump(Path outputDirectory) {
        Cmd c = cmd("cluster-info", "dump").noNamespace();
        if (outputDirectory != null) {
            c.flag("--output-directory", outputDirectory.toAbsolutePath().toString());
        }
        return c.runOk();
    }

    public String topNodes() {
        return cmd("top", "nodes").noNamespace().runOk();
    }

    public String topPods(String namespace, boolean allContainers) {
        Cmd c = cmd("top", "pods").ns(namespace);
        if (allContainers) c.flag("--containers");
        return c.runOk();
    }

    public String topPod(String pod, String namespace, boolean allContainers) {
        Cmd c = cmd("top", "pod", pod).ns(namespace);
        if (allContainers) c.flag("--containers");
        return c.runOk();
    }

    public String cordon(String node) {
        return cmd("cordon", node).noNamespace().runOk();
    }

    public String uncordon(String node) {
        return cmd("uncordon", node).noNamespace().runOk();
    }

    /**
     * {@code kubectl drain NODE [--ignore-daemonsets] [--delete-emptydir-data] [--force]}.
     */
    public String drain(
            String node, boolean ignoreDaemonSets, boolean deleteEmptyDirData,
            boolean force, Integer gracePeriod, Duration timeout) {
        Cmd c = cmd("drain", node).noNamespace();
        if (ignoreDaemonSets) c.flag("--ignore-daemonsets");
        if (deleteEmptyDirData) c.flag("--delete-emptydir-data");
        if (force) c.force();
        if (gracePeriod != null) c.gracePeriod(gracePeriod);
        if (timeout != null) c.timeout(timeout);
        return c.runOk();
    }

    public String taint(String node, String key, String value, String effect) {
        // effect: NoSchedule | PreferNoSchedule | NoExecute
        String spec = key + (value == null ? "" : "=" + value) + ":" + effect;
        return cmd("taint", "nodes", node, spec).noNamespace().runOk();
    }

    public String taintRemove(String node, String key, String effect) {
        String spec = key + (effect == null ? "-" : ":" + effect + "-");
        return cmd("taint", "nodes", node, spec).noNamespace().runOk();
    }

    // =========================================================================
    // Troubleshooting: describe / logs / attach / exec / port-forward / proxy /
    //                  cp / auth / debug / events
    // =========================================================================

    public String describe(String type, String name, String namespace) {
        return cmd("describe", type, name).ns(namespace).runOk();
    }

    public String describeByLabel(String type, String namespace, String labelSelector) {
        return cmd("describe", type).ns(namespace).label(labelSelector).runOk();
    }

    public String logs(String pod, String namespace, String container, Integer tail, boolean previous) {
        Cmd c = cmd("logs", pod).ns(namespace);
        if (container != null) c.flag("-c", container);
        if (tail != null) c.flag("--tail", tail);
        if (previous) c.flag("--previous");
        ProcessRunner.CommandResult r = raw(c.argv());
        if (!r.ok()) throw K8sException.ofExit("logs", r.exitCode(), r.output());
        return r.output();
    }

    public String logs(String pod, String namespace, LogsOptions opts) {
        LogsOptions o = opts == null ? LogsOptions.defaults() : opts;
        Cmd c = cmd("logs", pod).ns(namespace);
        if (o.container != null) c.flag("-c", o.container);
        if (o.tail != null) c.flag("--tail", o.tail);
        if (o.previous) c.flag("--previous");
        if (o.timestamps) c.flag("--timestamps");
        if (o.since != null) c.flag("--since", o.since);
        if (o.sinceTime != null) c.flag("--since-time", o.sinceTime);
        if (o.limitBytes != null) c.flag("--limit-bytes", o.limitBytes);
        if (o.allContainers) c.flag("--all-containers=true");
        if (o.prefix) c.flag("--prefix=true");
        if (o.selector != null) c.label(o.selector);
        ProcessRunner.CommandResult r = raw(c.argv(), o.timeout == null ? options.timeout : o.timeout, null);
        if (!r.ok()) throw K8sException.ofExit("logs", r.exitCode(), r.output());
        return r.output();
    }

    /** Follow logs in background — caller reads {@link Process#getInputStream()}. */
    public Process logsFollow(String pod, String namespace, String container, Integer tail) throws IOException {
        List<String> cmdLine = new ArrayList<>();
        cmdLine.add(options.kubectlBin);
        injectGlobalFlags(cmdLine);
        cmdLine.add("logs");
        cmdLine.add("-f");
        cmdLine.add(pod);
        String ns = namespace == null ? options.namespace : namespace;
        if (ns != null && !ns.isBlank()) {
            cmdLine.add("-n");
            cmdLine.add(ns);
        }
        if (container != null) {
            cmdLine.add("-c");
            cmdLine.add(container);
        }
        if (tail != null) {
            cmdLine.add("--tail");
            cmdLine.add(String.valueOf(tail));
        }
        ProcessBuilder pb = new ProcessBuilder(cmdLine);
        pb.redirectErrorStream(true);
        if (!options.extraEnv.isEmpty()) pb.environment().putAll(options.extraEnv);
        return pb.start();
    }

    public static final class LogsOptions {
        public String container;
        public Integer tail;
        public boolean previous;
        public boolean timestamps;
        public String since;       // e.g. "1h"
        public String sinceTime;   // RFC3339
        public Integer limitBytes;
        public boolean allContainers;
        public boolean prefix;
        public String selector;
        public Duration timeout;

        public static LogsOptions defaults() { return new LogsOptions(); }
        public LogsOptions container(String v) { this.container = v; return this; }
        public LogsOptions tail(int v) { this.tail = v; return this; }
        public LogsOptions previous(boolean v) { this.previous = v; return this; }
        public LogsOptions timestamps(boolean v) { this.timestamps = v; return this; }
        public LogsOptions since(String v) { this.since = v; return this; }
        public LogsOptions sinceTime(String v) { this.sinceTime = v; return this; }
        public LogsOptions limitBytes(int v) { this.limitBytes = v; return this; }
        public LogsOptions allContainers(boolean v) { this.allContainers = v; return this; }
        public LogsOptions prefix(boolean v) { this.prefix = v; return this; }
        public LogsOptions selector(String v) { this.selector = v; return this; }
        public LogsOptions timeout(Duration v) { this.timeout = v; return this; }
    }

    /**
     * {@code kubectl attach POD [-c CONTAINER] [-i] [-t]}. Long-running; prefer Process API.
     */
    public Process attach(String pod, String namespace, String container, boolean stdin, boolean tty)
            throws IOException {
        List<String> cmdLine = new ArrayList<>();
        cmdLine.add(options.kubectlBin);
        injectGlobalFlags(cmdLine);
        cmdLine.add("attach");
        cmdLine.add(pod);
        String ns = namespace == null ? options.namespace : namespace;
        if (ns != null && !ns.isBlank()) {
            cmdLine.add("-n");
            cmdLine.add(ns);
        }
        if (container != null) {
            cmdLine.add("-c");
            cmdLine.add(container);
        }
        if (stdin) cmdLine.add("-i");
        if (tty) cmdLine.add("-t");
        ProcessBuilder pb = new ProcessBuilder(cmdLine);
        if (!options.extraEnv.isEmpty()) pb.environment().putAll(options.extraEnv);
        return pb.start();
    }

    public String exec(String pod, String namespace, List<String> command, boolean tty) {
        Objects.requireNonNull(command, "command");
        Cmd c = cmd("exec");
        if (tty) c.flag("-t");
        c.add(pod).ns(namespace).add("--").add(command);
        return c.runOk();
    }

    public String exec(String pod, String namespace, String container, List<String> command,
                       boolean stdin, boolean tty) {
        Objects.requireNonNull(command, "command");
        Cmd c = cmd("exec");
        if (stdin) c.flag("-i");
        if (tty) c.flag("-t");
        c.add(pod).ns(namespace);
        if (container != null) c.flag("-c", container);
        c.add("--").add(command);
        return c.runOk();
    }

    /**
     * Start {@code kubectl port-forward} in background. Caller must destroy the process.
     */
    public Process portForward(String typeName, String namespace, int localPort, int remotePort)
            throws IOException {
        return portForward(typeName, namespace, List.of(localPort + ":" + remotePort), null);
    }

    public Process portForward(
            String typeName, String namespace, List<String> portMappings, String address)
            throws IOException {
        List<String> cmdLine = new ArrayList<>();
        cmdLine.add(options.kubectlBin);
        injectGlobalFlags(cmdLine);
        cmdLine.add("port-forward");
        String ns = namespace == null ? options.namespace : namespace;
        if (ns != null && !ns.isBlank()) {
            cmdLine.add("-n");
            cmdLine.add(ns);
        }
        if (address != null) {
            cmdLine.add("--address");
            cmdLine.add(address);
        }
        cmdLine.add(typeName);
        if (portMappings != null) cmdLine.addAll(portMappings);
        ProcessBuilder pb = new ProcessBuilder(cmdLine);
        pb.redirectErrorStream(true);
        if (!options.extraEnv.isEmpty()) pb.environment().putAll(options.extraEnv);
        return pb.start();
    }

    /**
     * {@code kubectl proxy [--port=0] [--www=] [--www-prefix=] [--api-prefix=]}.
     * Returns the background process.
     */
    public Process proxy(Integer port, String acceptHosts, String www, String wwwPrefix, String apiPrefix)
            throws IOException {
        List<String> cmdLine = new ArrayList<>();
        cmdLine.add(options.kubectlBin);
        injectGlobalFlags(cmdLine);
        cmdLine.add("proxy");
        if (port != null) {
            cmdLine.add("--port");
            cmdLine.add(String.valueOf(port));
        }
        if (acceptHosts != null) {
            cmdLine.add("--accept-hosts");
            cmdLine.add(acceptHosts);
        }
        if (www != null) {
            cmdLine.add("--www");
            cmdLine.add(www);
        }
        if (wwwPrefix != null) {
            cmdLine.add("--www-prefix");
            cmdLine.add(wwwPrefix);
        }
        if (apiPrefix != null) {
            cmdLine.add("--api-prefix");
            cmdLine.add(apiPrefix);
        }
        ProcessBuilder pb = new ProcessBuilder(cmdLine);
        pb.redirectErrorStream(true);
        if (!options.extraEnv.isEmpty()) pb.environment().putAll(options.extraEnv);
        return pb.start();
    }

    public Process proxy(int port) throws IOException {
        return proxy(port, null, null, null, null);
    }

    /** {@code kubectl cp <src> <dst>} — either side may be {@code pod:path} or local path. */
    public String cp(String src, String dst, String namespace, String container) {
        Cmd c = cmd("cp", src, dst).ns(namespace);
        if (container != null) c.flag("-c", container);
        return c.runOk();
    }

    public String cpToPod(Path local, String pod, String remotePath, String namespace, String container) {
        return cp(local.toAbsolutePath().toString(), pod + ":" + remotePath, namespace, container);
    }

    public String cpFromPod(String pod, String remotePath, Path local, String namespace, String container) {
        return cp(pod + ":" + remotePath, local.toAbsolutePath().toString(), namespace, container);
    }

    // ---- auth ----

    public String authCanI(String verb, String resource, String namespace, boolean allNamespaces) {
        Cmd c = cmd("auth", "can-i", verb, resource);
        if (allNamespaces) c.allNamespaces();
        else c.ns(namespace);
        return c.runOk();
    }

    public String authCanIAs(String verb, String resource, String namespace, String asUser, List<String> asGroups) {
        Cmd c = cmd("auth", "can-i", verb, resource).ns(namespace);
        if (asUser != null) c.flag("--as", asUser);
        if (asGroups != null) {
            for (String g : asGroups) c.flag("--as-group", g);
        }
        return c.runOk();
    }

    public String authReconcile(Path file) {
        return cmd("auth", "reconcile").file(file).noNamespace().runOk();
    }

    public String authWhoAmI() {
        // kubectl auth whoami (1.26+)
        return cmd("auth", "whoami").noNamespace().runOk();
    }

    // ---- debug ----

    /**
     * {@code kubectl debug} — ephemeral container / node copy debugging (1.25+).
     */
    public String debugPod(String pod, String namespace, String image, boolean copy, String container) {
        Cmd c = cmd("debug", pod).ns(namespace);
        if (image != null) c.flag("--image", image);
        if (copy) c.flag("--copy-to", pod + "-debug");
        if (container != null) c.flag("--target", container);
        return c.runOk();
    }

    public String debugNode(String node, String image) {
        Cmd c = cmd("debug", "node/" + node).noNamespace();
        if (image != null) c.flag("--image", image);
        return c.runOk();
    }

    // ---- events ----

    public String events(String namespace, boolean forAllNamespaces, String output) {
        Cmd c = cmd("events");
        if (forAllNamespaces) c.allNamespaces();
        else c.ns(namespace);
        if (output != null) c.output(output);
        return c.runOk();
    }

    public String eventsFor(String namespace, String forResource) {
        return cmd("events").ns(namespace).flag("--for", forResource).runOk();
    }

    // =========================================================================
    // Advanced: diff / apply / patch / replace / wait / kustomize
    // =========================================================================

    public String diff(Path file) {
        return cmd("diff").file(file).noNamespace().runOk();
    }

    public String diffStdin(String yaml) {
        return cmd("diff").file("-").noNamespace().stdin(yaml).runOk();
    }

    public String apply(Path file) {
        return cmd("apply").file(file).noNamespace().runOk();
    }

    public String applyStdin(String yaml) {
        return cmd("apply").file("-").noNamespace().stdin(yaml).runOk();
    }

    public String applyDryRun(Path file, String strategy) {
        return cmd("apply").file(file).dryRun(strategy).noNamespace().runOk();
    }

    public String applyWithOptions(Path file, ApplyOptions opts) {
        ApplyOptions o = opts == null ? ApplyOptions.defaults() : opts;
        Cmd c = cmd("apply").file(file).noNamespace();
        if (o.recursive) c.recursive();
        if (o.forceConflicts) c.flag("--force-conflicts");
        if (o.serverSide) c.flag("--server-side");
        if (o.fieldManager != null) c.flag("--field-manager", o.fieldManager);
        if (o.dryRun != null) c.dryRun(o.dryRun);
        if (o.prune) {
            c.flag("--prune");
            if (o.pruneAllowlist != null) {
                for (String p : o.pruneAllowlist) c.flag("--prune-allowlist", p);
            }
        }
        if (o.wait) c.flag("--wait=true");
        if (o.timeout != null) c.timeout(o.timeout);
        return c.runOk();
    }

    public String applyKustomize(Path dir) {
        return cmd("apply").kustomize(dir).noNamespace().runOk();
    }

    public static final class ApplyOptions {
        public boolean recursive;
        public boolean serverSide;
        public boolean forceConflicts;
        public String fieldManager;
        public String dryRun;
        public boolean prune;
        public List<String> pruneAllowlist;
        public boolean wait;
        public Duration timeout;

        public static ApplyOptions defaults() { return new ApplyOptions(); }
        public ApplyOptions recursive(boolean v) { this.recursive = v; return this; }
        public ApplyOptions serverSide(boolean v) { this.serverSide = v; return this; }
        public ApplyOptions forceConflicts(boolean v) { this.forceConflicts = v; return this; }
        public ApplyOptions fieldManager(String v) { this.fieldManager = v; return this; }
        public ApplyOptions dryRun(String v) { this.dryRun = v; return this; }
        public ApplyOptions prune(boolean v) { this.prune = v; return this; }
        public ApplyOptions pruneAllowlist(List<String> v) { this.pruneAllowlist = v; return this; }
        public ApplyOptions wait(boolean v) { this.wait = v; return this; }
        public ApplyOptions timeout(Duration v) { this.timeout = v; return this; }
    }

    /**
     * {@code kubectl patch TYPE NAME --type=TYPE -p PATCH}.
     *
     * @param patchType {@code strategic} | {@code merge} | {@code json}
     */
    public String patch(String type, String name, String namespace, String patchType, String patchBody) {
        return cmd("patch", type, name).ns(namespace)
                .flag("--type", patchType == null ? "strategic" : patchType)
                .flag("-p", patchBody)
                .runOk();
    }

    public String patchFile(String type, String name, String namespace, String patchType, Path patchFile) {
        return cmd("patch", type, name).ns(namespace)
                .flag("--type", patchType == null ? "strategic" : patchType)
                .flag("--patch-file", patchFile.toAbsolutePath().toString())
                .runOk();
    }

    public String patchJson(String type, String name, String namespace, Object patchBody) {
        String body = patchBody instanceof String s ? s : Json.encode(patchBody);
        return patch(type, name, namespace, "json", body);
    }

    public String patchMerge(String type, String name, String namespace, Object patchBody) {
        String body = patchBody instanceof String s ? s : Json.encode(patchBody);
        return patch(type, name, namespace, "merge", body);
    }

    public String replace(Path file) {
        return cmd("replace").file(file).noNamespace().runOk();
    }

    public String replaceStdin(String yaml) {
        return cmd("replace").file("-").noNamespace().stdin(yaml).runOk();
    }

    public String replaceForce(Path file) {
        return cmd("replace").file(file).force().noNamespace().runOk();
    }

    public String wait(String typeName, String namespace, String forCondition, Duration timeout) {
        Cmd c = cmd("wait", typeName).ns(namespace);
        c.flagEq("--for", forCondition == null ? "condition=Available" : forCondition);
        Duration t = timeout == null ? options.timeout : timeout;
        c.timeout(t);
        return requireOk("wait", c.argv(), t.plusSeconds(30), null);
    }

    public String waitForDelete(String typeName, String namespace, Duration timeout) {
        return wait(typeName, namespace, "delete", timeout);
    }

    public String waitJsonPath(String typeName, String namespace, String jsonPathExpr, Duration timeout) {
        // --for=jsonpath='{...}'=value  mixed; pass full for= string
        return wait(typeName, namespace, "jsonpath=" + jsonPathExpr, timeout);
    }

    public String kustomize(Path dir) {
        return cmd("kustomize", dir.toAbsolutePath().toString()).noNamespace().runOk();
    }

    public String kustomize(String urlOrDir) {
        return cmd("kustomize", urlOrDir).noNamespace().runOk();
    }

    // =========================================================================
    // Settings: label / annotate / completion
    // =========================================================================

    public String annotate(
            String type, String name, String namespace, Map<String, String> annotations, boolean overwrite) {
        Cmd c = cmd("annotate", type, name).ns(namespace);
        if (overwrite) c.flag("--overwrite");
        if (annotations != null) {
            for (Map.Entry<String, String> e : annotations.entrySet()) {
                if (e.getValue() == null) c.add(e.getKey() + "-");
                else c.add(e.getKey() + "=" + e.getValue());
            }
        }
        return c.runOk();
    }

    public String label(
            String type, String name, String namespace, Map<String, String> labels, boolean overwrite) {
        Cmd c = cmd("label", type, name).ns(namespace);
        if (overwrite) c.flag("--overwrite");
        if (labels != null) {
            for (Map.Entry<String, String> e : labels.entrySet()) {
                if (e.getValue() == null) c.add(e.getKey() + "-");
                else c.add(e.getKey() + "=" + e.getValue());
            }
        }
        return c.runOk();
    }

    public String completion(String shell) {
        // bash | zsh | fish | powershell
        return cmd("completion", shell == null ? "bash" : shell).noNamespace().runOk();
    }

    // =========================================================================
    // Other: api-resources / api-versions / config / plugin / version
    // =========================================================================

    public String apiResources() {
        return cmd("api-resources").noNamespace().runOk();
    }

    public String apiResources(boolean namespaced, String apiGroup, String output) {
        Cmd c = cmd("api-resources").noNamespace();
        if (namespaced) c.flag("--namespaced=true");
        else c.flag("--namespaced=false");
        if (apiGroup != null) c.flag("--api-group", apiGroup);
        if (output != null) c.output(output);
        return c.runOk();
    }

    public String apiVersions() {
        return cmd("api-versions").noNamespace().runOk();
    }

    public String version() {
        return version(true, true);
    }

    public String version(boolean client, boolean server) {
        Cmd c = cmd("version").noNamespace();
        // Newer kubectl dropped --short; --client limits to client-only.
        if (client && !server) c.flag("--client");
        // When server=true we let kubectl contact the apiserver (may fail offline).
        if (!client && server) {
            // no dedicated --server-only flag; request full output
        }
        return c.runOk();
    }

    public Map<String, Object> versionJson() {
        String out = cmd("version").noNamespace().output("json").runOk();
        try {
            return Json.decodeObject(out.trim());
        } catch (IOException e) {
            throw new K8sException("version json: " + e.getMessage(), e);
        }
    }

    public String pluginList() {
        return cmd("plugin", "list").noNamespace().runOk();
    }

    // ---- config (kubeconfig manipulation) ----

    public ConfigCmd config() {
        return new ConfigCmd(this);
    }

    public static final class ConfigCmd {
        private final Kubectl k;
        ConfigCmd(Kubectl k) { this.k = k; }

        public String view() {
            return k.cmd("config", "view").noNamespace().runOk();
        }

        public String viewJson() {
            return k.cmd("config", "view").noNamespace().output("json").runOk();
        }

        public String viewRaw() {
            return k.cmd("config", "view", "--raw").noNamespace().runOk();
        }

        public String currentContext() {
            return k.cmd("config", "current-context").noNamespace().runOk().trim();
        }

        public String getContexts() {
            return k.cmd("config", "get-contexts").noNamespace().runOk();
        }

        public String getClusters() {
            return k.cmd("config", "get-clusters").noNamespace().runOk();
        }

        public String getUsers() {
            return k.cmd("config", "get-users").noNamespace().runOk();
        }

        public String useContext(String name) {
            return k.cmd("config", "use-context", name).noNamespace().runOk();
        }

        public String setContext(String name, String cluster, String user, String namespace) {
            Cmd c = k.cmd("config", "set-context", name).noNamespace();
            if (cluster != null) c.flag("--cluster", cluster);
            if (user != null) c.flag("--user", user);
            if (namespace != null) c.flag("--namespace", namespace);
            return c.runOk();
        }

        public String setCluster(String name, String server, Path caFile, Boolean insecure) {
            Cmd c = k.cmd("config", "set-cluster", name).noNamespace();
            if (server != null) c.flag("--server", server);
            if (caFile != null) c.flag("--certificate-authority", caFile.toAbsolutePath().toString());
            if (insecure != null) c.flagEq("--insecure-skip-tls-verify", insecure);
            return c.runOk();
        }

        public String setCredentialsToken(String name, String token) {
            return k.cmd("config", "set-credentials", name)
                    .noNamespace()
                    .flag("--token", token)
                    .runOk();
        }

        public String setCredentialsBasic(String name, String username, String password) {
            return k.cmd("config", "set-credentials", name)
                    .noNamespace()
                    .flag("--username", username)
                    .flag("--password", password)
                    .runOk();
        }

        public String setCredentialsClientCert(String name, Path cert, Path key) {
            return k.cmd("config", "set-credentials", name)
                    .noNamespace()
                    .flag("--client-certificate", cert.toAbsolutePath().toString())
                    .flag("--client-key", key.toAbsolutePath().toString())
                    .runOk();
        }

        public String set(String propertyName, String value) {
            return k.cmd("config", "set", propertyName, value).noNamespace().runOk();
        }

        public String unset(String propertyName) {
            return k.cmd("config", "unset", propertyName).noNamespace().runOk();
        }

        public String deleteCluster(String name) {
            return k.cmd("config", "delete-cluster", name).noNamespace().runOk();
        }

        public String deleteContext(String name) {
            return k.cmd("config", "delete-context", name).noNamespace().runOk();
        }

        public String deleteUser(String name) {
            return k.cmd("config", "delete-user", name).noNamespace().runOk();
        }

        public String renameContext(String oldName, String newName) {
            return k.cmd("config", "rename-context", oldName, newName).noNamespace().runOk();
        }
    }

    // =========================================================================
    // Higher-level helpers (deployment ready counts, scale+wait)
    // =========================================================================

    public int readyReplicas(String deploymentName, String namespace) {
        try {
            Map<String, Object> dep = getJson("deployment", deploymentName, namespace);
            Object status = dep.get("status");
            if (status instanceof Map<?, ?> m) {
                Object ready = m.get("readyReplicas");
                if (ready instanceof Number n) return n.intValue();
                if (ready != null) {
                    try { return Integer.parseInt(String.valueOf(ready)); } catch (Exception ignored) {}
                }
            }
            return 0;
        } catch (K8sException e) {
            return 0;
        }
    }

    public int desiredReplicas(String deploymentName, String namespace) {
        try {
            Map<String, Object> dep = getJson("deployment", deploymentName, namespace);
            Object spec = dep.get("spec");
            if (spec instanceof Map<?, ?> m) {
                Object r = m.get("replicas");
                if (r instanceof Number n) return n.intValue();
                if (r != null) {
                    try { return Integer.parseInt(String.valueOf(r)); } catch (Exception ignored) {}
                }
            }
            Object status = dep.get("status");
            if (status instanceof Map<?, ?> m) {
                Object r = m.get("replicas");
                if (r instanceof Number n) return n.intValue();
            }
            return 0;
        } catch (K8sException e) {
            return 0;
        }
    }

    /**
     * Scale and poll until readyReplicas &gt;= desired or timeout.
     *
     * @return ready replica count observed
     */
    public int scaleAndWait(String deploymentName, int replicas, String namespace, Duration wait) {
        scaleDeployment(deploymentName, replicas, namespace);
        if (replicas <= 0) return 0;
        Duration w = wait == null ? options.timeout : wait;
        long deadline = System.currentTimeMillis() + w.toMillis();
        int ready = 0;
        while (System.currentTimeMillis() < deadline) {
            ready = readyReplicas(deploymentName, namespace);
            if (ready >= replicas) return ready;
            try {
                TimeUnit.MILLISECONDS.sleep(1500);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                break;
            }
        }
        return ready;
    }

    /** Convenience: {@code kubectl get -o name} → list of {@code kind/name}. */
    public List<String> getNames(String type, String namespace, String labelSelector) {
        String out = getAll(type, namespace, labelSelector, "name");
        List<String> names = new ArrayList<>();
        if (out == null) return names;
        for (String line : out.split("\n")) {
            line = line.trim();
            if (!line.isEmpty()) names.add(line);
        }
        return names;
    }
}
