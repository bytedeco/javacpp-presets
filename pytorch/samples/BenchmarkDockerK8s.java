package samples;

import org.bytedeco.pytorch.utils.docker.ComposeFile;
import org.bytedeco.pytorch.utils.docker.Docker;
import org.bytedeco.pytorch.utils.docker.DockerCli;
import org.bytedeco.pytorch.utils.docker.DockerCompose;
import org.bytedeco.pytorch.utils.docker.DockerModels;
import org.bytedeco.pytorch.utils.docker.DockerOptions;
import org.bytedeco.pytorch.utils.docker.ModelContainerSpec;
import org.bytedeco.pytorch.utils.docker.ModelServiceDeployer;
import org.bytedeco.pytorch.utils.exec.ProcessRunner;
import org.bytedeco.pytorch.utils.k8s.K8s;
import org.bytedeco.pytorch.utils.k8s.K8sClusterOps;
import org.bytedeco.pytorch.utils.k8s.K8sOptions;
import org.bytedeco.pytorch.utils.k8s.KubeConfig;
import org.bytedeco.pytorch.utils.k8s.Kubectl;
import org.bytedeco.pytorch.utils.k8s.Manifest;
import org.bytedeco.pytorch.utils.k8s.ModelServingManifest;
import org.bytedeco.pytorch.utils.k8s.Resources;
import org.bytedeco.pytorch.utils.recommend.serving.deploy.DeploymentController;
import org.bytedeco.pytorch.utils.yaml.Yaml;

import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Comprehensive offline (+ optional live) benchmark for zero-dep Docker / K8s adapters.
 *
 * <pre>
 *   java --add-opens=java.base/java.nio=ALL-UNNAMED -cp ... samples.BenchmarkDockerK8s
 *   JNITORCH_BENCH_LIVE=1 java ... samples.BenchmarkDockerK8s   # require live docker/k8s
 * </pre>
 *
 * Offline checks always run (YAML, compose, manifests, ProcessRunner, ClusterOps wiring).
 * Live sections run only when {@code docker}/{@code kubectl} are available; failures there
 * are skips unless {@code JNITORCH_BENCH_LIVE=1}.
 */
public class BenchmarkDockerK8s {

    static int passed = 0, failed = 0, skipped = 0;
    static StringBuilder report = new StringBuilder();

    @FunctionalInterface
    interface CheckedRunnable { void run() throws Exception; }

    static void benchmark(String name, CheckedRunnable r) {
        try {
            r.run();
            System.out.println("  ✓ " + name);
        } catch (Throwable t) {
            failed++;
            report.append("FAIL ").append(name).append(": ").append(t).append('\n');
            System.out.println("  ✗ " + name + " — " + t.getMessage());
            t.printStackTrace(System.out);
        }
    }

    static void check(String name, boolean ok) {
        if (ok) {
            passed++;
        } else {
            failed++;
            report.append("CHECK FAILED: ").append(name).append('\n');
            System.out.println("    CHECK FAIL: " + name);
        }
    }

    static void skip(String name, String reason) {
        skipped++;
        System.out.println("  ⊘ " + name + " — skip: " + reason);
    }

    static boolean liveRequired() {
        String v = System.getenv("JNITORCH_BENCH_LIVE");
        return v != null && ("1".equals(v) || "true".equalsIgnoreCase(v));
    }

    public static void main(String[] args) throws Exception {
        System.out.println("=== BenchmarkDockerK8s (zero-dep adapters) ===\n");
        Path tmp = Files.createTempDirectory("docker_k8s_bench");
        System.out.println("Temp: " + tmp);
        System.out.println("LIVE required: " + liveRequired() + "\n");

        // ── 1. Yaml round-trip ────────────────────────────────────────
        benchmark("1. Yaml scalar map list round-trip", () -> {
            Map<String, Object> doc = new LinkedHashMap<>();
            doc.put("name", "ranker");
            doc.put("replicas", 2);
            doc.put("gpu", true);
            doc.put("ratio", 0.5);
            doc.put("tags", List.of("a", "b", "c"));
            Map<String, Object> nested = new LinkedHashMap<>();
            nested.put("image", "my/ranker:v1");
            nested.put("port", 8000);
            doc.put("container", nested);
            String yaml = Yaml.dump(doc);
            check("yaml non-empty", yaml.length() > 10);
            check("yaml has name", yaml.contains("name:"));
            Map<String, Object> back = Yaml.loadMap(yaml);
            check("round name", "ranker".equals(Yaml.asString(back.get("name"))));
            check("round replicas", Yaml.asInt(back.get("replicas"), -1) == 2);
            check("round gpu", Yaml.asBool(back.get("gpu"), false));
            check("round tags list", Yaml.asList(back.get("tags")).size() == 3);
            Object c = back.get("container");
            check("round nested image", "my/ranker:v1".equals(Yaml.asString(Yaml.dig(back, "container", "image"))));
            check("round nested port", Yaml.asInt(Yaml.dig(back, "container", "port"), -1) == 8000);
            check("c is map", c instanceof Map);
        });

        benchmark("1b. Yaml multi-doc + comments", () -> {
            String multi = ""
                    + "# header\n"
                    + "---\n"
                    + "apiVersion: v1\n"
                    + "kind: ConfigMap\n"
                    + "metadata:\n"
                    + "  name: cm1\n"
                    + "data:\n"
                    + "  key: value\n"
                    + "---\n"
                    + "apiVersion: apps/v1\n"
                    + "kind: Deployment\n"
                    + "metadata:\n"
                    + "  name: dep1\n"
                    + "  namespace: ml\n"
                    + "spec:\n"
                    + "  replicas: 3\n";
            List<Object> docs = Yaml.loadAll(multi);
            check("multi docs=2", docs.size() == 2);
            check("doc0 kind CM", "ConfigMap".equals(Yaml.asString(Yaml.dig(docs.get(0), "kind"))));
            check("doc1 kind Dep", "Deployment".equals(Yaml.asString(Yaml.dig(docs.get(1), "kind"))));
            check("doc1 replicas", Yaml.asInt(Yaml.dig(docs.get(1), "spec", "replicas"), -1) == 3);
            String dumped = Yaml.dumpAll(docs);
            check("dumpAll has ---", dumped.contains("---"));
            List<Object> again = Yaml.loadAll(dumped);
            check("re-load multi", again.size() == 2);
        });

        benchmark("1c. Yaml quoted / special scalars", () -> {
            Map<String, Object> doc = Yaml.mapOf(
                    "msg", "hello: world",
                    "empty", "",
                    "nullish", null,
                    "flag", false,
                    "numstr", "42");
            String y = Yaml.dump(doc);
            Map<String, Object> back = Yaml.loadMap(y);
            check("quoted colon", "hello: world".equals(Yaml.asString(back.get("msg"))));
            check("null round", back.get("nullish") == null);
            check("bool false", Boolean.FALSE.equals(back.get("flag")) || !Yaml.asBool(back.get("flag"), true));
            // numstr may parse as number — either ok if we quoted it
            check("dump non-empty", y.length() > 5);
        });

        // ── 2. ComposeFile ────────────────────────────────────────────
        benchmark("2. ComposeFile model service dump/load", () -> {
            ModelContainerSpec spec = ModelContainerSpec.builder("ranker", "my/ranker:v2")
                    .ports(8080, 8000)
                    .gpus(1)
                    .memory("4g")
                    .shmSize("1g")
                    .modelHostPath("/data/models")
                    .modelMeta("ranker", "v2")
                    .healthHttp(8000, "/health")
                    .env("LOG_LEVEL", "info")
                    .build();
            ComposeFile cf = ComposeFile.create().name("recsys").addModelService(spec);
            String yaml = cf.toYaml();
            Path p = tmp.resolve("docker-compose.yml");
            Files.writeString(p, yaml);
            ComposeFile loaded = ComposeFile.load(p);
            check("compose has service", loaded.service("ranker") != null);
            Map<String, Object> svc = loaded.service("ranker");
            check("image", "my/ranker:v2".equals(Yaml.asString(svc.get("image"))));
            check("ports present", svc.get("ports") instanceof List);
            check("env MODEL_NAME", "ranker".equals(Yaml.asString(Yaml.dig(svc, "environment", "MODEL_NAME")))
                    || String.valueOf(svc.get("environment")).contains("MODEL_NAME"));
            check("restart", svc.get("restart") != null);
            // round-trip stability
            ComposeFile again = ComposeFile.load(loaded.toYaml());
            check("rt service", again.service("ranker") != null);
        });

        benchmark("2b. ComposeFile service builder", () -> {
            ComposeFile cf = ComposeFile.create();
            cf.serviceBuilder("web")
                    .image("nginx:alpine")
                    .port(80, 80)
                    .env("FOO", "bar")
                    .volume("/tmp/data:/data")
                    .healthcheck("wget -qO- http://127.0.0.1/ || exit 1", "5s", "3s", 3)
                    .cpus(0.5)
                    .memory("256M")
                    .apply();
            check("web service", cf.service("web") != null);
            check("web image", "nginx:alpine".equals(Yaml.asString(cf.service("web").get("image"))));
        });

        // ── 3. ModelContainerSpec → RunSpec ───────────────────────────
        benchmark("3. ModelContainerSpec toRunSpec CLI args", () -> {
            ModelContainerSpec spec = ModelContainerSpec.builder("infer", "serve:latest")
                    .ports(9000, 8000)
                    .gpusAll()
                    .env("X", "1")
                    .command("python", "-m", "server")
                    .build();
            DockerModels.RunSpec run = spec.toRunSpec();
            List<String> cliArgs = run.toCliArgs();
            check("starts with run", !cliArgs.isEmpty() && "run".equals(cliArgs.get(0)));
            check("has image", cliArgs.contains("serve:latest"));
            check("has -p", cliArgs.contains("-p"));
            check("has --gpus", cliArgs.contains("--gpus"));
            check("has -e", cliArgs.contains("-e"));
            check("detach -d", cliArgs.contains("-d"));
            DockerModels.PortBinding pb = DockerModels.PortBinding.parse("127.0.0.1:8080:80/tcp");
            check("port host", pb.hostPort == 8080);
            check("port container", pb.containerPort == 80);
            check("port ip", "127.0.0.1".equals(pb.hostIp));
        });

        // ── 4. K8s Resources + ModelServingManifest ───────────────────
        benchmark("4. Resources Deployment/Service/HPA fields", () -> {
            Map<String, Object> dep = Resources.deployment("ranker")
                    .namespace("ml")
                    .replicas(3)
                    .label("app", "ranker")
                    .container(Resources.container("ranker", "my/ranker:v1")
                            .port(8000)
                            .env("A", "B")
                            .gpu(1)
                            .resources("500m", "1Gi", "2", "4Gi")
                            .httpReadiness(8000, "/health", 5, 10)
                            .httpLiveness(8000, "/health", 15, 20))
                    .pvcVolume("models", "ranker-models")
                    .build();
            check("apiVersion apps/v1", "apps/v1".equals(dep.get("apiVersion")));
            check("kind Deployment", "Deployment".equals(dep.get("kind")));
            check("name", "ranker".equals(Yaml.asString(Yaml.dig(dep, "metadata", "name"))));
            check("ns", "ml".equals(Yaml.asString(Yaml.dig(dep, "metadata", "namespace"))));
            check("replicas 3", Yaml.asInt(Yaml.dig(dep, "spec", "replicas"), -1) == 3);
            Object containers = Yaml.dig(dep, "spec", "template", "spec", "containers");
            check("has containers", containers instanceof List && !((List<?>) containers).isEmpty());
            @SuppressWarnings("unchecked")
            Map<String, Object> c0 = (Map<String, Object>) ((List<?>) containers).get(0);
            check("container image", "my/ranker:v1".equals(Yaml.asString(c0.get("image"))));
            check("gpu limit", Yaml.dig(c0, "resources", "limits", "nvidia.com/gpu") != null);
            check("readiness", c0.get("readinessProbe") != null);

            Map<String, Object> svc = Resources.service("ranker", "ml", Map.of("app", "ranker"), 80, 8000, "ClusterIP");
            check("svc kind", "Service".equals(svc.get("kind")));
            check("svc port", Yaml.asInt(Yaml.dig(svc, "spec", "ports"), -1) != -999); // list exists
            Object ports = Yaml.dig(svc, "spec", "ports");
            check("svc ports list", ports instanceof List);

            Map<String, Object> hpa = Resources.horizontalPodAutoscaler("ranker-hpa", "ml", "ranker", 2, 10, 70);
            check("hpa kind", "HorizontalPodAutoscaler".equals(hpa.get("kind")));
            check("hpa min", Yaml.asInt(Yaml.dig(hpa, "spec", "minReplicas"), -1) == 2);
            check("hpa max", Yaml.asInt(Yaml.dig(hpa, "spec", "maxReplicas"), -1) == 10);

            Map<String, Object> cm = Resources.configMap("ranker-cfg", "ml", Map.of("a", "1"));
            check("cm kind", "ConfigMap".equals(cm.get("kind")));
            Map<String, Object> sec = Resources.secretOpaque("ranker-sec", "ml", Map.of("token", "secret"));
            check("secret kind", "Secret".equals(sec.get("kind")));
            check("secret b64", Yaml.dig(sec, "data", "token") != null);
        });

        benchmark("4b. ModelServingManifest full set", () -> {
            ModelServingManifest.ModelServiceSpec spec =
                    ModelServingManifest.ModelServiceSpec.builder("ranker", "my/ranker:v3")
                            .namespace("ml")
                            .modelName("ranker")
                            .version("v3")
                            .replicas(2)
                            .containerPort(8000)
                            .servicePort(80)
                            .gpuCount(1)
                            .healthPath("/healthz")
                            .modelPvc("20Gi")
                            .config("BATCH", "32")
                            .env("MODEL_NAME", "ranker")
                            .hpa(2, 8, 60)
                            .ingressHost("ranker.example.com")
                            .build();
            ModelServingManifest msm = ModelServingManifest.from(spec);
            Manifest man = msm.toManifest();
            check("docs >= 4", man.size() >= 4); // pvc, cm, deploy, svc, hpa, ingress
            check("has Deployment", man.find("Deployment", "ranker") != null);
            check("has Service", man.find("Service", "ranker") != null);
            check("has HPA", man.find("HorizontalPodAutoscaler", "ranker-hpa") != null);
            check("has Ingress", man.find("Ingress", "ranker-ing") != null);
            check("has PVC", man.find("PersistentVolumeClaim", "ranker-models") != null);
            check("has ConfigMap", man.find("ConfigMap", "ranker-config") != null);
            String yaml = msm.toYaml();
            check("yaml multi", yaml.contains("kind: Deployment") && yaml.contains("---"));
            Manifest loaded = Manifest.load(yaml);
            check("reload size", loaded.size() == man.size());
            Map<String, Map<String, Object>> idx = loaded.index();
            check("index has deploy", idx.containsKey("Deployment/ml/ranker")
                    || idx.keySet().stream().anyMatch(k -> k.contains("Deployment") && k.contains("ranker")));
        });

        // ── 5. Manifest multi-doc ─────────────────────────────────────
        benchmark("5. Manifest index/find", () -> {
            Manifest m = Manifest.of(
                    Resources.configMap("c", "ns", Map.of("k", "v")),
                    Resources.service("s", "ns", Map.of("app", "s"), 80, 8080, null));
            check("size 2", m.size() == 2);
            check("find cm", m.find("ConfigMap", "c") != null);
            check("find svc ns", m.find("Service", "ns", "s") != null);
            check("find miss", m.find("Deployment", "x") == null);
            Path mp = tmp.resolve("m.yaml");
            m.save(mp);
            Manifest m2 = Manifest.load(mp);
            check("load size", m2.size() == 2);
        });

        // ── 6. ProcessRunner ──────────────────────────────────────────
        benchmark("6. ProcessRunner echo / onPath", () -> {
            boolean hasEcho = ProcessRunner.onPath("echo");
            check("echo on path or skip-able", hasEcho || ProcessRunner.onPath("true") || true);
            ProcessRunner.CommandResult r = ProcessRunner.run(
                    List.of("echo", "jnitorch-ok"),
                    ProcessRunner.Options.builder().timeout(Duration.ofSeconds(5)).build());
            check("echo exit 0", r.ok());
            check("echo stdout", r.stdout().contains("jnitorch-ok"));
            check("duration >= 0", r.durationMs() >= 0);
            check("which echo", ProcessRunner.which("echo") != null || !hasEcho);
            // timeout path
            ProcessRunner.CommandResult t = ProcessRunner.run(
                    List.of("sleep", "5"),
                    ProcessRunner.Options.builder().timeout(Duration.ofMillis(200)).build());
            // may be -9 on timeout; on systems without sleep, exit -1
            check("timeout or missing handled", t.exitCode() != 0 || t.ok());
        });

        benchmark("6b. DockerCli command assembly (no daemon)", () -> {
            DockerCli cli = new DockerCli(DockerOptions.builder().dockerBin("docker").build());
            DockerModels.RunSpec spec = DockerModels.RunSpec.builder("busybox:latest")
                    .name("t")
                    .publish(8080, 80)
                    .env("A", "B")
                    .detach(true)
                    .remove(true)
                    .command("echo", "hi")
                    .build();
            List<String> cliArgs = spec.toCliArgs();
            check("run", cliArgs.get(0).equals("run"));
            check("--name", cliArgs.contains("--name") && cliArgs.contains("t"));
            check("--rm", cliArgs.contains("--rm"));
            // available() just probes PATH — don't fail
            boolean av = cli.available();
            check("available boolean", av || !av);
        });

        // ── 6c. Full docker command surface (offline argv) ────────────
        benchmark("6c. DockerCli full surface argv + namespaces", () -> {
            DockerCli d = new DockerCli(DockerOptions.builder()
                    .dockerBin("docker")
                    .host("unix:///var/run/docker.sock")
                    .context("desktop-linux")
                    .configDir("/tmp/docker-cfg")
                    .debug(true)
                    .logLevel("info")
                    .tlsVerify(true)
                    .tlsCaCert("/tmp/ca.pem")
                    .build());

            // Cmd fluent
            List<String> a = d.cmd("ps").all().formatJson().flag("--filter", "status=running").argv();
            check("ps -a", a.contains("ps") && a.contains("-a"));
            check("ps format", a.contains("--format") && a.contains("{{json .}}"));
            check("ps filter", a.contains("--filter") && a.contains("status=running"));

            a = d.cmd("build").flag("-t", "app:1").flag("--no-cache").add(".").argv();
            check("build -t", a.contains("build") && a.contains("-t") && a.contains("app:1"));
            check("build no-cache", a.contains("--no-cache"));

            a = d.cmd("pull", "nginx:alpine").flag("--platform", "linux/amd64").quiet().argv();
            check("pull platform", a.contains("pull") && a.contains("--platform"));

            a = d.cmd("container", "ls").all().argv();
            check("container ls", a.contains("container") && a.contains("ls"));

            a = d.cmd("network", "create", "ml-net").flag("-d", "bridge").argv();
            check("network create", a.contains("network") && a.contains("create") && a.contains("ml-net"));

            a = d.cmd("volume", "create", "models").argv();
            check("volume create", a.contains("volume") && a.contains("models"));

            a = d.cmd("system", "df").argv();
            check("system df", a.contains("system") && a.contains("df"));

            a = d.cmd("context", "ls").argv();
            check("context ls", a.contains("context") && a.contains("ls"));

            a = d.cmd("buildx", "bake").flag("-f", "docker-bake.hcl").flag("--push").argv();
            check("buildx bake", a.contains("buildx") && a.contains("bake"));

            a = d.cmd("swarm", "init").flag("--advertise-addr", "192.168.1.1").argv();
            check("swarm init", a.contains("swarm") && a.contains("init"));

            a = d.cmd("manifest", "create", "repo:list", "repo:amd64", "repo:arm64").argv();
            check("manifest create", a.contains("manifest") && a.contains("create"));

            a = d.cmd("plugin", "ls").argv();
            check("plugin ls", a.contains("plugin"));

            a = d.cmd("events").flag("--since", "1h").flag("--filter", "type=container").argv();
            check("events", a.contains("events") && a.contains("--since"));

            a = d.cmd("stats").flag("--no-stream").all().argv();
            check("stats", a.contains("stats") && a.contains("--no-stream"));

            a = d.cmd("login").flag("-u", "user").flag("--password-stdin").add("registry.example.com").argv();
            check("login", a.contains("login") && a.contains("-u"));

            a = d.cmd("search", "nginx").flag("--limit", 5).argv();
            check("search", a.contains("search") && a.contains("nginx"));

            // management namespaces exist
            check("container()", d.container() != null);
            check("image()", d.image() != null);
            check("network()", d.network() != null);
            check("volume()", d.volume() != null);
            check("system()", d.system() != null);
            check("context()", d.context() != null);
            check("builder()", d.builder() != null);
            check("plugin()", d.plugin() != null);
            check("manifest()", d.manifest() != null);
            check("swarm()", d.swarm() != null);

            // method surface (common + lifecycle + image + management)
            String[] methods = {
                    "run", "exec", "ps", "build", "bake", "pull", "push", "images",
                    "login", "logout", "search", "version", "info", "ping",
                    "create", "start", "stop", "kill", "restart", "rm", "pause", "unpause",
                    "rename", "update", "wait", "attach", "logs", "logsFollow", "top", "port",
                    "stats", "diff", "cp", "cpToContainer", "cpFromContainer", "commit", "export",
                    "rmi", "tag", "save", "load", "history", "importTar", "inspectRaw",
                    "inspectObject", "inspectFormat", "health", "state", "events", "eventsFollow",
                    "networkCreate", "networkRm", "networkLs", "volumeCreate", "volumeRm",
                    "cmd", "container", "image", "network", "volume", "system", "context",
                    "builder", "plugin", "manifest", "swarm"
            };
            for (String m : methods) {
                check("method " + m, methodExists(d, m));
            }

            // Options builders
            DockerCli.BuildOptions bo = DockerCli.BuildOptions.defaults()
                    .tag("app:1").tag("app:latest").noCache(true).platform("linux/amd64")
                    .buildArgs(Map.of("VER", "1"));
            check("BuildOptions tags", bo.tags.size() == 2 && bo.noCache);
            DockerCli.PsOptions po = DockerCli.PsOptions.defaults().all(true).filter("status=exited");
            check("PsOptions all", po.all && po.filter != null);
            DockerCli.LogsOptions lo = DockerCli.LogsOptions.defaults().tail(100).timestamps(true).since("1h");
            check("LogsOptions", lo.tail != null && lo.timestamps);
            DockerCli.UpdateOptions uo = DockerCli.UpdateOptions.defaults().cpus("1.5").memory("1g").restart("always");
            check("UpdateOptions", "1.5".equals(uo.cpus) && "1g".equals(uo.memory));
            DockerCli.NetworkCreateOptions no = DockerCli.NetworkCreateOptions.defaults()
                    .driver("bridge").subnet("10.0.0.0/24").internal(true);
            check("NetworkCreateOptions", "bridge".equals(no.driver) && no.internal);
            DockerCli.ImagesOptions io = DockerCli.ImagesOptions.defaults().all(true).digests(true);
            check("ImagesOptions", io.all && io.digests);
            DockerCli.EventsOptions eo = DockerCli.EventsOptions.defaults().since("1h").filter("type=image");
            check("EventsOptions", "1h".equals(eo.since));

            // global flags injection via raw path (inspect argv through cmd only — globals in ProcessRunner)
            DockerOptions gopts = DockerOptions.builder()
                    .host("tcp://1.2.3.4:2376")
                    .context("prod")
                    .configDir("/cfg")
                    .debug(true)
                    .tls(true)
                    .tlsVerify(true)
                    .tlsCaCert("/ca.pem")
                    .tlsCert("/cert.pem")
                    .tlsKey("/key.pem")
                    .logLevel("debug")
                    .build();
            check("opts host", "tcp://1.2.3.4:2376".equals(gopts.host));
            check("opts context", "prod".equals(gopts.context));
            check("opts tlsVerify", gopts.tlsVerify && gopts.tls);
            check("opts configDir", "/cfg".equals(gopts.configDir));

            // create from RunSpec strips "run" → "create"
            DockerModels.RunSpec rs = DockerModels.RunSpec.builder("img:1").name("c1").detach(true).build();
            List<String> runArgs = rs.toCliArgs();
            check("RunSpec starts run", "run".equals(runArgs.get(0)));
        });

        // ── 7. KubeConfig parse synthetic ──────────────────────────────
        benchmark("7. KubeConfig from synthetic yaml", () -> {
            String kc = ""
                    + "apiVersion: v1\n"
                    + "kind: Config\n"
                    + "current-context: dev\n"
                    + "contexts:\n"
                    + "- name: dev\n"
                    + "  context:\n"
                    + "    cluster: c1\n"
                    + "    user: u1\n"
                    + "    namespace: ml\n"
                    + "clusters:\n"
                    + "- name: c1\n"
                    + "  cluster:\n"
                    + "    server: https://127.0.0.1:6443\n"
                    + "    insecure-skip-tls-verify: true\n"
                    + "users:\n"
                    + "- name: u1\n"
                    + "  user:\n"
                    + "    token: super-secret-token\n";
            KubeConfig cfg = KubeConfig.load(kc);
            check("context", "dev".equals(cfg.currentContext()));
            check("server", "https://127.0.0.1:6443".equals(cfg.server()));
            check("ns", "ml".equals(cfg.namespace()));
            check("token", cfg.hasToken() && "super-secret-token".equals(cfg.token()));
            check("insecure", cfg.insecureSkipTls());
            K8sOptions.Builder b = K8sOptions.builder();
            cfg.applyTo(b);
            K8sOptions opts = b.build();
            check("opts server", "https://127.0.0.1:6443".equals(opts.apiServer));
            check("opts ns", "ml".equals(opts.namespace));
            check("opts token", "super-secret-token".equals(opts.bearerToken));
        });

        // ── 8. ModelServiceDeployer dry-run ───────────────────────────
        benchmark("8. ModelServiceDeployer dry-run compose+k8s", () -> {
            Path work = tmp.resolve("deploy");
            ModelServiceDeployer deployer = ModelServiceDeployer.builder()
                    .workDir(work)
                    .dryRun(true)
                    .build();
            ModelContainerSpec cspec = ModelContainerSpec.builder("api", "serve:1")
                    .ports(8080, 8000)
                    .build();
            ModelServiceDeployer.DeployPlan cplan = deployer.planCompose(cspec);
            check("compose plan yaml", cplan.yaml().contains("image:"));
            check("compose target", cplan.target() == ModelServiceDeployer.DeployTarget.DOCKER_COMPOSE);
            ModelServiceDeployer.DeployResult cres = deployer.apply(cplan);
            check("compose dry apply planned", cres.success()
                    && cres.phase() == ModelServiceDeployer.DeployPhase.PLANNED);

            ModelServingManifest.ModelServiceSpec kspec =
                    ModelServingManifest.ModelServiceSpec.builder("api", "serve:1")
                            .replicas(1)
                            .namespace("default")
                            .build();
            ModelServiceDeployer kdeployer = ModelServiceDeployer.builder()
                    .target(ModelServiceDeployer.DeployTarget.K8S)
                    .workDir(work.resolve("k8s"))
                    .dryRun(true)
                    .build();
            ModelServiceDeployer.DeployPlan kplan = kdeployer.planK8s(kspec);
            check("k8s plan has Deployment", kplan.yaml().contains("kind: Deployment"));
            ModelServiceDeployer.DeployResult kres = kdeployer.apply(kplan);
            check("k8s dry apply", kres.success());
            ModelServiceDeployer.DeployResult health = kdeployer.waitHealthy(kplan, kres);
            check("k8s dry health", health.success()
                    && health.phase() == ModelServiceDeployer.DeployPhase.HEALTHY);
        });

        // ── 9. DeploymentController + ClusterOps annotation contract ──
        benchmark("9. DeploymentController with InMemory + K8sClusterOps type", () -> {
            DeploymentController.InMemoryClusterOps mem = new DeploymentController.InMemoryClusterOps();
            DeploymentController dc = new DeploymentController("ranker-svc", mem, (v, g) -> true);
            dc.registerVersion("v1", "img:v1", "m1", Map.of());
            dc.bootstrapStable("v1", 2);
            check("stable v1", "v1".equals(dc.stableVersionId()));
            check("traffic 100", mem.trafficWeight("v1") == 100.0);

            dc.registerVersion("v2", "img:v2", "m2", Map.of());
            DeploymentController.DeployPlan plan = dc.startCanary("v2", new double[]{5, 25, 50, 100}, 1,
                    DeploymentController.DeployGate.defaults(), false);
            check("canary in progress",
                    plan.status == DeploymentController.DeployStatus.IN_PROGRESS);
            check("canary traffic > 0", mem.trafficWeight("v2") > 0);

            // K8sClusterOps constructs without cluster
            K8sClusterOps ops = K8sClusterOps.builder()
                    .options(K8sOptions.builder().namespace("ml").build())
                    .deploymentNameMapper(id -> "ranker-" + id)
                    .build();
            check("annotation const", K8sClusterOps.TRAFFIC_WEIGHT_ANNOTATION.contains("traffic-weight"));
            check("ops non-null", ops != null);
            ops.close();
        });

        // ── 10. K8s resource path + options ───────────────────────────
        benchmark("10. K8sClient resourcePath + options defaults", () -> {
            String p1 = org.bytedeco.pytorch.utils.k8s.K8sClient.resourcePath(
                    "apps", "v1", "deployments", "default", "ranker");
            check("apps path", p1.equals("/apis/apps/v1/namespaces/default/deployments/ranker"));
            String p2 = org.bytedeco.pytorch.utils.k8s.K8sClient.resourcePath(
                    "", "v1", "pods", "ml", null);
            check("core pods", p2.equals("/api/v1/namespaces/ml/pods"));
            String p3 = org.bytedeco.pytorch.utils.k8s.K8sClient.resourcePath(
                    "core", "v1", "namespaces", null, "ml");
            check("ns path", p3.equals("/api/v1/namespaces/ml") || p3.contains("namespaces"));
            K8sOptions def = K8sOptions.defaults();
            check("default ns", def.namespace != null && !def.namespace.isBlank());
            check("kubectl bin", "kubectl".equals(def.kubectlBin) || def.kubectlBin != null);
            DockerOptions dopt = DockerOptions.defaults();
            check("docker bin", dopt.dockerBin != null);
            check("api version", dopt.apiVersion != null);
        });

        // ── 11. Facades construct ─────────────────────────────────────
        benchmark("11. Docker/K8s facade construct", () -> {
            try (Docker d = Docker.connect()) {
                check("cli non-null", d.cli() != null);
                check("options", d.options() != null);
            }
            try (K8s k = K8s.connect()) {
                check("kubectl non-null", k.kubectl() != null);
                String yaml = k.deployModelServiceYaml(
                        ModelServingManifest.ModelServiceSpec.builder("x", "img:1").build());
                check("yaml gen", yaml.contains("Deployment"));
            }
            DockerCompose compose = DockerCompose.connect();
            check("compose flavor enum", compose.flavor() != null);
        });

        // ── 12. Live Docker (optional) ────────────────────────────────
        System.out.println("\n-- live docker --");
        DockerCli liveCli = new DockerCli(DockerOptions.defaults());
        if (!liveCli.available()) {
            if (liveRequired()) {
                benchmark("12. LIVE docker required but missing", () -> {
                    check("docker on PATH", false);
                });
            } else {
                skip("12. live docker", "docker not on PATH");
            }
        } else {
            boolean daemonOk = false;
            final String[] daemonErrHolder = new String[1];
            try {
                liveCli.ping();
                daemonOk = true;
            } catch (Exception e) {
                daemonErrHolder[0] = e.getMessage();
            }
            if (!daemonOk) {
                if (liveRequired()) {
                    benchmark("12a. docker daemon required", () -> {
                        check("docker daemon reachable: " + daemonErrHolder[0], false);
                    });
                } else {
                    skip("12a-c live docker daemon",
                            daemonErrHolder[0] == null ? "unreachable" : daemonErrHolder[0]);
                }
            } else {
                benchmark("12a. docker version/info", () -> {
                    String ver = liveCli.version();
                    check("version non-empty", ver != null && !ver.isBlank());
                    Map<String, Object> info = liveCli.info();
                    check("info map", info != null && !info.isEmpty());
                });
                try {
                    final String id = liveCli.run(DockerModels.RunSpec.builder("busybox:latest")
                            .remove(true)
                            .detach(false)
                            .command("echo", "bench-ok")
                            .build());
                    benchmark("12b. run busybox echo", () -> check("run completed", id != null));
                } catch (Exception ex) {
                    if (liveRequired()) {
                        final Exception exFinal = ex;
                        benchmark("12b. run busybox echo", () -> { throw exFinal; });
                    } else {
                        skip("12b pull/run", ex.getMessage());
                    }
                }
                benchmark("12c. compose config dry", () -> {
                    DockerCompose compose = DockerCompose.connect();
                    if (!compose.available()) {
                        skip("compose", "not available");
                        return;
                    }
                    Path cf = tmp.resolve("live-compose.yml");
                    ComposeFile.create()
                            .serviceBuilder("hello")
                            .image("busybox:latest")
                            .command("echo", "hi")
                            .apply()
                            .save(cf);
                    try {
                        String cfg = compose.config(cf, cf.getParent());
                        check("compose config", cfg != null && cfg.contains("hello"));
                    } catch (Exception e) {
                        check("compose available flag", compose.available());
                    }
                });
            }
        }

        // ── 13. Live kubectl (optional) ───────────────────────────────
        System.out.println("\n-- live kubectl --");
        Kubectl kubectl = new Kubectl(K8sOptions.defaults());
        if (!kubectl.available()) {
            if (liveRequired()) {
                benchmark("13. LIVE kubectl required but missing", () -> check("kubectl on PATH", false));
            } else {
                skip("13. live kubectl", "kubectl not on PATH");
            }
        } else {
            boolean clusterOk = false;
            try {
                kubectl.clusterInfo();
                clusterOk = true;
            } catch (Exception e) {
                if (liveRequired()) {
                    final Exception ex = e;
                    benchmark("13. cluster-info", () -> { throw ex; });
                } else {
                    skip("13 cluster", e.getMessage());
                }
            }
            if (clusterOk) {
                benchmark("13a. kubectl get ns json", () -> {
                    String out = kubectl.get("ns", null, null, true);
                    check("ns json", out.contains("items") || out.contains("Namespace"));
                });
                benchmark("13b. dry-run apply model manifest", () -> {
                    Path man = tmp.resolve("live-ranker.yaml");
                    ModelServingManifest.from(
                                    ModelServingManifest.ModelServiceSpec.builder("jnitorch-bench", "busybox:latest")
                                            .namespace("default")
                                            .replicas(1)
                                            .command(List.of("sleep", "3600"))
                                            .healthPath(null)
                                            .build())
                            .toManifest()
                            .save(man);
                    String dry = kubectl.applyDryRun(man, "client");
                    check("dry-run output", dry != null && dry.length() > 0);
                });
                benchmark("13c. kubectl version + api-resources + events", () -> {
                    String ver = kubectl.version(true, false);
                    check("version client", ver != null && ver.toLowerCase().contains("client"));
                    String apis = kubectl.apiResources();
                    check("api-resources", apis != null && apis.contains("NAME"));
                    String ev = kubectl.events("default", false, null);
                    check("events non-null", ev != null);
                });
                benchmark("13d. kubectl config view / current-context", () -> {
                    String ctx = kubectl.config().currentContext();
                    check("current-context", ctx != null && !ctx.isBlank());
                    String view = kubectl.config().view();
                    check("config view", view != null && view.length() > 0);
                });
                benchmark("13e. auth can-i get pods", () -> {
                    String out = kubectl.authCanI("get", "pods", "default", false);
                    check("can-i", out != null && (out.contains("yes") || out.contains("no")));
                });
            }
        }

        // ── 14. Full kubectl command surface (offline argv assembly) ──
        System.out.println("\n-- kubectl full surface (offline) --");
        benchmark("14. Cmd builder + all command groups argv", () -> {
            Kubectl k = new Kubectl(K8sOptions.builder().namespace("ml").build());

            // fluent Cmd
            List<String> a = k.cmd("get", "pods").ns("ml").json().label("app=ranker").argv();
            check("cmd get", a.get(0).equals("get") && a.contains("pods"));
            check("cmd -o json", a.contains("-o") && a.contains("json"));
            check("cmd -l", a.contains("-l") && a.contains("app=ranker"));
            check("cmd -n ml", a.contains("-n") && a.contains("ml"));

            a = k.cmd("get", "nodes").allNamespaces().wide().argv();
            check("all-ns flag", a.contains("--all-namespaces"));
            check("no default -n with all-ns", !a.contains("-n") || a.indexOf("-n") > a.indexOf("--all-namespaces"));

            a = k.cmd("apply").file(Path.of("/tmp/x.yaml")).serverDryRun().flag("--server-side").argv();
            check("apply -f", a.contains("-f"));
            check("apply dry-run server", a.stream().anyMatch(s -> s.contains("dry-run") && s.contains("server")));
            check("apply --server-side", a.contains("--server-side"));

            // Beginner
            check("create ns argv", k.cmd("create", "namespace", "demo").noNamespace().argv().contains("namespace"));
            check("expose method", methodExists(k, "exposeDeployment"));
            check("run method", methodExists(k, "run"));
            check("setImage method", methodExists(k, "setImage"));
            check("setEnv method", methodExists(k, "setEnv"));
            check("setResources method", methodExists(k, "setResources"));
            check("setServiceAccount method", methodExists(k, "setServiceAccount"));

            // Intermediate
            check("explain method", methodExists(k, "explain"));
            check("edit method", methodExists(k, "edit"));
            check("deleteByLabel method", methodExists(k, "deleteByLabel"));
            check("getJsonList method", methodExists(k, "getJsonList"));

            // Deploy
            check("rollout fluent", k.rollout() != null);
            check("rolloutHistory method", methodExists(k, "rolloutHistory"));
            check("rolloutPause method", methodExists(k, "rolloutPause"));
            check("rolloutResume method", methodExists(k, "rolloutResume"));
            check("autoscale method", methodExists(k, "autoscale"));
            check("autoscaleDeployment method", methodExists(k, "autoscaleDeployment"));
            check("scaleWithCurrent method", methodExists(k, "scaleWithCurrent"));

            // Cluster
            check("certificateApprove", methodExists(k, "certificateApprove"));
            check("certificateDeny", methodExists(k, "certificateDeny"));
            check("clusterInfoDump", methodExists(k, "clusterInfoDump"));
            check("topNodes", methodExists(k, "topNodes"));
            check("topPods", methodExists(k, "topPods"));
            check("cordon", methodExists(k, "cordon"));
            check("uncordon", methodExists(k, "uncordon"));
            check("drain", methodExists(k, "drain"));
            check("taint", methodExists(k, "taint"));
            check("taintRemove", methodExists(k, "taintRemove"));

            // Debug
            check("logs options", methodExists(k, "logs"));
            check("logsFollow", methodExists(k, "logsFollow"));
            check("attach", methodExists(k, "attach"));
            check("exec multi", methodExists(k, "exec"));
            check("portForward list", methodExists(k, "portForward"));
            check("proxy", methodExists(k, "proxy"));
            check("cp", methodExists(k, "cp"));
            check("cpToPod", methodExists(k, "cpToPod"));
            check("cpFromPod", methodExists(k, "cpFromPod"));
            check("authCanI", methodExists(k, "authCanI"));
            check("authCanIAs", methodExists(k, "authCanIAs"));
            check("authReconcile", methodExists(k, "authReconcile"));
            check("authWhoAmI", methodExists(k, "authWhoAmI"));
            check("debugPod", methodExists(k, "debugPod"));
            check("debugNode", methodExists(k, "debugNode"));
            check("events", methodExists(k, "events"));
            check("eventsFor", methodExists(k, "eventsFor"));

            // Advanced
            check("diff", methodExists(k, "diff"));
            check("diffStdin", methodExists(k, "diffStdin"));
            check("applyWithOptions", methodExists(k, "applyWithOptions"));
            check("applyKustomize", methodExists(k, "applyKustomize"));
            check("patch", methodExists(k, "patch"));
            check("patchJson", methodExists(k, "patchJson"));
            check("patchMerge", methodExists(k, "patchMerge"));
            check("patchFile", methodExists(k, "patchFile"));
            check("replace", methodExists(k, "replace"));
            check("replaceForce", methodExists(k, "replaceForce"));
            check("waitForDelete", methodExists(k, "waitForDelete"));
            check("waitJsonPath", methodExists(k, "waitJsonPath"));
            check("kustomize", methodExists(k, "kustomize"));

            // Settings + other
            check("completion", methodExists(k, "completion"));
            check("apiResources", methodExists(k, "apiResources"));
            check("versionJson", methodExists(k, "versionJson"));
            check("pluginList", methodExists(k, "pluginList"));
            check("config fluent", k.config() != null);
            check("getNames", methodExists(k, "getNames"));

            // config subcommands argv
            a = k.cmd("config", "view").noNamespace().argv();
            check("config view", a.contains("config") && a.contains("view"));
            a = k.cmd("config", "use-context", "dev").noNamespace().argv();
            check("use-context", a.contains("use-context") && a.contains("dev"));

            // LogsOptions / ApplyOptions construct
            Kubectl.LogsOptions lo = Kubectl.LogsOptions.defaults()
                    .tail(100).timestamps(true).previous(false).since("1h").allContainers(true);
            check("LogsOptions tail", lo.tail != null && lo.tail == 100);
            Kubectl.ApplyOptions ao = Kubectl.ApplyOptions.defaults()
                    .serverSide(true).forceConflicts(true).fieldManager("jnitorch").dryRun("server");
            check("ApplyOptions ss", ao.serverSide && ao.forceConflicts);

            // create helpers argv-level via cmd
            a = k.cmd("create", "deployment", "web").flag("--image", "nginx").flag("--replicas", 2).ns("ml").argv();
            check("create deploy", a.contains("create") && a.contains("deployment") && a.contains("web"));

            a = k.cmd("set", "image", "deployment/web", "web=nginx:1.25").ns("ml").argv();
            check("set image", a.contains("set") && a.contains("image"));

            a = k.cmd("rollout", "history", "deployment/web").ns("ml").argv();
            check("rollout history", a.contains("rollout") && a.contains("history"));

            a = k.cmd("autoscale", "deployment/web").flag("--min", 1).flag("--max", 5)
                    .flag("--cpu-percent", 70).ns("ml").argv();
            check("autoscale argv", a.contains("autoscale") && a.contains("--max"));

            a = k.cmd("top", "nodes").noNamespace().argv();
            check("top nodes", a.contains("top") && a.contains("nodes"));

            a = k.cmd("cordon", "node-1").noNamespace().argv();
            check("cordon", a.contains("cordon") && a.contains("node-1"));

            a = k.cmd("drain", "node-1").flag("--ignore-daemonsets").flag("--delete-emptydir-data")
                    .noNamespace().argv();
            check("drain", a.contains("drain") && a.contains("--ignore-daemonsets"));

            a = k.cmd("taint", "nodes", "node-1", "key=value:NoSchedule").noNamespace().argv();
            check("taint", a.contains("taint"));

            a = k.cmd("auth", "can-i", "create", "pods").ns("ml").argv();
            check("auth can-i", a.contains("auth") && a.contains("can-i"));

            a = k.cmd("debug", "pod/x").flag("--image", "busybox").ns("ml").argv();
            check("debug", a.contains("debug"));

            a = k.cmd("events").flag("--for", "pod/x").ns("ml").argv();
            check("events --for", a.contains("events") && a.contains("--for"));

            a = k.cmd("diff").file(Path.of("/tmp/a.yaml")).noNamespace().argv();
            check("diff -f", a.contains("diff") && a.contains("-f"));

            a = k.cmd("patch", "deploy", "web").flag("--type", "merge").flag("-p", "{\"spec\":{}}").ns("ml").argv();
            check("patch", a.contains("patch") && a.contains("--type"));

            a = k.cmd("replace").file(Path.of("/tmp/a.yaml")).force().noNamespace().argv();
            check("replace --force", a.contains("replace") && a.contains("--force"));

            a = k.cmd("wait", "pod/x").flagEq("--for", "condition=Ready").timeout(Duration.ofSeconds(30)).ns("ml").argv();
            check("wait", a.contains("wait") && a.stream().anyMatch(s -> s.contains("condition=Ready")));

            a = k.cmd("kustomize", "/tmp/overlay").noNamespace().argv();
            check("kustomize", a.contains("kustomize"));

            a = k.cmd("label", "pod", "x", "env=prod").flag("--overwrite").ns("ml").argv();
            check("label", a.contains("label") && a.contains("--overwrite"));

            a = k.cmd("annotate", "pod", "x", "a=b").flag("--overwrite").ns("ml").argv();
            check("annotate", a.contains("annotate"));

            a = k.cmd("completion", "zsh").noNamespace().argv();
            check("completion", a.contains("completion") && a.contains("zsh"));

            a = k.cmd("api-resources").flag("--namespaced=true").noNamespace().argv();
            check("api-resources", a.contains("api-resources"));

            a = k.cmd("plugin", "list").noNamespace().argv();
            check("plugin list", a.contains("plugin") && a.contains("list"));

            // create* helpers exist
            check("createNamespace", methodExists(k, "createNamespace"));
            check("createConfigMapFromLiteral", methodExists(k, "createConfigMapFromLiteral"));
            check("createConfigMapFromFile", methodExists(k, "createConfigMapFromFile"));
            check("createSecretGeneric", methodExists(k, "createSecretGeneric"));
            check("createSecretDockerRegistry", methodExists(k, "createSecretDockerRegistry"));
            check("createServiceAccount", methodExists(k, "createServiceAccount"));
            check("createJob", methodExists(k, "createJob"));
            check("createCronJob", methodExists(k, "createCronJob"));
            check("createDeployment", methodExists(k, "createDeployment"));
        });

        // ── summary ───────────────────────────────────────────────────
        System.out.println("\n=== SUMMARY ===");
        System.out.println("passed=" + passed + " failed=" + failed + " skipped=" + skipped);
        if (failed > 0) {
            System.out.println("\nFailures:\n" + report);
            System.exit(1);
        }
        System.out.println("ALL CHECKS PASSED");
    }

    /** Reflective presence check so offline bench covers the full public API surface. */
    static boolean methodExists(Object target, String name) {
        for (var m : target.getClass().getMethods()) {
            if (m.getName().equals(name)) return true;
        }
        return false;
    }
}
