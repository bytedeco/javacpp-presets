package org.bytedeco.pytorch.utils.docker;

import java.util.List;
import java.util.Objects;

/**
 * Facade for zero-dep Docker access: CLI (default) + optional Engine REST.
 *
 * <pre>{@code
 * try (Docker docker = Docker.connect()) {
 *     if (!docker.cliAvailable()) throw new IllegalStateException("docker not on PATH");
 *     docker.ping();
 *     System.out.println(docker.cli().version());
 * }
 * }</pre>
 */
public final class Docker implements AutoCloseable {

    private final DockerOptions options;
    private final DockerCli cli;
    private DockerEngine engine; // lazy
    private DockerCompose compose; // lazy

    private Docker(DockerOptions options) {
        this.options = Objects.requireNonNull(options, "options");
        this.cli = new DockerCli(this.options);
    }

    public static Docker connect() {
        return connect(DockerOptions.defaults());
    }

    public static Docker connect(DockerOptions options) {
        return new Docker(options == null ? DockerOptions.defaults() : options);
    }

    public static Docker connect(String dockerHost) {
        return connect(DockerOptions.builder().fromEnv().host(dockerHost).build());
    }

    public DockerOptions options() {
        return options;
    }

    public DockerCli cli() {
        return cli;
    }

    public boolean cliAvailable() {
        return cli.available();
    }

    /** Lazy Engine REST client (unix socket or tcp). */
    public synchronized DockerEngine engine() {
        if (engine == null) {
            engine = DockerEngine.connect(options);
        }
        return engine;
    }

    public synchronized DockerCompose compose() {
        if (compose == null) {
            compose = DockerCompose.connect(options);
        }
        return compose;
    }

    /**
     * Health check: prefer CLI {@code docker info}, fallback Engine {@code /_ping}.
     */
    public void ping() {
        if (cli.available()) {
            cli.ping();
            return;
        }
        engine().ping();
    }

    public String version() {
        if (cli.available()) return cli.version();
        Object v = engine().version().get("Version");
        return v == null ? "" : String.valueOf(v);
    }

    public List<DockerModels.ContainerInfo> ps(boolean all) {
        return cli.ps(all);
    }

    public String run(DockerModels.RunSpec spec) {
        return cli.run(spec);
    }

    public String runModel(ModelContainerSpec spec) {
        return cli.run(spec.toRunSpec());
    }

    public void stop(String idOrName) {
        cli.stop(idOrName);
    }

    public void rm(String idOrName, boolean force) {
        cli.rm(idOrName, force);
    }

    public void pull(String image) {
        cli.pull(image);
    }

    @Override
    public synchronized void close() {
        if (engine != null) {
            engine.close();
            engine = null;
        }
    }
}
