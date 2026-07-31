/**
 * Zero-dependency Docker / Compose adapters for model-service deploy.
 *
 * <p>Design goals:
 * <ul>
 *   <li>No {@code docker-java} / Docker SDK Maven dependency</li>
 *   <li>Full {@code docker} CLI surface via {@link DockerCli}
 *       (run/exec/ps/build/bake/pull/push/images/login/logout/search/version/info,
 *       container/image/network/volume/system/context/builder/plugin/manifest/swarm,
 *       attach/commit/cp/create/diff/events/export/history/import/inspect/kill/load/
 *       logs/pause/port/rename/restart/rm/rmi/save/start/stats/stop/tag/top/unpause/
 *       update/wait) plus fluent {@link DockerCli.Cmd} and global options
 *       ({@code --host/--context/--config/--tls*})</li>
 *   <li>Optional Engine REST over TCP or Unix domain socket</li>
 *   <li>Compose + model-service YAML via {@link org.bytedeco.pytorch.utils.yaml.Yaml}</li>
 * </ul>
 *
 * <pre>{@code
 * try (Docker docker = Docker.connect()) {
 *     docker.ping();
 *     String id = docker.cli().run(DockerModels.RunSpec.builder("nginx:alpine")
 *         .name("demo").publish("8080:80").detach(true).build());
 *     docker.cli().network().create("ml-net");
 *     docker.cli().cmd("system", "df").runOk();
 *     docker.cli().stop(id);
 *     docker.cli().rm(id, true);
 * }
 * }</pre>
 *
 * @see Docker
 * @see DockerCli
 * @see DockerCompose
 * @see ModelServiceDeployer
 * @see <a href="https://docs.docker.com/reference/cli/docker/">docker CLI reference</a>
 * @see <a href="https://docs.docker.com/engine/api/">Docker Engine API</a>
 * @see <a href="https://docs.docker.com/compose/compose-file/">Compose specification</a>
 */
package org.bytedeco.pytorch.deploy.docker;
