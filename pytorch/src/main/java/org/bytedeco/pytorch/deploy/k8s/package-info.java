/**
 * Zero-dependency Kubernetes adapters for model-service deploy.
 *
 * <p>Design goals:
 * <ul>
 *   <li>No {@code client-java} / fabric8 / official SDK Maven dependency</li>
 *   <li>Full {@code kubectl} CLI surface via {@link Kubectl}
 *       (create/expose/run/set, get/edit/delete, rollout/scale/autoscale,
 *       certificate/cluster-info/top/cordon/drain/taint,
 *       describe/logs/attach/exec/port-forward/proxy/cp/auth/debug/events,
 *       diff/apply/patch/replace/wait/kustomize, label/annotate/completion,
 *       api-resources/api-versions/config/plugin/version) plus fluent {@link Kubectl.Cmd}</li>
 *   <li>Optional apiserver REST (JSON) with bearer token / in-cluster SA</li>
 *   <li>Manifest YAML via {@link org.bytedeco.pytorch.utils.yaml.Yaml}</li>
 *   <li>{@link K8sClusterOps} implements
 *       {@link org.bytedeco.pytorch.deploy.serving.deploy.DeploymentController.ClusterOps}</li>
 * </ul>
 *
 * <pre>{@code
 * try (K8s k8s = K8s.connect()) {
 *     k8s.kubectl().apply(Path.of("ranker.yaml"));
 *     k8s.kubectl().rollout().status("deployment/ranker", "default", Duration.ofMinutes(2));
 *     k8s.kubectl().cmd("get", "pods").label("app=ranker").json().runOk();
 * }
 * }</pre>
 *
 * @see K8s
 * @see Kubectl
 * @see K8sClusterOps
 * @see ModelServingManifest
 * @see <a href="https://kubernetes.io/docs/reference/kubectl/">kubectl reference</a>
 * @see <a href="https://kubernetes.io/docs/reference/kubernetes-api/">Kubernetes API</a>
 */
package org.bytedeco.pytorch.deploy.k8s;
