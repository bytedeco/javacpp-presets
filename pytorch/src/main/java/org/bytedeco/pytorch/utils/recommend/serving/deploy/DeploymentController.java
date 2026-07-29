/*
 * Deployment / release strategies for recommendation online services.
 *
 * Covers industry-standard progressive delivery:
 *   - Canary release (金丝雀) — small % traffic to new version, ramp up
 *   - Blue/Green (蓝绿) — two full environments, switch at gateway
 *   - Rolling / surge (滚动) — replace pods batch by batch
 *   - In-place upgrade (原地升级) — same pod, new binary/model (Ali/K8s in-place)
 *   - Rollback (回滚) — instant revert to last known good
 *   - Replica scaling (扩缩容) — HPA-style + predictive for recsys peaks
 *
 * References:
 *   - Kubernetes Deployment / Argo Rollouts / Flagger
 *   - Netflix Spinnaker pipelines
 *   - Google SRE progressive rollouts
 *   - Alibaba / ByteDance / Tencent release platforms for recsys
 *   - Meta "gatekeeper" + canary for ranking models
 */
package org.bytedeco.pytorch.utils.recommend.serving.deploy;

import org.bytedeco.pytorch.utils.recommend.abtest.TrafficSplitter;

import java.time.Instant;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.function.Consumer;

/**
 * Central deployment controller for recommendation serving clusters.
 *
 * <p>This is an in-process state machine encoding the control logic that
 * production systems implement via K8s controllers / Spinnaker / internal
 * release platforms. It does NOT talk to a real cluster — callers plug in
 * {@link ClusterOps} for actual pod mutations.
 */
public final class DeploymentController {

    /** Deployment plan status. */
    public enum DeployStatus {
        PENDING,
        IN_PROGRESS,
        PAUSED,
        SUCCEEDED,
        FAILED,
        ROLLED_BACK
    }

    /** A deployable service version (code + model + config bundle). */
    public static final class ServiceVersion {
        public final String versionId;
        public final String imageOrArtifact;
        public final String modelVersionId;
        public final Map<String, String> config;
        public final Instant createdAt;

        public ServiceVersion(
                String versionId, String imageOrArtifact, String modelVersionId, Map<String, String> config) {
            if (versionId == null || versionId.isEmpty()) {
                throw new IllegalArgumentException("versionId required");
            }
            this.versionId = versionId;
            this.imageOrArtifact = imageOrArtifact != null ? imageOrArtifact : "";
            this.modelVersionId = modelVersionId != null ? modelVersionId : "";
            this.config = config == null
                    ? Collections.emptyMap()
                    : Collections.unmodifiableMap(new LinkedHashMap<>(config));
            this.createdAt = Instant.now();
        }

        @Override
        public String toString() {
            return "ServiceVersion{id=" + versionId + ", model=" + modelVersionId + "}";
        }
    }

    /** Replica set snapshot for one version. */
    static final class ReplicaSet {
        final String versionId;
        int desired;
        int ready;
        int unavailable;

        ReplicaSet(String versionId, int desired) {
            this.versionId = versionId;
            this.desired = desired;
            this.ready = 0;
            this.unavailable = desired;
        }

        synchronized void markReady(int n) {
            ready = Math.min(desired, Math.max(0, n));
            unavailable = Math.max(0, desired - ready);
        }

        synchronized void scale(int newDesired) {
            desired = Math.max(0, newDesired);
            if (ready > desired) ready = desired;
            unavailable = Math.max(0, desired - ready);
        }

        synchronized boolean allReady() {
            return desired > 0 && ready >= desired;
        }

        @Override
        public String toString() {
            return versionId + "(desired=" + desired + ", ready=" + ready + ")";
        }
    }

    /** Abstraction over real cluster operations (K8s, Nomad, internal). */
    public interface ClusterOps {
        /** Scale version to desired replicas; return actual ready count after wait. */
        int scale(String versionId, int desiredReplicas) throws Exception;

        /** Replace N pods of version in-place with new artifact (in-place upgrade). */
        int inplaceRestart(String versionId, int podCount) throws Exception;

        /** Health percent [0,100] for a version (from readiness + custom checks). */
        double healthPercent(String versionId);

        /** Set gateway traffic weight for version (0-100); remaining goes to others. */
        void setTrafficWeight(String versionId, double weightPercent) throws Exception;
    }

    /** Observability hook. */
    public interface MetricsProbe {
        /** Return true if version is healthy enough to continue ramp. */
        boolean passGate(String versionId, DeployGate gate);
    }

    /** Gate criteria between canary stages. */
    public static final class DeployGate {
        public final double maxErrorRate;
        public final double maxP99LatencyMs;
        public final double maxRelativeMetricDrop;
        public final long minObserveSeconds;

        public DeployGate(double maxErrorRate, double maxP99LatencyMs,
                          double maxRelativeMetricDrop, long minObserveSeconds) {
            this.maxErrorRate = maxErrorRate;
            this.maxP99LatencyMs = maxP99LatencyMs;
            this.maxRelativeMetricDrop = maxRelativeMetricDrop;
            this.minObserveSeconds = minObserveSeconds;
        }

        public static DeployGate defaults() {
            return new DeployGate(0.01, 200.0, 0.02, 60L);
        }
    }

    private final String serviceName;
    private final ClusterOps clusterOps;
    private final MetricsProbe metricsProbe;
    private final CopyOnWriteArrayList<Consumer<DeployEvent>> listeners = new CopyOnWriteArrayList<>();
    private final LinkedHashMap<String, ServiceVersion> versions = new LinkedHashMap<>();
    private final LinkedHashMap<String, ReplicaSet> replicaSets = new LinkedHashMap<>();
    private String stableVersionId;
    private String activePlanId;
    private DeployPlan activePlan;

    public DeploymentController(String serviceName, ClusterOps clusterOps, MetricsProbe metricsProbe) {
        this.serviceName = Objects.requireNonNull(serviceName, "serviceName");
        this.clusterOps = Objects.requireNonNull(clusterOps, "clusterOps");
        this.metricsProbe = metricsProbe;
    }

    public void addListener(Consumer<DeployEvent> listener) {
        listeners.add(Objects.requireNonNull(listener));
    }

    public String serviceName() {
        return serviceName;
    }

    public String stableVersionId() {
        return stableVersionId;
    }

    public synchronized ServiceVersion registerVersion(ServiceVersion version) {
        versions.put(version.versionId, version);
        replicaSets.computeIfAbsent(version.versionId, id -> new ReplicaSet(id, 0));
        emit(DeployEvent.info(serviceName, "register_version", version.versionId));
        return version;
    }

    public synchronized ServiceVersion registerVersion(
            String versionId, String artifact, String modelVersionId, Map<String, String> config) {
        return registerVersion(new ServiceVersion(versionId, artifact, modelVersionId, config));
    }

    /**
     * Bootstrap stable production version at full replicas / 100% traffic.
     */
    public synchronized void bootstrapStable(String versionId, int replicas) throws Exception {
        requireVersion(versionId);
        int ready = clusterOps.scale(versionId, replicas);
        ReplicaSet rs = replicaSets.get(versionId);
        rs.scale(replicas);
        rs.markReady(ready);
        clusterOps.setTrafficWeight(versionId, 100.0);
        stableVersionId = versionId;
        emit(DeployEvent.info(serviceName, "bootstrap_stable", versionId + " replicas=" + replicas));
    }

    // ---- canary -------------------------------------------------------------

    /**
     * Start a canary release ramping {@code canaryVersionId} through stages.
     *
     * @param canaryVersionId new version
     * @param stagesPercent   e.g. {@link TrafficSplitter#defaultCanaryStages()}
     * @param canaryReplicas  replicas dedicated to canary (often small)
     * @param gate            promotion gate between stages
     * @param autoPromote     if true, advance stages when gate passes
     */
    public synchronized DeployPlan startCanary(
            String canaryVersionId,
            double[] stagesPercent,
            int canaryReplicas,
            DeployGate gate,
            boolean autoPromote) throws Exception {
        requireVersion(canaryVersionId);
        if (stableVersionId == null) {
            throw new IllegalStateException("no stable version; bootstrapStable first");
        }
        if (stableVersionId.equals(canaryVersionId)) {
            throw new IllegalArgumentException("canary == stable");
        }
        if (activePlan != null && activePlan.status == DeployStatus.IN_PROGRESS) {
            throw new IllegalStateException("another plan in progress: " + activePlan.planId);
        }
        double[] stages = stagesPercent == null || stagesPercent.length == 0
                ? TrafficSplitter.defaultCanaryStages()
                : stagesPercent.clone();
        DeployPlan plan = DeployPlan.canary(
                nextPlanId("canary"), stableVersionId, canaryVersionId, stages, canaryReplicas, gate, autoPromote);
        activePlan = plan;
        activePlanId = plan.planId;

        // Provision canary replicas
        int ready = clusterOps.scale(canaryVersionId, canaryReplicas);
        ReplicaSet rs = replicaSets.get(canaryVersionId);
        rs.scale(canaryReplicas);
        rs.markReady(ready);

        // Set first stage traffic
        advanceCanaryTraffic(plan, 0);
        plan.status = DeployStatus.IN_PROGRESS;
        plan.stageIndex = 0;
        emit(DeployEvent.planStarted(serviceName, plan));
        if (autoPromote) {
            // Caller should loop promoteCanary() on a schedule; we only set stage 0 here.
        }
        return plan;
    }

    /**
     * Evaluate gate and promote canary to next stage (or complete at 100%).
     */
    public synchronized DeployPlan promoteCanary() throws Exception {
        DeployPlan plan = requireActiveCanary();
        DeployGate gate = plan.gate != null ? plan.gate : DeployGate.defaults();
        if (metricsProbe != null && !metricsProbe.passGate(plan.targetVersionId, gate)) {
            plan.lastMessage = "gate_failed";
            emit(DeployEvent.gateFailed(serviceName, plan));
            return plan;
        }
        int next = plan.stageIndex + 1;
        if (next >= plan.stagesPercent.length) {
            // Fully promoted — make canary the new stable.
            clusterOps.setTrafficWeight(plan.targetVersionId, 100.0);
            clusterOps.setTrafficWeight(plan.stableVersionId, 0.0);
            // Scale down old stable optionally left to caller.
            stableVersionId = plan.targetVersionId;
            plan.status = DeployStatus.SUCCEEDED;
            plan.stageIndex = plan.stagesPercent.length - 1;
            plan.lastMessage = "canary_complete";
            emit(DeployEvent.planSucceeded(serviceName, plan));
            activePlan = plan;
            return plan;
        }
        advanceCanaryTraffic(plan, next);
        plan.stageIndex = next;
        plan.lastMessage = "promoted_to_stage_" + next;
        emit(DeployEvent.stagePromoted(serviceName, plan));
        return plan;
    }

    private void advanceCanaryTraffic(DeployPlan plan, int stageIndex) throws Exception {
        double pct = TrafficSplitter.canaryPercent(stageIndex, plan.stagesPercent);
        clusterOps.setTrafficWeight(plan.targetVersionId, pct);
        clusterOps.setTrafficWeight(plan.stableVersionId, 100.0 - pct);
        plan.currentTrafficPercent = pct;
    }

    // ---- blue / green -------------------------------------------------------

    /**
     * Blue/green: provision full green at same replica count, switch 100% when ready + gate.
     */
    public synchronized DeployPlan startBlueGreen(
            String greenVersionId, DeployGate gate, boolean autoSwitch) throws Exception {
        requireVersion(greenVersionId);
        if (stableVersionId == null) {
            throw new IllegalStateException("no stable (blue) version");
        }
        if (activePlan != null && activePlan.status == DeployStatus.IN_PROGRESS) {
            throw new IllegalStateException("another plan in progress");
        }
        ReplicaSet blue = replicaSets.get(stableVersionId);
        int replicas = blue == null ? 1 : Math.max(1, blue.desired);
        DeployPlan plan = DeployPlan.blueGreen(
                nextPlanId("bg"), stableVersionId, greenVersionId, replicas, gate, autoSwitch);
        activePlan = plan;
        activePlanId = plan.planId;

        int ready = clusterOps.scale(greenVersionId, replicas);
        ReplicaSet green = replicaSets.get(greenVersionId);
        green.scale(replicas);
        green.markReady(ready);
        plan.status = DeployStatus.IN_PROGRESS;
        plan.lastMessage = "green_provisioned ready=" + ready;
        emit(DeployEvent.planStarted(serviceName, plan));

        if (autoSwitch) {
            return switchBlueGreen();
        }
        return plan;
    }

    /**
     * Flip traffic from blue to green (100%).
     */
    public synchronized DeployPlan switchBlueGreen() throws Exception {
        DeployPlan plan = requireActive(DeployPlan.Strategy.BLUE_GREEN);
        DeployGate gate = plan.gate != null ? plan.gate : DeployGate.defaults();
        if (metricsProbe != null && !metricsProbe.passGate(plan.targetVersionId, gate)) {
            plan.lastMessage = "gate_failed_before_switch";
            emit(DeployEvent.gateFailed(serviceName, plan));
            return plan;
        }
        double health = clusterOps.healthPercent(plan.targetVersionId);
        if (health < 100.0) {
            plan.lastMessage = "green_not_fully_healthy health=" + health;
            emit(DeployEvent.gateFailed(serviceName, plan));
            return plan;
        }
        clusterOps.setTrafficWeight(plan.targetVersionId, 100.0);
        clusterOps.setTrafficWeight(plan.stableVersionId, 0.0);
        plan.currentTrafficPercent = 100.0;
        stableVersionId = plan.targetVersionId;
        plan.status = DeployStatus.SUCCEEDED;
        plan.lastMessage = "blue_green_switched";
        emit(DeployEvent.planSucceeded(serviceName, plan));
        return plan;
    }

    // ---- rolling ------------------------------------------------------------

    /**
     * Rolling update: replace stable pods in batches of {@code batchSize}
     * with target version (surge-style: scale up target, scale down stable).
     */
    public synchronized DeployPlan startRolling(
            String targetVersionId, int batchSize, DeployGate gate) throws Exception {
        requireVersion(targetVersionId);
        if (stableVersionId == null) {
            throw new IllegalStateException("no stable version");
        }
        if (activePlan != null && activePlan.status == DeployStatus.IN_PROGRESS) {
            throw new IllegalStateException("another plan in progress");
        }
        ReplicaSet stable = replicaSets.get(stableVersionId);
        int total = stable == null ? batchSize : stable.desired;
        DeployPlan plan = DeployPlan.rolling(
                nextPlanId("roll"), stableVersionId, targetVersionId, total, batchSize, gate);
        activePlan = plan;
        activePlanId = plan.planId;
        plan.status = DeployStatus.IN_PROGRESS;
        emit(DeployEvent.planStarted(serviceName, plan));
        return advanceRolling();
    }

    /**
     * Advance one rolling batch.
     */
    public synchronized DeployPlan advanceRolling() throws Exception {
        DeployPlan plan = requireActive(DeployPlan.Strategy.ROLLING);
        int batch = plan.batchSize;
        int already = plan.rolledReplicas;
        int remaining = plan.totalReplicas - already;
        if (remaining <= 0) {
            stableVersionId = plan.targetVersionId;
            plan.status = DeployStatus.SUCCEEDED;
            plan.currentTrafficPercent = 100.0;
            clusterOps.setTrafficWeight(plan.targetVersionId, 100.0);
            plan.lastMessage = "rolling_complete";
            emit(DeployEvent.planSucceeded(serviceName, plan));
            return plan;
        }
        DeployGate gate = plan.gate != null ? plan.gate : DeployGate.defaults();
        if (already > 0 && metricsProbe != null && !metricsProbe.passGate(plan.targetVersionId, gate)) {
            plan.lastMessage = "gate_failed";
            emit(DeployEvent.gateFailed(serviceName, plan));
            return plan;
        }
        int step = Math.min(batch, remaining);
        // Surge: add target replicas, remove stable replicas.
        ReplicaSet targetRs = replicaSets.get(plan.targetVersionId);
        ReplicaSet stableRs = replicaSets.get(plan.stableVersionId);
        int newTargetDesired = targetRs.desired + step;
        int newStableDesired = Math.max(0, stableRs.desired - step);
        int tReady = clusterOps.scale(plan.targetVersionId, newTargetDesired);
        targetRs.scale(newTargetDesired);
        targetRs.markReady(tReady);
        int sReady = clusterOps.scale(plan.stableVersionId, newStableDesired);
        stableRs.scale(newStableDesired);
        stableRs.markReady(sReady);
        // Traffic proportional to ready replicas.
        int totalReady = Math.max(1, targetRs.ready + stableRs.ready);
        double pct = 100.0 * targetRs.ready / totalReady;
        clusterOps.setTrafficWeight(plan.targetVersionId, pct);
        clusterOps.setTrafficWeight(plan.stableVersionId, 100.0 - pct);
        plan.rolledReplicas = already + step;
        plan.currentTrafficPercent = pct;
        plan.lastMessage = "rolled " + plan.rolledReplicas + "/" + plan.totalReplicas;
        emit(DeployEvent.stagePromoted(serviceName, plan));
        if (plan.rolledReplicas >= plan.totalReplicas) {
            stableVersionId = plan.targetVersionId;
            plan.status = DeployStatus.SUCCEEDED;
            clusterOps.setTrafficWeight(plan.targetVersionId, 100.0);
            plan.currentTrafficPercent = 100.0;
            plan.lastMessage = "rolling_complete";
            emit(DeployEvent.planSucceeded(serviceName, plan));
        }
        return plan;
    }

    // ---- in-place -----------------------------------------------------------

    /**
     * In-place upgrade: restart pods of the stable version with new artifact
     * without changing version id traffic mapping. Used for model-file hot
     * swap or minor binary patches (common in Alibaba / ByteDance model push).
     *
     * <p>Note: true atomic in-place requires cluster support; here we model
     * batch in-place restarts with gate checks.
     */
    public synchronized DeployPlan startInPlace(
            String versionId, int batchSize, DeployGate gate) throws Exception {
        requireVersion(versionId);
        ReplicaSet rs = replicaSets.get(versionId);
        int total = rs == null ? 0 : rs.desired;
        if (total <= 0) {
            throw new IllegalStateException("version has no replicas: " + versionId);
        }
        if (activePlan != null && activePlan.status == DeployStatus.IN_PROGRESS) {
            throw new IllegalStateException("another plan in progress");
        }
        DeployPlan plan = DeployPlan.inPlace(
                nextPlanId("inplace"), versionId, total, batchSize, gate);
        activePlan = plan;
        activePlanId = plan.planId;
        plan.status = DeployStatus.IN_PROGRESS;
        emit(DeployEvent.planStarted(serviceName, plan));
        return advanceInPlace();
    }

    public synchronized DeployPlan advanceInPlace() throws Exception {
        DeployPlan plan = requireActive(DeployPlan.Strategy.IN_PLACE);
        int remaining = plan.totalReplicas - plan.rolledReplicas;
        if (remaining <= 0) {
            plan.status = DeployStatus.SUCCEEDED;
            plan.lastMessage = "inplace_complete";
            emit(DeployEvent.planSucceeded(serviceName, plan));
            return plan;
        }
        DeployGate gate = plan.gate != null ? plan.gate : DeployGate.defaults();
        if (plan.rolledReplicas > 0 && metricsProbe != null
                && !metricsProbe.passGate(plan.targetVersionId, gate)) {
            plan.lastMessage = "gate_failed";
            emit(DeployEvent.gateFailed(serviceName, plan));
            return plan;
        }
        int step = Math.min(plan.batchSize, remaining);
        int n = clusterOps.inplaceRestart(plan.targetVersionId, step);
        plan.rolledReplicas += step;
        plan.lastMessage = "inplace restarted=" + n + " progress="
                + plan.rolledReplicas + "/" + plan.totalReplicas;
        emit(DeployEvent.stagePromoted(serviceName, plan));
        if (plan.rolledReplicas >= plan.totalReplicas) {
            plan.status = DeployStatus.SUCCEEDED;
            plan.lastMessage = "inplace_complete";
            emit(DeployEvent.planSucceeded(serviceName, plan));
        }
        return plan;
    }

    // ---- rollback -----------------------------------------------------------

    /**
     * Rollback active plan (or explicitly to {@code toVersionId}).
     * Sets 100% traffic to rollback target and marks plan ROLLED_BACK.
     */
    public synchronized DeployPlan rollback(String toVersionId) throws Exception {
        String target = toVersionId != null ? toVersionId : stableVersionId;
        if (target == null) {
            throw new IllegalStateException("no rollback target");
        }
        requireVersion(target);
        clusterOps.setTrafficWeight(target, 100.0);
        // Zero out other known versions' traffic.
        for (String vid : versions.keySet()) {
            if (!vid.equals(target)) {
                try {
                    clusterOps.setTrafficWeight(vid, 0.0);
                } catch (Exception ignored) {
                }
            }
        }
        stableVersionId = target;
        DeployPlan plan = activePlan;
        if (plan == null) {
            plan = DeployPlan.rollback(nextPlanId("rollback"), target);
        }
        plan.status = DeployStatus.ROLLED_BACK;
        plan.currentTrafficPercent = 100.0;
        plan.lastMessage = "rolled_back_to=" + target;
        activePlan = plan;
        emit(DeployEvent.rolledBack(serviceName, plan));
        return plan;
    }

    /**
     * Abort / pause current plan without traffic changes.
     */
    public synchronized DeployPlan pause() {
        if (activePlan == null) {
            throw new IllegalStateException("no active plan");
        }
        activePlan.status = DeployStatus.PAUSED;
        activePlan.lastMessage = "paused";
        emit(DeployEvent.info(serviceName, "paused", activePlan.planId));
        return activePlan;
    }

    public synchronized DeployPlan resume() {
        if (activePlan == null) {
            throw new IllegalStateException("no active plan");
        }
        if (activePlan.status != DeployStatus.PAUSED) {
            throw new IllegalStateException("plan not paused");
        }
        activePlan.status = DeployStatus.IN_PROGRESS;
        activePlan.lastMessage = "resumed";
        emit(DeployEvent.info(serviceName, "resumed", activePlan.planId));
        return activePlan;
    }

    public DeployPlan activePlan() {
        return activePlan;
    }

    public Map<String, ReplicaSetView> replicaSnapshot() {
        Map<String, ReplicaSetView> m = new LinkedHashMap<>();
        for (Map.Entry<String, ReplicaSet> e : replicaSets.entrySet()) {
            ReplicaSet rs = e.getValue();
            m.put(e.getKey(), new ReplicaSetView(rs.versionId, rs.desired, rs.ready, rs.unavailable));
        }
        return m;
    }

    // ---- helpers ------------------------------------------------------------

    private void requireVersion(String versionId) {
        if (!versions.containsKey(versionId)) {
            throw new IllegalArgumentException("unknown version: " + versionId);
        }
    }

    private DeployPlan requireActiveCanary() {
        return requireActive(DeployPlan.Strategy.CANARY);
    }

    private DeployPlan requireActive(DeployPlan.Strategy strategy) {
        if (activePlan == null) {
            throw new IllegalStateException("no active plan");
        }
        if (activePlan.strategy != strategy) {
            throw new IllegalStateException("active plan is " + activePlan.strategy + " not " + strategy);
        }
        if (activePlan.status != DeployStatus.IN_PROGRESS && activePlan.status != DeployStatus.PAUSED) {
            throw new IllegalStateException("plan not active: " + activePlan.status);
        }
        return activePlan;
    }

    private String nextPlanId(String prefix) {
        return prefix + "-" + System.currentTimeMillis();
    }

    private void emit(DeployEvent event) {
        for (Consumer<DeployEvent> l : listeners) {
            try {
                l.accept(event);
            } catch (RuntimeException ignored) {
            }
        }
    }

    // ---- nested public types ------------------------------------------------

    public static final class ReplicaSetView {
        public final String versionId;
        public final int desired;
        public final int ready;
        public final int unavailable;

        public ReplicaSetView(String versionId, int desired, int ready, int unavailable) {
            this.versionId = versionId;
            this.desired = desired;
            this.ready = ready;
            this.unavailable = unavailable;
        }

        @Override
        public String toString() {
            return versionId + "[desired=" + desired + " ready=" + ready + "]";
        }
    }

    public static final class DeployPlan {
        public enum Strategy { CANARY, BLUE_GREEN, ROLLING, IN_PLACE, ROLLBACK }

        public final String planId;
        public final Strategy strategy;
        public final String stableVersionId;
        public final String targetVersionId;
        public final double[] stagesPercent;
        public final int canaryReplicas;
        public final int totalReplicas;
        public final int batchSize;
        public final DeployGate gate;
        public final boolean autoPromote;
        public DeployStatus status;
        public int stageIndex;
        public int rolledReplicas;
        public double currentTrafficPercent;
        public String lastMessage;
        public final long createdAtMs;

        private DeployPlan(
                String planId,
                Strategy strategy,
                String stableVersionId,
                String targetVersionId,
                double[] stagesPercent,
                int canaryReplicas,
                int totalReplicas,
                int batchSize,
                DeployGate gate,
                boolean autoPromote) {
            this.planId = planId;
            this.strategy = strategy;
            this.stableVersionId = stableVersionId;
            this.targetVersionId = targetVersionId;
            this.stagesPercent = stagesPercent;
            this.canaryReplicas = canaryReplicas;
            this.totalReplicas = totalReplicas;
            this.batchSize = batchSize;
            this.gate = gate;
            this.autoPromote = autoPromote;
            this.status = DeployStatus.PENDING;
            this.stageIndex = -1;
            this.rolledReplicas = 0;
            this.currentTrafficPercent = 0.0;
            this.lastMessage = "";
            this.createdAtMs = System.currentTimeMillis();
        }

        static DeployPlan canary(
                String id, String stable, String target, double[] stages,
                int canaryReplicas, DeployGate gate, boolean auto) {
            return new DeployPlan(id, Strategy.CANARY, stable, target, stages,
                    canaryReplicas, 0, 0, gate, auto);
        }

        static DeployPlan blueGreen(
                String id, String blue, String green, int replicas, DeployGate gate, boolean auto) {
            return new DeployPlan(id, Strategy.BLUE_GREEN, blue, green, new double[] {100.0},
                    replicas, replicas, 0, gate, auto);
        }

        static DeployPlan rolling(
                String id, String stable, String target, int total, int batch, DeployGate gate) {
            return new DeployPlan(id, Strategy.ROLLING, stable, target, null,
                    0, total, batch, gate, false);
        }

        static DeployPlan inPlace(
                String id, String version, int total, int batch, DeployGate gate) {
            return new DeployPlan(id, Strategy.IN_PLACE, version, version, null,
                    0, total, batch, gate, false);
        }

        static DeployPlan rollback(String id, String target) {
            return new DeployPlan(id, Strategy.ROLLBACK, target, target, null,
                    0, 0, 0, null, false);
        }

        @Override
        public String toString() {
            return String.format(Locale.ROOT,
                    "DeployPlan{id=%s strategy=%s status=%s stable=%s target=%s traffic=%.1f%% msg=%s}",
                    planId, strategy, status, stableVersionId, targetVersionId,
                    currentTrafficPercent, lastMessage);
        }
    }

    public static final class DeployEvent {
        public enum Type {
            INFO, PLAN_STARTED, STAGE_PROMOTED, GATE_FAILED, PLAN_SUCCEEDED, ROLLED_BACK
        }

        public final Type type;
        public final String serviceName;
        public final String message;
        public final DeployPlan plan;
        public final long timestampMs;

        private DeployEvent(Type type, String serviceName, String message, DeployPlan plan) {
            this.type = type;
            this.serviceName = serviceName;
            this.message = message;
            this.plan = plan;
            this.timestampMs = System.currentTimeMillis();
        }

        static DeployEvent info(String svc, String action, String detail) {
            return new DeployEvent(Type.INFO, svc, action + ": " + detail, null);
        }

        static DeployEvent planStarted(String svc, DeployPlan plan) {
            return new DeployEvent(Type.PLAN_STARTED, svc, "started " + plan.planId, plan);
        }

        static DeployEvent stagePromoted(String svc, DeployPlan plan) {
            return new DeployEvent(Type.STAGE_PROMOTED, svc, plan.lastMessage, plan);
        }

        static DeployEvent gateFailed(String svc, DeployPlan plan) {
            return new DeployEvent(Type.GATE_FAILED, svc, plan.lastMessage, plan);
        }

        static DeployEvent planSucceeded(String svc, DeployPlan plan) {
            return new DeployEvent(Type.PLAN_SUCCEEDED, svc, plan.lastMessage, plan);
        }

        static DeployEvent rolledBack(String svc, DeployPlan plan) {
            return new DeployEvent(Type.ROLLED_BACK, svc, plan.lastMessage, plan);
        }

        @Override
        public String toString() {
            return "DeployEvent{type=" + type + ", svc=" + serviceName + ", msg=" + message + "}";
        }
    }

    /**
     * In-memory ClusterOps for tests / local simulation.
     */
    public static final class InMemoryClusterOps implements ClusterOps {
        private final Map<String, Integer> desired = new LinkedHashMap<>();
        private final Map<String, Integer> ready = new LinkedHashMap<>();
        private final Map<String, Double> traffic = new LinkedHashMap<>();
        private final Map<String, Double> health = new LinkedHashMap<>();

        @Override
        public int scale(String versionId, int desiredReplicas) {
            desired.put(versionId, desiredReplicas);
            ready.put(versionId, desiredReplicas); // instant ready in sim
            return desiredReplicas;
        }

        @Override
        public int inplaceRestart(String versionId, int podCount) {
            return podCount;
        }

        @Override
        public double healthPercent(String versionId) {
            return health.getOrDefault(versionId, 100.0);
        }

        @Override
        public void setTrafficWeight(String versionId, double weightPercent) {
            traffic.put(versionId, weightPercent);
        }

        public double trafficWeight(String versionId) {
            return traffic.getOrDefault(versionId, 0.0);
        }

        public void setHealth(String versionId, double pct) {
            health.put(versionId, pct);
        }
    }
}
