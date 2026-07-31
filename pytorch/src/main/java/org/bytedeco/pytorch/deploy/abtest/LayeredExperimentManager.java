/*
 * Multi-layer experiment manager — the runtime heart of online AB.
 *
 * Mirrors production experiment platforms:
 *   Meta XP, ByteDance Libra, Google Experiment Framework,
 *   Alibaba ABTest, Tencent Tab / TabEX.
 *
 * Responsibilities:
 *   - Register layers and experiments with capacity / mutex checks
 *   - Resolve multi-layer assignments for one request unit
 *   - Expose parameter overlays (variant parameters merge)
 *   - Lifecycle transitions with basic audit events
 */
package org.bytedeco.pytorch.deploy.abtest;

import java.time.Instant;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.function.Consumer;

/**
 * Thread-safe manager for layered recommendation experiments.
 *
 * <p>Typical recsys layer layout (industry common):
 * <pre>
 *   layer_recall      — recall channel / ANN index / inverted index experiments
 *   layer_coarse      — coarse rank model / formula experiments
 *   layer_fine        — fine rank (CTR/CVR) model experiments
 *   layer_rerank      — diversity / business rule / re-ranker experiments
 *   layer_mix         — multi-queue mix / insert experiments
 *   layer_ui          — UI card / layout (often device-level)
 * </pre>
 */
public final class LayeredExperimentManager {

    private final ConcurrentHashMap<String, ExperimentLayer> layers = new ConcurrentHashMap<>();
    private final ConcurrentHashMap<String, Experiment> experimentsById = new ConcurrentHashMap<>();
    private final CopyOnWriteArrayList<Consumer<AuditEvent>> listeners = new CopyOnWriteArrayList<>();
    private final boolean useReservedRanges;

    public LayeredExperimentManager() {
        this(true);
    }

    /**
     * @param useReservedRanges if true, same-layer assignment uses disjoint
     *                          reserved percentage windows (Alibaba-style);
     *                          if false, each experiment uses independent salt
     *                          traffic windows (simpler but can double-expose
     *                          in edge cases when salts collide conceptually).
     */
    public LayeredExperimentManager(boolean useReservedRanges) {
        this.useReservedRanges = useReservedRanges;
    }

    public void addListener(Consumer<AuditEvent> listener) {
        listeners.add(Objects.requireNonNull(listener));
    }

    public ExperimentLayer createLayer(String id, String name, DiversionUnit unit) {
        ExperimentLayer layer = new ExperimentLayer(id, name, unit);
        ExperimentLayer prev = layers.putIfAbsent(id, layer);
        if (prev != null) {
            throw new IllegalStateException("layer already exists: " + id);
        }
        emit(AuditEvent.layerCreated(id, name));
        return layer;
    }

    public ExperimentLayer getLayer(String id) {
        return layers.get(id);
    }

    public List<ExperimentLayer> listLayers() {
        return new ArrayList<>(layers.values());
    }

    public void register(Experiment experiment) {
        Objects.requireNonNull(experiment, "experiment");
        ExperimentLayer layer = layers.get(experiment.layerId());
        if (layer == null) {
            throw new IllegalStateException("layer not found: " + experiment.layerId()
                    + " — createLayer() first");
        }
        layer.addExperiment(experiment);
        experimentsById.put(experiment.id(), experiment);
        emit(AuditEvent.experimentRegistered(experiment.id(), experiment.layerId(), experiment.status()));
    }

    public Experiment getExperiment(String experimentId) {
        return experimentsById.get(experimentId);
    }

    public List<Experiment> listExperiments() {
        return new ArrayList<>(experimentsById.values());
    }

    /**
     * Transition experiment status with capacity re-validation.
     */
    public synchronized Experiment transition(String experimentId, ExperimentStatus newStatus) {
        Experiment current = requireExperiment(experimentId);
        ExperimentStatus old = current.status();
        validateTransition(old, newStatus);
        Experiment updated = current.withStatus(newStatus);
        ExperimentLayer layer = layers.get(updated.layerId());
        layer.updateExperiment(updated);
        experimentsById.put(experimentId, updated);
        emit(AuditEvent.statusChanged(experimentId, old, newStatus));
        return updated;
    }

    /**
     * Adjust experiment traffic percent (canary ramp style for experiments).
     */
    public synchronized Experiment setTrafficPercent(String experimentId, double percent) {
        if (percent < 0.0 || percent > 100.0) {
            throw new IllegalArgumentException("percent must be in [0, 100]");
        }
        Experiment current = requireExperiment(experimentId);
        Experiment updated = current.withTrafficPercent(percent);
        ExperimentLayer layer = layers.get(updated.layerId());
        layer.updateExperiment(updated);
        experimentsById.put(experimentId, updated);
        emit(AuditEvent.trafficChanged(experimentId, current.trafficPercent(), percent));
        return updated;
    }

    /**
     * Resolve all layer assignments for one diversion unit.
     *
     * <p>Returns one assignment per layer that matched an active experiment.
     * Empty list means unit is in holdout / default production path for all layers.
     */
    public List<BucketAssigner.Assignment> resolve(String unitId) {
        return resolve(unitId, System.currentTimeMillis());
    }

    public List<BucketAssigner.Assignment> resolve(String unitId, long nowEpochMs) {
        Objects.requireNonNull(unitId, "unitId");
        List<BucketAssigner.Assignment> result = new ArrayList<>();
        for (ExperimentLayer layer : layers.values()) {
            BucketAssigner.Assignment a = useReservedRanges
                    ? layer.assignWithReservedRanges(unitId, nowEpochMs)
                    : layer.assign(unitId, nowEpochMs);
            if (a != null) {
                result.add(a);
            }
        }
        return result;
    }

    /**
     * Merge variant parameter overlays from all matched assignments.
     * Later layers override earlier keys if conflict (deterministic by layer id sort).
     */
    public Map<String, String> resolveParameters(String unitId) {
        List<BucketAssigner.Assignment> assignments = resolve(unitId);
        // Stable merge: sort by layerId then experimentId.
        assignments.sort((a, b) -> {
            int c = a.layerId().compareTo(b.layerId());
            return c != 0 ? c : a.experimentId().compareTo(b.experimentId());
        });
        Map<String, String> merged = new LinkedHashMap<>();
        for (BucketAssigner.Assignment a : assignments) {
            Experiment exp = experimentsById.get(a.experimentId());
            if (exp == null) continue;
            Variant v = exp.variant(a.variantId());
            if (v == null) continue;
            merged.putAll(v.parameters());
            // Always expose assignment meta for logging / ranking pipeline.
            merged.put("exp." + exp.id() + ".variant", v.id());
            merged.put("exp." + exp.id() + ".layer", exp.layerId());
        }
        return Collections.unmodifiableMap(merged);
    }

    /**
     * Snapshot of current exposure decision for logging (Hive / ClickHouse / Kafka).
     */
    public ExposureLog buildExposureLog(String unitId, String requestId, Map<String, String> extra) {
        List<BucketAssigner.Assignment> assignments = resolve(unitId);
        Map<String, String> params = resolveParameters(unitId);
        return new ExposureLog(unitId, requestId, Instant.now().toEpochMilli(), assignments, params, extra);
    }

    private Experiment requireExperiment(String experimentId) {
        Experiment e = experimentsById.get(experimentId);
        if (e == null) {
            throw new IllegalArgumentException("unknown experiment: " + experimentId);
        }
        return e;
    }

    private static void validateTransition(ExperimentStatus from, ExperimentStatus to) {
        if (from == to) {
            return;
        }
        if (from.isTerminal()) {
            throw new IllegalStateException("cannot transition from terminal state " + from);
        }
        // Allowed graph (pragmatic subset of production state machines).
        boolean ok;
        switch (from) {
            case DRAFT:
                ok = to == ExperimentStatus.REVIEW || to == ExperimentStatus.KILLED;
                break;
            case REVIEW:
                ok = to == ExperimentStatus.AA_RUNNING
                        || to == ExperimentStatus.RUNNING
                        || to == ExperimentStatus.DRAFT
                        || to == ExperimentStatus.KILLED;
                break;
            case AA_RUNNING:
                ok = to == ExperimentStatus.RUNNING
                        || to == ExperimentStatus.PAUSED
                        || to == ExperimentStatus.COMPLETED
                        || to == ExperimentStatus.KILLED
                        || to == ExperimentStatus.ROLLED_BACK;
                break;
            case RUNNING:
                ok = to == ExperimentStatus.PAUSED
                        || to == ExperimentStatus.COMPLETED
                        || to == ExperimentStatus.KILLED
                        || to == ExperimentStatus.ROLLED_BACK;
                break;
            case PAUSED:
                ok = to == ExperimentStatus.RUNNING
                        || to == ExperimentStatus.COMPLETED
                        || to == ExperimentStatus.KILLED
                        || to == ExperimentStatus.ROLLED_BACK;
                break;
            default:
                ok = false;
        }
        if (!ok) {
            throw new IllegalStateException("illegal transition " + from + " -> " + to);
        }
    }

    private void emit(AuditEvent event) {
        for (Consumer<AuditEvent> l : listeners) {
            try {
                l.accept(event);
            } catch (RuntimeException ignored) {
                // listeners must not break diversion path
            }
        }
    }

    // ---- nested types -------------------------------------------------------

    /** Immutable exposure log row for analytics pipeline. */
    public static final class ExposureLog {
        private final String unitId;
        private final String requestId;
        private final long timestampMs;
        private final List<BucketAssigner.Assignment> assignments;
        private final Map<String, String> parameters;
        private final Map<String, String> extra;

        public ExposureLog(
                String unitId,
                String requestId,
                long timestampMs,
                List<BucketAssigner.Assignment> assignments,
                Map<String, String> parameters,
                Map<String, String> extra) {
            this.unitId = unitId;
            this.requestId = requestId;
            this.timestampMs = timestampMs;
            this.assignments = Collections.unmodifiableList(new ArrayList<>(assignments));
            this.parameters = Collections.unmodifiableMap(new LinkedHashMap<>(parameters));
            this.extra = extra == null
                    ? Collections.emptyMap()
                    : Collections.unmodifiableMap(new LinkedHashMap<>(extra));
        }

        public String unitId() {
            return unitId;
        }

        public String requestId() {
            return requestId;
        }

        public long timestampMs() {
            return timestampMs;
        }

        public List<BucketAssigner.Assignment> assignments() {
            return assignments;
        }

        public Map<String, String> parameters() {
            return parameters;
        }

        public Map<String, String> extra() {
            return extra;
        }
    }

    /** Lightweight audit event for experiment ops. */
    public static final class AuditEvent {
        public enum Type {
            LAYER_CREATED,
            EXPERIMENT_REGISTERED,
            STATUS_CHANGED,
            TRAFFIC_CHANGED
        }

        private final Type type;
        private final String experimentId;
        private final String layerId;
        private final String detail;
        private final long timestampMs;

        private AuditEvent(Type type, String experimentId, String layerId, String detail) {
            this.type = type;
            this.experimentId = experimentId;
            this.layerId = layerId;
            this.detail = detail;
            this.timestampMs = System.currentTimeMillis();
        }

        static AuditEvent layerCreated(String layerId, String name) {
            return new AuditEvent(Type.LAYER_CREATED, null, layerId, name);
        }

        static AuditEvent experimentRegistered(String expId, String layerId, ExperimentStatus status) {
            return new AuditEvent(Type.EXPERIMENT_REGISTERED, expId, layerId, status.name());
        }

        static AuditEvent statusChanged(String expId, ExperimentStatus from, ExperimentStatus to) {
            return new AuditEvent(Type.STATUS_CHANGED, expId, null, from + "->" + to);
        }

        static AuditEvent trafficChanged(String expId, double from, double to) {
            return new AuditEvent(Type.TRAFFIC_CHANGED, expId, null, from + "->" + to);
        }

        public Type type() {
            return type;
        }

        public String experimentId() {
            return experimentId;
        }

        public String layerId() {
            return layerId;
        }

        public String detail() {
            return detail;
        }

        public long timestampMs() {
            return timestampMs;
        }

        @Override
        public String toString() {
            return "AuditEvent{type=" + type + ", exp=" + experimentId
                    + ", layer=" + layerId + ", detail=" + detail + "}";
        }
    }
}
