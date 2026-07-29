/*
 * Model registry — source of truth for recommendation model versions.
 *
 * Mirrors MLflow / SageMaker Model Registry / internal platforms at
 * Meta, Google, Alibaba (PAI), ByteDance, Netflix Metaflow.
 */
package org.bytedeco.pytorch.utils.recommend.modelops;

import java.util.ArrayList;
import java.util.Collections;
import java.util.EnumSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.function.Consumer;

/** Thread-safe model version registry with stage transitions. */
public final class ModelRegistry {

    private final ConcurrentHashMap<String, ConcurrentHashMap<String, ModelVersion>> models =
            new ConcurrentHashMap<>();
    /** modelName -> prod versionId */
    private final ConcurrentHashMap<String, String> productionPointer = new ConcurrentHashMap<>();
    private final CopyOnWriteArrayList<Consumer<RegistryEvent>> listeners = new CopyOnWriteArrayList<>();

    public void addListener(Consumer<RegistryEvent> listener) {
        listeners.add(Objects.requireNonNull(listener));
    }

    public synchronized ModelVersion register(ModelVersion version) {
        Objects.requireNonNull(version, "version");
        ConcurrentHashMap<String, ModelVersion> versions =
                models.computeIfAbsent(version.modelName(), k -> new ConcurrentHashMap<>());
        if (versions.containsKey(version.versionId())) {
            throw new IllegalStateException("version already registered: " + version.fullyQualifiedId());
        }
        versions.put(version.versionId(), version);
        emit(RegistryEvent.registered(version));
        return version;
    }

    public ModelVersion get(String modelName, String versionId) {
        ConcurrentHashMap<String, ModelVersion> versions = models.get(modelName);
        if (versions == null) return null;
        return versions.get(versionId);
    }

    public ModelVersion require(String modelName, String versionId) {
        ModelVersion v = get(modelName, versionId);
        if (v == null) {
            throw new IllegalArgumentException("unknown model version: " + modelName + ":" + versionId);
        }
        return v;
    }

    public List<ModelVersion> listVersions(String modelName) {
        ConcurrentHashMap<String, ModelVersion> versions = models.get(modelName);
        if (versions == null) return List.of();
        List<ModelVersion> list = new ArrayList<>(versions.values());
        list.sort((a, b) -> b.createdAt().compareTo(a.createdAt()));
        return list;
    }

    public List<String> listModels() {
        return new ArrayList<>(models.keySet());
    }

    public ModelVersion productionOf(String modelName) {
        String vid = productionPointer.get(modelName);
        if (vid == null) return null;
        return get(modelName, vid);
    }

    /**
     * Transition stage with validation of allowed edges.
     */
    public synchronized ModelVersion transition(String modelName, String versionId, ModelStage to) {
        ModelVersion current = require(modelName, versionId);
        ModelStage from = current.stage();
        validateTransition(from, to);
        ModelVersion updated = current.withStage(to);
        models.get(modelName).put(versionId, updated);

        if (to == ModelStage.PROD) {
            // Archive previous prod
            String prevProd = productionPointer.put(modelName, versionId);
            if (prevProd != null && !prevProd.equals(versionId)) {
                ModelVersion prev = get(modelName, prevProd);
                if (prev != null && prev.stage() == ModelStage.PROD) {
                    ModelVersion archived = prev.withStage(ModelStage.ARCHIVED);
                    models.get(modelName).put(prevProd, archived);
                    emit(RegistryEvent.stageChanged(archived, ModelStage.PROD, ModelStage.ARCHIVED));
                }
            }
        }
        if (from == ModelStage.PROD && to != ModelStage.PROD) {
            productionPointer.remove(modelName, versionId);
        }
        emit(RegistryEvent.stageChanged(updated, from, to));
        return updated;
    }

    /**
     * Promote along the happy path one step:
     * TRAINED->OFFLINE_PASS->SHADOW->CANARY->PROD
     */
    public synchronized ModelVersion promote(String modelName, String versionId) {
        ModelVersion current = require(modelName, versionId);
        ModelStage next;
        switch (current.stage()) {
            case TRAINED:
                next = ModelStage.OFFLINE_PASS;
                break;
            case OFFLINE_PASS:
                next = ModelStage.SHADOW;
                break;
            case SHADOW:
                next = ModelStage.CANARY;
                break;
            case CANARY:
                next = ModelStage.PROD;
                break;
            default:
                throw new IllegalStateException("cannot auto-promote from " + current.stage());
        }
        return transition(modelName, versionId, next);
    }

    /**
     * Instant rollback: point PROD to a previous archived/canary version.
     */
    public synchronized ModelVersion rollback(String modelName, String toVersionId) {
        ModelVersion target = require(modelName, toVersionId);
        if (target.stage() == ModelStage.REJECTED) {
            throw new IllegalStateException("cannot rollback to REJECTED version");
        }
        // Mark current prod archived if different
        String cur = productionPointer.get(modelName);
        if (cur != null && !cur.equals(toVersionId)) {
            ModelVersion curV = get(modelName, cur);
            if (curV != null && curV.stage() == ModelStage.PROD) {
                models.get(modelName).put(cur, curV.withStage(ModelStage.ARCHIVED));
                emit(RegistryEvent.stageChanged(curV.withStage(ModelStage.ARCHIVED),
                        ModelStage.PROD, ModelStage.ARCHIVED));
            }
        }
        ModelVersion updated = target.withStage(ModelStage.PROD);
        models.get(modelName).put(toVersionId, updated);
        productionPointer.put(modelName, toVersionId);
        emit(RegistryEvent.rolledBack(modelName, cur, toVersionId));
        return updated;
    }

    public synchronized ModelVersion reject(String modelName, String versionId, String reason) {
        ModelVersion updated = transition(modelName, versionId, ModelStage.REJECTED);
        emit(RegistryEvent.rejected(updated, reason));
        return updated;
    }

    public synchronized ModelVersion updateMetrics(
            String modelName, String versionId, Map<String, Double> metrics) {
        ModelVersion current = require(modelName, versionId);
        ModelVersion updated = current.withOfflineMetrics(metrics);
        models.get(modelName).put(versionId, updated);
        return updated;
    }

    private static void validateTransition(ModelStage from, ModelStage to) {
        if (from == to) return;
        EnumSet<ModelStage> allowed;
        switch (from) {
            case TRAINED:
                allowed = EnumSet.of(ModelStage.OFFLINE_PASS, ModelStage.REJECTED, ModelStage.ARCHIVED);
                break;
            case OFFLINE_PASS:
                allowed = EnumSet.of(ModelStage.SHADOW, ModelStage.CANARY, ModelStage.REJECTED, ModelStage.ARCHIVED);
                break;
            case SHADOW:
                allowed = EnumSet.of(ModelStage.CANARY, ModelStage.REJECTED, ModelStage.ARCHIVED);
                break;
            case CANARY:
                allowed = EnumSet.of(ModelStage.PROD, ModelStage.SHADOW, ModelStage.REJECTED, ModelStage.ARCHIVED);
                break;
            case PROD:
                allowed = EnumSet.of(ModelStage.ARCHIVED, ModelStage.CANARY); // canary demote rare
                break;
            case ARCHIVED:
                allowed = EnumSet.of(ModelStage.PROD, ModelStage.CANARY); // rollback revive
                break;
            case REJECTED:
                allowed = EnumSet.noneOf(ModelStage.class);
                break;
            default:
                allowed = EnumSet.noneOf(ModelStage.class);
        }
        if (!allowed.contains(to)) {
            throw new IllegalStateException("illegal model stage transition " + from + " -> " + to);
        }
    }

    private void emit(RegistryEvent event) {
        for (Consumer<RegistryEvent> l : listeners) {
            try {
                l.accept(event);
            } catch (RuntimeException ignored) {
            }
        }
    }

    public static final class RegistryEvent {
        public enum Type { REGISTERED, STAGE_CHANGED, ROLLED_BACK, REJECTED }

        public final Type type;
        public final String modelName;
        public final String versionId;
        public final ModelStage from;
        public final ModelStage to;
        public final String detail;
        public final long timestampMs;

        private RegistryEvent(
                Type type, String modelName, String versionId,
                ModelStage from, ModelStage to, String detail) {
            this.type = type;
            this.modelName = modelName;
            this.versionId = versionId;
            this.from = from;
            this.to = to;
            this.detail = detail;
            this.timestampMs = System.currentTimeMillis();
        }

        static RegistryEvent registered(ModelVersion v) {
            return new RegistryEvent(Type.REGISTERED, v.modelName(), v.versionId(),
                    null, v.stage(), v.artifactUri());
        }

        static RegistryEvent stageChanged(ModelVersion v, ModelStage from, ModelStage to) {
            return new RegistryEvent(Type.STAGE_CHANGED, v.modelName(), v.versionId(),
                    from, to, from + "->" + to);
        }

        static RegistryEvent rolledBack(String model, String fromVid, String toVid) {
            return new RegistryEvent(Type.ROLLED_BACK, model, toVid, ModelStage.PROD, ModelStage.PROD,
                    "from=" + fromVid + " to=" + toVid);
        }

        static RegistryEvent rejected(ModelVersion v, String reason) {
            return new RegistryEvent(Type.REJECTED, v.modelName(), v.versionId(),
                    null, ModelStage.REJECTED, reason);
        }

        @Override
        public String toString() {
            return "RegistryEvent{type=" + type + ", model=" + modelName
                    + ":" + versionId + ", detail=" + detail + "}";
        }
    }
}
