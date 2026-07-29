/*
 * Experiment layer (traffic domain / mutually exclusive domain).
 *
 * Industry terminology:
 *   - Meta XP: "layer" / domain with exclusive experiments
 *   - ByteDance Libra: "流量层" (traffic layer)
 *   - Alibaba: "互斥域" (mutex domain)
 *   - Google: experiment layer / namespace
 *   - Tencent: 实验层
 *
 * Rules:
 *   1. Experiments in the SAME layer share a fixed 100% bucket space and
 *      must not over-subscribe traffic (sum of trafficPercent <= 100).
 *   2. Experiments in DIFFERENT layers are orthogonal: each unit is hashed
 *      independently so multi-layer exposure is approximately independent.
 *   3. Diversion unit should be consistent within a layer to avoid SRM.
 */
package org.bytedeco.pytorch.utils.recommend.abtest;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.CopyOnWriteArrayList;

/**
 * A named traffic layer holding mutually exclusive experiments.
 */
public final class ExperimentLayer {

    private final String id;
    private final String name;
    private final DiversionUnit diversionUnit;
    private final String description;
    private final CopyOnWriteArrayList<Experiment> experiments;

    public ExperimentLayer(String id, String name, DiversionUnit diversionUnit) {
        this(id, name, diversionUnit, "");
    }

    public ExperimentLayer(String id, String name, DiversionUnit diversionUnit, String description) {
        if (id == null || id.isEmpty()) {
            throw new IllegalArgumentException("layer id required");
        }
        this.id = id;
        this.name = name != null ? name : id;
        this.diversionUnit = diversionUnit != null ? diversionUnit : DiversionUnit.USER_ID;
        this.description = description != null ? description : "";
        this.experiments = new CopyOnWriteArrayList<>();
    }

    public String id() {
        return id;
    }

    public String name() {
        return name;
    }

    public DiversionUnit diversionUnit() {
        return diversionUnit;
    }

    public String description() {
        return description;
    }

    public List<Experiment> experiments() {
        return Collections.unmodifiableList(new ArrayList<>(experiments));
    }

    /**
     * Sum of trafficPercent across experiments that currently accept traffic.
     */
    public double usedTrafficPercent() {
        double used = 0.0;
        for (Experiment e : experiments) {
            if (e.status().acceptsTraffic()) {
                used += e.trafficPercent();
            }
        }
        return used;
    }

    public double remainingTrafficPercent() {
        return Math.max(0.0, 100.0 - usedTrafficPercent());
    }

    /**
     * Register an experiment into this layer with capacity check.
     *
     * @throws IllegalStateException if layer capacity would be exceeded or
     *         diversion unit mismatches
     */
    public synchronized void addExperiment(Experiment experiment) {
        Objects.requireNonNull(experiment, "experiment");
        if (!id.equals(experiment.layerId())) {
            throw new IllegalArgumentException(
                    "experiment layerId=" + experiment.layerId() + " != layer id=" + id);
        }
        if (experiment.diversionUnit() != diversionUnit) {
            throw new IllegalArgumentException(
                    "diversion unit mismatch: layer=" + diversionUnit
                            + " experiment=" + experiment.diversionUnit()
                            + " (mixed units in one layer cause SRM)");
        }
        for (Experiment existing : experiments) {
            if (existing.id().equals(experiment.id())) {
                throw new IllegalStateException("experiment already in layer: " + experiment.id());
            }
        }
        if (experiment.status().acceptsTraffic()) {
            double used = usedTrafficPercent();
            if (BucketAssigner.layerCapacityExceeded(used, experiment.trafficPercent())) {
                throw new IllegalStateException(String.format(
                        "layer '%s' capacity exceeded: used=%.2f%% + new=%.2f%% > 100%%",
                        id, used, experiment.trafficPercent()));
            }
        }
        experiments.add(experiment);
    }

    /**
     * Replace experiment definition (e.g. status / traffic change) with capacity re-check.
     */
    public synchronized void updateExperiment(Experiment updated) {
        Objects.requireNonNull(updated, "updated");
        int idx = -1;
        for (int i = 0; i < experiments.size(); i++) {
            if (experiments.get(i).id().equals(updated.id())) {
                idx = i;
                break;
            }
        }
        if (idx < 0) {
            throw new IllegalArgumentException("experiment not in layer: " + updated.id());
        }
        // Capacity check excluding the old version of this experiment.
        double used = 0.0;
        for (int i = 0; i < experiments.size(); i++) {
            if (i == idx) continue;
            Experiment e = experiments.get(i);
            if (e.status().acceptsTraffic()) {
                used += e.trafficPercent();
            }
        }
        if (updated.status().acceptsTraffic()
                && BucketAssigner.layerCapacityExceeded(used, updated.trafficPercent())) {
            throw new IllegalStateException(String.format(
                    "layer '%s' capacity exceeded on update: used=%.2f%% + new=%.2f%% > 100%%",
                    id, used, updated.trafficPercent()));
        }
        experiments.set(idx, updated);
    }

    public synchronized boolean removeExperiment(String experimentId) {
        return experiments.removeIf(e -> e.id().equals(experimentId));
    }

    public Experiment find(String experimentId) {
        for (Experiment e : experiments) {
            if (e.id().equals(experimentId)) {
                return e;
            }
        }
        return null;
    }

    /**
     * Assign unit against the first matching active experiment in this layer.
     *
     * <p>Production systems pre-allocate disjoint bucket ranges per experiment.
     * This reference implementation evaluates experiments in registration order
     * and relies on independent salts + traffic windows; for strict disjoint
     * ranges use {@link #assignWithReservedRanges}.
     */
    public BucketAssigner.Assignment assign(String unitId, long nowEpochMs) {
        for (Experiment e : experiments) {
            BucketAssigner.Assignment a = BucketAssigner.assign(e, unitId, nowEpochMs);
            if (a != null) {
                return a;
            }
        }
        return null;
    }

    /**
     * Strict disjoint range assignment: each active experiment occupies a
     * contiguous percentage window in registration order.
     *
     * <pre>
     *   exp0: [0, t0)
     *   exp1: [t0, t0+t1)
     *   ...
     * </pre>
     *
     * This matches Alibaba / Tencent mutex-domain bucket allocation more closely
     * than independent traffic windows.
     */
    public BucketAssigner.Assignment assignWithReservedRanges(String unitId, long nowEpochMs) {
        long bucketCount = 1000L;
        for (Experiment e : experiments) {
            if (e.status().acceptsTraffic()) {
                bucketCount = e.bucketCount();
                break;
            }
        }
        long bucket = BucketAssigner.bucketOf(id, unitId, bucketCount);
        double pct = (bucket * 100.0) / (double) bucketCount;

        double cursor = 0.0;
        for (Experiment e : experiments) {
            if (!e.status().acceptsTraffic()) {
                continue;
            }
            if (e.startTime() != null && nowEpochMs < e.startTime().toEpochMilli()) {
                continue;
            }
            if (e.endTime() != null && nowEpochMs >= e.endTime().toEpochMilli()) {
                continue;
            }
            double next = cursor + e.trafficPercent();
            if (pct >= cursor && pct < next) {
                Variant variant = BucketAssigner.pickVariant(e.variants(), e.salt(), unitId);
                return new BucketAssigner.Assignment(
                        e.id(),
                        e.layerId(),
                        variant.id(),
                        variant.isControl(),
                        bucket,
                        unitId,
                        e.diversionUnit(),
                        nowEpochMs);
            }
            cursor = next;
        }
        return null;
    }

    @Override
    public String toString() {
        return "ExperimentLayer{id='" + id + "', used=" + usedTrafficPercent()
                + "%, experiments=" + experiments.size() + "}";
    }
}
