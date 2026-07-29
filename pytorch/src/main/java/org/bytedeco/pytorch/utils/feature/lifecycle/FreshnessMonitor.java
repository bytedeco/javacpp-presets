/*
 * Feature freshness SLO monitor (event/ingest lag).
 * Industry: online features often minutes-level; batch daily — Meta/Google/Alibaba SLO practice.
 */
package org.bytedeco.pytorch.utils.feature.lifecycle;

import org.bytedeco.pytorch.utils.feature.offline.FileOfflineStore;

import java.time.Duration;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/** Tracks last event timestamp per view and compares against SLO. */
public final class FreshnessMonitor {

    public static final class Status {
        public final String viewName;
        public final long lastEventTsMs;
        public final long lagMs;
        public final long sloMs;
        public final boolean alert;

        public Status(String viewName, long lastEventTsMs, long lagMs, long sloMs, boolean alert) {
            this.viewName = viewName;
            this.lastEventTsMs = lastEventTsMs;
            this.lagMs = lagMs;
            this.sloMs = sloMs;
            this.alert = alert;
        }

        @Override
        public String toString() {
            return "Freshness{view=" + viewName + ", lagMs=" + lagMs + ", sloMs=" + sloMs + ", alert=" + alert + "}";
        }
    }

    private final ConcurrentHashMap<String, Long> lastEventTs = new ConcurrentHashMap<>();
    private final ConcurrentHashMap<String, Long> sloMs = new ConcurrentHashMap<>();

    public void setSlo(String project, String viewName, Duration slo) {
        sloMs.put(key(project, viewName), slo.toMillis());
    }

    public void observe(String project, String viewName, long eventTsMs) {
        lastEventTs.merge(key(project, viewName), eventTsMs, Math::max);
    }

    public void observeRows(String project, String viewName, List<Map<String, Object>> rows, String tsCol) {
        if (rows == null) return;
        String col = tsCol != null ? tsCol : "event_timestamp";
        long max = Long.MIN_VALUE;
        for (Map<String, Object> r : rows) {
            long ts = FileOfflineStore.toEpochMillis(r.get(col));
            if (ts > max) max = ts;
        }
        if (max > Long.MIN_VALUE) observe(project, viewName, max);
    }

    public Status check(String project, String viewName) {
        return check(project, viewName, System.currentTimeMillis());
    }

    public Status check(String project, String viewName, long nowMs) {
        String k = key(project, viewName);
        long last = lastEventTs.getOrDefault(k, 0L);
        long slo = sloMs.getOrDefault(k, Duration.ofHours(24).toMillis());
        long lag = last <= 0 ? Long.MAX_VALUE / 4 : Math.max(0L, nowMs - last);
        return new Status(viewName, last, lag, slo, lag > slo);
    }

    private static String key(String project, String viewName) {
        return (project == null || project.isEmpty() ? "default" : project) + "/" + viewName;
    }
}
