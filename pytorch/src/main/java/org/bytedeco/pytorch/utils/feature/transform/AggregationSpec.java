/*
 * Window aggregation spec — Feathub / Flink-style TUMBLE / HOP / SLIDING.
 *
 * Used to describe stream feature computations; in-process engine can simulate
 * simple COUNT/SUM/AVG/MAX/MIN over event lists for demos and benchmarks.
 */
package org.bytedeco.pytorch.utils.feature.transform;

import org.bytedeco.pytorch.utils.feature.offline.FileOfflineStore;

import java.time.Duration;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;

/** Aggregation descriptor + simple in-process executor. */
public final class AggregationSpec implements FeatureTransform {

    public enum AggFunc {
        COUNT, SUM, AVG, MAX, MIN, LAST
    }

    public enum WindowType {
        /** Fixed non-overlapping windows. */
        TUMBLE,
        /** Sliding with hop size. */
        HOP,
        /** Sliding window ending at each event (simplified). */
        SLIDING
    }

    private final String name;
    private final AggFunc func;
    private final String inputColumn;
    private final String outputColumn;
    private final List<String> groupByKeys;
    private final String timestampColumn;
    private final Duration windowSize;
    private final Duration hopSize;
    private final WindowType windowType;

    private AggregationSpec(Builder b) {
        this.name = b.name != null ? b.name : "agg";
        this.func = Objects.requireNonNull(b.func, "func");
        this.inputColumn = b.inputColumn != null ? b.inputColumn : "";
        this.outputColumn = b.outputColumn != null ? b.outputColumn
                : (func.name().toLowerCase(Locale.ROOT) + "_" + (inputColumn.isEmpty() ? "x" : inputColumn));
        this.groupByKeys = List.copyOf(b.groupByKeys);
        this.timestampColumn = b.timestampColumn != null ? b.timestampColumn : "event_timestamp";
        this.windowSize = b.windowSize != null ? b.windowSize : Duration.ofMinutes(5);
        this.hopSize = b.hopSize != null ? b.hopSize : b.windowSize;
        this.windowType = b.windowType != null ? b.windowType : WindowType.TUMBLE;
    }

    public static Builder builder(AggFunc func) {
        return new Builder(func);
    }

    public static AggregationSpec count(String output, String... groupBy) {
        Builder b = builder(AggFunc.COUNT).outputColumn(output);
        for (String g : groupBy) b.groupBy(g);
        return b.build();
    }

    public static AggregationSpec sum(String input, String output, String... groupBy) {
        Builder b = builder(AggFunc.SUM).inputColumn(input).outputColumn(output);
        for (String g : groupBy) b.groupBy(g);
        return b.build();
    }

    public String name() { return name; }
    public AggFunc func() { return func; }
    public String inputColumn() { return inputColumn; }
    public String outputColumn() { return outputColumn; }
    public List<String> groupByKeys() { return groupByKeys; }
    public String timestampColumn() { return timestampColumn; }
    public Duration windowSize() { return windowSize; }
    public Duration hopSize() { return hopSize; }
    public WindowType windowType() { return windowType; }

    /** Feathub-like descriptor string. */
    public String descriptor() {
        return func.name() + "(" + inputColumn + ") OVER " + windowType.name()
                + " " + windowSize + (windowType == WindowType.HOP ? " HOP " + hopSize : "")
                + " GROUP BY " + String.join(",", groupByKeys);
    }

    @Override
    public List<Map<String, Object>> apply(List<Map<String, Object>> rows) {
        if (rows == null || rows.isEmpty()) return List.of();
        long winMs = Math.max(1L, windowSize.toMillis());
        long hopMs = hopSize != null ? Math.max(1L, hopSize.toMillis()) : winMs;

        // group by keys + window bucket
        Map<String, List<Map<String, Object>>> buckets = new LinkedHashMap<>();
        for (Map<String, Object> row : rows) {
            long ts = FileOfflineStore.toEpochMillis(row.get(timestampColumn));
            long bucket;
            switch (windowType) {
                case HOP:
                case SLIDING:
                    bucket = (ts / hopMs) * hopMs;
                    break;
                case TUMBLE:
                default:
                    bucket = (ts / winMs) * winMs;
                    break;
            }
            StringBuilder gk = new StringBuilder();
            for (String k : groupByKeys) {
                gk.append(row.get(k)).append('|');
            }
            gk.append(bucket);
            buckets.computeIfAbsent(gk.toString(), x -> new ArrayList<>()).add(row);
        }

        List<Map<String, Object>> out = new ArrayList<>();
        for (Map.Entry<String, List<Map<String, Object>>> e : buckets.entrySet()) {
            List<Map<String, Object>> group = e.getValue();
            Map<String, Object> sample = group.get(0);
            Map<String, Object> row = new LinkedHashMap<>();
            for (String k : groupByKeys) {
                row.put(k, sample.get(k));
            }
            long ts = FileOfflineStore.toEpochMillis(sample.get(timestampColumn));
            long bucket = (ts / (windowType == WindowType.TUMBLE ? winMs : hopMs))
                    * (windowType == WindowType.TUMBLE ? winMs : hopMs);
            row.put(timestampColumn, bucket + winMs - 1); // window end as event ts
            row.put(outputColumn, aggregate(group));
            row.put("_window_start", bucket);
            row.put("_window_end", bucket + winMs);
            out.add(row);
        }
        return out;
    }

    private Object aggregate(List<Map<String, Object>> group) {
        switch (func) {
            case COUNT:
                return (long) group.size();
            case SUM: {
                double s = 0;
                for (Map<String, Object> r : group) {
                    Object v = r.get(inputColumn);
                    if (v instanceof Number) s += ((Number) v).doubleValue();
                }
                return s;
            }
            case AVG: {
                double s = 0;
                int n = 0;
                for (Map<String, Object> r : group) {
                    Object v = r.get(inputColumn);
                    if (v instanceof Number) {
                        s += ((Number) v).doubleValue();
                        n++;
                    }
                }
                return n == 0 ? 0.0 : s / n;
            }
            case MAX: {
                double m = Double.NEGATIVE_INFINITY;
                boolean any = false;
                for (Map<String, Object> r : group) {
                    Object v = r.get(inputColumn);
                    if (v instanceof Number) {
                        m = Math.max(m, ((Number) v).doubleValue());
                        any = true;
                    }
                }
                return any ? m : null;
            }
            case MIN: {
                double m = Double.POSITIVE_INFINITY;
                boolean any = false;
                for (Map<String, Object> r : group) {
                    Object v = r.get(inputColumn);
                    if (v instanceof Number) {
                        m = Math.min(m, ((Number) v).doubleValue());
                        any = true;
                    }
                }
                return any ? m : null;
            }
            case LAST: {
                Map<String, Object> last = group.get(group.size() - 1);
                return inputColumn.isEmpty() ? last : last.get(inputColumn);
            }
            default:
                return null;
        }
    }

    @Override
    public String toString() {
        return "AggregationSpec{" + descriptor() + "}";
    }

    public static final class Builder {
        private String name;
        private final AggFunc func;
        private String inputColumn;
        private String outputColumn;
        private final List<String> groupByKeys = new ArrayList<>();
        private String timestampColumn = "event_timestamp";
        private Duration windowSize = Duration.ofMinutes(5);
        private Duration hopSize;
        private WindowType windowType = WindowType.TUMBLE;

        private Builder(AggFunc func) {
            this.func = func;
        }

        public Builder name(String name) { this.name = name; return this; }
        public Builder inputColumn(String inputColumn) { this.inputColumn = inputColumn; return this; }
        public Builder outputColumn(String outputColumn) { this.outputColumn = outputColumn; return this; }
        public Builder groupBy(String key) { if (key != null) groupByKeys.add(key); return this; }
        public Builder timestampColumn(String timestampColumn) { this.timestampColumn = timestampColumn; return this; }
        public Builder windowSize(Duration windowSize) { this.windowSize = windowSize; return this; }
        public Builder hopSize(Duration hopSize) { this.hopSize = hopSize; return this; }
        public Builder windowType(WindowType windowType) { this.windowType = windowType; return this; }

        public AggregationSpec build() {
            return new AggregationSpec(this);
        }
    }
}
