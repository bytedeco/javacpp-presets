/*
 * Feature validation, freshness, drift, schema evolution, ACL, quality report.
 */
package org.bytedeco.pytorch.feature.lifecycle;

import org.bytedeco.pytorch.feature.core.FeatureView;
import org.bytedeco.pytorch.feature.core.Field;
import org.bytedeco.pytorch.feature.core.ValueType;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.Set;

/** Schema / null-rate / cardinality / embedding validators. */
public final class FeatureValidator {

    public static final class Issue {
        public final String code;
        public final String message;
        public final String feature;
        public final boolean error;

        public Issue(String code, String message, String feature, boolean error) {
            this.code = code;
            this.message = message;
            this.feature = feature;
            this.error = error;
        }

        @Override
        public String toString() {
            return (error ? "ERROR" : "WARN") + "[" + code + "] " + feature + ": " + message;
        }
    }

    public static final class Report {
        public final List<Issue> issues;
        public final long rowsChecked;
        public final boolean ok;

        public Report(List<Issue> issues, long rowsChecked) {
            this.issues = Collections.unmodifiableList(issues);
            this.rowsChecked = rowsChecked;
            boolean anyError = false;
            for (Issue i : issues) if (i.error) { anyError = true; break; }
            this.ok = !anyError;
        }

        @Override
        public String toString() {
            return "ValidationReport{ok=" + ok + ", rows=" + rowsChecked + ", issues=" + issues.size() + "}";
        }
    }

    private final double maxNullRate;
    private final long minRows;

    public FeatureValidator() {
        this(0.5, 1);
    }

    public FeatureValidator(double maxNullRate, long minRows) {
        this.maxNullRate = maxNullRate;
        this.minRows = minRows;
    }

    public Report validate(FeatureView view, List<Map<String, Object>> rows) {
        Objects.requireNonNull(view, "view");
        List<Issue> issues = new ArrayList<>();
        if (rows == null || rows.size() < minRows) {
            issues.add(new Issue("TOO_FEW_ROWS", "rows < " + minRows, view.name(), true));
            return new Report(issues, rows == null ? 0 : rows.size());
        }
        List<String> joinKeys = view.joinKeys().isEmpty() ? view.entityNames() : view.joinKeys();
        for (String jk : joinKeys) {
            long nulls = 0;
            Set<Object> distinct = new HashSet<>();
            for (Map<String, Object> r : rows) {
                Object v = r.get(jk);
                if (v == null) nulls++;
                else distinct.add(v);
            }
            double nr = nulls * 1.0 / rows.size();
            if (nr > 0) {
                issues.add(new Issue("NULL_JOIN_KEY", "null rate=" + nr, jk, nr > 0.01));
            }
            if (distinct.size() < Math.min(rows.size(), 1)) {
                issues.add(new Issue("LOW_CARDINALITY_KEY", "distinct=" + distinct.size(), jk, false));
            }
        }
        for (Field f : view.schema()) {
            long nulls = 0;
            for (Map<String, Object> r : rows) {
                if (r.get(f.name()) == null) nulls++;
            }
            double nr = nulls * 1.0 / rows.size();
            if (nr > maxNullRate) {
                issues.add(new Issue("HIGH_NULL_RATE", "null rate=" + String.format(Locale.ROOT, "%.3f", nr),
                        f.name(), true));
            }
            if (f.valueType() == ValueType.EMBEDDING) {
                int dim = f.embeddingDim();
                for (Map<String, Object> r : rows) {
                    Object v = r.get(f.name());
                    if (v == null) continue;
                    if (v instanceof float[]) {
                        if (((float[]) v).length != dim) {
                            issues.add(new Issue("EMB_DIM_MISMATCH",
                                    "len=" + ((float[]) v).length + " expected=" + dim, f.name(), true));
                            break;
                        }
                    } else if (v instanceof double[]) {
                        if (((double[]) v).length != dim) {
                            issues.add(new Issue("EMB_DIM_MISMATCH",
                                    "len=" + ((double[]) v).length + " expected=" + dim, f.name(), true));
                            break;
                        }
                    } else {
                        issues.add(new Issue("EMB_TYPE", "not float[]/double[]", f.name(), true));
                        break;
                    }
                }
            }
        }
        return new Report(issues, rows.size());
    }
}
