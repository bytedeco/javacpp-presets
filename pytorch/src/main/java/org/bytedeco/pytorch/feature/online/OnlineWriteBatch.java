/*
 * Batch of online writes (materialization output unit).
 */
package org.bytedeco.pytorch.feature.online;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Objects;

/** Immutable write batch. */
public final class OnlineWriteBatch {

    private final List<OnlineFeatureRow> rows;
    private final long createdAtMs;

    private OnlineWriteBatch(List<OnlineFeatureRow> rows) {
        this.rows = Collections.unmodifiableList(new ArrayList<>(rows));
        this.createdAtMs = System.currentTimeMillis();
    }

    public static OnlineWriteBatch of(List<OnlineFeatureRow> rows) {
        return new OnlineWriteBatch(Objects.requireNonNull(rows, "rows"));
    }

    public static OnlineWriteBatch of(OnlineFeatureRow... rows) {
        List<OnlineFeatureRow> list = new ArrayList<>();
        if (rows != null) {
            for (OnlineFeatureRow r : rows) {
                if (r != null) list.add(r);
            }
        }
        return new OnlineWriteBatch(list);
    }

    public static Builder builder() {
        return new Builder();
    }

    public List<OnlineFeatureRow> rows() {
        return rows;
    }

    public int size() {
        return rows.size();
    }

    public long createdAtMs() {
        return createdAtMs;
    }

    public static final class Builder {
        private final List<OnlineFeatureRow> rows = new ArrayList<>();

        public Builder add(OnlineFeatureRow row) {
            if (row != null) rows.add(row);
            return this;
        }

        public Builder addAll(List<OnlineFeatureRow> more) {
            if (more != null) rows.addAll(more);
            return this;
        }

        public OnlineWriteBatch build() {
            return new OnlineWriteBatch(rows);
        }
    }
}
