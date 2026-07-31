/*
 * Result of a materialization job (offline → online).
 */
package org.bytedeco.pytorch.feature.materialize;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/** Immutable materialization outcome. */
public final class MaterializationResult {

    private final String jobId;
    private final List<String> viewNames;
    private final long rowsRead;
    private final long rowsWritten;
    private final long entitiesTouched;
    private final long startMs;
    private final long endMs;
    private final long elapsedNanos;
    private final boolean success;
    private final String error;
    private final Map<String, Long> perViewWritten;
    private final long watermarkMs;

    private MaterializationResult(Builder b) {
        this.jobId = b.jobId != null ? b.jobId : "";
        this.viewNames = Collections.unmodifiableList(new ArrayList<>(b.viewNames));
        this.rowsRead = b.rowsRead;
        this.rowsWritten = b.rowsWritten;
        this.entitiesTouched = b.entitiesTouched;
        this.startMs = b.startMs;
        this.endMs = b.endMs;
        this.elapsedNanos = b.elapsedNanos;
        this.success = b.success;
        this.error = b.error != null ? b.error : "";
        this.perViewWritten = Collections.unmodifiableMap(new LinkedHashMap<>(b.perViewWritten));
        this.watermarkMs = b.watermarkMs;
    }

    public static Builder builder() {
        return new Builder();
    }

    public String jobId() { return jobId; }
    public List<String> viewNames() { return viewNames; }
    public long rowsRead() { return rowsRead; }
    public long rowsWritten() { return rowsWritten; }
    public long entitiesTouched() { return entitiesTouched; }
    public long startMs() { return startMs; }
    public long endMs() { return endMs; }
    public long elapsedNanos() { return elapsedNanos; }
    public double elapsedMs() { return elapsedNanos / 1_000_000.0; }
    public boolean success() { return success; }
    public String error() { return error; }
    public Map<String, Long> perViewWritten() { return perViewWritten; }
    public long watermarkMs() { return watermarkMs; }

    @Override
    public String toString() {
        return "MaterializationResult{job=" + jobId
                + ", views=" + viewNames
                + ", read=" + rowsRead
                + ", written=" + rowsWritten
                + ", entities=" + entitiesTouched
                + ", ok=" + success
                + ", ms=" + elapsedMs()
                + (error.isEmpty() ? "" : ", err=" + error)
                + "}";
    }

    public static final class Builder {
        private String jobId;
        private final List<String> viewNames = new ArrayList<>();
        private long rowsRead;
        private long rowsWritten;
        private long entitiesTouched;
        private long startMs;
        private long endMs;
        private long elapsedNanos;
        private boolean success = true;
        private String error;
        private final Map<String, Long> perViewWritten = new LinkedHashMap<>();
        private long watermarkMs;

        public Builder jobId(String jobId) { this.jobId = jobId; return this; }
        public Builder viewNames(List<String> names) {
            this.viewNames.clear();
            if (names != null) this.viewNames.addAll(names);
            return this;
        }
        public Builder rowsRead(long rowsRead) { this.rowsRead = rowsRead; return this; }
        public Builder rowsWritten(long rowsWritten) { this.rowsWritten = rowsWritten; return this; }
        public Builder entitiesTouched(long entitiesTouched) { this.entitiesTouched = entitiesTouched; return this; }
        public Builder startMs(long startMs) { this.startMs = startMs; return this; }
        public Builder endMs(long endMs) { this.endMs = endMs; return this; }
        public Builder elapsedNanos(long elapsedNanos) { this.elapsedNanos = elapsedNanos; return this; }
        public Builder success(boolean success) { this.success = success; return this; }
        public Builder error(String error) { this.error = error; return this; }
        public Builder perViewWritten(String view, long n) { perViewWritten.put(view, n); return this; }
        public Builder watermarkMs(long watermarkMs) { this.watermarkMs = watermarkMs; return this; }

        public MaterializationResult build() {
            return new MaterializationResult(this);
        }
    }
}
