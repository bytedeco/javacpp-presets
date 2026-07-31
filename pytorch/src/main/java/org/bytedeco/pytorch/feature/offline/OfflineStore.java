/*
 * Offline store SPI — batch historical feature retrieval
 * (Feast OfflineStore / Databricks Feature Store offline reads).
 */
package org.bytedeco.pytorch.feature.offline;

import org.bytedeco.pytorch.feature.core.FeatureView;

import java.time.Instant;
import java.util.List;
import java.util.Map;
import java.util.Optional;

/**
 * Offline feature storage and range reads.
 *
 * <p>Rows are represented as {@code Map&lt;String,Object&gt;} with at least entity join keys
 * and a timestamp column (default {@code event_timestamp} as epoch millis Long).
 */
public interface OfflineStore extends AutoCloseable {

    /** Upsert / append feature rows for a view (demo & materialize source). */
    void put(String project, String viewName, List<Map<String, Object>> rows);

    /** Replace all rows for a view. */
    void replace(String project, String viewName, List<Map<String, Object>> rows);

    /** All rows for a view (may be large — prefer {@link #readRange}). */
    List<Map<String, Object>> readAll(String project, String viewName);

    /**
     * Rows with timestamp in {@code [start, end]} inclusive (epoch millis on timestamp column).
     */
    List<Map<String, Object>> readRange(String project, String viewName,
                                        Instant start, Instant end,
                                        String timestampColumn);

    default List<Map<String, Object>> readRange(FeatureView view, Instant start, Instant end) {
        String ts = view.source() != null ? view.source().timestampColumn() : "event_timestamp";
        return readRange(view.project(), view.name(), start, end, ts);
    }

    Optional<Long> latestTimestamp(String project, String viewName, String timestampColumn);

    default Optional<Long> latestTimestamp(FeatureView view) {
        String ts = view.source() != null ? view.source().timestampColumn() : "event_timestamp";
        return latestTimestamp(view.project(), view.name(), ts);
    }

    long rowCount(String project, String viewName);

    @Override
    default void close() {}
}
