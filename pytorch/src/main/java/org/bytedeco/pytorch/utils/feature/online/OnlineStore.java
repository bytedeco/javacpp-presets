/*
 * Online store SPI — low-latency KV feature serving
 * (Feast OnlineStore / Redis / DynamoDB pattern).
 */
package org.bytedeco.pytorch.utils.feature.online;

import java.util.Collection;
import java.util.List;
import java.util.Map;
import java.util.Optional;

/** Online feature storage for serving path. */
public interface OnlineStore extends AutoCloseable {

    void onlineWrite(OnlineWriteBatch batch);

    default void onlineWrite(List<OnlineFeatureRow> rows) {
        onlineWrite(OnlineWriteBatch.of(rows));
    }

    Optional<OnlineFeatureRow> onlineRead(String project, String viewName, String entityKey);

    Map<String, OnlineFeatureRow> onlineReadBatch(String project, String viewName,
                                                  Collection<String> entityKeys);

    long size(String project, String viewName);

    void delete(String project, String viewName, String entityKey);

    void clearView(String project, String viewName);

    @Override
    default void close() {}
}
