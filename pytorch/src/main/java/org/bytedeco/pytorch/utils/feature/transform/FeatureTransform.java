/*
 * Feature transform SPI — batch/stream feature computation descriptor.
 */
package org.bytedeco.pytorch.utils.feature.transform;

import java.util.List;
import java.util.Map;

/** Transform that maps input rows → output feature columns. */
@FunctionalInterface
public interface FeatureTransform {

    /**
     * Apply transform over input rows.
     *
     * @param rows input event / base feature rows
     * @return output rows (may be aggregated — fewer rows than input)
     */
    List<Map<String, Object>> apply(List<Map<String, Object>> rows);

    default String name() {
        return getClass().getSimpleName();
    }
}
