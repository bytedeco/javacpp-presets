package org.bytedeco.pytorch.dataframe.geo;

/**
 * Spatial relationship predicates (OGC simple features subset).
 */
public enum SpatialPredicate {
    WITHIN,
    INTERSECTS,
    CONTAINS,
    DISJOINT,
    TOUCHES,
    CROSSES,
    EQUALS,
    /** Distance ≤ tolerance (meters for geographic approx via haversine on points). */
    DWITHIN
}
