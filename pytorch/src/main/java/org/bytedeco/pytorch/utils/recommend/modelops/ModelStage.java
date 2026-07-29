/*
 * Model lifecycle stages for recommendation modelops.
 *
 * TRAINED -> OFFLINE_PASS -> SHADOW -> CANARY -> PROD -> ARCHIVED
 *                                    \-> REJECTED
 */
package org.bytedeco.pytorch.utils.recommend.modelops;

/** Model lifecycle stage. */
public enum ModelStage {
    /** Training finished; artifact registered. */
    TRAINED,
    /** Passed offline evaluation ship bar. */
    OFFLINE_PASS,
    /** Receiving mirrored traffic; not serving users. */
    SHADOW,
    /** Partial online traffic via canary / experiment. */
    CANARY,
    /** Full production serving. */
    PROD,
    /** Rolled back or superseded; kept for audit. */
    ARCHIVED,
    /** Failed validation; do not promote. */
    REJECTED
}
