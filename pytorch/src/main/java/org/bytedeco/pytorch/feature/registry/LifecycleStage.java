/*
 * Feature lifecycle stages — mirror modelops ModelStage for feature assets.
 *
 * Happy path: DRAFT → VALIDATED → PROD → DEPRECATED → ARCHIVED
 */
package org.bytedeco.pytorch.feature.registry;

import java.util.EnumSet;
import java.util.Locale;
import java.util.Set;

/** Lifecycle stage for registered feature resources. */
public enum LifecycleStage {
    DRAFT,
    VALIDATED,
    PROD,
    DEPRECATED,
    ARCHIVED;

    private static final Set<LifecycleStage> FROM_DRAFT = EnumSet.of(VALIDATED, ARCHIVED);
    private static final Set<LifecycleStage> FROM_VALIDATED = EnumSet.of(PROD, DRAFT, ARCHIVED);
    private static final Set<LifecycleStage> FROM_PROD = EnumSet.of(DEPRECATED, ARCHIVED);
    private static final Set<LifecycleStage> FROM_DEPRECATED = EnumSet.of(PROD, ARCHIVED);
    private static final Set<LifecycleStage> FROM_ARCHIVED = EnumSet.noneOf(LifecycleStage.class);

    public boolean canTransitionTo(LifecycleStage to) {
        if (to == null || to == this) return false;
        switch (this) {
            case DRAFT:
                return FROM_DRAFT.contains(to);
            case VALIDATED:
                return FROM_VALIDATED.contains(to);
            case PROD:
                return FROM_PROD.contains(to);
            case DEPRECATED:
                return FROM_DEPRECATED.contains(to);
            case ARCHIVED:
                return FROM_ARCHIVED.contains(to);
            default:
                return false;
        }
    }

    /** One-step promote along happy path; null if terminal. */
    public LifecycleStage nextPromote() {
        switch (this) {
            case DRAFT:
                return VALIDATED;
            case VALIDATED:
                return PROD;
            case PROD:
                return DEPRECATED;
            case DEPRECATED:
                return ARCHIVED;
            default:
                return null;
        }
    }

    public static LifecycleStage parse(String raw) {
        if (raw == null || raw.isEmpty()) return DRAFT;
        return LifecycleStage.valueOf(raw.trim().toUpperCase(Locale.ROOT));
    }
}
