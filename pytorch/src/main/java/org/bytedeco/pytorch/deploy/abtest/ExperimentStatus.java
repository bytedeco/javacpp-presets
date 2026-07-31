/*
 * Online experiment lifecycle states used by Meta XP, ByteDance Libra,
 * Google Experiment Framework and Alibaba A/B platforms.
 *
 * Typical flow:
 *   DRAFT -> REVIEW -> AA_RUNNING -> RUNNING -> PAUSED -> COMPLETED
 *                                      \-> ROLLED_BACK / KILLED
 */
package org.bytedeco.pytorch.deploy.abtest;

/** Lifecycle state of an online recommendation experiment. */
public enum ExperimentStatus {
    /** Spec only; not exposed to traffic. */
    DRAFT,
    /** Waiting for design review / SRM pre-check / guardrail review. */
    REVIEW,
    /** A/A (null) test phase to detect SRM / metric bias before real treatment. */
    AA_RUNNING,
    /** Live traffic assignment; treatment variants receiving traffic. */
    RUNNING,
    /** Temporarily no new assignments; existing sticky users keep bucket. */
    PAUSED,
    /** Experiment ended; decision recorded (ship / no-ship). */
    COMPLETED,
    /** Guardrail breach or operator kill; traffic forced to control. */
    KILLED,
    /** Full traffic rolled back to control / previous stable version. */
    ROLLED_BACK;

    public boolean acceptsTraffic() {
        return this == AA_RUNNING || this == RUNNING;
    }

    public boolean isTerminal() {
        return this == COMPLETED || this == KILLED || this == ROLLED_BACK;
    }
}
