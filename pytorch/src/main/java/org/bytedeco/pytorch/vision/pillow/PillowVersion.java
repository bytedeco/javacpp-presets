package org.bytedeco.pytorch.vision.pillow;

/**
 * Module version aligned with the upstream Pillow reference used for API mapping.
 * Runtime code must not depend on CPython; this is documentation + Features only.
 */
public final class PillowVersion {
    private PillowVersion() {}

    /** Java module version string for this port. */
    public static final String VERSION = "0.1.0-p1";

    /** Upstream Pillow reference tag / version this API maps against. */
    public static final String UPSTREAM_REF = "13.0.0.dev0";

    public static String version() {
        return VERSION;
    }

    public static String upstream_ref() {
        return UPSTREAM_REF;
    }

    public static String upstreamRef() {
        return UPSTREAM_REF;
    }
}
