/*
 * Experiment unit / diversion key type.
 *
 * Industry practice:
 *   - Meta / Instagram: user_id primary, device_id for logged-out
 *   - TikTok / ByteDance: device_id + uid layered
 *   - Google Ads / YouTube: cookie / GAID / user
 *   - Alibaba / Taobao: user_id for logged-in, utdid for guest
 *   - Tencent: qimei / openid
 *
 * Mixing unit types in the same layer causes SRM and leakage.
 */
package org.bytedeco.pytorch.utils.recommend.abtest;

/** Diversion unit for experiment assignment. */
public enum DiversionUnit {
    /** Logged-in user identifier (primary for recsys personalization). */
    USER_ID,
    /** Device / install id (logged-out, cold start, app install experiments). */
    DEVICE_ID,
    /** Session id (short-lived UX experiments; not for long-horizon metrics). */
    SESSION_ID,
    /** Request id (pure request-level; no stickiness across requests). */
    REQUEST_ID,
    /** Page / scene id (page-level layout experiments). */
    PAGE_ID,
    /** Composite hash of multiple keys (e.g. user_id + scene). */
    COMPOSITE;

    public boolean stickyAcrossSessions() {
        return this == USER_ID || this == DEVICE_ID || this == COMPOSITE;
    }
}
