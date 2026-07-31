/*
 * Gateway / edge traffic routing for recommendation services.
 *
 * Responsibilities at L7 gateway (Envoy, Istio, Nginx, API Gateway, MSE):
 *   - Percentage / weight split across service versions (canary, blue-green)
 *   - Sticky session by userId / deviceId (consistent hash)
 *   - Header / cookie based routing (debug, internal dogfood, force canary)
 *   - Region / zone affinity
 *   - Shadow (mirrored) traffic to candidate without affecting response
 *
 * This package is pure routing logic; wire it into your actual gateway filter.
 */
package org.bytedeco.pytorch.deploy.serving.gateway;

import org.bytedeco.pytorch.deploy.abtest.BucketAssigner;
import org.bytedeco.pytorch.deploy.abtest.TrafficSplitter;

import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.atomic.AtomicLong;

/** L7 traffic router for recsys serving versions. */
public final class TrafficRouter {

    /** One upstream cluster / version. */
    public static final class Upstream {
        public final String id;
        public final String address; // host:port or logical name
        public volatile double weightPercent;
        public final boolean shadowOnly;

        public Upstream(String id, String address, double weightPercent) {
            this(id, address, weightPercent, false);
        }

        public Upstream(String id, String address, double weightPercent, boolean shadowOnly) {
            if (id == null || id.isEmpty()) throw new IllegalArgumentException("id required");
            this.id = id;
            this.address = address != null ? address : id;
            this.weightPercent = weightPercent;
            this.shadowOnly = shadowOnly;
        }

        @Override
        public String toString() {
            return id + "@" + address + "(" + weightPercent + "%" + (shadowOnly ? ",shadow" : "") + ")";
        }
    }

    /** Incoming request view used for routing decisions. */
    public static final class RouteRequest {
        public final String requestId;
        public final String userId;
        public final String deviceId;
        public final String path;
        public final String region;
        public final Map<String, String> headers;

        public RouteRequest(
                String requestId,
                String userId,
                String deviceId,
                String path,
                String region,
                Map<String, String> headers) {
            this.requestId = requestId != null ? requestId : "";
            this.userId = userId != null ? userId : "";
            this.deviceId = deviceId != null ? deviceId : "";
            this.path = path != null ? path : "/";
            this.region = region != null ? region : "";
            this.headers = headers == null
                    ? Collections.emptyMap()
                    : Collections.unmodifiableMap(new LinkedHashMap<>(headers));
        }

        public String stickyKey() {
            if (!userId.isEmpty()) return userId;
            if (!deviceId.isEmpty()) return deviceId;
            return requestId;
        }

        public String header(String name) {
            // case-insensitive lookup
            for (Map.Entry<String, String> e : headers.entrySet()) {
                if (e.getKey().equalsIgnoreCase(name)) return e.getValue();
            }
            return null;
        }
    }

    /** Routing decision. */
    public static final class RouteDecision {
        public final String primaryUpstreamId;
        public final String primaryAddress;
        public final List<String> shadowUpstreamIds;
        public final String reason;
        public final boolean forced;

        public RouteDecision(
                String primaryUpstreamId,
                String primaryAddress,
                List<String> shadowUpstreamIds,
                String reason,
                boolean forced) {
            this.primaryUpstreamId = primaryUpstreamId;
            this.primaryAddress = primaryAddress;
            this.shadowUpstreamIds = Collections.unmodifiableList(new ArrayList<>(
                    shadowUpstreamIds == null ? List.of() : shadowUpstreamIds));
            this.reason = reason;
            this.forced = forced;
        }

        @Override
        public String toString() {
            return "RouteDecision{primary=" + primaryUpstreamId
                    + ", shadow=" + shadowUpstreamIds + ", reason=" + reason + "}";
        }
    }

    /** Header-based force rule. */
    public static final class HeaderRule {
        public final String headerName;
        public final String headerValue; // null = any non-empty
        public final String upstreamId;

        public HeaderRule(String headerName, String headerValue, String upstreamId) {
            this.headerName = Objects.requireNonNull(headerName);
            this.headerValue = headerValue;
            this.upstreamId = Objects.requireNonNull(upstreamId);
        }
    }

    /** Region affinity rule: prefer upstream for a region. */
    public static final class RegionAffinity {
        public final String region;
        public final String upstreamId;

        public RegionAffinity(String region, String upstreamId) {
            this.region = Objects.requireNonNull(region);
            this.upstreamId = Objects.requireNonNull(upstreamId);
        }
    }

    private final String routeName;
    private final String salt;
    private final ConcurrentHashMap<String, Upstream> upstreams = new ConcurrentHashMap<>();
    private final CopyOnWriteArrayList<HeaderRule> headerRules = new CopyOnWriteArrayList<>();
    private final CopyOnWriteArrayList<RegionAffinity> regionAffinities = new CopyOnWriteArrayList<>();
    private final ConcurrentHashMap<String, AtomicLong> hitCounts = new ConcurrentHashMap<>();
    private volatile boolean sticky = true;

    public TrafficRouter(String routeName) {
        this(routeName, routeName);
    }

    public TrafficRouter(String routeName, String salt) {
        this.routeName = Objects.requireNonNull(routeName, "routeName");
        this.salt = salt != null ? salt : routeName;
    }

    public String routeName() {
        return routeName;
    }

    public TrafficRouter sticky(boolean sticky) {
        this.sticky = sticky;
        return this;
    }

    public void addUpstream(Upstream upstream) {
        upstreams.put(upstream.id, upstream);
    }

    public void addUpstream(String id, String address, double weightPercent) {
        addUpstream(new Upstream(id, address, weightPercent));
    }

    public void removeUpstream(String id) {
        upstreams.remove(id);
    }

    public Upstream getUpstream(String id) {
        return upstreams.get(id);
    }

    /**
     * Set traffic weights; non-shadow weights should roughly sum to 100.
     */
    public synchronized void setWeights(Map<String, Double> weights) {
        Objects.requireNonNull(weights, "weights");
        for (Map.Entry<String, Double> e : weights.entrySet()) {
            Upstream u = upstreams.get(e.getKey());
            if (u != null) {
                u.weightPercent = e.getValue();
            }
        }
    }

    /**
     * Canary helper: set canary percent, rest to stable.
     */
    public synchronized void setCanaryPercent(String stableId, String canaryId, double canaryPercent) {
        Upstream stable = requireUpstream(stableId);
        Upstream canary = requireUpstream(canaryId);
        double c = Math.max(0.0, Math.min(100.0, canaryPercent));
        canary.weightPercent = c;
        stable.weightPercent = 100.0 - c;
    }

    public void addHeaderRule(HeaderRule rule) {
        headerRules.add(Objects.requireNonNull(rule));
    }

    public void addHeaderRule(String headerName, String headerValue, String upstreamId) {
        addHeaderRule(new HeaderRule(headerName, headerValue, upstreamId));
    }

    public void addRegionAffinity(String region, String upstreamId) {
        regionAffinities.add(new RegionAffinity(region, upstreamId));
    }

    /**
     * Route one request.
     *
     * <p>Priority:
     * <ol>
     *   <li>Header force rules (dogfood / debug / internal)</li>
     *   <li>Region affinity if configured and weight allows</li>
     *   <li>Sticky or random weighted split among non-shadow upstreams</li>
     *   <li>Attach shadow upstreams for mirror</li>
     * </ol>
     */
    public RouteDecision route(RouteRequest request) {
        Objects.requireNonNull(request, "request");

        // 1) Header rules
        for (HeaderRule rule : headerRules) {
            String v = request.header(rule.headerName);
            if (v == null) continue;
            if (rule.headerValue == null || rule.headerValue.equals(v)) {
                Upstream u = upstreams.get(rule.upstreamId);
                if (u != null && !u.shadowOnly) {
                    hit(u.id);
                    return new RouteDecision(u.id, u.address, shadowIds(),
                            "header:" + rule.headerName, true);
                }
            }
        }

        // 2) Region affinity — soft preference: if region upstream has weight > 0 use it
        if (!request.region.isEmpty()) {
            for (RegionAffinity ra : regionAffinities) {
                if (ra.region.equalsIgnoreCase(request.region)) {
                    Upstream u = upstreams.get(ra.upstreamId);
                    if (u != null && !u.shadowOnly && u.weightPercent > 0) {
                        hit(u.id);
                        return new RouteDecision(u.id, u.address, shadowIds(),
                                "region:" + request.region, false);
                    }
                }
            }
        }

        // 3) Weighted split
        List<TrafficSplitter.WeightedTarget> targets = new ArrayList<>();
        Map<String, Upstream> primary = new LinkedHashMap<>();
        for (Upstream u : upstreams.values()) {
            if (u.shadowOnly) continue;
            if (u.weightPercent <= 0) continue;
            targets.add(new TrafficSplitter.WeightedTarget(u.id, u.weightPercent));
            primary.put(u.id, u);
        }
        if (targets.isEmpty()) {
            throw new IllegalStateException("no routable upstream with positive weight on " + routeName);
        }
        String chosenId;
        if (sticky) {
            chosenId = TrafficSplitter.selectSticky(request.stickyKey(), salt, targets);
        } else {
            chosenId = TrafficSplitter.selectRandom(targets);
        }
        Upstream chosen = primary.get(chosenId);
        hit(chosenId);
        return new RouteDecision(chosen.id, chosen.address, shadowIds(),
                sticky ? "sticky_weight" : "random_weight", false);
    }

    /**
     * Consistent-hash ring style select among upstreams (equal weight),
     * useful for session affinity to cache-warm rankers.
     */
    public String consistentHash(String key, List<String> upstreamIds) {
        if (upstreamIds == null || upstreamIds.isEmpty()) {
            throw new IllegalArgumentException("upstreamIds empty");
        }
        long bucket = BucketAssigner.bucketOf(salt, key, upstreamIds.size() * 1000L);
        int idx = (int) (bucket % upstreamIds.size());
        return upstreamIds.get(idx);
    }

    public Map<String, Long> hitCounts() {
        Map<String, Long> m = new LinkedHashMap<>();
        for (Map.Entry<String, AtomicLong> e : hitCounts.entrySet()) {
            m.put(e.getKey(), e.getValue().get());
        }
        return m;
    }

    public Map<String, Double> currentWeights() {
        Map<String, Double> m = new LinkedHashMap<>();
        for (Upstream u : upstreams.values()) {
            m.put(u.id, u.weightPercent);
        }
        return m;
    }

    public String explain() {
        StringBuilder sb = new StringBuilder();
        sb.append("TrafficRouter{name=").append(routeName).append(" sticky=").append(sticky).append('\n');
        for (Upstream u : upstreams.values()) {
            sb.append("  ").append(u).append('\n');
        }
        sb.append(String.format(Locale.ROOT, "  headerRules=%d regionRules=%d\n",
                headerRules.size(), regionAffinities.size()));
        sb.append('}');
        return sb.toString();
    }

    private List<String> shadowIds() {
        List<String> shadows = new ArrayList<>();
        for (Upstream u : upstreams.values()) {
            if (u.shadowOnly && u.weightPercent > 0) {
                shadows.add(u.id);
            }
        }
        return shadows;
    }

    private void hit(String id) {
        hitCounts.computeIfAbsent(id, k -> new AtomicLong()).incrementAndGet();
    }

    private Upstream requireUpstream(String id) {
        Upstream u = upstreams.get(id);
        if (u == null) throw new IllegalArgumentException("unknown upstream: " + id);
        return u;
    }

    /**
     * Percentage-based path splitter (e.g. /recommend vs /recommend-canary rewrite).
     * Returns chosen path suffix label.
     */
    public static String splitByPercent(String key, String salt, double percent,
                                        String controlLabel, String treatmentLabel) {
        return TrafficSplitter.selectByPercent(key, salt, percent, controlLabel, treatmentLabel);
    }

    /**
     * Hash a header value the same way Envoy ring hash roughly does (for tests).
     */
    public static int headerHash(String value) {
        if (value == null) value = "";
        return BucketAssigner.murmur3_32(value.getBytes(StandardCharsets.UTF_8), BucketAssigner.DEFAULT_SEED);
    }
}
