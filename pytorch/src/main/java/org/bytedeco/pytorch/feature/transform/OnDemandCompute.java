/*
 * On-demand compute helpers — request-time pure functions used by OnDemandFeatureView.
 */
package org.bytedeco.pytorch.feature.transform;

import org.bytedeco.pytorch.feature.core.OnDemandFeatureView;

import java.time.Instant;
import java.time.ZoneOffset;
import java.time.ZonedDateTime;
import java.util.HashMap;
import java.util.Map;

/** Common request-time feature computations used in ranking / CTR systems. */
public final class OnDemandCompute {

    private OnDemandCompute() {}

    /** hour_of_day + day_of_week from request epoch millis or "now". */
    public static OnDemandFeatureView.ComputeFn timeContext(String tsKey) {
        return (req, sources) -> {
            Map<String, Object> out = new HashMap<>();
            long ts = System.currentTimeMillis();
            Object raw = req.get(tsKey != null ? tsKey : "request_ts");
            if (raw instanceof Number) ts = ((Number) raw).longValue();
            ZonedDateTime zdt = Instant.ofEpochMilli(ts).atZone(ZoneOffset.UTC);
            out.put("hour_of_day", (long) zdt.getHour());
            out.put("day_of_week", (long) zdt.getDayOfWeek().getValue());
            out.put("is_weekend", zdt.getDayOfWeek().getValue() >= 6 ? 1L : 0L);
            return out;
        };
    }

    /** price_diff = request.price - item.avg_price (sources view "item_stats"). */
    public static OnDemandFeatureView.ComputeFn priceDiff(String requestPriceKey,
                                                         String itemView,
                                                         String itemAvgPriceKey,
                                                         String outputKey) {
        return (req, sources) -> {
            Map<String, Object> out = new HashMap<>();
            double reqPrice = toDouble(req.get(requestPriceKey));
            Map<String, Object> item = sources.getOrDefault(itemView, Map.of());
            double avg = toDouble(item.get(itemAvgPriceKey));
            out.put(outputKey != null ? outputKey : "price_diff", reqPrice - avg);
            out.put("price_ratio", avg == 0.0 ? 0.0 : reqPrice / avg);
            return out;
        };
    }

    /** Cross: hash bucket of user_id x item_id (simple multiplicative hash). */
    public static OnDemandFeatureView.ComputeFn crossHash(String userKey, String itemKey,
                                                         String outputKey, long buckets) {
        long b = Math.max(2L, buckets);
        return (req, sources) -> {
            Map<String, Object> out = new HashMap<>();
            long u = toLong(req.get(userKey));
            // try entity-like keys in sources first view values — also check req
            long i = toLong(req.get(itemKey));
            long h = Math.floorMod(u * 1315423911L ^ i * 2654435761L, b);
            out.put(outputKey != null ? outputKey : "cross_bucket", h);
            return out;
        };
    }

    /** Compose multiple compute fns (later wins on key conflict). */
    public static OnDemandFeatureView.ComputeFn compose(OnDemandFeatureView.ComputeFn... fns) {
        return (req, sources) -> {
            Map<String, Object> out = new HashMap<>();
            if (fns != null) {
                for (OnDemandFeatureView.ComputeFn fn : fns) {
                    if (fn == null) continue;
                    Map<String, Object> part = fn.apply(req, sources);
                    if (part != null) out.putAll(part);
                }
            }
            return out;
        };
    }

    private static double toDouble(Object v) {
        if (v instanceof Number) return ((Number) v).doubleValue();
        if (v instanceof String) {
            try { return Double.parseDouble((String) v); } catch (Exception e) { return 0.0; }
        }
        return 0.0;
    }

    private static long toLong(Object v) {
        if (v instanceof Number) return ((Number) v).longValue();
        if (v instanceof String) {
            try { return Long.parseLong((String) v); } catch (Exception e) { return v.hashCode(); }
        }
        return v == null ? 0L : v.hashCode();
    }
}
