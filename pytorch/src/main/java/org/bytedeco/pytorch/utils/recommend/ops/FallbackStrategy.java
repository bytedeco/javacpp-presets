/*
 * Fallback strategies when ranking cascade fails or degrades to emergency.
 *
 * Common production fallbacks:
 *   1. Hot / trending list (global or regional)
 *   2. Category default list
 *   3. User last-N history re-play
 *   4. Operations force-insert pool
 *   5. Static bootstrap JSON embedded in client (last resort)
 */
package org.bytedeco.pytorch.utils.recommend.ops;

import org.bytedeco.pytorch.utils.recommend.serving.pipeline.Candidate;
import org.bytedeco.pytorch.utils.recommend.serving.pipeline.RequestContext;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.function.Function;

/** Fallback provider chain. */
public final class FallbackStrategy {

    /** One fallback source. */
    public interface FallbackSource {
        String name();

        /**
         * @return candidates or empty if this source cannot serve
         */
        List<Candidate> supply(RequestContext ctx, int limit);
    }

    private final CopyOnWriteArrayList<FallbackSource> sources = new CopyOnWriteArrayList<>();
    private final ConcurrentHashMap<String, Long> hitCounts = new ConcurrentHashMap<>();

    public FallbackStrategy addSource(FallbackSource source) {
        sources.add(Objects.requireNonNull(source));
        return this;
    }

    public FallbackStrategy addHotList(String name, List<Candidate> items) {
        List<Candidate> fixed = Collections.unmodifiableList(new ArrayList<>(items));
        return addSource(new FallbackSource() {
            @Override
            public String name() {
                return name;
            }

            @Override
            public List<Candidate> supply(RequestContext ctx, int limit) {
                List<Candidate> out = new ArrayList<>();
                for (int i = 0; i < Math.min(limit, fixed.size()); i++) {
                    out.add(fixed.get(i).copy().tag("fallback", name));
                }
                return out;
            }
        });
    }

    public FallbackStrategy addFunctional(String name, Function<RequestContext, List<Candidate>> fn) {
        return addSource(new FallbackSource() {
            @Override
            public String name() {
                return name;
            }

            @Override
            public List<Candidate> supply(RequestContext ctx, int limit) {
                List<Candidate> got = fn.apply(ctx);
                if (got == null || got.isEmpty()) return List.of();
                List<Candidate> out = new ArrayList<>();
                for (int i = 0; i < Math.min(limit, got.size()); i++) {
                    out.add(got.get(i).copy().tag("fallback", name));
                }
                return out;
            }
        });
    }

    /**
     * Try sources in order until one returns non-empty result.
     */
    public List<Candidate> supply(RequestContext ctx, int limit) {
        for (FallbackSource src : sources) {
            try {
                List<Candidate> got = src.supply(ctx, limit);
                if (got != null && !got.isEmpty()) {
                    hitCounts.merge(src.name(), 1L, Long::sum);
                    return got;
                }
            } catch (RuntimeException ignored) {
                // try next
            }
        }
        return List.of();
    }

    /**
     * Merge first non-empty from each source up to limit (multi-source blend).
     */
    public List<Candidate> supplyMerged(RequestContext ctx, int limit) {
        Map<String, Candidate> merged = new LinkedHashMap<>();
        for (FallbackSource src : sources) {
            if (merged.size() >= limit) break;
            try {
                List<Candidate> got = src.supply(ctx, limit);
                if (got == null) continue;
                for (Candidate c : got) {
                    if (merged.size() >= limit) break;
                    merged.putIfAbsent(c.itemId(), c.copy().tag("fallback", src.name()));
                }
                if (!got.isEmpty()) {
                    hitCounts.merge(src.name(), 1L, Long::sum);
                }
            } catch (RuntimeException ignored) {
            }
        }
        return new ArrayList<>(merged.values());
    }

    public Map<String, Long> hitCounts() {
        return new LinkedHashMap<>(hitCounts);
    }

    public List<String> sourceNames() {
        List<String> names = new ArrayList<>();
        for (FallbackSource s : sources) names.add(s.name());
        return names;
    }
}
