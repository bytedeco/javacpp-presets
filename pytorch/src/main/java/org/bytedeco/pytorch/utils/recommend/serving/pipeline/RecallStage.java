/*
 * Multi-channel recall stage.
 *
 * Production recall (TikTok, Taobao, YouTube, Meta):
 *   - Multiple channels run in parallel: u2i, i2i, hot, geo, graph, ANN (DSSM/YouTubeDNN)
 *   - Per-channel quota + global merge dedup
 *   - Channel-level timeout isolation (one slow channel must not block all)
 *   - Experiment params switch channel on/off and quotas
 *
 * This class provides the orchestration skeleton; channel retrieval is pluggable
 * via {@link RecallChannel}.
 */
package org.bytedeco.pytorch.utils.recommend.serving.pipeline;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;

/** Recall stage with multi-channel fan-out. */
public final class RecallStage implements RankStage {

    /** Pluggable recall channel (ANN, inverted index, hot list, ...). */
    public interface RecallChannel {
        String name();

        /**
         * Retrieve candidates for this channel.
         *
         * @param ctx   request context
         * @param quota max items to return from this channel
         */
        List<Candidate> retrieve(RequestContext ctx, int quota) throws Exception;
    }

    private final List<RecallChannel> channels;
    private final int defaultPerChannelQuota;
    private final int defaultTotalQuota;
    private final long defaultChannelTimeoutMs;
    private final ExecutorService executor;
    private final boolean ownsExecutor;

    public RecallStage(List<RecallChannel> channels) {
        this(channels, 200, 1000, 50L, null);
    }

    public RecallStage(
            List<RecallChannel> channels,
            int defaultPerChannelQuota,
            int defaultTotalQuota,
            long defaultChannelTimeoutMs,
            ExecutorService executor) {
        this.channels = new ArrayList<>(Objects.requireNonNull(channels, "channels"));
        this.defaultPerChannelQuota = defaultPerChannelQuota;
        this.defaultTotalQuota = defaultTotalQuota;
        this.defaultChannelTimeoutMs = defaultChannelTimeoutMs;
        if (executor != null) {
            this.executor = executor;
            this.ownsExecutor = false;
        } else {
            this.executor = Executors.newCachedThreadPool(r -> {
                Thread t = new Thread(r, "recall-channel");
                t.setDaemon(true);
                return t;
            });
            this.ownsExecutor = true;
        }
    }

    @Override
    public String name() {
        return "recall";
    }

    @Override
    public StageResult execute(RequestContext ctx, List<Candidate> input) {
        long t0 = System.currentTimeMillis();
        if (ctx.deadlineExceeded()) {
            return StageResult.timeout(name(), input == null ? List.of() : input, 0L);
        }
        int totalQuota = ctx.expParamInt("recall.total_quota", defaultTotalQuota);
        int perQuota = ctx.expParamInt("recall.per_channel_quota", defaultPerChannelQuota);
        long channelTimeout = ctx.expParamInt("recall.channel_timeout_ms", (int) defaultChannelTimeoutMs);

        // Optional channel allow-list from experiment: "recall.channels=u2i,i2i,hot"
        String allow = ctx.expParam("recall.channels", "");
        List<RecallChannel> active = new ArrayList<>();
        for (RecallChannel ch : channels) {
            String enabledKey = "recall.channel." + ch.name() + ".enabled";
            String en = ctx.expParam(enabledKey, "true");
            if (!"true".equalsIgnoreCase(en)) {
                continue;
            }
            if (!allow.isEmpty()) {
                boolean ok = false;
                for (String a : allow.split(",")) {
                    if (a.trim().equals(ch.name())) {
                        ok = true;
                        break;
                    }
                }
                if (!ok) continue;
            }
            active.add(ch);
        }

        Map<String, Candidate> merged = new LinkedHashMap<>();
        List<Future<ChannelOutcome>> futures = new ArrayList<>();
        for (RecallChannel ch : active) {
            int q = ctx.expParamInt("recall.channel." + ch.name() + ".quota", perQuota);
            Callable<ChannelOutcome> task = () -> {
                try {
                    List<Candidate> got = ch.retrieve(ctx, q);
                    return new ChannelOutcome(ch.name(), got, null);
                } catch (Exception ex) {
                    return new ChannelOutcome(ch.name(), List.of(), ex);
                }
            };
            futures.add(executor.submit(task));
        }

        int successChannels = 0;
        int failedChannels = 0;
        for (int i = 0; i < futures.size(); i++) {
            Future<ChannelOutcome> f = futures.get(i);
            ChannelOutcome outcome;
            try {
                long remain = Math.max(1L, ctx.remainingBudgetMs());
                long wait = Math.min(channelTimeout, remain);
                outcome = f.get(wait, TimeUnit.MILLISECONDS);
            } catch (TimeoutException te) {
                f.cancel(true);
                failedChannels++;
                continue;
            } catch (InterruptedException ie) {
                Thread.currentThread().interrupt();
                failedChannels++;
                continue;
            } catch (ExecutionException ee) {
                failedChannels++;
                continue;
            }
            if (outcome.error != null) {
                failedChannels++;
                continue;
            }
            successChannels++;
            for (Candidate c : outcome.candidates) {
                c.addRecallChannel(outcome.channelName);
                Candidate existing = merged.get(c.itemId());
                if (existing == null) {
                    merged.put(c.itemId(), c);
                } else {
                    // Merge channels; keep higher score.
                    existing.addRecallChannel(outcome.channelName);
                    if (c.score() > existing.score()) {
                        existing.score(c.score());
                    }
                }
            }
        }

        List<Candidate> out = new ArrayList<>(merged.values());
        // Sort by recall score descending then truncate.
        out.sort((a, b) -> Double.compare(b.score(), a.score()));
        if (out.size() > totalQuota) {
            out = new ArrayList<>(out.subList(0, totalQuota));
        }
        for (int i = 0; i < out.size(); i++) {
            out.get(i).rank(i);
            out.get(i).putScore("recall_score", out.get(i).score());
        }
        long latency = System.currentTimeMillis() - t0;
        boolean degraded = successChannels == 0 && !active.isEmpty();
        String msg = "channels_ok=" + successChannels + " fail=" + failedChannels
                + " merged=" + out.size();
        if (degraded) {
            // Ultimate fallback: empty list (caller pipeline may inject hot list).
            return StageResult.degraded(name(), out, msg);
        }
        return StageResult.ok(name(), out, latency);
    }

    public void shutdown() {
        if (ownsExecutor) {
            executor.shutdownNow();
        }
    }

    private static final class ChannelOutcome {
        final String channelName;
        final List<Candidate> candidates;
        final Exception error;

        ChannelOutcome(String channelName, List<Candidate> candidates, Exception error) {
            this.channelName = channelName;
            this.candidates = candidates;
            this.error = error;
        }
    }

    // ---- simple built-in channels for tests / demos -------------------------

    /** Static list channel (hot / operations force-insert pool). */
    public static RecallChannel staticChannel(String name, List<Candidate> fixed) {
        Objects.requireNonNull(fixed, "fixed");
        return new RecallChannel() {
            @Override
            public String name() {
                return name;
            }

            @Override
            public List<Candidate> retrieve(RequestContext ctx, int quota) {
                List<Candidate> out = new ArrayList<>();
                for (int i = 0; i < Math.min(quota, fixed.size()); i++) {
                    out.add(fixed.get(i).copy());
                }
                return out;
            }
        };
    }

    /** Function-based channel. */
    public static RecallChannel functional(String name, ChannelFn fn) {
        return new RecallChannel() {
            @Override
            public String name() {
                return name;
            }

            @Override
            public List<Candidate> retrieve(RequestContext ctx, int quota) throws Exception {
                return fn.retrieve(ctx, quota);
            }
        };
    }

    @FunctionalInterface
    public interface ChannelFn {
        List<Candidate> retrieve(RequestContext ctx, int quota) throws Exception;
    }
}
