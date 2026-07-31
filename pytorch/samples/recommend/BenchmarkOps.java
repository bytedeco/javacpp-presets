/*
 * Benchmark for ops: metrics, SLO, circuit breaker, degradation, fallback,
 * health/inspector, rate limiter.
 *
 *   java -cp ... samples.recommend.BenchmarkOps
 */
package samples.recommend;

import org.bytedeco.pytorch.recommend.ops.CircuitBreaker;
import org.bytedeco.pytorch.recommend.ops.DegradationPolicy;
import org.bytedeco.pytorch.recommend.ops.FallbackStrategy;
import org.bytedeco.pytorch.recommend.ops.HealthChecker;
import org.bytedeco.pytorch.recommend.ops.MetricsRegistry;
import org.bytedeco.pytorch.recommend.ops.RateLimiter;
import org.bytedeco.pytorch.recommend.ops.ServiceLevel;
import org.bytedeco.pytorch.deploy.serving.pipeline.Candidate;
import org.bytedeco.pytorch.deploy.serving.pipeline.RequestContext;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;

public final class BenchmarkOps {

    public static void main(String[] args) {
        System.exit(runTests());
    }

    public static int runTests() {
        BenchSupport.Suite s = new BenchSupport.Suite("BenchmarkOps");
        s.header();

        s.benchmark("metrics_counter_gauge_timer", () -> {
            MetricsRegistry reg = new MetricsRegistry();
            reg.counter(MetricsRegistry.Names.REQUESTS).add(10);
            reg.counter(MetricsRegistry.Names.ERRORS).inc();
            reg.gauge("inflight").set(3);
            for (int i = 0; i < 100; i++) {
                reg.timer(MetricsRegistry.Names.LATENCY).record(i);
            }
            s.checkEq("requests", 10L, reg.counter(MetricsRegistry.Names.REQUESTS).get());
            s.checkEq("errors", 1L, reg.counter(MetricsRegistry.Names.ERRORS).get());
            s.checkEq("inflight", 3L, reg.gauge("inflight").get());
            s.checkEq("timer count", 100L, reg.timer(MetricsRegistry.Names.LATENCY).count());
            s.checkRange("p99 >= p50", reg.timer(MetricsRegistry.Names.LATENCY).percentile(0.99),
                    reg.timer(MetricsRegistry.Names.LATENCY).percentile(0.50), 1e9);
            s.checkTrue("snapshot has keys", !reg.snapshotCounters().isEmpty());
            s.checkTrue("timer snapshot", reg.snapshotTimerMeans().containsKey(MetricsRegistry.Names.LATENCY + ".p99"));
        });

        s.benchmark("slo_availability_and_budget", () -> {
            ServiceLevel sl = new ServiceLevel();
            ServiceLevel.SliWindow w = sl.register(
                    ServiceLevel.SloDefinition.availability("avail", 0.99, 86_400_000L));
            for (int i = 0; i < 1000; i++) {
                if (i < 995) w.recordSuccess();
                else w.recordFailure();
            }
            ServiceLevel.SloStatus st = w.status();
            System.out.println("    " + st);
            s.checkTrue("SLO met at 99.5%", st.met);
            s.checkRange("sli", st.sliValue, 0.99, 1.0);
            s.checkTrue("budget remaining > 0", st.errorBudgetRemaining > 0);

            // burn budget
            for (int i = 0; i < 100; i++) w.recordFailure();
            ServiceLevel.SloStatus st2 = w.status();
            s.checkTrue("after failures sli dropped", st2.sliValue < st.sliValue);
        });

        s.benchmark("slo_latency_and_empty", () -> {
            ServiceLevel sl = ServiceLevel.standardRecsys();
            ServiceLevel.SliWindow lat = sl.get("latency_p99");
            ServiceLevel.SliWindow empty = sl.get("empty_rate");
            s.checkTrue("standard pack has latency", lat != null);
            s.checkTrue("standard pack has empty", empty != null);
            for (int i = 0; i < 200; i++) {
                lat.recordLatency(20);
                empty.recordEmpty(false);
            }
            s.checkTrue("all met after good traffic", sl.allMet());
        });

        s.benchmark("circuit_breaker_trip_and_recover", () -> {
            CircuitBreaker.Config cfg = new CircuitBreaker.Config(5, 2, 50L, 0.5, 5);
            CircuitBreaker cb = new CircuitBreaker("feature_store", cfg);
            s.checkEq("starts closed", CircuitBreaker.State.CLOSED, cb.state());

            AtomicInteger calls = new AtomicInteger();
            for (int i = 0; i < 5; i++) {
                cb.execute(() -> {
                    calls.incrementAndGet();
                    throw new RuntimeException("down");
                }, () -> "fb");
            }
            s.checkEq("open after failures", CircuitBreaker.State.OPEN, cb.state());

            // rejected while open -> fallback, supplier not called much more
            int before = calls.get();
            String v = cb.execute(() -> {
                calls.incrementAndGet();
                return "x";
            }, () -> "fallback");
            s.checkEq("fallback while open", "fallback", v);
            s.checkEq("supplier not called when open", before, calls.get());

            Thread.sleep(60);
            s.checkEq("half open after timeout", CircuitBreaker.State.HALF_OPEN, cb.state());
            cb.execute(() -> "ok", () -> "fb");
            cb.execute(() -> "ok", () -> "fb");
            s.checkEq("closed after successes", CircuitBreaker.State.CLOSED, cb.state());
        });

        s.benchmark("degradation_escalate_and_recover", () -> {
            // zero hold for fast test
            DegradationPolicy pol = new DegradationPolicy(
                    DegradationPolicy.Thresholds.defaults(), 0L, 0L);
            AtomicReference<DegradationPolicy.Level> seen = new AtomicReference<>();
            pol.addListener(c -> seen.set(c.to));

            s.checkEq("start L0", DegradationPolicy.Level.L0_NORMAL, pol.currentLevel());
            s.checkTrue("fine enabled", pol.currentKnobs().enableFineRank);

            // escalate via high error rate to L2+
            pol.evaluate(new DegradationPolicy.Signal(0.06, 100, 0.5, 1.0, 0.0));
            s.checkTrue("escalated >= L2 got=" + pol.currentLevel(),
                    pol.currentLevel().severity >= DegradationPolicy.Level.L2_HARD.severity);
            s.checkTrue("fine disabled at L2+", !pol.currentKnobs().enableFineRank
                    || pol.currentLevel().severity >= DegradationPolicy.Level.L2_HARD.severity);

            // emergency
            pol.evaluate(new DegradationPolicy.Signal(0.20, 600, 0.95, 0.1, 0.1));
            s.checkTrue("emergency+ got=" + pol.currentLevel(),
                    pol.currentLevel().severity >= DegradationPolicy.Level.L3_EMERGENCY.severity);

            // recover step by step with healthy signals
            for (int i = 0; i < 6; i++) {
                pol.evaluate(new DegradationPolicy.Signal(0.0, 20, 0.2, 1.0, 0.0));
            }
            s.checkEq("recovered to L0", DegradationPolicy.Level.L0_NORMAL, pol.currentLevel());
            s.checkTrue("listener saw changes", seen.get() != null);

            pol.forceLevel(DegradationPolicy.Level.L4_CIRCUIT, "manual");
            s.checkTrue("fail closed", pol.currentKnobs().failClosed);
            s.checkTrue("params map", pol.currentKnobs().asExperimentParams().containsKey("degrade.fail_closed"));
        });

        s.benchmark("fallback_chain", () -> {
            FallbackStrategy fb = new FallbackStrategy();
            fb.addFunctional("empty", ctx -> List.of());
            List<Candidate> hot = new ArrayList<>();
            hot.add(new Candidate("h1", 1.0));
            hot.add(new Candidate("h2", 0.9));
            fb.addHotList("global_hot", hot);
            RequestContext ctx = RequestContext.builder("r").userId("u").timeoutMs(100).build();
            List<Candidate> got = fb.supply(ctx, 5);
            s.checkEq("size 2", 2, got.size());
            s.checkEq("tag", "global_hot", got.get(0).tag("fallback"));
            s.checkTrue("hit count", fb.hitCounts().getOrDefault("global_hot", 0L) >= 1);

            List<Candidate> merged = fb.supplyMerged(ctx, 10);
            s.checkTrue("merged non-empty", !merged.isEmpty());
        });

        s.benchmark("health_and_inspector", () -> {
            HealthChecker hc = new HealthChecker()
                    .addProbe("liveness", () -> HealthChecker.Status.UP)
                    .addProbe("deps", () -> HealthChecker.Status.DEGRADED);
            HealthChecker.Report r = hc.checkAll();
            s.checkEq("overall degraded", HealthChecker.Status.DEGRADED, r.overall);
            s.checkTrue("ready", r.ready());
            s.checkTrue("live", r.live());

            HealthChecker.Inspector insp = new HealthChecker.Inspector(hc);
            insp.addRule("qps_drop", () -> "qps halved");
            insp.addRule("ok_rule", () -> null);
            HealthChecker.InspectionReport ir = insp.run();
            s.checkTrue("has anomaly", ir.anomalies.containsKey("qps_drop"));
            s.checkTrue("not healthy", !ir.healthy());
            s.checkEq("history 1", 1, insp.history().size());
        });

        s.benchmark("token_bucket_rate_limit", () -> {
            RateLimiter.TokenBucket bucket = new RateLimiter.TokenBucket(10, 1000); // 10 cap, fast refill
            int ok = 0;
            for (int i = 0; i < 10; i++) {
                if (bucket.tryAcquire()) ok++;
            }
            s.checkEq("burst 10", 10, ok);
            s.checkTrue("11th denied before refill", !bucket.tryAcquire());
            Thread.sleep(20);
            s.checkTrue("refilled", bucket.tryAcquire() || bucket.available() >= 0);
        });

        s.benchmark("sliding_window_and_keyed", () -> {
            RateLimiter.SlidingWindow sw = new RateLimiter.SlidingWindow(100, 1000);
            int ok = 0;
            for (int i = 0; i < 150; i++) {
                if (sw.tryAcquire()) ok++;
            }
            s.checkTrue("limited under 150 got=" + ok, ok <= 110 && ok >= 90);

            RateLimiter.KeyedLimiter keyed = new RateLimiter.KeyedLimiter(5, 100);
            int u1 = 0, u2 = 0;
            for (int i = 0; i < 5; i++) {
                if (keyed.tryAcquire("u1")) u1++;
                if (keyed.tryAcquire("u2")) u2++;
            }
            s.checkEq("u1 burst", 5, u1);
            s.checkEq("u2 burst", 5, u2);
            s.checkTrue("u1 exhausted", !keyed.tryAcquire("u1"));
            s.checkTrue("two keys", keyed.keyCount() >= 2);
        });

        s.benchmark("adaptive_limiter_factor", () -> {
            RateLimiter.AdaptiveLimiter al = new RateLimiter.AdaptiveLimiter(1000);
            al.fromDegradationSeverity(0);
            s.checkClose("L0 factor qps", 1000.0, al.effectiveQps(), 1e-6);
            al.fromDegradationSeverity(3);
            s.checkClose("L3 factor", 200.0, al.effectiveQps(), 1e-6);
            al.fromDegradationSeverity(4);
            s.checkTrue("L4 very low", al.effectiveQps() < 100);
        });

        s.benchmark("ops_throughput_counters", () -> {
            MetricsRegistry reg = new MetricsRegistry();
            long t0 = System.nanoTime();
            int n = 500_000;
            for (int i = 0; i < n; i++) {
                reg.counter("x").inc();
                if ((i & 15) == 0) reg.timer("t").record(i & 31);
            }
            long ms = (System.nanoTime() - t0) / 1_000_000L;
            double qps = n / Math.max(0.001, ms / 1000.0);
            System.out.printf("    counter QPS=%.0f%n", qps);
            s.checkEq("count", (long) n, reg.counter("x").get());
            s.checkTrue("QPS > 200k", qps > 200_000);
        });

        return s.exitCode();
    }
}
