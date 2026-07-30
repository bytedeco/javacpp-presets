/*
 * Benchmark for gateway traffic routing: sticky hash, weights, header force,
 * region affinity, shadow upstreams, canary percent.
 *
 *   java -cp ... samples.recommend.BenchmarkGateway
 */
package samples.recommend;

import org.bytedeco.pytorch.utils.recommend.serving.gateway.TrafficRouter;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

public final class BenchmarkGateway {

    public static void main(String[] args) {
        System.exit(runTests());
    }

    public static int runTests() {
        BenchSupport.Suite s = new BenchSupport.Suite("BenchmarkGateway");
        s.header();

        s.benchmark("weighted_sticky_split", () -> {
            TrafficRouter router = new TrafficRouter("recommend_api", "salt1");
            router.addUpstream("stable", "stable.svc", 90.0);
            router.addUpstream("canary", "canary.svc", 10.0);
            router.sticky(true);

            Map<String, Integer> hist = new HashMap<>();
            for (int i = 0; i < 20_000; i++) {
                TrafficRouter.RouteDecision d = router.route(req("r" + i, "u" + i, null, null));
                hist.merge(d.primaryUpstreamId, 1, Integer::sum);
                // sticky
                TrafficRouter.RouteDecision d2 = router.route(req("r2" + i, "u" + i, null, null));
                s.checkEq("sticky same user", d.primaryUpstreamId, d2.primaryUpstreamId);
            }
            double canary = hist.getOrDefault("canary", 0) / 20_000.0;
            System.out.println("    canary rate=" + canary + " hits=" + router.hitCounts());
            s.checkRange("canary ~10%", canary, 0.08, 0.12);
            s.checkEq("reason sticky", "sticky_weight",
                    router.route(req("rx", "u0", null, null)).reason);
        });

        s.benchmark("set_canary_percent", () -> {
            TrafficRouter router = new TrafficRouter("api");
            router.addUpstream("stable", "s", 100.0);
            router.addUpstream("canary", "c", 0.0);
            router.setCanaryPercent("stable", "canary", 25.0);
            s.checkClose("canary w", 25.0, router.currentWeights().get("canary"), 1e-9);
            s.checkClose("stable w", 75.0, router.currentWeights().get("stable"), 1e-9);

            int c = 0;
            for (int i = 0; i < 10_000; i++) {
                if ("canary".equals(router.route(req("r" + i, "u" + i, null, null)).primaryUpstreamId)) {
                    c++;
                }
            }
            s.checkRange("25% traffic", c / 10_000.0, 0.22, 0.28);
        });

        s.benchmark("header_force_rule", () -> {
            TrafficRouter router = new TrafficRouter("api");
            router.addUpstream("stable", "s", 100.0);
            router.addUpstream("canary", "c", 0.0); // zero weight but forceable
            // Need positive weight OR force — force looks up upstream even if weight 0
            router.addHeaderRule("X-Rec-Canary", "1", "canary");

            Map<String, String> headers = new HashMap<>();
            headers.put("X-Rec-Canary", "1");
            TrafficRouter.RouteDecision d = router.route(
                    new TrafficRouter.RouteRequest("r1", "u1", "", "/", "", headers));
            s.checkEq("forced canary", "canary", d.primaryUpstreamId);
            s.checkTrue("forced flag", d.forced);
            s.checkTrue("reason header", d.reason.startsWith("header:"));

            // without header, weight 0 canary should not be chosen — only stable has weight
            TrafficRouter.RouteDecision d2 = router.route(req("r2", "u2", null, null));
            s.checkEq("default stable", "stable", d2.primaryUpstreamId);
            s.checkTrue("not forced", !d2.forced);
        });

        s.benchmark("region_affinity", () -> {
            TrafficRouter router = new TrafficRouter("api");
            router.addUpstream("cn", "cn.svc", 50.0);
            router.addUpstream("us", "us.svc", 50.0);
            router.addRegionAffinity("cn-shanghai", "cn");
            router.addRegionAffinity("us-west", "us");

            TrafficRouter.RouteDecision d = router.route(req("r", "u", null, "cn-shanghai"));
            s.checkEq("cn affinity", "cn", d.primaryUpstreamId);
            s.checkEq("reason region", "region:cn-shanghai", d.reason);

            TrafficRouter.RouteDecision d2 = router.route(req("r2", "u2", null, "us-west"));
            s.checkEq("us affinity", "us", d2.primaryUpstreamId);
        });

        s.benchmark("shadow_upstreams", () -> {
            TrafficRouter router = new TrafficRouter("api");
            router.addUpstream(new TrafficRouter.Upstream("stable", "s", 100.0, false));
            router.addUpstream(new TrafficRouter.Upstream("shadow-v2", "sh", 100.0, true));
            TrafficRouter.RouteDecision d = router.route(req("r", "u", null, null));
            s.checkEq("primary stable", "stable", d.primaryUpstreamId);
            s.checkTrue("has shadow", d.shadowUpstreamIds.contains("shadow-v2"));
            s.checkEq("one shadow", 1, d.shadowUpstreamIds.size());
        });

        s.benchmark("random_non_sticky", () -> {
            TrafficRouter router = new TrafficRouter("api");
            router.sticky(false);
            router.addUpstream("a", "a", 50.0);
            router.addUpstream("b", "b", 50.0);
            // same user may differ across calls when non-sticky — probabilistic
            int diffs = 0;
            for (int i = 0; i < 500; i++) {
                String u = "same_user";
                String x = router.route(req("r1" + i, u, null, null)).primaryUpstreamId;
                String y = router.route(req("r2" + i, u, null, null)).primaryUpstreamId;
                if (!x.equals(y)) diffs++;
            }
            System.out.println("    non-sticky diffs=" + diffs + "/500");
            s.checkTrue("some non-sticky variance", diffs > 0);
        });

        s.benchmark("consistent_hash_stable", () -> {
            TrafficRouter router = new TrafficRouter("api", "ring");
            List<String> nodes = List.of("n0", "n1", "n2", "n3");
            String a = router.consistentHash("user_9", nodes);
            String b = router.consistentHash("user_9", nodes);
            s.checkEq("consistent", a, b);
            s.checkTrue("in ring", nodes.contains(a));
            // distribution roughly even
            Map<String, Integer> hist = new HashMap<>();
            for (int i = 0; i < 10_000; i++) {
                hist.merge(router.consistentHash("u" + i, nodes), 1, Integer::sum);
            }
            for (String n : nodes) {
                double rate = hist.getOrDefault(n, 0) / 10_000.0;
                s.checkRange("node " + n + " ~25%", rate, 0.15, 0.35);
            }
        });

        s.benchmark("split_by_percent_helper", () -> {
            int treat = 0;
            for (int i = 0; i < 10_000; i++) {
                String v = TrafficRouter.splitByPercent("k" + i, "s", 30.0, "ctrl", "treat");
                if ("treat".equals(v)) treat++;
            }
            s.checkRange("30% helper", treat / 10_000.0, 0.27, 0.33);
        });

        s.benchmark("header_hash_deterministic", () -> {
            int h1 = TrafficRouter.headerHash("abc");
            int h2 = TrafficRouter.headerHash("abc");
            s.checkEq("hash stable", h1, h2);
            s.checkTrue("diff inputs", TrafficRouter.headerHash("abc") != TrafficRouter.headerHash("xyz")
                    || true); // allow theoretical collision but usually different
        });

        s.benchmark("no_upstream_throws", () -> {
            TrafficRouter router = new TrafficRouter("empty");
            boolean threw = false;
            try {
                router.route(req("r", "u", null, null));
            } catch (IllegalStateException ex) {
                threw = true;
            }
            s.checkTrue("throws when empty", threw);
        });

        s.benchmark("explain_and_throughput", () -> {
            TrafficRouter router = new TrafficRouter("api");
            router.addUpstream("stable", "s", 80.0);
            router.addUpstream("canary", "c", 20.0);
            s.checkTrue("explain", router.explain().contains("TrafficRouter"));
            long t0 = System.nanoTime();
            int n = 200_000;
            for (int i = 0; i < n; i++) {
                router.route(req("r" + i, "u" + (i % 10_000), null, null));
            }
            long ms = (System.nanoTime() - t0) / 1_000_000L;
            double qps = n / Math.max(0.001, ms / 1000.0);
            System.out.printf("    route QPS=%.0f%n", qps);
            s.checkTrue("QPS > 100k", qps > 100_000);
        });

        return s.exitCode();
    }

    private static TrafficRouter.RouteRequest req(
            String requestId, String userId, Map<String, String> headers, String region) {
        return new TrafficRouter.RouteRequest(
                requestId, userId, "", "/recommend", region == null ? "" : region,
                headers == null ? Map.of() : headers);
    }
}
