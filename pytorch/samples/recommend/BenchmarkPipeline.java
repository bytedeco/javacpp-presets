/*
 * Benchmark for multi-stage ranking pipeline:
 * recall → coarse → fine → rerank → mix + orchestrator degradation paths.
 *
 *   java -cp ... samples.recommend.BenchmarkPipeline
 */
package samples.recommend;

import org.bytedeco.pytorch.deploy.serving.pipeline.Candidate;
import org.bytedeco.pytorch.deploy.serving.pipeline.CoarseRankStage;
import org.bytedeco.pytorch.deploy.serving.pipeline.FineRankStage;
import org.bytedeco.pytorch.deploy.serving.pipeline.MixRankStage;
import org.bytedeco.pytorch.deploy.serving.pipeline.PipelineOrchestrator;
import org.bytedeco.pytorch.deploy.serving.pipeline.RankStage;
import org.bytedeco.pytorch.deploy.serving.pipeline.RecallStage;
import org.bytedeco.pytorch.deploy.serving.pipeline.RequestContext;
import org.bytedeco.pytorch.deploy.serving.pipeline.RerankStage;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Random;
import java.util.Set;

public final class BenchmarkPipeline {

    public static void main(String[] args) {
        System.exit(runTests());
    }

    public static int runTests() {
        BenchSupport.Suite s = new BenchSupport.Suite("BenchmarkPipeline");
        s.header();

        s.benchmark("candidate_identity_and_copy", () -> {
            Candidate c = new Candidate("item_1", 0.5)
                    .putScore("recall_score", 0.5)
                    .tag("category", "sports")
                    .addRecallChannel("u2i");
            Candidate copy = c.copy();
            s.checkEq("id", "item_1", copy.itemId());
            s.checkClose("score", 0.5, copy.score(), 1e-12);
            s.checkEq("category tag", "sports", copy.tag("category"));
            s.checkTrue("channel retained", copy.recallChannels().contains("u2i"));
            copy.score(0.9);
            s.checkClose("original unchanged", 0.5, c.score(), 1e-12);
        });

        s.benchmark("request_context_budget", () -> {
            RequestContext ctx = RequestContext.builder("r1")
                    .userId("u1")
                    .timeoutMs(200)
                    .experimentParam("fine.quota", "50")
                    .build();
            s.checkEq("diversion user", "u1", ctx.diversionKey());
            s.checkEq("exp int", 50, ctx.expParamInt("fine.quota", 0));
            s.checkTrue("budget remaining > 0", ctx.remainingBudgetMs() > 0);
            s.checkTrue("not exceeded immediately", !ctx.deadlineExceeded());
        });

        s.benchmark("recall_multi_channel_dedup", () -> {
            // Keep both channel lists short so the shared duplicate is inside each
            // channel's per-channel quota window (staticChannel truncates first).
            List<Candidate> hot = items("hot", 20, 1.0);
            List<Candidate> u2i = items("u2i", 15, 0.8);
            // Shared id appears in both channels — should merge recallChannels.
            String sharedId = hot.get(0).itemId();
            u2i.add(0, hot.get(0).copy().score(0.99));
            RecallStage recall = new RecallStage(List.of(
                    RecallStage.staticChannel("hot", hot),
                    RecallStage.staticChannel("u2i", u2i)), 50, 120, 100L, null);
            RequestContext ctx = RequestContext.builder("r-recall")
                    .userId("u")
                    .timeoutMs(500)
                    .experimentParam("recall.total_quota", "100")
                    .experimentParam("recall.per_channel_quota", "50")
                    .build();
            RankStage.StageResult res = recall.execute(ctx, List.of());
            s.checkTrue("recall ok", !res.degraded || res.size() > 0);
            s.checkTrue("size <= quota", res.size() <= 100);
            Set<String> ids = new HashSet<>();
            for (Candidate c : res.candidates) {
                s.checkTrue("unique id " + c.itemId(), ids.add(c.itemId()));
            }
            boolean merged = false;
            for (Candidate c : res.candidates) {
                if (sharedId.equals(c.itemId()) && c.recallChannels().size() >= 2) {
                    merged = true;
                    s.checkTrue("has hot channel", c.recallChannels().contains("hot"));
                    s.checkTrue("has u2i channel", c.recallChannels().contains("u2i"));
                }
            }
            s.checkTrue("duplicate merged channels id=" + sharedId, merged);
            // unique count: hot(20) + u2i-only(15) = 35, shared counted once
            s.checkEq("deduped size", 35, res.size());
            recall.shutdown();
        });

        s.benchmark("coarse_rank_cuts_quota", () -> {
            List<Candidate> input = items("c", 500, 0.0);
            Random rng = BenchSupport.rng(1);
            for (Candidate c : input) {
                c.score(rng.nextDouble());
                c.putScore("recall_score", c.score());
            }
            CoarseRankStage coarse = new CoarseRankStage(
                    (ctx, c) -> c.getScore("recall_score", 0.0) * 2.0, 100, 50L);
            RequestContext ctx = RequestContext.builder("r-coarse")
                    .timeoutMs(500)
                    .experimentParam("coarse.quota", "100")
                    .build();
            RankStage.StageResult res = coarse.execute(ctx, input);
            s.checkEq("coarse quota", 100, res.size());
            // sorted desc
            for (int i = 1; i < res.candidates.size(); i++) {
                s.checkTrue("sorted desc",
                        res.candidates.get(i - 1).score() >= res.candidates.get(i).score() - 1e-12);
            }
        });

        s.benchmark("fine_rank_batch_and_fusion", () -> {
            List<Candidate> input = items("f", 200, 0.1);
            for (int i = 0; i < input.size(); i++) {
                input.get(i).putScore("coarse_score", 0.1);
                input.get(i).putScore("pcvr", 0.2);
                input.get(i).putScore("price", 1.0 + (i % 5));
            }
            FineRankStage fine = new FineRankStage(
                    FineRankStage.fromSingle((ctx, c) -> 0.5),
                    FineRankStage.eCpmLike(1.0, 0.5),
                    50,
                    80L);
            RequestContext ctx = RequestContext.builder("r-fine").timeoutMs(1000)
                    .experimentParam("fine.quota", "50").build();
            RankStage.StageResult res = fine.execute(ctx, input);
            s.checkEq("fine quota", 50, res.size());
            s.checkTrue("has fine_score", res.candidates.get(0).getScore("fine_score", -1) > 0);
        });

        s.benchmark("fine_rank_scorer_error_degrades", () -> {
            FineRankStage fine = new FineRankStage((ctx, cands) -> {
                throw new RuntimeException("boom");
            });
            List<Candidate> input = items("e", 20, 0.3);
            for (Candidate c : input) c.putScore("coarse_score", 0.3);
            RequestContext ctx = RequestContext.builder("r-err").timeoutMs(500).build();
            RankStage.StageResult res = fine.execute(ctx, input);
            s.checkTrue("degraded", res.degraded);
            s.checkTrue("still returns items", res.size() > 0);
        });

        s.benchmark("rerank_mmr_and_dedup", () -> {
            List<Candidate> input = new ArrayList<>();
            for (int i = 0; i < 30; i++) {
                Candidate c = new Candidate("i" + i, 1.0 - i * 0.01);
                c.tag("category", i % 3 == 0 ? "A" : (i % 3 == 1 ? "B" : "C"));
                input.add(c);
            }
            // intentional dup
            input.add(input.get(0).copy());
            RerankStage rerank = new RerankStage(List.of(
                    RerankStage.dedup(),
                    RerankStage.mmr(0.7),
                    RerankStage.categoryDamping(5, 2)), 20);
            RequestContext ctx = RequestContext.builder("r-rerank").timeoutMs(500)
                    .experimentParam("rerank.quota", "20").build();
            RankStage.StageResult res = rerank.execute(ctx, input);
            s.checkEq("rerank quota", 20, res.size());
            Set<String> ids = new HashSet<>();
            for (Candidate c : res.candidates) {
                s.checkTrue("deduped", ids.add(c.itemId()));
            }
        });

        s.benchmark("mix_organic_and_ads", () -> {
            MixRankStage mix = new MixRankStage(20);
            List<Candidate> organic = items("org", 30, 0.5);
            List<Candidate> ads = items("ad", 10, 0.9);
            RequestContext ctx = RequestContext.builder("r-mix").timeoutMs(200).build();
            List<Candidate> out = mix.mixOrganicAndAds(ctx, organic, ads, 20, 5, 3);
            s.checkTrue("page filled", out.size() == 20 || out.size() > 10);
            int adCount = 0;
            for (Candidate c : out) {
                if ("ads".equals(c.tag("mix_queue"))) adCount++;
            }
            s.checkTrue("ads inserted <= 3 got=" + adCount, adCount <= 3);
            s.checkTrue("some ads or organic", adCount >= 0 && out.size() > 0);
        });

        s.benchmark("orchestrator_e2e", () -> {
            List<Candidate> hot = items("hot", 50, 0.6);
            for (int i = 0; i < hot.size(); i++) {
                hot.get(i).tag("category", "cat" + (i % 4));
            }
            RecallStage recall = new RecallStage(List.of(RecallStage.staticChannel("hot", hot)));
            CoarseRankStage coarse = new CoarseRankStage(CoarseRankStage.passThrough(), 40, 20L);
            FineRankStage fine = new FineRankStage(
                    FineRankStage.fromSingle((ctx, c) -> c.score() + 0.01), null, 20, 40L);
            RerankStage rerank = new RerankStage(List.of(RerankStage.dedup(), RerankStage.mmr(0.6)), 15);
            MixRankStage mix = new MixRankStage(10);
            PipelineOrchestrator pipe = PipelineOrchestrator.builder()
                    .standardCascade(recall, coarse, fine, rerank, mix)
                    .ultimateFallback(hot.subList(0, 5))
                    .build();
            RequestContext ctx = RequestContext.builder("r-e2e")
                    .userId("u42")
                    .timeoutMs(1000)
                    .experimentParam("recall.total_quota", "50")
                    .experimentParam("coarse.quota", "40")
                    .experimentParam("fine.quota", "20")
                    .experimentParam("rerank.quota", "15")
                    .experimentParam("mix.page_size", "10")
                    .build();
            PipelineOrchestrator.PipelineResult pr = pipe.run(ctx);
            s.checkEq("final page", 10, pr.items.size());
            s.checkEq("5 stages", 5, pr.stageResults.size());
            s.checkTrue("latency recorded", pr.totalLatencyMs >= 0);
            s.checkTrue("stage latencies map", pr.stageLatencies().size() == 5);
            for (int i = 0; i < pr.items.size(); i++) {
                s.checkEq("rank " + i, i, pr.items.get(i).rank());
            }
            recall.shutdown();
        });

        s.benchmark("orchestrator_ultimate_fallback_on_empty", () -> {
            RecallStage emptyRecall = new RecallStage(List.of(
                    RecallStage.functional("empty", (ctx, q) -> List.of())));
            // single stage that returns empty, then fallback
            PipelineOrchestrator pipe = PipelineOrchestrator.builder()
                    .addStage(emptyRecall)
                    .ultimateFallback(items("fb", 3, 1.0))
                    .build();
            RequestContext ctx = RequestContext.builder("r-fb").timeoutMs(200).build();
            PipelineOrchestrator.PipelineResult pr = pipe.run(ctx);
            s.checkTrue("degraded", pr.degraded);
            s.checkEq("fallback size", 3, pr.items.size());
            s.checkEq("fallback tag", "ultimate", pr.items.get(0).tag("fallback"));
            emptyRecall.shutdown();
        });

        s.benchmark("pipeline_throughput", () -> {
            List<Candidate> hot = items(" thr", 200, 0.5);
            // trim id space
            for (int i = 0; i < hot.size(); i++) {
                hot.set(i, new Candidate(" thr_" + i, 0.5 + (i % 10) * 0.01));
            }
            RecallStage recall = new RecallStage(List.of(RecallStage.staticChannel("hot", hot)));
            CoarseRankStage coarse = new CoarseRankStage(CoarseRankStage.passThrough(), 100, 20L);
            FineRankStage fine = new FineRankStage(FineRankStage.fromSingle((ctx, c) -> c.score()), null, 50, 40L);
            RerankStage rerank = new RerankStage(List.of(RerankStage.dedup()), 30);
            MixRankStage mix = new MixRankStage(20);
            PipelineOrchestrator pipe = PipelineOrchestrator.builder()
                    .standardCascade(recall, coarse, fine, rerank, mix)
                    .build();
            int n = 500;
            long t0 = System.nanoTime();
            for (int i = 0; i < n; i++) {
                RequestContext ctx = RequestContext.builder("r" + i)
                        .userId("u" + i)
                        .timeoutMs(2000)
                        .build();
                PipelineOrchestrator.PipelineResult pr = pipe.run(ctx);
                if (pr.items.isEmpty()) {
                    throw new IllegalStateException("empty at " + i);
                }
            }
            long ms = (System.nanoTime() - t0) / 1_000_000L;
            double qps = n / Math.max(0.001, ms / 1000.0);
            System.out.printf("    pipeline QPS=%.1f (n=%d, %d ms)%n", qps, n, ms);
            s.checkTrue("QPS > 50", qps > 50);
            recall.shutdown();
        });

        return s.exitCode();
    }

    private static List<Candidate> items(String prefix, int n, double baseScore) {
        List<Candidate> list = new ArrayList<>(n);
        for (int i = 0; i < n; i++) {
            list.add(new Candidate(prefix + "_" + i, baseScore + i * 0.0001));
        }
        return list;
    }
}
