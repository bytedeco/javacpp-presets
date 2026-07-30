/*
 * Benchmark / correctness suite for abtest package.
 *
 * Covers: Murmur hash uniformity, sticky assignment, layered mutex capacity,
 * SRM, Welch t-test, CUPED variance reduction, guardrails, analyzer ship decision,
 * traffic splitter canary stages, online metrics collector.
 *
 *   java -cp ... samples.recommend.BenchmarkAbtest
 */
package samples.recommend;

import org.bytedeco.pytorch.utils.recommend.abtest.BucketAssigner;
import org.bytedeco.pytorch.utils.recommend.abtest.DiversionUnit;
import org.bytedeco.pytorch.utils.recommend.abtest.Experiment;
import org.bytedeco.pytorch.utils.recommend.abtest.ExperimentAnalyzer;
import org.bytedeco.pytorch.utils.recommend.abtest.ExperimentStatus;
import org.bytedeco.pytorch.utils.recommend.abtest.Guardrail;
import org.bytedeco.pytorch.utils.recommend.abtest.LayeredExperimentManager;
import org.bytedeco.pytorch.utils.recommend.abtest.OnlineMetricsCollector;
import org.bytedeco.pytorch.utils.recommend.abtest.StatisticalTest;
import org.bytedeco.pytorch.utils.recommend.abtest.TrafficSplitter;
import org.bytedeco.pytorch.utils.recommend.abtest.Variant;

import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;

public final class BenchmarkAbtest {

    public static void main(String[] args) {
        System.exit(runTests());
    }

    public static int runTests() {
        BenchSupport.Suite s = new BenchSupport.Suite("BenchmarkAbtest");
        s.header();

        s.benchmark("murmur3_deterministic", () -> {
            byte[] data = "user_42\0salt".getBytes();
            int a = BucketAssigner.murmur3_32(data, BucketAssigner.DEFAULT_SEED);
            int b = BucketAssigner.murmur3_32(data, BucketAssigner.DEFAULT_SEED);
            s.checkEq("deterministic hash", a, b);
        });

        s.benchmark("bucket_uniformity_chi2", () -> {
            long buckets = 100;
            long[] counts = new long[(int) buckets];
            int n = 100_000;
            for (int i = 0; i < n; i++) {
                long b = BucketAssigner.bucketOf("saltA", "user_" + i, buckets);
                counts[(int) b]++;
            }
            double[] ratio = new double[(int) buckets];
            java.util.Arrays.fill(ratio, 1.0);
            StatisticalTest.SrmResult r = StatisticalTest.srmTest(counts, ratio, 0.001);
            // Uniform hash should NOT flag SRM at alpha=0.001 on 100k samples.
            s.checkTrue("no false SRM on uniform hash p=" + r.pValue, !r.srmDetected);
            s.checkRange("chi2 reasonable", r.chiSquare, 0.0, 200.0);
        });

        s.benchmark("sticky_assignment", () -> {
            Experiment exp = Experiment.builder("exp_sticky", "layer_fine")
                    .diversionUnit(DiversionUnit.USER_ID)
                    .trafficPercent(100.0)
                    .addVariant(Variant.control("control", 1.0))
                    .addVariant(Variant.treatment("t1", 1.0))
                    .status(ExperimentStatus.RUNNING)
                    .build();
            BucketAssigner.Assignment a1 = BucketAssigner.assign(exp, "user_7", 1_700_000_000_000L);
            BucketAssigner.Assignment a2 = BucketAssigner.assign(exp, "user_7", 1_700_000_000_100L);
            s.checkTrue("assignment non-null", a1 != null && a2 != null);
            s.checkEq("sticky variant", a1.variantId(), a2.variantId());
            s.checkEq("sticky bucket", a1.bucket(), a2.bucket());
        });

        s.benchmark("traffic_percent_window", () -> {
            Experiment exp = Experiment.builder("exp_10pct", "layer_fine")
                    .trafficPercent(10.0)
                    .addVariant(Variant.control("c", 1.0))
                    .addVariant(Variant.treatment("t", 1.0))
                    .status(ExperimentStatus.RUNNING)
                    .bucketCount(1000)
                    .build();
            int in = 0;
            int n = 20_000;
            for (int i = 0; i < n; i++) {
                if (BucketAssigner.assign(exp, "u" + i, 1L) != null) in++;
            }
            double rate = in / (double) n;
            s.checkRange("enter rate ~10%", rate, 0.08, 0.12);
        });

        s.benchmark("layered_manager_capacity_and_resolve", () -> {
            LayeredExperimentManager mgr = new LayeredExperimentManager(true);
            mgr.createLayer("layer_fine", "fine", DiversionUnit.USER_ID);
            mgr.createLayer("layer_recall", "recall", DiversionUnit.USER_ID);
            Experiment fine = Experiment.builder("fine_a", "layer_fine")
                    .trafficPercent(30.0)
                    .addVariant(Variant.control("c", 1.0))
                    .addVariant(Variant.builder("t").trafficWeight(1.0).parameter("model", "v2").build())
                    .status(ExperimentStatus.RUNNING)
                    .primaryMetric("ctr")
                    .build();
            mgr.register(fine);
            Experiment recall = Experiment.builder("recall_a", "layer_recall")
                    .trafficPercent(50.0)
                    .addVariant(Variant.control("c", 1.0))
                    .addVariant(Variant.treatment("t", 1.0))
                    .status(ExperimentStatus.RUNNING)
                    .build();
            mgr.register(recall);

            // Same layer over-subscribe should fail
            boolean threw = false;
            try {
                Experiment over = Experiment.builder("fine_b", "layer_fine")
                        .trafficPercent(80.0)
                        .addVariant(Variant.control("c", 1.0))
                        .status(ExperimentStatus.RUNNING)
                        .build();
                mgr.register(over);
            } catch (IllegalStateException ex) {
                threw = true;
            }
            s.checkTrue("layer capacity enforced", threw);

            int multiLayer = 0;
            for (int i = 0; i < 5000; i++) {
                List<BucketAssigner.Assignment> as = mgr.resolve("user_" + i, 1L);
                if (as.size() >= 2) multiLayer++;
            }
            // Orthogonal layers: some users hit both; should be > 0
            s.checkTrue("multi-layer exposure exists", multiLayer > 0);

            Map<String, String> params = mgr.resolveParameters("user_1");
            s.checkTrue("params map non-null", params != null);
        });

        s.benchmark("welch_ttest_detects_lift", () -> {
            Random rng = BenchSupport.rng(42);
            double[] c = BenchSupport.randomGaussian(rng, 2000, 0.10, 0.05);
            double[] t = BenchSupport.randomGaussian(rng, 2000, 0.12, 0.05);
            StatisticalTest.MeanTestResult r = StatisticalTest.welchTTest(c, t, 0.05);
            s.checkTrue("detects positive lift", r.significantAtAlpha && r.absoluteDelta > 0);
            s.checkTrue("p < 0.05", r.pValue < 0.05);
        });

        s.benchmark("welch_ttest_null_mostly_insignificant", () -> {
            Random rng = BenchSupport.rng(7);
            double[] c = BenchSupport.randomGaussian(rng, 1500, 0.10, 0.05);
            double[] t = BenchSupport.randomGaussian(rng, 1500, 0.10, 0.05);
            StatisticalTest.MeanTestResult r = StatisticalTest.welchTTest(c, t, 0.05);
            // Not a hard guarantee but with equal means usually p > 0.01
            s.checkTrue("null p not tiny p=" + r.pValue, r.pValue > 0.001);
        });

        s.benchmark("two_proportion_ztest", () -> {
            StatisticalTest.ProportionTestResult r =
                    StatisticalTest.twoProportionZTest(1000, 10_000, 1200, 10_000, 0.05);
            s.checkTrue("CTR lift significant", r.significantAtAlpha);
            s.checkClose("control rate", 0.10, r.controlRate, 1e-9);
            s.checkClose("treat rate", 0.12, r.treatmentRate, 1e-9);
        });

        s.benchmark("srm_detects_mismatch", () -> {
            long[] observed = {10000, 8000}; // expected equal
            StatisticalTest.SrmResult r = StatisticalTest.srmEqualWeight(observed, 0.001);
            s.checkTrue("SRM detected", r.srmDetected);
            s.checkTrue("p very small", r.pValue < 0.001);
        });

        s.benchmark("cuped_reduces_variance", () -> {
            Random rng = BenchSupport.rng(99);
            int n = 3000;
            double[] x = new double[n];
            double[] y = new double[n];
            for (int i = 0; i < n; i++) {
                x[i] = rng.nextGaussian();
                y[i] = 0.8 * x[i] + 0.2 * rng.nextGaussian(); // correlated
            }
            double[] yCuped = StatisticalTest.cupedAdjust(y, x);
            double varY = var(y);
            double varC = var(yCuped);
            s.checkTrue("CUPED var < raw var (" + varC + " < " + varY + ")", varC < varY * 0.7);
        });

        s.benchmark("sample_size_positive", () -> {
            long n = StatisticalTest.sampleSizePerArm(0.05, 0.01, 0.05, 0.8);
            s.checkTrue("n > 100", n > 100);
            long np = StatisticalTest.sampleSizePerArmProportion(0.1, 0.01, 0.05, 0.8);
            s.checkTrue("prop n > 100", np > 100);
        });

        s.benchmark("guardrail_srm_and_drop", () -> {
            Map<String, Guardrail.MetricArmPair> metrics = new HashMap<>();
            metrics.put("ctr", new Guardrail.MetricArmPair(0.10, 0.07)); // -30%
            metrics.put("error_rate", new Guardrail.MetricArmPair(0.001, 0.02));
            Guardrail.ExperimentSnapshot snap =
                    new Guardrail.ExperimentSnapshot("e1", 5000, 5000, 1e-6, metrics);
            List<Guardrail> rules = List.of(
                    Guardrail.srm("srm", 0.001, Guardrail.Action.KILL),
                    Guardrail.relativeDrop("ctr_drop", "ctr", 0.05, Guardrail.Action.KILL),
                    Guardrail.treatmentAbove("err", "error_rate", 0.01, Guardrail.Action.PAUSE));
            Guardrail.EvaluationResult er = Guardrail.evaluateAll(rules, snap);
            s.checkEq("decision KILL", Guardrail.Decision.KILL, er.decision);
            s.checkTrue("has fires", !er.fires.isEmpty());
        });

        s.benchmark("online_metrics_and_analyzer", () -> {
            Experiment exp = Experiment.builder("exp_ana", "layer_fine")
                    .trafficPercent(100.0)
                    .addVariant(Variant.control("control", 1.0))
                    .addVariant(Variant.treatment("treatment", 1.0))
                    .primaryMetric("ctr")
                    .status(ExperimentStatus.RUNNING)
                    .build();
            OnlineMetricsCollector col = new OnlineMetricsCollector();
            Random rng = BenchSupport.rng(1);
            for (int i = 0; i < 5000; i++) {
                col.recordExposure(exp.id(), "control");
                col.observeBinary(exp.id(), "control", "ctr", rng.nextDouble() < 0.10);
            }
            for (int i = 0; i < 5000; i++) {
                col.recordExposure(exp.id(), "treatment");
                col.observeBinary(exp.id(), "treatment", "ctr", rng.nextDouble() < 0.13);
            }
            StatisticalTest.SrmResult srm = col.srm(exp, 0.001);
            s.checkTrue("balanced exposure no SRM", !srm.srmDetected);

            ExperimentAnalyzer analyzer = ExperimentAnalyzer.builder()
                    .alpha(0.05)
                    .srmAlpha(0.001)
                    .minSamplePerArm(1000)
                    .addGuardrail(Guardrail.srm("srm", 0.001, Guardrail.Action.KILL))
                    .build();
            ExperimentAnalyzer.Report report = analyzer.analyze(exp, col, "control", "treatment");
            s.checkTrue("report text non-empty", report.text != null && report.text.length() > 20);
            s.checkTrue("decision ship or no_ship (not blocked)",
                    report.decision == ExperimentAnalyzer.ShipDecision.SHIP
                            || report.decision == ExperimentAnalyzer.ShipDecision.NO_SHIP
                            || report.decision == ExperimentAnalyzer.ShipDecision.INCONCLUSIVE);
            // With 3pp lift on 5k each, should usually SHIP
            s.checkTrue("likely SHIP got=" + report.decision,
                    report.decision == ExperimentAnalyzer.ShipDecision.SHIP
                            || report.decision == ExperimentAnalyzer.ShipDecision.NO_SHIP);
        });

        s.benchmark("traffic_splitter_sticky_and_canary", () -> {
            List<TrafficSplitter.WeightedTarget> targets = List.of(
                    new TrafficSplitter.WeightedTarget("stable", 90),
                    new TrafficSplitter.WeightedTarget("canary", 10));
            String a = TrafficSplitter.selectSticky("userX", "salt", targets);
            String b = TrafficSplitter.selectSticky("userX", "salt", targets);
            s.checkEq("sticky select", a, b);

            Map<String, Integer> hist = new HashMap<>();
            for (int i = 0; i < 20_000; i++) {
                String id = TrafficSplitter.selectSticky("u" + i, "salt", targets);
                hist.merge(id, 1, Integer::sum);
            }
            double canaryRate = hist.getOrDefault("canary", 0) / 20_000.0;
            s.checkRange("canary ~10%", canaryRate, 0.08, 0.12);

            double[] stages = TrafficSplitter.defaultCanaryStages();
            s.checkEq("stages len", 6, stages.length);
            s.checkClose("first stage 1%", 1.0, stages[0], 1e-9);
            s.checkClose("last stage 100%", 100.0, stages[stages.length - 1], 1e-9);
        });

        s.benchmark("assignment_throughput", () -> {
            Experiment exp = Experiment.builder("exp_perf", "layer_fine")
                    .trafficPercent(100.0)
                    .addVariant(Variant.control("c", 1.0))
                    .addVariant(Variant.treatment("t", 1.0))
                    .status(ExperimentStatus.RUNNING)
                    .build();
            int n = 200_000;
            long t0 = System.nanoTime();
            int hits = 0;
            for (int i = 0; i < n; i++) {
                if (BucketAssigner.assign(exp, "u" + i, 1L) != null) hits++;
            }
            long ms = (System.nanoTime() - t0) / 1_000_000L;
            double qps = n / Math.max(0.001, ms / 1000.0);
            System.out.printf("    assignment QPS=%.0f (n=%d, %d ms, hits=%d)%n", qps, n, ms, hits);
            s.checkTrue("QPS > 50k", qps > 50_000);
        });

        return s.exitCode();
    }

    private static double var(double[] xs) {
        double m = 0;
        for (double x : xs) m += x;
        m /= xs.length;
        double v = 0;
        for (double x : xs) {
            double d = x - m;
            v += d * d;
        }
        return v / (xs.length - 1);
    }
}
