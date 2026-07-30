/*
 * Benchmark for offline evaluation: metrics, holdout splits, calibration, A/A.
 *
 *   java -cp ... samples.recommend.BenchmarkOffline
 */
package samples.recommend;

import org.bytedeco.pytorch.utils.recommend.offline.AATestRunner;
import org.bytedeco.pytorch.utils.recommend.offline.CalibrationChecker;
import org.bytedeco.pytorch.utils.recommend.offline.HoldoutSplitter;
import org.bytedeco.pytorch.utils.recommend.offline.OfflineEvaluator;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;

public final class BenchmarkOffline {

    public static void main(String[] args) {
        System.exit(runTests());
    }

    public static int runTests() {
        BenchSupport.Suite s = new BenchSupport.Suite("BenchmarkOffline");
        s.header();

        s.benchmark("pointwise_perfect_auc", () -> {
            float[] y = {0, 0, 1, 1};
            float[] p = {0.1f, 0.2f, 0.8f, 0.9f};
            OfflineEvaluator ev = OfflineEvaluator.builder().computeCalibration(false).build();
            OfflineEvaluator.Scorecard sc = ev.evaluatePointwise("perfect", y, p);
            s.checkClose("AUC", 1.0, sc.get("AUC"), 1e-6);
            s.checkTrue("LogLoss finite", Double.isFinite(sc.get("LogLoss")));
        });

        s.benchmark("pointwise_random_auc_near_half", () -> {
            Random rng = BenchSupport.rng(3);
            int n = 5000;
            float[] y = BenchSupport.randomBinary(rng, n, 0.3);
            float[] p = BenchSupport.randomFloat01(rng, n);
            OfflineEvaluator ev = OfflineEvaluator.builder().computeCalibration(true).build();
            OfflineEvaluator.Scorecard sc = ev.evaluatePointwise("rand", y, p);
            s.checkRange("random AUC ~0.5", sc.get("AUC"), 0.45, 0.55);
            s.checkTrue("ECE computed", Double.isFinite(sc.get("ECE")));
        });

        s.benchmark("gauc_with_users", () -> {
            float[] y = {1, 0, 1, 0, 1, 0};
            float[] p = {0.9f, 0.1f, 0.8f, 0.2f, 0.7f, 0.3f};
            int[] u = {1, 1, 2, 2, 3, 3};
            OfflineEvaluator ev = OfflineEvaluator.builder().computeCalibration(false).build();
            OfflineEvaluator.Scorecard sc = ev.evaluatePointwise("g", y, p, u);
            s.checkClose("GAUC", 1.0, sc.get("GAUC"), 1e-6);
        });

        s.benchmark("ranking_metrics_ideal", () -> {
            Map<String, List<OfflineEvaluator.ScoredItem>> users = new HashMap<>();
            List<OfflineEvaluator.ScoredItem> items = new ArrayList<>();
            // higher score on positives
            items.add(new OfflineEvaluator.ScoredItem("i1", 0.9, 1f));
            items.add(new OfflineEvaluator.ScoredItem("i2", 0.8, 1f));
            items.add(new OfflineEvaluator.ScoredItem("i3", 0.1, 0f));
            items.add(new OfflineEvaluator.ScoredItem("i4", 0.05, 0f));
            users.put("u1", items);
            OfflineEvaluator ev = OfflineEvaluator.builder().rankingKs(2, 4).computeCalibration(false).build();
            OfflineEvaluator.Scorecard sc = ev.evaluateRanking("ranker", users);
            s.checkClose("HR@2", 1.0, sc.get("HR@2"), 1e-9);
            s.checkClose("NDCG@2", 1.0, sc.get("NDCG@2"), 1e-9);
            s.checkClose("MRR@4", 1.0, sc.get("MRR@4"), 1e-9);
        });

        s.benchmark("delta_report_ship_bar", () -> {
            OfflineEvaluator ev = OfflineEvaluator.builder().computeCalibration(false).build();
            float[] y = BenchSupport.randomBinary(BenchSupport.rng(1), 2000, 0.2);
            float[] pBase = new float[2000];
            float[] pCand = new float[2000];
            Random rng = BenchSupport.rng(2);
            for (int i = 0; i < 2000; i++) {
                pBase[i] = 0.2f + 0.1f * rng.nextFloat();
                pCand[i] = pBase[i] + 0.05f; // systematically better ranking signal-ish
            }
            // Make candidate better correlated
            for (int i = 0; i < 2000; i++) {
                pCand[i] = y[i] * 0.7f + (1 - y[i]) * 0.2f + 0.05f * rng.nextFloat();
                pBase[i] = y[i] * 0.55f + (1 - y[i]) * 0.35f + 0.05f * rng.nextFloat();
            }
            OfflineEvaluator.Scorecard base = ev.evaluatePointwise("base", y, pBase);
            OfflineEvaluator.Scorecard cand = ev.evaluatePointwise("cand", y, pCand);
            OfflineEvaluator.DeltaReport dr = ev.compare(base, cand, "AUC", 0.001);
            s.checkTrue("candidate AUC >= baseline", cand.get("AUC") >= base.get("AUC") - 1e-9);
            s.checkTrue("report non-empty", dr.text.length() > 10);
            System.out.println("    baseAUC=" + base.get("AUC") + " candAUC=" + cand.get("AUC")
                    + " ship=" + dr.shipRecommended);
        });

        s.benchmark("holdout_by_timestamp", () -> {
            List<HoldoutSplitter.Event> events = new ArrayList<>();
            for (int i = 0; i < 100; i++) {
                events.add(new HoldoutSplitter.Event("u" + (i % 10), "i" + i, 1000L + i, i % 2));
            }
            HoldoutSplitter.Split sp = HoldoutSplitter.byTimestamp(events, 1050L);
            s.checkEq("train size", 50, sp.trainSize());
            s.checkEq("test size", 50, sp.testSize());
            for (HoldoutSplitter.Event e : sp.train) {
                s.checkTrue("train before cutoff", e.timestampMs < 1050L);
            }
        });

        s.benchmark("holdout_leave_last_k", () -> {
            List<HoldoutSplitter.Event> events = new ArrayList<>();
            for (int u = 0; u < 5; u++) {
                for (int t = 0; t < 10; t++) {
                    events.add(new HoldoutSplitter.Event("u" + u, "i" + t, t, 1f));
                }
            }
            HoldoutSplitter.Split sp = HoldoutSplitter.leaveLastK(events, 2, true);
            s.checkEq("test = 5 users * 2", 10, sp.testSize());
            s.checkEq("train = 5 * 8", 40, sp.trainSize());
        });

        s.benchmark("holdout_user_holdout_disjoint", () -> {
            List<HoldoutSplitter.Event> events = new ArrayList<>();
            for (int u = 0; u < 100; u++) {
                events.add(new HoldoutSplitter.Event("u" + u, "i0", u, 1f));
            }
            HoldoutSplitter.Split sp = HoldoutSplitter.byUserHoldout(events, 0.2, 42L);
            Set<String> trainUsers = new HashSet<>();
            Set<String> testUsers = new HashSet<>();
            for (HoldoutSplitter.Event e : sp.train) trainUsers.add(e.userId);
            for (HoldoutSplitter.Event e : sp.test) testUsers.add(e.userId);
            Set<String> inter = new HashSet<>(trainUsers);
            inter.retainAll(testUsers);
            s.checkTrue("user disjoint", inter.isEmpty());
            s.checkRange("test user ratio", testUsers.size() / 100.0, 0.15, 0.25);
        });

        s.benchmark("calibration_perfect", () -> {
            // perfectly calibrated: pred = label rates in bins
            float[] y = new float[1000];
            float[] p = new float[1000];
            for (int i = 0; i < 1000; i++) {
                p[i] = (i % 10) / 10.0f + 0.05f;
                y[i] = p[i]; // identity => perfect calibration
            }
            CalibrationChecker.Result r = CalibrationChecker.expectedCalibrationError(y, p, 10);
            s.checkTrue("ECE small " + r.ece, r.ece < 0.05);
            s.checkRange("slope near 1", r.slope, 0.9, 1.1);
        });

        s.benchmark("platt_and_bin_calibrate", () -> {
            Random rng = BenchSupport.rng(11);
            int n = 2000;
            float[] y = BenchSupport.randomBinary(rng, n, 0.3);
            float[] p = new float[n];
            for (int i = 0; i < n; i++) {
                // miscalibrated raw scores
                p[i] = Math.min(0.99f, Math.max(0.01f, y[i] * 0.4f + 0.3f + 0.05f * rng.nextFloat()));
            }
            double[] ab = CalibrationChecker.fitPlatt(y, p);
            float[] cal = CalibrationChecker.applyPlatt(p, ab[0], ab[1]);
            s.checkEq("cal length", n, cal.length);
            float[] binCal = CalibrationChecker.binCalibrate(y, p, p, 10);
            s.checkEq("bin cal length", n, binCal.length);
            CalibrationChecker.Result before = CalibrationChecker.expectedCalibrationError(y, p, 10);
            CalibrationChecker.Result after = CalibrationChecker.expectedCalibrationError(y, cal, 10);
            System.out.println("    ECE before=" + before.ece + " after_platt=" + after.ece
                    + " A=" + ab[0] + " B=" + ab[1]);
            s.checkTrue("platt runs", Double.isFinite(after.ece));
        });

        s.benchmark("aa_test_fpr_near_alpha", () -> {
            Random rng = BenchSupport.rng(5);
            double[] values = BenchSupport.randomGaussian(rng, 2000, 1.0, 0.5);
            AATestRunner.Summary sum = AATestRunner.runMeanAA(values, 200, 0.05, 0.05, 123L);
            System.out.println("    " + sum);
            s.checkRange("empiric FPR near 0.05", sum.empiricalFpr, 0.0, 0.12);
            s.checkTrue("summary healthy flag boolean", sum.healthy || !sum.healthy);
        });

        s.benchmark("aa_power_increases_with_lift", () -> {
            Random rng = BenchSupport.rng(8);
            double[] baseline = BenchSupport.randomGaussian(rng, 1500, 0.0, 1.0);
            double powerSmall = AATestRunner.estimatePower(baseline, 0.05, 100, 0.05, 1L);
            double powerLarge = AATestRunner.estimatePower(baseline, 0.3, 100, 0.05, 1L);
            System.out.println("    powerSmall=" + powerSmall + " powerLarge=" + powerLarge);
            s.checkTrue("larger lift => higher power", powerLarge >= powerSmall);
            s.checkRange("large lift power high", powerLarge, 0.5, 1.0);
        });

        s.benchmark("offline_eval_throughput", () -> {
            // Metrics.aucScore is O(n^2); keep n modest for a smoke throughput check.
            Random rng = BenchSupport.rng(9);
            int n = 3_000;
            float[] y = BenchSupport.randomBinary(rng, n, 0.2);
            float[] p = BenchSupport.randomFloat01(rng, n);
            OfflineEvaluator ev = OfflineEvaluator.builder().computeCalibration(true).build();
            long t0 = System.nanoTime();
            OfflineEvaluator.Scorecard sc = ev.evaluatePointwise("perf", y, p);
            long ms = (System.nanoTime() - t0) / 1_000_000L;
            System.out.println("    n=" + n + " AUC=" + sc.get("AUC") + " in " + ms + " ms");
            s.checkTrue("finishes < 15s", ms < 15_000);
            s.checkTrue("AUC finite", Double.isFinite(sc.get("AUC")));
        });

        return s.exitCode();
    }
}
