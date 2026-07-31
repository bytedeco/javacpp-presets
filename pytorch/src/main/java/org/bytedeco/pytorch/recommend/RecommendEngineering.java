/*
 * Engineering-side facade for the recommend stack.
 *
 * Algorithm models live under {@code models.*}; this class documents and
 * wires the production engineering modules added alongside them:
 *
 * <pre>
 *   offline  →  abtest  →  serving/pipeline  →  serving/deploy
 *                    ↘ serving/gateway ↘ ops ↘ modelops
 * </pre>
 *
 * Design sources (not copied code — encoded industry practice):
 *   Meta XP / feed ranking, Microsoft ExP + CUPED, Google SRE + YouTube cascade,
 *   ByteDance Libra + multi-stage rank, Alibaba TPP / mutex domains / Sentinel,
 *   Tencent Tab, Netflix blender / Spinnaker, Apple-scale progressive delivery.
 *
 * All modules are pure Java, dependency-free beyond the existing recommend
 * stack, so they run in unit tests and as reference logic inside Flink/K8s
 * adapters.
 */
package org.bytedeco.pytorch.recommend;

import org.bytedeco.pytorch.deploy.abtest.DiversionUnit;
import org.bytedeco.pytorch.deploy.abtest.Experiment;
import org.bytedeco.pytorch.deploy.abtest.ExperimentAnalyzer;
import org.bytedeco.pytorch.deploy.abtest.ExperimentStatus;
import org.bytedeco.pytorch.deploy.abtest.Guardrail;
import org.bytedeco.pytorch.deploy.abtest.LayeredExperimentManager;
import org.bytedeco.pytorch.deploy.abtest.OnlineMetricsCollector;
import org.bytedeco.pytorch.deploy.abtest.TrafficSplitter;
import org.bytedeco.pytorch.deploy.abtest.Variant;
import org.bytedeco.pytorch.recommend.modelops.ModelRegistry;
import org.bytedeco.pytorch.deploy.offline.OfflineEvaluator;
import org.bytedeco.pytorch.recommend.ops.DegradationPolicy;
import org.bytedeco.pytorch.recommend.ops.FallbackStrategy;
import org.bytedeco.pytorch.recommend.ops.HealthChecker;
import org.bytedeco.pytorch.recommend.ops.MetricsRegistry;
import org.bytedeco.pytorch.recommend.ops.ServiceLevel;
import org.bytedeco.pytorch.deploy.serving.deploy.DeploymentController;
import org.bytedeco.pytorch.deploy.serving.deploy.ReplicaScaler;
import org.bytedeco.pytorch.deploy.serving.gateway.TrafficRouter;
import org.bytedeco.pytorch.deploy.serving.pipeline.Candidate;
import org.bytedeco.pytorch.deploy.serving.pipeline.CoarseRankStage;
import org.bytedeco.pytorch.deploy.serving.pipeline.FineRankStage;
import org.bytedeco.pytorch.deploy.serving.pipeline.MixRankStage;
import org.bytedeco.pytorch.deploy.serving.pipeline.PipelineOrchestrator;
import org.bytedeco.pytorch.deploy.serving.pipeline.RecallStage;
import org.bytedeco.pytorch.deploy.serving.pipeline.RerankStage;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Factories for a batteries-included engineering stack used by samples / demos.
 */
public final class RecommendEngineering {

    private RecommendEngineering() {}

    public static String version() {
        return "0.1.0-recommend-engineering";
    }

    /**
     * Standard recsys experiment layers (recall / coarse / fine / rerank / mix / ui).
     */
    public static LayeredExperimentManager standardExperimentLayers() {
        LayeredExperimentManager mgr = new LayeredExperimentManager(true);
        mgr.createLayer("layer_recall", "Recall channel experiments", DiversionUnit.USER_ID);
        mgr.createLayer("layer_coarse", "Coarse rank experiments", DiversionUnit.USER_ID);
        mgr.createLayer("layer_fine", "Fine rank model experiments", DiversionUnit.USER_ID);
        mgr.createLayer("layer_rerank", "Re-rank / diversity experiments", DiversionUnit.USER_ID);
        mgr.createLayer("layer_mix", "Mix-rank / insert experiments", DiversionUnit.USER_ID);
        mgr.createLayer("layer_ui", "UI / layout experiments", DiversionUnit.DEVICE_ID);
        return mgr;
    }

    /**
     * Example model experiment on the fine-rank layer: 50/50 control vs treatment.
     */
    public static Experiment buildFineRankExperiment(
            String id, double trafficPercent, String controlModel, String treatmentModel) {
        return Experiment.builder(id, "layer_fine")
                .name("fine-rank-model-exp")
                .owner("recsys")
                .diversionUnit(DiversionUnit.USER_ID)
                .trafficPercent(trafficPercent)
                .hypothesis("treatment fine-rank model improves CTR without hurting dwell")
                .primaryMetric("ctr")
                .primaryMetric("dwell_time")
                .guardrailMetric("error_rate")
                .guardrailMetric("p99_latency_ms")
                .addVariant(Variant.builder("control")
                        .control(true)
                        .trafficWeight(1.0)
                        .parameter("fine.model_id", controlModel)
                        .build())
                .addVariant(Variant.builder("treatment")
                        .control(false)
                        .trafficWeight(1.0)
                        .parameter("fine.model_id", treatmentModel)
                        .build())
                .status(ExperimentStatus.DRAFT)
                .build();
    }

    public static OfflineEvaluator standardOfflineEvaluator() {
        return OfflineEvaluator.builder()
                .rankingKs(5, 10, 20)
                .computeCalibration(true)
                .calibrationBins(10)
                .build();
    }

    public static ExperimentAnalyzer standardAnalyzer() {
        return ExperimentAnalyzer.builder()
                .alpha(0.05)
                .srmAlpha(0.001)
                .minSamplePerArm(1000L)
                .addGuardrail(Guardrail.srm("srm", 0.001, Guardrail.Action.KILL))
                .addGuardrail(Guardrail.relativeDrop("ctr_drop", "ctr", 0.02, Guardrail.Action.KILL))
                .addGuardrail(Guardrail.treatmentAbove("err_rate", "error_rate", 0.01, Guardrail.Action.PAUSE))
                .addGuardrail(Guardrail.minSample("min_n", 1000L))
                .build();
    }

    public static PipelineOrchestrator demoPipeline(List<Candidate> hotList) {
        RecallStage recall = new RecallStage(Arrays.asList(
                RecallStage.staticChannel("hot", hotList)));
        CoarseRankStage coarse = new CoarseRankStage(CoarseRankStage.passThrough());
        FineRankStage fine = new FineRankStage(FineRankStage.fromSingle((ctx, c) -> c.score()));
        RerankStage rerank = new RerankStage(Arrays.asList(
                RerankStage.dedup(),
                RerankStage.mmr(0.7)));
        MixRankStage mix = new MixRankStage(20);
        return PipelineOrchestrator.builder()
                .standardCascade(recall, coarse, fine, rerank, mix)
                .ultimateFallback(hotList)
                .build();
    }

    public static DeploymentController demoDeployController(String serviceName) {
        DeploymentController.InMemoryClusterOps ops = new DeploymentController.InMemoryClusterOps();
        return new DeploymentController(serviceName, ops, (versionId, gate) -> true);
    }

    public static TrafficRouter demoRouter(String stableId, String canaryId) {
        TrafficRouter router = new TrafficRouter("recommend_api");
        router.addUpstream(stableId, "stable.rank.svc", 100.0);
        router.addUpstream(canaryId, "canary.rank.svc", 0.0);
        router.addHeaderRule("X-Rec-Canary", "1", canaryId);
        return router;
    }

    public static ServiceLevel standardSlo() {
        return ServiceLevel.standardRecsys();
    }

    public static MetricsRegistry metrics() {
        return new MetricsRegistry();
    }

    public static DegradationPolicy degradationPolicy() {
        return new DegradationPolicy();
    }

    public static FallbackStrategy hotListFallback(List<Candidate> hot) {
        return new FallbackStrategy().addHotList("global_hot", hot);
    }

    public static HealthChecker basicHealth() {
        return new HealthChecker()
                .addProbe("liveness", () -> HealthChecker.Status.UP)
                .addProbe("model_loaded", () -> HealthChecker.Status.UP);
    }

    public static ReplicaScaler defaultScaler() {
        return new ReplicaScaler(ReplicaScaler.Config.defaults());
    }

    public static ModelRegistry modelRegistry() {
        return new ModelRegistry();
    }

    public static OnlineMetricsCollector onlineMetrics() {
        return new OnlineMetricsCollector();
    }

    public static double[] defaultCanaryStages() {
        return TrafficSplitter.defaultCanaryStages();
    }

    /** Module index for documentation / discovery. */
    public static List<String> modules() {
        List<String> m = new ArrayList<>();
        m.add("abtest: LayeredExperimentManager, BucketAssigner, StatisticalTest(CUPED/SRM/Welch), Guardrail, ExperimentAnalyzer");
        m.add("offline: OfflineEvaluator, HoldoutSplitter, CalibrationChecker, AATestRunner");
        m.add("serving.pipeline: Recall → Coarse → Fine → Rerank → Mix + PipelineOrchestrator");
        m.add("serving.deploy: Canary / BlueGreen / Rolling / InPlace / Rollback + ReplicaScaler");
        m.add("serving.gateway: TrafficRouter (sticky, header, region, shadow)");
        m.add("ops: MetricsRegistry, ServiceLevel(SLO), HealthChecker/Inspector, CircuitBreaker, DegradationPolicy, FallbackStrategy, RateLimiter");
        m.add("modelops: ModelRegistry, ShadowServing, DriftDetector(PSI), OnlineLearningHook, FeatureStoreSnapshot");
        return m;
    }
}
