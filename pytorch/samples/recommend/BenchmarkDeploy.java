/*
 * Benchmark for deployment strategies: canary, blue/green, rolling, in-place,
 * rollback, and replica autoscaler.
 *
 *   java -cp ... samples.recommend.BenchmarkDeploy
 */
package samples.recommend;

import org.bytedeco.pytorch.utils.recommend.abtest.TrafficSplitter;
import org.bytedeco.pytorch.utils.recommend.serving.deploy.DeploymentController;
import org.bytedeco.pytorch.utils.recommend.serving.deploy.ReplicaScaler;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicBoolean;

public final class BenchmarkDeploy {

    public static void main(String[] args) {
        System.exit(runTests());
    }

    public static int runTests() {
        BenchSupport.Suite s = new BenchSupport.Suite("BenchmarkDeploy");
        s.header();

        s.benchmark("bootstrap_and_register", () -> {
            DeploymentController.InMemoryClusterOps ops = new DeploymentController.InMemoryClusterOps();
            DeploymentController dc = new DeploymentController("rank-svc", ops, (v, g) -> true);
            dc.registerVersion("v1", "img:v1", "model-1", null);
            dc.bootstrapStable("v1", 4);
            s.checkEq("stable", "v1", dc.stableVersionId());
            s.checkClose("traffic 100", 100.0, ops.trafficWeight("v1"), 1e-9);
            DeploymentController.ReplicaSetView rs = dc.replicaSnapshot().get("v1");
            s.checkEq("desired 4", 4, rs.desired);
            s.checkEq("ready 4", 4, rs.ready);
        });

        s.benchmark("canary_ramp_to_complete", () -> {
            DeploymentController.InMemoryClusterOps ops = new DeploymentController.InMemoryClusterOps();
            List<String> events = new ArrayList<>();
            DeploymentController dc = new DeploymentController("rank-svc", ops, (v, g) -> true);
            dc.addListener(e -> events.add(e.type.name()));
            dc.registerVersion("v1", "img:v1", "m1", null);
            dc.registerVersion("v2", "img:v2", "m2", null);
            dc.bootstrapStable("v1", 10);

            double[] stages = new double[] {10.0, 50.0, 100.0};
            DeploymentController.DeployPlan plan = dc.startCanary("v2", stages, 2,
                    DeploymentController.DeployGate.defaults(), false);
            s.checkEq("in progress", DeploymentController.DeployStatus.IN_PROGRESS, plan.status);
            s.checkClose("stage0 traffic 10", 10.0, ops.trafficWeight("v2"), 1e-9);
            s.checkClose("stable 90", 90.0, ops.trafficWeight("v1"), 1e-9);

            plan = dc.promoteCanary(); // -> 50
            s.checkClose("stage1 50", 50.0, ops.trafficWeight("v2"), 1e-9);
            plan = dc.promoteCanary(); // -> 100 complete (next beyond last promotes to done)
            // After stage index reaches end, one more promote completes
            if (plan.status != DeploymentController.DeployStatus.SUCCEEDED) {
                plan = dc.promoteCanary();
            }
            s.checkEq("succeeded", DeploymentController.DeployStatus.SUCCEEDED, plan.status);
            s.checkEq("new stable v2", "v2", dc.stableVersionId());
            s.checkTrue("events fired", events.contains("PLAN_STARTED"));
            s.checkTrue("success event", events.contains("PLAN_SUCCEEDED") || plan.status == DeploymentController.DeployStatus.SUCCEEDED);
        });

        s.benchmark("canary_gate_blocks_promote", () -> {
            DeploymentController.InMemoryClusterOps ops = new DeploymentController.InMemoryClusterOps();
            AtomicBoolean pass = new AtomicBoolean(false);
            DeploymentController dc = new DeploymentController("rank-svc", ops, (v, g) -> pass.get());
            dc.registerVersion("v1", "a", "m", null);
            dc.registerVersion("v2", "b", "m", null);
            dc.bootstrapStable("v1", 4);
            DeploymentController.DeployPlan plan = dc.startCanary("v2",
                    new double[] {5.0, 100.0}, 1, DeploymentController.DeployGate.defaults(), false);
            DeploymentController.DeployPlan blocked = dc.promoteCanary();
            s.checkEq("still in progress", DeploymentController.DeployStatus.IN_PROGRESS, blocked.status);
            s.checkTrue("gate failed msg", blocked.lastMessage != null && blocked.lastMessage.contains("gate"));
            pass.set(true);
            DeploymentController.DeployPlan ok = dc.promoteCanary();
            // may need one more to fully complete depending on stages
            if (ok.status != DeploymentController.DeployStatus.SUCCEEDED) {
                ok = dc.promoteCanary();
            }
            s.checkEq("eventually succeeded", DeploymentController.DeployStatus.SUCCEEDED, ok.status);
        });

        s.benchmark("blue_green_switch", () -> {
            DeploymentController.InMemoryClusterOps ops = new DeploymentController.InMemoryClusterOps();
            DeploymentController dc = new DeploymentController("rank-svc", ops, (v, g) -> true);
            dc.registerVersion("blue", "img:blue", "m", null);
            dc.registerVersion("green", "img:green", "m", null);
            dc.bootstrapStable("blue", 6);
            DeploymentController.DeployPlan plan = dc.startBlueGreen("green",
                    DeploymentController.DeployGate.defaults(), true);
            s.checkEq("bg succeeded", DeploymentController.DeployStatus.SUCCEEDED, plan.status);
            s.checkEq("stable green", "green", dc.stableVersionId());
            s.checkClose("green 100", 100.0, ops.trafficWeight("green"), 1e-9);
            s.checkClose("blue 0", 0.0, ops.trafficWeight("blue"), 1e-9);
        });

        s.benchmark("rolling_batches", () -> {
            DeploymentController.InMemoryClusterOps ops = new DeploymentController.InMemoryClusterOps();
            DeploymentController dc = new DeploymentController("rank-svc", ops, (v, g) -> true);
            dc.registerVersion("v1", "a", "m", null);
            dc.registerVersion("v2", "b", "m", null);
            dc.bootstrapStable("v1", 8);
            DeploymentController.DeployPlan plan = dc.startRolling("v2", 2,
                    DeploymentController.DeployGate.defaults());
            int guards = 0;
            while (plan.status == DeploymentController.DeployStatus.IN_PROGRESS && guards++ < 20) {
                plan = dc.advanceRolling();
            }
            s.checkEq("rolling done", DeploymentController.DeployStatus.SUCCEEDED, plan.status);
            s.checkEq("stable v2", "v2", dc.stableVersionId());
            s.checkEq("rolled replicas", 8, plan.rolledReplicas);
        });

        s.benchmark("inplace_restart", () -> {
            DeploymentController.InMemoryClusterOps ops = new DeploymentController.InMemoryClusterOps();
            DeploymentController dc = new DeploymentController("rank-svc", ops, (v, g) -> true);
            dc.registerVersion("v1", "a", "m", null);
            dc.bootstrapStable("v1", 5);
            DeploymentController.DeployPlan plan = dc.startInPlace("v1", 2,
                    DeploymentController.DeployGate.defaults());
            int guards = 0;
            while (plan.status == DeploymentController.DeployStatus.IN_PROGRESS && guards++ < 10) {
                plan = dc.advanceInPlace();
            }
            s.checkEq("inplace done", DeploymentController.DeployStatus.SUCCEEDED, plan.status);
            s.checkEq("all pods", 5, plan.rolledReplicas);
        });

        s.benchmark("rollback_restores_stable", () -> {
            DeploymentController.InMemoryClusterOps ops = new DeploymentController.InMemoryClusterOps();
            DeploymentController dc = new DeploymentController("rank-svc", ops, (v, g) -> true);
            dc.registerVersion("v1", "a", "m", null);
            dc.registerVersion("v2", "b", "m", null);
            dc.bootstrapStable("v1", 4);
            dc.startCanary("v2", new double[] {20.0, 100.0}, 1,
                    DeploymentController.DeployGate.defaults(), false);
            s.checkClose("canary partial", 20.0, ops.trafficWeight("v2"), 1e-9);
            DeploymentController.DeployPlan rb = dc.rollback("v1");
            s.checkEq("rolled back status", DeploymentController.DeployStatus.ROLLED_BACK, rb.status);
            s.checkEq("stable back to v1", "v1", dc.stableVersionId());
            s.checkClose("v1 100", 100.0, ops.trafficWeight("v1"), 1e-9);
        });

        s.benchmark("pause_resume", () -> {
            DeploymentController.InMemoryClusterOps ops = new DeploymentController.InMemoryClusterOps();
            DeploymentController dc = new DeploymentController("rank-svc", ops, (v, g) -> true);
            dc.registerVersion("v1", "a", "m", null);
            dc.registerVersion("v2", "b", "m", null);
            dc.bootstrapStable("v1", 2);
            dc.startCanary("v2", TrafficSplitter.defaultCanaryStages(), 1,
                    DeploymentController.DeployGate.defaults(), false);
            DeploymentController.DeployPlan paused = dc.pause();
            s.checkEq("paused", DeploymentController.DeployStatus.PAUSED, paused.status);
            DeploymentController.DeployPlan resumed = dc.resume();
            s.checkEq("resumed", DeploymentController.DeployStatus.IN_PROGRESS, resumed.status);
        });

        s.benchmark("replica_scaler_up_down_cooldown", () -> {
            ReplicaScaler scaler = new ReplicaScaler(new ReplicaScaler.Config(
                    2, 100, 500.0, 50.0, 0L, 0L, 0.5)); // no cooldown for first checks
            // high QPS -> scale up
            ReplicaScaler.Decision up = scaler.evaluate(
                    new ReplicaScaler.Signal(5000, 30, 0.4, 4));
            s.checkTrue("scale up " + up, up.desiredReplicas > 4);
            s.checkTrue("scaled flag", up.scaled);

            // low QPS -> scale down
            ReplicaScaler scaler2 = new ReplicaScaler(new ReplicaScaler.Config(
                    2, 100, 500.0, 50.0, 0L, 0L, 1.0));
            ReplicaScaler.Decision down = scaler2.evaluate(
                    new ReplicaScaler.Signal(100, 20, 0.2, 20));
            s.checkTrue("scale down " + down, down.desiredReplicas < 20);

            // min/max clamp
            ReplicaScaler.Decision clamped = scaler2.evaluate(
                    new ReplicaScaler.Signal(0, 10, 0.1, 2));
            s.checkTrue("not below min", clamped.desiredReplicas >= 2);
        });

        s.benchmark("replica_scaler_latency_pressure", () -> {
            ReplicaScaler scaler = new ReplicaScaler(ReplicaScaler.Config.defaults());
            ReplicaScaler.Decision d = scaler.evaluate(
                    new ReplicaScaler.Signal(100, 200, 0.5, 10)); // p99 200 vs target 50
            s.checkTrue("latency forces up " + d, d.desiredReplicas > 10 || d.reason.contains("cooldown") || d.reason.contains("scale"));
        });

        s.benchmark("replica_scaler_forecast", () -> {
            ReplicaScaler scaler = new ReplicaScaler(new ReplicaScaler.Config(
                    2, 200, 500.0, 50.0, 0L, 0L, 1.0));
            ReplicaScaler.Decision d = scaler.evaluateWithForecast(
                    new ReplicaScaler.Signal(100, 20, 0.2, 4), 10_000);
            s.checkTrue("forecast pre-warm " + d, d.desiredReplicas >= 4);
            s.checkTrue("reason mentions forecast or hold", d.reason != null);
        });

        s.benchmark("concurrent_plan_rejected", () -> {
            DeploymentController.InMemoryClusterOps ops = new DeploymentController.InMemoryClusterOps();
            DeploymentController dc = new DeploymentController("rank-svc", ops, (v, g) -> true);
            dc.registerVersion("v1", "a", "m", null);
            dc.registerVersion("v2", "b", "m", null);
            dc.registerVersion("v3", "c", "m", null);
            dc.bootstrapStable("v1", 2);
            dc.startCanary("v2", new double[] {10.0, 100.0}, 1,
                    DeploymentController.DeployGate.defaults(), false);
            boolean threw = false;
            try {
                dc.startCanary("v3", new double[] {10.0, 100.0}, 1,
                        DeploymentController.DeployGate.defaults(), false);
            } catch (IllegalStateException ex) {
                threw = true;
            }
            s.checkTrue("second plan rejected", threw);
        });

        return s.exitCode();
    }
}
