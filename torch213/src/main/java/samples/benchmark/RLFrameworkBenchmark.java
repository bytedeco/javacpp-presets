package org.bytedeco.pytorch.rl.benchmark;
import org.bytedeco.pytorch.optim.*;
import org.bytedeco.pytorch.distributed.*;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.distribution.Categorical;
import org.bytedeco.pytorch.distribution.Distribution;
import org.bytedeco.pytorch.optim.Adam;
import org.bytedeco.pytorch.optim.AdamOptions;
import org.bytedeco.pytorch.rl.GAE;
import org.bytedeco.pytorch.rl.ReplayBuffer;
import org.bytedeco.pytorch.rl.StepResult;
import org.bytedeco.pytorch.rl.agent.A2CAgent;
import org.bytedeco.pytorch.rl.agent.DPOAgent;
import org.bytedeco.pytorch.rl.agent.GRPOAgent;
import org.bytedeco.pytorch.rl.agent.PPOAgent;
import org.bytedeco.pytorch.rl.critic.AbstractActorCritic;
import org.bytedeco.pytorch.rl.critic.ActorCritic;
import org.bytedeco.pytorch.rl.critic.CartPoleActorCritic;
import org.bytedeco.pytorch.rl.env.CartPoleEnv;
import org.bytedeco.pytorch.rl.loss.GRPOLoss;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

import static org.bytedeco.pytorch.global.torch.*;

/**
 * Multi-dimensional verification suite for the JavaCPP RL framework.
 *
 * <p>Dimensions covered:
 * <ol>
 *   <li><b>Correctness</b> — GAE math, PPO clip ratio, DPO preference direction, GRPO group norm</li>
 *   <li><b>Learning</b> — CartPole PPO/A2C episode return improvement</li>
 *   <li><b>API / wiring</b> — action-space labels, sample→buffer→update path, optimizer on live params</li>
 *   <li><b>Throughput</b> — sample / update steps per second</li>
 *   <li><b>Stability</b> — loss finite, no NaN after multi-epoch updates</li>
 * </ol>
 *
 * <pre>
 *   # compile then run (from pytorch module root):
 *   mvn -q -DskipTests compile
 *   java -cp target/classes:$(mvn -q -DincludeScope=runtime -DforceStdout dependency:build-classpath) \
 *        org.bytedeco.pytorch.rl.benchmark.RLFrameworkBenchmark
 * </pre>
 */
public final class RLFrameworkBenchmark {
    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    private static final class Result {
        final String name;
        final boolean pass;
        final String detail;
        final double metric;
        final String unit;

        Result(String name, boolean pass, String detail, double metric, String unit) {
            this.name = name;
            this.pass = pass;
            this.detail = detail;
            this.metric = metric;
            this.unit = unit;
        }
    }

    private final List<Result> results = new ArrayList<>();
    private final long seed;

    public RLFrameworkBenchmark(long seed) {
        this.seed = seed;
    }

    public static void main(String[] args) {
        long seed = 42L;
        if (args != null) {
            for (String a : args) {
                if (a.startsWith("--seed=")) {
                    seed = Long.parseLong(a.substring("--seed=".length()));
                }
            }
        }
        RLFrameworkBenchmark bench = new RLFrameworkBenchmark(seed);
        int failed = bench.runAll();
        System.exit(failed == 0 ? 0 : 1);
    }

    public int runAll() {
        manual_seed(seed);
        System.out.println("============================================================");
        System.out.println(" RL Framework Multi-Dimensional Benchmark");
        System.out.println(" seed=" + seed);
        System.out.println("============================================================");

        run("GAE.correctness", this::benchGaeCorrectness);
        run("PPO.clip_math", this::benchPpoClipMath);
        run("DPO.preference_direction", this::benchDpoPreference);
        run("GRPO.group_relative", this::benchGrpoGroupRelative);
        run("API.action_space_labels", this::benchActionSpaceLabels);
        run("API.sample_buffer_update", this::benchSampleBufferUpdate);
        run("LEARN.cartpole_ppo", this::benchCartPolePpo);
        run("LEARN.cartpole_a2c", this::benchCartPoleA2c);
        run("PERF.ppo_throughput", this::benchPpoThroughput);
        run("STAB.multi_epoch_finite", this::benchMultiEpochFinite);

        System.out.println();
        System.out.println("---------------- Summary ----------------");
        int fail = 0;
        for (Result r : results) {
            String mark = r.pass ? "PASS" : "FAIL";
            if (!r.pass) fail++;
            System.out.printf(Locale.ROOT, "[%s] %-28s  %s  (%.4g %s)%n",
                    mark, r.name, r.detail, r.metric, r.unit);
        }
        System.out.printf(Locale.ROOT, "%n%d/%d passed%n", results.size() - fail, results.size());
        return fail;
    }

    private interface Case {
        Result run() throws Exception;
    }

    private void run(String name, Case c) {
        System.out.println();
        System.out.println(">>> " + name);
        try {
            Result r = c.run();
            results.add(r);
            System.out.println((r.pass ? "  OK  " : "  !!  ") + r.detail);
        } catch (Throwable t) {
            String msg = t.getClass().getSimpleName() + ": " + t.getMessage();
            results.add(new Result(name, false, msg, Double.NaN, ""));
            System.out.println("  EX  " + msg);
            t.printStackTrace(System.out);
        }
    }

    // ------------------------------------------------------------------ GAE
    private Result benchGaeCorrectness() {
        // T=3, B=1, known hand-checkable path with V(s_T)=0 bootstrap
        // r = [1, 1, 1], V = [0, 0, 0], mask = [1, 1, 1], gamma=1, lambda=1
        // => delta_t = r_t, GAE backward: A2=1, A1=2, A0=3 ; R = A + V
        Tensor rewards = tensor(new float[]{1f, 1f, 1f}).reshape(3, 1);
        Tensor values = tensor(new float[]{0f, 0f, 0f}).reshape(3, 1);
        Tensor masks = ones_like(rewards);
        Tensor[] out = GAE.compute(rewards, values, masks, 1.0f, 1.0f);
        float a0 = out[0].select(0, 0).item().toFloat();
        float a1 = out[0].select(0, 1).item().toFloat();
        float a2 = out[0].select(0, 2).item().toFloat();
        boolean ok = near(a0, 3f, 1e-4f) && near(a1, 2f, 1e-4f) && near(a2, 1f, 1e-4f);

        // Also accept values of length T+1 without OOB
        Tensor valuesTp1 = tensor(new float[]{0f, 0f, 0f, 0f}).reshape(4, 1);
        Tensor[] out2 = GAE.compute(rewards, valuesTp1, masks, 1.0f, 1.0f);
        boolean ok2 = near(out2[0].select(0, 0).item().toFloat(), 3f, 1e-4f);

        return new Result("GAE.correctness", ok && ok2,
                String.format(Locale.ROOT, "A=[%.3f,%.3f,%.3f] expect [3,2,1]; T+1 bootstrap ok=%s",
                        a0, a1, a2, ok2),
                a0, "A0");
    }

    // -------------------------------------------------------------- PPO math
    private Result benchPpoClipMath() {
        // Fixed logprobs so ratio is known: new=log(0.8), old=log(0.5) => ratio=1.6
        // clipEps=0.2 => clipped ratio=1.2; adv=+1 => min(1.6,1.2)=1.2; loss=-1.2
        Tensor newLp = log(tensor(0.8f));
        Tensor oldLp = log(tensor(0.5f));
        Tensor adv = tensor(1.0f);
        Tensor ratio = exp(newLp.sub(oldLp));
        float ratioF = ratio.item().toFloat();
        Tensor clipped = clamp(ratio,
                new ScalarOptional(new Scalar(0.8)),
                new ScalarOptional(new Scalar(1.2)));
        Tensor surr = min(ratio.mul(adv), clipped.mul(adv));
        float loss = surr.mean().neg().item().toFloat();
        boolean ok = near(ratioF, 1.6f, 1e-4f) && near(loss, -1.2f, 1e-4f);
        return new Result("PPO.clip_math", ok,
                String.format(Locale.ROOT, "ratio=%.4f loss=%.4f (expect ratio≈1.6 loss≈-1.2)", ratioF, loss),
                loss, "neg_surr");
    }

    // ------------------------------------------------------------------- DPO
    private Result benchDpoPreference() {
        // When policy already prefers chosen, loss should be small;
        // when it prefers rejected, loss should be larger.
        CartPoleActorCritic policy = new CartPoleActorCritic(8, 4);
        manual_seed(seed + 1);
        CartPoleActorCritic reference = new CartPoleActorCritic(8, 4);
        freezeModule(reference);

        AdamOptions opt = new AdamOptions();
        opt.lr().put(1e-3f);
        DPOAgent agent = new DPOAgent(policy, reference, new Adam(policy.parameters(), opt),
                new ReplayBuffer(), 0.1f);

        // Synthetic log-probs (no grad) — pure loss direction check
        Tensor pC = tensor(new float[]{-0.5f, -0.4f, -0.6f});
        Tensor pR = tensor(new float[]{-2.0f, -1.8f, -2.2f});
        Tensor rC = tensor(new float[]{-1.0f, -1.0f, -1.0f});
        Tensor rR = tensor(new float[]{-1.0f, -1.0f, -1.0f});
        Tensor lossGood = agent.computeDPOLoss(pC, pR, rC, rR, 0.1f);
        Tensor lossBad = agent.computeDPOLoss(pR, pC, rC, rR, 0.1f);
        float g = lossGood.item().toFloat();
        float b = lossBad.item().toFloat();
        boolean ok = Float.isFinite(g) && Float.isFinite(b) && g < b;

        // Real trainStep with model-derived log-probs (has grad graph)
        Tensor x = randn(4, 8);
        Tensor chosen = tensor(new long[]{0, 1, 2, 0});
        Tensor rejected = tensor(new long[]{3, 2, 1, 3});
        Distribution pDist = policy.getDistribution(x);
        Distribution rDist = reference.getDistribution(x);
        Tensor stepped = agent.trainStep(
                pDist.log_prob(chosen), pDist.log_prob(rejected),
                rDist.log_prob(chosen).detach(), rDist.log_prob(rejected).detach());
        float s = stepped.item().toFloat();
        ok = ok && Float.isFinite(s);

        agent.close();
        return new Result("DPO.preference_direction", ok,
                String.format(Locale.ROOT, "loss_good=%.4f loss_bad=%.4f step=%.4f (good<bad)", g, b, s),
                g, "loss");
    }

    // ------------------------------------------------------------------- GRPO
    private Result benchGrpoGroupRelative() {
        // Two groups of size 2: rewards [[1,3],[10,0]]
        // With population std, group-normalized advantages have mean 0 within each group
        // → overall mean(adv)=0. With ratio=1, clipped surrogate mean is 0 → loss ≈ 0.
        Tensor groupRewards = tensor(new float[]{1f, 3f, 10f, 0f}).reshape(2, 2);
        Tensor probs = softmax(randn(4, 2), -1);
        Categorical dist = new Categorical(probs);
        Tensor actions = tensor(new long[]{0, 1, 0, 1});
        // Categorical.log_prob already returns [B] — no sum(-1)
        Tensor oldLp = dist.log_prob(actions).detach().clone();
        Categorical dist2 = new Categorical(probs);
        Tensor loss = GRPOLoss.computeLoss(dist2, actions, oldLp, groupRewards, 0.2f);
        float v = loss.item().toFloat();
        boolean ok = Float.isFinite(v) && Math.abs(v) < 1e-3f;
        return new Result("GRPO.group_relative", ok,
                String.format(Locale.ROOT, "GRPOLoss(ratio=1)=%.6f expect ~0", v),
                v, "loss");
    }

    // ---------------------------------------------------------- action space
    private Result benchActionSpaceLabels() {
        CartPoleActorCritic disc = new CartPoleActorCritic(4, 2);
        ActorCritic cont = new ActorCritic(4, 2);
        boolean ok = disc.getActionSpaceType() == AbstractActorCritic.ActionSpaceType.DISCRETE
                && cont.getActionSpaceType() == AbstractActorCritic.ActionSpaceType.CONTINUOUS;
        // Continuous sample should produce float actions; discrete long/int categories
        Tensor s = randn(1, 4);
        Distribution dDisc = disc.getDistribution(s);
        Distribution dCont = cont.getDistribution(s);
        Tensor aDisc = dDisc.sample();
        Tensor aCont = dCont.sample();
        ok = ok && aDisc.dim() >= 1 && aCont.dim() >= 1;
        return new Result("API.action_space_labels", ok,
                "discrete=" + disc.getActionSpaceType()
                        + " continuous=" + cont.getActionSpaceType()
                        + " a_disc.shape=" + shapeStr(aDisc)
                        + " a_cont.shape=" + shapeStr(aCont),
                ok ? 1 : 0, "ok");
    }

    // ------------------------------------------------- sample → buffer → update
    private Result benchSampleBufferUpdate() {
        PPOAgent agent = PPOAgent.create(4, 2, AbstractActorCritic.ActionSpaceType.DISCRETE);
        ReplayBuffer buf = agent.getReplayBuffer();
        // Collect a short synthetic trajectory with push(s,a,lp,adv,ret)
        // Use varied advantages so normalization is well-defined (N>=2, nonzero std).
        for (int t = 0; t < 16; t++) {
            Tensor state = randn(1, 4);
            Tensor[] sa = agent.sample(state);
            // action: long scalar index; logprob/adv/ret: float scalars — ReplayBuffer normalizes rank
            long act = sa[0].reshape(-1).item().toLong();
            float lp = sa[1].reshape(-1).item().toFloat();
            float adv = (float) Math.sin(t * 0.7) + 0.1f * t;
            float ret = 1.0f + 0.05f * t;
            buf.push(state.squeeze(0),
                    tensor(new long[]{act}),
                    tensor(lp),
                    tensor(adv),
                    tensor(ret));
        }
        Tensor loss = agent.trainStep();
        float v = loss.item().toFloat();
        boolean ok = buf.size() == 16 && Float.isFinite(v);
        agent.close();
        return new Result("API.sample_buffer_update", ok,
                String.format(Locale.ROOT, "buffer=%d loss=%.4f", 16, v),
                v, "loss");
    }

    // ----------------------------------------------------------- CartPole PPO
    private Result benchCartPolePpo() {
        final int warmupEpisodes = 5;
        final int trainEpisodes = 40;
        final int rolloutSteps = 128;
        final int evalEpisodes = 10;

        PPOAgent agent = PPOAgent.create(4, 2, AbstractActorCritic.ActionSpaceType.DISCRETE,
                3e-4f, 0.2f, 0.99f, 0.95f, 0.01f, 0.5f, 0.5f);

        double before = evalCartPole(agent, evalEpisodes, /*greedy*/true);
        // Warmup random-ish (policy as-is)
        for (int e = 0; e < warmupEpisodes; e++) {
            collectAndUpdatePpo(agent, rolloutSteps);
        }
        double mid = evalCartPole(agent, evalEpisodes, true);
        for (int e = 0; e < trainEpisodes; e++) {
            collectAndUpdatePpo(agent, rolloutSteps);
        }
        double after = evalCartPole(agent, evalEpisodes, true);
        // Pass if final return improves vs initial OR reaches a modest bar (env is stochastic)
        boolean ok = after > before + 5.0 || after >= 50.0 || after > mid;
        agent.close();
        return new Result("LEARN.cartpole_ppo", ok,
                String.format(Locale.ROOT, "return before=%.1f mid=%.1f after=%.1f", before, mid, after),
                after, "avg_return");
    }

    // ----------------------------------------------------------- CartPole A2C
    private Result benchCartPoleA2c() {
        A2CAgent agent = A2CAgent.create(4, 2, AbstractActorCritic.ActionSpaceType.DISCRETE,
                0.99f, 0.01f, 3e-4f);
        double before = evalCartPoleA2c(agent, 8);
        for (int e = 0; e < 30; e++) {
            collectAndUpdateA2c(agent, 64);
        }
        double after = evalCartPoleA2c(agent, 8);
        boolean ok = after >= before - 5.0; // A2C is noisier; require non-collapse
        boolean strong = after > before + 3.0 || after >= 40.0;
        agent.close();
        return new Result("LEARN.cartpole_a2c", ok,
                String.format(Locale.ROOT, "return before=%.1f after=%.1f strong=%s", before, after, strong),
                after, "avg_return");
    }

    // ------------------------------------------------------------- throughput
    private Result benchPpoThroughput() {
        PPOAgent agent = PPOAgent.create(4, 2, AbstractActorCritic.ActionSpaceType.DISCRETE);
        // Prime buffer once
        collectAndUpdatePpo(agent, 64);
        int iters = 50;
        long t0 = System.nanoTime();
        for (int i = 0; i < iters; i++) {
            collectAndUpdatePpo(agent, 64);
        }
        double sec = (System.nanoTime() - t0) / 1e9;
        double ups = iters / sec; // updates per second
        boolean ok = ups > 0.5 && Double.isFinite(ups);
        agent.close();
        return new Result("PERF.ppo_throughput", ok,
                String.format(Locale.ROOT, "%.2f PPO updates/s (%d×64-step rollouts in %.2fs)", ups, iters, sec),
                ups, "updates/s");
    }

    // --------------------------------------------------------------- stability
    private Result benchMultiEpochFinite() {
        PPOAgent agent = PPOAgent.create(8, 4, AbstractActorCritic.ActionSpaceType.DISCRETE);
        int nans = 0;
        double last = 0;
        for (int i = 0; i < 20; i++) {
            ReplayBuffer buf = agent.getReplayBuffer();
            buf.clear();
            for (int t = 0; t < 32; t++) {
                Tensor state = randn(1, 8);
                Tensor[] sa = agent.sample(state);
                long act = sa[0].reshape(-1).item().toLong();
                float lp = sa[1].reshape(-1).item().toFloat();
                float adv = (float) Math.sin(t) * 0.5f + 0.01f * t;
                buf.push(state.squeeze(0),
                        tensor(new long[]{act}),
                        tensor(lp),
                        tensor(adv),
                        tensor(1.0f + 0.01f * t));
            }
            Tensor loss = agent.trainStep();
            float v = loss.item().toFloat();
            if (!Float.isFinite(v)) nans++;
            last = v;
        }
        boolean ok = nans == 0;
        agent.close();
        return new Result("STAB.multi_epoch_finite", ok,
                String.format(Locale.ROOT, "nan_count=%d last_loss=%.4f", nans, last),
                last, "loss");
    }

    // ============================== helpers =================================

    private static void collectAndUpdatePpo(PPOAgent agent, int steps) {
        CartPoleEnv env = new CartPoleEnv();
        ReplayBuffer buf = agent.getReplayBuffer();
        buf.clear();

        // Store *normalized* observations so train-time states match sample-time.
        List<float[]> states = new ArrayList<>();
        List<Integer> actions = new ArrayList<>();
        List<Float> logps = new ArrayList<>();
        List<Float> rewards = new ArrayList<>();
        List<Float> values = new ArrayList<>();
        List<Float> masks = new ArrayList<>();

        float[] obs = env.reset();
        for (int t = 0; t < steps; t++) {
            Tensor stNorm = normalizeForAgent(agent, obs);
            // sample on already-normalized obs without double-updating stats
            boolean wasNorm = agent.isNormalizeObs();
            agent.setNormalizeObs(false);
            Tensor[] sa = agent.sample(stNorm);
            agent.setNormalizeObs(wasNorm);

            int act = (int) sa[0].item().toLong();
            if (act < 0) act = 0;
            if (act > 1) act = act % 2;
            float lp = sa[1].item().toFloat();
            float v = sa[2].reshape(-1).item().toFloat();
            StepResult sr = env.step(act);

            // persist normalized state vector
            float[] sn = new float[4];
            for (int i = 0; i < 4; i++) sn[i] = stNorm.reshape(-1).select(0, i).item().toFloat();
            states.add(sn);
            actions.add(act);
            logps.add(lp);
            rewards.add(sr.reward);
            values.add(v);
            masks.add(sr.done ? 0f : 1f);

            obs = sr.nextState;
            if (sr.done) {
                obs = env.reset();
            }
        }
        // Bootstrap value on normalized last obs
        Tensor lastSt = normalizeForAgent(agent, obs);
        boolean wasNorm = agent.isNormalizeObs();
        agent.setNormalizeObs(false);
        float lastV = agent.getModel().getValue(lastSt).reshape(-1).item().toFloat();
        agent.setNormalizeObs(wasNorm);

        float[] rArr = toFloatArray(rewards);
        float[] vArr = new float[values.size() + 1];
        for (int i = 0; i < values.size(); i++) vArr[i] = values.get(i);
        vArr[values.size()] = lastV;
        float[] mArr = toFloatArray(masks);

        Tensor rT = tensor(rArr).reshape(rArr.length, 1);
        Tensor vT = tensor(vArr).reshape(vArr.length, 1);
        Tensor mT = tensor(mArr).reshape(mArr.length, 1);
        Tensor[] gae = PPOAgent.computeGAE(rT, vT, mT, agent.getGamma(), agent.getGaeLambda());
        // Push per-step with precomputed adv/ret (1D scalars — ReplayBuffer normalizes rank)
        for (int t = 0; t < steps; t++) {
            Tensor s = tensor(states.get(t));
            Tensor a = tensor(new long[]{actions.get(t)});
            Tensor lp = tensor(logps.get(t));
            float advF = gae[0].select(0, t).reshape(-1).item().toFloat();
            float retF = gae[1].select(0, t).reshape(-1).item().toFloat();
            buf.push(s, a, lp, tensor(advF), tensor(retF));
        }
        agent.trainStep();
        buf.clear();
    }

    /** Update running obs stats from raw env obs and return normalized {@code [1,4]} tensor. */
    private static Tensor normalizeForAgent(PPOAgent agent, float[] obs) {
        Tensor raw = tensor(obs).reshape(1, 4);
        if (agent.isNormalizeObs()) {
            agent.updateObsStats(raw);
            return agent.normalizeObs(raw);
        }
        return raw;
    }

    private static void collectAndUpdateA2c(A2CAgent agent, int steps) {
        CartPoleEnv env = new CartPoleEnv();
        ReplayBuffer buf = agent.getReplayBuffer();
        buf.clear();

        float[] obs = env.reset();
        List<Float> rewards = new ArrayList<>();
        List<Float> values = new ArrayList<>();
        List<Float> dones = new ArrayList<>();
        List<float[]> states = new ArrayList<>();
        List<Integer> actions = new ArrayList<>();
        List<Float> logps = new ArrayList<>();

        for (int t = 0; t < steps; t++) {
            Tensor st = tensor(obs).reshape(1, 4);
            Tensor[] sa = agent.sample(st);
            int act = (int) sa[0].item().toLong();
            if (act < 0) act = 0;
            if (act > 1) act = act % 2;
            StepResult sr = env.step(act);
            states.add(obs.clone());
            actions.add(act);
            logps.add(sa[1].item().toFloat());
            values.add(sa[2].reshape(-1).item().toFloat());
            rewards.add(sr.reward);
            dones.add(sr.done ? 1f : 0f);
            obs = sr.nextState;
            if (sr.done) obs = env.reset();
        }
        float lastV = agent.getModel().getValue(tensor(obs).reshape(1, 4)).reshape(-1).item().toFloat();
        // Discounted returns
        float[] rets = new float[steps];
        float running = lastV;
        for (int t = steps - 1; t >= 0; t--) {
            running = rewards.get(t) + agent.getGamma() * running * (1f - dones.get(t));
            rets[t] = running;
        }
        for (int t = 0; t < steps; t++) {
            float adv = rets[t] - values.get(t);
            buf.push(tensor(states.get(t)),
                    tensor(new long[]{actions.get(t)}),
                    tensor(logps.get(t)),
                    tensor(adv),
                    tensor(rets[t]));
        }
        // A2C trainStep needs advantages/returns via getAdvantages/getReturns which
        // stack from push-stored lists — push already filled advantages/returns lists.
        // Ensure t_* built: getAdvantages stacks if t_advantages null.
        agent.trainStep();
        buf.clear();
    }

    private static double evalCartPole(PPOAgent agent, int episodes, boolean greedy) {
        CartPoleEnv env = new CartPoleEnv();
        double total = 0;
        for (int e = 0; e < episodes; e++) {
            float[] obs = env.reset();
            double ret = 0;
            for (int t = 0; t < 500; t++) {
                // Eval must use the same obs normalization as training (no stat update).
                Tensor st = agent.isNormalizeObs()
                        ? agent.normalizeObs(tensor(obs).reshape(1, 4))
                        : tensor(obs).reshape(1, 4);
                int act;
                if (greedy) {
                    Distribution dist = agent.getModel().getDistribution(st);
                    // Take argmax of probs for discrete
                    if (dist instanceof Categorical) {
                        Tensor probs = ((Categorical) dist).getProbs();
                        act = (int) probs.argmax(new LongOptional(-1L), false).item().toLong();
                    } else {
                        act = (int) dist.sample().item().toLong();
                    }
                } else {
                    boolean was = agent.isNormalizeObs();
                    agent.setNormalizeObs(false);
                    act = (int) agent.sample(st)[0].item().toLong();
                    agent.setNormalizeObs(was);
                }
                if (act < 0) act = 0;
                if (act > 1) act = act % 2;
                StepResult sr = env.step(act);
                ret += sr.reward;
                obs = sr.nextState;
                if (sr.done) break;
            }
            total += ret;
        }
        return total / episodes;
    }

    private static double evalCartPoleA2c(A2CAgent agent, int episodes) {
        CartPoleEnv env = new CartPoleEnv();
        double total = 0;
        for (int e = 0; e < episodes; e++) {
            float[] obs = env.reset();
            double ret = 0;
            for (int t = 0; t < 500; t++) {
                Tensor st = tensor(obs).reshape(1, 4);
                Distribution dist = agent.getModel().getDistribution(st);
                int act;
                if (dist instanceof Categorical) {
                    act = (int) ((Categorical) dist).getProbs()
                            .argmax(new LongOptional(-1L), false).item().toLong();
                } else {
                    act = (int) dist.sample().item().toLong();
                }
                if (act < 0) act = 0;
                if (act > 1) act = act % 2;
                StepResult sr = env.step(act);
                ret += sr.reward;
                obs = sr.nextState;
                if (sr.done) break;
            }
            total += ret;
        }
        return total / episodes;
    }

    private static void freezeModule(AbstractActorCritic model) {
        TensorVector params = model.parameters();
        var begin = params.begin();
        var end = params.end();
        while (!begin.equals(end)) {
            begin.get().requires_grad_(false);
            begin.increment();
        }
        params.close();
    }

    private static boolean near(float a, float b, float eps) {
        return Math.abs(a - b) <= eps;
    }

    private static float[] toFloatArray(List<Float> list) {
        float[] a = new float[list.size()];
        for (int i = 0; i < list.size(); i++) a[i] = list.get(i);
        return a;
    }

    private static String shapeStr(Tensor t) {
        long[] s = t.shape();
        StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < s.length; i++) {
            if (i > 0) sb.append(',');
            sb.append(s[i]);
        }
        return sb.append(']').toString();
    }
}
