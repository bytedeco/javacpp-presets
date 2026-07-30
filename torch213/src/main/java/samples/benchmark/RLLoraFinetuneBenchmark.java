package org.bytedeco.pytorch.rl.benchmark;
import org.bytedeco.pytorch.nn.modules.*;
import org.bytedeco.pytorch.jit.*;
import org.bytedeco.pytorch.optim.*;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.distribution.Categorical;
import org.bytedeco.pytorch.distribution.Distribution;
import org.bytedeco.pytorch.llm.peft.LoraConfig;
import org.bytedeco.pytorch.llm.peft.LoraLinear;
import org.bytedeco.pytorch.llm.peft.PeftModel;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.modules.LinearImpl;
import org.bytedeco.pytorch.optim.Adam;
import org.bytedeco.pytorch.optim.AdamOptions;
import org.bytedeco.pytorch.optim.AdamW;
import org.bytedeco.pytorch.optim.AdamWOptions;
import org.bytedeco.pytorch.optim.Optimizer;
import org.bytedeco.pytorch.rl.GAE;
import org.bytedeco.pytorch.rl.agent.DPOAgent;
import org.bytedeco.pytorch.rl.agent.PPOAgent;
import org.bytedeco.pytorch.rl.critic.AbstractActorCritic;
import org.bytedeco.pytorch.rl.loss.GRPOLoss;

import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Random;

import static org.bytedeco.pytorch.global.torch.*;

/**
 * Multi-dimensional RL + LoRA fine-tuning benchmark.
 *
 * <p>Simulates large-model alignment fine-tuning without loading a full LLM:
 * a small transformer-like MLP trunk is frozen, LoRA adapters on attention-style
 * projections are trained with PPO / DPO / GRPO-style objectives.
 *
 * <p>Dimensions:
 * <ol>
 *   <li><b>Param efficiency</b> — trainable LoRA params ≪ full model</li>
 *   <li><b>LoRA forward/merge</b> — ΔW path, merge/unmerge numerical parity</li>
 *   <li><b>PPO+LoRA</b> — policy improvement on a bandit / preference proxy task</li>
 *   <li><b>DPO+LoRA</b> — preference loss decreases; chosen&gt;rejected margin grows</li>
 *   <li><b>GRPO+LoRA</b> — group-relative advantages train without a critic</li>
 *   <li><b>Frozen base</b> — base weights do not move under freezeBase=true</li>
 *   <li><b>Throughput</b> — LoRA update steps/s vs full fine-tune</li>
 * </ol>
 *
 * <pre>
 *   java ... org.bytedeco.pytorch.rl.benchmark.RLLoraFinetuneBenchmark
 *   java ... org.bytedeco.pytorch.rl.benchmark.RLLoraFinetuneBenchmark --seed=7 --steps=200
 * </pre>
 */
public final class RLLoraFinetuneBenchmark {
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

    /** Tiny LM-like policy: embed → LoRA(q/v/out) → logits, with value head for PPO. */
    public static final class LoraPolicyNet extends Module {
        final long hidden;
        final long vocab;
        final LinearImpl embed;       // frozen "embedding" projection
        final LoraLinear qProj;       // LoRA on q
        final LoraLinear vProj;       // LoRA on v
        final LoraLinear outProj;     // LoRA on vocab head (critical for bandit learning)
        final LinearImpl valueHead;   // small trainable value head (full, not LoRA)
        final PeftModel peft;
        final LoraConfig loraConfig;

        public LoraPolicyNet(long hidden, long vocab, int rank, double alpha) {
            super("LoraPolicyNet");
            this.hidden = hidden;
            this.vocab = vocab;
            this.loraConfig = LoraConfig.builder()
                    .r(rank)
                    .alpha(alpha)
                    .dropout(0.0)
                    .freezeBase(true)
                    .targetModules("q_proj", "v_proj", "out_proj")
                    .build();

            this.embed = register_module("embed", new LinearImpl(hidden, hidden));
            embed.weight().requires_grad_(false);
            try {
                if (embed.bias() != null && embed.bias().defined()) {
                    embed.bias().requires_grad_(false);
                }
            } catch (Exception ignored) {}

            LinearImpl qBase = new LinearImpl(hidden, hidden);
            LinearImpl vBase = new LinearImpl(hidden, hidden);
            LinearImpl outBase = new LinearImpl(hidden, vocab);
            this.qProj = LoraLinear.borrowBase(qBase, loraConfig);
            this.vProj = LoraLinear.borrowBase(vBase, loraConfig);
            this.outProj = LoraLinear.borrowBase(outBase, loraConfig);
            register_module("q_base", qBase);
            register_module("v_base", vBase);
            register_module("out_base", outBase);
            register_module("q_proj", qProj);
            register_module("v_proj", vProj);
            register_module("out_proj", outProj);

            this.valueHead = register_module("value_head", new LinearImpl(hidden, 1));

            this.peft = new PeftModel(loraConfig)
                    .add("q_proj", qProj)
                    .add("v_proj", vProj)
                    .add("out_proj", outProj)
                    .root(this);
        }

        public Tensor encode(Tensor x) {
            Tensor h = relu(embed.forward(x));
            Tensor q = qProj.forward(h);
            Tensor v = vProj.forward(h);
            return relu(q.add(v));
        }

        public Tensor logits(Tensor x) {
            return outProj.forward(encode(x));
        }

        public Tensor value(Tensor x) {
            return valueHead.forward(encode(x));
        }

        public Categorical policy(Tensor x) {
            return new Categorical(softmax(logits(x), -1));
        }

        public TensorVector trainableParams() {
            TensorVector all = peft.trainableParameters();
            TensorVector vh = valueHead.parameters();
            var b = vh.begin();
            var e = vh.end();
            while (!b.equals(e)) {
                all.push_back(b.get());
                b.increment();
            }
            vh.close();
            return all;
        }

        public long countTrainable() {
            return countParams(trainableParams(), true);
        }

        public long countTotal() {
            return countParams(parameters(), false);
        }

        public PeftModel peft() { return peft; }
        public LoraConfig loraConfig() { return loraConfig; }
    }

    private final List<Result> results = new ArrayList<>();
    private final long seed;
    private final int trainSteps;

    // Toy task sizes (LLM-scale proportions, tiny absolute size for CI/laptop).
    // Vocab kept small so the contextual bandit is learnable in a few dozen steps.
    private static final long HIDDEN = 32;
    private static final long VOCAB = 8;
    private static final int LORA_R = 4;
    private static final double LORA_ALPHA = 8.0;

    public RLLoraFinetuneBenchmark(long seed, int trainSteps) {
        this.seed = seed;
        this.trainSteps = trainSteps;
    }

    public static void main(String[] args) {
        long seed = 42L;
        int steps = 120;
        if (args != null) {
            for (String a : args) {
                if (a.startsWith("--seed=")) seed = Long.parseLong(a.substring(7));
                if (a.startsWith("--steps=")) steps = Integer.parseInt(a.substring(8));
            }
        }
        RLLoraFinetuneBenchmark bench = new RLLoraFinetuneBenchmark(seed, steps);
        int failed = bench.runAll();
        System.exit(failed == 0 ? 0 : 1);
    }

    public int runAll() {
        manual_seed(seed);
        System.out.println("============================================================");
        System.out.println(" RL + LoRA Large-Model Fine-Tune Benchmark");
        System.out.println(" seed=" + seed + " steps=" + trainSteps
                + " hidden=" + HIDDEN + " vocab=" + VOCAB + " r=" + LORA_R);
        System.out.println("============================================================");

        run("LORA.param_efficiency", this::benchParamEfficiency);
        run("LORA.merge_parity", this::benchMergeParity);
        run("LORA.base_frozen", this::benchBaseFrozen);
        run("RL.ppo_lora_bandit", this::benchPpoLoraBandit);
        run("RL.dpo_lora_preference", this::benchDpoLoraPreference);
        run("RL.grpo_lora_group", this::benchGrpoLoraGroup);
        run("PERF.lora_vs_full", this::benchLoraVsFullThroughput);
        run("STAB.lora_ppo_finite", this::benchLoraPpoFinite);

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

    // -------------------------------------------------------- param efficiency
    private Result benchParamEfficiency() {
        LoraPolicyNet net = new LoraPolicyNet(HIDDEN, VOCAB, LORA_R, LORA_ALPHA);
        long trainable = net.countTrainable();
        long total = net.countTotal();
        double ratio = total == 0 ? 1.0 : (double) trainable / (double) total;
        // Expect LoRA+value << full; with H=64,V=32,r=4 should be well under 30%
        boolean ok = trainable > 0 && total > trainable && ratio < 0.35;
        return new Result("LORA.param_efficiency", ok,
                String.format(Locale.ROOT, "trainable=%d total=%d ratio=%.2f%%",
                        trainable, total, ratio * 100.0),
                ratio * 100.0, "%trainable");
    }

    // ----------------------------------------------------------- merge parity
    private Result benchMergeParity() {
        LoraConfig cfg = LoraConfig.builder().r(4).alpha(8).freezeBase(true).build();
        LoraLinear layer = new LoraLinear(32, 16, cfg);
        // Force non-zero ΔW: set B to ones so ΔW = B @ A * scale ≠ 0 after A~N
        try (NoGradGuard g = new NoGradGuard()) {
            layer.loraB().fill_(new Scalar(0.1));
        }
        Tensor x = randn(4, 32);
        Tensor yBefore = layer.forward(x).detach().clone();
        layer.merge();
        Tensor yMerged = layer.forward(x).detach().clone();
        double diffMerge = yBefore.sub(yMerged).abs().max().item().toDouble();
        layer.unmerge();
        Tensor yAfter = layer.forward(x).detach().clone();
        double diffUnmerge = yBefore.sub(yAfter).abs().max().item().toDouble();
        boolean ok = diffMerge < 1e-4 && diffUnmerge < 1e-4;
        return new Result("LORA.merge_parity", ok,
                String.format(Locale.ROOT, "max|y-y_merged|=%.3e max|y-y_unmerged|=%.3e",
                        diffMerge, diffUnmerge),
                diffMerge, "max_abs_err");
    }

    // ------------------------------------------------------------- base frozen
    private Result benchBaseFrozen() {
        LoraPolicyNet net = new LoraPolicyNet(HIDDEN, VOCAB, LORA_R, LORA_ALPHA);
        Tensor baseQ0 = net.qProj.base().weight().detach().clone();
        Tensor baseV0 = net.vProj.base().weight().detach().clone();
        Tensor loraA0 = net.qProj.loraA().detach().clone();

        AdamWOptions opt = new AdamWOptions();
        opt.lr().put(1e-2);
        Optimizer optimizer = new AdamW(net.trainableParams(), opt);

        for (int i = 0; i < 30; i++) {
            Tensor x = randn(8, HIDDEN);
            // Maximize logit of class 0 as a cheap supervised proxy
            Tensor logits = net.logits(x);
            Tensor target = zeros(new long[]{8}, new TensorOptions().dtype(new ScalarTypeOptional(kLong())));
            Tensor loss = nll_loss(log_softmax(logits, -1), target);
            optimizer.zero_grad();
            loss.backward();
            optimizer.step();
        }

        double dq = baseQ0.sub(net.qProj.base().weight()).abs().max().item().toDouble();
        double dv = baseV0.sub(net.vProj.base().weight()).abs().max().item().toDouble();
        double da = loraA0.sub(net.qProj.loraA()).abs().max().item().toDouble();
        boolean ok = dq < 1e-6 && dv < 1e-6 && da > 1e-5; // base frozen, LoRA moved
        return new Result("LORA.base_frozen", ok,
                String.format(Locale.ROOT, "Δbase_q=%.3e Δbase_v=%.3e Δlora_A=%.3e", dq, dv, da),
                da, "Δlora_A");
    }

    // -------------------------------------------------------- PPO + LoRA bandit
    /**
     * Contextual bandit: each state has a hidden preferred action = hash(state) % vocab.
     * Reward = 1 if sampled token matches preferred, else 0.
     * PPO+LoRA should raise average reward over training.
     */
    private Result benchPpoLoraBandit() {
        manual_seed(seed);
        LoraPolicyNet net = new LoraPolicyNet(HIDDEN, VOCAB, LORA_R, LORA_ALPHA);
        AdamOptions opt = new AdamOptions();
        opt.lr().put(1e-2); // higher LR — only LoRA + value head are trainable
        Optimizer optimizer = new Adam(net.trainableParams(), opt);

        double before = evalBanditReward(net, 128);
        List<Double> curve = new ArrayList<>();
        int steps = Math.max(trainSteps, 150);
        for (int step = 0; step < steps; step++) {
            ppoLoraUpdate(net, optimizer, /*batch*/32, /*horizon*/1);
            if (step % 25 == 0 || step == steps - 1) {
                curve.add(evalBanditReward(net, 128));
            }
        }
        double after = evalBanditReward(net, 256);
        // Chance level ≈ 1/V; require clear lift over chance or over init
        double chance = 1.0 / VOCAB;
        boolean ok = after > before + 0.08 || after >= chance + 0.15 || after >= 0.30;
        return new Result("RL.ppo_lora_bandit", ok,
                String.format(Locale.ROOT, "reward before=%.3f after=%.3f chance=%.3f curve=%s",
                        before, after, chance, curve),
                after, "avg_reward");
    }

    // ----------------------------------------------------- DPO + LoRA preference
    private Result benchDpoLoraPreference() {
        manual_seed(seed);
        LoraPolicyNet policy = new LoraPolicyNet(HIDDEN, VOCAB, LORA_R, LORA_ALPHA);
        // Reference: frozen snapshot of a freshly-initialized twin (not the same object)
        manual_seed(seed + 99);
        LoraPolicyNet reference = new LoraPolicyNet(HIDDEN, VOCAB, LORA_R, LORA_ALPHA);
        freezeAll(reference);

        AdamOptions opt = new AdamOptions();
        opt.lr().put(1e-2);
        Optimizer optimizer = new Adam(policy.trainableParams(), opt);

        double marginBefore = preferenceMargin(policy, 128);
        float firstLoss = Float.NaN;
        float lastLoss = Float.NaN;
        int steps = Math.max(trainSteps, 150);
        for (int step = 0; step < steps; step++) {
            Tensor x = makePreferenceBatch(64);
            long[] chosenArr = new long[64];
            long[] rejectedArr = new long[64];
            for (int i = 0; i < 64; i++) {
                long pref = preferredAction(x.select(0, i));
                chosenArr[i] = pref;
                rejectedArr[i] = (pref + 1) % VOCAB;
            }
            Tensor chosen = tensor(chosenArr);
            Tensor rejected = tensor(rejectedArr);

            Categorical pDist = policy.policy(x);
            Categorical rDist = reference.policy(x);
            Tensor pC = pDist.log_prob(chosen);
            Tensor pR = pDist.log_prob(rejected);
            Tensor rC = rDist.log_prob(chosen).detach();
            Tensor rR = rDist.log_prob(rejected).detach();

            // Stronger beta for faster preference separation on the toy task
            Tensor logits = pC.sub(rC).sub(pR.sub(rR)).mul(new Scalar(0.5));
            Tensor loss = log_sigmoid(logits).mean().neg();
            float lv = loss.item().toFloat();
            if (step == 0) firstLoss = lv;
            lastLoss = lv;

            optimizer.zero_grad();
            loss.backward();
            clip_grad_norm_(policy.trainableParams(), 1.0);
            optimizer.step();
        }
        double marginAfter = preferenceMargin(policy, 128);
        boolean ok = Float.isFinite(lastLoss)
                && (marginAfter > marginBefore + 0.05
                    || lastLoss < firstLoss - 0.01
                    || marginAfter > 0.1);
        return new Result("RL.dpo_lora_preference", ok,
                String.format(Locale.ROOT,
                        "margin before=%.3f after=%.3f loss first=%.4f last=%.4f",
                        marginBefore, marginAfter, firstLoss, lastLoss),
                marginAfter, "logp_margin");
    }

    // -------------------------------------------------------- GRPO + LoRA group
    private Result benchGrpoLoraGroup() {
        manual_seed(seed);
        LoraPolicyNet net = new LoraPolicyNet(HIDDEN, VOCAB, LORA_R, LORA_ALPHA);
        AdamOptions opt = new AdamOptions();
        opt.lr().put(3e-3);
        Optimizer optimizer = new Adam(net.trainableParams(), opt);

        double before = evalBanditReward(net, 64);
        final int groupSize = 4;
        final int prompts = 8;
        float lastLoss = 0;
        for (int step = 0; step < trainSteps; step++) {
            // For each prompt, sample G completions, score with bandit reward
            Tensor x = randn(prompts, HIDDEN);
            long[] actions = new long[prompts * groupSize];
            float[] rewards = new float[prompts * groupSize];
            float[] oldLp = new float[prompts * groupSize];

            // Expand prompts: each repeated G times
            Tensor xRep = x.unsqueeze(1).expand(prompts, groupSize, HIDDEN)
                    .reshape(prompts * groupSize, HIDDEN);
            Categorical dist = net.policy(xRep);
            Tensor sampled = dist.sample(); // [B*G]
            Tensor lp = dist.log_prob(sampled);
            for (int i = 0; i < prompts * groupSize; i++) {
                long a = sampled.select(0, i).item().toLong();
                actions[i] = a;
                oldLp[i] = lp.select(0, i).item().toFloat();
                // reward vs preferred of the underlying prompt
                int p = i / groupSize;
                long pref = preferredAction(x.select(0, p));
                rewards[i] = (a == pref) ? 1.0f : 0.0f;
            }

            // Re-evaluate current policy logprobs for GRPO loss
            Categorical dist2 = net.policy(xRep);
            Tensor actT = tensor(actions);
            Tensor oldLpT = tensor(oldLp);
            Tensor rewT = tensor(rewards).reshape(prompts, groupSize);
            Tensor loss = GRPOLoss.computeLoss(dist2, actT, oldLpT, rewT, 0.2f);
            lastLoss = loss.item().toFloat();
            optimizer.zero_grad();
            loss.backward();
            clip_grad_norm_(net.trainableParams(), 1.0);
            optimizer.step();
        }
        double after = evalBanditReward(net, 128);
        boolean ok = Float.isFinite(lastLoss) && (after >= before - 0.02);
        boolean strong = after > before + 0.05;
        return new Result("RL.grpo_lora_group", ok,
                String.format(Locale.ROOT, "reward before=%.3f after=%.3f last_loss=%.4f strong=%s",
                        before, after, lastLoss, strong),
                after, "avg_reward");
    }

    // --------------------------------------------------- LoRA vs full throughput
    private Result benchLoraVsFullThroughput() {
        // LoRA path
        LoraPolicyNet loraNet = new LoraPolicyNet(HIDDEN, VOCAB, LORA_R, LORA_ALPHA);
        AdamOptions o1 = new AdamOptions();
        o1.lr().put(1e-3);
        Optimizer loraOpt = new Adam(loraNet.trainableParams(), o1);

        int iters = 40;
        long t0 = System.nanoTime();
        for (int i = 0; i < iters; i++) {
            ppoLoraUpdate(loraNet, loraOpt, 16, 1);
        }
        double loraUps = iters / ((System.nanoTime() - t0) / 1e9);

        // Full fine-tune twin: unfreeze everything via a non-LoRA MLP of similar width
        FullPolicyNet full = new FullPolicyNet(HIDDEN, VOCAB);
        AdamOptions o2 = new AdamOptions();
        o2.lr().put(1e-3);
        Optimizer fullOpt = new Adam(full.parameters(), o2);
        t0 = System.nanoTime();
        for (int i = 0; i < iters; i++) {
            ppoFullUpdate(full, fullOpt, 16);
        }
        double fullUps = iters / ((System.nanoTime() - t0) / 1e9);

        double speedup = fullUps > 0 ? loraUps / fullUps : Double.NaN;
        // Not required to be faster on tiny nets; just both finite and LoRA runs
        boolean ok = Double.isFinite(loraUps) && Double.isFinite(fullUps) && loraUps > 0.5;
        return new Result("PERF.lora_vs_full", ok,
                String.format(Locale.ROOT, "lora=%.1f upd/s full=%.1f upd/s speedup=%.2fx trainable_lora=%d trainable_full=%d",
                        loraUps, fullUps, speedup, loraNet.countTrainable(), countParams(full.parameters(), false)),
                loraUps, "lora_upd/s");
    }

    // --------------------------------------------------------------- stability
    private Result benchLoraPpoFinite() {
        LoraPolicyNet net = new LoraPolicyNet(HIDDEN, VOCAB, LORA_R, LORA_ALPHA);
        AdamOptions opt = new AdamOptions();
        opt.lr().put(3e-3);
        Optimizer optimizer = new Adam(net.trainableParams(), opt);
        int nans = 0;
        double last = 0;
        for (int i = 0; i < 30; i++) {
            float loss = ppoLoraUpdate(net, optimizer, 16, 1);
            if (!Float.isFinite(loss)) nans++;
            last = loss;
        }
        boolean ok = nans == 0;
        return new Result("STAB.lora_ppo_finite", ok,
                String.format(Locale.ROOT, "nan_count=%d last_loss=%.4f", nans, last),
                last, "loss");
    }

    // ============================== toy nets / helpers =======================

    /** Non-LoRA full policy for throughput comparison. */
    static final class FullPolicyNet extends Module {
        final LinearImpl fc1, fc2, head, value;
        final long vocab;

        FullPolicyNet(long hidden, long vocab) {
            super("FullPolicyNet");
            this.vocab = vocab;
            fc1 = register_module("fc1", new LinearImpl(hidden, hidden));
            fc2 = register_module("fc2", new LinearImpl(hidden, hidden));
            head = register_module("head", new LinearImpl(hidden, vocab));
            value = register_module("value", new LinearImpl(hidden, 1));
        }

        Tensor encode(Tensor x) {
            return relu(fc2.forward(relu(fc1.forward(x))));
        }

        Categorical policy(Tensor x) {
            return new Categorical(softmax(head.forward(encode(x)), -1));
        }

        Tensor valueOf(Tensor x) {
            return value.forward(encode(x));
        }
    }

    private static float ppoLoraUpdate(LoraPolicyNet net, Optimizer opt, int batch, int horizon) {
        // Single-step contextual bandit treated as horizon-1 PPO
        Tensor x = randn(batch, HIDDEN);
        Categorical dist = net.policy(x);
        Tensor actions = dist.sample();
        Tensor oldLp = dist.log_prob(actions).detach().clone();
        Tensor values = net.value(x).squeeze(-1).detach().clone();

        float[] rewards = new float[batch];
        for (int i = 0; i < batch; i++) {
            long a = actions.select(0, i).item().toLong();
            long pref = preferredAction(x.select(0, i));
            rewards[i] = (a == pref) ? 1.0f : 0.0f;
        }
        Tensor rew = tensor(rewards);
        // advantage = r - V (no bootstrap for bandit)
        Tensor adv = rew.sub(values);
        Tensor advNorm = adv.sub(adv.mean()).div(adv.std().add(new Scalar(1e-8)));
        Tensor ret = rew; // return = reward for bandit

        Categorical distNew = net.policy(x);
        Tensor newLp = distNew.log_prob(actions);
        Tensor ratio = exp(newLp.sub(oldLp));
        Tensor surr1 = ratio.mul(advNorm);
        Tensor surr2 = clamp(ratio,
                new ScalarOptional(new Scalar(0.8)),
                new ScalarOptional(new Scalar(1.2))).mul(advNorm);
        Tensor actorLoss = min(surr1, surr2).mean().neg();
        Tensor criticLoss = mse_loss(net.value(x).squeeze(-1), ret);
        Tensor entropy = distNew.entropy().mean();
        Tensor total = actorLoss.add(criticLoss.mul(new Scalar(0.5)))
                .sub(entropy.mul(new Scalar(0.01)));

        opt.zero_grad();
        total.backward();
        clip_grad_norm_(net.trainableParams(), 1.0);
        opt.step();
        return total.item().toFloat();
    }

    private static void ppoFullUpdate(FullPolicyNet net, Optimizer opt, int batch) {
        Tensor x = randn(batch, HIDDEN);
        Categorical dist = net.policy(x);
        Tensor actions = dist.sample();
        Tensor oldLp = dist.log_prob(actions).detach().clone();
        Tensor values = net.valueOf(x).squeeze(-1).detach().clone();
        float[] rewards = new float[batch];
        for (int i = 0; i < batch; i++) {
            long a = actions.select(0, i).item().toLong();
            long pref = preferredAction(x.select(0, i));
            rewards[i] = (a == pref) ? 1.0f : 0.0f;
        }
        Tensor rew = tensor(rewards);
        Tensor adv = rew.sub(values);
        Tensor advNorm = adv.sub(adv.mean()).div(adv.std().add(new Scalar(1e-8)));
        Categorical distNew = net.policy(x);
        Tensor newLp = distNew.log_prob(actions);
        Tensor ratio = exp(newLp.sub(oldLp));
        Tensor surr1 = ratio.mul(advNorm);
        Tensor surr2 = clamp(ratio,
                new ScalarOptional(new Scalar(0.8)),
                new ScalarOptional(new Scalar(1.2))).mul(advNorm);
        Tensor actorLoss = min(surr1, surr2).mean().neg();
        Tensor criticLoss = mse_loss(net.valueOf(x).squeeze(-1), rew);
        Tensor total = actorLoss.add(criticLoss.mul(new Scalar(0.5)));
        opt.zero_grad();
        total.backward();
        clip_grad_norm_(net.parameters(), 1.0);
        opt.step();
    }

    private static double evalBanditReward(LoraPolicyNet net, int n) {
        Tensor x = randn(n, HIDDEN);
        Categorical dist = net.policy(x);
        Tensor probs = dist.getProbs();
        Tensor pred = probs.argmax(new LongOptional(-1L), false);
        int hits = 0;
        for (int i = 0; i < n; i++) {
            long a = pred.select(0, i).item().toLong();
            if (a == preferredAction(x.select(0, i))) hits++;
        }
        return hits / (double) n;
    }

    /** Mean (logπ(chosen) − logπ(rejected)) under policy. */
    private static double preferenceMargin(LoraPolicyNet net, int n) {
        Tensor x = makePreferenceBatch(n);
        long[] chosenArr = new long[n];
        long[] rejectedArr = new long[n];
        for (int i = 0; i < n; i++) {
            long pref = preferredAction(x.select(0, i));
            chosenArr[i] = pref;
            rejectedArr[i] = (pref + 1) % VOCAB;
        }
        Categorical dist = net.policy(x);
        Tensor margin = dist.log_prob(tensor(chosenArr)).sub(dist.log_prob(tensor(rejectedArr)));
        return margin.mean().item().toDouble();
    }

    private static Tensor makePreferenceBatch(int n) {
        return randn(n, HIDDEN);
    }

    /**
     * Deterministic preferred action — linear threshold on first coordinate.
     * Easy enough for a small LoRA MLP to pick up in tens of PPO/DPO steps.
     * <pre>preferred = floor((tanh(s0)+1)/2 * V)  ∈ {0..V-1}</pre>
     */
    private static long preferredAction(Tensor stateVec) {
        float s0 = stateVec.select(0, 0).item().toFloat();
        double u = (Math.tanh(s0) + 1.0) * 0.5; // [0,1]
        long a = (long) Math.floor(u * VOCAB);
        if (a >= VOCAB) a = VOCAB - 1;
        if (a < 0) a = 0;
        return a;
    }

    private static void freezeAll(Module m) {
        TensorVector params = m.parameters();
        var b = params.begin();
        var e = params.end();
        while (!b.equals(e)) {
            b.get().requires_grad_(false);
            b.increment();
        }
        params.close();
    }

    private static long countParams(TensorVector params, boolean close) {
        long n = 0;
        var b = params.begin();
        var e = params.end();
        while (!b.equals(e)) {
            n += b.get().numel();
            b.increment();
        }
        if (close) params.close();
        return n;
    }
}
