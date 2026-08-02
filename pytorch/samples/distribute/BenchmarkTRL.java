package distribute;
import org.bytedeco.pytorch.nn.modules.*;
import org.bytedeco.pytorch.optim.*;
import org.bytedeco.pytorch.optim.options.*;

import org.bytedeco.pytorch.NoGradGuard;
import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorVector;
import org.bytedeco.pytorch.llm.peft.LoraConfig;
import org.bytedeco.pytorch.llm.trl.BaseTrainer;
import org.bytedeco.pytorch.llm.trl.DPOTrainer;
import org.bytedeco.pytorch.llm.trl.GRPOTrainer;
import org.bytedeco.pytorch.llm.trl.LlmForward;
import org.bytedeco.pytorch.llm.trl.LogProbUtils;
import org.bytedeco.pytorch.llm.trl.ORPOTrainer;
import org.bytedeco.pytorch.llm.trl.PPOTrainer;
import org.bytedeco.pytorch.llm.trl.RewardTrainer;
import org.bytedeco.pytorch.llm.trl.SFTTrainer;
import org.bytedeco.pytorch.llm.trl.TrainerCallback;
import org.bytedeco.pytorch.llm.trl.config.DPOConfig;
import org.bytedeco.pytorch.llm.trl.config.GRPOConfig;
import org.bytedeco.pytorch.llm.trl.config.ORPOConfig;
import org.bytedeco.pytorch.llm.trl.config.PPOConfig;
import org.bytedeco.pytorch.llm.trl.config.RewardConfig;
import org.bytedeco.pytorch.llm.trl.config.SFTConfig;
import org.bytedeco.pytorch.llm.trl.config.TrainerConfig;
import org.bytedeco.pytorch.llm.trl.loss.DPOLoss;
import org.bytedeco.pytorch.llm.trl.loss.GRPOLoss;
import org.bytedeco.pytorch.llm.trl.loss.ORPOLoss;
import org.bytedeco.pytorch.llm.trl.loss.PPOLoss;
import org.bytedeco.pytorch.llm.trl.loss.RewardModelLoss;
import org.bytedeco.pytorch.llm.trl.loss.SFTLoss;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.modules.LinearImpl;
import org.bytedeco.pytorch.optim.Adam;
import org.bytedeco.pytorch.optim.options.AdamOptions;
import org.bytedeco.pytorch.llm.transformers.CausalLM;
import org.bytedeco.pytorch.llm.transformers.PretrainedConfig;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;

import static org.bytedeco.pytorch.global.torch.manual_seed;
import static org.bytedeco.pytorch.global.torch.randn;
import static org.bytedeco.pytorch.global.torch.tensor;
import static org.bytedeco.pytorch.global.torch.zeros;
import static org.bytedeco.pytorch.global.torch.ones;

/**
 * Multi-dimensional benchmark / usability suite for {@code org.bytedeco.pytorch.llm.trl}.
 *
 * <p>Verifies that the TRL-style trainers are usable for LLM fine-tuning by exercising:
 * <ul>
 *   <li>Config builders &amp; hyper-parameter surface (all trainer types)</li>
 *   <li>Unit losses in isolation (SFT / DPO / ORPO / PPO / GRPO / Reward)</li>
 *   <li>LogProbUtils causal shift + mask contracts</li>
 *   <li>End-to-end micro training loops on {@link CausalLM} tinyGpt2</li>
 *   <li>Gradient accumulation, grad clip, callbacks, train/eval mode</li>
 *   <li>DPO loss-type matrix (sigmoid / hinge / ipo) ± reference-free</li>
 *   <li>PPO precomputed + online PolicyValueForward + GAE</li>
 *   <li>GRPO group-normalize / clipped / with KL beta</li>
 *   <li>ORPO reference-free preference path</li>
 *   <li>Reward model Bradley-Terry ± margin ± center</li>
 *   <li>Config matrix: lr × accum × maxSteps × beta × clip</li>
 *   <li>Optional LoRA-on-CausalLM + SFT (PEFT path)</li>
 *   <li>Throughput / step latency smoke numbers</li>
 * </ul>
 *
 * <p>Run:
 * <pre>
 *   javac -cp target/classes:$(cat target/cp.txt) -d target/samples-compile samples/BenchmarkTRL.java
 *   java  -cp target/samples-compile:target/classes:$(cat target/cp.txt) distribute.BenchmarkTRL
 * </pre>
 */
public class BenchmarkTRL {

    static int passed = 0, failed = 0, skipped = 0;
    static final List<String> failures = new ArrayList<>();
    static final List<String> timings = new ArrayList<>();

    // Tiny model geometry (matches PretrainedConfig.tinyGpt2 defaults)
    static final int VOCAB = 256;
    static final int SEQ = 8;
    static final int BATCH = 2;

    static void section(String t) {
        System.out.println("\n=== " + t + " ===");
    }

    static void check(String name, boolean ok) {
        if (ok) {
            passed++;
            System.out.println("  PASS  " + name);
        } else {
            failed++;
            failures.add(name);
            System.out.println("  FAIL  " + name);
        }
    }

    static void checkFinite(String name, double v) {
        boolean ok = !Double.isNaN(v) && !Double.isInfinite(v);
        check(name + " finite=" + fmt(v), ok);
    }

    static void skip(String name, String reason) {
        skipped++;
        System.out.println("  SKIP  " + name + " (" + reason + ")");
    }

    static String fmt(double v) {
        return String.format(Locale.US, "%.6g", v);
    }

    static long nowNs() {
        return System.nanoTime();
    }

    static void recordTiming(String name, long ns, int steps) {
        double ms = ns / 1e6;
        double per = steps > 0 ? ms / steps : ms;
        String line = String.format(Locale.US, "%-40s  total=%.2f ms  steps=%d  per_step=%.3f ms",
                name, ms, steps, per);
        timings.add(line);
        System.out.println("  TIME  " + line);
    }

    // ------------------------------------------------------------------ helpers

    static Tensor longIds(int B, int T, int vocab, long seed) {
        long[] flat = new long[B * T];
        // deterministic pseudo-random in [0, vocab)
        long s = seed;
        for (int i = 0; i < flat.length; i++) {
            s = s * 6364136223846793005L + 1L;
            flat[i] = Math.floorMod(s, vocab);
        }
        return tensor(flat).reshape(B, T);
    }

    static Tensor floatVec(double... vals) {
        return floatVec(false, vals);
    }

    /**
     * @param requiresGrad when true, marks the leaf so {@code trainingStep} can
     *                     {@code backward} through precomputed log-prob / reward paths
     *                     (simulates logps that came from a differentiable forward).
     */
    static Tensor floatVec(boolean requiresGrad, double... vals) {
        float[] f = new float[vals.length];
        for (int i = 0; i < vals.length; i++) f[i] = (float) vals[i];
        // 1-D via 2-D then squeeze to keep API portable
        float[][] row = new float[][]{f};
        Tensor t = tensor(row).squeeze(0);
        if (requiresGrad) {
            t = t.requires_grad_(true);
        }
        return t;
    }

    static Tensor onesMask(int B, int T) {
        return ones(new long[]{B, T});
    }

    static CausalLM tinyModel() {
        return CausalLM.fromConfig(PretrainedConfig.tinyGpt2());
    }

    static CausalLM tinyModelCloneWeights(CausalLM src) {
        // Fresh model with same config (independent params) — used as frozen reference.
        return CausalLM.fromConfig(src.config());
    }

    static Adam adam(Module m, double lr) {
        return new Adam(m.parameters(), new AdamOptions(lr));
    }

    static LlmForward asForward(CausalLM m) {
        return (ids, mask) -> m.forward(ids);
    }

    static Map<String, Tensor> sftBatch(int B, int T, int vocab, long seed) {
        Map<String, Tensor> b = new LinkedHashMap<>();
        Tensor ids = longIds(B, T, vocab, seed);
        b.put("input_ids", ids);
        b.put("labels", ids);
        b.put("attention_mask", onesMask(B, T));
        return b;
    }

    static Map<String, Tensor> prefBatch(int B, int T, int vocab, long seed) {
        Map<String, Tensor> b = new LinkedHashMap<>();
        b.put("chosen_input_ids", longIds(B, T, vocab, seed));
        b.put("rejected_input_ids", longIds(B, T, vocab, seed + 17));
        b.put("chosen_attention_mask", onesMask(B, T));
        b.put("rejected_attention_mask", onesMask(B, T));
        return b;
    }

    static Map<String, Tensor> prefLogpsBatch(int B) {
        Map<String, Tensor> b = new LinkedHashMap<>();
        // chosen slightly higher logp than rejected → DPO prefers them
        double[] c = new double[B];
        double[] r = new double[B];
        double[] rc = new double[B];
        double[] rr = new double[B];
        for (int i = 0; i < B; i++) {
            c[i] = -2.0 - 0.1 * i;
            r[i] = -4.0 - 0.1 * i;
            rc[i] = -3.0;
            rr[i] = -3.0;
        }
        // Policy logps must require grad so BaseTrainer.trainingStep can backward.
        b.put("policy_chosen_logps", floatVec(true, c));
        b.put("policy_rejected_logps", floatVec(true, r));
        b.put("ref_chosen_logps", floatVec(rc));
        b.put("ref_rejected_logps", floatVec(rr));
        return b;
    }

    static Map<String, Tensor> ppoPrecomputedBatch(int n) {
        Map<String, Tensor> b = new LinkedHashMap<>();
        double[] oldLp = new double[n];
        double[] newLp = new double[n];
        double[] adv = new double[n];
        double[] ret = new double[n];
        double[] oldV = new double[n];
        double[] val = new double[n];
        double[] ent = new double[n];
        for (int i = 0; i < n; i++) {
            oldLp[i] = -1.5 - 0.05 * i;
            newLp[i] = -1.4 - 0.04 * i; // slight policy improvement
            adv[i] = (i % 2 == 0) ? 0.5 : -0.3;
            ret[i] = 1.0 + 0.1 * i;
            oldV[i] = 0.8 + 0.05 * i;
            val[i] = 0.9 + 0.05 * i;
            ent[i] = 0.2;
        }
        // Differentiable side: new_logprobs + values (+ entropy)
        b.put("old_logprobs", floatVec(oldLp));
        b.put("new_logprobs", floatVec(true, newLp));
        b.put("advantages", floatVec(adv));
        b.put("returns", floatVec(ret));
        b.put("old_values", floatVec(oldV));
        b.put("values", floatVec(true, val));
        b.put("entropy", floatVec(true, ent));
        return b;
    }

    static Map<String, Tensor> grpoPrecomputedBatch(int groups, int G) {
        int n = groups * G;
        Map<String, Tensor> b = new LinkedHashMap<>();
        double[] rewards = new double[n];
        double[] logps = new double[n];
        double[] oldLp = new double[n];
        double[] refLp = new double[n];
        for (int g = 0; g < groups; g++) {
            for (int i = 0; i < G; i++) {
                int idx = g * G + i;
                rewards[idx] = i;                 // within-group ranking
                logps[idx] = -2.0 - 0.1 * i;
                oldLp[idx] = -2.1 - 0.1 * i;
                refLp[idx] = -2.5;
            }
        }
        b.put("rewards", floatVec(rewards));
        b.put("logprobs", floatVec(true, logps)); // policy logps need grad
        b.put("old_logprobs", floatVec(oldLp));
        b.put("ref_logprobs", floatVec(refLp));
        return b;
    }

    static boolean requiresGradSomewhere(Module m) {
        TensorVector pv = m.parameters();
        for (long i = 0, n = pv.size(); i < n; i++) {
            Tensor p = pv.get(i);
            if (p != null && !p.isNull() && p.defined() && p.requires_grad()) {
                return true;
            }
        }
        return false;
    }

    // ================================================================== D1 Configs

    static void d1Configs() {
        section("D1 Config builders (all trainer types)");

        TrainerConfig tc = TrainerConfig.builder()
                .learningRate(2e-5)
                .maxSteps(100)
                .loggingSteps(5)
                .gradientAccumulationSteps(4)
                .maxGradNorm(1.0)
                .fp16(false)
                .seed(7L)
                .build();
        check("TrainerConfig lr", Math.abs(tc.learningRate() - 2e-5) < 1e-12);
        check("TrainerConfig maxSteps", tc.maxSteps() == 100);
        check("TrainerConfig accum", tc.gradientAccumulationSteps() == 4);
        check("TrainerConfig maxGradNorm", Math.abs(tc.maxGradNorm() - 1.0) < 1e-12);
        check("TrainerConfig seed", tc.seed() == 7L);
        check("TrainerConfig fp16 default false", !tc.fp16());

        SFTConfig sft = SFTConfig.builder()
                .learningRate(1e-4)
                .maxSteps(50)
                .maxSeqLength(512)
                .ignoreIndex(-100L)
                .packing(true)
                .build();
        check("SFTConfig maxSeqLength", sft.maxSeqLength() == 512);
        check("SFTConfig ignoreIndex", sft.ignoreIndex() == -100L);
        check("SFTConfig packing", sft.packing());
        check("SFTConfig inherits lr", Math.abs(sft.learningRate() - 1e-4) < 1e-12);

        DPOConfig dpo = DPOConfig.builder()
                .beta(0.2)
                .lossType("ipo")
                .referenceFree(true)
                .labelSmoothing(0.05)
                .learningRate(5e-6)
                .build();
        check("DPOConfig beta", Math.abs(dpo.beta() - 0.2) < 1e-12);
        check("DPOConfig lossType=ipo", "ipo".equals(dpo.lossType()));
        check("DPOConfig referenceFree", dpo.referenceFree());
        check("DPOConfig labelSmoothing", Math.abs(dpo.labelSmoothing() - 0.05) < 1e-12);

        ORPOConfig orpo = ORPOConfig.builder()
                .beta(0.15)
                .lengthNormalize(true)
                .maxSteps(20)
                .build();
        check("ORPOConfig beta", Math.abs(orpo.beta() - 0.15) < 1e-12);
        check("ORPOConfig lengthNormalize", orpo.lengthNormalize());

        PPOConfig ppo = PPOConfig.builder()
                .clipRange(0.15)
                .clipRangeVf(0.1)
                .vfCoef(0.25)
                .entCoef(0.02)
                .gamma(0.98)
                .gaeLambda(0.9)
                .ppoEpochs(2)
                .miniBatchSize(8)
                .build();
        check("PPOConfig clipRange", Math.abs(ppo.clipRange() - 0.15) < 1e-12);
        check("PPOConfig vfCoef", Math.abs(ppo.vfCoef() - 0.25) < 1e-12);
        check("PPOConfig gamma", Math.abs(ppo.gamma() - 0.98) < 1e-12);
        check("PPOConfig gaeLambda", Math.abs(ppo.gaeLambda() - 0.9) < 1e-12);
        check("PPOConfig ppoEpochs", ppo.ppoEpochs() == 2);

        GRPOConfig grpo = GRPOConfig.builder()
                .numGenerations(8)
                .beta(0.05)
                .clipRange(0.25)
                .temperature(0.7)
                .maxCompletionLength(128)
                .build();
        check("GRPOConfig numGenerations", grpo.numGenerations() == 8);
        check("GRPOConfig beta", Math.abs(grpo.beta() - 0.05) < 1e-12);
        check("GRPOConfig temperature", Math.abs(grpo.temperature() - 0.7) < 1e-12);
        check("GRPOConfig maxCompletionLength", grpo.maxCompletionLength() == 128);

        RewardConfig rew = RewardConfig.builder()
                .margin(0.5)
                .centerRewards(true)
                .learningRate(3e-5)
                .build();
        check("RewardConfig margin", Math.abs(rew.margin() - 0.5) < 1e-12);
        check("RewardConfig centerRewards", rew.centerRewards());
    }

    // ================================================================== D2 Unit losses

    static void d2UnitLosses() {
        section("D2 Unit losses (no trainer loop)");
        manual_seed(0);

        // SFT
        Tensor logits = randn(new long[]{BATCH, SEQ, VOCAB});
        Tensor labels = longIds(BATCH, SEQ, VOCAB, 1L);
        Tensor sftLoss = SFTLoss.compute(logits, labels);
        check("SFTLoss defined", sftLoss != null && sftLoss.defined());
        checkFinite("SFTLoss", sftLoss.item_double());

        // DPO loss types
        Tensor pC = floatVec(-1.0, -1.2);
        Tensor pR = floatVec(-2.0, -2.5);
        Tensor rC = floatVec(-1.5, -1.5);
        Tensor rR = floatVec(-1.5, -1.5);
        for (String type : new String[]{"sigmoid", "hinge", "ipo"}) {
            Tensor l = DPOLoss.compute(pC, pR, rC, rR, 0.1, type);
            checkFinite("DPOLoss/" + type, l.item_double());
        }
        // reference-free: zeros ref → still finite
        Tensor z = zeros(new long[]{2});
        checkFinite("DPOLoss/ref-free", DPOLoss.compute(pC, pR, z, z, 0.1).item_double());

        // ORPO
        checkFinite("ORPOLoss", ORPOLoss.compute(pC, pR, 0.1).item_double());

        // Reward BT
        Tensor chosenR = floatVec(1.0, 0.5);
        Tensor rejectedR = floatVec(-0.5, 0.0);
        checkFinite("RewardModelLoss", RewardModelLoss.compute(chosenR, rejectedR).item_double());
        // chosen > rejected → loss should be < ln(2) ≈ 0.693
        double bt = RewardModelLoss.compute(chosenR, rejectedR).item_double();
        check("RewardModelLoss < ln2 when chosen>rejected", bt < 0.693);

        // PPO
        Tensor newLp = floatVec(-1.0, -1.1, -0.9, -1.2);
        Tensor oldLp = floatVec(-1.05, -1.0, -1.0, -1.1);
        Tensor adv = floatVec(0.5, -0.2, 0.3, -0.1);
        Tensor values = floatVec(0.8, 0.7, 0.9, 0.6);
        Tensor returns = floatVec(1.0, 0.5, 1.1, 0.4);
        Tensor oldV = floatVec(0.75, 0.65, 0.85, 0.55);
        Tensor ent = floatVec(0.2, 0.2, 0.2, 0.2);
        PPOLoss.Result pr = PPOLoss.compute(newLp, oldLp, adv, values, returns, oldV, ent,
                0.2, 0.2, 0.5, 0.01);
        checkFinite("PPOLoss.total", pr.total.item_double());
        checkFinite("PPOLoss.policy", pr.policy.item_double());
        checkFinite("PPOLoss.value", pr.value.item_double());
        checkFinite("PPOLoss.entropy", pr.entropy.item_double());

        // GRPO
        Tensor rewards = floatVec(1.0, 2.0, 0.5, 3.0); // 2 groups of 2
        Tensor logps = floatVec(-1.0, -1.5, -2.0, -0.5);
        Tensor grpoL = GRPOLoss.compute(logps, rewards, 2);
        checkFinite("GRPOLoss", grpoL.item_double());
        Tensor advG = GRPOLoss.groupNormalize(rewards, 2);
        check("GRPO groupNormalize numel", advG.numel() == 4);
        // within each group mean≈0
        float[] a = new float[4];
        // read via item on selects
        double g0 = advG.select(0, 0).item_double() + advG.select(0, 1).item_double();
        double g1 = advG.select(0, 2).item_double() + advG.select(0, 3).item_double();
        check("GRPO group mean≈0 g0", Math.abs(g0) < 1e-5);
        check("GRPO group mean≈0 g1", Math.abs(g1) < 1e-5);

        Tensor clipped = GRPOLoss.computeClipped(logps, floatVec(-1.1, -1.4, -1.9, -0.6), rewards, 2, 0.2);
        checkFinite("GRPOLoss.clipped", clipped.item_double());
    }

    // ================================================================== D3 LogProbUtils

    static void d3LogProbUtils() {
        section("D3 LogProbUtils causal shift + mask");
        manual_seed(1);
        CausalLM model = tinyModel();
        model.eval();
        Tensor ids = longIds(BATCH, SEQ, model.vocabSize(), 42L);
        Tensor logits;
        try (NoGradGuard g = new NoGradGuard()) {
            logits = model.forward(ids);
        }
        check("logits shape [B,T,V]", logits.dim() == 3
                && logits.size(0) == BATCH
                && logits.size(1) == SEQ
                && logits.size(2) == model.vocabSize());

        Tensor lp = LogProbUtils.sequenceLogProbs(logits, ids, null);
        check("sequenceLogProbs [B]", lp.dim() == 1 && lp.size(0) == BATCH);
        checkFinite("sequenceLogProbs[0]", lp.select(0, 0).item_double());
        // logprobs are ≤ 0
        check("logprobs ≤ 0", lp.select(0, 0).item_double() <= 1e-6);

        Tensor meanLp = LogProbUtils.sequenceMeanLogProbs(logits, ids, null);
        check("sequenceMeanLogProbs [B]", meanLp.dim() == 1 && meanLp.size(0) == BATCH);
        checkFinite("sequenceMeanLogProbs", meanLp.select(0, 0).item_double());

        // Mask: zero out first half of tokens → sum |masked| ≤ |unmasked|
        Tensor mask = onesMask(BATCH, SEQ);
        // zero positions 0..T/2 (prompt region); after shift, label positions 1.. use mask[1:]
        for (int t = 0; t < SEQ / 2; t++) {
            mask.select(0, 0).select(0, t).fill_(new Scalar(0.0));
            mask.select(0, 1).select(0, t).fill_(new Scalar(0.0));
        }
        Tensor lpMasked = LogProbUtils.sequenceLogProbs(logits, ids, mask);
        // masked sum magnitude should be smaller (fewer tokens) in absolute value typically
        checkFinite("masked logprobs", lpMasked.select(0, 0).item_double());
        check("masked |lp| ≤ unmasked |lp| (approx)",
                Math.abs(lpMasked.select(0, 0).item_double()) <= Math.abs(lp.select(0, 0).item_double()) + 1e-4);
    }

    // ================================================================== D4 SFT trainer

    static void d4SftTrainer() {
        section("D4 SFTTrainer end-to-end on CausalLM");
        manual_seed(2);
        CausalLM model = tinyModel();
        check("model has trainable params", requiresGradSomewhere(model));

        SFTConfig cfg = SFTConfig.builder()
                .learningRate(1e-3)
                .maxSteps(4)
                .loggingSteps(1)
                .gradientAccumulationSteps(1)
                .maxGradNorm(1.0)
                .maxSeqLength(SEQ)
                .ignoreIndex(0L) // disable ignore path; labels are valid ids
                .build();
        Adam opt = adam(model, cfg.learningRate());

        AtomicInteger logs = new AtomicInteger();
        AtomicInteger steps = new AtomicInteger();
        AtomicReference<Double> lastLoss = new AtomicReference<>(Double.NaN);
        TrainerCallback cb = new TrainerCallback() {
            @Override public void onTrainBegin(BaseTrainer t) { /* ok */ }
            @Override public void onStepEnd(BaseTrainer t, int step, Map<String, Double> m) {
                steps.incrementAndGet();
                if (m.containsKey("loss")) lastLoss.set(m.get("loss"));
            }
            @Override public void onLog(BaseTrainer t, int step, Map<String, Double> m) {
                logs.incrementAndGet();
            }
            @Override public void onTrainEnd(BaseTrainer t) { /* ok */ }
        };

        try (SFTTrainer trainer = new SFTTrainer(model, asForward(model), opt, cfg)) {
            trainer.addCallback(cb);
            check("SFTTrainer.model()", trainer.model() == model);
            check("SFTTrainer.config maxSteps", trainer.config().maxSteps() == 4);

            long t0 = nowNs();
            List<Double> losses = new ArrayList<>();
            for (int i = 0; i < 4; i++) {
                double loss = trainer.trainingStep(sftBatch(BATCH, SEQ, model.vocabSize(), 100L + i));
                losses.add(loss);
                checkFinite("SFT step" + i, loss);
            }
            recordTiming("SFTTrainer.trainingStep×4", nowNs() - t0, 4);

            check("SFT globalStep==4", trainer.globalStep() == 4);
            check("SFT callbacks onStepEnd fired", steps.get() == 4);
            check("SFT callbacks onLog fired", logs.get() >= 1);
            checkFinite("SFT last logged loss", lastLoss.get());

            // train() via BatchSupplier
            SFTConfig cfg2 = SFTConfig.builder()
                    .learningRate(1e-3).maxSteps(3).loggingSteps(10)
                    .ignoreIndex(0L).maxSeqLength(SEQ).build();
            CausalLM m2 = tinyModel();
            Adam opt2 = adam(m2, cfg2.learningRate());
            try (SFTTrainer t2 = new SFTTrainer(m2, asForward(m2), opt2, cfg2)) {
                final int[] i = {0};
                t2.train(() -> {
                    if (i[0]++ >= 3) return null;
                    return sftBatch(BATCH, SEQ, m2.vocabSize(), 200L + i[0]);
                });
                check("SFT train(supplier) globalStep==3", t2.globalStep() == 3);
            }

            // convenience ctor (Module.forward only)
            CausalLM m3 = tinyModel();
            SFTConfig cfg3 = SFTConfig.builder().learningRate(1e-3).maxSteps(1).ignoreIndex(0L).build();
            try (SFTTrainer t3 = new SFTTrainer(m3, adam(m3, 1e-3), cfg3)) {
                double l = t3.trainingStep(sftBatch(1, SEQ, m3.vocabSize(), 7L));
                checkFinite("SFT convenience ctor step", l);
            }

            // train/eval mode flip
            trainer.eval();
            check("SFT isTraining=false after eval", !trainer.isTraining());
            trainer.train();
            check("SFT isTraining=true after train()", trainer.isTraining());
        }
    }

    // ================================================================== D5 DPO trainer

    static void d5DpoTrainer() {
        section("D5 DPOTrainer (precomputed + online + loss types + ref-free)");
        manual_seed(3);

        // --- precomputed logps path (fast, no LM forward) ---
        CausalLM policy = tinyModel(); // only needed as Module holder
        for (String lossType : new String[]{"sigmoid", "hinge", "ipo"}) {
            DPOConfig cfg = DPOConfig.builder()
                    .learningRate(1e-3)
                    .maxSteps(2)
                    .loggingSteps(1)
                    .beta(0.1)
                    .lossType(lossType)
                    .referenceFree(false)
                    .build();
            try (DPOTrainer tr = new DPOTrainer(policy, asForward(policy), adam(policy, 1e-3), cfg)) {
                double l = tr.trainingStep(prefLogpsBatch(BATCH));
                checkFinite("DPO precomputed/" + lossType, l);
            }
        }

        // --- reference-free precomputed ---
        {
            DPOConfig cfg = DPOConfig.builder()
                    .learningRate(1e-3).maxSteps(1).beta(0.1)
                    .referenceFree(true).lossType("sigmoid").build();
            Map<String, Tensor> b = prefLogpsBatch(BATCH);
            b.remove("ref_chosen_logps");
            b.remove("ref_rejected_logps");
            try (DPOTrainer tr = new DPOTrainer(policy, asForward(policy), adam(policy, 1e-3), cfg)) {
                checkFinite("DPO reference-free precomputed", tr.trainingStep(b));
            }
        }

        // --- online with policy + frozen reference ---
        {
            CausalLM pol = tinyModel();
            CausalLM ref = tinyModelCloneWeights(pol);
            DPOConfig cfg = DPOConfig.builder()
                    .learningRate(1e-3)
                    .maxSteps(3)
                    .loggingSteps(1)
                    .beta(0.1)
                    .lossType("sigmoid")
                    .referenceFree(false)
                    .gradientAccumulationSteps(1)
                    .maxGradNorm(1.0)
                    .build();
            long t0 = nowNs();
            try (DPOTrainer tr = new DPOTrainer(
                    pol, asForward(pol), ref, asForward(ref), adam(pol, cfg.learningRate()), cfg)) {
                check("DPO has reference", tr.reference() == ref);
                List<Double> losses = new ArrayList<>();
                for (int i = 0; i < 3; i++) {
                    double l = tr.trainingStep(prefBatch(BATCH, SEQ, pol.vocabSize(), 300L + i));
                    losses.add(l);
                    checkFinite("DPO online step" + i, l);
                }
                check("DPO online globalStep==3", tr.globalStep() == 3);
            }
            recordTiming("DPOTrainer.online×3", nowNs() - t0, 3);
        }

        // --- online reference-free (no ref model) ---
        {
            CausalLM pol = tinyModel();
            DPOConfig cfg = DPOConfig.builder()
                    .learningRate(1e-3).maxSteps(2).beta(0.1)
                    .referenceFree(true).build();
            try (DPOTrainer tr = new DPOTrainer(pol, asForward(pol), adam(pol, 1e-3), cfg)) {
                check("DPO ref-free reference()==null", tr.reference() == null);
                checkFinite("DPO online ref-free",
                        tr.trainingStep(prefBatch(1, SEQ, pol.vocabSize(), 11L)));
            }
        }

        // --- length-normalized logps flag ---
        {
            CausalLM pol = tinyModel();
            DPOConfig cfg = DPOConfig.builder()
                    .learningRate(1e-3).maxSteps(1).beta(0.1).referenceFree(true).build();
            try (DPOTrainer tr = new DPOTrainer(
                    pol, asForward(pol), null, null, adam(pol, 1e-3), cfg, /*lengthNormalize=*/true)) {
                checkFinite("DPO lengthNormalize",
                        tr.trainingStep(prefBatch(1, SEQ, pol.vocabSize(), 13L)));
            }
        }
    }

    // ================================================================== D6 ORPO trainer

    static void d6OrpoTrainer() {
        section("D6 ORPOTrainer (precomputed + online, ± length norm)");
        manual_seed(4);

        CausalLM policy = tinyModel();
        // precomputed
        {
            ORPOConfig cfg = ORPOConfig.builder()
                    .learningRate(1e-3).maxSteps(2).beta(0.1).build();
            try (ORPOTrainer tr = new ORPOTrainer(policy, asForward(policy), adam(policy, 1e-3), cfg)) {
                Map<String, Tensor> b = new LinkedHashMap<>();
                b.put("policy_chosen_logps", floatVec(true, -1.0, -1.2));
                b.put("policy_rejected_logps", floatVec(true, -2.5, -3.0));
                checkFinite("ORPO precomputed", tr.trainingStep(b));
            }
        }

        // online
        {
            CausalLM pol = tinyModel();
            ORPOConfig cfg = ORPOConfig.builder()
                    .learningRate(1e-3).maxSteps(3).beta(0.1)
                    .lengthNormalize(false).loggingSteps(1).build();
            long t0 = nowNs();
            try (ORPOTrainer tr = new ORPOTrainer(pol, asForward(pol), adam(pol, 1e-3), cfg)) {
                for (int i = 0; i < 3; i++) {
                    checkFinite("ORPO online step" + i,
                            tr.trainingStep(prefBatch(BATCH, SEQ, pol.vocabSize(), 400L + i)));
                }
                check("ORPO globalStep==3", tr.globalStep() == 3);
            }
            recordTiming("ORPOTrainer.online×3", nowNs() - t0, 3);
        }

        // length-normalized
        {
            CausalLM pol = tinyModel();
            ORPOConfig cfg = ORPOConfig.builder()
                    .learningRate(1e-3).maxSteps(1).beta(0.1).lengthNormalize(true).build();
            try (ORPOTrainer tr = new ORPOTrainer(pol, asForward(pol), adam(pol, 1e-3), cfg)) {
                check("ORPO lengthNormalize flag", tr.orpoConfig().lengthNormalize());
                checkFinite("ORPO lengthNormalize step",
                        tr.trainingStep(prefBatch(1, SEQ, pol.vocabSize(), 19L)));
            }
        }
    }

    // ================================================================== D7 PPO trainer

    static void d7PpoTrainer() {
        section("D7 PPOTrainer (precomputed + online + GAE)");
        manual_seed(5);

        // precomputed rollout
        {
            CausalLM model = tinyModel();
            PPOConfig cfg = PPOConfig.builder()
                    .learningRate(1e-3).maxSteps(3).loggingSteps(1)
                    .clipRange(0.2).clipRangeVf(0.2).vfCoef(0.5).entCoef(0.01)
                    .build();
            long t0 = nowNs();
            try (PPOTrainer tr = new PPOTrainer(model, adam(model, 1e-3), cfg)) {
                for (int i = 0; i < 3; i++) {
                    checkFinite("PPO precomputed step" + i, tr.trainingStep(ppoPrecomputedBatch(4)));
                }
                check("PPO globalStep==3", tr.globalStep() == 3);
            }
            recordTiming("PPOTrainer.precomputed×3", nowNs() - t0, 3);
        }

        // online PolicyValueForward — reuse LM logits + a cheap value head (mean pooled)
        {
            CausalLM model = tinyModel();
            // detach-free value head: linear over hidden via last-logit mean as proxy value
            // (keeps graph connected through model params for a real fine-tune step)
            LinearImpl valueHead = new LinearImpl(model.vocabSize(), 1);
            PPOTrainer.PolicyValueForward pvf = (ids, mask) -> {
                Tensor logits = model.forward(ids);          // [B,T,V]
                Tensor values = valueHead.forward(logits.mean(new long[]{1L})).squeeze(-1); // [B]
                // crude entropy proxy: zeros (PPOLoss tolerates it)
                Tensor ent = zeros(new long[]{logits.size(0)});
                return new PPOTrainer.PolicyValueOutput(logits, values, ent);
            };
            // optimizer over model + value head
            TensorVector params = model.parameters();
            TensorVector vhParams = valueHead.parameters();
            for (long i = 0; i < vhParams.size(); i++) {
                params.put(vhParams.get(i));
            }
            Adam opt = new Adam(params, new AdamOptions(1e-3));
            PPOConfig cfg = PPOConfig.builder()
                    .learningRate(1e-3).maxSteps(2)
                    .clipRange(0.2).clipRangeVf(0.2).vfCoef(0.5).entCoef(0.0)
                    .build();
            try (PPOTrainer tr = new PPOTrainer(model, pvf, opt, cfg, /*normalizeAdv=*/true)) {
                Map<String, Tensor> b = new LinkedHashMap<>();
                Tensor ids = longIds(BATCH, SEQ, model.vocabSize(), 501L);
                b.put("input_ids", ids);
                b.put("labels", ids);
                b.put("attention_mask", onesMask(BATCH, SEQ));
                // old logprobs / advantages / returns / old_values required
                b.put("old_logprobs", floatVec(-10.0, -11.0));
                b.put("advantages", floatVec(0.4, -0.2));
                b.put("returns", floatVec(1.0, 0.5));
                b.put("old_values", floatVec(0.8, 0.6));
                checkFinite("PPO online step", tr.trainingStep(b));
                check("PPO online globalStep==1", tr.globalStep() == 1);
            }
        }

        // GAE static helper
        {
            // T=4 steps, values length T+1=5
            Tensor rewards = floatVec(1.0, 0.5, 0.0, 1.0);
            Tensor values = floatVec(0.5, 0.6, 0.4, 0.7, 0.0); // bootstrap 0
            Tensor masks = floatVec(1.0, 1.0, 1.0, 0.0);       // last done
            Tensor[] out = PPOTrainer.computeGae(rewards, values, masks, 0.99, 0.95);
            check("GAE returns pair", out != null && out.length == 2);
            check("GAE advantages numel", out[0].numel() == 4);
            check("GAE returns numel", out[1].numel() == 4);
            checkFinite("GAE adv[0]", out[0].select(0, 0).item_double());
            checkFinite("GAE ret[0]", out[1].select(0, 0).item_double());
        }
    }

    // ================================================================== D8 GRPO trainer

    static void d8GrpoTrainer() {
        section("D8 GRPOTrainer (precomputed ± clip ± KL, online)");
        manual_seed(6);

        int G = 4;
        // precomputed no-clip
        {
            CausalLM pol = tinyModel();
            GRPOConfig cfg = GRPOConfig.builder()
                    .learningRate(1e-3).maxSteps(2)
                    .numGenerations(G).beta(0.0).clipRange(0.0)
                    .build();
            try (GRPOTrainer tr = new GRPOTrainer(pol, asForward(pol), null, null, adam(pol, 1e-3), cfg,
                    /*useClipping=*/false)) {
                checkFinite("GRPO precomputed no-clip",
                        tr.trainingStep(grpoPrecomputedBatch(2, G)));
            }
        }

        // precomputed clipped + KL
        {
            CausalLM pol = tinyModel();
            GRPOConfig cfg = GRPOConfig.builder()
                    .learningRate(1e-3).maxSteps(2)
                    .numGenerations(G).beta(0.04).clipRange(0.2)
                    .build();
            long t0 = nowNs();
            try (GRPOTrainer tr = new GRPOTrainer(pol, asForward(pol), null, null, adam(pol, 1e-3), cfg,
                    /*useClipping=*/true)) {
                for (int i = 0; i < 2; i++) {
                    checkFinite("GRPO clipped+KL step" + i,
                            tr.trainingStep(grpoPrecomputedBatch(2, G)));
                }
                check("GRPO globalStep==2", tr.globalStep() == 2);
            }
            recordTiming("GRPOTrainer.clipped×2", nowNs() - t0, 2);
        }

        // online logprob recompute
        {
            CausalLM pol = tinyModel();
            CausalLM ref = tinyModelCloneWeights(pol);
            GRPOConfig cfg = GRPOConfig.builder()
                    .learningRate(1e-3).maxSteps(2)
                    .numGenerations(2).beta(0.04).clipRange(0.2)
                    .build();
            try (GRPOTrainer tr = new GRPOTrainer(
                    pol, asForward(pol), ref, asForward(ref), adam(pol, 1e-3), cfg)) {
                // B*G = 2 prompts * 2 gens = 4 rows
                Map<String, Tensor> b = new LinkedHashMap<>();
                b.put("input_ids", longIds(4, SEQ, pol.vocabSize(), 600L));
                b.put("attention_mask", onesMask(4, SEQ));
                b.put("labels", longIds(4, SEQ, pol.vocabSize(), 600L));
                b.put("rewards", floatVec(1.0, 0.5, 2.0, 0.0));
                b.put("old_logprobs", floatVec(-8.0, -9.0, -7.5, -10.0));
                checkFinite("GRPO online", tr.trainingStep(b));
            }
        }

        // static group normalize
        {
            Tensor r = floatVec(1, 3, 2, 4);
            Tensor a = GRPOTrainer.groupNormalizeAdvantages(r, 2);
            check("GRPOTrainer.groupNormalizeAdvantages numel", a.numel() == 4);
        }
    }

    // ================================================================== D9 Reward trainer

    static void d9RewardTrainer() {
        section("D9 RewardTrainer (precomputed + online forward)");
        manual_seed(7);

        // precomputed rewards
        {
            CausalLM holder = tinyModel();
            RewardConfig cfg = RewardConfig.builder()
                    .learningRate(1e-3).maxSteps(3).margin(0.0).centerRewards(false).build();
            try (RewardTrainer tr = new RewardTrainer(holder, adam(holder, 1e-3), cfg)) {
                Map<String, Tensor> b = new LinkedHashMap<>();
                b.put("chosen_rewards", floatVec(true, 1.0, 0.8));
                b.put("rejected_rewards", floatVec(true, -0.5, 0.1));
                checkFinite("Reward precomputed", tr.trainingStep(b));
            }
        }

        // with margin + center
        {
            CausalLM holder = tinyModel();
            RewardConfig cfg = RewardConfig.builder()
                    .learningRate(1e-3).maxSteps(2).margin(0.5).centerRewards(true).build();
            try (RewardTrainer tr = new RewardTrainer(holder, adam(holder, 1e-3), cfg)) {
                Map<String, Tensor> b = new LinkedHashMap<>();
                b.put("chosen_rewards", floatVec(true, 2.0, 1.5));
                b.put("rejected_rewards", floatVec(true, 0.0, 0.5));
                checkFinite("Reward margin+center", tr.trainingStep(b));
            }
        }

        // online RewardForward: mean-pool LM logits → scalar via linear
        {
            CausalLM backbone = tinyModel();
            LinearImpl head = new LinearImpl(backbone.vocabSize(), 1);
            RewardTrainer.RewardForward rf = (ids, mask) -> {
                Tensor logits = backbone.forward(ids);                 // [B,T,V]
                Tensor pooled = logits.mean(new long[]{1L});           // [B,V]
                return head.forward(pooled).squeeze(-1);               // [B]
            };
            TensorVector params = backbone.parameters();
            TensorVector hp = head.parameters();
            for (long i = 0; i < hp.size(); i++) params.put(hp.get(i));
            Adam opt = new Adam(params, new AdamOptions(1e-3));
            RewardConfig cfg = RewardConfig.builder()
                    .learningRate(1e-3).maxSteps(2).margin(0.0).build();
            long t0 = nowNs();
            try (RewardTrainer tr = new RewardTrainer(backbone, rf, opt, cfg)) {
                for (int i = 0; i < 2; i++) {
                    Map<String, Tensor> b = prefBatch(BATCH, SEQ, backbone.vocabSize(), 700L + i);
                    checkFinite("Reward online step" + i, tr.trainingStep(b));
                }
                check("Reward globalStep==2", tr.globalStep() == 2);
            }
            recordTiming("RewardTrainer.online×2", nowNs() - t0, 2);
        }
    }

    // ================================================================== D10 Grad accum + clip + callbacks

    static void d10GradAccumClipCallbacks() {
        section("D10 Gradient accumulation / clip / callback contract");
        manual_seed(8);
        CausalLM model = tinyModel();
        SFTConfig cfg = SFTConfig.builder()
                .learningRate(1e-3)
                .maxSteps(2)                       // 2 optimizer steps
                .gradientAccumulationSteps(2)      // → 4 micro-batches
                .maxGradNorm(0.5)
                .loggingSteps(1)
                .ignoreIndex(0L)
                .build();
        Adam opt = adam(model, cfg.learningRate());

        AtomicInteger begin = new AtomicInteger();
        AtomicInteger end = new AtomicInteger();
        AtomicInteger stepEnd = new AtomicInteger();
        AtomicInteger logN = new AtomicInteger();
        List<Double> stepLosses = new ArrayList<>();

        TrainerCallback cb = new TrainerCallback() {
            @Override public void onTrainBegin(BaseTrainer t) { begin.incrementAndGet(); }
            @Override public void onTrainEnd(BaseTrainer t) { end.incrementAndGet(); }
            @Override public void onStepEnd(BaseTrainer t, int step, Map<String, Double> m) {
                stepEnd.incrementAndGet();
                if (m.containsKey("loss")) stepLosses.add(m.get("loss"));
            }
            @Override public void onLog(BaseTrainer t, int step, Map<String, Double> m) {
                logN.incrementAndGet();
                check("log metrics has loss_avg or loss",
                        m.containsKey("loss_avg") || m.containsKey("loss"));
            }
        };

        try (SFTTrainer tr = new SFTTrainer(model, asForward(model), opt, cfg)) {
            tr.addCallback(cb);
            final int[] i = {0};
            tr.train(() -> {
                if (i[0] >= 4) return null; // 4 micro = 2 opt steps with accum=2
                return sftBatch(BATCH, SEQ, model.vocabSize(), 800L + (i[0]++));
            });
            check("accum: onTrainBegin once", begin.get() == 1);
            check("accum: onTrainEnd once", end.get() == 1);
            check("accum: globalStep==2 (not 4)", tr.globalStep() == 2);
            check("accum: onStepEnd==2", stepEnd.get() == 2);
            check("accum: onLog fired", logN.get() >= 1);
            check("accum: step losses recorded", stepLosses.size() == 2);
            for (int k = 0; k < stepLosses.size(); k++) {
                checkFinite("accum loss[" + k + "]", stepLosses.get(k));
            }
        }

        // maxGradNorm=0 disables clipping path (should still train)
        {
            CausalLM m = tinyModel();
            SFTConfig c = SFTConfig.builder()
                    .learningRate(1e-3).maxSteps(1).maxGradNorm(0.0).ignoreIndex(0L).build();
            try (SFTTrainer tr = new SFTTrainer(m, asForward(m), adam(m, 1e-3), c)) {
                checkFinite("maxGradNorm=0 still trains",
                        tr.trainingStep(sftBatch(1, SEQ, m.vocabSize(), 9L)));
            }
        }
    }

    // ================================================================== D11 Config matrix

    static void d11ConfigMatrix() {
        section("D11 Multi-config matrix (SFT × DPO × GRPO)");
        manual_seed(9);

        double[] lrs = {1e-3, 5e-4};
        int[] accums = {1, 2};
        double[] betas = {0.1, 0.5};
        String[] dpoTypes = {"sigmoid", "ipo"};

        int matrixPass = 0, matrixTotal = 0;

        // SFT matrix
        for (double lr : lrs) {
            for (int accum : accums) {
                matrixTotal++;
                CausalLM m = tinyModel();
                SFTConfig cfg = SFTConfig.builder()
                        .learningRate(lr).maxSteps(1)
                        .gradientAccumulationSteps(accum)
                        .ignoreIndex(0L).loggingSteps(0)
                        .build();
                try (SFTTrainer tr = new SFTTrainer(m, asForward(m), adam(m, lr), cfg)) {
                    // run `accum` micro-steps to complete 1 optimizer step
                    double last = Double.NaN;
                    for (int a = 0; a < accum; a++) {
                        last = tr.trainingStep(sftBatch(1, SEQ, m.vocabSize(), 900L + a));
                    }
                    boolean ok = !Double.isNaN(last) && !Double.isInfinite(last) && tr.globalStep() == 1;
                    if (ok) matrixPass++;
                    check(String.format(Locale.US, "SFT matrix lr=%.1e accum=%d", lr, accum), ok);
                } catch (Throwable t) {
                    check("SFT matrix lr=" + lr + " accum=" + accum + " ex=" + t.getClass().getSimpleName(), false);
                }
            }
        }

        // DPO matrix (precomputed — fast)
        CausalLM holder = tinyModel();
        for (double beta : betas) {
            for (String type : dpoTypes) {
                matrixTotal++;
                DPOConfig cfg = DPOConfig.builder()
                        .learningRate(1e-3).maxSteps(1).beta(beta).lossType(type).build();
                try (DPOTrainer tr = new DPOTrainer(holder, asForward(holder), adam(holder, 1e-3), cfg)) {
                    double l = tr.trainingStep(prefLogpsBatch(2));
                    boolean ok = !Double.isNaN(l) && !Double.isInfinite(l);
                    if (ok) matrixPass++;
                    check(String.format(Locale.US, "DPO matrix beta=%.2f type=%s", beta, type), ok);
                } catch (Throwable t) {
                    check("DPO matrix beta=" + beta + " type=" + type + " ex=" + t.getClass().getSimpleName(), false);
                }
            }
        }

        // GRPO matrix group sizes
        for (int G : new int[]{2, 4}) {
            matrixTotal++;
            CausalLM pol = tinyModel();
            GRPOConfig cfg = GRPOConfig.builder()
                    .learningRate(1e-3).maxSteps(1).numGenerations(G).beta(0.0).clipRange(0.2)
                    .build();
            try (GRPOTrainer tr = new GRPOTrainer(pol, asForward(pol), null, null, adam(pol, 1e-3), cfg, true)) {
                double l = tr.trainingStep(grpoPrecomputedBatch(1, G));
                boolean ok = !Double.isNaN(l) && !Double.isInfinite(l);
                if (ok) matrixPass++;
                check("GRPO matrix G=" + G, ok);
            } catch (Throwable t) {
                check("GRPO matrix G=" + G + " ex=" + t.getClass().getSimpleName(), false);
            }
        }

        check("config matrix majority green (" + matrixPass + "/" + matrixTotal + ")",
                matrixPass == matrixTotal);
    }

    // ================================================================== D12 PEFT + SFT

    static void d12PeftSft() {
        section("D12 LoRA-on-CausalLM + SFTTrainer (PEFT fine-tune path)");
        manual_seed(10);
        try {
            CausalLM model = tinyModel();
            LoraConfig lora = LoraConfig.builder()
                    .r(4)
                    .alpha(8)
                    .dropout(0.0)
                    .build();
            int n = model.attachLora(lora);
            check("attachLora count>0", n > 0);
            check("loraAdapters non-empty", !model.loraAdapters().isEmpty());

            // Count trainable: base may still require grad depending on freeze; just ensure step works
            SFTConfig cfg = SFTConfig.builder()
                    .learningRate(1e-3).maxSteps(2).ignoreIndex(0L).loggingSteps(1)
                    .maxGradNorm(1.0).build();
            // Optimize all params that require grad (LoRA + possibly base)
            Adam opt = adam(model, cfg.learningRate());
            long t0 = nowNs();
            try (SFTTrainer tr = new SFTTrainer(model, asForward(model), opt, cfg)) {
                for (int i = 0; i < 2; i++) {
                    checkFinite("LoRA+SFT step" + i,
                            tr.trainingStep(sftBatch(BATCH, SEQ, model.vocabSize(), 1000L + i)));
                }
                check("LoRA+SFT globalStep==2", tr.globalStep() == 2);
            }
            recordTiming("LoRA+SFTTrainer×2", nowNs() - t0, 2);
        } catch (Throwable t) {
            // PEFT path is optional for TRL core usability — report but don't hide
            check("LoRA+SFT path exception-free: " + t.getClass().getSimpleName() + ": " + t.getMessage(),
                    false);
            t.printStackTrace(System.out);
        }
    }

    // ================================================================== D13 Loss decreases smoke

    static void d13LossTrend() {
        section("D13 Overfit smoke: SFT loss trends down on fixed batch");
        manual_seed(11);
        CausalLM model = tinyModel();
        SFTConfig cfg = SFTConfig.builder()
                .learningRate(5e-3)          // aggressive for tiny overfit
                .maxSteps(20)
                .loggingSteps(5)
                .ignoreIndex(0L)
                .maxGradNorm(1.0)
                .build();
        Adam opt = adam(model, cfg.learningRate());
        Map<String, Tensor> fixed = sftBatch(BATCH, SEQ, model.vocabSize(), 42L);

        List<Double> losses = new ArrayList<>();
        long t0 = nowNs();
        try (SFTTrainer tr = new SFTTrainer(model, asForward(model), opt, cfg)) {
            for (int i = 0; i < 20; i++) {
                losses.add(tr.trainingStep(fixed));
            }
        }
        recordTiming("SFT overfit×20", nowNs() - t0, 20);

        double first = losses.get(0);
        double last = losses.get(losses.size() - 1);
        double mid = losses.get(losses.size() / 2);
        checkFinite("overfit first", first);
        checkFinite("overfit last", last);
        // Not a hard guarantee on tiny random data, but with lr=5e-3 we expect non-increase overall.
        // Accept either strict decrease OR last <= first * 1.05 (numerical noise / plateau).
        boolean improved = last <= first * 1.05;
        check(String.format(Locale.US, "overfit last(%.4f) <= first(%.4f)*1.05 (mid=%.4f)",
                last, first, mid), improved);

        // Also verify all finite
        boolean allFinite = true;
        for (double l : losses) {
            if (Double.isNaN(l) || Double.isInfinite(l)) { allFinite = false; break; }
        }
        check("overfit all 20 losses finite", allFinite);
    }

    // ================================================================== D14 Error contracts

    static void d14ErrorContracts() {
        section("D14 Error contracts (missing batch keys)");
        CausalLM model = tinyModel();

        // SFT missing input_ids
        {
            SFTConfig cfg = SFTConfig.builder().maxSteps(1).ignoreIndex(0L).build();
            try (SFTTrainer tr = new SFTTrainer(model, asForward(model), adam(model, 1e-3), cfg)) {
                boolean threw = false;
                try {
                    tr.trainingStep(new HashMap<>());
                } catch (IllegalArgumentException ex) {
                    threw = ex.getMessage() != null && ex.getMessage().contains("input_ids");
                }
                check("SFT missing input_ids throws", threw);
            }
        }

        // DPO missing chosen_input_ids (online path)
        {
            DPOConfig cfg = DPOConfig.builder().maxSteps(1).referenceFree(true).build();
            try (DPOTrainer tr = new DPOTrainer(model, asForward(model), adam(model, 1e-3), cfg)) {
                boolean threw = false;
                try {
                    Map<String, Tensor> b = new HashMap<>();
                    b.put("rejected_input_ids", longIds(1, SEQ, model.vocabSize(), 1L));
                    tr.trainingStep(b);
                } catch (IllegalArgumentException ex) {
                    threw = ex.getMessage() != null && ex.getMessage().contains("chosen_input_ids");
                }
                check("DPO missing chosen_input_ids throws", threw);
            }
        }

        // PPO missing old_logprobs
        {
            PPOConfig cfg = PPOConfig.builder().maxSteps(1).build();
            try (PPOTrainer tr = new PPOTrainer(model, adam(model, 1e-3), cfg)) {
                boolean threw = false;
                try {
                    tr.trainingStep(new HashMap<>());
                } catch (IllegalArgumentException ex) {
                    threw = ex.getMessage() != null && ex.getMessage().contains("old_logprobs");
                }
                check("PPO missing old_logprobs throws", threw);
            }
        }

        // GRPO missing rewards
        {
            GRPOConfig cfg = GRPOConfig.builder().maxSteps(1).numGenerations(2).build();
            try (GRPOTrainer tr = new GRPOTrainer(model, asForward(model), adam(model, 1e-3), cfg)) {
                boolean threw = false;
                try {
                    tr.trainingStep(new HashMap<>());
                } catch (IllegalArgumentException ex) {
                    threw = ex.getMessage() != null && ex.getMessage().contains("rewards");
                }
                check("GRPO missing rewards throws", threw);
            }
        }

        // Reward missing both paths
        {
            RewardConfig cfg = RewardConfig.builder().maxSteps(1).build();
            try (RewardTrainer tr = new RewardTrainer(model, adam(model, 1e-3), cfg)) {
                boolean threw = false;
                try {
                    tr.trainingStep(new HashMap<>());
                } catch (IllegalStateException | IllegalArgumentException ex) {
                    threw = true;
                }
                check("Reward missing rewards throws", threw);
            }
        }
    }

    // ================================================================== D15 Multi-trainer pipeline smoke

    static void d15PipelineSmoke() {
        section("D15 Fine-tune pipeline smoke: SFT → DPO → (optional) PPO");
        manual_seed(12);

        // Stage 1: SFT
        CausalLM model = tinyModel();
        SFTConfig sftCfg = SFTConfig.builder()
                .learningRate(1e-3).maxSteps(2).ignoreIndex(0L).build();
        try (SFTTrainer sft = new SFTTrainer(model, asForward(model), adam(model, 1e-3), sftCfg)) {
            for (int i = 0; i < 2; i++) {
                checkFinite("pipeline SFT " + i,
                        sft.trainingStep(sftBatch(BATCH, SEQ, model.vocabSize(), 1100L + i)));
            }
        }

        // Stage 2: DPO on same model (preference align)
        CausalLM ref = tinyModelCloneWeights(model);
        DPOConfig dpoCfg = DPOConfig.builder()
                .learningRate(5e-4).maxSteps(2).beta(0.1).lossType("sigmoid").build();
        try (DPOTrainer dpo = new DPOTrainer(
                model, asForward(model), ref, asForward(ref), adam(model, 5e-4), dpoCfg)) {
            for (int i = 0; i < 2; i++) {
                checkFinite("pipeline DPO " + i,
                        dpo.trainingStep(prefBatch(BATCH, SEQ, model.vocabSize(), 1200L + i)));
            }
        }

        // Stage 3: PPO refine with precomputed rollout (no need for full online gen)
        PPOConfig ppoCfg = PPOConfig.builder()
                .learningRate(5e-4).maxSteps(2).clipRange(0.2).vfCoef(0.5).entCoef(0.01).build();
        try (PPOTrainer ppo = new PPOTrainer(model, adam(model, 5e-4), ppoCfg)) {
            for (int i = 0; i < 2; i++) {
                checkFinite("pipeline PPO " + i, ppo.trainingStep(ppoPrecomputedBatch(4)));
            }
        }

        check("pipeline model still has grads enabled", requiresGradSomewhere(model));
    }

    // ================================================================== main

    public static void main(String[] args) throws Exception {
        System.out.println("============================================================");
        System.out.println(" TRL multi-dimensional fine-tuning usability benchmark");
        System.out.println(" package: org.bytedeco.pytorch.llm.trl");
        System.out.println("============================================================");

        long wall0 = nowNs();
        try {
            d1Configs();
            d2UnitLosses();
            d3LogProbUtils();
            d4SftTrainer();
            d5DpoTrainer();
            d6OrpoTrainer();
            d7PpoTrainer();
            d8GrpoTrainer();
            d9RewardTrainer();
            d10GradAccumClipCallbacks();
            d11ConfigMatrix();
            d12PeftSft();
            d13LossTrend();
            d14ErrorContracts();
            d15PipelineSmoke();
        } catch (Throwable t) {
            failed++;
            failures.add("UNCAUGHT: " + t);
            System.out.println("\n!! UNCAUGHT EXCEPTION !!");
            t.printStackTrace(System.out);
        }
        long wallNs = nowNs() - wall0;

        System.out.println("\n==================== TIMINGS ====================");
        for (String line : timings) {
            System.out.println("  " + line);
        }
        System.out.printf(Locale.US, "  %-40s  total=%.2f ms%n", "WALL", wallNs / 1e6);

        System.out.println("\n============================================================");
        System.out.println("TRL  passed=" + passed + "  failed=" + failed + "  skipped=" + skipped);
        if (!failures.isEmpty()) {
            System.out.println("Failures:");
            for (String f : failures) System.out.println("  - " + f);
        }
        System.out.println("============================================================");
        if (failed > 0) {
            System.out.println("SOME DIMENSIONS RED — TRL fine-tune path has issues");
            System.exit(1);
        }
        System.out.println("ALL DIMENSIONS GREEN — TRL usable for LLM fine-tuning");
    }
}
