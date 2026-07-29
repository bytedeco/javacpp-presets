/*
 * GRPO trainer for generative recommenders (MiniOneRec-style).
 *
 * Reference:
 *   - MiniOneRec RL stage (GRPO + constrained decoding)
 *     https://github.com/AkaliKong/MiniOneRec
 *   - OneRec Iterative Preference Alignment
 *
 * Algorithm (Group Relative Policy Optimization):
 *   For each context prefix:
 *     1) Sample G completions via constrained beam over SID trie
 *     2) Reward each completion (hit@target, rank-aware, optional CF)
 *     3) Standardize rewards within the group
 *     4) Policy gradient with group advantages + KL penalty to SFT reference
 *
 * Works with OneRec / OneRecV2 / OpenOneRec. Reference model is a frozen copy
 * of initial SFT weights (caller may pass the same module architecture with
 * cloned params, or null to skip KL).
 */
package org.bytedeco.pytorch.utils.recommend.trainers;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.PointerScope;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.Device;
import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.global.torch.ScalarType;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.optim.Adam;
import org.bytedeco.pytorch.optim.Optimizer;
import org.bytedeco.pytorch.optim.options.AdamOptions;
import org.bytedeco.pytorch.utils.recommend.DeviceSupport;
import org.bytedeco.pytorch.utils.recommend.TensorHelpers;
import org.bytedeco.pytorch.utils.recommend.data.Batch;
import org.bytedeco.pytorch.utils.recommend.models.generative.ConstrainedBeamSearch;
import org.bytedeco.pytorch.utils.recommend.models.generative.OneRec;
import org.bytedeco.pytorch.utils.recommend.models.generative.OneRecV2;
import org.bytedeco.pytorch.utils.recommend.models.generative.OpenOneRec;
import org.bytedeco.pytorch.utils.recommend.models.generative.SemanticID;
import org.bytedeco.pytorch.utils.tqdm.Tqdm;
import org.bytedeco.pytorch.utils.tqdm.TqdmBar;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class GRPOTrainer {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    @FunctionalInterface
    public interface RewardFn {
        /**
         * @param generatedSid encoded code tokens length L
         * @param targetSid    encoded target code tokens length L (may be null)
         * @return scalar reward (higher better)
         */
        double reward(int[] generatedSid, int[] targetSid);
    }

    private final Module policy;
    private final Module reference; // nullable
    private final SemanticID.Trie trie;
    private final int numLevels;
    private Optimizer optimizer;
    private String device;
    private float learningRate;
    private int groupSize;
    private int beamSize;
    private float klCoeff;
    private float clipEps;
    private RewardFn rewardFn;
    private boolean verbose;

    public GRPOTrainer(Module policy, SemanticID.Trie trie, int numLevels) {
        this(policy, null, trie, numLevels);
    }

    public GRPOTrainer(Module policy, Module reference, SemanticID.Trie trie, int numLevels) {
        this.policy = policy;
        this.reference = reference;
        this.trie = trie;
        this.numLevels = numLevels;
        this.device = DeviceSupport.backend();
        this.learningRate = 5e-6f;
        this.groupSize = 4;
        this.beamSize = 8;
        this.klCoeff = 0.05f;
        this.clipEps = 0.2f;
        this.verbose = true;
        this.rewardFn = GRPOTrainer::defaultHitReward;
        if (reference != null) {
            reference.eval();
        }
    }

    public GRPOTrainer learningRate(float lr) { this.learningRate = lr; return this; }
    public GRPOTrainer device(String d) { this.device = d; return this; }
    public GRPOTrainer groupSize(int g) { this.groupSize = Math.max(2, g); return this; }
    public GRPOTrainer beamSize(int b) { this.beamSize = Math.max(groupSize, b); return this; }
    public GRPOTrainer klCoeff(float k) { this.klCoeff = Math.max(0f, k); return this; }
    public GRPOTrainer clipEps(float e) { this.clipEps = Math.max(0f, e); return this; }
    public GRPOTrainer rewardFn(RewardFn fn) { this.rewardFn = fn; return this; }
    public GRPOTrainer verbose(boolean v) { this.verbose = v; return this; }
    public GRPOTrainer withOptimizer(Optimizer opt) { this.optimizer = opt; return this; }

    public static double defaultHitReward(int[] generated, int[] target) {
        if (generated == null || target == null) return 0.0;
        if (generated.length != target.length) return 0.0;
        for (int i = 0; i < generated.length; i++) {
            if (generated[i] != target[i]) {
                // partial credit by prefix match
                return i / (double) target.length;
            }
        }
        return 1.0;
    }

    /** Rank-aware: full hit=1, partial prefix fraction, plus small bonus for any trie-legal. */
    public static RewardFn rankAwareReward(SemanticID.Trie trie) {
        return (gen, tgt) -> {
            double base = defaultHitReward(gen, tgt);
            boolean legal = trie != null && trie.contains(gen);
            return base + (legal ? 0.05 : -0.05);
        };
    }

    private void ensureOptimizer() {
        if (optimizer == null) {
            optimizer = new Adam(policy.parameters(), new AdamOptions(learningRate));
        }
    }

    private Tensor forwardLogits(Module m, Tensor tokens) {
        if (m instanceof OneRec) return ((OneRec) m).forward(tokens);
        if (m instanceof OneRecV2) return ((OneRecV2) m).forward(tokens);
        if (m instanceof OpenOneRec) return ((OpenOneRec) m).forward(tokens);
        throw new IllegalArgumentException("Unsupported policy: " + m.getClass().getName());
    }

    /**
     * Average log-prob of completion tokens conditioned on context.
     * context [1,Tc], completion int[L] → scalar tensor with grad through policy.
     */
    private Tensor completionLogProb(Module m, Tensor context, int[] completion) {
        Tensor comp = TensorHelpers.tensor(completion, 1L, (long) completion.length)
                .toType(ScalarType.Long);
        try {
            comp = comp.to(new Device(device), ScalarType.Long);
        } catch (Throwable ignored) {}
        Tensor full = TensorHelpers.cat(new Tensor[]{context, comp}, 1);
        // teacher-force: input=full[:, :-1] but we only score the completion region
        Tensor input = full.narrow(1, 0, full.size(1) - 1);
        Tensor target = full.narrow(1, 1, full.size(1) - 1);
        Tensor logits = forwardLogits(m, input);
        Tensor logProb = torch.log_softmax(logits, 2L);
        Tensor tokLp = logProb.gather(2, target.toType(ScalarType.Long).unsqueeze(2)).squeeze(2);
        // mean over last L positions (completion)
        long T = tokLp.size(1);
        Tensor compLp = tokLp.narrow(1, T - completion.length, completion.length);
        return compLp.mean();
    }

    /**
     * One GRPO step on a single-user context batch (tokens = context prefix only,
     * labels/targets optional via batch.targets as Long codes flattened, or
     * last L tokens of an original SFT sequence passed separately).
     *
     * @param contextTokens [1, T] prefix (BOS+hist)
     * @param targetSid     length-L encoded target codes (may be null)
     * @return metrics map
     */
    public Map<String, Float> step(Tensor contextTokens, int[] targetSid) {
        ensureOptimizer();
        policy.train(true);

        // 1) sample group with constrained beam (no grad)
        List<int[]> group;
        try (PointerScope scope = new PointerScope()) {
            // sampling uses forward under the hood — keep outside long-lived scope for adam
        }
        group = ConstrainedBeamSearch.sampleGroup(
                policy, contextTokens, trie, groupSize, numLevels, device, System.nanoTime());
        if (group.isEmpty()) {
            Map<String, Float> empty = new HashMap<>();
            empty.put("loss", 0f);
            empty.put("reward_mean", 0f);
            return empty;
        }

        // 2) rewards + group normalize
        double[] rewards = new double[group.size()];
        double rSum = 0;
        for (int i = 0; i < group.size(); i++) {
            rewards[i] = rewardFn.reward(group.get(i), targetSid);
            rSum += rewards[i];
        }
        double rMean = rSum / group.size();
        double rVar = 0;
        for (double r : rewards) rVar += (r - rMean) * (r - rMean);
        double rStd = Math.sqrt(rVar / group.size()) + 1e-8;
        double[] adv = new double[group.size()];
        for (int i = 0; i < group.size(); i++) adv[i] = (rewards[i] - rMean) / rStd;

        // 3) policy loss
        optimizer.zero_grad();
        Tensor totalLoss = null;
        double klSum = 0;
        for (int i = 0; i < group.size(); i++) {
            Tensor logp = completionLogProb(policy, contextTokens, group.get(i));
            Tensor pg = logp.mul(new Scalar(-adv[i])); // minimise -A * logπ
            Tensor loss_i = pg;
            if (reference != null && klCoeff > 0f) {
                Tensor logpRef;
                try (org.bytedeco.pytorch.NoGradGuard g = new org.bytedeco.pytorch.NoGradGuard()) {
                    logpRef = completionLogProb(reference, contextTokens, group.get(i));
                }
                // KL ~ logπ - logπ_ref (token-average already)
                Tensor kl = logp.sub(logpRef.detach());
                klSum += TensorHelpers.itemSafe(kl);
                loss_i = loss_i.add(kl.mul(new Scalar(klCoeff)));
            }
            totalLoss = totalLoss == null ? loss_i : totalLoss.add(loss_i);
        }
        totalLoss = totalLoss.div(new Scalar((double) group.size()));
        totalLoss.backward();
        try {
            torch.clip_grad_norm_(policy.parameters(), 1.0);
        } catch (Throwable ignored) {}
        optimizer.step();

        Map<String, Float> m = new HashMap<>();
        m.put("loss", (float) TensorHelpers.itemSafe(totalLoss));
        m.put("reward_mean", (float) rMean);
        m.put("reward_std", (float) rStd);
        m.put("kl", (float) (klSum / group.size()));
        m.put("hit", targetSid == null ? 0f :
                (float) (defaultHitReward(group.get(0), targetSid) >= 1.0 ? 1.0 : 0.0));
        return m;
    }

    /**
     * Run GRPO over a loader of SFT-style batches.
     * Each batch row: tokens = [BOS+hist+targetCodes...]; we split off last L as target
     * and use the prefix as context.
     */
    public Map<String, Float> fit(Iterable<Batch> data, int maxSteps) {
        ensureOptimizer();
        double lossSum = 0, rewardSum = 0, hitSum = 0;
        int n = 0;
        Iterator<Batch> it = data.iterator();
        int total = maxSteps > 0 ? maxSteps : 100;
        TqdmBar<Integer> bar = Tqdm.range(total)
                .setDescription("GRPO")
                .setUnit("step")
                .colour("magenta")
                .setMinInterval(0.2);
        try {
            while (bar.hasNext() && (maxSteps <= 0 || n < maxSteps)) {
                bar.next();
                if (!it.hasNext()) {
                    // restart
                    it = data.iterator();
                    if (!it.hasNext()) break;
                }
                Batch batch = it.next();
                if (batch == null || batch.tokens == null) continue;
                // use first row of batch
                Tensor row = batch.tokens.select(0, 0).unsqueeze(0); // [1,T]
                long[] flat = TensorHelpers.toLongArray(
                        row.reshape(-1L).to(ScalarType.Long).cpu().contiguous());
                // strip trailing pads
                int len = flat.length;
                while (len > numLevels + 1 && flat[len - 1] == SemanticID.PAD) len--;
                if (len <= numLevels) continue;
                int[] target = new int[numLevels];
                for (int i = 0; i < numLevels; i++) {
                    target[i] = (int) flat[len - numLevels + i];
                }
                int prefLen = len - numLevels;
                int[] pref = new int[prefLen];
                for (int i = 0; i < prefLen; i++) pref[i] = (int) flat[i];
                Tensor context = TensorHelpers.tensor(pref, 1L, (long) prefLen).toType(ScalarType.Long);
                try {
                    context = context.to(new Device(device), ScalarType.Long);
                } catch (Throwable ignored) {}

                Map<String, Float> sm;
                try (PointerScope scope = new PointerScope()) {
                    sm = step(context, target);
                }
                lossSum += sm.getOrDefault("loss", 0f);
                rewardSum += sm.getOrDefault("reward_mean", 0f);
                hitSum += sm.getOrDefault("hit", 0f);
                n++;
                Map<String, Object> pf = new HashMap<>();
                pf.put("loss", String.format("%.4f", sm.getOrDefault("loss", 0f)));
                pf.put("R", String.format("%.3f", sm.getOrDefault("reward_mean", 0f)));
                pf.put("hit", String.format("%.2f", sm.getOrDefault("hit", 0f)));
                bar.set_postfix(pf);
            }
        } finally {
            bar.close();
        }
        Map<String, Float> out = new HashMap<>();
        out.put("loss", n > 0 ? (float) (lossSum / n) : 0f);
        out.put("reward_mean", n > 0 ? (float) (rewardSum / n) : 0f);
        out.put("hit_rate", n > 0 ? (float) (hitSum / n) : 0f);
        out.put("steps", (float) n);
        if (verbose) {
            System.out.printf("GRPO done  steps=%d  loss=%.4f  reward=%.4f  hit=%.4f%n",
                    n, out.get("loss"), out.get("reward_mean"), out.get("hit_rate"));
        }
        return out;
    }
}
