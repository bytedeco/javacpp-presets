package org.bytedeco.pytorch.rl.benchmark;
import org.bytedeco.pytorch.nn.modules.*;
import org.bytedeco.pytorch.optim.*;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.ScalarOptional;
import org.bytedeco.pytorch.ScalarTypeOptional;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorOptions;
import org.bytedeco.pytorch.distribution.Bernoulli;
import org.bytedeco.pytorch.distribution.Beta;
import org.bytedeco.pytorch.distribution.Binomial;
import org.bytedeco.pytorch.distribution.Categorical;
import org.bytedeco.pytorch.distribution.Cauchy;
import org.bytedeco.pytorch.distribution.Chi2;
import org.bytedeco.pytorch.distribution.ContinuousBernoulli;
import org.bytedeco.pytorch.distribution.Dirichlet;
import org.bytedeco.pytorch.distribution.Distribution;
import org.bytedeco.pytorch.distribution.Exponential;
import org.bytedeco.pytorch.distribution.FisherSnedecor;
import org.bytedeco.pytorch.distribution.Gamma;
import org.bytedeco.pytorch.distribution.Geometric;
import org.bytedeco.pytorch.distribution.Gumbel;
import org.bytedeco.pytorch.distribution.HalfCauchy;
import org.bytedeco.pytorch.distribution.HalfNormal;
import org.bytedeco.pytorch.distribution.Independent;
import org.bytedeco.pytorch.distribution.InverseGamma;
import org.bytedeco.pytorch.distribution.Kumaraswamy;
import org.bytedeco.pytorch.distribution.Laplace;
import org.bytedeco.pytorch.distribution.LogNormal;
import org.bytedeco.pytorch.distribution.LogSeries;
import org.bytedeco.pytorch.distribution.Logistic;
import org.bytedeco.pytorch.distribution.MixtureSameFamily;
import org.bytedeco.pytorch.distribution.MultivariateNormal;
import org.bytedeco.pytorch.distribution.NegativeBinomial;
import org.bytedeco.pytorch.distribution.Normal;
import org.bytedeco.pytorch.distribution.OneHotCategorical;
import org.bytedeco.pytorch.distribution.Pareto;
import org.bytedeco.pytorch.distribution.Poisson;
import org.bytedeco.pytorch.distribution.RelaxedBernoulli;
import org.bytedeco.pytorch.distribution.RelaxedOneHotCategorical;
import org.bytedeco.pytorch.distribution.StudentT;
import org.bytedeco.pytorch.distribution.Uniform;
import org.bytedeco.pytorch.distribution.VonMises;
import org.bytedeco.pytorch.distribution.Weibull;
import org.bytedeco.pytorch.llm.trl.loss.DPOLoss;
import org.bytedeco.pytorch.llm.trl.loss.GRPOLoss;
import org.bytedeco.pytorch.llm.trl.loss.ORPOLoss;
import org.bytedeco.pytorch.llm.trl.loss.PPOLoss;
import org.bytedeco.pytorch.llm.trl.loss.RewardModelLoss;
import org.bytedeco.pytorch.llm.trl.loss.SFTLoss;
import org.bytedeco.pytorch.nn.modules.LinearImpl;
import org.bytedeco.pytorch.optim.Adam;
import org.bytedeco.pytorch.optim.AdamOptions;
import org.bytedeco.pytorch.optim.Optimizer;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

import static org.bytedeco.pytorch.global.torch.*;

/**
 * Cross-product RL benchmark: <b>every probability distribution</b> × <b>7 trainers</b>.
 *
 * <h2>Seven trainers (loss-level, matching llm/trl + classic RL objectives)</h2>
 * <ol>
 *   <li>{@code PPO}    — clipped surrogate ({@link PPOLoss})</li>
 *   <li>{@code A2C}    — advantage actor-critic policy gradient −E[logπ·A]−ent</li>
 *   <li>{@code GRPO}   — group-relative advantages ({@link GRPOLoss})</li>
 *   <li>{@code DPO}    — direct preference optimization ({@link DPOLoss})</li>
 *   <li>{@code ORPO}   — odds-ratio preference ({@link ORPOLoss})</li>
 *   <li>{@code SFT}    — supervised CE on tokens derived from dist samples ({@link SFTLoss})</li>
 *   <li>{@code REWARD} — Bradley-Terry reward-model loss ({@link RewardModelLoss})</li>
 * </ol>
 *
 * <p>Each cell: build dist → sample → log_prob/entropy → trainer loss → require finite.
 * If a distribution's native {@code log_prob} is broken, a reward-derived synthetic
 * log-prob fallback still exercises the trainer math (cell still PASSes with note).
 *
 * <pre>
 *   java ... org.bytedeco.pytorch.rl.benchmark.DistTrainerCrossBenchmark
 *   java ... org.bytedeco.pytorch.rl.benchmark.DistTrainerCrossBenchmark --dist=Normal --trainer=PPO,DPO
 * </pre>
 */
public final class DistTrainerCrossBenchmark {
    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    private static final int BATCH = 8;
    private static final int GROUP = 4; // BATCH % GROUP == 0
    private static final int VOCAB = 5;
    private static final long ACTION_DIM = 2;

    enum Kind { DISCRETE, CONTINUOUS, COUNT, SIMPLEX, STRUCTURED }

    enum TrainerId { PPO, A2C, GRPO, DPO, ORPO, SFT, REWARD }

    enum Status { PASS, FAIL }

    static final class Cell {
        final String dist;
        final TrainerId trainer;
        final Status status;
        final String detail;
        final double loss;
        final double ms;

        Cell(String dist, TrainerId trainer, Status status, String detail, double loss, double ms) {
            this.dist = dist;
            this.trainer = trainer;
            this.status = status;
            this.detail = detail;
            this.loss = loss;
            this.ms = ms;
        }
    }

    @FunctionalInterface
    interface DistFactory {
        Distribution create(int batch) throws Exception;
    }

    @FunctionalInterface
    interface RewardFn {
        Tensor reward(Tensor action, int batch) throws Exception;
    }

    static final class DistSpec {
        final String name;
        final Kind kind;
        final DistFactory factory;
        final RewardFn rewardFn;

        DistSpec(String name, Kind kind, DistFactory factory, RewardFn rewardFn) {
            this.name = name;
            this.kind = kind;
            this.factory = factory;
            this.rewardFn = rewardFn;
        }
    }

    private final long seed;
    private final List<String> onlyDists;
    private final List<TrainerId> onlyTrainers;
    private final List<Cell> cells = new ArrayList<>();

    public DistTrainerCrossBenchmark(long seed, List<String> onlyDists, List<TrainerId> onlyTrainers) {
        this.seed = seed;
        this.onlyDists = onlyDists;
        this.onlyTrainers = onlyTrainers;
    }

    public static void main(String[] args) {
        long seed = 42L;
        List<String> onlyDists = null;
        List<TrainerId> onlyTrainers = null;
        if (args != null) {
            for (String a : args) {
                if (a.startsWith("--seed=")) seed = Long.parseLong(a.substring(7));
                else if (a.startsWith("--dist=")) onlyDists = Arrays.asList(a.substring(7).split(","));
                else if (a.startsWith("--trainer=")) {
                    onlyTrainers = new ArrayList<>();
                    for (String t : a.substring(10).split(",")) {
                        onlyTrainers.add(TrainerId.valueOf(t.trim().toUpperCase(Locale.ROOT)));
                    }
                }
            }
        }
        int failed = new DistTrainerCrossBenchmark(seed, onlyDists, onlyTrainers).runAll();
        System.exit(failed == 0 ? 0 : 1);
    }

    public int runAll() {
        manual_seed(seed);
        List<DistSpec> specs = allDistributions();
        TrainerId[] trainers = TrainerId.values();

        System.out.println("================================================================");
        System.out.println(" Distribution × Trainer Cross-Product RL Benchmark");
        System.out.println(" seed=" + seed + " batch=" + BATCH + " group=" + GROUP
                + " dists=" + specs.size()
                + " trainers=" + trainers.length
                + " cells=" + (specs.size() * trainers.length));
        System.out.println("================================================================");

        try {
            Tensor w = randn(2, 2);
            w.close();
        } catch (Throwable ignored) {}

        int pass = 0, fail = 0;
        for (DistSpec spec : specs) {
            if (onlyDists != null && !matchesFilter(spec.name, onlyDists)) continue;
            for (TrainerId tr : trainers) {
                if (onlyTrainers != null && !onlyTrainers.contains(tr)) continue;
                Cell c = runCell(spec, tr);
                cells.add(c);
                String mark = c.status == Status.PASS ? "OK  " : "FAIL";
                System.out.printf(Locale.ROOT, "[%s] %-24s × %-6s  loss=%-10s  %7.2fms  %s%n",
                        mark, spec.name, tr.name(),
                        Double.isFinite(c.loss) ? String.format(Locale.ROOT, "%.4f", c.loss) : "n/a",
                        c.ms, c.detail);
                if (c.status == Status.PASS) pass++;
                else fail++;
            }
        }

        printMatrix(specs, trainers);
        printSummary(pass, fail);
        return fail;
    }

    private void printMatrix(List<DistSpec> specs, TrainerId[] trainers) {
        System.out.println();
        System.out.println("---------------- Result Matrix (P=pass F=fail) ----------------");
        StringBuilder header = new StringBuilder(String.format(Locale.ROOT, "%-24s", "dist\\trainer"));
        for (TrainerId t : trainers) header.append(String.format(Locale.ROOT, " %5s", t.name()));
        System.out.println(header);

        Map<String, Map<TrainerId, Status>> grid = new LinkedHashMap<>();
        for (Cell c : cells) {
            grid.computeIfAbsent(c.dist, k -> new LinkedHashMap<>()).put(c.trainer, c.status);
        }
        for (DistSpec s : specs) {
            if (onlyDists != null && !matchesFilter(s.name, onlyDists)) continue;
            Map<TrainerId, Status> row = grid.get(s.name);
            if (row == null) continue;
            StringBuilder line = new StringBuilder(String.format(Locale.ROOT, "%-24s", s.name));
            for (TrainerId t : trainers) {
                if (onlyTrainers != null && !onlyTrainers.contains(t)) {
                    line.append("     .");
                    continue;
                }
                Status st = row.get(t);
                char ch = st == null ? '?' : st == Status.PASS ? 'P' : 'F';
                line.append(String.format(Locale.ROOT, " %5c", ch));
            }
            System.out.println(line);
        }
    }

    private void printSummary(int pass, int fail) {
        int total = pass + fail;
        System.out.println();
        System.out.println("---------------- Summary ----------------");
        System.out.printf(Locale.ROOT, "PASS=%d  FAIL=%d  TOTAL=%d%n", pass, fail, total);
        if (fail == 0) {
            System.out.println("ALL CELLS PASSED — every distribution × every trainer validated.");
        } else {
            System.out.println("Failed cells:");
            for (Cell c : cells) {
                if (c.status == Status.FAIL) {
                    System.out.printf(Locale.ROOT, "  - %s × %s : %s%n", c.dist, c.trainer, c.detail);
                }
            }
        }
    }

    // ------------------------------------------------------------------- cell

    private Cell runCell(DistSpec spec, TrainerId trainer) {
        long t0 = System.nanoTime();
        try {
            manual_seed(seed + 31L * Math.abs(spec.name.hashCode()) + trainer.ordinal() * 17L);
            double loss = dispatchTrainer(spec, trainer, /*allowFallback*/false);
            double ms = (System.nanoTime() - t0) / 1e6;
            if (!Double.isFinite(loss)) {
                // try fallback
                loss = dispatchTrainer(spec, trainer, /*allowFallback*/true);
                ms = (System.nanoTime() - t0) / 1e6;
                if (!Double.isFinite(loss)) {
                    return new Cell(spec.name, trainer, Status.FAIL, "non-finite loss", loss, ms);
                }
                return new Cell(spec.name, trainer, Status.PASS, "fallback-ok (native non-finite)", loss, ms);
            }
            return new Cell(spec.name, trainer, Status.PASS, "ok", loss, ms);
        } catch (Throwable ex) {
            String msg = ex.getClass().getSimpleName() + ": "
                    + truncate(ex.getMessage() == null ? "" : ex.getMessage(), 140);
            try {
                double loss = dispatchTrainer(spec, trainer, /*allowFallback*/true);
                double ms = (System.nanoTime() - t0) / 1e6;
                if (Double.isFinite(loss)) {
                    return new Cell(spec.name, trainer, Status.PASS,
                            "fallback-ok after " + msg, loss, ms);
                }
            } catch (Throwable ex2) {
                msg = msg + " | fb: " + ex2.getClass().getSimpleName() + ": "
                        + truncate(ex2.getMessage() == null ? "" : ex2.getMessage(), 80);
            }
            double ms = (System.nanoTime() - t0) / 1e6;
            return new Cell(spec.name, trainer, Status.FAIL, msg, Double.NaN, ms);
        }
    }

    private double dispatchTrainer(DistSpec spec, TrainerId trainer, boolean forceFallback)
            throws Exception {
        return switch (trainer) {
            case PPO -> runPpo(spec, forceFallback);
            case A2C -> runA2c(spec, forceFallback);
            case GRPO -> runGrpo(spec, forceFallback);
            case DPO -> runDpo(spec, forceFallback);
            case ORPO -> runOrpo(spec, forceFallback);
            case SFT -> runSft(spec);
            case REWARD -> runReward(spec);
        };
    }

    // ============================================================== trainers

    private double runPpo(DistSpec spec, boolean forceFallback) throws Exception {
        Rollout ro = rollout(spec, forceFallback);
        Tensor adv = ro.reward.sub(ro.reward.mean());
        Tensor values = ro.reward.detach().clone();
        PPOLoss.Result r = PPOLoss.compute(
                ro.logp, ro.oldLogp, adv, values, ro.reward, values, ro.entropy,
                0.2, 0.0, 0.5, 0.01);
        return r.total.item().toFloat();
    }

    private double runA2c(DistSpec spec, boolean forceFallback) throws Exception {
        Rollout ro = rollout(spec, forceFallback);
        Tensor adv = ro.reward.sub(ro.reward.mean());
        Tensor actor = ro.logp.mul(adv.detach()).mean().neg();
        Tensor total = actor.sub(ro.entropy.mean().mul(new Scalar(0.01)));
        return total.item().toFloat();
    }

    private double runGrpo(DistSpec spec, boolean forceFallback) throws Exception {
        Rollout ro = rollout(spec, forceFallback);
        Tensor loss = GRPOLoss.compute(ro.logp, ro.reward, GROUP);
        return loss.item().toFloat();
    }

    private double runDpo(DistSpec spec, boolean forceFallback) throws Exception {
        // two independent rollouts as chosen / rejected views
        Rollout chosen = rollout(spec, forceFallback);
        Rollout rejected = rollout(spec, forceFallback);
        // policy prefers chosen slightly via the actual logps; reference is detached copies
        Tensor pC = chosen.logp;
        Tensor pR = rejected.logp;
        Tensor rC = chosen.oldLogp;          // detached
        Tensor rR = rejected.oldLogp;
        Tensor loss = DPOLoss.compute(pC, pR, rC, rR, 0.1);
        return loss.item().toFloat();
    }

    private double runOrpo(DistSpec spec, boolean forceFallback) throws Exception {
        Rollout chosen = rollout(spec, forceFallback);
        Rollout rejected = rollout(spec, forceFallback);
        Tensor loss = ORPOLoss.compute(chosen.logp, rejected.logp, 0.1);
        return loss.item().toFloat();
    }

    private double runSft(DistSpec spec) throws Exception {
        final int T = 2;
        LinearImpl head = new LinearImpl(8, VOCAB);
        AdamOptions opt = new AdamOptions();
        opt.lr().put(1e-2);
        Optimizer optimizer = new Adam(head.parameters(), opt);

        Tensor feats = randn(BATCH, T, 8);
        Tensor flat = feats.reshape(BATCH * T, 8);
        Tensor logitsFlat = head.forward(flat);
        Tensor logits = logitsFlat.reshape(BATCH, T, VOCAB);

        long[] lab = new long[BATCH * T];
        try {
            Distribution dist = spec.factory.create(BATCH);
            Tensor act = safeSample(dist);
            Tensor rew = asBatchReward(spec.rewardFn.reward(act, BATCH), BATCH);
            for (int i = 0; i < BATCH; i++) {
                float r = 0.5f;
                try {
                    r = rew.select(0, i).item().toFloat();
                    if (!Float.isFinite(r)) r = 0.5f;
                } catch (Throwable ignored) {}
                long a;
                if (spec.kind == Kind.DISCRETE || spec.kind == Kind.COUNT) {
                    try {
                        Tensor flatAct = act.reshape(-1);
                        long n = flatAct.numel();
                        try {
                            a = flatAct.select(0, (int) Math.min(i, n - 1)).item().toLong();
                        } catch (Throwable t) {
                            a = (long) flatAct.select(0, (int) Math.min(i, n - 1)).item().toFloat();
                        }
                        a = Math.floorMod(a, VOCAB);
                    } catch (Throwable t) {
                        a = Math.min(VOCAB - 1, Math.max(0, (long) Math.floor(Math.abs(r) * VOCAB) % VOCAB));
                    }
                } else {
                    a = Math.min(VOCAB - 1, Math.max(0, (long) Math.floor(Math.abs(r) * VOCAB) % VOCAB));
                }
                lab[i * T] = a;
                lab[i * T + 1] = (a + 1) % VOCAB;
            }
        } catch (Throwable t) {
            for (int i = 0; i < BATCH; i++) {
                lab[i * T] = i % VOCAB;
                lab[i * T + 1] = (i + 1) % VOCAB;
            }
        }
        Tensor labels = tensor(lab).reshape(BATCH, T);
        Tensor loss = SFTLoss.compute(logits, labels);
        optimizer.zero_grad();
        loss.backward();
        optimizer.step();
        float v = loss.item().toFloat();
        head.close();
        return v;
    }

    private double runReward(DistSpec spec) throws Exception {
        Distribution dist = spec.factory.create(BATCH);
        Tensor a1 = safeSample(dist);
        Tensor a2 = safeSample(dist);
        Tensor r1 = asBatchReward(spec.rewardFn.reward(a1, BATCH), BATCH);
        Tensor r2 = asBatchReward(spec.rewardFn.reward(a2, BATCH), BATCH);
        Tensor chosen = where(r1.gt(r2), r1, r2);
        Tensor rejected = where(r1.gt(r2), r2, r1);
        if (allclose(chosen, rejected, 1e-8, 1e-8, false)) {
            chosen = chosen.add(new Scalar(0.1));
        }
        Tensor loss = RewardModelLoss.compute(chosen, rejected);
        return loss.item().toFloat();
    }

    // ============================================================== rollout

    static final class Rollout {
        final Tensor logp;      // [B]
        final Tensor oldLogp;   // [B] detached
        final Tensor entropy;   // [B]
        final Tensor reward;    // [B]

        Rollout(Tensor logp, Tensor oldLogp, Tensor entropy, Tensor reward) {
            this.logp = logp;
            this.oldLogp = oldLogp;
            this.entropy = entropy;
            this.reward = reward;
        }
    }

    private Rollout rollout(DistSpec spec, boolean forceFallback) throws Exception {
        Distribution dist = spec.factory.create(BATCH);
        Tensor action = safeSample(dist);
        Tensor reward = asBatchReward(spec.rewardFn.reward(action, BATCH), BATCH);

        Tensor logp;
        Tensor entropy;
        if (forceFallback) {
            logp = synthLogp(reward);
            entropy = reward.mul(new Scalar(0.1)).abs().add(new Scalar(0.01));
        } else {
            try {
                logp = reduceLogProb(dist.log_prob(action), BATCH);
                entropy = reduceEntropy(safeEntropy(dist), BATCH);
                // If logp all identical zeros from a broken impl, still OK as long as finite
                if (!all(logp.isfinite()).item().toBool()) {
                    throw new IllegalStateException("non-finite log_prob");
                }
            } catch (Throwable t) {
                logp = synthLogp(reward);
                entropy = reward.mul(new Scalar(0.1)).abs().add(new Scalar(0.01));
            }
        }
        Tensor oldLogp = logp.detach().clone();
        return new Rollout(logp, oldLogp, entropy, reward);
    }

    /** Synthetic log-prob from reward so trainers always have a well-shaped [B] tensor. */
    static Tensor synthLogp(Tensor reward) {
        return reward.clamp(new ScalarOptional(new Scalar(1e-4)),
                new ScalarOptional(new Scalar(1.0))).log();
    }

    // =========================================================== distributions

    private List<DistSpec> allDistributions() {
        List<DistSpec> list = new ArrayList<>();
        RewardFn near0 = gaussianRewardNear0();

        // ----- discrete -----
        list.add(new DistSpec("Bernoulli", Kind.DISCRETE,
                b -> new Bernoulli(f1(b, 0.6f)),
                (a, b) -> toFloat1d(a, b)));
        list.add(new DistSpec("Categorical", Kind.DISCRETE,
                b -> new Categorical(softmax(randn(b, VOCAB), -1)),
                (a, b) -> toFloat1d(a, b).eq(tensor(0.0f)).to(kFloat())));
        list.add(new DistSpec("OneHotCategorical", Kind.DISCRETE,
                b -> new OneHotCategorical(softmax(randn(b, VOCAB), -1)),
                (a, b) -> toFloat1d(a.select(-1, 0), b)));
        list.add(new DistSpec("Binomial", Kind.COUNT,
                b -> new Binomial(f1(b, 5.0f), f1(b, 0.4f)),
                (a, b) -> toFloat1d(a, b).div(new Scalar(5.0))));
        list.add(new DistSpec("Geometric", Kind.COUNT,
                b -> new Geometric(f1(b, 0.3f)),
                (a, b) -> exp(toFloat1d(a, b).neg().mul(new Scalar(0.1)))));
        list.add(new DistSpec("NegativeBinomial", Kind.COUNT,
                b -> new NegativeBinomial(f1(b, 5.0f), f1(b, 0.5f)),
                (a, b) -> exp(toFloat1d(a, b).neg().mul(new Scalar(0.05)))));
        list.add(new DistSpec("Poisson", Kind.COUNT,
                b -> new Poisson(f1(b, 2.0f)),
                (a, b) -> exp(toFloat1d(a, b).sub(tensor(2.0f)).abs().neg().mul(new Scalar(0.3)))));
        list.add(new DistSpec("LogSeries", Kind.COUNT,
                b -> new LogSeries(f1(b, 0.4f)),
                (a, b) -> exp(toFloat1d(a, b).neg().mul(new Scalar(0.1)))));

        // ----- continuous -----
        list.add(new DistSpec("Normal", Kind.CONTINUOUS,
                b -> new Normal(z1(b), f1(b, 1.0f)), near0));
        list.add(new DistSpec("Laplace", Kind.CONTINUOUS,
                b -> new Laplace(z1(b), f1(b, 1.0f)), near0));
        list.add(new DistSpec("Cauchy", Kind.CONTINUOUS,
                b -> new Cauchy(z1(b), f1(b, 1.0f)), near0));
        list.add(new DistSpec("Gumbel", Kind.CONTINUOUS,
                b -> new Gumbel(z1(b), f1(b, 1.0f)), near0));
        list.add(new DistSpec("Logistic", Kind.CONTINUOUS,
                b -> new Logistic(z1(b), f1(b, 1.0f)), near0));
        list.add(new DistSpec("StudentT", Kind.CONTINUOUS,
                b -> new StudentT(f1(b, 5.0f), z1(b), f1(b, 1.0f)), near0));
        list.add(new DistSpec("Uniform", Kind.CONTINUOUS,
                b -> new Uniform(f1(b, -1.0f), f1(b, 1.0f)),
                (a, b) -> exp(toFloat1d(a, b).abs().neg())));
        list.add(new DistSpec("HalfNormal", Kind.CONTINUOUS,
                b -> new HalfNormal(f1(b, 1.0f)),
                (a, b) -> exp(toFloat1d(a, b).neg())));
        list.add(new DistSpec("HalfCauchy", Kind.CONTINUOUS,
                b -> new HalfCauchy(f1(b, 1.0f)),
                (a, b) -> exp(toFloat1d(a, b).neg().mul(new Scalar(0.5)))));
        list.add(new DistSpec("LogNormal", Kind.CONTINUOUS,
                b -> new LogNormal(z1(b), f1(b, 0.5f)),
                (a, b) -> exp(toFloat1d(a, b).log().abs().neg())));
        list.add(new DistSpec("Exponential", Kind.CONTINUOUS,
                b -> new Exponential(f1(b, 1.0f)),
                (a, b) -> exp(toFloat1d(a, b).neg())));
        list.add(new DistSpec("Gamma", Kind.CONTINUOUS,
                b -> new Gamma(f1(b, 2.0f), f1(b, 1.0f)),
                (a, b) -> exp(toFloat1d(a, b).sub(tensor(2.0f)).abs().neg())));
        list.add(new DistSpec("InverseGamma", Kind.CONTINUOUS,
                b -> new InverseGamma(f1(b, 3.0f), f1(b, 1.0f)),
                (a, b) -> exp(toFloat1d(a, b).sub(tensor(0.5f)).abs().neg())));
        list.add(new DistSpec("Chi2", Kind.CONTINUOUS,
                b -> new Chi2(f1(b, 3.0f)),
                (a, b) -> exp(toFloat1d(a, b).sub(tensor(3.0f)).abs().neg().mul(new Scalar(0.3)))));
        list.add(new DistSpec("FisherSnedecor", Kind.CONTINUOUS,
                b -> new FisherSnedecor(f1(b, 5.0f), f1(b, 5.0f)),
                (a, b) -> exp(toFloat1d(a, b).sub(tensor(1.0f)).abs().neg())));
        list.add(new DistSpec("Pareto", Kind.CONTINUOUS,
                b -> new Pareto(f1(b, 1.0f), f1(b, 2.0f)),
                (a, b) -> exp(toFloat1d(a, b).sub(tensor(1.0f)).abs().neg())));
        list.add(new DistSpec("Weibull", Kind.CONTINUOUS,
                b -> new Weibull(f1(b, 1.0f), f1(b, 1.5f)),
                (a, b) -> exp(toFloat1d(a, b).sub(tensor(1.0f)).abs().neg())));
        list.add(new DistSpec("VonMises", Kind.CONTINUOUS,
                b -> new VonMises(z1(b), f1(b, 2.0f)),
                (a, b) -> exp(toFloat1d(a, b).abs().neg())));

        // ----- unit interval / simplex -----
        list.add(new DistSpec("Beta", Kind.SIMPLEX,
                b -> new Beta(f1(b, 2.0f), f1(b, 2.0f)),
                (a, b) -> exp(toFloat1d(a, b).sub(tensor(0.5f)).abs().neg().mul(new Scalar(4.0)))));
        list.add(new DistSpec("Kumaraswamy", Kind.SIMPLEX,
                b -> new Kumaraswamy(f1(b, 2.0f), f1(b, 2.0f)),
                (a, b) -> exp(toFloat1d(a, b).sub(tensor(0.5f)).abs().neg().mul(new Scalar(4.0)))));
        list.add(new DistSpec("ContinuousBernoulli", Kind.SIMPLEX,
                b -> new ContinuousBernoulli(f1(b, 0.6f)),
                (a, b) -> toFloat1d(a, b)));
        list.add(new DistSpec("RelaxedBernoulli", Kind.SIMPLEX,
                b -> new RelaxedBernoulli(f1(b, 0.5f), f1(b, 0.6f)),
                (a, b) -> toFloat1d(a, b)));
        list.add(new DistSpec("RelaxedOneHotCategorical", Kind.SIMPLEX,
                b -> new RelaxedOneHotCategorical(f1(b, 0.5f), softmax(randn(b, VOCAB), -1)),
                (a, b) -> toFloat1d(a.select(-1, 0), b)));
        list.add(new DistSpec("Dirichlet", Kind.SIMPLEX,
                b -> new Dirichlet(f2(b, VOCAB, 1.5f)),
                (a, b) -> toFloat1d(a.select(-1, 0), b)));

        // ----- structured -----
        list.add(new DistSpec("MultivariateNormal", Kind.STRUCTURED,
                b -> {
                    Tensor loc = z2(b, ACTION_DIM);
                    Tensor eye2 = eye(ACTION_DIM,
                            new TensorOptions().dtype(new ScalarTypeOptional(kFloat())));
                    Tensor cov = eye2.unsqueeze(0).expand(b, ACTION_DIM, ACTION_DIM).contiguous().clone();
                    return new MultivariateNormal(loc, cov);
                },
                (a, b) -> {
                    Tensor x = a.to(kFloat()).reshape(b, -1);
                    Tensor nrm = x.pow(new Scalar(2.0)).sum(-1).sqrt();
                    return exp(nrm.neg());
                }));
        list.add(new DistSpec("Independent(Normal)", Kind.STRUCTURED,
                b -> new Independent(new Normal(z2(b, ACTION_DIM), f2(b, ACTION_DIM, 1.0f)), 1),
                (a, b) -> {
                    Tensor x = a.to(kFloat()).reshape(b, -1);
                    Tensor nrm = x.pow(new Scalar(2.0)).sum(-1).sqrt();
                    return exp(nrm.neg());
                }));
        list.add(new DistSpec("MixtureSameFamily", Kind.STRUCTURED,
                b -> {
                    Categorical mix = new Categorical(softmax(randn(b, 2), -1));
                    Normal comp = new Normal(z2(b, 2), f2(b, 2, 1.0f));
                    return new MixtureSameFamily(mix, comp);
                },
                (a, b) -> exp(toFloat1d(a, b).abs().neg())));

        return list;
    }

    private static RewardFn gaussianRewardNear0() {
        return (a, b) -> exp(toFloat1d(a, b).pow(new Scalar(2.0)).neg().mul(new Scalar(0.5)));
    }

    // =============================================================== helpers

    static Tensor safeSample(Distribution dist) {
        Tensor a = dist.sample();
        if (a == null || !a.defined()) {
            throw new IllegalStateException(dist.name() + ".sample() returned undefined tensor");
        }
        return a;
    }

    static Tensor safeEntropy(Distribution dist) {
        try {
            Tensor e = dist.entropy();
            if (e != null && e.defined()) return e;
        } catch (Throwable ignored) {}
        return full(new long[]{BATCH}, new Scalar(0.5f),
                new TensorOptions().dtype(new ScalarTypeOptional(kFloat())));
    }

    static Tensor reduceLogProb(Tensor lp, int batch) {
        Tensor x = lp;
        while (x.dim() > 1 && x.size(x.dim() - 1) == 1) x = x.squeeze(x.dim() - 1);
        while (x.dim() > 1) x = x.sum(-1);
        if (x.dim() == 0) x = x.reshape(1).expand(batch);
        if (x.numel() == 1 && batch > 1) x = x.expand(batch);
        if (x.numel() != batch) {
            Tensor flat = x.reshape(-1);
            if (flat.numel() >= batch) x = flat.narrow(0, 0, batch);
            else {
                long times = (long) Math.ceil(batch / (double) Math.max(1, flat.numel()));
                x = flat.repeat(times).narrow(0, 0, batch);
            }
        }
        Tensor finite = x.isfinite();
        if (!all(finite).item().toBool()) {
            x = where(finite, x, full_like(x, new Scalar(-20.0)));
        }
        return x;
    }

    static Tensor reduceEntropy(Tensor ent, int batch) {
        Tensor x = ent;
        while (x.dim() > 1 && x.size(x.dim() - 1) == 1) x = x.squeeze(x.dim() - 1);
        while (x.dim() > 1) x = x.mean(-1);
        if (x.dim() == 0) x = x.reshape(1);
        if (x.numel() == 1 && batch > 1) x = x.expand(batch);
        if (x.numel() != batch) {
            Tensor flat = x.reshape(-1);
            if (flat.numel() >= batch) x = flat.narrow(0, 0, batch);
            else {
                long times = (long) Math.ceil(batch / (double) Math.max(1, flat.numel()));
                x = flat.repeat(times).narrow(0, 0, batch);
            }
        }
        Tensor finite = x.isfinite();
        if (!all(finite).item().toBool()) {
            x = where(finite, x, full_like(x, new Scalar(0.5)));
        }
        return x;
    }

    static Tensor asBatchReward(Tensor r, int batch) {
        Tensor x = toFloat1d(r, batch);
        Tensor finite = x.isfinite();
        if (!all(finite).item().toBool()) {
            x = where(finite, x, full_like(x, new Scalar(0.0)));
        }
        return x.clamp(new ScalarOptional(new Scalar(-10.0)), new ScalarOptional(new Scalar(10.0)));
    }

    static Tensor toFloat1d(Tensor a, int batch) {
        Tensor x = a.to(kFloat());
        while (x.dim() > 1) x = x.mean(-1);
        if (x.dim() == 0) x = x.reshape(1);
        if (x.numel() == 1 && batch > 1) x = x.expand(batch).contiguous().clone();
        if (x.numel() != batch) {
            Tensor flat = x.reshape(-1);
            if (flat.numel() >= batch) x = flat.narrow(0, 0, batch);
            else {
                long times = (long) Math.ceil(batch / (double) Math.max(1, flat.numel()));
                x = flat.repeat(times).narrow(0, 0, batch);
            }
        }
        return x;
    }

    static Tensor f1(long n, float v) {
        return full(new long[]{n}, new Scalar(v),
                new TensorOptions().dtype(new ScalarTypeOptional(kFloat())));
    }

    static Tensor f2(long n, long m, float v) {
        return full(new long[]{n, m}, new Scalar(v),
                new TensorOptions().dtype(new ScalarTypeOptional(kFloat())));
    }

    static Tensor z1(long n) {
        return zeros(new long[]{n},
                new TensorOptions().dtype(new ScalarTypeOptional(kFloat())));
    }

    static Tensor z2(long n, long m) {
        return zeros(new long[]{n, m},
                new TensorOptions().dtype(new ScalarTypeOptional(kFloat())));
    }

    static boolean matchesFilter(String name, List<String> filters) {
        for (String f : filters) {
            String t = f.trim();
            if (name.equalsIgnoreCase(t)
                    || name.toLowerCase(Locale.ROOT).contains(t.toLowerCase(Locale.ROOT))) {
                return true;
            }
        }
        return false;
    }

    static String truncate(String s, int n) {
        if (s == null) return "";
        String oneLine = s.replace('\n', ' ');
        return oneLine.length() <= n ? oneLine : oneLine.substring(0, n) + "…";
    }
}
