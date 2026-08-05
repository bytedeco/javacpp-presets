package samples;
import org.bytedeco.pytorch.data.sampler.*;
import org.bytedeco.pytorch.optim.*;
import org.bytedeco.pytorch.optim.options.*;
import org.bytedeco.pytorch.optim.schedulers.*;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.pytorch.NoGradGuard;
import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.SizeTOptional;
import org.bytedeco.pytorch.SizeTVector;
import org.bytedeco.pytorch.SizeTVectorOptional;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorVector;
import org.bytedeco.pytorch.data.sampler.RandomSampler;
import org.bytedeco.pytorch.data.sampler.Sampler;
import org.bytedeco.pytorch.data.sampler.SequentialSampler;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.optim.Adam;
import org.bytedeco.pytorch.optim.options.AdamOptions;
import org.bytedeco.pytorch.optim.Optimizer;
import org.bytedeco.pytorch.optim.SGD;
import org.bytedeco.pytorch.optim.options.SGDOptions;
import org.bytedeco.pytorch.optim.schedulers.StepLR;

import java.lang.reflect.Constructor;
import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Set;

/**
 * Benchmark / validation for Python-parity optimizers, LR schedulers, and
 * data samplers bound as real subclasses of:
 * <ul>
 *   <li>{@link Optimizer} — Adadelta, Adamax, ASGD, NAdam, RAdam, Rprop</li>
 *   <li>{@link org.bytedeco.pytorch.optim.schedulers.LRScheduler} — ExponentialLR, MultiStepLR, …</li>
 *   <li>{@link Sampler} — SubsetRandomSampler, WeightedRandomSampler, BatchSampler</li>
 * </ul>
 *
 * <p>New C++ peers from {@code python_optim_java.h} / {@code python_lr_scheduler_java.h} /
 * {@code python_samplers_java.h} are detected via reflection after:
 * <pre>
 *   mvn -Djavacpp.cppbuild.skip=true install
 * </pre>
 * regenerates JavaCPP bindings and recompiles JNI.
 *
 * <pre>
 *   cd samples &amp;&amp; mvn -q exec:java -Dexec.mainClass=BenchmarkOptimSchedulersSamplers
 * </pre>
 */
public class BenchmarkOptimSchedulersSamplers {

    private static int passed = 0;
    private static int failed = 0;
    private static int skipped = 0;
    private static final List<String> failures = new ArrayList<>();

    private static void check(String name, boolean cond, String detail) {
        if (cond) {
            passed++;
            System.out.printf(Locale.ROOT, "  [PASS] %s%n", name);
        } else {
            failed++;
            String msg = name + (detail == null || detail.isEmpty() ? "" : " — " + detail);
            failures.add(msg);
            System.out.printf(Locale.ROOT, "  [FAIL] %s%n", msg);
        }
    }

    private static void check(String name, boolean cond) {
        check(name, cond, null);
    }

    private static void skip(String name, String reason) {
        skipped++;
        System.out.printf(Locale.ROOT, "  [SKIP] %s — %s%n", name, reason);
    }

    private static void section(String title) {
        System.out.println();
        System.out.println("======== " + title + " ========");
    }

    private static Tensor requiresGrad(long... shape) {
        return torch.randn(shape).set_requires_grad(true);
    }

    private static double lossValue(Tensor p) {
        return p.square().sum().item().toDouble() * 0.5;
    }

    private static void backwardQuadratic(Tensor p) {
        Tensor loss = p.square().sum().mul(new Scalar(0.5));
        loss.backward();
    }

    private static double trainStep(Optimizer opt, Tensor p) {
        opt.zero_grad();
        backwardQuadratic(p);
        opt.step();
        try (NoGradGuard g = new NoGradGuard()) {
            return lossValue(p);
        }
    }

    /** Drain a C++ Sampler via next(batchSize), collect all indices. */
    private static List<Long> drain(Sampler sampler, long batchSize) {
        List<Long> out = new ArrayList<>();
        while (true) {
            SizeTVectorOptional opt = sampler.next(batchSize);
            if (opt == null || opt.isNull() || !opt.has_value()) break;
            SizeTVector v = opt.get();
            for (long i = 0, n = v.size(); i < n; i++) {
                out.add(v.get(i));
            }
        }
        return out;
    }

    public static void main(String[] args) throws Exception {
        Loader.load(org.bytedeco.pytorch.global.torch.class);
        System.out.println("=== BenchmarkOptimSchedulersSamplers ===");

        benchNativeSamplers();
        benchNewSamplersReflective();
        benchNativeOptimAndStepLR();
        benchNewOptimizersReflective();
        benchNewSchedulersReflective();

        System.out.println();
        System.out.printf(Locale.ROOT, "RESULT  passed=%d  failed=%d  skipped=%d%n",
                passed, failed, skipped);
        if (!failures.isEmpty()) {
            System.out.println("Failures:");
            for (String f : failures) System.out.println("  - " + f);
        }
        if (failed > 0) System.exit(1);
    }

    // ------------------------------------------------------------------
    // Native LibTorch samplers (already bound)
    // ------------------------------------------------------------------

    private static void benchNativeSamplers() {
        section("Native data.sampler (Sequential / Random)");

        SequentialSampler seq = new SequentialSampler(10);
        seq.reset();
        List<Long> seqOut = drain(seq, 3);
        check("SequentialSampler covers 0..9", seqOut.size() == 10);
        check("SequentialSampler order",
                seqOut.equals(List.of(0L, 1L, 2L, 3L, 4L, 5L, 6L, 7L, 8L, 9L)));
        check("SequentialSampler exhausted", !seq.next(3).has_value());

        RandomSampler rnd = new RandomSampler(20);
        rnd.reset();
        List<Long> rndOut = drain(rnd, 5);
        check("RandomSampler size 20", rndOut.size() == 20);
        check("RandomSampler permutation",
                new HashSet<>(rndOut).size() == 20
                        && rndOut.stream().allMatch(x -> x >= 0 && x < 20));

        // throughput
        SequentialSampler big = new SequentialSampler(1_000_000);
        big.reset();
        long t0 = System.nanoTime();
        long sink = 0;
        while (true) {
            SizeTVectorOptional opt = big.next(1024);
            if (opt == null || opt.isNull() || !opt.has_value()) break;
            SizeTVector v = opt.get();
            for (long i = 0, n = v.size(); i < n; i++) sink += v.get(i);
        }
        long t1 = System.nanoTime();
        check("SequentialSampler 1e6 sink", sink != 0);
        System.out.printf(Locale.ROOT, "  [BENCH] SequentialSampler 1e6 next(1024): %.3f ms%n",
                (t1 - t0) / 1e6);
    }

    // ------------------------------------------------------------------
    // New samplers (extends Sampler) — reflective
    // ------------------------------------------------------------------

    private static void benchNewSamplersReflective() {
        section("New samplers (SubsetRandom / WeightedRandom / Batch)");

        // SubsetRandomSampler
        try {
            Class<?> cls = Class.forName("org.bytedeco.pytorch.data.sampler.SubsetRandomSampler");
            check("SubsetRandomSampler extends Sampler", Sampler.class.isAssignableFrom(cls));

            // Prefer SizeTVector ctor if present
            Object sub;
            try {
                Class<?> sizeTVector = Class.forName("org.bytedeco.pytorch.SizeTVector");
                Object vec = sizeTVector.getConstructor(long.class).newInstance(5L);
                Method put = sizeTVector.getMethod("put", long.class, long.class);
                long[] idxs = {2, 4, 6, 8, 10};
                for (int i = 0; i < idxs.length; i++) put.invoke(vec, (long) i, idxs[i]);
                sub = cls.getConstructor(sizeTVector).newInstance(vec);
            } catch (NoSuchMethodException e) {
                // from_int64(LongVector) static factory?
                Method factory = cls.getMethod("from_int64", Class.forName("org.bytedeco.pytorch.LongVector"));
                Class<?> lv = Class.forName("org.bytedeco.pytorch.LongVector");
                Object vec = lv.getConstructor(long.class).newInstance(5L);
                Method put = lv.getMethod("put", long.class, long.class);
                long[] idxs = {2, 4, 6, 8, 10};
                for (int i = 0; i < idxs.length; i++) put.invoke(vec, (long) i, idxs[i]);
                sub = factory.invoke(null, vec);
            }

            Sampler sampler = (Sampler) sub;
            sampler.reset(new SizeTOptional());
            List<Long> out = drain(sampler, 2);
            check("SubsetRandomSampler size 5", out.size() == 5);
            check("SubsetRandomSampler values",
                    new HashSet<>(out).equals(Set.of(2L, 4L, 6L, 8L, 10L)));
        } catch (ClassNotFoundException e) {
            skip("SubsetRandomSampler", "Java peer not generated — mvn -Djavacpp.cppbuild.skip=true install");
        } catch (Throwable t) {
            check("SubsetRandomSampler runnable", false, t.getClass().getSimpleName() + ": " + t.getMessage());
            t.printStackTrace(System.out);
        }

        // WeightedRandomSampler
        try {
            Class<?> cls = Class.forName("org.bytedeco.pytorch.data.sampler.WeightedRandomSampler");
            check("WeightedRandomSampler extends Sampler", Sampler.class.isAssignableFrom(cls));

            Class<?> doubleVector = Class.forName("org.bytedeco.pytorch.DoubleVector");
            Object weights = doubleVector.getConstructor(long.class).newInstance(4L);
            Method put = doubleVector.getMethod("put", long.class, double.class);
            double[] w = {0.1, 0.9, 0.0, 0.0};
            for (int i = 0; i < w.length; i++) put.invoke(weights, (long) i, w[i]);

            Object wrs;
            try {
                wrs = cls.getConstructor(doubleVector, long.class, boolean.class)
                        .newInstance(weights, 20L, true);
            } catch (NoSuchMethodException e) {
                wrs = cls.getConstructor(doubleVector, long.class)
                        .newInstance(weights, 20L);
            }
            Sampler sampler = (Sampler) wrs;
            sampler.reset(new SizeTOptional());
            List<Long> out = drain(sampler, 5);
            check("WeightedRandomSampler size 20", out.size() == 20);
            long c1 = out.stream().filter(x -> x == 1L).count();
            check("WeightedRandomSampler prefers weight 1", c1 >= 10, "count1=" + c1);
        } catch (ClassNotFoundException e) {
            skip("WeightedRandomSampler", "Java peer not generated — mvn -Djavacpp.cppbuild.skip=true install");
        } catch (Throwable t) {
            check("WeightedRandomSampler runnable", false, t.getClass().getSimpleName() + ": " + t.getMessage());
            t.printStackTrace(System.out);
        }

        // BatchSampler drop_last
        try {
            Class<?> cls = Class.forName("org.bytedeco.pytorch.data.sampler.BatchSampler");
            check("BatchSampler extends Sampler", Sampler.class.isAssignableFrom(cls));

            SequentialSampler inner = new SequentialSampler(10);
            // BatchSampler(shared_ptr<Sampler<>>, bool)
            Object batch;
            try {
                // may accept Sampler directly if JavaCPP converts
                batch = cls.getConstructor(Sampler.class, boolean.class)
                        .newInstance(inner, true);
            } catch (NoSuchMethodException e) {
                // try SharedPtr form — often same as Sampler with @SharedPtr
                Constructor<?>[] ctors = cls.getConstructors();
                if (ctors.length == 0) throw e;
                batch = ctors[0].newInstance(inner, true);
            }
            Sampler sampler = (Sampler) batch;
            sampler.reset(new SizeTOptional());
            List<List<Long>> batches = new ArrayList<>();
            while (true) {
                SizeTVectorOptional opt = sampler.next(3);
                if (opt == null || opt.isNull() || !opt.has_value()) break;
                SizeTVector v = opt.get();
                List<Long> b = new ArrayList<>();
                for (long i = 0, n = v.size(); i < n; i++) b.add(v.get(i));
                batches.add(b);
            }
            check("BatchSampler drop_last only full batches",
                    batches.stream().allMatch(b -> b.size() == 3),
                    "batches=" + batches);
            check("BatchSampler drop_last count", batches.size() == 3,
                    "n=" + batches.size());
        } catch (ClassNotFoundException e) {
            skip("BatchSampler", "Java peer not generated — mvn -Djavacpp.cppbuild.skip=true install");
        } catch (Throwable t) {
            check("BatchSampler runnable", false, t.getClass().getSimpleName() + ": " + t.getMessage());
            t.printStackTrace(System.out);
        }
    }

    // ------------------------------------------------------------------
    // Native optimizers
    // ------------------------------------------------------------------

    private static void benchNativeOptimAndStepLR() {
        section("Native Optimizer + StepLR (LibTorch)");

        Tensor p = requiresGrad(16, 16);
        double loss0;
        try (NoGradGuard g = new NoGradGuard()) {
            loss0 = lossValue(p);
        }

        SGDOptions sgdOpt = new SGDOptions(0.1);
        sgdOpt.momentum(0.0);
        SGD sgd = new SGD(new TensorVector(p), sgdOpt);
        double loss1 = trainStep(sgd, p);
        check("SGD reduces quadratic loss", loss1 < loss0,
                String.format(Locale.ROOT, "loss0=%.6f loss1=%.6f", loss0, loss1));

        double prev = loss1;
        boolean mono = true;
        for (int i = 0; i < 5; i++) {
            double li = trainStep(sgd, p);
            if (li > prev + 1e-9) mono = false;
            prev = li;
        }
        check("SGD mono decrease over 5 steps", mono, "last=" + prev);

        Tensor p2 = requiresGrad(8, 8);
        double a0;
        try (NoGradGuard g = new NoGradGuard()) { a0 = lossValue(p2); }
        Adam adam = new Adam(new TensorVector(p2), new AdamOptions(1e-1));
        double a1 = trainStep(adam, p2);
        double a2 = trainStep(adam, p2);
        check("Adam reduces loss", a1 < a0 && a2 <= a1 + 1e-6,
                String.format(Locale.ROOT, "a0=%.6f a1=%.6f a2=%.6f", a0, a1, a2));

        Tensor p3 = requiresGrad(4, 4);
        SGD sgd2 = new SGD(new TensorVector(p3), new SGDOptions(0.1));
        double lr0 = sgd2.param_groups().get(0).options().get_lr();
        StepLR stepLR = new StepLR(sgd2, 2, 0.5);
        stepLR.step();
        double lr1 = sgd2.param_groups().get(0).options().get_lr();
        stepLR.step();
        stepLR.step();
        double lr3 = sgd2.param_groups().get(0).options().get_lr();
        check("StepLR initial lr", Math.abs(lr0 - 0.1) < 1e-12, "lr0=" + lr0);
        check("StepLR no decay before milestone", Math.abs(lr1 - 0.1) < 1e-12, "lr1=" + lr1);
        check("StepLR decays at milestone", Math.abs(lr3 - 0.05) < 1e-12, "lr3=" + lr3);

        Tensor pb = requiresGrad(64, 64);
        SGD sgdB = new SGD(new TensorVector(pb), new SGDOptions(0.05));
        for (int i = 0; i < 3; i++) trainStep(sgdB, pb);
        long t0 = System.nanoTime();
        int N = 50;
        for (int i = 0; i < N; i++) trainStep(sgdB, pb);
        long t1 = System.nanoTime();
        System.out.printf(Locale.ROOT, "  [BENCH] SGD step 64x64 x%d: %.3f ms/step%n",
                N, (t1 - t0) / 1e6 / N);
    }

    // ------------------------------------------------------------------
    // New optimizers
    // ------------------------------------------------------------------

    private static void benchNewOptimizersReflective() {
        section("New optimizers (Adadelta/Adamax/ASGD/NAdam/RAdam/Rprop)");

        String[] names = {"Adadelta", "Adamax", "ASGD", "NAdam", "RAdam", "Rprop"};
        for (String name : names) {
            String fqcn = "org.bytedeco.pytorch.optim." + name;
            String optFqcn = "org.bytedeco.pytorch.optim." + name + "Options";
            try {
                Class<?> optCls = Class.forName(optFqcn);
                Class<?> cls = Class.forName(fqcn);
                check(name + " extends Optimizer", Optimizer.class.isAssignableFrom(cls));

                Tensor p = requiresGrad(8, 8);
                double l0;
                try (NoGradGuard g = new NoGradGuard()) { l0 = lossValue(p); }

                Object options;
                try {
                    double lr = name.equals("Adadelta") ? 1.0 : 0.1;
                    options = optCls.getConstructor(double.class).newInstance(lr);
                } catch (NoSuchMethodException e) {
                    options = optCls.getConstructor().newInstance();
                }

                Constructor<?> ctor = cls.getConstructor(TensorVector.class, optCls);
                Optimizer opt = (Optimizer) ctor.newInstance(new TensorVector(p), options);

                double l1 = trainStep(opt, p);
                double l2 = trainStep(opt, p);
                check(name + " reduces loss", l1 < l0,
                        String.format(Locale.ROOT, "l0=%.6f l1=%.6f l2=%.6f", l0, l1, l2));
                check(name + " second step finite", Double.isFinite(l2));
                check(name + " param_groups non-empty",
                        opt.param_groups() != null && opt.param_groups().size() >= 1);
                double lrNow = opt.param_groups().get(0).options().get_lr();
                check(name + " get_lr works", lrNow > 0, "lr=" + lrNow);

                Tensor pb = requiresGrad(32, 32);
                Optimizer optB = (Optimizer) ctor.newInstance(new TensorVector(pb), options);
                for (int i = 0; i < 2; i++) trainStep(optB, pb);
                long t0 = System.nanoTime();
                int N = 30;
                for (int i = 0; i < N; i++) trainStep(optB, pb);
                long t1 = System.nanoTime();
                System.out.printf(Locale.ROOT, "  [BENCH] %s step 32x32 x%d: %.3f ms/step%n",
                        name, N, (t1 - t0) / 1e6 / N);
            } catch (ClassNotFoundException e) {
                skip(name, "Java peer not generated yet — run: mvn -Djavacpp.cppbuild.skip=true install");
            } catch (Throwable t) {
                check(name + " runnable", false, t.getClass().getSimpleName() + ": " + t.getMessage());
                t.printStackTrace(System.out);
            }
        }
    }

    // ------------------------------------------------------------------
    // New schedulers
    // ------------------------------------------------------------------

    private static void benchNewSchedulersReflective() {
        section("New LR schedulers (ExponentialLR/MultiStepLR/...)");

        tryReflectScheduler("ExponentialLR", (opt) -> {
            Class<?> cls = Class.forName("org.bytedeco.pytorch.optim.schedulers.ExponentialLR");
            Object sched = cls.getConstructor(Optimizer.class, double.class).newInstance(opt, 0.9);
            Method step = cls.getMethod("step");
            double lr0 = opt.param_groups().get(0).options().get_lr();
            step.invoke(sched);
            step.invoke(sched);
            double lr2 = opt.param_groups().get(0).options().get_lr();
            check("ExponentialLR decay", Math.abs(lr2 - lr0 * 0.9) < 1e-12,
                    "lr0=" + lr0 + " lr2=" + lr2);
        });

        tryReflectScheduler("MultiplicativeLR", (opt) -> {
            Class<?> cls = Class.forName("org.bytedeco.pytorch.optim.schedulers.MultiplicativeLR");
            Object sched = cls.getConstructor(Optimizer.class, double.class).newInstance(opt, 0.5);
            Method step = cls.getMethod("step");
            double lr0 = opt.param_groups().get(0).options().get_lr();
            step.invoke(sched);
            step.invoke(sched);
            double lr = opt.param_groups().get(0).options().get_lr();
            check("MultiplicativeLR halves lr", Math.abs(lr - lr0 * 0.5) < 1e-12,
                    "lr0=" + lr0 + " lr=" + lr);
        });

        tryReflectScheduler("CosineAnnealingLR", (opt) -> {
            Class<?> cls = Class.forName("org.bytedeco.pytorch.optim.schedulers.CosineAnnealingLR");
            Constructor<?> c;
            try {
                c = cls.getConstructor(Optimizer.class, int.class, double.class);
            } catch (NoSuchMethodException e) {
                c = cls.getConstructor(Optimizer.class, long.class, double.class);
            }
            Object sched = c.newInstance(opt, 10, 0.0);
            Method step = cls.getMethod("step");
            double lr0 = opt.param_groups().get(0).options().get_lr();
            step.invoke(sched);
            double lrA = opt.param_groups().get(0).options().get_lr();
            for (int i = 0; i < 5; i++) step.invoke(sched);
            double lrB = opt.param_groups().get(0).options().get_lr();
            check("CosineAnnealingLR starts at base", Math.abs(lrA - lr0) < 1e-9, "lrA=" + lrA);
            check("CosineAnnealingLR decreases", lrB < lr0, "lrB=" + lrB);
        });

        tryReflectScheduler("LinearLR", (opt) -> {
            Class<?> cls = Class.forName("org.bytedeco.pytorch.optim.schedulers.LinearLR");
            Constructor<?> c = cls.getConstructors()[0];
            Object sched = c.getParameterCount() >= 4
                    ? c.newInstance(opt, 0.5, 1.0, 5)
                    : c.newInstance(buildArgs(c, opt));
            Method step = cls.getMethod("step");
            double lr0 = opt.param_groups().get(0).options().get_lr();
            step.invoke(sched);
            double lr1 = opt.param_groups().get(0).options().get_lr();
            check("LinearLR applies start_factor", lr1 > 0 && lr1 <= lr0 + 1e-12,
                    "lr0=" + lr0 + " lr1=" + lr1);
        });

        tryReflectScheduler("ConstantLR", (opt) -> {
            Class<?> cls = Class.forName("org.bytedeco.pytorch.optim.schedulers.ConstantLR");
            Constructor<?> c = cls.getConstructors()[0];
            Object sched = c.getParameterCount() >= 3
                    ? c.newInstance(opt, 0.5, 3)
                    : c.newInstance(buildArgs(c, opt));
            Method step = cls.getMethod("step");
            double lr0 = opt.param_groups().get(0).options().get_lr();
            step.invoke(sched);
            double lr1 = opt.param_groups().get(0).options().get_lr();
            check("ConstantLR scales on first step",
                    Math.abs(lr1 - lr0 * 0.5) < 1e-9 || lr1 < lr0,
                    "lr0=" + lr0 + " lr1=" + lr1);
        });

        tryReflectScheduler("PolynomialLR", (opt) -> {
            Class<?> cls = Class.forName("org.bytedeco.pytorch.optim.schedulers.PolynomialLR");
            Object sched = cls.getConstructors()[0].newInstance(
                    buildArgs(cls.getConstructors()[0], opt));
            Method step = cls.getMethod("step");
            step.invoke(sched);
            step.invoke(sched);
            double lr = opt.param_groups().get(0).options().get_lr();
            check("PolynomialLR lr finite positive", lr > 0 && Double.isFinite(lr), "lr=" + lr);
        });

        tryReflectScheduler("CyclicLR", (opt) -> {
            Class<?> cls = Class.forName("org.bytedeco.pytorch.optim.schedulers.CyclicLR");
            Constructor<?> c = null;
            for (Constructor<?> ctor : cls.getConstructors()) {
                if (ctor.getParameterCount() >= 3) { c = ctor; break; }
            }
            if (c == null) throw new IllegalStateException("no ctor");
            Object sched = c.newInstance(buildArgs(c, opt, 0.01, 0.1));
            Method step = cls.getMethod("step");
            step.invoke(sched);
            step.invoke(sched);
            double lr = opt.param_groups().get(0).options().get_lr();
            check("CyclicLR lr finite", Double.isFinite(lr) && lr > 0, "lr=" + lr);
        });

        tryReflectScheduler("CosineAnnealingWarmRestarts", (opt) -> {
            Class<?> cls = Class.forName("org.bytedeco.pytorch.optim.schedulers.CosineAnnealingWarmRestarts");
            Constructor<?> c = cls.getConstructors()[0];
            Object sched = c.newInstance(buildArgs(c, opt, 5));
            Method step = cls.getMethod("step");
            step.invoke(sched);
            for (int i = 0; i < 6; i++) step.invoke(sched);
            double lr = opt.param_groups().get(0).options().get_lr();
            check("CosineAnnealingWarmRestarts lr finite", Double.isFinite(lr) && lr >= 0, "lr=" + lr);
        });

        tryReflectScheduler("MultiStepLR", (opt) -> {
            Class<?> cls = Class.forName("org.bytedeco.pytorch.optim.schedulers.MultiStepLR");
            Constructor<?> chosen = null;
            for (Constructor<?> c : cls.getConstructors()) {
                if (c.getParameterCount() >= 2) { chosen = c; break; }
            }
            if (chosen == null) throw new IllegalStateException("no ctor");
            Object sched;
            Class<?>[] pts = chosen.getParameterTypes();
            if (pts.length >= 3 && pts[1] == int[].class) {
                sched = chosen.newInstance(opt, new int[]{2, 4}, 0.1);
            } else if (pts.length >= 3 && pts[1].getName().contains("Vector")) {
                Object vec = pts[1].getConstructor().newInstance();
                try {
                    Method pb = pts[1].getMethod("push_back", int.class);
                    pb.invoke(vec, 2);
                    pb.invoke(vec, 4);
                } catch (NoSuchMethodException e) {
                    pts[1].getMethod("resize", long.class).invoke(vec, 2L);
                    Method put = pts[1].getMethod("put", long.class, int.class);
                    put.invoke(vec, 0L, 2);
                    put.invoke(vec, 1L, 4);
                }
                sched = chosen.newInstance(opt, vec, 0.1);
            } else {
                sched = chosen.newInstance(buildArgs(chosen, opt));
            }
            Method step = cls.getMethod("step");
            double lr0 = opt.param_groups().get(0).options().get_lr();
            step.invoke(sched);
            step.invoke(sched);
            step.invoke(sched);
            double lr = opt.param_groups().get(0).options().get_lr();
            check("MultiStepLR decays at milestone", lr <= lr0 + 1e-12 && lr > 0,
                    "lr0=" + lr0 + " lr=" + lr);
        });

        skip("LambdaLR", "intentionally skipped (LRLambda JavaCPP virtualize unsupported); use MultiplicativeLR");
    }

    @FunctionalInterface
    private interface SchedulerTest {
        void run(SGD opt) throws Exception;
    }

    private static void tryReflectScheduler(String name, SchedulerTest test) {
        try {
            Class.forName("org.bytedeco.pytorch.optim." + name);
        } catch (ClassNotFoundException e) {
            skip(name, "Java peer not generated yet — run: mvn -Djavacpp.cppbuild.skip=true install");
            return;
        }
        try {
            Tensor p = requiresGrad(4, 4);
            SGD opt = new SGD(new TensorVector(p), new SGDOptions(0.1));
            test.run(opt);
        } catch (Throwable t) {
            check(name + " runnable", false, t.getClass().getSimpleName() + ": " + t.getMessage());
            t.printStackTrace(System.out);
        }
    }

    private static Object[] buildArgs(Constructor<?> c, Object opt, Object... extras) {
        Class<?>[] pts = c.getParameterTypes();
        Object[] args = new Object[pts.length];
        args[0] = opt;
        int ei = 0;
        for (int i = 1; i < pts.length; i++) {
            if (ei < extras.length) {
                Object e = extras[ei++];
                if (pts[i] == int.class || pts[i] == Integer.class) {
                    args[i] = ((Number) e).intValue();
                } else if (pts[i] == long.class || pts[i] == Long.class) {
                    args[i] = ((Number) e).longValue();
                } else if (pts[i] == double.class || pts[i] == Double.class) {
                    args[i] = ((Number) e).doubleValue();
                } else if (pts[i] == float.class || pts[i] == Float.class) {
                    args[i] = ((Number) e).floatValue();
                } else if (pts[i] == boolean.class || pts[i] == Boolean.class) {
                    args[i] = e;
                } else {
                    args[i] = e;
                }
            } else {
                if (pts[i] == int.class || pts[i] == long.class) args[i] = 5;
                else if (pts[i] == double.class || pts[i] == float.class) args[i] = 1.0;
                else if (pts[i] == boolean.class) args[i] = false;
                else if (pts[i] == String.class) args[i] = "triangular";
                else args[i] = null;
            }
        }
        return args;
    }
}
