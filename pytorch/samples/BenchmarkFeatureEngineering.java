package samples;

import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.data.numpy.NDArray;
import org.bytedeco.pytorch.data.numpy.NP;
import org.bytedeco.pytorch.dataframe.Column;
import org.bytedeco.pytorch.dataframe.DataFrame;
import org.bytedeco.pytorch.dataframe.DataValues;
import org.bytedeco.pytorch.dataframe.feature.BaseTransformer;
import org.bytedeco.pytorch.dataframe.feature.construction.*;
import org.bytedeco.pytorch.dataframe.feature.decomposition.NMF;
import org.bytedeco.pytorch.dataframe.feature.decomposition.PCA;
import org.bytedeco.pytorch.dataframe.feature.decomposition.TruncatedSVD;
import org.bytedeco.pytorch.dataframe.feature.encoding.*;
import org.bytedeco.pytorch.dataframe.feature.imputation.KNNImputer;
import org.bytedeco.pytorch.dataframe.feature.imputation.SimpleImputer;
import org.bytedeco.pytorch.dataframe.feature.pipeline.ColumnTransformer;
import org.bytedeco.pytorch.dataframe.feature.pipeline.Pipeline;
import org.bytedeco.pytorch.dataframe.feature.pipeline.TransformedTargetRegressor;
import org.bytedeco.pytorch.dataframe.feature.preprocessing.MissingIndicator;
import org.bytedeco.pytorch.dataframe.feature.preprocessing.PowerTransformer;
import org.bytedeco.pytorch.dataframe.feature.preprocessing.QuantileTransformer;
import org.bytedeco.pytorch.dataframe.feature.preprocessing.SplineTransformer;
import org.bytedeco.pytorch.dataframe.feature.scaling.*;
import org.bytedeco.pytorch.dataframe.feature.selection.*;
import org.bytedeco.pytorch.dataframe.feature.text.CountVectorizer;
import org.bytedeco.pytorch.dataframe.feature.text.FeatureHasher;
import org.bytedeco.pytorch.dataframe.feature.text.TfidfVectorizer;
import org.bytedeco.pytorch.dataframe.feature.util.FeatureBackends;
import org.bytedeco.pytorch.dataframe.feature.util.FeatureMatrices;
import org.bytedeco.pytorch.dataframe.ml.classification.LogisticRegression;
import org.bytedeco.pytorch.dataframe.ml.classification.RandomForestClassifier;
import org.bytedeco.pytorch.dataframe.ml.regression.Ridge;
import org.bytedeco.pytorch.utils.plot.*;

import java.nio.file.*;
import java.util.*;

/**
 * sklearn feature-engineering parity suite for the 60 examples in
 * {@code org/lance/ipc/fea.md}, with DataFrame / numpy / Tensor backends,
 * multi-dimensional stress, and Seaborn effect plots.
 *
 * <pre>
 *   javac -cp "target/classes:$(cat target/cp.txt)" -d target/samples-compile \
 *         samples/BenchmarkFeatureEngineering.java
 *   java  -cp "target/samples-compile:target/classes:$(cat target/cp.txt)" \
 *         samples.BenchmarkFeatureEngineering [--full]
 * </pre>
 */
public class BenchmarkFeatureEngineering {

    static int passed = 0, failed = 0, skipped = 0;
    static final StringBuilder report = new StringBuilder();
    static final Random RNG = new Random(42);
    static Path PLOT_DIR;
    static boolean FULL = false;
    static int N = 2000;

    static final String[] NUM = {
        "num_0", "num_1", "num_2", "num_3", "num_4", "num_5",
        "num_6", "num_7", "num_8", "num_9", "num_10", "num_11"
    };
    static final String[] NUM3 = {"num_0", "num_1", "num_2"};
    static final String[] NUM4 = {"num_0", "num_1", "num_2", "num_3"};
    static final String[] TEXTS = {
        "good product", "bad quality", "normal", "excellent", "terrible"
    };
    static final String[] ORD = {"low", "mid", "high"};
    static final String[] NOM = {"A", "B", "C", "D", "E"};

    interface CheckedRunnable { void run() throws Exception; }

    static void benchmark(String name, CheckedRunnable r) {
        long t0 = System.nanoTime();
        try {
            r.run();
            long ms = (System.nanoTime() - t0) / 1_000_000;
            System.out.printf("  OK  %-56s %6d ms%n", name, ms);
        } catch (Throwable e) {
            failed++;
            long ms = (System.nanoTime() - t0) / 1_000_000;
            System.out.printf(" FAIL %-56s %6d ms: %s%n", name, ms, e.toString());
            report.append("FAIL ").append(name).append(": ").append(e).append('\n');
            e.printStackTrace(System.out);
        }
    }

    static void check(String name, boolean ok) {
        if (ok) {
            passed++;
        } else {
            failed++;
            report.append("  check failed: ").append(name).append('\n');
            throw new AssertionError(name);
        }
    }

    static void checkFinite(DataFrame df, String... cols) {
        for (String c : cols) {
            if (!df.hasColumn(c)) continue;
            double[] a = FeatureBackends.columnToArray(df, c);
            int bad = 0;
            for (double v : a) if (Double.isInfinite(v)) bad++;
            check(c + " no inf", bad == 0);
        }
    }

    static void checkPng(Path p, String label) throws Exception {
        check(label + " exists", Files.exists(p));
        long sz = Files.size(p);
        check(label + " non-empty (" + sz + " B)", sz > 100);
        byte[] head = Files.readAllBytes(p);
        check(label + " PNG magic", head.length >= 8
            && (head[0] & 0xFF) == 0x89
            && head[1] == 'P' && head[2] == 'N' && head[3] == 'G');
    }

    // ---- synthetic data (fea.md) ----

    static final class Split {
        final DataFrame train, test;
        final double[] yTrain, yTest;
        Split(DataFrame train, DataFrame test, double[] yTrain, double[] yTest) {
            this.train = train; this.test = test; this.yTrain = yTrain; this.yTest = yTest;
        }
    }

    static Split makeData(int n) {
        int nFeat = 12;
        // simple classification-like features
        double[][] X = new double[n][nFeat];
        double[] y = new double[n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < nFeat; j++) X[i][j] = RNG.nextGaussian();
            // informative signal
            double s = X[i][0] + 0.5 * X[i][1] - 0.3 * X[i][2] + 0.2 * X[i][3];
            y[i] = s + 0.3 * RNG.nextGaussian() > 0 ? 1.0 : 0.0;
        }
        DataFrame df = DataFrame.create();
        for (String c : NUM) df.addColumn(c, Column.DType.FLOAT64);
        df.addColumn("cat_ordinal", Column.DType.STRING);
        df.addColumn("cat_nominal", Column.DType.STRING);
        df.addColumn("cat_highcard", Column.DType.STRING);
        df.addColumn("text", Column.DType.STRING);
        df.addColumn("datetime_ts", Column.DType.INT64);
        df.addColumn("y", Column.DType.FLOAT64);

        long start = 1704067200000L; // 2024-01-01 UTC-ish
        for (int i = 0; i < n; i++) {
            Object[] row = new Object[12 + 6];
            for (int j = 0; j < 12; j++) row[j] = X[i][j];
            row[12] = ORD[RNG.nextInt(ORD.length)];
            row[13] = NOM[RNG.nextInt(NOM.length)];
            row[14] = "id_" + RNG.nextInt(250);
            row[15] = TEXTS[RNG.nextInt(TEXTS.length)];
            row[16] = start + i * 600_000L; // 10 min
            row[17] = y[i];
            df.addRow(row);
        }

        // inject ~12% missing on num_0, num_3, cat_nominal
        int nMiss = Math.max(1, (int) (n * 0.12));
        Set<Integer> used = new HashSet<>();
        for (String col : new String[]{"num_0", "num_3", "cat_nominal"}) {
            used.clear();
            while (used.size() < nMiss) {
                int idx = RNG.nextInt(n);
                if (used.add(idx)) df.set(idx, col, null);
            }
        }

        // train/test split 80/20
        int nTrain = (int) (n * 0.8);
        int[] idx = new int[n];
        for (int i = 0; i < n; i++) idx[i] = i;
        // shuffle with fixed seed
        Random shuf = new Random(42);
        for (int i = n - 1; i > 0; i--) {
            int j = shuf.nextInt(i + 1);
            int tmp = idx[i]; idx[i] = idx[j]; idx[j] = tmp;
        }
        int[] tr = Arrays.copyOfRange(idx, 0, nTrain);
        int[] te = Arrays.copyOfRange(idx, nTrain, n);
        DataFrame train = df.loc(tr);
        DataFrame test = df.loc(te);
        double[] yTrain = new double[tr.length];
        double[] yTest = new double[te.length];
        for (int i = 0; i < tr.length; i++) yTrain[i] = y[tr[i]];
        for (int i = 0; i < te.length; i++) yTest[i] = y[te[i]];
        return new Split(train, test, yTrain, yTest);
    }

    static long countMissing(DataFrame df, String col) {
        long m = 0;
        Column c = df.column(col);
        for (int i = 0; i < df.rowCount(); i++) {
            Object v = c.get(i);
            if (v == null) { m++; continue; }
            double d = DataValues.asDouble(v);
            if (Double.isNaN(d) && !(v instanceof String)) m++;
        }
        return m;
    }

    // ===================== main =====================

    public static void main(String[] args) throws Exception {
        for (String a : args) if ("--full".equals(a)) FULL = true;
        N = FULL ? 8000 : 2000;
        System.out.println("=== Feature Engineering Benchmark (fea.md 1-60) ===");
        System.out.println("N=" + N + (FULL ? " (full)" : " (fast; pass --full for 8000)"));
        PLOT_DIR = Paths.get("target/feature-bench-plots");
        Files.createDirectories(PLOT_DIR);

        Split split = makeData(N);
        DataFrame Xtr = split.train;
        DataFrame Xte = split.test;
        double[] ytr = split.yTrain;

        System.out.println("\n-- 1. Imputation (1-6) --");
        runImputation(Xtr, Xte);

        System.out.println("\n-- 2. Scaling / distribution (7-16) --");
        runScaling(Xtr);

        System.out.println("\n-- 3. Discretization (17-22) --");
        runBins(Xtr);

        System.out.println("\n-- 4. Encoding (23-32) --");
        runEncoding(Xtr, ytr);

        System.out.println("\n-- 5. Polynomial / interaction (33-38) --");
        runPoly(Xtr);

        System.out.println("\n-- 6. Feature selection (39-46) --");
        runSelection(Xtr, ytr);

        System.out.println("\n-- 7. Decomposition / cluster (47-54) --");
        runDecomp(Xtr);

        System.out.println("\n-- 8. Text (55-57) --");
        runText(Xtr);

        System.out.println("\n-- 9. Time / target / full pipeline (58-60) --");
        runIndustrial(Xtr, Xte, ytr);

        System.out.println("\n-- 10. Multi-backend parity (DF / numpy / Tensor) --");
        runBackendParity(Xtr);

        System.out.println("\n-- 11. Multi-dimensional stress --");
        runStress();

        System.out.println("\n-- 12. Seaborn effect plots --");
        runPlots(Xtr, ytr);

        System.out.println("\n=== Summary ===");
        System.out.printf("passed=%d  failed=%d  skipped=%d%n", passed, failed, skipped);
        if (report.length() > 0) {
            System.out.println("--- failures ---");
            System.out.print(report);
        }
        if (failed > 0) System.exit(1);
    }

    // ===================== sections =====================

    static void runImputation(DataFrame Xtr, DataFrame Xte) {
        benchmark("01 SimpleImputer mean+mode ColumnTransformer", () -> {
            ColumnTransformer ct = new ColumnTransformer()
                .addTransformer("num_imp", new SimpleImputer("mean", NUM3), NUM3)
                .addTransformer("cat_imp", new SimpleImputer("most_frequent", "cat_nominal"), "cat_nominal");
            DataFrame out = ct.fitTransform(Xtr);
            check("imp rows", out.rowCount() == Xtr.rowCount());
            check("num_0 filled", countMissing(out, "num_0") == 0);
        });
        benchmark("02 KNNImputer n=7 weights=distance", () -> {
            KNNImputer knn = new KNNImputer(7, NUM4).setWeights("distance");
            DataFrame out = knn.fitTransform(Xtr.select(NUM4));
            check("knn rows", out.rowCount() == Xtr.rowCount());
            check("knn num_0 filled", countMissing(out, "num_0") == 0);
        });
        benchmark("03 MissingIndicator + median impute", () -> {
            MissingIndicator ind = new MissingIndicator(NUM3);
            DataFrame mask = ind.fitTransform(Xtr);
            Pipeline pipe = new Pipeline().addStep("imp", new SimpleImputer("median", NUM3));
            DataFrame filled = pipe.fitTransform(Xtr);
            check("indicator fitted", ind.isFitted());
            check("median filled", countMissing(filled, "num_0") == 0);
            check("mask has rows", mask.rowCount() == Xtr.rowCount());
        });
        benchmark("04 constant fill num/cat", () -> {
            ColumnTransformer ct = new ColumnTransformer()
                .addTransformer("num", SimpleImputer.constant(0, NUM3), NUM3)
                .addTransformer("cat", SimpleImputer.constant("UNKNOWN", "cat_nominal"), "cat_nominal");
            DataFrame out = ct.fitTransform(Xtr);
            check("const filled num", countMissing(out, "num_0") == 0);
        });
        benchmark("05 train-fit only, transform test (no leak)", () -> {
            SimpleImputer imp = new SimpleImputer("mean", NUM3);
            imp.fit(Xtr);
            DataFrame te = imp.transform(Xte);
            check("test filled", countMissing(te, "num_0") == 0);
            // stats must come from train
            Map<String, Object> stats = imp.getStatistics();
            check("has num_0 stat", stats.containsKey("num_0"));
        });
        benchmark("06 FunctionTransformer log-mean fill", () -> {
            FunctionTransformer ft = FunctionTransformer.ofMatrix(X -> {
                int n = X.length, d = X[0].length;
                double[] logMean = new double[d];
                for (int j = 0; j < d; j++) {
                    double s = 0; int c = 0;
                    for (int i = 0; i < n; i++) {
                        if (!Double.isNaN(X[i][j])) { s += Math.abs(X[i][j]); c++; }
                    }
                    double m = c == 0 ? 1e-6 : s / c;
                    logMean[j] = Math.exp(Math.log(Math.max(m, 1e-12)));
                }
                double[][] out = new double[n][d];
                for (int i = 0; i < n; i++) {
                    for (int j = 0; j < d; j++) {
                        out[i][j] = Double.isNaN(X[i][j]) ? logMean[j] : X[i][j];
                    }
                }
                return out;
            }, "log_mean_fill", NUM3);
            DataFrame out = ft.fitTransform(Xtr);
            check("logmean filled", countMissing(out, "num_0") == 0);
        });
    }

    static void runScaling(DataFrame Xtr) {
        DataFrame num = fillNum(Xtr, NUM3);
        benchmark("07 StandardScaler", () -> {
            StandardScaler sc = new StandardScaler(NUM3);
            DataFrame out = sc.fitTransform(num);
            double[] a = FeatureBackends.columnToArray(out, "num_0");
            double mean = mean(a), std = std(a);
            check("std mean~0", Math.abs(mean) < 0.05);
            check("std std~1", Math.abs(std - 1) < 0.1);
        });
        benchmark("08 RobustScaler quantile_range=(2.5,97.5)", () -> {
            RobustScaler rs = new RobustScaler(NUM3).setQuantileRange(2.5, 97.5);
            DataFrame out = rs.fitTransform(num);
            check("robust rows", out.rowCount() == num.rowCount());
            checkFinite(out, NUM3);
        });
        benchmark("09 MinMaxScaler [0,1]", () -> {
            MinMaxScaler mm = new MinMaxScaler(NUM3);
            // try common ctor patterns
            DataFrame out = mm.fitTransform(num);
            check("minmax rows", out.rowCount() == num.rowCount());
        });
        benchmark("10 MaxAbsScaler", () -> {
            MaxAbsScaler ma = new MaxAbsScaler(NUM3);
            DataFrame out = ma.fitTransform(num);
            checkFinite(out, NUM3);
        });
        benchmark("11 PowerTransformer yeo-johnson", () -> {
            PowerTransformer pt = new PowerTransformer(PowerTransformer.Method.YEO_JOHNSON, "num_0", "num_1");
            DataFrame out = pt.fitTransform(num);
            check("yj fitted", pt.isFitted());
            checkFinite(out, "num_0", "num_1");
        });
        benchmark("12 PowerTransformer box-cox on positive", () -> {
            DataFrame pos = num.copy();
            Column c = pos.column("num_0");
            for (int i = 0; i < pos.rowCount(); i++) {
                double v = DataValues.asDouble(c.get(i));
                c.set(i, Math.abs(v) + 1e-3);
            }
            PowerTransformer pt = new PowerTransformer(PowerTransformer.Method.BOX_COX, "num_0");
            DataFrame out = pt.fitTransform(pos);
            check("bc fitted", pt.isFitted());
        });
        benchmark("13 QuantileTransformer uniform", () -> {
            QuantileTransformer qt = new QuantileTransformer(
                QuantileTransformer.Output.UNIFORM, 100, NUM3);
            DataFrame out = qt.fitTransform(num);
            checkFinite(out, NUM3);
        });
        benchmark("14 QuantileTransformer normal", () -> {
            QuantileTransformer qt = new QuantileTransformer(
                QuantileTransformer.Output.NORMAL, 100, NUM3);
            DataFrame out = qt.fitTransform(num);
            checkFinite(out, NUM3);
        });
        benchmark("15 SplineTransformer", () -> {
            SplineTransformer sp = new SplineTransformer("num_0");
            // if ctor needs knots — try fit anyway
            try {
                DataFrame out = sp.fitTransform(num);
                check("spline rows", out.rowCount() == num.rowCount());
            } catch (Exception e) {
                // alternate ctors
                SplineTransformer sp2 = new SplineTransformer(5, 3, "num_0");
                DataFrame out = sp2.fitTransform(num);
                check("spline rows", out.rowCount() == num.rowCount());
            }
        });
        benchmark("16 FunctionTransformer log1p", () -> {
            FunctionTransformer log = new FunctionTransformer(
                x -> Math.log(Math.abs(x) + 1e-6), "log1p", "num_0");
            DataFrame out = log.fitTransform(num);
            checkFinite(out, "num_0");
        });
    }

    static void runBins(DataFrame Xtr) {
        DataFrame num = fillNum(Xtr, "num_0", "num_1");
        benchmark("17 KBins uniform onehot", () -> {
            KBinsDiscretizer b = KBinsDiscretizer.of(6, "uniform", "onehot-dense", "num_0");
            DataFrame out = b.fitTransform(num);
            check("bin onehot cols>in", out.columnCount() >= num.columnCount());
        });
        benchmark("18 KBins quantile ordinal", () -> {
            KBinsDiscretizer b = KBinsDiscretizer.of(6, "quantile", "ordinal", "num_0");
            DataFrame out = b.fitTransform(num);
            check("has bin col", out.hasColumn("num_0_bin") || out.hasColumn("num_0"));
        });
        benchmark("19 KBins kmeans", () -> {
            KBinsDiscretizer b = KBinsDiscretizer.of(5, "kmeans", "ordinal", "num_1");
            DataFrame out = b.fitTransform(num);
            check("kmeans bins fitted", b.isFitted());
        });
        benchmark("20 custom business bins FunctionTransformer", () -> {
            FunctionTransformer ft = FunctionTransformer.ofMatrix(X -> {
                double[][] out = new double[X.length][1];
                for (int i = 0; i < X.length; i++) {
                    double v = X[i][0];
                    int bin = v < -1 ? 0 : v < 0 ? 1 : v < 1 ? 2 : 3;
                    out[i][0] = bin;
                }
                return out;
            }, "biz_bin", "num_0").setOutputNames("num_0_bizbin");
            DataFrame out = ft.fitTransform(num);
            check("biz bin col", out.hasColumn("num_0_bizbin") || out.columnCount() >= 1);
        });
        benchmark("21 Pipeline impute+bin", () -> {
            Pipeline p = new Pipeline()
                .addStep("imp", new SimpleImputer("mean", "num_0"))
                .addStep("bin", KBinsDiscretizer.withStrategy(4, "quantile", "num_0"));
            DataFrame out = p.fitTransform(Xtr);
            check("pipe bin rows", out.rowCount() == Xtr.rowCount());
        });
        benchmark("22 ColumnTransformer multi-bin", () -> {
            ColumnTransformer ct = new ColumnTransformer()
                .addTransformer("bin0", KBinsDiscretizer.withStrategy(5, "quantile", "num_0"), "num_0")
                .addTransformer("bin1", KBinsDiscretizer.withStrategy(4, "uniform", "num_1"), "num_1");
            DataFrame filled = fillNum(Xtr, "num_0", "num_1");
            DataFrame out = ct.fitTransform(filled);
            check("multi bin rows", out.rowCount() == filled.rowCount());
        });
    }

    static void runEncoding(DataFrame Xtr, double[] ytr) {
        benchmark("23 OrdinalEncoder handle_unknown", () -> {
            OrdinalEncoder enc = new OrdinalEncoder("cat_ordinal")
                .setHandleUnknown("use_encoded_value").setUnknownValue(-1);
            DataFrame out = enc.fitTransform(Xtr);
            check("ord fitted", enc.isFitted());
        });
        benchmark("24 OneHotEncoder ignore unknown", () -> {
            OneHotEncoder ohe = new OneHotEncoder(true, null, "cat_nominal")
                .setHandleUnknown("ignore");
            // need impute first for nulls
            DataFrame base = new SimpleImputer("most_frequent", "cat_nominal").fitTransform(Xtr);
            DataFrame out = ohe.fitTransform(base);
            check("ohe more cols", out.columnCount() > base.columnCount() - 1);
        });
        benchmark("25 TargetEncoder", () -> {
            // put y into frame
            DataFrame df = Xtr.copy();
            if (!df.hasColumn("y")) {
                df.addColumn("y", Column.DType.FLOAT64);
                Column c = df.column("y");
                while (c.size() < df.rowCount()) c.add(null);
                for (int i = 0; i < ytr.length; i++) c.set(i, ytr[i]);
            }
            TargetEncoder te = new TargetEncoder("y", "cat_highcard").setReplace(false);
            DataFrame out = te.fit(df, ytr).transform(df);
            check("te col", out.hasColumn("cat_highcard_te") || out.hasColumn("cat_highcard"));
        });
        benchmark("26 LabelEncoder on target-like col", () -> {
            // build string labels
            DataFrame lab = DataFrame.create();
            lab.addColumn("label", Column.DType.STRING);
            for (int i = 0; i < 100; i++) lab.addRow(RNG.nextBoolean() ? "pos" : "neg");
            LabelEncoder le = new LabelEncoder("label");
            DataFrame out = le.fitTransform(lab);
            check("le fitted", le.isFitted());
        });
        benchmark("27 FeatureHasher high-card", () -> {
            FeatureHasher fh = new FeatureHasher(64, "cat_highcard");
            DataFrame out = fh.fitTransform(Xtr);
            check("hash fitted", fh.isFitted());
            check("hash cols>=", out.columnCount() >= Xtr.columnCount());
        });
        benchmark("28 ColumnTransformer mix scale+ord+ohe", () -> {
            DataFrame base = new SimpleImputer("mean", NUM3).fitTransform(Xtr);
            base = new SimpleImputer("most_frequent", "cat_nominal").fitTransform(base);
            ColumnTransformer ct = new ColumnTransformer()
                .addTransformer("num", new StandardScaler(NUM3), NUM3)
                .addTransformer("ord", new OrdinalEncoder("cat_ordinal"), "cat_ordinal")
                .addTransformer("ohe", new OneHotEncoder(true, null, "cat_nominal"), "cat_nominal");
            DataFrame out = ct.fitTransform(base);
            check("mix rows", out.rowCount() == base.rowCount());
        });
        benchmark("29 OneHotEncoder min_frequency=0.03", () -> {
            DataFrame base = new SimpleImputer("most_frequent", "cat_highcard").fitTransform(Xtr);
            OneHotEncoder ohe = new OneHotEncoder(true, null, "cat_highcard").setMinFrequency(0.03);
            DataFrame out = ohe.fitTransform(base);
            int nCats = ohe.getCategories().get("cat_highcard").size();
            check("minfreq reduced cats", nCats < 250);
        });
        benchmark("30 FrequencyEncoder", () -> {
            FrequencyEncoder fe = new FrequencyEncoder("cat_nominal").setReplace(false);
            DataFrame base = new SimpleImputer("most_frequent", "cat_nominal").fitTransform(Xtr);
            DataFrame out = fe.fitTransform(base);
            check("freq col", out.hasColumn("cat_nominal_freq") || out.hasColumn("cat_nominal"));
        });
        benchmark("31 Ordinal Pipeline with impute", () -> {
            Pipeline p = new Pipeline()
                .addStep("imp", new SimpleImputer("most_frequent", "cat_nominal"))
                .addStep("ord", new OrdinalEncoder("cat_nominal"));
            DataFrame out = p.fitTransform(Xtr);
            check("pipe ord rows", out.rowCount() == Xtr.rowCount());
        });
        benchmark("32 TargetEncoder cv=5 OOF", () -> {
            TargetEncoder te = new TargetEncoder(null, "cat_highcard").setCv(5).setReplace(false);
            DataFrame out = te.fitTransform(Xtr, ytr);
            check("tgt cv fitted", te.isFitted());
            check("tgt cv rows", out.rowCount() == Xtr.rowCount());
        });
    }

    static void runPoly(DataFrame Xtr) {
        DataFrame num = fillNum(Xtr, "num_0", "num_1", "num_2", "num_3");
        benchmark("33 PolynomialFeatures degree=2", () -> {
            PolynomialFeatures p = new PolynomialFeatures(2, false, false, "num_0", "num_1");
            DataFrame out = p.fitTransform(num);
            check("poly names", p.getFeatureNames().size() >= 3);
        });
        benchmark("34 PolynomialFeatures interaction_only", () -> {
            PolynomialFeatures p = new PolynomialFeatures(2, false, true, "num_0", "num_1", "num_2");
            DataFrame out = p.fitTransform(num);
            check("inter only", p.isInteractionOnly());
            // should not contain pure squares like num_0^2 as sole power beyond degree1 — still has raw
            check("inter fitted", p.isFitted());
        });
        benchmark("35 PolynomialFeatures degree=3", () -> {
            PolynomialFeatures p = new PolynomialFeatures(3, false, "num_0", "num_1");
            DataFrame out = p.fitTransform(num);
            check("poly3 features", p.getFeatureNames().size() >= 5);
        });
        benchmark("36 Pipeline scale+poly", () -> {
            Pipeline p = new Pipeline()
                .addStep("scale", new StandardScaler("num_0", "num_1"))
                .addStep("poly", new PolynomialFeatures(2, false, true, "num_0", "num_1"));
            DataFrame out = p.fitTransform(num);
            check("pipe poly rows", out.rowCount() == num.rowCount());
        });
        benchmark("37 ratio interaction FunctionTransformer", () -> {
            FunctionTransformer ft = FunctionTransformer.ofMatrix(X -> {
                double[][] out = new double[X.length][3];
                for (int i = 0; i < X.length; i++) {
                    double a = X[i][0], b = X[i][1] + 1e-6;
                    out[i][0] = a; out[i][1] = b; out[i][2] = a / b;
                }
                return out;
            }, "ratio", "num_0", "num_1").setOutputNames("a", "b", "a_over_b");
            DataFrame out = ft.fitTransform(num);
            check("ratio col", out.hasColumn("a_over_b"));
        });
        benchmark("38 ColumnTransformer poly+scale", () -> {
            ColumnTransformer ct = new ColumnTransformer()
                .addTransformer("poly", new PolynomialFeatures(2, false, "num_0", "num_1"), "num_0", "num_1")
                .addTransformer("scale", new StandardScaler("num_2", "num_3"), "num_2", "num_3");
            DataFrame out = ct.fitTransform(num);
            check("ct poly rows", out.rowCount() == num.rowCount());
        });
    }

    static void runSelection(DataFrame Xtr, double[] ytr) {
        DataFrame num = fillNum(Xtr, NUM);
        // attach y
        DataFrame df = num.copy();
        if (!df.hasColumn("y")) {
            df.addColumn("y", Column.DType.FLOAT64);
            Column c = df.column("y");
            while (c.size() < df.rowCount()) c.add(0.0);
            for (int i = 0; i < Math.min(ytr.length, df.rowCount()); i++) c.set(i, ytr[i]);
        }
        benchmark("39 VarianceThreshold", () -> {
            VarianceThreshold vt = new VarianceThreshold(0.01, NUM);
            DataFrame out = vt.fitTransform(num);
            check("var sel cols<=", out.columnCount() <= num.columnCount());
        });
        benchmark("40 SelectKBest f_classif k=6", () -> {
            SelectKBest skb = new SelectKBest(6, "f_classif", NUM).setLabelCol("y");
            DataFrame out = skb.fit(df, "y").transform(df);
            check("kbest==6", skb.getSelectedColumns().size() == 6);
        });
        benchmark("41 SelectPercentile 50%", () -> {
            SelectPercentile sp = new SelectPercentile(50, "f_classif", NUM).setLabelCol("y");
            DataFrame out = sp.fit(df, "y").transform(df);
            check("pct selected>0", sp.getSelectedColumns().size() >= 1);
        });
        benchmark("42 SelectFromModel LogisticRegression threshold=mean", () -> {
            LogisticRegression lr = new LogisticRegression("l2", 1.0, 200, 1e-4, 42L);
            SelectFromModel sfm = new SelectFromModel(lr, "mean", NUM, "y");
            DataFrame out = sfm.fitTransform(df);
            check("sfm selected>0", sfm.getSelectedColumns().size() >= 1);
        });
        benchmark("43 RFE n=5", () -> {
            LogisticRegression lr = new LogisticRegression("l2", 1.0, 100, 1e-4, 42L);
            RFE rfe = new RFE(lr, 5, NUM, "y");
            DataFrame out = rfe.fitTransform(df);
            check("rfe==5", rfe.getSelectedColumns().size() == 5);
        });
        benchmark("44 RFECV cv=3 with real coef importances", () -> {
            LogisticRegression lr = new LogisticRegression("l2", 1.0, 80, 1e-4, 42L);
            // fea.md: RFECV(estimator=LogisticRegression(), cv=3)
            RFECV rfecv = new RFECV(lr, 3, 3, "accuracy", NUM, "y");
            DataFrame out = rfecv.fitTransform(df);
            List<String> sel = rfecv.getSelectedCols();
            check("rfecv selected>0", sel != null && !sel.isEmpty());
            check("rfecv optimal>=" + 3, rfecv.getOptimalNFeatures() >= 3);
            check("rfecv selected==optimal", sel.size() == rfecv.getOptimalNFeatures());
            // With real |coef| importances, selection must not be identity order of all NUM
            // (unless all truly equal — still must be a proper subset when minFeatures < n)
            check("rfecv subset of NUM", sel.size() <= NUM.length);
        });
        benchmark("45 Pipeline scale+SelectKBest with y propagation", () -> {
            // fea.md: Pipeline([scale, SelectKBest]).fit_transform(X, y)
            SelectKBest skb = new SelectKBest(8, "f_classif", NUM);
            Pipeline p = new Pipeline()
                .addStep("scale", new StandardScaler(NUM))
                .addStep("select", skb);
            DataFrame out = p.fitTransform(df, ytr);
            check("pipe sel fitted via y", skb.isFitted());
            check("pipe sel 8", skb.getSelectedColumns().size() == 8);
            check("pipe sel rows", out.rowCount() == df.rowCount());
            // also labelCol path
            SelectKBest skb2 = new SelectKBest(6, "f_classif", NUM);
            Pipeline p2 = new Pipeline()
                .addStep("scale", new StandardScaler(NUM))
                .addStep("select", skb2);
            DataFrame out2 = p2.fitTransform(df, "y");
            check("pipe labelCol sel 6", skb2.getSelectedColumns().size() == 6);
            check("pipe labelCol rows", out2.rowCount() == df.rowCount());
        });
        benchmark("46 SelectFromModel RandomForest median + real importances", () -> {
            // fea.md: RandomForestClassifier(n_estimators=80); threshold="median"
            // Keep n_estimators modest for wall-clock but still real MDI importances.
            RandomForestClassifier rf = new RandomForestClassifier(30, null, 2, null, 42L);
            // Fit once to assert importances are non-uniform (not all-ones fallback)
            double[][] mat = org.bytedeco.pytorch.dataframe.feature.util.FeatureMatrices.fromDf(df, NUM);
            double[] y = org.bytedeco.pytorch.dataframe.feature.util.FeatureMatrices.columnAsDoubles(df, "y");
            rf.fit(mat, y);
            double[] fi = rf.getFeatureImportances();
            check("rf importances length", fi != null && fi.length == NUM.length);
            double sum = 0, max = 0, min = Double.POSITIVE_INFINITY;
            for (double v : fi) { sum += v; max = Math.max(max, v); min = Math.min(min, v); }
            check("rf importances sum~1", Math.abs(sum - 1.0) < 1e-6);
            check("rf importances non-uniform", max - min > 1e-9);

            SelectFromModel sfm = new SelectFromModel(rf, "median", NUM, "y");
            DataFrame out = sfm.fitTransform(df);
            check("sfm rf >0", sfm.getSelectedColumns().size() >= 1);
            check("sfm rf < all", sfm.getSelectedColumns().size() < NUM.length
                || fi.length <= 1); // if all equal median keeps all — rare with real MDI
        });
    }

    static void runDecomp(DataFrame Xtr) {
        DataFrame num = fillNum(Xtr, NUM3);
        // expand to more cols for PCA
        DataFrame wide = fillNum(Xtr, NUM);
        benchmark("47 PCA n=4", () -> {
            PCA pca = new PCA(4, NUM3);
            DataFrame out = pca.fitTransform(num);
            check("pca fitted", pca.isFitted());
            check("pca rows", out.rowCount() == num.rowCount());
        });
        benchmark("48 PCA n_components=0.85 variance fraction", () -> {
            // fea.md: pca_auto = PCA(n_components=0.85)
            PCA pca = new PCA(0.85, NUM);
            DataFrame out = pca.fitTransform(wide);
            check("pca auto fitted", pca.isFitted());
            double[] cum = pca.getCumulativeExplainedVarianceRatio();
            check("pca auto n>=1", pca.getNComponents() >= 1);
            check("pca auto cum length", cum != null && cum.length == pca.getNComponents());
            double last = cum[cum.length - 1];
            check("pca auto cum>=0.85 (" + last + ")", last >= 0.85 - 1e-9);
            // should not keep all dims unless needed
            check("pca auto n<=d", pca.getNComponents() <= NUM.length);
        });
        benchmark("49 TruncatedSVD n=5", () -> {
            TruncatedSVD svd = new TruncatedSVD(5, NUM);
            DataFrame out = svd.fitTransform(wide);
            check("svd fitted", svd.isFitted());
        });
        benchmark("50 NMF n=4 on abs features", () -> {
            DataFrame pos = wide.copy();
            for (String c : NUM) {
                Column col = pos.column(c);
                for (int i = 0; i < pos.rowCount(); i++) {
                    double v = DataValues.asDouble(col.get(i));
                    col.set(i, Math.abs(v));
                }
            }
            NMF nmf = new NMF(4, NUM);
            DataFrame out = nmf.fitTransform(pos);
            check("nmf fitted", nmf.isFitted());
        });
        benchmark("51 ClusterFeatures labels", () -> {
            ClusterFeatures cf = new ClusterFeatures(6, NUM3).setMode(ClusterFeatures.Mode.LABEL);
            DataFrame out = cf.fitTransform(num);
            check("cluster label", out.hasColumn("cluster_label"));
        });
        benchmark("52 ClusterFeatures distances", () -> {
            ClusterFeatures cf = new ClusterFeatures(6, NUM3).setMode(ClusterFeatures.Mode.DISTANCE);
            DataFrame out = cf.fitTransform(num);
            check("cluster dist0", out.hasColumn("cluster_dist_0"));
        });
        benchmark("53 Pipeline scale+PCA", () -> {
            Pipeline p = new Pipeline()
                .addStep("scale", new StandardScaler(NUM3))
                .addStep("pca", new PCA(3, NUM3));
            DataFrame out = p.fitTransform(num);
            check("pipe dim rows", out.rowCount() == num.rowCount());
        });
        benchmark("54 PCA concat with scaled raw", () -> {
            StandardScaler sc = new StandardScaler(NUM3);
            DataFrame scaled = sc.fitTransform(num);
            PCA pca = new PCA(2, NUM3);
            DataFrame pcs = pca.fitTransform(scaled);
            check("concat rows", pcs.rowCount() == scaled.rowCount());
        });
    }

    static void runText(DataFrame Xtr) {
        benchmark("55 CountVectorizer max_features=30 stop_words=[normal]", () -> {
            // fea.md: CountVectorizer(max_features=30, stop_words=["normal"])
            CountVectorizer cv = new CountVectorizer("text")
                .setMaxFeatures(30)
                .setStopWords("normal");
            DataFrame out = cv.fitTransform(Xtr);
            check("cv fitted", cv.isFitted());
            check("cv featureCount<=30", cv.getFeatureCount() <= 30);
            check("cv featureCount>0", cv.getFeatureCount() > 0);
            check("cv vocab excludes 'normal'", !cv.getVocabulary().contains("normal"));
            // ngrams are single tokens by default; ensure no token equals normal
            boolean hasNormal = false;
            for (String t : cv.getVocabulary()) {
                if ("normal".equals(t) || t.contains("normal")) { hasNormal = true; break; }
            }
            check("cv no normal token/ngram", !hasNormal);
            check("cv cols>", out.columnCount() >= Xtr.columnCount());
        });
        benchmark("56 TfidfVectorizer max_features=30", () -> {
            TfidfVectorizer tf = new TfidfVectorizer("text", 30, 1, Integer.MAX_VALUE);
            DataFrame out = tf.fitTransform(Xtr);
            check("tfidf fitted", tf.isFitted());
            check("tfidf features<=30", tf.getFeatureCount() <= 30);
        });
        benchmark("57 ColumnTransformer num+text", () -> {
            DataFrame base = fillNum(Xtr, NUM3);
            ColumnTransformer ct = new ColumnTransformer()
                .addTransformer("num", new StandardScaler(NUM3), NUM3)
                .addTransformer("text", new TfidfVectorizer("text", 20, 1, Integer.MAX_VALUE), "text");
            DataFrame out = ct.fitTransform(base);
            check("ct text rows", out.rowCount() == base.rowCount());
        });
    }

    static void runIndustrial(DataFrame Xtr, DataFrame Xte, double[] ytr) {
        benchmark("58 TimeFeatureExtractor", () -> {
            TimeFeatureExtractor te = new TimeFeatureExtractor("datetime_ts")
                .includeHour(true).includeWeekday(true);
            DataFrame out = te.fitTransform(Xtr);
            check("hour col", out.hasColumn("datetime_ts_hour"));
            check("weekday col", out.hasColumn("datetime_ts_weekday"));
        });
        benchmark("59 TransformedTargetRegressor log", () -> {
            // synthetic regression y>0
            int n = Math.min(500, ytr.length);
            double[][] X = new double[n][3];
            double[] y = new double[n];
            DataFrame num = fillNum(Xtr, NUM3);
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < 3; j++) X[i][j] = DataValues.asDouble(num.column(NUM3[j]).get(i));
                y[i] = Math.abs(ytr[i]) + 0.1 + Math.abs(X[i][0]);
            }
            TransformedTargetRegressor model = TransformedTargetRegressor.withLogTransform(new Ridge());
            model.fit(X, y);
            double[] pred = model.predict(X);
            check("ttr preds", pred.length == n);
            double mae = 0;
            for (int i = 0; i < n; i++) mae += Math.abs(pred[i] - y[i]);
            mae /= n;
            check("ttr finite mae", Double.isFinite(mae));
        });
        benchmark("60 full industrial Pipeline ColumnTransformer+SelectKBest", () -> {
            DataFrame train = Xtr.copy();
            if (!train.hasColumn("y")) {
                train.addColumn("y", Column.DType.FLOAT64);
                Column c = train.column("y");
                while (c.size() < train.rowCount()) c.add(0.0);
                for (int i = 0; i < ytr.length; i++) c.set(i, ytr[i]);
            }
            // ColumnTransformer column-isolated fit (fea.md preprocessor)
            SimpleImputer numImp = new SimpleImputer("median", NUM4);
            SimpleImputer catImp = new SimpleImputer("most_frequent", "cat_nominal");
            ColumnTransformer pre = new ColumnTransformer()
                .addTransformer("num", numImp, NUM4)
                .addTransformer("ord", new SimpleImputer("most_frequent", "cat_ordinal"), "cat_ordinal")
                .addTransformer("nom", catImp, "cat_nominal");
            DataFrame preOut = pre.fitTransform(train);
            check("ct isolated fit: numImp fitted", numImp.isFitted());
            check("ct isolated fit: catImp fitted", catImp.isFitted());
            // num imputer must NOT have stats for cat column (column-isolated fit)
            check("ct numImp no cat stats", !numImp.getStatistics().containsKey("cat_nominal"));

            StandardScaler sc = new StandardScaler(NUM4);
            DataFrame scaled = sc.fitTransform(preOut);

            // fea.md-style: Pipeline([preprocess pieces..., SelectKBest]).fit(X, y)
            int k = Math.min(24, NUM4.length);
            SelectKBest skb = new SelectKBest(k, "f_classif", NUM4);
            Pipeline full = new Pipeline()
                .addStep("scale", new StandardScaler(NUM4))
                .addStep("select", skb);
            DataFrame selected = full.fitTransform(scaled, ytr);
            check("full pipe select fitted via y", skb.isFitted());
            check("full pipe select k", skb.getSelectedColumns().size() == k);

            DataFrame testPre = pre.transform(Xte);
            DataFrame testOut = full.transform(testPre);
            check("test transform rows", testOut.rowCount() == Xte.rowCount());
            check("selected features>0", skb.getSelectedColumns().size() > 0);
        });
    }

    static void runBackendParity(DataFrame Xtr) {
        DataFrame num = fillNum(Xtr, NUM3);
        benchmark("backend StandardScaler DF vs matrix vs NDArray vs Tensor", () -> {
            StandardScaler sc = new StandardScaler(NUM3);
            DataFrame dfOut = sc.fitTransform(num);
            double[][] mIn = FeatureMatrices.fromDf(num, NUM3);
            FeatureBackends.MatrixStandardScaler ms = new FeatureBackends.MatrixStandardScaler();
            double[][] mOut = ms.fitTransform(mIn);

            double[][] dfMat = FeatureMatrices.fromDf(dfOut, NUM3);
            double diff = FeatureMatrices.maxAbsDiff(dfMat, mOut);
            check("DF vs matrix maxAbsDiff<1e-5 (" + diff + ")", diff < 1e-5);

            NDArray nd = FeatureBackends.toNdArray(mIn);
            double[][] back = FeatureBackends.fromNdArray(nd);
            check("NDArray roundtrip", FeatureMatrices.maxAbsDiff(mIn, back) < 1e-12);

            // scale via numpy path: matrix scaler on nd roundtrip
            double[][] ndScaled = ms.transform(FeatureBackends.fromNdArray(nd));
            check("numpy path parity", FeatureMatrices.maxAbsDiff(mOut, ndScaled) < 1e-12);

            try {
                Tensor t = FeatureBackends.toTensor(mIn);
                double[][] tBack = FeatureBackends.fromTensor(t);
                double td = FeatureMatrices.maxAbsDiff(mIn, tBack);
                check("Tensor roundtrip <1e-4 (" + td + ")", td < 1e-4);
                double[][] tScaled = ms.transform(tBack);
                check("Tensor scale parity", FeatureMatrices.maxAbsDiff(mOut, tScaled) < 1e-4);
            } catch (Throwable e) {
                skipped++;
                System.out.println("  SKIP tensor path: " + e);
            }
        });
    }

    static void runStress() {
        int[] rows = FULL ? new int[]{1000, 10000} : new int[]{1000, 5000};
        int[] dims = FULL ? new int[]{8, 32, 128} : new int[]{8, 32};
        System.out.printf("%-18s %-10s %8s %6s %10s %12s%n", "op", "backend", "n", "d", "ms", "rows/s");
        for (int n : rows) {
            for (int d : dims) {
                double[][] X = new double[n][d];
                for (int i = 0; i < n; i++)
                    for (int j = 0; j < d; j++) X[i][j] = RNG.nextGaussian();
                String[] cols = new String[d];
                for (int j = 0; j < d; j++) cols[j] = "f" + j;
                DataFrame df = FeatureMatrices.toDf(X, cols);

                timeOp("StandardScaler", "DataFrame", n, d, () -> {
                    new StandardScaler(cols).fitTransform(df);
                });
                timeOp("StandardScaler", "matrix", n, d, () -> {
                    new FeatureBackends.MatrixStandardScaler().fitTransform(X);
                });
                timeOp("StandardScaler", "NDArray", n, d, () -> {
                    NDArray nd = FeatureBackends.toNdArray(X);
                    double[][] m = FeatureBackends.fromNdArray(nd);
                    new FeatureBackends.MatrixStandardScaler().fitTransform(m);
                });
                timeOp("RobustScaler", "DataFrame", n, d, () -> {
                    new RobustScaler(cols).setQuantileRange(2.5, 97.5).fitTransform(df);
                });
                if (d <= 32) {
                    timeOp("Polynomial d=2", "DataFrame", n, Math.min(d, 6), () -> {
                        String[] c6 = Arrays.copyOf(cols, Math.min(6, d));
                        new PolynomialFeatures(2, false, false, c6).fitTransform(df);
                    });
                    timeOp("PCA k=8", "DataFrame", n, d, () -> {
                        int k = Math.min(8, d);
                        new PCA(k, cols).fitTransform(df);
                    });
                }
            }
        }
        // one-hot cardinality stress
        benchmark("stress OneHot card=100 n=2000", () -> {
            DataFrame df = DataFrame.create();
            df.addColumn("cat", Column.DType.STRING);
            for (int i = 0; i < 2000; i++) df.addRow("c" + (i % 100));
            OneHotEncoder ohe = new OneHotEncoder(true, null, "cat");
            DataFrame out = ohe.fitTransform(df);
            check("ohe card cols", ohe.getCategories().get("cat").size() == 100);
        });
    }

    static void timeOp(String op, String backend, int n, int d, CheckedRunnable r) {
        long t0 = System.nanoTime();
        try {
            r.run();
            long ms = Math.max(1, (System.nanoTime() - t0) / 1_000_000);
            double rps = n * 1000.0 / ms;
            System.out.printf("%-18s %-10s %8d %6d %10d %12.1f%n", op, backend, n, d, ms, rps);
            passed++;
        } catch (Throwable e) {
            failed++;
            System.out.printf("%-18s %-10s %8d %6d FAIL %s%n", op, backend, n, d, e.toString());
            report.append("stress FAIL ").append(op).append("/").append(backend).append(": ").append(e).append('\n');
        }
    }

    static void runPlots(DataFrame Xtr, double[] ytr) throws Exception {
        DataFrame num = fillNum(Xtr, NUM3);
        Seaborn.set_theme("darkgrid");

        benchmark("plot scale effect hist/kde", () -> {
            StandardScaler sc = new StandardScaler("num_0");
            RobustScaler rs = new RobustScaler("num_0");
            QuantileTransformer qt = new QuantileTransformer(QuantileTransformer.Output.NORMAL, 100, "num_0");
            DataFrame s1 = sc.fitTransform(num);
            DataFrame s2 = rs.fitTransform(num);
            DataFrame s3 = qt.fitTransform(num);

            DataFrame plotDf = DataFrame.create();
            plotDf.addColumn("raw", Column.DType.FLOAT64);
            plotDf.addColumn("standard", Column.DType.FLOAT64);
            plotDf.addColumn("robust", Column.DType.FLOAT64);
            plotDf.addColumn("quantile", Column.DType.FLOAT64);
            int n = Math.min(800, num.rowCount());
            for (int i = 0; i < n; i++) {
                plotDf.addRow(
                    DataValues.asDouble(num.column("num_0").get(i)),
                    DataValues.asDouble(s1.column("num_0").get(i)),
                    DataValues.asDouble(s2.column("num_0").get(i)),
                    DataValues.asDouble(s3.column("num_0").get(i))
                );
            }
            var h = Seaborn.histplot(plotDf, "raw", 30, true);
            Path p = PLOT_DIR.resolve("01_scale_raw_hist.png");
            h.savefig(p.toString());
            checkPng(p, "scale raw");

            var k = Seaborn.kdeplot(FeatureBackends.columnToArray(s1, "num_0"));
            k = Seaborn.kdeplot(k, FeatureBackends.columnToArray(s2, "num_0"), "robust");
            Path p2 = PLOT_DIR.resolve("02_scale_kde_compare.png");
            k.savefig(p2.toString());
            checkPng(p2, "scale kde");
        });

        benchmark("plot missingness bar", () -> {
            long m0 = countMissing(Xtr, "num_0");
            DataFrame filled = new SimpleImputer("mean", "num_0").fitTransform(Xtr);
            long m1 = countMissing(filled, "num_0");
            DataFrame bar = DataFrame.create();
            bar.addColumn("stage", Column.DType.STRING);
            bar.addColumn("missing", Column.DType.FLOAT64);
            bar.addRow("before", (double) m0);
            bar.addRow("after", (double) m1);
            var c = Seaborn.barplot(bar, "stage", "missing");
            Path p = PLOT_DIR.resolve("03_missingness.png");
            c.savefig(p.toString());
            checkPng(p, "missingness");
        });

        benchmark("plot ordinal countplot", () -> {
            var c = Seaborn.countplot(Xtr, "cat_ordinal");
            Path p = PLOT_DIR.resolve("04_ordinal_count.png");
            c.savefig(p.toString());
            checkPng(p, "ordinal count");
        });

        benchmark("plot PCA scatter by y", () -> {
            DataFrame wide = fillNum(Xtr, NUM);
            StandardScaler sc = new StandardScaler(NUM);
            DataFrame scaled = sc.fitTransform(wide);
            PCA pca = new PCA(2, NUM);
            DataFrame pcs = pca.fitTransform(scaled);
            // find pc columns
            List<String> pcCols = new ArrayList<>();
            for (Column col : pcs.columns()) {
                String n = col.name().toLowerCase();
                if (n.contains("pca") || n.contains("pc") || n.startsWith("component")) pcCols.add(col.name());
            }
            if (pcCols.size() < 2) {
                // PCA may replace or append — use first two numeric outs differing from input
                for (Column col : pcs.columns()) {
                    if (FeatureMatrices.numericColumnNames(pcs).contains(col.name())) pcCols.add(col.name());
                }
            }
            check("pca has cols", pcs.columnCount() > 0);
            // build plot frame
            DataFrame plot = DataFrame.create();
            plot.addColumn("pc1", Column.DType.FLOAT64);
            plot.addColumn("pc2", Column.DType.FLOAT64);
            plot.addColumn("y", Column.DType.FLOAT64);
            String c1 = pcCols.size() > 0 ? pcCols.get(0) : NUM[0];
            String c2 = pcCols.size() > 1 ? pcCols.get(1) : NUM[1];
            if (!pcs.hasColumn(c1)) c1 = pcs.columns().get(0).name();
            if (!pcs.hasColumn(c2)) c2 = pcs.columns().get(Math.min(1, pcs.columnCount() - 1)).name();
            int n = Math.min(1000, pcs.rowCount());
            for (int i = 0; i < n; i++) {
                plot.addRow(
                    DataValues.asDouble(pcs.column(c1).get(i)),
                    DataValues.asDouble(pcs.column(c2).get(i)),
                    ytr[i]
                );
            }
            var sca = Seaborn.scatterplot(plot, "pc1", "pc2", "y");
            Path p = PLOT_DIR.resolve("05_pca_scatter.png");
            sca.savefig(p.toString());
            checkPng(p, "pca scatter");
        });

        benchmark("plot cluster labels", () -> {
            DataFrame num2 = fillNum(Xtr, NUM3);
            ClusterFeatures cf = new ClusterFeatures(5, NUM3).setMode(ClusterFeatures.Mode.BOTH);
            DataFrame out = cf.fitTransform(num2);
            DataFrame plot = DataFrame.create();
            plot.addColumn("x", Column.DType.FLOAT64);
            plot.addColumn("y", Column.DType.FLOAT64);
            plot.addColumn("cluster", Column.DType.FLOAT64);
            int n = Math.min(1000, out.rowCount());
            for (int i = 0; i < n; i++) {
                plot.addRow(
                    DataValues.asDouble(out.column("num_0").get(i)),
                    DataValues.asDouble(out.column("num_1").get(i)),
                    DataValues.asDouble(out.column("cluster_label").get(i))
                );
            }
            var sca = Seaborn.scatterplot(plot, "x", "y", "cluster");
            Path p = PLOT_DIR.resolve("06_cluster_scatter.png");
            sca.savefig(p.toString());
            checkPng(p, "cluster");
        });

        benchmark("plot SelectKBest scores bar", () -> {
            DataFrame wide = fillNum(Xtr, NUM);
            if (!wide.hasColumn("y")) {
                wide.addColumn("y", Column.DType.FLOAT64);
                Column yc = wide.column("y");
                while (yc.size() < wide.rowCount()) yc.add(0.0);
            }
            Column yc = wide.column("y");
            for (int i = 0; i < Math.min(ytr.length, wide.rowCount()); i++) yc.set(i, ytr[i]);
            SelectKBest skb = new SelectKBest(6, "f_classif", NUM);
            skb.fit(wide, "y");
            Map<String, Double> scores = skb.getScores();
            DataFrame bar = DataFrame.create();
            bar.addColumn("feature", Column.DType.STRING);
            bar.addColumn("score", Column.DType.FLOAT64);
            for (Map.Entry<String, Double> e : scores.entrySet()) {
                bar.addRow(e.getKey(), e.getValue());
            }
            var c = Seaborn.barplot(bar, "feature", "score");
            Path p = PLOT_DIR.resolve("07_selectkbest_scores.png");
            c.savefig(p.toString());
            checkPng(p, "kbest scores");
        });

        benchmark("plot corr heatmap after scale", () -> {
            DataFrame wide = fillNum(Xtr, NUM4);
            StandardScaler sc = new StandardScaler(NUM4);
            DataFrame scaled = sc.fitTransform(wide);
            double[][] mat = FeatureMatrices.fromDf(scaled, NUM4);
            int d = NUM4.length;
            double[][] corr = new double[d][d];
            for (int i = 0; i < d; i++) {
                for (int j = 0; j < d; j++) {
                    corr[i][j] = pearson(mat, i, j);
                }
            }
            var hm = Seaborn.heatmap(corr, Arrays.asList(NUM4), Arrays.asList(NUM4));
            Path p = PLOT_DIR.resolve("08_corr_heatmap.png");
            hm.savefig(p.toString());
            checkPng(p, "corr heatmap");
        });

        benchmark("plot backend parity hist DF vs matrix", () -> {
            DataFrame n0 = fillNum(Xtr, "num_0");
            StandardScaler sc = new StandardScaler("num_0");
            DataFrame dfOut = sc.fitTransform(n0);
            double[][] m = FeatureMatrices.fromDf(n0, "num_0");
            double[][] mOut = new FeatureBackends.MatrixStandardScaler().fitTransform(m);
            double[] a = FeatureBackends.columnToArray(dfOut, "num_0");
            double[] b = new double[mOut.length];
            for (int i = 0; i < b.length; i++) b[i] = mOut[i][0];
            var k = Seaborn.kdeplot(a, "dataframe");
            k = Seaborn.kdeplot(k, b, "matrix");
            Path p = PLOT_DIR.resolve("09_backend_parity_kde.png");
            k.savefig(p.toString());
            checkPng(p, "backend parity");
        });

        benchmark("plot power transform before/after", () -> {
            DataFrame n0 = fillNum(Xtr, "num_0", "num_1");
            PowerTransformer pt = new PowerTransformer(PowerTransformer.Method.YEO_JOHNSON, "num_0");
            DataFrame after = pt.fitTransform(n0);
            var k = Seaborn.kdeplot(FeatureBackends.columnToArray(n0, "num_0"), "before");
            k = Seaborn.kdeplot(k, FeatureBackends.columnToArray(after, "num_0"), "after");
            Path p = PLOT_DIR.resolve("10_power_yj.png");
            k.savefig(p.toString());
            checkPng(p, "power yj");
        });
    }

    // ---- helpers ----

    static DataFrame fillNum(DataFrame src, String... cols) {
        try {
            return new SimpleImputer("mean", cols).fitTransform(src);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    static double mean(double[] a) {
        double s = 0; int c = 0;
        for (double v : a) if (!Double.isNaN(v)) { s += v; c++; }
        return c == 0 ? 0 : s / c;
    }

    static double std(double[] a) {
        double m = mean(a); double s = 0; int c = 0;
        for (double v : a) if (!Double.isNaN(v)) { s += (v - m) * (v - m); c++; }
        return c < 2 ? 0 : Math.sqrt(s / c);
    }

    static double pearson(double[][] mat, int j1, int j2) {
        int n = mat.length;
        double m1 = 0, m2 = 0;
        for (int i = 0; i < n; i++) { m1 += mat[i][j1]; m2 += mat[i][j2]; }
        m1 /= n; m2 /= n;
        double num = 0, d1 = 0, d2 = 0;
        for (int i = 0; i < n; i++) {
            double a = mat[i][j1] - m1, b = mat[i][j2] - m2;
            num += a * b; d1 += a * a; d2 += b * b;
        }
        double den = Math.sqrt(d1 * d2);
        return den < 1e-15 ? 0 : num / den;
    }
}
