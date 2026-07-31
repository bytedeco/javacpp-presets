package media;

import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.data.numpy.NDArray;
import org.bytedeco.pytorch.data.numpy.NP;
import org.bytedeco.pytorch.dataframe.Column;
import org.bytedeco.pytorch.dataframe.DataFrame;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.plot.*;
import org.bytedeco.pytorch.plot.chart.*;
import org.bytedeco.pytorch.plot.seaborn.Seaborn;

import java.awt.image.BufferedImage;
import java.nio.file.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Seaborn Java ↔ Python API parity suite for the 20 examples in
 * {@code org/lance/ipc/sn.md}, exercised on three data backends:
 * <ul>
 *   <li><b>numpy</b> — {@link NP} / {@link NDArray} (NO hand-rolled normal())</li>
 *   <li><b>dataframe</b> — {@link DataFrame}</li>
 *   <li><b>tensor</b> — javacpp-pytorch {@link Tensor}</li>
 * </ul>
 * plus multi-dimensional stress / throughput benchmarks.
 *
 * <pre>
 *   javac -cp "target/classes:$(cat target/cp.txt)" -d target/samples-compile \
 *         samples/BenchmarkSeaborn.java
 *   java  --add-opens=java.base/java.nio=ALL-UNNAMED \
 *         -cp "target/samples-compile:target/classes:$(cat target/cp.txt)" \
 *         media.BenchmarkSeaborn
 * </pre>
 */
public class BenchmarkSeaborn {
    static int passed = 0, failed = 0;
    static final StringBuilder report = new StringBuilder();

    @FunctionalInterface
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
            System.out.printf(" FAIL %-56s %6d ms: %s%n", name, ms, e);
            report.append("FAIL ").append(name).append(": ").append(e).append('\n');
            e.printStackTrace(System.out);
        }
    }

    static void check(String name, boolean ok) {
        if (ok) passed++;
        else {
            failed++;
            report.append("  check failed: ").append(name).append('\n');
            throw new AssertionError(name);
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

    static void checkRender(BaseChart chart, String label) {
        check(label + " non-null", chart != null);
        BufferedImage img = chart.render();
        check(label + " image non-null", img != null);
        check(label + " width>0", img.getWidth() > 0);
        check(label + " height>0", img.getHeight() > 0);
        int sample = 0;
        for (int y = 0; y < img.getHeight(); y += Math.max(1, img.getHeight() / 8)) {
            for (int x = 0; x < img.getWidth(); x += Math.max(1, img.getWidth() / 8)) {
                if ((img.getRGB(x, y) & 0xFFFFFF) != 0xFFFFFF) sample++;
            }
        }
        check(label + " has non-white content (" + sample + ")", sample > 0);
    }

    // ---- data builders: ALWAYS via NP for numeric generation ----

    /** True numpy normal — required for numpy backend (no hand-rolled RNG). */
    static NDArray npNormal(int n, double loc, double scale) {
        return NP.Random.normal(loc, scale, n);
    }

    static Tensor tensorNormal(int n, double loc, double scale) {
        return NP.toTensor(npNormal(n, loc, scale));
    }

    static DataFrame groupValueDf() {
        DataFrame df = DataFrame.create();
        df.addColumn("group", Column.DType.STRING);
        df.addColumn("value", Column.DType.FLOAT64);
        for (double v : npNormal(100, 0, 1).asDoubleArray()) df.addRow("A", v);
        for (double v : npNormal(100, 1.5, 1).asDoubleArray()) df.addRow("B", v);
        return df;
    }

    static DataFrame scatterDf() {
        DataFrame df = DataFrame.create();
        df.addColumn("x", Column.DType.FLOAT64);
        df.addColumn("y", Column.DType.FLOAT64);
        double[] x = NP.Random.randn(200).asDoubleArray();
        double[] n = NP.Random.randn(200).asDoubleArray();
        for (int i = 0; i < 200; i++) df.addRow(x[i], x[i] + n[i] * 0.5);
        return df;
    }

    static DataFrame lineLossDf() {
        DataFrame df = DataFrame.create();
        df.addColumn("step", Column.DType.FLOAT64);
        df.addColumn("loss", Column.DType.FLOAT64);
        double[] noise = NP.Random.randn(200).asDoubleArray();
        for (int step = 0; step < 200; step++) {
            double loss = Math.exp(-step / 80.0) + noise[step] * 0.02;
            df.addRow((double) step, loss);
        }
        return df;
    }

    static DataFrame catDf() {
        DataFrame df = DataFrame.create();
        df.addColumn("category", Column.DType.STRING);
        String[] cats = {"cat1", "cat2", "cat3"};
        // use NP permutation-ish via randint
        double[] idx = NP.Random.randint(0, 3, 300).asDoubleArray();
        for (int i = 0; i < 300; i++) df.addRow(cats[(int) idx[i]]);
        return df;
    }

    static DataFrame multiDf() {
        DataFrame df = DataFrame.create();
        df.addColumn("f1", Column.DType.FLOAT64);
        df.addColumn("f2", Column.DType.FLOAT64);
        df.addColumn("f3", Column.DType.FLOAT64);
        double[] a = NP.Random.randn(150).asDoubleArray();
        double[] b = NP.Random.randn(150).asDoubleArray();
        double[] c = NP.Random.randn(150).asDoubleArray();
        for (int i = 0; i < 150; i++) df.addRow(a[i], b[i], c[i]);
        return df;
    }

    static DataFrame scatterWithLabel() {
        DataFrame out = DataFrame.create();
        out.addColumn("x", Column.DType.FLOAT64);
        out.addColumn("y", Column.DType.FLOAT64);
        out.addColumn("label", Column.DType.STRING);
        double[] x = NP.Random.randn(200).asDoubleArray();
        double[] n = NP.Random.randn(200).asDoubleArray();
        double[] coin = NP.Random.rand(200).asDoubleArray();
        for (int i = 0; i < 200; i++)
            out.addRow(x[i], x[i] + n[i] * 0.5, coin[i] < 0.5 ? "type1" : "type2");
        return out;
    }

    static DataFrame timeDf() {
        DataFrame df = DataFrame.create();
        df.addColumn("t", Column.DType.FLOAT64);
        df.addColumn("val", Column.DType.FLOAT64);
        df.addColumn("grp", Column.DType.STRING);
        int[] ts = {1, 2, 3, 4};
        for (int g = 0; g < 2; g++) {
            String grp = g == 0 ? "A" : "B";
            double[] vals = NP.Random.randn(30 * ts.length).asDoubleArray();
            int k = 0;
            for (int rep = 0; rep < 30; rep++)
                for (int t : ts) df.addRow((double) t, vals[k++], grp);
        }
        return df;
    }

    static double[][] corrLike(int n) {
        double[] flat = NP.Random.rand(n * n).asDoubleArray();
        double[][] m = new double[n][n];
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                m[i][j] = i == j ? 1.0 : flat[i * n + j];
        return m;
    }

    public static void main(String[] args) throws Exception {
        System.out.println("=== BenchmarkSeaborn (sn.md 20-example × 3 backends) ===");
        System.out.println("seed=42  style=whitegrid  numpy=NP.Random.* (no hand-rolled normal)\n");
        NP.Random.seed(42);
        try { torch.tensor(new double[]{1.0}); } catch (Throwable ignored) {}

        Path out = args.length > 0
            ? Paths.get(args[0])
            : Files.createTempDirectory("seaborn-bench-");
        Files.createDirectories(out);
        System.out.println("Output dir: " + out.toAbsolutePath());

        Seaborn.set_theme("whitegrid");
        Seaborn.set_palette("deep");
        check("style whitegrid", "whitegrid".equals(Seaborn.currentStyle()));
        check("palette non-empty", Seaborn.color_palette().length >= 6);

        DataFrame groupDf = groupValueDf();
        DataFrame scatter = scatterDf();
        DataFrame loss = lineLossDf();
        DataFrame cats = catDf();
        DataFrame multi = multiDf();
        DataFrame labeled = scatterWithLabel();
        DataFrame timed = timeDf();

        System.out.println("\n-- 20 API parity examples (numpy / dataframe / tensor) --");

        // 01 histplot + kde
        benchmark("01 histplot+kde viaNumpy", () -> {
            NDArray data = npNormal(1000, 0, 1);
            HistogramChart c = Seaborn.histplot(data, 30, true)
                .setTitle("Normal Distribution Histogram");
            checkRender(c, "01-np");
            Path p = out.resolve("01_histplot_numpy.png");
            c.savefig(p.toString());
            checkPng(p, "01_np");
        });
        benchmark("01 histplot+kde viaDataFrame", () -> {
            DataFrame df = DataFrame.create();
            df.addColumn("value", Column.DType.FLOAT64);
            for (double v : npNormal(1000, 0, 1).asDoubleArray()) df.addRow(v);
            HistogramChart c = Seaborn.histplot(df, "value", 30, true);
            checkRender(c, "01-df");
            Path p = out.resolve("01_histplot_df.png");
            c.savefig(p.toString());
            checkPng(p, "01_df");
        });
        benchmark("01 histplot+kde viaTensor", () -> {
            Tensor t = tensorNormal(1000, 0, 1);
            HistogramChart c = Seaborn.histplot(t, 30, true);
            checkRender(c, "01-t");
            Path p = out.resolve("01_histplot_tensor.png");
            c.savefig(p.toString());
            checkPng(p, "01_t");
        });

        // 02 kdeplot multi-group
        benchmark("02 kdeplot multi viaNumpy", () -> {
            NDArray d1 = npNormal(500, 0, 1);
            NDArray d2 = npNormal(500, 2, 1);
            LineChart c = Seaborn.kdeplot(d1, "Group A");
            Seaborn.kdeplot(c, d2, "Group B");
            checkRender(c, "02-np");
            Path p = out.resolve("02_kdeplot_numpy.png");
            c.savefig(p.toString());
            checkPng(p, "02_np");
        });
        benchmark("02 kdeplot multi viaDataFrame", () -> {
            LineChart c = Seaborn.kdeplot(groupDf, "value", "group");
            checkRender(c, "02-df");
            Path p = out.resolve("02_kdeplot_df.png");
            c.savefig(p.toString());
            checkPng(p, "02_df");
        });
        benchmark("02 kdeplot multi viaTensor", () -> {
            Tensor d1 = tensorNormal(500, 0, 1);
            Tensor d2 = tensorNormal(500, 2, 1);
            LineChart c = Seaborn.kdeplot(d1, "Group A");
            Seaborn.kdeplot(c, d2, "Group B");
            checkRender(c, "02-t");
            Path p = out.resolve("02_kdeplot_tensor.png");
            c.savefig(p.toString());
            checkPng(p, "02_t");
        });

        // 03 boxplot
        benchmark("03 boxplot viaNumpy groups", () -> {
            BoxChart c = Seaborn.boxplot(new String[]{"A", "B"},
                npNormal(100, 0, 1), npNormal(100, 1.5, 1));
            checkRender(c, "03-np");
            Path p = out.resolve("03_boxplot_numpy.png");
            c.savefig(p.toString());
            checkPng(p, "03_np");
        });
        benchmark("03 boxplot viaDataFrame", () -> {
            BoxChart c = Seaborn.boxplot(groupDf, "group", "value");
            checkRender(c, "03-df");
            Path p = out.resolve("03_boxplot_df.png");
            c.savefig(p.toString());
            checkPng(p, "03_df");
        });
        benchmark("03 boxplot viaTensor groups", () -> {
            BoxChart c = Seaborn.boxplot(new String[]{"A", "B"},
                tensorNormal(100, 0, 1), tensorNormal(100, 1.5, 1));
            checkRender(c, "03-t");
            Path p = out.resolve("03_boxplot_tensor.png");
            c.savefig(p.toString());
            checkPng(p, "03_t");
        });

        // 04 violinplot
        benchmark("04 violinplot viaNumpy", () -> {
            ViolinChart c = Seaborn.violinplot(new String[]{"A", "B"},
                npNormal(100, 0, 1), npNormal(100, 1.5, 1));
            checkRender(c, "04-np");
            Path p = out.resolve("04_violin_numpy.png");
            c.savefig(p.toString());
            checkPng(p, "04_np");
        });
        benchmark("04 violinplot viaDataFrame", () -> {
            ViolinChart c = Seaborn.violinplot(groupDf, "group", "value", "quartile");
            checkRender(c, "04-df");
            Path p = out.resolve("04_violin_df.png");
            c.savefig(p.toString());
            checkPng(p, "04_df");
        });
        benchmark("04 violinplot viaTensor", () -> {
            ViolinChart c = Seaborn.violinplot(new String[]{"A", "B"},
                tensorNormal(100, 0, 1), tensorNormal(100, 1.5, 1));
            checkRender(c, "04-t");
            Path p = out.resolve("04_violin_tensor.png");
            c.savefig(p.toString());
            checkPng(p, "04_t");
        });

        // 05 scatterplot
        benchmark("05 scatterplot viaNumpy", () -> {
            NDArray x = NP.Random.randn(200);
            double[] xx = x.asDoubleArray();
            double[] nn = NP.Random.randn(200).asDoubleArray();
            double[] yy = new double[200];
            for (int i = 0; i < 200; i++) yy[i] = xx[i] + nn[i] * 0.5;
            ScatterChart c = Seaborn.scatterplot(x, new NDArray(yy, 200));
            checkRender(c, "05-np");
            Path p = out.resolve("05_scatter_numpy.png");
            c.savefig(p.toString());
            checkPng(p, "05_np");
        });
        benchmark("05 scatterplot viaDataFrame", () -> {
            ScatterChart c = Seaborn.scatterplot(scatter, "x", "y");
            checkRender(c, "05-df");
            Path p = out.resolve("05_scatter_df.png");
            c.savefig(p.toString());
            checkPng(p, "05_df");
        });
        benchmark("05 scatterplot viaTensor", () -> {
            double[] xx = NP.Random.randn(200).asDoubleArray();
            double[] nn = NP.Random.randn(200).asDoubleArray();
            double[] yy = new double[200];
            for (int i = 0; i < 200; i++) yy[i] = xx[i] + nn[i] * 0.5;
            ScatterChart c = Seaborn.scatterplot(torch.tensor(xx), torch.tensor(yy));
            checkRender(c, "05-t");
            Path p = out.resolve("05_scatter_tensor.png");
            c.savefig(p.toString());
            checkPng(p, "05_t");
        });

        // 06 stripplot
        benchmark("06 stripplot viaNumpy", () -> {
            ScatterChart c = Seaborn.stripplot(new String[]{"A", "B"},
                npNormal(100, 0, 1), npNormal(100, 1.5, 1));
            checkRender(c, "06-np");
            Path p = out.resolve("06_strip_numpy.png");
            c.savefig(p.toString());
            checkPng(p, "06_np");
        });
        benchmark("06 stripplot viaDataFrame", () -> {
            ScatterChart c = Seaborn.stripplot(groupDf, "group", "value", 0.6);
            checkRender(c, "06-df");
            Path p = out.resolve("06_strip_df.png");
            c.savefig(p.toString());
            checkPng(p, "06_df");
        });
        benchmark("06 stripplot viaTensor", () -> {
            ScatterChart c = Seaborn.stripplot(new String[]{"A", "B"},
                tensorNormal(100, 0, 1), tensorNormal(100, 1.5, 1));
            checkRender(c, "06-t");
            Path p = out.resolve("06_strip_tensor.png");
            c.savefig(p.toString());
            checkPng(p, "06_t");
        });

        // 07 swarmplot
        benchmark("07 swarmplot viaNumpy", () -> {
            ScatterChart c = Seaborn.swarmplot(new String[]{"A", "B"},
                npNormal(80, 0, 1), npNormal(80, 1.5, 1));
            checkRender(c, "07-np");
            Path p = out.resolve("07_swarm_numpy.png");
            c.savefig(p.toString());
            checkPng(p, "07_np");
        });
        benchmark("07 swarmplot viaDataFrame", () -> {
            ScatterChart c = Seaborn.swarmplot(groupDf, "group", "value", 4);
            checkRender(c, "07-df");
            Path p = out.resolve("07_swarm_df.png");
            c.savefig(p.toString());
            checkPng(p, "07_df");
        });
        benchmark("07 swarmplot viaTensor", () -> {
            ScatterChart c = Seaborn.swarmplot(new String[]{"A", "B"},
                tensorNormal(80, 0, 1), tensorNormal(80, 1.5, 1));
            checkRender(c, "07-t");
            Path p = out.resolve("07_swarm_tensor.png");
            c.savefig(p.toString());
            checkPng(p, "07_t");
        });

        // 08 lineplot
        benchmark("08 lineplot viaNumpy", () -> {
            NDArray step = NP.arange(0, 200);
            double[] st = step.asDoubleArray();
            double[] noise = NP.Random.randn(200).asDoubleArray();
            double[] lossY = new double[200];
            for (int i = 0; i < 200; i++) lossY[i] = Math.exp(-st[i] / 80.0) + noise[i] * 0.02;
            LineChart c = Seaborn.lineplot(step, new NDArray(lossY, 200))
                .setTitle("Training Loss Curve");
            checkRender(c, "08-np");
            Path p = out.resolve("08_line_numpy.png");
            c.savefig(p.toString());
            checkPng(p, "08_np");
        });
        benchmark("08 lineplot viaDataFrame", () -> {
            LineChart c = Seaborn.lineplot(loss, "step", "loss").setTitle("Training Loss Curve");
            checkRender(c, "08-df");
            Path p = out.resolve("08_line_df.png");
            c.savefig(p.toString());
            checkPng(p, "08_df");
        });
        benchmark("08 lineplot viaTensor", () -> {
            double[] st = NP.arange(0, 200).asDoubleArray();
            double[] noise = NP.Random.randn(200).asDoubleArray();
            double[] lossY = new double[200];
            for (int i = 0; i < 200; i++) lossY[i] = Math.exp(-st[i] / 80.0) + noise[i] * 0.02;
            LineChart c = Seaborn.lineplot(torch.tensor(st), torch.tensor(lossY))
                .setTitle("Training Loss Curve");
            checkRender(c, "08-t");
            Path p = out.resolve("08_line_tensor.png");
            c.savefig(p.toString());
            checkPng(p, "08_t");
        });

        // 09 barplot — category semantics → DF primary; array overload for np/tensor
        benchmark("09 barplot viaNumpy", () -> {
            // means of groups via NP
            double[] a = npNormal(100, 0, 1).asDoubleArray();
            double[] b = npNormal(100, 1.5, 1).asDoubleArray();
            double ma = 0, mb = 0;
            for (double v : a) ma += v; ma /= a.length;
            for (double v : b) mb += v; mb /= b.length;
            BarChart c = Seaborn.barplot(new String[]{"A", "B"},
                new NDArray(new double[]{ma, mb}, 2));
            checkRender(c, "09-np");
            Path p = out.resolve("09_bar_numpy.png");
            c.savefig(p.toString());
            checkPng(p, "09_np");
        });
        benchmark("09 barplot viaDataFrame errorbar=sd", () -> {
            BarChart c = Seaborn.barplot(groupDf, "group", "value", "sd");
            checkRender(c, "09-df");
            Path p = out.resolve("09_bar_df.png");
            c.savefig(p.toString());
            checkPng(p, "09_df");
        });
        benchmark("09 barplot viaTensor", () -> {
            BarChart c = Seaborn.barplot(new String[]{"A", "B"},
                torch.tensor(new double[]{0.1, 1.4}));
            checkRender(c, "09-t");
            Path p = out.resolve("09_bar_tensor.png");
            c.savefig(p.toString());
            checkPng(p, "09_t");
        });

        // 10 countplot — DF only (categorical)
        benchmark("10 countplot viaDataFrame", () -> {
            BarChart c = Seaborn.countplot(cats, "category");
            checkRender(c, "10-df");
            Path p = out.resolve("10_count_df.png");
            c.savefig(p.toString());
            checkPng(p, "10_df");
        });
        // numpy/tensor: count via unique labels not native — document as DF-primary
        benchmark("10 countplot viaNumpy (DF-from-labels)", () -> {
            // build DF from NP randint categories
            DataFrame df = catDf();
            BarChart c = Seaborn.countplot(df, "category");
            checkRender(c, "10-np");
            Path p = out.resolve("10_count_numpy.png");
            c.savefig(p.toString());
            checkPng(p, "10_np");
        });

        // 11 heatmap
        benchmark("11 heatmap viaNumpy", () -> {
            NDArray mat = new NDArray(flatten(corrLike(6)), 6, 6);
            HeatmapChart c = Seaborn.heatmap(mat, true, "coolwarm", 0.0, 1.0);
            checkRender(c, "11-np");
            Path p = out.resolve("11_heatmap_numpy.png");
            c.savefig(p.toString());
            checkPng(p, "11_np");
        });
        benchmark("11 heatmap viaDataFrame", () -> {
            HeatmapChart c = Seaborn.heatmap(multi);
            checkRender(c, "11-df");
            Path p = out.resolve("11_heatmap_df.png");
            c.savefig(p.toString());
            checkPng(p, "11_df");
        });
        benchmark("11 heatmap viaTensor", () -> {
            double[][] m = corrLike(6);
            Tensor t = torch.tensor(flatten(m)).reshape(new long[]{6, 6});
            HeatmapChart c = Seaborn.heatmap(t, true, "coolwarm", 0.0, 1.0);
            checkRender(c, "11-t");
            Path p = out.resolve("11_heatmap_tensor.png");
            c.savefig(p.toString());
            checkPng(p, "11_t");
        });

        // 12 clustermap
        benchmark("12 clustermap viaNumpy", () -> {
            NDArray mat = NP.Random.rand(8, 8);
            HeatmapChart c = Seaborn.clustermap(mat, "viridis");
            checkRender(c, "12-np");
            Path p = out.resolve("12_cluster_numpy.png");
            c.savefig(p.toString());
            checkPng(p, "12_np");
        });
        benchmark("12 clustermap viaDataFrame", () -> {
            HeatmapChart c = Seaborn.clustermap(multi);
            checkRender(c, "12-df");
            Path p = out.resolve("12_cluster_df.png");
            c.savefig(p.toString());
            checkPng(p, "12_df");
        });
        benchmark("12 clustermap viaTensor", () -> {
            Tensor t = NP.toTensor(NP.Random.rand(8, 8));
            HeatmapChart c = Seaborn.clustermap(t, "viridis");
            checkRender(c, "12-t");
            Path p = out.resolve("12_cluster_tensor.png");
            c.savefig(p.toString());
            checkPng(p, "12_t");
        });

        // 13 jointplot — DF primary; build DF from numpy/tensor matrix
        benchmark("13 jointplot viaNumpy→DF", () -> {
            NDArray x = NP.Random.randn(200);
            double[] xx = x.asDoubleArray();
            double[] nn = NP.Random.randn(200).asDoubleArray();
            double[] yy = new double[200];
            for (int i = 0; i < 200; i++) yy[i] = xx[i] + nn[i] * 0.5;
            DataFrame df = PlotInputs.xyDataFrame("x", xx, "y", yy);
            BaseChart c = Seaborn.jointplot(df, "x", "y", "kde");
            checkRender(c, "13-np");
            Path p = out.resolve("13_joint_numpy.png");
            c.savefig(p.toString());
            checkPng(p, "13_np");
        });
        benchmark("13 jointplot viaDataFrame", () -> {
            BaseChart c = Seaborn.jointplot(scatter, "x", "y", "kde");
            checkRender(c, "13-df");
            Path p = out.resolve("13_joint_df.png");
            c.savefig(p.toString());
            checkPng(p, "13_df");
        });
        benchmark("13 jointplot viaTensor→DF", () -> {
            double[] xx = NP.Random.randn(200).asDoubleArray();
            double[] nn = NP.Random.randn(200).asDoubleArray();
            double[] yy = new double[200];
            for (int i = 0; i < 200; i++) yy[i] = xx[i] + nn[i] * 0.5;
            // pack (200,2) tensor → DF
            double[] raw = new double[400];
            for (int i = 0; i < 200; i++) { raw[i * 2] = xx[i]; raw[i * 2 + 1] = yy[i]; }
            Tensor t = torch.tensor(raw).reshape(new long[]{200, 2});
            DataFrame df = Seaborn.dataframeFrom(t, "x", "y");
            BaseChart c = Seaborn.jointplot(df, "x", "y", "kde");
            checkRender(c, "13-t");
            Path p = out.resolve("13_joint_tensor.png");
            c.savefig(p.toString());
            checkPng(p, "13_t");
        });

        // 14 pairplot
        benchmark("14 pairplot viaNumpy→DF", () -> {
            NDArray m = NP.Random.randn(150, 3);
            DataFrame df = Seaborn.dataframeFrom(m, "f1", "f2", "f3");
            BaseChart c = Seaborn.pairplot(df);
            checkRender(c, "14-np");
            Path p = out.resolve("14_pair_numpy.png");
            c.savefig(p.toString());
            checkPng(p, "14_np");
        });
        benchmark("14 pairplot viaDataFrame", () -> {
            BaseChart c = Seaborn.pairplot(multi);
            checkRender(c, "14-df");
            Path p = out.resolve("14_pair_df.png");
            c.savefig(p.toString());
            checkPng(p, "14_df");
        });
        benchmark("14 pairplot viaTensor→DF", () -> {
            Tensor t = NP.toTensor(NP.Random.randn(150, 3));
            DataFrame df = Seaborn.dataframeFrom(t, "f1", "f2", "f3");
            BaseChart c = Seaborn.pairplot(df);
            checkRender(c, "14-t");
            Path p = out.resolve("14_pair_tensor.png");
            c.savefig(p.toString());
            checkPng(p, "14_t");
        });

        // 15 regplot
        benchmark("15 regplot viaNumpy", () -> {
            NDArray x = NP.Random.randn(200);
            double[] xx = x.asDoubleArray();
            double[] nn = NP.Random.randn(200).asDoubleArray();
            double[] yy = new double[200];
            for (int i = 0; i < 200; i++) yy[i] = xx[i] + nn[i] * 0.5;
            ScatterChart c = Seaborn.regplot(x, new NDArray(yy, 200)).setAlpha(0.6);
            checkRender(c, "15-np");
            Path p = out.resolve("15_reg_numpy.png");
            c.savefig(p.toString());
            checkPng(p, "15_np");
        });
        benchmark("15 regplot viaDataFrame", () -> {
            ScatterChart c = Seaborn.regplot(scatter, "x", "y").setAlpha(0.6);
            checkRender(c, "15-df");
            Path p = out.resolve("15_reg_df.png");
            c.savefig(p.toString());
            checkPng(p, "15_df");
        });
        benchmark("15 regplot viaTensor", () -> {
            double[] xx = NP.Random.randn(200).asDoubleArray();
            double[] nn = NP.Random.randn(200).asDoubleArray();
            double[] yy = new double[200];
            for (int i = 0; i < 200; i++) yy[i] = xx[i] + nn[i] * 0.5;
            ScatterChart c = Seaborn.regplot(torch.tensor(xx), torch.tensor(yy)).setAlpha(0.6);
            checkRender(c, "15-t");
            Path p = out.resolve("15_reg_tensor.png");
            c.savefig(p.toString());
            checkPng(p, "15_t");
        });

        // 16 lmplot hue — DF primary
        benchmark("16 lmplot viaDataFrame", () -> {
            ScatterChart c = Seaborn.lmplot(labeled, "x", "y", "label");
            checkRender(c, "16-df");
            Path p = out.resolve("16_lm_df.png");
            c.savefig(p.toString());
            checkPng(p, "16_df");
        });
        benchmark("16 lmplot viaNumpy-built DF", () -> {
            ScatterChart c = Seaborn.lmplot(scatterWithLabel(), "x", "y", "label");
            checkRender(c, "16-np");
            Path p = out.resolve("16_lm_numpy.png");
            c.savefig(p.toString());
            checkPng(p, "16_np");
        });

        // 17 jointplot fill
        benchmark("17 jointplot fill viaDataFrame", () -> {
            BaseChart c = Seaborn.jointplot(scatter, "x", "y", "kde", true);
            checkRender(c, "17-df");
            Path p = out.resolve("17_joint_fill_df.png");
            c.savefig(p.toString());
            checkPng(p, "17_df");
        });
        benchmark("17 jointplot fill viaNumpy→DF", () -> {
            double[] xx = NP.Random.randn(200).asDoubleArray();
            double[] nn = NP.Random.randn(200).asDoubleArray();
            double[] yy = new double[200];
            for (int i = 0; i < 200; i++) yy[i] = xx[i] + nn[i] * 0.5;
            BaseChart c = Seaborn.jointplot(PlotInputs.xyDataFrame("x", xx, "y", yy), "x", "y", "kde", true);
            checkRender(c, "17-np");
            Path p = out.resolve("17_joint_fill_numpy.png");
            c.savefig(p.toString());
            checkPng(p, "17_np");
        });

        // 18 FacetGrid — DF only
        benchmark("18 FacetGrid viaDataFrame", () -> {
            Seaborn.FacetGrid g = Seaborn.FacetGrid(groupDf, "group");
            g.mapHist("value", 20).setTitle("FacetGrid histplot");
            BaseChart c = g.render();
            checkRender(c, "18-df");
            Path p = out.resolve("18_facet_df.png");
            c.savefig(p.toString());
            checkPng(p, "18_df");
        });
        benchmark("18 FacetGrid viaNumpy-built DF", () -> {
            Seaborn.FacetGrid g = Seaborn.FacetGrid(groupValueDf(), "group");
            checkRender(g.mapHist("value", 20).render(), "18-np");
            Path p = out.resolve("18_facet_numpy.png");
            g.render().savefig(p.toString());
            checkPng(p, "18_np");
        });

        // 19 pointplot — DF primary
        benchmark("19 pointplot viaDataFrame", () -> {
            LineChart c = Seaborn.pointplot(timed, "t", "val", "grp");
            checkRender(c, "19-df");
            Path p = out.resolve("19_point_df.png");
            c.savefig(p.toString());
            checkPng(p, "19_df");
        });
        benchmark("19 pointplot viaNumpy-built DF", () -> {
            LineChart c = Seaborn.pointplot(timeDf(), "t", "val", "grp");
            checkRender(c, "19-np");
            Path p = out.resolve("19_point_numpy.png");
            c.savefig(p.toString());
            checkPng(p, "19_np");
        });

        // 20 ecdfplot
        benchmark("20 ecdfplot viaNumpy", () -> {
            LineChart c = Seaborn.ecdfplot(npNormal(500, 0, 1));
            checkRender(c, "20-np");
            Path p = out.resolve("20_ecdf_numpy.png");
            c.savefig(p.toString());
            checkPng(p, "20_np");
        });
        benchmark("20 ecdfplot viaDataFrame hue", () -> {
            LineChart c = Seaborn.ecdfplot(groupDf, "value", "group")
                .setTitle("Empirical Cumulative Distribution");
            checkRender(c, "20-df");
            Path p = out.resolve("20_ecdf_df.png");
            c.savefig(p.toString());
            checkPng(p, "20_df");
        });
        benchmark("20 ecdfplot viaTensor", () -> {
            LineChart c = Seaborn.ecdfplot(tensorNormal(500, 0, 1));
            checkRender(c, "20-t");
            Path p = out.resolve("20_ecdf_tensor.png");
            c.savefig(p.toString());
            checkPng(p, "20_t");
        });

        // ---- extras ----
        System.out.println("\n-- API surface extras --");
        benchmark("histplot DF+kde / barplot se-ci / joint scatter-reg", () -> {
            checkRender(Seaborn.histplot(groupDf, "value", 25, true), "hist-df-kde");
            checkRender(Seaborn.barplot(groupDf, "group", "value", "se"), "bar-se");
            checkRender(Seaborn.barplot(groupDf, "group", "value", "ci"), "bar-ci");
            checkRender(Seaborn.jointplot(scatter, "x", "y", "scatter"), "jp-sc");
            checkRender(Seaborn.jointplot(scatter, "x", "y", "reg"), "jp-reg");
        });
        benchmark("palette + cmap variants", () -> {
            for (String name : new String[]{"deep", "muted", "bright", "colorblind", "pastel", "dark"}) {
                Seaborn.set_palette(name);
                check("palette " + name, Seaborn.color_palette(name, 8).length == 8);
            }
            Seaborn.set_palette("deep");
            double[][] m = corrLike(5);
            for (String cmap : new String[]{"viridis", "plasma", "magma", "blues"}) {
                checkRender(Seaborn.heatmap(m, true, cmap, null, null).setTitle(cmap), "cmap-" + cmap);
            }
        });

        // ---- stress: MUST use NP for numpy path ----
        System.out.println("\n-- multi-dimensional stress (NP-backed) --");
        for (int N : new int[]{1_000, 10_000, 50_000}) {
            final int n = N;
            benchmark("stress histplot N=" + n + " viaNumpy", () -> {
                long t0 = System.nanoTime();
                HistogramChart c = Seaborn.histplot(npNormal(n, 0, 1), 50, n <= 10_000);
                BufferedImage img = c.render();
                long ms = (System.nanoTime() - t0) / 1_000_000;
                check("stress hist " + n + " w", img.getWidth() > 0);
                System.out.println("      hist N=" + n + " render: " + ms + " ms");
            });
            benchmark("stress histplot N=" + n + " viaTensor", () -> {
                HistogramChart c = Seaborn.histplot(tensorNormal(n, 0, 1), 50, false);
                check("stress hist-t " + n, c.render().getWidth() > 0);
            });
        }
        benchmark("stress kdeplot N=20k viaNumpy", () -> {
            long t0 = System.nanoTime();
            LineChart c = Seaborn.kdeplot(npNormal(20_000, 0, 1));
            checkRender(c, "stress-kde");
            long ms = (System.nanoTime() - t0) / 1_000_000;
            check("stress kde < 8s", ms < 8000);
            System.out.println("      kde 20k: " + ms + " ms");
        });
        benchmark("stress pairplot 4×2k viaNumpy→DF", () -> {
            NDArray m = NP.Random.randn(2000, 4);
            DataFrame big = Seaborn.dataframeFrom(m, "a", "b", "c", "d");
            long t0 = System.nanoTime();
            BaseChart c = Seaborn.pairplot(big);
            checkRender(c, "stress-pair");
            long ms = (System.nanoTime() - t0) / 1_000_000;
            check("stress pair < 15s", ms < 15_000);
            System.out.println("      pairplot 2k×4: " + ms + " ms");
            c.savefig(out.resolve("stress_pairplot.png").toString());
        });
        benchmark("stress concurrent 8-thread (NP data)", () -> {
            int threads = 8;
            ExecutorService pool = Executors.newFixedThreadPool(threads);
            AtomicInteger ok = new AtomicInteger();
            AtomicInteger err = new AtomicInteger();
            CountDownLatch latch = new CountDownLatch(threads);
            long t0 = System.nanoTime();
            for (int t = 0; t < threads; t++) {
                final int id = t;
                pool.submit(() -> {
                    try {
                        switch (id % 4) {
                            case 0 -> Seaborn.histplot(npNormal(5000, 0, 1), 40, true).render();
                            case 1 -> Seaborn.kdeplot(npNormal(3000, id, 1)).render();
                            case 2 -> Seaborn.scatterplot(NP.Random.randn(200), NP.Random.randn(200)).render();
                            default -> Seaborn.boxplot(new String[]{"A", "B"},
                                npNormal(100, 0, 1), npNormal(100, 1, 1)).render();
                        }
                        ok.incrementAndGet();
                    } catch (Throwable e) {
                        err.incrementAndGet();
                        e.printStackTrace(System.out);
                    } finally {
                        latch.countDown();
                    }
                });
            }
            check("concurrent await", latch.await(30, TimeUnit.SECONDS));
            pool.shutdownNow();
            long ms = (System.nanoTime() - t0) / 1_000_000;
            check("concurrent all ok", ok.get() == threads && err.get() == 0);
            System.out.println("      concurrent 8×render: " + ms + " ms  ok=" + ok.get());
        });
        benchmark("stress throughput 100 hist viaNumpy", () -> {
            long t0 = System.nanoTime();
            for (int i = 0; i < 100; i++) {
                Seaborn.histplot(npNormal(200, 0, 1), 15).setSize(320, 240).render();
            }
            long ms = (System.nanoTime() - t0) / 1_000_000;
            double per = ms / 100.0;
            check("throughput < 50ms/plot avg", per < 50);
            System.out.printf("      100×hist(200): total=%d ms  avg=%.2f ms/plot%n", ms, per);
        });

        System.out.println("\n=== Results: " + passed + " passed, " + failed + " failed ===");
        System.out.println("Artifacts: " + out.toAbsolutePath());
        System.out.println("Note: numeric data for ALL backends generated via NP.Random (no hand-rolled normal).");
        if (failed > 0) {
            System.out.println(report);
            System.exit(1);
        }
        System.out.println("All Seaborn tri-backend benchmarks passed.");
    }

    private static double[] flatten(double[][] m) {
        int rows = m.length, cols = rows == 0 ? 0 : m[0].length;
        double[] f = new double[rows * cols];
        for (int i = 0; i < rows; i++) System.arraycopy(m[i], 0, f, i * cols, cols);
        return f;
    }
}
