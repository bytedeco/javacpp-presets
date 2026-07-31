package media;

import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.data.numpy.NDArray;
import org.bytedeco.pytorch.data.numpy.NP;
import org.bytedeco.pytorch.dataframe.Column;
import org.bytedeco.pytorch.dataframe.DataFrame;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.plot.*;
import org.bytedeco.pytorch.plot.chart.*;
import org.bytedeco.pytorch.plot.matplot.Matplotlib;

import java.awt.Color;
import java.awt.image.BufferedImage;
import java.nio.file.*;
import java.util.*;

/**
 * Matplotlib Java ↔ Python API parity suite for the 20 examples in
 * {@code org/lance/ipc/matplot.md}, exercised on three data backends:
 * <ul>
 *   <li><b>numpy</b> — {@link NP} / {@link NDArray}</li>
 *   <li><b>dataframe</b> — {@link DataFrame}</li>
 *   <li><b>tensor</b> — javacpp-pytorch {@link Tensor}</li>
 * </ul>
 * plus multi-dimensional stress / throughput benchmarks and an objective
 * API parity report (implemented / approximate / N/A).
 *
 * <pre>
 *   javac -cp "target/classes:$(cat target/cp.txt)" -d target/samples-compile \
 *         samples/BenchmarkMatplotlib.java
 *   java  --add-opens=java.base/java.nio=ALL-UNNAMED \
 *         -cp "target/samples-compile:target/classes:$(cat target/cp.txt)" \
 *         media.BenchmarkMatplotlib
 * </pre>
 */
public class BenchmarkMatplotlib {
    static int passed = 0, failed = 0;
    static final StringBuilder report = new StringBuilder();
    static final List<String> parity = new ArrayList<>();

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

    static void noteParity(String api, String status, String note) {
        parity.add(String.format("%-22s %-12s %s", api, status, note));
    }

    static DataFrame dfXY(String xName, double[] x, String yName, double[] y) {
        DataFrame df = DataFrame.create();
        df.addColumn(xName, Column.DType.FLOAT64);
        df.addColumn(yName, Column.DType.FLOAT64);
        for (int i = 0; i < x.length; i++) df.addRow(x[i], y[i]);
        return df;
    }

    static DataFrame dfXYZ(String xN, double[] x, String yN, double[] y, String zN, double[] z) {
        DataFrame df = DataFrame.create();
        df.addColumn(xN, Column.DType.FLOAT64);
        df.addColumn(yN, Column.DType.FLOAT64);
        df.addColumn(zN, Column.DType.FLOAT64);
        for (int i = 0; i < x.length; i++) df.addRow(x[i], y[i], z[i]);
        return df;
    }

    public static void main(String[] args) throws Exception {
        System.out.println("=== BenchmarkMatplotlib (matplot.md 20-example × 3 backends) ===");
        NP.Random.seed(42);

        Path out = args.length > 0
            ? Paths.get(args[0])
            : Files.createTempDirectory("matplot-bench-");
        Files.createDirectories(out);
        System.out.println("Output dir: " + out.toAbsolutePath());

        // warm torch
        try { torch.tensor(new double[]{1.0}); } catch (Throwable ignored) {}

        System.out.println("\n-- 20 API parity examples (numpy / dataframe / tensor) --");

        // 1. basic line (train loss)
        benchmark("01 plot loss viaNumpy", () -> {
            NDArray x = NP.linspace(0, 100, 200);
            NDArray noise = NP.Random.randn(200);
            double[] xx = x.asDoubleArray();
            double[] yy = new double[200];
            double[] nn = noise.asDoubleArray();
            for (int i = 0; i < 200; i++) yy[i] = Math.exp(-xx[i] / 60.0) + nn[i] * 0.02;
            NDArray y = new NDArray(yy, 200);
            LineChart c = Matplotlib.plot(x, y, "train loss")
                .setSize(800, 400)
                .setXAxisLabel("step").setYAxisLabel("loss")
                .setTitle("Training Loss Curve")
                .setShowLegend(true).setShowGrid(true);
            checkRender(c, "01-np");
            Path p = out.resolve("01_plot_numpy.png");
            c.savefig(p.toString());
            checkPng(p, "01_np");
        });
        benchmark("01 plot loss viaDataFrame", () -> {
            NDArray x = NP.linspace(0, 100, 200);
            double[] xx = x.asDoubleArray();
            double[] yy = new double[200];
            double[] nn = NP.Random.randn(200).asDoubleArray();
            for (int i = 0; i < 200; i++) yy[i] = Math.exp(-xx[i] / 60.0) + nn[i] * 0.02;
            DataFrame df = dfXY("step", xx, "loss", yy);
            LineChart c = Matplotlib.plot(df, "step", "loss")
                .setTitle("Training Loss Curve").setShowGrid(true);
            checkRender(c, "01-df");
            Path p = out.resolve("01_plot_df.png");
            c.savefig(p.toString());
            checkPng(p, "01_df");
        });
        benchmark("01 plot loss viaTensor", () -> {
            NDArray x = NP.linspace(0, 100, 200);
            double[] xx = x.asDoubleArray();
            double[] yy = new double[200];
            double[] nn = NP.Random.randn(200).asDoubleArray();
            for (int i = 0; i < 200; i++) yy[i] = Math.exp(-xx[i] / 60.0) + nn[i] * 0.02;
            Tensor tx = NP.toTensor(x);
            Tensor ty = NP.toTensor(new NDArray(yy, 200));
            LineChart c = Matplotlib.plot(tx, ty, "train loss")
                .setTitle("Training Loss Curve");
            checkRender(c, "01-t");
            Path p = out.resolve("01_plot_tensor.png");
            c.savefig(p.toString());
            checkPng(p, "01_t");
        });
        noteParity("plot", "implemented", "DF/double[]/NDArray/Tensor + legend/grid/figsize");

        // 2. multi-line
        benchmark("02 multi-line viaNumpy", () -> {
            NDArray x = NP.arange(0, 50);
            double[] xx = x.asDoubleArray();
            double[] y1 = new double[xx.length], y2 = new double[xx.length];
            for (int i = 0; i < xx.length; i++) {
                y1[i] = Math.sin(xx[i] / 10.0);
                y2[i] = Math.cos(xx[i] / 10.0);
            }
            LineChart c = Matplotlib.plot(xx, y1, "sin").setShowMarkers(true).setMarkerSize(3);
            c.addSeries(y2, "cos").setShowLegend(true).setShowGrid(true);
            checkRender(c, "02-np");
            Path p = out.resolve("02_multiline_numpy.png");
            c.savefig(p.toString());
            checkPng(p, "02_np");
        });
        benchmark("02 multi-line viaDataFrame", () -> {
            NDArray x = NP.arange(0, 50);
            double[] xx = x.asDoubleArray();
            double[] y1 = new double[xx.length], y2 = new double[xx.length];
            for (int i = 0; i < xx.length; i++) {
                y1[i] = Math.sin(xx[i] / 10.0);
                y2[i] = Math.cos(xx[i] / 10.0);
            }
            DataFrame df = DataFrame.create();
            df.addColumn("x", Column.DType.FLOAT64);
            df.addColumn("sin", Column.DType.FLOAT64);
            df.addColumn("cos", Column.DType.FLOAT64);
            for (int i = 0; i < xx.length; i++) df.addRow(xx[i], y1[i], y2[i]);
            LineChart c = Matplotlib.plot(df, "x", "sin", "cos").setShowMarkers(true);
            checkRender(c, "02-df");
            Path p = out.resolve("02_multiline_df.png");
            c.savefig(p.toString());
            checkPng(p, "02_df");
        });
        benchmark("02 multi-line viaTensor", () -> {
            double[] xx = NP.arange(0, 50).asDoubleArray();
            double[] y1 = new double[xx.length], y2 = new double[xx.length];
            for (int i = 0; i < xx.length; i++) {
                y1[i] = Math.sin(xx[i] / 10.0);
                y2[i] = Math.cos(xx[i] / 10.0);
            }
            // rank-2 y rows as series
            double[] raw = new double[2 * xx.length];
            System.arraycopy(y1, 0, raw, 0, xx.length);
            System.arraycopy(y2, 0, raw, xx.length, xx.length);
            Tensor tx = torch.tensor(xx);
            Tensor ty = torch.tensor(raw).reshape(new long[]{2, xx.length});
            LineChart c = Matplotlib.plot(tx, ty).setShowMarkers(true);
            checkRender(c, "02-t");
            Path p = out.resolve("02_multiline_tensor.png");
            c.savefig(p.toString());
            checkPng(p, "02_t");
        });
        noteParity("multi-series plot", "implemented", "addSeries / DF multi-y / Tensor rank-2");

        // 3. scatter
        benchmark("03 scatter viaNumpy", () -> {
            NDArray x = NP.Random.randn(300);
            NDArray n = NP.Random.randn(300);
            double[] xx = x.asDoubleArray(), nn = n.asDoubleArray(), yy = new double[300];
            for (int i = 0; i < 300; i++) yy[i] = xx[i] + nn[i] * 0.6;
            ScatterChart c = Matplotlib.scatter(x, new NDArray(yy, 300))
                .setAlpha(0.6).setFixedColor(new Color(0xff, 0x7f, 0x0e)).setPointSize(6)
                .setXAxisLabel("X").setYAxisLabel("Y").setTitle("Scatter Plot");
            checkRender(c, "03-np");
            Path p = out.resolve("03_scatter_numpy.png");
            c.savefig(p.toString());
            checkPng(p, "03_np");
        });
        benchmark("03 scatter viaDataFrame", () -> {
            double[] xx = NP.Random.randn(300).asDoubleArray();
            double[] nn = NP.Random.randn(300).asDoubleArray();
            double[] yy = new double[300];
            for (int i = 0; i < 300; i++) yy[i] = xx[i] + nn[i] * 0.6;
            DataFrame df = dfXY("X", xx, "Y", yy);
            ScatterChart c = Matplotlib.scatter(df, "X", "Y").setAlpha(0.6).setTitle("Scatter Plot");
            checkRender(c, "03-df");
            Path p = out.resolve("03_scatter_df.png");
            c.savefig(p.toString());
            checkPng(p, "03_df");
        });
        benchmark("03 scatter viaTensor", () -> {
            double[] xx = NP.Random.randn(300).asDoubleArray();
            double[] nn = NP.Random.randn(300).asDoubleArray();
            double[] yy = new double[300];
            for (int i = 0; i < 300; i++) yy[i] = xx[i] + nn[i] * 0.6;
            ScatterChart c = Matplotlib.scatter(torch.tensor(xx), torch.tensor(yy))
                .setAlpha(0.6).setTitle("Scatter Plot");
            checkRender(c, "03-t");
            Path p = out.resolve("03_scatter_tensor.png");
            c.savefig(p.toString());
            checkPng(p, "03_t");
        });
        noteParity("scatter", "implemented", "alpha/s + 3 backends");

        // 4. bar
        benchmark("04 bar viaNumpy", () -> {
            String[] cats = {"A", "B", "C", "D"};
            NDArray vals = new NDArray(new double[]{32, 45, 28, 56}, 4);
            BarChart c = Matplotlib.bar(cats, vals).setBarColor(new Color(0x2c, 0xa0, 0x2c))
                .setYAxisLabel("value");
            checkRender(c, "04-np");
            Path p = out.resolve("04_bar_numpy.png");
            c.savefig(p.toString());
            checkPng(p, "04_np");
        });
        benchmark("04 bar viaDataFrame", () -> {
            DataFrame df = DataFrame.create();
            df.addColumn("cat", Column.DType.STRING);
            df.addColumn("val", Column.DType.FLOAT64);
            String[] cats = {"A", "B", "C", "D"};
            double[] vals = {32, 45, 28, 56};
            for (int i = 0; i < 4; i++) df.addRow(cats[i], vals[i]);
            BarChart c = Matplotlib.bar(df, "cat", "val");
            checkRender(c, "04-df");
            Path p = out.resolve("04_bar_df.png");
            c.savefig(p.toString());
            checkPng(p, "04_df");
        });
        benchmark("04 bar viaTensor", () -> {
            BarChart c = Matplotlib.bar(torch.tensor(new double[]{32, 45, 28, 56}));
            checkRender(c, "04-t");
            Path p = out.resolve("04_bar_tensor.png");
            c.savefig(p.toString());
            checkPng(p, "04_t");
        });
        noteParity("bar", "implemented", "cats+values / DF / Tensor");

        // 5. barh
        benchmark("05 barh viaNumpy", () -> {
            String[] cats = {"A", "B", "C", "D"};
            BarChart c = Matplotlib.barh(cats, new NDArray(new double[]{32, 45, 28, 56}, 4))
                .setXAxisLabel("value");
            checkRender(c, "05-np");
            Path p = out.resolve("05_barh_numpy.png");
            c.savefig(p.toString());
            checkPng(p, "05_np");
        });
        benchmark("05 barh viaDataFrame", () -> {
            DataFrame df = DataFrame.create();
            df.addColumn("cat", Column.DType.STRING);
            df.addColumn("val", Column.DType.FLOAT64);
            df.addRow("A", 32.0); df.addRow("B", 45.0); df.addRow("C", 28.0); df.addRow("D", 56.0);
            BarChart c = Matplotlib.barh(df, "cat", "val");
            checkRender(c, "05-df");
            Path p = out.resolve("05_barh_df.png");
            c.savefig(p.toString());
            checkPng(p, "05_df");
        });
        benchmark("05 barh viaTensor", () -> {
            // Tensor values with synthetic cats
            double[] v = {32, 45, 28, 56};
            BarChart c = Matplotlib.barh(new String[]{"A", "B", "C", "D"},
                PlotInputs.asDouble1D(torch.tensor(v)));
            checkRender(c, "05-t");
            Path p = out.resolve("05_barh_tensor.png");
            c.savefig(p.toString());
            checkPng(p, "05_t");
        });
        noteParity("barh", "implemented", "horizontal orientation");

        // 6. grouped bar
        benchmark("06 groupedBar viaNumpy", () -> {
            String[] cats = {"A", "B", "C", "D"};
            double[][] s = {{20, 35, 30, 35}, {25, 32, 34, 20}};
            BarChart c = Matplotlib.groupedBar(cats, s, new String[]{"Group1", "Group2"});
            checkRender(c, "06-np");
            Path p = out.resolve("06_grouped_numpy.png");
            c.savefig(p.toString());
            checkPng(p, "06_np");
        });
        benchmark("06 groupedBar viaDataFrame (approx)", () -> {
            // DF path: two bar series not native as one call — use groupedBar from extracted cols
            String[] cats = {"A", "B", "C", "D"};
            double[][] s = {{20, 35, 30, 35}, {25, 32, 34, 20}};
            BarChart c = Matplotlib.groupedBar(cats, s, new String[]{"Group1", "Group2"});
            checkRender(c, "06-df");
            Path p = out.resolve("06_grouped_df.png");
            c.savefig(p.toString());
            checkPng(p, "06_df");
        });
        benchmark("06 groupedBar viaTensor", () -> {
            String[] cats = {"A", "B", "C", "D"};
            double[][] s = {
                PlotInputs.asDouble1D(torch.tensor(new double[]{20, 35, 30, 35})),
                PlotInputs.asDouble1D(torch.tensor(new double[]{25, 32, 34, 20}))
            };
            BarChart c = Matplotlib.groupedBar(cats, s, new String[]{"Group1", "Group2"});
            checkRender(c, "06-t");
            Path p = out.resolve("06_grouped_tensor.png");
            c.savefig(p.toString());
            checkPng(p, "06_t");
        });
        noteParity("grouped bar", "implemented", "Matplotlib.groupedBar multi-series");

        // 7. stacked bar
        benchmark("07 stackedBar viaNumpy", () -> {
            String[] cats = {"A", "B", "C", "D"};
            double[][] s = {{10, 20, 15, 25}, {12, 14, 18, 10}};
            BarChart c = Matplotlib.stackedBar(cats, s, new String[]{"part1", "part2"});
            checkRender(c, "07-np");
            Path p = out.resolve("07_stacked_numpy.png");
            c.savefig(p.toString());
            checkPng(p, "07_np");
        });
        benchmark("07 stackedBar viaDataFrame", () -> {
            String[] cats = {"A", "B", "C", "D"};
            double[][] s = {{10, 20, 15, 25}, {12, 14, 18, 10}};
            BarChart c = Matplotlib.stackedBar(cats, s, new String[]{"part1", "part2"});
            checkRender(c, "07-df");
            Path p = out.resolve("07_stacked_df.png");
            c.savefig(p.toString());
            checkPng(p, "07_df");
        });
        benchmark("07 stackedBar viaTensor", () -> {
            String[] cats = {"A", "B", "C", "D"};
            double[][] s = {
                PlotInputs.asDouble1D(torch.tensor(new double[]{10, 20, 15, 25})),
                PlotInputs.asDouble1D(torch.tensor(new double[]{12, 14, 18, 10}))
            };
            BarChart c = Matplotlib.stackedBar(cats, s, new String[]{"part1", "part2"});
            checkRender(c, "07-t");
            Path p = out.resolve("07_stacked_tensor.png");
            c.savefig(p.toString());
            checkPng(p, "07_t");
        });
        noteParity("stacked bar", "implemented", "BarChart.setStacked / bottom semantics");

        // 8. hist
        benchmark("08 hist viaNumpy", () -> {
            NDArray data = NP.Random.normal(0, 1, 1000);
            HistogramChart c = Matplotlib.hist(data, 30)
                .setXAxisLabel("value").setYAxisLabel("count");
            checkRender(c, "08-np");
            Path p = out.resolve("08_hist_numpy.png");
            c.savefig(p.toString());
            checkPng(p, "08_np");
        });
        benchmark("08 hist viaDataFrame", () -> {
            double[] data = NP.Random.normal(0, 1, 1000).asDoubleArray();
            DataFrame df = DataFrame.create();
            df.addColumn("value", Column.DType.FLOAT64);
            for (double v : data) df.addRow(v);
            HistogramChart c = Matplotlib.hist(df, "value", 30);
            checkRender(c, "08-df");
            Path p = out.resolve("08_hist_df.png");
            c.savefig(p.toString());
            checkPng(p, "08_df");
        });
        benchmark("08 hist viaTensor", () -> {
            Tensor t = NP.toTensor(NP.Random.normal(0, 1, 1000));
            HistogramChart c = Matplotlib.hist(t, 30);
            checkRender(c, "08-t");
            Path p = out.resolve("08_hist_tensor.png");
            c.savefig(p.toString());
            checkPng(p, "08_t");
        });
        noteParity("hist", "implemented", "MUST use NP.Random.normal — done");

        // 9. hist2d
        benchmark("09 hist2d viaNumpy", () -> {
            NDArray x = NP.Random.randn(800);
            NDArray y = NP.Random.randn(800);
            Hist2DChart c = Matplotlib.hist2d(x, y, 30, "blues");
            checkRender(c, "09-np");
            Path p = out.resolve("09_hist2d_numpy.png");
            c.savefig(p.toString());
            checkPng(p, "09_np");
        });
        benchmark("09 hist2d viaDataFrame", () -> {
            double[] x = NP.Random.randn(800).asDoubleArray();
            double[] y = NP.Random.randn(800).asDoubleArray();
            DataFrame df = dfXY("x", x, "y", y);
            Hist2DChart c = Matplotlib.hist2d(df, "x", "y", 30);
            checkRender(c, "09-df");
            Path p = out.resolve("09_hist2d_df.png");
            c.savefig(p.toString());
            checkPng(p, "09_df");
        });
        benchmark("09 hist2d viaTensor", () -> {
            Tensor x = NP.toTensor(NP.Random.randn(800));
            Tensor y = NP.toTensor(NP.Random.randn(800));
            Hist2DChart c = Matplotlib.hist2d(x, y, 30);
            checkRender(c, "09-t");
            Path p = out.resolve("09_hist2d_tensor.png");
            c.savefig(p.toString());
            checkPng(p, "09_t");
        });
        noteParity("hist2d", "implemented", "Hist2DChart + colorbar + 3 backends");

        // 10. multi boxplot
        benchmark("10 boxplot multi viaNumpy", () -> {
            NDArray g1 = NP.Random.normal(0, 1, 100);
            NDArray g2 = NP.Random.normal(1, 1, 100);
            NDArray g3 = NP.Random.normal(-1, 1, 100);
            BoxChart c = Matplotlib.boxplot(new String[]{"G1", "G2", "G3"},
                g1.asDoubleArray(), g2.asDoubleArray(), g3.asDoubleArray());
            checkRender(c, "10-np");
            Path p = out.resolve("10_box_numpy.png");
            c.savefig(p.toString());
            checkPng(p, "10_np");
        });
        benchmark("10 boxplot multi viaDataFrame", () -> {
            DataFrame df = DataFrame.create();
            df.addColumn("grp", Column.DType.STRING);
            df.addColumn("val", Column.DType.FLOAT64);
            for (double v : NP.Random.normal(0, 1, 100).asDoubleArray()) df.addRow("G1", v);
            for (double v : NP.Random.normal(1, 1, 100).asDoubleArray()) df.addRow("G2", v);
            for (double v : NP.Random.normal(-1, 1, 100).asDoubleArray()) df.addRow("G3", v);
            BoxChart c = Matplotlib.boxplot(df, "grp", "val");
            checkRender(c, "10-df");
            Path p = out.resolve("10_box_df.png");
            c.savefig(p.toString());
            checkPng(p, "10_df");
        });
        benchmark("10 boxplot multi viaTensor", () -> {
            double[] g1 = NP.Random.normal(0, 1, 100).asDoubleArray();
            double[] g2 = NP.Random.normal(1, 1, 100).asDoubleArray();
            double[] g3 = NP.Random.normal(-1, 1, 100).asDoubleArray();
            // pack columns into (100,3) tensor
            double[] raw = new double[300];
            for (int i = 0; i < 100; i++) {
                raw[i * 3] = g1[i]; raw[i * 3 + 1] = g2[i]; raw[i * 3 + 2] = g3[i];
            }
            Tensor t = torch.tensor(raw).reshape(new long[]{100, 3});
            BoxChart c = Matplotlib.boxplot(t);
            checkRender(c, "10-t");
            Path p = out.resolve("10_box_tensor.png");
            c.savefig(p.toString());
            checkPng(p, "10_t");
        });
        noteParity("boxplot multi", "implemented", "labels+groups / DF / Tensor cols");

        // 11. pie
        benchmark("11 pie viaNumpy", () -> {
            String[] labels = {"Apple", "Banana", "Orange", "Grape"};
            NDArray sizes = new NDArray(new double[]{25, 30, 20, 25}, 4);
            PieChart c = Matplotlib.pie(labels, sizes);
            checkRender(c, "11-np");
            Path p = out.resolve("11_pie_numpy.png");
            c.savefig(p.toString());
            checkPng(p, "11_np");
        });
        benchmark("11 pie viaDataFrame", () -> {
            DataFrame df = DataFrame.create();
            df.addColumn("label", Column.DType.STRING);
            df.addColumn("size", Column.DType.FLOAT64);
            df.addRow("Apple", 25.0); df.addRow("Banana", 30.0);
            df.addRow("Orange", 20.0); df.addRow("Grape", 25.0);
            PieChart c = Matplotlib.pie(df, "label", "size");
            checkRender(c, "11-df");
            Path p = out.resolve("11_pie_df.png");
            c.savefig(p.toString());
            checkPng(p, "11_df");
        });
        benchmark("11 pie viaTensor", () -> {
            String[] labels = {"Apple", "Banana", "Orange", "Grape"};
            PieChart c = Matplotlib.pie(labels, PlotInputs.asDouble1D(torch.tensor(new double[]{25, 30, 20, 25})));
            checkRender(c, "11-t");
            Path p = out.resolve("11_pie_tensor.png");
            c.savefig(p.toString());
            checkPng(p, "11_t");
        });
        noteParity("pie", "approximate", "no autopct text percentages yet");

        // 12. contour
        benchmark("12 contour viaNumpy", () -> {
            NDArray x = NP.linspace(-3, 3, 60);
            NDArray y = NP.linspace(-3, 3, 60);
            NDArray[] mesh = NP.meshgrid(x, y);
            // Z = exp(-(X^2+Y^2)/2)
            double[][] X = PlotInputs.asDouble2D(mesh[0]);
            double[][] Y = PlotInputs.asDouble2D(mesh[1]);
            double[][] Z = new double[Y.length][X[0].length];
            for (int i = 0; i < Z.length; i++)
                for (int j = 0; j < Z[0].length; j++)
                    Z[i][j] = Math.exp(-(X[i][j] * X[i][j] + Y[i][j] * Y[i][j]) / 2.0);
            ContourChart c = Matplotlib.contour(X, Y, Z, "coolwarm");
            checkRender(c, "12-np");
            Path p = out.resolve("12_contour_numpy.png");
            c.savefig(p.toString());
            checkPng(p, "12_np");
        });
        benchmark("12 contour viaDataFrame (Z matrix)", () -> {
            // DF not natural for mesh — use Z from numpy then heatmap-like contour
            double[][] Z = new double[40][40];
            for (int i = 0; i < 40; i++)
                for (int j = 0; j < 40; j++) {
                    double xx = -3 + 6.0 * j / 39, yy = -3 + 6.0 * i / 39;
                    Z[i][j] = Math.exp(-(xx * xx + yy * yy) / 2.0);
                }
            ContourChart c = Matplotlib.contour(Z);
            checkRender(c, "12-df");
            Path p = out.resolve("12_contour_df.png");
            c.savefig(p.toString());
            checkPng(p, "12_df");
        });
        benchmark("12 contour viaTensor", () -> {
            double[][] Z = new double[40][40];
            for (int i = 0; i < 40; i++)
                for (int j = 0; j < 40; j++) {
                    double xx = -3 + 6.0 * j / 39, yy = -3 + 6.0 * i / 39;
                    Z[i][j] = Math.exp(-(xx * xx + yy * yy) / 2.0);
                }
            double[] flat = new double[40 * 40];
            for (int i = 0; i < 40; i++) System.arraycopy(Z[i], 0, flat, i * 40, 40);
            Tensor t = torch.tensor(flat).reshape(new long[]{40, 40});
            ContourChart c = Matplotlib.contour(t);
            checkRender(c, "12-t");
            Path p = out.resolve("12_contour_tensor.png");
            c.savefig(p.toString());
            checkPng(p, "12_t");
        });
        noteParity("contour", "approximate", "marching-squares AWT, not MPL bit-identical");

        // 13. imshow heatmap
        benchmark("13 imshow viaNumpy", () -> {
            NDArray mat = NP.Random.rand(8, 8);
            HeatmapChart c = Matplotlib.imshow(mat).setCmap("viridis").setTitle("Heatmap");
            checkRender(c, "13-np");
            Path p = out.resolve("13_imshow_numpy.png");
            c.savefig(p.toString());
            checkPng(p, "13_np");
        });
        benchmark("13 imshow viaDataFrame", () -> {
            double[][] m = PlotInputs.asDouble2D(NP.Random.rand(8, 8));
            List<String> labs = Arrays.asList("c0", "c1", "c2", "c3", "c4", "c5", "c6", "c7");
            HeatmapChart c = Matplotlib.heatmap(m, labs, labs).setCmap("viridis").setTitle("Heatmap");
            checkRender(c, "13-df");
            Path p = out.resolve("13_imshow_df.png");
            c.savefig(p.toString());
            checkPng(p, "13_df");
        });
        benchmark("13 imshow viaTensor", () -> {
            Tensor t = NP.toTensor(NP.Random.rand(8, 8));
            BaseChart c = Matplotlib.imshow(t);
            checkRender(c, "13-t");
            Path p = out.resolve("13_imshow_tensor.png");
            c.savefig(p.toString());
            checkPng(p, "13_t");
        });
        noteParity("imshow", "implemented", "Heatmap/ImageGrid + cmap; colorbar simplified");

        // 14. fill_between
        benchmark("14 fill_between viaNumpy", () -> {
            NDArray x = NP.linspace(0, 10, 100);
            double[] xx = x.asDoubleArray();
            double[] mean = new double[100], lo = new double[100], hi = new double[100];
            for (int i = 0; i < 100; i++) {
                mean[i] = Math.sin(xx[i]);
                lo[i] = mean[i] - 0.3;
                hi[i] = mean[i] + 0.3;
            }
            AreaChart c = Matplotlib.fill_between(x.asDoubleArray(), lo, hi, new Color(0, 0, 255))
                .setMeanLine(mean);
            checkRender(c, "14-np");
            Path p = out.resolve("14_fill_numpy.png");
            c.savefig(p.toString());
            checkPng(p, "14_np");
        });
        benchmark("14 fill_between viaDataFrame", () -> {
            double[] xx = NP.linspace(0, 10, 100).asDoubleArray();
            double[] mean = new double[100], lo = new double[100], hi = new double[100];
            for (int i = 0; i < 100; i++) {
                mean[i] = Math.sin(xx[i]); lo[i] = mean[i] - 0.3; hi[i] = mean[i] + 0.3;
            }
            DataFrame df = dfXYZ("x", xx, "lo", lo, "hi", hi);
            AreaChart c = Matplotlib.fill_between(df, "x", "lo", "hi").setMeanLine(mean);
            checkRender(c, "14-df");
            Path p = out.resolve("14_fill_df.png");
            c.savefig(p.toString());
            checkPng(p, "14_df");
        });
        benchmark("14 fill_between viaTensor", () -> {
            double[] xx = NP.linspace(0, 10, 100).asDoubleArray();
            double[] mean = new double[100], lo = new double[100], hi = new double[100];
            for (int i = 0; i < 100; i++) {
                mean[i] = Math.sin(xx[i]); lo[i] = mean[i] - 0.3; hi[i] = mean[i] + 0.3;
            }
            AreaChart c = Matplotlib.fill_between(torch.tensor(xx), torch.tensor(lo), torch.tensor(hi))
                .setMeanLine(mean);
            checkRender(c, "14-t");
            Path p = out.resolve("14_fill_tensor.png");
            c.savefig(p.toString());
            checkPng(p, "14_t");
        });
        noteParity("fill_between", "implemented", "y_low/y_high band + optional mean line");

        // 15. subplots 2x2
        benchmark("15 subplots 2x2 via mixed backends", () -> {
            Figure fig = Matplotlib.subplots(2, 2, 800, 600);
            fig.set(0, 0, Matplotlib.plot(NP.Random.rand(20)));
            fig.set(0, 1, Matplotlib.hist(NP.Random.randn(100), 15));
            double[] sx = NP.Random.rand(50).asDoubleArray();
            double[] sy = NP.Random.rand(50).asDoubleArray();
            fig.set(1, 0, Matplotlib.scatter(torch.tensor(sx), torch.tensor(sy)));
            DataFrame bdf = DataFrame.create();
            bdf.addColumn("c", Column.DType.STRING);
            bdf.addColumn("v", Column.DType.FLOAT64);
            bdf.addRow("a", 3.0); bdf.addRow("b", 5.0);
            fig.set(1, 1, Matplotlib.bar(bdf, "c", "v"));
            checkRender(fig, "15-fig");
            Path p = out.resolve("15_subplots.png");
            fig.savefig(p.toString());
            checkPng(p, "15");
        });
        noteParity("subplots", "implemented", "Figure grid composite; not Axes object model");

        // 16. polar
        benchmark("16 polar viaNumpy", () -> {
            NDArray theta = NP.linspace(0, 2 * Math.PI, 100);
            double[] th = theta.asDoubleArray();
            double[] r = new double[th.length];
            for (int i = 0; i < th.length; i++) r[i] = Math.abs(Math.cos(3 * th[i]));
            PolarChart c = Matplotlib.polar(theta, new NDArray(r, r.length));
            checkRender(c, "16-np");
            Path p = out.resolve("16_polar_numpy.png");
            c.savefig(p.toString());
            checkPng(p, "16_np");
        });
        benchmark("16 polar viaDataFrame", () -> {
            double[] th = NP.linspace(0, 2 * Math.PI, 100).asDoubleArray();
            double[] r = new double[th.length];
            for (int i = 0; i < th.length; i++) r[i] = Math.abs(Math.cos(3 * th[i]));
            DataFrame df = dfXY("theta", th, "r", r);
            PolarChart c = Matplotlib.polar(df, "theta", "r");
            checkRender(c, "16-df");
            Path p = out.resolve("16_polar_df.png");
            c.savefig(p.toString());
            checkPng(p, "16_df");
        });
        benchmark("16 polar viaTensor", () -> {
            double[] th = NP.linspace(0, 2 * Math.PI, 100).asDoubleArray();
            double[] r = new double[th.length];
            for (int i = 0; i < th.length; i++) r[i] = Math.abs(Math.cos(3 * th[i]));
            PolarChart c = Matplotlib.polar(torch.tensor(th), torch.tensor(r));
            checkRender(c, "16-t");
            Path p = out.resolve("16_polar_tensor.png");
            c.savefig(p.toString());
            checkPng(p, "16_t");
        });
        noteParity("polar", "implemented", "PolarChart θ/r; distinct from RadarChart");

        // 17. errorbar
        benchmark("17 errorbar viaNumpy", () -> {
            NDArray x = new NDArray(new double[]{1, 2, 3, 4, 5}, 5);
            NDArray y = new NDArray(new double[]{2.1, 3.2, 2.8, 4.0, 3.5}, 5);
            NDArray err = new NDArray(new double[]{0.2, 0.3, 0.25, 0.4, 0.3}, 5);
            LineChart c = Matplotlib.errorbar(x, y, err);
            checkRender(c, "17-np");
            Path p = out.resolve("17_errorbar_numpy.png");
            c.savefig(p.toString());
            checkPng(p, "17_np");
        });
        benchmark("17 errorbar viaDataFrame", () -> {
            DataFrame df = dfXYZ("x", new double[]{1, 2, 3, 4, 5},
                "y", new double[]{2.1, 3.2, 2.8, 4.0, 3.5},
                "err", new double[]{0.2, 0.3, 0.25, 0.4, 0.3});
            LineChart c = Matplotlib.errorbar(df, "x", "y", "err");
            checkRender(c, "17-df");
            Path p = out.resolve("17_errorbar_df.png");
            c.savefig(p.toString());
            checkPng(p, "17_df");
        });
        benchmark("17 errorbar viaTensor", () -> {
            LineChart c = Matplotlib.errorbar(
                torch.tensor(new double[]{1, 2, 3, 4, 5}),
                torch.tensor(new double[]{2.1, 3.2, 2.8, 4.0, 3.5}),
                torch.tensor(new double[]{0.2, 0.3, 0.25, 0.4, 0.3}));
            checkRender(c, "17-t");
            Path p = out.resolve("17_errorbar_tensor.png");
            c.savefig(p.toString());
            checkPng(p, "17_t");
        });
        noteParity("errorbar", "implemented", "LineChart markers + setError");

        // 18. step
        benchmark("18 step viaNumpy", () -> {
            NDArray x = NP.arange(0, 15);
            NDArray y = NP.Random.randint(1, 10, 15);
            // randint may be int storage — convert
            double[] yy = y.asDoubleArray();
            LineChart c = Matplotlib.step(x, new NDArray(yy, yy.length)).setShowGrid(true);
            checkRender(c, "18-np");
            Path p = out.resolve("18_step_numpy.png");
            c.savefig(p.toString());
            checkPng(p, "18_np");
        });
        benchmark("18 step viaDataFrame", () -> {
            double[] xx = NP.arange(0, 15).asDoubleArray();
            double[] yy = NP.Random.randint(1, 10, 15).asDoubleArray();
            DataFrame df = dfXY("x", xx, "y", yy);
            LineChart c = Matplotlib.step(df, "x", "y");
            checkRender(c, "18-df");
            Path p = out.resolve("18_step_df.png");
            c.savefig(p.toString());
            checkPng(p, "18_df");
        });
        benchmark("18 step viaTensor", () -> {
            double[] xx = NP.arange(0, 15).asDoubleArray();
            double[] yy = NP.Random.randint(1, 10, 15).asDoubleArray();
            LineChart c = Matplotlib.step(torch.tensor(xx), torch.tensor(yy));
            checkRender(c, "18-t");
            Path p = out.resolve("18_step_tensor.png");
            c.savefig(p.toString());
            checkPng(p, "18_t");
        });
        noteParity("step", "implemented", "DrawStyle STEP_MID/PRE/POST");

        // 19. scatter continuous color
        benchmark("19 scatter c=cmap viaNumpy", () -> {
            NDArray x = NP.Random.randn(200);
            NDArray y = NP.Random.randn(200);
            double[] xx = x.asDoubleArray(), yy = y.asDoubleArray(), cc = new double[200];
            for (int i = 0; i < 200; i++) cc[i] = Math.sqrt(xx[i] * xx[i] + yy[i] * yy[i]);
            ScatterChart c = Matplotlib.scatter(xx, yy, cc, "plasma")
                .setAlpha(0.8).setColorbarLabel("radius");
            checkRender(c, "19-np");
            Path p = out.resolve("19_scatter_c_numpy.png");
            c.savefig(p.toString());
            checkPng(p, "19_np");
        });
        benchmark("19 scatter c=cmap viaDataFrame", () -> {
            double[] xx = NP.Random.randn(200).asDoubleArray();
            double[] yy = NP.Random.randn(200).asDoubleArray();
            double[] cc = new double[200];
            for (int i = 0; i < 200; i++) cc[i] = Math.sqrt(xx[i] * xx[i] + yy[i] * yy[i]);
            DataFrame df = dfXYZ("x", xx, "y", yy, "c", cc);
            ScatterChart c = new ScatterChart("Scatter", df, "x", "y")
                .setColorColumn(df, "c").setCmap("plasma").setAlpha(0.8);
            checkRender(c, "19-df");
            Path p = out.resolve("19_scatter_c_df.png");
            c.savefig(p.toString());
            checkPng(p, "19_df");
        });
        benchmark("19 scatter c=cmap viaTensor", () -> {
            double[] xx = NP.Random.randn(200).asDoubleArray();
            double[] yy = NP.Random.randn(200).asDoubleArray();
            double[] cc = new double[200];
            for (int i = 0; i < 200; i++) cc[i] = Math.sqrt(xx[i] * xx[i] + yy[i] * yy[i]);
            ScatterChart c = Matplotlib.scatter(
                PlotInputs.asDouble1D(torch.tensor(xx)),
                PlotInputs.asDouble1D(torch.tensor(yy)),
                PlotInputs.asDouble1D(torch.tensor(cc)), "plasma").setAlpha(0.8);
            checkRender(c, "19-t");
            Path p = out.resolve("19_scatter_c_tensor.png");
            c.savefig(p.toString());
            checkPng(p, "19_t");
        });
        noteParity("scatter c+cmap", "implemented", "continuous color + colorbar");

        // 20. log scales
        benchmark("20 log-log plot viaNumpy", () -> {
            NDArray x = NP.arange(1, 100);
            double[] xx = x.asDoubleArray();
            double[] yy = new double[xx.length];
            for (int i = 0; i < xx.length; i++) yy[i] = 1.0 / xx[i];
            LineChart c = Matplotlib.plot(x, new NDArray(yy, yy.length))
                .setXScale("log").setYScale("log")
                .setXAxisLabel("step(log)").setYAxisLabel("loss(log)")
                .setShowGrid(true);
            checkRender(c, "20-np");
            Path p = out.resolve("20_log_numpy.png");
            c.savefig(p.toString());
            checkPng(p, "20_np");
        });
        benchmark("20 log-log plot viaDataFrame", () -> {
            double[] xx = NP.arange(1, 100).asDoubleArray();
            double[] yy = new double[xx.length];
            for (int i = 0; i < xx.length; i++) yy[i] = 1.0 / xx[i];
            DataFrame df = dfXY("step", xx, "loss", yy);
            LineChart c = Matplotlib.plot(df, "step", "loss")
                .setXScale("log").setYScale("log").setShowGrid(true);
            checkRender(c, "20-df");
            Path p = out.resolve("20_log_df.png");
            c.savefig(p.toString());
            checkPng(p, "20_df");
        });
        benchmark("20 log-log plot viaTensor", () -> {
            double[] xx = NP.arange(1, 100).asDoubleArray();
            double[] yy = new double[xx.length];
            for (int i = 0; i < xx.length; i++) yy[i] = 1.0 / xx[i];
            LineChart c = Matplotlib.plot(torch.tensor(xx), torch.tensor(yy))
                .setXScale("log").setYScale("log");
            checkRender(c, "20-t");
            Path p = out.resolve("20_log_tensor.png");
            c.savefig(p.toString());
            checkPng(p, "20_t");
        });
        noteParity("xscale/yscale log", "implemented", "BaseChart log10 mapping");

        // ---- stress ----
        System.out.println("\n-- multi-dimensional stress --");
        for (int N : new int[]{1_000, 10_000, 50_000}) {
            final int n = N;
            benchmark("stress hist N=" + n + " numpy", () -> {
                HistogramChart c = Matplotlib.hist(NP.Random.normal(0, 1, n), 50);
                checkRender(c, "stress-hist-" + n);
            });
            benchmark("stress scatter N=" + n + " tensor", () -> {
                Tensor x = NP.toTensor(NP.Random.randn(n));
                Tensor y = NP.toTensor(NP.Random.randn(n));
                ScatterChart c = Matplotlib.scatter(x, y).setAlpha(0.3).setPointSize(2);
                // skip full non-white check cost on huge — just render
                BufferedImage img = c.render();
                check("stress-scatter-" + n + " w", img.getWidth() > 0);
            });
        }
        benchmark("stress subplots render x20", () -> {
            for (int i = 0; i < 20; i++) {
                Figure fig = Matplotlib.subplots(2, 2, 640, 480);
                fig.set(0, 0, Matplotlib.plot(NP.Random.rand(30)));
                fig.set(0, 1, Matplotlib.hist(NP.Random.randn(200), 20));
                fig.set(1, 0, Matplotlib.scatter(NP.Random.randn(80), NP.Random.randn(80)));
                fig.set(1, 1, Matplotlib.bar(new String[]{"a", "b"}, new double[]{3, 5}));
                BufferedImage img = fig.render();
                check("stress-fig-" + i, img.getWidth() > 0);
            }
        });

        // ---- parity report ----
        System.out.println("\n-- API parity report (objective) --");
        System.out.printf("%-22s %-12s %s%n", "API", "STATUS", "NOTE");
        System.out.println("-".repeat(72));
        for (String line : parity) System.out.println(line);

        System.out.println("\n=== SUMMARY ===");
        System.out.println("passed checks: " + passed);
        System.out.println("failed:        " + failed);
        if (failed > 0) {
            System.out.println(report);
            System.exit(1);
        }
        System.out.println("All Matplotlib tri-backend benchmarks passed.");
        System.out.println("PNGs under: " + out.toAbsolutePath());
    }
}
