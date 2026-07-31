package dataframe;

import org.bytedeco.pytorch.dataframe.Column;
import org.bytedeco.pytorch.dataframe.DataFrame;
import org.bytedeco.pytorch.plot.chart.LineChart;
import org.bytedeco.pytorch.plot.matplot.Matplotlib;
import org.bytedeco.pytorch.plot.seaborn.Seaborn;

import java.nio.file.*;
import java.util.*;

/**
 * Plot package correctness suite (savefig only — no interactive show).
 *
 * <pre>
 *   java --add-opens=java.base/java.nio=ALL-UNNAMED -cp ... BenchmarkDataFramePlot
 * </pre>
 */
public class BenchmarkDataFramePlot {
    static int passed = 0, failed = 0;
    static final StringBuilder report = new StringBuilder();

    interface CheckedRunnable { void run() throws Exception; }

    static void benchmark(String name, CheckedRunnable r) {
        long t0 = System.nanoTime();
        try {
            r.run();
            long ms = (System.nanoTime() - t0) / 1_000_000;
            System.out.println("  OK  " + name + " (" + ms + " ms)");
        } catch (Throwable e) {
            failed++;
            long ms = (System.nanoTime() - t0) / 1_000_000;
            System.out.println(" FAIL " + name + " (" + ms + " ms): " + e);
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
        check(label + " non-empty (" + sz + " bytes)", sz > 100);
        // PNG magic
        byte[] head = Files.readAllBytes(p);
        check(label + " PNG magic", head.length >= 8
            && (head[0] & 0xFF) == 0x89
            && head[1] == 'P' && head[2] == 'N' && head[3] == 'G');
    }

    static DataFrame seed() {
        DataFrame df = DataFrame.create();
        df.addColumn("x", Column.DType.FLOAT64);
        df.addColumn("y", Column.DType.FLOAT64);
        df.addColumn("y2", Column.DType.FLOAT64);
        df.addColumn("cat", Column.DType.STRING);
        df.addColumn("score", Column.DType.FLOAT64);
        String[] cats = {"A", "B", "C"};
        for (int i = 0; i < 30; i++) {
            df.addRow((double) i, Math.sin(i / 5.0), Math.cos(i / 5.0),
                cats[i % 3], 5.0 + (i % 7));
        }
        return df;
    }

    public static void main(String[] args) throws Exception {
        System.out.println("=== BenchmarkDataFramePlot ===");
        Path tmp = Files.createTempDirectory("df-plot-");
        DataFrame df = seed();

        try {
            benchmark("1. line chart savefig", () -> {
                Path p = tmp.resolve("line.png");
                Matplotlib.plot(df, "x", "y", "y2")
                    .setTitle("Line")
                    .setXAxisLabel("x")
                    .setYAxisLabel("y")
                    .savefig(p.toString());
                checkPng(p, "line");
            });

            benchmark("2. scatter chart", () -> {
                Path p = tmp.resolve("scatter.png");
                Matplotlib.scatter(df, "x", "y")
                    .setTitle("Scatter")
                    .savefig(p.toString());
                checkPng(p, "scatter");
            });

            benchmark("3. bar chart", () -> {
                Path p = tmp.resolve("bar.png");
                // aggregate-ish small frame
                DataFrame b = DataFrame.create();
                b.addColumn("cat", Column.DType.STRING);
                b.addColumn("val", Column.DType.FLOAT64);
                b.addRow("A", 10.0);
                b.addRow("B", 20.0);
                b.addRow("C", 15.0);
                Matplotlib.bar(b, "cat", "val").setTitle("Bar").savefig(p.toString());
                checkPng(p, "bar");
            });

            benchmark("4. histogram", () -> {
                Path p = tmp.resolve("hist.png");
                Matplotlib.hist(df, "score", 8).setTitle("Hist").savefig(p.toString());
                checkPng(p, "hist");
            });

            benchmark("5. heatmap", () -> {
                Path p = tmp.resolve("heat.png");
                double[][] m = {
                    {1, 2, 3},
                    {4, 5, 6},
                    {7, 8, 9}
                };
                Matplotlib.heatmap(m, Arrays.asList("r1", "r2", "r3"),
                        Arrays.asList("c1", "c2", "c3"))
                    .setShowValues(true)
                    .setTitle("Heat")
                    .savefig(p.toString());
                checkPng(p, "heat");
            });

            benchmark("6. boxplot", () -> {
                Path p = tmp.resolve("box.png");
                Matplotlib.boxplot(df, "cat", "score").setTitle("Box").savefig(p.toString());
                checkPng(p, "box");
            });

            benchmark("7. df.plot() fluent API", () -> {
                Path p = tmp.resolve("dfplot.png");
                df.plot().line("x", "y").setTitle("df.plot").savefig(p.toString());
                checkPng(p, "dfplot");
            });

            benchmark("8. seaborn facade", () -> {
                Path p = tmp.resolve("sns.png");
                Seaborn.scatterplot(df, "x", "y").setTitle("sns").savefig(p.toString());
                checkPng(p, "sns");
            });

            benchmark("9. array API + last savefig", () -> {
                Path p = tmp.resolve("arr.png");
                double[] x = {0, 1, 2, 3, 4};
                double[] y = {0, 1, 4, 9, 16};
                Matplotlib.plot(x, y, "sq");
                Matplotlib.savefig(p.toString());
                checkPng(p, "arr");
            });

            benchmark("10. render returns image with expected size", () -> {
                LineChart c = Matplotlib.plot(df, "x", "y").setSize(640, 480);
                java.awt.image.BufferedImage img = c.render();
                check("width 640", img.getWidth() == 640);
                check("height 480", img.getHeight() == 480);
            });

        } finally {
            try {
                Files.walk(tmp).sorted(Comparator.reverseOrder()).forEach(p -> {
                    try { Files.deleteIfExists(p); } catch (Exception ignored) {}
                });
            } catch (Exception ignored) {}
        }

        System.out.println("\n=== Results: " + passed + " passed, " + failed + " failed ===");
        if (failed > 0) {
            System.out.println(report);
            System.exit(1);
        }
    }
}
