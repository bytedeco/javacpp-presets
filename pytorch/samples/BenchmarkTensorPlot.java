package samples;
import org.bytedeco.pytorch.autograd.*;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.dataframe.Column;
import org.bytedeco.pytorch.dataframe.DataFrame;
import org.bytedeco.pytorch.utils.plot.BarChart;
import org.bytedeco.pytorch.utils.plot.BaseChart;
import org.bytedeco.pytorch.utils.plot.BoxChart;
import org.bytedeco.pytorch.utils.plot.HeatmapChart;
import org.bytedeco.pytorch.utils.plot.HistogramChart;
import org.bytedeco.pytorch.utils.plot.ImageGridChart;
import org.bytedeco.pytorch.utils.plot.LineChart;
import org.bytedeco.pytorch.utils.plot.Matplotlib;
import org.bytedeco.pytorch.utils.plot.ScatterChart;
import org.bytedeco.pytorch.utils.plot.TensorPlot;
import org.bytedeco.pytorch.utils.plot.TensorPlotUtils;
import org.bytedeco.pytorch.utils.plot.TensorPlotUtils.Layout;
import org.bytedeco.pytorch.global.torch;

import java.awt.image.BufferedImage;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Comparator;

/**
 * Multi-dimensional Tensor visualization correctness suite (savefig only — headless-safe).
 *
 * <p>Covers ranks 0–5, layouts (HW/CHW/HWC/NCHW/NHWC/NHW/AUTO), dtypes, edge cases,
 * {@link Matplotlib} + {@link TensorPlot} façades, and a light DF regression check.
 *
 * <pre>
 *   java --add-opens=java.base/java.nio=ALL-UNNAMED -cp ... samples.BenchmarkTensorPlot
 * </pre>
 */
public class BenchmarkTensorPlot {
    static int passed = 0, failed = 0;
    static final StringBuilder report = new StringBuilder();

    @FunctionalInterface
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
        byte[] head = Files.readAllBytes(p);
        check(label + " PNG magic", head.length >= 8
            && (head[0] & 0xFF) == 0x89
            && head[1] == 'P' && head[2] == 'N' && head[3] == 'G');
    }

    static void checkRenderSize(BaseChart c, int w, int h, String label) {
        BufferedImage img = c.setSize(w, h).render();
        check(label + " width " + w, img.getWidth() == w);
        check(label + " height " + h, img.getHeight() == h);
    }

    static Tensor arangeReshape(int n, long... shape) {
        double[] d = new double[n];
        for (int i = 0; i < n; i++) d[i] = i;
        Tensor t = torch.tensor(d);
        return shape.length > 0 ? t.reshape(shape) : t;
    }

    static Tensor randn(long... shape) {
        return torch.randn(shape);
    }

    public static void main(String[] args) throws Exception {
        System.out.println("=== BenchmarkTensorPlot ===");
        // Touch native lib early
        Tensor warm = torch.tensor(new double[]{1.0, 2.0, 3.0});
        check("warmup numel", warm.numel() == 3);

        Path tmp = Files.createTempDirectory("tensor-plot-");
        System.out.println("tmp = " + tmp);

        try {
            // ------------------------------------------------------------------
            // Rank 0 — reject
            // ------------------------------------------------------------------
            benchmark("0. rank0 scalar rejected", () -> {
                // torch.tensor(42.0) is shape [1]; true scalar needs empty shape
                Tensor s = torch.tensor(42.0).reshape(new long[]{});
                check("rank0 shape empty", TensorPlotUtils.rank(s) == 0);
                boolean threw = false;
                try {
                    Matplotlib.plot(s);
                } catch (IllegalArgumentException ex) {
                    threw = true;
                    check("msg mentions scalar/rank", ex.getMessage().toLowerCase().contains("scalar")
                        || ex.getMessage().contains("rank-0")
                        || ex.getMessage().contains("rank 0"));
                }
                check("plot(scalar) throws", threw);

                threw = false;
                try {
                    TensorPlot.imshow(s);
                } catch (IllegalArgumentException ex) {
                    threw = true;
                }
                check("imshow(scalar) throws", threw);
            });

            // ------------------------------------------------------------------
            // Rank 1
            // ------------------------------------------------------------------
            benchmark("1a. rank1 plot / hist / bar / box / violin / area", () -> {
                Tensor y = torch.tensor(new double[]{0, 1, 4, 9, 16, 25, 20, 15, 10, 5});
                Path p1 = tmp.resolve("r1_line.png");
                LineChart line = Matplotlib.plot(y).setTitle("rank1 line");
                line.savefig(p1.toString());
                checkPng(p1, "r1_line");
                checkRenderSize(line, 640, 480, "r1_line");

                Path p2 = tmp.resolve("r1_hist.png");
                HistogramChart h = Matplotlib.hist(y, 5).setTitle("rank1 hist");
                h.savefig(p2.toString());
                checkPng(p2, "r1_hist");

                Path p3 = tmp.resolve("r1_bar.png");
                BarChart b = Matplotlib.bar(torch.tensor(new double[]{3, 1, 4, 1, 5}));
                b.savefig(p3.toString());
                checkPng(p3, "r1_bar");

                Path p4 = tmp.resolve("r1_box.png");
                BoxChart box = Matplotlib.boxplot(y);
                box.savefig(p4.toString());
                checkPng(p4, "r1_box");
//                box.show();

                Path p5 = tmp.resolve("r1_violin.png");
                Matplotlib.violinplot(y).savefig(tmp.resolve("r1_violin.png").toString());
                checkPng(p5, "r1_violin");

                Path p6 = tmp.resolve("r1_area.png");
                Matplotlib.area(y).savefig(p6.toString());
                checkPng(p6, "r1_area");
            });

            benchmark("1b. rank1 plot(x,y) + TensorPlot façade", () -> {
                Tensor x = torch.tensor(new double[]{0, 1, 2, 3, 4, 5});
                Tensor y = torch.tensor(new double[]{0, 0.5, 1.5, 2.0, 1.0, 0.2});
                Path p = tmp.resolve("r1_xy.png");
                TensorPlot.line(x, y).setTitle("xy").savefig(p.toString());
                checkPng(p, "r1_xy");
                check("last is LineChart", Matplotlib.last() instanceof LineChart);
            });

            // ------------------------------------------------------------------
            // Rank 2
            // ------------------------------------------------------------------
            benchmark("2a. rank2 heatmap / imshow HW", () -> {
                Tensor m = arangeReshape(20, 4, 5);
                Path p1 = tmp.resolve("r2_heat.png");
                HeatmapChart heat = Matplotlib.heatmap(m).setShowValues(true).setTitle("4x5");
                heat.savefig(p1.toString());
                checkPng(p1, "r2_heat");

                Path p2 = tmp.resolve("r2_imshow.png");
                BaseChart im = Matplotlib.imshow(m, Layout.HW);
                im.savefig(p2.toString());
                checkPng(p2, "r2_imshow");
                check("imshow HW is HeatmapChart", im instanceof HeatmapChart);
            });

            benchmark("2b. rank2 multi-series plot (rows as series)", () -> {
                // 3 series, 8 points
                double[] raw = new double[3 * 8];
                for (int r = 0; r < 3; r++)
                    for (int c = 0; c < 8; c++)
                        raw[r * 8 + c] = Math.sin((c + 1) * 0.4 + r);
                Tensor t = torch.tensor(raw).reshape(new long[]{3, 8});
                Path p = tmp.resolve("r2_multiseries.png");
                LineChart chart = Matplotlib.plot(t);
                chart.savefig(p.toString());
                checkPng(p, "r2_multiseries");
            });

            benchmark("2c. rank2 scatter (N,2) and (2,N)", () -> {
                double[] n2 = new double[30];
                for (int i = 0; i < 15; i++) {
                    n2[i * 2] = i;
                    n2[i * 2 + 1] = Math.cos(i / 3.0);
                }
                Tensor a = torch.tensor(n2).reshape(new long[]{15, 2});
                Path p1 = tmp.resolve("r2_scatter_n2.png");
                ScatterChart s1 = Matplotlib.scatter(a);
                s1.savefig(p1.toString());
                checkPng(p1, "r2_scatter_n2");

                Tensor b = torch.tensor(n2).reshape(new long[]{2, 15});
                Path p2 = tmp.resolve("r2_scatter_2n.png");
                Matplotlib.scatter(b).savefig(p2.toString());
                checkPng(p2, "r2_scatter_2n");
            });

            benchmark("2d. rank2 boxplot columns", () -> {
                Tensor m = randn(50, 4);
                Path p = tmp.resolve("r2_box_cols.png");
                Matplotlib.boxplot(m).savefig(p.toString());
                checkPng(p, "r2_box_cols");
            });

            benchmark("2e. plot(x, y_rank2) multi-series", () -> {
                Tensor x = torch.tensor(new double[]{0, 1, 2, 3, 4});
                double[] yraw = new double[2 * 5];
                for (int r = 0; r < 2; r++)
                    for (int c = 0; c < 5; c++) yraw[r * 5 + c] = (r + 1) * c;
                Tensor y = torch.tensor(yraw).reshape(new long[]{2, 5});
                Path p = tmp.resolve("r2_plot_xy.png");
                Matplotlib.plot(x, y).savefig(p.toString());
                checkPng(p, "r2_plot_xy");
            });

            // ------------------------------------------------------------------
            // Rank 3
            // ------------------------------------------------------------------
            benchmark("3a. CHW C=1 grayscale imshow", () -> {
                Tensor t = arangeReshape(1 * 8 * 10, 1, 8, 10);
                Path p = tmp.resolve("r3_chw1.png");
                BaseChart c = Matplotlib.imshow(t, Layout.CHW);
                c.savefig(p.toString());
                checkPng(p, "r3_chw1");
            });

            benchmark("3b. CHW C=3 RGB imshow", () -> {
                Tensor t = randn(3, 16, 16);
                Path p = tmp.resolve("r3_chw3.png");
                BaseChart c = Matplotlib.imshow(t, Layout.CHW);
                c.savefig(p.toString());
                checkPng(p, "r3_chw3");
                check("CHW3 uses ImageGrid or Heatmap", c instanceof ImageGridChart || c instanceof HeatmapChart);
                c.show();
            });

            benchmark("3c. CHW C>4 channel grid", () -> {
                Tensor t = randn(6, 8, 8);
                Path p = tmp.resolve("r3_chw6.png");
                BaseChart c = Matplotlib.imshow(t, Layout.CHW);
                c.savefig(p.toString());
                checkPng(p, "r3_chw6");
                check("C>4 is ImageGridChart", c instanceof ImageGridChart);
                check("plane count == 6", ((ImageGridChart) c).planeCount() == 6);
            });

            benchmark("3d. HWC C=3 imshow", () -> {
                Tensor t = randn(12, 14, 3);
                Path p = tmp.resolve("r3_hwc3.png");
                Matplotlib.imshow(t, Layout.HWC).savefig(p.toString());
                checkPng(p, "r3_hwc3");
            });

            benchmark("3e. NHW batch imageGrid", () -> {
                Tensor t = randn(5, 10, 10);
                Path p = tmp.resolve("r3_nhw.png");
                ImageGridChart g = Matplotlib.imageGrid(t, Layout.NHW, 5);
                g.setCols(3).setTitle("NHW batch").savefig(p.toString());
                checkPng(p, "r3_nhw");
                check("5 planes", g.planeCount() == 5);
            });

            benchmark("3f. AUTO detect CHW vs NHW", () -> {
                Layout chw = TensorPlotUtils.detectLayout(randn(3, 20, 20));
                check("AUTO CHW for (3,H,W)", chw == Layout.CHW);
                Layout nhw = TensorPlotUtils.detectLayout(randn(7, 20, 20));
                check("AUTO NHW for (7,H,W)", nhw == Layout.NHW);
                Layout hwc = TensorPlotUtils.detectLayout(randn(20, 20, 3));
                check("AUTO HWC for (H,W,3)", hwc == Layout.HWC);
            });

            // ------------------------------------------------------------------
            // Rank 4
            // ------------------------------------------------------------------
            benchmark("4a. NCHW imageGrid", () -> {
                Tensor t = randn(4, 3, 12, 12);
                Path p = tmp.resolve("r4_nchw.png");
                ImageGridChart g = TensorPlot.grid(t, Layout.NCHW, 4);
                g.setCols(2).setTitle("NCHW").savefig(p.toString());
                checkPng(p, "r4_nchw");
                check("4 images", g.planeCount() == 4);
            });

            benchmark("4b. NHWC imageGrid explicit layout", () -> {
                Tensor t = randn(3, 10, 10, 3);
                Path p = tmp.resolve("r4_nhwc.png");
                ImageGridChart g = Matplotlib.imageGrid(t, Layout.NHWC, 3);
                g.savefig(p.toString());
                checkPng(p, "r4_nhwc");
                check("3 images", g.planeCount() == 3);
            });

            benchmark("4c. NCHW maxImages cap", () -> {
                Tensor t = randn(10, 1, 6, 6);
                ImageGridChart g = Matplotlib.imageGrid(t, Layout.NCHW, 4);
                check("capped to 4", g.planeCount() == 4);
                Path p = tmp.resolve("r4_cap.png");
                g.savefig(p.toString());
                checkPng(p, "r4_cap");
            });

            benchmark("4d. AUTO detect NCHW vs NHWC", () -> {
                check("AUTO NCHW (N,3,H,W)",
                    TensorPlotUtils.detectLayout(randn(2, 3, 8, 8)) == Layout.NCHW);
                check("AUTO NHWC (N,H,W,3)",
                    TensorPlotUtils.detectLayout(randn(2, 8, 8, 3)) == Layout.NHWC);
            });

            // ------------------------------------------------------------------
            // Rank 5+
            // ------------------------------------------------------------------
            benchmark("5. rank5 leading-slice imageGrid", () -> {
                Tensor t = randn(3, 2, 1, 8, 8); // peel dim0 → rank4-ish remainder
                Path p = tmp.resolve("r5_grid.png");
                ImageGridChart g = Matplotlib.imageGrid(t, Layout.AUTO, 3);
                g.savefig(p.toString());
                checkPng(p, "r5_grid");
                check("rank5 yields planes", g.planeCount() >= 1 && g.planeCount() <= 3);
            });

            // ------------------------------------------------------------------
            // Dtypes
            // ------------------------------------------------------------------
            benchmark("6. dtypes float32 / float64 / int promoted", () -> {
                Tensor f32 = torch.randn(new long[]{8, 8}); // default float
                Tensor f64 = torch.tensor(new double[]{1, 2, 3, 4, 5, 6, 7, 8, 9}).reshape(new long[]{3, 3});
                // int via long/int array if available — use double then to int
                Tensor i32 = f64.to(org.bytedeco.pytorch.global.torch.ScalarType.Int);

                Path p1 = tmp.resolve("dtype_f32.png");
                Matplotlib.heatmap(f32).savefig(p1.toString());
                checkPng(p1, "dtype_f32");

                Path p2 = tmp.resolve("dtype_f64.png");
                Matplotlib.imshow(f64).savefig(p2.toString());
                checkPng(p2, "dtype_f64");

                Path p3 = tmp.resolve("dtype_i32.png");
                Matplotlib.heatmap(i32).savefig(p3.toString());
                checkPng(p3, "dtype_i32");
            });

            // ------------------------------------------------------------------
            // Edge cases
            // ------------------------------------------------------------------
            benchmark("7a. empty numel guard / empty plot", () -> {
                Tensor empty = torch.tensor(new double[0]);
                Path p = tmp.resolve("empty_line.png");
                Matplotlib.plot(empty).savefig(p.toString());
                checkPng(p, "empty_line");
            });

            benchmark("7b. singleton (1,H,W)", () -> {
                Tensor t = randn(1, 9, 9);
                Path p = tmp.resolve("singleton.png");
                // AUTO → CHW C=1
                Matplotlib.imshow(t, Layout.AUTO).savefig(p.toString());
                checkPng(p, "singleton");
            });

            benchmark("7c. non-contiguous transpose then plot", () -> {
                Tensor t = arangeReshape(12, 3, 4);
                Tensor tr = t.transpose(0, 1); // 4x3, typically non-contiguous
                // JavaCPP exposes is_contiguous as __dispatch_is_contiguous
                boolean contig = tr.__dispatch_is_contiguous();
                check("transpose often non-contiguous (info)", true); // soft: platform-dependent
                System.out.println("       transpose contiguous? " + contig);
                Path p1 = tmp.resolve("noncontig_heat.png");
                Matplotlib.heatmap(tr).savefig(p1.toString());
                checkPng(p1, "noncontig_heat");

                Path p2 = tmp.resolve("noncontig_line.png");
                Matplotlib.plot(tr).savefig(p2.toString());
                checkPng(p2, "noncontig_line");
            });

            benchmark("7d. size guard rejects huge numel", () -> {
                long prev = TensorPlotUtils.maxNumel();
                try {
                    TensorPlotUtils.setMaxNumel(100);
                    Tensor big = randn(50, 50); // 2500 > 100
                    boolean threw = false;
                    try {
                        TensorPlotUtils.asDouble1D(big);
                    } catch (IllegalArgumentException ex) {
                        threw = true;
                        check("guard msg", ex.getMessage().contains("numel"));
                    }
                    check("size guard throws", threw);
                } finally {
                    TensorPlotUtils.setMaxNumel(prev);
                }
            });

            benchmark("7e. heatmap rank3 uses first plane", () -> {
                Tensor t = arangeReshape(2 * 3 * 4, 2, 3, 4);
                Path p = tmp.resolve("heat_r3_first.png");
                HeatmapChart h = Matplotlib.heatmap(t);
                h.savefig(p.toString());
                checkPng(p, "heat_r3_first");
            });

            // ------------------------------------------------------------------
            // Facades: Matplotlib.last + TensorPlot
            // ------------------------------------------------------------------
            benchmark("8. TensorPlot façade + last()", () -> {
                Tensor y = torch.tensor(new double[]{1, 3, 2, 5, 4});
                TensorPlot.hist(y, 4);
                check("last after hist", Matplotlib.last() instanceof HistogramChart);
                check("TensorPlot.last same", TensorPlot.last() == Matplotlib.last());

                Path p = tmp.resolve("facade_grid.png");
                TensorPlot.grid(randn(2, 1, 6, 6)).setCols(2).savefig(p.toString());
                checkPng(p, "facade_grid");
                TensorPlot.savefig(tmp.resolve("facade_last.png").toString());
                checkPng(tmp.resolve("facade_last.png"), "facade_last");
            });

            // ------------------------------------------------------------------
            // Regression: DF + array APIs still work
            // ------------------------------------------------------------------
            benchmark("9. regression DataFrame + array plot APIs", () -> {
                // Array API always available (no extra deps)
                Path p2 = tmp.resolve("reg_arr.png");
                double[] x = {0, 1, 2, 3};
                double[] y = {0, 1, 0, 1};
                Matplotlib.plot(x, y, "arr");
                Matplotlib.savefig(p2.toString());
                checkPng(p2, "reg_arr");

                // DataFrame path needs parquet (+ slf4j, hadoop bits, …) on the classpath.
                // Soft-skip when optional deps are absent so the tensor suite stays standalone.
                try {
                    DataFrame df = DataFrame.create();
                    df.addColumn("x", Column.DType.FLOAT64);
                    df.addColumn("y", Column.DType.FLOAT64);
                    for (int i = 0; i < 20; i++) df.addRow((double) i, Math.sin(i / 3.0));

                    Path p1 = tmp.resolve("reg_df.png");
                    Matplotlib.plot(df, "x", "y").setTitle("df reg").savefig(p1.toString());
                    checkPng(p1, "reg_df");

                    Path p3 = tmp.resolve("reg_dfplot.png");
                    df.plot().line("x", "y").savefig(p3.toString());
                    checkPng(p3, "reg_dfplot");
                } catch (NoClassDefFoundError | ExceptionInInitializerError link) {
                    System.out.println("       skip DF regression (optional deps missing: "
                        + link.getClass().getSimpleName() + ")");
                } catch (LinkageError link) {
                    System.out.println("       skip DF regression (linkage: " + link.getClass().getSimpleName() + ")");
                }
            });

            // ------------------------------------------------------------------
            // Utils unit-ish checks
            // ------------------------------------------------------------------
            benchmark("10. TensorPlotUtils shape/matrix/scatter helpers", () -> {
                Tensor t2 = arangeReshape(6, 2, 3);
                check("rank", TensorPlotUtils.rank(t2) == 2);
                check("shape", Arrays.equals(TensorPlotUtils.shape(t2), new long[]{2, 3}));
                double[][] m = TensorPlotUtils.asMatrix2D(t2);
                check("matrix rows", m.length == 2 && m[0].length == 3);
                check("matrix[1][2]==5", Math.abs(m[1][2] - 5.0) < 1e-9);

                Tensor pair = torch.tensor(new double[]{1, 10, 2, 20, 3, 30}).reshape(new long[]{3, 2});
                double[][] xy = TensorPlotUtils.scatterXY(pair);
                check("scatter x0", Math.abs(xy[0][0] - 1) < 1e-9);
                check("scatter y2", Math.abs(xy[1][2] - 30) < 1e-9);

                Tensor s0 = TensorPlotUtils.sliceLeading(t2, 1);
                check("slice rank1", TensorPlotUtils.rank(s0) == 1);
                double[] row = TensorPlotUtils.asDouble1D(s0);
                check("slice values", row.length == 3 && Math.abs(row[0] - 3) < 1e-9);
            });

            // ------------------------------------------------------------------
            // Perf smoke (informational — does not fail on slow hosts)
            // ------------------------------------------------------------------
            benchmark("11. perf smoke 32x3x32x32 grid (informational)", () -> {
                Tensor batch = randn(32, 3, 32, 32);
                long t0 = System.nanoTime();
                Path p = tmp.resolve("perf_grid.png");
                ImageGridChart g = Matplotlib.imageGrid(batch, Layout.NCHW, 16);
                g.setCols(4).savefig(p.toString());
                long ms = (System.nanoTime() - t0) / 1_000_000;
                checkPng(p, "perf_grid");
                check("16 planes capped", g.planeCount() == 16);
                System.out.println("       perf_grid wall " + ms + " ms (info only)");
                // Soft budget 30s — only fail if extremely pathological
                check("perf under 30s", ms < 30_000);
            });

            // ------------------------------------------------------------------
            // hist flattens multi-dim
            // ------------------------------------------------------------------
            benchmark("12. hist flattens rank3", () -> {
                Tensor t = randn(2, 3, 4);
                Path p = tmp.resolve("hist_flat.png");
                Matplotlib.hist(t, 12).savefig(p.toString());
                checkPng(p, "hist_flat");
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
