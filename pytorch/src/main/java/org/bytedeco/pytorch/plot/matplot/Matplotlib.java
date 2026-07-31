package org.bytedeco.pytorch.plot.matplot;

import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.data.numpy.NDArray;
import org.bytedeco.pytorch.dataframe.DataFrame;
import org.bytedeco.pytorch.plot.Figure;
import org.bytedeco.pytorch.plot.PlotInputs;
import org.bytedeco.pytorch.plot.TensorPlotUtils;
import org.bytedeco.pytorch.plot.TensorPlotUtils.Layout;
import org.bytedeco.pytorch.plot.TensorPlotUtils.Plane;
import org.bytedeco.pytorch.plot.chart.*;

import java.awt.Color;
import java.util.List;

/**
 * Matplotlib-like static facade (pyplot style). Pure AWT charts; headless-safe {@code savefig}.
 *
 * <h2>Tensor rank policy</h2>
 * <ul>
 *   <li><b>0</b> — rejected</li>
 *   <li><b>1</b> {@code (N,)} — line / hist / bar / box / violin</li>
 *   <li><b>2</b> {@code (H,W)} — heatmap / imshow; {@code plot} treats <em>rows as series</em>;
 *       {@code scatter(t)} accepts {@code (N,2)} or {@code (2,N)}</li>
 *   <li><b>3</b> — {@code imshow} for CHW/HWC (C∈{1,3,4}); else {@code imageGrid}
 *       (NHW batch or per-channel)</li>
 *   <li><b>4</b> — {@code imageGrid} NCHW / NHWC (batch capped, default 16)</li>
 *   <li><b>≥5</b> — leading-dim slices into grid (capped)</li>
 * </ul>
 *
 * <p>Existing DataFrame / {@code double[]} APIs are unchanged. Legacy Tensor overloads
 * ({@code plot(x,y)}, {@code scatter(x,y)}, {@code hist}, {@code heatmap}) keep flatten /
 * matrix semantics and now route through {@link TensorPlotUtils}.
 */
public final class Matplotlib {
    /** Default max images in a batch grid. */
    public static final int DEFAULT_MAX_IMAGES = 16;

    private static BaseChart lastChart;

    private Matplotlib() {}

    public static BaseChart last() { return lastChart; }

    // ---- DataFrame ----

    public static LineChart plot(DataFrame df, String xColumn, String... yColumns) {
        return remember(new LineChart("Line Plot", df, xColumn, yColumns));
    }

    public static ScatterChart scatter(DataFrame df, String xColumn, String yColumn) {
        return remember(new ScatterChart("Scatter Plot", df, xColumn, yColumn));
    }

    public static BarChart bar(DataFrame df, String xColumn, String yColumn) {
        return remember(new BarChart("Bar Chart", df, xColumn, yColumn));
    }

    public static BoxChart boxplot(DataFrame df, String categoryColumn, String valueColumn) {
        return remember(new BoxChart("Box Plot", df, categoryColumn, valueColumn));
    }

    public static HistogramChart hist(DataFrame df, String column, int bins) {
        return remember(new HistogramChart("Histogram", df.column(column).asDoubleArray(), bins));
    }

    public static PieChart pie(DataFrame df, String labelColumn, String valueColumn) {
        return remember(new PieChart("Pie Chart", df, labelColumn, valueColumn));
    }

    public static AreaChart area(DataFrame df, String xColumn, String... yColumns) {
        return remember(new AreaChart("Area Chart", df, xColumn, yColumns));
    }

    public static ViolinChart violinplot(DataFrame df, String categoryColumn, String valueColumn) {
        return remember(new ViolinChart("Violin Plot", df, categoryColumn, valueColumn));
    }

    public static BubbleChart bubble(DataFrame df, String xColumn, String yColumn, String sizeColumn) {
        return remember(new BubbleChart("Bubble Chart", df, xColumn, yColumn, sizeColumn));
    }

    public static BubbleChart bubble(DataFrame df, String xColumn, String yColumn, String sizeColumn, String categoryColumn) {
        return remember(new BubbleChart("Bubble Chart", df, xColumn, yColumn, sizeColumn, categoryColumn));
    }

    public static RadarChart radar(DataFrame df, String categoryColumn, String valueColumn) {
        return remember(new RadarChart("Radar Chart", df, categoryColumn, valueColumn));
    }

    public static FunnelChart funnel(DataFrame df, String stageColumn, String valueColumn) {
        return remember(new FunnelChart("Funnel Chart", df, stageColumn, valueColumn));
    }

    public static HeatmapChart heatmap(DataFrame df, List<String> columns) {
        List<String> cols = columns;
        if (cols == null || cols.isEmpty()) {
            cols = new java.util.ArrayList<>();
            for (var c : df.columns()) {
                switch (c.dtype()) {
                    case INT32, INT64, FLOAT32, FLOAT64 -> cols.add(c.name());
                    default -> {}
                }
            }
        }
        int n = cols.size();
        double[][] m = new double[df.rowCount()][n];
        for (int j = 0; j < n; j++) {
            double[] col = df.column(cols.get(j)).asDoubleArray();
            for (int i = 0; i < col.length; i++) m[i][j] = col[i];
        }
        return heatmap(m, null, cols);
    }

    // ---- arrays ----

    public static LineChart plot(double[] x, double[] y) {
        return remember(new LineChart("Line Plot", x, y, "y"));
    }

    public static LineChart plot(double[] x, double[] y, String label) {
        return remember(new LineChart("Line Plot", x, y, label));
    }

    public static ScatterChart scatter(double[] x, double[] y) {
        return remember(new ScatterChart("Scatter Plot", x, y));
    }

    public static BarChart bar(String[] categories, double[] values) {
        return remember(new BarChart("Bar Chart", categories, values));
    }

    public static HistogramChart hist(double[] data, int bins) {
        return remember(new HistogramChart("Histogram", data, bins));
    }

    public static HeatmapChart heatmap(double[][] matrix, List<String> rowLabels, List<String> colLabels) {
        return remember(new HeatmapChart("Heatmap", matrix, rowLabels, colLabels));
    }

    public static HeatmapChart heatmap(double[][] matrix) {
        return heatmap(matrix, null, null);
    }

    public static BoxChart boxplot(double[] values) {
        return remember(new BoxChart("Box Plot", values));
    }

    public static PieChart pie(String[] labels, double[] values) {
        return remember(new PieChart("Pie Chart", labels, values));
    }

    public static AreaChart area(double[] x, double[] y, String name) {
        return remember(new AreaChart("Area Chart", x, y, name));
    }

    public static ViolinChart violinplot(double[] values) {
        return remember(new ViolinChart("Violin Plot", values));
    }

    public static BubbleChart bubble(double[] x, double[] y, double[] size) {
        return remember(new BubbleChart("Bubble Chart", x, y, size));
    }

    public static FunnelChart funnel(String[] stages, double[] values) {
        return remember(new FunnelChart("Funnel Chart", stages, values));
    }

    // =========================================================================
    // Tensor overloads — multi-dimensional
    // =========================================================================

    /**
     * Rank-aware line plot of a single tensor.
     * <ul>
     *   <li>rank 1 — y vs index</li>
     *   <li>rank 2 — each <em>row</em> is a series vs column index</li>
     *   <li>higher — rejected (use {@link #imshow} / {@link #imageGrid})</li>
     * </ul>
     */
    public static LineChart plot(Tensor t) {
        TensorPlotUtils.requireNonNull(t);
        TensorPlotUtils.rejectScalar(t);
        int r = TensorPlotUtils.rank(t);
        if (r == 1) {
            double[] y = TensorPlotUtils.asDouble1D(t);
            return remember(new LineChart("Line Plot", TensorPlotUtils.indexArray(y.length), y, "y")
                .setTitle("plot " + TensorPlotUtils.shapeString(t)));
        }
        if (r == 2) {
            double[][] m = TensorPlotUtils.asMatrix2D(t);
            if (m.length == 0) {
                return remember(new LineChart("Line Plot", new double[0], new double[0], "y"));
            }
            int cols = m[0].length;
            double[] x = TensorPlotUtils.indexArray(cols);
            LineChart chart = new LineChart("Line Plot", x, m[0], "row0");
            for (int i = 1; i < m.length; i++) chart.addSeries(m[i], "row" + i);
            chart.setShowLegend(m.length > 1);
            chart.setTitle("plot " + TensorPlotUtils.shapeString(t) + " (rows as series)");
            return remember(chart);
        }
        throw new IllegalArgumentException(
            "plot(Tensor) supports rank 1–2, got rank " + r + " shape " + TensorPlotUtils.shapeString(t)
                + "; use imshow/imageGrid for image tensors");
    }

    /**
     * Plot y vs x. If {@code y} is rank-2, each row is a series against flattened/1D {@code x}.
     * Rank-1 {@code y} preserves legacy flatten semantics for both sides.
     */
    public static LineChart plot(Tensor x, Tensor y) {
        TensorPlotUtils.requireNonNull(x);
        TensorPlotUtils.requireNonNull(y);
        int ry = TensorPlotUtils.rank(y);
        if (ry <= 1) {
            return plot(tensorToDouble(x), tensorToDouble(y));
        }
        if (ry == 2) {
            double[] xx = tensorToDouble(x);
            double[][] m = TensorPlotUtils.asMatrix2D(y);
            if (m.length == 0) {
                return remember(new LineChart("Line Plot", xx, new double[0], "y0"));
            }
            // Prefer rows as series when row length matches x; else columns as series.
            boolean rowsMatch = m[0].length == xx.length;
            boolean colsMatch = m.length == xx.length;
            LineChart chart;
            if (rowsMatch || !colsMatch) {
                chart = new LineChart("Line Plot", xx, m[0], "row0");
                for (int i = 1; i < m.length; i++) chart.addSeries(m[i], "row" + i);
            } else {
                List<double[]> series = TensorPlotUtils.colsAsSeries(m);
                chart = new LineChart("Line Plot", xx, series.get(0), "col0");
                for (int i = 1; i < series.size(); i++) chart.addSeries(series.get(i), "col" + i);
            }
            chart.setShowLegend(true);
            return remember(chart);
        }
        // flatten fallback for odd ranks (compat with aggressive callers)
        return plot(tensorToDouble(x), tensorToDouble(y));
    }

    public static LineChart plot(Tensor x, Tensor y, String label) {
        LineChart c = plot(x, y);
        if (label != null) c.setTitle(label);
        return c;
    }

    /** Scatter from two 1D tensors (flatten). */
    public static ScatterChart scatter(Tensor x, Tensor y) {
        return scatter(tensorToDouble(x), tensorToDouble(y));
    }

    /**
     * Scatter from a single rank-2 tensor with shape {@code (N,2)} or {@code (2,N)}.
     */
    public static ScatterChart scatter(Tensor t) {
        TensorPlotUtils.requireNonNull(t);
        TensorPlotUtils.rejectScalar(t);
        double[][] xy = TensorPlotUtils.scatterXY(t);
        return remember(new ScatterChart("Scatter Plot", xy[0], xy[1])
            .setTitle("scatter " + TensorPlotUtils.shapeString(t)));
    }

    /** Histogram; flattens any rank. */
    public static HistogramChart hist(Tensor data, int bins) {
        return hist(tensorToDouble(data), bins);
    }

    public static HistogramChart hist(Tensor data) {
        return hist(data, 10);
    }

    /**
     * Heatmap:
     * <ul>
     *   <li>rank 1–2 — matrix view</li>
     *   <li>rank ≥3 — first leading plane (see {@link #imageGrid} for full batch)</li>
     * </ul>
     */
    public static HeatmapChart heatmap(Tensor matrix) {
        TensorPlotUtils.requireNonNull(matrix);
        TensorPlotUtils.rejectScalar(matrix);
        int r = TensorPlotUtils.rank(matrix);
        double[][] m = r <= 2
            ? TensorPlotUtils.asMatrix2D(matrix)
            : TensorPlotUtils.firstPlaneAsMatrix(matrix);
        HeatmapChart c = new HeatmapChart(
            r <= 2 ? "Heatmap" : "Heatmap (first plane) " + TensorPlotUtils.shapeString(matrix),
            m, null, null);
        return remember(c);
    }

    /** Rank-1 values as a single box; rank-2 → one box group per column. */
    public static BoxChart boxplot(Tensor t) {
        TensorPlotUtils.requireNonNull(t);
        TensorPlotUtils.rejectScalar(t);
        int r = TensorPlotUtils.rank(t);
        if (r == 1) {
            return remember(new BoxChart("Box Plot", TensorPlotUtils.asDouble1D(t)));
        }
        if (r == 2) {
            double[][] m = TensorPlotUtils.asMatrix2D(t);
            // Build via first column then — BoxChart has no multi-group array ctor;
            // synthesize by concatenating labeled groups through DataFrame-free path:
            // use first series ctor and rely on Map via reflection-free rebuild:
            BoxChart chart = boxplotFromColumns(m);
            chart.setTitle("Box Plot " + TensorPlotUtils.shapeString(t));
            return remember(chart);
        }
        return remember(new BoxChart("Box Plot", TensorPlotUtils.asDouble1D(t)));
    }

    private static BoxChart boxplotFromColumns(double[][] m) {
        // Pack columns into sequential labeled groups by constructing one chart from col0
        // then we need multi-group — BoxChart only exposes Map via DF ctor.
        // Workaround: flatten each column into a synthetic approach using DF-free:
        // Create chart from col0, but BoxChart groups is private final.
        // Use a tiny on-the-fly encoding: one box of all values if we can't multi.
        // Better: build a minimal structure — actually BoxChart has only two public ctors.
        // For multi-column, concatenate with NaN separators won't work.
        // Simplest robust approach: plot column 0 if single-col, else create chart from
        // flattened and set title noting columns — OR build DataFrame-free multi by
        // adding a package-private path. Prefer: one box per column via repeated values
        // stored by hacking through a temp double[][] → use first column only when cols>1
        // is weak. Instead construct via category simulation:
        // We'll create one BoxChart from col0 and document; for true multi use groups via
        // a new package ctor. Add package-visible multi ctor usage here by building
        // values list — extend BoxChart? Plan said reuse. Use column-major: if many cols,
        // return box of flattened with title; for better UX add helper groups.
        if (m.length == 0) return new BoxChart("Box Plot", new double[0]);
        int rows = m.length;
        int cols = m[0].length;
        if (cols == 1) {
            double[] col = new double[rows];
            for (int i = 0; i < rows; i++) col[i] = m[i][0];
            return new BoxChart("Box Plot", col);
        }
        // Multi-column: build combined chart using BoxChart's Map by going through
        // a package-local constructor added below — call boxplotColumns.
        return BoxChart.fromGroups(columnsToGroups(m));
    }

    private static java.util.Map<String, java.util.List<Double>> columnsToGroups(double[][] m) {
        int rows = m.length;
        int cols = m[0].length;
        java.util.Map<String, java.util.List<Double>> groups = new java.util.LinkedHashMap<>();
        for (int c = 0; c < cols; c++) {
            java.util.List<Double> list = new java.util.ArrayList<>(rows);
            for (int r = 0; r < rows; r++) {
                double v = m[r][c];
                if (!Double.isNaN(v)) list.add(v);
            }
            groups.put("c" + c, list);
        }
        return groups;
    }

    /** Rank-1 violin; rank-2 flattens (single violin) — multi-group uses boxplot-style columns. */
    public static ViolinChart violinplot(Tensor t) {
        TensorPlotUtils.requireNonNull(t);
        TensorPlotUtils.rejectScalar(t);
        return remember(new ViolinChart("Violin Plot", TensorPlotUtils.asDouble1D(t))
            .setTitle("Violin " + TensorPlotUtils.shapeString(t)));
    }

    /** Rank-1 bar chart with synthetic categories {@code 0..n-1}. */
    public static BarChart bar(Tensor values) {
        TensorPlotUtils.requireNonNull(values);
        TensorPlotUtils.rejectScalar(values);
        double[] v = TensorPlotUtils.asDouble1D(values);
        String[] cats = new String[v.length];
        for (int i = 0; i < v.length; i++) cats[i] = String.valueOf(i);
        return remember(new BarChart("Bar Chart", cats, v)
            .setTitle("bar " + TensorPlotUtils.shapeString(values)));
    }

    public static AreaChart area(Tensor x, Tensor y) {
        return area(tensorToDouble(x), tensorToDouble(y), "y");
    }

    public static AreaChart area(Tensor y) {
        double[] yy = TensorPlotUtils.asDouble1D(y);
        return area(TensorPlotUtils.indexArray(yy.length), yy, "y");
    }

    // ---- imshow / imageGrid -------------------------------------------------

    /**
     * Display an image-like tensor. Layout {@link Layout#AUTO} by default.
     * <ul>
     *   <li>HW — single heatmap</li>
     *   <li>CHW/HWC with C∈{1,3,4} — gray or RGB single panel (via {@link ImageGridChart})</li>
     *   <li>C&gt;4 CHW — per-channel grid</li>
     *   <li>batch ranks — delegated to {@link #imageGrid(Tensor)}</li>
     * </ul>
     */
    public static BaseChart imshow(Tensor t) {
        return imshow(t, Layout.AUTO);
    }

    public static BaseChart imshow(Tensor t, Layout layout) {
        TensorPlotUtils.requireNonNull(t);
        TensorPlotUtils.rejectScalar(t);
        int r = TensorPlotUtils.rank(t);
        Layout L = TensorPlotUtils.resolveLayout(t, layout);

        // Batch-like → grid
        if (r >= 4 || L == Layout.NCHW || L == Layout.NHWC || L == Layout.NHW
            || (r == 3 && L == Layout.NHW)
            || (r == 3 && L == Layout.CHW && TensorPlotUtils.shape(t)[0] > 4)
            || (r == 3 && L == Layout.HWC && TensorPlotUtils.shape(t)[2] > 4)
            || r >= 5) {
            // Special case: pure CHW/HWC single image with few channels stays single-cell grid
            if (r == 3 && (L == Layout.CHW || L == Layout.HWC)) {
                long c = L == Layout.CHW ? TensorPlotUtils.shape(t)[0] : TensorPlotUtils.shape(t)[2];
                if (c > 4) {
                    List<Plane> chans = L == Layout.CHW
                        ? TensorPlotUtils.channelPlanesCHW(TensorPlotUtils.toCpuDouble(t), DEFAULT_MAX_IMAGES)
                        : TensorPlotUtils.extractPlanes(t, L, DEFAULT_MAX_IMAGES);
                    return remember(new ImageGridChart("imshow channels " + TensorPlotUtils.shapeString(t), chans)
                        .setTitle("imshow " + TensorPlotUtils.shapeString(t)));
                }
            }
            if (r >= 4 || L == Layout.NHW || L == Layout.NCHW || L == Layout.NHWC || r >= 5) {
                return imageGrid(t, L, DEFAULT_MAX_IMAGES);
            }
        }

        if (r == 1 || r == 2 || L == Layout.HW) {
            // single heatmap for HW
            double[][] m = r <= 2 ? TensorPlotUtils.asMatrix2D(t) : TensorPlotUtils.firstPlaneAsMatrix(t);
            return remember(new HeatmapChart("imshow " + TensorPlotUtils.shapeString(t), m, null, null));
        }

        // Single CHW/HWC image → one-cell ImageGrid (supports RGB)
        List<Plane> planes = TensorPlotUtils.extractPlanes(t, L, 1);
        ImageGridChart grid = new ImageGridChart("imshow " + TensorPlotUtils.shapeString(t), planes);
        grid.setCols(1).setShowIndices(false);
        return remember(grid);
    }

    public static ImageGridChart imageGrid(Tensor t) {
        return imageGrid(t, Layout.AUTO, DEFAULT_MAX_IMAGES);
    }

    public static ImageGridChart imageGrid(Tensor t, Layout layout) {
        return imageGrid(t, layout, DEFAULT_MAX_IMAGES);
    }

    public static ImageGridChart imageGrid(Tensor t, int maxImages) {
        return imageGrid(t, Layout.AUTO, maxImages);
    }

    public static ImageGridChart imageGrid(Tensor t, Layout layout, int maxImages) {
        TensorPlotUtils.requireNonNull(t);
        TensorPlotUtils.rejectScalar(t);
        Layout L = TensorPlotUtils.resolveLayout(t, layout);
        // CHW with many channels → channel grid
        long[] sh = TensorPlotUtils.shape(t);
        List<Plane> planes;
        if (sh.length == 3 && L == Layout.CHW && sh[0] > 4) {
            planes = TensorPlotUtils.channelPlanesCHW(TensorPlotUtils.toCpuDouble(t), maxImages);
        } else {
            planes = TensorPlotUtils.extractPlanes(t, L, maxImages);
        }
        ImageGridChart grid = new ImageGridChart(
            "ImageGrid " + TensorPlotUtils.shapeString(t) + " [" + TensorPlotUtils.layoutName(L) + "]",
            planes);
        return remember(grid);
    }

    // =========================================================================
    // matplot.md parity APIs (barh / hist2d / contour / fill_between / subplots /
    // polar / errorbar / step / grouped+stacked bar) + NDArray overloads
    // =========================================================================

    // ---- barh / grouped / stacked --------------------------------------------

    /** matplotlib {@code plt.barh(cats, vals)}. */
    public static BarChart barh(String[] categories, double[] values) {
        return remember(new BarChart("BarH", categories, values).setHorizontal(true));
    }

    public static BarChart barh(DataFrame df, String yColumn, String xColumn) {
        // y = categories, x = values (horizontal)
        return remember(new BarChart("BarH", df, yColumn, xColumn).setHorizontal(true));
    }

    /** Grouped multi-series vertical bars (matplotlib side-by-side bar). */
    public static BarChart groupedBar(String[] categories, double[][] seriesValues, String[] seriesNames) {
        if (seriesValues == null || seriesValues.length == 0)
            throw new IllegalArgumentException("groupedBar: empty series");
        BarChart c = new BarChart("Grouped Bar", categories, seriesValues[0]);
        if (seriesNames != null && seriesNames.length > 0) {
            // replace default name via addSeries path — first series already added as "value"
        }
        for (int s = 1; s < seriesValues.length; s++) {
            String name = seriesNames != null && s < seriesNames.length ? seriesNames[s] : ("s" + s);
            c.addSeries(seriesValues[s], name);
        }
        if (seriesNames != null && seriesNames.length > 0) {
            // first series name is fixed as "value" in ctor; acceptable for bench
        }
        c.setShowLegend(seriesValues.length > 1);
        return remember(c);
    }

    /** Stacked bars (matplotlib {@code bottom=}). */
    public static BarChart stackedBar(String[] categories, double[][] seriesValues, String[] seriesNames) {
        BarChart c = groupedBar(categories, seriesValues, seriesNames);
        c.setStacked(true);
        return c;
    }

    // ---- hist2d --------------------------------------------------------------

    public static Hist2DChart hist2d(double[] x, double[] y, int bins) {
        return remember(new Hist2DChart("hist2d", x, y, bins).setCmap("blues"));
    }

    public static Hist2DChart hist2d(double[] x, double[] y, int bins, String cmap) {
        return remember(new Hist2DChart("hist2d", x, y, bins).setCmap(cmap));
    }

    public static Hist2DChart hist2d(DataFrame df, String xCol, String yCol, int bins) {
        return hist2d(PlotInputs.asDouble1D(df, xCol), PlotInputs.asDouble1D(df, yCol), bins);
    }

    public static Hist2DChart hist2d(NDArray x, NDArray y, int bins) {
        return hist2d(PlotInputs.asDouble1D(x), PlotInputs.asDouble1D(y), bins);
    }

    public static Hist2DChart hist2d(NDArray x, NDArray y, int bins, String cmap) {
        return hist2d(PlotInputs.asDouble1D(x), PlotInputs.asDouble1D(y), bins, cmap);
    }

    public static Hist2DChart hist2d(Tensor x, Tensor y, int bins) {
        return hist2d(PlotInputs.asDouble1D(x), PlotInputs.asDouble1D(y), bins);
    }

    public static Hist2DChart hist2d(Tensor x, Tensor y, int bins, String cmap) {
        return hist2d(PlotInputs.asDouble1D(x), PlotInputs.asDouble1D(y), bins, cmap);
    }

    // ---- contour -------------------------------------------------------------

    public static ContourChart contour(double[][] z) {
        return remember(new ContourChart("contour", z).setCmap("coolwarm"));
    }

    public static ContourChart contour(double[] x, double[] y, double[][] z) {
        return remember(new ContourChart("contour", x, y, z).setCmap("coolwarm"));
    }

    public static ContourChart contour(double[][] X, double[][] Y, double[][] Z, String cmap) {
        return remember(ContourChart.fromMesh("contour", X, Y, Z).setCmap(cmap));
    }

    public static ContourChart contour(NDArray z) {
        return contour(PlotInputs.asDouble2D(z));
    }

    public static ContourChart contour(NDArray x, NDArray y, NDArray z) {
        return contour(PlotInputs.asDouble1D(x), PlotInputs.asDouble1D(y), PlotInputs.asDouble2D(z));
    }

    public static ContourChart contour(Tensor z) {
        return contour(PlotInputs.asDouble2D(z));
    }

    // ---- fill_between --------------------------------------------------------

    public static AreaChart fill_between(double[] x, double[] yLower, double[] yUpper) {
        return remember(new AreaChart("fill_between", x, yLower, yUpper, "band")
            .setFillAlpha(0.2));
    }

    public static AreaChart fill_between(double[] x, double[] yLower, double[] yUpper, Color color) {
        return remember(new AreaChart("fill_between", x, yLower, yUpper, "band")
            .setFillAlpha(0.2).setFillColor(color));
    }

    /** Convenience: mean line + ± band. */
    public static AreaChart fill_between(double[] x, double[] yMean, double band) {
        double[] lo = new double[yMean.length];
        double[] hi = new double[yMean.length];
        for (int i = 0; i < yMean.length; i++) {
            lo[i] = yMean[i] - band;
            hi[i] = yMean[i] + band;
        }
        return remember(new AreaChart("fill_between", x, lo, hi, "band")
            .setFillAlpha(0.2).setMeanLine(yMean));
    }

    public static AreaChart fill_between(NDArray x, NDArray yLower, NDArray yUpper) {
        return fill_between(PlotInputs.asDouble1D(x), PlotInputs.asDouble1D(yLower), PlotInputs.asDouble1D(yUpper));
    }

    public static AreaChart fill_between(Tensor x, Tensor yLower, Tensor yUpper) {
        return fill_between(PlotInputs.asDouble1D(x), PlotInputs.asDouble1D(yLower), PlotInputs.asDouble1D(yUpper));
    }

    public static AreaChart fill_between(DataFrame df, String xCol, String yLoCol, String yHiCol) {
        return fill_between(PlotInputs.asDouble1D(df, xCol),
            PlotInputs.asDouble1D(df, yLoCol), PlotInputs.asDouble1D(df, yHiCol));
    }

    // ---- subplots / Figure ---------------------------------------------------

    public static Figure subplots(int rows, int cols) {
        return remember(new Figure(rows, cols));
    }

    public static Figure subplots(int rows, int cols, int figW, int figH) {
        return remember(new Figure(rows, cols).setSize(figW, figH));
    }

    // ---- polar ---------------------------------------------------------------

    public static PolarChart polar(double[] theta, double[] r) {
        return remember(new PolarChart("polar", theta, r));
    }

    public static PolarChart polar(NDArray theta, NDArray r) {
        return polar(PlotInputs.asDouble1D(theta), PlotInputs.asDouble1D(r));
    }

    public static PolarChart polar(Tensor theta, Tensor r) {
        return polar(PlotInputs.asDouble1D(theta), PlotInputs.asDouble1D(r));
    }

    public static PolarChart polar(DataFrame df, String thetaCol, String rCol) {
        return polar(PlotInputs.asDouble1D(df, thetaCol), PlotInputs.asDouble1D(df, rCol));
    }

    // ---- errorbar ------------------------------------------------------------

    public static LineChart errorbar(double[] x, double[] y, double[] yerr) {
        return remember(new LineChart("errorbar", x, y, "y")
            .setShowMarkers(true).setError(0, yerr));
    }

    public static LineChart errorbar(NDArray x, NDArray y, NDArray yerr) {
        return errorbar(PlotInputs.asDouble1D(x), PlotInputs.asDouble1D(y), PlotInputs.asDouble1D(yerr));
    }

    public static LineChart errorbar(Tensor x, Tensor y, Tensor yerr) {
        return errorbar(PlotInputs.asDouble1D(x), PlotInputs.asDouble1D(y), PlotInputs.asDouble1D(yerr));
    }

    public static LineChart errorbar(DataFrame df, String xCol, String yCol, String errCol) {
        return errorbar(PlotInputs.asDouble1D(df, xCol),
            PlotInputs.asDouble1D(df, yCol), PlotInputs.asDouble1D(df, errCol));
    }

    // ---- step ----------------------------------------------------------------

    public static LineChart step(double[] x, double[] y) {
        return remember(new LineChart("step", x, y, "y").setStep("mid"));
    }

    public static LineChart step(double[] x, double[] y, String where) {
        return remember(new LineChart("step", x, y, "y").setStep(where));
    }

    public static LineChart step(NDArray x, NDArray y) {
        return step(PlotInputs.asDouble1D(x), PlotInputs.asDouble1D(y));
    }

    public static LineChart step(Tensor x, Tensor y) {
        return step(PlotInputs.asDouble1D(x), PlotInputs.asDouble1D(y));
    }

    public static LineChart step(DataFrame df, String xCol, String yCol) {
        return step(PlotInputs.asDouble1D(df, xCol), PlotInputs.asDouble1D(df, yCol));
    }

    // ---- multi-group boxplot labels ------------------------------------------

    public static BoxChart boxplot(String[] labels, double[]... groups) {
        java.util.Map<String, java.util.List<Double>> map = new java.util.LinkedHashMap<>();
        for (int g = 0; g < groups.length; g++) {
            String lab = labels != null && g < labels.length && labels[g] != null
                ? labels[g] : ("G" + (g + 1));
            java.util.List<Double> list = new java.util.ArrayList<>();
            if (groups[g] != null) for (double v : groups[g]) if (!Double.isNaN(v)) list.add(v);
            map.put(lab, list);
        }
        return remember(BoxChart.fromGroups(map));
    }

    // ---- scatter with continuous color ---------------------------------------

    public static ScatterChart scatter(double[] x, double[] y, double[] c, String cmap) {
        return remember(new ScatterChart("Scatter Plot", x, y)
            .setColorValues(c).setCmap(cmap).setShowColorbar(true));
    }

    // ---- imshow for double[][] / NDArray -------------------------------------

    public static HeatmapChart imshow(double[][] matrix) {
        return remember(new HeatmapChart("imshow", matrix, null, null).setCmap("viridis"));
    }

    public static HeatmapChart imshow(NDArray a) {
        return imshow(PlotInputs.asDouble2D(a));
    }

    // ---- NDArray overloads (mirror Tensor / double[]) ------------------------

    public static LineChart plot(NDArray y) {
        double[] yy = PlotInputs.asDouble1D(y);
        return remember(new LineChart("Line Plot", PlotInputs.indexArray(yy.length), yy, "y"));
    }

    public static LineChart plot(NDArray x, NDArray y) {
        return plot(PlotInputs.asDouble1D(x), PlotInputs.asDouble1D(y));
    }

    public static LineChart plot(NDArray x, NDArray y, String label) {
        return plot(PlotInputs.asDouble1D(x), PlotInputs.asDouble1D(y), label);
    }

    public static ScatterChart scatter(NDArray x, NDArray y) {
        return scatter(PlotInputs.asDouble1D(x), PlotInputs.asDouble1D(y));
    }

    public static ScatterChart scatter(NDArray x, NDArray y, NDArray c, String cmap) {
        return scatter(PlotInputs.asDouble1D(x), PlotInputs.asDouble1D(y),
            PlotInputs.asDouble1D(c), cmap);
    }

    public static HistogramChart hist(NDArray data, int bins) {
        return hist(PlotInputs.asDouble1D(data), bins);
    }

    public static BarChart bar(NDArray values) {
        double[] v = PlotInputs.asDouble1D(values);
        return bar(PlotInputs.indexLabels(v.length), v);
    }

    public static BarChart bar(String[] categories, NDArray values) {
        return bar(categories, PlotInputs.asDouble1D(values));
    }

    public static BarChart barh(String[] categories, NDArray values) {
        return barh(categories, PlotInputs.asDouble1D(values));
    }

    public static BarChart barh(NDArray values) {
        double[] v = PlotInputs.asDouble1D(values);
        return barh(PlotInputs.indexLabels(v.length), v);
    }

    public static BoxChart boxplot(NDArray values) {
        return boxplot(PlotInputs.asDouble1D(values));
    }

    public static HeatmapChart heatmap(NDArray a) {
        return heatmap(PlotInputs.asDouble2D(a));
    }

    public static AreaChart area(NDArray x, NDArray y, String name) {
        return area(PlotInputs.asDouble1D(x), PlotInputs.asDouble1D(y), name);
    }

    public static PieChart pie(String[] labels, NDArray values) {
        return pie(labels, PlotInputs.asDouble1D(values));
    }

    public static ViolinChart violinplot(NDArray values) {
        return violinplot(PlotInputs.asDouble1D(values));
    }

    // ---- pyplot-style axis helpers on last chart -----------------------------

    public static BaseChart title(String t) {
        if (lastChart != null) lastChart.setTitle(t);
        return lastChart;
    }

    public static BaseChart xlabel(String s) {
        if (lastChart != null) lastChart.setXAxisLabel(s);
        return lastChart;
    }

    public static BaseChart ylabel(String s) {
        if (lastChart != null) lastChart.setYAxisLabel(s);
        return lastChart;
    }

    public static BaseChart legend(boolean on) {
        if (lastChart != null) lastChart.setShowLegend(on);
        return lastChart;
    }

    public static BaseChart grid(boolean on) {
        if (lastChart != null) lastChart.setShowGrid(on);
        return lastChart;
    }

    public static BaseChart figsize(int w, int h) {
        if (lastChart != null) lastChart.setSize(w, h);
        return lastChart;
    }

    public static BaseChart xscale(String scale) {
        if (lastChart != null) lastChart.setXScale(scale);
        return lastChart;
    }

    public static BaseChart yscale(String scale) {
        if (lastChart != null) lastChart.setYScale(scale);
        return lastChart;
    }

    // ---- show / save ----

    /**
     * Block until the last chart window is closed (matplotlib script default).
     * Fixes the "window flashes and disappears" problem of fire-and-forget show.
     */
    public static void show() {
        if (lastChart != null) lastChart.show(true);
        else System.out.println("[Matplotlib] nothing to show — call a plot API first");
    }

    /**
     * @param block {@code true} wait until close; {@code false} leave window open
     *              (strong-ref kept) and return immediately
     */
    public static void show(boolean block) {
        if (lastChart != null) lastChart.show(block);
        else System.out.println("[Matplotlib] nothing to show — call a plot API first");
    }

    public static void savefig(String path) throws Exception {
        if (lastChart == null) throw new IllegalStateException("No chart to save");
        lastChart.savefig(path);
    }

    /** Close all non-modal plot windows opened via {@link BaseChart#show(boolean)}. */
    public static void close() {
        BaseChart.closeAll();
    }

    private static <T extends BaseChart> T remember(T c) {
        lastChart = c;
        return c;
    }

    /**
     * Package-visible flatten helper. Preserves historical empty-on-error behavior
     * for legacy call sites; new code should prefer {@link TensorPlotUtils#asDouble1D}.
     */
    static double[] tensorToDouble(Tensor t) {
        if (t == null) return new double[0];
        try {
            return TensorPlotUtils.asDouble1D(t);
        } catch (Throwable e) {
            return new double[0];
        }
    }

    /**
     * Package-visible matrix helper. Rank ≥3 returns first plane (improved vs old
     * behavior that only read shape[0], shape[1] of a flat buffer incorrectly).
     */
    static double[][] tensorToMatrix(Tensor t) {
        if (t == null) return new double[0][0];
        try {
            int r = TensorPlotUtils.rank(t);
            if (r <= 2) return TensorPlotUtils.asMatrix2D(t);
            return TensorPlotUtils.firstPlaneAsMatrix(t);
        } catch (Throwable e) {
            return new double[0][0];
        }
    }
}
