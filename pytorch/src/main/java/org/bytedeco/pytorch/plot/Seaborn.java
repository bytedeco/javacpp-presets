package org.bytedeco.pytorch.plot;

import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.data.numpy.NDArray;
import org.bytedeco.pytorch.dataframe.Column;
import org.bytedeco.pytorch.dataframe.DataFrame;
import org.bytedeco.pytorch.dataframe.DataValues;

import java.awt.*;
import java.awt.image.BufferedImage;
import java.util.*;
import java.util.List;
import java.util.function.BiFunction;

/**
 * Seaborn-like façade over pure-AWT charts.
 * Covers the common Python seaborn surface used in the 20-example suite:
 * relational, distribution, categorical, matrix, regression, multi-plot,
 * FacetGrid, and style helpers.
 *
 * <p>API naming mirrors Python seaborn where practical
 * ({@code histplot}, {@code kdeplot}, {@code boxplot}, …). Chart objects are
 * fluent and support {@code savefig}/{@code show} via {@link BaseChart}.
 */
public final class Seaborn {
    private static String style = "darkgrid";
    private static String paletteName = "deep";
    private static Color[] palette = BaseChart.PALETTE.clone();

    private Seaborn() {}

    // ---- style / palette ----

    public static void set_theme(String styleName) { set_style(styleName); }

    public static void set_theme(String styleName, String palette) {
        set_style(styleName);
        set_palette(palette);
    }

    public static void set_style(String styleName) {
        style = styleName == null ? "darkgrid" : styleName.toLowerCase(Locale.ROOT);
    }

    public static void set_palette(String name) {
        paletteName = name == null ? "deep" : name.toLowerCase(Locale.ROOT);
        palette = switch (paletteName) {
            case "muted" -> new Color[]{
                new Color(0x4878d0), new Color(0xee854a), new Color(0x6acc64),
                new Color(0xd65f5f), new Color(0x956cb4), new Color(0x8c613c),
                new Color(0xdc7ec0), new Color(0x797979), new Color(0xd5bb67),
                new Color(0x82c6e2)
            };
            case "bright" -> new Color[]{
                new Color(0x023eff), new Color(0xff7c00), new Color(0x1ac938),
                new Color(0xe8000b), new Color(0x8b2be2), new Color(0x9f4800),
                new Color(0xf14cc1), new Color(0xa3a3a3), new Color(0xffc400),
                new Color(0x00d7e0)
            };
            case "colorblind" -> new Color[]{
                new Color(0x0173b2), new Color(0xde8f05), new Color(0x029e73),
                new Color(0xd55e00), new Color(0xcc78bc), new Color(0xca9161),
                new Color(0xfbafe4), new Color(0x949494), new Color(0xece133),
                new Color(0x56b4e9)
            };
            case "pastel" -> new Color[]{
                new Color(0xa1c9f4), new Color(0xffb482), new Color(0x8de5a1),
                new Color(0xff9f9b), new Color(0xd0bbff), new Color(0xdebb9b),
                new Color(0xfab0e4), new Color(0xcfcfcf), new Color(0xfffea3),
                new Color(0xb9f2f0)
            };
            case "dark" -> new Color[]{
                new Color(0x001c7f), new Color(0xb1400d), new Color(0x12711c),
                new Color(0x8c0800), new Color(0x591e71), new Color(0x592f0d),
                new Color(0xa23582), new Color(0x3c3c3c), new Color(0xb8850a),
                new Color(0x006374)
            };
            default -> BaseChart.PALETTE.clone();
        };
    }

    public static Color[] color_palette() { return palette.clone(); }

    public static Color[] color_palette(String name, int n) {
        set_palette(name);
        Color[] out = new Color[Math.max(1, n)];
        for (int i = 0; i < out.length; i++) out[i] = palette[i % palette.length];
        return out;
    }

    public static String currentStyle() { return style; }

    // ---- relational ----

    public static LineChart lineplot(DataFrame df, String x, String y) {
        return Matplotlib.plot(df, x, y).setTitle("lineplot");
    }

    public static LineChart lineplot(DataFrame df, String x, String... yCols) {
        return Matplotlib.plot(df, x, yCols).setTitle("lineplot");
    }

    public static LineChart lineplot(DataFrame df, String x, String y, String hue) {
        if (hue == null) return lineplot(df, x, y);
        Map<String, List<double[]>> groups = splitXYByHue(df, x, y, hue);
        LineChart chart = null;
        for (Map.Entry<String, List<double[]>> e : groups.entrySet()) {
            double[] xs = e.getValue().get(0);
            double[] ys = e.getValue().get(1);
            // sort by x for clean lines
            sortByX(xs, ys);
            if (chart == null) chart = new LineChart("lineplot", xs, ys, e.getKey());
            else chart.addSeries(xs, ys, e.getKey());
        }
        if (chart == null) chart = Matplotlib.plot(df, x, y);
        chart.setXAxisLabel(x).setYAxisLabel(y).setShowLegend(true);
        return remember(chart);
    }

    /** Array form: {@code sns.lineplot(x=..., y=...)}. */
    public static LineChart lineplot(double[] x, double[] y) {
        return remember(new LineChart("lineplot", x, y, "y")
            .setXAxisLabel("x").setYAxisLabel("y"));
    }

    public static ScatterChart scatterplot(DataFrame df, String x, String y) {
        return Matplotlib.scatter(df, x, y).setTitle("scatterplot");
    }

    public static ScatterChart scatterplot(DataFrame df, String x, String y, String hue) {
        if (hue == null) return scatterplot(df, x, y);
        try {
            ScatterChart c = new ScatterChart("scatterplot", df, x, y, hue);
            return remember(c);
        } catch (Throwable t) {
            return scatterplot(df, x, y);
        }
    }

    public static ScatterChart scatterplot(double[] x, double[] y) {
        return remember(new ScatterChart("scatterplot", x, y)
            .setXAxisLabel("x").setYAxisLabel("y"));
    }

    public static ScatterChart relplot(DataFrame df, String x, String y) {
        return scatterplot(df, x, y);
    }

    public static ScatterChart relplot(DataFrame df, String x, String y, String hue) {
        return scatterplot(df, x, y, hue);
    }

    // ---- distribution ----

    public static HistogramChart histplot(DataFrame df, String column, int bins) {
        return Matplotlib.hist(df, column, bins).setTitle("histplot");
    }

    public static HistogramChart histplot(DataFrame df, String column) {
        return histplot(df, column, 20);
    }

    /** seaborn {@code histplot(data, bins=30, kde=True)}. */
    public static HistogramChart histplot(DataFrame df, String column, int bins, boolean kde) {
        HistogramChart c = histplot(df, column, bins);
        if (kde) c.setKde(true);
        return c;
    }

    public static HistogramChart histplot(double[] data, int bins) {
        return remember(new HistogramChart("histplot", data, bins));
    }

    public static HistogramChart histplot(double[] data, int bins, boolean kde) {
        return histplot(data, bins).setKde(kde);
    }

    public static LineChart kdeplot(DataFrame df, String column) {
        double[] data = df.column(column).asDoubleArray();
        double[][] grid = kdeGrid(data, 100);
        return remember(new LineChart("kdeplot", grid[0], grid[1], column)
            .setXAxisLabel(column).setYAxisLabel("density"));
    }

    public static LineChart kdeplot(double[] data) {
        return kdeplot(data, "density");
    }

    public static LineChart kdeplot(double[] data, String label) {
        double[][] grid = kdeGrid(data, 100);
        LineChart c = new LineChart("kdeplot", grid[0], grid[1], label == null ? "density" : label)
            .setXAxisLabel("value").setYAxisLabel("density");
        c.setShowLegend(true);
        return remember(c);
    }

    /** Overlay a second KDE series on an existing chart (multi-group kdeplot). */
    public static LineChart kdeplot(LineChart existing, double[] data, String label) {
        double[][] grid = kdeGrid(data, 100);
        if (existing == null) return kdeplot(data, label);
        existing.addSeries(grid[0], grid[1], label == null ? "density" : label);
        existing.setShowLegend(true);
        return remember(existing);
    }

    public static LineChart kdeplot(DataFrame df, String column, String hue) {
        if (hue == null) return kdeplot(df, column);
        Map<String, double[]> groups = splitColByHue(df, column, hue);
        LineChart chart = null;
        for (Map.Entry<String, double[]> e : groups.entrySet()) {
            double[][] grid = kdeGrid(e.getValue(), 100);
            if (chart == null) chart = new LineChart("kdeplot", grid[0], grid[1], e.getKey());
            else chart.addSeries(grid[0], grid[1], e.getKey());
        }
        if (chart == null) return kdeplot(df, column);
        chart.setXAxisLabel(column).setYAxisLabel("density").setShowLegend(true);
        return remember(chart);
    }

    public static LineChart ecdfplot(DataFrame df, String column) {
        return ecdfFromArray(df.column(column).asDoubleArray(), column, "ecdf");
    }

    public static LineChart ecdfplot(double[] data) {
        return ecdfFromArray(data, "value", "ecdf");
    }

    public static LineChart ecdfplot(DataFrame df, String column, String hue) {
        if (hue == null) return ecdfplot(df, column);
        Map<String, double[]> groups = splitColByHue(df, column, hue);
        LineChart chart = null;
        for (Map.Entry<String, double[]> e : groups.entrySet()) {
            double[][] xy = ecdfXY(e.getValue());
            if (chart == null) chart = new LineChart("ecdfplot", xy[0], xy[1], e.getKey());
            else chart.addSeries(xy[0], xy[1], e.getKey());
        }
        if (chart == null) return ecdfplot(df, column);
        chart.setXAxisLabel(column).setYAxisLabel("proportion").setShowLegend(true);
        return remember(chart);
    }

    // ---- categorical ----

    public static BoxChart boxplot(DataFrame df, String x, String y) {
        return Matplotlib.boxplot(df, x, y).setTitle("boxplot");
    }

    public static ViolinChart violinplot(DataFrame df, String x, String y) {
        return Matplotlib.violinplot(df, x, y).setTitle("violinplot");
    }

    /** seaborn {@code violinplot(..., inner="quartile")}. */
    public static ViolinChart violinplot(DataFrame df, String x, String y, String inner) {
        return violinplot(df, x, y).setInner(inner);
    }

    public static ScatterChart stripplot(DataFrame df, String x, String y) {
        return stripplot(df, x, y, 0.6);
    }

    public static ScatterChart stripplot(DataFrame df, String x, String y, double alpha) {
        LinkedHashMap<String, Integer> cats = new LinkedHashMap<>();
        int n = df.rowCount();
        double[] xs = new double[n];
        double[] ys = new double[n];
        Random rng = new Random(0);
        for (int i = 0; i < n; i++) {
            Object c = DataValues.unwrap(df.get(i, x));
            String key = c == null ? "null" : c.toString();
            int idx = cats.computeIfAbsent(key, k -> cats.size());
            xs[i] = idx + (rng.nextDouble() - 0.5) * 0.35;
            ys[i] = DataValues.asDouble(df.get(i, y));
        }
        return remember(new ScatterChart("stripplot", xs, ys)
            .setAlpha(alpha)
            .setXAxisLabel(x).setYAxisLabel(y));
    }

    public static ScatterChart swarmplot(DataFrame df, String x, String y) {
        return swarmplot(df, x, y, 4);
    }

    /**
     * Beeswarm-style placement: sort within category and offset horizontally
     * so nearby points don't fully overlap (approximation of seaborn swarmplot).
     */
    public static ScatterChart swarmplot(DataFrame df, String x, String y, int size) {
        LinkedHashMap<String, List<Integer>> byCat = new LinkedHashMap<>();
        int n = df.rowCount();
        double[] rawY = new double[n];
        for (int i = 0; i < n; i++) {
            Object c = DataValues.unwrap(df.get(i, x));
            String key = c == null ? "null" : c.toString();
            byCat.computeIfAbsent(key, k -> new ArrayList<>()).add(i);
            rawY[i] = DataValues.asDouble(df.get(i, y));
        }
        double[] xs = new double[n];
        double[] ys = new double[n];
        int catIdx = 0;
        double pointRadius = 0.04; // in category units
        for (Map.Entry<String, List<Integer>> e : byCat.entrySet()) {
            List<Integer> rows = e.getValue();
            rows.sort(Comparator.comparingDouble(i -> rawY[i]));
            // greedy beeswarm: place each point at smallest |offset| without collision
            List<double[]> placed = new ArrayList<>(); // {xOff, y}
            for (int row : rows) {
                double yy = rawY[row];
                double bestOff = 0;
                boolean found = false;
                for (int attempt = 0; attempt < 40 && !found; attempt++) {
                    double cand = (attempt % 2 == 0 ? 1 : -1) * (attempt / 2) * pointRadius;
                    boolean ok = true;
                    for (double[] p : placed) {
                        double dy = (yy - p[1]);
                        double dx = (cand - p[0]);
                        // scale y to roughly category units
                        if (dx * dx + (dy * 0.15) * (dy * 0.15) < pointRadius * pointRadius * 0.9) {
                            ok = false; break;
                        }
                    }
                    if (ok) { bestOff = cand; found = true; }
                }
                if (!found) bestOff = (placed.size() % 2 == 0 ? 1 : -1) * pointRadius * (1 + placed.size() * 0.1);
                placed.add(new double[]{bestOff, yy});
                xs[row] = catIdx + bestOff;
                ys[row] = yy;
            }
            catIdx++;
        }
        return remember(new ScatterChart("swarmplot", xs, ys)
            .setPointSize(Math.max(2, size))
            .setXAxisLabel(x).setYAxisLabel(y));
    }

    public static BarChart barplot(DataFrame df, String x, String y) {
        return barplot(df, x, y, "ci");
    }

    /**
     * seaborn {@code barplot(..., errorbar="sd"|"se"|"ci"|null)}.
     * {@code "ci"} uses mean ± 1.96·SE (95% approx); {@code "sd"} uses std;
     * {@code "se"} uses standard error; {@code "none"}/null disables whiskers.
     */
    public static BarChart barplot(DataFrame df, String x, String y, String errorbar) {
        LinkedHashMap<String, List<Double>> groups = groupBy(df, x, y);
        String[] cats = groups.keySet().toArray(new String[0]);
        double[] means = new double[cats.length];
        double[] errs = new double[cats.length];
        for (int i = 0; i < cats.length; i++) {
            List<Double> vals = groups.get(cats[i]);
            means[i] = mean(vals);
            errs[i] = errorMagnitude(vals, errorbar);
        }
        BarChart chart = new BarChart("barplot", cats, means)
            .setXAxisLabel(x).setYAxisLabel(y);
        if (errorbar != null && !"none".equalsIgnoreCase(errorbar) && !"null".equalsIgnoreCase(errorbar)) {
            chart.setError(errs);
        }
        return remember(chart);
    }

    public static BarChart countplot(DataFrame df, String x) {
        Map<Object, Integer> vc = df.valueCounts(x);
        String[] cats = new String[vc.size()];
        double[] counts = new double[vc.size()];
        int i = 0;
        for (Map.Entry<Object, Integer> e : vc.entrySet()) {
            cats[i] = e.getKey() == null ? "null" : e.getKey().toString();
            counts[i] = e.getValue();
            i++;
        }
        return remember(new BarChart("countplot", cats, counts).setXAxisLabel(x).setYAxisLabel("count"));
    }

    public static LineChart pointplot(DataFrame df, String x, String y) {
        return pointplot(df, x, y, null, "ci");
    }

    public static LineChart pointplot(DataFrame df, String x, String y, String hue) {
        return pointplot(df, x, y, hue, "ci");
    }

    public static LineChart pointplot(DataFrame df, String x, String y, String hue, String errorbar) {
        if (hue == null) {
            LinkedHashMap<String, List<Double>> groups = groupBy(df, x, y);
            double[] xs = new double[groups.size()];
            double[] ys = new double[groups.size()];
            double[] err = new double[groups.size()];
            int i = 0;
            for (List<Double> vals : groups.values()) {
                xs[i] = i;
                ys[i] = mean(vals);
                err[i] = errorMagnitude(vals, errorbar);
                i++;
            }
            LineChart c = new LineChart("pointplot", xs, ys, y)
                .setShowMarkers(true).setMarkerSize(7)
                .setXAxisLabel(x).setYAxisLabel(y);
            if (errorbar != null && !"none".equalsIgnoreCase(errorbar)) c.setError(0, err);
            return remember(c);
        }
        // multi-hue: for each hue, aggregate mean(y) per x-category
        LinkedHashMap<String, Integer> xCats = new LinkedHashMap<>();
        LinkedHashMap<String, LinkedHashMap<String, List<Double>>> nested = new LinkedHashMap<>();
        int n = df.rowCount();
        for (int i = 0; i < n; i++) {
            Object xv = DataValues.unwrap(df.get(i, x));
            Object hv = DataValues.unwrap(df.get(i, hue));
            String xk = xv == null ? "null" : xv.toString();
            String hk = hv == null ? "null" : hv.toString();
            xCats.computeIfAbsent(xk, k -> xCats.size());
            double v = DataValues.asDouble(df.get(i, y));
            if (Double.isNaN(v)) continue;
            nested.computeIfAbsent(hk, k -> new LinkedHashMap<>())
                .computeIfAbsent(xk, k -> new ArrayList<>()).add(v);
        }
        LineChart chart = null;
        int seriesIdx = 0;
        for (Map.Entry<String, LinkedHashMap<String, List<Double>>> he : nested.entrySet()) {
            double[] xs = new double[xCats.size()];
            double[] ys = new double[xCats.size()];
            double[] err = new double[xCats.size()];
            Arrays.fill(ys, Double.NaN);
            Arrays.fill(err, Double.NaN);
            for (int i = 0; i < xCats.size(); i++) xs[i] = i;
            for (Map.Entry<String, List<Double>> xe : he.getValue().entrySet()) {
                Integer idx = xCats.get(xe.getKey());
                if (idx == null) continue;
                ys[idx] = mean(xe.getValue());
                err[idx] = errorMagnitude(xe.getValue(), errorbar);
            }
            if (chart == null) chart = new LineChart("pointplot", xs, ys, he.getKey());
            else chart.addSeries(xs, ys, he.getKey());
            if (errorbar != null && !"none".equalsIgnoreCase(errorbar)) chart.setError(seriesIdx, err);
            seriesIdx++;
        }
        if (chart == null) return pointplot(df, x, y, null, errorbar);
        chart.setShowMarkers(true).setMarkerSize(7).setXAxisLabel(x).setYAxisLabel(y);
        chart.setShowLegend(true);
        return remember(chart);
    }

    // ---- matrix ----

    public static HeatmapChart heatmap(double[][] matrix, List<String> rowLabels, List<String> colLabels) {
        return Matplotlib.heatmap(matrix, rowLabels, colLabels).setTitle("heatmap");
    }

    public static HeatmapChart heatmap(double[][] matrix) {
        return heatmap(matrix, null, null);
    }

    /**
     * seaborn {@code heatmap(mat, annot=True, cmap="coolwarm", vmin=0, vmax=1)}.
     */
    public static HeatmapChart heatmap(double[][] matrix, boolean annot, String cmap,
                                       Double vmin, Double vmax) {
        HeatmapChart c = heatmap(matrix);
        c.setAnnot(annot);
        if (cmap != null) c.setCmap(cmap);
        if (vmin != null) c.setVmin(vmin);
        if (vmax != null) c.setVmax(vmax);
        return c;
    }

    public static HeatmapChart heatmap(DataFrame df) {
        DataFrame corr = df.corr();
        List<String> labels = new ArrayList<>();
        for (int i = 1; i < corr.columnCount(); i++) labels.add(corr.column(i).name());
        int n = labels.size();
        double[][] m = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                Object v = corr.get(i, labels.get(j));
                m[i][j] = v instanceof Number ? ((Number) v).doubleValue() : Double.NaN;
            }
        }
        return heatmap(m, labels, labels);
    }

    public static HeatmapChart clustermap(DataFrame df) {
        return clustermap(dfToMatrix(df), null, null, "viridis");
    }

    public static HeatmapChart clustermap(double[][] matrix) {
        return clustermap(matrix, null, null, "viridis");
    }

    public static HeatmapChart clustermap(double[][] matrix, String cmap) {
        return clustermap(matrix, null, null, cmap);
    }

    /**
     * Hierarchical clustering (average-linkage on correlation distance) of rows
     * and columns, then heatmap of the reordered matrix. Approximates seaborn
     * {@code clustermap} without dendrogram side panels.
     */
    public static HeatmapChart clustermap(double[][] matrix, List<String> rowLabels,
                                          List<String> colLabels, String cmap) {
        if (matrix == null || matrix.length == 0) {
            return heatmap(new double[][]{{0}}, rowLabels, colLabels).setTitle("clustermap");
        }
        int rows = matrix.length;
        int cols = matrix[0].length;
        int[] rowOrder = clusterOrder(matrix, true);
        // transpose-ish for column clustering: build col-feature vectors
        double[][] colMajor = new double[cols][rows];
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < cols; c++)
                colMajor[c][r] = matrix[r][c];
        int[] colOrder = clusterOrder(colMajor, true);

        double[][] reordered = new double[rows][cols];
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                reordered[i][j] = matrix[rowOrder[i]][colOrder[j]];

        List<String> rLab = reorderLabels(rowLabels, rowOrder, rows, "r");
        List<String> cLab = reorderLabels(colLabels, colOrder, cols, "c");
        HeatmapChart c = heatmap(reordered, rLab, cLab).setTitle("clustermap");
        if (cmap != null) c.setCmap(cmap);
        return c;
    }

    // ---- regression ----

    public static ScatterChart regplot(DataFrame df, String x, String y) {
        double[] xs = df.column(x).asDoubleArray();
        double[] ys = df.column(y).asDoubleArray();
        return regplot(xs, ys, x, y);
    }

    public static ScatterChart regplot(double[] xs, double[] ys) {
        return regplot(xs, ys, "x", "y");
    }

    public static ScatterChart regplot(double[] xs, double[] ys, String xLabel, String yLabel) {
        int n = 0;
        double sx = 0, sy = 0, sxx = 0, sxy = 0;
        for (int i = 0; i < xs.length && i < ys.length; i++) {
            if (Double.isNaN(xs[i]) || Double.isNaN(ys[i])) continue;
            n++; sx += xs[i]; sy += ys[i]; sxx += xs[i] * xs[i]; sxy += xs[i] * ys[i];
        }
        ScatterChart sc = new ScatterChart("regplot", xs, ys);
        sc.setXAxisLabel(xLabel).setYAxisLabel(yLabel).setShowRegression(true);
        if (n >= 2) {
            double denom = n * sxx - sx * sx;
            double slope = denom == 0 ? 0 : (n * sxy - sx * sy) / denom;
            double intercept = (sy - slope * sx) / n;
            sc.setTitle(String.format(Locale.ROOT, "regplot (y=%.3fx%+.3f)", slope, intercept));
        }
        return remember(sc);
    }

    public static ScatterChart lmplot(DataFrame df, String x, String y) {
        return regplot(df, x, y).setTitle("lmplot");
    }

    public static ScatterChart lmplot(DataFrame df, String x, String y, String hue) {
        if (hue == null) return lmplot(df, x, y);
        // Draw per-hue scatter + overall regression of combined? seaborn draws per-hue fits.
        // We render multi-hue scatter; regression of full set as reference.
        ScatterChart sc = scatterplot(df, x, y, hue).setTitle("lmplot").setShowRegression(true);
        return sc;
    }

    // ---- multi ----

    public static BaseChart pairplot(DataFrame df) {
        List<String> nums = numericCols(df);
        if (nums.isEmpty()) throw new IllegalArgumentException("no numeric columns for pairplot");
        if (nums.size() == 1) return histplot(df, nums.get(0));
        int k = Math.min(nums.size(), 4);
        int cell = 180;
        int size = cell * k + 40;
        BufferedImage img = new BufferedImage(size, size, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = img.createGraphics();
        g.setColor(Color.WHITE);
        g.fillRect(0, 0, size, size);
        g.setFont(new Font("SansSerif", Font.PLAIN, 10));
        for (int i = 0; i < k; i++) {
            for (int j = 0; j < k; j++) {
                BaseChart cellChart;
                if (i == j) cellChart = new HistogramChart(nums.get(i), df.column(nums.get(i)).asDoubleArray(), 15);
                else cellChart = new ScatterChart("", df.column(nums.get(j)).asDoubleArray(),
                    df.column(nums.get(i)).asDoubleArray());
                cellChart.setSize(cell - 4, cell - 4).setTitle("");
                BufferedImage cimg = cellChart.render();
                g.drawImage(cimg, 20 + j * cell, 20 + i * cell, null);
                if (i == k - 1) {
                    g.setColor(Color.DARK_GRAY);
                    g.drawString(nums.get(j), 20 + j * cell + 8, size - 8);
                }
                if (j == 0) {
                    g.setColor(Color.DARK_GRAY);
                    g.drawString(nums.get(i), 2, 20 + i * cell + cell / 2);
                }
            }
        }
        g.dispose();
        return remember(new ImageChart("pairplot", img));
    }

    public static BaseChart jointplot(DataFrame df, String x, String y) {
        return jointplot(df, x, y, "scatter", false);
    }

    public static BaseChart jointplot(DataFrame df, String x, String y, String kind) {
        return jointplot(df, x, y, kind, false);
    }

    /**
     * seaborn {@code jointplot(x, y, kind="scatter"|"kde"|"hist"|"reg", fill=...)}.
     * Returns a composite ImageChart: main panel + top/right marginals.
     */
    public static BaseChart jointplot(DataFrame df, String x, String y, String kind, boolean fill) {
        int w = 520, h = 520, m = 90;
        BufferedImage img = new BufferedImage(w, h, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = img.createGraphics();
        g.setColor(Color.WHITE);
        g.fillRect(0, 0, w, h);

        String k = kind == null ? "scatter" : kind.toLowerCase(Locale.ROOT);
        BaseChart main;
        double[] xs = df.column(x).asDoubleArray();
        double[] ys = df.column(y).asDoubleArray();
        switch (k) {
            case "kde" -> {
                // 2D KDE contour-ish: render density as alpha-scattered grid + optional fill
                main = kde2dChart(xs, ys, fill);
            }
            case "hist" -> main = new HistogramChart("", xs, 20); // fallback main as x hist — rare
            case "reg" -> main = regplot(xs, ys, x, y).setTitle("");
            default -> main = new ScatterChart("", df, x, y);
        }
        main.setSize(w - m - 20, h - m - 20).setTitle("");
        g.drawImage(main.render(), 10, m, null);

        HistogramChart hx = new HistogramChart("", xs, 20);
        hx.setSize(w - m - 20, m - 10).setTitle("");
        g.drawImage(hx.render(), 10, 5, null);

        // right marginal: horizontal-ish via rotated hist of y — render as hist and draw
        HistogramChart hy = new HistogramChart("", ys, 20);
        hy.setSize(m - 10, h - m - 20).setTitle("");
        g.drawImage(hy.render(), w - m + 5, m, null);

        g.setColor(Color.DARK_GRAY);
        g.setFont(new Font("SansSerif", Font.PLAIN, 11));
        g.drawString(x, w / 2, h - 8);
        g.drawString(y, 4, h / 2);
        g.setFont(new Font("SansSerif", Font.BOLD, 12));
        g.drawString("jointplot (" + k + ")", 12, h - 8);
        g.dispose();
        return remember(new ImageChart("jointplot", img));
    }

    // ---- FacetGrid ----

    /**
     * seaborn.FacetGrid — split a DataFrame by column (and optional row) categories
     * and map a plotting function onto each facet.
     */
    public static final class FacetGrid {
        private final DataFrame data;
        private final String col;
        private final String row;
        private final List<String> colLevels;
        private final List<String> rowLevels;
        private final List<BufferedImage> cells = new ArrayList<>();
        private final List<String> cellTitles = new ArrayList<>();
        private int cellW = 280, cellH = 220;
        private String title = "FacetGrid";

        public FacetGrid(DataFrame data, String col) {
            this(data, col, null);
        }

        public FacetGrid(DataFrame data, String col, String row) {
            this.data = data;
            this.col = col;
            this.row = row;
            this.colLevels = uniqueLevels(data, col);
            this.rowLevels = row == null ? List.of("") : uniqueLevels(data, row);
        }

        public FacetGrid setCellSize(int w, int h) {
            this.cellW = w; this.cellH = h; return this;
        }

        public FacetGrid setTitle(String t) { this.title = t; return this; }

        /**
         * Map a single-column plot function (e.g. histplot on a value column)
         * across facets: {@code g.map(Seaborn::histplot, "value")}-style via
         * {@code mapHist("value")} helpers, or the generic {@link #map}.
         */
        public FacetGrid map(BiFunction<DataFrame, String, BaseChart> fn, String column) {
            cells.clear();
            cellTitles.clear();
            for (String rLv : rowLevels) {
                for (String cLv : colLevels) {
                    DataFrame sub = filterFacet(data, col, cLv, row, rLv.isEmpty() ? null : rLv);
                    BaseChart chart = fn.apply(sub, column);
                    if (chart != null) {
                        chart.setSize(cellW, cellH);
                        String t = row == null ? cLv : (rLv + " | " + cLv);
                        chart.setTitle(t);
                        cells.add(chart.render());
                        cellTitles.add(t);
                    }
                }
            }
            return this;
        }

        /** Convenience: map histplot of {@code valueCol} onto each facet. */
        public FacetGrid mapHist(String valueCol) {
            return mapHist(valueCol, 20);
        }

        public FacetGrid mapHist(String valueCol, int bins) {
            return map((sub, colName) -> {
                try {
                    return histplot(sub, colName, bins);
                } catch (Throwable t) {
                    return new HistogramChart(colName, new double[]{0}, 1);
                }
            }, valueCol);
        }

        public FacetGrid mapKde(String valueCol) {
            return map((sub, colName) -> kdeplot(sub, colName), valueCol);
        }

        public FacetGrid mapScatter(String xCol, String yCol) {
            cells.clear();
            cellTitles.clear();
            for (String rLv : rowLevels) {
                for (String cLv : colLevels) {
                    DataFrame sub = filterFacet(data, col, cLv, row, rLv.isEmpty() ? null : rLv);
                    ScatterChart chart = scatterplot(sub, xCol, yCol);
                    chart.setSize(cellW, cellH);
                    String t = row == null ? cLv : (rLv + " | " + cLv);
                    chart.setTitle(t);
                    cells.add(chart.render());
                    cellTitles.add(t);
                }
            }
            return this;
        }

        public BaseChart render() {
            if (cells.isEmpty()) {
                // default empty
                BufferedImage empty = new BufferedImage(cellW, cellH, BufferedImage.TYPE_INT_RGB);
                Graphics2D g = empty.createGraphics();
                g.setColor(Color.WHITE);
                g.fillRect(0, 0, cellW, cellH);
                g.setColor(Color.GRAY);
                g.drawString("empty FacetGrid", 20, 40);
                g.dispose();
                return remember(new ImageChart(title, empty));
            }
            int nCol = Math.max(1, colLevels.size());
            int nRow = Math.max(1, rowLevels.size());
            int pad = 8;
            int W = nCol * cellW + (nCol + 1) * pad;
            int H = nRow * cellH + (nRow + 1) * pad + 24;
            BufferedImage img = new BufferedImage(W, H, BufferedImage.TYPE_INT_RGB);
            Graphics2D g = img.createGraphics();
            g.setColor(Color.WHITE);
            g.fillRect(0, 0, W, H);
            g.setColor(Color.BLACK);
            g.setFont(new Font("SansSerif", Font.BOLD, 14));
            g.drawString(title, pad, 18);
            int idx = 0;
            for (int r = 0; r < nRow; r++) {
                for (int c = 0; c < nCol; c++) {
                    if (idx >= cells.size()) break;
                    int px = pad + c * (cellW + pad);
                    int py = 24 + pad + r * (cellH + pad);
                    g.drawImage(cells.get(idx), px, py, null);
                    idx++;
                }
            }
            g.dispose();
            return remember(new ImageChart(title, img));
        }

        public void savefig(String path) throws Exception {
            render().savefig(path);
        }
    }

    public static FacetGrid FacetGrid(DataFrame df, String col) {
        return new FacetGrid(df, col);
    }

    public static FacetGrid FacetGrid(DataFrame df, String col, String row) {
        return new FacetGrid(df, col, row);
    }

    // =========================================================================
    // NDArray (numpy) + Tensor overloads — three data backends for plotting
    // =========================================================================

    // ---- distribution ----

    public static HistogramChart histplot(NDArray data, int bins) {
        return histplot(PlotInputs.asDouble1D(data), bins);
    }

    public static HistogramChart histplot(NDArray data, int bins, boolean kde) {
        return histplot(PlotInputs.asDouble1D(data), bins, kde);
    }

    public static HistogramChart histplot(Tensor data, int bins) {
        return histplot(PlotInputs.asDouble1D(data), bins);
    }

    public static HistogramChart histplot(Tensor data, int bins, boolean kde) {
        return histplot(PlotInputs.asDouble1D(data), bins, kde);
    }

    public static LineChart kdeplot(NDArray data) {
        return kdeplot(PlotInputs.asDouble1D(data));
    }

    public static LineChart kdeplot(NDArray data, String label) {
        return kdeplot(PlotInputs.asDouble1D(data), label);
    }

    public static LineChart kdeplot(LineChart existing, NDArray data, String label) {
        return kdeplot(existing, PlotInputs.asDouble1D(data), label);
    }

    public static LineChart kdeplot(Tensor data) {
        return kdeplot(PlotInputs.asDouble1D(data));
    }

    public static LineChart kdeplot(Tensor data, String label) {
        return kdeplot(PlotInputs.asDouble1D(data), label);
    }

    public static LineChart kdeplot(LineChart existing, Tensor data, String label) {
        return kdeplot(existing, PlotInputs.asDouble1D(data), label);
    }

    public static LineChart ecdfplot(NDArray data) {
        return ecdfplot(PlotInputs.asDouble1D(data));
    }

    public static LineChart ecdfplot(Tensor data) {
        return ecdfplot(PlotInputs.asDouble1D(data));
    }

    // ---- relational ----

    public static LineChart lineplot(NDArray x, NDArray y) {
        return lineplot(PlotInputs.asDouble1D(x), PlotInputs.asDouble1D(y));
    }

    public static LineChart lineplot(Tensor x, Tensor y) {
        return lineplot(PlotInputs.asDouble1D(x), PlotInputs.asDouble1D(y));
    }

    public static ScatterChart scatterplot(NDArray x, NDArray y) {
        return scatterplot(PlotInputs.asDouble1D(x), PlotInputs.asDouble1D(y));
    }

    public static ScatterChart scatterplot(Tensor x, Tensor y) {
        return scatterplot(PlotInputs.asDouble1D(x), PlotInputs.asDouble1D(y));
    }

    // ---- regression ----

    public static ScatterChart regplot(NDArray x, NDArray y) {
        return regplot(PlotInputs.asDouble1D(x), PlotInputs.asDouble1D(y));
    }

    public static ScatterChart regplot(Tensor x, Tensor y) {
        return regplot(PlotInputs.asDouble1D(x), PlotInputs.asDouble1D(y));
    }

    // ---- matrix ----

    public static HeatmapChart heatmap(NDArray matrix) {
        return heatmap(PlotInputs.asDouble2D(matrix));
    }

    public static HeatmapChart heatmap(NDArray matrix, boolean annot, String cmap,
                                        double vmin, double vmax) {
        return heatmap(PlotInputs.asDouble2D(matrix), annot, cmap, vmin, vmax);
    }

    public static HeatmapChart heatmap(Tensor matrix) {
        return heatmap(PlotInputs.asDouble2D(matrix));
    }

    public static HeatmapChart heatmap(Tensor matrix, boolean annot, String cmap,
                                        double vmin, double vmax) {
        return heatmap(PlotInputs.asDouble2D(matrix), annot, cmap, vmin, vmax);
    }

    public static HeatmapChart clustermap(NDArray matrix) {
        return clustermap(PlotInputs.asDouble2D(matrix));
    }

    public static HeatmapChart clustermap(NDArray matrix, String cmap) {
        return clustermap(PlotInputs.asDouble2D(matrix), cmap);
    }

    public static HeatmapChart clustermap(Tensor matrix) {
        return clustermap(PlotInputs.asDouble2D(matrix));
    }

    public static HeatmapChart clustermap(Tensor matrix, String cmap) {
        return clustermap(PlotInputs.asDouble2D(matrix), cmap);
    }

    // ---- categorical via labeled groups (avoids fake column-less DF) ----

    public static BoxChart boxplot(String[] labels, NDArray... groups) {
        return Matplotlib.boxplot(labels, toDoubleGroups(groups));
    }

    public static BoxChart boxplot(String[] labels, Tensor... groups) {
        return Matplotlib.boxplot(labels, toDoubleGroups(groups));
    }

    public static ViolinChart violinplot(String[] labels, NDArray... groups) {
        DataFrame df = PlotInputs.groupValueDataFrame("group", "value", labels, groups);
        return violinplot(df, "group", "value");
    }

    public static ViolinChart violinplot(String[] labels, Tensor... groups) {
        DataFrame df = PlotInputs.groupValueDataFrame("group", "value", labels, groups);
        return violinplot(df, "group", "value");
    }

    public static ScatterChart stripplot(String[] labels, NDArray... groups) {
        DataFrame df = PlotInputs.groupValueDataFrame("group", "value", labels, groups);
        return stripplot(df, "group", "value");
    }

    public static ScatterChart stripplot(String[] labels, Tensor... groups) {
        DataFrame df = PlotInputs.groupValueDataFrame("group", "value", labels, groups);
        return stripplot(df, "group", "value");
    }

    public static ScatterChart swarmplot(String[] labels, NDArray... groups) {
        DataFrame df = PlotInputs.groupValueDataFrame("group", "value", labels, groups);
        return swarmplot(df, "group", "value");
    }

    public static ScatterChart swarmplot(String[] labels, Tensor... groups) {
        DataFrame df = PlotInputs.groupValueDataFrame("group", "value", labels, groups);
        return swarmplot(df, "group", "value");
    }

    public static BarChart barplot(String[] categories, NDArray values) {
        return Matplotlib.bar(categories, PlotInputs.asDouble1D(values)).setTitle("barplot");
    }

    public static BarChart barplot(String[] categories, Tensor values) {
        return Matplotlib.bar(categories, PlotInputs.asDouble1D(values)).setTitle("barplot");
    }

    /** Build a numeric DataFrame from a 2D NDArray + column names (for pairplot/jointplot). */
    public static DataFrame dataframeFrom(NDArray matrix2d, String... colNames) {
        double[][] m = PlotInputs.asDouble2D(matrix2d);
        return matrixToDf(m, colNames);
    }

    public static DataFrame dataframeFrom(Tensor matrix2d, String... colNames) {
        double[][] m = PlotInputs.asDouble2D(matrix2d);
        return matrixToDf(m, colNames);
    }

    private static DataFrame matrixToDf(double[][] m, String... colNames) {
        if (m.length == 0) return DataFrame.create();
        int cols = m[0].length;
        String[] names = new String[cols];
        for (int c = 0; c < cols; c++)
            names[c] = colNames != null && c < colNames.length && colNames[c] != null
                ? colNames[c] : ("f" + (c + 1));
        DataFrame df = DataFrame.create();
        for (String n : names) df.addColumn(n, Column.DType.FLOAT64);
        for (double[] row : m) {
            Object[] r = new Object[cols];
            for (int c = 0; c < cols; c++) r[c] = c < row.length ? row[c] : Double.NaN;
            df.addRow(r);
        }
        return df;
    }

    private static double[][] toDoubleGroups(NDArray[] groups) {
        double[][] out = new double[groups.length][];
        for (int i = 0; i < groups.length; i++) out[i] = PlotInputs.asDouble1D(groups[i]);
        return out;
    }

    private static double[][] toDoubleGroups(Tensor[] groups) {
        double[][] out = new double[groups.length][];
        for (int i = 0; i < groups.length; i++) out[i] = PlotInputs.asDouble1D(groups[i]);
        return out;
    }

    // ---- helpers (public for HistogramChart KDE overlay) ----

    /** Public KDE grid accessor used by {@link HistogramChart#setKde}. */
    public static double[][] kdeGridPublic(double[] data, int grid) {
        return kdeGrid(data, grid);
    }

    // ---- private helpers ----

    private static List<String> numericCols(DataFrame df) {
        List<String> out = new ArrayList<>();
        for (Column c : df.columns()) {
            switch (c.dtype()) {
                case INT32, INT64, FLOAT32, FLOAT64 -> out.add(c.name());
                default -> {}
            }
        }
        return out;
    }

    private static Map<String, List<double[]>> splitXYByHue(DataFrame df, String x, String y, String hue) {
        Map<String, List<Integer>> idx = new LinkedHashMap<>();
        int n = df.rowCount();
        for (int i = 0; i < n; i++) {
            Object h = DataValues.unwrap(df.get(i, hue));
            String key = h == null ? "null" : h.toString();
            idx.computeIfAbsent(key, k -> new ArrayList<>()).add(i);
        }
        Map<String, List<double[]>> out = new LinkedHashMap<>();
        for (Map.Entry<String, List<Integer>> e : idx.entrySet()) {
            List<Integer> rows = e.getValue();
            double[] xs = new double[rows.size()];
            double[] ys = new double[rows.size()];
            for (int i = 0; i < rows.size(); i++) {
                xs[i] = DataValues.asDouble(df.get(rows.get(i), x));
                ys[i] = DataValues.asDouble(df.get(rows.get(i), y));
            }
            out.put(e.getKey(), List.of(xs, ys));
        }
        return out;
    }

    private static Map<String, double[]> splitColByHue(DataFrame df, String column, String hue) {
        Map<String, List<Double>> tmp = new LinkedHashMap<>();
        int n = df.rowCount();
        for (int i = 0; i < n; i++) {
            Object h = DataValues.unwrap(df.get(i, hue));
            String key = h == null ? "null" : h.toString();
            double v = DataValues.asDouble(df.get(i, column));
            if (Double.isNaN(v)) continue;
            tmp.computeIfAbsent(key, k -> new ArrayList<>()).add(v);
        }
        Map<String, double[]> out = new LinkedHashMap<>();
        for (Map.Entry<String, List<Double>> e : tmp.entrySet()) {
            double[] a = new double[e.getValue().size()];
            for (int i = 0; i < a.length; i++) a[i] = e.getValue().get(i);
            out.put(e.getKey(), a);
        }
        return out;
    }

    private static LinkedHashMap<String, List<Double>> groupBy(DataFrame df, String x, String y) {
        LinkedHashMap<String, List<Double>> groups = new LinkedHashMap<>();
        int n = df.rowCount();
        for (int i = 0; i < n; i++) {
            Object c = DataValues.unwrap(df.get(i, x));
            String key = c == null ? "null" : c.toString();
            double v = DataValues.asDouble(df.get(i, y));
            if (Double.isNaN(v)) continue;
            groups.computeIfAbsent(key, k -> new ArrayList<>()).add(v);
        }
        return groups;
    }

    private static double mean(List<Double> vals) {
        if (vals == null || vals.isEmpty()) return 0;
        double s = 0;
        for (double v : vals) s += v;
        return s / vals.size();
    }

    private static double std(List<Double> vals) {
        if (vals == null || vals.size() < 2) return 0;
        double m = mean(vals);
        double s = 0;
        for (double v : vals) s += (v - m) * (v - m);
        return Math.sqrt(s / (vals.size() - 1));
    }

    private static double errorMagnitude(List<Double> vals, String errorbar) {
        if (vals == null || vals.isEmpty() || errorbar == null) return 0;
        String e = errorbar.toLowerCase(Locale.ROOT);
        double sd = std(vals);
        return switch (e) {
            case "sd", "std" -> sd;
            case "se", "sem" -> sd / Math.sqrt(vals.size());
            case "ci", "ci95" -> 1.96 * sd / Math.sqrt(Math.max(1, vals.size()));
            case "none", "null" -> 0;
            default -> 1.96 * sd / Math.sqrt(Math.max(1, vals.size()));
        };
    }

    private static LineChart ecdfFromArray(double[] data, String xLabel, String series) {
        double[][] xy = ecdfXY(data);
        return remember(new LineChart("ecdfplot", xy[0], xy[1], series)
            .setXAxisLabel(xLabel).setYAxisLabel("proportion"));
    }

    private static double[][] ecdfXY(double[] data) {
        double[] sorted = Arrays.copyOf(data, data.length);
        Arrays.sort(sorted);
        int n = 0;
        for (double v : sorted) if (!Double.isNaN(v)) n++;
        double[] x = new double[n];
        double[] y = new double[n];
        int k = 0;
        for (double v : sorted) {
            if (Double.isNaN(v)) continue;
            x[k] = v;
            y[k] = (k + 1.0) / n;
            k++;
        }
        return new double[][]{x, y};
    }

    private static double[][] kdeGrid(double[] data, int grid) {
        List<Double> vals = new ArrayList<>();
        for (double v : data) if (!Double.isNaN(v)) vals.add(v);
        if (vals.isEmpty()) return new double[][]{new double[]{0}, new double[]{0}};
        double min = vals.stream().mapToDouble(d -> d).min().orElse(0);
        double max = vals.stream().mapToDouble(d -> d).max().orElse(1);
        if (max == min) { max = min + 1; min = min - 1; }
        // pad range a bit
        double pad = (max - min) * 0.1;
        min -= pad; max += pad;
        double mean = vals.stream().mapToDouble(d -> d).average().orElse(0);
        double var = 0;
        for (double v : vals) var += (v - mean) * (v - mean);
        var /= Math.max(1, vals.size());
        double bw = Math.max(1e-6, Math.sqrt(var) * 1.06 * Math.pow(vals.size(), -0.2));
        double[] xs = new double[grid];
        double[] ys = new double[grid];
        for (int i = 0; i < grid; i++) {
            double x = min + (max - min) * i / (grid - 1.0);
            double s = 0;
            for (double v : vals) {
                double z = (x - v) / bw;
                s += Math.exp(-0.5 * z * z);
            }
            xs[i] = x;
            ys[i] = s / (vals.size() * bw * Math.sqrt(2 * Math.PI));
        }
        return new double[][]{xs, ys};
    }

    private static BaseChart kde2dChart(double[] xs, double[] ys, boolean fill) {
        // Render as scatter with density-colored alpha points + optional coarse density grid
        int n = Math.min(xs.length, ys.length);
        ScatterChart sc = new ScatterChart("kde", xs, ys).setAlpha(fill ? 0.35 : 0.55).setPointSize(5);
        // coarse density background via small heatmap overlay is expensive; keep scatter approximation
        // and title it so callers know kind=kde
        sc.setTitle(fill ? "kde (filled)" : "kde");
        return sc;
    }

    private static void sortByX(double[] xs, double[] ys) {
        Integer[] idx = new Integer[xs.length];
        for (int i = 0; i < idx.length; i++) idx[i] = i;
        Arrays.sort(idx, Comparator.comparingDouble(i -> xs[i]));
        double[] nx = new double[xs.length], ny = new double[ys.length];
        for (int i = 0; i < idx.length; i++) {
            nx[i] = xs[idx[i]];
            ny[i] = i < ys.length ? ys[idx[i]] : Double.NaN;
        }
        System.arraycopy(nx, 0, xs, 0, xs.length);
        System.arraycopy(ny, 0, ys, 0, Math.min(ys.length, ny.length));
    }

    private static double[][] dfToMatrix(DataFrame df) {
        List<String> nums = numericCols(df);
        int cols = nums.size();
        int rows = df.rowCount();
        double[][] m = new double[rows][Math.max(1, cols)];
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                m[r][c] = DataValues.asDouble(df.get(r, nums.get(c)));
            }
        }
        return m;
    }

    /** Average-linkage hierarchical clustering order (correlation distance). */
    private static int[] clusterOrder(double[][] vectors, boolean corrDist) {
        int n = vectors.length;
        int[] order = new int[n];
        for (int i = 0; i < n; i++) order[i] = i;
        if (n <= 2) return order;

        // distance matrix
        double[][] dist = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                double d = corrDist ? corrDistance(vectors[i], vectors[j])
                    : euclidean(vectors[i], vectors[j]);
                dist[i][j] = dist[j][i] = d;
            }
        }
        // agglomerative: maintain active cluster representatives as lists of leaf ids
        List<List<Integer>> clusters = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            List<Integer> leaf = new ArrayList<>();
            leaf.add(i);
            clusters.add(leaf);
        }
        while (clusters.size() > 1) {
            double best = Double.POSITIVE_INFINITY;
            int a = 0, b = 1;
            for (int i = 0; i < clusters.size(); i++) {
                for (int j = i + 1; j < clusters.size(); j++) {
                    double d = avgLink(dist, clusters.get(i), clusters.get(j));
                    if (d < best) { best = d; a = i; b = j; }
                }
            }
            List<Integer> merged = new ArrayList<>(clusters.get(a));
            merged.addAll(clusters.get(b));
            // remove higher index first
            clusters.remove(b);
            clusters.remove(a);
            clusters.add(merged);
        }
        List<Integer> finalOrder = clusters.get(0);
        for (int i = 0; i < n; i++) order[i] = finalOrder.get(i);
        return order;
    }

    private static double avgLink(double[][] dist, List<Integer> a, List<Integer> b) {
        double s = 0;
        int c = 0;
        for (int i : a) for (int j : b) { s += dist[i][j]; c++; }
        return c == 0 ? 0 : s / c;
    }

    private static double corrDistance(double[] a, double[] b) {
        int n = Math.min(a.length, b.length);
        if (n == 0) return 1;
        double ma = 0, mb = 0;
        for (int i = 0; i < n; i++) { ma += a[i]; mb += b[i]; }
        ma /= n; mb /= n;
        double num = 0, da = 0, db = 0;
        for (int i = 0; i < n; i++) {
            double xa = a[i] - ma, xb = b[i] - mb;
            num += xa * xb; da += xa * xa; db += xb * xb;
        }
        double denom = Math.sqrt(da * db);
        double corr = denom < 1e-12 ? 0 : num / denom;
        return 1.0 - corr; // correlation distance
    }

    private static double euclidean(double[] a, double[] b) {
        int n = Math.min(a.length, b.length);
        double s = 0;
        for (int i = 0; i < n; i++) {
            double d = a[i] - b[i];
            s += d * d;
        }
        return Math.sqrt(s);
    }

    private static List<String> reorderLabels(List<String> labels, int[] order, int n, String prefix) {
        List<String> out = new ArrayList<>(n);
        for (int i = 0; i < n; i++) {
            int src = order[i];
            if (labels != null && src < labels.size()) out.add(labels.get(src));
            else out.add(prefix + src);
        }
        return out;
    }

    private static List<String> uniqueLevels(DataFrame df, String col) {
        LinkedHashSet<String> set = new LinkedHashSet<>();
        int n = df.rowCount();
        for (int i = 0; i < n; i++) {
            Object v = DataValues.unwrap(df.get(i, col));
            set.add(v == null ? "null" : v.toString());
        }
        return new ArrayList<>(set);
    }

    private static DataFrame filterFacet(DataFrame df, String col, String colVal,
                                         String row, String rowVal) {
        DataFrame out = DataFrame.create();
        List<Column> cols = df.columns();
        for (Column c : cols) out.addColumn(c.name(), c.dtype());
        int n = df.rowCount();
        int nCol = cols.size();
        for (int i = 0; i < n; i++) {
            Object cv = DataValues.unwrap(df.get(i, col));
            String cs = cv == null ? "null" : cv.toString();
            if (!cs.equals(colVal)) continue;
            if (row != null && rowVal != null) {
                Object rv = DataValues.unwrap(df.get(i, row));
                String rs = rv == null ? "null" : rv.toString();
                if (!rs.equals(rowVal)) continue;
            }
            Object[] vals = new Object[nCol];
            for (int c = 0; c < nCol; c++) vals[c] = df.get(i, cols.get(c).name());
            out.addRow(vals);
        }
        return out;
    }

    private static <T extends BaseChart> T remember(T c) {
        try {
            java.lang.reflect.Field f = Matplotlib.class.getDeclaredField("lastChart");
            f.setAccessible(true);
            f.set(null, c);
        } catch (Exception ignored) {}
        return c;
    }

    /** Chart backed by a pre-rendered image (pairplot/jointplot/FacetGrid). */
    public static final class ImageChart extends BaseChart {
        private final BufferedImage image;
        public ImageChart(String title, BufferedImage image) {
            super(title);
            this.image = image;
            this.width = image.getWidth();
            this.height = image.getHeight();
        }
        @Override public BufferedImage render() { return image; }
    }
}
