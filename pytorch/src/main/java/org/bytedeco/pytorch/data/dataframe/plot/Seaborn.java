package org.bytedeco.pytorch.data.dataframe.plot;

import org.bytedeco.pytorch.data.dataframe.Column;
import org.bytedeco.pytorch.data.dataframe.DataFrame;
import org.bytedeco.pytorch.data.dataframe.DataValues;

import java.awt.*;
import java.awt.image.BufferedImage;
import java.util.*;
import java.util.List;

/**
 * Seaborn-like façade over pure-AWT charts.
 * Covers the common Python seaborn surface: relational, distribution,
 * categorical, matrix, regression, multi-plot, and style helpers.
 */
public final class Seaborn {
    private static String style = "darkgrid";
    private static String paletteName = "deep";
    private static Color[] palette = BaseChart.PALETTE.clone();

    private Seaborn() {}

    // ---- style / palette ----

    public static void set_theme(String styleName) { set_style(styleName); }

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
            if (chart == null) chart = new LineChart("lineplot", xs, ys, e.getKey());
            else chart.addSeries(ys, e.getKey()); // shares first series x-axis
        }
        if (chart == null) chart = Matplotlib.plot(df, x, y);
        return remember(chart.setXAxisLabel(x).setYAxisLabel(y));
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

    public static ScatterChart relplot(DataFrame df, String x, String y) {
        return scatterplot(df, x, y);
    }

    // ---- distribution ----

    public static HistogramChart histplot(DataFrame df, String column, int bins) {
        return Matplotlib.hist(df, column, bins).setTitle("histplot");
    }

    public static HistogramChart histplot(DataFrame df, String column) {
        return histplot(df, column, 20);
    }

    public static LineChart kdeplot(DataFrame df, String column) {
        double[] data = df.column(column).asDoubleArray();
        double[][] grid = kdeGrid(data, 100);
        return remember(new LineChart("kdeplot", grid[0], grid[1], column)
            .setXAxisLabel(column).setYAxisLabel("density"));
    }

    public static LineChart ecdfplot(DataFrame df, String column) {
        double[] data = df.column(column).asDoubleArray();
        Arrays.sort(data);
        int n = 0;
        for (double v : data) if (!Double.isNaN(v)) n++;
        double[] x = new double[n];
        double[] y = new double[n];
        int k = 0;
        for (double v : data) {
            if (Double.isNaN(v)) continue;
            x[k] = v;
            y[k] = (k + 1.0) / n;
            k++;
        }
        return remember(new LineChart("ecdfplot", x, y, "ecdf")
            .setXAxisLabel(column).setYAxisLabel("proportion"));
    }

    // ---- categorical ----

    public static BoxChart boxplot(DataFrame df, String x, String y) {
        return Matplotlib.boxplot(df, x, y).setTitle("boxplot");
    }

    public static ViolinChart violinplot(DataFrame df, String x, String y) {
        return Matplotlib.violinplot(df, x, y).setTitle("violinplot");
    }

    public static ScatterChart stripplot(DataFrame df, String x, String y) {
        LinkedHashMap<String, Integer> cats = new LinkedHashMap<>();
        int n = df.rowCount();
        double[] xs = new double[n];
        double[] ys = new double[n];
        Random rng = new Random(0);
        for (int i = 0; i < n; i++) {
            Object c = DataValues.unwrap(df.get(i, x));
            String key = c == null ? "null" : c.toString();
            int idx = cats.computeIfAbsent(key, k -> cats.size());
            xs[i] = idx + (rng.nextDouble() - 0.5) * 0.25;
            ys[i] = DataValues.asDouble(df.get(i, y));
        }
        return remember(new ScatterChart("stripplot", xs, ys)
            .setXAxisLabel(x).setYAxisLabel(y));
    }

    public static ScatterChart swarmplot(DataFrame df, String x, String y) {
        return stripplot(df, x, y).setTitle("swarmplot");
    }

    public static BarChart barplot(DataFrame df, String x, String y) {
        LinkedHashMap<String, List<Double>> groups = new LinkedHashMap<>();
        int n = df.rowCount();
        for (int i = 0; i < n; i++) {
            Object c = DataValues.unwrap(df.get(i, x));
            String key = c == null ? "null" : c.toString();
            double v = DataValues.asDouble(df.get(i, y));
            if (Double.isNaN(v)) continue;
            groups.computeIfAbsent(key, k -> new ArrayList<>()).add(v);
        }
        String[] cats = groups.keySet().toArray(new String[0]);
        double[] means = new double[cats.length];
        for (int i = 0; i < cats.length; i++) {
            List<Double> vals = groups.get(cats[i]);
            means[i] = vals.stream().mapToDouble(d -> d).average().orElse(0);
        }
        return remember(new BarChart("barplot", cats, means).setXAxisLabel(x).setYAxisLabel(y));
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
        LinkedHashMap<String, List<Double>> groups = new LinkedHashMap<>();
        int n = df.rowCount();
        for (int i = 0; i < n; i++) {
            Object c = DataValues.unwrap(df.get(i, x));
            String key = c == null ? "null" : c.toString();
            double v = DataValues.asDouble(df.get(i, y));
            if (Double.isNaN(v)) continue;
            groups.computeIfAbsent(key, k -> new ArrayList<>()).add(v);
        }
        double[] xs = new double[groups.size()];
        double[] ys = new double[groups.size()];
        int i = 0;
        for (List<Double> vals : groups.values()) {
            xs[i] = i;
            ys[i] = vals.stream().mapToDouble(d -> d).average().orElse(0);
            i++;
        }
        return remember(new LineChart("pointplot", xs, ys, y).setXAxisLabel(x).setYAxisLabel(y));
    }

    // ---- matrix ----

    public static HeatmapChart heatmap(double[][] matrix, List<String> rowLabels, List<String> colLabels) {
        return Matplotlib.heatmap(matrix, rowLabels, colLabels).setTitle("heatmap");
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
        return heatmap(df).setTitle("clustermap");
    }

    // ---- regression ----

    public static ScatterChart regplot(DataFrame df, String x, String y) {
        double[] xs = df.column(x).asDoubleArray();
        double[] ys = df.column(y).asDoubleArray();
        int n = 0;
        double sx = 0, sy = 0, sxx = 0, sxy = 0;
        for (int i = 0; i < xs.length && i < ys.length; i++) {
            if (Double.isNaN(xs[i]) || Double.isNaN(ys[i])) continue;
            n++; sx += xs[i]; sy += ys[i]; sxx += xs[i] * xs[i]; sxy += xs[i] * ys[i];
        }
        ScatterChart sc = new ScatterChart("regplot", xs, ys);
        sc.setXAxisLabel(x).setYAxisLabel(y);
        if (n >= 2) {
            double denom = n * sxx - sx * sx;
            double slope = denom == 0 ? 0 : (n * sxy - sx * sy) / denom;
            double intercept = (sy - slope * sx) / n;
            sc.setTitle(String.format("regplot (y=%.3fx%+.3f)", slope, intercept));
        }
        return remember(sc);
    }

    public static ScatterChart lmplot(DataFrame df, String x, String y) {
        return regplot(df, x, y).setTitle("lmplot");
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
        int w = 500, h = 500, m = 80;
        BufferedImage img = new BufferedImage(w, h, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = img.createGraphics();
        g.setColor(Color.WHITE);
        g.fillRect(0, 0, w, h);

        ScatterChart sc = new ScatterChart("", df, x, y);
        sc.setSize(w - m - 20, h - m - 20).setTitle("");
        g.drawImage(sc.render(), 10, m, null);

        HistogramChart hx = new HistogramChart("", df.column(x).asDoubleArray(), 20);
        hx.setSize(w - m - 20, m - 10).setTitle("");
        g.drawImage(hx.render(), 10, 5, null);

        HistogramChart hy = new HistogramChart("", df.column(y).asDoubleArray(), 20);
        hy.setSize(m - 10, h - m - 20).setTitle("");
        g.drawImage(hy.render(), w - m + 5, m, null);

        g.setColor(Color.DARK_GRAY);
        g.setFont(new Font("SansSerif", Font.PLAIN, 11));
        g.drawString(x, w / 2, h - 8);
        g.drawString(y, 4, h / 2);
        g.dispose();
        return remember(new ImageChart("jointplot", img));
    }

    // ---- helpers ----

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

    private static double[][] kdeGrid(double[] data, int grid) {
        List<Double> vals = new ArrayList<>();
        for (double v : data) if (!Double.isNaN(v)) vals.add(v);
        if (vals.isEmpty()) return new double[][]{new double[]{0}, new double[]{0}};
        double min = vals.stream().mapToDouble(d -> d).min().orElse(0);
        double max = vals.stream().mapToDouble(d -> d).max().orElse(1);
        if (max == min) { max = min + 1; min = min - 1; }
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

    private static <T extends BaseChart> T remember(T c) {
        try {
            java.lang.reflect.Field f = Matplotlib.class.getDeclaredField("lastChart");
            f.setAccessible(true);
            f.set(null, c);
        } catch (Exception ignored) {}
        return c;
    }

    /** Chart backed by a pre-rendered image (pairplot/jointplot). */
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
