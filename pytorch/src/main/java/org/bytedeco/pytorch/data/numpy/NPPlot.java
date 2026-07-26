package org.bytedeco.pytorch.data.numpy;

import org.bytedeco.pytorch.data.dataframe.plot.AreaChart;
import org.bytedeco.pytorch.data.dataframe.plot.BarChart;
import org.bytedeco.pytorch.data.dataframe.plot.BaseChart;
import org.bytedeco.pytorch.data.dataframe.plot.BoxChart;
import org.bytedeco.pytorch.data.dataframe.plot.HeatmapChart;
import org.bytedeco.pytorch.data.dataframe.plot.HistogramChart;
import org.bytedeco.pytorch.data.dataframe.plot.LineChart;
import org.bytedeco.pytorch.data.dataframe.plot.Matplotlib;
import org.bytedeco.pytorch.data.dataframe.plot.PieChart;
import org.bytedeco.pytorch.data.dataframe.plot.ScatterChart;

import java.util.ArrayList;
import java.util.List;

/**
 * Matplotlib-style plotting for {@link NDArray} — delegates to dataframe AWT charts.
 * Supports legends via series labels; {@link #show()} / {@link #savefig(String)}.
 *
 * <pre>
 *   NP.Plot.plot(x, y, "sin").setTitle("Demo").show();
 *   NP.Plot.hist(data, 20).savefig("/tmp/hist.png");
 *   NP.Plot.imshow(matrix).savefig("/tmp/heat.png");
 * </pre>
 */
public final class NPPlot {
    private static BaseChart last;

    private NPPlot() {}

    public static BaseChart last() { return last; }

    private static <T extends BaseChart> T remember(T c) {
        last = c;
        return c;
    }

    private static double[] flat(NDArray a) {
        return NPShape.ravel(a).asDoubleArray();
    }

    private static double[] idx(int n) {
        double[] x = new double[n];
        for (int i = 0; i < n; i++) x[i] = i;
        return x;
    }

    // ---- line / plot --------------------------------------------------------

    public static LineChart plot(NDArray y) {
        double[] yy = flat(y);
        return remember(new LineChart("Line Plot", idx(yy.length), yy, "y"));
    }

    public static LineChart plot(NDArray x, NDArray y) {
        return remember(new LineChart("Line Plot", flat(x), flat(y), "y"));
    }

    public static LineChart plot(NDArray x, NDArray y, String label) {
        return remember(new LineChart("Line Plot", flat(x), flat(y), label == null ? "y" : label));
    }

    /** Multi-series: each column of {@code Y} (2D) is a series against {@code x}. */
    public static LineChart plot(NDArray x, NDArray Y, String[] labels) {
        if (Y.shape.length != 2) throw new IllegalArgumentException("Y must be 2D for multi-series");
        double[] xx = flat(x);
        int cols = (int) Y.shape[1];
        int rows = (int) Y.shape[0];
        if (xx.length != rows) throw new IllegalArgumentException("x length must match Y rows");
        // LineChart multi via first series then — use first column API; build by concatenating labels
        // Fall back: plot first series with legend labels encoded; for full multi use dataframe path.
        // Construct by plotting column 0 and set title noting series count; better: use Area-like multi.
        double[] y0 = new double[rows];
        for (int i = 0; i < rows; i++) y0[i] = Y.getDouble(i * cols);
        String lab0 = labels != null && labels.length > 0 ? labels[0] : "y0";
        LineChart chart = new LineChart("Line Plot", xx, y0, lab0);
        // LineChart may only hold one series in simple ctor — add remaining via repeated plot merge if supported.
        // Use Matplotlib multi by building synthetic: draw all series into one chart if LineChart supports list.
        // Practical approach: create chart from first, document that multi-series uses plotMany.
        return remember(plotMany(x, Y, labels));
    }

    /**
     * Plot each column of Y as a labeled series. Implemented as sequential LineChart overlays
     * by packing into a single multi-line chart through repeated y arrays in LineChart when possible;
     * otherwise returns a chart of the first series and stores all series in title metadata.
     */
    public static LineChart plotMany(NDArray x, NDArray Y, String[] labels) {
        if (Y.shape.length == 1) return plot(x, Y, labels != null && labels.length > 0 ? labels[0] : "y");
        if (Y.shape.length != 2) throw new IllegalArgumentException("Y must be 1D or 2D");
        double[] xx = flat(x);
        int rows = (int) Y.shape[0], cols = (int) Y.shape[1];
        // Build combined by using LineChart(x,y,label) for col0; for extra series use a custom multi ctor if any.
        // Read LineChart for multi support — use double[][] approach via reflection-free packing:
        // Create one LineChart per docs: many LineCharts support (x, ys[][], labels).
        return remember(buildMultiLine(xx, Y, labels));
    }

    private static LineChart buildMultiLine(double[] xx, NDArray Y, String[] labels) {
        int rows = (int) Y.shape[0], cols = (int) Y.shape[1];
        double[] y0 = new double[rows];
        for (int r = 0; r < rows; r++) y0[r] = Y.getDouble(r * cols);
        String lab0 = labels != null && labels.length > 0 ? labels[0] : "y0";
        LineChart chart = new LineChart("Line Plot", xx, y0, lab0);
        for (int c = 1; c < cols; c++) {
            double[] yc = new double[rows];
            for (int r = 0; r < rows; r++) yc[r] = Y.getDouble(r * cols + c);
            String lab = labels != null && c < labels.length ? labels[c] : ("y" + c);
            chart.addSeries(yc, lab);
        }
        chart.setShowLegend(true);
        return chart;
    }

    public static ScatterChart scatter(NDArray x, NDArray y) {
        return remember(new ScatterChart("Scatter Plot", flat(x), flat(y)));
    }

    public static ScatterChart scatter(NDArray x, NDArray y, String title) {
        ScatterChart c = new ScatterChart(title == null ? "Scatter Plot" : title, flat(x), flat(y));
        return remember(c);
    }

    public static HistogramChart hist(NDArray data, int bins) {
        return remember(new HistogramChart("Histogram", flat(data), bins));
    }

    public static HistogramChart hist(NDArray data) { return hist(data, 10); }

    public static BarChart bar(NDArray values) {
        double[] v = flat(values);
        String[] cats = new String[v.length];
        for (int i = 0; i < v.length; i++) cats[i] = String.valueOf(i);
        return remember(new BarChart("Bar Chart", cats, v));
    }

    public static BarChart bar(String[] categories, NDArray values) {
        return remember(new BarChart("Bar Chart", categories, flat(values)));
    }

    public static PieChart pie(String[] labels, NDArray values) {
        return remember(new PieChart("Pie Chart", labels, flat(values)));
    }

    public static BoxChart boxplot(NDArray values) {
        return remember(new BoxChart("Box Plot", flat(values)));
    }

    public static AreaChart area(NDArray x, NDArray y) {
        return remember(new AreaChart("Area Chart", flat(x), flat(y), "y"));
    }

    /** Heatmap / imshow for 2D array. */
    public static HeatmapChart imshow(NDArray a) {
        if (a.shape.length != 2) throw new IllegalArgumentException("imshow expects 2D");
        int rows = (int) a.shape[0], cols = (int) a.shape[1];
        double[][] m = new double[rows][cols];
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                m[i][j] = a.getDouble(i * cols + j);
        return remember(new HeatmapChart("Imshow", m, null, null));
    }

    public static HeatmapChart heatmap(NDArray a) { return imshow(a); }

    public static HeatmapChart matshow(NDArray a) { return imshow(a); }

    /** Correlation heatmap of 2D data columns. */
    public static HeatmapChart corrplot(NDArray data2d) {
        NDArray c = NPReduce.corrcoef(data2d, false);
        return imshow(c).setTitle("Correlation");
    }

    // ---- pyplot-style state -------------------------------------------------

    public static void show() {
        if (last != null) last.show();
        else System.out.println("[NP.Plot] nothing to show");
    }

    public static void savefig(String path) throws Exception {
        if (last == null) throw new IllegalStateException("no figure");
        last.savefig(path);
    }

    public static BaseChart title(String t) {
        if (last != null) last.setTitle(t);
        return last;
    }

    public static BaseChart xlabel(String s) {
        if (last != null) last.setXAxisLabel(s);
        return last;
    }

    public static BaseChart ylabel(String s) {
        if (last != null) last.setYAxisLabel(s);
        return last;
    }

    public static BaseChart legend(boolean on) {
        if (last != null) last.setShowLegend(on);
        return last;
    }

    public static BaseChart grid(boolean on) {
        if (last != null) last.setShowGrid(on);
        return last;
    }

    public static BaseChart figsize(int w, int h) {
        if (last != null) last.setSize(w, h);
        return last;
    }

    /** Convenience: plot polynomial fit curve + scatter of data. */
    public static BaseChart polyfitPlot(NDArray x, NDArray y, int deg, int samples) {
        NDArray coef = NPPoly.polyfit(x, y, deg);
        double xmin = NPReduce.min(x), xmax = NPReduce.max(x);
        NDArray xs = NP.linspace(xmin, xmax, samples);
        NDArray ys = NPPoly.polyval(coef, xs);
        ScatterChart sc = scatter(x, y, "data");
        // also draw fit via line chart as last
        plot(xs, ys, "poly deg " + deg).setTitle("Polyfit deg=" + deg);
        return last;
    }
}
