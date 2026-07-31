package org.bytedeco.pytorch.data.numpy;

import org.bytedeco.pytorch.plot.chart.AreaChart;
import org.bytedeco.pytorch.plot.chart.BarChart;
import org.bytedeco.pytorch.plot.chart.BaseChart;
import org.bytedeco.pytorch.plot.chart.BoxChart;
import org.bytedeco.pytorch.plot.chart.HeatmapChart;
import org.bytedeco.pytorch.plot.chart.HistogramChart;
import org.bytedeco.pytorch.plot.chart.LineChart;
import org.bytedeco.pytorch.plot.matplot.Matplotlib;
import org.bytedeco.pytorch.plot.chart.PieChart;
import org.bytedeco.pytorch.plot.chart.ScatterChart;

/**
 * Matplotlib-style plotting for {@link NDArray} — thin wrapper that delegates to
 * {@link Matplotlib} so numpy / DataFrame / Tensor entry points share one AWT backend.
 *
 * <pre>
 *   NP.Plot.plot(x, y, "sin").setTitle("Demo").show();
 *   NP.Plot.hist(data, 20).savefig("/tmp/hist.png");
 *   NP.Plot.imshow(matrix).savefig("/tmp/heat.png");
 * </pre>
 */
public final class NPPlot {
    private NPPlot() {}

    public static BaseChart last() { return Matplotlib.last(); }

    // ---- line / plot --------------------------------------------------------

    public static LineChart plot(NDArray y) {
        return Matplotlib.plot(y);
    }

    public static LineChart plot(NDArray x, NDArray y) {
        return Matplotlib.plot(x, y);
    }

    public static LineChart plot(NDArray x, NDArray y, String label) {
        return Matplotlib.plot(x, y, label);
    }

    /** Multi-series: each column of {@code Y} (2D) is a series against {@code x}. */
    public static LineChart plot(NDArray x, NDArray Y, String[] labels) {
        return plotMany(x, Y, labels);
    }

    public static LineChart plotMany(NDArray x, NDArray Y, String[] labels) {
        if (Y.shape.length == 1) return plot(x, Y, labels != null && labels.length > 0 ? labels[0] : "y");
        if (Y.shape.length != 2) throw new IllegalArgumentException("Y must be 1D or 2D");
        double[] xx = Y.shape.length >= 1 ? flat(x) : flat(x);
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
        // keep last chart in Matplotlib state
        Matplotlib.plot(xx, y0, lab0); // sets last; then we return richer multi chart
        return chart;
    }

    public static ScatterChart scatter(NDArray x, NDArray y) {
        return Matplotlib.scatter(x, y);
    }

    public static ScatterChart scatter(NDArray x, NDArray y, String title) {
        ScatterChart c = Matplotlib.scatter(x, y);
        if (title != null) c.setTitle(title);
        return c;
    }

    public static HistogramChart hist(NDArray data, int bins) {
        return Matplotlib.hist(data, bins);
    }

    public static HistogramChart hist(NDArray data) { return hist(data, 10); }

    public static BarChart bar(NDArray values) {
        return Matplotlib.bar(values);
    }

    public static BarChart bar(String[] categories, NDArray values) {
        return Matplotlib.bar(categories, values);
    }

    public static PieChart pie(String[] labels, NDArray values) {
        return Matplotlib.pie(labels, values);
    }

    public static BoxChart boxplot(NDArray values) {
        return Matplotlib.boxplot(values);
    }

    public static AreaChart area(NDArray x, NDArray y) {
        return Matplotlib.area(x, y, "y");
    }

    /** Heatmap / imshow for 2D array. */
    public static HeatmapChart imshow(NDArray a) {
        return Matplotlib.imshow(a);
    }

    public static HeatmapChart heatmap(NDArray a) { return imshow(a); }

    public static HeatmapChart matshow(NDArray a) { return imshow(a); }

    /** Correlation heatmap of 2D data columns. */
    public static HeatmapChart corrplot(NDArray data2d) {
        NDArray c = NPReduce.corrcoef(data2d, false);
        return imshow(c).setTitle("Correlation");
    }

    // ---- pyplot-style state -------------------------------------------------

    /** Block until the plot window is closed. */
    public static void show() { Matplotlib.show(true); }

    /** @param block true = wait for close; false = non-blocking open window */
    public static void show(boolean block) { Matplotlib.show(block); }

    public static void savefig(String path) throws Exception { Matplotlib.savefig(path); }

    public static BaseChart title(String t) { return Matplotlib.title(t); }

    public static BaseChart xlabel(String s) { return Matplotlib.xlabel(s); }

    public static BaseChart ylabel(String s) { return Matplotlib.ylabel(s); }

    public static BaseChart legend(boolean on) { return Matplotlib.legend(on); }

    public static BaseChart grid(boolean on) { return Matplotlib.grid(on); }

    public static BaseChart figsize(int w, int h) { return Matplotlib.figsize(w, h); }

    /** Convenience: plot polynomial fit curve + scatter of data. */
    public static BaseChart polyfitPlot(NDArray x, NDArray y, int deg, int samples) {
        NDArray coef = NPPoly.polyfit(x, y, deg);
        double xmin = NPReduce.min(x), xmax = NPReduce.max(x);
        NDArray xs = NP.linspace(xmin, xmax, samples);
        NDArray ys = NPPoly.polyval(coef, xs);
        scatter(x, y, "data");
        plot(xs, ys, "poly deg " + deg).setTitle("Polyfit deg=" + deg);
        return last();
    }

    private static double[] flat(NDArray a) {
        return NPShape.ravel(a).asDoubleArray();
    }
}
