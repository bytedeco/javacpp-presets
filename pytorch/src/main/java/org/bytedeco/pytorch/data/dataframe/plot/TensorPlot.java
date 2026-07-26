package org.bytedeco.pytorch.data.dataframe.plot;

import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.data.dataframe.plot.TensorPlotUtils.Layout;

/**
 * Tensor-first plotting façade. Delegates to {@link Matplotlib} so DataFrame,
 * array, and tensor entry points share one AWT backend.
 *
 * <pre>
 *   TensorPlot.line(t).savefig("/tmp/line.png");
 *   TensorPlot.imshow(img, Layout.NCHW).savefig("/tmp/img.png");
 *   TensorPlot.grid(batch).setCols(4).savefig("/tmp/grid.png");
 *   TensorPlot.hist(t, 32);
 * </pre>
 *
 * @see Matplotlib
 * @see TensorPlotUtils
 */
public final class TensorPlot {
    private TensorPlot() {}

    public static BaseChart last() { return Matplotlib.last(); }

    // ---- 1D / 2D series -----------------------------------------------------

    /** Rank-aware line plot (rank 1 y-vs-index; rank 2 rows-as-series). */
    public static LineChart line(Tensor t) { return Matplotlib.plot(t); }

    public static LineChart line(Tensor x, Tensor y) { return Matplotlib.plot(x, y); }

    public static LineChart plot(Tensor t) { return Matplotlib.plot(t); }

    public static LineChart plot(Tensor x, Tensor y) { return Matplotlib.plot(x, y); }

    public static ScatterChart scatter(Tensor x, Tensor y) { return Matplotlib.scatter(x, y); }

    /** Shape (N,2) or (2,N). */
    public static ScatterChart scatter(Tensor t) { return Matplotlib.scatter(t); }

    public static HistogramChart hist(Tensor t, int bins) { return Matplotlib.hist(t, bins); }

    public static HistogramChart hist(Tensor t) { return Matplotlib.hist(t); }

    public static BoxChart box(Tensor t) { return Matplotlib.boxplot(t); }

    public static BoxChart boxplot(Tensor t) { return Matplotlib.boxplot(t); }

    public static ViolinChart violin(Tensor t) { return Matplotlib.violinplot(t); }

    public static ViolinChart violinplot(Tensor t) { return Matplotlib.violinplot(t); }

    public static BarChart bar(Tensor values) { return Matplotlib.bar(values); }

    public static AreaChart area(Tensor y) { return Matplotlib.area(y); }

    public static AreaChart area(Tensor x, Tensor y) { return Matplotlib.area(x, y); }

    public static HeatmapChart heatmap(Tensor t) { return Matplotlib.heatmap(t); }

    // ---- images / grids -----------------------------------------------------

    public static BaseChart imshow(Tensor t) { return Matplotlib.imshow(t); }

    public static BaseChart imshow(Tensor t, Layout layout) { return Matplotlib.imshow(t, layout); }

    public static ImageGridChart grid(Tensor t) { return Matplotlib.imageGrid(t); }

    public static ImageGridChart grid(Tensor t, Layout layout) { return Matplotlib.imageGrid(t, layout); }

    public static ImageGridChart grid(Tensor t, Layout layout, int maxImages) {
        return Matplotlib.imageGrid(t, layout, maxImages);
    }

    public static ImageGridChart imageGrid(Tensor t) { return Matplotlib.imageGrid(t); }

    public static ImageGridChart imageGrid(Tensor t, Layout layout, int maxImages) {
        return Matplotlib.imageGrid(t, layout, maxImages);
    }

    // ---- output -------------------------------------------------------------

    public static void show() { Matplotlib.show(); }

    public static void savefig(String path) throws Exception { Matplotlib.savefig(path); }
}
