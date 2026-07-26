package org.bytedeco.pytorch.data.dataframe.plot;

import org.bytedeco.pytorch.data.dataframe.DataFrame;

/**
 * Fluent plot handle returned by {@link DataFrame#plot()}.
 *
 * <pre>
 *   df.plot().line("x", "y").setTitle("t").savefig("out.png");
 *   df.plot().scatter("x", "y").show();
 *   df.plot().hist("score", 20);
 *   df.plot().pie("label", "value");
 * </pre>
 */
public final class DataFramePlot {
    private final DataFrame df;
    private BaseChart chart;

    public DataFramePlot(DataFrame df) {
        this.df = df;
    }

    public LineChart line(String x, String... y) {
        chart = Matplotlib.plot(df, x, y);
        return (LineChart) chart;
    }

    public ScatterChart scatter(String x, String y) {
        chart = Matplotlib.scatter(df, x, y);
        return (ScatterChart) chart;
    }

    public BarChart bar(String x, String y) {
        chart = Matplotlib.bar(df, x, y);
        return (BarChart) chart;
    }

    public HistogramChart hist(String column, int bins) {
        chart = Matplotlib.hist(df, column, bins);
        return (HistogramChart) chart;
    }

    public BoxChart box(String category, String value) {
        chart = Matplotlib.boxplot(df, category, value);
        return (BoxChart) chart;
    }

    public PieChart pie(String label, String value) {
        chart = Matplotlib.pie(df, label, value);
        return (PieChart) chart;
    }

    public AreaChart area(String x, String... y) {
        chart = Matplotlib.area(df, x, y);
        return (AreaChart) chart;
    }

    public ViolinChart violin(String category, String value) {
        chart = Matplotlib.violinplot(df, category, value);
        return (ViolinChart) chart;
    }

    public BubbleChart bubble(String x, String y, String size) {
        chart = Matplotlib.bubble(df, x, y, size);
        return (BubbleChart) chart;
    }

    public RadarChart radar(String category, String value) {
        chart = Matplotlib.radar(df, category, value);
        return (RadarChart) chart;
    }

    public FunnelChart funnel(String stage, String value) {
        chart = Matplotlib.funnel(df, stage, value);
        return (FunnelChart) chart;
    }

    public HeatmapChart heatmap() {
        chart = Seaborn.heatmap(df);
        return (HeatmapChart) chart;
    }

    public BaseChart chart() { return chart; }

    public void show() {
        if (chart != null) chart.show();
        else Matplotlib.show();
    }

    public void savefig(String path) throws Exception {
        if (chart != null) chart.savefig(path);
        else Matplotlib.savefig(path);
    }
}
