package org.bytedeco.pytorch.utils.plot;

import org.bytedeco.pytorch.dataframe.DataFrame;

import java.awt.*;
import java.awt.image.BufferedImage;
import java.util.*;
import java.util.List;

/**
 * Bar chart with optional error bars, horizontal orientation (barh),
 * grouped multi-series, and stacked mode (matplotlib bar / barh / bottom=).
 */
public final class BarChart extends BaseChart {
    private final List<String> categories = new ArrayList<>();
    private final List<double[]> series = new ArrayList<>();
    private final List<String> seriesNames = new ArrayList<>();
    private double[] errorLow;   // absolute y low (mean - err)
    private double[] errorHigh;  // absolute y high (mean + err)
    private boolean horizontal = false;
    private boolean stacked = false;
    private Color barColor = null; // optional single-series override

    public BarChart(String title, String[] cats, double[] values) {
        super(title);
        Collections.addAll(categories, cats);
        series.add(values);
        seriesNames.add("value");
    }

    public BarChart(String title, DataFrame df, String xCol, String yCol) {
        super(title);
        int n = df.rowCount();
        for (int i = 0; i < n; i++) {
            Object v = df.get(i, xCol);
            categories.add(v == null ? "" : v.toString());
        }
        series.add(toDoubles(df, yCol));
        seriesNames.add(yCol);
        this.xAxisLabel = xCol;
        this.yAxisLabel = yCol;
    }

    public BarChart addSeries(double[] values, String name) {
        series.add(values);
        seriesNames.add(name == null ? ("s" + series.size()) : name);
        return this;
    }

    /** Symmetric error bars around first series values (absolute half-width). */
    public BarChart setError(double[] err) {
        if (err == null || series.isEmpty()) return this;
        double[] vals = series.get(0);
        this.errorLow = new double[vals.length];
        this.errorHigh = new double[vals.length];
        for (int i = 0; i < vals.length; i++) {
            double e = i < err.length ? Math.abs(err[i]) : 0;
            errorLow[i] = vals[i] - e;
            errorHigh[i] = vals[i] + e;
        }
        return this;
    }

    /** Absolute low/high error whiskers for first series. */
    public BarChart setErrorRange(double[] low, double[] high) {
        this.errorLow = low;
        this.errorHigh = high;
        return this;
    }

    /** matplotlib {@code plt.barh}. */
    public BarChart setHorizontal(boolean v) { this.horizontal = v; return this; }

    /** matplotlib stacked bars via {@code bottom=}. */
    public BarChart setStacked(boolean v) { this.stacked = v; return this; }

    public BarChart setBarColor(Color c) { this.barColor = c; return this; }

    @Override public BarChart setTitle(String t) { super.setTitle(t); return this; }
    @Override public BarChart setXAxisLabel(String l) { super.setXAxisLabel(l); return this; }
    @Override public BarChart setYAxisLabel(String l) { super.setYAxisLabel(l); return this; }
    @Override public BarChart setSize(int w, int h) { super.setSize(w, h); return this; }
    @Override public BarChart setShowLegend(boolean v) { super.setShowLegend(v); return this; }

    @Override
    public BufferedImage render() {
        BufferedImage img = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = img.createGraphics();
        int left = 70, right = 20, top = 40, bottom = 70;
        int plotW = width - left - right, plotH = height - top - bottom;

        int n = categories.size();
        int nSeries = series.size();

        if (horizontal) {
            renderHorizontal(g, left, top, plotW, plotH, n, nSeries);
        } else {
            renderVertical(g, left, top, plotW, plotH, n, nSeries);
        }

        // category labels
        g.setColor(Color.DARK_GRAY);
        g.setFont(new Font("SansSerif", Font.PLAIN, 10));
        if (!horizontal) {
            double groupW = plotW / (double) Math.max(1, n);
            for (int i = 0; i < n; i++) {
                String lab = categories.get(i);
                if (lab.length() > 8) lab = lab.substring(0, 8);
                FontMetrics fm = g.getFontMetrics();
                int x = left + (int) (i * groupW + groupW / 2 - fm.stringWidth(lab) / 2.0);
                g.drawString(lab, x, top + plotH + 28);
            }
        } else {
            double groupH = plotH / (double) Math.max(1, n);
            for (int i = 0; i < n; i++) {
                String lab = categories.get(i);
                if (lab.length() > 10) lab = lab.substring(0, 10);
                FontMetrics fm = g.getFontMetrics();
                int y = top + (int) (i * groupH + groupH / 2 + 4);
                g.drawString(lab, left - fm.stringWidth(lab) - 6, y);
            }
        }

        if (showLegend && seriesNames.size() > 1) {
            int lx = left + plotW - 100, ly = top + 8;
            g.setFont(new Font("SansSerif", Font.PLAIN, 11));
            for (int s = 0; s < seriesNames.size(); s++) {
                g.setColor(PALETTE[s % PALETTE.length]);
                g.fillRect(lx, ly + s * 16, 12, 12);
                g.setColor(Color.BLACK);
                g.drawString(seriesNames.get(s), lx + 16, ly + s * 16 + 11);
            }
        }
        g.dispose();
        return img;
    }

    private void renderVertical(Graphics2D g, int left, int top, int plotW, int plotH, int n, int nSeries) {
        double yMin = 0, yMax = Double.NEGATIVE_INFINITY;
        if (stacked) {
            for (int i = 0; i < n; i++) {
                double sum = 0;
                for (int s = 0; s < nSeries; s++) {
                    double[] vals = series.get(s);
                    if (i < vals.length) sum += vals[i];
                }
                yMax = Math.max(yMax, sum);
            }
        } else {
            for (double[] s : series) yMax = Math.max(yMax, max(s));
        }
        if (errorHigh != null) yMax = Math.max(yMax, max(errorHigh));
        if (errorLow != null) yMin = Math.min(yMin, min(errorLow));
        if (yMax <= yMin) yMax = yMin + 1;
        double[] yr = {yMin, yMax}; padRange(yr); yr[0] = Math.min(0, yr[0]);

        drawAxesFrame(g, left, top, plotW, plotH, 0, Math.max(1, n), yr[0], yr[1]);

        double groupW = plotW / (double) Math.max(1, n);
        double barW = stacked ? groupW * 0.7 : groupW * 0.8 / Math.max(1, nSeries);

        double[] bottoms = stacked ? new double[n] : null;

        for (int s = 0; s < nSeries; s++) {
            Color c = (barColor != null && nSeries == 1) ? barColor : PALETTE[s % PALETTE.length];
            g.setColor(c);
            double[] vals = series.get(s);
            for (int i = 0; i < n && i < vals.length; i++) {
                double base = stacked && bottoms != null ? bottoms[i] : 0;
                double topVal = base + vals[i];
                int x;
                if (stacked) {
                    x = left + (int) (i * groupW + groupW * 0.15);
                } else {
                    x = left + (int) (i * groupW + groupW * 0.1 + s * barW);
                }
                int y0 = mapY(base, yr[0], yr[1], top, plotH);
                int y1 = mapY(topVal, yr[0], yr[1], top, plotH);
                int topY = Math.min(y0, y1);
                int h = Math.max(1, Math.abs(y0 - y1));
                int bw = Math.max(1, (int) barW - 1);
                g.fillRect(x, topY, bw, h);

                if (s == 0 && !stacked && errorLow != null && errorHigh != null && i < errorLow.length) {
                    int cx = x + bw / 2;
                    int ey0 = mapY(errorLow[i], yr[0], yr[1], top, plotH);
                    int ey1 = mapY(errorHigh[i], yr[0], yr[1], top, plotH);
                    g.setColor(Color.DARK_GRAY);
                    g.setStroke(new BasicStroke(1.5f));
                    g.drawLine(cx, ey0, cx, ey1);
                    int cap = Math.max(3, bw / 4);
                    g.drawLine(cx - cap, ey0, cx + cap, ey0);
                    g.drawLine(cx - cap, ey1, cx + cap, ey1);
                    g.setColor(c);
                }
                if (stacked && bottoms != null) bottoms[i] = topVal;
            }
        }
    }

    private void renderHorizontal(Graphics2D g, int left, int top, int plotW, int plotH, int n, int nSeries) {
        double xMin = 0, xMax = Double.NEGATIVE_INFINITY;
        if (stacked) {
            for (int i = 0; i < n; i++) {
                double sum = 0;
                for (int s = 0; s < nSeries; s++) {
                    double[] vals = series.get(s);
                    if (i < vals.length) sum += vals[i];
                }
                xMax = Math.max(xMax, sum);
            }
        } else {
            for (double[] s : series) xMax = Math.max(xMax, max(s));
        }
        if (xMax <= xMin) xMax = xMin + 1;
        double[] xr = {xMin, xMax}; padRange(xr); xr[0] = Math.min(0, xr[0]);

        drawAxesFrame(g, left, top, plotW, plotH, xr[0], xr[1], 0, Math.max(1, n));

        double groupH = plotH / (double) Math.max(1, n);
        double barH = stacked ? groupH * 0.7 : groupH * 0.8 / Math.max(1, nSeries);
        double[] lefts = stacked ? new double[n] : null;

        for (int s = 0; s < nSeries; s++) {
            Color c = (barColor != null && nSeries == 1) ? barColor : PALETTE[s % PALETTE.length];
            g.setColor(c);
            double[] vals = series.get(s);
            for (int i = 0; i < n && i < vals.length; i++) {
                double base = stacked && lefts != null ? lefts[i] : 0;
                double rightVal = base + vals[i];
                int y;
                if (stacked) {
                    y = top + (int) (i * groupH + groupH * 0.15);
                } else {
                    y = top + (int) (i * groupH + groupH * 0.1 + s * barH);
                }
                int x0 = mapX(base, xr[0], xr[1], left, plotW);
                int x1 = mapX(rightVal, xr[0], xr[1], left, plotW);
                int leftX = Math.min(x0, x1);
                int w = Math.max(1, Math.abs(x1 - x0));
                int bh = Math.max(1, (int) barH - 1);
                g.fillRect(leftX, y, w, bh);
                if (stacked && lefts != null) lefts[i] = rightVal;
            }
        }
    }
}
