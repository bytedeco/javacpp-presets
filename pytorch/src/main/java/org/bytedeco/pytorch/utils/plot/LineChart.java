package org.bytedeco.pytorch.utils.plot;

import org.bytedeco.pytorch.dataframe.DataFrame;

import java.awt.*;
import java.awt.image.BufferedImage;
import java.util.*;
import java.util.List;

/** Line chart (matplotlib.pyplot.plot) with multi-x series, markers, step, errorbar. */
public final class LineChart extends BaseChart {
    public enum DrawStyle { DEFAULT, STEP_PRE, STEP_MID, STEP_POST }

    private double[] x;
    private final List<double[]> xs = new ArrayList<>(); // optional per-series x; null → share this.x
    private final List<double[]> ys = new ArrayList<>();
    private final List<String> labels = new ArrayList<>();
    private boolean showMarkers = false;
    private int markerSize = 6;
    private DrawStyle drawStyle = DrawStyle.DEFAULT;
    private final List<double[]> errorBars = new ArrayList<>(); // optional symmetric err per series

    public LineChart(String title, double[] x, double[] y, String label) {
        super(title);
        this.x = x;
        xs.add(null);
        ys.add(y);
        labels.add(label == null ? "y" : label);
        errorBars.add(null);
    }

    public LineChart(String title, DataFrame df, String xCol, String... yCols) {
        super(title);
        this.x = toDoubles(df, xCol);
        this.xAxisLabel = xCol;
        for (String yc : yCols) {
            xs.add(null);
            ys.add(toDoubles(df, yc));
            labels.add(yc);
            errorBars.add(null);
        }
    }

    public LineChart addSeries(double[] y, String label) {
        xs.add(null);
        ys.add(y);
        labels.add(label == null ? "y" + ys.size() : label);
        errorBars.add(null);
        return this;
    }

    /** Add a series with its own x coordinates (needed for multi-group kdeplot / ecdf). */
    public LineChart addSeries(double[] xCoords, double[] y, String label) {
        xs.add(xCoords);
        ys.add(y);
        labels.add(label == null ? "y" + ys.size() : label);
        errorBars.add(null);
        return this;
    }

    public LineChart setShowMarkers(boolean v) { this.showMarkers = v; return this; }
    public LineChart setMarkerSize(int s) { this.markerSize = s; return this; }

    /** matplotlib {@code plt.step(..., where="pre"|"mid"|"post")}. */
    public LineChart setDrawStyle(DrawStyle style) {
        this.drawStyle = style == null ? DrawStyle.DEFAULT : style;
        return this;
    }

    public LineChart setStep(String where) {
        if (where == null) return setDrawStyle(DrawStyle.STEP_PRE);
        switch (where.toLowerCase()) {
            case "mid": return setDrawStyle(DrawStyle.STEP_MID);
            case "post": return setDrawStyle(DrawStyle.STEP_POST);
            default: return setDrawStyle(DrawStyle.STEP_PRE);
        }
    }

    /** Symmetric error bars for series index (pointplot / lineplot ci / errorbar). */
    public LineChart setError(int seriesIdx, double[] err) {
        while (errorBars.size() <= seriesIdx) errorBars.add(null);
        errorBars.set(seriesIdx, err);
        return this;
    }

    @Override public LineChart setTitle(String t) { super.setTitle(t); return this; }
    @Override public LineChart setXAxisLabel(String l) { super.setXAxisLabel(l); return this; }
    @Override public LineChart setYAxisLabel(String l) { super.setYAxisLabel(l); return this; }
    @Override public LineChart setSize(int w, int h) { super.setSize(w, h); return this; }
    @Override public LineChart setXScale(String s) { super.setXScale(s); return this; }
    @Override public LineChart setYScale(String s) { super.setYScale(s); return this; }
    @Override public LineChart setShowGrid(boolean v) { super.setShowGrid(v); return this; }
    @Override public LineChart setShowLegend(boolean v) { super.setShowLegend(v); return this; }

    @Override
    public BufferedImage render() {
        BufferedImage img = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = img.createGraphics();
        int left = 60, right = 20, top = 40, bottom = 50;
        int plotW = width - left - right, plotH = height - top - bottom;

        double xMin = Double.POSITIVE_INFINITY, xMax = Double.NEGATIVE_INFINITY;
        double yMin = Double.POSITIVE_INFINITY, yMax = Double.NEGATIVE_INFINITY;
        for (int s = 0; s < ys.size(); s++) {
            double[] xx = seriesX(s);
            double[] y = ys.get(s);
            xMin = Math.min(xMin, min(xx));
            xMax = Math.max(xMax, max(xx));
            yMin = Math.min(yMin, min(y));
            yMax = Math.max(yMax, max(y));
            double[] err = s < errorBars.size() ? errorBars.get(s) : null;
            if (err != null) {
                for (int i = 0; i < y.length && i < err.length; i++) {
                    if (Double.isNaN(y[i]) || Double.isNaN(err[i])) continue;
                    yMin = Math.min(yMin, y[i] - Math.abs(err[i]));
                    yMax = Math.max(yMax, y[i] + Math.abs(err[i]));
                }
            }
        }
        if (xMax <= xMin) { xMin = 0; xMax = 1; }
        if (yMax <= yMin) { yMin = 0; yMax = 1; }
        double[] xr = {xMin, xMax}, yr = {yMin, yMax};
        padRange(xr); padRange(yr);
        if (isLog(xScale)) ensurePositiveRange(xr);
        if (isLog(yScale)) ensurePositiveRange(yr);

        drawAxesFrame(g, left, top, plotW, plotH, xr[0], xr[1], yr[0], yr[1]);

        for (int s = 0; s < ys.size(); s++) {
            g.setColor(PALETTE[s % PALETTE.length]);
            g.setStroke(new BasicStroke(2f));
            double[] xx = seriesX(s);
            double[] y = ys.get(s);
            int n = Math.min(xx.length, y.length);
            drawSeriesLine(g, xx, y, n, xr, yr, left, top, plotW, plotH);
            double[] err = s < errorBars.size() ? errorBars.get(s) : null;
            if (showMarkers || err != null) {
                for (int i = 0; i < n; i++) {
                    if (Double.isNaN(xx[i]) || Double.isNaN(y[i])) continue;
                    int px = mapXScaled(xx[i], xr[0], xr[1], left, plotW);
                    int py = mapYScaled(y[i], yr[0], yr[1], top, plotH);
                    if (err != null && i < err.length && !Double.isNaN(err[i])) {
                        int ey0 = mapYScaled(y[i] - Math.abs(err[i]), yr[0], yr[1], top, plotH);
                        int ey1 = mapYScaled(y[i] + Math.abs(err[i]), yr[0], yr[1], top, plotH);
                        g.drawLine(px, ey0, px, ey1);
                        g.drawLine(px - 3, ey0, px + 3, ey0);
                        g.drawLine(px - 3, ey1, px + 3, ey1);
                    }
                    if (showMarkers) {
                        g.fillOval(px - markerSize / 2, py - markerSize / 2, markerSize, markerSize);
                    }
                }
            }
        }

        if (showLegend && !labels.isEmpty()) {
            int lx = left + plotW - 120, ly = top + 10;
            g.setFont(new Font("SansSerif", Font.PLAIN, 11));
            for (int s = 0; s < labels.size(); s++) {
                g.setColor(PALETTE[s % PALETTE.length]);
                g.fillRect(lx, ly + s * 16, 12, 12);
                g.setColor(Color.BLACK);
                g.drawString(labels.get(s), lx + 16, ly + s * 16 + 11);
            }
        }
        g.dispose();
        return img;
    }

    private void drawSeriesLine(Graphics2D g, double[] xx, double[] y, int n,
                                 double[] xr, double[] yr, int left, int top, int plotW, int plotH) {
        if (drawStyle == DrawStyle.DEFAULT) {
            for (int i = 1; i < n; i++) {
                if (Double.isNaN(xx[i - 1]) || Double.isNaN(y[i - 1]) || Double.isNaN(xx[i]) || Double.isNaN(y[i]))
                    continue;
                int x0 = mapXScaled(xx[i - 1], xr[0], xr[1], left, plotW);
                int y0 = mapYScaled(y[i - 1], yr[0], yr[1], top, plotH);
                int x1 = mapXScaled(xx[i], xr[0], xr[1], left, plotW);
                int y1 = mapYScaled(y[i], yr[0], yr[1], top, plotH);
                g.drawLine(x0, y0, x1, y1);
            }
            return;
        }
        // step styles
        for (int i = 1; i < n; i++) {
            if (Double.isNaN(xx[i - 1]) || Double.isNaN(y[i - 1]) || Double.isNaN(xx[i]) || Double.isNaN(y[i]))
                continue;
            int x0 = mapXScaled(xx[i - 1], xr[0], xr[1], left, plotW);
            int y0 = mapYScaled(y[i - 1], yr[0], yr[1], top, plotH);
            int x1 = mapXScaled(xx[i], xr[0], xr[1], left, plotW);
            int y1 = mapYScaled(y[i], yr[0], yr[1], top, plotH);
            switch (drawStyle) {
                case STEP_PRE:
                    g.drawLine(x0, y0, x0, y1);
                    g.drawLine(x0, y1, x1, y1);
                    break;
                case STEP_POST:
                    g.drawLine(x0, y0, x1, y0);
                    g.drawLine(x1, y0, x1, y1);
                    break;
                case STEP_MID:
                default: {
                    int xm = (x0 + x1) / 2;
                    g.drawLine(x0, y0, xm, y0);
                    g.drawLine(xm, y0, xm, y1);
                    g.drawLine(xm, y1, x1, y1);
                    break;
                }
            }
        }
    }

    private double[] seriesX(int s) {
        if (s < xs.size() && xs.get(s) != null) return xs.get(s);
        return x;
    }
}
