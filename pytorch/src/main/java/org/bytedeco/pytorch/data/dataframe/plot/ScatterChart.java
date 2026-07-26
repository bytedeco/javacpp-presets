package org.bytedeco.pytorch.data.dataframe.plot;

import org.bytedeco.pytorch.data.dataframe.DataFrame;

import java.awt.*;
import java.awt.image.BufferedImage;

/** Scatter chart. */
public final class ScatterChart extends BaseChart {
    private final double[] x, y;
    private double[] colors; // optional third dim mapped to palette index
    private int pointSize = 6;

    public ScatterChart(String title, double[] x, double[] y) {
        super(title);
        this.x = x; this.y = y;
    }

    public ScatterChart(String title, DataFrame df, String xCol, String yCol) {
        super(title);
        this.x = toDoubles(df, xCol);
        this.y = toDoubles(df, yCol);
        this.xAxisLabel = xCol;
        this.yAxisLabel = yCol;
    }

    /** Scatter with categorical hue column (mapped to palette indices). */
    public ScatterChart(String title, DataFrame df, String xCol, String yCol, String hueCol) {
        this(title, df, xCol, yCol);
        if (hueCol != null) {
            java.util.LinkedHashMap<String, Integer> map = new java.util.LinkedHashMap<>();
            int n = df.rowCount();
            this.colors = new double[n];
            for (int i = 0; i < n; i++) {
                Object v = df.get(i, hueCol);
                String key = v == null ? "null" : v.toString();
                int idx = map.computeIfAbsent(key, k -> map.size());
                this.colors[i] = idx;
            }
        }
    }

    public ScatterChart setColorColumn(DataFrame df, String col) {
        this.colors = toDoubles(df, col);
        return this;
    }

    public ScatterChart setPointSize(int s) { this.pointSize = s; return this; }

    @Override public ScatterChart setTitle(String t) { super.setTitle(t); return this; }
    @Override public ScatterChart setXAxisLabel(String l) { super.setXAxisLabel(l); return this; }
    @Override public ScatterChart setYAxisLabel(String l) { super.setYAxisLabel(l); return this; }
    @Override public ScatterChart setSize(int w, int h) { super.setSize(w, h); return this; }

    @Override
    public BufferedImage render() {
        BufferedImage img = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = img.createGraphics();
        int left = 60, right = 20, top = 40, bottom = 50;
        int plotW = width - left - right, plotH = height - top - bottom;
        double[] xr = {min(x), max(x)}, yr = {min(y), max(y)};
        padRange(xr); padRange(yr);
        drawAxesFrame(g, left, top, plotW, plotH, xr[0], xr[1], yr[0], yr[1]);

        int n = Math.min(x.length, y.length);
        for (int i = 0; i < n; i++) {
            if (Double.isNaN(x[i]) || Double.isNaN(y[i])) continue;
            int px = mapX(x[i], xr[0], xr[1], left, plotW);
            int py = mapY(y[i], yr[0], yr[1], top, plotH);
            if (colors != null && i < colors.length && !Double.isNaN(colors[i])) {
                int idx = Math.floorMod((int) Math.round(colors[i]), PALETTE.length);
                g.setColor(PALETTE[idx]);
            } else {
                g.setColor(PALETTE[0]);
            }
            g.fillOval(px - pointSize / 2, py - pointSize / 2, pointSize, pointSize);
        }
        g.dispose();
        return img;
    }
}
