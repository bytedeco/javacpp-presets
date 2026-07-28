package org.bytedeco.pytorch.utils.plot;

import org.bytedeco.pytorch.dataframe.DataFrame;

import java.awt.*;
import java.awt.image.BufferedImage;
import java.util.Locale;

/**
 * Scatter chart with alpha / regression-line / continuous color mapping
 * (seaborn scatter/regplot + matplotlib scatter c= + cmap).
 */
public final class ScatterChart extends BaseChart {
    private final double[] x, y;
    private double[] colors; // categorical index OR continuous values
    private boolean continuousColor = false;
    private String cmap = "plasma";
    private boolean showColorbar = false;
    private String colorbarLabel = "";
    private int pointSize = 6;
    private float alpha = 1.0f;
    private boolean showRegression = false;
    private double[] regX, regY; // optional explicit regression line
    private Color fixedColor = null;

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
            this.continuousColor = false;
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
        this.continuousColor = true;
        this.showColorbar = true;
        return this;
    }

    /** Continuous color values (matplotlib scatter c=). */
    public ScatterChart setColorValues(double[] c) {
        this.colors = c;
        this.continuousColor = true;
        this.showColorbar = true;
        return this;
    }

    public ScatterChart setCmap(String name) {
        this.cmap = name == null ? "plasma" : name.toLowerCase(Locale.ROOT);
        return this;
    }

    public ScatterChart setShowColorbar(boolean v) { this.showColorbar = v; return this; }
    public ScatterChart setColorbarLabel(String lab) { this.colorbarLabel = lab == null ? "" : lab; return this; }

    public ScatterChart setPointSize(int s) { this.pointSize = s; return this; }
    public ScatterChart setAlpha(double a) {
        this.alpha = (float) Math.max(0, Math.min(1, a));
        return this;
    }
    public ScatterChart setFixedColor(Color c) { this.fixedColor = c; return this; }

    /** Draw OLS fit line through the points (seaborn regplot). */
    public ScatterChart setShowRegression(boolean v) { this.showRegression = v; return this; }
    public ScatterChart setRegressionLine(double[] rx, double[] ry) {
        this.regX = rx; this.regY = ry; this.showRegression = true; return this;
    }

    @Override public ScatterChart setTitle(String t) { super.setTitle(t); return this; }
    @Override public ScatterChart setXAxisLabel(String l) { super.setXAxisLabel(l); return this; }
    @Override public ScatterChart setYAxisLabel(String l) { super.setYAxisLabel(l); return this; }
    @Override public ScatterChart setSize(int w, int h) { super.setSize(w, h); return this; }
    @Override public ScatterChart setXScale(String s) { super.setXScale(s); return this; }
    @Override public ScatterChart setYScale(String s) { super.setYScale(s); return this; }

    @Override
    public BufferedImage render() {
        BufferedImage img = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = img.createGraphics();
        int colorbarW = (continuousColor && showColorbar) ? 50 : 0;
        int left = 60, right = 20 + colorbarW, top = 40, bottom = 50;
        int plotW = width - left - right, plotH = height - top - bottom;
        double[] xr = {min(x), max(x)}, yr = {min(y), max(y)};
        padRange(xr); padRange(yr);
        if (isLog(xScale)) ensurePositiveRange(xr);
        if (isLog(yScale)) ensurePositiveRange(yr);
        drawAxesFrame(g, left, top, plotW, plotH, xr[0], xr[1], yr[0], yr[1]);

        double cMin = 0, cMax = 1;
        if (continuousColor && colors != null) {
            cMin = min(colors);
            cMax = max(colors);
            if (cMax <= cMin) cMax = cMin + 1;
        }

        int n = Math.min(x.length, y.length);
        for (int i = 0; i < n; i++) {
            if (Double.isNaN(x[i]) || Double.isNaN(y[i])) continue;
            int px = mapXScaled(x[i], xr[0], xr[1], left, plotW);
            int py = mapYScaled(y[i], yr[0], yr[1], top, plotH);
            Color base;
            if (fixedColor != null && colors == null) {
                base = fixedColor;
            } else if (colors != null && i < colors.length && !Double.isNaN(colors[i])) {
                if (continuousColor) {
                    float t = (float) ((colors[i] - cMin) / (cMax - cMin));
                    t = Math.max(0f, Math.min(1f, t));
                    base = HeatmapChart.mapColor(t, cmap);
                } else {
                    int idx = Math.floorMod((int) Math.round(colors[i]), PALETTE.length);
                    base = PALETTE[idx];
                }
            } else {
                base = fixedColor != null ? fixedColor : PALETTE[0];
            }
            if (alpha < 1f) {
                g.setColor(new Color(base.getRed(), base.getGreen(), base.getBlue(),
                    Math.round(alpha * 255)));
            } else {
                g.setColor(base);
            }
            g.fillOval(px - pointSize / 2, py - pointSize / 2, pointSize, pointSize);
        }

        if (showRegression) {
            double[] lx = regX, ly = regY;
            if (lx == null || ly == null) {
                int cnt = 0;
                double sx = 0, sy = 0, sxx = 0, sxy = 0;
                for (int i = 0; i < n; i++) {
                    if (Double.isNaN(x[i]) || Double.isNaN(y[i])) continue;
                    cnt++; sx += x[i]; sy += y[i]; sxx += x[i] * x[i]; sxy += x[i] * y[i];
                }
                if (cnt >= 2) {
                    double denom = cnt * sxx - sx * sx;
                    double slope = denom == 0 ? 0 : (cnt * sxy - sx * sy) / denom;
                    double intercept = (sy - slope * sx) / cnt;
                    lx = new double[]{xr[0], xr[1]};
                    ly = new double[]{slope * xr[0] + intercept, slope * xr[1] + intercept};
                }
            }
            if (lx != null && ly != null && lx.length >= 2) {
                g.setColor(new Color(0xC4, 0x4E, 0x52));
                g.setStroke(new BasicStroke(2f));
                for (int i = 1; i < Math.min(lx.length, ly.length); i++) {
                    int x0 = mapXScaled(lx[i - 1], xr[0], xr[1], left, plotW);
                    int y0 = mapYScaled(ly[i - 1], yr[0], yr[1], top, plotH);
                    int x1 = mapXScaled(lx[i], xr[0], xr[1], left, plotW);
                    int y1 = mapYScaled(ly[i], yr[0], yr[1], top, plotH);
                    g.drawLine(x0, y0, x1, y1);
                }
            }
        }

        if (continuousColor && showColorbar && colors != null) {
            int cbX = left + plotW + 12;
            int cbY = top;
            int cbH = plotH;
            int cbW = 14;
            for (int i = 0; i < cbH; i++) {
                float t = 1f - (float) i / Math.max(1, cbH - 1);
                g.setColor(HeatmapChart.mapColor(t, cmap));
                g.fillRect(cbX, cbY + i, cbW, 1);
            }
            g.setColor(Color.DARK_GRAY);
            g.drawRect(cbX, cbY, cbW, cbH);
            g.setFont(new Font("SansSerif", Font.PLAIN, 10));
            g.drawString(formatTick(cMax), cbX + cbW + 3, cbY + 10);
            g.drawString(formatTick(cMin), cbX + cbW + 3, cbY + cbH);
            if (colorbarLabel != null && !colorbarLabel.isEmpty()) {
                g.drawString(colorbarLabel, cbX - 4, cbY - 6);
            }
        }
        g.dispose();
        return img;
    }
}
