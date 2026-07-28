package org.bytedeco.pytorch.utils.plot;

import org.bytedeco.pytorch.dataframe.DataFrame;

import java.awt.*;
import java.awt.geom.Path2D;
import java.awt.image.BufferedImage;
import java.util.ArrayList;
import java.util.List;

/**
 * Area chart (filled line) and matplotlib {@code fill_between(x, y1, y2)}.
 */
public final class AreaChart extends BaseChart {
    private final List<double[]> xs = new ArrayList<>();
    private final List<double[]> ys = new ArrayList<>();
    private final List<double[]> yLows = new ArrayList<>(); // null → fill to 0
    private final List<String> names = new ArrayList<>();
    private float fillAlpha = 90 / 255f;
    private Color fillColor = null;
    private boolean drawMeanLine = false;
    private double[] meanLineY;

    public AreaChart(String title, double[] x, double[] y, String name) {
        super(title);
        xs.add(x);
        ys.add(y);
        yLows.add(null);
        names.add(name == null ? "y" : name);
    }

    /** fill_between(x, yLower, yUpper). */
    public AreaChart(String title, double[] x, double[] yLower, double[] yUpper, String name) {
        super(title);
        xs.add(x);
        ys.add(yUpper);
        yLows.add(yLower);
        names.add(name == null ? "band" : name);
    }

    public AreaChart(String title, DataFrame df, String xCol, String... yCols) {
        super(title);
        double[] x = toDoubles(df, xCol);
        this.xAxisLabel = xCol;
        for (String yc : yCols) {
            xs.add(x);
            ys.add(toDoubles(df, yc));
            yLows.add(null);
            names.add(yc);
        }
        if (yCols.length == 1) this.yAxisLabel = yCols[0];
    }

    public AreaChart addBand(double[] x, double[] yLower, double[] yUpper, String name) {
        xs.add(x);
        ys.add(yUpper);
        yLows.add(yLower);
        names.add(name == null ? "band" : name);
        return this;
    }

    public AreaChart setFillAlpha(double a) {
        this.fillAlpha = (float) Math.max(0, Math.min(1, a));
        return this;
    }

    public AreaChart setFillColor(Color c) { this.fillColor = c; return this; }

    /** Optional center line drawn on top of the band (CI mean). */
    public AreaChart setMeanLine(double[] y) {
        this.meanLineY = y;
        this.drawMeanLine = y != null;
        return this;
    }

    @Override public AreaChart setTitle(String t) { super.setTitle(t); return this; }
    @Override public AreaChart setXAxisLabel(String l) { super.setXAxisLabel(l); return this; }
    @Override public AreaChart setYAxisLabel(String l) { super.setYAxisLabel(l); return this; }
    @Override public AreaChart setSize(int w, int h) { super.setSize(w, h); return this; }

    @Override
    public BufferedImage render() {
        BufferedImage img = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = img.createGraphics();
        int left = 60, right = 20, top = 40, bottom = 50;
        int plotW = width - left - right, plotH = height - top - bottom;

        double xMin = Double.POSITIVE_INFINITY, xMax = Double.NEGATIVE_INFINITY;
        double yMin = Double.POSITIVE_INFINITY, yMax = Double.NEGATIVE_INFINITY;
        for (int s = 0; s < ys.size(); s++) {
            xMin = Math.min(xMin, min(xs.get(s)));
            xMax = Math.max(xMax, max(xs.get(s)));
            yMax = Math.max(yMax, max(ys.get(s)));
            yMin = Math.min(yMin, min(ys.get(s)));
            double[] lo = yLows.get(s);
            if (lo != null) {
                yMin = Math.min(yMin, min(lo));
                yMax = Math.max(yMax, max(lo));
            } else {
                yMin = Math.min(yMin, 0);
            }
        }
        if (meanLineY != null) {
            yMin = Math.min(yMin, min(meanLineY));
            yMax = Math.max(yMax, max(meanLineY));
        }
        if (yMax <= yMin) yMax = yMin + 1;
        if (xMax <= xMin) { xMin = 0; xMax = 1; }
        double[] xr = {xMin, xMax}; padRange(xr);
        double[] yr = {yMin, yMax}; padRange(yr);

        drawAxesFrame(g, left, top, plotW, plotH, xr[0], xr[1], yr[0], yr[1]);

        for (int s = 0; s < ys.size(); s++) {
            double[] x = xs.get(s);
            double[] yHi = ys.get(s);
            double[] yLo = yLows.get(s);
            Path2D path = new Path2D.Double();
            boolean started = false;
            int firstX = 0, lastX = 0;
            // upper edge left→right
            for (int i = 0; i < x.length && i < yHi.length; i++) {
                if (Double.isNaN(x[i]) || Double.isNaN(yHi[i])) continue;
                int px = mapX(x[i], xr[0], xr[1], left, plotW);
                int py = mapY(yHi[i], yr[0], yr[1], top, plotH);
                if (!started) {
                    path.moveTo(px, py);
                    firstX = px;
                    started = true;
                } else {
                    path.lineTo(px, py);
                }
                lastX = px;
            }
            if (started) {
                // lower edge right→left
                if (yLo != null) {
                    for (int i = Math.min(x.length, yLo.length) - 1; i >= 0; i--) {
                        if (Double.isNaN(x[i]) || Double.isNaN(yLo[i])) continue;
                        int px = mapX(x[i], xr[0], xr[1], left, plotW);
                        int py = mapY(yLo[i], yr[0], yr[1], top, plotH);
                        path.lineTo(px, py);
                    }
                } else {
                    int y0 = mapY(0, yr[0], yr[1], top, plotH);
                    path.lineTo(lastX, y0);
                    path.lineTo(firstX, y0);
                }
                path.closePath();
                Color c = fillColor != null ? fillColor : PALETTE[s % PALETTE.length];
                int alpha = Math.round(fillAlpha * 255);
                g.setColor(new Color(c.getRed(), c.getGreen(), c.getBlue(), alpha));
                g.fill(path);
                g.setColor(c);
                g.setStroke(new BasicStroke(1.8f));
                // outline upper
                Path2D line = new Path2D.Double();
                boolean st = false;
                for (int i = 0; i < x.length && i < yHi.length; i++) {
                    if (Double.isNaN(x[i]) || Double.isNaN(yHi[i])) continue;
                    int px = mapX(x[i], xr[0], xr[1], left, plotW);
                    int py = mapY(yHi[i], yr[0], yr[1], top, plotH);
                    if (!st) { line.moveTo(px, py); st = true; }
                    else line.lineTo(px, py);
                }
                g.draw(line);
            }
        }

        if (drawMeanLine && meanLineY != null && !xs.isEmpty()) {
            double[] x = xs.get(0);
            g.setColor(fillColor != null ? fillColor : PALETTE[0]);
            g.setStroke(new BasicStroke(2f));
            for (int i = 1; i < Math.min(x.length, meanLineY.length); i++) {
                if (Double.isNaN(x[i - 1]) || Double.isNaN(meanLineY[i - 1])
                    || Double.isNaN(x[i]) || Double.isNaN(meanLineY[i])) continue;
                int x0 = mapX(x[i - 1], xr[0], xr[1], left, plotW);
                int y0 = mapY(meanLineY[i - 1], yr[0], yr[1], top, plotH);
                int x1 = mapX(x[i], xr[0], xr[1], left, plotW);
                int y1 = mapY(meanLineY[i], yr[0], yr[1], top, plotH);
                g.drawLine(x0, y0, x1, y1);
            }
        }

        if (showLegend && names.size() > 1) {
            g.setFont(new Font("SansSerif", Font.PLAIN, 11));
            for (int s = 0; s < names.size(); s++) {
                g.setColor(PALETTE[s % PALETTE.length]);
                g.fillRect(left + 8, top + 8 + s * 16, 12, 10);
                g.setColor(Color.DARK_GRAY);
                g.drawString(names.get(s), left + 24, top + 17 + s * 16);
            }
        }
        g.dispose();
        return img;
    }
}
