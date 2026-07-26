package org.bytedeco.pytorch.data.dataframe.plot;

import org.bytedeco.pytorch.data.dataframe.DataFrame;

import java.awt.*;
import java.awt.geom.Path2D;
import java.awt.image.BufferedImage;
import java.util.ArrayList;
import java.util.List;

/** Area chart (filled line). */
public final class AreaChart extends BaseChart {
    private final List<double[]> xs = new ArrayList<>();
    private final List<double[]> ys = new ArrayList<>();
    private final List<String> names = new ArrayList<>();

    public AreaChart(String title, double[] x, double[] y, String name) {
        super(title);
        xs.add(x);
        ys.add(y);
        names.add(name == null ? "y" : name);
    }

    public AreaChart(String title, DataFrame df, String xCol, String... yCols) {
        super(title);
        double[] x = toDoubles(df, xCol);
        this.xAxisLabel = xCol;
        for (String yc : yCols) {
            xs.add(x);
            ys.add(toDoubles(df, yc));
            names.add(yc);
        }
        if (yCols.length == 1) this.yAxisLabel = yCols[0];
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
        double yMin = 0, yMax = Double.NEGATIVE_INFINITY;
        for (int s = 0; s < ys.size(); s++) {
            xMin = Math.min(xMin, min(xs.get(s)));
            xMax = Math.max(xMax, max(xs.get(s)));
            yMax = Math.max(yMax, max(ys.get(s)));
            yMin = Math.min(yMin, min(ys.get(s)));
        }
        if (yMax <= yMin) yMax = yMin + 1;
        double[] xr = {xMin, xMax}; padRange(xr);
        double[] yr = {Math.min(0, yMin), yMax}; padRange(yr); yr[0] = Math.min(0, yr[0]);

        drawAxesFrame(g, left, top, plotW, plotH, xr[0], xr[1], yr[0], yr[1]);

        for (int s = 0; s < ys.size(); s++) {
            double[] x = xs.get(s);
            double[] y = ys.get(s);
            Path2D path = new Path2D.Double();
            boolean started = false;
            int firstX = 0, lastX = 0;
            for (int i = 0; i < x.length && i < y.length; i++) {
                if (Double.isNaN(x[i]) || Double.isNaN(y[i])) continue;
                int px = mapX(x[i], xr[0], xr[1], left, plotW);
                int py = mapY(y[i], yr[0], yr[1], top, plotH);
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
                int y0 = mapY(0, yr[0], yr[1], top, plotH);
                path.lineTo(lastX, y0);
                path.lineTo(firstX, y0);
                path.closePath();
                Color c = PALETTE[s % PALETTE.length];
                g.setColor(new Color(c.getRed(), c.getGreen(), c.getBlue(), 90));
                g.fill(path);
                g.setColor(c);
                g.setStroke(new BasicStroke(1.8f));
                // outline top
                Path2D line = new Path2D.Double();
                boolean st = false;
                for (int i = 0; i < x.length && i < y.length; i++) {
                    if (Double.isNaN(x[i]) || Double.isNaN(y[i])) continue;
                    int px = mapX(x[i], xr[0], xr[1], left, plotW);
                    int py = mapY(y[i], yr[0], yr[1], top, plotH);
                    if (!st) { line.moveTo(px, py); st = true; }
                    else line.lineTo(px, py);
                }
                g.draw(line);
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
