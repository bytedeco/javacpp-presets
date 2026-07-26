package org.bytedeco.pytorch.data.dataframe.plot;

import org.bytedeco.pytorch.data.dataframe.DataFrame;

import java.awt.*;
import java.awt.image.BufferedImage;
import java.util.*;
import java.util.List;

/** Line chart (matplotlib.pyplot.plot). */
public final class LineChart extends BaseChart {
    private double[] x;
    private final List<double[]> ys = new ArrayList<>();
    private final List<String> labels = new ArrayList<>();

    public LineChart(String title, double[] x, double[] y, String label) {
        super(title);
        this.x = x;
        ys.add(y);
        labels.add(label == null ? "y" : label);
    }

    public LineChart(String title, DataFrame df, String xCol, String... yCols) {
        super(title);
        this.x = toDoubles(df, xCol);
        this.xAxisLabel = xCol;
        for (String yc : yCols) {
            ys.add(toDoubles(df, yc));
            labels.add(yc);
        }
    }

    public LineChart addSeries(double[] y, String label) {
        ys.add(y);
        labels.add(label == null ? "y" + ys.size() : label);
        return this;
    }

    @Override public LineChart setTitle(String t) { super.setTitle(t); return this; }
    @Override public LineChart setXAxisLabel(String l) { super.setXAxisLabel(l); return this; }
    @Override public LineChart setYAxisLabel(String l) { super.setYAxisLabel(l); return this; }
    @Override public LineChart setSize(int w, int h) { super.setSize(w, h); return this; }

    @Override
    public BufferedImage render() {
        BufferedImage img = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = img.createGraphics();
        int left = 60, right = 20, top = 40, bottom = 50;
        int plotW = width - left - right, plotH = height - top - bottom;

        double xMin = min(x), xMax = max(x);
        double yMin = Double.POSITIVE_INFINITY, yMax = Double.NEGATIVE_INFINITY;
        for (double[] y : ys) {
            yMin = Math.min(yMin, min(y));
            yMax = Math.max(yMax, max(y));
        }
        double[] xr = {xMin, xMax}, yr = {yMin, yMax};
        padRange(xr); padRange(yr);

        drawAxesFrame(g, left, top, plotW, plotH, xr[0], xr[1], yr[0], yr[1]);

        for (int s = 0; s < ys.size(); s++) {
            g.setColor(PALETTE[s % PALETTE.length]);
            g.setStroke(new BasicStroke(2f));
            double[] y = ys.get(s);
            int n = Math.min(x.length, y.length);
            for (int i = 1; i < n; i++) {
                if (Double.isNaN(x[i - 1]) || Double.isNaN(y[i - 1]) || Double.isNaN(x[i]) || Double.isNaN(y[i]))
                    continue;
                int x0 = mapX(x[i - 1], xr[0], xr[1], left, plotW);
                int y0 = mapY(y[i - 1], yr[0], yr[1], top, plotH);
                int x1 = mapX(x[i], xr[0], xr[1], left, plotW);
                int y1 = mapY(y[i], yr[0], yr[1], top, plotH);
                g.drawLine(x0, y0, x1, y1);
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
}
