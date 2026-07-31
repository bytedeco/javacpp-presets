package org.bytedeco.pytorch.plot;

import org.bytedeco.pytorch.dataframe.DataFrame;
import org.bytedeco.pytorch.dataframe.DataValues;

import java.awt.*;
import java.awt.image.BufferedImage;

/** Bubble chart: x, y, size (and optional category color). */
public final class BubbleChart extends BaseChart {
    private final double[] x;
    private final double[] y;
    private final double[] size;
    private final String[] categories;

    public BubbleChart(String title, double[] x, double[] y, double[] size) {
        this(title, x, y, size, null);
    }

    public BubbleChart(String title, double[] x, double[] y, double[] size, String[] categories) {
        super(title);
        this.x = x;
        this.y = y;
        this.size = size;
        this.categories = categories;
    }

    public BubbleChart(String title, DataFrame df, String xCol, String yCol, String sizeCol) {
        this(title, df, xCol, yCol, sizeCol, null);
    }

    public BubbleChart(String title, DataFrame df, String xCol, String yCol, String sizeCol, String catCol) {
        super(title);
        this.x = toDoubles(df, xCol);
        this.y = toDoubles(df, yCol);
        this.size = toDoubles(df, sizeCol);
        this.xAxisLabel = xCol;
        this.yAxisLabel = yCol;
        if (catCol != null) {
            this.categories = new String[df.rowCount()];
            for (int i = 0; i < df.rowCount(); i++) {
                Object v = DataValues.unwrap(df.get(i, catCol));
                this.categories[i] = v == null ? "" : v.toString();
            }
        } else {
            this.categories = null;
        }
    }

    @Override public BubbleChart setTitle(String t) { super.setTitle(t); return this; }
    @Override public BubbleChart setXAxisLabel(String l) { super.setXAxisLabel(l); return this; }
    @Override public BubbleChart setYAxisLabel(String l) { super.setYAxisLabel(l); return this; }
    @Override public BubbleChart setSize(int w, int h) { super.setSize(w, h); return this; }

    @Override
    public BufferedImage render() {
        BufferedImage img = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = img.createGraphics();
        int left = 60, right = 20, top = 40, bottom = 50;
        int plotW = width - left - right, plotH = height - top - bottom;

        double[] xr = {min(x), max(x)}; padRange(xr);
        double[] yr = {min(y), max(y)}; padRange(yr);
        double sMin = min(size), sMax = max(size);
        if (sMax <= sMin) sMax = sMin + 1;

        drawAxesFrame(g, left, top, plotW, plotH, xr[0], xr[1], yr[0], yr[1]);

        java.util.LinkedHashMap<String, Integer> catIndex = new java.util.LinkedHashMap<>();
        if (categories != null) {
            for (String c : categories) catIndex.putIfAbsent(c == null ? "" : c, catIndex.size());
        }

        for (int i = 0; i < x.length && i < y.length; i++) {
            if (Double.isNaN(x[i]) || Double.isNaN(y[i])) continue;
            double sv = i < size.length ? size[i] : sMin;
            if (Double.isNaN(sv)) sv = sMin;
            int r = 4 + (int) Math.round(20.0 * (sv - sMin) / (sMax - sMin));
            int px = mapX(x[i], xr[0], xr[1], left, plotW);
            int py = mapY(y[i], yr[0], yr[1], top, plotH);
            int ci = 0;
            if (categories != null && i < categories.length) {
                ci = catIndex.getOrDefault(categories[i] == null ? "" : categories[i], 0);
            }
            Color c = PALETTE[ci % PALETTE.length];
            g.setColor(new Color(c.getRed(), c.getGreen(), c.getBlue(), 140));
            g.fillOval(px - r, py - r, 2 * r, 2 * r);
            g.setColor(c);
            g.drawOval(px - r, py - r, 2 * r, 2 * r);
        }
        g.dispose();
        return img;
    }
}
