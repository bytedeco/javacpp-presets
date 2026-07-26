package org.bytedeco.pytorch.data.dataframe.plot;

import org.bytedeco.pytorch.data.dataframe.DataFrame;

import java.awt.*;
import java.awt.image.BufferedImage;
import java.util.*;
import java.util.List;

/** Bar chart. */
public final class BarChart extends BaseChart {
    private final List<String> categories = new ArrayList<>();
    private final List<double[]> series = new ArrayList<>();
    private final List<String> seriesNames = new ArrayList<>();

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

    @Override public BarChart setTitle(String t) { super.setTitle(t); return this; }
    @Override public BarChart setXAxisLabel(String l) { super.setXAxisLabel(l); return this; }
    @Override public BarChart setYAxisLabel(String l) { super.setYAxisLabel(l); return this; }
    @Override public BarChart setSize(int w, int h) { super.setSize(w, h); return this; }

    @Override
    public BufferedImage render() {
        BufferedImage img = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = img.createGraphics();
        int left = 60, right = 20, top = 40, bottom = 70;
        int plotW = width - left - right, plotH = height - top - bottom;

        double yMin = 0, yMax = Double.NEGATIVE_INFINITY;
        for (double[] s : series) yMax = Math.max(yMax, max(s));
        if (yMax <= yMin) yMax = yMin + 1;
        double[] yr = {yMin, yMax}; padRange(yr); yr[0] = Math.min(0, yr[0]);

        drawAxesFrame(g, left, top, plotW, plotH, 0, Math.max(1, categories.size()), yr[0], yr[1]);

        int n = categories.size();
        int nSeries = series.size();
        double groupW = plotW / (double) Math.max(1, n);
        double barW = groupW * 0.8 / nSeries;

        for (int s = 0; s < nSeries; s++) {
            g.setColor(PALETTE[s % PALETTE.length]);
            double[] vals = series.get(s);
            for (int i = 0; i < n && i < vals.length; i++) {
                int x = left + (int) (i * groupW + groupW * 0.1 + s * barW);
                int y0 = mapY(0, yr[0], yr[1], top, plotH);
                int y1 = mapY(vals[i], yr[0], yr[1], top, plotH);
                int topY = Math.min(y0, y1);
                int h = Math.abs(y0 - y1);
                g.fillRect(x, topY, Math.max(1, (int) barW - 1), Math.max(1, h));
            }
        }

        // category labels
        g.setColor(Color.DARK_GRAY);
        g.setFont(new Font("SansSerif", Font.PLAIN, 10));
        for (int i = 0; i < n; i++) {
            String lab = categories.get(i);
            if (lab.length() > 8) lab = lab.substring(0, 8);
            FontMetrics fm = g.getFontMetrics();
            int x = left + (int) (i * groupW + groupW / 2 - fm.stringWidth(lab) / 2.0);
            g.drawString(lab, x, top + plotH + 28);
        }
        g.dispose();
        return img;
    }
}
