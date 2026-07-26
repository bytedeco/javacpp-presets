package org.bytedeco.pytorch.data.dataframe.plot;

import org.bytedeco.pytorch.data.dataframe.DataFrame;
import org.bytedeco.pytorch.data.dataframe.DataValues;

import java.awt.*;
import java.awt.geom.Path2D;
import java.awt.image.BufferedImage;
import java.util.*;
import java.util.List;

/** Violin plot (KDE mirrored density + box markers). */
public final class ViolinChart extends BaseChart {
    private final LinkedHashMap<String, List<Double>> groups = new LinkedHashMap<>();

    public ViolinChart(String title, DataFrame df, String categoryCol, String valueCol) {
        super(title);
        this.xAxisLabel = categoryCol;
        this.yAxisLabel = valueCol;
        int n = df.rowCount();
        for (int i = 0; i < n; i++) {
            Object cat = DataValues.unwrap(df.get(i, categoryCol));
            String key = cat == null ? "null" : cat.toString();
            double v = DataValues.asDouble(df.get(i, valueCol));
            if (Double.isNaN(v)) continue;
            groups.computeIfAbsent(key, k -> new ArrayList<>()).add(v);
        }
    }

    public ViolinChart(String title, double[] values) {
        super(title);
        List<Double> list = new ArrayList<>();
        for (double v : values) if (!Double.isNaN(v)) list.add(v);
        groups.put("all", list);
    }

    @Override public ViolinChart setTitle(String t) { super.setTitle(t); return this; }
    @Override public ViolinChart setXAxisLabel(String l) { super.setXAxisLabel(l); return this; }
    @Override public ViolinChart setYAxisLabel(String l) { super.setYAxisLabel(l); return this; }
    @Override public ViolinChart setSize(int w, int h) { super.setSize(w, h); return this; }

    @Override
    public BufferedImage render() {
        BufferedImage img = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = img.createGraphics();
        int left = 60, right = 20, top = 40, bottom = 60;
        int plotW = width - left - right, plotH = height - top - bottom;

        double yMin = Double.POSITIVE_INFINITY, yMax = Double.NEGATIVE_INFINITY;
        for (List<Double> vals : groups.values()) {
            for (double v : vals) { yMin = Math.min(yMin, v); yMax = Math.max(yMax, v); }
        }
        if (yMax <= yMin) { yMin = 0; yMax = 1; }
        double[] yr = {yMin, yMax}; padRange(yr);

        List<String> cats = new ArrayList<>(groups.keySet());
        drawAxesFrame(g, left, top, plotW, plotH, 0, Math.max(1, cats.size()), yr[0], yr[1]);

        double slot = plotW / (double) Math.max(1, cats.size());
        int grid = 40;

        for (int ci = 0; ci < cats.size(); ci++) {
            List<Double> vals = groups.get(cats.get(ci));
            if (vals == null || vals.isEmpty()) continue;
            double[] dens = kde(vals, yr[0], yr[1], grid);
            double dMax = 0;
            for (double d : dens) dMax = Math.max(dMax, d);
            if (dMax <= 0) dMax = 1;

            double cx = left + (ci + 0.5) * slot;
            double halfMax = slot * 0.35;
            Path2D path = new Path2D.Double();
            for (int i = 0; i < grid; i++) {
                double yv = yr[0] + (yr[1] - yr[0]) * i / (grid - 1.0);
                int py = mapY(yv, yr[0], yr[1], top, plotH);
                double half = halfMax * dens[i] / dMax;
                if (i == 0) path.moveTo(cx - half, py);
                else path.lineTo(cx - half, py);
            }
            for (int i = grid - 1; i >= 0; i--) {
                double yv = yr[0] + (yr[1] - yr[0]) * i / (grid - 1.0);
                int py = mapY(yv, yr[0], yr[1], top, plotH);
                double half = halfMax * dens[i] / dMax;
                path.lineTo(cx + half, py);
            }
            path.closePath();
            Color c = PALETTE[ci % PALETTE.length];
            g.setColor(new Color(c.getRed(), c.getGreen(), c.getBlue(), 120));
            g.fill(path);
            g.setColor(c);
            g.draw(path);

            // median line
            List<Double> sorted = new ArrayList<>(vals);
            Collections.sort(sorted);
            double med = sorted.get(sorted.size() / 2);
            int my = mapY(med, yr[0], yr[1], top, plotH);
            g.setStroke(new BasicStroke(2f));
            g.drawLine((int) (cx - halfMax * 0.4), my, (int) (cx + halfMax * 0.4), my);
        }

        g.setColor(Color.DARK_GRAY);
        g.setFont(new Font("SansSerif", Font.PLAIN, 10));
        for (int i = 0; i < cats.size(); i++) {
            String lab = cats.get(i);
            if (lab.length() > 10) lab = lab.substring(0, 10);
            FontMetrics fm = g.getFontMetrics();
            int x = left + (int) ((i + 0.5) * slot - fm.stringWidth(lab) / 2.0);
            g.drawString(lab, x, top + plotH + 28);
        }
        g.dispose();
        return img;
    }

    private static double[] kde(List<Double> vals, double yMin, double yMax, int grid) {
        double[] dens = new double[grid];
        if (vals.isEmpty()) return dens;
        double mean = 0;
        for (double v : vals) mean += v;
        mean /= vals.size();
        double var = 0;
        for (double v : vals) var += (v - mean) * (v - mean);
        var /= Math.max(1, vals.size());
        double bw = Math.max(1e-6, Math.sqrt(var) * 1.06 * Math.pow(vals.size(), -0.2));
        for (int i = 0; i < grid; i++) {
            double y = yMin + (yMax - yMin) * i / (grid - 1.0);
            double s = 0;
            for (double v : vals) {
                double z = (y - v) / bw;
                s += Math.exp(-0.5 * z * z);
            }
            dens[i] = s / (vals.size() * bw * Math.sqrt(2 * Math.PI));
        }
        return dens;
    }
}
