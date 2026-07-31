package org.bytedeco.pytorch.plot.chart;

import org.bytedeco.pytorch.dataframe.DataFrame;

import java.awt.*;
import java.awt.image.BufferedImage;
import java.util.*;
import java.util.List;

/** Box plot by category. */
public final class BoxChart extends BaseChart {
    private final Map<String, List<Double>> groups = new LinkedHashMap<>();

    public BoxChart(String title, DataFrame df, String categoryCol, String valueCol) {
        super(title);
        int n = df.rowCount();
        for (int i = 0; i < n; i++) {
            Object cat = df.get(i, categoryCol);
            Object val = df.get(i, valueCol);
            if (!(val instanceof Number)) continue;
            String key = cat == null ? "null" : cat.toString();
            groups.computeIfAbsent(key, k -> new ArrayList<>()).add(((Number) val).doubleValue());
        }
        this.xAxisLabel = categoryCol;
        this.yAxisLabel = valueCol;
    }

    public BoxChart(String title, double[] values) {
        super(title);
        List<Double> list = new ArrayList<>();
        for (double v : values) if (!Double.isNaN(v)) list.add(v);
        groups.put("data", list);
    }

    /** Multi-group box plot from pre-built category → values map (tensor column groups). */
    public static BoxChart fromGroups(Map<String, List<Double>> groupMap) {
        BoxChart chart = new BoxChart("Box Plot");
        if (groupMap != null) {
            for (Map.Entry<String, List<Double>> e : groupMap.entrySet()) {
                List<Double> vals = new ArrayList<>();
                if (e.getValue() != null) {
                    for (Double v : e.getValue()) if (v != null && !Double.isNaN(v)) vals.add(v);
                }
                chart.groups.put(e.getKey() == null ? "null" : e.getKey(), vals);
            }
        }
        return chart;
    }

    private BoxChart(String title) {
        super(title);
    }

    @Override public BoxChart setTitle(String t) { super.setTitle(t); return this; }
    @Override public BoxChart setXAxisLabel(String l) { super.setXAxisLabel(l); return this; }
    @Override public BoxChart setYAxisLabel(String l) { super.setYAxisLabel(l); return this; }
    @Override public BoxChart setSize(int w, int h) { super.setSize(w, h); return this; }

    @Override
    public BufferedImage render() {
        BufferedImage img = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = img.createGraphics();
        int left = 60, right = 20, top = 40, bottom = 60;
        int plotW = width - left - right, plotH = height - top - bottom;

        double yMin = Double.POSITIVE_INFINITY, yMax = Double.NEGATIVE_INFINITY;
        for (List<Double> vals : groups.values()) {
            for (double v : vals) {
                yMin = Math.min(yMin, v);
                yMax = Math.max(yMax, v);
            }
        }
        if (yMax <= yMin) { yMin = 0; yMax = 1; }
        double[] yr = {yMin, yMax}; padRange(yr);

        int n = Math.max(1, groups.size());
        drawAxesFrame(g, left, top, plotW, plotH, 0, n, yr[0], yr[1]);

        int i = 0;
        double groupW = plotW / (double) n;
        g.setStroke(new BasicStroke(1.5f));
        for (Map.Entry<String, List<Double>> e : groups.entrySet()) {
            List<Double> vals = new ArrayList<>(e.getValue());
            if (vals.isEmpty()) { i++; continue; }
            Collections.sort(vals);
            double q1 = quantile(vals, 0.25);
            double med = quantile(vals, 0.5);
            double q3 = quantile(vals, 0.75);
            double iqr = q3 - q1;
            double whiskLo = q1 - 1.5 * iqr;
            double whiskHi = q3 + 1.5 * iqr;
            double wLo = vals.get(0), wHi = vals.get(vals.size() - 1);
            for (double v : vals) {
                if (v >= whiskLo) { wLo = v; break; }
            }
            for (int k = vals.size() - 1; k >= 0; k--) {
                if (vals.get(k) <= whiskHi) { wHi = vals.get(k); break; }
            }

            int cx = left + (int) (i * groupW + groupW / 2);
            int boxW = (int) (groupW * 0.4);
            int yQ1 = mapY(q1, yr[0], yr[1], top, plotH);
            int yQ3 = mapY(q3, yr[0], yr[1], top, plotH);
            int yMed = mapY(med, yr[0], yr[1], top, plotH);
            int yWLo = mapY(wLo, yr[0], yr[1], top, plotH);
            int yWHi = mapY(wHi, yr[0], yr[1], top, plotH);

            g.setColor(PALETTE[i % PALETTE.length]);
            int boxTop = Math.min(yQ1, yQ3);
            int boxH = Math.max(1, Math.abs(yQ3 - yQ1));
            g.drawRect(cx - boxW / 2, boxTop, boxW, boxH);
            g.drawLine(cx - boxW / 2, yMed, cx + boxW / 2, yMed);
            g.drawLine(cx, yQ3, cx, yWHi);
            g.drawLine(cx, yQ1, cx, yWLo);
            g.drawLine(cx - boxW / 4, yWHi, cx + boxW / 4, yWHi);
            g.drawLine(cx - boxW / 4, yWLo, cx + boxW / 4, yWLo);

            g.setColor(Color.DARK_GRAY);
            g.setFont(new Font("SansSerif", Font.PLAIN, 10));
            String lab = e.getKey();
            if (lab.length() > 10) lab = lab.substring(0, 10);
            FontMetrics fm = g.getFontMetrics();
            g.drawString(lab, cx - fm.stringWidth(lab) / 2, top + plotH + 28);
            i++;
        }
        g.dispose();
        return img;
    }

    private static double quantile(List<Double> sorted, double q) {
        if (sorted.isEmpty()) return Double.NaN;
        double pos = q * (sorted.size() - 1);
        int i = (int) Math.floor(pos);
        int j = Math.min(sorted.size() - 1, i + 1);
        double f = pos - i;
        return sorted.get(i) * (1 - f) + sorted.get(j) * f;
    }
}
