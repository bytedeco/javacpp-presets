package org.bytedeco.pytorch.data.dataframe.plot;

import org.bytedeco.pytorch.data.dataframe.DataFrame;
import org.bytedeco.pytorch.data.dataframe.DataValues;

import java.awt.*;
import java.awt.geom.Arc2D;
import java.awt.image.BufferedImage;
import java.util.ArrayList;
import java.util.List;

/** Pie chart (pure AWT). */
public final class PieChart extends BaseChart {
    private final List<String> labels = new ArrayList<>();
    private final List<Double> values = new ArrayList<>();

    public PieChart(String title, String[] labels, double[] values) {
        super(title);
        for (String l : labels) this.labels.add(l == null ? "" : l);
        for (double v : values) this.values.add(v);
    }

    public PieChart(String title, DataFrame df, String labelCol, String valueCol) {
        super(title);
        int n = df.rowCount();
        for (int i = 0; i < n; i++) {
            Object lab = df.get(i, labelCol);
            labels.add(lab == null ? "" : lab.toString());
            values.add(DataValues.asDouble(df.get(i, valueCol)));
        }
    }

    @Override public PieChart setTitle(String t) { super.setTitle(t); return this; }
    @Override public PieChart setSize(int w, int h) { super.setSize(w, h); return this; }

    @Override
    public BufferedImage render() {
        BufferedImage img = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = img.createGraphics();
        g.setColor(background);
        g.fillRect(0, 0, width, height);
        g.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);

        if (title != null && !title.isEmpty()) {
            g.setColor(Color.BLACK);
            g.setFont(new Font("SansSerif", Font.BOLD, 16));
            FontMetrics fm = g.getFontMetrics();
            g.drawString(title, (width - fm.stringWidth(title)) / 2, 24);
        }

        double total = 0;
        for (double v : values) if (!Double.isNaN(v) && v > 0) total += v;
        if (total <= 0) total = 1;

        int size = Math.min(width, height) - 120;
        int cx = width / 2 - 40;
        int cy = height / 2 + 10;
        int x = cx - size / 2, y = cy - size / 2;

        double angle = 90;
        for (int i = 0; i < values.size(); i++) {
            double v = values.get(i);
            if (Double.isNaN(v) || v <= 0) continue;
            double extent = -360.0 * v / total;
            g.setColor(PALETTE[i % PALETTE.length]);
            g.fill(new Arc2D.Double(x, y, size, size, angle, extent, Arc2D.PIE));
            angle += extent;
        }
        g.setColor(Color.DARK_GRAY);
        g.drawOval(x, y, size, size);

        if (showLegend) {
            g.setFont(new Font("SansSerif", Font.PLAIN, 11));
            int lx = width - 140, ly = 50;
            for (int i = 0; i < labels.size(); i++) {
                g.setColor(PALETTE[i % PALETTE.length]);
                g.fillRect(lx, ly + i * 18, 12, 12);
                g.setColor(Color.DARK_GRAY);
                String lab = labels.get(i);
                if (lab.length() > 12) lab = lab.substring(0, 12);
                double pct = total > 0 && i < values.size() ? 100.0 * Math.max(0, values.get(i)) / total : 0;
                g.drawString(String.format("%s (%.0f%%)", lab, pct), lx + 16, ly + i * 18 + 11);
            }
        }
        g.dispose();
        return img;
    }
}
