package org.bytedeco.pytorch.utils.plot;

import java.awt.*;
import java.awt.image.BufferedImage;
import java.util.ArrayList;
import java.util.List;

/**
 * Polar line chart (matplotlib {@code plt.subplot(projection="polar")}).
 * Theta in radians, r ≥ 0. Distinct from {@link RadarChart} (categorical spokes).
 */
public final class PolarChart extends BaseChart {
    private final List<double[]> thetas = new ArrayList<>();
    private final List<double[]> rs = new ArrayList<>();
    private final List<String> labels = new ArrayList<>();

    public PolarChart(String title, double[] theta, double[] r) {
        super(title == null ? "polar" : title);
        thetas.add(theta);
        rs.add(r);
        labels.add("r");
    }

    public PolarChart addSeries(double[] theta, double[] r, String label) {
        thetas.add(theta);
        rs.add(r);
        labels.add(label == null ? ("r" + thetas.size()) : label);
        return this;
    }

    @Override public PolarChart setTitle(String t) { super.setTitle(t); return this; }
    @Override public PolarChart setSize(int w, int h) { super.setSize(w, h); return this; }
    @Override public PolarChart setShowLegend(boolean v) { super.setShowLegend(v); return this; }

    @Override
    public BufferedImage render() {
        BufferedImage img = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = img.createGraphics();
        g.setColor(background);
        g.fillRect(0, 0, width, height);
        g.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);

        g.setColor(Color.BLACK);
        g.setFont(new Font("SansSerif", Font.BOLD, 16));
        if (title != null && !title.isEmpty()) {
            FontMetrics fm = g.getFontMetrics();
            g.drawString(title, (width - fm.stringWidth(title)) / 2, 24);
        }

        int cx = width / 2;
        int cy = height / 2 + 10;
        int radius = Math.min(width, height) / 2 - 50;

        double rMax = 0;
        for (double[] rr : rs) rMax = Math.max(rMax, max(rr));
        if (rMax <= 0) rMax = 1;

        // concentric circles + angle spokes
        g.setColor(new Color(220, 220, 220));
        for (int k = 1; k <= 4; k++) {
            int rad = radius * k / 4;
            g.drawOval(cx - rad, cy - rad, rad * 2, rad * 2);
        }
        g.setColor(new Color(180, 180, 180));
        for (int a = 0; a < 12; a++) {
            double th = a * Math.PI / 6;
            int x2 = cx + (int) Math.round(radius * Math.cos(th));
            int y2 = cy - (int) Math.round(radius * Math.sin(th));
            g.drawLine(cx, cy, x2, y2);
        }
        g.setColor(Color.GRAY);
        g.drawOval(cx - radius, cy - radius, radius * 2, radius * 2);

        // angle labels
        g.setFont(new Font("SansSerif", Font.PLAIN, 10));
        g.setColor(Color.DARK_GRAY);
        String[] labs = {"0", "π/6", "π/3", "π/2", "2π/3", "5π/6", "π",
            "7π/6", "4π/3", "3π/2", "5π/3", "11π/6"};
        for (int a = 0; a < 12; a++) {
            double th = a * Math.PI / 6;
            int lx = cx + (int) Math.round((radius + 14) * Math.cos(th)) - 8;
            int ly = cy - (int) Math.round((radius + 14) * Math.sin(th)) + 4;
            g.drawString(labs[a], lx, ly);
        }

        for (int s = 0; s < rs.size(); s++) {
            g.setColor(PALETTE[s % PALETTE.length]);
            g.setStroke(new BasicStroke(2f));
            double[] th = thetas.get(s);
            double[] rr = rs.get(s);
            int n = Math.min(th.length, rr.length);
            int prevX = 0, prevY = 0;
            boolean hasPrev = false;
            for (int i = 0; i < n; i++) {
                if (Double.isNaN(th[i]) || Double.isNaN(rr[i])) { hasPrev = false; continue; }
                double rad = radius * (rr[i] / rMax);
                int px = cx + (int) Math.round(rad * Math.cos(th[i]));
                int py = cy - (int) Math.round(rad * Math.sin(th[i]));
                if (hasPrev) g.drawLine(prevX, prevY, px, py);
                prevX = px; prevY = py; hasPrev = true;
            }
        }

        if (showLegend && labels.size() > 1) {
            g.setFont(new Font("SansSerif", Font.PLAIN, 11));
            for (int s = 0; s < labels.size(); s++) {
                g.setColor(PALETTE[s % PALETTE.length]);
                g.fillRect(12, 40 + s * 16, 12, 12);
                g.setColor(Color.BLACK);
                g.drawString(labels.get(s), 28, 51 + s * 16);
            }
        }
        g.dispose();
        return img;
    }
}
