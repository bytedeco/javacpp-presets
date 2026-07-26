package org.bytedeco.pytorch.data.dataframe.plot;

import org.bytedeco.pytorch.data.dataframe.DataFrame;
import org.bytedeco.pytorch.data.dataframe.DataValues;

import java.awt.*;
import java.awt.geom.Path2D;
import java.awt.image.BufferedImage;
import java.util.ArrayList;
import java.util.List;

/** Funnel chart for stage conversion visualization. */
public final class FunnelChart extends BaseChart {
    private final List<String> stages = new ArrayList<>();
    private final List<Double> values = new ArrayList<>();

    public FunnelChart(String title, String[] stages, double[] values) {
        super(title);
        for (String s : stages) this.stages.add(s == null ? "" : s);
        for (double v : values) this.values.add(v);
    }

    public FunnelChart(String title, DataFrame df, String stageCol, String valueCol) {
        super(title);
        int n = df.rowCount();
        for (int i = 0; i < n; i++) {
            Object s = DataValues.unwrap(df.get(i, stageCol));
            stages.add(s == null ? "" : s.toString());
            values.add(DataValues.asDouble(df.get(i, valueCol)));
        }
    }

    @Override public FunnelChart setTitle(String t) { super.setTitle(t); return this; }
    @Override public FunnelChart setSize(int w, int h) { super.setSize(w, h); return this; }

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
            g.drawString(title, (width - fm.stringWidth(title)) / 2, 28);
        }

        int n = Math.min(stages.size(), values.size());
        if (n == 0) { g.dispose(); return img; }

        double maxV = 0;
        for (double v : values) if (!Double.isNaN(v)) maxV = Math.max(maxV, v);
        if (maxV <= 0) maxV = 1;

        int top = 50, bottom = 30, left = 40, right = 40;
        int plotH = height - top - bottom;
        int plotW = width - left - right;
        double rowH = plotH / (double) n;
        int cx = left + plotW / 2;

        g.setFont(new Font("SansSerif", Font.PLAIN, 12));
        for (int i = 0; i < n; i++) {
            double v = values.get(i);
            if (Double.isNaN(v) || v < 0) v = 0;
            double next = (i + 1 < n && !Double.isNaN(values.get(i + 1))) ? Math.max(0, values.get(i + 1)) : v * 0.6;
            double wTop = plotW * 0.9 * (v / maxV);
            double wBot = plotW * 0.9 * (next / maxV);
            if (i == n - 1) wBot = wTop * 0.55;

            double y0 = top + i * rowH;
            double y1 = top + (i + 1) * rowH - 4;

            Path2D trap = new Path2D.Double();
            trap.moveTo(cx - wTop / 2, y0);
            trap.lineTo(cx + wTop / 2, y0);
            trap.lineTo(cx + wBot / 2, y1);
            trap.lineTo(cx - wBot / 2, y1);
            trap.closePath();

            Color c = PALETTE[i % PALETTE.length];
            g.setColor(new Color(c.getRed(), c.getGreen(), c.getBlue(), 200));
            g.fill(trap);
            g.setColor(c.darker());
            g.draw(trap);

            g.setColor(Color.WHITE);
            FontMetrics fm = g.getFontMetrics();
            String lab = stages.get(i) + "  " + formatTick(v);
            int tx = cx - fm.stringWidth(lab) / 2;
            int ty = (int) ((y0 + y1) / 2 + fm.getAscent() / 2.0 - 2);
            g.drawString(lab, tx, ty);
        }
        g.dispose();
        return img;
    }
}
