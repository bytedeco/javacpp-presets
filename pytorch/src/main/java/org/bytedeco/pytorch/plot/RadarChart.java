package org.bytedeco.pytorch.plot;

import org.bytedeco.pytorch.dataframe.DataFrame;
import org.bytedeco.pytorch.dataframe.DataValues;

import java.awt.*;
import java.awt.geom.Path2D;
import java.awt.image.BufferedImage;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/** Radar / spider chart. */
public final class RadarChart extends BaseChart {
    private final List<String> categories = new ArrayList<>();
    private final Map<String, double[]> series = new LinkedHashMap<>();

    /** Categories from labels; each series is values aligned to categories. */
    public RadarChart(String title, List<String> categories, Map<String, double[]> series) {
        super(title);
        if (categories != null) this.categories.addAll(categories);
        if (series != null) this.series.putAll(series);
        this.width = 700;
        this.height = 700;
    }

    /**
     * One series from category + value columns (one row per spoke).
     */
    public RadarChart(String title, DataFrame df, String categoryCol, String valueCol) {
        super(title);
        this.width = 700;
        this.height = 700;
        int n = df.rowCount();
        double[] vals = new double[n];
        for (int i = 0; i < n; i++) {
            Object c = DataValues.unwrap(df.get(i, categoryCol));
            categories.add(c == null ? "" : c.toString());
            vals[i] = DataValues.asDouble(df.get(i, valueCol));
        }
        series.put(valueCol, vals);
    }

    public RadarChart addSeries(String name, double[] values) {
        series.put(name, values);
        return this;
    }

    @Override public RadarChart setTitle(String t) { super.setTitle(t); return this; }
    @Override public RadarChart setSize(int w, int h) { super.setSize(w, h); return this; }

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

        int n = Math.max(3, categories.isEmpty() ? 3 : categories.size());
        int cx = width / 2, cy = height / 2 + 10;
        int radius = Math.min(width, height) / 2 - 70;

        double maxV = 1e-9;
        for (double[] s : series.values())
            for (double v : s) if (!Double.isNaN(v)) maxV = Math.max(maxV, v);

        // grid
        g.setColor(new Color(210, 210, 210));
        for (int ring = 1; ring <= 5; ring++) {
            double r = radius * ring / 5.0;
            Path2D poly = new Path2D.Double();
            for (int i = 0; i < n; i++) {
                double ang = -Math.PI / 2 + 2 * Math.PI * i / n;
                double px = cx + r * Math.cos(ang);
                double py = cy + r * Math.sin(ang);
                if (i == 0) poly.moveTo(px, py); else poly.lineTo(px, py);
            }
            poly.closePath();
            g.draw(poly);
        }
        g.setColor(Color.GRAY);
        for (int i = 0; i < n; i++) {
            double ang = -Math.PI / 2 + 2 * Math.PI * i / n;
            g.drawLine(cx, cy, (int) (cx + radius * Math.cos(ang)), (int) (cy + radius * Math.sin(ang)));
        }

        // labels
        g.setColor(Color.DARK_GRAY);
        g.setFont(new Font("SansSerif", Font.PLAIN, 11));
        for (int i = 0; i < n; i++) {
            double ang = -Math.PI / 2 + 2 * Math.PI * i / n;
            String lab = i < categories.size() ? categories.get(i) : ("C" + i);
            if (lab.length() > 10) lab = lab.substring(0, 10);
            FontMetrics fm = g.getFontMetrics();
            int lx = (int) (cx + (radius + 18) * Math.cos(ang) - fm.stringWidth(lab) / 2.0);
            int ly = (int) (cy + (radius + 18) * Math.sin(ang) + 4);
            g.drawString(lab, lx, ly);
        }

        int si = 0;
        for (Map.Entry<String, double[]> e : series.entrySet()) {
            double[] vals = e.getValue();
            Path2D path = new Path2D.Double();
            for (int i = 0; i < n; i++) {
                double v = i < vals.length && !Double.isNaN(vals[i]) ? vals[i] : 0;
                double r = radius * (v / maxV);
                double ang = -Math.PI / 2 + 2 * Math.PI * i / n;
                double px = cx + r * Math.cos(ang);
                double py = cy + r * Math.sin(ang);
                if (i == 0) path.moveTo(px, py); else path.lineTo(px, py);
            }
            path.closePath();
            Color c = PALETTE[si % PALETTE.length];
            g.setColor(new Color(c.getRed(), c.getGreen(), c.getBlue(), 80));
            g.fill(path);
            g.setColor(c);
            g.setStroke(new BasicStroke(2f));
            g.draw(path);
            si++;
        }

        if (showLegend && series.size() > 1) {
            g.setFont(new Font("SansSerif", Font.PLAIN, 11));
            int li = 0;
            for (String name : series.keySet()) {
                g.setColor(PALETTE[li % PALETTE.length]);
                g.fillRect(20, 50 + li * 16, 12, 10);
                g.setColor(Color.DARK_GRAY);
                g.drawString(name, 36, 59 + li * 16);
                li++;
            }
        }
        g.dispose();
        return img;
    }
}
