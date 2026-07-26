package org.bytedeco.pytorch.data.dataframe.plot;

import java.awt.*;
import java.awt.image.BufferedImage;
import java.util.List;

/** Heatmap chart. */
public final class HeatmapChart extends BaseChart {
    private final double[][] matrix;
    private final List<String> rowLabels;
    private final List<String> colLabels;
    private boolean showValues = false;

    public HeatmapChart(String title, double[][] matrix, List<String> rowLabels, List<String> colLabels) {
        super(title);
        this.matrix = matrix;
        this.rowLabels = rowLabels;
        this.colLabels = colLabels;
    }

    public HeatmapChart setShowValues(boolean v) { this.showValues = v; return this; }

    @Override public HeatmapChart setTitle(String t) { super.setTitle(t); return this; }
    @Override public HeatmapChart setSize(int w, int h) { super.setSize(w, h); return this; }

    @Override
    public BufferedImage render() {
        BufferedImage img = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = img.createGraphics();
        g.setColor(Color.WHITE);
        g.fillRect(0, 0, width, height);
        g.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);

        int rows = matrix.length;
        int cols = rows == 0 ? 0 : matrix[0].length;
        int left = 80, top = 40, right = 40, bottom = 60;
        int plotW = width - left - right, plotH = height - top - bottom;
        if (rows == 0 || cols == 0) { g.dispose(); return img; }

        double lo = Double.POSITIVE_INFINITY, hi = Double.NEGATIVE_INFINITY;
        for (double[] row : matrix) for (double v : row) {
            if (!Double.isNaN(v)) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
        }
        if (hi <= lo) { hi = lo + 1; }

        if (title != null && !title.isEmpty()) {
            g.setColor(Color.BLACK);
            g.setFont(new Font("SansSerif", Font.BOLD, 16));
            FontMetrics fm = g.getFontMetrics();
            g.drawString(title, (width - fm.stringWidth(title)) / 2, 24);
        }

        int cellW = plotW / cols, cellH = plotH / rows;
        g.setFont(new Font("SansSerif", Font.PLAIN, 10));
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                double v = matrix[r][c];
                float t = (float) ((v - lo) / (hi - lo));
                if (t < 0) t = 0; if (t > 1) t = 1;
                // blue → white → red
                Color color = t < 0.5
                    ? lerp(new Color(0x21, 0x66, 0xac), Color.WHITE, t * 2)
                    : lerp(Color.WHITE, new Color(0xb2, 0x18, 0x2b), (t - 0.5f) * 2);
                g.setColor(color);
                int x = left + c * cellW, y = top + r * cellH;
                g.fillRect(x, y, cellW, cellH);
                g.setColor(Color.LIGHT_GRAY);
                g.drawRect(x, y, cellW, cellH);
                if (showValues && !Double.isNaN(v)) {
                    g.setColor(Color.BLACK);
                    String s = formatTick(v);
                    FontMetrics fm = g.getFontMetrics();
                    g.drawString(s, x + (cellW - fm.stringWidth(s)) / 2, y + cellH / 2 + 4);
                }
            }
        }

        g.setColor(Color.DARK_GRAY);
        if (rowLabels != null) {
            for (int r = 0; r < rows && r < rowLabels.size(); r++) {
                String lab = rowLabels.get(r);
                FontMetrics fm = g.getFontMetrics();
                g.drawString(lab, left - fm.stringWidth(lab) - 4, top + r * cellH + cellH / 2 + 4);
            }
        }
        if (colLabels != null) {
            for (int c = 0; c < cols && c < colLabels.size(); c++) {
                String lab = colLabels.get(c);
                FontMetrics fm = g.getFontMetrics();
                g.drawString(lab, left + c * cellW + (cellW - fm.stringWidth(lab)) / 2, top + plotH + 14);
            }
        }
        g.dispose();
        return img;
    }

    private static Color lerp(Color a, Color b, float t) {
        int r = (int) (a.getRed() + (b.getRed() - a.getRed()) * t);
        int g = (int) (a.getGreen() + (b.getGreen() - a.getGreen()) * t);
        int bl = (int) (a.getBlue() + (b.getBlue() - a.getBlue()) * t);
        return new Color(
            Math.min(255, Math.max(0, r)),
            Math.min(255, Math.max(0, g)),
            Math.min(255, Math.max(0, bl)));
    }
}
