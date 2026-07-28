package org.bytedeco.pytorch.utils.plot;

import java.awt.*;
import java.awt.image.BufferedImage;
import java.util.List;
import java.util.Locale;

/** Heatmap chart with cmap / vmin / vmax / annot (seaborn heatmap surface). */
public final class HeatmapChart extends BaseChart {
    private final double[][] matrix;
    private final List<String> rowLabels;
    private final List<String> colLabels;
    private boolean showValues = false;
    private String cmap = "coolwarm";
    private Double vmin = null;
    private Double vmax = null;

    public HeatmapChart(String title, double[][] matrix, List<String> rowLabels, List<String> colLabels) {
        super(title);
        this.matrix = matrix;
        this.rowLabels = rowLabels;
        this.colLabels = colLabels;
    }

    public HeatmapChart setShowValues(boolean v) { this.showValues = v; return this; }
    /** Alias of setShowValues — seaborn {@code annot=True}. */
    public HeatmapChart setAnnot(boolean v) { return setShowValues(v); }
    public HeatmapChart setCmap(String name) {
        this.cmap = name == null ? "coolwarm" : name.toLowerCase(Locale.ROOT);
        return this;
    }
    public HeatmapChart setVmin(double v) { this.vmin = v; return this; }
    public HeatmapChart setVmax(double v) { this.vmax = v; return this; }

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
        if (vmin != null) lo = vmin;
        if (vmax != null) hi = vmax;
        if (hi <= lo) { hi = lo + 1; }

        if (title != null && !title.isEmpty()) {
            g.setColor(Color.BLACK);
            g.setFont(new Font("SansSerif", Font.BOLD, 16));
            FontMetrics fm = g.getFontMetrics();
            g.drawString(title, (width - fm.stringWidth(title)) / 2, 24);
        }

        int cellW = Math.max(1, plotW / cols), cellH = Math.max(1, plotH / rows);
        g.setFont(new Font("SansSerif", Font.PLAIN, 10));
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                double v = matrix[r][c];
                float t = (float) ((v - lo) / (hi - lo));
                if (t < 0) t = 0; if (t > 1) t = 1;
                Color color = mapColor(t, cmap);
                g.setColor(color);
                int x = left + c * cellW, y = top + r * cellH;
                g.fillRect(x, y, cellW, cellH);
                g.setColor(Color.LIGHT_GRAY);
                g.drawRect(x, y, cellW, cellH);
                if (showValues && !Double.isNaN(v)) {
                    double lum = 0.299 * color.getRed() + 0.587 * color.getGreen() + 0.114 * color.getBlue();
                    g.setColor(lum < 140 ? Color.WHITE : Color.BLACK);
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

    static Color mapColor(float t, String cmap) {
        if (t < 0) t = 0; if (t > 1) t = 1;
        return switch (cmap == null ? "coolwarm" : cmap) {
            case "viridis" -> viridis(t);
            case "plasma" -> plasma(t);
            case "magma" -> magma(t);
            case "inferno" -> inferno(t);
            case "blues" -> lerp(new Color(0xF7, 0xFB, 0xFF), new Color(0x08, 0x30, 0x6B), t);
            case "greens" -> lerp(new Color(0xF7, 0xFC, 0xF5), new Color(0x00, 0x41, 0x0B), t);
            case "rocket" -> rocket(t);
            case "mako" -> mako(t);
            default -> // coolwarm diverging
                t < 0.5f
                    ? lerp(new Color(0x21, 0x66, 0xac), Color.WHITE, t * 2)
                    : lerp(Color.WHITE, new Color(0xb2, 0x18, 0x2b), (t - 0.5f) * 2);
        };
    }

    private static Color viridis(float t) {
        // sampled control points approximating matplotlib viridis
        Color[] stops = {
            new Color(0x44, 0x01, 0x54), new Color(0x3B, 0x52, 0x84),
            new Color(0x21, 0x9A, 0x8C), new Color(0x5C, 0xC8, 0x63),
            new Color(0xFD, 0xE7, 0x25)
        };
        return multiLerp(stops, t);
    }

    private static Color plasma(float t) {
        Color[] stops = {
            new Color(0x0D, 0x08, 0x87), new Color(0x6A, 0x00, 0xA8),
            new Color(0xB1, 0x2A, 0x90), new Color(0xE1, 0x64, 0x62),
            new Color(0xF0, 0xF9, 0x21)
        };
        return multiLerp(stops, t);
    }

    private static Color magma(float t) {
        Color[] stops = {
            new Color(0x00, 0x00, 0x04), new Color(0x51, 0x15, 0x5A),
            new Color(0xB6, 0x36, 0x79), new Color(0xFB, 0x88, 0x61),
            new Color(0xFC, 0xF7, 0xB8)
        };
        return multiLerp(stops, t);
    }

    private static Color inferno(float t) {
        Color[] stops = {
            new Color(0x00, 0x00, 0x04), new Color(0x57, 0x0F, 0x6D),
            new Color(0xBB, 0x37, 0x54), new Color(0xF8, 0x8E, 0x20),
            new Color(0xFC, 0xFF, 0xA4)
        };
        return multiLerp(stops, t);
    }

    private static Color rocket(float t) {
        Color[] stops = {
            new Color(0x03, 0x01, 0x2d), new Color(0x5c, 0x1a, 0x6e),
            new Color(0xb6, 0x36, 0x79), new Color(0xee, 0x80, 0x5d),
            new Color(0xfa, 0xeb, 0xdd)
        };
        return multiLerp(stops, t);
    }

    private static Color mako(float t) {
        Color[] stops = {
            new Color(0x0B, 0x04, 0x05), new Color(0x35, 0x27, 0x4A),
            new Color(0x2E, 0x6B, 0x8E), new Color(0x35, 0xB7, 0x79),
            new Color(0xDE, 0xF5, 0xE5)
        };
        return multiLerp(stops, t);
    }

    private static Color multiLerp(Color[] stops, float t) {
        if (t <= 0) return stops[0];
        if (t >= 1) return stops[stops.length - 1];
        float scaled = t * (stops.length - 1);
        int i = (int) scaled;
        float f = scaled - i;
        return lerp(stops[i], stops[Math.min(i + 1, stops.length - 1)], f);
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
