package org.bytedeco.pytorch.utils.plot;

import java.awt.*;
import java.awt.image.BufferedImage;
import java.util.Locale;

/**
 * 2D histogram (matplotlib {@code plt.hist2d}). Counts binned into a heatmap.
 */
public final class Hist2DChart extends BaseChart {
    private final double[] x;
    private final double[] y;
    private final int binsX;
    private final int binsY;
    private String cmap = "blues";
    private boolean showColorbar = true;

    public Hist2DChart(String title, double[] x, double[] y, int bins) {
        this(title, x, y, bins, bins);
    }

    public Hist2DChart(String title, double[] x, double[] y, int binsX, int binsY) {
        super(title == null ? "hist2d" : title);
        this.x = x;
        this.y = y;
        this.binsX = Math.max(1, binsX);
        this.binsY = Math.max(1, binsY);
    }

    public Hist2DChart setCmap(String name) {
        this.cmap = name == null ? "blues" : name.toLowerCase(Locale.ROOT);
        return this;
    }

    public Hist2DChart setShowColorbar(boolean v) { this.showColorbar = v; return this; }

    @Override public Hist2DChart setTitle(String t) { super.setTitle(t); return this; }
    @Override public Hist2DChart setXAxisLabel(String l) { super.setXAxisLabel(l); return this; }
    @Override public Hist2DChart setYAxisLabel(String l) { super.setYAxisLabel(l); return this; }
    @Override public Hist2DChart setSize(int w, int h) { super.setSize(w, h); return this; }

    @Override
    public BufferedImage render() {
        double xMin = min(x), xMax = max(x);
        double yMin = min(y), yMax = max(y);
        if (xMax <= xMin) { xMin -= 0.5; xMax += 0.5; }
        if (yMax <= yMin) { yMin -= 0.5; yMax += 0.5; }
        double[][] counts = new double[binsY][binsX];
        int n = Math.min(x.length, y.length);
        for (int i = 0; i < n; i++) {
            if (Double.isNaN(x[i]) || Double.isNaN(y[i])) continue;
            int bx = (int) ((x[i] - xMin) / (xMax - xMin) * binsX);
            int by = (int) ((y[i] - yMin) / (yMax - yMin) * binsY);
            if (bx < 0) bx = 0; if (bx >= binsX) bx = binsX - 1;
            if (by < 0) by = 0; if (by >= binsY) by = binsY - 1;
            // image row 0 is top → invert y so low y is bottom
            counts[binsY - 1 - by][bx]++;
        }

        BufferedImage img = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = img.createGraphics();
        int cbW = showColorbar ? 40 : 0;
        int left = 60, right = 20 + cbW, top = 40, bottom = 50;
        int plotW = width - left - right, plotH = height - top - bottom;

        double cMax = 0;
        for (double[] row : counts) for (double v : row) cMax = Math.max(cMax, v);
        if (cMax <= 0) cMax = 1;

        drawAxesFrame(g, left, top, plotW, plotH, xMin, xMax, yMin, yMax);

        int cellW = Math.max(1, plotW / binsX);
        int cellH = Math.max(1, plotH / binsY);
        for (int r = 0; r < binsY; r++) {
            for (int c = 0; c < binsX; c++) {
                float t = (float) (counts[r][c] / cMax);
                g.setColor(HeatmapChart.mapColor(t, cmap));
                int px = left + c * plotW / binsX;
                int py = top + r * plotH / binsY;
                g.fillRect(px, py, cellW + 1, cellH + 1);
            }
        }
        g.setColor(Color.GRAY);
        g.drawRect(left, top, plotW, plotH);

        if (showColorbar) {
            int cbX = left + plotW + 10;
            for (int i = 0; i < plotH; i++) {
                float t = 1f - (float) i / Math.max(1, plotH - 1);
                g.setColor(HeatmapChart.mapColor(t, cmap));
                g.fillRect(cbX, top + i, 12, 1);
            }
            g.setColor(Color.DARK_GRAY);
            g.drawRect(cbX, top, 12, plotH);
            g.setFont(new Font("SansSerif", Font.PLAIN, 10));
            g.drawString(formatTick(cMax), cbX + 14, top + 10);
            g.drawString("0", cbX + 14, top + plotH);
        }
        g.dispose();
        return img;
    }
}
