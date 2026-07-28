package org.bytedeco.pytorch.utils.plot;

import java.awt.*;
import java.awt.image.BufferedImage;

/** Histogram chart with optional KDE overlay (seaborn histplot kde=True). */
public final class HistogramChart extends BaseChart {
    private final double[] data;
    private final int bins;
    private boolean kde = false;
    private boolean density = false;

    public HistogramChart(String title, double[] data, int bins) {
        super(title);
        this.data = data;
        this.bins = Math.max(1, bins);
        this.xAxisLabel = "value";
        this.yAxisLabel = "count";
    }

    /** Overlay Gaussian KDE curve (seaborn {@code histplot(..., kde=True)}). */
    public HistogramChart setKde(boolean v) { this.kde = v; return this; }

    /** Normalize bar heights to density (integrates ≈ 1). */
    public HistogramChart setDensity(boolean v) {
        this.density = v;
        if (v) this.yAxisLabel = "density";
        return this;
    }

    @Override public HistogramChart setTitle(String t) { super.setTitle(t); return this; }
    @Override public HistogramChart setXAxisLabel(String l) { super.setXAxisLabel(l); return this; }
    @Override public HistogramChart setYAxisLabel(String l) { super.setYAxisLabel(l); return this; }
    @Override public HistogramChart setSize(int w, int h) { super.setSize(w, h); return this; }

    @Override
    public BufferedImage render() {
        BufferedImage img = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = img.createGraphics();
        int left = 60, right = 20, top = 40, bottom = 50;
        int plotW = width - left - right, plotH = height - top - bottom;

        double lo = min(data), hi = max(data);
        if (hi <= lo) hi = lo + 1;
        double widthBin = (hi - lo) / bins;
        int[] counts = new int[bins];
        int nValid = 0;
        for (double v : data) {
            if (Double.isNaN(v)) continue;
            nValid++;
            int b = (int) ((v - lo) / widthBin);
            if (b >= bins) b = bins - 1;
            if (b < 0) b = 0;
            counts[b]++;
        }
        double[] heights = new double[bins];
        double maxH = 1e-12;
        for (int i = 0; i < bins; i++) {
            heights[i] = density && nValid > 0
                ? counts[i] / (nValid * widthBin)
                : counts[i];
            if (heights[i] > maxH) maxH = heights[i];
        }

        // KDE may exceed histogram peak; expand y-range if needed
        double[][] kdeGrid = null;
        if (kde) {
            kdeGrid = Seaborn.kdeGridPublic(data, 120);
            for (double d : kdeGrid[1]) if (d > maxH) maxH = d;
            if (!density) {
                // scale KDE to count units so it overlays the histogram
                for (int i = 0; i < kdeGrid[1].length; i++)
                    kdeGrid[1][i] *= nValid * widthBin;
                for (double d : kdeGrid[1]) if (d > maxH) maxH = d;
            }
        }

        drawAxesFrame(g, left, top, plotW, plotH, lo, hi, 0, maxH * 1.1);

        g.setColor(PALETTE[0]);
        for (int i = 0; i < bins; i++) {
            double x0 = lo + i * widthBin;
            double x1 = lo + (i + 1) * widthBin;
            int px0 = mapX(x0, lo, hi, left, plotW);
            int px1 = mapX(x1, lo, hi, left, plotW);
            int py0 = mapY(0, 0, maxH * 1.1, top, plotH);
            int py1 = mapY(heights[i], 0, maxH * 1.1, top, plotH);
            g.fillRect(px0, Math.min(py0, py1), Math.max(1, px1 - px0 - 1), Math.abs(py0 - py1));
            g.setColor(PALETTE[0].darker());
            g.drawRect(px0, Math.min(py0, py1), Math.max(1, px1 - px0 - 1), Math.abs(py0 - py1));
            g.setColor(PALETTE[0]);
        }

        if (kdeGrid != null) {
            g.setColor(new Color(0xC4, 0x4E, 0x52));
            g.setStroke(new BasicStroke(2f));
            double[] kx = kdeGrid[0], ky = kdeGrid[1];
            for (int i = 1; i < kx.length; i++) {
                if (Double.isNaN(kx[i - 1]) || Double.isNaN(ky[i - 1])
                    || Double.isNaN(kx[i]) || Double.isNaN(ky[i])) continue;
                int x0 = mapX(kx[i - 1], lo, hi, left, plotW);
                int y0 = mapY(ky[i - 1], 0, maxH * 1.1, top, plotH);
                int x1 = mapX(kx[i], lo, hi, left, plotW);
                int y1 = mapY(ky[i], 0, maxH * 1.1, top, plotH);
                g.drawLine(x0, y0, x1, y1);
            }
        }
        g.dispose();
        return img;
    }
}
