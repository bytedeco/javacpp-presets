package org.bytedeco.pytorch.data.dataframe.plot;

import java.awt.*;
import java.awt.image.BufferedImage;

/** Histogram chart. */
public final class HistogramChart extends BaseChart {
    private final double[] data;
    private final int bins;

    public HistogramChart(String title, double[] data, int bins) {
        super(title);
        this.data = data;
        this.bins = Math.max(1, bins);
        this.xAxisLabel = "value";
        this.yAxisLabel = "count";
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
        for (double v : data) {
            if (Double.isNaN(v)) continue;
            int b = (int) ((v - lo) / widthBin);
            if (b >= bins) b = bins - 1;
            if (b < 0) b = 0;
            counts[b]++;
        }
        int maxC = 1;
        for (int c : counts) if (c > maxC) maxC = c;

        drawAxesFrame(g, left, top, plotW, plotH, lo, hi, 0, maxC * 1.1);

        g.setColor(PALETTE[0]);
        for (int i = 0; i < bins; i++) {
            double x0 = lo + i * widthBin;
            double x1 = lo + (i + 1) * widthBin;
            int px0 = mapX(x0, lo, hi, left, plotW);
            int px1 = mapX(x1, lo, hi, left, plotW);
            int py0 = mapY(0, 0, maxC * 1.1, top, plotH);
            int py1 = mapY(counts[i], 0, maxC * 1.1, top, plotH);
            g.fillRect(px0, Math.min(py0, py1), Math.max(1, px1 - px0 - 1), Math.abs(py0 - py1));
            g.setColor(PALETTE[0].darker());
            g.drawRect(px0, Math.min(py0, py1), Math.max(1, px1 - px0 - 1), Math.abs(py0 - py1));
            g.setColor(PALETTE[0]);
        }
        g.dispose();
        return img;
    }
}
