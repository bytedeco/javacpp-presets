package org.bytedeco.pytorch.plot;

import org.bytedeco.pytorch.plot.chart.BaseChart;

import java.awt.*;
import java.awt.image.BufferedImage;

/**
 * Multi-axes figure (matplotlib {@code plt.subplots(nrows, ncols)}).
 * Holds a grid of {@link BaseChart} children and composites them into one image.
 *
 * <pre>
 *   Figure fig = Matplotlib.subplots(2, 2);
 *   fig.set(0, 0, Matplotlib.plot(x, y));
 *   fig.set(0, 1, Matplotlib.hist(data, 20));
 *   fig.savefig("/tmp/grid.png");
 * </pre>
 */
public final class Figure extends BaseChart {
    private final int rows;
    private final int cols;
    private final BaseChart[][] axes;
    private int hspace = 16;
    private int wspace = 16;
    private int outerPad = 12;

    public Figure(int rows, int cols) {
        super("Figure");
        if (rows < 1 || cols < 1) throw new IllegalArgumentException("rows/cols must be ≥ 1");
        this.rows = rows;
        this.cols = cols;
        this.axes = new BaseChart[rows][cols];
        // default size scales with grid
        this.width = Math.max(800, cols * 400);
        this.height = Math.max(600, rows * 320);
    }

    public int rows() { return rows; }
    public int cols() { return cols; }

    public BaseChart get(int r, int c) {
        check(r, c);
        return axes[r][c];
    }

    public Figure set(int r, int c, BaseChart chart) {
        check(r, c);
        axes[r][c] = chart;
        return this;
    }

    /** Flat index access: row-major. */
    public Figure set(int flatIndex, BaseChart chart) {
        int r = flatIndex / cols;
        int c = flatIndex % cols;
        return set(r, c, chart);
    }

    public Figure setHspace(int px) { this.hspace = Math.max(0, px); return this; }
    public Figure setWspace(int px) { this.wspace = Math.max(0, px); return this; }

    @Override public Figure setTitle(String t) { super.setTitle(t); return this; }
    @Override public Figure setSize(int w, int h) { super.setSize(w, h); return this; }

    private void check(int r, int c) {
        if (r < 0 || r >= rows || c < 0 || c >= cols)
            throw new IndexOutOfBoundsException("axes[" + r + "," + c + "] out of " + rows + "x" + cols);
    }

    @Override
    public BufferedImage render() {
        BufferedImage img = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = img.createGraphics();
        g.setColor(background);
        g.fillRect(0, 0, width, height);
        g.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);

        int titleH = 0;
        if (title != null && !title.isEmpty()) {
            g.setColor(Color.BLACK);
            g.setFont(new Font("SansSerif", Font.BOLD, 16));
            FontMetrics fm = g.getFontMetrics();
            g.drawString(title, (width - fm.stringWidth(title)) / 2, 22);
            titleH = 28;
        }

        int availW = width - 2 * outerPad - wspace * (cols - 1);
        int availH = height - 2 * outerPad - titleH - hspace * (rows - 1);
        int cellW = Math.max(40, availW / cols);
        int cellH = Math.max(40, availH / rows);

        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                int x = outerPad + c * (cellW + wspace);
                int y = outerPad + titleH + r * (cellH + hspace);
                BaseChart ch = axes[r][c];
                if (ch == null) {
                    g.setColor(new Color(245, 245, 245));
                    g.fillRect(x, y, cellW, cellH);
                    g.setColor(Color.LIGHT_GRAY);
                    g.drawRect(x, y, cellW, cellH);
                    g.setColor(Color.GRAY);
                    g.drawString("(" + r + "," + c + ")", x + 8, y + 20);
                    continue;
                }
                // temporarily size child to cell
                int ow = ch.width, oh = ch.height;
                ch.setSize(cellW, cellH);
                BufferedImage cell = ch.render();
                ch.setSize(ow, oh);
                g.drawImage(cell, x, y, null);
            }
        }
        g.dispose();
        return img;
    }
}
