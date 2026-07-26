package org.bytedeco.pytorch.data.dataframe.plot;

import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.atomic.AtomicReference;
import javax.imageio.ImageIO;
import javax.swing.*;

/**
 * Base chart with fluent config, AWT rendering (headless-safe savefig),
 * and optional Swing display.
 */
public abstract class BaseChart {
    protected String title;
    protected String xAxisLabel = "";
    protected String yAxisLabel = "";
    protected int width = 800;
    protected int height = 600;
    protected Color background = Color.WHITE;
    protected boolean showGrid = true;
    protected boolean showLegend = true;

    protected BaseChart(String title) {
        this.title = title == null ? "" : title;
    }

    public BaseChart setTitle(String t) { this.title = t; return this; }
    public BaseChart setXAxisLabel(String label) { this.xAxisLabel = label; return this; }
    public BaseChart setYAxisLabel(String label) { this.yAxisLabel = label; return this; }
    public BaseChart setSize(int width, int height) {
        this.width = width; this.height = height; return this;
    }
    public BaseChart setShowGrid(boolean v) { this.showGrid = v; return this; }
    public BaseChart setShowLegend(boolean v) { this.showLegend = v; return this; }

    /** Render to a BufferedImage (pure AWT — works headless). */
    public abstract BufferedImage render();

    /** Save as PNG/JPEG based on extension. */
    public void savefig(String path) throws Exception {
        BufferedImage img = render();
        String lower = path.toLowerCase();
        String fmt = lower.endsWith(".jpg") || lower.endsWith(".jpeg") ? "jpg" : "png";
        File f = new File(path);
        if (f.getParentFile() != null) f.getParentFile().mkdirs();
        ImageIO.write(img, fmt, f);
    }

    /** Show in a Swing window (no-op if headless). */
    public void show() {
        if (GraphicsEnvironment.isHeadless()) {
            System.out.println("[plot] headless — skip show(); use savefig()");
            return;
        }
        try {
            BufferedImage img = render();
            CountDownLatch latch = new CountDownLatch(1);
            AtomicReference<JFrame> frameRef = new AtomicReference<>();
            SwingUtilities.invokeLater(() -> {
                JFrame frame = new JFrame(title == null || title.isEmpty() ? "Plot" : title);
                frame.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
                frame.add(new JLabel(new ImageIcon(img)));
                frame.pack();
                frame.setLocationRelativeTo(null);
                frame.setVisible(true);
                frameRef.set(frame);
                latch.countDown();
            });
            latch.await();
        } catch (Exception e) {
            System.err.println("[plot] show failed: " + e.getMessage());
        }
    }

    // ---- shared drawing helpers ----

    protected static final Color[] PALETTE = {
        new Color(0x1f77b4), new Color(0xff7f0e), new Color(0x2ca02c),
        new Color(0xd62728), new Color(0x9467bd), new Color(0x8c564b),
        new Color(0xe377c2), new Color(0x7f7f7f), new Color(0xbcbd22),
        new Color(0x17becf)
    };

    protected void drawAxesFrame(Graphics2D g, int left, int top, int plotW, int plotH,
                                 double xMin, double xMax, double yMin, double yMax) {
        g.setColor(background);
        g.fillRect(0, 0, width, height);
        g.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);

        // title
        g.setColor(Color.BLACK);
        g.setFont(new Font("SansSerif", Font.BOLD, 16));
        if (title != null && !title.isEmpty()) {
            FontMetrics fm = g.getFontMetrics();
            g.drawString(title, (width - fm.stringWidth(title)) / 2, 24);
        }

        // plot background
        g.setColor(new Color(250, 250, 250));
        g.fillRect(left, top, plotW, plotH);
        g.setColor(Color.GRAY);
        g.drawRect(left, top, plotW, plotH);

        if (showGrid) {
            g.setColor(new Color(220, 220, 220));
            for (int i = 1; i < 5; i++) {
                int x = left + i * plotW / 5;
                int y = top + i * plotH / 5;
                g.drawLine(x, top, x, top + plotH);
                g.drawLine(left, y, left + plotW, y);
            }
        }

        // axis labels
        g.setColor(Color.DARK_GRAY);
        g.setFont(new Font("SansSerif", Font.PLAIN, 12));
        if (xAxisLabel != null && !xAxisLabel.isEmpty()) {
            FontMetrics fm = g.getFontMetrics();
            g.drawString(xAxisLabel, left + (plotW - fm.stringWidth(xAxisLabel)) / 2, height - 12);
        }
        if (yAxisLabel != null && !yAxisLabel.isEmpty()) {
            // vertical-ish: draw near left
            g.drawString(yAxisLabel, 8, top + plotH / 2);
        }

        // tick labels
        g.setFont(new Font("SansSerif", Font.PLAIN, 10));
        for (int i = 0; i <= 5; i++) {
            double xv = xMin + (xMax - xMin) * i / 5.0;
            double yv = yMin + (yMax - yMin) * i / 5.0;
            int x = left + i * plotW / 5;
            int y = top + plotH - i * plotH / 5;
            String xs = formatTick(xv);
            String ys = formatTick(yv);
            FontMetrics fm = g.getFontMetrics();
            g.drawString(xs, x - fm.stringWidth(xs) / 2, top + plotH + 14);
            g.drawString(ys, left - fm.stringWidth(ys) - 4, y + 4);
        }
    }

    protected static String formatTick(double v) {
        if (Double.isNaN(v) || Double.isInfinite(v)) return "";
        if (Math.abs(v) >= 1000 || (Math.abs(v) > 0 && Math.abs(v) < 0.01))
            return String.format("%.2e", v);
        if (Math.abs(v - Math.rint(v)) < 1e-9) return String.format("%.0f", v);
        return String.format("%.2f", v);
    }

    protected static int mapX(double x, double xMin, double xMax, int left, int plotW) {
        if (xMax == xMin) return left + plotW / 2;
        return left + (int) Math.round((x - xMin) / (xMax - xMin) * plotW);
    }

    protected static int mapY(double y, double yMin, double yMax, int top, int plotH) {
        if (yMax == yMin) return top + plotH / 2;
        return top + plotH - (int) Math.round((y - yMin) / (yMax - yMin) * plotH);
    }

    protected static double[] toDoubles(org.bytedeco.pytorch.data.dataframe.Column col) {
        return col.asDoubleArray();
    }

    protected static double[] toDoubles(org.bytedeco.pytorch.data.dataframe.DataFrame df, String col) {
        return df.column(col).asDoubleArray();
    }

    protected static double[] indexArray(int n) {
        double[] x = new double[n];
        for (int i = 0; i < n; i++) x[i] = i;
        return x;
    }

    protected static double min(double[] a) {
        double m = Double.POSITIVE_INFINITY;
        for (double v : a) if (!Double.isNaN(v) && v < m) m = v;
        return m == Double.POSITIVE_INFINITY ? 0 : m;
    }

    protected static double max(double[] a) {
        double m = Double.NEGATIVE_INFINITY;
        for (double v : a) if (!Double.isNaN(v) && v > m) m = v;
        return m == Double.NEGATIVE_INFINITY ? 1 : m;
    }

    protected static void padRange(double[] minMax) {
        double span = minMax[1] - minMax[0];
        if (span == 0) span = Math.abs(minMax[0]) * 0.1 + 1;
        minMax[0] -= span * 0.05;
        minMax[1] += span * 0.05;
    }
}
