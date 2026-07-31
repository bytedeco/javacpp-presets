package org.bytedeco.pytorch.plot.chart;

import org.bytedeco.pytorch.dataframe.Column;
import org.bytedeco.pytorch.dataframe.DataFrame;

import java.awt.*;
import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;
import java.awt.image.BufferedImage;
import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.atomic.AtomicInteger;
import javax.imageio.ImageIO;
import javax.swing.*;

/**
 * Base chart with fluent config, AWT rendering (headless-safe savefig),
 * and optional Swing display.
 *
 * <p>{@link #show()} blocks until the user closes the window (matplotlib-script
 * semantics). {@link #show(boolean) show(false)} opens a non-blocking window that
 * stays alive via a strong reference until closed. Headless environments fall
 * back to writing a temp PNG and printing its path.
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
    /** "linear" (default) or "log" — matplotlib xscale/yscale. */
    protected String xScale = "linear";
    protected String yScale = "linear";

    /**
     * Strong refs to open non-modal windows so they are not GC'd / flash-closed
     * when the caller returns. Removed on windowClosed.
     */
    private static final List<Window> OPEN_WINDOWS = new ArrayList<>();
    private static final AtomicInteger WINDOW_SEQ = new AtomicInteger();

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

    /** matplotlib {@code plt.xscale("log"|"linear")}. */
    public BaseChart setXScale(String scale) {
        this.xScale = normalizeScale(scale);
        return this;
    }

    /** matplotlib {@code plt.yscale("log"|"linear")}. */
    public BaseChart setYScale(String scale) {
        this.yScale = normalizeScale(scale);
        return this;
    }

    public String getXScale() { return xScale; }
    public String getYScale() { return yScale; }

    private static String normalizeScale(String scale) {
        if (scale == null) return "linear";
        String s = scale.toLowerCase().trim();
        if ("log".equals(s) || "log10".equals(s) || "logit".equals(s)) return "log";
        return "linear";
    }

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

    /**
     * Display the chart and <b>block</b> until the user closes the window
     * (matplotlib interactive script default). Equivalent to {@code show(true)}.
     *
     * <p>If the environment is headless, writes a temp PNG and prints the path
     * instead of opening a GUI.
     */
    public void show() {
        show(true);
    }

    /**
     * Display the chart in a Swing window.
     *
     * @param block {@code true} — wait until the user closes the window (use in
     *              demos / scripts so the figure does not flash past);
     *              {@code false} — return immediately while keeping the window
     *              open via a strong reference until the user closes it
     */
    public void show(boolean block) {
        if (GraphicsEnvironment.isHeadless()) {
            try {
                File tmp = File.createTempFile(
                    "plot-" + WINDOW_SEQ.incrementAndGet() + "-", ".png");
                savefig(tmp.getAbsolutePath());
                System.out.println("[plot] headless — saved " + tmp.getAbsolutePath()
                    + "  (use savefig() / open that file; GUI show() unavailable)");
            } catch (Exception e) {
                System.out.println("[plot] headless — skip show(); use savefig(): " + e.getMessage());
            }
            return;
        }

        final BufferedImage img;
        try {
            img = render();
        } catch (Exception e) {
            System.err.println("[plot] render failed: " + e.getMessage());
            e.printStackTrace(System.err);
            return;
        }
        if (img == null) {
            System.err.println("[plot] render() returned null");
            return;
        }
        final String winTitle = (title == null || title.isEmpty()) ? "Plot" : title;

        try {
            if (block) {
                // Blocking: modal dialog is correct on EDT; latch+frame off EDT.
                if (SwingUtilities.isEventDispatchThread()) {
                    showModalBlocking(img, winTitle);
                } else {
                    CountDownLatch closed = new CountDownLatch(1);
                    SwingUtilities.invokeAndWait(() -> openFrame(img, winTitle, closed, true));
                    closed.await();
                }
            } else {
                // Non-blocking: still create on EDT; keep strong ref until closed.
                if (SwingUtilities.isEventDispatchThread()) {
                    openFrame(img, winTitle, null, true);
                } else {
                    SwingUtilities.invokeAndWait(() -> openFrame(img, winTitle, null, true));
                }
            }
        } catch (InterruptedException ie) {
            Thread.currentThread().interrupt();
            System.err.println("[plot] show interrupted — window may still be open");
        } catch (Exception e) {
            System.err.println("[plot] show failed: " + e.getMessage());
            e.printStackTrace(System.err);
        }
    }

    /**
     * Modal JDialog — blocks the EDT properly without deadlock (user must close
     * the dialog to continue). Used when {@link #show(boolean)} is called on the EDT.
     */
    private void showModalBlocking(BufferedImage img, String winTitle) {
        JDialog dialog = new JDialog((Frame) null, winTitle, true);
        dialog.setDefaultCloseOperation(WindowConstants.DISPOSE_ON_CLOSE);
        dialog.setContentPane(buildPlotPanel(img, winTitle));
        dialog.pack();
        dialog.setLocationRelativeTo(null);
        dialog.setResizable(true);
        // Ensure it stays above and focused
        dialog.setAlwaysOnTop(true);
        dialog.toFront();
        dialog.setVisible(true); // blocks until dispose
        dialog.setAlwaysOnTop(false);
    }

    /**
     * Create a non-modal JFrame. If {@code closed} is non-null, countDown on close
     * so an off-EDT caller can await. Always registers a strong ref while open.
     */
    private void openFrame(BufferedImage img, String winTitle,
                           CountDownLatch closed, boolean requestFocus) {
        JFrame frame = new JFrame(winTitle);
        frame.setDefaultCloseOperation(WindowConstants.DISPOSE_ON_CLOSE);
        frame.setContentPane(buildPlotPanel(img, winTitle));
        frame.pack();
        frame.setLocationByPlatform(true);
        if (frame.getLocation().x == 0 && frame.getLocation().y == 0) {
            frame.setLocationRelativeTo(null);
        }
        frame.setResizable(true);

        synchronized (OPEN_WINDOWS) {
            OPEN_WINDOWS.add(frame);
        }

        frame.addWindowListener(new WindowAdapter() {
            @Override public void windowClosed(WindowEvent e) {
                synchronized (OPEN_WINDOWS) {
                    OPEN_WINDOWS.remove(frame);
                }
                if (closed != null) closed.countDown();
            }

            @Override public void windowClosing(WindowEvent e) {
                // ensure dispose path fires windowClosed
                frame.dispose();
            }
        });

        frame.setVisible(true);
        if (requestFocus) {
            frame.toFront();
            frame.requestFocus();
            // macOS sometimes needs a nudge to raise the window
            frame.setAlwaysOnTop(true);
            frame.setAlwaysOnTop(false);
        }
    }

    /** Scrollable image + hint bar so the user knows how to dismiss. */
    private static JPanel buildPlotPanel(BufferedImage img, String winTitle) {
        JPanel root = new JPanel(new BorderLayout(0, 4));
        root.setBorder(BorderFactory.createEmptyBorder(6, 6, 6, 6));

        JLabel image = new JLabel(new ImageIcon(img));
        image.setHorizontalAlignment(SwingConstants.CENTER);
        JScrollPane scroll = new JScrollPane(image);
        scroll.setBorder(BorderFactory.createLineBorder(Color.LIGHT_GRAY));
        root.add(scroll, BorderLayout.CENTER);

        JLabel hint = new JLabel("  " + winTitle
            + "  —  close this window to continue  (savefig still available for offline view)");
        hint.setFont(new Font("SansSerif", Font.PLAIN, 12));
        hint.setForeground(new Color(0x44, 0x44, 0x44));
        root.add(hint, BorderLayout.SOUTH);
        return root;
    }

    /** How many plot windows are currently open (non-modal). */
    public static int openWindowCount() {
        synchronized (OPEN_WINDOWS) {
            return OPEN_WINDOWS.size();
        }
    }

    /** Close all non-modal plot windows opened via {@link #show(boolean)}. */
    public static void closeAll() {
        List<Window> copy;
        synchronized (OPEN_WINDOWS) {
            copy = new ArrayList<>(OPEN_WINDOWS);
            OPEN_WINDOWS.clear();
        }
        for (Window w : copy) {
            try {
                w.dispose();
            } catch (Exception ignored) {}
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

        // tick labels (honor log scales)
        g.setFont(new Font("SansSerif", Font.PLAIN, 10));
        boolean xLog = isLog(xScale);
        boolean yLog = isLog(yScale);
        for (int i = 0; i <= 5; i++) {
            double xv = tickValue(xMin, xMax, i, 5, xLog);
            double yv = tickValue(yMin, yMax, i, 5, yLog);
            int x = left + i * plotW / 5;
            int y = top + plotH - i * plotH / 5;
            String xs = formatTick(xv);
            String ys = formatTick(yv);
            FontMetrics fm = g.getFontMetrics();
            g.drawString(xs, x - fm.stringWidth(xs) / 2, top + plotH + 14);
            g.drawString(ys, left - fm.stringWidth(ys) - 4, y + 4);
        }
    }

    protected static boolean isLog(String scale) {
        return scale != null && scale.toLowerCase().startsWith("log");
    }

    protected static double tickValue(double min, double max, int i, int n, boolean log) {
        if (!log) return min + (max - min) * i / (double) n;
        double lo = Math.log10(Math.max(min, Double.MIN_NORMAL));
        double hi = Math.log10(Math.max(max, Double.MIN_NORMAL * 10));
        return Math.pow(10, lo + (hi - lo) * i / (double) n);
    }

    protected static String formatTick(double v) {
        if (Double.isNaN(v) || Double.isInfinite(v)) return "";
        if (Math.abs(v) >= 1000 || (Math.abs(v) > 0 && Math.abs(v) < 0.01))
            return String.format("%.2e", v);
        if (Math.abs(v - Math.rint(v)) < 1e-9) return String.format("%.0f", v);
        return String.format("%.2f", v);
    }

    /** Linear map (legacy static — used when scale is linear). */
    protected static int mapX(double x, double xMin, double xMax, int left, int plotW) {
        if (xMax == xMin) return left + plotW / 2;
        return left + (int) Math.round((x - xMin) / (xMax - xMin) * plotW);
    }

    protected static int mapY(double y, double yMin, double yMax, int top, int plotH) {
        if (yMax == yMin) return top + plotH / 2;
        return top + plotH - (int) Math.round((y - yMin) / (yMax - yMin) * plotH);
    }

    /** Scale-aware X mapping (uses {@link #xScale}). */
    protected int mapXScaled(double x, double xMin, double xMax, int left, int plotW) {
        if (isLog(xScale)) {
            double xv = Math.max(x, Double.MIN_NORMAL);
            double lo = Math.log10(Math.max(xMin, Double.MIN_NORMAL));
            double hi = Math.log10(Math.max(xMax, Double.MIN_NORMAL * 10));
            if (hi == lo) return left + plotW / 2;
            return left + (int) Math.round((Math.log10(xv) - lo) / (hi - lo) * plotW);
        }
        return mapX(x, xMin, xMax, left, plotW);
    }

    /** Scale-aware Y mapping (uses {@link #yScale}). */
    protected int mapYScaled(double y, double yMin, double yMax, int top, int plotH) {
        if (isLog(yScale)) {
            double yv = Math.max(y, Double.MIN_NORMAL);
            double lo = Math.log10(Math.max(yMin, Double.MIN_NORMAL));
            double hi = Math.log10(Math.max(yMax, Double.MIN_NORMAL * 10));
            if (hi == lo) return top + plotH / 2;
            return top + plotH - (int) Math.round((Math.log10(yv) - lo) / (hi - lo) * plotH);
        }
        return mapY(y, yMin, yMax, top, plotH);
    }

    /** Ensure log-scale ranges stay strictly positive. */
    protected static void ensurePositiveRange(double[] minMax) {
        if (minMax[0] <= 0) minMax[0] = Math.max(Double.MIN_NORMAL, minMax[1] * 1e-6);
        if (minMax[1] <= minMax[0]) minMax[1] = minMax[0] * 10;
    }

    protected static double[] toDoubles(Column col) {
        return col.asDoubleArray();
    }

    protected static double[] toDoubles(DataFrame df, String col) {
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
